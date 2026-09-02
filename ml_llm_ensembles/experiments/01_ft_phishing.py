#!/usr/bin/env python3
# Experiment 01: full ModernBERT fine-tuning for binary phishing detection.
# Exact-text deduplication precedes a stratified outer split. Epoch selection
# uses a validation partition drawn only from outer training data; the test set
# is evaluated once after selection. Raw and provenance-stripped variants write
# separate checkpoints and results.

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_llm_ensembles.utils.datasets import load_phishing_dataset, strip_provenance

SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    p.add_argument("--output-dir", type=Path, default=_ROOT / "models" / "modernbert-phishing-ft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--val-frac", type=float, default=0.125,
                   help="Validation fraction CARVED FROM TRAIN (not from test). "
                        "0.125 of the 80%% train ≈ 10%% of the full dataset.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--strip-provenance", action="store_true",
                   help="Remove Enron source-corpus shortcut tokens before training "
                        "(robustness ablation; report raw vs ablated).")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    import os
    import random
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 classification_report)
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from tqdm import tqdm

    # ── Seed training and split RNGs ──────────────────────────────────────────
    # The split was already deterministic via random_state, but model init, batch
    # order, and GPU kernels were not pinned. Seed python/numpy/torch + a
    # DataLoader generator. (Full bitwise GPU determinism also needs
    # deterministic algorithms/cuBLAS flags; we pin seeds, which is enough for
    # run-to-run stability without the throughput hit.)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    loader_gen = torch.Generator().manual_seed(args.seed)

    # Result JSON is variant-specific so raw and --strip-provenance runs do NOT
    # overwrite each other.
    variant = "stripped" if args.strip_provenance else "raw"
    _suf = os.environ.get("EXPERIMENT_RESULT_SUFFIX", "")
    out_json = Path(__file__).with_name(f"01_ft_phishing.{variant}{_suf}.result.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    # ── Load (dedup happens INSIDE the loader, before any split) ──────────────
    df = load_phishing_dataset("zefang-liu")
    _lim = os.environ.get("EXPERIMENT_LIMIT")
    if _lim and int(_lim) < len(df):
        df = df.sample(n=int(_lim), random_state=args.seed).reset_index(drop=True)
        print(f"  [smoke] capped to {len(df)} rows")
    if args.strip_provenance:
        # Ablation: strip "enron"/"forwarded by"/*@enron.com so the model cannot
        # lean on source-corpus origin instead of phishing semantics.
        df = df.copy()
        df["text"] = df["text"].map(strip_provenance)
        print("  [ablation] stripped provenance tokens from all text")

    # ── THREE-WAY split (the core fix) ────────────────────────────────────────
    # 1) Outer 80/20 -> TEST is identical to router/meta (seed 42, stratified).
    train_full, test_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=df["label"])
    # 2) Carve VALIDATION out of TRAIN only. TEST is never touched until the end.
    train_df, val_df = train_test_split(
        train_full, test_size=args.val_frac, random_state=args.seed,
        stratify=train_full["label"])
    for d in (train_df, val_df, test_df):
        d.reset_index(drop=True, inplace=True)
    print(f"Split: train {len(train_df)} | val {len(val_df)} | test {len(test_df)} "
          f"(test prior {test_df['label'].mean():.3f})")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    class DS(Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(texts, padding=True, truncation=True,
                                 max_length=args.max_length, return_tensors="pt")
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            return {"input_ids": self.enc["input_ids"][i],
                    "attention_mask": self.enc["attention_mask"][i],
                    "labels": self.labels[i]}

    train_loader = DataLoader(DS(train_df.text.tolist(), train_df.label.tolist()),
                              batch_size=args.batch_size, shuffle=True,
                              generator=loader_gen)
    val_loader = DataLoader(DS(val_df.text.tolist(), val_df.label.tolist()),
                            batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(DS(test_df.text.tolist(), test_df.label.tolist()),
                             batch_size=args.batch_size * 2, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2,
        id2label={0: "legitimate", 1: "phishing"},
        label2id={"legitimate": 0, "phishing": 1}).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total = len(train_loader) * args.epochs
    warm = total // 10
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm
        else max(0.0, (total - s) / max(1, total - warm)))

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        probs, labels = [], []
        for batch in loader:
            y = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            p = torch.softmax(model(**batch).logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist()); labels.extend(y.numpy().tolist())
        probs, labels = np.array(probs), np.array(labels)
        preds = (probs >= 0.5).astype(int)
        return {
            "aucpr": float(average_precision_score(labels, probs)),
            "rocauc": float(roc_auc_score(labels, probs)),
            "accuracy": float((preds == labels).mean()),
            "report": classification_report(labels, preds, output_dict=True,
                                            zero_division=0),
        }

    # ── Train; select best epoch on VALIDATION ONLY ───────────────────────────
    best_val, best_epoch, best_state = -1.0, 0, None
    import copy
    for epoch in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item()
        val = evaluate(val_loader)
        print(f"  epoch {epoch}: train_loss={tot/len(train_loader):.4f} "
              f"VAL aucpr={val['aucpr']:.4f} rocauc={val['rocauc']:.4f}")
        if val["aucpr"] > best_val:           # selection target = VALIDATION
            best_val, best_epoch = val["aucpr"], epoch
            best_state = copy.deepcopy(model.state_dict())

    # ── Restore val-best weights; score TEST exactly once ─────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(test_loader)
    print(f"\n=== val-selected (epoch {best_epoch}, val AUCPR {best_val:.4f}) ===")
    print(f"TEST  AUCPR={test['aucpr']:.4f}  ROC-AUC={test['rocauc']:.4f}  "
          f"acc={test['accuracy']:.4f}  prior={test_df['label'].mean():.4f}")

    result = {
        "experiment": "01_ft_phishing",
        "fix": "val-based checkpoint selection (was test-based); 3-way split",
        "deduped": True, "strip_provenance": args.strip_provenance,
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "test_prior": float(test_df["label"].mean()),
        "best_epoch": best_epoch, "val_aucpr_selected": best_val,
        "test": {k: test[k] for k in ("aucpr", "rocauc", "accuracy")},
        "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
        "full_finetune": True,  # NOTE: paper text says LoRA; this matches the original artifact
    }
    result["variant"] = variant
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_json}")
    if not args.no_save:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"saved val-selected checkpoint -> {args.output_dir}")


if __name__ == "__main__":
    main()
