#!/usr/bin/env python3
# Experiment 02: LoRA fine-tuning for serialized Kitsune Mirai flows.
# Validation is drawn only from the outer training side and selects the best
# epoch by AUCPR. The test set is scored after selection. Train, validation, and
# inference use the same compact key=value serializer. The adapter is merged
# before saving so downstream code loads a standard classifier checkpoint.

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_llm_ensembles.utils.datasets import load_network_dataset
from ml_llm_ensembles.utils.prompts import format_network_row_kitsune_pcap_flow_ft

SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/kitsune")
    p.add_argument("--output", type=Path, default=_ROOT / "models" / "modernbert-mirai-flows-ft")
    p.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--val-frac", type=float, default=0.125,
                   help="Validation fraction carved FROM TRAIN (not from test).")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    import os, random
    import numpy as np
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments,
                              set_seed)

    # Seed training and split RNGs.
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load via the SAME loader the experiments use (matching aggregation+split)
    flow_df = load_network_dataset("kitsune-mirai-pcap-flows", args.data_dir)
    _lim = os.environ.get("EXPERIMENT_LIMIT")
    if _lim and int(_lim) < len(flow_df):
        flow_df = flow_df.sample(n=int(_lim), random_state=args.seed).reset_index(drop=True)
        print(f"  [smoke] capped to {len(flow_df)} flows")

    # ── THREE-WAY split (the core fix) ────────────────────────────────────────
    # Outer 80/20 -> TEST identical to router/meta flows test (seed 42, stratified).
    train_full, test_df = train_test_split(
        flow_df, test_size=0.2, random_state=args.seed, stratify=flow_df["label"])
    # Carve VALIDATION from TRAIN only. TEST never seen until the final eval.
    train_df, val_df = train_test_split(
        train_full, test_size=args.val_frac, random_state=args.seed,
        stratify=train_full["label"])
    for d in (train_df, val_df, test_df):
        d.reset_index(drop=True, inplace=True)
    print(f"Split: train {len(train_df)} | val {len(val_df)} | test {len(test_df)} "
          f"(test prior {test_df['label'].mean():.3f})")

    # ── Serialize with the same format used by downstream inference ──────────
    ser = format_network_row_kitsune_pcap_flow_ft
    def texts(d): return [ser(r) for _, r in d.iterrows()]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    def tok(b): return tokenizer(b["text"], truncation=True, max_length=args.max_length)
    mk = lambda d: Dataset.from_dict(
        {"text": texts(d), "label": d["label"].tolist()}
    ).map(tok, batched=True, remove_columns=["text"])
    train_ds, val_ds, test_ds = mk(train_df), mk(val_df), mk(test_df)

    # ── Class-weighted loss for imbalance (as original) ───────────────────────
    n_attack = int(train_df["label"].sum()); n_benign = len(train_df) - n_attack
    pos_weight = n_benign / max(n_attack, 1)
    print(f"  imbalance: benign {n_benign}, attack {n_attack}, pos_weight {pos_weight:.2f}")

    base = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2, ignore_mismatched_sizes=True)
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=0.05, target_modules=["Wqkv", "Wo"], bias="none"))
    model.print_trainable_parameters()

    def weighted_loss_fn(outputs, labels, num_items_in_batch=None, **kw):
        w = torch.tensor([1.0, pos_weight], device=outputs.logits.device)
        return torch.nn.functional.cross_entropy(outputs.logits, labels, weight=w)

    # Keep prob of class 1 (NOT argmax) so we can select on validation AUCPR.
    def preprocess_logits(logits, labels):
        if isinstance(logits, (list, tuple)): logits = logits[0]
        return torch.softmax(logits, dim=-1)[:, 1]

    def compute_metrics(eval_pred):
        prob1, lbls = eval_pred
        prob1 = np.asarray(prob1).ravel()
        preds = (prob1 >= 0.5).astype(int)
        return {
            "aucpr": float(average_precision_score(lbls, prob1)),
            "rocauc": float(roc_auc_score(lbls, prob1)),
            "f1_macro": float(f1_score(lbls, preds, average="macro", zero_division=0)),
            "accuracy": float((preds == lbls).mean()),
        }

    targs = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr, warmup_steps=100, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="aucpr", greater_is_better=True,   # selection = VAL AUCPR
        fp16=(device == "cuda"), logging_steps=50, report_to="none",
        label_names=["labels"], seed=args.seed, data_seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        compute_loss_func=weighted_loss_fn,
        preprocess_logits_for_metrics=preprocess_logits,
    )
    print("\n=== Training (select best epoch on VALIDATION) ===")
    trainer.train()
    val_metrics = trainer.evaluate(val_ds)
    print(f"  best-checkpoint VAL: aucpr={val_metrics['eval_aucpr']:.4f} "
          f"rocauc={val_metrics['eval_rocauc']:.4f}")

    # ── Score TEST exactly once with the val-selected model ───────────────────
    test_metrics = trainer.evaluate(test_ds)
    print(f"\nTEST  AUCPR={test_metrics['eval_aucpr']:.4f}  "
          f"ROC-AUC={test_metrics['eval_rocauc']:.4f}  "
          f"acc={test_metrics['eval_accuracy']:.4f}  prior={test_df['label'].mean():.4f}")

    result = {
        "experiment": "02_ft_flows",
        "fix": "eval on validation (was test); val-AUCPR selection; 3-way split",
        "serializer": "format_network_row_kitsune_pcap_flow_ft (compact; matches inference)",
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha},
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "test_prior": float(test_df["label"].mean()),
        "val_aucpr_selected": float(val_metrics["eval_aucpr"]),
        "test": {"aucpr": float(test_metrics["eval_aucpr"]),
                 "rocauc": float(test_metrics["eval_rocauc"]),
                 "accuracy": float(test_metrics["eval_accuracy"])},
        "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
        "within_capture_caveat": True,
    }
    _suf = os.environ.get("EXPERIMENT_RESULT_SUFFIX", "")
    out_json = Path(__file__).with_name(f"02_ft_flows{_suf}.result.json")
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_json}")
    if not args.no_save:
        # MERGE the LoRA adapter into the base weights before saving. The
        # downstream FT-tier loader (utils/prompts.py _load_modernbert_ft) calls
        # AutoModelForSequenceClassification.from_pretrained(model_dir), which
        # expects a plain merged checkpoint: NOT a bare PEFT adapter dir. Saving
        # the adapter (model.save_pretrained on the PEFT model) would silently
        # produce a dir the router/inference path cannot load. merge_and_unload
        # returns the base model with LoRA folded in. (Mirrors the original
        # flows script's save contract.)
        args.output.mkdir(parents=True, exist_ok=True)
        merged = model.merge_and_unload()
        merged.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        print(f"saved MERGED val-selected checkpoint -> {args.output}")


if __name__ == "__main__":
    main()
