#!/usr/bin/env python3
# Experiment 03: confidence-gated phishing classifiers.
# XGBoost and TabPFN gates use hand-crafted or frozen-ModernBERT features. Rows
# below the confidence threshold are escalated to a decoder or to experiment
# 01's validation-selected classifier. Exact-text deduplication precedes the
# outer split; cache coverage is reported for decoder-dependent rows.

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_llm_ensembles.utils.datasets import load_phishing_dataset, strip_provenance
from ml_llm_ensembles.utils.features import build_phishing_email_feature_matrix
from ml_llm_ensembles.utils.models import train_xgb, train_tabpfn, build_modernbert_features, xgb_device
from ml_llm_ensembles.utils.prompts import DOMAIN_PROMPTS, classify_with_cache

SEED = 42
THRESHOLD = 0.7
CACHE_DIR = _ROOT / "results" / "cache"
CACHE_FILE = CACHE_DIR / "llm_cache.json"
DECODERS = ["mistral", "gemma3:12b", "gpt-oss", "llama3.1:8b", "llama3.2"]
CLAUDE = "claude-sonnet-4-6"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=DECODERS)
    p.add_argument("--claude", action="store_true", help="include Claude (cache-only; no new calls)")
    p.add_argument("--ft-dir", type=Path, default=None,
                   help="validation-selected fine-tuned checkpoint from experiment 01. If unset, "
                        "auto-resolves: models/modernbert-phishing-ft, or "
                        "...-ft-stripped under --strip-provenance.")
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--strip-provenance", action="store_true")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    import os, random
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 classification_report)

    random.seed(args.seed); np.random.seed(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))

    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    prompt = DOMAIN_PROMPTS["phishing"]

    # ── Load (dedup inside loader, before split) + optional provenance ablation
    df = load_phishing_dataset("zefang-liu")
    _lim = os.environ.get("EXPERIMENT_LIMIT")
    if _lim and int(_lim) < len(df):
        df = df.sample(n=int(_lim), random_state=args.seed).reset_index(drop=True)
        print(f"  [smoke] capped to {len(df)} rows")
    if args.strip_provenance:
        df = df.copy(); df["text"] = df["text"].map(strip_provenance)
        print("  [ablation] stripped provenance tokens")

    # Match the router text variant to the corresponding experiment-01 checkpoint.
    if args.ft_dir is None:
        args.ft_dir = (_ROOT / "models" / "modernbert-phishing-ft-stripped"
                       if args.strip_provenance
                       else _ROOT / "models" / "modernbert-phishing-ft")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=df["label"])
    train_df = train_df.reset_index(drop=True); test_df = test_df.reset_index(drop=True)
    y_train, y_test = train_df["label"].values, test_df["label"].values
    texts_test = test_df["text"].tolist()
    print(f"Train {len(train_df)} | Test {len(test_df)} (prior {y_test.mean():.3f})")

    # ── Tier-1 gate models: XGB and TabPFN, on hand-crafted feats and BERT emb ─
    # The router gate can be any calibrated tier-1. We build all four so the panel
    # has a symmetric gate matrix: {XGB, TabPFN} × {hand-crafted, frozen-BERT}.
    device = xgb_device()
    X_tr = build_phishing_email_feature_matrix(train_df).values.astype(float)
    X_te = build_phishing_email_feature_matrix(test_df).values.astype(float)
    xgb = train_xgb(X_tr, y_train, random_state=args.seed)
    xgb_te = xgb.predict_proba(X_te)                       # (N,2)
    pfn = train_tabpfn(X_tr, y_train, random_state=args.seed, device=device)
    pfn_te = pfn.predict_proba(X_te)

    Xb_tr, Xb_te = build_modernbert_features(
        train_df["text"].tolist(), texts_test, cache_dir=CACHE_DIR)
    xgb_bert = train_xgb(Xb_tr, y_train, random_state=args.seed)
    xgb_bert_te = xgb_bert.predict_proba(Xb_te)
    pfn_bert = train_tabpfn(Xb_tr, y_train, random_state=args.seed, device=device)
    pfn_bert_te = pfn_bert.predict_proba(Xb_te)

    def metrics(y, preds, probs):
        return {"aucpr": float(average_precision_score(y, probs)),
                "rocauc": float(roc_auc_score(y, probs)),
                "accuracy": float((np.asarray(preds) == y).mean())}

    rows = {}
    rows["XGB Only"] = metrics(y_test, xgb_te[:, 1] >= 0.5, xgb_te[:, 1])
    rows["TabPFN Only"] = metrics(y_test, pfn_te[:, 1] >= 0.5, pfn_te[:, 1])
    rows["XGB+BERT Only"] = metrics(y_test, xgb_bert_te[:, 1] >= 0.5, xgb_bert_te[:, 1])
    rows["TabPFN+BERT Only"] = metrics(y_test, pfn_bert_te[:, 1] >= 0.5, pfn_bert_te[:, 1])

    # Returns (pred, P(attack)). The returned label is used for accuracy/F1; the
    # class-conditioned probability is used for ranking. Do not re-threshold the
    # probability to replace the returned label.
    MIN_EVAL = 20   # floor so a small cached SUBSET (e.g. Claude) still reports
                    # a row instead of being dropped; coverage is always shown.
    def llm_prob(text, model, backend):
        r = classify_with_cache(text, model, cache, prompt, backend, cache_only=True)
        if r is None:
            return None
        pred, conf = r
        p1 = conf if backend == "modernbert-ft" else (conf if pred == 1 else 1.0 - conf)
        return int(pred), float(p1)

    def eval_llm_only(model, backend):
        preds, P, Y = [], [], []
        for t, y in zip(texts_test, y_test):
            r = llm_prob(t, model, backend)
            if r is None: continue
            preds.append(r[0]); P.append(r[1]); Y.append(y)   # pred=returned label
        if len(Y) < MIN_EVAL or len(set(Y)) < 2: return None
        Y = np.array(Y); P = np.array(P)
        m = metrics(Y, preds, P)                  # acc from returned preds; AUCPR from P
        m["coverage"] = len(Y) / len(y_test); m["n_evaluated"] = len(Y)
        return m

    def eval_router(model, backend, gate_probs):
        """gate_probs: (N,2) tier-1 probs. Route when max<thr -> LLM; else gate."""
        conf = gate_probs.max(axis=1)
        routed = conf < args.threshold
        P, PR, Y, n_routed = [], [], [], 0
        for i, (t, y) in enumerate(zip(texts_test, y_test)):
            if not routed[i]:
                PR.append(int(gate_probs[i, 1] > 0.5)); P.append(float(gate_probs[i, 1])); Y.append(y)
            else:
                r = llm_prob(t, model, backend)
                if r is None: continue
                PR.append(r[0]); P.append(r[1]); Y.append(y); n_routed += 1
        if len(Y) < MIN_EVAL or len(set(Y)) < 2: return None
        Y = np.array(Y); P = np.array(P)
        m = metrics(Y, PR, P)
        m["routed_pct"] = 100.0 * n_routed / len(Y); m["coverage"] = len(Y) / len(y_test)
        m["n_evaluated"] = len(Y)
        return m

    all_models = list(args.models) + ([CLAUDE] if args.claude else [])
    for model in all_models:
        backend = "claude" if model == CLAUDE else "ollama"
        lo = eval_llm_only(model, backend)
        if lo: rows[f"{model} | LLM Only"] = lo
        r = eval_router(model, backend, xgb_te)
        if r: rows[f"{model} | Router"] = r
        rb = eval_router(model, backend, xgb_bert_te)
        if rb: rows[f"{model} | Router+BERT"] = rb
        rp = eval_router(model, backend, pfn_te)
        if rp: rows[f"{model} | Router-PFN"] = rp
        rpb = eval_router(model, backend, pfn_bert_te)
        if rpb: rows[f"{model} | Router-PFN+BERT"] = rpb

    # ── ModernBERT-FT tier: direct inference from experiment 01 ───────────────
    # The decoder cache key does not identify classifier checkpoint contents, so
    # this tier always loads the selected checkpoint and performs direct inference.
    if args.ft_dir.exists():
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(str(args.ft_dir))
        ftm = AutoModelForSequenceClassification.from_pretrained(str(args.ft_dir)).to(dev).eval()
        ft_all = []
        with torch.no_grad():
            for i in range(0, len(texts_test), 64):
                b = tok(texts_test[i:i + 64], padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(dev)
                ft_all.extend(torch.softmax(ftm(**b).logits, dim=-1)[:, 1].cpu().numpy().tolist())
        ft_all = np.array(ft_all)   # P(attack) for every test text, from THIS checkpoint
        print(f"  FT tier: live inference from {args.ft_dir.name} ({len(ft_all)} texts)")

        rows["ModernBERT-FT | LLM Only"] = metrics(y_test, ft_all >= 0.5, ft_all)

        # FT escalation tier under EVERY tier-1 gate (complete gate matrix).
        for gname, gate in [("Router", xgb_te), ("Router+BERT", xgb_bert_te),
                            ("Router-PFN", pfn_te), ("Router-PFN+BERT", pfn_bert_te)]:
            routed = gate.max(axis=1) < args.threshold
            PR = np.where(routed, ft_all >= 0.5, gate[:, 1] > 0.5).astype(int)
            P = np.where(routed, ft_all, gate[:, 1])
            m = metrics(y_test, PR, P); m["routed_pct"] = 100.0 * float(routed.mean())
            rows[f"ModernBERT-FT | {gname}"] = m
    else:
        print(f"  [warn] FT checkpoint {args.ft_dir} missing: run exp 01 first; FT tier skipped.")

    print(f"\n{'config':32s} {'AUCPR':>7s} {'ROC':>7s} {'acc':>7s} {'route%':>7s} {'cov':>5s}")
    for k, m in rows.items():
        print(f"{k:32s} {m['aucpr']:7.4f} {m['rocauc']:7.4f} {m['accuracy']:7.4f} "
              f"{m.get('routed_pct', float('nan')):7.1f} {m.get('coverage', 1.0):5.2f}")

    variant = "stripped" if args.strip_provenance else "raw"
    out = {"experiment": "03_router_phishing", "variant": variant,
           "deduped": True, "threshold": args.threshold, "seed": args.seed,
           "n_train": len(train_df), "n_test": len(test_df),
           "test_prior": float(y_test.mean()),
           "ft_checkpoint": str(args.ft_dir), "rows": rows}
    _suf = os.environ.get("EXPERIMENT_RESULT_SUFFIX", "")
    out_json = Path(__file__).with_name(f"03_router_phishing.{variant}{_suf}.result.json")
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
