#!/usr/bin/env python3
# Experiment 04: confidence-gated classifiers on Kitsune Mirai flows.
# Gate preprocessing is fitted on training rows only. Decoder tiers use the
# descriptive flow serialization; the fine-tuned tier uses experiment 02's
# compact serializer and merged checkpoint. This is a within-capture split.

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_llm_ensembles.utils.datasets import load_network_dataset
from ml_llm_ensembles.utils.models import train_xgb, train_tabpfn, build_modernbert_features, xgb_device
from ml_llm_ensembles.utils.prompts import (
    DOMAIN_PROMPTS, classify_with_cache,
    format_network_row_kitsune_pcap_flow, format_network_row_kitsune_pcap_flow_ft,
    KITSUNE_PCAP_FLOW_FEATURE_COLS,
)

SEED = 42
THRESHOLD = 0.7
CACHE_DIR = _ROOT / "results" / "cache"
CACHE_FILE = CACHE_DIR / "llm_cache.json"
DECODERS = ["mistral", "gemma3:12b", "gpt-oss", "llama3.1:8b", "llama3.2"]
CLAUDE = "claude-sonnet-4-6"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/kitsune")
    p.add_argument("--models", nargs="*", default=DECODERS)
    p.add_argument("--claude", action="store_true", help="include Claude (cache-only)")
    p.add_argument("--ft-dir", type=Path, default=_ROOT / "models" / "modernbert-mirai-flows-ft",
                   help="merged validation-selected flows checkpoint from experiment 02")
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--sweep", default="0.5,0.6,0.7,0.8,0.9",
                   help="router threshold sweep for the cost/quality figure.")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    import os, random
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import average_precision_score, roc_auc_score

    random.seed(args.seed); np.random.seed(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    prompt = DOMAIN_PROMPTS["network"]

    # ── Load flows (1 row/flow) and split (TEST identical to exp-02/meta) ─────
    df = load_network_dataset("kitsune-mirai-pcap-flows", args.data_dir)
    _lim = os.environ.get("EXPERIMENT_LIMIT")
    if _lim and int(_lim) < len(df):
        df = df.sample(n=int(_lim), random_state=args.seed).reset_index(drop=True)
        print(f"  [smoke] capped to {len(df)} flows")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=df["label"])
    train_df = train_df.reset_index(drop=True); test_df = test_df.reset_index(drop=True)
    y_train, y_test = train_df["label"].values, test_df["label"].values
    print(f"Train {len(train_df)} | Test {len(test_df)} (prior {y_test.mean():.3f})")

    # ── Hand-crafted flow features with training-only imputation ─────────────
    cols = [c for c in KITSUNE_PCAP_FLOW_FEATURE_COLS if c in train_df.columns]
    def numeric(d):
        return (d[cols].apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan))
    device = xgb_device()
    imp = SimpleImputer(strategy="median").fit(numeric(train_df))   # FIT ON TRAIN ONLY
    X_tr = imp.transform(numeric(train_df)); X_te = imp.transform(numeric(test_df))
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = neg / max(pos, 1)
    # Tier-1 gates: {XGB, TabPFN} on hand-crafted flow stats (symmetric gate matrix)
    xgb = train_xgb(X_tr, y_train, scale_pos_weight=spw, random_state=args.seed)
    xgb_te = xgb.predict_proba(X_te)
    pfn = train_tabpfn(X_tr, y_train, random_state=args.seed, device=device)
    pfn_te = pfn.predict_proba(X_te)

    # ── Frozen BERT embeddings (TabLLM serialization) -> XGB and TabPFN gates ──
    tr_text = [format_network_row_kitsune_pcap_flow(r) for _, r in train_df.iterrows()]
    te_text = [format_network_row_kitsune_pcap_flow(r) for _, r in test_df.iterrows()]
    Xb_tr, Xb_te = build_modernbert_features(
        tr_text, te_text, cache_dir=CACHE_DIR,
        cache_filename=f"modernbert_embeddings_kitsune-mirai-pcap-flows_{len(train_df)}.npz")
    xgb_bert = train_xgb(Xb_tr, y_train, scale_pos_weight=spw, random_state=args.seed)
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

    # Decoder LLM cached output. Returns (pred, P(attack)). The PREDICTION is the
    # model's returned label (used for accuracy/F1);
    # only the ranking PROBABILITY is class-conditioned (used for AUCPR). Do NOT
    # re-threshold P(attack) to derive the label: that flips low-confidence
    # benign/attack calls and diverges from the runner's reported accuracy.
    MIN_EVAL = 20   # floor so a small cached SUBSET (e.g. Claude) still reports,
                    # rather than being silently dropped; coverage is always shown.
    def llm_prob(text, model, backend):
        r = classify_with_cache(text, model, cache, prompt, backend, cache_only=True)
        if r is None: return None
        pred, conf = r
        return int(pred), float(conf if pred == 1 else 1.0 - conf)

    def eval_llm_only(model, backend):
        preds, P, Y = [], [], []
        for t, y in zip(te_text, y_test):
            r = llm_prob(t, model, backend)
            if r is None: continue
            preds.append(r[0]); P.append(r[1]); Y.append(y)   # pred=returned label
        if len(Y) < MIN_EVAL or len(set(Y)) < 2: return None
        Y, P = np.array(Y), np.array(P)
        m = metrics(Y, preds, P)                  # acc from returned preds; AUCPR from P
        m["coverage"] = len(Y) / len(y_test); m["n_evaluated"] = len(Y)
        return m

    def eval_router(model, backend, gate_probs, threshold):
        routed = gate_probs.max(axis=1) < threshold
        P, PR, Y, nr = [], [], [], 0
        for i, (t, y) in enumerate(zip(te_text, y_test)):
            if not routed[i]:
                PR.append(int(gate_probs[i, 1] > 0.5)); P.append(float(gate_probs[i, 1])); Y.append(y)
            else:
                r = llm_prob(t, model, backend)
                if r is None: continue
                PR.append(r[0]); P.append(r[1]); Y.append(y); nr += 1   # pred=returned label
        if len(Y) < MIN_EVAL or len(set(Y)) < 2: return None
        Y, P = np.array(Y), np.array(P)
        m = metrics(Y, PR, P)
        m["routed_pct"] = 100.0 * nr / len(Y); m["coverage"] = len(Y) / len(y_test)
        m["n_evaluated"] = len(Y)
        return m

    all_models = list(args.models) + ([CLAUDE] if args.claude else [])
    for model in all_models:
        backend = "claude" if model == CLAUDE else "ollama"
        lo = eval_llm_only(model, backend)
        if lo: rows[f"{model} | LLM Only"] = lo
        r = eval_router(model, backend, xgb_te, args.threshold)
        if r: rows[f"{model} | Router"] = r
        rb = eval_router(model, backend, xgb_bert_te, args.threshold)
        if rb: rows[f"{model} | Router+BERT"] = rb
        rp = eval_router(model, backend, pfn_te, args.threshold)
        if rp: rows[f"{model} | Router-PFN"] = rp
        rpb = eval_router(model, backend, pfn_bert_te, args.threshold)
        if rpb: rows[f"{model} | Router-PFN+BERT"] = rpb

    # ── Threshold sweep (cost/quality figure): Router AUCPR vs routed% ─────────
    # Sweep EVERY tier-1 gate, not just XGB: the "routing decays as routed% rises"
    # claim should hold across the whole gate family {XGB, TabPFN} × {plain, BERT}.
    sweep_gates = {"XGB": xgb_te, "TabPFN": pfn_te,
                   "XGB+BERT": xgb_bert_te, "TabPFN+BERT": pfn_bert_te}
    sweep = {}
    for model in args.models:
        for gname, gate in sweep_gates.items():
            for thr in [float(x) for x in args.sweep.split(",")]:
                r = eval_router(model, "ollama", gate, thr)
                if r: sweep[f"{model}@{gname}@{thr}"] = {
                    "aucpr": r["aucpr"], "routed_pct": r["routed_pct"]}

    # ── ModernBERT-FT tier: LIVE inference from exp-02 MERGED checkpoint ──────
    if args.ft_dir.exists():
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(str(args.ft_dir))
        ftm = AutoModelForSequenceClassification.from_pretrained(str(args.ft_dir)).to(dev).eval()
        ft_text = [format_network_row_kitsune_pcap_flow_ft(r) for _, r in test_df.iterrows()]
        ft_all = []
        with torch.no_grad():
            for i in range(0, len(ft_text), 64):
                b = tok(ft_text[i:i + 64], padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(dev)
                ft_all.extend(torch.softmax(ftm(**b).logits, dim=-1)[:, 1].cpu().numpy().tolist())
        ft_all = np.array(ft_all)
        print(f"  FT tier: live inference from {args.ft_dir.name}")
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
        print(f"  [warn] FT checkpoint {args.ft_dir} missing: run exp 02 first; FT tier skipped.")

    print(f"\n{'config':32s} {'AUCPR':>7s} {'ROC':>7s} {'acc':>7s} {'route%':>7s} {'cov':>5s}")
    for k, m in rows.items():
        print(f"{k:32s} {m['aucpr']:7.4f} {m['rocauc']:7.4f} {m['accuracy']:7.4f} "
              f"{m.get('routed_pct', float('nan')):7.1f} {m.get('coverage', 1.0):5.2f}")
    print("\nthreshold sweep (Router, XGB gate):")
    for k, v in sweep.items():
        print(f"  {k:24s} aucpr={v['aucpr']:.4f} routed%={v['routed_pct']:.1f}")

    out = {"experiment": "04_router_flows",
           # flows are 1 row per 5-tuple aggregation; the loader does NOT run an
           # explicit drop_duplicates (unlike the AfterImage/CIC loaders), so do
           # not claim network dedup here.
           "network_dedup": "n/a (one row per 5-tuple flow; no explicit dedup)",
           "within_capture_caveat": True, "threshold": args.threshold,
           "seed": args.seed, "n_train": len(train_df), "n_test": len(test_df),
           "test_prior": float(y_test.mean()), "ft_checkpoint": str(args.ft_dir),
           "rows": rows, "threshold_sweep": sweep}
    _suf = os.environ.get("EXPERIMENT_RESULT_SUFFIX", "")
    out_json = Path(__file__).with_name(f"04_router_flows{_suf}.result.json")
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
