#!/usr/bin/env python3
# Experiment 06: out-of-fold stacking on Kitsune Mirai flows.
# Numeric preprocessing is fitted within each OOF training fold. A separate
# imputer fitted on the complete outer training set transforms the test set.
# Decoder features are cache-backed and their coverage is reported. The split is
# flow-level and within one capture; it is not host- or time-disjoint.

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
    format_network_row_kitsune_pcap_flow, KITSUNE_PCAP_FLOW_FEATURE_COLS,
)

SEED = 42
N_FOLDS = 5
META_TRAIN = 5000        # cap on LLM-covered training rows
CACHE_DIR = _ROOT / "results" / "cache"
CACHE_FILE = CACHE_DIR / "llm_cache.json"
DECODERS = ["mistral", "gemma3:12b", "llama3.2"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/kitsune")
    p.add_argument("--models", nargs="*", default=DECODERS)
    p.add_argument("--folds", type=int, default=N_FOLDS)
    p.add_argument("--meta-train-samples", type=int, default=META_TRAIN)
    p.add_argument("--stage2", nargs="+", default=["logreg", "xgboost"],
                   choices=["logreg", "xgboost"])
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    import os, random
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from xgboost import XGBClassifier

    random.seed(args.seed); np.random.seed(args.seed)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    try:
        import torch; torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass
    device = xgb_device()
    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    prompt = DOMAIN_PROMPTS["network"]

    # ── Load flows; split (TEST identical to exp 02/04/router) ────────────────
    df = load_network_dataset("kitsune-mirai-pcap-flows", args.data_dir)
    _lim = os.environ.get("EXPERIMENT_LIMIT")
    if _lim and int(_lim) < len(df):
        df = df.sample(n=int(_lim), random_state=args.seed).reset_index(drop=True)
        print(f"  [smoke] capped to {len(df)} flows")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=df["label"])
    train_df = train_df.reset_index(drop=True); test_df = test_df.reset_index(drop=True)
    y_tr, y_te = train_df["label"].values, test_df["label"].values
    print(f"Train {len(train_df)} | Test {len(test_df)} (prior {y_te.mean():.3f})")

    cols = [c for c in KITSUNE_PCAP_FLOW_FEATURE_COLS if c in train_df.columns]
    def numeric(d):
        return (d[cols].apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan).values)
    Xraw_tr, Xraw_te = numeric(train_df), numeric(test_df)

    def fit_xgb(X, y):
        neg, pos = int((y == 0).sum()), int((y == 1).sum())
        return train_xgb(X, y, scale_pos_weight=neg / max(pos, 1), random_state=args.seed)
    def fit_pfn(X, y):
        return train_tabpfn(X, y, random_state=args.seed, device=device)

    def metrics(y, preds, prob1):
        return {"aucpr": float(average_precision_score(y, prob1)),
                "rocauc": float(roc_auc_score(y, prob1)),
                "accuracy": float((np.asarray(preds) == y).mean())}

    # ── OOF with PER-FOLD imputation (fit on fold-train only) ─────────────────
    def oof_impute(Xraw, y, fit_fn, label):
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        P = np.zeros((len(y), 2))
        for k, (tr, va) in enumerate(skf.split(Xraw, y)):
            imp = SimpleImputer(strategy="median").fit(Xraw[tr])   # fold-train only
            m = fit_fn(imp.transform(Xraw[tr]), y[tr])
            P[va] = m.predict_proba(imp.transform(Xraw[va]))
            print(f"    {label} OOF fold {k+1}/{args.folds}")
        return P

    def oof_plain(X, y, fit_fn, label):   # dense embeddings, no imputation
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        P = np.zeros((len(y), 2))
        for k, (tr, va) in enumerate(skf.split(X, y)):
            P[va] = fit_fn(X[tr], y[tr]).predict_proba(X[va])
            print(f"    {label} OOF fold {k+1}/{args.folds}")
        return P

    # Train-only imputer for the full-train -> TEST transform.
    imp_full = SimpleImputer(strategy="median").fit(Xraw_tr)
    Xtr, Xte = imp_full.transform(Xraw_tr), imp_full.transform(Xraw_te)

    xgb_oof = oof_impute(Xraw_tr, y_tr, fit_xgb, "XGB")
    xgb_te = fit_xgb(Xtr, y_tr).predict_proba(Xte)
    pfn_oof = oof_impute(Xraw_tr, y_tr, fit_pfn, "TabPFN")
    pfn_te = fit_pfn(Xtr, y_tr).predict_proba(Xte)

    tr_text = [format_network_row_kitsune_pcap_flow(r) for _, r in train_df.iterrows()]
    te_text = [format_network_row_kitsune_pcap_flow(r) for _, r in test_df.iterrows()]
    Xb_tr, Xb_te = build_modernbert_features(
        tr_text, te_text, cache_dir=CACHE_DIR,
        cache_filename=f"modernbert_embeddings_kitsune-mirai-pcap-flows_{len(train_df)}.npz")
    bxgb_oof = oof_plain(Xb_tr, y_tr, fit_xgb, "XGB+BERT")
    bxgb_te = fit_xgb(Xb_tr, y_tr).predict_proba(Xb_te)
    bpfn_oof = oof_plain(Xb_tr, y_tr, fit_pfn, "TabPFN+BERT")
    bpfn_te = fit_pfn(Xb_tr, y_tr).predict_proba(Xb_te)

    # ── Stratified meta-train subset (LLM-covered train rows) ─────────────────
    def strat_subset(y, n, seed):
        if n >= len(y): return np.arange(len(y))
        rng = np.random.default_rng(seed); idx = []
        for c in np.unique(y):
            ci = np.where(y == c)[0]; k = max(1, round(n * len(ci) / len(y)))
            idx.append(rng.choice(ci, size=min(k, len(ci)), replace=False))
        return np.sort(np.concatenate(idx))
    meta_idx = strat_subset(y_tr, args.meta_train_samples, args.seed)
    print(f"meta-train subset: {len(meta_idx)}/{len(y_tr)} (LLM-covered, stage-2 train)")

    def llm_feats(texts, model):
        probs = np.full((len(texts), 2), 0.5); preds = np.zeros(len(texts), int)
        mask = np.ones(len(texts), bool)
        for i, t in enumerate(texts):
            r = classify_with_cache(t, model, cache, prompt, "ollama", cache_only=True)
            if r is None: mask[i] = False; continue
            pred, conf = r; p1 = conf if pred == 1 else 1.0 - conf
            probs[i] = [1.0 - p1, p1]; preds[i] = pred
        return probs, preds, mask

    def make_stage2(kind):
        if kind == "logreg":
            return LogisticRegression(random_state=args.seed, max_iter=1000)
        return XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                             random_state=args.seed, eval_metric="logloss", verbosity=0)

    def base_row(prob1):
        m = metrics(y_te, prob1 >= 0.5, prob1)
        m["coverage_test"] = 1.0; m["n_evaluated"] = len(y_te)   # full-test base rows
        return m
    rows = {}
    rows["XGB Only"] = base_row(xgb_te[:, 1])
    rows["TabPFN Only"] = base_row(pfn_te[:, 1])
    rows["XGB+BERT Only"] = base_row(bxgb_te[:, 1])
    rows["TabPFN+BERT Only"] = base_row(bpfn_te[:, 1])

    # no-LLM stack on the meta subset (so it's comparable to the LLM stacks)
    def stack(name, tr_list, te_list, ytr, yte, cov_tr=1.0, cov_te=1.0, n_te=None):
        for s2 in args.stage2:
            mdl = make_stage2(s2); mdl.fit(np.hstack(tr_list), ytr)
            prob1 = mdl.predict_proba(np.hstack(te_list))[:, 1]
            preds = mdl.predict(np.hstack(te_list))
            m = metrics(yte, preds, prob1)
            # Persist coverage so partially-cached LLM rows are not mistaken for
            # complete-test numbers in the saved JSON.
            m["coverage_train"] = float(cov_tr); m["coverage_test"] = float(cov_te)
            m["n_evaluated"] = int(n_te if n_te is not None else len(yte))
            rows[f"{name} [{'LR' if s2=='logreg' else 'XGB'}]"] = m

    mi = meta_idx
    stack("XGB+TabPFN", [xgb_oof[mi], pfn_oof[mi]], [xgb_te, pfn_te], y_tr[mi], y_te)

    for model in args.models:
        ltr, _, mtr = llm_feats([tr_text[i] for i in mi], model)   # train: meta subset
        lte, lpred_te, mte = llm_feats(te_text, model)             # test: full
        print(f"  LLM {model}: train cov {mtr.mean():.0%} ({mtr.sum()}/{len(mi)}), "
              f"test cov {mte.mean():.0%} ({mte.sum()}/{len(y_te)})")
        if mte.sum() < 20 or len(set(y_te[mte])) < 2:
            print(f"  [skip] {model}: insufficient test coverage"); continue
        cov_tr, cov_te, n_te = float(mtr.mean()), float(mte.mean()), int(mte.sum())
        # LLM-Only: accuracy from RETURNED label, AUCPR from class-cond P1
        m = metrics(y_te[mte], lpred_te[mte], lte[mte, 1])
        m["coverage_test"] = cov_te; m["n_evaluated"] = n_te
        rows[f"{model} | LLM Only"] = m
        # Stacks: train rows = meta subset ∩ cached; test rows = full ∩ cached.
        tr = mtr; te = mte
        ytr, yte = y_tr[mi][tr], y_te[te]
        combos = [
            (f"XGB+{model}", [xgb_oof[mi][tr], ltr[tr]], [xgb_te[te], lte[te]]),
            (f"TabPFN+{model}", [pfn_oof[mi][tr], ltr[tr]], [pfn_te[te], lte[te]]),
            (f"XGB+TabPFN+{model}", [xgb_oof[mi][tr], pfn_oof[mi][tr], ltr[tr]],
                                    [xgb_te[te], pfn_te[te], lte[te]]),
            (f"XGB+BERT+{model}", [xgb_oof[mi][tr], bxgb_oof[mi][tr], ltr[tr]],
                                  [xgb_te[te], bxgb_te[te], lte[te]]),
            (f"TabPFN+BERT+{model}", [bpfn_oof[mi][tr], ltr[tr]], [bpfn_te[te], lte[te]]),
        ]
        for name, trl, tel in combos:
            stack(name, trl, tel, ytr, yte, cov_tr=cov_tr, cov_te=cov_te, n_te=n_te)

    print(f"\n{'config':30s} {'AUCPR':>7s} {'ROC':>7s} {'acc':>7s} {'cov':>5s}")
    for k, m in rows.items():
        print(f"{k:30s} {m['aucpr']:7.4f} {m['rocauc']:7.4f} {m['accuracy']:7.4f} "
              f"{m.get('coverage_test', 1.0):5.2f}")

    out = {"experiment": "06_meta_flows", "within_capture_caveat": True,
           "folds": args.folds, "seed": args.seed, "stage2": args.stage2,
           "meta_train_samples": int(len(meta_idx)),
           "n_train": len(train_df), "n_test": len(test_df),
           "test_prior": float(y_te.mean()), "rows": rows}
    _suf = os.environ.get("EXPERIMENT_RESULT_SUFFIX", "")
    out_json = Path(__file__).with_name(f"06_meta_flows{_suf}.result.json")
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
