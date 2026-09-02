#!/usr/bin/env python3
# Experiment 09: out-of-fold stacking on CIC-IDS2017 flow statistics.
# The experiment uses a conventional stratified random split after exact-row
# deduplication. Timestamp information is not retained, so this is a
# within-benchmark comparison rather than a temporal generalization result.

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_llm_ensembles.utils.datasets import load_network_dataset
from ml_llm_ensembles.utils.models import train_xgb, train_tabpfn, build_modernbert_features, xgb_device
from ml_llm_ensembles.utils.prompts import DOMAIN_PROMPTS, classify_with_cache, format_network_row_cicids

SEED = 42
N_FOLDS = 5
CACHE_DIR = _ROOT / "results" / "cache"
CACHE_FILE = CACHE_DIR / "llm_cache.json"
DECODERS = ["mistral", "gemma3:12b", "llama3.2"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/cicids2017")
    p.add_argument("--models", nargs="*", default=DECODERS)
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--folds", type=int, default=N_FOLDS)
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

    # ── Load CIC-IDS2017, sample, RANDOM stratified split (matches campaign) ──
    df = load_network_dataset("cicids2017", args.data_dir)
    if args.samples and args.samples < len(df):
        df = df.sample(n=args.samples, random_state=args.seed).reset_index(drop=True)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=df["label"])
    train_df = train_df.reset_index(drop=True); test_df = test_df.reset_index(drop=True)
    y_tr, y_te = train_df["label"].values, test_df["label"].values
    feat_cols = [c for c in train_df.columns if c != "label"]
    print(f"Train {len(train_df)} | Test {len(test_df)} | {len(feat_cols)} feats "
          f"(prior {y_te.mean():.3f}): RANDOM split (cicids leakage caveat applies)")

    def numeric(d):
        return (d[feat_cols].apply(pd.to_numeric, errors="coerce")
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

    def oof_impute(Xraw, y, fit_fn, label, impute=True):
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        P = np.zeros((len(y), 2))
        for k, (tr, va) in enumerate(skf.split(Xraw, y)):
            if impute:
                imp = SimpleImputer(strategy="median").fit(Xraw[tr])
                Xt, Xv = imp.transform(Xraw[tr]), imp.transform(Xraw[va])
            else:
                Xt, Xv = Xraw[tr], Xraw[va]
            P[va] = fit_fn(Xt, y[tr]).predict_proba(Xv)
            print(f"    {label} OOF fold {k+1}/{args.folds}")
        return P

    imp_full = SimpleImputer(strategy="median").fit(Xraw_tr)
    Xtr, Xte = imp_full.transform(Xraw_tr), imp_full.transform(Xraw_te)

    xgb_oof = oof_impute(Xraw_tr, y_tr, fit_xgb, "XGB")
    xgb_te = fit_xgb(Xtr, y_tr).predict_proba(Xte)
    pfn_oof = oof_impute(Xraw_tr, y_tr, fit_pfn, "TabPFN")
    pfn_te = fit_pfn(Xtr, y_tr).predict_proba(Xte)

    tr_text = [format_network_row_cicids(r) for _, r in train_df.iterrows()]
    te_text = [format_network_row_cicids(r) for _, r in test_df.iterrows()]
    Xb_tr, Xb_te = build_modernbert_features(
        tr_text, te_text, cache_dir=CACHE_DIR,
        cache_filename=f"modernbert_embeddings_cicids2017_{len(train_df)}.npz")
    bxgb_oof = oof_impute(Xb_tr, y_tr, fit_xgb, "XGB+BERT", impute=False)
    bxgb_te = fit_xgb(Xb_tr, y_tr).predict_proba(Xb_te)
    bpfn_oof = oof_impute(Xb_tr, y_tr, fit_pfn, "TabPFN+BERT", impute=False)
    bpfn_te = fit_pfn(Xb_tr, y_tr).predict_proba(Xb_te)

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
        m["coverage_test"] = 1.0; m["n_evaluated"] = len(y_te)
        return m
    rows = {}
    rows["XGB Only"] = base_row(xgb_te[:, 1])
    rows["TabPFN Only"] = base_row(pfn_te[:, 1])
    rows["XGB+BERT Only"] = base_row(bxgb_te[:, 1])
    rows["TabPFN+BERT Only"] = base_row(bpfn_te[:, 1])

    def stack(name, tr_list, te_list, ytr, yte, cov_tr=1.0, cov_te=1.0, n_te=None):
        for s2 in args.stage2:
            mdl = make_stage2(s2); mdl.fit(np.hstack(tr_list), ytr)
            prob1 = mdl.predict_proba(np.hstack(te_list))[:, 1]
            preds = mdl.predict(np.hstack(te_list))
            m = metrics(yte, preds, prob1)
            m["coverage_train"] = float(cov_tr); m["coverage_test"] = float(cov_te)
            m["n_evaluated"] = int(n_te if n_te is not None else len(yte))
            rows[f"{name} [{'LR' if s2=='logreg' else 'XGB'}]"] = m

    stack("XGB+TabPFN", [xgb_oof, pfn_oof], [xgb_te, pfn_te], y_tr, y_te)

    for model in args.models:
        ltr, _, mtr = llm_feats(tr_text, model)
        lte, lpred_te, mte = llm_feats(te_text, model)
        cov_tr, cov_te, n_te = float(mtr.mean()), float(mte.mean()), int(mte.sum())
        print(f"  LLM {model}: train cov {cov_tr:.0%}, test cov {cov_te:.0%} (n_test={n_te})")
        if mte.sum() < 20 or len(set(y_te[mte])) < 2:
            print(f"  [skip] {model}: insufficient test coverage"); continue
        m = metrics(y_te[mte], lpred_te[mte], lte[mte, 1])
        m["coverage_test"] = cov_te; m["n_evaluated"] = n_te
        rows[f"{model} | LLM Only"] = m
        tr, te = mtr, mte
        ytr, yte = y_tr[tr], y_te[te]
        combos = [
            (f"XGB+{model}", [xgb_oof[tr], ltr[tr]], [xgb_te[te], lte[te]]),
            (f"TabPFN+{model}", [pfn_oof[tr], ltr[tr]], [pfn_te[te], lte[te]]),
            (f"XGB+TabPFN+{model}", [xgb_oof[tr], pfn_oof[tr], ltr[tr]], [xgb_te[te], pfn_te[te], lte[te]]),
            (f"XGB+BERT+{model}", [xgb_oof[tr], bxgb_oof[tr], ltr[tr]], [xgb_te[te], bxgb_te[te], lte[te]]),
            (f"TabPFN+BERT+{model}", [bpfn_oof[tr], ltr[tr]], [bpfn_te[te], lte[te]]),
        ]
        for name, trl, tel in combos:
            stack(name, trl, tel, ytr, yte, cov_tr=cov_tr, cov_te=cov_te, n_te=n_te)

    print(f"\n{'config':30s} {'AUCPR':>7s} {'ROC':>7s} {'acc':>7s} {'cov':>5s}")
    for k, m in rows.items():
        print(f"{k:30s} {m['aucpr']:7.4f} {m['rocauc']:7.4f} {m['accuracy']:7.4f} "
              f"{m.get('coverage_test', 1.0):5.2f}")

    out = {"experiment": "09_meta_cicids", "split": "random (cicids leakage caveat)",
           "within_capture_caveat": True, "folds": args.folds, "seed": args.seed,
           "stage2": args.stage2, "n_features": len(feat_cols),
           "n_train": len(train_df), "n_test": len(test_df),
           "test_prior": float(y_te.mean()), "rows": rows}
    _suf = os.environ.get("EXPERIMENT_RESULT_SUFFIX", "")
    out_json = Path(__file__).with_name(f"09_meta_cicids{_suf}.result.json")
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
