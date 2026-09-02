#!/usr/bin/env python3
# Experiment 05: out-of-fold stacking for binary phishing detection.
# Trained base models generate outer-training features with stratified OOF
# prediction and are refit on the complete training side for test scoring.
# Cached zero-shot scores do not require fitting. Raw and provenance-stripped
# variants use the same split policy and report coverage explicitly.

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
N_FOLDS = 5
CACHE_DIR = _ROOT / "results" / "cache"
CACHE_FILE = CACHE_DIR / "llm_cache.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=["llama3.2"],
                   help="decoder LLM(s) used as a stacking base (cache-only).")
    p.add_argument("--folds", type=int, default=N_FOLDS)
    p.add_argument("--stage2", nargs="+", default=["logreg", "xgboost"],
                   choices=["logreg", "xgboost"])
    p.add_argument("--strip-provenance", action="store_true")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    import os, random
    import numpy as np
    from sklearn.model_selection import StratifiedKFold, train_test_split
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
    prompt = DOMAIN_PROMPTS["phishing"]

    # ── Load (dedup before split) + optional provenance ablation ──────────────
    df = load_phishing_dataset("zefang-liu")
    _lim = os.environ.get("EXPERIMENT_LIMIT")
    if _lim and int(_lim) < len(df):
        df = df.sample(n=int(_lim), random_state=args.seed).reset_index(drop=True)
        print(f"  [smoke] capped to {len(df)} rows")
    if args.strip_provenance:
        df = df.copy(); df["text"] = df["text"].map(strip_provenance)
        print("  [ablation] stripped provenance tokens")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=df["label"])
    train_df = train_df.reset_index(drop=True); test_df = test_df.reset_index(drop=True)
    y_tr, y_te = train_df["label"].values, test_df["label"].values
    print(f"Train {len(train_df)} | Test {len(test_df)} (prior {y_te.mean():.3f})")

    def metrics(y, preds, prob1):
        return {"aucpr": float(average_precision_score(y, prob1)),
                "rocauc": float(roc_auc_score(y, prob1)),
                "accuracy": float((np.asarray(preds) == y).mean())}

    def oof(X, y, fit_fn, label):
        """Out-of-fold [P0,P1] on TRAIN: fold model never predicts its own train rows."""
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        P = np.zeros((len(y), 2))
        for k, (tr, va) in enumerate(skf.split(X, y)):
            P[va] = fit_fn(X[tr], y[tr]).predict_proba(X[va])
            print(f"    {label} OOF fold {k+1}/{args.folds}")
        return P

    # ── Base: XGB (hand-crafted, per-row features -> no impute leak) ───────────
    Xtr = build_phishing_email_feature_matrix(train_df).values.astype(float)
    Xte = build_phishing_email_feature_matrix(test_df).values.astype(float)
    fit_xgb = lambda X, y: train_xgb(X, y, random_state=args.seed)
    xgb_oof = oof(Xtr, y_tr, fit_xgb, "XGB")
    xgb_te = fit_xgb(Xtr, y_tr).predict_proba(Xte)

    # ── Base: TabPFN-3 (same features, no training; GPU) ──────────────────────
    fit_pfn = lambda X, y: train_tabpfn(X, y, random_state=args.seed, device=device)
    pfn_oof = oof(Xtr, y_tr, fit_pfn, "TabPFN")
    pfn_te = fit_pfn(Xtr, y_tr).predict_proba(Xte)

    # ── Base: frozen-BERT embeddings -> XGB and TabPFN heads ──────────────────
    Xb_tr, Xb_te = build_modernbert_features(
        train_df["text"].tolist(), test_df["text"].tolist(), cache_dir=CACHE_DIR)
    bxgb_oof = oof(Xb_tr, y_tr, fit_xgb, "XGB+BERT")
    bxgb_te = fit_xgb(Xb_tr, y_tr).predict_proba(Xb_te)
    bpfn_oof = oof(Xb_tr, y_tr, fit_pfn, "TabPFN+BERT")
    bpfn_te = fit_pfn(Xb_tr, y_tr).predict_proba(Xb_te)

    # ── LLM base (cache-only). Track BOTH returned label and class-cond prob. ──
    def llm_feats(texts, model, backend):
        probs = np.full((len(texts), 2), 0.5); preds = np.zeros(len(texts), int)
        mask = np.ones(len(texts), bool)
        for i, t in enumerate(texts):
            r = classify_with_cache(t, model, cache, prompt, backend, cache_only=True)
            if r is None: mask[i] = False; continue
            pred, conf = r
            p1 = conf if pred == 1 else 1.0 - conf
            probs[i] = [1.0 - p1, p1]; preds[i] = pred
        return probs, preds, mask

    def make_stage2(kind):
        if kind == "logreg":
            return LogisticRegression(random_state=args.seed, max_iter=1000)
        return XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                             random_state=args.seed, eval_metric="logloss", verbosity=0)

    def stack(name, tr_list, te_list, m_y_tr, m_y_te, cov_tr=1.0, cov_te=1.0, n_te=None):
        Xs_tr = np.hstack(tr_list); Xs_te = np.hstack(te_list)
        for s2 in args.stage2:
            mdl = make_stage2(s2); mdl.fit(Xs_tr, m_y_tr)
            prob1 = mdl.predict_proba(Xs_te)[:, 1]; preds = mdl.predict(Xs_te)
            m = metrics(m_y_te, preds, prob1)
            # Record coverage so partial (e.g. stripped-text) rows are NOT mistaken
            # for complete-test numbers.
            m["coverage_train"] = float(cov_tr); m["coverage_test"] = float(cov_te)
            m["n_evaluated"] = int(n_te if n_te is not None else len(m_y_te))
            rows[f"{name} [{ 'LR' if s2=='logreg' else 'XGB' }]"] = m

    def base_row(prob1):
        m = metrics(y_te, prob1 >= 0.5, prob1)
        m["coverage_test"] = 1.0; m["n_evaluated"] = len(y_te)   # full-test base rows
        return m
    rows = {}
    # base-only rows (full-train model on TEST)
    rows["XGB Only"] = base_row(xgb_te[:, 1])
    rows["TabPFN Only"] = base_row(pfn_te[:, 1])
    rows["XGB+BERT Only"] = base_row(bxgb_te[:, 1])
    rows["TabPFN+BERT Only"] = base_row(bpfn_te[:, 1])
    # no-LLM stacks
    stack("XGB+TabPFN", [xgb_oof, pfn_oof], [xgb_te, pfn_te], y_tr, y_te)

    for model in args.models:
        backend = "ollama"
        ltr, _, mtr = llm_feats(train_df["text"].tolist(), model, backend)
        lte, lpred_te, mte = llm_feats(test_df["text"].tolist(), model, backend)
        cov_tr, cov_te, n_te = mtr.mean(), mte.mean(), int(mte.sum())
        print(f"  LLM {model} coverage train={cov_tr:.0%} test={cov_te:.0%} (n_test={n_te})")
        # NOTE: with --strip-provenance the cache key changes (stripped text ->
        # different hash), so cache-only coverage drops (~86%). LLM-dependent
        # stripped rows are PARTIAL-EVALUATION and are NOT directly comparable to
        # the raw rows; coverage is recorded on every row so this is explicit.
        if cov_tr < 0.99 or cov_te < 0.99:
            print(f"  [warn] {model} coverage < 99%: LLM rows are partial-eval (see coverage_*)")
        # LLM-Only: accuracy from RETURNED label, AUCPR from class-cond P1 (fix)
        m = metrics(y_te[mte], lpred_te[mte], lte[mte, 1])
        m["coverage_test"] = float(cov_te); m["n_evaluated"] = n_te
        rows[f"{model} | LLM Only"] = m
        # Stacks use covered rows on each side independently.
        TR = mtr; TE = mte
        combos = [
            (f"XGB+{model}", [xgb_oof[TR], ltr[TR]], [xgb_te[TE], lte[TE]]),
            (f"TabPFN+{model}", [pfn_oof[TR], ltr[TR]], [pfn_te[TE], lte[TE]]),
            (f"XGB+TabPFN+{model}", [xgb_oof[TR], pfn_oof[TR], ltr[TR]], [xgb_te[TE], pfn_te[TE], lte[TE]]),
            (f"XGB+BERT+{model}", [xgb_oof[TR], bxgb_oof[TR], ltr[TR]], [xgb_te[TE], bxgb_te[TE], lte[TE]]),
            (f"TabPFN+BERT+{model}", [bpfn_oof[TR], ltr[TR]], [bpfn_te[TE], lte[TE]]),
        ]
        for name, trl, tel in combos:
            stack(name, trl, tel, y_tr[TR], y_te[TE], cov_tr=cov_tr, cov_te=cov_te, n_te=n_te)

    print(f"\n{'config':30s} {'AUCPR':>7s} {'ROC':>7s} {'acc':>7s} {'cov':>5s}")
    for k, m in rows.items():
        print(f"{k:30s} {m['aucpr']:7.4f} {m['rocauc']:7.4f} {m['accuracy']:7.4f} "
              f"{m.get('coverage_test', 1.0):5.2f}")

    variant = "stripped" if args.strip_provenance else "raw"
    out = {"experiment": "05_meta_phishing", "variant": variant, "deduped": True,
           "folds": args.folds, "seed": args.seed, "stage2": args.stage2,
           "n_train": len(train_df), "n_test": len(test_df),
           "test_prior": float(y_te.mean()), "rows": rows}
    _suf = os.environ.get("EXPERIMENT_RESULT_SUFFIX", "")
    out_json = Path(__file__).with_name(f"05_meta_phishing.{variant}{_suf}.result.json")
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
