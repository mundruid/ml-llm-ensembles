#!/usr/bin/env python3
# Experiment 08: XGBoost and TabPFN on AfterImage features.
# The temporal policy orders rows separately within each class and is not a
# single global cutoff. The opt-in random-row reference is susceptible to shared
# sliding-window state. Neither policy is presented as cross-capture evidence.

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_llm_ensembles.utils.datasets import load_network_dataset
from ml_llm_ensembles.utils.models import train_xgb, train_tabpfn, xgb_device

SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/kitsune")
    p.add_argument("--samples", type=int, default=100000)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--split", choices=["random", "temporal"], default="temporal",
                   help="temporal (per-class first80/last20 in capture order) or "
                        "random (within-capture reference; leakage-prone).")
    p.add_argument("--allow-leaky-random", action="store_true",
                   help="required with --split random; acknowledges the known "
                        "AfterImage leakage from i.i.d. row splitting.")
    p.add_argument("--stage2", nargs="+", default=["logreg", "xgboost"],
                   choices=["logreg", "xgboost"])
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    import os, random
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
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

    # ── Load AfterImage; guard against the loader picking the wrong file ──────
    df = load_network_dataset("kitsune-mirai", args.data_dir)
    feat_cols = [c for c in df.columns if c != "label"]
    if len(feat_cols) < 100:
        sys.exit(f"ERROR: only {len(feat_cols)} feature cols: loader likely picked "
                 "the wrong file (expected approximately 115 AfterImage statistics).")
    df = df.copy()
    df["_capture_idx"] = np.arange(len(df))
    print(f"AfterImage: {len(df)} rows, {len(feat_cols)} feature cols")

    if args.samples and args.samples < len(df):
        # Preserve capture order by sorting the sampled indices.
        rng = np.random.default_rng(args.seed)
        sub = np.sort(rng.choice(len(df), size=args.samples, replace=False))
        df = df.iloc[sub].reset_index(drop=True)

    # ── Split: random (leakage-prone) or temporal (within-capture) ────────────
    if args.split == "random" and not args.allow_leaky_random:
        sys.exit("ERROR: --split random is known leakage-prone for AfterImage. "
                 "Re-run with --allow-leaky-random if you explicitly want the "
                 "campaign reproduction, or use the default --split temporal.")

    if args.split == "temporal":
        # Chronological split, computed PER CLASS. The capture has a single
        # benign-to-attack transition (Mirai infects partway through), so a global
        # first80/last20 cut lands entirely in the attack era and yields an
        # all-attack test set (no usable comparison). Splitting each class's own
        # timeline at 80/20 keeps both classes represented while still holding
        # out each row's "future" relative to training. Mirrors the split used in
        # run_experiments_router.py and cited in the paper.
        tr_idx, te_idx = [], []
        for lab in sorted(df["label"].unique()):
            idx = np.flatnonzero((df["label"] == lab).to_numpy())
            cut = int(len(idx) * 0.8)
            tr_idx.append(idx[:cut]); te_idx.append(idx[cut:])
        tr_idx, te_idx = np.sort(np.concatenate(tr_idx)), np.sort(np.concatenate(te_idx))
        train_df, test_df = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
        print(f"TEMPORAL split (per-class first80/last20): train {len(train_df)} | test {len(test_df)}")
    else:
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=args.seed, stratify=df["label"])
        train_df = train_df.reset_index(drop=True); test_df = test_df.reset_index(drop=True)
        print(f"RANDOM split (LEAKAGE-PRONE for windowed features): "
              f"train {len(train_df)} | test {len(test_df)}")

    split_diag = {
        "train_capture_min": int(train_df["_capture_idx"].min()),
        "train_capture_max": int(train_df["_capture_idx"].max()),
        "test_capture_min": int(test_df["_capture_idx"].min()),
        "test_capture_max": int(test_df["_capture_idx"].max()),
        "train_capture_monotone": bool(train_df["_capture_idx"].is_monotonic_increasing),
        "test_capture_monotone": bool(test_df["_capture_idx"].is_monotonic_increasing),
    }
    if args.split == "temporal":
        # Per class: every test row must come later in the capture than every
        # training row of the same class.
        ok = True
        for lab in sorted(df["label"].unique()):
            tr_max = train_df.loc[train_df["label"] == lab, "_capture_idx"].max()
            te_min = test_df.loc[test_df["label"] == lab, "_capture_idx"].min()
            split_diag[f"class{int(lab)}_train_capture_max"] = int(tr_max)
            split_diag[f"class{int(lab)}_test_capture_min"] = int(te_min)
            ok &= bool(tr_max < te_min)
        split_diag["temporal_order_respected"] = ok
        if not ok:
            sys.exit("ERROR: temporal split violated per-class capture order.")
    else:
        split_diag["temporal_order_respected"] = False

    y_tr, y_te = train_df["label"].values, test_df["label"].values
    print(f"  test prior {y_te.mean():.3f}")

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
        y = np.asarray(y)
        m = {"accuracy": float((np.asarray(preds) == y).mean()),
             "coverage_test": 1.0, "n_evaluated": int(len(y))}
        if len(set(y.tolist())) < 2:        # single-class test -> ranking metrics undefined
            m["aucpr"] = None; m["rocauc"] = None; m["degenerate"] = "single-class test"
        else:
            m["aucpr"] = float(average_precision_score(y, prob1))
            m["rocauc"] = float(roc_auc_score(y, prob1))
        return m

    def oof(Xraw, y, fit_fn, label):
        # Match OOF construction to the outer split. Temporal uses contiguous,
        # order-respecting blocks (KFold shuffle=False); random -> shuffled
        # stratified folds. (Contiguous blocks are order-respecting, not strictly
        # forward-chaining; good enough for this within-capture leakage check.)
        if args.split == "temporal":
            it = KFold(n_splits=args.folds, shuffle=False).split(Xraw)
        else:
            it = StratifiedKFold(n_splits=args.folds, shuffle=True,
                                 random_state=args.seed).split(Xraw, y)
        P = np.zeros((len(y), 2))
        for k, (tr, va) in enumerate(it):
            imp = SimpleImputer(strategy="median").fit(Xraw[tr])
            P[va] = fit_fn(imp.transform(Xraw[tr]), y[tr]).predict_proba(imp.transform(Xraw[va]))
            print(f"    {label} OOF fold {k+1}/{args.folds}")
        return P

    def make_stage2(kind):
        if kind == "logreg":
            return LogisticRegression(random_state=args.seed, max_iter=1000)
        return XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                             random_state=args.seed, eval_metric="logloss", verbosity=0)

    rows = {}
    note = None
    if len(set(y_tr.tolist())) < 2 or len(set(y_te.tolist())) < 2:
        # AfterImage's Mirai attack is time-localized; a temporal split can leave a
        # train or test block single-class -> not cleanly evaluable. Record the fact
        # and stop (this itself shows the random-split 1.000 is not a trustworthy
        # generalization estimate, rather than crashing).
        note = (f"degenerate {args.split} split (AfterImage attack burst): "
                f"train classes {sorted(set(y_tr.tolist()))}, test {sorted(set(y_te.tolist()))}")
        print(f"[{args.split}] {note} -> recording note, skipping fits.")
    else:
        imp_full = SimpleImputer(strategy="median").fit(Xraw_tr)
        Xtr, Xte = imp_full.transform(Xraw_tr), imp_full.transform(Xraw_te)
        xgb_te = fit_xgb(Xtr, y_tr).predict_proba(Xte)
        pfn_te = fit_pfn(Xtr, y_tr).predict_proba(Xte)
        rows["XGB Only"] = metrics(y_te, xgb_te[:, 1] >= 0.5, xgb_te[:, 1])
        rows["TabPFN Only"] = metrics(y_te, pfn_te[:, 1] >= 0.5, pfn_te[:, 1])
        # OOF stack only with reliable folds. Temporal contiguous folds on the
        # AfterImage burst go single-class (XGB cannot fit) -> base-only there.
        if args.split == "temporal":
            print("[temporal] base-only: skipping XGB+TabPFN OOF stack "
                  "(contiguous temporal folds are single-class).")
        else:
            xgb_oof = oof(Xraw_tr, y_tr, fit_xgb, "XGB")
            pfn_oof = oof(Xraw_tr, y_tr, fit_pfn, "TabPFN")
            for s2 in args.stage2:
                mdl = make_stage2(s2); mdl.fit(np.hstack([xgb_oof, pfn_oof]), y_tr)
                Xs_te = np.hstack([xgb_te, pfn_te])
                prob1 = mdl.predict_proba(Xs_te)[:, 1]; preds = mdl.predict(Xs_te)
                rows[f"XGB+TabPFN [{'LR' if s2=='logreg' else 'XGB'}]"] = metrics(y_te, preds, prob1)

    print(f"\n{'config':24s} {'AUCPR':>7s} {'ROC':>7s} {'acc':>7s}")
    for k, m in rows.items():
        a = "n/a" if m.get("aucpr") is None else f"{m['aucpr']:.4f}"
        r = "n/a" if m.get("rocauc") is None else f"{m['rocauc']:.4f}"
        print(f"{k:24s} {a:>7} {r:>7} {m['accuracy']:7.4f}")

    out = {"experiment": "08_meta_afterimage", "split": args.split,
           "split_guard": ("explicitly allowed leaky random split"
                           if args.split == "random" else
                           "default temporal split"),
           "leakage_caveat": "windowed/incremental features; random split is leakage-prone",
           "within_capture_caveat": True, "folds": args.folds, "seed": args.seed,
           "n_features": len(feat_cols), "n_train": len(train_df), "n_test": len(test_df),
           "test_prior": float(y_te.mean()), "split_diagnostics": split_diag,
           "note": note, "rows": rows}
    _suf = os.environ.get("EXPERIMENT_RESULT_SUFFIX", "")
    out_json = Path(__file__).with_name(f"08_meta_afterimage.{args.split}{_suf}.result.json")
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
