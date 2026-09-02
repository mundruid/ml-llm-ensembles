#!/usr/bin/env python3
# Experiment 11: CTU-13 split-policy and operational evaluation.
# Scenario holds out designated captures; host groups by the internal endpoint
# but can retain overlap through the other endpoint; temporal uses one global
# cutoff and forward-chaining OOF. XGBoost scores the complete test partition.
# Expensive models use a uniform test sample, and decoder rows use a balanced
# subset marked representative_test=false. Thresholds are selected on training
# OOF scores. Background labels are unverified negatives.

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_llm_ensembles.utils.datasets import load_ctu13, CTU13_DEFAULT_TEST_SCENARIOS
from ml_llm_ensembles.utils.models import train_xgb, train_tabpfn, build_modernbert_features, xgb_device
from ml_llm_ensembles.utils.prompts import (
    DOMAIN_PROMPTS, classify_with_cache, format_network_row_ctu13, CTU13_FEATURE_COLS,
)
from ml_llm_ensembles.utils.splits import make_split, oof_folds, endpoint_overlap_diag
from ml_llm_ensembles.utils.evaluation import operational_report, format_operational_table

SEED = 42
CACHE_DIR = _ROOT / "results" / "cache"
CACHE_FILE = CACHE_DIR / "llm_cache.json"
DECODERS = ["mistral", "gemma3:12b", "llama3.2"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/ctu13")
    p.add_argument("--split", choices=["scenario", "host", "temporal", "random"], default="scenario")
    p.add_argument("--test-scenarios", nargs="+", type=int, default=CTU13_DEFAULT_TEST_SCENARIOS)
    p.add_argument("--scenarios", nargs="+", type=int, default=None, help="subset of 1..13 to load")
    p.add_argument("--background", choices=["benign", "drop"], default="benign")
    p.add_argument("--train-rows", type=int, default=2_000_000,
                   help="uniform thinning of the TRAINING side only (0 = all)")
    p.add_argument("--expensive-test-rows", type=int, default=300_000,
                   help="uniform TEST subsample scored by TabPFN / BERT (0 = all)")
    p.add_argument("--models", nargs="*", default=DECODERS)
    p.add_argument("--allow-llm-calls", action="store_true", help="fill cache misses via Ollama and persist")
    p.add_argument("--llm-test-pos", type=int, default=1000)
    p.add_argument("--llm-test-neg", type=int, default=5000)
    p.add_argument("--llm-train-rows", type=int, default=2000, help="class-balanced LLM meta-train subset")
    p.add_argument("--tabpfn-rows", type=int, default=10_000)
    p.add_argument("--bert-rows", type=int, default=20_000)
    p.add_argument("--no-bert", action="store_true")
    p.add_argument("--no-tabpfn", action="store_true")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--stage2", nargs="+", default=["logreg", "xgboost"], choices=["logreg", "xgboost"])
    p.add_argument("--priors", nargs="+", type=float, default=[0.01, 0.001])
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--tag", default="", help="suffix for the result file name")
    return p.parse_args()


def save_cache_atomic(cache):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache)); os.replace(tmp, CACHE_FILE)


def texts_sha(texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode()); h.update(b"\0")
    return h.hexdigest()[:16]


def main():
    args = parse_args()
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from xgboost import XGBClassifier

    random.seed(args.seed); np.random.seed(args.seed)
    try:
        import torch; torch.manual_seed(args.seed)
    except Exception:
        pass
    device = xgb_device()
    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    prompt = DOMAIN_PROMPTS["network"]
    rng = np.random.default_rng(args.seed)

    # ── Load (complete scenarios; smoke mode thins per scenario) ─────────────
    lim = os.environ.get("EXPERIMENT_LIMIT")
    sps = max(int(lim), 50) if lim else None
    df = load_ctu13(args.data_dir, scenarios=args.scenarios, background=args.background,
                    sample_per_scenario=sps, seed=args.seed)
    if args.split == "temporal":
        df = df.sort_values("start_time", kind="stable").reset_index(drop=True)

    # ── Split ────────────────────────────────────────────────────────────────
    group_col = {"host": "host", "scenario": "scenario"}.get(args.split)
    # make_split/oof_folds know the generic "group" policy; "host" is our name for
    # group-on-the-internal-endpoint. Translate here (scenario stays scenario).
    split_policy = "group" if args.split == "host" else args.split
    tr_idx, te_idx, split_diag = make_split(
        df, split_policy, seed=args.seed, time_col="start_time", group_col=group_col,
        scenario_col="scenario", test_scenarios=args.test_scenarios)
    if args.split == "host":
        split_diag.update(endpoint_overlap_diag(df, tr_idx, te_idx))
    # training-side thinning only (uniform -> keeps the training prior in expectation)
    n_train_full = int(len(tr_idx))
    if args.train_rows and args.train_rows < len(tr_idx):
        tr_idx = np.sort(rng.choice(tr_idx, size=args.train_rows, replace=False))
    train_df = df.iloc[tr_idx].reset_index(drop=True)
    test_df = df.iloc[te_idx].reset_index(drop=True)
    del df
    y_tr, y_te = train_df["label"].values, test_df["label"].values
    if y_te.sum() == 0 or y_tr.sum() == 0:
        sys.exit(f"ERROR: degenerate split ({args.split}): train pos {y_tr.sum()}, test pos {y_te.sum()}")
    if args.split == "scenario":   # captures are disjoint in time; sum per-scenario spans
        hours = float(sum((g.max() - g.min()) / 3600.0 for _, g in test_df.groupby("scenario")["start_time"]))
    else:
        hours = float((test_df["start_time"].max() - test_df["start_time"].min()) / 3600.0)
    print(f"\nSPLIT={args.split}  train {len(train_df)} of {n_train_full} (prior {y_tr.mean():.4f}) | "
          f"test {len(test_df)} COMPLETE (prior {y_te.mean():.4f}, {hours:.1f} capture-hours)")
    print(f"  diag: {split_diag}")
    groups_tr = train_df[group_col].values if group_col else None
    times_tr = train_df["start_time"].values

    feat_cols = [c for c in CTU13_FEATURE_COLS if c in train_df.columns]
    def numeric(d):
        return d[feat_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
    Xraw_tr, Xraw_te = numeric(train_df), numeric(test_df)

    def fit_xgb(X, y):
        neg, pos = int((y == 0).sum()), int((y == 1).sum())
        return train_xgb(X, y, scale_pos_weight=neg / max(pos, 1), random_state=args.seed)
    def fit_pfn(X, y):
        return train_tabpfn(X, y, random_state=args.seed, device=device)

    def subset(y, n, seed, balanced=False, n_pos=None, n_neg=None):
        """Stratified (prior-preserving) or class-balanced subset of indices."""
        r = np.random.default_rng(seed); idx = []
        if n_pos is not None:
            for c, k in ((1, n_pos), (0, n_neg)):
                ci = np.where(y == c)[0]; idx.append(r.choice(ci, size=min(k, len(ci)), replace=False))
            return np.sort(np.concatenate(idx))
        if n >= len(y):
            return np.arange(len(y))
        for c in np.unique(y):
            ci = np.where(y == c)[0]
            k = n // 2 if balanced else max(1, round(n * len(ci) / len(y)))
            idx.append(r.choice(ci, size=min(k, len(ci)), replace=False))
        return np.sort(np.concatenate(idx))

    # ── OOF: regime-aware folds, per-fold imputation, NaN where no OOF exists ─
    def oof(Xraw, y, fit_fn, label, groups=None, times=None, impute=True):
        P = np.full((len(y), 2), np.nan, dtype=np.float32)
        for k, (tr, va) in enumerate(oof_folds(split_policy, Xraw, y, groups, args.folds, args.seed, times)):
            if len(np.unique(y[tr])) < 2:
                print(f"    {label} fold {k+1}: single-class train fold, no OOF for its val block"); continue
            if impute:
                imp = SimpleImputer(strategy="median").fit(Xraw[tr])
                Xt, Xv = imp.transform(Xraw[tr]), imp.transform(Xraw[va])
            else:
                Xt, Xv = Xraw[tr], Xraw[va]
            P[va] = fit_fn(Xt, y[tr]).predict_proba(Xv)
            print(f"    {label} OOF fold {k+1}/{args.folds} (train {len(tr)}, val {len(va)})")
        return P

    imp_full = SimpleImputer(strategy="median").fit(Xraw_tr)
    Xtr, Xte = imp_full.transform(Xraw_tr), imp_full.transform(Xraw_te)

    print("\n[XGB] full train, complete test"); xgb_oof = oof(Xraw_tr, y_tr, fit_xgb, "XGB", groups_tr, times_tr)
    xgb_te = fit_xgb(Xtr, y_tr).predict_proba(Xte).astype(np.float32)
    # bases: name -> (oof on its train subset, test scores, train-subset idx, test idx scored, exposure)
    full_te = np.arange(len(y_te))
    bases = {"XGB": (xgb_oof, xgb_te, np.arange(len(y_tr)), full_te, 1.0)}

    exp_te = full_te
    if args.expensive_test_rows and args.expensive_test_rows < len(y_te):
        exp_te = np.sort(rng.choice(len(y_te), size=args.expensive_test_rows, replace=False))
    exposure = len(exp_te) / len(y_te)

    if not args.no_tabpfn:
        pi = subset(y_tr, args.tabpfn_rows, args.seed)
        print(f"\n[TabPFN] train subset {len(pi)} (pos {int(y_tr[pi].sum())}); test rows {len(exp_te)} (uniform)")
        g = groups_tr[pi] if groups_tr is not None else None
        pfn_oof = oof(Xraw_tr[pi], y_tr[pi], fit_pfn, "TabPFN", g, times_tr[pi])
        pfn_te = fit_pfn(Xtr[pi], y_tr[pi]).predict_proba(Xte[exp_te]).astype(np.float32)
        bases["TabPFN"] = (pfn_oof, pfn_te, pi, exp_te, exposure)

    if not args.no_bert:
        bi = subset(y_tr, args.bert_rows, args.seed)
        tr_text_b = [format_network_row_ctu13(r) for _, r in train_df.iloc[bi].iterrows()]
        te_text_b = [format_network_row_ctu13(r) for _, r in test_df.iloc[exp_te].iterrows()]
        fname = (f"modernbert_ctu13_{args.split}_{args.background}_s{args.seed}_"
                 f"{texts_sha(tr_text_b)}_{texts_sha(te_text_b)}.npz")
        print(f"\n[XGB+BERT] train subset {len(bi)}; encoding {len(te_text_b)} test rows -> {fname}")
        Xb_tr, Xb_te = build_modernbert_features(tr_text_b, te_text_b, cache_dir=CACHE_DIR, cache_filename=fname)
        g = groups_tr[bi] if groups_tr is not None else None
        bxgb_oof = oof(Xb_tr, y_tr[bi], fit_xgb, "XGB+BERT", g, times_tr[bi], impute=False)
        bxgb_te = fit_xgb(Xb_tr, y_tr[bi]).predict_proba(Xb_te).astype(np.float32)
        bases["XGB+BERT"] = (bxgb_oof, bxgb_te, bi, exp_te, exposure)

    # ── Metrics ──────────────────────────────────────────────────────────────
    def metrics(y, preds, prob1):
        both = len(set(y.tolist())) > 1
        return {"aucpr": float(average_precision_score(y, prob1)) if both else None,
                "rocauc": float(roc_auc_score(y, prob1)) if both else None,
                "accuracy": float((np.asarray(preds) == y).mean())}

    rows, operational = {}, {}
    for name, (o, t, tidx, teidx, expo) in bases.items():
        y_eval = y_te[teidx]; m = metrics(y_eval, t[:, 1] >= 0.5, t[:, 1])
        m["coverage_test"] = 1.0; m["n_evaluated"] = int(len(teidx)); m["representative_test"] = True
        m["test_subsample"] = "none" if expo == 1.0 else f"uniform ({expo:.3f})"
        rows[f"{name} Only"] = m
        ok = ~np.isnan(o[:, 1])
        operational[f"{name} Only"] = operational_report(
            y_eval, t[:, 1], y_train=y_tr[tidx][ok], score_train=o[ok, 1], hours=hours,
            exposure_fraction=expo, priors=args.priors)

    # ── LLMs: class-balanced test subset, projected to the observed prior ────
    llm_te_idx = subset(y_te, None, args.seed, n_pos=args.llm_test_pos, n_neg=args.llm_test_neg)
    llm_tr_idx = subset(y_tr, args.llm_train_rows, args.seed, balanced=True)
    tr_text_l = [format_network_row_ctu13(r) for _, r in train_df.iloc[llm_tr_idx].iterrows()]
    te_text_l = [format_network_row_ctu13(r) for _, r in test_df.iloc[llm_te_idx].iterrows()]
    yte_l = y_te[llm_te_idx]
    print(f"\n[LLM] balanced subsets: test {len(llm_te_idx)} (pos {int(yte_l.sum())}, neg {int((yte_l==0).sum())}), "
          f"meta-train {len(llm_tr_idx)}; cache_only={not args.allow_llm_calls}")

    def llm_feats(texts, model):
        probs = np.full((len(texts), 2), 0.5); preds = np.zeros(len(texts), int); mask = np.ones(len(texts), bool)
        for i, t in enumerate(texts):
            r = classify_with_cache(t, model, cache, prompt, "ollama", cache_only=not args.allow_llm_calls)
            if r is None:
                mask[i] = False; continue
            pred, conf = r; p1 = conf if pred == 1 else 1.0 - conf
            probs[i] = [1.0 - p1, p1]; preds[i] = pred
        return probs, preds, mask

    def make_stage2(kind):
        if kind == "logreg":
            return LogisticRegression(random_state=args.seed, max_iter=1000)
        return XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                             random_state=args.seed, eval_metric="logloss", verbosity=0)

    def subset_row(y, preds, prob1, cov, n, extra=None):
        m = metrics(y, preds, prob1)
        m["aucpr_subset"] = m.pop("aucpr"); m["aucpr"] = None
        m["representative_test"] = False; m["evaluation_prior"] = float(y.mean())
        m["subset_n"] = int(len(y)); m["coverage_test"] = float(cov); m["n_evaluated"] = int(n)
        m["note"] = "class-balanced LLM test subset; natural-prior precision only via operational[...]['projected']"
        if extra: m.update(extra)
        return m

    def op_subset(y, prob1, y_train=None, s_train=None):
        return operational_report(y, prob1, y_train=y_train, score_train=s_train, hours=hours,
                                  exposure_fraction=len(y) / len(y_te),   # only used for per-hour projections
                                  priors=list(args.priors) + [float(y_te.mean())],
                                  representative_test=False, evaluation_prior=float(y.mean()))

    # XGB on the same balanced subset, for a like-for-like comparison with the LLM rows
    rows["XGB Only (LLM subset)"] = subset_row(yte_l, xgb_te[llm_te_idx, 1] >= 0.5, xgb_te[llm_te_idx, 1], 1.0, len(yte_l))
    operational["XGB Only (LLM subset)"] = op_subset(yte_l, xgb_te[llm_te_idx, 1], y_tr, xgb_oof[:, 1])

    for model in args.models:
        ltr, _, mtr = llm_feats(tr_text_l, model)
        lte, lpred, mte = llm_feats(te_text_l, model)
        if args.allow_llm_calls:
            save_cache_atomic(cache)
        cov_te, n_te = float(mte.mean()), int(mte.sum())
        print(f"  LLM {model}: train cov {mtr.mean():.0%}, test cov {cov_te:.0%} (n={n_te})")
        if n_te < 20 or len(set(yte_l[mte].tolist())) < 2:
            print(f"  [skip] {model}: insufficient coverage"); continue
        rows[f"{model} | LLM Only"] = subset_row(yte_l[mte], lpred[mte], lte[mte, 1], cov_te, n_te)
        operational[f"{model} | LLM Only"] = op_subset(yte_l[mte], lte[mte, 1])
        # stacks: OOF base features on the meta-train subset (+) LLM; rows lacking OOF are dropped
        ytr_s = y_tr[llm_tr_idx][mtr]
        for bname, (o, t, bidx, teidx, _) in bases.items():
            if not np.array_equal(teidx, full_te):
                t_full = np.full((len(y_te), 2), np.nan, dtype=np.float32); t_full[teidx] = t
            else:
                t_full = t
            pos = {int(i): k for k, i in enumerate(bidx)}
            sel = llm_tr_idx[mtr]
            keep = np.array([int(i) in pos and not np.isnan(o[pos[int(i)], 1]) for i in sel])
            te_keep = ~np.isnan(t_full[llm_te_idx][mte, 1])
            if keep.sum() < 50 or len(set(ytr_s[keep].tolist())) < 2 or te_keep.sum() < 20:
                continue
            o_rows = o[[pos[int(i)] for i in sel[keep]]]
            for s2 in args.stage2:
                mdl = make_stage2(s2); mdl.fit(np.hstack([o_rows, ltr[mtr][keep]]), ytr_s[keep])
                Xs2 = np.hstack([t_full[llm_te_idx][mte][te_keep], lte[mte][te_keep]])
                p1 = mdl.predict_proba(Xs2)[:, 1]; yk = yte_l[mte][te_keep]
                key = f"{bname}+{model} [{'LR' if s2 == 'logreg' else 'XGB'}]"
                rows[key] = subset_row(yk, p1 >= 0.5, p1, cov_te, int(te_keep.sum()))
                # stage-2 training scores (in-sample) only pick the threshold; flagged as such
                s_tr = mdl.predict_proba(np.hstack([o_rows, ltr[mtr][keep]]))[:, 1]
                operational[key] = op_subset(yk, p1, ytr_s[keep], s_tr)
                operational[key]["threshold_source"] = "stage-2 in-sample train scores"
        # Router: XGB gate tau=0.7, escalate to the LLM
        gate = xgb_te[llm_te_idx][mte, 1]; conf = np.maximum(gate, 1 - gate); routed = conf < 0.7
        p1 = np.where(routed, lte[mte, 1], gate)
        key = f"Router XGB->{model} (tau=0.7)"
        rows[key] = subset_row(yte_l[mte], p1 >= 0.5, p1, cov_te, n_te, {"routed_pct": float(100 * routed.mean())})
        operational[key] = op_subset(yte_l[mte], p1)

    # ── Print ────────────────────────────────────────────────────────────────
    print(f"\n{'config':34s} {'AUCPR':>7s} {'ROC':>7s} {'acc':>7s} {'cov':>5s}  note")
    for k, m in rows.items():
        a = m.get("aucpr") if m.get("representative_test", True) else m.get("aucpr_subset")
        r = m.get("rocauc")
        print(f"{k:34s} {'  n/a  ' if a is None else f'{a:7.4f}'} {'  n/a  ' if r is None else f'{r:7.4f}'} "
              f"{m['accuracy']:7.4f} {m.get('coverage_test', 1.0):5.2f}  "
              f"{'' if m.get('representative_test', True) else 'BALANCED SUBSET'}")
    print("\nOPERATIONAL (observed prior on complete test unless marked; projections labelled @prior):")
    for k, rep in operational.items():
        print(format_operational_table(k, rep))

    out = {"experiment": "11_ctu13_operational", "dataset": "ctu13", "split": args.split,
           "split_diag": split_diag, "background_policy": args.background,
           "prior_definition": "observed prevalence of the Botnet label among all captured flows; "
                               "Background treated as negative but unverified",
           "test_scenarios": args.test_scenarios if args.split == "scenario" else None,
           "smoke_sample_per_scenario": sps, "train_rows_used": int(len(train_df)),
           "train_rows_available": n_train_full, "folds": args.folds, "seed": args.seed,
           "stage2": args.stage2, "n_features": len(feat_cols),
           "n_train": int(len(train_df)), "n_test": int(len(test_df)), "test_complete": True,
           "train_prior": float(y_tr.mean()), "test_prior": float(y_te.mean()),
           "test_capture_hours": hours, "expensive_test_rows": int(len(exp_te)),
           "priors_projected": args.priors,
           "llm_subset": {"test_pos": args.llm_test_pos, "test_neg": args.llm_test_neg,
                          "train_rows": int(len(llm_tr_idx)), "balanced": True},
           "rows": rows, "operational": operational}
    suf = os.environ.get("EXPERIMENT_RESULT_SUFFIX", "")
    tag = f".{args.split}" + (f".{args.tag}" if args.tag else "")
    out_json = Path(__file__).with_name(f"11_ctu13_operational{tag}{suf}.result.json")
    out_json.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
