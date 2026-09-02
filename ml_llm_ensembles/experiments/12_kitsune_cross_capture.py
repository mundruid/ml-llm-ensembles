#!/usr/bin/env python3
# Experiment 12: within- and cross-capture Kitsune packet evaluation.
# Cross mode transfers between complete Mirai and SYN DoS captures. Within mode
# uses a flow-grouped split or one global temporal cutoff. The per-packet form
# preserves SYN DoS's observed 0.25% packet prevalence; five-tuple aggregation
# does not. Decoder subsets are marked representative_test=false.

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

from ml_llm_ensembles.utils.datasets import load_network_dataset
from ml_llm_ensembles.utils.models import train_xgb, train_tabpfn, build_modernbert_features, xgb_device
from ml_llm_ensembles.utils.prompts import (
    DOMAIN_PROMPTS, classify_with_cache, format_network_row_kitsune_pcap, KITSUNE_PCAP_FEATURE_COLS,
)
from ml_llm_ensembles.utils.splits import temporal_split, group_split, oof_folds
from ml_llm_ensembles.utils.evaluation import operational_report, format_operational_table

SEED = 42
CACHE_DIR = _ROOT / "results" / "cache"
CACHE_FILE = CACHE_DIR / "llm_cache.json"
DECODERS = ["mistral", "gemma3:12b", "llama3.2"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/kitsune")
    p.add_argument("--mode", choices=["cross", "within"], default="cross")
    p.add_argument("--within-split", choices=["temporal", "flow"], default="flow")
    p.add_argument("--capture", default="kitsune-syndos-pcap")
    p.add_argument("--reference", default="kitsune-mirai-pcap")
    p.add_argument("--models", nargs="*", default=DECODERS)
    p.add_argument("--allow-llm-calls", action="store_true")
    p.add_argument("--llm-test-pos", type=int, default=1000)
    p.add_argument("--llm-test-neg", type=int, default=5000)
    p.add_argument("--llm-train-rows", type=int, default=2000)
    p.add_argument("--tabpfn-rows", type=int, default=10_000)
    p.add_argument("--expensive-test-rows", type=int, default=300_000)
    p.add_argument("--bert-rows", type=int, default=0, help="0 = frozen-BERT base off")
    p.add_argument("--no-tabpfn", action="store_true")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--stage2", nargs="+", default=["logreg", "xgboost"], choices=["logreg", "xgboost"])
    p.add_argument("--priors", nargs="+", type=float, default=[0.01, 0.001])
    p.add_argument("--seed", type=int, default=SEED)
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


def load_pkts(name, data_dir):
    import pandas as pd
    pkl = CACHE_DIR / f"pkts_{name}.pkl"
    if pkl.exists():
        df = pd.read_pickle(pkl); print(f"  loaded cached packets {pkl.name}: {df.shape}")
    else:
        df = load_network_dataset(name, data_dir); CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_pickle(pkl)
    return df


def main():
    args = parse_args()
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from xgboost import XGBClassifier

    random.seed(args.seed); np.random.seed(args.seed)
    device = xgb_device()
    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    prompt = DOMAIN_PROMPTS["network"]
    lim = os.environ.get("EXPERIMENT_LIMIT")
    rng = np.random.default_rng(args.seed)

    def smoke_cap(df):
        """Smoke-only thinning: stratified WITHOUT replacement from the original
        frame, guaranteeing >=5 positives when available; never duplicates rows."""
        if not lim or int(lim) >= len(df):
            return df
        n = max(int(lim), 200); y = df["label"].values
        pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
        k_pos = min(len(pos), max(5, round(n * len(pos) / len(y))))
        k_neg = min(len(neg), n - k_pos)
        keep = np.sort(np.concatenate([rng.choice(pos, k_pos, replace=False), rng.choice(neg, k_neg, replace=False)]))
        return df.iloc[keep].reset_index(drop=True)

    # ── Runs: (name, train_df, test_df, diag, fold_policy) ───────────────────
    runs = []
    if args.mode == "cross":
        a = smoke_cap(load_pkts(args.reference, args.data_dir))
        b = smoke_cap(load_pkts(args.capture, args.data_dir))
        runs.append((f"{args.reference}_to_{args.capture}", a, b,
                     {"policy": "cross-capture", "train": args.reference, "test": args.capture}, "group"))
        runs.append((f"{args.capture}_to_{args.reference}", b, a,
                     {"policy": "cross-capture", "train": args.capture, "test": args.reference}, "group"))
    else:
        df = smoke_cap(load_pkts(args.capture, args.data_dir))
        if args.within_split == "temporal":
            df = df.sort_values("time", kind="stable").reset_index(drop=True)
            tr, te, diag = temporal_split(df, "time", 0.2); pol = "temporal"
        else:
            tr, te, diag = group_split(df, "flow_id", 0.2, args.seed); pol = "group"
        runs.append((f"{args.capture}_within_{args.within_split}", df.iloc[tr].reset_index(drop=True),
                     df.iloc[te].reset_index(drop=True), diag, pol))

    cols = list(KITSUNE_PCAP_FEATURE_COLS)
    suf = os.environ.get("EXPERIMENT_RESULT_SUFFIX", "")
    written = []
    for run_name, train_df, test_df, diag, fold_policy in runs:
        y_tr, y_te = train_df["label"].values, test_df["label"].values
        hours = float((test_df["time"].max() - test_df["time"].min()) / 3600.0)
        print(f"\n===== {run_name}: train {len(train_df)} (prior {y_tr.mean():.4f}) | "
              f"test {len(test_df)} COMPLETE (prior {y_te.mean():.5f}, {hours:.2f} h) =====\n  {diag}")
        if y_tr.sum() == 0 or y_te.sum() == 0:
            note = f"degenerate: train pos {int(y_tr.sum())}, test pos {int(y_te.sum())}"
            print("  " + note)
            out_json = Path(__file__).with_name(f"12_kitsune_cross_capture.{run_name}{suf}.result.json")
            out_json.write_text(json.dumps({"experiment": "12_kitsune_cross_capture", "run": run_name,
                                            "split_diag": diag, "note": note, "n_test": int(len(test_df)),
                                            "test_prior": float(y_te.mean()), "rows": {}}, indent=2))
            written.append(out_json); continue
        groups_tr = train_df["flow_id"].values if fold_policy == "group" else None
        times_tr = train_df["time"].values

        def numeric(d):
            return d[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
        Xraw_tr, Xraw_te = numeric(train_df), numeric(test_df)

        def fit_xgb(X, y):
            neg, pos = int((y == 0).sum()), int((y == 1).sum())
            return train_xgb(X, y, scale_pos_weight=neg / max(pos, 1), random_state=args.seed)
        def fit_pfn(X, y):
            return train_tabpfn(X, y, random_state=args.seed, device=device)

        def subset(y, n, seed, balanced=False, n_pos=None, n_neg=None):
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

        def oof(Xraw, y, fit_fn, label, groups=None, times=None, impute=True):
            P = np.full((len(y), 2), np.nan, dtype=np.float32)
            for k, (tr, va) in enumerate(oof_folds(fold_policy, Xraw, y, groups, args.folds, args.seed, times)):
                if len(np.unique(y[tr])) < 2:
                    print(f"    {label} fold {k+1}: single-class, no OOF for its val block"); continue
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
        xgb_oof = oof(Xraw_tr, y_tr, fit_xgb, "XGB", groups_tr, times_tr)
        xgb_te = fit_xgb(Xtr, y_tr).predict_proba(Xte).astype(np.float32)
        full_te = np.arange(len(y_te))
        bases = {"XGB": (xgb_oof, xgb_te, np.arange(len(y_tr)), full_te, 1.0)}
        exp_te = full_te
        if args.expensive_test_rows and args.expensive_test_rows < len(y_te):
            exp_te = np.sort(rng.choice(len(y_te), size=args.expensive_test_rows, replace=False))
        exposure = len(exp_te) / len(y_te)
        if not args.no_tabpfn:
            pi = subset(y_tr, args.tabpfn_rows, args.seed)
            g = groups_tr[pi] if groups_tr is not None else None
            bases["TabPFN"] = (oof(Xraw_tr[pi], y_tr[pi], fit_pfn, "TabPFN", g, times_tr[pi]),
                               fit_pfn(Xtr[pi], y_tr[pi]).predict_proba(Xte[exp_te]).astype(np.float32), pi, exp_te, exposure)
        if args.bert_rows:
            bi = subset(y_tr, args.bert_rows, args.seed)
            tr_text_b = [format_network_row_kitsune_pcap(r) for _, r in train_df.iloc[bi].iterrows()]
            te_text_b = [format_network_row_kitsune_pcap(r) for _, r in test_df.iloc[exp_te].iterrows()]
            fname = f"modernbert_kitsune_{run_name}_s{args.seed}_{texts_sha(tr_text_b)}_{texts_sha(te_text_b)}.npz"
            Xb_tr, Xb_te = build_modernbert_features(tr_text_b, te_text_b, cache_dir=CACHE_DIR, cache_filename=fname)
            g = groups_tr[bi] if groups_tr is not None else None
            bases["XGB+BERT"] = (oof(Xb_tr, y_tr[bi], fit_xgb, "XGB+BERT", g, times_tr[bi], impute=False),
                                 fit_xgb(Xb_tr, y_tr[bi]).predict_proba(Xb_te).astype(np.float32), bi, exp_te, exposure)

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

        llm_te_idx = subset(y_te, None, args.seed, n_pos=args.llm_test_pos, n_neg=args.llm_test_neg)
        llm_tr_idx = subset(y_tr, args.llm_train_rows, args.seed, balanced=True)
        tr_text_l = [format_network_row_kitsune_pcap(r) for _, r in train_df.iloc[llm_tr_idx].iterrows()]
        te_text_l = [format_network_row_kitsune_pcap(r) for _, r in test_df.iloc[llm_te_idx].iterrows()]
        yte_l = y_te[llm_te_idx]

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
                                      exposure_fraction=len(y) / len(y_te),
                                      priors=list(args.priors) + [float(y_te.mean())],
                                      representative_test=False, evaluation_prior=float(y.mean()))

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
                    p1 = mdl.predict_proba(np.hstack([t_full[llm_te_idx][mte][te_keep], lte[mte][te_keep]]))[:, 1]
                    yk = yte_l[mte][te_keep]
                    key = f"{bname}+{model} [{'LR' if s2 == 'logreg' else 'XGB'}]"
                    rows[key] = subset_row(yk, p1 >= 0.5, p1, cov_te, int(te_keep.sum()))
                    s_tr = mdl.predict_proba(np.hstack([o_rows, ltr[mtr][keep]]))[:, 1]
                    operational[key] = op_subset(yk, p1, ytr_s[keep], s_tr)
                    operational[key]["threshold_source"] = "stage-2 in-sample train scores"
            gate = xgb_te[llm_te_idx][mte, 1]; conf = np.maximum(gate, 1 - gate); routed = conf < 0.7
            p1 = np.where(routed, lte[mte, 1], gate)
            key = f"Router XGB->{model} (tau=0.7)"
            rows[key] = subset_row(yte_l[mte], p1 >= 0.5, p1, cov_te, n_te, {"routed_pct": float(100 * routed.mean())})
            operational[key] = op_subset(yte_l[mte], p1)

        print(f"\n{'config':34s} {'AUCPR':>7s} {'ROC':>7s} {'acc':>7s} {'cov':>5s}  note")
        for k, m in rows.items():
            a = m.get("aucpr") if m.get("representative_test", True) else m.get("aucpr_subset")
            r = m.get("rocauc")
            print(f"{k:34s} {'  n/a  ' if a is None else f'{a:7.4f}'} {'  n/a  ' if r is None else f'{r:7.4f}'} "
                  f"{m['accuracy']:7.4f} {m.get('coverage_test', 1.0):5.2f}  "
                  f"{'' if m.get('representative_test', True) else 'BALANCED SUBSET'}")
        print("\nOPERATIONAL:")
        for k, rep in operational.items():
            print(format_operational_table(k, rep))

        out = {"experiment": "12_kitsune_cross_capture", "run": run_name, "mode": args.mode,
               "representation": "per-packet (12 named fields)", "split_diag": diag,
               "fold_policy": fold_policy, "folds": args.folds, "seed": args.seed, "stage2": args.stage2,
               "n_train": int(len(train_df)), "n_test": int(len(test_df)), "test_complete": True,
               "train_prior": float(y_tr.mean()), "test_prior": float(y_te.mean()),
               "test_capture_hours": hours, "expensive_test_rows": int(len(exp_te)),
               "priors_projected": args.priors,
               "llm_subset": {"test_pos": args.llm_test_pos, "test_neg": args.llm_test_neg,
                              "train_rows": int(len(llm_tr_idx)), "balanced": True},
               "rows": rows, "operational": operational}
        out_json = Path(__file__).with_name(f"12_kitsune_cross_capture.{run_name}{suf}.result.json")
        out_json.write_text(json.dumps(out, indent=2, default=float))
        written.append(out_json)
    for w in written:
        print(f"wrote {w}")


if __name__ == "__main__":
    main()
