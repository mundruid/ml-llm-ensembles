#!/usr/bin/env python3
# Experiment 13: binary phishing detection under hard-negative and source shift.
#
# RQ1 hard negatives: does training with spam/scam negatives reduce false
#     positives without an unacceptable loss of phishing recall?
# RQ2 source shift: do the effects persist on complete held-out source-partition
#     bundles (phishing D + ham E + spam/scam F)?
# RQ3 prior shift: given measured subtype-conditional rates, under which
#     hypothetical prevalences and negative mixtures is precision useful?
#
# CLAIM BOUNDARY. The evaluation preserves held-out messages and estimates
# phishing recall and subtype-specific false-positive rates. It then calculates
# the implications of those conditional rates under explicitly hypothetical
# phishing prevalences and negative-traffic compositions. No test set is
# resampled to manufacture a deployment prior.
#
# REGIMES
#   controlled  group-aware train / calibration / test partitions drawn from all
#               development sources (campaign/template groups never cross)
#   external    predeclared bundle rule: every (D, E, F) with D a phishing
#               source, E a ham source, F a spam/scam source, such that the
#               remaining sources still contain phishing, ham and spam/scam.
#               D, E, F touch nothing before final scoring.
# VARIANTS     raw text / provenance-stripped text (strip_provenance)
# CONDITIONS   A ham-only negatives; B matched-budget (same negative count,
#              fraction --hard-frac of ham replaced by spam/scam); C additive
#              (all ham + all spam/scam). Positives, evaluation rows, groups,
#              hyperparameters, seed, calibration and threshold policy fixed.
# MODELS       xgb (hand-crafted features), bert_lr (frozen ModernBERT-base
#              embeddings + logistic regression), and modernbert_ft (a fresh,
#              split-local fine-tune selected on calibration AUCPR).
#
# Decoder LLMs, routers and stacks are out of scope by design (small balanced
# subsets cannot resolve rare FPRs). Smoke mode uses --synthetic-smoke: a
# deterministic synthetic six-source corpus, XGB only, no downloads.
# =============================================================================

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_llm_ensembles.utils.datasets import strip_provenance
from ml_llm_ensembles.utils.features import build_phishing_email_feature_matrix
from ml_llm_ensembles.utils.models import train_xgb, build_modernbert_features
from ml_llm_ensembles.utils.phishing_taxonomy import (
    ANNOTATION_POLICY_VERSION, NEGATIVE_SUBTYPES, load_manifest, synthetic_corpus, dedup_and_group,
    partition_overlap_diag, ordered_hash_fingerprint,
)
from ml_llm_ensembles.utils.modernbert_ft import make_ft_backend, BASE_MODEL_DEFAULT, _sha_texts as ft_sha_texts
from ml_llm_ensembles.utils.operational_phishing import (
    DEFAULT_PIS, DEFAULT_SPAM_SHARES, conditional_rates, deployment_grid, group_bootstrap,
    mixture_weights, projection, PlattCalibrator, IsotonicCalibrator, calibration_metrics, select_thresholds,
)

SEED = 42
CACHE_DIR = _ROOT / "results" / "cache"
SPAM_LIKE = ["bulk_spam", "scam_fraud"]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="data/phishing-operational/manifest.csv")
    p.add_argument("--synthetic-smoke", action="store_true", help="deterministic synthetic corpus; no files")
    p.add_argument("--regimes", nargs="+", default=["controlled", "external"], choices=["controlled", "external"])
    p.add_argument("--variants", nargs="+", default=["raw", "stripped"], choices=["raw", "stripped"])
    p.add_argument("--models", nargs="+", default=["xgb", "bert_lr"],
                   choices=["xgb", "bert_lr", "modernbert_ft"])
    p.add_argument("--conditions", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    p.add_argument("--bundles", default="all", help="'all', 'first', or a bundle_id")
    p.add_argument("--hard-frac", type=float, default=0.5, help="condition B: fraction of ham replaced")
    p.add_argument("--min-source-rows", type=int, default=50, help="held-out source must have >= this many rows")
    p.add_argument("--controlled-split", nargs=3, type=float, default=[0.6, 0.2, 0.2], metavar=("TR", "CAL", "TE"))
    p.add_argument("--calibration", choices=["none", "platt", "isotonic"], default="platt")
    p.add_argument("--role-purity", type=float, default=0.9,
                   help="min fraction of a source's rows in its role subtype(s) to be held out as that role")
    p.add_argument("--pis", nargs="+", type=float, default=DEFAULT_PIS)
    p.add_argument("--spam-shares", nargs="+", type=float, default=DEFAULT_SPAM_SHARES)
    p.add_argument("--spam-split-bulk", type=float, default=0.5)
    p.add_argument("--target-precision", type=float, default=0.9)
    p.add_argument("--target-recall", type=float, default=0.9)
    p.add_argument("--pi-for-threshold", type=float, default=0.01)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--random-reference", action="store_true", help="also run the leakage-prone random row split")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--seed", type=int, default=SEED)
    # ── modernbert_ft: split-local full fine-tuning from answerdotai/ModernBERT-base ──
    # Conservative A100 defaults: bs 32 @ len 512 bf16 fits ModernBERT-base comfortably;
    # 3 epochs with patience 2 matches experiment 01's budget.
    ft = p.add_argument_group("modernbert_ft")
    ft.add_argument("--ft-conditions", nargs="+", default=None, choices=["A", "B", "C"],
                    help="hard-negative conditions for modernbert_ft; must be a subset of "
                         "--conditions (default: same as --conditions)")
    ft.add_argument("--ft-scope", choices=["full", "staged", "minimal"], default="staged",
                    help="compute scope for fine-tuning: full = every regime/variant/bundle; "
                         "staged = controlled (both variants, all ft-conditions) + the first "
                         "(predeclared, rule-ordered) external bundle; minimal = controlled raw A/B "
                         "+ first external bundle raw A/B")
    ft.add_argument("--ft-backend", choices=["real", "fake"], default="real",
                    help="'fake' = deterministic test backend, no downloads (forced under --synthetic-smoke)")
    ft.add_argument("--ft-base-model", default=BASE_MODEL_DEFAULT)
    ft.add_argument("--ft-epochs", type=int, default=3)
    ft.add_argument("--ft-batch-size", type=int, default=32)
    ft.add_argument("--ft-eval-batch-size", type=int, default=64)
    ft.add_argument("--ft-learning-rate", type=float, default=2e-5)
    ft.add_argument("--ft-weight-decay", type=float, default=0.01)
    ft.add_argument("--ft-max-length", type=int, default=512)
    ft.add_argument("--ft-gradient-accumulation", type=int, default=1)
    ft.add_argument("--ft-patience", type=int, default=2, help="early stop after N non-improving epochs (cal AUCPR)")
    ft.add_argument("--ft-output-dir", type=Path, default=_ROOT / "models" / "exp13-modernbert-ft")
    ft.add_argument("--ft-resume", action="store_true",
                    help="reuse a checkpoint ONLY when its stored metadata matches this run's identity exactly")
    ft.add_argument("--ft-controlled-only", action="store_true")
    ft.add_argument("--ft-bundles", default="all", help="'all', 'first', or a bundle_id (fine-tuning only)")
    return p


def parse_args():
    return build_parser().parse_args()


def ft_in_scope(args, regime, variant, cname, bundle):
    """Compute-scope gate for modernbert_ft (full / staged / minimal; see --ft-scope).
    The 'first' bundle is predeclared by the deterministic bundle-enumeration order,
    not chosen after looking at results. FT never runs on the leaky random reference."""
    if regime == "controlled-random":
        return False
    rank = (bundle or {}).get("_rank", 0)
    explicit = args.ft_bundles not in ("all", "first")     # a named bundle_id
    if regime == "external":
        if args.ft_controlled_only:
            return False
        if explicit and bundle["bundle_id"] != args.ft_bundles:
            return False
        if args.ft_bundles == "first" and rank != 0:
            return False
    if args.ft_scope == "full":
        return True
    # An EXPLICITLY named bundle overrides the staged/minimal first-bundle gate
    # (you asked for that bundle by id); the variant/condition limits of
    # 'minimal' still apply.
    bundle_ok = regime == "controlled" or rank == 0 or (regime == "external" and explicit)
    if args.ft_scope == "staged":
        return bundle_ok
    # minimal
    if variant != "raw" or cname not in ("A", "B"):
        return False
    return bundle_ok


def sha_texts(texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode()); h.update(b"\0")
    return h.hexdigest()[:16]


# ── Source roles and bundle rule ──────────────────────────────────────────────
# Hold-out-eligible roles. The spam/scam role requires KNOWN spam or scam:
# a source that is mostly unknown_nonphishing gets the development-only role
# 'unknown_negative' (usable as a hard negative in training, never as the
# external spam/scam source F, where it would leave the spam/scam FPRs and most
# prevalence projections unavailable).
SOURCE_FAMILY = {
    "nazario_early": "nazario", "nazario_recent": "nazario",
    "spamassassin_easy_ham": "spamassassin", "spamassassin_easy_ham_2": "spamassassin",
    "spamassassin_hard_ham": "spamassassin", "spamassassin_spam": "spamassassin",
    "spamassassin_spam_2": "spamassassin", "enron": "enron",
}


def source_family(src: str) -> str:
    """Corpus family: partitions of one collection share collection processes and
    fingerprints, so a sibling-partition holdout is WITHIN-FAMILY generalization,
    not unseen-corpus generalization. Unknown sources are their own family."""
    if src in SOURCE_FAMILY:
        return SOURCE_FAMILY[src]
    for fam in ("synth_ham", "synth_phish"):
        if src.startswith(fam):
            return fam
    return src


ROLE_SUBTYPES = {"phishing": {"phishing"}, "ham": {"ham"},
                 "spam_like": {"bulk_spam", "scam_fraud"},
                 "unknown_negative": {"unknown_nonphishing"}}
HOLDOUT_ROLES = ("phishing", "ham", "spam_like")


def source_roles(df, purity=0.9):
    """Explicit role per source_corpus, gated by a purity threshold: a source is
    eligible as role r only if >= purity of its rows carry r's subtype(s).
    Anything below purity for every role is 'mixed'. Only phishing / ham /
    spam_like sources can be held out; 'mixed' and 'unknown_negative' sources are
    development-only. Returns (roles, composition). Mixed corpora
    should be listed in the manifest as one asset per partition (e.g.
    spamassassin_easy_ham / spamassassin_spam) so each source has one role."""
    roles, comp = {}, {}
    for src, g in df.groupby("source_corpus"):
        fr = g["subtype"].value_counts(normalize=True).to_dict()
        comp[src] = {"n": int(len(g)), "subtype_counts": g["subtype"].value_counts().to_dict(),
                     "subtype_fractions": {k: round(float(v), 4) for k, v in fr.items()}}
        role = "mixed"
        for r, subs in ROLE_SUBTYPES.items():
            if sum(fr.get(x, 0.0) for x in subs) >= purity:
                role = r; break
        roles[src] = role
        comp[src]["role"] = role
    return roles, comp


def enumerate_bundles(df, roles, comp, min_rows, purity):
    """Predeclared rule: all (D,E,F) triples of distinct PURE sources with roles
    phishing/ham/spam_like, each with >= min_rows, such that the development
    remainder still holds at least one source of each role. Mixed (sub-purity)
    sources are excluded from hold-out and stay in development. The actual
    subtype composition of every held-out source is recorded in the bundle."""
    counts = df["source_corpus"].value_counts()
    by_role = {r: sorted(s for s, rr in roles.items() if rr == r and counts[s] >= min_rows)
               for r in HOLDOUT_ROLES}
    bundles = []
    for D, E, F in product(by_role["phishing"], by_role["ham"], by_role["spam_like"]):
        dev = [s for s in roles if s not in (D, E, F)]
        dev_roles = {roles[s] for s in dev}
        if {"phishing", "ham", "spam_like"} <= dev_roles:
            bid = "b_" + hashlib.sha256(f"{D}|{E}|{F}".encode()).hexdigest()[:8]
            dev_fams = {source_family(x) for x in dev}
            tiers = {role: ("family-disjoint" if source_family(src) not in dev_fams
                            else "within-family-partition")
                     for role, src in (("phishing", D), ("ham", E), ("spam_scam", F))}
            bundles.append({"bundle_id": bid, "phishing_source": D, "ham_source": E, "spam_scam_source": F,
                            "development_sources": dev, "role_purity": purity,
                            "holdout_tier_by_role": tiers,
                            "fully_family_disjoint": all(v == "family-disjoint" for v in tiers.values()),
                            "held_out_composition": {D: comp[D], E: comp[E], F: comp[F]}})
    return bundles


# ── Partitions ────────────────────────────────────────────────────────────────
def required_subtypes(df, spam_shares):
    """Subtypes every partition must contain: phishing always; ham whenever any
    requested mixture gives ham weight; each spam-like subtype that exists in the
    corpus whenever any requested mixture gives spam weight."""
    present = set(df["subtype"].unique())
    req = {"phishing"}
    if any(s < 1.0 for s in spam_shares):
        req |= {"ham"} & present
    if any(s > 0.0 for s in spam_shares):
        req |= {"bulk_spam", "scam_fraud"} & present
    return req


def group_partition(df, fracs, seed, required=None, max_salt=20):
    """Group-aware split into len(fracs) partitions by hashing the group key with
    a salted seed (deterministic, order-independent). If `required` subtypes are
    given, salts seed..seed+max_salt-1 are tried until every partition contains
    every required subtype; otherwise the run FAILS CLEARLY rather than emitting
    partially unavailable results. Returns (list of index arrays, salt used)."""
    keys = df["campaign_or_template_group"].astype(str).values
    st = df["subtype"].values
    edges = np.cumsum(fracs) / sum(fracs)
    for salt in range(max_salt):
        u = np.array([int(hashlib.sha256(f"{seed + salt}:{k}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                      for k in keys])
        part = np.searchsorted(edges, u, side="right")
        parts = [np.flatnonzero(part == i) for i in range(len(fracs))]
        if required is None or all(required <= set(st[p]) for p in parts):
            return parts, salt
    sys.exit(f"ERROR: no group-aware partition (salts {seed}..{seed + max_salt - 1}) gives every partition "
             f"the required subtypes {sorted(required)}; the corpus is too thin or too template-concentrated "
             f"for the requested mixtures: add data or reduce --spam-shares coverage.")


def hard_negative_conditions(train, seed, hard_frac, which):
    """Return {condition: (index array into train, description)} holding the
    phishing positives fixed. Never resamples with replacement."""
    rng = np.random.default_rng(seed)
    st = train["subtype"].values
    pos = np.flatnonzero(train["label_binary"].values == 1)
    ham = np.flatnonzero(st == "ham"); hard = np.flatnonzero(np.isin(st, SPAM_LIKE + ["unknown_nonphishing"]))
    out = {}
    if "A" in which:
        out["A"] = (np.sort(np.concatenate([pos, ham])), {"negative_budget_policy": "ham only", "n_pos": len(pos),
                    "n_ham": len(ham), "n_hard": 0, "n_neg_total": len(ham)})
    if "B" in which:
        k = min(int(round(hard_frac * len(ham))), len(hard))
        keep_ham = rng.choice(ham, size=len(ham) - k, replace=False) if k else ham
        pick_hard = rng.choice(hard, size=k, replace=False) if k else np.array([], int)
        out["B"] = (np.sort(np.concatenate([pos, keep_ham, pick_hard])),
                    {"negative_budget_policy": f"matched budget: {k} of {len(ham)} ham replaced by spam/scam "
                                               f"(requested fraction {hard_frac}, available hard {len(hard)})",
                     "n_pos": len(pos), "n_ham": len(keep_ham), "n_hard": int(k), "n_neg_total": len(keep_ham) + int(k),
                     "applicable": bool(k > 0)})
    if "C" in which:
        out["C"] = (np.sort(np.concatenate([pos, ham, hard])),
                    {"negative_budget_policy": "additive: all ham + all spam/scam (more data AND more diversity; "
                                               "not a pure diversity effect)", "n_pos": len(pos), "n_ham": len(ham),
                     "n_hard": len(hard), "n_neg_total": len(ham) + len(hard), "applicable": bool(len(hard) > 0)})
    return out


# ── Models ────────────────────────────────────────────────────────────────────
class XGBText:
    name = "xgb"
    def __init__(self, seed): self.seed = seed
    def features(self, texts):
        return build_phishing_email_feature_matrix(pd.DataFrame({"text": texts})).values.astype(float)
    def fit(self, X, y):
        neg, pos = int((y == 0).sum()), int((y == 1).sum())
        self.m = train_xgb(X, y, scale_pos_weight=neg / max(pos, 1), random_state=self.seed); return self
    def score(self, X): return self.m.predict_proba(X)[:, 1]


class BertLR:
    name = "bert_lr"
    def __init__(self, seed, cache_tag): self.seed, self.cache_tag = seed, cache_tag
    def features(self, texts):
        fname = f"modernbert_phishop_{self.cache_tag}_{sha_texts(texts)}_{len(texts)}.npz"
        X, _ = build_modernbert_features(list(texts), list(texts[:1]), cache_dir=CACHE_DIR, cache_filename=fname)
        return X
    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=self.seed).fit(X, y); return self
    def score(self, X): return self.m.predict_proba(X)[:, 1]


# ── Corpus-artifact diagnostic ────────────────────────────────────────────────
def corpus_artifact_diagnostic(train_texts, train_src, test_texts, test_src, seed, char_only=False):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, recall_score
    vec = (TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200_000) if char_only
           else TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000))
    Xtr = vec.fit_transform(train_texts); Xte = vec.transform(test_texts)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed).fit(Xtr, train_src)
    pred = clf.predict(Xte); labels = sorted(set(train_src) | set(test_src))
    return {"features": "char_wb 3-5" if char_only else "word 1-2gram", "class_weight": "balanced",
            "macro_f1": float(f1_score(test_src, pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(test_src, pred)),
            "per_source_recall": dict(zip(labels, map(float, recall_score(test_src, pred, labels=labels, average=None, zero_division=0)))),
            "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(test_src, pred, labels=labels).tolist()},
            "n_train": int(len(train_texts)), "n_test": int(len(test_texts))}


# ── One (regime, variant, bundle) run ─────────────────────────────────────────
def corpus_artifact_variant_diagnostic(d, seed):
    """Source-corpus predictability diagnostic on its OWN dedicated group-aware
    70/30 split of the full (variant-transformed) corpus, so every source appears
    on both sides. Independent of the phishing model's partitions; never enters
    phishing-model rankings."""
    (tr_i, te_i), salt = group_partition(d, [0.7, 0.3], seed + 1000, required=None)
    tr, te = d.iloc[tr_i], d.iloc[te_i]
    both = sorted(set(tr["source_corpus"]) & set(te["source_corpus"]))
    dropped = sorted((set(d["source_corpus"]) - set(both)))
    tr = tr[tr["source_corpus"].isin(both)]; te = te[te["source_corpus"].isin(both)]
    return {"split": "dedicated group-aware 70/30 split; every source on both sides; "
                     "independent of the phishing model's train/calibration/test partitions",
            "salt": salt, "sources_evaluated": both, "sources_dropped_missing_one_side": dropped,
            "word": corpus_artifact_diagnostic(tr["text"].tolist(), tr["source_corpus"].tolist(),
                                               te["text"].tolist(), te["source_corpus"].tolist(), seed),
            "char": corpus_artifact_diagnostic(tr["text"].tolist(), tr["source_corpus"].tolist(),
                                               te["text"].tolist(), te["source_corpus"].tolist(), seed, char_only=True),
            "note": "high source predictability shows corpus fingerprints; connect to phishing metrics via the "
                    "raw/stripped variants and external bundles, not by itself"}


def run_one(args, df, corpus_report, dedup_diag, regime, variant, bundle, out_dir, plot_rows, cad):
    from sklearn.metrics import average_precision_score, roc_auc_score
    seed = args.seed
    d = df.copy()
    if variant == "stripped":
        d["text"] = d["text"].map(strip_provenance)
    req = required_subtypes(d, args.spam_shares)

    # ── partitions ───────────────────────────────────────────────────────────
    if regime == "controlled":
        (tr_i, cal_i, te_i), salt = group_partition(d, args.controlled_split, seed, required=req)
        train, cal, test = d.iloc[tr_i], d.iloc[cal_i], d.iloc[te_i]
        split_policy = {"policy": "group-aware hashed partition on campaign_or_template_group",
                        "fractions": args.controlled_split, "seed": seed, "salt": salt,
                        "required_subtypes_per_partition": sorted(req)}
        tag = f"controlled.{variant}"
    elif regime == "controlled-random":
        # leakage-prone ROW split kept only as an explicit reference; groups may cross
        rng_ = np.random.default_rng(seed); u = rng_.uniform(size=len(d))
        edges = np.cumsum(args.controlled_split) / sum(args.controlled_split)
        part = np.searchsorted(edges, u, side="right")
        train, cal, test = d[part == 0], d[part == 1], d[part == 2]
        split_policy = {"policy": "RANDOM ROW split (leakage-prone reference; groups may cross)", "seed": seed,
                        "leakage_prone": True}
        tag = f"controlled-random.{variant}"
    else:
        held = {bundle["phishing_source"], bundle["ham_source"], bundle["spam_scam_source"]}
        dev = d[~d["source_corpus"].isin(held)]; test = d[d["source_corpus"].isin(held)]
        # Near-duplicate control across sources: a campaign/template group that spans a
        # development source AND a held-out source would leak template identity into the
        # external test. Conservative fix: DROP those rows from development (the held-out
        # test stays complete) and record exactly what was dropped.
        shared = set(dev["campaign_or_template_group"]) & set(test["campaign_or_template_group"])
        dropped_by_src = {}
        if shared:
            m = dev["campaign_or_template_group"].isin(shared)
            dropped_by_src = dev.loc[m, "source_corpus"].value_counts().to_dict()
            dev = dev[~m]
        req_dev = required_subtypes(dev, args.spam_shares)   # dev-present subtypes only
        (tr_i, cal_i), salt = group_partition(dev, [0.75, 0.25], seed, required=req_dev)
        train, cal = dev.iloc[tr_i], dev.iloc[cal_i]
        split_policy = {"policy": "external bundle: development sources split group-aware 75/25 into "
                                  "train/calibration; test = complete held-out sources D+E+F", "seed": seed,
                        "salt": salt, "required_subtypes_dev": sorted(req_dev),
                        "cross_source_template_groups_dropped_from_dev": {
                            "n_groups": len(shared), "n_rows": int(sum(dropped_by_src.values())),
                            "by_source": dropped_by_src}, **bundle}
        tag = f"external.{bundle['bundle_id']}.{variant}"
    train, cal, test = (x.reset_index(drop=True) for x in (train, cal, test))
    overlap = partition_overlap_diag({"train": train, "calibration": cal, "test": test})
    if not overlap["clean"] and regime != "controlled-random":
        raise RuntimeError(f"partition overlap detected: {overlap}")
    if regime == "controlled-random" and any(overlap["hash_overlap"].values()):
        raise RuntimeError("exact duplicates crossed partitions even in the random reference (dedup failed)")
    if regime == "external":
        assert not (set(train["source_corpus"]) | set(cal["source_corpus"])) & set(test["source_corpus"])
    def counts(x):
        return {"n": int(len(x)), "subtype": x["subtype"].value_counts().to_dict(),
                "source": x["source_corpus"].value_counts().to_dict(),
                "groups": int(x["campaign_or_template_group"].nunique()),
                "mapping_confidence": x["mapping_confidence"].value_counts().to_dict()}
    part_counts = {"train": counts(train), "calibration": counts(cal), "test": counts(test)}
    y_cal, st_cal = cal["label_binary"].values, cal["subtype"].values
    y_te, st_te, g_te = test["label_binary"].values, test["subtype"].values, test["campaign_or_template_group"].values
    if y_te.sum() == 0 or y_cal.sum() == 0:
        print(f"  [{tag}] degenerate: no positives in test/calibration; skipped"); return None
    print(f"\n===== {tag}: train {len(train)} | cal {len(cal)} | test {len(test)} "
          f"(test subtypes {part_counts['test']['subtype']}) =====")

    conds = hard_negative_conditions(train, seed, args.hard_frac, args.conditions)
    qs = [mixture_weights(s, args.spam_split_bulk) for s in args.spam_shares]
    results_models = {}
    rows = {}
    def scored_conditions(mname):
        """Yield (cname, cdesc, s_cal_raw, s_te_raw, ft_extra) per applicable condition.
        Fitting sees TRAIN rows only; modernbert_ft additionally sees CALIBRATION rows
        for epoch selection inside fit(). TEST rows are scored here strictly AFTER
        fit()/selection and are never passed into any fit or selection step."""
        if mname == "modernbert_ft":
            cal_texts = cal["text"].tolist(); te_texts = test["text"].tolist()
            ft_conds = args.ft_conditions or list(conds.keys())
            for cname, (idx, cdesc) in conds.items():
                if cname not in ft_conds or not ft_in_scope(args, regime, variant, cname, bundle):
                    continue
                if cdesc.get("applicable", True) is False:
                    results_models[f"{mname}|{cname}"] = {"condition": cname, **cdesc,
                                                          "skipped": "no hard negatives in training partition"}
                    continue
                tr_texts = train["text"].values[idx].tolist()
                y_tr_c = train["label_binary"].values[idx]
                hp = {"epochs": args.ft_epochs, "batch_size": args.ft_batch_size,
                      "eval_batch_size": args.ft_eval_batch_size, "learning_rate": args.ft_learning_rate,
                      "weight_decay": args.ft_weight_decay, "max_length": args.ft_max_length,
                      "gradient_accumulation": args.ft_gradient_accumulation, "patience": args.ft_patience}
                ident = dict(regime=regime, bundle_id=bundle["bundle_id"] if bundle else None,
                             variant=variant, condition=cname, seed=seed,
                             policy_version=ANNOTATION_POLICY_VERSION,
                             train_fingerprint=ft_sha_texts(tr_texts),
                             cal_fingerprint=ft_sha_texts(cal_texts),
                             base_model=args.ft_base_model)
                backend = make_ft_backend(args.ft_backend, args.ft_output_dir, ident, hp, args.ft_resume)
                print(f"  [ft] {tag} {cname}: {args.ft_backend} backend -> {backend.name}")
                backend.fit(tr_texts, y_tr_c, cal_texts, y_cal)
                backend.attach_counts({
                    "train_counts": {"subtype": train.iloc[idx]["subtype"].value_counts().to_dict(),
                                     "source": train.iloc[idx]["source_corpus"].value_counts().to_dict()},
                    "cal_counts": part_counts["calibration"],
                    "partition_fingerprints": {"train_condition": ident["train_fingerprint"],
                                               "calibration": ident["cal_fingerprint"],
                                               "test": ordered_hash_fingerprint(test)}})
                t0 = time.time()
                s_cal_raw = backend.score(cal_texts)
                s_te_raw = backend.score(te_texts)      # test scored only after selection
                extra = {"base_model": args.ft_base_model, "backend": args.ft_backend,
                         "checkpoint": str(backend.dir), "identity": backend.name,
                         "selected_epoch": backend.meta.get("selected_epoch"),
                         "selection_metric": backend.meta.get("selection_metric"),
                         "calibration_aucpr_by_epoch": backend.meta.get("calibration_aucpr_by_epoch"),
                         "hyperparams": hp, "resumed": backend.resumed,
                         "train_seconds": backend.meta.get("train_seconds"),
                         "inference_seconds": round(time.time() - t0, 2),
                         "n_train": int(len(tr_texts)), "n_cal": int(len(cal_texts)),
                         "n_test": int(len(te_texts))}
                yield cname, cdesc, s_cal_raw, s_te_raw, extra
            return
        model = XGBText(seed) if mname == "xgb" else BertLR(seed, f"{tag}")
        # features are computed once per partition and indexed per condition
        Xtr_all = model.features(train["text"].tolist()); Xcal = model.features(cal["text"].tolist())
        Xte = model.features(test["text"].tolist())
        for cname, (idx, cdesc) in conds.items():
            if cdesc.get("applicable", True) is False:
                results_models[f"{mname}|{cname}"] = {"condition": cname, **cdesc, "skipped": "no hard negatives in training partition"}
                continue
            model.fit(Xtr_all[idx], train["label_binary"].values[idx])
            yield cname, cdesc, model.score(Xcal), model.score(Xte), None

    for mname in args.models:
        for cname, cdesc, s_cal_raw, s_te_raw, ft_extra in scored_conditions(mname):
            if args.calibration == "none":
                s_cal, s_te, calib = s_cal_raw, s_te_raw, {"method": "none"}
            else:
                C = (PlattCalibrator() if args.calibration == "platt" else IsotonicCalibrator()).fit(s_cal_raw, y_cal)
                s_cal, s_te = C(s_cal_raw), C(s_te_raw)
                calib = {"method": args.calibration, "fit_on": "calibration partition only",
                         "calibration_composition": part_counts["calibration"]["subtype"],
                         "before": calibration_metrics(y_cal, s_cal_raw), "after": calibration_metrics(y_cal, s_cal),
                         "note": "in-sample calibration metrics on the calibration partition; test metrics below"}
            # NOTE: for Platt fitted on cal, s_cal is in-sample for the calibrator only (thresholds still cal-only)
            thr = select_thresholds(y_cal, st_cal, s_cal, recall_target=args.target_recall,
                                    precision_target=args.target_precision, pi_for_precision=args.pi_for_threshold,
                                    q_for_precision=mixture_weights(0.5, args.spam_split_bulk))
            both = len(set(y_te.tolist())) > 1
            std = {"aucpr": float(average_precision_score(y_te, s_te)) if both else None,
                   "rocauc": float(roc_auc_score(y_te, s_te)) if both else None,
                   "test_prior_curated": float(y_te.mean()),
                   "calibration_test": calibration_metrics(y_te, s_te) if args.calibration != "none" else None}
            ops = {}
            for pname, pinfo in thr.items():
                if pname == "_meta" or not pinfo["attainable"]:
                    if pname != "_meta":
                        ops[pname] = {"attainable": False, "threshold": None, "policy": pinfo["policy"]}
                    continue
                t = pinfo["threshold"]
                rates = conditional_rates(y_te, st_te, s_te, t, g_te)
                grid = deployment_grid(rates, args.pis, args.spam_shares, args.spam_split_bulk, args.target_precision)
                boot = group_bootstrap(y_te, st_te, s_te, t, g_te, args.pis, qs, args.n_boot, seed)
                # observed-mixture self-consistency: projecting at the test's own prior and
                # its own negative composition must reproduce the measured precision
                neg = st_te != "phishing"; q_obs = {s: float((st_te[neg] == s).mean()) for s in NEGATIVE_SUBTYPES}
                pred = s_te >= t; tp = int((pred & (y_te == 1)).sum()); fp = int((pred & (y_te == 0)).sum())
                meas = tp / (tp + fp) if (tp + fp) else None
                p_obs = projection(rates, float(y_te.mean()), q_obs)
                if meas is not None and p_obs["available"] and abs(p_obs["precision_point"] - meas) > 1e-6:
                    raise AssertionError(f"observed-mixture self-check failed: {p_obs['precision_point']} vs {meas}")
                ops[pname] = {"threshold": float(t), "attainable": True, "policy": pinfo["policy"],
                              "conditional_rates": rates, "deployment_grid": grid, "uncertainty": boot,
                              "observed_mixture_check": {"q_observed": q_obs, "measured_precision": meas,
                                                         "projected_precision": p_obs.get("precision_point")}}
                for cell in grid:
                    p = cell["projection"]
                    plot_rows.append({"run": tag, "model": mname, "condition": cname, "threshold_policy": pname,
                                      "prior": cell["prior"], "spam_share": cell["spam_share"],
                                      "precision_point": p.get("precision_point"), "precision_conservative": p.get("precision_conservative"),
                                      "support_status": cell["support"]["support_status"],
                                      "ci_lo": boot["projected_precision_ci95"].get(f"{cell['prior']:g}|{args.spam_shares.index(cell['spam_share'])}", [None, None])[0],
                                      "ci_hi": boot["projected_precision_ci95"].get(f"{cell['prior']:g}|{args.spam_shares.index(cell['spam_share'])}", [None, None])[1]})
            results_models[f"{mname}|{cname}"] = {"model": mname, "condition": cname, **cdesc,
                                                  "calibration": calib, "thresholds": thr, "standard_metrics": std,
                                                  "operating_points": ops,
                                                  **({"ft": ft_extra} if ft_extra else {})}
            rows[f"{mname} | {cname}"] = {"aucpr": std["aucpr"], "rocauc": std["rocauc"],
                                          "accuracy": float(((s_te >= 0.5).astype(int) == y_te).mean()),
                                          "coverage_test": 1.0, "n_evaluated": int(len(y_te)),
                                          "representative_test": True,
                                          "note": "curated held-out test set; prior is a property of corpus construction"}
            r0 = ops.get("fixed@0.50", {}).get("conditional_rates")
            if r0:
                print(f"  {mname:8s} {cname}: AUCPR={std['aucpr']:.4f}  TPR={r0['TPR_phishing']['point']:.3f}  "
                      + "  ".join(f"FPR_{s}={r0[f'FPR_{s}']['point']:.4f}({r0[f'FPR_{s}']['denominator']})"
                                  for s in NEGATIVE_SUBTYPES if r0[f'FPR_{s}']['denominator']))

    suf = os.environ.get("EXPERIMENT_RESULT_SUFFIX", "")
    out = {"experiment": "13_phishing_operational", "regime": regime, "variant": variant, "seed": seed,
           "n_test": int(len(test)), "test_prior": float(y_te.mean()),
           "prior_note": "CURATED test prior: a property of corpus construction, not a deployment prior",
           "bundle": bundle,
           "label_policy": "strict (phishing vs all non-phishing); the broad-malicious sensitivity is "
                           "deliberately not wired into this runner",
           "annotation_policy_version": ANNOTATION_POLICY_VERSION,
           "manifest_fingerprint": corpus_report["manifest_fingerprint"], "sources": corpus_report["assets"],
           "taxonomy_caveats": "bulk_spam rows mapped with confidence 'proxy' come from generic spam corpora that may "
                               "contain phishing; Background-style 'unknown_nonphishing' rows are not verified benign",
           "dedup_diagnostics": dedup_diag, "partition_overlap": overlap, "split_policy": split_policy,
           "partition_counts": part_counts,
           "partition_fingerprints": {k: ordered_hash_fingerprint(v) for k, v in
                                      (("train", train), ("calibration", cal), ("test", test))},
           "hard_negative_conditions": {k: v[1] for k, v in conds.items()},
           "test_complete": True,
           "ft_config": ({"scope": args.ft_scope, "backend": args.ft_backend,
                          "base_model": args.ft_base_model, "output_dir": str(args.ft_output_dir),
                          "conditions": args.ft_conditions or args.conditions,
                          "note": "a single-bundle fine-tuning scope is a case study, not general "
                                  "external-domain evidence"}
                         if "modernbert_ft" in args.models else None),
           "models": results_models, "rows": rows,
           "corpus_artifact_diagnostic": cad,
           "plot_data": str(out_dir / f"13_phishing_operational.plot_data{suf}.csv"),
           "limitations": ["curated corpora: class ratios are properties of corpus construction, not of a stream",
                           "projections are hypothetical mixtures; support_status marks what the denominators resolve",
                           "no FP/hour: corpora are not one observed continuous interval",
                           "template-prefix grouping is a proxy for campaigns when no campaign metadata exists"]}
    out_json = out_dir / f"13_phishing_operational.{tag}{suf}.result.json"
    out_json.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {out_json}")
    # compact per-run summary at fixed@0.50 (feeds the external-bundle summary artifact)
    summary = {"tag": tag, "bundle_id": bundle["bundle_id"] if bundle else None, "n_test": int(len(test)),
               "test_prior": float(y_te.mean()),
               "holdout_tier_by_role": (bundle or {}).get("holdout_tier_by_role"),
               "fully_family_disjoint": (bundle or {}).get("fully_family_disjoint"), "cells": {}}
    for key, rm in results_models.items():
        op = rm.get("operating_points", {}).get("fixed@0.50")
        if not op or not op.get("attainable"):
            continue
        r = op["conditional_rates"]
        cell = {"aucpr": rm["standard_metrics"]["aucpr"],
                "tpr_phishing": r["TPR_phishing"]["point"],
                **{f"fpr_{st}": r[f"FPR_{st}"]["point"] for st in NEGATIVE_SUBTYPES
                   if r[f"FPR_{st}"]["denominator"] > 0}}
        # one selected projection cell for cross-bundle comparison (recorded, not chosen post hoc)
        sel_pi = 0.01 if 0.01 in args.pis else args.pis[0]
        sel_sh = 0.5 if 0.5 in args.spam_shares else args.spam_shares[len(args.spam_shares) // 2]
        for c in op["deployment_grid"]:
            if c["prior"] == sel_pi and c["spam_share"] == sel_sh and c["projection"].get("available"):
                cell["projected_precision_point"] = c["projection"]["precision_point"]
                cell["projected_precision_conservative"] = c["projection"]["precision_conservative"]
                cell["projection_cell"] = {"prior": sel_pi, "spam_share": sel_sh}
                break
        summary["cells"][key] = cell
    return out_json, summary


def write_external_summary(variant, summaries, n_bundles, out_dir, suf, ft_scope=None):
    """Median and range across external bundles per model|condition cell, at the
    fixed@0.50 operating point. Written even with zero bundles (n_bundles=0) so
    verification can require the file. Never entered into AUCPR rankings."""
    cells = {}
    for smry in summaries:
        for key, v in smry["cells"].items():
            cells.setdefault(key, []).append({"bundle_id": smry["bundle_id"], **v})
    def agg(vals):
        vv = [v for v in vals if v is not None]
        return {"median": float(np.median(vv)), "min": float(np.min(vv)), "max": float(np.max(vv)),
                "n": len(vv)} if vv else None
    out = {"experiment": "13_phishing_operational", "artifact": "external_bundle_summary", "variant": variant,
           "n_bundles_valid": int(n_bundles), "n_bundles_run": len(summaries),
           "single_bundle_case_study": bool(n_bundles == 1),
           "per_cell": {key: {"per_bundle": rows,
                              **{metric: agg([r.get(metric) for r in rows])
                                 for metric in ["aucpr", "tpr_phishing",
                                                "fpr_ham", "fpr_bulk_spam", "fpr_scam_fraud",
                                                "fpr_unknown_nonphishing",
                                                "projected_precision_point",
                                                "projected_precision_conservative"]}}
                        for key, rows in cells.items()},
           "holdout_tiers": {sm["bundle_id"]: sm.get("holdout_tier_by_role") for sm in summaries},
           "n_fully_family_disjoint": sum(1 for sm in summaries if sm.get("fully_family_disjoint")),
           "family_caveat": "a bundle whose held-out source shares a corpus family with a development "
                            "source tests WITHIN-FAMILY partition generalization (shared collection "
                            "process and fingerprints), not unseen-corpus generalization; see "
                            "holdout_tier_by_role per bundle",
           "ft_scope": ft_scope,
           "note": "median and range across complete external bundles at fixed@0.50; diagnostic artifact, "
                   "excluded from AUCPR rankings"
                   + ("; modernbert_ft cells may cover FEWER bundles than n_bundles_run (see each cell's n): "
                      "under a staged/minimal scope a one-bundle FT cell is a case study, not general "
                      "external-domain evidence" if ft_scope else "")}
    path = out_dir / f"13_phishing_operational.external_summary.{variant}{suf}.result.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {path}")


def write_plots(plot_rows, out_dir, suf, enabled=True):
    path = out_dir / f"13_phishing_operational.plot_data{suf}.csv"
    if not plot_rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(plot_rows[0].keys())); w.writeheader(); w.writerows(plot_rows)
    print(f"wrote {path}")
    if not enabled:
        return
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(plot_rows)
    df = df[df["threshold_policy"] == "fixed@0.50"]
    for run, g in df.groupby("run"):
        for model, gm in g.groupby("model"):
            for kind in ("precision_point", "precision_conservative"):
                shares = sorted(gm["spam_share"].unique())
                fig, axes = plt.subplots(1, len(shares), figsize=(3.6 * len(shares), 3.4), sharey=True)
                axes = np.atleast_1d(axes)
                for ax, sh in zip(axes, shares):
                    for cond, gc in gm[gm["spam_share"] == sh].groupby("condition"):
                        gc = gc.sort_values("prior")
                        ax.plot(gc["prior"], gc[kind], marker="o", label={"A": "ham-only", "B": "matched-budget ham+spam/scam", "C": "additive ham+spam/scam"}[cond])
                        ext = gc[gc["support_status"] == "extrapolative"]
                        if len(ext):
                            ax.scatter(ext["prior"], ext[kind], marker="x", color="k", zorder=3)
                    ax.set_xscale("log"); ax.set_xlim(8e-5, 1.2e-1); ax.set_ylim(0, 1.02)
                    ax.set_title(f"{int(sh*100)}% spam/scam among negatives", fontsize=9); ax.set_xlabel("phishing prevalence (hypothetical)")
                axes[0].set_ylabel("projected precision"); axes[0].legend(fontsize=7)
                fig.suptitle(f"{run} / {model} / {kind.replace('precision_', '')} (x = extrapolative)", fontsize=9)
                fig.tight_layout()
                fig.savefig(out_dir / f"13_phishing_operational.{run}.{model}.{kind.replace('precision_', '')}{suf}.png", dpi=130)
                plt.close(fig)


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(__file__).parent
    lim = os.environ.get("EXPERIMENT_LIMIT")

    if args.ft_conditions and not set(args.ft_conditions) <= set(args.conditions):
        sys.exit("--ft-conditions must be a subset of --conditions")
    if args.synthetic_smoke:
        if "modernbert_ft" in args.models and args.ft_backend == "real":
            print("[synthetic-smoke] forcing --ft-backend fake (no downloads, no real fine-tune)")
            args.ft_backend = "fake"
        df, report = synthetic_corpus(args.seed, n_per_source=max(int(lim) if lim else 160, 60))
        args.n_boot = min(args.n_boot, 25)
    else:
        df, report = load_manifest(args.manifest)
        if lim and int(lim) < len(df):        # smoke-style cap: per-source stratified, deterministic
            df = (df.groupby("source_corpus", group_keys=False)
                    .apply(lambda g: g.sample(n=min(len(g), max(int(lim) // df["source_corpus"].nunique(), 20)),
                                              random_state=args.seed))).reset_index(drop=True)
    df, dedup_diag = dedup_and_group(df)
    print(f"corpus: {len(df)} rows after dedup ({dedup_diag['exact_duplicates_removed']} exact dups removed, "
          f"{dedup_diag['cross_source_duplicate_hashes']} cross-source hashes); subtypes {df['subtype'].value_counts().to_dict()}")
    roles, comp = source_roles(df, args.role_purity)
    bundles = enumerate_bundles(df, roles, comp, args.min_source_rows, args.role_purity)
    print(f"source roles (purity {args.role_purity}; 'mixed' = development-only): {roles}\n"
          f"valid external bundles: {[(b['bundle_id'], b['phishing_source'], b['ham_source'], b['spam_scam_source']) for b in bundles]}")
    for i, b in enumerate(bundles):
        b["_rank"] = i          # predeclared, rule-ordered; rank 0 = the staged/minimal FT bundle
    if args.bundles == "first":
        bundles = bundles[:1]
    elif args.bundles != "all":
        bundles = [b for b in bundles if b["bundle_id"] == args.bundles]

    plot_rows = []
    suf = os.environ.get("EXPERIMENT_RESULT_SUFFIX", "")
    for variant in args.variants:
        dvar = df.copy()
        if variant == "stripped":
            dvar["text"] = dvar["text"].map(strip_provenance)
        cad = corpus_artifact_variant_diagnostic(dvar, args.seed)
        cad["raw_vs_stripped_variant"] = variant
        if "controlled" in args.regimes:
            run_one(args, df, report, dedup_diag, "controlled", variant, None, out_dir, plot_rows, cad)
            if args.random_reference:
                run_one(args, df, report, dedup_diag, "controlled-random", variant, None, out_dir, plot_rows, cad)
        if "external" in args.regimes:
            if not bundles:
                print("no complete external bundle exists (need unseen pure phishing + ham + spam/scam sources "
                      "with each role still present in development); external regime skipped")
            summaries = []
            for b in bundles:
                r = run_one(args, df, report, dedup_diag, "external", variant, b, out_dir, plot_rows, cad)
                if r:
                    summaries.append(r[1])
            write_external_summary(variant, summaries, len(bundles), out_dir, suf,
                                   ft_scope=args.ft_scope if "modernbert_ft" in args.models else None)
    if len(bundles) == 1 and "external" in args.regimes:
        print("NOTE: exactly one complete external bundle: report it as a single external-domain case study")
    write_plots(plot_rows, out_dir, suf, enabled=not args.no_plots)


if __name__ == "__main__":
    main()
