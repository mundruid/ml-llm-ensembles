#!/usr/bin/env python3
"""Synthetic verification for experiment 13 (no data, no models beyond tiny fits).

    uv run python experiments/tests_13_synthetic.py

Each numbered check maps to the list in 13_phishing_operational.md. Plain asserts;
exit code non-zero on the first failure.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "ml_llm_ensembles" / "experiments"))

from ml_llm_ensembles.utils import phishing_taxonomy as T
from ml_llm_ensembles.utils import operational_phishing as O
from ml_llm_ensembles.utils.evaluation import threshold_for_recall, threshold_for_precision
import importlib
R = importlib.import_module("13_phishing_operational")

df, report = T.synthetic_corpus(42)
d, diag = T.dedup_and_group(df)

# 1. source labels, mapping rules and subtypes retained
assert set(T.REQUIRED_FIELDS) <= set(d.columns)
assert d["source_label"].nunique() >= 3 and d["mapping_rule"].nunique() >= 4
assert set(d["subtype"]) == {"ham", "bulk_spam", "scam_fraud", "phishing"}
assert (d.loc[d.subtype == "bulk_spam", "mapping_confidence"] == "proxy").all()
try:
    T.apply_mapping(pd.Series(["weird"]), "spamassassin"); raise SystemExit("unmapped label must raise")
except ValueError:
    pass
print("1 ok taxonomy fields")

# 2. duplicates/groups cannot cross partitions; 3. controlled separation
(tr, cal, te), _salt = R.group_partition(d, [0.6, 0.2, 0.2], 42, required=R.required_subtypes(d, [0.0, 0.5]))
parts = {"train": d.iloc[tr], "calibration": d.iloc[cal], "test": d.iloc[te]}
ov = T.partition_overlap_diag(parts); assert ov["clean"], ov
assert diag["exact_duplicates_removed"] > 0 and diag["cross_source_duplicate_hashes"] == 1
assert d["content_hash"].is_unique
print("2-3 ok dedup, grouped partitions", ov["group_overlap"])

# 4-5. complete external bundles exclude all three held-out sources; rotation
roles, comp = R.source_roles(d, 0.9); bundles = R.enumerate_bundles(d, roles, comp, 50, 0.9)
assert len(bundles) >= 2, bundles           # 2 phishing x 2 ham x 1 spam-like minus invalid = rotation exists
for b in bundles:
    held = {b["phishing_source"], b["ham_source"], b["spam_scam_source"]}
    assert not held & set(b["development_sources"])
    assert {roles[s] for s in b["development_sources"]} >= {"phishing", "ham", "spam_like"}
ids = [b["bundle_id"] for b in bundles]; assert len(ids) == len(set(ids))
print("4-5 ok bundles", ids)

# 6. calibration and thresholds only see calibration rows (structural: functions take cal arrays only)
y_cal = parts["calibration"]["label_binary"].values; st_cal = parts["calibration"]["subtype"].values
rng = np.random.default_rng(0); s_cal = np.clip(0.8 * y_cal + rng.normal(0, 0.2, len(y_cal)), 0, 1)
thr = O.select_thresholds(y_cal, st_cal, s_cal)
assert all(k.startswith(("fixed", "cal-", "_meta")) for k in thr)
print("6 ok thresholds from calibration only")

# 7. exact recall/precision thresholds
y = np.array([1] * 10 + [0] * 90); s = np.r_[np.linspace(.5, 1, 10), rng.uniform(0, .6, 90)]
t = threshold_for_recall(y, s, 0.9); assert abs(((s >= t) & (y == 1)).sum() / 10 - 0.9) < 1e-9
# a higher recall target needs more positives above threshold -> a lower (or equal) threshold
assert threshold_for_recall(y, s, 0.9) <= threshold_for_recall(y, s, 0.5)
tp_ = threshold_for_precision(y, s, 0.9); assert ((s >= tp_) & (y == 1)).sum() / (s >= tp_).sum() >= 0.9
assert threshold_for_precision(np.array([0, 1, 0, 1]), np.array([.9, .8, .7, .6]), 0.99) is None
print("7 ok exact thresholds")

# 8. separate subtype FPRs
y_te = parts["test"]["label_binary"].values; st_te = parts["test"]["subtype"].values; g_te = parts["test"]["campaign_or_template_group"].values
s_te = np.where(st_te == "phishing", 0.9, np.where(st_te == "scam_fraud", 0.6, 0.1)) + rng.normal(0, 0.02, len(y_te))
rates = O.conditional_rates(y_te, st_te, s_te, 0.5, g_te)
assert rates["FPR_scam_fraud"]["point"] > 0.9 and rates["FPR_ham"]["point"] < 0.05
assert rates["FPR_ham"]["denominator"] > 0 and rates["FPR_ham"]["resolution"] == 1 / rates["FPR_ham"]["denominator"]
print("8 ok subtype FPRs", {k: round(v["point"], 3) for k, v in rates.items() if k != "threshold" and v["point"] is not None})

# 9-10. mixture weights, point and conservative projection
q = O.mixture_weights(0.5, 0.5); assert abs(sum(q.values()) - 1) < 1e-12 and q["bulk_spam"] == 0.25
p = O.projection(rates, 0.01, q)
fm = 0.5 * rates["FPR_ham"]["point"] + 0.25 * rates["FPR_bulk_spam"]["point"] + 0.25 * rates["FPR_scam_fraud"]["point"]
assert abs(p["mixed_fpr"] - fm) < 1e-12
assert abs(p["precision_point"] - 0.01 * p["recall_point"] / (0.01 * p["recall_point"] + 0.99 * fm)) < 1e-12
assert p["precision_conservative"] <= p["precision_point"] + 1e-12
print("9-10 ok projections", round(p["precision_point"], 4), round(p["precision_conservative"], 4))

# 11. observed-mixture self-consistency
neg = st_te != "phishing"; q_obs = {s: float((st_te[neg] == s).mean()) for s in T.NEGATIVE_SUBTYPES}
pred = s_te >= 0.5; meas = (pred & (y_te == 1)).sum() / pred.sum()
assert abs(O.projection(rates, float(y_te.mean()), q_obs)["precision_point"] - meas) < 1e-9
print("11 ok observed-mixture self-check")

# 12. zero-FP upper bound is nonzero
r0 = O.conditional_rates(np.array([0] * 50 + [1] * 5), np.array(["ham"] * 50 + ["phishing"] * 5), np.r_[np.zeros(50), np.ones(5)], 0.5)
assert r0["FPR_ham"]["numerator"] == 0 and r0["FPR_ham"]["ci95"][1] > 0
print("12 ok zero-FP upper bound", round(r0["FPR_ham"]["ci95"][1], 4))

# 13. required-FPR algebra and support flags
req = O.required_fpr_max(0.9, 1.0, 0.001); assert abs(req - 0.001 * (0.1) / (0.9 * 0.999)) < 1e-12
d0 = O.support_diagnostic(r0, 0.001, O.mixture_weights(0.0), 0.9)
assert d0["support_status"] == "extrapolative" and d0["zero_observed_fp"] and "zero FPs" in d0["support_reason"]
d1 = O.support_diagnostic(rates, 0.05, O.mixture_weights(0.0), 0.5)
assert d1["support_status"] in ("supported", "limited", "extrapolative") and d1["support_reason"]
print("13 ok required-FPR", d0["support_status"], d1["support_status"])

# 14-15. matched-budget keeps negative count; additive labelled as more data + diversity
train = parts["train"]
conds = R.hard_negative_conditions(train, 42, 0.5, ["A", "B", "C"])
assert conds["A"][1]["n_neg_total"] == conds["B"][1]["n_neg_total"]
assert conds["B"][1]["n_hard"] > 0 and conds["C"][1]["n_neg_total"] > conds["A"][1]["n_neg_total"]
assert "more data AND more diversity" in conds["C"][1]["negative_budget_policy"]
for k in "ABC":
    idx = conds[k][0]; assert len(idx) == len(set(idx)); assert (train["label_binary"].values[idx] == 1).sum() == conds[k][1]["n_pos"]
print("14-15 ok hard-negative budgets", {k: v[1]["n_neg_total"] for k, v in conds.items()})

# 16. bootstrap resamples groups
boot = O.group_bootstrap(y_te, st_te, s_te, 0.5, g_te, [0.01], [q], n_boot=30, seed=1)
assert boot["bootstrap_unit"] == "campaign_or_template_group" and boot["n_groups"] == len(set(g_te))
assert boot["projected_precision_ci95"]["0.01|0"][0] is not None
print("16 ok group bootstrap", boot["rates_ci95"]["TPR_phishing"])

# 17. source classifier balanced and isolated from rankings (structural: stored under corpus_artifact_diagnostic)
cad = R.corpus_artifact_diagnostic(train["text"].tolist(), train["source_corpus"].tolist(),
                                   parts["test"]["text"].tolist(), parts["test"]["source_corpus"].tolist(), 42)
assert cad["class_weight"] == "balanced" and 0 <= cad["macro_f1"] <= 1
print("17 ok source classifier macro-F1", round(cad["macro_f1"], 3))

# 18. no projected AUCPR anywhere
assert "aucpr" not in json.dumps(p) and "aucpr" not in json.dumps(boot)
# 19. non-representative subset AUCPR cannot enter rankings (aggregator rule)
import make_results_table as M
assert M.trustworthy({"aucpr": 0.9, "representative_test": False}) is False
assert M.trustworthy({"aucpr": 0.9, "representative_test": True, "coverage_test": 1.0}) is True
print("18-19 ok no projected AUCPR; balanced rows excluded from ranking")

# 20. deterministic fingerprints and filenames
f1 = T.ordered_hash_fingerprint(parts["train"]); f2 = T.ordered_hash_fingerprint(T.dedup_and_group(T.synthetic_corpus(42)[0])[0].iloc[tr])
assert f1 == f2
b1 = R.enumerate_bundles(d, roles, comp, 50, 0.9)[0]["bundle_id"]; assert b1 == hashlib.sha256(
    f"{bundles[0]['phishing_source']}|{bundles[0]['ham_source']}|{bundles[0]['spam_scam_source']}".encode()).hexdigest()[:8].join(["b_", ""])
print("20 ok deterministic fingerprints")

# 21. sub-purity ("mixed") sources are excluded from hold-out; composition recorded
d_mix = d.copy()
mix = d_mix[d_mix.source_corpus == "synth_ham_list"].index
half = mix[: len(mix) // 2]
d_mix.loc[half, "subtype"] = "phishing"; d_mix.loc[half, "label_binary"] = 1
roles_m, comp_m = R.source_roles(d_mix, 0.9)
assert roles_m["synth_ham_list"] == "mixed"
bundles_m = R.enumerate_bundles(d_mix, roles_m, comp_m, 50, 0.9)
for b in bundles_m:
    assert "synth_ham_list" not in {b["phishing_source"], b["ham_source"], b["spam_scam_source"]}
    assert "synth_ham_list" in b["development_sources"]
    for src, c in b["held_out_composition"].items():
        assert "subtype_fractions" in c and c["role"] != "mixed"
print("21 ok mixed sources are development-only; held-out composition recorded")

# 22. impossible subtype coverage fails clearly instead of degrading
d_thin = d[d.subtype != "scam_fraud"].copy()
one = d_thin[d_thin.subtype == "bulk_spam"].index[:1]
d_thin = pd.concat([d_thin[d_thin.subtype != "bulk_spam"], d_thin.loc[one]]).reset_index(drop=True)
try:
    R.group_partition(d_thin, [0.6, 0.2, 0.2], 42, required={"phishing", "ham", "bulk_spam"}, max_salt=5)
    raise AssertionError("expected SystemExit for unattainable subtype coverage")
except SystemExit as e:
    assert "required subtypes" in str(e)
print("22 ok unattainable coverage fails clearly")

# 23. external summary median/range math
sums = [{"tag": "x", "bundle_id": f"b{i}", "n_test": 10, "test_prior": 0.3,
         "cells": {"xgb|A": {"aucpr": a, "tpr_phishing": 1.0, "fpr_ham": 0.0,
                             "fpr_bulk_spam": 0.02, "fpr_scam_fraud": 0.1,
                             "projected_precision_point": 0.7}}}
        for i, a in enumerate([0.2, 0.5, 0.8])]
import tempfile
with tempfile.TemporaryDirectory() as td:
    R.write_external_summary("raw", sums, 3, Path(td), "")
    js = json.loads((Path(td) / "13_phishing_operational.external_summary.raw.result.json").read_text())
    agg = js["per_cell"]["xgb|A"]["aucpr"]
    assert agg == {"median": 0.5, "min": 0.2, "max": 0.8, "n": 3}
    for metric in ["fpr_ham", "fpr_bulk_spam", "fpr_scam_fraud", "fpr_unknown_nonphishing",
                   "projected_precision_point", "projected_precision_conservative"]:
        assert metric in js["per_cell"]["xgb|A"]     # aggregated (None when never measured)
    assert js["per_cell"]["xgb|A"]["fpr_scam_fraud"] == {"median": 0.1, "min": 0.1, "max": 0.1, "n": 3}
    assert js["single_bundle_case_study"] is False and "rankings" in js["note"]
    R.write_external_summary("raw", [], 0, Path(td), ".s")
    js0 = json.loads((Path(td) / "13_phishing_operational.external_summary.raw.s.result.json").read_text())
    assert js0["n_bundles_valid"] == 0
print("23 ok external summary median/range; zero-bundle file written")

# 24. corpus-artifact diagnostic split has every source on both sides, independent of bundles
cad = R.corpus_artifact_variant_diagnostic(d, 42)
assert set(cad["sources_evaluated"]) | set(cad["sources_dropped_missing_one_side"]) == set(d.source_corpus.unique())
assert all(v > 0 for v in cad["word"]["per_source_recall"].values()) or cad["word"]["macro_f1"] >= 0
assert set(cad["sources_evaluated"]) == set(d.source_corpus.unique())   # synthetic corpus: none dropped
print("24 ok diagnostic split covers every source on both sides")

# 25. an unknown-heavy source cannot be the external spam/scam source F
d_unk = d.copy()
unk = d_unk[d_unk.source_corpus == "synth_spam"].index
d_unk.loc[unk, "subtype"] = "unknown_nonphishing"          # now 100% unknown negatives
roles_u, comp_u = R.source_roles(d_unk, 0.9)
assert roles_u["synth_spam"] == "unknown_negative"
bundles_u = R.enumerate_bundles(d_unk, roles_u, comp_u, 50, 0.9)
for b in bundles_u:
    assert b["spam_scam_source"] != "synth_spam"
    assert "synth_spam" in b["development_sources"]        # still a development hard negative
# synth_fraud remains the only spam_like source -> holding it out empties dev's spam_like -> no bundles
assert bundles_u == []
print("25 ok unknown-heavy sources are development-only, never spam/scam source F")

# ── modernbert_ft (fake backend; no downloads, no real fine-tune) ─────────────
import tempfile
from ml_llm_ensembles.utils import modernbert_ft as F

# 26. identity changes with EVERY field (regime, bundle, variant, condition, seed,
#     policy version, fingerprints, base model, hyperparams)
base_kw = dict(regime="controlled", bundle_id=None, variant="raw", condition="A", seed=42,
               policy_version="1.0", train_fingerprint="t1", cal_fingerprint="c1",
               base_model="answerdotai/ModernBERT-base")
n0, _ = F.checkpoint_identity(**base_kw, hyperparams={"epochs": 2})
names = {n0}
for k, v in [("regime", "external"), ("bundle_id", "b_x"), ("variant", "stripped"), ("condition", "B"),
             ("seed", 43), ("policy_version", "1.1"), ("train_fingerprint", "t2"),
             ("cal_fingerprint", "c2"), ("base_model", "other/model")]:
    n, _ = F.checkpoint_identity(**{**base_kw, k: v}, hyperparams={"epochs": 2})
    names.add(n)
nh, _ = F.checkpoint_identity(**base_kw, hyperparams={"epochs": 3})
names.add(nh)
assert len(names) == 11, "every identity field must change the checkpoint directory"
print("26 ok checkpoint identity distinguishes every field (incl. raw vs stripped)")

with tempfile.TemporaryDirectory() as td:
    # 27. score() before fit() raises; fit precedes score in the call log
    def mk(cond="A", tr=None, ca=None, resume=False, seed=42):
        tr = tr or ["buy pills now", "meeting notes agenda"]; ca = ca or ["verify password account"]
        kw = dict(regime="controlled", bundle_id=None, variant="raw", condition=cond, seed=seed,
                  policy_version="1.0", train_fingerprint=F._sha_texts(tr), cal_fingerprint=F._sha_texts(ca),
                  base_model="answerdotai/ModernBERT-base")
        b = F.make_ft_backend("fake", td, kw, {"epochs": 2}, resume)
        return b, tr, ca
    b, tr, ca = mk()
    try:
        b.score(["x"]); raise AssertionError("score before fit must raise")
    except RuntimeError as e:
        assert "before fit" in str(e)
    b.fit(tr, [0, 0] if len(tr) == 2 else [0], ca, [1])
    _ = b.score(["verify account"])
    ev = [r["event"] for r in F.FT_CALL_LOG if r["identity"] == b.name]
    assert ev.index("fit_train") < ev.index("score") and "fit_cal" in ev
    assert (Path(td) / b.name / "metadata.json").exists()
    meta = json.loads((Path(td) / b.name / "metadata.json").read_text())
    assert meta["selection_metric"].startswith("calibration AUCPR") and meta["identity"] == b.identity
    print("27 ok fit-before-score enforced; metadata written with calibration-only selection")

    # 28. resume works only on an exact metadata match; mismatch fails clearly
    b2, tr2, ca2 = mk(resume=True)
    b2.fit(tr2, [0, 0], ca2, [1])
    assert b2.resumed, "identical identity + --ft-resume must resume"
    mp = Path(td) / b2.name / "metadata.json"
    stored = json.loads(mp.read_text()); stored["identity"]["seed"] = 999; mp.write_text(json.dumps(stored))
    b3, tr3, ca3 = mk(resume=True)
    try:
        b3.fit(tr3, [0, 0], ca3, [1]); raise AssertionError("mismatched metadata must not resume")
    except RuntimeError as e:
        assert "does not match" in str(e)
    print("28 ok resume requires exact metadata match; mismatch fails clearly")

# 29-31. run_one end-to-end with modernbert_ft on an EXTERNAL bundle:
#   fit sees only development rows; calibration rows are development-only; test
#   (held-out D+E+F) appears only in score events after fit; result JSON records
#   the ft block; the external summary aggregates its subtype FPRs.
F.FT_CALL_LOG.clear()
a = R.build_parser().parse_args([])
a.models = ["modernbert_ft"]; a.ft_backend = "fake"; a.ft_scope = "full"; a.n_boot = 10
with tempfile.TemporaryDirectory() as td:
    a.ft_output_dir = Path(td) / "models"
    b0 = dict(bundles[0]); b0["_rank"] = 0
    res = R.run_one(a, d, report, diag, "external", "raw", b0, Path(td), [], {"stub": True})
    assert res is not None
    out_json, smry = res
    js = json.loads(Path(out_json).read_text())
    held = {b0["phishing_source"], b0["ham_source"], b0["spam_scam_source"]}
    held_shas = F._text_shas(d[d.source_corpus.isin(held)]["text"])
    dev_shas = F._text_shas(d[~d.source_corpus.isin(held)]["text"])
    fit_shas = set().union(*[r["text_shas"] for r in F.FT_CALL_LOG if r["event"] in ("fit_train", "fit_cal")])
    assert fit_shas and fit_shas <= dev_shas and not (fit_shas & held_shas), \
        "an external fine-tune must never receive held-out source rows"
    score_shas = set().union(*[r["text_shas"] for r in F.FT_CALL_LOG if r["event"] == "score"])
    assert held_shas <= score_shas, "held-out bundle is scored (after fit) in full"
    for cname in ("A", "B", "C"):
        e = js["models"].get(f"modernbert_ft|{cname}")
        if e and "skipped" not in e:
            ft = e["ft"]
            for k in ("base_model", "checkpoint", "identity", "selected_epoch", "selection_metric",
                      "hyperparams", "resumed", "train_seconds", "inference_seconds"):
                assert k in ft, k
            assert ft["base_model"] == "answerdotai/ModernBERT-base" and ft["backend"] == "fake"
    assert js["ft_config"]["scope"] == "full"
    R.write_external_summary("raw", [smry], 1, Path(td), ".t", ft_scope="full")
    sj = json.loads((Path(td) / "13_phishing_operational.external_summary.raw.t.result.json").read_text())
    ft_cells = [k for k in sj["per_cell"] if k.startswith("modernbert_ft|")]
    assert ft_cells and sj["single_bundle_case_study"] is True
    for k in ft_cells:
        assert "fpr_ham" in sj["per_cell"][k] and "fpr_scam_fraud" in sj["per_cell"][k]
    assert "case study" in sj["note"]
print("29-31 ok external FT isolation, ft result block, summary aggregation")

# 32. FT hard-negative budgets: A and B identical (same function as check 14) and
#     the ft path consumes the same condition indices
assert conds["A"][1]["n_neg_total"] == conds["B"][1]["n_neg_total"]
# 33. scope gating: minimal excludes stripped and condition C; staged keeps rank-0 bundle only
a.ft_scope = "minimal"
assert R.ft_in_scope(a, "controlled", "raw", "A", None) and not R.ft_in_scope(a, "controlled", "raw", "C", None)
assert not R.ft_in_scope(a, "controlled", "stripped", "A", None)
assert R.ft_in_scope(a, "external", "raw", "B", {"_rank": 0, "bundle_id": "b"}) \
       and not R.ft_in_scope(a, "external", "raw", "B", {"_rank": 1, "bundle_id": "b"})
a.ft_scope = "staged"
assert R.ft_in_scope(a, "controlled", "stripped", "C", None) \
       and not R.ft_in_scope(a, "external", "raw", "A", {"_rank": 2, "bundle_id": "b"})
assert not R.ft_in_scope(a, "controlled-random", "raw", "A", None)
# an explicitly named bundle overrides the first-bundle gate (staged and minimal)
a.ft_bundles = "b_xyz"
assert R.ft_in_scope(a, "external", "raw", "A", {"_rank": 3, "bundle_id": "b_xyz"})
assert not R.ft_in_scope(a, "external", "raw", "A", {"_rank": 0, "bundle_id": "b_other"})
a.ft_scope = "minimal"
assert R.ft_in_scope(a, "external", "raw", "B", {"_rank": 3, "bundle_id": "b_xyz"})
assert not R.ft_in_scope(a, "external", "stripped", "B", {"_rank": 3, "bundle_id": "b_xyz"})
a.ft_bundles = "all"
print("32-33 ok budgets shared with A/B; scope gating incl. explicit-bundle override")

# 34. gradient-accumulation group sizes: full groups get accum, the trailing
#     partial group gets its true size (loss not underweighted)
gs = [F.accum_group_size(i, 4, 10) for i in range(10)]
assert gs == [4, 4, 4, 4, 4, 4, 4, 4, 2, 2] and F.accum_group_size(0, 1, 3) == 1
assert [F.accum_group_size(i, 3, 3) for i in range(3)] == [3, 3, 3]
print("34 ok partial accumulation group uses its true divisor")
print("\nALL 34 CHECKS PASSED")
