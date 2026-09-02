"""
Operational phishing evaluation: subtype-conditional rates, calibration,
threshold policies, deployment projections and support diagnostics.
=====================================================================
Implements the arithmetic of experiments/13_phishing_operational.md. Every
projection is a calculation under a HYPOTHETICAL prevalence pi and negative
mixture q; no rows are resampled to manufacture a prior, and no AUCPR is ever
projected from a single operating point.
"""
from __future__ import annotations

import math

import numpy as np

from ml_llm_ensembles.utils.evaluation import wilson, threshold_for_recall
from ml_llm_ensembles.utils.phishing_taxonomy import NEGATIVE_SUBTYPES

DEFAULT_PIS = [0.05, 0.01, 0.001, 0.0001]
DEFAULT_SPAM_SHARES = [0.0, 0.1, 0.5, 0.9]


# ── Conditional rates ─────────────────────────────────────────────────────────
def conditional_rates(y, subtype, score, thr, groups=None) -> dict:
    """TPR on phishing and FPR per negative subtype at threshold thr, each with
    numerator, denominator, Wilson 95% interval, empirical resolution 1/n and the
    number of independent groups (if groups are given)."""
    y = np.asarray(y).astype(int); st = np.asarray(subtype, dtype=object)
    pred = (np.asarray(score, dtype=float) >= thr).astype(int)
    groups = None if groups is None else np.asarray(groups, dtype=object)
    def rate(mask):
        n = int(mask.sum()); k = int((pred[mask] == 1).sum())
        lo, hi = wilson(k, n)
        return {"numerator": k, "denominator": n, "point": (k / n) if n else None,
                "ci95": [lo, hi] if n else [None, None], "resolution": (1.0 / n) if n else None,
                "n_groups": int(len(set(groups[mask]))) if (groups is not None and n) else None}
    out = {"threshold": float(thr), "TPR_phishing": rate(st == "phishing")}
    for s in NEGATIVE_SUBTYPES:
        out[f"FPR_{s}"] = rate(st == s)
    return out


# ── Mixtures and projections ──────────────────────────────────────────────────
def mixture_weights(spam_share: float, spam_split_bulk: float = 0.5, unknown_share: float = 0.0) -> dict:
    """q over negative subtypes. spam_share is the share of bulk_spam+scam_fraud
    among negatives; spam_split_bulk says how much of it is bulk spam (rest is
    scam/fraud). unknown_share is carved out of the ham share."""
    q_spam = spam_share * spam_split_bulk; q_fraud = spam_share * (1 - spam_split_bulk)
    q_ham = max(0.0, 1.0 - spam_share - unknown_share)
    q = {"ham": q_ham, "bulk_spam": q_spam, "scam_fraud": q_fraud, "unknown_nonphishing": unknown_share}
    assert abs(sum(q.values()) - 1.0) < 1e-9
    return q


def mixed_fpr(rates: dict, q: dict, bound: str = "point") -> float | None:
    """Weighted FPR; subtypes with q>0 but no denominator make the result None."""
    tot = 0.0
    for s, w in q.items():
        if w == 0:
            continue
        r = rates[f"FPR_{s}"]
        if r["denominator"] == 0:
            return None
        v = r["point"] if bound == "point" else r["ci95"][1] if bound == "upper" else r["ci95"][0]
        tot += w * v
    return tot


def project_precision(pi: float, tpr: float, fpr_mix: float) -> float:
    d = pi * tpr + (1 - pi) * fpr_mix
    return (pi * tpr / d) if d > 0 else 0.0


def projection(rates: dict, pi: float, q: dict) -> dict:
    """Point and conservative (TPR lower, FPR upper) projection at (pi, q)."""
    tpr = rates["TPR_phishing"]["point"]; tpr_lo = rates["TPR_phishing"]["ci95"][0]
    fm = mixed_fpr(rates, q, "point"); fm_up = mixed_fpr(rates, q, "upper")
    if tpr is None or fm is None:
        return {"prior": pi, "q": q, "available": False}
    out = {"prior": pi, "q": q, "available": True,
           "mixed_fpr": fm, "mixed_fpr_upper": fm_up,
           "precision_point": project_precision(pi, tpr, fm),
           "precision_conservative": project_precision(pi, tpr_lo, fm_up),
           "recall_point": tpr, "recall_lower": tpr_lo,
           "fp_per_1k": 1000 * (1 - pi) * fm, "fp_per_million": 1e6 * (1 - pi) * fm,
           "fp_per_million_upper": 1e6 * (1 - pi) * fm_up,
           "alerts_per_million": 1e6 * (pi * tpr + (1 - pi) * fm)}
    return out


def required_fpr_max(target_precision: float, recall: float, pi: float) -> float:
    return pi * recall * (1 - target_precision) / (target_precision * (1 - pi))


def support_diagnostic(rates: dict, pi: float, q: dict, target_precision: float = 0.9) -> dict:
    """Compare the FPR the target would require with what was measured and what
    the denominators can resolve; label supported / limited / extrapolative."""
    proj = projection(rates, pi, q)
    if not proj["available"]:
        return {"prior": pi, "q": q, "support_status": "extrapolative",
                "support_reason": "a subtype with nonzero mixture weight has no held-out examples"}
    r = proj["recall_point"]
    req = required_fpr_max(target_precision, r, pi) if r > 0 else 0.0
    # resolution of the mixture: weighted resolution of the contributing subtypes
    res = sum(w * rates[f"FPR_{s}"]["resolution"] for s, w in q.items() if w > 0)
    zero_fp = all(rates[f"FPR_{s}"]["numerator"] == 0 for s, w in q.items() if w > 0)
    met_point = proj["precision_point"] >= target_precision
    met_cons = proj["precision_conservative"] >= target_precision
    resolved = req >= res and not zero_fp
    if met_cons:
        status, reason = "supported", (f"conservative projected precision {proj['precision_conservative']:.3f} "
                                       f">= {target_precision}")
    elif met_point and resolved:
        status, reason = "limited", (f"point precision {proj['precision_point']:.3f} meets the target but the "
                                     f"conservative projection {proj['precision_conservative']:.3f} does not")
    elif not resolved:
        status, reason = "extrapolative", (f"required mixed FPR {req:.2e} is below the empirical resolution "
                                           f"{res:.2e} or no false positive was observed (zero FPs are not zero risk)")
    else:
        status, reason = "limited", (f"measured mixed FPR {proj['mixed_fpr']:.2e} exceeds the required "
                                     f"{req:.2e}; point precision {proj['precision_point']:.3f}")
    return {"prior": pi, "q": q, "target_precision": target_precision, "required_fpr_max": req,
            "measured_mixed_fpr": proj["mixed_fpr"], "conservative_mixed_fpr_upper": proj["mixed_fpr_upper"],
            "empirical_resolution": res, "zero_observed_fp": zero_fp,
            "target_met_point": bool(met_point), "target_certified_conservative": bool(met_cons),
            "statistically_resolved": bool(resolved), "support_status": status, "support_reason": reason}


def deployment_grid(rates: dict, pis=DEFAULT_PIS, spam_shares=DEFAULT_SPAM_SHARES,
                    spam_split_bulk: float = 0.5, target_precision: float = 0.9) -> list[dict]:
    rows = []
    for pi in pis:
        for sh in spam_shares:
            q = mixture_weights(sh, spam_split_bulk)
            p = projection(rates, pi, q); d = support_diagnostic(rates, pi, q, target_precision)
            rows.append({"prior": pi, "spam_share": sh, "spam_split_bulk": spam_split_bulk, "projection": p,
                         "support": d})
    return rows


# ── Group bootstrap ───────────────────────────────────────────────────────────
def group_bootstrap(y, subtype, score, thr, groups, pis, qs, n_boot=200, seed=42) -> dict:
    """Resample campaign/template GROUPS with replacement; percentile intervals for
    TPR, each subtype FPR, and projected precision at every (pi, q)."""
    y = np.asarray(y).astype(int); st = np.asarray(subtype, dtype=object)
    pred = (np.asarray(score, dtype=float) >= thr).astype(int); g = np.asarray(groups, dtype=object)
    ug = np.unique(g); idx_by_g = {k: np.flatnonzero(g == k) for k in ug}
    rng = np.random.default_rng(seed)
    keys = ["TPR_phishing"] + [f"FPR_{s}" for s in NEGATIVE_SUBTYPES]
    samples = {k: [] for k in keys}
    proj = {f"{pi:g}|{i}": [] for pi in pis for i in range(len(qs))}
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by_g[k] for k in pick])
        yy, ss, pp = y[idx], st[idx], pred[idx]
        r = {}
        m = ss == "phishing"; r["TPR_phishing"] = (pp[m] == 1).mean() if m.any() else np.nan
        for s in NEGATIVE_SUBTYPES:
            m = ss == s; r[f"FPR_{s}"] = (pp[m] == 1).mean() if m.any() else np.nan
        for k in keys:
            samples[k].append(r[k])
        for pi in pis:
            for i, q in enumerate(qs):
                fm = sum(w * r[f"FPR_{s}"] for s, w in q.items() if w > 0)
                proj[f"{pi:g}|{i}"].append(project_precision(pi, r["TPR_phishing"], fm) if not np.isnan(fm) else np.nan)
    def pct(v):
        v = np.asarray(v, dtype=float); v = v[~np.isnan(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]
    return {"method": "group bootstrap (campaign/template groups resampled with replacement)",
            "bootstrap_unit": "campaign_or_template_group", "n_groups": int(len(ug)),
            "replicates": int(n_boot), "seed": int(seed),
            "rates_ci95": {k: pct(v) for k, v in samples.items()},
            "projected_precision_ci95": {k: pct(v) for k, v in proj.items()},
            "limitations": "percentile intervals; groups are the template-prefix key unless campaign "
                           "metadata was supplied; residual within-source dependence is not modelled"}


# ── Calibration ───────────────────────────────────────────────────────────────
class PlattCalibrator:
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(C=1e6, max_iter=1000)
    def fit(self, s, y):
        s = np.asarray(s, dtype=float).reshape(-1, 1); self.lr.fit(s, np.asarray(y).astype(int)); return self
    def __call__(self, s):
        return self.lr.predict_proba(np.asarray(s, dtype=float).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self.ir = IsotonicRegression(out_of_bounds="clip")
    def fit(self, s, y):
        self.ir.fit(np.asarray(s, dtype=float), np.asarray(y).astype(int)); return self
    def __call__(self, s):
        return self.ir.predict(np.asarray(s, dtype=float))


def calibration_metrics(y, p, n_bins: int = 10) -> dict:
    """Brier score, ECE (equal-width bins, |acc-conf| weighted by bin mass) and
    reliability-diagram data."""
    y = np.asarray(y).astype(int); p = np.clip(np.asarray(p, dtype=float), 0, 1)
    edges = np.linspace(0, 1, n_bins + 1); bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0; rel = []
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            rel.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": 0}); continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(acc - conf)
        rel.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                    "mean_confidence": conf, "empirical_positive_rate": acc})
    return {"brier": float(np.mean((p - y) ** 2)), "ece": float(ece),
            "ece_definition": f"{n_bins} equal-width bins, sum_b (n_b/n)*|mean_y_b - mean_p_b|",
            "reliability": rel}


# ── Threshold policies (calibration partition only) ───────────────────────────
def threshold_for_projected_precision(y_cal, st_cal, s_cal, target, pi, q, conservative=False):
    """Lowest calibration-score threshold whose PROJECTED precision at (pi, q)
    meets the target, enumerating every unique score exactly."""
    s = np.asarray(s_cal, dtype=float)
    cands = np.unique(s)
    best = None
    for thr in cands:                               # ascending: first hit is the lowest
        r = conditional_rates(y_cal, st_cal, s, thr)
        p = projection(r, pi, q)
        if not p["available"]:
            return None, {"attainable": False, "reason": "mixture subtype absent from calibration data"}
        v = p["precision_conservative"] if conservative else p["precision_point"]
        if v >= target and r["TPR_phishing"]["numerator"] > 0:
            best = float(thr); break
    return best, {"attainable": best is not None, "candidates": int(len(cands))}


def select_thresholds(y_cal, st_cal, s_cal, *, recall_target=0.9, precision_target=0.9,
                      pi_for_precision=0.01, q_for_precision=None) -> dict:
    q = q_for_precision or mixture_weights(0.5)
    out = {"fixed@0.50": {"threshold": 0.5, "attainable": True, "policy": "fixed"}}
    t = threshold_for_recall(y_cal, s_cal, recall_target)
    out[f"cal-recall>={recall_target}"] = {"threshold": t, "attainable": t is not None,
                                          "policy": "highest calibration threshold with recall >= target"}
    for cons in (False, True):
        t, d = threshold_for_projected_precision(y_cal, st_cal, s_cal, precision_target, pi_for_precision, q, cons)
        name = f"cal-{'conservative-' if cons else ''}proj-precision>={precision_target}@pi={pi_for_precision:g}"
        out[name] = {"threshold": t, **d, "policy": ("lowest calibration threshold whose "
                     f"{'conservative ' if cons else ''}projected precision at pi={pi_for_precision:g}, q={q} >= target")}
    out["_meta"] = {"n_cal": int(len(np.asarray(y_cal))), "n_cal_pos": int(np.asarray(y_cal).sum()),
                    "candidate_thresholds": int(len(np.unique(np.asarray(s_cal, dtype=float)))),
                    "q_for_precision": q, "pi_for_precision": pi_for_precision}
    return out
