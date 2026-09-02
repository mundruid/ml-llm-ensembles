"""
Threshold-based operational evaluation.
=======================================
AUCPR is interpreted with the prevalence of its evaluation set. Deployment also
requires threshold metrics and an estimate of alert burden. This module reports:

  * precision / recall / FPR (with Wilson 95% intervals) at a fixed threshold,
    and at thresholds chosen on TRAINING (OOF) scores for a target recall or
    precision, never on test. Thresholds are found by exact enumeration of the
    unique training scores; an unattainable target is reported as such, not
    silently replaced;
  * false-positive burden: FPs per 1,000 flows and per hour of capture. When the
    scored rows are a uniform fraction f of the capture, pass
    exposure_fraction=f and the per-hour figures are divided by f;
  * the same operating point PROJECTED to another class prior pi from the
    class-conditional TPR and FPR:
        precision(pi) = pi*TPR / (pi*TPR + (1-pi)*FPR).
    This is exact under prior-probability shift with representative within-
    class sampling, so a class-balanced LLM subset yields a precision ESTIMATE
    at an observed or specified prior. It is labelled as a projection, carries the
    subset's Wilson intervals, and does not produce a projected AUCPR
    (that would need every PR operating point re-weighted).

Nothing here subsamples the data. The observed prevalence of a capture is
reported as-is, and every alternate-prior result is labelled as a projection.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _rates(y, score, thr):
    y = np.asarray(y).astype(int); pred = (np.asarray(score) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return {"threshold": float(thr), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": float(prec), "precision_ci95": wilson(tp, tp + fp),
            "recall": float(tpr), "recall_ci95": wilson(tp, tp + fn),
            "fpr": float(fpr), "fpr_ci95": wilson(fp, fp + tn),
            "alerts": tp + fp}


def threshold_for_recall(y_train, score_train, target: float = 0.9):
    """HIGHEST threshold whose training recall >= target (exact order statistic).
    Returns None if there are no positives."""
    y = np.asarray(y_train); s = np.asarray(score_train, dtype=float)
    pos = np.sort(s[y == 1])[::-1]              # descending
    if len(pos) == 0:
        return None
    k = int(math.ceil(target * len(pos)))       # need at least k positives at/above thr
    return float(pos[min(k, len(pos)) - 1])


def threshold_for_precision(y_train, score_train, target: float = 0.9):
    """LOWEST threshold (max recall) whose training precision >= target, over all
    unique score values exactly. Returns None if unattainable."""
    y = np.asarray(y_train).astype(int); s = np.asarray(score_train, dtype=float)
    order = np.argsort(-s, kind="stable"); ss = s[order]; yy = y[order]
    tp_cum = np.cumsum(yy); n_cum = np.arange(1, len(yy) + 1)
    # evaluate at the last index of each unique score (threshold = that score)
    last_of_value = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])
    prec = tp_cum[last_of_value] / n_cum[last_of_value]
    ok = np.flatnonzero(prec >= target)
    if len(ok) == 0:
        return None
    return float(ss[last_of_value[ok[-1]]])     # lowest qualifying threshold


def project_to_prior(tpr: float, fpr: float, pi: float, flows_per_hour: float | None = None):
    """Precision / alert volume implied by (TPR, FPR) at class prior pi."""
    denom = pi * tpr + (1 - pi) * fpr
    prec = (pi * tpr / denom) if denom > 0 else 0.0
    out = {"prior": float(pi), "precision": float(prec), "recall": float(tpr),
           "fpr": float(fpr), "fp_per_1k_flows": float(1000 * (1 - pi) * fpr),
           "alerts_per_1k_flows": float(1000 * denom)}
    if flows_per_hour:
        out["fp_per_hour"] = float(flows_per_hour * (1 - pi) * fpr)
        out["alerts_per_hour"] = float(flows_per_hour * denom)
    return out


def operational_report(y_test, score_test, *, y_train=None, score_train=None,
                       hours: float | None = None, exposure_fraction: float = 1.0,
                       priors=(0.01, 0.001), recall_target: float = 0.9,
                       precision_target: float = 0.9, fixed_threshold: float = 0.5,
                       representative_test: bool = True, evaluation_prior: float | None = None):
    """Full operational block for one scorer on one test set.

    y_train/score_train (OOF or held-in training scores) pick the operating
    thresholds; when absent only the fixed threshold is reported. `hours` is the
    wall-clock span of the capture the test rows come from; `exposure_fraction`
    is the fraction of that capture's rows actually scored (uniform subsample).
    representative_test=False marks a class-balanced or otherwise non-
    representative subset: its AUCPR is then stored as aucpr_subset and its
    natural-prior numbers exist only inside `projected`."""
    y = np.asarray(y_test).astype(int); s = np.asarray(score_test, dtype=float)
    n = len(y); pi_eval = float(y.mean())
    fph = (n / exposure_fraction / hours) if hours else None
    has_both = len(set(y.tolist())) > 1
    ap = float(average_precision_score(y, s)) if has_both else None
    roc = float(roc_auc_score(y, s)) if has_both else None
    rep = {"n": int(n), "n_pos": int(y.sum()), "n_neg": int(n - y.sum()),
           "representative_test": bool(representative_test),
           "evaluation_prior": pi_eval if evaluation_prior is None else float(evaluation_prior),
           "hours": hours, "exposure_fraction": float(exposure_fraction),
           "rocauc": roc, "operating_points": {}}
    if representative_test:
        rep["aucpr"] = ap; rep["natural_prior"] = pi_eval
    else:
        rep["aucpr_subset"] = ap; rep["aucpr"] = None
        rep["note"] = "non-representative (class-balanced) subset: aucpr_subset is NOT natural-prior AUCPR"
    points = {"fixed@%.2f" % fixed_threshold: (fixed_threshold, True)}
    if y_train is not None and score_train is not None:
        tr_ = threshold_for_recall(y_train, score_train, recall_target)
        tp_ = threshold_for_precision(y_train, score_train, precision_target)
        points[f"train-recall>={recall_target}"] = (tr_, tr_ is not None)
        points[f"train-precision>={precision_target}"] = (tp_, tp_ is not None)
    for name, (thr, attainable) in points.items():
        if not attainable:
            rep["operating_points"][name] = {"attainable": False, "threshold": None}
            continue
        r = _rates(y, s, thr); r["attainable"] = True
        r["fp_per_1k_flows"] = 1000 * r["fp"] / n
        if fph:
            r["fp_per_hour"] = r["fp"] / exposure_fraction / hours
            r["alerts_per_hour"] = r["alerts"] / exposure_fraction / hours
        r["projected"] = {f"{p:g}": project_to_prior(r["recall"], r["fpr"], p, fph) for p in priors}
        if representative_test:
            # self-consistency: projection at the observed evaluation prior must reproduce the
            # measured precision (exact identity up to float rounding)
            nat = project_to_prior(r["recall"], r["fpr"], pi_eval, fph)
            if r["alerts"] > 0 and abs(nat["precision"] - r["precision"]) > 1e-6:
                raise AssertionError(f"projection self-check failed at {name}: "
                                     f"{nat['precision']} vs {r['precision']}")
            r["projected"]["natural"] = nat
        rep["operating_points"][name] = r
    return rep


def format_operational_table(name: str, rep: dict) -> str:
    a = rep.get("aucpr") if rep.get("representative_test", True) else rep.get("aucpr_subset")
    tag = "" if rep.get("representative_test", True) else " [balanced subset]"
    lines = [f"  {name}{tag}: n={rep['n']} pos={rep['n_pos']} eval-prior={rep['evaluation_prior']:.4f} "
             f"AUCPR={'n/a' if a is None else round(a, 4)}"]
    for pt, r in rep["operating_points"].items():
        if not r.get("attainable", True):
            lines.append(f"    {pt:24s} unattainable on training scores"); continue
        proj = "  ".join(f"@{k}: P={v['precision']:.3f} FP/1k={v['fp_per_1k_flows']:.2f}"
                         for k, v in r["projected"].items() if k != "natural")
        lo, hi = r["precision_ci95"]
        lines.append(f"    {pt:24s} thr={r['threshold']:.3f} P={r['precision']:.3f} [{lo:.3f},{hi:.3f}] "
                     f"R={r['recall']:.3f} FPR={r['fpr']:.5f} FP/1k={r['fp_per_1k_flows']:.2f}"
                     + (f" FP/h={r['fp_per_hour']:.1f}" if 'fp_per_hour' in r else "")
                     + f"  | {proj}")
    return "\n".join(lines)
