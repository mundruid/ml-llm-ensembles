"""
Leakage-aware train/test split policies for network datasets.
==============================================================
Every policy returns (train_idx, test_idx) as sorted numpy int arrays plus a
diagnostics dict that the experiment scripts persist next to their results, so
the split regime a number was produced under is always on record.

Policies
  random    stratified i.i.d. row split (the leaky reference; only for contrast)
  temporal  ONE global time cutoff T: every training row has time <= T and every
            test row has time > T. The realized class priors on each side are
            whatever the timeline gives; they are reported, not engineered.
  group     GroupShuffleSplit on an arbitrary group column (host, flow, ...)
  scenario  hold out named captures/campaigns entirely (leave-scenarios-out)

OOF fold construction mirrors the outer policy:
  temporal  -> forward-chaining blocked folds (TimeSeriesSplit). Rows in the
               first block never receive an OOF prediction; callers must mask
               them out of stage-2 training rather than fill a placeholder.
  group /
  scenario  -> StratifiedGroupKFold on the supplied groups
  random    -> StratifiedKFold
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit,
    train_test_split,
)


def random_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed,
                              stratify=df["label"].values)
    return np.sort(tr), np.sort(te), {"policy": "random", "leakage_prone": True}


def temporal_split(df: pd.DataFrame, time_col: str, test_size: float = 0.2):
    """Single global chronological cutoff: train = earliest (1-test_size) of ALL
    rows, test = the rest. No per-class cutoffs, so there is one T with
    max(train time) <= T < min(test time). If the attack is concentrated at one
    end of the timeline one side can end up single-class; the caller must check
    the realized priors and record a degenerate run rather than re-splitting."""
    t = df[time_col].values.astype(float)
    order = np.argsort(t, kind="stable")
    cut = int(len(order) * (1 - test_size))
    tr, te = np.sort(order[:cut]), np.sort(order[cut:])
    diag = {"policy": "temporal", "time_col": time_col,
            "cutoff_time": float(t[tr].max()),
            "train_time_max": float(t[tr].max()), "test_time_min": float(t[te].min()),
            "temporal_order_respected": bool(t[tr].max() <= t[te].min()),
            "train_prior": float(df["label"].values[tr].mean()),
            "test_prior": float(df["label"].values[te].mean())}
    return tr, te, diag


def group_split(df: pd.DataFrame, group_col: str, test_size: float = 0.2, seed: int = 42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, df["label"].values, groups=df[group_col].values))
    tr, te = np.sort(tr), np.sort(te)
    g_tr = set(df[group_col].values[tr]); g_te = set(df[group_col].values[te])
    diag = {"policy": "group", "group_col": group_col,
            "n_groups_train": len(g_tr), "n_groups_test": len(g_te),
            "group_overlap": len(g_tr & g_te)}
    return tr, te, diag


def endpoint_overlap_diag(df: pd.DataFrame, tr, te, src_col="src_ip", dst_col="dst_ip"):
    """How much host identity still crosses a split that only grouped on one
    endpoint: fraction of test rows whose src OR dst IP appears anywhere in the
    training rows as src or dst. Reported so 'host-grouped' is never read as
    'host-disjoint'."""
    train_eps = set(df[src_col].values[tr]) | set(df[dst_col].values[tr])
    s = pd.Series(df[src_col].values[te]).isin(train_eps).values
    d = pd.Series(df[dst_col].values[te]).isin(train_eps).values
    return {"test_rows_with_any_endpoint_seen_in_train": float((s | d).mean()),
            "test_rows_with_src_seen_in_train": float(s.mean()),
            "test_rows_with_dst_seen_in_train": float(d.mean())}


def scenario_split(df: pd.DataFrame, scenario_col: str, test_scenarios):
    test_scenarios = [type(df[scenario_col].iloc[0])(s) for s in test_scenarios]
    mask = df[scenario_col].isin(test_scenarios).values
    tr, te = np.flatnonzero(~mask), np.flatnonzero(mask)
    diag = {"policy": "scenario", "scenario_col": scenario_col,
            "train_scenarios": sorted(map(str, set(df[scenario_col].values[tr]))),
            "test_scenarios": sorted(map(str, set(df[scenario_col].values[te])))}
    return tr, te, diag


def make_split(df: pd.DataFrame, policy: str, *, seed: int = 42, test_size: float = 0.2,
               time_col: str = "start_time", group_col: str | None = None,
               scenario_col: str = "scenario", test_scenarios=None):
    if policy == "random":
        return random_split(df, test_size, seed)
    if policy == "temporal":
        return temporal_split(df, time_col, test_size)
    if policy == "group":
        assert group_col, "group policy needs group_col"
        return group_split(df, group_col, test_size, seed)
    if policy == "scenario":
        assert test_scenarios, "scenario policy needs test_scenarios"
        return scenario_split(df, scenario_col, test_scenarios)
    raise ValueError(f"unknown split policy {policy!r}")


def oof_folds(policy: str, X, y, groups=None, n_folds: int = 5, seed: int = 42, times=None):
    """Fold iterator for OOF stacking that respects the outer split regime.

    temporal: forward-chaining blocks on time order (X must be passed with
    `times`; folds are built on the time-sorted order and mapped back). The first
    block gets no validation prediction. group/scenario: StratifiedGroupKFold on
    `groups`. random: StratifiedKFold."""
    if policy == "temporal":
        assert times is not None, "temporal OOF needs times"
        order = np.argsort(np.asarray(times, dtype=float), kind="stable")
        for tr, va in TimeSeriesSplit(n_splits=n_folds).split(order):
            yield order[tr], order[va]
        return
    if policy in ("group", "scenario"):
        assert groups is not None
        yield from StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                        random_state=seed).split(X, y, groups)
        return
    yield from StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X, y)
