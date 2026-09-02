#!/usr/bin/env python3
"""Inspect or populate the local decoder-prediction cache.

The utility reconstructs each experiment's texts, prompt, seed-42 split, and
sampling policy so cache keys match the experiment entry points. It supports
local Ollama models only; API-backed models are excluded. The default is
read-only. Passing --execute explicitly authorizes local inference and cache
writes. Existing entries are retained.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from ml_llm_ensembles.utils.datasets import (
    load_phishing_dataset, strip_provenance, load_network_dataset,
)
from ml_llm_ensembles.utils.prompts import (
    DOMAIN_PROMPTS, classify_with_cache, cache_key,
    format_network_row_kitsune_pcap_flow,
    format_network_row_kitsune_pcap,
    format_network_row_cicids,
)

SEED = 42
META_TRAIN = 5000  # matches 06_meta_flows.py
OPEN_MODELS = ["mistral", "gemma3:12b", "gpt-oss", "llama3.2"]
META_NET_MODELS = ["mistral", "gemma3:12b", "llama3.2"]  # roster used by 06/07/09
CACHE_FILE = _ROOT / "results" / "cache" / "llm_cache.json"
PHISH = DOMAIN_PROMPTS["phishing"]
NET = DOMAIN_PROMPTS["network"]


def strat_subset(y, n, seed):
    """Verbatim copy of 06_meta_flows.py's meta-train subsampler (keys must match)."""
    if n >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        k = max(1, round(n * len(ci) / len(y)))
        idx.append(rng.choice(ci, size=min(k, len(ci)), replace=False))
    return np.sort(np.concatenate(idx))


def _phishing(strip):
    df = load_phishing_dataset("zefang-liu")
    if strip:
        df = df.copy()
        df["text"] = df["text"].map(strip_provenance)
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])
    return tr["text"].tolist(), te["text"].tolist(), tr["label"].values


def _flows(data_dir):
    df = load_network_dataset("kitsune-mirai-pcap-flows", data_dir)
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])
    tr_t = [format_network_row_kitsune_pcap_flow(r) for _, r in tr.iterrows()]
    te_t = [format_network_row_kitsune_pcap_flow(r) for _, r in te.iterrows()]
    return tr_t, te_t, tr["label"].values


def _pcap(data_dir):
    df = load_network_dataset("kitsune-mirai-pcap", data_dir)
    df = df.sample(n=5000, random_state=SEED).reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tri, tei = next(gss.split(df, df["label"], groups=df["flow_id"]))
    tr, te = df.iloc[tri], df.iloc[tei]
    tr_t = [format_network_row_kitsune_pcap(r) for _, r in tr.iterrows()]
    te_t = [format_network_row_kitsune_pcap(r) for _, r in te.iterrows()]
    return tr_t, te_t


def _cicids(data_dir):
    df = load_network_dataset("cicids2017", data_dir)
    df = df.sample(n=5000, random_state=SEED).reset_index(drop=True)
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])
    tr_t = [format_network_row_cicids(r) for _, r in tr.iterrows()]
    te_t = [format_network_row_cicids(r) for _, r in te.iterrows()]
    return tr_t, te_t


# name -> builder(args) returning (prompt, models, texts_the_LLM_must_cover)
def build_one(name, args):
    if name == "03_router_phishing_raw":
        _, te, _ = _phishing(False);                 return PHISH, OPEN_MODELS, te
    if name == "03_router_phishing_strip":
        _, te, _ = _phishing(True);                  return PHISH, OPEN_MODELS, te
    if name == "04_router_flows":
        _, te, _ = _flows(args.data_dir);            return NET, OPEN_MODELS, te
    if name == "05_meta_phishing_raw":
        tr, te, _ = _phishing(False);                return PHISH, ["llama3.2"], tr + te
    if name == "05_meta_phishing_strip":
        tr, te, _ = _phishing(True);                 return PHISH, ["llama3.2"], tr + te
    if name == "06_meta_flows":
        tr, te, ytr = _flows(args.data_dir)
        mi = strat_subset(ytr, META_TRAIN, SEED)     # meta-subset train + full test
        return NET, META_NET_MODELS + ["gpt-oss"], [tr[i] for i in mi] + te
    if name == "07_meta_pcap":
        tr, te = _pcap(args.data_dir);               return NET, META_NET_MODELS, tr + te
    if name == "09_meta_cicids":
        tr, te = _cicids(args.cicids_dir);           return NET, META_NET_MODELS, tr + te
    raise ValueError(name)


ALL = ["03_router_phishing_raw", "03_router_phishing_strip", "04_router_flows",
       "05_meta_phishing_raw", "05_meta_phishing_strip", "06_meta_flows",
       "07_meta_pcap", "09_meta_cicids"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiments", nargs="*", default=ALL, choices=ALL, metavar="EXP")
    ap.add_argument("--data-dir", default="data/kitsune")
    ap.add_argument("--cicids-dir", default="data/cicids2017")
    ap.add_argument("--workers", type=int, default=4,
                    help="Concurrent Ollama requests (local, no rate limit).")
    ap.add_argument("--execute", action="store_true",
                    help="Run local Ollama inference for cache misses and write the cache")
    args = ap.parse_args()

    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    print(f"Cache: {CACHE_FILE}  ({len(cache)} entries)\n")

    grand_misses, incomplete = 0, []
    for exp in args.experiments:
        prompt, models, texts = build_one(exp, args)
        print(f"=== {exp}  ({len(texts)} texts, {len(models)} models) ===")
        for model in models:
            keys = [cache_key(t, model, prompt) for t in texts]
            missing = [t for t, k in zip(texts, keys) if k not in cache]
            cov = (len(texts) - len(missing)) / len(texts) if texts else 1.0
            print(f"  {model:14s} coverage {cov:5.1%}  missing={len(missing):5d}", end="")
            grand_misses += len(missing)
            if not missing or not args.execute:
                if missing:
                    incomplete.append((exp, model, len(missing)))
                print()
                continue
            t0 = time.perf_counter()

            def warm_one(text):
                classify_with_cache(text, model, cache, prompt, "ollama", cache_only=False)

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                list(ex.map(warm_one, missing))
            CACHE_FILE.write_text(json.dumps(cache))  # persist after each model
            still = sum(1 for t in missing if cache_key(t, model, prompt) not in cache)
            if still:
                incomplete.append((exp, model, still))
            print(f"  -> filled {len(missing) - still}/{len(missing)} in {time.perf_counter() - t0:.0f}s")
        print()

    print(f"Total cache misses across requested experiments: {grand_misses}")
    if not args.execute:
        print("(inspection only; pass --execute to run local inference)")
        return
    print(f"Cache now has {len(cache)} entries.")
    if incomplete:
        print("\n[FAIL] models still below 100% coverage:")
        for exp, model, n in incomplete:
            print(f"  {exp}: {model} ({n} missing)")
        sys.exit(1)
    print("\n[OK] every requested model is at 100% coverage.")


if __name__ == "__main__":
    main()
