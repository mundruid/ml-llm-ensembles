#!/usr/bin/env python3
"""Inspect or populate the local decoder-prediction cache.

The utility reconstructs each experiment's texts, prompt, seed-42 split, and
sampling policy so cache keys match the experiment entry points. It supports
local Ollama models and, for the router experiments only, the System One
models named by --systemone; other API-backed models are excluded. The default
is read-only. Passing --execute explicitly authorizes inference and cache
writes -- for --systemone jev that means billed TypeSafe calls. Existing
entries are retained.
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
    backend_for, classify_with_cache, cache_key, prompt_for,
    format_network_row_kitsune_pcap_flow,
    format_network_row_kitsune_pcap,
    format_network_row_cicids,
)

SEED = 42
META_TRAIN = 5000  # matches 06_meta_flows.py
OPEN_MODELS = ["mistral", "gemma3:12b", "gpt-oss", "llama3.2"]
META_NET_MODELS = ["mistral", "gemma3:12b", "llama3.2"]  # roster used by 06/07/09
CACHE_FILE = _ROOT / "results" / "cache" / "llm_cache.json"
PHISH, NET = "phishing", "network"   # domain, resolved to a prompt per model
# --systemone choice -> model id. Jev is pinned; jev-latest/jev-preview move on
# release, which would silently change cached answers under the same key.
SYSTEMONE_CHOICES = {"jev": "jev-1.13.0", "laya": "convaiinnovations/laya"}
ROUTER_EXPERIMENTS = ("03_router_phishing_raw", "03_router_phishing_strip",
                      "04_router_flows")


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


# name -> builder(args) returning (domain, models, texts_the_LLM_must_cover)
def build_one(name, args):
    extra = ([SYSTEMONE_CHOICES[s] for s in args.systemone]
             if name in ROUTER_EXPERIMENTS else [])
    if name == "03_router_phishing_raw":
        _, te, _ = _phishing(False);                 return PHISH, OPEN_MODELS + extra, te
    if name == "03_router_phishing_strip":
        _, te, _ = _phishing(True);                  return PHISH, OPEN_MODELS + extra, te
    if name == "04_router_flows":
        _, te, _ = _flows(args.data_dir);            return NET, OPEN_MODELS + extra, te
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
    ap.add_argument("--only", nargs="*", default=[], metavar="MODEL",
                    help="Warm only these model ids (e.g. --only jev-1.13.0 "
                         "convaiinnovations/laya); default is every model.")
    ap.add_argument("--systemone", nargs="*", default=[], choices=list(SYSTEMONE_CHOICES),
                    help="Also warm these System One models on the router experiments. "
                         "jev needs TYPESAFE_API_KEY; laya needs laya-serve running.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Concurrent requests. Ollama and laya-serve are local; Jev "
                         "allows 1,200 req/min, and 429s are retried in the client.")
    ap.add_argument("--execute", action="store_true",
                    help="Run inference for cache misses and write the cache "
                         "(with --systemone jev this makes billed API calls)")
    args = ap.parse_args()

    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    print(f"Cache: {CACHE_FILE}  ({len(cache)} entries)\n")

    grand_misses, incomplete = 0, []
    for exp in args.experiments:
        domain, models, texts = build_one(exp, args)
        if args.only:
            models = [m for m in models if m in args.only]
        print(f"=== {exp}  ({len(texts)} texts, {len(models)} models) ===")
        for model in models:
            prompt = prompt_for(model, domain)
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
            print()
            done = 0

            def warm_one(text, model=model, prompt=prompt):
                classify_with_cache(text, model, cache, prompt,
                                    backend_for(model), cache_only=False)

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for _ in ex.map(warm_one, missing):
                    done += 1
                    # a model can take ~25 min here; without a tick the run
                    # looks hung and gets killed mid-way.
                    if done % 25 == 0 or done == len(missing):
                        el = time.perf_counter() - t0
                        eta = el / done * (len(missing) - done)
                        print(f"\r    {done}/{len(missing)}  {el:.0f}s elapsed  "
                              f"~{eta / 60:.0f}m left", end="", flush=True)
            CACHE_FILE.write_text(json.dumps(cache))  # persist after each model
            still = sum(1 for t in missing if cache_key(t, model, prompt) not in cache)
            if still:
                incomplete.append((exp, model, still))
            print(f"\n  -> filled {len(missing) - still}/{len(missing)} in {time.perf_counter() - t0:.0f}s")
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
