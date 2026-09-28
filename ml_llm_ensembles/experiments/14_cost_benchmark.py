#!/usr/bin/env python3
# Experiment 14: measured inference cost for the Table 6 model families.
# Measures single-record latency, throughput, hardware use, and dollar cost per
# million records. This experiment does not compute AUCPR and does not change any
# accuracy result; it fills the cost columns the paper previously left unmeasured.
#
# The prediction cache is deliberately bypassed: a cache hit returns a stored
# label in microseconds, so timing anything through classify_with_cache would
# measure a dict lookup instead of inference.

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# run_all.sh sources .env for the numbered suite, but this experiment is run on its
# own, so it loads the file itself. Existing environment variables win.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_ROOT / ".env")

from ml_llm_ensembles.utils.cost import (
    API_PRICES, DEFAULT_CPU_HOURLY_USD, DEFAULT_GPU_HOURLY_USD,
    api_dollars_per_million, local_dollars_per_million,
)
from ml_llm_ensembles.utils.models import MODERNBERT_MODEL, train_xgb, _texts_fingerprint

SEED = 42
WARMUP = 5
CACHE_DIR = _ROOT / "results" / "cache"
DECODERS = ["llama3.2", "mistral", "gemma3:12b", "gpt-oss"]
CLAUDE = "claude-sonnet-4-6"
MAX_TEXT_LENGTH = 1000   # mirrors utils.prompts, applied to every decoder input


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", choices=["phishing", "flows"], default="phishing")
    p.add_argument("--data-dir", default="data/kitsune")
    p.add_argument("--n", type=int, default=300, help="records timed per configuration")
    p.add_argument("--train-n", type=int, default=5000,
                   help="rows used to fit the XGBoost heads; inference latency is "
                        "independent of this, only the fixed tree shape matters")
    p.add_argument("--configs", nargs="*", default=None,
                   help="subset of config names; default is every local config plus --claude")
    p.add_argument("--decoders", nargs="*", default=DECODERS)
    p.add_argument("--claude", action="store_true",
                   help="include the metered frontier API (requires ALLOW_LIVE_LLM_CALLS=1)")
    p.add_argument("--gemini", action="store_true",
                   help="include the second frontier API (requires ALLOW_LIVE_LLM_CALLS=1)")
    p.add_argument("--gemini-model", default="gemini-3.1-pro-preview",
                   help="the Gemini model the paper's decoder panel used")
    p.add_argument("--gemini-price", nargs=2, type=float, metavar=("IN", "OUT"),
                   help="USD per million input and output tokens. Required with --gemini: "
                        "no Gemini list price is hardcoded, so it cannot go stale unnoticed")
    p.add_argument("--ft-dir", type=Path, default=None,
                   help="fine-tuned ModernBERT checkpoint; defaults per domain under models/")
    p.add_argument("--xgb-device", default="cpu",
                   help="Table 6 claims XGBoost classifies on CPU; benchmark that claim")
    p.add_argument("--concurrency", type=int, default=1,
                   help="in-flight requests. Now always 1 to measure honest latency.")
    p.add_argument("--gpu-hourly-usd", type=float, default=DEFAULT_GPU_HOURLY_USD)
    p.add_argument("--cpu-hourly-usd", type=float, default=DEFAULT_CPU_HOURLY_USD,
                   help="rate for the CPU-only rows; Table 6 classifies XGBoost as CPU/edge")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", type=Path, default=Path(__file__).with_suffix(".json"))
    args = p.parse_args()
    if args.gemini and not args.gemini_price:
        p.error("--gemini needs --gemini-price <USD/Mtok in> <USD/Mtok out>. No Gemini list "
                "price is hardcoded: a stale rate would become a wrong dollar figure in the "
                "paper with nothing to flag it. Check the vendor's current pricing page.")
    return args


# ── Hardware probe ────────────────────────────────────────────────────────────

class HardwareProbe:
    """Samples device VRAM, system CPU, and our RSS on a background thread.

    VRAM is read device-wide rather than per-process because Ollama serves the
    decoders from a separate process; the figure therefore includes whatever else
    occupies the GPU, and the idle baseline is recorded so it can be subtracted.
    """

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self.rss_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_vram_mb = _vram_mb()

    def _loop(self):
        import psutil
        proc = psutil.Process()
        psutil.cpu_percent()          # first call primes the interval, returns 0.0
        # A sub-second config never reaches the loop below, so take one sample up
        # front. RSS included: without it such a config reports 0.0 MB resident,
        # which reads as a measurement rather than as the absence of one.
        self.samples.append((_vram_mb(), 0.0))
        self.rss_mb = proc.memory_info().rss / 1e6
        while not self._stop.wait(self.interval):
            self.samples.append((_vram_mb(), psutil.cpu_percent()))
            self.rss_mb = max(self.rss_mb, proc.memory_info().rss / 1e6)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval * 2)
        vram = [v for v, _ in self.samples]
        cpu = [c for _, c in self.samples]
        peak = max(vram) if vram else None
        return {
            "peak_vram_mb": round(peak, 1) if peak is not None else None,
            "baseline_vram_mb": round(self.baseline_vram_mb, 1),
            # Device-wide peak less the pre-load reading: this model's own footprint.
            "vram_delta_mb": round(peak - self.baseline_vram_mb, 1) if peak is not None else None,
            "mean_cpu_percent": round(sum(cpu) / len(cpu), 1) if cpu else None,
            "peak_rss_mb": round(self.rss_mb, 1),
        }


# ponytail: device-level nvidia-smi sampling. It attributes a decoder correctly
# (Ollama is a separate process) but understates the SECOND torch model measured in
# this process: PyTorch's caching allocator serves those weights from blocks it has
# already reserved from the driver, so the device total never rises. Upgrade path if
# the encoder footprints need to be exact: run one config per subprocess.
def _vram_mb() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0.0


# ── Timing ────────────────────────────────────────────────────────────────────

def measure(name: str, fn, items: list, concurrency: int, batch_fn=None) -> dict:
    """Time fn over items and return latency, throughput, and hardware use.

    fn returns either None or a {"input_tokens", "output_tokens"} dict. Warm-up
    records are timed but discarded so that weight loading, CUDA context setup,
    and the Ollama model load do not land in the reported distribution.

    The probe is constructed before the warm-up so that its VRAM baseline is read
    while this config's weights are still unloaded; peak minus baseline is then
    this model's own footprint rather than whatever else shares the device.

    Note: concurrency is now ignored. All items are processed sequentially to
    measure honest single-record latency and throughput.
    """
    import numpy as np

    probe = HardwareProbe()
    print(f"\n── {name}: warming up on {WARMUP} records ...")
    for x in items[:WARMUP]:
        fn(x)

    probe.start()
    t0 = time.perf_counter()
    # Sequential processing - no concurrency
    results = []
    for x in items:
        results.append(_timed(fn, x))
    wall = time.perf_counter() - t0
    hardware = probe.stop()

    lat = np.array([dt for dt, _ in results])
    toks = [tk for _, tk in results if tk]
    row = {
        "n": len(items),
        "concurrency": 1,  # Always sequential now
        "latency_p50_s": round(float(np.percentile(lat, 50)), 4),
        "latency_p95_s": round(float(np.percentile(lat, 95)), 4),
        "latency_mean_s": round(float(lat.mean()), 4),
        "wall_s": round(wall, 2),
        "throughput_rps": round(len(items) / wall, 3),
        "hardware": hardware,
    }
    # A backend may report a count as None rather than omitting it: Ollama nulls
    # prompt_eval_count when the prompt prefix was served from its cache. Average over
    # the counts that exist and record how many were missing, rather than folding the
    # gaps in as zeros, which would silently understate the mean.
    if toks:
        ins = [t["input_tokens"] for t in toks if t["input_tokens"] is not None]
        outs = [t["output_tokens"] for t in toks if t["output_tokens"] is not None]
        if ins:
            row["mean_input_tokens"] = round(sum(ins) / len(ins), 1)
        if outs:
            row["mean_output_tokens"] = round(sum(outs) / len(outs), 1)
        row["n_input_counts_missing"] = len(toks) - len(ins)
        row["n_output_counts_missing"] = len(toks) - len(outs)

    if batch_fn is not None:
        t0 = time.perf_counter()
        batch_fn(items)
        batched = time.perf_counter() - t0
        row["batched_throughput_rps"] = round(len(items) / batched, 1)

    print(f"   p50 {row['latency_p50_s']:.4f}s | p95 {row['latency_p95_s']:.4f}s | "
          f"{row['throughput_rps']:.2f} rec/s | VRAM +{hardware['vram_delta_mb']} MB")
    return row


def _timed(fn, x) -> tuple[float, dict | None]:
    t0 = time.perf_counter()
    out = fn(x)
    return time.perf_counter() - t0, out if isinstance(out, dict) else None


# ── Per-family inference callables ────────────────────────────────────────────

def ollama_fn(model: str, prompt_template: str):
    import ollama

    def run(text: str):
        r = ollama.chat(
            model=model,
            messages=[{"role": "user",
                       "content": prompt_template.format(text=text[:MAX_TEXT_LENGTH])}],
            options={"temperature": 0.1},
        )
        # Passed through as-is, None included: measure() tracks missing counts.
        return {"input_tokens": r.get("prompt_eval_count"),
                "output_tokens": r.get("eval_count")}
    return run


def gemini_fn(model: str, prompt_template: str):
    # Same reasoning as claude_fn: the shared helper discards usage_metadata, and
    # going through the cache would time a dict lookup.
    from ml_llm_ensembles.utils.prompts import _get_gemini_client
    client = _get_gemini_client()

    def run(text: str):
        r = client.models.generate_content(
            model=model, contents=prompt_template.format(text=text[:MAX_TEXT_LENGTH]),
        )
        u = r.usage_metadata
        return {"input_tokens": u.prompt_token_count,
                "output_tokens": u.candidates_token_count}
    return run


def claude_fn(model: str, prompt_template: str):
    # Calls the client directly rather than through classify_with_cache: the cache
    # would short-circuit the call, and usage tokens are needed for the dollar column.
    from ml_llm_ensembles.utils.prompts import _get_claude_client
    client = _get_claude_client()

    def run(text: str):
        r = client.messages.create(
            model=model, max_tokens=100,
            messages=[{"role": "user",
                       "content": prompt_template.format(text=text[:MAX_TEXT_LENGTH])}],
        )
        return {"input_tokens": r.usage.input_tokens,
                "output_tokens": r.usage.output_tokens}
    return run


def modernbert_ft_fn(ft_dir: Path):
    # Deliberately not pre-loaded: _classify_modernbert_ft loads on first call, so the
    # weights land inside the measure window where the VRAM probe can attribute them.
    from ml_llm_ensembles.utils.prompts import _classify_modernbert_ft

    def run(text: str):
        _classify_modernbert_ft(text, str(ft_dir))
    return run


def modernbert_ft_batch_fn(ft_dir: Path, batch_size: int = 32):
    import torch
    from ml_llm_ensembles.utils import prompts as P

    def run(texts: list[str]):
        # Loaded here, not at construction: this runs after the sequential pass, which
        # has already loaded the weights inside the measure window where they belong.
        P._load_modernbert_ft(str(ft_dir))
        for i in range(0, len(texts), batch_size):
            chunk = [t[:MAX_TEXT_LENGTH] for t in texts[i:i + batch_size]]
            enc = P._modernbert_ft_tokenizer(
                chunk, return_tensors="pt", truncation=True, max_length=512, padding=True,
            ).to(P._modernbert_ft_device)
            with torch.no_grad():
                P._modernbert_ft_model(**enc)
        # CUDA kernels are queued asynchronously. The single-record path syncs
        # implicitly by reading the probability back; this one never reads its
        # output, so without an explicit sync it would time queue submission.
        if P._modernbert_ft_device == "cuda":
            torch.cuda.synchronize()
    return run


def lazy_frozen_bert_encoder():
    """Defer the encoder load to the first call, for the same reason as above."""
    holder: dict = {}

    def encode(texts: list[str]):
        if "fn" not in holder:
            holder["fn"] = frozen_bert_encoder()
        return holder["fn"](texts)
    return encode


def frozen_bert_encoder():
    """Single-text frozen ModernBERT encode, mean-pooled — the encoder half of the
    'XGBoost on frozen BERT' row. build_modernbert_features only encodes in bulk."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODERNBERT_MODEL)
    model = AutoModel.from_pretrained(MODERNBERT_MODEL).to(device).eval()

    def encode(texts: list[str]):
        enc = tok(texts, return_tensors="pt", truncation=True, max_length=512,
                  padding=True).to(device)
        with torch.no_grad():
            out = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        return ((out * mask).sum(1) / mask.sum(1)).cpu().numpy()
    return encode


# ── Data ──────────────────────────────────────────────────────────────────────

def load_domain(args):
    """Returns (train_df, sample_df, decoder_texts, ft_texts, X_train, X_sample)."""
    import numpy as np
    import pandas as pd

    if args.domain == "phishing":
        from ml_llm_ensembles.utils.datasets import load_phishing_dataset
        from ml_llm_ensembles.utils.features import build_phishing_email_feature_matrix
        from ml_llm_ensembles.utils.prompts import DOMAIN_PROMPTS

        df = load_phishing_dataset("zefang-liu")
        train_df, sample_df = _split(df, args)
        X_train = build_phishing_email_feature_matrix(train_df).values
        X_sample = build_phishing_email_feature_matrix(sample_df).values
        texts = sample_df["text"].astype(str).tolist()
        return train_df, sample_df, texts, texts, X_train, X_sample, DOMAIN_PROMPTS["phishing"]

    from ml_llm_ensembles.utils.datasets import load_network_dataset
    from ml_llm_ensembles.utils.prompts import (
        DOMAIN_PROMPTS, KITSUNE_PCAP_FLOW_FEATURE_COLS,
        format_network_row_kitsune_pcap_flow, format_network_row_kitsune_pcap_flow_ft,
    )
    from sklearn.impute import SimpleImputer

    # The pcap parse costs minutes; cache the aggregated flows so repeat runs are cheap.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    flows_cache = CACHE_DIR / "bench_flows_mirai.parquet"
    if flows_cache.exists():
        df = pd.read_parquet(flows_cache)
    else:
        df = load_network_dataset("kitsune-mirai-pcap-flows", args.data_dir)
        df.to_parquet(flows_cache)

    train_df, sample_df = _split(df, args)
    cols = [c for c in KITSUNE_PCAP_FLOW_FEATURE_COLS if c in train_df.columns]

    def numeric(d):
        return (d[cols].apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan))

    imp = SimpleImputer(strategy="median").fit(numeric(train_df))
    decoder_texts = [format_network_row_kitsune_pcap_flow(r) for _, r in sample_df.iterrows()]
    ft_texts = [format_network_row_kitsune_pcap_flow_ft(r) for _, r in sample_df.iterrows()]
    return (train_df, sample_df, decoder_texts, ft_texts,
            imp.transform(numeric(train_df)), imp.transform(numeric(sample_df)),
            DOMAIN_PROMPTS["network"])


def _split(df, args):
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=df["label"])
    train_df = train_df.sample(n=min(args.train_n, len(train_df)),
                               random_state=args.seed).reset_index(drop=True)
    sample_df = test_df.sample(n=min(args.n, len(test_df)),
                               random_state=args.seed).reset_index(drop=True)
    return train_df, sample_df


def main():
    args = parse_args()
    import gc

    import numpy as np
    import torch

    np.random.seed(args.seed)
    if args.ft_dir is None:
        args.ft_dir = _ROOT / "models" / (
            "modernbert-phishing-ft" if args.domain == "phishing"
            else "modernbert-mirai-flows-ft")

    print(f"Loading {args.domain} ...")
    (train_df, sample_df, decoder_texts, ft_texts,
     X_train, X_sample, prompt) = load_domain(args)
    y_train = train_df["label"].values
    print(f"Fit rows {len(train_df)} | timed records {len(sample_df)}")

    # ── Build the configuration table ────────────────────────────────────────
    configs: dict[str, tuple] = {}   # name -> (fn, items, batch_fn, family)

    xgb = train_xgb(X_train, y_train, random_state=args.seed)
    xgb.set_params(device=args.xgb_device)
    configs["xgboost"] = (
        lambda r: xgb.predict_proba(r),
        [X_sample[i:i + 1] for i in range(len(X_sample))],
        lambda rows: xgb.predict_proba(X_sample),
        "local-cpu" if args.xgb_device == "cpu" else "local-gpu",
    )

    # The frozen-BERT head is fitted on the same serialization its row is timed on.
    # Fitting needs the encoder now, but the timed row must load it itself, so this
    # instance is dropped and the config gets a fresh lazy one.
    fit_encode = frozen_bert_encoder()
    head_texts = (train_df["text"].astype(str).tolist() if args.domain == "phishing"
                  else _flow_train_texts(train_df))
    emb_train = np.vstack([fit_encode(c) for c in _chunks(head_texts, 32)])
    del fit_encode
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    encode = lazy_frozen_bert_encoder()
    xgb_bert = train_xgb(emb_train, y_train, random_state=args.seed)
    xgb_bert.set_params(device=args.xgb_device)
    configs["xgboost-frozen-bert"] = (
        lambda t: xgb_bert.predict_proba(encode([t])),
        decoder_texts,
        lambda ts: [xgb_bert.predict_proba(encode(c)) for c in _chunks(ts, 32)],
        "local-gpu",
    )

    if args.ft_dir.exists():
        configs["modernbert-ft"] = (
            modernbert_ft_fn(args.ft_dir), ft_texts,
            modernbert_ft_batch_fn(args.ft_dir), "local-gpu")
    else:
        print(f"  [skip] modernbert-ft: no checkpoint at {args.ft_dir}")

    for m in args.decoders:
        configs[m] = (ollama_fn(m, prompt), decoder_texts, None, "local-gpu")

    if args.claude:
        if os.environ.get("ALLOW_LIVE_LLM_CALLS") != "1":
            sys.exit("--claude bills a metered API. Set ALLOW_LIVE_LLM_CALLS=1 to proceed.")
        configs[CLAUDE] = (claude_fn(CLAUDE, prompt), decoder_texts, None, "api")

    if args.gemini:
        if os.environ.get("ALLOW_LIVE_LLM_CALLS") != "1":
            sys.exit("--gemini bills a metered API. Set ALLOW_LIVE_LLM_CALLS=1 to proceed.")
        configs[args.gemini_model] = (
            gemini_fn(args.gemini_model, prompt), decoder_texts, None, "api")

    if args.configs:
        configs = {k: v for k, v in configs.items() if k in args.configs}

    prices = dict(API_PRICES)
    if args.gemini_price:
        prices[args.gemini_model] = tuple(args.gemini_price)

    # ── Measure ──────────────────────────────────────────────────────────────
    results = {}
    for name, (fn, items, batch_fn, family) in configs.items():
        if name in args.decoders:
            _unload_ollama()
        row = measure(name, fn, items, args.concurrency, batch_fn)
        if family == "api":
            # The probe measures this machine, which for a hosted model is just the
            # harness: its torch allocations and resident set, plus whatever else shares
            # the GPU. Attributing that to the API is wrong in both directions, and the
            # VRAM delta is small enough to come out negative on noise, so drop it.
            row["hardware"] = {"note": "hosted API: no local accelerator; local CPU and "
                                       "memory readings are the harness, not this model"}
            if "mean_input_tokens" not in row or "mean_output_tokens" not in row:
                sys.exit(f"{name}: the API reported no token usage, so this row cannot be "
                         f"priced. The calls were made and billed; do not retry blindly.")
            price_in, price_out = prices[name]
            usd = api_dollars_per_million(row["mean_input_tokens"],
                                          row["mean_output_tokens"], price_in, price_out)
            row["usd_per_million_records"] = float(f"{usd:.4g}")
            row["cost_basis"] = f"metered tokens at ${price_in}/${price_out} per Mtok"
        else:
            rate = args.cpu_hourly_usd if family == "local-cpu" else args.gpu_hourly_usd
            unit = "CPU-hour" if family == "local-cpu" else "GPU-hour"
            # Use single-record throughput for cost calculation (not batched)
            rps = row["throughput_rps"]
            row["usd_per_million_records"] = float(
                f"{local_dollars_per_million(rps, rate):.4g}")
            row["cost_basis"] = f"amortized at ${rate}/{unit}"
        results[name] = row

    out = {
        "domain": args.domain,
        "seed": args.seed,
        "n_timed": len(sample_df),
        "sample_fingerprint": _texts_fingerprint(decoder_texts),
        "xgb_device": args.xgb_device,
        "gpu_hourly_usd": args.gpu_hourly_usd,
        "cpu_hourly_usd": args.cpu_hourly_usd,
        "results": results,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")
    _print_table(results)


def _unload_ollama(settle_s: float = 5.0):
    """Evict every model Ollama currently holds, before any decoder is timed.

    Whatever is actually loaded, not just this run's roster: a model left resident by
    an earlier run is still evicted by Ollama to make room for the new one, which
    inflates the new model's VRAM baseline and can make its measured delta negative.
    The settle matters for the same reason — eviction is asynchronous."""
    ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=60)
    for line in ps.stdout.strip().splitlines()[1:]:      # skip the header row
        if line.split():
            subprocess.run(["ollama", "stop", line.split()[0]], capture_output=True, timeout=60)
    time.sleep(settle_s)


def _chunks(xs, k):
    for i in range(0, len(xs), k):
        yield xs[i:i + k]


def _flow_train_texts(train_df):
    from ml_llm_ensembles.utils.prompts import format_network_row_kitsune_pcap_flow
    return [format_network_row_kitsune_pcap_flow(r) for _, r in train_df.iterrows()]


def _print_table(results: dict):
    print(f"\n{'model':<24} {'p50 s':>8} {'p95 s':>8} {'rec/s':>9} "
          f"{'batch rec/s':>12} {'VRAM +MB':>9} {'USD/1M':>10}")
    for name, r in results.items():
        batch = r.get("batched_throughput_rps")
        batch_s = f"{batch:.1f}" if batch else "--"
        vram = r["hardware"].get("vram_delta_mb")
        vram_s = "n/a" if vram is None else f"{vram:.0f}"
        print(f"{name:<24} {r['latency_p50_s']:>8.4f} {r['latency_p95_s']:>8.4f} "
              f"{r['throughput_rps']:>9.2f} {batch_s:>12} {vram_s:>9} "
              f"{r['usd_per_million_records']:>10.4g}")


if __name__ == "__main__":
    main()
