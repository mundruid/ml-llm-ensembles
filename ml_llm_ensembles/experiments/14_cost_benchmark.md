# Experiment 14: measured inference cost for Table 6

Table 6 compares the model families by cost *structure*; its caption states that wall-clock
latency and dollar cost were not benchmarked. This experiment measures them. It computes no
AUCPR and changes no accuracy result: it fills the cost columns only.

## What is measured

For each configuration, on a fixed sample of test records:

| Quantity | Definition |
| --- | --- |
| `latency_p50_s`, `latency_p95_s` | single-record wall-clock, after a discarded warm-up |
| `throughput_rps` | records per second in the timed regime (`--concurrency` in flight) |
| `batched_throughput_rps` | records per second when the whole sample is classified in batches; defined only for the single-pass families, which is the regime the Section 4.4 argument is about |
| `hardware.vram_delta_mb` | device VRAM peak minus the pre-load baseline, i.e. this model's own footprint |
| `hardware.mean_cpu_percent`, `hardware.peak_rss_mb` | system CPU and this process's resident set |
| `usd_per_million_records` | see below |

## The prediction cache is bypassed

`classify_with_cache` returns a stored label in microseconds. Timing through it would measure a
dictionary lookup, not inference, so this experiment calls `ollama.chat` and the Anthropic client
directly. This is also why it needs token counts that the shared helpers discard: `_classify_claude`
does not return `usage`, and `_classify_ollama` does not return `prompt_eval_count`/`eval_count`.

## Dollar cost is two different quantities

They are kept apart in `utils/cost.py` because they are not comparable measurements:

- **Hosted API**: metered. Measured input and output tokens at the published list price. This is a
  real cost. Claude Sonnet's prices are recorded in `API_PRICES`; Gemini's are deliberately not,
  and `--gemini` refuses to run without `--gemini-price`. A hardcoded rate for a preview model
  would go stale silently and turn into a wrong dollar figure in the paper with nothing to flag
  it, so the rate is supplied per run and echoed into that row's `cost_basis`.
- **Local models**: amortized. Measured throughput against an assumed hardware rental rate
  (`--gpu-hourly-usd`, `--cpu-hourly-usd`). This is an *assumption*, recorded in each row's
  `cost_basis` field, and any figure derived from it must be reported with the rate attached.

XGBoost is charged a CPU rate because Table 6 places it on CPU; `--xgb-device cpu` is the default
so that the benchmark tests the claim the table actually makes.

## Attribution caveats

VRAM is read device-wide via `nvidia-smi`, because Ollama serves the decoders from a separate
process. Per-model attribution therefore depends on ordering, and the experiment arranges for it:
the whole decoder roster is evicted before any decoder is timed, and the encoder weights are loaded
lazily inside the measure window rather than at configuration time. `baseline_vram_mb` is recorded
alongside the peak so the subtraction can be checked. Configurations are not run in separate
processes, so a resident model from an *earlier* configuration still sits in the baseline; this
does not affect the delta but does mean the absolute peak is not a single model's footprint.

One known limit remains, and it is marked in the source: the *second* torch model measured in
a single process has its VRAM understated, because PyTorch's caching allocator serves its
weights from blocks already reserved from the driver, so the device total never rises. In the
default ordering this affects `modernbert-ft`, which under-reports by roughly the weight size.
To get an exact figure for an encoder row, measure it in its own process with `--configs`.
Decoder rows are unaffected, since Ollama serves them from a separate process.

Batched throughput is measured after the probe stops, so the hardware figures describe the
single-record regime.

## Reproduction

```bash
uv run python ml_llm_ensembles/experiments/tests_14_cost.py

uv run python ml_llm_ensembles/experiments/14_cost_benchmark.py --domain phishing --n 300
uv run python ml_llm_ensembles/experiments/14_cost_benchmark.py --domain flows --n 300

ALLOW_LIVE_LLM_CALLS=1 uv run python ml_llm_ensembles/experiments/14_cost_benchmark.py \
  --domain phishing --n 300 --claude --configs claude-sonnet-4-6

ALLOW_LIVE_LLM_CALLS=1 uv run python ml_llm_ensembles/experiments/14_cost_benchmark.py \
  --domain phishing --n 300 --decoders --gemini --gemini-price <IN> <OUT> \
  --configs gemini-3.1-pro-preview
```

The decoder roster defaults to `llama3.2 mistral gemma3:12b gpt-oss`, covering the 1B-20B range
Table 6 claims and both the reasoning and non-reasoning cases discussed in Section 4.4.
Llama 3.1 8B is excluded. The frontier API is gated behind `ALLOW_LIVE_LLM_CALLS=1` because it
bills; the local decoders are not, because they do not.

The flow-level run parses the Mirai pcap once and caches the aggregated flows to
`results/cache/bench_flows_mirai.parquet`; subsequent runs reuse it.

## Single-seed, single-machine

All rows come from one machine and one seed (42). Latency and throughput are properties of that
hardware, not of the model in the abstract; the JSON records the device and the rates used.
