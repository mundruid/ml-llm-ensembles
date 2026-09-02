# Experiment 04: confidence-gated Kitsune flow classifiers

This experiment evaluates standalone and confidence-gated models on five-tuple flows from the
Kitsune Mirai capture. The gate families match experiment 03: XGBoost or TabPFN over numeric flow
features or frozen ModernBERT embeddings.

## Method

The seed-42 stratified split is shared with experiment 02. Numeric missing values are imputed using
statistics fitted on training rows only. Zero-shot decoder tiers use the descriptive flow
serialization; the fine-tuned ModernBERT tier uses the compact serialization on which it was
trained. The latter is loaded from experiment 02's merged checkpoint rather than the prediction
cache.

The primary gate threshold is 0.7, and the script also evaluates a threshold sweep. Each row
records AUCPR, ROC-AUC, routing rate, and prediction coverage. Decoder predictions are cache-only
unless the caller explicitly enables a supported live backend.

## Reproduction

```bash
uv run python ml_llm_ensembles/experiments/04_router_flows.py \
  --ft-dir models/modernbert-mirai-flows-ft
```

## Interpretation

The experiment measures the accuracy-cost trade-off within one capture. It does not establish
generalization across hosts, time periods, or attack captures; experiment 12 addresses capture
transfer directly.
