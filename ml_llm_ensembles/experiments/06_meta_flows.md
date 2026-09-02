# Experiment 06: out-of-fold stacking for Kitsune flows

This experiment evaluates XGBoost, TabPFN, frozen ModernBERT features, zero-shot decoders, and
logistic-regression or XGBoost stage-two combinations on aggregated Kitsune Mirai flows.

## Method

The outer split is stratified and uses seed 42. Numeric preprocessing is fitted independently
inside every OOF training fold and then applied to that fold's validation rows. A separate imputer
is fitted on the complete outer training side for final test prediction. Each flow is one row, so
plain stratified folds are used.

Decoder features are drawn from the configured cache. The stage-two training subset is capped by
`--meta-train-samples`; sampling is stratified and occurs only within the outer training side.
Coverage and evaluated-row counts are stored with each result.

## Reproduction

```bash
uv run python ml_llm_ensembles/experiments/06_meta_flows.py
```

## Interpretation

This is a within-capture flow evaluation. The stage-two comparison shows whether a weak decoder
feature is ignored or degrades the trained bases, but it does not measure cross-capture transfer.
