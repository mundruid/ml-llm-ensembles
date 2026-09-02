# Experiment 09: out-of-fold stacking on CIC-IDS2017

This experiment applies XGBoost, TabPFN, frozen ModernBERT features, zero-shot decoders, and
stage-two combinations to named CIC-IDS2017 flow statistics.

## Method

The loader removes exact duplicates and the run samples 5,000 rows with seed 42 before a
stratified random split. Trained bases generate five-fold OOF features. Numeric imputation is
fitted within each fold and separately on the complete outer training side for test prediction.
Decoder-only accuracy uses the returned class; class-conditioned probabilities are used for AUCPR
and stacking.

## Reproduction

```bash
uv run python ml_llm_ensembles/experiments/09_meta_cicids.py
```

## Interpretation

This is a conventional random benchmark split. The pipeline discards timestamps, and published
CIC-IDS2017 analyses have identified duplicated and temporally related records. Exact-duplicate
removal reduces one source of dependence but does not make the evaluation capture- or
time-disjoint. Near-ceiling results should therefore be read as within-benchmark discrimination.
