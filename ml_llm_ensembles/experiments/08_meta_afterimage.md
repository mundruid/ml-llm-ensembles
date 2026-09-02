# Experiment 08: XGBoost and TabPFN on AfterImage features

This experiment compares XGBoost, TabPFN, and their stage-two combinations on 115 incremental
AfterImage statistics from the Kitsune Mirai capture. Decoder and text-embedding models are omitted
because the inputs are opaque numeric state variables.

## Split policies

`--split temporal` applies a chronological 80/20 cutoff separately within each class. This keeps
both labels when their capture ranges differ, but it is not a single global deployment-time
boundary. `--split random --allow-leaky-random` provides an explicitly labelled random-row
reference. Consecutive AfterImage rows share sliding-window state, so neither policy establishes
independence from all temporal state; the random reference is particularly susceptible.

The script validates that approximately 115 feature columns were loaded. Imputation is fitted
inside each OOF fold and on the complete training partition for final test transformation.

## Reproduction

```bash
uv run python ml_llm_ensembles/experiments/08_meta_afterimage.py --split temporal
uv run python ml_llm_ensembles/experiments/08_meta_afterimage.py --split random --allow-leaky-random
```

## Interpretation

The preserved temporal run is degenerate: its test side contains one class and therefore has no
model-comparison rows. It must not be cited as a temporal AUCPR result. The random-row result is a
within-capture benchmark reference, not evidence of temporal or cross-capture generalization.
