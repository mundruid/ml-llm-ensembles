# Experiment 03: confidence-gated phishing classifiers

This experiment compares standalone phishing classifiers with two-tier systems that retain a
gate prediction when its maximum class probability is at least 0.7 and otherwise invoke an
escalation model.

## Models

The tier-one gates are XGBoost or TabPFN, each using either hand-crafted email features or frozen
ModernBERT embeddings. Escalation models include zero-shot decoders and the validation-selected
ModernBERT checkpoint from experiment 01. Model scores are converted to a consistent phishing
probability before AUCPR and ROC-AUC are calculated.

Raw and provenance-stripped variants use separate checkpoints and result files. Exact-text
deduplication precedes the common seed-42 stratified split. Decoder predictions are read from the
cache by default; partial cache coverage is reported and is not eligible for ranking as a complete
test result.

## Reproduction

```bash
uv run python ml_llm_ensembles/experiments/03_router_phishing.py \
  --ft-dir models/modernbert-phishing-ft
uv run python ml_llm_ensembles/experiments/03_router_phishing.py --strip-provenance \
  --ft-dir models/modernbert-phishing-ft-stripped
```

## Interpretation

Routing changes both predictive performance and the fraction of examples sent to the more
expensive tier. Comparisons therefore report AUCPR, ROC-AUC, routing rate, and coverage together.
The split is in-corpus rather than source-disjoint.
