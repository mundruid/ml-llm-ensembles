# Experiment 01: fine-tuned ModernBERT for phishing

This experiment fine-tunes `answerdotai/ModernBERT-base` for binary phishing detection. It uses
the deduplicated phishing dataset and reports raw-text and provenance-stripped variants.

## Method

The outer stratified 80/20 split (seed 42) defines the test set. A validation partition is drawn
only from the outer training side. The model is trained on the remaining rows, the best epoch is
selected by validation AUCPR, and the selected checkpoint is evaluated once on the test set.
Exact-text deduplication occurs before splitting.

The phishing model is fully fine-tuned. The optional `--strip-provenance` variant removes source
markers before the split and writes a separate checkpoint. This ablation tests sensitivity to
corpus-specific tokens; it does not change the binary target.

## Reproduction

```bash
uv run python ml_llm_ensembles/experiments/01_ft_phishing.py \
  --output-dir models/modernbert-phishing-ft
uv run python ml_llm_ensembles/experiments/01_ft_phishing.py --strip-provenance \
  --output-dir models/modernbert-phishing-ft-stripped
```

The resulting checkpoints are consumed by experiment 03. Results are written as ignored working
artifacts; the published summary is in `RESULTS_TABLE.md`.

## Interpretation

This is a conventional random held-out evaluation on a curated corpus. It measures in-corpus
discrimination and should not be interpreted as an inbox prevalence or a source-disjoint test.
