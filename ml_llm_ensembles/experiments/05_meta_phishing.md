# Experiment 05: out-of-fold stacking for phishing

This experiment combines phishing classifiers with logistic-regression and XGBoost stage-two
models. It evaluates hand-crafted features, frozen ModernBERT embeddings, TabPFN, and a zero-shot
decoder feature.

## Method

The outer seed-42 stratified split is applied after exact-text deduplication. Every trained base
model produces training features through five-fold stratified out-of-fold prediction. The base is
then refit on the complete outer training set and applied to the test set. Consequently, a
stage-two row never contains an in-sample prediction from a trained base model.

The zero-shot decoder is not trained on these rows, so its cached probability does not require an
OOF fit. Decoder labels are used for standalone accuracy; class-conditioned probabilities are used
for ranking and stacking. Raw and provenance-stripped variants are reported separately.

## Reproduction

```bash
uv run python ml_llm_ensembles/experiments/05_meta_phishing.py
uv run python ml_llm_ensembles/experiments/05_meta_phishing.py --strip-provenance
```

## Interpretation

The held-out data come from the same curated corpus mixture as training. Experiment 13 separately
tests hard-negative composition and held-out source partitions.
