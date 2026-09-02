# Experiment 07: flow-grouped stacking for packet features

This experiment applies the model and stacking panel to named packet fields from the Kitsune Mirai
capture. It is designed to prevent packets from the same five-tuple flow from appearing on both
sides of a learned comparison.

## Method

The outer split uses `GroupShuffleSplit` keyed by `flow_id`. Every trained base produces training
features with `StratifiedGroupKFold`, using the same key; fold-level assertions verify zero flow
overlap. Numeric imputation is fitted within each fold, and the final test transform uses an
imputer fitted on the complete outer training partition.

The evaluated bases are XGBoost, TabPFN, and frozen-ModernBERT variants. Logistic-regression and
XGBoost stage-two models combine their OOF probabilities with cached zero-shot decoder scores.
Coverage and the number of evaluated rows are reported for each decoder-dependent row.

## Reproduction

```bash
uv run python ml_llm_ensembles/experiments/07_meta_pcap.py
```

## Interpretation

The default 5,000-packet sample yields a small test set (291 packets in the recorded run), so model
differences have substantial sampling uncertainty. Grouping controls exact flow identity, not all
host, campaign, or temporal dependence.
