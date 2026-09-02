# Experiment 11: CTU-13 generalization and operational burden

This experiment evaluates network classifiers on the thirteen CTU-13 botnet scenarios and reports
threshold metrics at the prevalence observed in each held-out partition. Background traffic is
treated as negative by default, although it is not verified benign; `--background drop` provides a
Botnet-versus-Normal sensitivity analysis.

## Splits

| Policy | Outer split and OOF policy |
| --- | --- |
| `scenario` | Holds out complete designated captures; OOF folds group by scenario. |
| `host` | Groups by the internal 147.32/16 endpoint; the other endpoint may overlap. Residual endpoint overlap is recorded. |
| `temporal` | Uses one global cutoff; OOF prediction uses forward-chaining folds and excludes the initial block from stage-two training. |
| `random` | Stratified row split used only as a benchmark reference. |

XGBoost scores the complete held-out partition. TabPFN and frozen-embedding pipelines use a uniform
test sample and record its exposure fraction. Decoder rows use a class-balanced cost-controlled
subset, store `aucpr_subset` rather than AUCPR, and are not ranked as representative test results.

Thresholds for target recall or precision are selected from outer-training OOF scores. The test
report includes precision, recall, FPR, Wilson intervals, false positives per 1,000 flows, and
false positives per capture-hour. Prior projections use measured TPR/FPR and are labelled as
projections; no AUCPR is projected.

## Reproduction

```bash
uv run python ml_llm_ensembles/experiments/11_ctu13_operational.py --split scenario --models
uv run python ml_llm_ensembles/experiments/11_ctu13_operational.py --split host --models
uv run python ml_llm_ensembles/experiments/11_ctu13_operational.py --split temporal --models
```

The recorded decoder extension was run only for the scenario policy. The published result table
must state the scored subset and its 1:5 positive-to-negative composition.

## Interpretation

The scenario result supports capture hold-out generalization. The host policy is grouped but not
fully host-disjoint, and its overlap diagnostic must accompany any claim. The observed prevalence
is a property of CTU-13 labels and should not be described as a universal deployment prior.
