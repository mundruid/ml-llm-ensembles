# Experiment 12: Kitsune within- and cross-capture packet evaluation

This experiment compares within-capture discrimination with transfer between the Kitsune Mirai and
SYN DoS captures. SYN DoS contains 2,771,276 packets, of which 7,038 are labelled malicious
(approximately 0.25%).

## Representation and splits

The experiment uses the named per-packet representation. Five-tuple aggregation is unsuitable for
the prevalence question because it produces only about 740 flows and an attack prevalence near
66%. Cross mode trains on one complete capture and tests on the other, with flow-grouped OOF folds
on the training capture. Within mode uses either a flow-grouped split or one global temporal
cutoff.

XGBoost scores the complete test capture. TabPFN and optional frozen-embedding models use a uniform
test sample and record the exposure fraction. Decoder rows use a class-balanced subset and are
marked non-representative. The recorded decoder panel contains Mistral and Llama 3.2; Gemma 3 12B
was not run for this experiment.

## Reproduction

```bash
uv run python ml_llm_ensembles/experiments/12_kitsune_cross_capture.py --mode cross --models
uv run python ml_llm_ensembles/experiments/12_kitsune_cross_capture.py --mode within \
  --within-split flow --models
uv run python ml_llm_ensembles/experiments/12_kitsune_cross_capture.py --mode within \
  --within-split temporal --models
```

## Interpretation

The flow-grouped within-capture result is nearly perfectly separable, whereas cross-capture
performance falls close to the receiving capture's prior. This contrast is not itself proof of
leakage: same-capture packet fields can encode stable capture- and attack-specific patterns. The
global temporal SYN DoS split is degenerate because all attack packets occur in the final burst;
the run records that outcome instead of changing the split.
