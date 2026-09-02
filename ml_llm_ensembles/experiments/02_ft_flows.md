# Experiment 02: fine-tuned ModernBERT for Kitsune flows

This experiment fine-tunes ModernBERT on compact text serializations of five-tuple flows from the
Kitsune Mirai capture. LoRA is used during training and merged into a standard classifier
checkpoint for downstream inference.

## Method

An outer stratified 80/20 split (seed 42) defines the test set. A validation partition is drawn
only from the training side. Hugging Face Trainer evaluates on validation data, selects the best
epoch by validation AUCPR, and scores the test partition after selection.

Training, validation, and inference all use
`format_network_row_kitsune_pcap_flow_ft`. The compact `key=value` representation is part of the
model contract and must remain identical downstream. LoRA uses rank 8, alpha 16, and the `Wqkv`
and `Wo` target modules. Class weighting accounts for the training imbalance.

## Reproduction

```bash
uv run python ml_llm_ensembles/experiments/02_ft_flows.py \
  --output models/modernbert-mirai-flows-ft
```

The merged checkpoint is consumed by experiment 04. Results are written as ignored working
artifacts; the published summary is in `RESULTS_TABLE.md`.

## Interpretation

This is a within-capture, random flow-level split. Each row represents one flow, but hosts and time
ranges are not held out. Cross-capture transfer is evaluated separately in experiment 12.
