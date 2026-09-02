# Paper experiment artifact

This directory contains the validated experiment implementations and result artifacts used in
the paper on machine-learning and language-model ensembles for phishing and network-intrusion
detection. The numbered experiments cover fine-tuned encoders, confidence-gated routers,
out-of-fold meta-learners, grouped and temporal network evaluation, and operational phishing
evaluation under class-prior shift.

The code remains part of the parent Python package: experiment scripts import utilities from
`ml_llm_ensembles/utils/`. Raw datasets, model weights, API credentials, and prediction caches
are not distributed here. Dataset acquisition and licensing notes are in the repository README
and in `data/phishing-operational/`.

## Published scope

| Experiments | Purpose |
| --- | --- |
| 01--02 | ModernBERT fine-tuning with train/validation/test separation |
| 03--04 | Confidence-gated phishing and flow classifiers |
| 05--09 | Out-of-fold stacking on phishing and network representations |
| 11 | CTU-13 scenario, host-grouped, and temporal evaluation |
| 12 | Within- and cross-capture Kitsune packet evaluation |
| 13 | Hard-negative phishing training, source shift, and prior projections |

Experiment number 10 is intentionally unused in this release. It was a development-only feature
comparison without a completed canonical run. Decoder fine-tuning is also outside the validated
scope: the available historical decoder scripts do not use an independent validation partition
for checkpoint selection. The published decoder rows in this directory are zero-shot evaluations,
not a claim that the historical fine-tuned-decoder results are reproduced here.

## Results policy

Machine-level result JSON, prediction caches, plots, and checkpoints are not included in the
public artifact. A normal script invocation writes working results beside the script; those files
are ignored by Git. `RESULTS_TABLE.md` is the human-readable record of the completed panel.
`make_results_table.py` can regenerate a table from a private or newly reproduced result directory
supplied with `--results-dir`.

AUCPR must always be interpreted with the test-set class prevalence printed for the same block.
Rows scored on a class-balanced LLM subset contain `aucpr_subset`, are marked
`representative_test=false`, and are excluded from rankings. Experiment 13 projects precision at
hypothetical priors from subtype-conditional rates; it does not project AUCPR or claim to measure
a natural inbox prevalence.

## Reproduction

Create the environment from the repository root:

```bash
uv sync
```

Run the dependency-light checks:

```bash
uv run python ml_llm_ensembles/experiments/tests_13_synthetic.py
bash -n ml_llm_ensembles/experiments/run_all.sh
bash ml_llm_ensembles/experiments/run_all.sh smoke
```

The paper configuration is expensive and requires all documented datasets, an NVIDIA GPU,
pretrained model downloads, and the local model/cache prerequisites described below:

```bash
bash ml_llm_ensembles/experiments/run_all.sh paper
```

`paper` fails before training if the experiment-13 manifest is absent. Live decoder calls are
disabled by default; set `ALLOW_LIVE_LLM_CALLS=1` only after reviewing the cost and model roster.
The exact commands represented by previously published artifacts are recorded in the experiment
notes and JSON metadata.

## Data and compute

- Experiments 01, 03, and 05 use the phishing corpus supported by
  `load_phishing_dataset`.
- Experiments 02, 04, and 06--08 use Kitsune Mirai representations.
- Experiment 09 uses CIC-IDS2017.
- Experiment 11 uses all 13 CTU-13 bidirectional NetFlow scenarios.
- Experiment 12 uses the Mirai and SYN DoS packet captures.
- Experiment 13 uses a local manifest; see `data/phishing-operational/README.md` and
  `SOURCES.md`.

The complete runs require substantially more time and memory than smoke mode. Experiment 11
loads roughly 20 million flows, experiment 12 parses about 2.8 million SYN DoS packets, and the
experiment-13 paper panel includes 42 split-local ModernBERT fine-tunes. Model checkpoints are
regenerable and are not included in Git.

## Methodological boundaries

- The CTU-13 scenario policy holds out complete captures. Its host policy groups by the internal
  endpoint but is not fully endpoint-disjoint; the JSON records residual endpoint overlap.
- The Kitsune within-capture flow policy prevents a five-tuple from crossing the split. Its global
  temporal SYN DoS run is degenerate because the attack is concentrated in the final burst.
- The experiment-13 external bundles hold out source partitions. No bundle is fully
  family-disjoint for phishing and spam (`n_fully_family_disjoint=0`); Enron provides the only
  family-disjoint ham role. Projections use bulk spam only because no verified scam/fraud source
  was available.
- All released runs use seed 42. This is a single-seed study.

## License and citation

This code is released under the MIT License (see the repository-level [LICENSE](../../LICENSE)
file). Citation metadata for the accompanying paper will be added when the preprint is
available.
