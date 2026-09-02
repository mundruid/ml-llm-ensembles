# Experiment 13: hard negatives, corpus shift, and phishing prevalence

This experiment retains a binary target, phishing versus non-phishing, while preserving the
negative subtype (`ham`, `bulk_spam`, `scam_fraud`, or `unknown_nonphishing`) for evaluation.
Subtype labels do not turn the task into multiclass prediction. They allow false-positive rates to
be measured separately for ordinary mail and difficult non-phishing mail.

## Executed corpus panel

The completed panel contains 43,365 deduplicated messages from eight source partitions: Enron,
five SpamAssassin partitions, and early and recent partitions of the Nazario phishing collection.
The subtype counts are 31,888 ham, 9,669 phishing, and 1,808 bulk-spam proxy examples. No verified
scam/fraud source was available. Exact acquisition, mapping, licensing, and redistribution caveats
are recorded in `data/phishing-operational/SOURCES.md`.

The binary definition is fixed before loading data. Source labels are mapped through the versioned
taxonomy in `phishing_taxonomy.py`; unknown mappings raise instead of defaulting. Text is
normalized and SHA-256 deduplicated before group-aware partitioning. Campaign metadata is used
when available; otherwise the group is a template-prefix proxy.

## Evaluation regimes

The controlled regime hashes groups into 60/20/20 training, calibration, and test partitions while
requiring represented subtypes. The external regime combines a held-out phishing source partition,
a held-out ham source partition, and a held-out bulk-spam source partition. Groups duplicated
across source partitions are removed from the development side, while the held-out test remains
complete.

Six external bundles were valid. They are source-partition holdouts, not six fully unseen corpus
families: the phishing and spam roles are alternate partitions from the Nazario and SpamAssassin
families. Only Enron supplies a family-disjoint ham role, and
`n_fully_family_disjoint=0`. External results therefore measure source and era shift with partial
family overlap, not complete three-role domain generalization.

## Hard-negative intervention

All models predict the same binary label under three training conditions:

- A: phishing versus ham only;
- B: matched negative budget, replacing half of ham negatives with bulk-spam hard negatives;
- C: additive hard negatives, retaining the ham budget and adding bulk spam.

A and B have equal total negative counts. The comparison isolates negative composition; C also
tests additional negative volume. The evaluated models are XGBoost, logistic regression over
frozen ModernBERT embeddings, and split-local fine-tuned ModernBERT. The completed paper panel has
42 fine-tunes: three conditions, two text variants, and seven evaluation units (one controlled and
six external).

Every fine-tune starts from `answerdotai/ModernBERT-base`. Training receives only training text;
epoch selection uses calibration AUCPR; test text is scored after selection. Checkpoint identities
hash the split, condition, text fingerprints, policy version, model, and hyperparameters.

## Operational estimates

At each calibration-selected threshold the experiment estimates phishing TPR and separate FPRs
for each available negative subtype. For hypothetical phishing prevalence `pi` and negative
mixture weights `q`, it reports

```text
mixed_fpr = sum(q_s * FPR_s)
precision = pi * TPR / (pi * TPR + (1 - pi) * mixed_fpr)
```

The test examples are not resampled to manufacture those priors. Point projections are accompanied
by conservative projections using a lower confidence bound for TPR and upper bounds for FPR.
Support status compares the required FPR with the available negative denominator and marks poorly
resolved low-prior projections as extrapolative. Group-bootstrap intervals are primary; Wilson
intervals provide binomial sensitivity estimates.

All reported mixtures in the completed panel are bulk-spam-only among the spam-like subtype
(`spam_split_bulk=1.0`). They do not support claims about scam/fraud traffic. AUCPR is reported only
for an observed curated test set and is never projected from one threshold.

## Reproduction

Prepare `data/phishing-operational/manifest.csv` as described in the data README, then run:

```bash
uv run python ml_llm_ensembles/experiments/13_phishing_operational.py \
  --manifest data/phishing-operational/manifest.csv \
  --models xgb bert_lr modernbert_ft \
  --ft-scope full --min-source-rows 1000 --spam-split-bulk 1.0
```

The deterministic dependency-light test path does not read corpora or download a model:

```bash
uv run python ml_llm_ensembles/experiments/tests_13_synthetic.py
uv run python ml_llm_ensembles/experiments/13_phishing_operational.py --synthetic-smoke \
  --models xgb modernbert_ft --ft-backend fake --ft-scope minimal --n-boot 20
```

## Interpretation

The main question is whether exposure to hard-negative bulk spam lowers spam false positives
without materially reducing phishing recall, and whether that improvement survives source shift.
Projected precision at low prevalence is a conditional sensitivity analysis, not a measurement of
the phishing rate in a live inbox. Results use one seed and one spam family; both limit
generalization.
