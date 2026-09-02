# Phishing operational corpora (experiment 13)

Experiment 13 (`ml_llm_ensembles/experiments/13_phishing_operational.py`) reads a manifest that lists
locally stored e-mail corpora and the rule that maps each corpus's original labels onto the
annotation policy in `ml_llm_ensembles/utils/phishing_taxonomy.py`. Nothing is downloaded by
the runner; if a listed file is missing it stops and prints the expected paths. Raw corpora
and derived caches stay out of Git. The executed acquisition checksums and source limitations are
published in `ACQUISITION_SHA256.tsv` and `SOURCES.md`.

## Manifest schema (`manifest.csv`)

| column | meaning |
| --- | --- |
| `asset_id` | unique short id for the asset |
| `path` | path relative to this directory |
| `format` | `csv`, `parquet`, `jsonl`, `eml_dir` (one message per file; label = first sub-directory), `mbox` |
| `source_corpus` | corpus name; a held-out unit for external bundles |
| `source_version` | release/version string recorded in every result |
| `text_field` | column with the message text (csv/parquet/jsonl) |
| `label_field` | column with the original label; for `eml_dir`/`mbox` a constant label for the whole asset (leave blank for `eml_dir` to use the sub-directory name) |
| `mapping` | a registered rule name (below) or inline JSON `{"<source label>": ["<subtype>", "<confidence>"]}` |
| `license_note` | licence / redistribution terms |
| optional: `id_field`, `timestamp_field`, `group_field` | original id, timestamp, campaign/thread id columns when the corpus has them |

Registered mapping rules: `nazario_phishing`, `enron_ham`, `spamassassin`, `ling_spam`,
`trec_ceas`, `nigerian_fraud`, `zefang_liu`, `binary_generic`. A rule is applied only to the
asset that names it; the corpus name never sets the label by itself. Labels a rule does not
cover cause a hard error.

**Mixed corpora: one asset per partition.** External bundles hold out whole
`source_corpus` units, and a source qualifies for a role (phishing / ham / spam-scam) only if
at least `--role-purity` (default 0.9) of its rows carry that role's subtype(s); sub-purity
sources are `mixed` and stay development-only. The spam/scam role counts only KNOWN
`bulk_spam` + `scam_fraud` rows: a source that is mostly `unknown_nonphishing` is
`unknown_negative`, usable as a development hard negative but never as the held-out
spam/scam source F. So list a mixed corpus as separate assets with
distinct `source_corpus` names per partition, e.g. `spamassassin_easy_ham` and
`spamassassin_spam`, so each unit has one explicit role. Every bundle records the actual
subtype composition of its held-out sources.

## Compatible source formats

| corpus | typical role | rule | notes |
| --- | --- | --- | --- |
| Nazario phishing corpus (`phishing-*.mbox`) | phishing | `nazario_phishing` | all messages phishing (confirmed) |
| SpamAssassin public corpus (`easy_ham`, `hard_ham`, `spam` dirs) | ham, bulk spam | `spamassassin` | spam partition is `bulk_spam` with confidence `proxy` (may contain phishing) |
| Enron e-mail corpus | ham | `enron_ham` | `probable`: Enron contains some spam |
| TREC 2007 / CEAS 2008 | ham, bulk spam | `trec_ceas` | index file gives ham/spam |
| Ling-Spam | ham, bulk spam | `ling_spam` | |
| CLAIR "Nigerian" fraud collection | scam/fraud | `nigerian_fraud` | advance-fee fraud (confirmed) |
| `zefang-liu/phishing-email-dataset` (export to csv) | phishing, ham | `zefang_liu` | curated aggregate; `probable` on both classes |

Example: `manifest.example.csv`.

## Bundle rule

External bundles are enumerated before any model is fitted: every triple of distinct sources
(phishing-majority D, ham-majority E, spam/scam-majority F) with at least `--min-source-rows`
rows each, such that the remaining development sources still contain all three roles. All
valid bundles are run and reported individually with median and range; a single valid bundle
is reported as a single external-domain case study.

## Run

```bash
uv run python ml_llm_ensembles/experiments/13_phishing_operational.py --manifest data/phishing-operational/manifest.csv \
    --models xgb bert_lr modernbert_ft --ft-scope full \
    --min-source-rows 1000 --spam-split-bulk 1.0
uv run python ml_llm_ensembles/experiments/13_phishing_operational.py --synthetic-smoke --models xgb   # no data needed
```

`modernbert_ft` fine-tunes `answerdotai/ModernBERT-base` from scratch inside each split.
Task-specific checkpoints are ineligible because their source overlap with this panel is not
established. Checkpoints and metadata land under
`models/exp13-modernbert-ft/` (gitignored) with identities covering split, variant,
condition, seed and exact data fingerprints; see the appendix in
`ml_llm_ensembles/experiments/13_phishing_operational.md` for the executed scope and interpretation limits.
