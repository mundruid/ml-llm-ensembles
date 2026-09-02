# Experiment 13 source panel: provenance and limitations

The corpus snapshot was acquired on 2026-08-30 UTC. Exact download URLs, byte counts, and SHA-256
values are in `ACQUISITION_SHA256.tsv`. Raw corpora remain outside Git. Upstream terms should be
reviewed before downloading or redistributing any source; this repository redistributes no
messages.

| source_corpus | official URL | version | licence / terms | original labels | taxonomy mapping | concerns | decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nazario_early | https://monkey.org/~jose/phishing/ | 20051114 + phishing0-3 + private-phishing4 (2004-2007) | CC-BY-4.0 (LICENSE.txt shipped with corpus; README confirms) | all messages are hand-classified phishing | phishing / confirmed (`nazario_phishing`) | single collector's inbox; early mboxes anonymized; two eras of ONE collection, so era-independence is weaker than truly independent corpora (recorded) | include |
| nazario_recent | https://monkey.org/~jose/phishing/ | phishing-2015 .. phishing-2025 yearly mboxes | CC-BY-4.0 | same | phishing / confirmed | later mboxes not anonymized; same collector as nazario_early | include |
| spamassassin_easy_ham | https://spamassassin.apache.org/old/publiccorpus/ | 20030228 | Apache SpamAssassin public corpus, published for research/redistribution | partition = easy_ham | ham / confirmed (`spamassassin`) | 2003-era mail | include |
| spamassassin_easy_ham_2 | https://spamassassin.apache.org/old/publiccorpus/ | 20030228 | Apache SpamAssassin public corpus | easy_ham_2 | ham / confirmed | 2003-era mail | include |
| spamassassin_hard_ham | https://spamassassin.apache.org/old/publiccorpus/ | 20030228 | Apache SpamAssassin public corpus | hard_ham | ham / confirmed | only ~250 msgs: below --min-source-rows, development-only | include (dev-only) |
| spamassassin_spam | https://spamassassin.apache.org/old/publiccorpus/ | 20030228 | Apache SpamAssassin public corpus | spam | bulk_spam / proxy (may contain phishing) | ~500 msgs; dev-only under min-source-rows | include (dev-only) |
| spamassassin_spam_2 | https://spamassassin.apache.org/old/publiccorpus/ | 20050311 | Apache SpamAssassin public corpus | spam_2 | bulk_spam / proxy | ~1,400 msgs; only hold-out-eligible spam unit, so rotation is limited | include |
| enron | https://www.cs.cmu.edu/~enron/ | enron_mail_20150507.tar.gz | CMU-hosted public release (FERC public record) | none (legitimate corporate mail) | ham / probable (`enron_ham`; contains some spam) | normalized deterministic 30k sample (every k-th file) as `enron_sample.parquet`; real names/content of a public legal record | include |
| CLAIR/"Nigerian" fraud collection | (Kaggle mirror) | - | canonical distribution requires authenticated Kaggle access; unauthenticated mirrors have unverifiable provenance | - | would be scam_fraud / confirmed | cannot verify licence/provenance without credentials | **exclude** (documented) |
| TREC 2007 / CEAS | - | - | requires usage agreement | - | - | agreement not in place | **exclude** |
| zefang-liu aggregate | HuggingFace | - | - | curated aggregate | - | likely overlaps Enron and Nazario; component provenance not recoverable -> cannot serve as an independent external source | **exclude from exp 13** |

Consequences recorded for interpretation:
- scam_fraud has NO source in this panel: FPR_scam_fraud is unmeasurable; mixtures that weight
  it will be reported as unavailable/extrapolative by design.
- Both phishing units come from one collector (different eras); external "unseen phishing
  source" therefore means "unseen collection era", a weaker claim than an independent feed.
- spam/scam hold-out rotation has a single eligible unit (spamassassin_spam_2), so external
  bundles vary only in D and E.
- Generic spam is a proxy negative and may contain phishing (mapping confidence `proxy`).

## Executed panel (manifest fingerprint `7adf8392735ca2b6`)

- Loaded rows: nazario_early 8,544 / nazario_recent 3,466 / SA easy_ham 2,500 / easy_ham_2
  1,400 / hard_ham 250 / spam 500 / spam_2 1,396 / enron sample 30,000 (48,056 total).
- Global exact dedup removed 4,691 rows; 191 content hashes crossed sources, 190 of them
  between nazario_early and nazario_recent (the two eras republish messages) and 1 between
  the two easy_ham partitions. 16 template groups span more than one source.
- Post-dedup subtypes: ham 31,888 / phishing 9,669 / bulk_spam 1,808. scam_fraud absent
  (documented exclusion); mixtures weighting it will report unavailable/extrapolative.
- All 8 sources pass role purity 1.0. Six external bundles (D in {nazario_early,
  nazario_recent} x E in {enron, sa_easy_ham, sa_easy_ham_2}; F fixed to sa_spam_2 - the
  single hold-out-eligible spam unit, as anticipated).
- Near-duplicate control: template groups shared between a bundle held-out source and
  development are DROPPED FROM DEVELOPMENT (held-out test stays complete); per-bundle drop
  counts are recorded in each result split_policy. This is the required exact-hash control
  plus a template-prefix near-duplicate control; paraphrased overlap beyond that remains a
  stated limitation.

## Corpus-family relationships (holdout tiers)

Families: nazario = {nazario_early, nazario_recent}; spamassassin = {easy_ham, easy_ham_2,
hard_ham, spam, spam_2}; enron = {enron}. Partitions of one family share collection processes
and corpus fingerprints, so holding out one partition while a sibling stays in development is
WITHIN-FAMILY PARTITION generalization (temporal/partition shift), not unseen-corpus
generalization.

Structural limitation of this panel: ALL phishing lives in the nazario family and ALL
spam-like data in the spamassassin family, so a fully family-disjoint bundle is impossible
(holding out either family entirely removes that role from development). Every bundle
therefore records holdout_tier_by_role; in the 6 audited bundles the phishing and spam_scam
roles are always within-family-partition, and only the ham role is family-disjoint (when E =
enron with SA ham in dev, or E = an SA ham partition with enron in dev... the latter is
within-family; see per-bundle tiers). n_fully_family_disjoint = 0 is reported in the external
summary and must be stated wherever external results are presented.

## Projection mixture limitation

The panel has no scam_fraud source, so FPR_scam_fraud has denominator zero and any projection
mixture giving scam_fraud nonzero weight is correctly reported unavailable. Reported
projections therefore use --spam-split-bulk 1.0 (100% of the spam-like share is bulk_spam)
and are labelled BULK-SPAM-ONLY projections; they must not be described as spam/scam
evidence. Adding a licensed scam/fraud corpus later restores the mixed projection.
