# Operational and generalization results

This summary records the principal results from experiments 11--13. Machine-level result JSON is
not distributed in the public artifact. Values below were extracted from the preserved run archive
before publication; `RESULTS_TABLE.md` contains the broader model comparison.

## CTU-13

| Split | Test rows | Test attack prevalence | XGBoost AUCPR | Precision / recall at 0.50 | False positives per capture-hour |
| --- | ---: | ---: | ---: | ---: | ---: |
| scenario holdout | 10,233,415 | 2.52% | 0.938 | 0.871 / 0.874 | 887.9 |
| internal-host grouped | 5,667,278 | 4.17% | 0.938 | 0.791 / 0.878 | 251.2 |
| global temporal cutoff | 3,995,340 | 7.56% | 0.985 | 0.846 / 0.978 | 1,065.4 |

The scenario split is the primary capture-holdout result. The host policy groups on the internal
endpoint but is not fully endpoint-disjoint. Precision and per-hour burden are not directly
comparable across rows because the held-out scenarios, prevalences, and capture rates differ.
Background is an unverified negative class.

On the balanced 1:5 decoder subset for the scenario split, Mistral and Llama 3.2 have AUCPR 0.163
and 0.166, near the subset prior of 0.167; Gemma 3 12B reaches 0.465. These are subset AUCPRs and are
not estimates at the 2.52% observed CTU-13 prior.

## Kitsune capture transfer

| Train to test | Test rows | Test attack prevalence | XGBoost AUCPR | TabPFN AUCPR |
| --- | ---: | ---: | ---: | ---: |
| Mirai to SYN DoS | 2,771,276 | 0.254% | 0.0088 | 0.0234 |
| SYN DoS to Mirai | 764,137 | 84.08% | 0.8429 | 0.7591 |
| SYN DoS within capture, flow-grouped | 391,015 | 0.085% | 1.0000 | 1.0000 |

The within-capture score and cross-capture collapse show that separability within a capture does
not imply transfer to a different attack capture. The one-valued within-capture AUCPR is not, by
itself, evidence of train/test leakage: complete five-tuple flows are separated, but stable
capture-specific patterns remain available to the model. The global temporal SYN DoS run is
degenerate because its training side contains no positive packets.

Mistral and Llama 3.2 reach subset AUCPR 0.139 and 0.142 on Mirai-to-SYN-DoS and 0.172 and 0.164 in
the reverse direction, against a balanced-subset prior of 0.167. Gemma 3 12B was not run in this
panel.

## Phishing hard negatives and source shift

The prediction target remains binary. Conditions B and C add messages labelled non-phishing as
hard negatives; ham and bulk spam are retained only as evaluation subtypes.

For controlled raw-text ModernBERT at threshold 0.50, bulk-spam FPR falls from 0.111 in condition A
(ham-only negatives), to 0.015 in B (matched-budget hard negatives), and 0.0025 in C (additive hard
negatives). Phishing recall is 0.991, 0.991, and 0.989 respectively. At a hypothetical 1% phishing
prevalence with bulk spam comprising half of negatives, the corresponding point precision is
0.152, 0.502, and 0.756. Conservative values are 0.119, 0.336, and 0.499; this low-prior cell is
marked extrapolative because the negative denominators do not resolve the required FPR tightly.

Across six raw-text external bundles, ModernBERT's median bulk-spam FPR is 0.112, 0.053, and 0.048
for A, B, and C; median phishing recall remains approximately 0.95. Median AUCPR is 0.987, 0.992,
and 0.993. Two Enron-ham bundles are substantially harder than the median, including AUCPR 0.849
with ham FPR 0.243 and AUCPR 0.813 with ham FPR 0.358 under condition A.

These are source-partition bundles, not fully family-disjoint tests. Both phishing partitions come
from the Nazario family and all spam partitions from SpamAssassin; only the Enron ham role is
family-disjoint. There is no verified scam/fraud source, so every reported spam-mixture projection
is bulk-spam-only. The study uses one seed.
