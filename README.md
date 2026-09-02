# ml-llm-ensembles

Code to reproduce the experiments in ["We Don't Need a Large Language Model for
That: When Traditional Machine Learning and Encoders Beat Decoder LLMs in
Security Classification"](TODO: add arxiv link).

We compare traditional ML (XGBoost, TabPFN), encoder models (ModernBERT, frozen and
fine-tuned), and decoder LLMs (zero-shot, local and frontier) on phishing-email and
network-intrusion classification; evaluate two ML-LLM ensemble designs, a confidence-gated
**router** and a stacking **meta-learner**; and evaluate generalization and operational cost
under grouped and temporal splits at observed attack prevalence.

---

## Contributions

- **A controlled, multi-representation comparison.** One phishing-text dataset and four
  network representations, three of them derived from the *same* Mirai botnet capture at
  different aggregation levels, isolating feature representation as the variable that
  governs LLM classification ability.
- **A leakage-aware evaluation discipline.** Every AUCPR is read against the class prior;
  per-packet data is split flow-aware (grouped by 5-tuple); campaign, host-grouped, and
  temporal splits are used on the new captures. This discipline overturns an otherwise
  illusory "LLMs work" result.
- **Router and Meta-Learner ML-LLM ensembles**, characterizing when routing to an LLM helps
  vs. only adds cost, and when stacking adds accuracy vs. merely down-weights a weak LLM.
- **Evidence that learned representations, not prompting, unlock non-language features**:
  frozen encoder embeddings and fine-tuning recover most of the signal zero-shot LLMs miss
  on serialized flow statistics.
- **Generalization and operational evaluation on new captures.** Campaign-disjoint CTU-13
  evaluation at its observed prevalence with false-positive burden in operator units, and a
  within- vs cross-capture transfer study on a second Kitsune capture.
- **Subtype-decomposed phishing evaluation with hard-negative training.** A multi-source
  panel separates ham from bulk-spam false positives, revealing an operational failure mode
  invisible to pooled AUCPR and a training-data intervention that removes most of it.

## Ensemble architectures

```
Router: prioritizes efficiency
─────────────────────────────────────────
  Input record
      │
      ▼
  ┌─────────────┐   confidence ≥ threshold   ┌──────────────┐
  │  Tier 1: ML │ ─────────────────────────► │  Prediction  │
  └─────────────┘                             └──────────────┘
      │ confidence < threshold
      ▼
  ┌─────────────┐                             ┌──────────────┐
  │  Tier 2: LLM│ ─────────────────────────► │  Prediction  │
  └─────────────┘                             └──────────────┘

Meta-Learner: prioritizes accuracy
─────────────────────────────────────────────
  Input record
      │
      ├──────────────► XGBoost ─────► P(1)  ─┐
      ├──────────────► TabPFN  ─────► P(1)  ─┼──► Stage-2 model ──► Prediction
      └──────────────► LLM     ─────► P(1)  ─┘   (logistic regression / GBM)
```

---

## Repo structure

```
ml-llm-ensembles/
├── ml_llm_ensembles/
│   ├── experiments/              # the numbered experiment suite (see its README)
│   │   ├── 01-02  ModernBERT fine-tuning (train/validation/test separation)
│   │   ├── 03-04  confidence-gated routers (phishing, flows)
│   │   ├── 05-09  out-of-fold stacking across representations
│   │   ├── 11     CTU-13 scenario / host-grouped / temporal evaluation
│   │   ├── 12     Kitsune within- and cross-capture packet evaluation
│   │   ├── 13     hard-negative phishing training, source shift, prior projections
│   │   ├── run_all.sh            # single orchestrator: synthetic | smoke | paper
│   │   ├── warm_llm_cache.py     # inspect/populate the decoder prediction cache
│   │   ├── make_results_table.py # regenerate RESULTS_TABLE.md from a result archive
│   │   ├── OPERATIONAL_RESULTS.md# human-readable results for experiments 11-13
│   │   └── RESULTS_TABLE.md      # full model-comparison tables
│   └── utils/                    # loaders, features, models, prompts, splits,
│                                 #   operational metrics, phishing taxonomy, fine-tuning
├── data/                         # dataset placeholders; see data/README.md
├── pyproject.toml / uv.lock
└── .env.example
```

---

## Setup

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/), an NVIDIA GPU (recommended for
fine-tuning, TabPFN, and encoder-embedding steps), and, for the open-source decoder LLMs, a
local [Ollama](https://ollama.com) install.

```bash
uv sync
cp .env.example .env   # fill in the keys you need (see below)
```

`.env` keys, all optional except when the corresponding flag is used:

| Key | Needed for |
|---|---|
| `ANTHROPIC_API_KEY` | `--claude` (Claude Sonnet as the frontier LLM) |
| `GEMINI_API_KEY` | `--gemini` (optional second frontier LLM) |
| `HF_TOKEN` | Gated HuggingFace model downloads |
| `TABPFN_TOKEN` | `--tabpfn` (TabPFN base learner in the meta-learner) |

For the local decoder LLMs (Mistral, Gemma3, Llama3.2, GPT-OSS), pull them with Ollama
first, e.g. `ollama pull mistral`.

---

## Data

Datasets are **not** included in this repo. See [`data/README.md`](data/README.md) for
sources, expected paths, and sizes, and
[`data/phishing-operational/`](data/phishing-operational/) for the experiment-13 corpus
panel (manifest schema, provenance, and licensing notes).

---

## Reproducing the experiments

`ml_llm_ensembles/experiments/README.md` documents the suite in detail. Three modes:

```bash
bash ml_llm_ensembles/experiments/run_all.sh synthetic   # no data or GPU: unit tests + fake-backend smoke
bash ml_llm_ensembles/experiments/run_all.sh smoke       # reduced-size execution; data required
bash ml_llm_ensembles/experiments/run_all.sh paper       # complete paper configuration; GPU and time required
```

Result JSON is written next to the experiment scripts and summarized with
`make_results_table.py --results-dir <archive>`. Live decoder calls are disabled by default;
set `ALLOW_LIVE_LLM_CALLS=1` only after reviewing the cost and model roster. Decoder
predictions are cached (key: text, model, prompt hash), so completed generations are never
repeated; `warm_llm_cache.py` inspects or pre-populates the cache.

Scope notes: experiment number 10 is intentionally unused, and fine-tuned decoder results
are outside the validated scope of this release (see `ml_llm_ensembles/experiments/README.md`).

## License

MIT, see [LICENSE](LICENSE).
