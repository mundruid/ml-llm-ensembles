# data/

Raw datasets are **not** tracked in git (see `.gitignore`). Download them from the sources
below and place them in this directory exactly as shown. The loaders in
[`ml_llm_ensembles/utils/datasets.py`](../ml_llm_ensembles/utils/datasets.py) expect these
paths; if a file is missing they raise `FileNotFoundError` naming what to fetch.

The phishing benchmark corpus (`zefang-liu`) is **not** placed here: it is fetched
automatically via `datasets.load_dataset("zefang-liu/phishing-email-dataset")` and cached by
the `datasets` library (usually `~/.cache/huggingface/`).

## Expected structure

```
data/
├── cicids2017/
│   └── *.csv                      # 8 daily CSVs, ~847 MB total (experiment 09)
├── kitsune/
│   ├── Mirai_dataset.csv          # 115 AfterImage features, no header, ~1.3 GB (exp 08)
│   ├── Mirai_labels.csv           # one 0/1 label per row, ~2.2 MB (exps 02, 04, 06-08, 12)
│   ├── Mirai_pcap.pcap            # raw packet capture, ~72 MB (exps 02, 04, 06, 07, 12)
│   ├── SYN_DoS_pcap.pcap(.gz)     # second capture, ~2.6 GB unpacked (exp 12)
│   └── SYN_DoS_labels.csv(.gz)    # per-packet labels for the SYN DoS capture (exp 12)
├── ctu13/
│   └── CTU-13-Dataset/<n>/*.binetflow   # 13 scenario dirs from the official
│                                        # 1.9 GB tarball, ~75 GB extracted (exp 11)
└── phishing-operational/          # experiment 13 panel; see its README for the
    └── raw/...                    # manifest schema and acquisition steps
```

## Sources

| Directory | Source | Notes |
|---|---|---|
| `cicids2017/` | Kaggle: [`chethuhn/network-intrusion-dataset`](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) | Extract all 8 daily CSVs directly into `data/cicids2017/`. Raw column names carry stray whitespace; the loader strips it. |
| `kitsune/` (Mirai) | Kaggle: [`ymirsky/network-attack-dataset-kitsune`](https://www.kaggle.com/datasets/ymirsky/network-attack-dataset-kitsune) or UCI ML repository #516 | `Mirai_dataset.csv`, `Mirai_labels.csv`, `Mirai_pcap.pcap` (or compressed forms; the loaders extract). |
| `kitsune/` (SYN DoS) | UCI ML repository #516, `syn_dos/` | `SYN DoS_pcap.pcap.gz` and `SYN DoS_labels.csv.gz`, saved with the space replaced by an underscore as shown above. |
| `ctu13/` | [Stratosphere Laboratory CTU-13](https://www.stratosphereips.org/datasets-ctu13) | Download `CTU-13-Dataset.tar.bz2` (1.9 GB) and extract inside `data/ctu13/`. CC-BY 2.0. |
| `phishing-operational/` | Multiple public corpora | See [`phishing-operational/README.md`](phishing-operational/README.md) for the manifest schema, [`SOURCES.md`](phishing-operational/SOURCES.md) for provenance and licensing, and [`ACQUISITION_SHA256.tsv`](phishing-operational/ACQUISITION_SHA256.tsv) for archive checksums. |

Approximate totals once extracted: ~2.2 GB (Mirai + CIC-IDS2017), ~2.6 GB (SYN DoS),
~75 GB (CTU-13), ~0.5 GB (phishing panel). CTU-13 is only needed for experiment 11; you can
run everything else without it.
