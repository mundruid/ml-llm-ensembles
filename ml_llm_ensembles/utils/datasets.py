import re
import numpy as np
import pandas as pd
import requests
from io import StringIO
from pathlib import Path

from ml_llm_ensembles.utils.features import (
    aggregate_pcap_packets_to_flows,
    extract_packet_features,
    _PCAP_PROTO_INT_MAP,
)

DATASETS = {
    # ── Phishing (text) ────────────────────────────────────────────────────
    "zefang-liu": {
        "hf_id": "zefang-liu/phishing-email-dataset",
        "description": "Phishing email dataset with 'Safe Email' / 'Phishing Email' labels",
        "domain": "phishing",
    },
    # ── Network (tabular) ──────────────────────────────────────────────────
    "cicids2017": {
        "local_path": "data/cicids2017/",
        "description": "CIC-IDS2017 network flow dataset — 15-class, binarised to Benign=0 / Attack=1",
        "domain": "network",
    },
    "kitsune-mirai": {
        "local_path": "data/kitsune/",
        "description": "Kitsune Mirai botnet dataset — 115 AfterImage features, binary Benign=0 / Attack=1",
        "domain": "network",
    },
    "kitsune-mirai-pcap": {
        "local_path": "data/kitsune/",
        "description": "Kitsune Mirai pcap — packet-level named features (src/dst port, protocol, length, TTL, flags, interarrival)",
        "domain": "network",
    },
    "kitsune-mirai-pcap-flows": {
        "local_path": "data/kitsune/",
        "description": "Kitsune Mirai pcap aggregated to 5-tuple flows — per-flow stats (pkt_count, bytes, duration, IAT mean/std, TCP flag totals)",
        "domain": "network",
    },
    "kitsune-syndos-pcap": {
        "local_path": "data/kitsune/",
        "description": "Kitsune SYN DoS capture (UCI #516 syn_dos/) — packet-level named features; 2,771,276 packets, 0.25% malicious",
        "domain": "network",
    },
    "kitsune-syndos-pcap-flows": {
        "local_path": "data/kitsune/",
        "description": "Kitsune SYN DoS capture aggregated to 5-tuple flows (NOTE: only ~740 flows, ~66% attack; use the packet form for prevalence work)",
        "domain": "network",
    },
    "kitsune-ssdp-pcap": {
        "local_path": "data/kitsune/",
        "description": "Kitsune SSDP Flood capture (UCI #516 ssdp_flood/) — packet-level named features",
        "domain": "network",
    },
    "ctu13": {
        "local_path": "data/ctu13/",
        "description": "CTU-13 (Stratosphere 2011) — 13 botnet scenarios, bidirectional NetFlow, Botnet=1 vs Normal+Background=0",
        "domain": "network",
    },
}

PHISHING_DATASETS = [k for k, v in DATASETS.items() if v["domain"] == "phishing"]
NETWORK_DATASETS  = [k for k, v in DATASETS.items() if v["domain"] == "network"]


# ── Phishing loaders ───────────────────────────────────────────────────────────

def _load_zefang_liu() -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("zefang-liu/phishing-email-dataset")
    df = ds["train"].to_pandas()
    df = df.rename(columns={"Email Text": "text"})
    df["label"] = df["Email Type"].map({"Phishing Email": 1, "Safe Email": 0})
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df[["text", "label"]]


# Source-corpus provenance tokens in the zefang-liu phishing set: the "Safe"
# class is the Enron corpus and the "Phishing" class is a separate collection,
# so these tokens identify the SOURCE corpus, not phishing semantics. A
# classifier can exploit them as a spurious shortcut. strip_provenance removes
# them for the --strip-provenance robustness ablation. Email-address pattern is
# applied before the bare-word pattern so addresses are removed whole.
_PROVENANCE_PATTERNS = [
    re.compile(r"\b[\w.\-]+@enron\.com\b", re.IGNORECASE),
    re.compile(r"\bforwarded by\b", re.IGNORECASE),
    re.compile(r"\benron\b", re.IGNORECASE),
]


def strip_provenance(text: str) -> str:
    """Remove source-corpus provenance tokens (e.g. 'enron') that separate the
    phishing classes by origin rather than by phishing semantics."""
    for pat in _PROVENANCE_PATTERNS:
        text = pat.sub("", text)
    return text


def load_phishing_dataset(name: str) -> pd.DataFrame:
    """Returns DataFrame with columns: text (str), label (int 0/1)."""
    loaders = {
        "zefang-liu": _load_zefang_liu,
    }
    if name not in loaders:
        raise ValueError(f"Unknown phishing dataset '{name}'. Choose from: {list(loaders.keys())}")
    df = loaders[name]().reset_index(drop=True)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    n_before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} exact-duplicate texts ({len(df)} unique remain)")
    print(f"Dataset shape: {df.shape}")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    return df


# ── Network loaders ────────────────────────────────────────────────────────────

def load_cicids2017(data_dir: str | Path) -> pd.DataFrame:
    """
    Load CIC-IDS2017 network flow CSVs from data_dir.

    Expected: one or more CSV files (one per day) downloaded from Kaggle
    (chethuhn/network-intrusion-dataset). Labels are binarised:
    BENIGN -> 0, everything else -> 1.

    Returns a DataFrame with all numeric feature columns plus 'label' (int 0/1).
    Column named 'text' is NOT present — this is a tabular dataset.
    """
    import numpy as np

    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            "Download the dataset from https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset "
            "and extract the CSV files into that directory."
        )

    print(f"Loading {len(csv_files)} CSV file(s) from {data_dir} ...")
    dfs = []
    for f in csv_files:
        print(f"  Reading {f.name} ...")
        chunk = pd.read_csv(f, low_memory=False)
        # Strip leading/trailing whitespace from column names (known quirk)
        chunk.columns = chunk.columns.str.strip()
        dfs.append(chunk)

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Combined shape before cleaning: {df.shape}")

    # Strip whitespace from label values
    df["Label"] = df["Label"].astype(str).str.strip()

    # Binarise: BENIGN=0, everything else=1
    df["label"] = (df["Label"] != "BENIGN").astype(int)
    df = df.drop(columns=["Label"])

    # Convert all non-label columns to numeric, coercing errors to NaN
    feature_cols = [c for c in df.columns if c != "label"]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Replace inf values with NaN then drop metadata-like columns
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop columns that are completely non-numeric metadata
    drop_cols = [c for c in ["Flow ID", "Source IP", "Destination IP",
                              "Src IP", "Dst IP", "Timestamp"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Drop duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    print(f"  Shape after cleaning: {df.shape}")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    return df


def load_kitsune_mirai(data_dir: str | Path) -> pd.DataFrame:
    """
    Load the Kitsune Mirai botnet dataset from data_dir.

    Expected files (from Kaggle ymirsky/network-attack-dataset-kitsune):
      - Mirai_dataset.csv   (115 AfterImage features, no header)
      - Mirai_labels.csv    (one label per row: 0=benign, 1=attack)

    Also accepts:
      - Mirai_dataset.csv with a header row already present
      - A single combined file with the label as the last column

    Returns a DataFrame with 115 named feature columns plus 'label' (int 0/1).
    """
    import numpy as np
    from ml_llm_ensembles.utils.prompts import _KITSUNE_COL_NAMES

    data_dir = Path(data_dir)

    def _uniq(paths):
        seen = set()
        out = []
        for p in paths:
            if p not in seen:
                out.append(p)
                seen.add(p)
        return out

    def _sorted_existing(names):
        return [data_dir / name for name in names if (data_dir / name).exists()]

    # Prefer explicit dataset filenames and never treat label files as features.
    exact_feature_names = [
        "Mirai_dataset.csv",
        "Mirai_dataset.csv.gz",
        "mirai_dataset.csv",
        "mirai_dataset.csv.gz",
    ]
    exact_label_names = [
        "Mirai_labels.csv",
        "Mirai_labels.csv.gz",
        "mirai_labels.csv",
        "mirai_labels.csv.gz",
    ]

    feature_candidates = _sorted_existing(exact_feature_names)
    if not feature_candidates:
        feature_candidates = _uniq([
            p for p in sorted(data_dir.glob("*Mirai*dataset*"))
            if p.is_file() and "label" not in p.name.lower()
        ] + [
            p for p in sorted(data_dir.glob("*Mirai*.csv*"))
            if p.is_file() and "label" not in p.name.lower()
        ])

    label_candidates = _sorted_existing(exact_label_names)
    if not label_candidates:
        label_candidates = _uniq([
            p for p in sorted(data_dir.glob("*Mirai*label*"))
            if p.is_file()
        ] + [
            p for p in sorted(data_dir.glob("*label*.csv*"))
            if p.is_file()
        ])

    if not feature_candidates:
        found = ", ".join(sorted(p.name for p in data_dir.iterdir() if p.is_file())) or "(none)"
        raise FileNotFoundError(
            f"No Mirai dataset CSV found in {data_dir}. "
            "Download from https://www.kaggle.com/datasets/ymirsky/network-attack-dataset-kitsune "
            "and place Mirai_dataset.csv (and Mirai_labels.csv) in that directory. "
            f"Files present: {found}"
        )

    feature_file = feature_candidates[0]
    print(f"  Reading features from {feature_file.name} ...")

    # Detect if file has a header. Read the first row via header=None so pandas
    # doesn't mangle repeated values into non-numeric-looking names (e.g. the
    # AfterImage channels share seed stats at row 0, so a naive header=0 read
    # turns repeated "1.0" into "1.0.1", "1.0.2", ... which broke this check).
    first_row = pd.read_csv(feature_file, header=None, nrows=1).iloc[0]
    has_header = not all(
        str(c).replace(".", "", 1).replace("-", "", 1).lstrip("-").isdigit()
        for c in first_row
    )

    if has_header:
        df = pd.read_csv(feature_file, low_memory=False)
    else:
        df = pd.read_csv(feature_file, header=None, low_memory=False)
        if df.shape[1] == len(_KITSUNE_COL_NAMES):
            df.columns = _KITSUNE_COL_NAMES
        elif df.shape[1] == len(_KITSUNE_COL_NAMES) + 1:
            leading = df.iloc[:, 0].astype(float)
            if np.array_equal(leading.values, np.arange(len(df), dtype=float)):
                # Leading column is a literal 0..N-1 row index (the Kaggle
                # Mirai_dataset.csv release ships one) -- drop it, keep the
                # 115 named features.
                df = df.iloc[:, 1:]
                df.columns = _KITSUNE_COL_NAMES
            else:
                # Otherwise assume the trailing column is the label.
                df.columns = _KITSUNE_COL_NAMES + ["label"]
        else:
            df.columns = [f"f{i}" for i in range(df.shape[1])]

    print(f"  Shape: {df.shape}")

    # Load labels if separate file exists and label not already in df
    if "label" not in df.columns:
        if label_candidates:
            label_file = label_candidates[0]
            print(f"  Reading labels from {label_file.name} ...")
            labels = pd.read_csv(label_file, header=None).iloc[:, 0]
            df["label"] = labels.values[: len(df)]
        else:
            raise FileNotFoundError(
                f"No label file found in {data_dir} and no 'label' column in feature file. "
                "Provide Mirai_labels.csv alongside Mirai_dataset.csv."
            )

    df["label"] = df["label"].astype(int)

    # Replace inf values with NaN
    feature_cols = [c for c in df.columns if c != "label"]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df = df.drop_duplicates().reset_index(drop=True)

    print(f"  Shape after cleaning: {df.shape}")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    return df


def load_kitsune_capture_pcap(data_dir: str | Path, capture: str = "Mirai") -> pd.DataFrame:
    """Per-packet named features for ANY Kitsune capture (Mirai, SYN_DoS, SSDP_Flood, ...).

    Expects `<capture>_pcap.pcap` (or `.pcap.gz` / `.pcap.zip`) and
    `<capture>_labels.csv` (or `.csv.gz`) in data_dir, i.e. the UCI #516 layout with
    the space in the file name replaced by an underscore. The Mirai-specific loader
    below is a thin alias kept for backwards compatibility.
    """
    import gzip, shutil, zipfile
    from scapy.all import PcapReader

    data_dir = Path(data_dir)
    pcap_path = data_dir / f"{capture}_pcap.pcap"
    if not pcap_path.exists():
        gz = data_dir / f"{capture}_pcap.pcap.gz"; zp = data_dir / f"{capture}_pcap.pcap.zip"
        if gz.exists():
            print(f"  Extracting {gz.name} ...")
            with gzip.open(gz, "rb") as fin, open(pcap_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        elif zp.exists():
            print(f"  Extracting {zp.name} ...")
            with zipfile.ZipFile(zp) as z:
                z.extractall(data_dir)
        else:
            raise FileNotFoundError(
                f"No {pcap_path.name} / .gz / .zip in {data_dir}. Download the requested "
                "capture from UCI Machine Learning Repository dataset 516.")

    label_file = None
    for cand in (f"{capture}_labels.csv", f"{capture}_labels.csv.gz"):
        if (data_dir / cand).exists():
            label_file = data_dir / cand; break
    if label_file is None:
        raise FileNotFoundError(f"No {capture}_labels.csv(.gz) in {data_dir}.")
    print(f"  Reading labels from {label_file.name} ...")
    lab = pd.read_csv(label_file, header=None, low_memory=False)
    # Mirai labels are a single 0/1 column; the other UCI captures ship a header
    # row plus (index, label) columns. Take the last column, coerce, drop non-numeric.
    labels = pd.to_numeric(lab.iloc[:, -1], errors="coerce")
    labels = labels[labels.notna()].astype(float).round().astype(int).tolist()

    print(f"  Parsing {pcap_path.name} ({len(labels)} packets expected) ...")
    rows = []
    prev_time = None
    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            row, prev_time = extract_packet_features(pkt, prev_time)
            rows.append(row)
    df = pd.DataFrame(rows)
    n = min(len(df), len(labels))
    if abs(len(df) - len(labels)) > 1:
        print(f"  WARNING: {len(df)} packets parsed vs {len(labels)} labels; truncating to {n}")
    df = df.iloc[:n].reset_index(drop=True)
    df["label"] = labels[:n]
    df["protocol_int"] = df["protocol"].map(_PCAP_PROTO_INT_MAP)
    df["flow_id"] = (
        df["src_ip"].astype(str) + "_" + df["dst_ip"].astype(str) + "_"
        + df["src_port"].astype(str) + "_" + df["dst_port"].astype(str) + "_"
        + df["protocol_int"].astype(str)
    )
    print(f"  Parsed {len(df)} packets")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    return df


def load_kitsune_capture_pcap_flows(data_dir: str | Path, capture: str = "Mirai") -> pd.DataFrame:
    """Flow-level (5-tuple) aggregation of any Kitsune capture; keeps `start_time`."""
    pkt_df = load_kitsune_capture_pcap(data_dir, capture)
    print("  Aggregating packets into flows ...")
    return aggregate_pcap_packets_to_flows(pkt_df)


def load_kitsune_mirai_pcap(data_dir: str | Path) -> pd.DataFrame:
    """
    Load the Kitsune Mirai dataset from the raw pcap file, extracting per-packet
    named features that LLMs can reason about semantically.

    Features extracted per packet:
      - protocol: TCP / UDP / ICMP / OTHER
      - src_port, dst_port: integer port numbers (0 if not TCP/UDP)
      - pkt_len: total packet length in bytes
      - ttl: IP time-to-live
      - tcp_flags: SYN / ACK / FIN / RST / PSH / URG flags as 0/1 columns
      - interarrival_ms: time since previous packet in milliseconds

    Labels come from Mirai_labels.csv (same order as packets in the pcap).

    Returns a DataFrame with named feature columns plus 'label' (int 0/1).
    """
    import zipfile
    from scapy.all import PcapReader

    data_dir = Path(data_dir)
    pcap_path = data_dir / "Mirai_pcap.pcap"
    zip_path  = data_dir / "Mirai_pcap.pcap.zip"

    if not pcap_path.exists():
        if zip_path.exists():
            print(f"  Extracting {zip_path.name} ...")
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(data_dir)
        else:
            raise FileNotFoundError(
                f"Neither {pcap_path} nor {zip_path} found. "
                "Download Mirai_pcap.pcap.zip from the Kitsune Kaggle dataset."
            )

    label_candidates = list(data_dir.glob("*Mirai*label*")) + list(data_dir.glob("*label*.csv"))
    if not label_candidates:
        raise FileNotFoundError(
            f"No label file found in {data_dir}. "
            "Place Mirai_labels.csv alongside the pcap file."
        )
    label_file = label_candidates[0]
    print(f"  Reading labels from {label_file.name} ...")
    labels = pd.read_csv(label_file, header=None).iloc[:, 0].astype(int).tolist()

    print(f"  Parsing {pcap_path.name} ({len(labels)} packets expected) ...")

    rows = []
    prev_time = None
    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            row, prev_time = extract_packet_features(pkt, prev_time)
            rows.append(row)

    df = pd.DataFrame(rows)

    # Align with labels (pcap row count may differ by 1 due to header)
    n = min(len(df), len(labels))
    df = df.iloc[:n].reset_index(drop=True)
    df["label"] = labels[:n]

    df["protocol_int"] = df["protocol"].map(_PCAP_PROTO_INT_MAP)

    # 5-tuple flow id (same key used by aggregate_pcap_packets_to_flows) so a
    # train/test split can keep all packets of a flow on the same side.
    df["flow_id"] = (
        df["src_ip"].astype(str) + "_" + df["dst_ip"].astype(str) + "_"
        + df["src_port"].astype(str) + "_" + df["dst_port"].astype(str) + "_"
        + df["protocol_int"].astype(str)
    )

    print(f"  Parsed {len(df)} packets")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    return df


def load_kitsune_mirai_pcap_flows(data_dir: str | Path) -> pd.DataFrame:
    """
    Load Kitsune Mirai pcap and aggregate to per-flow rows (5-tuple groupby).

    This is the flow-level counterpart of load_kitsune_mirai_pcap. Each row is
    a flow with summary statistics suitable for both XGBoost and LLM prompts.
    A flow is labeled as attack (1) if any of its constituent packets is.
    """
    pkt_df = load_kitsune_mirai_pcap(data_dir)
    print("  Aggregating packets into flows ...")
    return aggregate_pcap_packets_to_flows(pkt_df)



# ── CTU-13 (Stratosphere, 2011): 13 botnet scenarios, bidirectional NetFlow ────
CTU13_SCENARIO_BOTNET = {
    1: "Neris", 2: "Neris", 3: "Rbot", 4: "Rbot", 5: "Virut", 6: "Menti", 7: "Sogou",
    8: "Murlo", 9: "Neris", 10: "Rbot", 11: "Rbot", 12: "NSIS.ay", 13: "Virut",
}
# Garcia et al. 2014 evaluation protocol: train on {3,4,5,7,10,11,12,13}, test on
# {1,2,6,8,9} so that every test botnet family is unseen or a different capture.
CTU13_DEFAULT_TEST_SCENARIOS = [1, 2, 6, 8, 9]
CTU13_INTERNAL_PREFIX = "147.32."      # CTU university /16; infected + normal hosts live here

_CTU_DIR_FWD = {"->", "?>"}
_CTU_DIR_BWD = {"<-", "<?"}
_CTU_DIR_BI = {"<->", "<?>"}


def _ctu_port(x):
    try:
        return int(str(x).strip(), 0)     # handles decimal and 0x hex (ICMP codes)
    except (ValueError, TypeError):
        return -1                          # sentinel; counted and reported by the loader


def load_ctu13(data_dir: str | Path, scenarios=None, background: str = "benign",
               sample_per_scenario: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Load CTU-13 bidirectional NetFlow scenarios into one DataFrame.

    Layout: <data_dir>/CTU-13-Dataset/<n>/*.binetflow (from the official tarball)
    or <data_dir>/<n>/*.binetflow.

    Label handling (the `Label` column is free text, e.g.
    'flow=From-Botnet-V42-UDP-DNS', 'flow=Background-...', 'flow=From-Normal-...'):
      * Botnet     -> 1
      * Normal     -> 0   (verified-normal hosts)
      * Background -> 0 when background='benign' (DEFAULT). Background is "all the
        rest of traffic that we don't know what it is for sure" (Stratosphere), so
        the resulting prior is the OBSERVED prevalence of the Botnet label among
        all captured flows, and measured FPR / precision are relative to noisy
        negatives. background='drop' keeps Botnet vs Normal only (clean labels,
        unrealistic prior; a sensitivity analysis, not the headline).
      Any label matching none of the three kinds raises.

    `sample_per_scenario` draws a uniform random subset per scenario. It exists
    for smoke tests and for TRAINING-side thinning; do NOT use it on the rows you
    report natural-prevalence metrics on (equal per-scenario caps re-weight
    scenarios of different sizes and shift the pooled prior).

    Returns numeric feature columns (prompts.CTU13_FEATURE_COLS) plus metadata:
      label, label_kind (category: botnet/normal/background), scenario (int),
      botnet (family), start_time (epoch s, float), src_ip, dst_ip (category),
      proto, dir, state (category), host (grouping key: the internal 147.32/16
      endpoint, source if both or neither are internal), flow_id (5-tuple).
    """
    import glob
    data_dir = Path(data_dir)
    root = data_dir / "CTU-13-Dataset" if (data_dir / "CTU-13-Dataset").exists() else data_dir
    scenarios = list(range(1, 14)) if scenarios is None else [int(x) for x in scenarios]
    usecols = ["StartTime", "Dur", "Proto", "SrcAddr", "Sport", "Dir", "DstAddr", "Dport",
               "State", "sTos", "dTos", "TotPkts", "TotBytes", "SrcBytes", "Label"]
    frames = []
    for sc in scenarios:
        files = sorted(glob.glob(str(root / str(sc) / "*.binetflow")))
        if not files:
            raise FileNotFoundError(f"No .binetflow under {root / str(sc)} (scenario {sc}). "
                                    "Extract CTU-13-Dataset.tar.bz2 into data/ctu13.")
        print(f"  CTU-13 scenario {sc:2d} ({CTU13_SCENARIO_BOTNET.get(sc, '?')}): {Path(files[0]).name}")
        df = pd.read_csv(files[0], usecols=usecols, low_memory=False,
                         dtype={"Sport": str, "Dport": str, "Proto": str, "Dir": str, "State": str,
                                "SrcAddr": str, "DstAddr": str, "Label": str})
        if sample_per_scenario and sample_per_scenario < len(df):
            df = df.sample(n=sample_per_scenario, random_state=seed + sc)
        df["scenario"] = sc
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    del frames

    lab = df["Label"].astype(str)
    is_bot = lab.str.contains("Botnet", case=False).values
    is_norm = lab.str.contains("Normal", case=False).values & ~is_bot
    is_bg = lab.str.contains("Background", case=False).values & ~is_bot & ~is_norm
    unknown = ~(is_bot | is_norm | is_bg)
    if unknown.any():
        raise ValueError(f"{int(unknown.sum())} CTU-13 labels match none of Botnet/Normal/Background, "
                         f"e.g. {lab[unknown].unique()[:5].tolist()}")
    if background == "drop":
        keep = is_bot | is_norm
        df = df[keep].reset_index(drop=True); is_bot, is_norm, is_bg = is_bot[keep], is_norm[keep], is_bg[keep]
    elif background != "benign":
        raise ValueError("background must be 'benign' or 'drop'")
    label_kind = np.where(is_bot, "botnet", np.where(is_norm, "normal", "background"))

    ts = pd.to_datetime(df["StartTime"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    if ts.isna().any():
        raise ValueError(f"{int(ts.isna().sum())} CTU-13 StartTime values failed to parse, "
                         f"e.g. {df['StartTime'][ts.isna()].head(3).tolist()}")
    sport = df["Sport"].map(_ctu_port); dport = df["Dport"].map(_ctu_port)
    n_bad_port = int((sport < 0).sum() + (dport < 0).sum())
    dir_s = df["Dir"].astype(str).str.strip()
    src = df["SrcAddr"].astype(str); dst = df["DstAddr"].astype(str)
    src_int = src.str.startswith(CTU13_INTERNAL_PREFIX).values
    dst_int = dst.str.startswith(CTU13_INTERNAL_PREFIX).values
    host = np.where(dst_int & ~src_int, dst.values, src.values)

    out = pd.DataFrame({
        "start_time": ts.values.astype("datetime64[ns]").astype("int64") / 1e9,
        "dur": pd.to_numeric(df["Dur"], errors="coerce"),
        "proto": df["Proto"].astype(str).str.lower().astype("category"),
        "src_ip": src.astype("category"), "dst_ip": dst.astype("category"),
        "sport": sport.astype("int64"), "dport": dport.astype("int64"),
        "dir": dir_s.astype("category"), "state": df["State"].astype(str).astype("category"),
        "stos": pd.to_numeric(df["sTos"], errors="coerce"),
        "dtos": pd.to_numeric(df["dTos"], errors="coerce"),
        "tot_pkts": pd.to_numeric(df["TotPkts"], errors="coerce"),
        "tot_bytes": pd.to_numeric(df["TotBytes"], errors="coerce"),
        "src_bytes": pd.to_numeric(df["SrcBytes"], errors="coerce"),
        "label": is_bot.astype(int), "label_kind": pd.Categorical(label_kind),
        "scenario": df["scenario"].values.astype("int16"),
        "host": pd.Categorical(host),
    })
    del df
    out["botnet"] = out["scenario"].map(CTU13_SCENARIO_BOTNET).astype("category")
    proto_map = {"tcp": 6, "udp": 17, "icmp": 1}
    out["proto_int"] = out["proto"].astype(str).map(proto_map).fillna(0).astype("int8")
    out["dir_fwd"] = dir_s.isin(_CTU_DIR_FWD).values.astype("int8")
    out["dir_bwd"] = dir_s.isin(_CTU_DIR_BWD).values.astype("int8")
    out["dir_bi"] = dir_s.isin(_CTU_DIR_BI).values.astype("int8")
    out["bytes_per_pkt"] = out["tot_bytes"] / out["tot_pkts"].replace(0, np.nan)
    out["src_byte_ratio"] = out["src_bytes"] / out["tot_bytes"].replace(0, np.nan)
    out["flow_id"] = (out["src_ip"].astype(str) + "_" + out["dst_ip"].astype(str) + "_"
                      + out["sport"].astype(str) + "_" + out["dport"].astype(str) + "_"
                      + out["proto_int"].astype(str)).astype("category")
    out = out.sort_values(["scenario", "start_time"], kind="stable").reset_index(drop=True)
    n_bot = int(out["label"].sum())
    print(f"  CTU-13: {len(out)} flows across {out['scenario'].nunique()} scenarios; "
          f"botnet {n_bot} ({n_bot / len(out):.4%}); normal {int((label_kind == 'normal').sum())}; "
          f"background policy = {background}; malformed ports -> -1: {n_bad_port}")
    return out


def load_network_dataset(name: str, data_dir: str | Path) -> pd.DataFrame:
    """
    Load a network tabular dataset by name.
    Returns a DataFrame with numeric feature columns plus 'label' (int 0/1).
    No 'text' column — use format_network_row_* from utils.prompts for LLM input.
    """
    loaders = {
        "cicids2017": load_cicids2017,
        "kitsune-mirai": load_kitsune_mirai,
        "kitsune-mirai-pcap": load_kitsune_mirai_pcap,
        "kitsune-mirai-pcap-flows": load_kitsune_mirai_pcap_flows,
        "kitsune-syndos-pcap": lambda d: load_kitsune_capture_pcap(d, "SYN_DoS"),
        "kitsune-syndos-pcap-flows": lambda d: load_kitsune_capture_pcap_flows(d, "SYN_DoS"),
        "kitsune-ssdp-pcap": lambda d: load_kitsune_capture_pcap(d, "SSDP_Flood"),
        "kitsune-ssdp-pcap-flows": lambda d: load_kitsune_capture_pcap_flows(d, "SSDP_Flood"),
        "ctu13": load_ctu13,
    }
    if name not in loaders:
        raise ValueError(f"Unknown network dataset '{name}'. Choose from: {list(loaders.keys())}")
    return loaders[name](data_dir)


def load_dataset(name: str, data_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Unified dataset loader. Dispatches by dataset name.

    For the phishing dataset (zefang-liu):
      Returns DataFrame with columns: text (str), label (int 0/1).

    For network datasets (cicids2017, kitsune-mirai):
      Returns DataFrame with numeric feature columns + label (int 0/1).
      data_dir is required.
    """
    if name in PHISHING_DATASETS:
        return load_phishing_dataset(name)
    if name in NETWORK_DATASETS:
        if data_dir is None:
            raise ValueError(f"data_dir is required for network dataset '{name}'.")
        return load_network_dataset(name, data_dir)
    raise ValueError(
        f"Unknown dataset '{name}'. "
        f"Phishing: {PHISHING_DATASETS}. Network: {NETWORK_DATASETS}."
    )
