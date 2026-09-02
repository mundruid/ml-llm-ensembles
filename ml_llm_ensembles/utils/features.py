import re
import math
from collections import Counter

import numpy as np
import pandas as pd

# ── Network flow feature utilities ────────────────────────────────────────────

# Columns to drop from CIC-IDS2017 (metadata, not useful as ML features)
CICIDS_DROP_COLS = [
    "Flow ID", "Source IP", "Destination IP", "Src IP", "Dst IP", "Timestamp",
]

# Columns known to contain Inf values in CIC-IDS2017 (replaced with NaN then imputed)
CICIDS_KNOWN_INF_COLS = ["Flow Bytes/s", "Flow Packets/s"]


def build_network_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Drop metadata columns, coerce to numeric, replace Inf, fill NaN with medians."""
    drop = [c for c in CICIDS_DROP_COLS if c in df.columns]
    if drop:
        df = df.drop(columns=drop)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))

    return df.astype(np.float64)

FEATURE_GROUPS = {
    "text_stats": ["text_length", "word_count", "avg_word_length"],
    "url_analysis": [
        "url_count",
        "has_url",
        "shortened_url_count",
        "ip_url_count",
        "suspicious_tld_count",
        "url_to_text_ratio",
    ],
    "keywords": ["phishing_keyword_count", "ambiguous_keyword_count"],
    "threat_language": ["threat_phrase_count"],
    "credential_harvesting": ["credential_request_count"],
    "brand_impersonation": ["brand_mention_count"],
    "generic_greetings": ["has_generic_greeting"],
    "special_characters": [
        "exclamation_count",
        "question_count",
        "dollar_sign_count",
        "at_sign_count",
    ],
    "casing_digits": ["uppercase_ratio", "numeric_ratio"],
    "html_structure": ["html_tag_count", "form_tag_count", "js_event_handler_count"],
    "obfuscation": ["leetspeak_count", "has_base64"],
    "attachments": ["attachment_reference_count"],
    "information_theory": ["char_entropy", "vocab_richness", "line_count"],
}

ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]


def extract_phishing_email_features(text: str) -> dict:
    """Extract all 30 features from a single text."""
    if not isinstance(text, str):
        text = ""
    features = {}
    text_lower = text.lower()
    words = text.split()

    # --- Text length features ---
    features['text_length'] = len(text)
    features['word_count'] = len(words)
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0

    # --- URL-related features ---
    urls = re.findall(r'https?://[^\s]+', text_lower)
    features['url_count'] = len(urls)
    features['has_url'] = 1 if urls else 0

    # Shortened URLs (bit.ly, tinyurl, t.co, goo.gl, etc.) — strong phishing signal
    shorteners = ['bit.ly', 'tinyurl', 't.co', 'goo.gl', 'ow.ly', 'is.gd',
                  'buff.ly', 'adf.ly', 'bl.ink', 'lnkd.in', 'rb.gy']
    features['shortened_url_count'] = sum(
        1 for url in urls if any(s in url for s in shorteners)
    )

    # IP-address URLs (http://192.168.1.1/...) — almost never legitimate
    features['ip_url_count'] = sum(
        1 for url in urls if re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url)
    )

    # Suspicious TLDs
    suspicious_tlds = ['.xyz', '.top', '.buzz', '.info', '.tk', '.ml', '.ga', '.cf', '.gq', '.pw']
    features['suspicious_tld_count'] = sum(
        1 for url in urls if any(url.rstrip('/').endswith(tld) or tld + '/' in url for tld in suspicious_tlds)
    )

    # URL-to-text ratio (phishing = short message, many links)
    features['url_to_text_ratio'] = len(urls) / max(len(words), 1)

    # --- Phishing-specific keywords (high precision, low marketing overlap) ---
    phishing_keywords = [
        'verify', 'account', 'suspended', 'confirm', 'password',
        'security', 'bank', 'credit', 'login', 'ssn', 'social security'
    ]
    features['phishing_keyword_count'] = sum(1 for kw in phishing_keywords if kw in text_lower)

    # Marketing-ambiguous keywords (separated so XGBoost can learn independent weight)
    ambiguous_keywords = ['urgent', 'click', 'expire', 'immediately', 'update', 'limited time']
    features['ambiguous_keyword_count'] = sum(1 for kw in ambiguous_keywords if kw in text_lower)

    # Threat / coercion language — phishing-specific, marketing uses positive framing
    threat_phrases = [
        'unauthorized', 'will be locked', 'will be closed', 'will be suspended',
        'suspicious activity', 'unusual activity', 'failure to', 'legal action',
        'your account has been', 'we detected'
    ]
    features['threat_phrase_count'] = sum(1 for phrase in threat_phrases if phrase in text_lower)

    # Credential request language — directly asking for sensitive info
    credential_phrases = [
        'enter your password', 'verify your identity', 'confirm your account',
        'update your payment', 'enter your ssn', 'verify your account',
        'confirm your identity', 'update your information'
    ]
    features['credential_request_count'] = sum(1 for phrase in credential_phrases if phrase in text_lower)

    # Brand impersonation — brand names in body text (phishing pretends to be these)
    impersonated_brands = [
        'paypal', 'netflix', 'amazon', 'apple id', 'microsoft',
        'wells fargo', 'chase bank', 'irs', 'fedex', 'ups',
        'dhl', 'usps', 'coinbase', 'binance'
    ]
    features['brand_mention_count'] = sum(1 for brand in impersonated_brands if brand in text_lower)

    # --- Generic greeting detection ---
    generic_greetings = [
        'dear customer', 'dear user', 'dear account holder', 'dear client',
        'dear member', 'dear valued', 'dear sir', 'dear madam'
    ]
    features['has_generic_greeting'] = 1 if any(g in text_lower for g in generic_greetings) else 0

    # --- Special character features ---
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['dollar_sign_count'] = text.count('$')
    features['at_sign_count'] = text.count('@')

    # --- Uppercase ratio ---
    alpha_chars = [c for c in text if c.isalpha()]
    features['uppercase_ratio'] = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) if alpha_chars else 0

    # --- Numeric character ratio ---
    features['numeric_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0

    # --- HTML / structural features ---
    html_tags = re.findall(r'<[^>]+>', text)
    features['html_tag_count'] = len(html_tags)

    # Form tags — embedded credential harvesting forms
    features['form_tag_count'] = len(re.findall(r'<form[\s>]', text_lower))

    # JavaScript / event handlers in HTML
    features['js_event_handler_count'] = len(
        re.findall(r'(?:onclick|onload|onmouseover|onerror|javascript:)', text_lower)
    )

    # --- Obfuscation features ---
    # Leetspeak / character substitution (p@ssw0rd, cl1ck, etc.)
    features['leetspeak_count'] = len(re.findall(r'[a-zA-Z]+[@$!0-9]+[a-zA-Z]+', text))

    # Base64-like or hex-encoded strings
    features['has_base64'] = 1 if re.search(r'[A-Za-z0-9+/]{20,}={0,2}', text) else 0

    # --- Attachment reference ---
    attachment_phrases = ['see attached', 'open the attachment', 'attached file',
                         'attached document', 'see the attached', 'open attached']
    features['attachment_reference_count'] = sum(1 for phrase in attachment_phrases if phrase in text_lower)

    # --- Statistical text features ---
    # Character entropy (Shannon) — templated phishing has distinct entropy profile
    if text:
        char_counts = Counter(text)
        total = len(text)
        features['char_entropy'] = -sum(
            (count / total) * math.log2(count / total) for count in char_counts.values()
        )
    else:
        features['char_entropy'] = 0.0

    # Vocabulary richness (distinct words / total words)
    lower_words = [w.lower() for w in words]
    features['vocab_richness'] = len(set(lower_words)) / len(lower_words) if lower_words else 0

    # Line count
    features['line_count'] = text.count('\n') + 1

    return features


def build_phishing_email_feature_matrix(
    df: pd.DataFrame, features: list[str] | None = None
) -> pd.DataFrame:
    """
    Apply extract_phishing_email_features to df['text'], return feature DataFrame.
    If features is None, return all 30. Otherwise filter to the given list.
    """
    print("Extracting features...")
    texts = df["text"].astype(str).fillna("")
    feature_dicts = texts.apply(extract_phishing_email_features).tolist()
    features_df = pd.DataFrame(feature_dicts)

    if features is not None:
        missing = set(features) - set(features_df.columns)
        if missing:
            raise ValueError(f"Unknown features: {missing}")
        features_df = features_df[features]

    print(f"Extracted {len(features_df.columns)} features")
    print(f"\nFeature list: {features_df.columns.tolist()}")
    return features_df


# ── Pcap / packet-level feature utilities ─────────────────────────────────────

_PCAP_PROTO_MAP = {6: "TCP", 17: "UDP", 1: "ICMP"}   # IP proto number → name
_PCAP_PROTO_INT_MAP = {"TCP": 6, "UDP": 17, "ICMP": 1, "OTHER": 0}  # name → int


def extract_packet_features(pkt, prev_time: float | None) -> tuple[dict, float]:
    """Extract per-packet named features from a Scapy packet.

    Returns (feature_row_dict, current_packet_time).
    interarrival_ms is capture-order (global) IAT — do NOT use for within-flow stats.
    """
    from scapy.all import IP, TCP, UDP

    t = float(pkt.time)
    interarrival_ms = (t - prev_time) * 1000.0 if prev_time is not None else 0.0

    row = {
        "time":            t,
        "src_ip":          "",
        "dst_ip":          "",
        "protocol":        "OTHER",
        "src_port":        0,
        "dst_port":        0,
        "pkt_len":         len(pkt),
        "ttl":             0,
        "tcp_flag_syn":    0,
        "tcp_flag_ack":    0,
        "tcp_flag_fin":    0,
        "tcp_flag_rst":    0,
        "tcp_flag_psh":    0,
        "tcp_flag_urg":    0,
        "interarrival_ms": round(interarrival_ms, 3),  # global IAT — do NOT use for flow stats
    }

    if IP in pkt:
        ip = pkt[IP]
        row["src_ip"]   = ip.src
        row["dst_ip"]   = ip.dst
        row["ttl"]      = ip.ttl
        row["protocol"] = _PCAP_PROTO_MAP.get(ip.proto, "OTHER")

        if TCP in pkt:
            tcp = pkt[TCP]
            row["src_port"]     = tcp.sport
            row["dst_port"]     = tcp.dport
            flags = tcp.flags
            row["tcp_flag_syn"] = int(flags & 0x02 != 0)
            row["tcp_flag_ack"] = int(flags & 0x10 != 0)
            row["tcp_flag_fin"] = int(flags & 0x01 != 0)
            row["tcp_flag_rst"] = int(flags & 0x04 != 0)
            row["tcp_flag_psh"] = int(flags & 0x08 != 0)
            row["tcp_flag_urg"] = int(flags & 0x20 != 0)
        elif UDP in pkt:
            udp = pkt[UDP]
            row["src_port"] = udp.sport
            row["dst_port"] = udp.dport

    return row, t


def aggregate_pcap_packets_to_flows(pkt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-packet rows into per-flow rows grouped by 5-tuple.

    IATs are computed from per-flow sorted timestamps via np.diff — NOT from
    the global `interarrival_ms` column, which is time since the previous packet
    in the entire capture and produces meaningless within-flow statistics.
    """
    import numpy as np

    required = {"time", "src_ip", "dst_ip"}
    missing = required - set(pkt_df.columns)
    if missing:
        raise ValueError(
            f"aggregate_pcap_packets_to_flows requires columns {required}; "
            f"missing: {missing}. Regenerate the DataFrame with load_kitsune_mirai_pcap."
        )

    key_cols = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol_int"]

    flows = []
    for key, grp in pkt_df.groupby(key_cols, sort=False):
        src_ip, dst_ip, src_port, dst_port, proto = key

        times    = grp["time"].sort_values().values
        iats     = np.diff(times) * 1000.0
        pkt_lens = grp["pkt_len"].values

        flow = {
            "src_ip":       src_ip,
            "dst_ip":       dst_ip,
            "src_port":     int(src_port),
            "dst_port":     int(dst_port),
            "protocol_int": int(proto),
            "protocol":     _PCAP_PROTO_MAP.get(int(proto), "OTHER"),
            "pkt_count":    int(len(grp)),
            "total_bytes":  int(pkt_lens.sum()),
            "mean_pkt_len": float(pkt_lens.mean()),
            "max_pkt_len":  int(pkt_lens.max()),
            "start_time":   float(times[0]),
            "duration_ms":  float((times[-1] - times[0]) * 1000.0),
            "mean_iat_ms":  float(iats.mean()) if len(iats) > 0 else 0.0,
            "std_iat_ms":   float(iats.std())  if len(iats) > 1 else 0.0,
            "syn_count":    int(grp["tcp_flag_syn"].sum()) if "tcp_flag_syn" in grp.columns else 0,
            "ack_count":    int(grp["tcp_flag_ack"].sum()) if "tcp_flag_ack" in grp.columns else 0,
            "fin_count":    int(grp["tcp_flag_fin"].sum()) if "tcp_flag_fin" in grp.columns else 0,
            "rst_count":    int(grp["tcp_flag_rst"].sum()) if "tcp_flag_rst" in grp.columns else 0,
            "psh_count":    int(grp["tcp_flag_psh"].sum()) if "tcp_flag_psh" in grp.columns else 0,
            "mean_ttl":     float(grp["ttl"].mean()) if "ttl" in grp.columns else 0.0,
            "label":        int(grp["label"].max()),
        }
        flows.append(flow)

    df = pd.DataFrame(flows)
    print(f"  Flows: {len(df)}  (attack: {int(df['label'].sum())}, benign: {int((df['label']==0).sum())})")
    return df
