"""
Domain-specific LLM prompt templates and classifiers.
======================================================
Each domain provides:
  - A prompt template with a {text} placeholder
  - A row formatter that converts a DataFrame row to the text slot

Backends supported: ollama, claude, gemini, modernbert-ft, decoder-ft
"""

import os
import re

MAX_TEXT_LENGTH = 1000

# ── Prompt templates ──────────────────────────────────────────────────────────

PHISHING_PROMPT = """\
Analyze the following email/message and determine if it is a phishing attempt or legitimate.

Message:
---
{text}
---

Respond with ONLY a JSON object in this exact format:
{{"prediction": 0 or 1, "confidence": 0.0 to 1.0}}

Where:
- prediction: 1 for phishing, 0 for legitimate
- confidence: your confidence level (0.0 to 1.0)

JSON response:"""


NETWORK_PROMPT = """\
Analyze the following network flow record and determine if it represents an attack or benign traffic.

Flow:
---
{text}
---

Respond with ONLY a JSON object in this exact format:
{{"prediction": 0 or 1, "confidence": 0.0 to 1.0}}

Where:
- prediction: 1 for attack/malicious, 0 for benign
- confidence: your confidence level (0.0 to 1.0)

JSON response:"""


DOMAIN_PROMPTS: dict[str, str] = {
    "phishing": PHISHING_PROMPT,
    "network": NETWORK_PROMPT,
}

# ── Network row → text formatter ──────────────────────────────────────────────

# CIC-IDS2017 columns to include in the LLM prompt (curated for semantic signal).
# Kept small and interpretable. Note: CIC-IDS2017 CSVs do NOT contain a 'Protocol'
# column — protocol is implicit in Destination Port (well-known services).
CICIDS_LLM_COLS = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Flow IAT Mean", "Flow IAT Std",
    "SYN Flag Count", "ACK Flag Count", "FIN Flag Count", "RST Flag Count",
    "Packet Length Mean",
]

_PROTO_NAMES = {6: "TCP", 17: "UDP", 1: "ICMP", 0: "OTHER"}

# Kitsune AfterImage feature names (115 features).
# NOTE: neither the Kaggle release nor the UCI mirror of this dataset publish a
# per-column data dictionary for this CSV. Because the exact within-block
# statistic ordering is not verifiable, the loader uses positional names and
# format_network_row_kitsune serializes them without assigning semantics.
_KITSUNE_COL_NAMES = [f"kitsune_f{i}" for i in range(115)]

# Kitsune LLM prompt columns -- a fixed positional subset (first 12 of the 115),
# not a claim about which stats are most "interpretable" (see note above).
KITSUNE_LLM_COLS = _KITSUNE_COL_NAMES[:12]


def format_network_row_cicids(row) -> str:
    """TabLLM-style natural-language sentence from a CIC-IDS2017 flow row."""
    def _get(name, default=0):
        return row[name] if name in row.index else default

    dst_port  = int(_get("Destination Port"))
    dur_us    = float(_get("Flow Duration"))
    dur_ms    = dur_us / 1000.0
    fwd_pkts  = int(_get("Total Fwd Packets"))
    bwd_pkts  = int(_get("Total Backward Packets"))
    fwd_bytes = int(_get("Total Length of Fwd Packets"))
    bwd_bytes = int(_get("Total Length of Bwd Packets"))
    iat_mean  = float(_get("Flow IAT Mean"))
    iat_std   = float(_get("Flow IAT Std"))
    pkt_len   = float(_get("Packet Length Mean"))
    syn       = int(_get("SYN Flag Count"))
    ack       = int(_get("ACK Flag Count"))
    fin       = int(_get("FIN Flag Count"))
    rst       = int(_get("RST Flag Count"))

    sentence = (
        f"A flow on destination port {_port_label(dst_port)}, "
        f"lasting {dur_ms:.2f} ms, with {fwd_pkts} forward packets "
        f"({fwd_bytes} bytes) and {bwd_pkts} backward packets ({bwd_bytes} bytes). "
        f"Mean inter-arrival {iat_mean:.1f} us (std {iat_std:.1f} us); "
        f"mean packet size {pkt_len:.0f} bytes. "
        f"TCP flags observed: SYN={syn}, ACK={ack}, FIN={fin}, RST={rst}."
    )
    return sentence


def format_network_row_kitsune(row, columns: list[str] | None = None) -> str:
    """Serialize a Kitsune packet row to natural language for an LLM prompt."""
    cols = columns or KITSUNE_LLM_COLS
    parts = []
    for col in cols:
        if col not in row.index:
            continue
        val = row[col]
        if isinstance(val, float):
            parts.append(f"{col}={val:.4g}")
        else:
            parts.append(f"{col}={val}")
    return "Packet stats: " + ", ".join(parts)


# Well-known port number → service name (for LLM context)
_WELL_KNOWN_PORTS = {
    20: "FTP-data", 21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP",
    53: "DNS", 67: "DHCP", 68: "DHCP", 80: "HTTP", 110: "POP3",
    123: "NTP", 143: "IMAP", 161: "SNMP", 179: "BGP", 443: "HTTPS",
    445: "SMB", 465: "SMTPS", 993: "IMAPS", 995: "POP3S",
    1080: "SOCKS", 3306: "MySQL", 3389: "RDP", 5900: "VNC",
    6881: "BitTorrent", 8080: "HTTP-alt", 8443: "HTTPS-alt",
    137: "NetBIOS-NS", 138: "NetBIOS-DGM", 139: "NetBIOS-SSN",
    1900: "SSDP", 3128: "HTTP-proxy", 5060: "SIP", 6666: "IRC", 6667: "IRC", 6697: "IRC-TLS",
    10240: "Mirai-C2",  # Mirai botnet known C2 port
    23231: "Mirai-report",
}


def _port_label(port: int) -> str:
    """Return 'port(Service)' or just 'port' for unknown ports."""
    name = _WELL_KNOWN_PORTS.get(int(port))
    return f"{int(port)}({name})" if name else str(int(port))


def format_network_row_kitsune_pcap(row) -> str:
    """
    Serialize a pcap-derived Kitsune packet row to natural language.

    Uses named, interpretable fields — protocol, ports, packet length, TTL,
    TCP flags, interarrival time — that an LLM has semantic priors about.
    """
    proto = row.get("protocol", "OTHER")
    src_port = int(row.get("src_port", 0))
    dst_port = int(row.get("dst_port", 0))
    pkt_len = int(row.get("pkt_len", 0))
    ttl = int(row.get("ttl", 0))
    interarrival = float(row.get("interarrival_ms", 0.0))

    parts = [
        f"protocol={proto}",
        f"src_port={_port_label(src_port)}",
        f"dst_port={_port_label(dst_port)}",
        f"pkt_len={pkt_len}",
        f"ttl={ttl}",
        f"interarrival_ms={interarrival:.2f}",
    ]

    # Add TCP flags only for TCP packets where any flag is set
    if proto == "TCP":
        flag_names = {
            "tcp_flag_syn": "SYN",
            "tcp_flag_ack": "ACK",
            "tcp_flag_fin": "FIN",
            "tcp_flag_rst": "RST",
            "tcp_flag_psh": "PSH",
            "tcp_flag_urg": "URG",
        }
        active = [name for col, name in flag_names.items() if int(row.get(col, 0)) == 1]
        parts.append(f"flags=[{','.join(active) if active else 'none'}]")

    return "Packet: " + ", ".join(parts)


# Columns to use as XGBoost features for the pcap-derived dataset
# (drop 'protocol' string column; use 'protocol_int' instead)
KITSUNE_PCAP_FEATURE_COLS = [
    "protocol_int", "src_port", "dst_port", "pkt_len", "ttl",
    "tcp_flag_syn", "tcp_flag_ack", "tcp_flag_fin",
    "tcp_flag_rst", "tcp_flag_psh", "tcp_flag_urg",
    "interarrival_ms",
]


# Columns to use as XGBoost features for the flow-aggregated pcap dataset
KITSUNE_PCAP_FLOW_FEATURE_COLS = [
    "protocol_int", "src_port", "dst_port",
    "pkt_count", "total_bytes", "mean_pkt_len", "max_pkt_len",
    "duration_ms", "mean_iat_ms", "std_iat_ms",
    "syn_count", "ack_count", "fin_count", "rst_count", "psh_count",
    "mean_ttl",
]


def format_network_row_kitsune_pcap_flow(row) -> str:
    """TabLLM natural-language sentence for LLMs. Do NOT use with modernbert-mirai-flows-ft — see format_network_row_kitsune_pcap_flow_ft."""
    proto    = row.get("protocol", "OTHER")
    src_port = int(row.get("src_port", 0))
    dst_port = int(row.get("dst_port", 0))
    pkts     = int(row.get("pkt_count", 0))
    bytes_   = int(row.get("total_bytes", 0))
    dur_ms   = float(row.get("duration_ms", 0.0))
    iat_mean = float(row.get("mean_iat_ms", 0.0))
    iat_std  = float(row.get("std_iat_ms", 0.0))
    mean_len = float(row.get("mean_pkt_len", 0.0))

    sentence = (
        f"A {proto} flow from port {_port_label(src_port)} to port {_port_label(dst_port)}, "
        f"with {pkts} packets totaling {bytes_} bytes over {dur_ms:.1f} ms "
        f"(mean inter-arrival {iat_mean:.2f} ms, std {iat_std:.2f} ms; "
        f"mean packet size {mean_len:.0f} bytes)"
    )

    if proto == "TCP":
        syn = int(row.get("syn_count", 0))
        ack = int(row.get("ack_count", 0))
        fin = int(row.get("fin_count", 0))
        rst = int(row.get("rst_count", 0))
        psh = int(row.get("psh_count", 0))
        sentence += (
            f". TCP flags observed: SYN={syn}, ACK={ack}, FIN={fin}, "
            f"RST={rst}, PSH={psh}"
        )

    return sentence + "."


CTU13_FEATURE_COLS = [
    "proto_int", "dir_fwd", "dir_bwd", "dir_bi", "dur", "sport", "dport", "stos", "dtos",
    "tot_pkts", "tot_bytes", "src_bytes", "bytes_per_pkt", "src_byte_ratio",
]


def format_network_row_ctu13(row) -> str:
    """TabLLM natural-language sentence for a CTU-13 bidirectional NetFlow row."""
    proto = str(row.get("proto", "other")).upper()
    sport = int(row.get("sport", 0) or 0)
    dport = int(row.get("dport", 0) or 0)
    dur = float(row.get("dur", 0.0) or 0.0)
    pkts = int(row.get("tot_pkts", 0) or 0)
    tot_b = int(row.get("tot_bytes", 0) or 0)
    src_b = int(row.get("src_bytes", 0) or 0)
    direction = str(row.get("dir", "->")).strip()
    state = str(row.get("state", "")).strip()
    sentence = (
        f"A {proto} flow from port {_port_label(sport)} to port {_port_label(dport)} "
        f"(direction {direction}), lasting {dur:.3f} s with {pkts} packets totaling "
        f"{tot_b} bytes, of which the source sent {src_b} bytes"
    )
    if state and state.lower() != "nan":
        sentence += f"; connection state {state}"
    return sentence + "."


def format_network_row_kitsune_pcap_flow_ft(row) -> str:
    """Compact key=value format matching finetune_modernbert_flows.py:format_flow().
    Must match training format exactly — using the TabLLM format causes AUCPR ~0.46."""
    _PROTO_INT_MAP = {6: "TCP", 17: "UDP", 1: "ICMP"}

    proto_int = int(row.get("protocol_int", 0))
    proto     = _PROTO_INT_MAP.get(proto_int, "OTHER")
    src_port  = int(row.get("src_port", 0))
    dst_port  = int(row.get("dst_port", 0))

    parts = [
        f"proto={proto}",
        f"src={_port_label(src_port)}",
        f"dst={_port_label(dst_port)}",
        f"pkts={int(row.get('pkt_count', 0))}",
        f"bytes={int(row.get('total_bytes', 0))}",
        f"duration_ms={float(row.get('duration_ms', 0.0)):.1f}",
        f"mean_iat_ms={float(row.get('mean_iat_ms', 0.0)):.2f}",
        f"std_iat_ms={float(row.get('std_iat_ms', 0.0)):.2f}",
        f"mean_pkt_len={float(row.get('mean_pkt_len', 0.0)):.1f}",
    ]
    if proto == "TCP":
        parts.append(
            f"flags=[SYN:{int(row.get('syn_count', 0))} "
            f"ACK:{int(row.get('ack_count', 0))} "
            f"FIN:{int(row.get('fin_count', 0))} "
            f"RST:{int(row.get('rst_count', 0))}]"
        )
    return "Network flow: " + ", ".join(parts)


# ── LLM classifiers ───────────────────────────────────────────────────────────

def _parse_llm_response(response_text: str) -> tuple[int, float]:
    json_match = re.search(r"\{[^}]+\}", response_text)
    if json_match:
        import json
        result = json.loads(json_match.group())
        return int(result.get("prediction", 0)), float(result.get("confidence", 0.5))
    if "phishing" in response_text.lower() or "attack" in response_text.lower():
        return 1, 0.6
    return 0, 0.5


def _classify_ollama(text: str, model: str, prompt_template: str) -> tuple[int, float]:
    import time
    import ollama
    truncated = text[:MAX_TEXT_LENGTH]
    last_err = None
    for attempt in range(3):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt_template.format(text=truncated)}],
                options={"temperature": 0.1},
            )
            return _parse_llm_response(response["message"]["content"].strip())
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    print(f"  [ollama error] {model}: {last_err}")
    return 0, 0.5


_claude_client = None


def _get_claude_client():
    global _claude_client
    if _claude_client is None:
        import anthropic
        # max_retries: the SDK already backs off on 429s honoring Retry-After;
        # raise the ceiling so transient rate-limit bursts don't exhaust
        # retries before they clear.
        _claude_client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=8,
        )
    return _claude_client


_modernbert_ft_model = None
_modernbert_ft_tokenizer = None
_modernbert_ft_device = None


def _load_modernbert_ft(model_dir: str):
    global _modernbert_ft_model, _modernbert_ft_tokenizer, _modernbert_ft_device
    if _modernbert_ft_model is not None:
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _modernbert_ft_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading fine-tuned ModernBERT from {model_dir} on {_modernbert_ft_device} ...")
    _modernbert_ft_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _modernbert_ft_model = (
        AutoModelForSequenceClassification.from_pretrained(model_dir)
        .to(_modernbert_ft_device)
        .eval()
    )


def _classify_modernbert_ft(text: str, model_dir: str) -> tuple[int, float]:
    import torch
    _load_modernbert_ft(model_dir)
    truncated = text[:MAX_TEXT_LENGTH]
    try:
        inputs = _modernbert_ft_tokenizer(
            truncated, return_tensors="pt", truncation=True, max_length=512,
        ).to(_modernbert_ft_device)
        with torch.no_grad():
            logits = _modernbert_ft_model(**inputs).logits
        prob = float(torch.softmax(logits, dim=-1)[0, 1])
        return int(prob >= 0.5), prob
    except Exception as e:
        print(f"  [modernbert-ft error]: {e}")
        return 0, 0.5


def _classify_claude(text: str, model: str, prompt_template: str) -> tuple[int, float] | None:
    """Returns None (instead of a fake prediction) on failure, so the caller
    knows not to cache it — caching a rate-limit fallback as ground truth
    would permanently poison that sample's cached prediction."""
    truncated = text[:MAX_TEXT_LENGTH]
    try:
        client = _get_claude_client()
        response = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt_template.format(text=truncated)}],
        )
        return _parse_llm_response(response.content[0].text.strip())
    except Exception as e:
        print(f"  [claude error] {model}: {e}")
        return None


_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _gemini_client


def _classify_gemini(text: str, model: str, prompt_template: str) -> tuple[int, float] | None:
    """Returns None (instead of a fake prediction) on failure, mirroring
    _classify_claude, so the caller knows not to cache it."""
    truncated = text[:MAX_TEXT_LENGTH]
    try:
        client = _get_gemini_client()
        response = client.models.generate_content(
            model=model,
            contents=prompt_template.format(text=truncated),
        )
        return _parse_llm_response((response.text or "").strip())
    except Exception as e:
        print(f"  [gemini error] {model}: {e}")
        return None


_decoder_ft_model = None
_decoder_ft_tokenizer = None
_decoder_ft_device = None


def _load_decoder_ft(model_dir: str):
    """Loads a LoRA-fine-tuned decoder sequence-classification checkpoint
    (see finetune_decoder_phishing.py / finetune_decoder_flows.py). Kept
    separate from _load_modernbert_ft so a run can evaluate both a
    fine-tuned encoder and a fine-tuned decoder checkpoint side by side."""
    global _decoder_ft_model, _decoder_ft_tokenizer, _decoder_ft_device
    if _decoder_ft_model is not None:
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _decoder_ft_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading fine-tuned decoder from {model_dir} on {_decoder_ft_device} ...")
    _decoder_ft_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _decoder_ft_model = (
        AutoModelForSequenceClassification.from_pretrained(model_dir)
        .to(_decoder_ft_device)
        .eval()
    )


def _classify_decoder_ft(text: str, model_dir: str) -> tuple[int, float]:
    import torch
    _load_decoder_ft(model_dir)
    truncated = text[:MAX_TEXT_LENGTH]
    try:
        inputs = _decoder_ft_tokenizer(
            truncated, return_tensors="pt", truncation=True, max_length=512,
        ).to(_decoder_ft_device)
        with torch.no_grad():
            logits = _decoder_ft_model(**inputs, return_dict=True).logits
        prob = float(torch.softmax(logits.float(), dim=-1)[0, 1])
        return int(prob >= 0.5), prob
    except Exception as e:
        print(f"  [decoder-ft error]: {e}")
        return 0, 0.5


def build_few_shot_prompt(base_template: str, examples_block: str) -> str:
    """Prepend few-shot examples into a {text}-style prompt template.

    Inserts examples_block immediately before the sample being classified,
    preserving the {text} placeholder for later substitution.
    """
    for marker in ("Flow:\n---\n{text}", "Message:\n---\n{text}"):
        if marker in base_template:
            return base_template.replace(marker, examples_block + marker, 1)
    return examples_block + base_template


def cache_key(text: str, model: str, prompt_template: str) -> str:
    """Cache key for one (sample, model, prompt) prediction: text:model:prompt_hash."""
    import hashlib
    prompt_hash = hashlib.sha256(prompt_template.encode()).hexdigest()[:8]
    return f"{hashlib.sha256(text.encode()).hexdigest()[:16]}:{model}:{prompt_hash}"


def classify_with_cache(
    text: str,
    model: str,
    cache: dict,
    prompt_template: str,
    backend: str = "ollama",
    cache_only: bool = False,
) -> tuple[int, float] | None:
    """Returns (pred, conf), or None if cache_only=True and the entry is not
    cached, OR if a live API call (claude/gemini) failed. Callers must treat
    None as "exclude this sample from metrics" in both cases — a failed call
    is not a (0, 0.5) prediction, it's a missing one, and silently faking a
    prediction would contaminate AUCPR with an unlabeled coin-flip."""
    key = cache_key(text, model, prompt_template)
    if key in cache:
        return tuple(cache[key])
    if cache_only:
        return None
    if backend == "ollama":
        pred, conf = _classify_ollama(text, model, prompt_template)
    elif backend == "claude":
        result = _classify_claude(text, model, prompt_template)
        if result is None:
            return None  # transient/quota failure — exclude, don't poison the cache
        pred, conf = result
    elif backend == "gemini":
        result = _classify_gemini(text, model, prompt_template)
        if result is None:
            return None  # transient/quota failure — exclude, don't poison the cache
        pred, conf = result
    elif backend == "modernbert-ft":
        pred, conf = _classify_modernbert_ft(text, model)
    elif backend == "decoder-ft":
        pred, conf = _classify_decoder_ft(text, model)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    cache[key] = [pred, conf]
    return pred, conf
