"""
Phishing annotation policy, manifest loading, deduplication and grouping.
=========================================================================
Implements the taxonomy of experiments/13_phishing_operational.md.

The scientific taxonomy is defined FIRST; dataset labels are mapped onto it by
explicit, versioned rules (source_label + metadata -> subtype, confidence).
Corpus identity never becomes the target label by itself: every row keeps its
exact original label, its source, the rule that mapped it and how confident
that mapping is.

Subtypes
  phishing              strict phishing (credential theft / malicious destination
                        or payload / impersonated login)
  scam_fraud            deceptive financial or social-engineering mail without a
                        confirmed phishing mechanism (advance-fee, 419, ...)
  bulk_spam             unsolicited bulk / promotional mail without confirmed
                        phishing behaviour
  ham                   ordinary legitimate mail
  unknown_nonphishing   used as a negative by its source but not reliably
                        separable into ham / bulk_spam / scam_fraud, or possibly
                        contaminated with phishing

Mapping confidence
  confirmed   the source's own labels establish the subtype
  probable    the source is documented to be overwhelmingly that subtype
  proxy       the source label supports "not ham" (or "not phishing") but not the
              finer distinction; generic spam corpora may contain phishing
  unknown     no reliable metadata
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ANNOTATION_POLICY_VERSION = "1.0"
SUBTYPES = ["ham", "bulk_spam", "scam_fraud", "phishing", "unknown_nonphishing"]
NEGATIVE_SUBTYPES = ["ham", "bulk_spam", "scam_fraud", "unknown_nonphishing"]
CONFIDENCES = ["confirmed", "probable", "proxy", "unknown"]
REQUIRED_FIELDS = [
    "text", "label_binary", "subtype", "source_label", "source_corpus", "source_version",
    "source_id", "content_hash", "mapping_rule", "mapping_confidence",
    "annotation_policy_version", "campaign_or_template_group", "timestamp",
]

# ── Mapping rules ─────────────────────────────────────────────────────────────
# rule name -> {normalized source_label: (subtype, confidence)}; "*" = any label.
# These encode what each corpus's ORIGINAL labels justify. They are candidates
# that a manifest row opts into explicitly; the manifest may also carry an
# inline JSON mapping. Anything not covered raises, never silently defaults.
MAPPING_RULES: dict[str, dict[str, tuple[str, str]]] = {
    # Nazario phishing corpus: every message is a phishing sample.
    "nazario_phishing": {"*": ("phishing", "confirmed")},
    # Enron: legitimate corporate mail; contains some spam so 'probable', not 'confirmed'.
    "enron_ham": {"*": ("ham", "probable")},
    # SpamAssassin public corpus partitions.
    "spamassassin": {"easy_ham": ("ham", "confirmed"), "easy_ham_2": ("ham", "confirmed"),
                     "hard_ham": ("ham", "confirmed"), "ham": ("ham", "confirmed"),
                     "spam": ("bulk_spam", "proxy"), "spam_2": ("bulk_spam", "proxy")},
    # Ling-Spam: linguistics list mail vs spam.
    "ling_spam": {"ham": ("ham", "confirmed"), "legit": ("ham", "confirmed"), "spam": ("bulk_spam", "proxy")},
    # TREC / CEAS style corpora: ham/spam index files.
    "trec_ceas": {"ham": ("ham", "confirmed"), "spam": ("bulk_spam", "proxy")},
    # CLAIR / "Nigerian" fraud collection: advance-fee fraud letters.
    "nigerian_fraud": {"*": ("scam_fraud", "confirmed")},
    # zefang-liu/phishing-email-dataset: curated aggregate of public sources.
    "zefang_liu": {"phishing email": ("phishing", "probable"), "safe email": ("ham", "probable"),
                   "1": ("phishing", "probable"), "0": ("ham", "probable")},
    # Generic binary corpus with no further metadata.
    "binary_generic": {"1": ("phishing", "probable"), "phishing": ("phishing", "probable"),
                       "0": ("unknown_nonphishing", "unknown"), "legitimate": ("unknown_nonphishing", "unknown"),
                       "ham": ("unknown_nonphishing", "unknown"), "spam": ("unknown_nonphishing", "unknown")},
}


def _norm_label(x) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())


def resolve_mapping(rule_spec: str) -> tuple[str, dict[str, tuple[str, str]]]:
    """A registered rule name, or inline JSON {label: [subtype, confidence]}."""
    rule_spec = str(rule_spec).strip()
    if rule_spec in MAPPING_RULES:
        return rule_spec, MAPPING_RULES[rule_spec]
    if rule_spec.startswith("{"):
        raw = json.loads(rule_spec)
        m = {_norm_label(k): (v[0], v[1]) for k, v in raw.items()}
        for st, cf in m.values():
            if st not in SUBTYPES or cf not in CONFIDENCES:
                raise ValueError(f"inline mapping has invalid subtype/confidence: {st}/{cf}")
        return "inline:" + hashlib.sha256(rule_spec.encode()).hexdigest()[:8], m
    raise ValueError(f"unknown mapping rule {rule_spec!r}; registered: {sorted(MAPPING_RULES)}")


def apply_mapping(source_labels: pd.Series, rule_spec: str):
    rule_name, m = resolve_mapping(rule_spec)
    labs = source_labels.map(_norm_label)
    subtype, conf = [], []
    unmapped = set()
    for l in labs:
        hit = m.get(l) or m.get("*")
        if hit is None:
            unmapped.add(l); subtype.append(None); conf.append(None); continue
        subtype.append(hit[0]); conf.append(hit[1])
    if unmapped:
        raise ValueError(f"rule {rule_name}: source labels without a mapping: {sorted(unmapped)[:10]}")
    return rule_name, np.array(subtype, dtype=object), np.array(conf, dtype=object)


# ── Text normalization, hashing, grouping ─────────────────────────────────────
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_NUM_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")

TEXT_NORMALIZATION = "v1: NFKC-free lowercase; strip non-printable; collapse whitespace"
GROUPING_METHOD = ("template-prefix-v1: SHA-256 of the first 25 whitespace tokens of the "
                   "normalized text after replacing URLs, e-mail addresses and digit runs "
                   "with placeholders; rows sharing the key are one campaign/template group. "
                   "Exact duplicates share a key by construction.")


def normalize_text(t: str) -> str:
    t = "" if t is None or (isinstance(t, float) and np.isnan(t)) else str(t)
    t = "".join(ch for ch in t if ch.isprintable() or ch in "\n\t")
    return _WS_RE.sub(" ", t.lower()).strip()


def content_hash(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def template_key(normalized: str, n_tokens: int = 25) -> str:
    t = _URL_RE.sub(" <url> ", normalized)
    t = _EMAIL_RE.sub(" <email> ", t)
    t = _NUM_RE.sub("<num>", t)
    toks = t.split()[:n_tokens]
    return hashlib.sha256(" ".join(toks).encode("utf-8")).hexdigest()[:24]


def annotate(df: pd.DataFrame, source_corpus: str, source_version: str, rule_spec: str,
             text_field: str = "text", label_field: str = "label", id_field: str | None = None,
             timestamp_field: str | None = None, group_field: str | None = None) -> pd.DataFrame:
    """Map one corpus frame onto the taxonomy and attach every required field."""
    rule_name, subtype, conf = apply_mapping(df[label_field], rule_spec)
    norm = df[text_field].map(normalize_text)
    out = pd.DataFrame({
        "text": df[text_field].astype(str).values,
        "subtype": subtype,
        "source_label": df[label_field].astype(str).values,
        "source_corpus": source_corpus,
        "source_version": str(source_version),
        "source_id": (df[id_field].astype(str).values if id_field and id_field in df.columns
                      else [f"{source_corpus}:{i}" for i in range(len(df))]),
        "content_hash": norm.map(content_hash).values,
        "mapping_rule": rule_name,
        "mapping_confidence": conf,
        "annotation_policy_version": ANNOTATION_POLICY_VERSION,
        "campaign_or_template_group": (df[group_field].astype(str).values if group_field and group_field in df.columns
                                       else norm.map(template_key).values),
        "timestamp": (pd.to_datetime(df[timestamp_field], errors="coerce", utc=True).values
                      if timestamp_field and timestamp_field in df.columns else pd.NaT),
    })
    out["group_method"] = "provided" if (group_field and group_field in df.columns) else "template-prefix-v1"
    out["label_binary"] = (out["subtype"] == "phishing").astype(int)   # strict policy
    return out


def label_policy(df: pd.DataFrame, policy: str = "strict") -> np.ndarray:
    """strict: phishing vs everything else. broad_malicious: phishing+scam_fraud vs ham+bulk_spam
    (unknown proxies excluded from the broad task; caller must drop them)."""
    if policy == "strict":
        return (df["subtype"] == "phishing").astype(int).values
    if policy == "broad_malicious":
        return df["subtype"].isin(["phishing", "scam_fraud"]).astype(int).values
    raise ValueError(policy)


# ── Manifest ──────────────────────────────────────────────────────────────────
MANIFEST_COLUMNS = ["asset_id", "path", "format", "source_corpus", "source_version",
                    "text_field", "label_field", "mapping", "license_note"]
OPTIONAL_MANIFEST_COLUMNS = ["id_field", "timestamp_field", "group_field", "label_from_path", "encoding"]


def manifest_fingerprint(manifest_path: Path) -> str:
    return hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()[:16]


def _read_eml_dir(root: Path, label_from_path: bool = True, encoding: str = "latin-1") -> pd.DataFrame:
    """Every file under root is one message; label = first path component below root."""
    import email
    from email import policy as _pol
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        raw = p.read_bytes()
        try:
            msg = email.message_from_bytes(raw, policy=_pol.default)
            subj = msg.get("subject", "") or ""
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_content(); break
                if not body:
                    for part in msg.walk():
                        if part.get_content_type() == "text/html":
                            body = re.sub(r"<[^>]+>", " ", part.get_content()); break
            else:
                body = msg.get_content() if msg.get_content_type() != "text/html" else re.sub(r"<[^>]+>", " ", msg.get_content())
            date = msg.get("date", None)
        except Exception:
            subj, body, date = "", raw.decode(encoding, errors="replace"), None
        rel = p.relative_to(root).parts
        rows.append({"text": f"{subj}\n{body}", "label": rel[0] if (label_from_path and len(rel) > 1) else "",
                     "source_id": str(p.relative_to(root)), "timestamp": date})
    return pd.DataFrame(rows)


def _read_mbox(path: Path, label: str = "") -> pd.DataFrame:
    """Byte-level mbox splitter: the stdlib mailbox module requires ASCII From_
    separator lines and crashes on real-world mboxes with 8-bit sender names.
    Messages are delimited by lines starting with b"From " at position 0."""
    import email
    import re as _re
    raw = Path(path).read_bytes()
    if raw.startswith(b"From "):
        chunks = _re.split(rb"\n(?=From )", raw)
    else:
        chunks = [raw]
    rows = []
    for i, chunk in enumerate(chunks):
        nl = chunk.find(b"\n")
        payload = chunk[nl + 1:] if chunk.startswith(b"From ") and nl != -1 else chunk
        if not payload.strip():
            continue
        try:
            msg = email.message_from_bytes(payload)
            subj = str(msg.get("subject", "") or "")
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        pl = part.get_payload(decode=True)
                        body = (pl or b"").decode("latin-1", errors="replace"); break
            else:
                pl = msg.get_payload(decode=True)
                body = (pl if pl is not None else str(msg.get_payload()).encode("latin-1", "replace")).decode("latin-1", errors="replace") if not isinstance(pl, str) else pl
            ts = msg.get("date", None)
        except Exception:
            subj, body, ts = "", payload.decode("latin-1", errors="replace"), None
        rows.append({"text": f"{subj}\n{body}", "label": label, "source_id": f"{path.name}:{i}",
                     "timestamp": ts})
    return pd.DataFrame(rows)


def load_manifest(manifest_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load every asset listed in the manifest and return (annotated frame, report).
    Fails with the list of missing files; never downloads or substitutes."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}. See data/phishing-operational/README.md")
    man = pd.read_csv(manifest_path, dtype=str).fillna("")
    missing_cols = [c for c in MANIFEST_COLUMNS if c not in man.columns]
    if missing_cols:
        raise ValueError(f"manifest missing columns {missing_cols}; required: {MANIFEST_COLUMNS}")
    base = manifest_path.parent
    missing_files = [str(base / r["path"]) for _, r in man.iterrows() if not (base / r["path"]).exists()]
    if missing_files:
        raise FileNotFoundError("expected data files not found (no download is attempted):\n  " + "\n  ".join(missing_files))
    frames, report = [], {"assets": []}
    for _, r in man.iterrows():
        p = base / r["path"]; fmt = r["format"].lower()
        if fmt in ("csv", "parquet", "jsonl"):
            raw = (pd.read_csv(p, low_memory=False) if fmt == "csv" else
                   pd.read_parquet(p) if fmt == "parquet" else pd.read_json(p, lines=True))
            tf, lf = r["text_field"], r["label_field"]
        elif fmt == "eml_dir":
            raw = _read_eml_dir(p, label_from_path=(r.get("label_from_path", "true").lower() != "false"),
                                encoding=r.get("encoding") or "latin-1")
            tf, lf = "text", "label"
            if r["label_field"] and r["label_field"] != "label":      # constant label for the whole dir
                raw["label"] = r["label_field"]
        elif fmt == "mbox":
            raw = _read_mbox(p, label=r["label_field"]); tf, lf = "text", "label"
        else:
            raise ValueError(f"asset {r['asset_id']}: unknown format {fmt!r}")
        ann = annotate(raw, r["source_corpus"], r["source_version"], r["mapping"], text_field=tf, label_field=lf,
                       id_field=r.get("id_field") or ("source_id" if "source_id" in raw.columns else None),
                       timestamp_field=r.get("timestamp_field") or ("timestamp" if "timestamp" in raw.columns else None),
                       group_field=r.get("group_field") or None)
        ann["asset_id"] = r["asset_id"]
        frames.append(ann)
        report["assets"].append({"asset_id": r["asset_id"], "source_corpus": r["source_corpus"], "format": fmt,
                                 "rows": int(len(ann)), "mapping_rule": ann["mapping_rule"].iloc[0] if len(ann) else None,
                                 "subtype_counts": ann["subtype"].value_counts().to_dict(),
                                 "confidence_counts": ann["mapping_confidence"].value_counts().to_dict(),
                                 "license_note": r["license_note"]})
    df = pd.concat(frames, ignore_index=True)
    report["manifest_fingerprint"] = manifest_fingerprint(manifest_path)
    report["annotation_policy_version"] = ANNOTATION_POLICY_VERSION
    return df, report


# ── Deduplication and grouping ────────────────────────────────────────────────
def dedup_and_group(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Global exact dedup on content_hash (first occurrence kept), cross-source
    duplicate report, group-size diagnostics. Never writes message text."""
    n0 = len(df)
    per_hash_sources = df.groupby("content_hash")["source_corpus"].nunique()
    cross = per_hash_sources[per_hash_sources > 1]
    dup_mask = df.duplicated("content_hash", keep="first")
    dropped_by_source = df.loc[dup_mask, "source_corpus"].value_counts().to_dict()
    out = df.loc[~dup_mask].reset_index(drop=True)
    gsizes = out["campaign_or_template_group"].value_counts()
    diag = {
        "rows_before": int(n0), "rows_after": int(len(out)),
        "exact_duplicates_removed": int(dup_mask.sum()),
        "duplicates_removed_by_source": dropped_by_source,
        "cross_source_duplicate_hashes": int(len(cross)),
        "cross_source_duplicate_rows_before_dedup": int(df["content_hash"].isin(cross.index).sum()),
        "text_normalization": TEXT_NORMALIZATION,
        "group_method": GROUPING_METHOD,
        "n_groups": int(len(gsizes)),
        "group_size_max": int(gsizes.max()) if len(gsizes) else 0,
        "group_size_mean": float(gsizes.mean()) if len(gsizes) else 0.0,
        "groups_size_gt1": int((gsizes > 1).sum()),
        "rows_in_groups_gt1": int(gsizes[gsizes > 1].sum()),
    }
    return out, diag


def partition_overlap_diag(parts: dict[str, pd.DataFrame]) -> dict:
    """Prove no content_hash and no group crosses any pair of partitions."""
    names = list(parts)
    diag = {"hash_overlap": {}, "group_overlap": {}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            h = len(set(parts[a]["content_hash"]) & set(parts[b]["content_hash"]))
            g = len(set(parts[a]["campaign_or_template_group"]) & set(parts[b]["campaign_or_template_group"]))
            diag["hash_overlap"][f"{a}|{b}"] = int(h); diag["group_overlap"][f"{a}|{b}"] = int(g)
    diag["clean"] = all(v == 0 for v in diag["hash_overlap"].values()) and \
                    all(v == 0 for v in diag["group_overlap"].values())
    return diag


def ordered_hash_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for x in df["content_hash"].values:
        h.update(x.encode()); h.update(b"\n")
    return h.hexdigest()[:16]


# ── Deterministic synthetic corpus (smoke / tests; no downloads) ──────────────
def synthetic_corpus(seed: int = 42, n_per_source: int = 160) -> tuple[pd.DataFrame, dict]:
    """Six synthetic 'source corpora' with distinct fingerprints and templated
    content: two ham, one bulk spam, one scam/fraud, two phishing. Texts are
    generated from templates with random fillers so template groups have size
    > 1 and exact duplicates occur. Used by --synthetic-smoke and the tests."""
    rng = np.random.default_rng(seed)
    words = ["meeting", "report", "invoice", "quarter", "schedule", "team", "project", "update",
             "budget", "review", "office", "client", "deadline", "agenda", "notes", "travel"]
    def filler(k):
        return " ".join(rng.choice(words, size=k))
    sources = {
        "synth_ham_corp":  ("ham", "enron_ham", ["Re: {f} please see attached {f}", "Forwarded by desk: {f} tomorrow {f}",
                                                  "Lunch {f}? {f}"]),
        "synth_ham_list":  ("ham", "trec_ceas", ["[list] {f} discussion {f}", "Digest {f} {f} unsubscribe info"]),
        "synth_spam":      ("spam", "spamassassin", ["BUY NOW cheap {f} {f} click http://deal.example/{n}",
                                                     "Limited offer {f} {f} {n}% off"]),
        "synth_fraud":     ("fraud", "nigerian_fraud", ["Dear friend, I am the widow of {f} with {n} million USD {f}",
                                                         "Confidential business proposal {f} transfer {n} USD {f}"]),
        "synth_phish_a":   ("phish", "nazario_phishing", ["Your account {f} is suspended, verify at http://login-{n}.example {f}",
                                                           "Security alert: confirm password {f} http://secure-{n}.example"]),
        "synth_phish_b":   ("phish", "nazario_phishing", ["Bank notice {f}: update credentials http://bank-{n}.example {f}",
                                                           "Payment failed {f}, re-enter card at http://pay-{n}.example"]),
    }
    frames = []
    for name, (lab, rule, templates) in sources.items():
        texts = []
        for i in range(n_per_source):
            t = templates[i % len(templates)]
            texts.append(t.format(f=filler(3), n=int(rng.integers(10, 99))))
        texts += texts[:5]        # deliberate exact duplicates
        raw = pd.DataFrame({"text": texts, "label": lab})
        if rule == "spamassassin":
            raw["label"] = "spam"
        if rule == "trec_ceas":
            raw["label"] = "ham"
        frames.append(annotate(raw, name, "synthetic-v1", rule))
    df = pd.concat(frames, ignore_index=True)
    # one cross-source duplicate on purpose
    df.loc[len(df) - 1, "text"] = df.loc[0, "text"]
    df.loc[len(df) - 1, "content_hash"] = df.loc[0, "content_hash"]
    df.loc[len(df) - 1, "campaign_or_template_group"] = df.loc[0, "campaign_or_template_group"]
    report = {"assets": [{"asset_id": s, "source_corpus": s, "rows": n_per_source + 5} for s in sources],
              "manifest_fingerprint": f"synthetic-seed{seed}", "annotation_policy_version": ANNOTATION_POLICY_VERSION}
    return df, report
