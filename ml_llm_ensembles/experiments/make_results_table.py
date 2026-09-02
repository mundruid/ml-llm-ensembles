#!/usr/bin/env python3
"""Build a human-readable table from a caller-supplied result directory.

Result JSON is intentionally excluded from the public artifact. Maintainers can
regenerate the checked-in summary from a private archive or newly reproduced
outputs with ``--results-dir``.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "RESULTS_TABLE.md"
EXCLUDED_CONFIG_TOKENS = ("llama3.1",)

# display order + human labels for the experiment files
ORDER = [
    ("01_ft_phishing.raw",        "01 FT phishing (raw)"),
    ("01_ft_phishing.stripped",   "01 FT phishing (strip-provenance)"),
    ("02_ft_flows",               "02 FT flows"),
    ("03_router_phishing.raw",    "03 router phishing (raw)"),
    ("03_router_phishing.stripped","03 router phishing (strip-provenance)"),
    ("04_router_flows",           "04 router flows"),
    ("05_meta_phishing.raw",      "05 meta phishing (raw)"),
    ("05_meta_phishing.stripped", "05 meta phishing (strip-provenance)"),
    ("06_meta_flows",             "06 meta flows"),
    ("07_meta_pcap",              "07 meta pcap (flow-aware)"),
    ("08_meta_afterimage.random", "08 afterimage (random split)"),
    ("08_meta_afterimage.temporal","08 afterimage (temporal split)"),
    ("09_meta_cicids",            "09 meta cicids"),
    ("11_ctu13_operational.scenario.ml", "11 CTU-13 (scenario hold-out; ML panel)"),
    ("11_ctu13_operational.scenario.llm", "11 CTU-13 (scenario hold-out; decoder subset)"),
    ("11_ctu13_operational.host.ml",     "11 CTU-13 (internal-host grouped; ML panel)"),
    ("11_ctu13_operational.temporal.ml", "11 CTU-13 (global temporal cutoff; ML panel)"),
    ("12_kitsune_cross_capture.kitsune-mirai-pcap_to_kitsune-syndos-pcap.ml", "12 train Mirai -> test SYN DoS; ML panel"),
    ("12_kitsune_cross_capture.kitsune-mirai-pcap_to_kitsune-syndos-pcap.llm", "12 train Mirai -> test SYN DoS; decoder subset"),
    ("12_kitsune_cross_capture.kitsune-syndos-pcap_to_kitsune-mirai-pcap.ml", "12 train SYN DoS -> test Mirai; ML panel"),
    ("12_kitsune_cross_capture.kitsune-syndos-pcap_to_kitsune-mirai-pcap.llm", "12 train SYN DoS -> test Mirai; decoder subset"),
    ("12_kitsune_cross_capture.kitsune-syndos-pcap_within_flow.ml", "12 SYN DoS within capture (flow-grouped)"),
    ("12_kitsune_cross_capture.kitsune-syndos-pcap_within_temporal.ml", "12 SYN DoS within capture (global temporal)"),
    # 13: standard held-out metrics only; projections / corpus-artifact diagnostics never enter rankings
    ("13_phishing_operational.controlled.raw.paper",      "13 phishing operational (controlled, raw; curated prior)"),
    ("13_phishing_operational.controlled.stripped.paper", "13 phishing operational (controlled, strip-provenance)"),
]


def fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


def rows_from(doc):
    """Yield (config, metrics-dict) for both FT ('test') and rows-style files."""
    if "rows" in doc and doc["rows"]:
        for cfg, m in doc["rows"].items():
            if cfg == "_note":
                continue
            if any(token in cfg.lower() for token in EXCLUDED_CONFIG_TOKENS):
                continue
            yield cfg, m
    elif "test" in doc:                       # fine-tune files
        m = dict(doc["test"]); m["val_selected"] = doc.get("val_aucpr_selected")
        yield "ModernBERT-FT", m


def trustworthy(m):
    """Eligible for the top-2 ranking: has an AUCPR, is scored on rows
    representative of its FULL held-out test partition (representative_test is
    about the scored subset, not about the prior: curated benchmark priors: the
    phishing sets, exp 13's controlled runs: remain rankable as conventional
    benchmark comparisons, while class-balanced scoring subsets are excluded),
    and isn't a low-coverage (non-representative cache subset) row."""
    if m.get("aucpr") is None or m.get("representative_test") is False:
        return False
    cov = m.get("coverage_test")
    if cov is None:
        cov = m.get("coverage")
    return cov is None or cov >= 0.9


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="directory containing result JSON files")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    src = args.results_dir
    body = []
    ranking = []          # (label, [(cfg, aucpr), ...top-2]) per task block
    total = 0
    for stem, label in ORDER:
        f = src / f"{stem}.result.json"
        if not f.exists():
            body += [f"## {label}", "", "_(not present yet)_", ""]
            continue
        d = json.loads(f.read_text())
        prior = d.get("test_prior"); ntest = d.get("n_test")
        head = f"## {label} :  prior {fmt(prior)}, n_test {ntest}"
        if d.get("note"):
            body += [head, "", f"> {d['note']}", ""]
            ranking.append((label, "degenerate", d["note"]))
            continue
        body += [head, "",
                  "| Config | AUCPR | ROC-AUC | Acc | route% | cov |",
                  "| --- | --- | --- | --- | --- | --- |"]
        items = list(rows_from(d))
        # sort by AUCPR desc within the block (None last)
        items.sort(key=lambda kv: (kv[1].get("aucpr") is not None, kv[1].get("aucpr") or 0), reverse=True)
        top2 = [(cfg, m["aucpr"]) for cfg, m in items if trustworthy(m)][:2]
        ranking.append((label, "ranked", top2))
        for cfg, m in items:
            rp = m.get("routed_pct")
            cov = m.get("coverage_test")
            if cov is None:
                cov = m.get("coverage")          # router rows use "coverage"
            covs = "" if cov is None else f"{cov:.2f}"
            # Flag low-coverage rows: these were scored on a non-representative
            # subset (model not warmed on this dataset) and are NOT trustworthy.
            flag = " ⚠low-cov" if (cov is not None and cov < 0.9) else ""
            if m.get("representative_test") is False:
                flag += f" ⚠balanced-subset(prior {m.get('evaluation_prior', 0):.2f})"
            auc = m.get("aucpr") if m.get("representative_test") is not False else m.get("aucpr_subset")
            body.append(
                f"| {cfg.replace('|', '\\|')}{flag} | {fmt(auc)} | {fmt(m.get('rocauc'))} | "
                f"{fmt(m.get('accuracy'))} | {'' if rp is None else f'{rp:.1f}'} | {covs} |")
            total += 1
        body.append("")

    # ── Experiment counts (dynamic) ───────────────────────────────────────────
    present = [s for s, _ in ORDER if (src / f"{s}.result.json").exists()]
    n_runs = len(present)                               # executed full-run results
    n_families = len({s[:2] for s in present})          # distinct 01..09 prefixes

    # ── Rank summary (#1 / #2 per task), prepended to the top of the page ──────
    rank = [
        "# Experiment results",
        "",
        "> **Takeaway:** Strong tabular models (TabPFN-3) and encoder representations "
        "(ModernBERT-FT) account for the best results; adding a decoder LLM does not. "
        "No router or meta-learner ensemble beats the best standalone model by a "
        "meaningful margin, decoder rows on the network datasets sit at or near their "
        "evaluation priors, and the operational experiments (11 to 13) show that grouped "
        "splits and subtype-level false-positive rates change the practical picture even "
        "where AUCPR looks saturated.",
        "",
        "## Scope: how to count the experiments",
        "",
        "The released suite contains numbered experiments 01--09 and 11--13; number 10 is unused.",
        "",
        f"- **{n_families} numbered experiment families**.",
        f"- **{n_runs} result panels**, including text, split, and model-roster variants.",
        f"- **{total} total configurations/rows** in the per-config tables below "
        "(every base model, ensemble, gate, and stage-2 combination).",
        "",
        f"So the full executed panel is **{n_runs} runs covering {total} "
        "configurations.**",
        "",
        "## Top performers per task (by AUCPR)",
        "",
        "Best two trustworthy configs per task. Low-coverage rows (LLM not warmed on "
        "that dataset; scored on a non-representative subset) are excluded. Full "
        "per-config tables below.",
        "",
        "| Task | #1 | #2 |",
        "| --- | --- | --- |",
    ]
    for entry in ranking:
        label = entry[0]
        if entry[1] == "degenerate":
            rank.append(f"| {label} | _n/a (degenerate split)_ | _n/a_ |")
            continue
        top2 = entry[2]
        def cell(i):
            if i >= len(top2):
                return "--"
            cfg, a = top2[i]
            return f"{cfg.replace('|', '\\|')} ({a:.4f})"   # escape pipes (router cfgs)
        rank.append(f"| {label} | {cell(0)} | {cell(1)} |")
    rank += [""]

    preamble = [
        "Auto-generated by `make_results_table.py` from a non-public result archive.",
        "AUCPR primary; read against each block's prior. `route%` = router escalation",
        "rate; `cov` = cache-only coverage (LLM rows). Re-run the generator to refresh.",
        "",
        "---",
        "",
    ]
    tail = ["---", f"**{total} configurations** across "
            f"{sum(1 for s,_ in ORDER if (src/f'{s}.result.json').exists())} experiment runs.", ""]
    args.output.write_text("\n".join(rank + preamble + body + tail))
    print(f"wrote {args.output}  ({total} configs)")


if __name__ == "__main__":
    main()
