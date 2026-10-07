#!/usr/bin/env python3
"""Pull the Jev and Laya rows out of the router result JSONs for the paper.

    uv run python -m ml_llm_ensembles.scripts.extract_systemone_results

Reads every *.result.json written by experiments 03 and 04, keeps the rows whose
model id is a System One model, and writes results/systemone_results.json plus a
LaTeX tabular on stdout. Cost, latency and memory are deliberately absent: they
come from 14_cost_benchmark.py, not from these runs.
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from ml_llm_ensembles.utils.prompts import SYSTEMONE_MODELS

METRICS = ["aucpr", "rocauc", "accuracy", "precision", "recall", "f1"]
EXP_DIR = _ROOT / "ml_llm_ensembles" / "experiments"
OUT = _ROOT / "results" / "systemone_results.json"


def main():
    out = {}
    for path in sorted(EXP_DIR.glob("0[34]_router_*.result.json")):
        res = json.loads(path.read_text())
        # rows are keyed "<model> | <config>"; keep only the System One models so a
        # renamed config (Router+BERT, Router-PFN) is picked up without editing this.
        rows = {k: v for k, v in res["rows"].items()
                if k.split(" | ")[0] in SYSTEMONE_MODELS}
        if not rows:
            continue
        out[path.name] = {
            "experiment": res["experiment"],
            "variant": res.get("variant"),
            "seed": res["seed"], "threshold": res["threshold"],
            "n_test": res["n_test"], "test_prior": res["test_prior"],
            "rows": rows,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}  ({sum(len(v['rows']) for v in out.values())} rows)\n")

    for name, blk in out.items():
        print(f"% {name}  n_test={blk['n_test']}  prior={blk['test_prior']:.3f}  "
              f"seed={blk['seed']}  tau={blk['threshold']}")
        print(r"\begin{tabular}{l" + "r" * len(METRICS) + "}")
        print(r"\toprule")
        print("Configuration & " + " & ".join(
            {"aucpr": "AUCPR", "rocauc": "ROC-AUC", "f1": "F1"}.get(m, m.capitalize())
            for m in METRICS) + r" \\")
        print(r"\midrule")
        for k, m in blk["rows"].items():
            label = k.replace("convaiinnovations/laya", "Laya").replace(
                "jev-1.13.0", "Jev").replace("_", r"\_")
            print(f"{label} & " + " & ".join(f"{m[x]:.4f}" for x in METRICS) + r" \\")
        print(r"\bottomrule")
        print(r"\end{tabular}" + "\n")


if __name__ == "__main__":
    main()
