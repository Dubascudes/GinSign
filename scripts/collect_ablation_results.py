#!/usr/bin/env python
"""Collect ablation results into summary tables.

Scans outputs/ablations/*/eval_*.json and groups by ablation axis.
Produces markdown tables + outputs/ablations/summary.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ABL = ROOT / "outputs" / "ablations"
BASELINE = ROOT / "outputs"


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def load_baselines():
    """Per-domain baseline results from single-domain training."""
    baselines = {}
    for p in BASELINE.glob("*/eval.json"):
        dom = p.parent.name
        if dom != "ablations":
            baselines[dom] = load_json(p)
    return baselines


def collect():
    results = {}
    for exp_dir in sorted(ABL.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp = exp_dir.name
        evals = {}
        for ep in exp_dir.glob("eval_*.json"):
            dom = ep.stem.replace("eval_", "")
            evals[dom] = load_json(ep)
        if evals:
            results[exp] = evals
    return results


def print_axis(title, experiments, baselines):
    print(f"\n## {title}\n")
    all_doms = sorted({d for evals in experiments.values() for d in evals})
    header = "| experiment | " + " | ".join(all_doms) + " |"
    sep = "|---|" + "|".join(["---:"] * len(all_doms)) + "|"
    print(header)
    print(sep)

    # Baseline row
    cells = []
    for d in all_doms:
        b = baselines.get(d)
        cells.append(f"{b['joint_acc']:.3f}" if b else "—")
    print(f"| *per-domain baseline* | " + " | ".join(cells) + " |")

    for exp_name, evals in sorted(experiments.items()):
        cells = []
        for d in all_doms:
            m = evals.get(d)
            cells.append(f"{m['joint_acc']:.3f}" if m else "—")
        print(f"| {exp_name} | " + " | ".join(cells) + " |")


def main():
    baselines = load_baselines()
    all_results = collect()

    loo = {k: v for k, v in all_results.items() if k.startswith("loo_")}
    joint = {k: v for k, v in all_results.items() if k.startswith("joint_")}
    cross = {k: v for k, v in all_results.items() if k.startswith("cross_")}
    prefix = {k: v for k, v in all_results.items() if k.startswith("prefix_")}

    if loo:
        print_axis("Leave-One-Out", loo, baselines)
    if joint:
        print_axis("Joint Multi-Domain", joint, baselines)
    if cross:
        print_axis("Cross-Dataset Transfer", cross, baselines)
    if prefix:
        # Group prefix ablations by transform type
        by_transform: dict = {}
        for k, v in prefix.items():
            # prefix_shuffle__warehouse -> transform=shuffle, domain=warehouse
            match = re.match(r"prefix_(.+?)__(.+)", k)
            if match:
                t_name = match.group(1)
                if t_name not in by_transform:
                    by_transform[t_name] = {}
                by_transform[t_name].update(v)
        for t_name, evals in sorted(by_transform.items()):
            print_axis(f"Prefix: {t_name}", {t_name: evals}, baselines)

    summary_path = ABL / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "baselines": baselines,
        "ablations": all_results,
    }, indent=2) + "\n")
    print(f"\n[written to {summary_path}]")


if __name__ == "__main__":
    main()
