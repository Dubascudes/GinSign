#!/usr/bin/env python
"""Collect per-domain eval.json + best metrics into one summary table.

Looks for outputs/<domain>/eval.json from each completed training run,
plus the best-checkpoint metrics stored inside best.pt. Writes
outputs/summary.json and prints a markdown table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

DOMAINS = [
    "warehouse",
    "traffic_light",
    "search_and_rescue",
    "cleanup_world",
    "GLTL",
    "conformal",
    "navi",
]


def _best_from_ckpt(path: Path) -> dict | None:
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("metrics")


def main() -> None:
    out_root = Path("outputs")
    rows = []
    for d in DOMAINS:
        run_dir = out_root / d
        eval_path = run_dir / "eval.json"
        best_metrics = _best_from_ckpt(run_dir / "best.pt")
        final_metrics = None
        if eval_path.exists():
            final_metrics = json.loads(eval_path.read_text())
        rows.append({"domain": d, "best": best_metrics, "final": final_metrics})

    summary_path = out_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, indent=2) + "\n")

    # Markdown table over BEST metrics (the saved checkpoint we'd actually use).
    print("| domain | n | pred_acc | pred_f1 | arg_full | joint |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        m = row["best"] or row["final"]
        if m is None:
            print(f"| {row['domain']} | — | — | — | — | — |")
            continue
        print(
            f"| {row['domain']} | {m.get('n','—')} | "
            f"{m.get('pred_acc',0):.3f} | "
            f"{m.get('pred_f1_macro',0):.3f} | "
            f"{m.get('arg_acc_full_tuple',0):.3f} | "
            f"{m.get('joint_acc',0):.3f} |"
        )
    print(f"\n[summary written to {summary_path}]")


if __name__ == "__main__":
    main()
