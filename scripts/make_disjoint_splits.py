#!/usr/bin/env python
"""Build template-disjoint and surface-disjoint re-splits for rebuttal experiments.

Template-disjoint: partition lifted-LTL skeletons (masked_tl) ~70/30 so no
skeleton appears in both train and test.

Surface-disjoint: partition distinct AP surface strings (action_ref + args_ref)
~70/30 so no AP surface form in test was ever seen in train; rows whose APs
straddle the partition are dropped.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ap_surfaces(row):
    out = []
    for info in (row.get("prop_dict") or {}).values():
        if info is None:
            continue
        ref = (info.get("action_ref", "") + " " + " ".join(info.get("args_ref", []))).strip()
        out.append(ref)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--corpus-dir", default="data/VLTL-Bench")
    ap.add_argument("--out-dir", default="data/rebuttal_splits")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for split in ("train", "test"):
        with open(ROOT / args.corpus_dir / split / f"{args.domain}.jsonl") as fh:
            rows += [json.loads(l) for l in fh]

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- template-disjoint (by masked_tl skeleton) ----
    by_skel = defaultdict(list)
    for r in rows:
        by_skel[" ".join(r["masked_tl"])].append(r)
    skels = sorted(by_skel)
    rng.shuffle(skels)
    total = len(rows)
    test_target = args.test_frac * total
    test_skels, n_test = set(), 0
    for s in skels:
        if n_test < test_target:
            test_skels.add(s)
            n_test += len(by_skel[s])
    tr = [r for s in skels if s not in test_skels for r in by_skel[s]]
    te = [r for s in test_skels for r in by_skel[s]]
    for name, part in (("train", tr), ("test", te)):
        with open(out_dir / f"{args.domain}_template_{name}.jsonl", "w") as fh:
            for r in part:
                fh.write(json.dumps(r) + "\n")
    print(f"[template-disjoint] {args.domain}: {len(tr)} train rows "
          f"({len(skels)-len(test_skels)} skeletons) / {len(te)} test rows "
          f"({len(test_skels)} skeletons)")

    # ---- surface-disjoint (by AP surface string) ----
    surf_counts = Counter(s for r in rows for s in ap_surfaces(r))
    surfaces = sorted(surf_counts)
    rng.shuffle(surfaces)
    test_target = args.test_frac * sum(surf_counts.values())
    test_surf, n_test = set(), 0
    for s in surfaces:
        if n_test < test_target:
            test_surf.add(s)
            n_test += surf_counts[s]
    tr, te, dropped = [], [], 0
    for r in rows:
        aps = ap_surfaces(r)
        in_test = [s in test_surf for s in aps]
        if all(in_test):
            te.append(r)
        elif not any(in_test):
            tr.append(r)
        else:
            dropped += 1
    for name, part in (("train", tr), ("test", te)):
        with open(out_dir / f"{args.domain}_surface_{name}.jsonl", "w") as fh:
            for r in part:
                fh.write(json.dumps(r) + "\n")
    print(f"[surface-disjoint] {args.domain}: {len(tr)} train / {len(te)} test "
          f"rows, {dropped} straddling rows dropped; "
          f"{len(surfaces)-len(test_surf)}/{len(test_surf)} train/test surfaces")


if __name__ == "__main__":
    main()
