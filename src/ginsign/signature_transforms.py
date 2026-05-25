"""Inference-time transforms on Signatures for prefix ablations.

Each function returns a new Signature with the transform applied.
The original is not modified.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Collection, List, Optional

from .signature_io import PredicateDef, Signature


def shuffle_constants(sig: Signature, seed: int = 42) -> Signature:
    """Return a copy with constant order randomized per sort."""
    rng = random.Random(seed)
    new_constants = {}
    for sort, cs in sig.constants.items():
        shuffled = list(cs)
        rng.shuffle(shuffled)
        new_constants[sort] = shuffled
    return Signature(
        name=sig.name,
        sorts=list(sig.sorts),
        predicates=dict(sig.predicates),
        constants=new_constants,
        provenance=f"{sig.provenance} + shuffle(seed={seed})",
    )


def inject_distractors(
    sig: Signature,
    donor_sigs: List[Signature],
    n_per_sort: int = 10,
    seed: int = 42,
) -> Signature:
    """Add n random constants from donor signatures to each sort.

    Constants are drawn from any donor sort. If a constant already
    exists in the target sort, it's skipped (no duplicates).
    """
    rng = random.Random(seed)
    all_donor_consts: list = []
    for ds in donor_sigs:
        for cs in ds.constants.values():
            all_donor_consts.extend(cs)
    all_donor_consts = list(set(all_donor_consts))

    new_constants = {}
    for sort, cs in sig.constants.items():
        existing = set(cs)
        pool = [c for c in all_donor_consts if c not in existing]
        k = min(n_per_sort, len(pool))
        distractors = rng.sample(pool, k) if k > 0 else []
        new_constants[sort] = list(cs) + distractors
    return Signature(
        name=sig.name,
        sorts=list(sig.sorts),
        predicates=dict(sig.predicates),
        constants=new_constants,
        provenance=f"{sig.provenance} + distractors(n={n_per_sort}, seed={seed})",
    )


def partial_signature(
    sig: Signature,
    drop_frac: float = 0.3,
    seed: int = 42,
    protect_constants: Optional[Collection[str]] = None,
) -> Signature:
    """Remove drop_frac of constants from each sort.

    Constants in protect_constants are never removed (these should be
    the gold constants from the eval corpus to avoid dropping the
    answer). Compute the protect set at the corpus level before calling.
    """
    rng = random.Random(seed)
    protect = set(protect_constants or [])
    new_constants = {}
    for sort, cs in sig.constants.items():
        droppable = [c for c in cs if c not in protect]
        n_drop = int(len(droppable) * drop_frac)
        to_drop = set(rng.sample(droppable, min(n_drop, len(droppable))))
        new_constants[sort] = [c for c in cs if c not in to_drop]
    return Signature(
        name=sig.name,
        sorts=list(sig.sorts),
        predicates=dict(sig.predicates),
        constants=new_constants,
        provenance=f"{sig.provenance} + partial(drop={drop_frac}, seed={seed})",
    )


def collect_gold_constants(jsonl_path: str, cluster_map: Optional[dict] = None) -> set:
    """Sweep a corpus and return the set of all gold args_canon values."""
    import json
    golds: set = set()
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            for _, info in (d.get("prop_dict") or {}).items():
                if info is None:
                    continue
                action = info.get("action_canon")
                if cluster_map and action in cluster_map:
                    action = cluster_map[action]
                for c in info.get("args_canon", []):
                    golds.add(c)
    return golds
