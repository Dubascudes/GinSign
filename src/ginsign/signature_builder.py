"""Extract Signature files from grounding corpora.

VLTL-Bench: sweep the corpus, cross-reference each (predicate, args)
against the canonical TYPE_CONSTANTS dict so we recover per-slot arg
sorts. Raise if any predicate-arity is inconsistent across the corpus
or any constant lands outside the canonical type assignment.

Priorwork: clean-build path that consumes a cleaned corpus, optionally
a predicate-cluster map (navi), and infers per-slot types by clustering
constants on predicate co-occurrence (see infer_priorwork_signature).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .signature_io import PredicateDef, Signature, save_signature
from ._vltl_canonical_types import TYPE_CONSTANTS


# ---------------------------------------------------------------------------
# Predicate clustering (navi)
# ---------------------------------------------------------------------------


def load_cluster_map(path: Path) -> Tuple[Dict[str, str], List[str], set]:
    """Return (raw -> canonical, canonical_order, drop_set).

    Drop set lists raw predicates that should be excluded from the
    signature entirely (e.g., bare-arity-only state verbs that have no
    sensible canonical analog).
    """
    raw_data = json.loads(path.read_text())
    c2r = raw_data["canonical_to_raw"]
    raw_to_canon: Dict[str, str] = {}
    for canon, raws in c2r.items():
        for r in raws:
            if r in raw_to_canon:
                raise ValueError(
                    f"Raw predicate {r!r} mapped to two canonicals: "
                    f"{raw_to_canon[r]!r} and {canon!r}"
                )
            raw_to_canon[r] = canon
    drop_set = set(raw_data.get("_drop", []))
    overlap = drop_set & set(raw_to_canon)
    if overlap:
        raise ValueError(
            f"cluster map: predicates appear in both canonical_to_raw and _drop: {overlap}"
        )
    return raw_to_canon, list(c2r.keys()), drop_set


# ---------------------------------------------------------------------------
# Shared corpus sweep
# ---------------------------------------------------------------------------


def iter_props(jsonl_path: Path) -> Iterable[Tuple[str, List[str]]]:
    """Yield (action_canon, args_canon) per prop across the corpus."""
    with jsonl_path.open() as fh:
        for line in fh:
            d = json.loads(line)
            for _, info in (d.get("prop_dict") or {}).items():
                action = info.get("action_canon")
                args = info.get("args_canon") or []
                if action is None:
                    continue
                yield action, list(args)


def sweep_corpus(jsonl_path: Path) -> Tuple[
    Dict[str, Counter],                                # pred -> arity Counter
    Dict[str, Dict[int, set]],                         # pred -> slot -> set(consts)
]:
    pred_arity: Dict[str, Counter] = defaultdict(Counter)
    pred_slot_consts: Dict[str, Dict[int, set]] = defaultdict(
        lambda: defaultdict(set)
    )
    for action, args in iter_props(jsonl_path):
        pred_arity[action][len(args)] += 1
        for r, c in enumerate(args):
            pred_slot_consts[action][r].add(c)
    return pred_arity, pred_slot_consts


# ---------------------------------------------------------------------------
# VLTL-Bench: canonical-type-anchored extraction
# ---------------------------------------------------------------------------


def _const_to_sort_for_domain(domain: str) -> Dict[str, str]:
    """Invert TYPE_CONSTANTS[domain] to constant -> sort."""
    if domain not in TYPE_CONSTANTS:
        raise KeyError(f"Unknown VLTL-Bench domain {domain!r}")
    rev: Dict[str, str] = {}
    for sort, cs in TYPE_CONSTANTS[domain].items():
        for c in cs:
            if c in rev:
                raise ValueError(
                    f"Canonical type table has {c!r} in two sorts for {domain}"
                )
            rev[c] = sort
    return rev


def build_vltl_signature(
    domain: str,
    corpus: Path,
    require_full_coverage: bool = True,
) -> Signature:
    """Build a Signature for a VLTL-Bench domain.

    Cross-checks corpus contents against TYPE_CONSTANTS[domain]:
      * every constant must appear in the canonical type table
      * each predicate's slot must map to a single sort (no mixing)
      * predicates must have a unique arity across the corpus
    """
    const_to_sort = _const_to_sort_for_domain(domain)
    pred_arity, pred_slot_consts = sweep_corpus(corpus)

    inconsistent = {p: dict(c) for p, c in pred_arity.items() if len(c) > 1}
    if inconsistent:
        raise ValueError(
            f"{domain}: predicates with inconsistent arity in corpus: {inconsistent}"
        )

    predicates: Dict[str, PredicateDef] = {}
    sorts_used: set = set()
    for pred in sorted(pred_arity):
        arity = next(iter(pred_arity[pred]))
        arg_sorts: List[List[str]] = []
        for r in range(arity):
            slot_consts = pred_slot_consts[pred][r]
            sorts_at_slot = {const_to_sort.get(c) for c in slot_consts}
            if None in sorts_at_slot:
                missing = sorted(c for c in slot_consts if c not in const_to_sort)
                msg = (
                    f"{domain}: predicate {pred!r} slot {r} contains constants "
                    f"missing from TYPE_CONSTANTS: {missing[:10]}"
                    + ("..." if len(missing) > 10 else "")
                )
                if require_full_coverage:
                    raise ValueError(msg)
                sorts_at_slot.discard(None)
            if not sorts_at_slot:
                raise ValueError(
                    f"{domain}: predicate {pred!r} slot {r} has no resolvable sorts"
                )
            slot_sort_list = sorted(sorts_at_slot)
            arg_sorts.append(slot_sort_list)
            sorts_used.update(slot_sort_list)
        predicates[pred] = PredicateDef(name=pred, arg_sorts=arg_sorts)

    sorts = sorted(set(TYPE_CONSTANTS[domain].keys()))
    constants = {sort: sorted(TYPE_CONSTANTS[domain].get(sort, [])) for sort in sorts}
    unused = sorts_used - set(sorts)
    if unused:
        raise ValueError(f"{domain}: internal error, unaccounted sorts {unused}")

    return Signature(
        name=domain,
        sorts=sorts,
        predicates=predicates,
        constants=constants,
        provenance=f"vltl-bench (auto-extracted from {corpus.name})",
    )


def verify_corpus_coverage(
    sig: Signature,
    corpus: Path,
    cluster_map: Optional[Dict[str, str]] = None,
    drop_set: Optional[set] = None,
) -> Dict[str, int]:
    """Verify every (pred, args) tuple in corpus is signature-covered.

    Pass cluster_map for priorwork signatures whose corpus uses raw
    predicate names that map to canonical names in the signature.
    drop_set excludes raw predicates that are intentionally absent
    from the signature.
    """
    drop_set = drop_set or set()
    n_props = 0
    n_uncovered = 0
    n_dropped = 0
    examples_uncovered: List[str] = []
    for raw_action, args in iter_props(corpus):
        if raw_action in drop_set:
            n_dropped += 1
            continue
        n_props += 1
        action = cluster_map.get(raw_action, raw_action) if cluster_map else raw_action
        if action not in sig.predicates:
            n_uncovered += 1
            if len(examples_uncovered) < 5:
                examples_uncovered.append(f"unknown predicate {action!r}")
            continue
        pdef = sig.predicates[action]
        if len(args) != pdef.arity:
            n_uncovered += 1
            if len(examples_uncovered) < 5:
                examples_uncovered.append(
                    f"arity mismatch for {action!r}: got {len(args)}, sig has {pdef.arity}"
                )
            continue
        for r, c in enumerate(args):
            allowed = sig.slot_constants(action, r)
            if c not in allowed:
                n_uncovered += 1
                if len(examples_uncovered) < 5:
                    sorts_at_slot = pdef.arg_sorts[r]
                    examples_uncovered.append(
                        f"{action}({', '.join(args)}): arg[{r}]={c!r} not in slot sorts {sorts_at_slot}"
                    )
                break
    if n_uncovered:
        raise ValueError(
            f"{sig.name}: {n_uncovered}/{n_props} props uncovered. Examples:\n"
            + "\n".join(f"  - {ex}" for ex in examples_uncovered)
        )
    return {"n_props": n_props, "n_uncovered": n_uncovered, "n_dropped": n_dropped}


# ---------------------------------------------------------------------------
# Priorwork: co-occurrence-based type inference
# ---------------------------------------------------------------------------


def _apply_clustering(action: str, cluster_map: Optional[Dict[str, str]]) -> str:
    if cluster_map is None:
        return action
    if action not in cluster_map:
        raise ValueError(
            f"Predicate {action!r} not present in cluster map; "
            f"either extend the map or skip clustering."
        )
    return cluster_map[action]


def _partition_constants_by_cooccurrence(
    pred_slot_consts: Dict[str, Dict[int, set]],
) -> Tuple[Dict[str, str], Dict[Tuple, str]]:
    """Group constants by their (predicate, slot) co-occurrence pattern.

    Returns (const -> type_name, partition_signature -> type_name). Each
    distinct membership pattern becomes one type, named after the
    canonical predicates that use it (e.g., 'type_go_to_take' for
    constants seen at both predicates' slots).
    """
    const_membership: Dict[str, set] = defaultdict(set)
    for pred, slot_to_consts in pred_slot_consts.items():
        for slot, consts in slot_to_consts.items():
            for c in consts:
                const_membership[c].add((pred, slot))

    partition_to_type: Dict[Tuple, str] = {}
    const_to_type: Dict[str, str] = {}
    for c, mem in const_membership.items():
        key = tuple(sorted(mem))
        if key not in partition_to_type:
            preds_in_partition = sorted({p for p, _ in key})
            type_name = "type_" + "_".join(preds_in_partition)
            # Disambiguate slot-level distinctions for the same pred set
            slots_in_partition = sorted({s for _, s in key})
            if len(slots_in_partition) > 1 or any(s != 0 for s in slots_in_partition):
                type_name += "_slots" + "_".join(str(s) for s in slots_in_partition)
            # Disambiguate if two distinct keys produced the same name
            base = type_name
            i = 2
            while type_name in partition_to_type.values():
                type_name = f"{base}_{i}"
                i += 1
            partition_to_type[key] = type_name
        const_to_type[c] = partition_to_type[key]
    return const_to_type, partition_to_type


def build_priorwork_signature(
    name: str,
    corpus: Path,
    cluster_map_path: Optional[Path] = None,
) -> Tuple[Signature, dict]:
    """Build a signature for a cleaned priorwork corpus + sidecar evidence."""
    raw_to_canon: Optional[Dict[str, str]] = None
    drop_set: set = set()
    if cluster_map_path is not None:
        raw_to_canon, _, drop_set = load_cluster_map(cluster_map_path)

    pred_arity: Dict[str, Counter] = defaultdict(Counter)
    pred_slot_consts: Dict[str, Dict[int, set]] = defaultdict(
        lambda: defaultdict(set)
    )
    raw_pred_counts: Counter = Counter()
    n_dropped = 0
    for action, args in iter_props(corpus):
        if action in drop_set:
            n_dropped += 1
            continue
        canon_action = _apply_clustering(action, raw_to_canon)
        raw_pred_counts[(action, canon_action)] += 1
        pred_arity[canon_action][len(args)] += 1
        for r, c in enumerate(args):
            pred_slot_consts[canon_action][r].add(c)

    # Sanity: each canonical must have a single arity. If clustering merges
    # predicates of different arities (e.g., drop covers let_go/0 and drop/1)
    # we drop the minority arity occurrences -- they are usually parse-bugs
    # we already filtered, but guard anyway.
    for p, arities in list(pred_arity.items()):
        if len(arities) > 1:
            raise ValueError(
                f"{name}: canonical predicate {p!r} has inconsistent arities {dict(arities)}. "
                f"Check cluster map or clean the source."
            )

    const_to_type, partition_to_type = _partition_constants_by_cooccurrence(
        pred_slot_consts
    )

    sorts = sorted(set(const_to_type.values()))
    constants: Dict[str, List[str]] = defaultdict(list)
    for c, t in const_to_type.items():
        constants[t].append(c)
    for t in constants:
        constants[t].sort()

    predicates: Dict[str, PredicateDef] = {}
    for pred in sorted(pred_arity):
        arity = next(iter(pred_arity[pred]))
        arg_sorts: List[List[str]] = []
        for r in range(arity):
            slot_consts = pred_slot_consts[pred][r]
            sorts_at_slot = sorted({const_to_type[c] for c in slot_consts})
            if not sorts_at_slot:
                raise ValueError(
                    f"{name}: predicate {pred!r} slot {r} has no resolvable type"
                )
            arg_sorts.append(sorts_at_slot)
        predicates[pred] = PredicateDef(name=pred, arg_sorts=arg_sorts)

    sig = Signature(
        name=name,
        sorts=sorts,
        predicates=predicates,
        constants=dict(constants),
        provenance=f"priorwork-synthesized from {corpus.name}"
                   + (f" + {cluster_map_path.name}" if cluster_map_path else ""),
        notes="Types inferred by predicate-slot co-occurrence partitioning.",
    )

    evidence = {
        "raw_pred_counts": {
            f"{raw}->{canon}": n for (raw, canon), n in raw_pred_counts.most_common()
        },
        "partition_membership": {
            t: [{"predicate": p, "slot": s} for p, s in key]
            for key, t in partition_to_type.items()
        },
        "type_sizes": {t: len(constants[t]) for t in sorts},
        "props_dropped_by_cluster_map": n_dropped,
    }
    return sig, evidence


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _vltl_main(domain: str, corpus: Path, out: Path) -> None:
    sig = build_vltl_signature(domain, corpus)
    stats = verify_corpus_coverage(sig, corpus)
    save_signature(sig, out)
    print(f"[{domain}] -> {out}")
    print(f"  sorts: {sig.sorts}")
    print(f"  predicates: {sig.predicate_names()}")
    print(f"  total constants: {sum(len(cs) for cs in sig.constants.values())}")
    print(f"  coverage: {stats['n_props'] - stats['n_uncovered']}/{stats['n_props']}")


def _priorwork_main(name: str, corpus: Path, out: Path, cluster_map: Optional[Path]) -> None:
    sig, evidence = build_priorwork_signature(name, corpus, cluster_map)
    if cluster_map:
        raw_to_canon, _, drop_set = load_cluster_map(cluster_map)
    else:
        raw_to_canon, drop_set = None, set()
    stats = verify_corpus_coverage(
        sig, corpus, cluster_map=raw_to_canon, drop_set=drop_set
    )
    save_signature(sig, out)
    evidence_path = out.with_suffix(".evidence.json")
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=False) + "\n")
    print(f"[{name}] -> {out}  (evidence -> {evidence_path.name})")
    print(f"  sorts ({len(sig.sorts)}): {sig.sorts}")
    print(f"  predicates: {sig.predicate_names()}")
    print(f"  total constants: {sum(len(cs) for cs in sig.constants.values())}")
    print(f"  type sizes: {evidence['type_sizes']}")
    print(f"  coverage: {stats['n_props'] - stats['n_uncovered']}/{stats['n_props']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_vltl = sub.add_parser("vltl", help="Build a VLTL-Bench domain signature")
    p_vltl.add_argument("--domain", required=True,
                        choices=sorted(TYPE_CONSTANTS.keys()))
    p_vltl.add_argument("--corpus", type=Path, required=True)
    p_vltl.add_argument("--out", type=Path, required=True)

    p_pw = sub.add_parser("priorwork", help="Build a priorwork signature")
    p_pw.add_argument("--name", required=True)
    p_pw.add_argument("--corpus", type=Path, required=True)
    p_pw.add_argument("--out", type=Path, required=True)
    p_pw.add_argument("--cluster-map", type=Path, default=None)

    args = parser.parse_args()
    if args.cmd == "vltl":
        _vltl_main(args.domain, args.corpus, args.out)
    elif args.cmd == "priorwork":
        _priorwork_main(args.name, args.corpus, args.out, args.cluster_map)


if __name__ == "__main__":
    main()
