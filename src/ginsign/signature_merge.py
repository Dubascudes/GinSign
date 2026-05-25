"""Merge multiple Signatures into a single union signature."""

from __future__ import annotations

from typing import List

from .signature_io import PredicateDef, Signature


def merge_signatures(sigs: List[Signature], name: str = "merged") -> Signature:
    """Union of sorts, predicates, and constants across signatures.

    Raises ValueError if two signatures define the same predicate name
    with different arities or incompatible arg_sorts.
    """
    if not sigs:
        raise ValueError("Cannot merge an empty list of signatures")

    all_sorts: set = set()
    predicates: dict[str, PredicateDef] = {}
    constants: dict[str, list] = {}

    for sig in sigs:
        all_sorts.update(sig.sorts)
        for sort in sig.sorts:
            if sort not in constants:
                constants[sort] = []
            seen = set(constants[sort])
            for c in sig.constants.get(sort, []):
                if c not in seen:
                    constants[sort].append(c)
                    seen.add(c)

        for pname, pdef in sig.predicates.items():
            if pname in predicates:
                existing = predicates[pname]
                if existing.arity != pdef.arity:
                    raise ValueError(
                        f"Predicate {pname!r} has arity {existing.arity} in one "
                        f"signature but {pdef.arity} in another"
                    )
                merged_arg_sorts = []
                for r in range(pdef.arity):
                    union = list(existing.arg_sorts[r])
                    for s in pdef.arg_sorts[r]:
                        if s not in union:
                            union.append(s)
                    merged_arg_sorts.append(sorted(union))
                predicates[pname] = PredicateDef(name=pname, arg_sorts=merged_arg_sorts)
            else:
                predicates[pname] = pdef

    sorts = sorted(all_sorts)
    for sort in constants:
        constants[sort].sort()

    provenance_parts = [s.name for s in sigs]
    return Signature(
        name=name,
        sorts=sorts,
        predicates=predicates,
        constants=constants,
        provenance=f"merged from {', '.join(provenance_parts)}",
    )
