"""Merge multiple Signatures into a single union signature."""

from __future__ import annotations

from typing import List

from .signature_io import PredicateDef, Signature


def merge_signatures(
    sigs: List[Signature],
    name: str = "merged",
    disambiguate: bool = True,
) -> Signature:
    """Union of sorts, predicates, and constants across signatures.

    When two signatures define the same predicate name with different
    arities: if disambiguate=True, rename both to '{domain}/{pred}';
    if False, raise ValueError.
    """
    if not sigs:
        raise ValueError("Cannot merge an empty list of signatures")

    all_sorts: set = set()
    predicates: dict[str, PredicateDef] = {}
    # Track which domain each predicate came from (for disambiguation)
    pred_origin: dict[str, str] = {}
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
                    if not disambiguate:
                        raise ValueError(
                            f"Predicate {pname!r} has arity {existing.arity} in "
                            f"{pred_origin[pname]!r} but {pdef.arity} in {sig.name!r}"
                        )
                    # Disambiguate: rename both to domain/pred
                    old_origin = pred_origin.pop(pname)
                    old_def = predicates.pop(pname)
                    qual_old = f"{old_origin}/{pname}"
                    qual_new = f"{sig.name}/{pname}"
                    predicates[qual_old] = PredicateDef(name=qual_old, arg_sorts=old_def.arg_sorts)
                    pred_origin[qual_old] = old_origin
                    predicates[qual_new] = PredicateDef(name=qual_new, arg_sorts=pdef.arg_sorts)
                    pred_origin[qual_new] = sig.name
                elif pname in pred_origin:
                    # Same arity — merge arg_sorts
                    merged_arg_sorts = []
                    for r in range(pdef.arity):
                        union = list(existing.arg_sorts[r])
                        for s in pdef.arg_sorts[r]:
                            if s not in union:
                                union.append(s)
                        merged_arg_sorts.append(sorted(union))
                    predicates[pname] = PredicateDef(name=pname, arg_sorts=merged_arg_sorts)
                else:
                    # Was already disambiguated by a prior conflict
                    qual = f"{sig.name}/{pname}"
                    predicates[qual] = PredicateDef(name=qual, arg_sorts=pdef.arg_sorts)
                    pred_origin[qual] = sig.name
            else:
                predicates[pname] = pdef
                pred_origin[pname] = sig.name

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
