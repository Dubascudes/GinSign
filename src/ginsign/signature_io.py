"""Many-sorted system signature: dataclass + JSON I/O.

A Signature is the canonical handoff between data and the grounder. It
carries the sorts (types), predicates with per-slot argument sorts, and
the constants of each sort. Validation runs at load time to catch
signatures whose predicates reference unknown sorts or whose constants
land in undeclared sorts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PredicateDef:
    """A predicate with a per-slot sort signature.

    arg_sorts[r] is the list of sorts accepted at slot r. A single-sort
    slot is `[sort]`; a polymorphic slot like S&R's `photo` over both
    persons and threats is `[sort_a, sort_b, ...]`.
    """

    name: str
    arg_sorts: List[List[str]]

    @property
    def arity(self) -> int:
        return len(self.arg_sorts)


@dataclass
class Signature:
    name: str
    sorts: List[str]
    predicates: Dict[str, PredicateDef]   # name -> PredicateDef
    constants: Dict[str, List[str]]       # sort -> [constant, ...]
    provenance: str = "unspecified"
    notes: Optional[str] = None
    _const_to_sort: Dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.validate()
        self._const_to_sort = {}
        for sort, cs in self.constants.items():
            for c in cs:
                if c in self._const_to_sort:
                    raise ValueError(
                        f"Constant {c!r} appears in multiple sorts: "
                        f"{self._const_to_sort[c]!r} and {sort!r}"
                    )
                self._const_to_sort[c] = sort

    def validate(self) -> None:
        sort_set = set(self.sorts)
        for p in self.predicates.values():
            for r, slot_sorts in enumerate(p.arg_sorts):
                if not slot_sorts:
                    raise ValueError(
                        f"Predicate {p.name!r} slot {r} has no accepted sorts"
                    )
                for s in slot_sorts:
                    if s not in sort_set:
                        raise ValueError(
                            f"Predicate {p.name!r} slot {r} references unknown sort "
                            f"{s!r}; declared sorts: {sorted(sort_set)}"
                        )
        for sort in self.constants:
            if sort not in sort_set:
                raise ValueError(
                    f"Constants are declared for unknown sort {sort!r}; "
                    f"declared sorts: {sorted(sort_set)}"
                )

    def slot_sorts(self, predicate: str, slot: int) -> List[str]:
        return list(self.predicates[predicate].arg_sorts[slot])

    def slot_constants(self, predicate: str, slot: int) -> List[str]:
        """Union of constants over all sorts accepted at this slot."""
        out: List[str] = []
        seen: set = set()
        for s in self.slot_sorts(predicate, slot):
            for c in self.constants.get(s, []):
                if c not in seen:
                    seen.add(c)
                    out.append(c)
        return out

    def sort_of(self, constant: str) -> Optional[str]:
        return self._const_to_sort.get(constant)

    def constants_of(self, sort: str) -> List[str]:
        return list(self.constants.get(sort, []))

    def predicate_names(self) -> List[str]:
        return sorted(self.predicates.keys())

    def arity_of(self, predicate: str) -> int:
        return self.predicates[predicate].arity


def _normalize_arg_sorts_out(arg_sorts: List[List[str]]) -> List:
    """Wire form: single-sort slots collapse to a bare string for readability."""
    out: List = []
    for slot in arg_sorts:
        if len(slot) == 1:
            out.append(slot[0])
        else:
            out.append(list(slot))
    return out


def _normalize_arg_sorts_in(raw) -> List[List[str]]:
    """Accept both forms: 'item' or ['person', 'threat'] per slot."""
    out: List[List[str]] = []
    for slot in raw:
        if isinstance(slot, str):
            out.append([slot])
        else:
            out.append(list(slot))
    return out


def signature_to_dict(sig: Signature) -> dict:
    return {
        "name": sig.name,
        "sorts": list(sig.sorts),
        "predicates": {
            name: {"arg_sorts": _normalize_arg_sorts_out(p.arg_sorts)}
            for name, p in sorted(sig.predicates.items())
        },
        "constants": {
            sort: sorted(sig.constants.get(sort, []))
            for sort in sig.sorts
        },
        "provenance": sig.provenance,
        "notes": sig.notes,
    }


def signature_from_dict(d: dict) -> Signature:
    preds = {
        name: PredicateDef(
            name=name,
            arg_sorts=_normalize_arg_sorts_in(pdef["arg_sorts"]),
        )
        for name, pdef in d["predicates"].items()
    }
    return Signature(
        name=d["name"],
        sorts=list(d["sorts"]),
        predicates=preds,
        constants={s: list(d["constants"].get(s, [])) for s in d["sorts"]},
        provenance=d.get("provenance", "unspecified"),
        notes=d.get("notes"),
    )


def save_signature(sig: Signature, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(signature_to_dict(sig), indent=2, sort_keys=False) + "\n")


def load_signature(path: str | Path) -> Signature:
    return signature_from_dict(json.loads(Path(path).read_text()))
