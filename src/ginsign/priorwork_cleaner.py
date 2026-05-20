"""Clean parse artifacts in the priorwork JSONL files.

Source artifacts found in the raw data:
  * navi: all args end with a stray ')' (TL parser dropped the wrong closing
    paren; e.g. 'pear)' instead of 'pear')
  * navi: 32 occurrences of 'let_go' with arity 0 ('never let go' as a bare
    predicate). The other 391 use arity 1 ('let go pear'). Bare let_go has
    no analog under the {go_to, take, drop} clustered signature; drop it.
  * conformal: ~12 props with whitespace-glitched constants like ' photo'
    (the action token leaked into the arg slot). Drop those props.

We never modify the originals; cleaned outputs land in data/priorwork_cleaned/.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def _clean_arg(arg: str) -> str:
    return arg.rstrip(")")


def _raw_prop_is_bad(info: dict) -> Tuple[bool, str]:
    """Reject parse-glitched props before cleaning erases the evidence.

    Allowed artifact: a single trailing ')' (navi). Anything else --
    whitespace, internal parens, empty -- is a parse glitch.
    """
    action = info.get("action_canon")
    args = info.get("args_canon", [])
    if action == "let_go" and len(args) == 0:
        return True, "bare let_go (no clustering target)"
    for c in args:
        if not isinstance(c, str) or not c:
            return True, "empty or non-string arg"
        body = c[:-1] if c.endswith(")") else c
        if not body:
            return True, f"arg is bare ')' after cleaning: {c!r}"
        if any(ch.isspace() for ch in body):
            return True, f"whitespace in arg {c!r}"
        if "(" in body or ")" in body:
            return True, f"unparseable paren in arg {c!r}"
    return False, ""


def _clean_prop(info: dict) -> dict:
    out = dict(info)
    out["args_canon"] = [_clean_arg(a) for a in info.get("args_canon", [])]
    out["args_ref"] = [_clean_arg(a) for a in info.get("args_ref", [])]
    return out


def clean_file(in_path: Path, out_path: Path) -> Dict[str, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    drop_reasons: Counter = Counter()
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            stats["rows_in"] += 1
            d = json.loads(line)
            old_props = d.get("prop_dict") or {}
            new_props: Dict[str, dict] = {}
            for pname, info in old_props.items():
                stats["props_in"] += 1
                bad, reason = _raw_prop_is_bad(info)
                if bad:
                    stats["props_dropped"] += 1
                    drop_reasons[reason] += 1
                    continue
                cleaned = _clean_prop(info)
                if cleaned != info:
                    stats["props_modified"] += 1
                new_props[pname] = cleaned
            if not new_props:
                stats["rows_dropped"] += 1
                continue
            d["prop_dict"] = new_props
            fout.write(json.dumps(d) + "\n")
            stats["rows_out"] += 1
    return {"stats": dict(stats), "drop_reasons": dict(drop_reasons)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-root", type=Path, default=Path("data/priorwork"))
    parser.add_argument("--out-root", type=Path, default=Path("data/priorwork_cleaned"))
    args = parser.parse_args()

    splits = ["train", "test", "total"]
    datasets = ["cleanup_world", "conformal", "GLTL", "navi"]
    grand_total = Counter()
    for split in splits:
        for ds in datasets:
            src = args.in_root / split / f"{ds}.jsonl"
            if not src.exists():
                continue
            dst = args.out_root / split / f"{ds}.jsonl"
            result = clean_file(src, dst)
            grand_total.update(result["stats"])
            print(f"[{split}/{ds}]")
            for k, v in result["stats"].items():
                print(f"  {k}: {v}")
            if result["drop_reasons"]:
                print(f"  drop_reasons: {result['drop_reasons']}")
    print(f"\nTOTAL: {dict(grand_total)}")


if __name__ == "__main__":
    main()
