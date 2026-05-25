#!/usr/bin/env python
"""Run GPT-5 lifted translation on priorwork test sets.

For each priorwork domain, reads the first N test entries from
eval_data/translation_eval/nl2tl/raw_nl/{domain}.jsonl, sends the
lifted NL sentence (grounded_sentence with prop_N placeholders) to
GPT-5, and writes the full entry + prediction to
eval_data/translation_eval/llm_direct/gpt-5/{domain}.jsonl.

Usage:
  python scripts/llm_translate.py --domains cleanup_world GLTL conformal navi
  python scripts/llm_translate.py --domains navi --limit 500
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

EVAL_DIR = ROOT / "eval_data" / "translation_eval"
INPUT_DIR = EVAL_DIR / "nl2tl" / "raw_nl"
OUTPUT_DIR = EVAL_DIR / "llm_direct" / "gpt-5"

SYSTEM_PROMPT = """\
You are a temporal logic translator. Given an English specification \
where atomic propositions have been replaced with placeholders \
(prop_1, prop_2, ...), output the equivalent Linear Temporal Logic (LTL) formula \
using those same placeholders as atoms.

Available LTL operators (use these exact tokens):
  finally    (eventually / diamond)
  globally   (always / box)
  next       (next step)
  until      (strong until)
  not        (negation)
  and        (conjunction)
  or         (disjunction)
  implies    (implication)
  double_implies  (biconditional / iff)

Use parentheses for grouping. Output ONLY the LTL formula, nothing else. \
Do not include any explanation, markdown, or formatting.\
"""

FEW_SHOT = [
    {
        "input": "Whenever prop_1 holds, prop_2 holds as well.",
        "output": "globally ( prop_1 implies prop_2 )",
    },
    {
        "input": "prop_1 must eventually happen.",
        "output": "finally prop_1",
    },
    {
        "input": "prop_1 must always hold, with at most a two-step grace period for recovery.",
        "output": "not globally ( not ( prop_1 and next prop_1 ) )",
    },
    {
        "input": "prop_2 persists until prop_1 holds, or else prop_2 holds forever.",
        "output": "( prop_2 until prop_1 ) or globally prop_2",
    },
    {
        "input": "If prop_1 ever holds, prop_2 must have held beforehand.",
        "output": "( finally prop_1 ) implies ( not prop_1 until ( prop_2 and not prop_1 ) )",
    },
]


def build_messages(lifted_sentence: str) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT:
        msgs.append({"role": "user", "content": ex["input"]})
        msgs.append({"role": "assistant", "content": ex["output"]})
    msgs.append({"role": "user", "content": lifted_sentence})
    return msgs


def translate_one(client: OpenAI, lifted_sentence: str, model: str) -> str:
    msgs = build_messages(lifted_sentence)
    resp = client.chat.completions.create(
        model=model,
        messages=msgs,
        max_completion_tokens=4096,
    )
    return resp.choices[0].message.content.strip()


def run_domain(
    client: OpenAI,
    domain: str,
    model: str,
    limit: int,
    skip_existing: bool,
) -> None:
    in_path = INPUT_DIR / f"{domain}.jsonl"
    out_path = OUTPUT_DIR / f"{domain}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[{domain}] input not found at {in_path}, skipping")
        return

    existing_ids: set = set()
    existing_rows: list[dict] = []
    if skip_existing and out_path.exists():
        with out_path.open() as f:
            for line in f:
                row = json.loads(line)
                existing_ids.add(row.get("id"))
                existing_rows.append(row)
        print(f"[{domain}] {len(existing_ids)} existing predictions found")

    rows: list[dict] = []
    with in_path.open() as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            rows.append(json.loads(line))

    todo = [r for r in rows if r.get("id") not in existing_ids]
    print(f"[{domain}] {len(rows)} total, {len(todo)} to translate")

    results = list(existing_rows)
    for i, row in enumerate(todo):
        lifted = " ".join(row.get("grounded_sentence", row.get("sentence", [])))
        try:
            pred = translate_one(client, lifted, model)
        except Exception as e:
            print(f"  [{domain}] row {row.get('id', i)} failed: {e}")
            pred = ""
            time.sleep(2)

        row["prediction"] = pred
        results.append(row)

        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            print(f"  [{domain}] {i+1}/{len(todo)} done")

    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_with_pred = sum(1 for r in results if r.get("prediction"))
    print(f"[{domain}] wrote {len(results)} rows ({n_with_pred} with predictions) to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domains", nargs="+", default=["cleanup_world", "GLTL", "conformal", "navi"])
    p.add_argument("--model", default="gpt-4.1")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = p.parse_args()

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    for domain in args.domains:
        run_domain(client, domain, args.model, args.limit, args.skip_existing)


if __name__ == "__main__":
    main()
