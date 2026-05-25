#!/usr/bin/env python3
"""Run lifting, translation, and grounding evaluations with a local LLM.

Designed for HiPerGator: serves the model via vLLM's OpenAI-compatible API,
then runs all three evaluation tasks against it.

Uses async concurrency to saturate the vLLM server with parallel requests.

Usage:
  # 1. Start vLLM server (in a separate terminal / SLURM job):
  #    vllm serve "openai/gpt-oss-120b"
  #
  # 2. Run this script pointing at the server:
  python scripts/eval_oss_llm.py \
      --api-base http://localhost:8000/v1 \
      --model "openai/gpt-oss-120b" \
      --tag gpt-oss-120b \
      --concurrency 32

All results are saved under eval_data/ in the same format as existing
baselines, so rebuild_tables.py / summarize_results.py can pick them up.
"""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
EVAL = ROOT / "eval_data"

ALL_DOMAINS = [
    "traffic_light", "search_and_rescue", "warehouse",
    "cleanup_world", "conformal", "GLTL", "navi",
]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_raw_nl(domain):
    return load_jsonl(EVAL / "translation_eval" / "nl2tl" / "raw_nl" / f"{domain}.jsonl")


def format_signature(domain):
    from ginsign.signature_io import load_signature
    sig = load_signature(ROOT / "data" / "signatures" / f"{domain}.json")
    lines = [f"Domain: {sig.name}", ""]
    lines.append("Types:")
    for s in sig.sorts:
        lines.append(f"  {s}: {', '.join(sig.constants_of(s))}")
    lines.append("")
    lines.append("Predicates:")
    for pname in sig.predicate_names():
        pdef = sig.predicates[pname]
        if pdef.arity == 0:
            lines.append(f"  {pname}()")
        else:
            slots = ["/".join(ss) for ss in pdef.arg_sorts]
            lines.append(f"  {pname}({', '.join(slots)})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Async batch runner
# ---------------------------------------------------------------------------

async def run_batch(client, model, requests, concurrency, label=""):
    """Run a list of (messages, max_tokens) requests with bounded concurrency.

    Returns list of response content strings in the same order.
    """
    sem = asyncio.Semaphore(concurrency)
    results = [None] * len(requests)
    done = 0
    t0 = time.time()

    async def process(idx, messages, max_tokens):
        nonlocal done
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=0, max_tokens=max_tokens,
                )
                results[idx] = resp.choices[0].message.content.strip()
            except Exception as e:
                results[idx] = ""
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                print(f"    {label} {done}/{len(requests)} ({rate:.1f} req/s)", flush=True)

    tasks = [
        process(i, msgs, mt)
        for i, (msgs, mt) in enumerate(requests)
    ]
    await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    print(f"    {label} {len(requests)}/{len(requests)} done in {elapsed:.0f}s "
          f"({len(requests)/elapsed:.1f} req/s)", flush=True)
    return results


# ---------------------------------------------------------------------------
# Task 1: Lifting
# ---------------------------------------------------------------------------

LIFTING_SYSTEM = """\
You are a sentence segmentation model. Given a natural-language specification \
about a robot or system, identify each atomic-proposition (AP) span — the \
contiguous subsequence of tokens that describes one action or property — and \
label them prop_1, prop_2, etc. in order of first appearance.

Output a JSON object with two keys:
  "lifted_sentence": list of tokens with AP spans replaced by prop_N
  "lifted_sentence_prop_ids": list of integers (same length as the ORIGINAL \
sentence), where 0 = not part of any AP, 1 = part of prop_1, 2 = part of prop_2, etc.

Output ONLY valid JSON, no explanation."""

LIFTING_FEWSHOT = [
    {
        "input": "The robot must find the bookbag and then deliver it to shipping.",
        "output": json.dumps({
            "lifted_sentence": ["The", "robot", "must", "prop_1", "and", "then", "prop_2"],
            "lifted_sentence_prop_ids": [0, 0, 0, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2],
        }),
    },
    {
        "input": "You must eventually, avoid set east light yellow.",
        "output": json.dumps({
            "lifted_sentence": ["You", "must", "eventually,", "avoid", "prop_1"],
            "lifted_sentence_prop_ids": [0, 0, 0, 0, 1, 1, 1, 1],
        }),
    },
]


async def run_lifting(client, model, domains, tag, concurrency):
    print(f"\n{'='*60}")
    print(f"  Task 1: Lifting ({tag})")
    print(f"{'='*60}")

    out_dir = EVAL / "lifting_eval" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        rows = load_raw_nl(domain)
        requests = []
        for row in rows:
            sentence = " ".join(row["sentence"])
            msgs = [{"role": "system", "content": LIFTING_SYSTEM}]
            for fs in LIFTING_FEWSHOT:
                msgs.append({"role": "user", "content": fs["input"]})
                msgs.append({"role": "assistant", "content": fs["output"]})
            msgs.append({"role": "user", "content": sentence})
            requests.append((msgs, 1024))

        results = await run_batch(client, model, requests, concurrency, f"[{domain}]")

        output = []
        for row, content in zip(rows, results):
            try:
                parsed = json.loads(content) if content else {}
                output.append({
                    "id": row["id"],
                    "lifted_sentence_prop_ids": parsed.get("lifted_sentence_prop_ids", [0] * len(row["sentence"])),
                    "grounded_sentence": parsed.get("lifted_sentence", list(row["sentence"])),
                })
            except json.JSONDecodeError:
                output.append({
                    "id": row["id"],
                    "lifted_sentence_prop_ids": [0] * len(row["sentence"]),
                    "grounded_sentence": list(row["sentence"]),
                })

        out_path = out_dir / f"{domain}.jsonl"
        with open(out_path, "w") as f:
            for r in output:
                f.write(json.dumps(r) + "\n")
        print(f"  [{domain}] {len(output)} samples -> {out_path}")


# ---------------------------------------------------------------------------
# Task 2: Lifted Translation
# ---------------------------------------------------------------------------

TRANSLATION_SYSTEM = """\
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
Do not include any explanation, markdown, or formatting."""

TRANSLATION_FEWSHOT = [
    ("Whenever prop_1 holds, prop_2 holds as well.",
     "globally ( prop_1 implies prop_2 )"),
    ("prop_1 must eventually happen.",
     "finally prop_1"),
    ("prop_1 must always hold, with at most a two-step grace period for recovery.",
     "not globally ( not ( prop_1 and next prop_1 ) )"),
    ("prop_2 persists until prop_1 holds, or else prop_2 holds forever.",
     "( prop_2 until prop_1 ) or globally prop_2"),
    ("If prop_1 ever holds, prop_2 must have held beforehand.",
     "( finally prop_1 ) implies ( not prop_1 until ( prop_2 and not prop_1 ) )"),
]


async def run_translation(client, model, domains, tag, concurrency):
    print(f"\n{'='*60}")
    print(f"  Task 2: Lifted Translation ({tag})")
    print(f"{'='*60}")

    out_dir = EVAL / "translation_eval" / "llm_direct" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        rows = load_raw_nl(domain)
        requests = []
        for row in rows:
            lifted = " ".join(row.get("grounded_sentence", row["sentence"]))
            msgs = [{"role": "system", "content": TRANSLATION_SYSTEM}]
            for inp, out in TRANSLATION_FEWSHOT:
                msgs.append({"role": "user", "content": inp})
                msgs.append({"role": "assistant", "content": out})
            msgs.append({"role": "user", "content": lifted})
            requests.append((msgs, 512))

        results = await run_batch(client, model, requests, concurrency, f"[{domain}]")

        output = []
        for row, prediction in zip(rows, results):
            out_row = dict(row)
            out_row["prediction"] = prediction or ""
            output.append(out_row)

        out_path = out_dir / f"{domain}.jsonl"
        with open(out_path, "w") as f:
            for r in output:
                f.write(json.dumps(r) + "\n")
        print(f"  [{domain}] {len(output)} samples -> {out_path}")


# ---------------------------------------------------------------------------
# Task 3: Grounding
# ---------------------------------------------------------------------------

GROUNDING_PROMPT = """Scenario Configuration: {scenario}
Sentence: {sentence}
Lifted Sentence: {lifted_sentence}

For each prop_n in the lifted sentence, identify which predicate from the \
Scenario Configuration it corresponds to, and which constants are its arguments. \
Use ONLY predicate names and constant names that appear in the Scenario Configuration above.

Output ONLY a JSON object (no explanation, no markdown) in this exact form:
{{
  "prop_1": {{"action_canon": "<predicate_name>", "args_canon": ["<const1>", "<const2>"]}},
  "prop_2": {{"action_canon": "<predicate_name>", "args_canon": ["<const1>"]}}
}}"""


async def run_grounding(client, model, domains, tag, concurrency):
    print(f"\n{'='*60}")
    print(f"  Task 3: Grounding ({tag})")
    print(f"{'='*60}")

    out_dir = EVAL / "grounding_eval" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        scenario = format_signature(domain)
        rows = load_raw_nl(domain)
        requests = []
        for row in rows:
            sentence = " ".join(row.get("sentence", []))
            lifted = " ".join(row.get("grounded_sentence", []))
            prompt = GROUNDING_PROMPT.format(
                scenario=scenario, sentence=sentence, lifted_sentence=lifted,
            )
            msgs = [{"role": "user", "content": prompt}]
            requests.append((msgs, 1024))

        results = await run_batch(client, model, requests, concurrency, f"[{domain}]")

        output = []
        for row, content in zip(rows, results):
            output.append({
                "id": "local",
                "custom_id": f"{domain}-{tag}-{row['id']}",
                "response": {
                    "status_code": 200,
                    "body": {"choices": [{"message": {"content": content or ""}}]},
                },
            })

        out_path = out_dir / f"{domain}_scenario.jsonl"
        with open(out_path, "w") as f:
            for r in output:
                f.write(json.dumps(r) + "\n")
        print(f"  [{domain}] {len(output)} samples -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def async_main(args):
    tag = args.tag or re.sub(r"[^a-zA-Z0-9_-]", "_", args.model)
    client = AsyncOpenAI(base_url=args.api_base, api_key=args.api_key)

    print(f"Model:       {args.model}")
    print(f"Tag:         {tag}")
    print(f"API:         {args.api_base}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Domains:     {args.domains}")
    print(f"Tasks:       {args.tasks}")

    if "lifting" in args.tasks:
        await run_lifting(client, args.model, args.domains, tag, args.concurrency)
    if "translation" in args.tasks:
        await run_translation(client, args.model, args.domains, tag, args.concurrency)
    if "grounding" in args.tasks:
        await run_grounding(client, args.model, args.domains, tag, args.concurrency)

    print(f"\nDone. Results saved with tag '{tag}'.")
    print(f"  Lifting:     eval_data/lifting_eval/{tag}/")
    print(f"  Translation: eval_data/translation_eval/llm_direct/{tag}/")
    print(f"  Grounding:   eval_data/grounding_eval/{tag}/")


def main():
    p = argparse.ArgumentParser(
        description="Run lifting, translation, and grounding evals with a local LLM.",
    )
    p.add_argument("--api-base", default="http://localhost:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--domains", nargs="+", default=ALL_DOMAINS)
    p.add_argument("--tasks", nargs="+", default=["lifting", "translation", "grounding"],
                    choices=["lifting", "translation", "grounding"])
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--concurrency", type=int, default=32,
                    help="Max concurrent requests to the vLLM server")
    args = p.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
