#!/usr/bin/env python
"""Few-shot LLM grounding baseline (rebuttal WLTJ W2).

Augments the zero-shot signature grounding prompt (paper A.6 /
scripts/batch_grounding.py) with k in-domain exemplars sampled from the
training split, and reruns gpt-4o on the same item ids as the existing
zero-shot runs. Writes eval_data/grounding_eval/gpt-4o-fewshot/<domain>_scenario.jsonl
in the same result format.
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from openai import AsyncOpenAI

from batch_grounding import GROUNDING_PROMPT, format_signature_for_prompt
from ginsign.signature_io import load_signature

VLTL = ROOT / "data/VLTL-Bench"
PW = ROOT / "data/priorwork_cleaned"
DOMAINS = {
    "warehouse": VLTL, "traffic_light": VLTL, "search_and_rescue": VLTL,
    "cleanup_world": PW, "GLTL": PW, "conformal": PW, "navi": PW,
}


def load_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh]


def exemplar_text(rows, k, seed=0):
    rng = random.Random(seed)
    pool = [r for r in rows if r.get("prop_dict")]
    exs = rng.sample(pool, k)
    blocks = []
    for i, r in enumerate(exs, 1):
        pd = {p: {"action_canon": info["action_canon"],
                  "args_canon": info.get("args_canon", [])}
              for p, info in r["prop_dict"].items() if info}
        blocks.append(
            f"Example {i}:\n"
            f"Sentence: {' '.join(r['sentence'])}\n"
            f"Lifted Sentence: {' '.join(r['grounded_sentence'])}\n"
            f"prop_dict: {json.dumps(pd, indent=1)}"
        )
    return "\n\n".join(blocks)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--distractors", type=int, default=0,
                    help="inject N distractor constants per sort into the "
                         "prompted signature (rebuttal WLTJ W3); implies "
                         "zero-shot unless --k > 0")
    args = ap.parse_args()

    client = AsyncOpenAI()
    suffix = f"-fewshot" if args.k > 0 else ""
    if args.distractors:
        suffix += f"-distractor{args.distractors}"
    out_root = ROOT / "eval_data/grounding_eval" / f"{args.model}{suffix}"
    out_root.mkdir(parents=True, exist_ok=True)

    for domain in args.domains:
        out_path = out_root / f"{domain}_scenario.jsonl"
        if out_path.exists() and sum(1 for _ in open(out_path)) > 0:
            print(f"[{domain}] exists, skipping")
            continue
        base = DOMAINS[domain]
        # same item ids as the existing zero-shot gpt-4o run
        zs_path = ROOT / "eval_data/grounding_eval/gpt-4o" / f"{domain}_scenario.jsonl"
        ids = [int(json.loads(l)["custom_id"].rsplit("-", 1)[1])
               for l in open(zs_path)]
        test = {r["id"]: r for r in load_jsonl(base / "test" / f"{domain}.jsonl")}
        train_rows = load_jsonl(base / "train" / f"{domain}.jsonl")
        sig = load_signature(ROOT / "data/signatures" / f"{domain}.json")
        if args.distractors:
            from ginsign.signature_transforms import inject_distractors
            donor_names = [d for d in DOMAINS if d != domain]
            donors = [load_signature(ROOT / "data/signatures" / f"{d}.json")
                      for d in donor_names]
            sig = inject_distractors(sig, donors, n_per_sort=args.distractors,
                                     seed=42)
        scenario = format_signature_for_prompt(sig)
        exemplars = exemplar_text(train_rows, args.k) if args.k > 0 else ""

        sem = asyncio.Semaphore(args.concurrency)

        async def one(eid, row):
            core = GROUNDING_PROMPT.format(
                scenario=scenario,
                sentence=" ".join(row["sentence"]),
                lifted_sentence=" ".join(row["grounded_sentence"]),
            )
            prompt = (f"In-domain examples:\n{exemplars}\n\n" + core
                      if exemplars else core)
            async with sem:
                for attempt in range(4):
                    try:
                        r = await client.chat.completions.create(
                            model=args.model, temperature=0.0, max_tokens=1024,
                            messages=[{"role": "user", "content": prompt}])
                        return {
                            "custom_id": f"{domain}-{args.model}-fewshot-{eid}",
                            "response": {"status_code": 200, "body": {
                                "model": r.model,
                                "choices": [{"index": 0, "message": {
                                    "role": "assistant",
                                    "content": r.choices[0].message.content}}],
                            }},
                        }
                    except Exception as e:
                        if attempt == 3:
                            return {"custom_id": f"{domain}-{args.model}-fewshot-{eid}",
                                    "response": None, "error": str(e)}
                        await asyncio.sleep(2 ** attempt * 2)

        tasks = [one(i, test[i]) for i in ids if i in test]
        print(f"[{domain}] {len(tasks)} requests (k={args.k})", flush=True)
        results = await asyncio.gather(*tasks)
        with open(out_path, "w") as fh:
            for r in results:
                fh.write(json.dumps(r) + "\n")
        nerr = sum(1 for r in results if r.get("response") is None)
        print(f"[{domain}] done, {nerr} errors", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
