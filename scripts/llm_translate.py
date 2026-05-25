#!/usr/bin/env python
"""GPT-5 lifted translation via the OpenAI Batch API.

Workflow:
  1. submit  - Build JSONL batch request, upload, create batch job
  2. poll    - Check batch status until complete
  3. collect - Download results, merge predictions into output JSONL

Usage:
  # Submit batch for all 4 priorwork domains:
  python scripts/llm_translate.py submit \
      --domains cleanup_world GLTL conformal navi --model gpt-5

  # Poll until done:
  python scripts/llm_translate.py poll --batch-id <id>

  # Collect results:
  python scripts/llm_translate.py collect --batch-id <id> \
      --domains cleanup_world GLTL conformal navi
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
BATCH_DIR = ROOT / "eval_data" / "batch_jobs"

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


def build_messages(lifted_sentence: str) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for inp, out in FEW_SHOT:
        msgs.append({"role": "user", "content": inp})
        msgs.append({"role": "assistant", "content": out})
    msgs.append({"role": "user", "content": lifted_sentence})
    return msgs


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

def cmd_submit(args):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    requests = []
    index = {}  # custom_id -> (domain, row_index)

    for domain in args.domains:
        in_path = INPUT_DIR / f"{domain}.jsonl"
        if not in_path.exists():
            print(f"[{domain}] input not found at {in_path}, skipping")
            continue
        with in_path.open() as f:
            for i, line in enumerate(f):
                if i >= args.limit:
                    break
                row = json.loads(line)
                lifted = " ".join(row.get("grounded_sentence", row.get("sentence", [])))
                custom_id = f"{domain}-{row.get('id', i)}"
                requests.append({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        "messages": build_messages(lifted),
                        "max_completion_tokens": 4096,
                    },
                })
                index[custom_id] = (domain, i, row)

    # Write batch request JSONL
    batch_file = BATCH_DIR / "translate_request.jsonl"
    with batch_file.open("w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    print(f"Wrote {len(requests)} requests to {batch_file}")

    # Save index for collect step
    index_file = BATCH_DIR / "translate_index.jsonl"
    with index_file.open("w") as f:
        for cid, (domain, i, row) in index.items():
            f.write(json.dumps({"custom_id": cid, "domain": domain, "row_index": i, "row": row}) + "\n")
    print(f"Wrote index to {index_file}")

    # Upload and create batch
    uploaded = client.files.create(file=batch_file.open("rb"), purpose="batch")
    print(f"Uploaded file: {uploaded.id}")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch created: {batch.id}  status={batch.status}")
    print(f"\nNext steps:")
    print(f"  python scripts/llm_translate.py poll --batch-id {batch.id}")
    print(f"  python scripts/llm_translate.py collect --batch-id {batch.id} --domains {' '.join(args.domains)}")

    # Save batch ID for convenience
    (BATCH_DIR / "latest_batch_id.txt").write_text(batch.id + "\n")


# ---------------------------------------------------------------------------
# Poll
# ---------------------------------------------------------------------------

def cmd_poll(args):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    batch_id = args.batch_id or (BATCH_DIR / "latest_batch_id.txt").read_text().strip()

    while True:
        batch = client.batches.retrieve(batch_id)
        done = batch.request_counts.completed
        failed = batch.request_counts.failed
        total = batch.request_counts.total
        print(f"[{batch.status}] {done}/{total} done, {failed} failed")
        if batch.status in ("completed", "failed", "cancelled", "expired"):
            if batch.output_file_id:
                print(f"Output file: {batch.output_file_id}")
            if batch.error_file_id:
                print(f"Error file: {batch.error_file_id}")
            break
        time.sleep(30)


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------

def cmd_collect(args):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    batch_id = args.batch_id or (BATCH_DIR / "latest_batch_id.txt").read_text().strip()

    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        print(f"Batch {batch_id} status={batch.status}, not collecting yet")
        return

    # Download results
    result_content = client.files.content(batch.output_file_id).text
    result_path = BATCH_DIR / "translate_results.jsonl"
    result_path.write_text(result_content)
    print(f"Downloaded {len(result_content.splitlines())} results to {result_path}")

    # Load index
    index: dict = {}
    index_path = BATCH_DIR / "translate_index.jsonl"
    with index_path.open() as f:
        for line in f:
            entry = json.loads(line)
            index[entry["custom_id"]] = entry

    # Parse results and group by domain
    per_domain: dict[str, list] = {}
    n_ok = 0
    n_empty = 0
    for line in result_content.splitlines():
        result = json.loads(line)
        cid = result["custom_id"]
        entry = index.get(cid)
        if entry is None:
            continue
        domain = entry["domain"]
        row = entry["row"]

        resp = result.get("response", {})
        body = resp.get("body", {})
        choices = body.get("choices", [])
        prediction = ""
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            prediction = content.strip()
        if prediction:
            n_ok += 1
        else:
            n_empty += 1

        row["prediction"] = prediction
        per_domain.setdefault(domain, []).append(row)

    # Write per-domain output files
    for domain in (args.domains or sorted(per_domain)):
        rows = per_domain.get(domain, [])
        if not rows:
            continue
        out_path = OUTPUT_DIR / f"{domain}.jsonl"
        with out_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n_with_pred = sum(1 for r in rows if r.get("prediction"))
        print(f"[{domain}] wrote {len(rows)} rows ({n_with_pred} with predictions) to {out_path}")

    print(f"\nTotal: {n_ok} predictions, {n_empty} empty")

    # Check for missing rows (failed in batch) and report
    all_cids_in_index = set(index.keys())
    all_cids_in_results = set()
    for line in result_content.splitlines():
        all_cids_in_results.add(json.loads(line)["custom_id"])
    missing = all_cids_in_index - all_cids_in_results
    if missing:
        print(f"\n{len(missing)} requests missing from results (failed in batch)")
        missing_path = BATCH_DIR / "translate_missing.jsonl"
        with missing_path.open("w") as f:
            for cid in sorted(missing):
                f.write(json.dumps(index[cid]) + "\n")
        print(f"Missing IDs written to {missing_path}")

    # Download error file if present
    batch = client.batches.retrieve(batch_id)
    if batch.error_file_id:
        err_content = client.files.content(batch.error_file_id).text
        err_path = BATCH_DIR / "translate_errors.jsonl"
        err_path.write_text(err_content)
        print(f"Error details written to {err_path}")


# ---------------------------------------------------------------------------
# Retry failed requests
# ---------------------------------------------------------------------------

def cmd_retry(args):
    """Resubmit just the missing/failed requests from a prior batch."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    missing_path = BATCH_DIR / "translate_missing.jsonl"
    if not missing_path.exists():
        print("No missing requests file found. Run 'collect' first.")
        return

    requests = []
    with missing_path.open() as f:
        for line in f:
            entry = json.loads(line)
            row = entry["row"]
            lifted = " ".join(row.get("grounded_sentence", row.get("sentence", [])))
            requests.append({
                "custom_id": entry["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": build_messages(lifted),
                    "max_completion_tokens": 4096,
                },
            })

    print(f"Retrying {len(requests)} failed requests")

    batch_file = BATCH_DIR / "translate_retry_request.jsonl"
    with batch_file.open("w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    uploaded = client.files.create(file=batch_file.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Retry batch: {batch.id}  status={batch.status}")
    (BATCH_DIR / "latest_batch_id.txt").write_text(batch.id + "\n")


# ---------------------------------------------------------------------------
# Merge: combine original + retry results into final outputs
# ---------------------------------------------------------------------------

def cmd_merge(args):
    """Merge results from original + retry batches into final per-domain files."""
    result_files = [
        BATCH_DIR / "translate_results.jsonl",
        BATCH_DIR / "translate_retry_results.jsonl",
    ]

    index: dict = {}
    index_path = BATCH_DIR / "translate_index.jsonl"
    with index_path.open() as f:
        for line in f:
            entry = json.loads(line)
            index[entry["custom_id"]] = entry

    per_domain: dict[str, dict] = {}  # domain -> {cid -> row}
    n_ok = 0
    for rf in result_files:
        if not rf.exists():
            continue
        for line in rf.read_text().splitlines():
            result = json.loads(line)
            cid = result["custom_id"]
            entry = index.get(cid)
            if entry is None:
                continue
            domain = entry["domain"]
            row = entry["row"]

            resp = result.get("response", {})
            choices = resp.get("body", {}).get("choices", [])
            prediction = ""
            if choices:
                prediction = choices[0].get("message", {}).get("content", "").strip()
            if not prediction:
                continue

            row["prediction"] = prediction
            per_domain.setdefault(domain, {})[cid] = row
            n_ok += 1

    for domain in sorted(per_domain):
        rows = list(per_domain[domain].values())
        out_path = OUTPUT_DIR / f"{domain}.jsonl"
        with out_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[{domain}] {len(rows)} predictions written to {out_path}")

    print(f"\nTotal merged: {n_ok} predictions")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--domains", nargs="+", default=["cleanup_world", "GLTL", "conformal", "navi"])
    p_submit.add_argument("--model", default="gpt-5")
    p_submit.add_argument("--limit", type=int, default=500)

    p_poll = sub.add_parser("poll")
    p_poll.add_argument("--batch-id", default=None)

    p_collect = sub.add_parser("collect")
    p_collect.add_argument("--batch-id", default=None)
    p_collect.add_argument("--domains", nargs="+", default=None)

    p_retry = sub.add_parser("retry")
    p_retry.add_argument("--model", default="gpt-5")

    sub.add_parser("merge")

    args = p.parse_args()
    if args.cmd == "submit":
        cmd_submit(args)
    elif args.cmd == "poll":
        cmd_poll(args)
    elif args.cmd == "collect":
        cmd_collect(args)
    elif args.cmd == "retry":
        cmd_retry(args)
    elif args.cmd == "merge":
        cmd_merge(args)


if __name__ == "__main__":
    main()
