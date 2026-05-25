#!/usr/bin/env python3
"""Submit and collect batched grounding API calls for prior-work domains.

Usage:
  python scripts/batch_grounding.py submit   # create batch files and submit
  python scripts/batch_grounding.py collect  # poll and download results
"""

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

EVAL = ROOT / "eval_data"
BATCH_DIR = ROOT / "logs" / "batch_grounding"
BATCH_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def format_signature_for_prompt(sig):
    from ginsign.signature_io import load_signature
    lines = [f"Domain: {sig.name}", ""]
    lines.append("Types:")
    for s in sig.sorts:
        consts = sig.constants_of(s)
        lines.append(f"  {s}: {', '.join(consts)}")
    lines.append("")
    lines.append("Predicates:")
    for pname in sig.predicate_names():
        pdef = sig.predicates[pname]
        if pdef.arity == 0:
            lines.append(f"  {pname}()")
        else:
            slot_strs = ["/".join(ss) for ss in pdef.arg_sorts]
            lines.append(f"  {pname}({', '.join(slot_strs)})")
    return "\n".join(lines)


GROUNDING_PROMPT = """Scenario Configuration: {scenario}
Sentence: {sentence}
Lifted Sentence: {lifted_sentence}

Return a dictionary of the types, predicates, and constants for each prop_n in the lifted sentence. The dictionary should be in this form:
prop_dict: {{
"prop_1": {{
"action_canon": *string*,
"args_canon": *list of strings*,
}},
"prop_2": {{
"action_canon": *string*,
"args_canon": *list of strings*,
}}
}}
Now, predict:
prop_dict:"""


def needed_jobs():
    """Return list of (model_dir, openai_model_id, domain) tuples still needed."""
    jobs = []
    models = [
        ("gpt-4o", "gpt-4o"),
        ("gpt-5", "gpt-5"),
    ]
    domains = ["cleanup_world", "conformal", "GLTL", "navi"]
    for mdir, mid in models:
        for d in domains:
            f = EVAL / "grounding_eval" / mdir / f"{d}_scenario.jsonl"
            if not f.exists() or sum(1 for _ in open(f)) < 10:
                jobs.append((mdir, mid, d))
    return jobs


def submit():
    from openai import OpenAI
    from ginsign.signature_io import load_signature

    client = OpenAI()
    jobs = needed_jobs()
    if not jobs:
        print("All OpenAI batches already complete.")
        return

    # Group by model
    by_model = {}
    for mdir, mid, domain in jobs:
        by_model.setdefault(mid, []).append((mdir, domain))

    for model_id, domain_list in by_model.items():
        # Build one JSONL batch file per model with all domains
        batch_file = BATCH_DIR / f"batch_{model_id}.jsonl"
        requests = []
        for mdir, domain in domain_list:
            sig = load_signature(ROOT / "data" / "signatures" / f"{domain}.json")
            scenario = format_signature_for_prompt(sig)
            gt_rows = load_jsonl(EVAL / "translation_eval" / "nl2tl" / "raw_nl" / f"{domain}.jsonl")
            for row in gt_rows:
                sid = row["id"]
                sentence = " ".join(row.get("sentence", []))
                lifted = " ".join(row.get("grounded_sentence", []))
                prompt = GROUNDING_PROMPT.format(
                    scenario=scenario, sentence=sentence, lifted_sentence=lifted
                )
                requests.append({
                    "custom_id": f"{domain}-{model_id}-{sid}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "max_tokens": 512,
                    }
                })

        with open(batch_file, "w") as f:
            for r in requests:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(requests)} requests to {batch_file}")

        # Upload and submit
        with open(batch_file, "rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        print(f"Uploaded file: {uploaded.id}")

        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"model": model_id, "task": "grounding_priorwork"},
        )
        print(f"Batch submitted: {batch.id} (model={model_id}, n={len(requests)})")

        # Save batch ID for collection
        with open(BATCH_DIR / f"batch_{model_id}_id.txt", "w") as f:
            f.write(batch.id + "\n")
            json.dump([(mdir, d) for mdir, d in domain_list], f)
            f.write("\n")

    # Also submit Anthropic batches
    submit_anthropic()


def submit_anthropic():
    import anthropic
    from ginsign.signature_io import load_signature

    client = anthropic.Anthropic()
    domains = ["cleanup_world", "conformal", "GLTL", "navi"]
    needed = []
    for d in domains:
        f = EVAL / "grounding_eval" / "claude-sonnet" / f"{d}_scenario.jsonl"
        if not f.exists() or sum(1 for _ in open(f)) < 10:
            needed.append(d)
    if not needed:
        print("All Anthropic batches already complete.")
        return

    requests = []
    for domain in needed:
        sig = load_signature(ROOT / "data" / "signatures" / f"{domain}.json")
        scenario = format_signature_for_prompt(sig)
        gt_rows = load_jsonl(EVAL / "translation_eval" / "nl2tl" / "raw_nl" / f"{domain}.jsonl")
        for row in gt_rows:
            sid = row["id"]
            sentence = " ".join(row.get("sentence", []))
            lifted = " ".join(row.get("grounded_sentence", []))
            prompt = GROUNDING_PROMPT.format(
                scenario=scenario, sentence=sentence, lifted_sentence=lifted
            )
            requests.append({
                "custom_id": f"{domain}-claude-sonnet-{sid}",
                "params": {
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 512,
                    "messages": [{"role": "user", "content": prompt}],
                }
            })

    batch = client.messages.batches.create(requests=requests)
    print(f"Anthropic batch submitted: {batch.id} (n={len(requests)})")
    with open(BATCH_DIR / "batch_anthropic_id.txt", "w") as f:
        f.write(batch.id + "\n")
        json.dump(needed, f)
        f.write("\n")


def collect():
    from openai import OpenAI
    client = OpenAI()

    # Collect OpenAI batches
    for model_id in ["gpt-4o", "gpt-5"]:
        id_file = BATCH_DIR / f"batch_{model_id}_id.txt"
        if not id_file.exists():
            continue
        lines = id_file.read_text().strip().split("\n")
        batch_id = lines[0]
        domain_list = json.loads(lines[1])

        batch = client.batches.retrieve(batch_id)
        print(f"{model_id}: status={batch.status}, completed={batch.request_counts.completed}/{batch.request_counts.total}")

        if batch.status != "completed":
            continue

        # Download results
        content = client.files.content(batch.output_file_id)
        all_results = [json.loads(line) for line in content.text.strip().split("\n")]

        # Split by domain
        by_domain = {}
        for r in all_results:
            cid = r.get("custom_id", "")
            for mdir, domain in domain_list:
                if cid.startswith(domain):
                    by_domain.setdefault((mdir, domain), []).append(r)
                    break

        for (mdir, domain), results in by_domain.items():
            out_dir = EVAL / "grounding_eval" / mdir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{domain}_scenario.jsonl"
            with open(out_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            print(f"  Saved {len(results)} results to {out_path}")

    # Collect Anthropic batch
    id_file = BATCH_DIR / "batch_anthropic_id.txt"
    if id_file.exists():
        lines = id_file.read_text().strip().split("\n")
        batch_id = lines[0]
        needed_domains = json.loads(lines[1])

        import anthropic
        client_a = anthropic.Anthropic()
        batch = client_a.messages.batches.retrieve(batch_id)
        print(f"claude-sonnet: status={batch.processing_status}, ended={batch.ended_at}")

        if batch.processing_status == "ended":
            by_domain = {d: [] for d in needed_domains}
            for result in client_a.messages.batches.results(batch_id):
                cid = result.custom_id
                for domain in needed_domains:
                    if cid.startswith(domain):
                        content = result.result.message.content[0].text if result.result.message.content else ""
                        by_domain[domain].append({
                            "id": "batch",
                            "custom_id": cid,
                            "response": {
                                "status_code": 200,
                                "body": {"choices": [{"message": {"content": content}}]}
                            }
                        })
                        break

            out_dir = EVAL / "grounding_eval" / "claude-sonnet"
            out_dir.mkdir(parents=True, exist_ok=True)
            for domain, results in by_domain.items():
                out_path = out_dir / f"{domain}_scenario.jsonl"
                with open(out_path, "w") as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")
                print(f"  Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "submit"
    if cmd == "submit":
        submit()
    elif cmd == "collect":
        collect()
    else:
        print(f"Usage: {sys.argv[0]} [submit|collect]")
