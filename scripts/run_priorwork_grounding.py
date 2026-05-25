#!/usr/bin/env python3
"""Run all grounding baselines on prior-work domains.

1. Lang2LTL (OpenAI embeddings)
2. LLM prompting grounding (GPT-4o, GPT-5, Claude Sonnet)
3. GinSign BERT pointer grounder (already has checkpoints)

Saves results to eval_data/grounding_eval/{method}/{domain}.jsonl
in a unified per-AP format for easy cross-referencing.
"""

import itertools
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

EVAL = ROOT / "eval_data"
SIG_DIR = ROOT / "data" / "signatures"
PRIORWORK_DOMAINS = ["cleanup_world", "conformal", "GLTL", "navi"]
EMBEDDING_MODEL = "text-embedding-3-large"
BATCH_SIZE = 2048


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_signature(domain):
    from ginsign.signature_io import load_signature as _load
    return _load(SIG_DIR / f"{domain}.json")


def load_test_aps(domain):
    """Load all (sample_id, prop_key, gt, nl_ref) from raw_nl."""
    path = EVAL / "translation_eval" / "nl2tl" / "raw_nl" / f"{domain}.jsonl"
    aps = []
    for row in load_jsonl(path):
        for pk in sorted(row.get("prop_dict", {})):
            pv = row["prop_dict"][pk]
            if pv is None:
                continue
            action_ref = pv.get("action_ref", pv["action_canon"])
            args_ref = pv.get("args_ref", pv.get("args_canon", []))
            nl_ref = (action_ref + " " + " ".join(args_ref)).strip()
            aps.append({
                "sample_id": row["id"],
                "prop_key": pk,
                "nl_ref": nl_ref,
                "gt_predicate": pv["action_canon"],
                "gt_args": pv.get("args_canon", []),
                "sentence": row.get("sentence", []),
                "grounded_sentence": row.get("grounded_sentence", []),
            })
    return aps


def atom_str(pred, args):
    return f"{pred}({','.join(args)})" if args else pred


def format_signature_for_prompt(sig):
    """Format a signature as the Scenario Configuration for the LLM prompt."""
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
            slot_strs = []
            for slot_sorts in pdef.arg_sorts:
                slot_strs.append("/".join(slot_sorts))
            lines.append(f"  {pname}({', '.join(slot_strs)})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Lang2LTL (embeddings)
# ---------------------------------------------------------------------------

def run_lang2ltl(domains):
    from openai import OpenAI
    client = OpenAI()

    def get_embeddings(texts):
        all_embs = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            embs = [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
            all_embs.extend(embs)
            if i + BATCH_SIZE < len(texts):
                time.sleep(0.1)
        return np.array(all_embs, dtype=np.float32)

    for domain in domains:
        print(f"\n  Lang2LTL: {domain}")
        sig = load_signature(domain)

        atoms = []
        for pname in sig.predicate_names():
            pdef = sig.predicates[pname]
            if pdef.arity == 0:
                atoms.append({"canon": pname, "pred": pname, "args": []})
            else:
                slots = [sig.slot_constants(pname, i) for i in range(pdef.arity)]
                for combo in itertools.product(*slots):
                    atoms.append({
                        "canon": atom_str(pname, list(combo)),
                        "pred": pname, "args": list(combo),
                    })
        print(f"    {len(atoms)} atoms, ", end="", flush=True)

        atom_embs = get_embeddings([a["canon"] for a in atoms])
        atom_norm = atom_embs / np.linalg.norm(atom_embs, axis=1, keepdims=True)

        aps = load_test_aps(domain)
        print(f"{len(aps)} APs")
        query_embs = get_embeddings([a["nl_ref"] for a in aps])
        query_norm = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)

        sims = query_norm @ atom_norm.T
        best = sims.argmax(axis=1)

        out_dir = EVAL / "grounding_eval" / "lang2ltl"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{domain}.jsonl", "w") as f:
            for i, ap in enumerate(aps):
                a = atoms[best[i]]
                f.write(json.dumps({
                    "domain": domain,
                    "sample_id": ap["sample_id"],
                    "prop_key": ap["prop_key"],
                    "nl_ref": ap["nl_ref"],
                    "gt_predicate": ap["gt_predicate"],
                    "gt_args": ap["gt_args"],
                    "pred_predicate": a["pred"],
                    "pred_args": a["args"],
                    "pred_canon_str": a["canon"],
                    "cosine_sim": float(sims[i, best[i]]),
                }) + "\n")
        print(f"    saved to {out_dir / domain}.jsonl")


# ---------------------------------------------------------------------------
# 2. LLM grounding (GPT-4o, GPT-5, Claude Sonnet)
# ---------------------------------------------------------------------------

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


def run_llm_grounding_openai(model_name, model_id, domains):
    from openai import OpenAI
    client = OpenAI()

    for domain in domains:
        print(f"\n  {model_name}: {domain}")
        sig = load_signature(domain)
        scenario = format_signature_for_prompt(sig)
        aps_by_sample = {}
        gt_rows = load_jsonl(EVAL / "translation_eval" / "nl2tl" / "raw_nl" / f"{domain}.jsonl")
        for row in gt_rows:
            aps_by_sample[row["id"]] = row

        sample_ids = sorted(aps_by_sample.keys())
        results = []
        for idx, sid in enumerate(sample_ids):
            row = aps_by_sample[sid]
            sentence = " ".join(row.get("sentence", []))
            lifted = " ".join(row.get("grounded_sentence", []))
            prompt = GROUNDING_PROMPT.format(
                scenario=scenario, sentence=sentence, lifted_sentence=lifted
            )
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=512,
                )
                content = resp.choices[0].message.content
            except Exception as e:
                content = ""
                print(f"    error on {sid}: {e}")

            results.append({
                "id": f"batch_manual",
                "custom_id": f"{domain}-{model_name}-{sid}",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{"message": {"content": content}}]
                    }
                }
            })
            if (idx + 1) % 50 == 0:
                print(f"    {idx+1}/{len(sample_ids)}", flush=True)

        out_dir = EVAL / "grounding_eval" / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{domain}_scenario.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"    saved {len(results)} predictions")


def run_llm_grounding_anthropic(domains):
    import anthropic
    client = anthropic.Anthropic()

    for domain in domains:
        print(f"\n  Claude Sonnet: {domain}")
        sig = load_signature(domain)
        scenario = format_signature_for_prompt(sig)
        gt_rows = load_jsonl(EVAL / "translation_eval" / "nl2tl" / "raw_nl" / f"{domain}.jsonl")
        aps_by_sample = {row["id"]: row for row in gt_rows}

        sample_ids = sorted(aps_by_sample.keys())
        results = []
        for idx, sid in enumerate(sample_ids):
            row = aps_by_sample[sid]
            sentence = " ".join(row.get("sentence", []))
            lifted = " ".join(row.get("grounded_sentence", []))
            prompt = GROUNDING_PROMPT.format(
                scenario=scenario, sentence=sentence, lifted_sentence=lifted
            )
            try:
                resp = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = resp.content[0].text
            except Exception as e:
                content = ""
                print(f"    error on {sid}: {e}")

            results.append({
                "id": f"batch_manual",
                "custom_id": f"{domain}-claude-sonnet-{sid}",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{"message": {"content": content}}]
                    }
                }
            })
            if (idx + 1) % 50 == 0:
                print(f"    {idx+1}/{len(sample_ids)}", flush=True)

        out_dir = EVAL / "grounding_eval" / "claude-sonnet"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{domain}_scenario.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"    saved {len(results)} predictions")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    domains = PRIORWORK_DOMAINS
    print(f"Running grounding baselines on: {domains}")

    print("\n=== Lang2LTL (embeddings) ===")
    run_lang2ltl(domains)

    print("\n=== GPT-4o grounding ===")
    run_llm_grounding_openai("gpt-4o", "gpt-4o", domains)

    print("\n=== GPT-5 grounding ===")
    run_llm_grounding_openai("gpt-5", "gpt-5", domains)

    print("\n=== Claude Sonnet grounding ===")
    run_llm_grounding_anthropic(domains)

    print("\nDone!")


if __name__ == "__main__":
    main()
