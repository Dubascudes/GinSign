#!/usr/bin/env python3
"""Lang2LTL grounding baseline via OpenAI embedding cosine similarity.

For each domain:
  1. Enumerate every valid grounded atom from the signature
     (predicate + type-filtered constant tuples).
  2. Embed every canonical atom string with text-embedding-3-large.
  3. For each lifted AP in the test set, embed the NL reference,
     find the most cosine-similar canonical atom, and record the
     predicted predicate + arguments.
  4. Save per-AP predictions to eval_data/grounding_eval/lang2ltl/.
"""

import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

EVAL = ROOT / "eval_data"
SIG_DIR = ROOT / "data" / "signatures"
DOMAINS = ["traffic_light", "search_and_rescue", "warehouse"]
EMBEDDING_MODEL = "text-embedding-3-large"
BATCH_SIZE = 2048


def load_signature(domain: str):
    from ginsign.signature_io import load_signature as _load
    return _load(SIG_DIR / f"{domain}.json")


def enumerate_grounded_atoms(sig) -> list[dict]:
    """Enumerate all valid grounded atoms from the signature.

    Returns a list of dicts with keys:
      canon_str:  e.g. "change(light_east,yellow)"
      predicate:  e.g. "change"
      args:       e.g. ["light_east", "yellow"]
    """
    atoms = []
    for pname in sig.predicate_names():
        pdef = sig.predicates[pname]
        if pdef.arity == 0:
            atoms.append({
                "canon_str": pname,
                "predicate": pname,
                "args": [],
            })
        else:
            slot_options = []
            for slot_idx in range(pdef.arity):
                consts = sig.slot_constants(pname, slot_idx)
                slot_options.append(consts)
            for combo in itertools.product(*slot_options):
                canon_str = f"{pname}({','.join(combo)})"
                atoms.append({
                    "canon_str": canon_str,
                    "predicate": pname,
                    "args": list(combo),
                })
    return atoms


def get_embeddings(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Embed a list of texts in batches, returning (N, D) array."""
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embs = [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
        all_embs.extend(embs)
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.1)
    return np.array(all_embs, dtype=np.float32)


def build_nl_reference(prop: dict) -> str:
    """Build a natural-language string for an AP from its ref fields."""
    action_ref = prop.get("action_ref", "")
    args_ref = prop.get("args_ref", [])
    parts = [action_ref] + args_ref
    return " ".join(p for p in parts if p)


def load_test_data(domain: str) -> list[dict]:
    path = EVAL / "translation_eval" / "nl2tl" / "raw_nl" / f"{domain}.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_domain(client: OpenAI, domain: str) -> list[dict]:
    """Run Lang2LTL grounding for one domain. Returns per-AP results."""
    print(f"\n{'='*60}")
    print(f"  Domain: {domain}")
    print(f"{'='*60}")

    sig = load_signature(domain)
    atoms = enumerate_grounded_atoms(sig)
    print(f"  Signature: {len(sig.predicate_names())} predicates, "
          f"{sum(len(v) for v in sig.constants.values())} constants")
    print(f"  Enumerated {len(atoms)} grounded atoms")

    # Embed all canonical atoms
    atom_texts = [a["canon_str"] for a in atoms]
    print(f"  Embedding {len(atom_texts)} canonical atoms...")
    atom_embs = get_embeddings(client, atom_texts)
    atom_embs_norm = atom_embs / np.linalg.norm(atom_embs, axis=1, keepdims=True)

    # Collect all NL AP references from test data
    test_rows = load_test_data(domain)
    queries = []
    for row in test_rows:
        prop_dict = row.get("prop_dict", {})
        for prop_key in sorted(prop_dict):
            prop_val = prop_dict[prop_key]
            if prop_val is None:
                continue
            nl_ref = build_nl_reference(prop_val)
            queries.append({
                "sample_id": row["id"],
                "prop_key": prop_key,
                "nl_ref": nl_ref,
                "gt_predicate": prop_val["action_canon"],
                "gt_args": prop_val.get("args_canon", []),
            })

    print(f"  {len(queries)} AP references to ground")

    # Embed all NL references
    query_texts = [q["nl_ref"] for q in queries]
    print(f"  Embedding {len(query_texts)} NL references...")
    query_embs = get_embeddings(client, query_texts)
    query_embs_norm = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)

    # Cosine similarity: (N_queries, N_atoms)
    sims = query_embs_norm @ atom_embs_norm.T
    best_idx = sims.argmax(axis=1)

    results = []
    for i, q in enumerate(queries):
        best_atom = atoms[best_idx[i]]
        results.append({
            "domain": domain,
            "sample_id": q["sample_id"],
            "prop_key": q["prop_key"],
            "nl_ref": q["nl_ref"],
            "gt_predicate": q["gt_predicate"],
            "gt_args": q["gt_args"],
            "pred_predicate": best_atom["predicate"],
            "pred_args": best_atom["args"],
            "pred_canon_str": best_atom["canon_str"],
            "cosine_sim": float(sims[i, best_idx[i]]),
        })

    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute predicate accuracy and argument F1 from per-AP results."""
    pred_correct = sum(1 for r in results if r["pred_predicate"] == r["gt_predicate"])
    pred_total = len(results)

    arg_f1_sum = 0.0
    arg_total = 0
    for r in results:
        gt_args = set(r["gt_args"])
        pred_args = set(r["pred_args"])
        if gt_args or pred_args:
            arg_total += 1
            tp = len(gt_args & pred_args)
            prec = tp / len(pred_args) if pred_args else 0
            rec = tp / len(gt_args) if gt_args else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            arg_f1_sum += f1

    joint_correct = sum(
        1 for r in results
        if r["pred_predicate"] == r["gt_predicate"]
        and r["pred_args"] == r["gt_args"]
    )

    return {
        "pred_acc": 100.0 * pred_correct / pred_total if pred_total else 0,
        "arg_f1": 100.0 * arg_f1_sum / arg_total if arg_total else 0,
        "joint_acc": 100.0 * joint_correct / pred_total if pred_total else 0,
        "n": pred_total,
    }


def main():
    import sys
    sys.path.insert(0, str(ROOT / "src"))

    client = OpenAI()

    out_dir = EVAL / "grounding_eval" / "lang2ltl"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Lang2LTL Grounding Baseline")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    all_results = {}
    for domain in DOMAINS:
        results = run_domain(client, domain)
        all_results[domain] = results

        # Save per-AP results
        out_path = out_dir / f"{domain}.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"  Saved {len(results)} results to {out_path}")

        # Compute and print metrics
        m = compute_metrics(results)
        print(f"\n  Results for {domain}:")
        print(f"    Predicate Accuracy: {m['pred_acc']:.1f}%")
        print(f"    Argument F1:        {m['arg_f1']:.1f}%")
        print(f"    Joint Accuracy:     {m['joint_acc']:.1f}%")
        print(f"    N:                  {m['n']}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Lang2LTL Embedding Grounding")
    print(f"{'='*70}")
    print(f"  {'Domain':<20s}  {'Pred Acc':>8s}  {'Arg F1':>8s}  {'Joint':>8s}  {'N':>5s}")
    print(f"  {'-'*55}")
    for domain in DOMAINS:
        m = compute_metrics(all_results[domain])
        dname = domain.replace("_", " ").title()
        print(f"  {dname:<20s}  {m['pred_acc']:>7.1f}%  {m['arg_f1']:>7.1f}%  "
              f"{m['joint_acc']:>7.1f}%  {m['n']:>5d}")


if __name__ == "__main__":
    main()
