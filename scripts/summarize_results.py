#!/usr/bin/env python3
"""Build the main results table: Lifted Translation × Grounding Method → GLE.

For each (lifting method, grounding method, domain) triple:
  1. Load the masked lifted TL prediction from the translation eval data.
  2. Load the per-AP grounding prediction from the grounding eval data.
  3. Substitute groundings into the masked formula.
  4. Compare the grounded prediction against ground-truth grounded TL (GLE).
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
EVAL = ROOT / "eval_data"
VLTL_DOMAINS = ["traffic_light", "search_and_rescue", "warehouse"]
PRIOR_DOMAINS = ["cleanup_world", "conformal", "GLTL", "navi"]
ALL_DOMAINS = VLTL_DOMAINS + PRIOR_DOMAINS
DOMAIN_SHORT = {
    "traffic_light": "TL",
    "search_and_rescue": "S&R",
    "warehouse": "WH",
    "cleanup_world": "CW",
    "conformal": "CF",
    "GLTL": "GLTL",
    "navi": "Navi",
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_ground_truth(domain):
    path = EVAL / "translation_eval" / "nl2tl" / "raw_nl" / f"{domain}.jsonl"
    return {r["id"]: r for r in load_jsonl(path)}


def parse_llm_content(content):
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        return None


def atom_str(predicate, args):
    return f"{predicate}({','.join(args)})" if args else predicate


# ---------------------------------------------------------------------------
# Grounding loaders: each returns {sample_id: {prop_key: canon_str}}
# ---------------------------------------------------------------------------

def load_llm_grounding(model_dir, domain):
    fpath = EVAL / "grounding_eval" / model_dir / f"{domain}_scenario.jsonl"
    if not fpath.exists():
        return {}
    preds = {}
    for row in load_jsonl(fpath):
        cid = row.get("custom_id", "")
        parts = cid.rsplit("-", 1)
        if len(parts) != 2:
            continue
        try:
            sid = int(parts[1])
        except ValueError:
            continue
        if not cid.startswith(domain):
            continue
        content = row["response"]["body"]["choices"][0]["message"]["content"]
        parsed = parse_llm_content(content)
        if parsed is None:
            continue
        prop_map = {}
        for pk, pv in parsed.items():
            if pv is None:
                continue
            prop_map[pk] = atom_str(
                pv.get("action_canon", ""), pv.get("args_canon", [])
            )
        preds[sid] = prop_map
    return preds


def load_lang2ltl_grounding(domain):
    fpath = EVAL / "grounding_eval" / "lang2ltl" / f"{domain}.jsonl"
    if not fpath.exists():
        return {}
    preds = {}
    for row in load_jsonl(fpath):
        sid = row["sample_id"]
        if sid not in preds:
            preds[sid] = {}
        preds[sid][row["prop_key"]] = atom_str(
            row["pred_predicate"], row["pred_args"]
        )
    return preds


def load_ginsign_grounding(domain):
    """Run the GinSign BERT pointer grounder checkpoint."""
    import torch
    from torch.utils.data import DataLoader
    from ginsign.grounding_dataset import GroundingDataset, make_collate_fn
    from ginsign.ptr_grounder import PointerJointGrounder
    from ginsign.signature_io import load_signature

    ckpt_path = ROOT / "outputs" / domain / "best.pt"
    sig_path = ROOT / "data" / "signatures" / f"{domain}.json"
    corpus = EVAL / "translation_eval" / "nl2tl" / "raw_nl" / f"{domain}.jsonl"
    if not ckpt_path.exists():
        return {}

    device = torch.device("cpu")
    sig = load_signature(sig_path)
    ds = GroundingDataset(str(corpus), sig)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=make_collate_fn(sig))

    model = PointerJointGrounder(bert_name="bert-base-cased").to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    pred_names = sig.predicate_names()

    # Reconstruct the (sample_id, prop_key) ordering that GroundingDataset uses
    flat_keys = []
    for row in load_jsonl(corpus):
        for pk in sorted(row.get("prop_dict", {})):
            pv = row["prop_dict"][pk]
            if pv is None:
                continue
            cn = pv["action_canon"]
            if cn not in sig.predicates:
                continue
            if len(pv.get("args_canon", [])) != sig.arity_of(cn):
                continue
            ar = pv.get("action_ref", cn)
            arefs = pv.get("args_ref", pv.get("args_canon", []))
            if not (ar + " " + " ".join(arefs)).strip():
                continue
            flat_keys.append((row["id"], pk))

    preds = {}
    idx = 0
    with torch.no_grad():
        for batch in loader:
            pred_logits, ap_repr, pred_cand_reprs, _ = model.ground_predicate(
                batch["ap_texts"], batch["predicate_lists"]
            )
            B = pred_logits.size(0)
            H = model.hidden_size
            pp_idx = pred_logits.argmax(dim=-1)
            pred_repr = pred_cand_reprs[torch.arange(B), pp_idx]

            gold_arg_idx = batch["gold_arg_idx"].to(device)
            slot_cands, slot_masks = [], []
            gold_arg_reprs = torch.zeros(
                B, len(batch["slot_candidate_lists"]), H, device=device
            )
            for r, (cr, tm) in enumerate(
                zip(batch["slot_candidate_lists"], batch["slot_type_masks"])
            ):
                _, _, cr_emb, sp = model.encode_stage(
                    batch["ap_texts"], cr, tm, device=device
                )
                slot_cands.append(cr_emb)
                slot_masks.append(sp.effective_mask())
                gr = gold_arg_idx[:, r]
                present = gr >= 0
                if present.any():
                    safe = gr.clamp(min=0)
                    gold_arg_reprs[:, r, :] = (
                        cr_emb[torch.arange(B), safe] * present.unsqueeze(-1).float()
                    )

            arg_preds = []
            for r in range(len(slot_cands)):
                prev = gold_arg_reprs[:, :r, :] if r > 0 else None
                logits = model.arg_decoder.step(
                    ap_repr, pred_repr, prev, slot_cands[r], slot_masks[r]
                )
                arg_preds.append(logits.argmax(dim=-1))

            for b in range(B):
                if idx >= len(flat_keys):
                    break
                sid, pk = flat_keys[idx]
                idx += 1
                pred_p = pred_names[pp_idx[b].item()]
                arity = sig.arity_of(pred_p)
                args = []
                for r in range(arity):
                    cl = batch["slot_candidate_lists"][r]
                    ai = arg_preds[r][b].item()
                    args.append(cl[b][ai] if ai < len(cl[b]) else "?")
                if sid not in preds:
                    preds[sid] = {}
                preds[sid][pk] = atom_str(pred_p, args)

    return preds


# ---------------------------------------------------------------------------
# Lifted translation loader
# ---------------------------------------------------------------------------

def load_lifted_translations(dirpath, domain):
    fpath = Path(dirpath) / f"{domain}.jsonl"
    if not fpath.exists():
        return {}
    return {r["id"]: r.get("prediction", "").strip()
            for r in load_jsonl(fpath) if r.get("prediction", "").strip()}


# ---------------------------------------------------------------------------
# Combine and evaluate
# ---------------------------------------------------------------------------

def normalize(s):
    return re.sub(r"\s+", " ", s.strip())


def ground_formula(masked_pred, grounding_map):
    result = masked_pred
    for pk in sorted(grounding_map, key=lambda k: int(k.split("_")[1]), reverse=True):
        result = result.replace(pk, grounding_map[pk])
    return result


def parse_atom(canon_str):
    """Parse 'pred(a,b)' → ('pred', ['a','b']) or 'pred' → ('pred', [])."""
    m = re.match(r"^([^(]+)\(([^)]*)\)$", canon_str)
    if m:
        pred = m.group(1)
        args = [a.strip().rstrip(")") for a in m.group(2).split(",") if a.strip()]
        return pred, args
    return canon_str, []


def clean_args(args):
    """Strip trailing parens from args (navi data artifact)."""
    return [a.rstrip(")") for a in args]


def evaluate_combination(lifted, groundings, gt_by_domain, domain):
    """Evaluate GLE by checking LE (masked structure) AND per-prop grounding.

    This avoids domain-specific TL atom format issues by comparing
    (predicate, args) tuples directly rather than string-substituting
    into the formula.
    """
    gt = gt_by_domain[domain]
    le_ok = gle_ok = total = 0

    for sid, masked_pred in lifted.items():
        if sid not in gt:
            continue
        gt_row = gt[sid]
        gt_masked = normalize(" ".join(gt_row.get("masked_tl", [])))
        total += 1

        # LE: does the masked formula structure match?
        le_match = gt_masked and normalize(masked_pred) == gt_masked
        if le_match:
            le_ok += 1

        # GLE: LE must pass AND every prop must be correctly grounded
        if not le_match:
            continue
        gmap = groundings.get(sid, {})
        if not gmap:
            continue
        gt_props = gt_row.get("prop_dict", {})
        all_correct = True
        for pk, gt_prop in gt_props.items():
            if gt_prop is None:
                continue
            # Only check props that appear in the formula
            if pk not in masked_pred:
                continue
            pred_atom = gmap.get(pk)
            if pred_atom is None:
                all_correct = False
                break
            pred_p, pred_a = parse_atom(pred_atom)
            gt_args = clean_args(gt_prop.get("args_canon", []))
            if pred_p != gt_prop["action_canon"] or clean_args(pred_a) != gt_args:
                all_correct = False
                break
        if all_correct:
            gle_ok += 1

    return {
        "le": 100.0 * le_ok / total if total else 0,
        "gle": 100.0 * gle_ok / total if total else 0,
        "n": total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    gt_by_domain = {d: load_ground_truth(d) for d in ALL_DOMAINS}

    lifting_methods = [
        ("LLM Prompting", EVAL / "translation_eval" / "llm_direct" / "gpt-5", VLTL_DOMAINS),
        ("NL2TL", EVAL / "translation_eval" / "nl2tl" / "llm_masked_nl" / "gpt4_1_lifting", ALL_DOMAINS),
        ("GraFT", EVAL / "translation_eval" / "nl2tl" / "gt_lifting", ALL_DOMAINS),
    ]

    grounding_specs = [
        ("GPT-4o",        lambda d: load_llm_grounding("gpt-4o", d)),
        ("GPT-5",         lambda d: load_llm_grounding("gpt-5", d)),
        ("Claude Sonnet", lambda d: load_llm_grounding("claude-sonnet", d)),
        ("Lang2LTL",      load_lang2ltl_grounding),
        ("GinSign",       load_ginsign_grounding),
    ]

    print("Loading grounding predictions...", flush=True)
    gcache = {}
    for gname, gfn in grounding_specs:
        gcache[gname] = {}
        for domain in ALL_DOMAINS:
            try:
                gcache[gname][domain] = gfn(domain)
            except Exception as e:
                print(f"  {gname}/{domain}: skipped ({e})")
                gcache[gname][domain] = {}
        print(f"  {gname}: loaded", flush=True)

    # Table
    print()
    cw = 7
    ncols = len(ALL_DOMAINS)
    total_w = 18 + 2 + 15 + ncols * (cw + 1) + 4
    print("=" * total_w)
    print("  Lifted Translation × Grounding → GLE (%)")
    print("=" * total_w)

    hdr = f"{'Lift+Translate':>18s}  {'Grounding':>15s}"
    for d in ALL_DOMAINS:
        hdr += f" {DOMAIN_SHORT[d]:>{cw}s}"
    print(f"\n{hdr}")
    print("-" * total_w)

    for lname, lpath, ldomains in lifting_methods:
        first = True
        for gname, _ in grounding_specs:
            label = lname if first else ""
            first = False
            print(f"{label:>18s}  {gname:>15s}", end="")
            for domain in ALL_DOMAINS:
                if domain not in ldomains:
                    print(f" {'—':>{cw}s}", end="")
                    continue
                lifted = load_lifted_translations(lpath, domain)
                grd = gcache[gname].get(domain, {})
                if not lifted or not grd:
                    print(f" {'—':>{cw}s}", end="")
                    continue
                r = evaluate_combination(lifted, grd, gt_by_domain, domain)
                print(f" {r['gle']:>{cw}.1f}", end="")
            print()

        # Ground truth row = LE ceiling (perfect grounding always passes)
        print(f"{'':>18s}  {'Ground Truth':>15s}", end="")
        for domain in ALL_DOMAINS:
            if domain not in ldomains:
                print(f" {'—':>{cw}s}", end="")
                continue
            lifted = load_lifted_translations(lpath, domain)
            gt = gt_by_domain[domain]
            # With perfect grounding, GLE = LE
            le_ok = total = 0
            for sid, masked_pred in lifted.items():
                if sid not in gt:
                    continue
                gt_masked = normalize(" ".join(gt[sid].get("masked_tl", [])))
                total += 1
                if gt_masked and normalize(masked_pred) == gt_masked:
                    le_ok += 1
            gle = 100.0 * le_ok / total if total else 0
            print(f" {gle:>{cw}.1f}", end="")
        print()
        print("-" * total_w)

    print()
    print("  GLE = Grounded Logical Equivalence (end-to-end correctness)")
    print("  Ground Truth row = LE ceiling (perfect grounding)")


if __name__ == "__main__":
    main()
