#!/usr/bin/env python3
"""Rebuild grounding, lifting, and end-to-end eval result tables from eval_data.

Reproduces Tables 3 and 4 from the ICML 2026 submission, plus a lifting table.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval_data"
DOMAINS = ["traffic_light", "search_and_rescue", "warehouse"]
DOMAIN_DISPLAY = {
    "traffic_light": "Traffic Light",
    "search_and_rescue": "Search & Rescue",
    "warehouse": "Warehouse",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_ground_truth(domain: str) -> dict[int, dict]:
    path = EVAL / "translation_eval" / "nl2tl" / "raw_nl" / f"{domain}.jsonl"
    return {row["id"]: row for row in load_jsonl(path)}


def parse_llm_content(content: str) -> dict | None:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def extract_batch_predictions(rows: list[dict]) -> dict[tuple[str, int], dict]:
    preds = {}
    for row in rows:
        cid = row.get("custom_id", "")
        parts = cid.rsplit("-", 1)
        if len(parts) != 2:
            continue
        try:
            sample_id = int(parts[1])
        except ValueError:
            continue
        prefix = parts[0]
        domain = None
        for d in DOMAINS:
            if prefix.startswith(d):
                domain = d
                break
        if domain is None:
            continue
        resp = row.get("response", {})
        body = resp.get("body", {})
        choices = body.get("choices", [])
        if not choices:
            continue
        content = choices[0].get("message", {}).get("content", "")
        parsed = parse_llm_content(content)
        if parsed is not None:
            preds[(domain, sample_id)] = parsed
    return preds


# ---------------------------------------------------------------------------
# Table 3: Grounding Evaluation  (F1 per AP)
# ---------------------------------------------------------------------------

def compute_grounding_metrics(
    preds: dict[tuple[str, int], dict],
    gt_by_domain: dict[str, dict[int, dict]],
    domain: str,
) -> dict:
    """Compute per-AP predicate accuracy, argument F1, and macro F1s."""
    gt = gt_by_domain[domain]

    # Per-AP metrics
    pred_correct = 0
    pred_total = 0
    arg_f1_sum = 0.0
    arg_total = 0

    # Per-class for macro F1
    pred_tp = defaultdict(int)
    pred_fp = defaultdict(int)
    pred_fn = defaultdict(int)
    arg_tp = defaultdict(int)
    arg_fp = defaultdict(int)
    arg_fn = defaultdict(int)

    for (d, sid), pred_dict in preds.items():
        if d != domain or sid not in gt:
            continue
        gt_props = gt[sid].get("prop_dict", {})

        for prop_key, gt_prop in gt_props.items():
            if gt_prop is None:
                continue
            gt_action = gt_prop["action_canon"]
            gt_args = set(gt_prop.get("args_canon", []))

            pred_prop = pred_dict.get(prop_key)
            if pred_prop is None:
                pred_total += 1
                pred_fn[gt_action] += 1
                for a in gt_args:
                    arg_fn[a] += 1
                if gt_args:
                    arg_total += 1
                continue

            # Predicate
            pred_action = pred_prop.get("action_canon", "")
            pred_total += 1
            if pred_action == gt_action:
                pred_correct += 1
                pred_tp[gt_action] += 1
            else:
                pred_fp[pred_action] += 1
                pred_fn[gt_action] += 1

            # Arguments (per-AP set F1)
            pred_args = set(pred_prop.get("args_canon", []))
            if gt_args or pred_args:
                arg_total += 1
                tp_count = len(gt_args & pred_args)
                prec = tp_count / len(pred_args) if pred_args else 0
                rec = tp_count / len(gt_args) if gt_args else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                arg_f1_sum += f1

            for a in gt_args & pred_args:
                arg_tp[a] += 1
            for a in pred_args - gt_args:
                arg_fp[a] += 1
            for a in gt_args - pred_args:
                arg_fn[a] += 1

    def macro_f1(tp, fp, fn):
        classes = set(tp) | set(fp) | set(fn)
        if not classes:
            return 0.0
        f1s = []
        for c in classes:
            t, f_p, f_n = tp[c], fp[c], fn[c]
            if t + f_p == 0 or t + f_n == 0:
                f1s.append(0.0)
                continue
            p = t / (t + f_p)
            r = t / (t + f_n)
            f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
        return 100.0 * sum(f1s) / len(f1s)

    pred_acc = 100.0 * pred_correct / pred_total if pred_total else 0
    arg_f1_avg = 100.0 * arg_f1_sum / arg_total if arg_total else 0
    pred_macro = macro_f1(pred_tp, pred_fp, pred_fn)
    arg_macro = macro_f1(arg_tp, arg_fp, arg_fn)

    return {
        "pred_acc": pred_acc,
        "pred_macro_f1": pred_macro,
        "arg_f1": arg_f1_avg,
        "arg_macro_f1": arg_macro,
        "pred_n": pred_total,
        "arg_n": arg_total,
    }


def load_lang2ltl_preds(domain: str) -> dict[tuple[str, int], dict]:
    """Load Lang2LTL per-AP results into the same format as LLM batch preds."""
    fpath = EVAL / "grounding_eval" / "lang2ltl" / f"{domain}.jsonl"
    if not fpath.exists():
        return {}
    preds: dict[tuple[str, int], dict] = {}
    for row in load_jsonl(fpath):
        key = (row["domain"], row["sample_id"])
        if key not in preds:
            preds[key] = {}
        preds[key][row["prop_key"]] = {
            "action_canon": row["pred_predicate"],
            "args_canon": row["pred_args"],
        }
    return preds


def compute_lang2ltl_metrics(domain: str, gt_by_domain: dict) -> dict | None:
    preds = load_lang2ltl_preds(domain)
    if not preds:
        return None
    return compute_grounding_metrics(preds, gt_by_domain, domain)


def build_grounding_table():
    print("=" * 100)
    print("TABLE 3: Grounding Evaluation  (per-AP metrics %)")
    print("=" * 100)

    gt_by_domain = {d: load_ground_truth(d) for d in DOMAINS}
    grounding_dir = EVAL / "grounding_eval"

    # LLM grounding models to include (scenario prompt)
    llm_models = [
        ("gpt-4o", "GPT-4o"),
        ("gpt-5", "GPT-5"),
        ("claude-sonnet", "Claude Sonnet"),
    ]

    # Table header
    header_domains = "".join(f"  {DOMAIN_DISPLAY[d]:^16s}" for d in DOMAINS)
    sub_header = "".join(f"  {'Pred':>5s} {'Arg':>5s} {'Jnt':>5s}" for _ in DOMAINS)
    print(f"\n{'Grounding Method':>22s}{header_domains}")
    print(f"{'':>22s}{sub_header}")
    print("-" * 73)

    def print_row(name: str, metrics_by_domain: list):
        print(f"{name:>22s}", end="")
        for m in metrics_by_domain:
            if m:
                print(f"  {m['pred_acc']:>5.1f} {m['arg_f1']:>5.1f} {m.get('joint_acc', 0):>5.1f}", end="")
            else:
                print(f"  {'—':>5s} {'—':>5s} {'—':>5s}", end="")
        print()

    # LLM grounding baselines (scenario prompt)
    for dir_name, display_name in llm_models:
        mdir = grounding_dir / dir_name
        cells = []
        for domain in DOMAINS:
            fpath = mdir / f"{domain}_scenario.jsonl"
            if not fpath.exists():
                cells.append(None)
                continue
            rows = load_jsonl(fpath)
            preds = extract_batch_predictions(rows)
            m = compute_grounding_metrics(preds, gt_by_domain, domain)
            # Compute joint accuracy
            gt = gt_by_domain[domain]
            joint_correct = 0
            joint_total = 0
            for (d, sid), pred_dict in preds.items():
                if d != domain or sid not in gt:
                    continue
                gt_props = gt[sid].get("prop_dict", {})
                for pk, gp in gt_props.items():
                    if gp is None:
                        continue
                    joint_total += 1
                    pp = pred_dict.get(pk)
                    if pp and pp.get("action_canon") == gp["action_canon"] \
                       and pp.get("args_canon", []) == gp.get("args_canon", []):
                        joint_correct += 1
            m["joint_acc"] = 100.0 * joint_correct / joint_total if joint_total else 0
            cells.append(m)
        print_row(display_name, cells)

    # Lang2LTL
    lang_cells = []
    for domain in DOMAINS:
        fpath = EVAL / "grounding_eval" / "lang2ltl" / f"{domain}.jsonl"
        if not fpath.exists():
            lang_cells.append(None)
            continue
        results = load_jsonl(fpath)
        pred_correct = sum(1 for r in results if r["pred_predicate"] == r["gt_predicate"])
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
        n = len(results)
        lang_cells.append({
            "pred_acc": 100.0 * pred_correct / n if n else 0,
            "arg_f1": 100.0 * arg_f1_sum / arg_total if arg_total else 0,
            "joint_acc": 100.0 * joint_correct / n if n else 0,
        })
    print_row("Lang2LTL", lang_cells)

    # GinSign — run the PointerJointGrounder checkpoint per domain
    ginsign_cells = []
    try:
        import torch
        from torch.utils.data import DataLoader
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from ginsign.eval import evaluate as ginsign_evaluate
        from ginsign.grounding_dataset import GroundingDataset, make_collate_fn
        from ginsign.ptr_grounder import PointerJointGrounder
        from ginsign.signature_io import load_signature as _load_sig

        device = torch.device("cpu")
        for domain in DOMAINS:
            ckpt_path = ROOT / "outputs" / domain / "best.pt"
            sig_path = ROOT / "data" / "signatures" / f"{domain}.json"
            corpus = EVAL / "translation_eval" / "nl2tl" / "raw_nl" / f"{domain}.jsonl"
            if not ckpt_path.exists():
                ginsign_cells.append(None)
                continue
            sig = _load_sig(sig_path)
            ds = GroundingDataset(str(corpus), sig)
            loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=make_collate_fn(sig))
            model = PointerJointGrounder(bert_name="bert-base-cased").to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            m = ginsign_evaluate(model, loader, sig, device)
            ginsign_cells.append({
                "pred_acc": 100.0 * m["pred_acc"],
                "arg_f1": 100.0 * m["arg_acc_full_tuple"],
                "joint_acc": 100.0 * m["joint_acc"],
            })
    except Exception as e:
        print(f"  [GinSign eval error: {e}]")
        ginsign_cells = [None] * len(DOMAINS)
    print_row("GinSign (proposed)", ginsign_cells)

    print(f"\n  Pred = predicate accuracy; Arg = per-AP argument F1; Jnt = joint accuracy")
    print()


# ---------------------------------------------------------------------------
# Lifting Evaluation
# ---------------------------------------------------------------------------

def compute_lifting_metrics(pred_rows: list[dict], gt: dict[int, dict]) -> dict:
    tp = fp = fn = 0
    total_tokens = correct_tokens = 0
    n_samples = n_exact = 0

    for prow in pred_rows:
        sid = prow["id"]
        if sid not in gt:
            continue
        gt_row = gt[sid]
        gt_ids = gt_row["lifted_sentence_prop_ids"]
        pred_ids = prow["lifted_sentence_prop_ids"]
        n_samples += 1

        if len(pred_ids) == len(gt_ids):
            exact = True
            for g, p in zip(gt_ids, pred_ids):
                total_tokens += 1
                if g == p:
                    correct_tokens += 1
                else:
                    exact = False
                g_prop = g > 0
                p_prop = p > 0
                if g_prop and p_prop and g == p:
                    tp += 1
                elif g_prop and p_prop and g != p:
                    fp += 1
                    fn += 1
                elif p_prop and not g_prop:
                    fp += 1
                elif g_prop and not p_prop:
                    fn += 1
            if exact:
                n_exact += 1
        else:
            gt_gs = gt_row.get("grounded_sentence", [])
            pred_gs = prow.get("grounded_sentence", [])
            gt_props = [t for t in gt_gs if t.startswith("prop_")]
            pred_props = [t for t in pred_gs if t.startswith("prop_")]
            for p in gt_props:
                if p in pred_props:
                    tp += 1
                else:
                    fn += 1
            for p in pred_props:
                if p not in gt_props:
                    fp += 1
            total_tokens += max(len(gt_ids), len(pred_ids))
            correct_tokens += sum(1 for a, b in zip(gt_gs, pred_gs) if a == b)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {
        "accuracy": 100 * correct_tokens / total_tokens if total_tokens else 0,
        "precision": 100 * prec,
        "recall": 100 * rec,
        "f1": 100 * f1,
        "exact_match": 100 * n_exact / n_samples if n_samples else 0,
        "n_samples": n_samples,
    }


def build_lifting_table():
    print("=" * 100)
    print("LIFTING EVALUATION  (Token-level metrics %)")
    print("=" * 100)

    gt_by_domain = {d: load_ground_truth(d) for d in DOMAINS}
    lifting_dir = EVAL / "lifting_eval"
    model_dirs = sorted(lifting_dir.iterdir())

    model_name_map = {
        "BERT": "BERT (GinSign)",
        "gpt-3.5-turbo": "GPT-3.5 Turbo",
        "gpt-4o-mini": "GPT-4o Mini",
        "gpt-4.1-mini": "GPT-4.1 Mini",
    }

    print(f"\n{'Method':>20s}  ", end="")
    for d in DOMAINS:
        dn = DOMAIN_DISPLAY[d]
        print(f" {dn:^18s}", end="")
    print(f"  |  {'Average':^18s}")
    print(f"{'':>20s}  ", end="")
    for _ in DOMAINS:
        print(f" {'Acc':>5s} {'F1':>5s} {'EM':>5s}", end="")
    print(f"  |  {'Acc':>5s} {'F1':>5s} {'EM':>5s}")
    print("-" * 100)

    for mdir in model_dirs:
        mname = model_name_map.get(mdir.name, mdir.name)
        all_m = []
        print(f"{mname:>20s}  ", end="")
        for domain in DOMAINS:
            fpath = mdir / f"{domain}.jsonl"
            if not fpath.exists():
                print(f" {'—':>5s} {'—':>5s} {'—':>5s}", end="")
                continue
            m = compute_lifting_metrics(load_jsonl(fpath), gt_by_domain[domain])
            all_m.append(m)
            print(f" {m['accuracy']:>5.1f} {m['f1']:>5.1f} {m['exact_match']:>5.1f}", end="")

        if all_m:
            avg_acc = np.mean([m["accuracy"] for m in all_m])
            avg_f1 = np.mean([m["f1"] for m in all_m])
            avg_em = np.mean([m["exact_match"] for m in all_m])
            print(f"  |  {avg_acc:>5.1f} {avg_f1:>5.1f} {avg_em:>5.1f}")
        else:
            print()
    print(f"\n  Acc = token accuracy; F1 = token-level prop-detection F1; EM = exact match (full sequence)")
    print()


# ---------------------------------------------------------------------------
# Table 4: End-to-End Translation (LE and GLE)
# ---------------------------------------------------------------------------

def normalize_ltl(formula: str) -> str:
    return re.sub(r"\s+", " ", formula.strip())


def is_masked(prediction: str) -> bool:
    return bool(re.search(r"\bprop_\d+\b", prediction))


def ground_prediction(prediction: str, prop_dict: dict) -> str:
    result = prediction
    for pk in sorted(prop_dict, key=lambda k: int(k.split("_")[1]), reverse=True):
        pv = prop_dict[pk]
        if pv is None:
            continue
        action = pv["action_canon"]
        args = pv.get("args_canon", [])
        grounded = f"{action}({','.join(args)})" if args else action
        result = result.replace(pk, grounded)
    return result


def mask_prediction(prediction: str, prop_dict: dict) -> str:
    result = prediction
    for pk in sorted(prop_dict, key=lambda k: int(k.split("_")[1]), reverse=True):
        pv = prop_dict[pk]
        if pv is None:
            continue
        action = pv["action_canon"]
        args = pv.get("args_canon", [])
        grounded = f"{action}({','.join(args)})" if args else action
        result = result.replace(grounded, pk)
    return result


def check_lifting_match(pred_row: dict, gt_row: dict) -> bool:
    """Check if the lifting in pred_row matches ground truth."""
    pred_ids = pred_row.get("lifted_sentence_prop_ids", [])
    gt_ids = gt_row.get("lifted_sentence_prop_ids", [])
    if len(pred_ids) != len(gt_ids):
        return False
    return pred_ids == gt_ids


def eval_translation_dir(
    dirpath: Path,
    gt_by_domain: dict[str, dict[int, dict]],
    domains: list[str] | None = None,
    mode: str = "auto",
) -> dict[str, dict]:
    """Evaluate translation predictions.

    mode: 'grounded' - predictions are fully grounded
          'masked' - predictions use prop_X placeholders
          'auto' - detect from content
    """
    if domains is None:
        domains = DOMAINS
    results = {}

    for domain in domains:
        fpath = dirpath / f"{domain}.jsonl"
        if not fpath.exists():
            continue

        gt = gt_by_domain[domain]
        rows = load_jsonl(fpath)
        le_ok = gle_ok = total = 0

        for row in rows:
            prediction = row.get("prediction", "").strip()
            total += 1
            if not prediction:
                continue

            sid = row.get("id")
            gt_row = gt.get(sid, {}) if sid else {}

            tl = row.get("tl", gt_row.get("tl", []))
            masked_tl = row.get("masked_tl", gt_row.get("masked_tl", []))
            prop_dict = row.get("prop_dict", gt_row.get("prop_dict", {}))

            if not tl:
                continue

            gt_grounded = normalize_ltl(" ".join(tl))
            gt_masked = normalize_ltl(" ".join(masked_tl)) if masked_tl else ""
            pred_norm = normalize_ltl(prediction)

            if mode == "auto":
                pred_is_masked = is_masked(pred_norm)
            elif mode == "masked":
                pred_is_masked = True
            else:
                pred_is_masked = False

            if pred_is_masked:
                # LE: compare masked prediction against masked GT
                le_match = (pred_norm == gt_masked) if gt_masked else False

                # GLE: ground the prediction using GT prop_dict, then
                # compare against GT grounded TL.  This correctly captures
                # both formula-structure errors AND prop-label mismatches:
                # if the model swapped or mis-assigned prop labels, the
                # substituted formula will differ from the GT grounded TL.
                if prop_dict:
                    grounded_pred = normalize_ltl(ground_prediction(prediction, prop_dict))
                    gle_match = (grounded_pred == gt_grounded)
                else:
                    gle_match = False
            else:
                # LE: mask the prediction, compare against masked GT
                if prop_dict and gt_masked:
                    masked_pred = normalize_ltl(mask_prediction(prediction, prop_dict))
                    le_match = (masked_pred == gt_masked)
                else:
                    le_match = False

                # GLE: compare grounded prediction against grounded GT
                gle_match = (pred_norm == gt_grounded)

            if le_match:
                le_ok += 1
            if gle_match:
                gle_ok += 1

        if total > 0:
            results[domain] = {
                "le": 100.0 * le_ok / total,
                "gle": 100.0 * gle_ok / total,
                "n": total,
            }
    return results


def build_translation_table():
    print("=" * 108)
    print("TABLE 4: End-to-End Translation Evaluation  (LE % / GLE %)")
    print("=" * 108)

    gt_by_domain = {d: load_ground_truth(d) for d in DOMAINS}
    trans_dir = EVAL / "translation_eval"

    baselines = [
        # (display_name, path, mode, domains)
        # --- Prior-work prompting baselines (masked output) ---
        ("NL2LTL (GPT-3.5T)", trans_dir / "nl2ltl" / "raw_nl" / "nl2ltl_gpt-3.5-turbo", "masked", None),
        ("NL2LTL (GPT-4.1m)", trans_dir / "nl2ltl" / "raw_nl" / "nl2ltl_gpt-4.1-mini", "masked", None),
        ("NL2LTL (GPT-4om)", trans_dir / "nl2ltl" / "raw_nl" / "nl2ltl_gpt-4o-mini", "masked", None),
        ("NL2Spec (GPT-3.5T)", trans_dir / "nl2spec" / "raw_nl" / "nl2spec_gpt-3.5-turbo", "masked", None),
        ("NL2Spec (GPT-4.1m)", trans_dir / "nl2spec" / "raw_nl" / "nl2spec_gpt-4.1-mini", "masked", None),
        ("NL2Spec (GPT-4om)", trans_dir / "nl2spec" / "raw_nl" / "nl2spec_gpt-4o-mini", "masked", None),

        # --- NL2TL seq2seq: T5 + LLM lifting (masked output) ---
        ("NL2TL (GPT-3.5T lift)", trans_dir / "nl2tl" / "llm_masked_nl" / "gpt3_5_lifting", "auto", None),
        ("NL2TL (GPT-4.1m lift)", trans_dir / "nl2tl" / "llm_masked_nl" / "gpt4_1_lifting", "auto", None),
        ("NL2TL (GPT-4o lift)", trans_dir / "nl2tl" / "llm_masked_nl" / "gpt4o_lifting", "auto", None),
        ("NL2TL (GT lift)", trans_dir / "nl2tl" / "gt_lifting", "masked", None),

        # --- Direct LLM translation (grounded output) ---
        ("Direct (GPT-5)", trans_dir / "direct" / "gpt-5", "grounded", DOMAINS),
        ("Direct (Llama-3.1-8B)", trans_dir / "direct" / "llama-3_1-8b", "grounded", DOMAINS),

        # --- LLM lifted+translated (masked output) ---
        ("LLM Lift+Trans (GPT-5)", trans_dir / "llm_direct" / "gpt-5", "auto", DOMAINS),
        ("LLM Lift+Trans (Llama)", trans_dir / "llm_direct" / "llama-3_1-8b", "auto", DOMAINS),

        # --- GinSign: T5 end-to-end grounded translation ---
        ("GinSign (proposed)", trans_dir / "nl2tl" / "raw_nl", "grounded", None),
    ]

    print(f"\n{'Baseline':>28s}  ", end="")
    for d in DOMAINS:
        dn = DOMAIN_DISPLAY[d]
        print(f" {dn:^13s}", end="")
    print(f"     n")
    print(f"{'':>28s}  ", end="")
    for _ in DOMAINS:
        print(f" {'LE':>5s} {'GLE':>5s}  ", end="")
    print()
    print("-" * 108)

    for name, path, mode, domains in baselines:
        if not path.exists():
            continue
        results = eval_translation_dir(path, gt_by_domain, domains=domains, mode=mode)
        if not results:
            continue
        print(f"{name:>28s}  ", end="")
        for d in DOMAINS:
            if d in results:
                r = results[d]
                print(f" {r['le']:>5.1f} {r['gle']:>5.1f}  ", end="")
            else:
                print(f" {'—':>5s} {'—':>5s}  ", end="")
        n_vals = [results[d]["n"] for d in DOMAINS if d in results]
        print(f"  {sum(n_vals):>5d}" if n_vals else "")

    print(f"\n  LE = Logical Equivalence (operator structure); GLE = Grounded Logical Equivalence")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    build_grounding_table()
    build_lifting_table()
    build_translation_table()


if __name__ == "__main__":
    main()
