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
    # Strip markdown fences (handle multiline)
    content = re.sub(r"^```(?:json)?\s*\n?", "", content)
    content = re.sub(r"\n?```\s*$", "", content)
    # Strip leading "prop_dict:" or "**prop_dict**:" prefix
    content = re.sub(r"^\*{0,2}prop_dict\*{0,2}\s*:\s*", "", content.strip())

    def _unwrap(d):
        if isinstance(d, dict) and "prop_dict" in d and isinstance(d["prop_dict"], dict):
            return d["prop_dict"]
        return d

    # Try direct parse
    try:
        d = json.loads(content.strip())
        if isinstance(d, dict):
            return _unwrap(d)
    except json.JSONDecodeError:
        pass

    # Fallback: find every top-level {...} block and try parsing each
    depth = 0
    start = -1
    for i, ch in enumerate(content):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    d = json.loads(content[start : i + 1])
                    if isinstance(d, dict):
                        return _unwrap(d)
                except json.JSONDecodeError:
                    pass
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


def _load_cluster_map(domain):
    """Load raw→canonical predicate mapping if it exists."""
    cm_path = ROOT / "data" / "signatures" / f"{domain}_clusters.json"
    if not cm_path.exists():
        return {}
    raw = json.loads(cm_path.read_text())
    mapping = {}
    for canon, raws in raw.get("canonical_to_raw", {}).items():
        for r in raws:
            mapping[r] = canon
    return mapping


def load_ginsign_grounding(domain):
    """Load pre-computed GinSign BERT pointer grounder predictions."""
    pred_path = ROOT / "results" / f"{domain}_predictions.jsonl"
    corpus = EVAL / "translation_eval" / "nl2tl" / "raw_nl" / f"{domain}.jsonl"
    if not pred_path.exists():
        return {}

    cluster_map = _load_cluster_map(domain)

    # Build (sample_id, prop_key, ap_text) from the corpus
    flat_keys = []
    for row in load_jsonl(corpus):
        for pk in sorted(row.get("prop_dict", {})):
            pv = row["prop_dict"][pk]
            if pv is None:
                continue
            action_ref = pv.get("action_ref", pv["action_canon"])
            args_ref = pv.get("args_ref", pv.get("args_canon", []))
            ap_text = (action_ref + " " + " ".join(args_ref)).strip()
            flat_keys.append((row["id"], pk, ap_text))

    # Index predictions by ap_text for matching
    pred_rows = load_jsonl(pred_path)
    pred_by_ap = {}
    for pr in pred_rows:
        pred_by_ap.setdefault(pr["ap_text"], []).append(pr)

    preds = {}
    for sid, pk, ap_text in flat_keys:
        candidates = pred_by_ap.get(ap_text, [])
        if not candidates:
            continue
        pr = candidates.pop(0)
        if sid not in preds:
            preds[sid] = {}
        preds[sid][pk] = atom_str(pr["pred_pred"], pr["pred_args"])

    return preds


# Also update evaluate_combination to apply cluster maps for GT comparison
_cluster_maps = {}


def _get_cluster_map(domain):
    if domain not in _cluster_maps:
        _cluster_maps[domain] = _load_cluster_map(domain)
    return _cluster_maps[domain]


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
            gt_action = gt_prop["action_canon"]
            cm = _get_cluster_map(domain)
            gt_action_canon = cm.get(gt_action, gt_action)
            if pred_p != gt_action_canon or clean_args(pred_a) != gt_args:
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
# Isolated Grounding Table
# ---------------------------------------------------------------------------

def compute_grounding_accuracy(groundings, gt_by_domain, domain):
    """Compute per-AP joint accuracy (predicate + all args correct)."""
    gt = gt_by_domain[domain]
    cm = _get_cluster_map(domain)
    correct = total = 0
    for sid, gmap in groundings.items():
        if sid not in gt:
            continue
        gt_props = gt[sid].get("prop_dict", {})
        for pk, gt_prop in gt_props.items():
            if gt_prop is None:
                continue
            total += 1
            pred_atom = gmap.get(pk)
            if pred_atom is None:
                continue
            pred_p, pred_a = parse_atom(pred_atom)
            gt_action = cm.get(gt_prop["action_canon"], gt_prop["action_canon"])
            gt_args = clean_args(gt_prop.get("args_canon", []))
            if pred_p == gt_action and clean_args(pred_a) == gt_args:
                correct += 1
    return 100.0 * correct / total if total else 0


def build_grounding_table(gt_by_domain, gcache):
    """Build isolated grounding evaluation table (per-AP joint accuracy)."""
    print()
    cw = 7
    total_w = 18 + len(ALL_DOMAINS) * (cw + 1) + 4
    print("=" * total_w)
    print("  Isolated Grounding Evaluation (per-AP joint accuracy %)")
    print("=" * total_w)

    hdr = f"{'Grounder':>18s}"
    for d in ALL_DOMAINS:
        hdr += f" {DOMAIN_SHORT[d]:>{cw}s}"
    print(f"\n{hdr}")
    print("-" * total_w)

    for gname in gcache:
        print(f"{gname:>18s}", end="")
        for domain in ALL_DOMAINS:
            grd = gcache[gname].get(domain, {})
            if not grd:
                print(f" {'—':>{cw}s}", end="")
                continue
            acc = compute_grounding_accuracy(grd, gt_by_domain, domain)
            print(f" {acc:>{cw}.1f}", end="")
        print()

    print()

    # LaTeX
    latex_path = ROOT / "paper" / "tables" / "grounding_table.tex"
    latex_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all data to find best per column
    all_data = {}
    for gname in gcache:
        all_data[gname] = {}
        for domain in ALL_DOMAINS:
            grd = gcache[gname].get(domain, {})
            if grd:
                all_data[gname][domain] = compute_grounding_accuracy(grd, gt_by_domain, domain)
            else:
                all_data[gname][domain] = None

    best_per_domain = {}
    for domain in ALL_DOMAINS:
        vals = [all_data[gn][domain] for gn in all_data if all_data[gn][domain] is not None]
        best_per_domain[domain] = max(vals) if vals else None

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Isolated grounding evaluation: per-AP joint accuracy (\%).")
    lines.append(r"  Each grounding method receives the ground-truth lifted APs and must predict")
    lines.append(r"  the correct predicate and all arguments from the system signature.}")
    lines.append(r"\label{tab:grounding}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{l|ccccccc}")
    lines.append(r"\toprule")
    lines.append(r"Grounder & TL & S\&R & WH & CW & CF & GLTL & Navi \\")
    lines.append(r"\midrule")

    for gname in gcache:
        label = "GinSign (ours)" if gname == "GinSign" else gname
        cells = []
        for domain in ALL_DOMAINS:
            v = all_data[gname][domain]
            if v is None:
                cells.append("---")
            elif best_per_domain[domain] is not None and v == best_per_domain[domain]:
                cells.append(r"\textbf{" + f"{v:.1f}" + "}")
            else:
                cells.append(f"{v:.1f}")
        lines.append(f"{label} & {' & '.join(cells)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(latex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  LaTeX grounding table written to {latex_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    gt_by_domain = {d: load_ground_truth(d) for d in ALL_DOMAINS}

    # Check if gpt-oss-120b results exist
    oss_trans = EVAL / "translation_eval" / "llm_direct" / "gpt-oss-120b"
    has_oss = oss_trans.exists() and any(oss_trans.glob("*.jsonl"))

    lifting_methods = [
        ("LLM Prompting", oss_trans if has_oss else EVAL / "translation_eval" / "llm_direct" / "gpt-5", ALL_DOMAINS if has_oss else VLTL_DOMAINS),
        ("NL2TL", EVAL / "translation_eval" / "nl2tl" / "llm_masked_nl" / "gpt4_1_lifting", ALL_DOMAINS),
        ("GraFT", EVAL / "translation_eval" / "nl2tl" / "gt_lifting", ALL_DOMAINS),
    ]

    grounding_specs = [
        ("GPT-4o",        lambda d: load_llm_grounding("gpt-4o", d)),
        ("GPT-oss-120B",  lambda d: load_llm_grounding("gpt-oss-120b", d)),
        ("Claude Sonnet", lambda d: load_llm_grounding("claude-sonnet", d)),
        ("Lang2LTL",      load_lang2ltl_grounding),
        ("GinSign",       load_ginsign_grounding),
    ]

    # Extended list for the isolated grounding table (includes VLTL-only models)
    all_grounding_specs = [
        ("GPT-3.5 Turbo", lambda d: load_llm_grounding("3_5_turbo", d)),
        ("GPT-4.1 Mini",  lambda d: load_llm_grounding("4_1_mini", d)),
        ("GPT-4o Mini",   lambda d: load_llm_grounding("4o_mini", d)),
        ("GPT-4o",        lambda d: load_llm_grounding("gpt-4o", d)),
        ("GPT-oss-120B",  lambda d: load_llm_grounding("gpt-oss-120b", d)),
        ("Claude Sonnet", lambda d: load_llm_grounding("claude-sonnet", d)),
        ("Lang2LTL",      load_lang2ltl_grounding),
        ("GinSign",       load_ginsign_grounding),
    ]

    print("Loading grounding predictions...", flush=True)
    gcache = {}
    gcache_all = {}
    loaded_names = set()
    for gname, gfn in all_grounding_specs:
        gcache_all[gname] = {}
        for domain in ALL_DOMAINS:
            try:
                gcache_all[gname][domain] = gfn(domain)
            except Exception as e:
                if gname not in loaded_names:
                    pass  # suppress duplicate skip messages
                gcache_all[gname][domain] = {}
        loaded_names.add(gname)
        print(f"  {gname}: loaded", flush=True)

    # Subset for the main cross-table
    for gname, _ in grounding_specs:
        gcache[gname] = gcache_all[gname]

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

        # Upper bound row = LE ceiling (perfect grounding always passes)
        print(f"{'':>18s}  {'Upper Bound':>15s}", end="")
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
    print("  Upper Bound row = LE ceiling (perfect grounding)")

    # --- LaTeX output ---
    latex_path = ROOT / "paper" / "tables" / "main_results.tex"
    latex_path.parent.mkdir(parents=True, exist_ok=True)

    n_grounding = len(grounding_specs)
    n_rows_per_block = n_grounding + 1  # +1 for upper bound

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Grounded Logical Equivalence (GLE, \%) across lifted translation frameworks and grounding methods.")
    lines.append(r"  GLE requires that the predicted formula is both structurally correct and that every atomic proposition")
    lines.append(r"  is correctly grounded in the system signature.")
    lines.append(r"  The \emph{Upper Bound} row shows the LE ceiling (perfect grounding) for each translation method.}")
    lines.append(r"\label{tab:main-results}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{ll|ccccccc}")
    lines.append(r"\toprule")
    lines.append(r"Lift + Translate & Grounding & TL & S\&R & WH & CW & CF & GLTL & Navi \\")
    lines.append(r"\midrule")

    for li, (lname, lpath, ldomains) in enumerate(lifting_methods):
        # Collect all GLE values for this block to find best per column
        block_data = {}
        for gname, _ in grounding_specs:
            block_data[gname] = {}
            for domain in ALL_DOMAINS:
                if domain not in ldomains:
                    block_data[gname][domain] = None
                    continue
                lifted = load_lifted_translations(lpath, domain)
                grd = gcache[gname].get(domain, {})
                if not lifted or not grd:
                    block_data[gname][domain] = None
                    continue
                r = evaluate_combination(lifted, grd, gt_by_domain, domain)
                block_data[gname][domain] = r["gle"]

        # Find best GLE per domain (among non-None values)
        best_per_domain = {}
        for domain in ALL_DOMAINS:
            vals = [block_data[gn][domain] for gn in block_data
                    if block_data[gn][domain] is not None]
            best_per_domain[domain] = max(vals) if vals else None

        # Grounding rows
        for gi, (gname, _) in enumerate(grounding_specs):
            lift_cell = r"\multirow{" + str(n_rows_per_block) + "}{*}{" + lname + "}" if gi == 0 else ""
            cells = []
            for domain in ALL_DOMAINS:
                v = block_data[gname][domain]
                if v is None:
                    cells.append("---")
                elif best_per_domain[domain] is not None and v == best_per_domain[domain]:
                    cells.append(r"\textbf{" + f"{v:.1f}" + "}")
                else:
                    cells.append(f"{v:.1f}")
            grounding_label = gname
            if gname == "GinSign":
                grounding_label = "GinSign (ours)"
            lines.append(f"{lift_cell} & {grounding_label} & {' & '.join(cells)} \\\\")

        # Upper bound row
        lines.append(r" \cdashline{2-9}")
        ub_cells = []
        for domain in ALL_DOMAINS:
            if domain not in ldomains:
                ub_cells.append(r"\textcolor{gray}{---}")
                continue
            lifted = load_lifted_translations(lpath, domain)
            gt = gt_by_domain[domain]
            le_ok = total = 0
            for sid, masked_pred in lifted.items():
                if sid not in gt:
                    continue
                gt_masked = normalize(" ".join(gt[sid].get("masked_tl", [])))
                total += 1
                if gt_masked and normalize(masked_pred) == gt_masked:
                    le_ok += 1
            gle = 100.0 * le_ok / total if total else 0
            ub_cells.append(r"\textcolor{gray}{" + f"{gle:.1f}" + "}")
        lines.append(r" & \textcolor{gray}{Upper Bound} & " + " & ".join(ub_cells) + r" \\")

        if li < len(lifting_methods) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    with open(latex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  LaTeX table written to {latex_path}")

    # --- Isolated grounding table ---
    build_grounding_table(gt_by_domain, gcache_all)


if __name__ == "__main__":
    main()
