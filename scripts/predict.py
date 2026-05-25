#!/usr/bin/env python
"""Run inference on a corpus and save per-example predictions.

Outputs a JSONL file where each line contains:
  - ap_text: the natural-language AP span
  - gold_pred / pred_pred: gold and predicted predicate names
  - gold_args / pred_args: gold and predicted argument lists (constant names)
  - pred_correct / args_correct / joint_correct: per-example correctness flags
  - pred_logits: full predicate logit vector (for analysis)
  - arg_logits_per_slot: per-slot logit vectors over candidates

Example:
  python scripts/predict.py \
      --checkpoint outputs/warehouse/best.pt \
      --signature  data/signatures/warehouse.json \
      --corpus     data/VLTL-Bench/test/warehouse.jsonl \
      --out        results/warehouse_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ginsign.grounding_dataset import GroundingDataset, make_collate_fn
from ginsign.ptr_grounder import PointerJointGrounder
from ginsign.signature_builder import load_cluster_map
from ginsign.signature_io import load_signature


@torch.no_grad()
def predict_all(model, loader, signature, device):
    model.eval()
    predicate_names = signature.predicate_names()
    records = []

    for batch in loader:
        gold_pred_idx = batch["gold_pred_idx"].to(device)
        gold_arg_idx = batch["gold_arg_idx"].to(device)
        B = gold_pred_idx.size(0)

        pred_logits, ap_repr, pred_cand_reprs, _ = model.ground_predicate(
            batch["ap_texts"], batch["predicate_lists"]
        )
        pred_pred_idx = pred_logits.argmax(dim=-1)

        H = model.hidden_size
        pred_repr = pred_cand_reprs[torch.arange(B), gold_pred_idx]

        slot_cand_reprs_list = []
        slot_valid_masks_list = []
        gold_arg_reprs = torch.zeros(
            B, len(batch["slot_candidate_lists"]), H, device=device
        )
        for r, (cands_r, tm_r) in enumerate(
            zip(batch["slot_candidate_lists"], batch["slot_type_masks"])
        ):
            _, _, cand_reprs_r, spans_r = model.encode_stage(
                batch["ap_texts"], cands_r, tm_r, device=device
            )
            slot_cand_reprs_list.append(cand_reprs_r)
            slot_valid_masks_list.append(spans_r.effective_mask())
            gold_r = gold_arg_idx[:, r]
            present = gold_r >= 0
            if present.any():
                safe_idx = gold_r.clamp(min=0)
                gold_arg_reprs[:, r, :] = (
                    cand_reprs_r[torch.arange(B), safe_idx]
                    * present.unsqueeze(-1).float()
                )

        per_slot_pred_idx = []
        per_slot_logits = []
        for r in range(len(slot_cand_reprs_list)):
            cand_r = slot_cand_reprs_list[r]
            mask_r = slot_valid_masks_list[r]
            prev = gold_arg_reprs[:, :r, :] if r > 0 else None
            logits_r = model.arg_decoder.step(
                ap_repr, pred_repr, prev, cand_r, mask_r
            )
            per_slot_pred_idx.append(logits_r.argmax(dim=-1))
            per_slot_logits.append(logits_r)

        for b in range(B):
            gold_p = predicate_names[gold_pred_idx[b].item()]
            pred_p = predicate_names[pred_pred_idx[b].item()]
            arity = signature.arity_of(gold_p)

            gold_args = []
            pred_args = []
            arg_slot_details = []
            all_args_correct = True

            for r in range(arity):
                gold_r_idx = gold_arg_idx[b, r].item()
                pred_r_idx = per_slot_pred_idx[r][b].item()
                cands_r = batch["slot_candidate_lists"][r][b]

                gold_name = cands_r[gold_r_idx] if 0 <= gold_r_idx < len(cands_r) else "<MISSING>"
                pred_name = cands_r[pred_r_idx] if 0 <= pred_r_idx < len(cands_r) else "<OOB>"
                correct = (gold_r_idx == pred_r_idx)
                if not correct:
                    all_args_correct = False

                gold_args.append(gold_name)
                pred_args.append(pred_name)

                valid_mask = slot_valid_masks_list[r][b]
                n_valid = int(valid_mask.sum().item())
                logits_valid = per_slot_logits[r][b][valid_mask].tolist()

                arg_slot_details.append({
                    "slot": r,
                    "gold": gold_name,
                    "pred": pred_name,
                    "correct": correct,
                    "n_candidates": n_valid,
                    "logits": logits_valid,
                })

            pred_correct = (gold_p == pred_p)
            joint_correct = pred_correct and all_args_correct

            pred_logits_list = pred_logits[b].tolist()

            records.append({
                "ap_text": batch["ap_texts"][b],
                "gold_pred": gold_p,
                "pred_pred": pred_p,
                "pred_correct": pred_correct,
                "pred_logits": {name: pred_logits_list[i] for i, name in enumerate(predicate_names)},
                "gold_args": gold_args,
                "pred_args": pred_args,
                "args_correct": all_args_correct,
                "joint_correct": joint_correct,
                "arg_details": arg_slot_details,
            })

    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--signature", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cluster-map", default=None)
    p.add_argument("--model-name", default="bert-base-cased")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else args.device if args.device != "auto" else "cpu"
    )

    sig = load_signature(args.signature)
    raw_to_canon, drop_set = None, set()
    if args.cluster_map:
        raw_to_canon, _, drop_set = load_cluster_map(Path(args.cluster_map))

    ds = GroundingDataset(args.corpus, sig, cluster_map=raw_to_canon, drop_set=drop_set)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=make_collate_fn(sig))

    model = PointerJointGrounder(bert_name=args.model_name).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    records = predict_all(model, loader, sig, device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    n = len(records)
    n_pred_ok = sum(r["pred_correct"] for r in records)
    n_joint_ok = sum(r["joint_correct"] for r in records)
    print(f"Wrote {n} predictions to {out_path}")
    print(f"  pred_acc={n_pred_ok/n:.4f}  joint_acc={n_joint_ok/n:.4f}")


if __name__ == "__main__":
    main()
