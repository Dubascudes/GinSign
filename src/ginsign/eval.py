"""Eval metrics for the pointer grounder.

Per-batch we recompute the model's predictions on the gold predicate's
arg-slot candidate sets (teacher-forced on predicate, free on args).
This isolates the joint-arg-decoder accuracy from predicate errors,
matching what compute_loss optimizes.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from typing import Iterable

import torch
import torch.nn.functional as F

from .grounding_dataset import collate_grounding_batch
from .ptr_grounder import PointerJointGrounder
from .signature_io import Signature


@torch.no_grad()
def evaluate(
    model: PointerJointGrounder,
    loader: Iterable,
    signature: Signature,
    device: torch.device,
    use_bf16: bool = False,
) -> dict:
    """Return metrics over the loader. Loader must yield collator dicts."""
    model.eval()
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if (
        use_bf16 and device.type == "cuda"
    ) else nullcontext()

    pred_correct = 0
    pred_total = 0
    # For macro-F1 over predicates
    per_pred_tp = defaultdict(int)
    per_pred_fp = defaultdict(int)
    per_pred_fn = defaultdict(int)

    # Argument metrics: per-slot accuracy and full-tuple correctness
    slot_correct: dict = defaultdict(int)
    slot_total: dict = defaultdict(int)
    tuple_correct = 0
    tuple_total = 0   # examples with at least one arg slot
    joint_correct = 0
    joint_total = 0   # examples = all

    predicate_names = signature.predicate_names()

    for batch in loader:
        gold_pred_idx = batch["gold_pred_idx"].to(device)
        gold_arg_idx = batch["gold_arg_idx"].to(device)

        with amp_ctx:
            pred_logits, ap_repr, pred_cand_reprs, _ = model.ground_predicate(
                batch["ap_texts"], batch["predicate_lists"]
            )
            pred_pred_idx = pred_logits.argmax(dim=-1)

            B = pred_logits.size(0)
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

            per_example_tuple_correct = torch.ones(B, dtype=torch.bool, device=device)
            per_example_has_args = torch.zeros(B, dtype=torch.bool, device=device)
            for r in range(len(slot_cand_reprs_list)):
                cand_r = slot_cand_reprs_list[r]
                mask_r = slot_valid_masks_list[r]
                prev = gold_arg_reprs[:, :r, :] if r > 0 else None
                logits_r = model.arg_decoder.step(
                    ap_repr, pred_repr, prev, cand_r, mask_r
                )
                pred_arg = logits_r.argmax(dim=-1)
                gold_r = gold_arg_idx[:, r]
                present = gold_r >= 0
                per_example_has_args |= present
                if present.any():
                    correct = (pred_arg == gold_r) & present
                    slot_correct[r] += int(correct.sum().item())
                    slot_total[r] += int(present.sum().item())
                    per_example_tuple_correct &= correct | ~present

        # Predicate metrics
        for gold_i, pred_i in zip(gold_pred_idx.tolist(), pred_pred_idx.tolist()):
            pred_total += 1
            gold_name = predicate_names[gold_i]
            pred_name = predicate_names[pred_i]
            if gold_i == pred_i:
                pred_correct += 1
                per_pred_tp[gold_name] += 1
            else:
                per_pred_fp[pred_name] += 1
                per_pred_fn[gold_name] += 1

        for b in range(B):
            joint_total += 1
            if per_example_has_args[b]:
                tuple_total += 1
                if per_example_tuple_correct[b]:
                    tuple_correct += 1
            else:
                # No args -> tuple trivially correct
                tuple_correct_for_this = True
            example_joint = (gold_pred_idx[b] == pred_pred_idx[b]) and (
                per_example_tuple_correct[b].item()
            )
            if bool(example_joint):
                joint_correct += 1

    # Aggregate
    pred_acc = pred_correct / max(pred_total, 1)

    f1s = []
    for p in predicate_names:
        tp, fp, fn = per_pred_tp[p], per_pred_fp[p], per_pred_fn[p]
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * prec * rec / (prec + rec))
    pred_f1_macro = sum(f1s) / len(f1s) if f1s else 0.0

    arg_acc_per_slot = {r: slot_correct[r] / max(slot_total[r], 1) for r in slot_total}
    arg_acc_full_tuple = tuple_correct / max(tuple_total, 1)
    joint_acc = joint_correct / max(joint_total, 1)

    return {
        "n": pred_total,
        "pred_acc": pred_acc,
        "pred_f1_macro": pred_f1_macro,
        "arg_acc_per_slot": arg_acc_per_slot,
        "arg_acc_full_tuple": arg_acc_full_tuple,
        "joint_acc": joint_acc,
    }
