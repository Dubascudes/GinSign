"""PyTorch Dataset adapter from a JSONL corpus + Signature to the dict
shape that PointerJointGrounder.compute_loss expects.

One JSONL row may contain several props (one per grounded atom in the
LTL formula). We flatten to one Dataset item per prop. The AP text we
feed the encoder is `action_ref + " " + " ".join(args_ref)`, which is
consistent across VLTL-Bench and priorwork even though their lifted-
sentence conventions differ.

Collation: build per-slot batched candidate lists by reading each
item's predicate arity from the signature; items whose arity is too
small for a given slot contribute a single padding candidate with
type_mask=False, and their gold_arg_idx for that slot is -1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .signature_io import Signature


@dataclass
class GroundingExample:
    ap_text: str
    gold_pred: str
    gold_args: List[str]


class GroundingDataset(Dataset):
    """Flat (one-prop-per-item) dataset over a grounding JSONL.

    Pass cluster_map / drop_set when the corpus uses raw predicate
    names that need clustering (navi). Examples whose predicates fall
    in drop_set are excluded.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        signature: Signature,
        cluster_map: Optional[Dict[str, str]] = None,
        drop_set: Optional[set] = None,
    ):
        self.path = Path(jsonl_path)
        self.signature = signature
        self.cluster_map = cluster_map or {}
        self.drop_set = drop_set or set()
        self.examples: List[GroundingExample] = []
        self._load()

    def _load(self) -> None:
        with self.path.open() as fh:
            for line in fh:
                d = json.loads(line)
                for _, info in (d.get("prop_dict") or {}).items():
                    raw_action = info.get("action_canon")
                    if raw_action is None or raw_action in self.drop_set:
                        continue
                    canonical = self.cluster_map.get(raw_action, raw_action)
                    if canonical not in self.signature.predicates:
                        continue
                    args = list(info.get("args_canon", []))
                    if len(args) != self.signature.arity_of(canonical):
                        continue
                    action_ref = info.get("action_ref", canonical)
                    args_ref = info.get("args_ref", args)
                    ap_text = (action_ref + " " + " ".join(args_ref)).strip()
                    if not ap_text:
                        continue
                    self.examples.append(
                        GroundingExample(
                            ap_text=ap_text,
                            gold_pred=canonical,
                            gold_args=args,
                        )
                    )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> GroundingExample:
        return self.examples[idx]


def collate_grounding_batch(
    examples: Sequence[GroundingExample],
    signature: Signature,
) -> dict:
    """Build the dict shape PointerJointGrounder.compute_loss consumes."""
    predicate_list = signature.predicate_names()
    pred_to_idx = {p: i for i, p in enumerate(predicate_list)}

    B = len(examples)
    max_arity = max((signature.arity_of(ex.gold_pred) for ex in examples), default=0)

    ap_texts = [ex.ap_text for ex in examples]
    predicate_lists = [list(predicate_list) for _ in examples]
    gold_pred_idx = torch.tensor([pred_to_idx[ex.gold_pred] for ex in examples], dtype=torch.long)

    slot_candidate_lists: List[List[List[str]]] = []  # [m][B][N_r]
    slot_type_masks: List[List[List[bool]]] = []
    gold_arg_idx_rows: List[List[int]] = [[] for _ in range(B)]

    for r in range(max_arity):
        per_batch_cands: List[List[str]] = []
        per_batch_mask: List[List[bool]] = []
        for b, ex in enumerate(examples):
            ar = signature.arity_of(ex.gold_pred)
            if r < ar:
                cands = signature.slot_constants(ex.gold_pred, r)
                if not cands:
                    raise ValueError(
                        f"signature {signature.name!r} predicate {ex.gold_pred!r} "
                        f"slot {r} has no constants"
                    )
                gold_c = ex.gold_args[r]
                try:
                    gold_idx = cands.index(gold_c)
                except ValueError:
                    raise ValueError(
                        f"gold arg {gold_c!r} not in candidate list for "
                        f"{ex.gold_pred!r} slot {r}: first 5 cands={cands[:5]}"
                    )
                per_batch_cands.append(cands)
                per_batch_mask.append([True] * len(cands))
                gold_arg_idx_rows[b].append(gold_idx)
            else:
                per_batch_cands.append(["[PAD]"])
                per_batch_mask.append([False])
                gold_arg_idx_rows[b].append(-1)
        slot_candidate_lists.append(per_batch_cands)
        slot_type_masks.append(per_batch_mask)

    gold_arg_idx = torch.tensor(
        gold_arg_idx_rows if max_arity > 0 else [[] for _ in range(B)],
        dtype=torch.long,
    ).reshape(B, max_arity)

    return {
        "ap_texts": ap_texts,
        "predicate_lists": predicate_lists,
        "gold_pred_idx": gold_pred_idx,
        "slot_candidate_lists": slot_candidate_lists,
        "slot_type_masks": slot_type_masks,
        "gold_arg_idx": gold_arg_idx,
    }


def make_collate_fn(signature: Signature):
    """Closure that DataLoader can use as collate_fn."""
    def _collate(batch):
        return collate_grounding_batch(batch, signature)
    return _collate
