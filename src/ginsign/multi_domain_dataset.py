"""Multi-domain dataset + collator for joint/LOO/cross-dataset training.

Each example is tagged with its own domain Signature so the collator
can build per-item predicate lists and slot candidate lists. The
model's compute_loss already accepts per-item ragged structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .grounding_dataset import GroundingDataset, GroundingExample
from .signature_io import Signature
from .signature_merge import merge_signatures


@dataclass
class DomainDataConfig:
    domain_name: str
    jsonl_path: str
    signature: Signature
    cluster_map: Optional[Dict[str, str]] = None
    drop_set: Optional[set] = None


@dataclass
class DomainGroundingExample:
    ap_text: str
    gold_pred: str
    gold_args: List[str]
    domain: str
    signature: Signature


class MultiDomainDataset(Dataset):
    """Concatenation of per-domain GroundingDatasets with domain tags."""

    def __init__(self, domain_configs: Sequence[DomainDataConfig]):
        self.examples: List[DomainGroundingExample] = []
        self.domain_names: List[str] = []
        for cfg in domain_configs:
            ds = GroundingDataset(
                cfg.jsonl_path, cfg.signature,
                cluster_map=cfg.cluster_map,
                drop_set=cfg.drop_set,
            )
            self.domain_names.append(cfg.domain_name)
            for ex in ds.examples:
                self.examples.append(DomainGroundingExample(
                    ap_text=ex.ap_text,
                    gold_pred=ex.gold_pred,
                    gold_args=ex.gold_args,
                    domain=cfg.domain_name,
                    signature=cfg.signature,
                ))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> DomainGroundingExample:
        return self.examples[idx]


def collate_multi_domain_batch(
    examples: Sequence[DomainGroundingExample],
    union_predicates: Optional[List[str]] = None,
) -> dict:
    """Build the dict shape PointerJointGrounder.compute_loss consumes.

    Each example carries its own Signature, so predicate lists and slot
    candidate lists are built per-item.

    union_predicates: if provided, every item's predicate list is padded/
    extended to this union set (for training, so the model sees all
    predicates across domains). If None, each item uses its own domain's
    predicate list.
    """
    B = len(examples)
    ap_texts = [ex.ap_text for ex in examples]

    if union_predicates is not None:
        pred_list_per_item = [list(union_predicates) for _ in examples]
    else:
        pred_list_per_item = [ex.signature.predicate_names() for ex in examples]

    gold_pred_idx_list = []
    for b, ex in enumerate(examples):
        preds = pred_list_per_item[b]
        try:
            idx = preds.index(ex.gold_pred)
        except ValueError:
            raise ValueError(
                f"gold pred {ex.gold_pred!r} not in predicate list for "
                f"item {b} (domain={ex.domain}): {preds[:10]}"
            )
        gold_pred_idx_list.append(idx)
    gold_pred_idx = torch.tensor(gold_pred_idx_list, dtype=torch.long)

    max_arity = max(
        (ex.signature.arity_of(ex.gold_pred) for ex in examples), default=0
    )

    slot_candidate_lists: List[List[List[str]]] = []
    slot_type_masks: List[List[List[bool]]] = []
    gold_arg_idx_rows: List[List[int]] = [[] for _ in range(B)]

    for r in range(max_arity):
        per_batch_cands: List[List[str]] = []
        per_batch_mask: List[List[bool]] = []
        for b, ex in enumerate(examples):
            ar = ex.signature.arity_of(ex.gold_pred)
            if r < ar:
                cands = ex.signature.slot_constants(ex.gold_pred, r)
                if not cands:
                    raise ValueError(
                        f"domain {ex.domain!r} pred {ex.gold_pred!r} slot {r} "
                        f"has no constants"
                    )
                gold_c = ex.gold_args[r]
                try:
                    gold_idx = cands.index(gold_c)
                except ValueError:
                    raise ValueError(
                        f"gold arg {gold_c!r} not in candidates for "
                        f"{ex.gold_pred!r} slot {r} (domain={ex.domain})"
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
        "predicate_lists": pred_list_per_item,
        "gold_pred_idx": gold_pred_idx,
        "slot_candidate_lists": slot_candidate_lists,
        "slot_type_masks": slot_type_masks,
        "gold_arg_idx": gold_arg_idx,
    }


def make_multi_domain_collate_fn(
    union_predicates: Optional[List[str]] = None,
):
    def _collate(batch):
        return collate_multi_domain_batch(batch, union_predicates=union_predicates)
    return _collate
