#!/usr/bin/env python
"""Evaluate a saved checkpoint on a corpus.

Example:
  python scripts/evaluate.py \
      --checkpoint outputs/warehouse/best.pt \
      --signature  data/signatures/warehouse.json \
      --corpus     data/VLTL-Bench/test/warehouse.jsonl
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ginsign.eval import evaluate
from ginsign.grounding_dataset import GroundingDataset, make_collate_fn
from ginsign.ptr_grounder import PointerJointGrounder
from ginsign.signature_builder import load_cluster_map
from ginsign.signature_io import load_signature


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--signature", required=True)
    p.add_argument("--corpus", required=True)
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

    ds = GroundingDataset(
        args.corpus, sig, cluster_map=raw_to_canon, drop_set=drop_set
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=make_collate_fn(sig),
    )

    model = PointerJointGrounder(bert_name=args.model_name).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    metrics = evaluate(model, loader, sig, device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
