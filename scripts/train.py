#!/usr/bin/env python
"""Train one pointer grounder per signature.

Examples:
  python scripts/train.py \
      --signature data/signatures/warehouse.json \
      --train     data/VLTL-Bench/train/warehouse.jsonl \
      --dev       data/VLTL-Bench/test/warehouse.jsonl \
      --output    outputs/warehouse \
      --epochs 3 --batch-size 16 --lr 2e-5

  python scripts/train.py \
      --signature data/signatures/navi.json \
      --train     data/priorwork_cleaned/train/navi.jsonl \
      --dev       data/priorwork_cleaned/test/navi.jsonl \
      --cluster-map data/signatures/navi_clusters.json \
      --output    outputs/navi
"""

import argparse
from dataclasses import fields

from ginsign.config import TrainingConfig
from ginsign.training import train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--signature", required=True, dest="signature_path")
    p.add_argument("--train", required=True, dest="train_corpus")
    p.add_argument("--dev", required=True, dest="dev_corpus")
    p.add_argument("--output", required=True, dest="output_dir")
    p.add_argument("--cluster-map", default=None)
    p.add_argument("--model-name", default="bert-base-cased")
    p.add_argument("--epochs", type=int, default=3, dest="max_epochs")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lambda-arg", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--limit-train", type=int, default=None, dest="limit_train_examples")
    p.add_argument("--limit-dev", type=int, default=None, dest="limit_dev_examples")
    p.add_argument("--save-best-metric", default="joint_acc")

    args = p.parse_args()

    valid_keys = {f.name for f in fields(TrainingConfig)}
    kwargs = {k: v for k, v in vars(args).items() if k in valid_keys}
    config = TrainingConfig(**kwargs)
    train(config)


if __name__ == "__main__":
    main()
