"""Configuration for per-domain grounder training runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    # Data
    signature_path: str
    train_corpus: str
    dev_corpus: str
    cluster_map: Optional[str] = None       # navi-only

    # Model
    model_name: str = "bert-base-cased"

    # Optimizer
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Training loop
    batch_size: int = 16
    eval_batch_size: int = -1               # -1 = 4x batch_size (no grads → can go bigger)
    max_epochs: int = 3
    max_steps: int = -1                     # -1 = no cap, use max_epochs
    eval_every: int = 500                   # in optimizer steps
    patience: int = 5                       # -1 = no early stopping
    seed: int = 0
    use_bf16: bool = False                  # autocast bf16 for train + eval
    compile_model: bool = False             # torch.compile the BERT encoder

    # Loss
    lambda_arg: float = 1.0

    # Output
    output_dir: str = "outputs/run"
    save_best_metric: str = "joint_acc"     # one of pred_acc / arg_acc_full_tuple / joint_acc

    # Logging
    wandb_project: Optional[str] = None     # None = disable wandb
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Misc
    num_workers: int = 0
    device: str = "auto"                    # "auto" / "cpu" / "cuda"
    limit_train_examples: Optional[int] = None   # smoke-test cap
    limit_dev_examples: Optional[int] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "TrainingConfig":
        return cls(**json.loads(s))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json() + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        return cls.from_json(Path(path).read_text())
