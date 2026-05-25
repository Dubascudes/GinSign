"""Custom training loop for PointerJointGrounder.

One model per signature: per-domain training. Outputs land in
config.output_dir/{best.pt, last.pt, config.json, metrics.jsonl, eval.json}.
"""

from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .eval import evaluate
from .grounding_dataset import GroundingDataset, make_collate_fn
from .ptr_grounder import PointerJointGrounder
from .signature_builder import load_cluster_map

try:
    import wandb as _wandb
except ImportError:
    _wandb = None
from .signature_io import Signature, load_signature


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _linear_warmup_decay(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 1.0 - progress)


def _build_dataset(
    corpus_path: str,
    signature: Signature,
    cluster_map_path: Optional[str],
    limit: Optional[int],
) -> GroundingDataset:
    raw_to_canon = None
    drop_set: set = set()
    if cluster_map_path:
        raw_to_canon, _, drop_set = load_cluster_map(Path(cluster_map_path))
    ds = GroundingDataset(
        corpus_path,
        signature,
        cluster_map=raw_to_canon,
        drop_set=drop_set,
    )
    if limit is not None and limit > 0 and limit < len(ds):
        # Deterministic subset
        ds.examples = ds.examples[:limit]
    return ds


def train(config: TrainingConfig) -> dict:
    """Run a per-domain training job and return the final dev metrics."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.save(out_dir / "config.json")
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("")  # truncate

    _set_seed(config.seed)
    device = _resolve_device(config.device)

    use_wandb = _wandb is not None and config.wandb_project is not None
    if use_wandb:
        _wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name or Path(config.output_dir).name,
            config=json.loads(config.to_json()),
            dir=str(out_dir),
        )

    signature = load_signature(config.signature_path)
    collate = make_collate_fn(signature)

    train_ds = _build_dataset(
        config.train_corpus, signature, config.cluster_map, config.limit_train_examples
    )
    dev_ds = _build_dataset(
        config.dev_corpus, signature, config.cluster_map, config.limit_dev_examples
    )
    print(
        f"[train] sig={signature.name}  "
        f"|train|={len(train_ds)}  |dev|={len(dev_ds)}  "
        f"device={device}"
    )

    eval_bs = config.eval_batch_size if config.eval_batch_size > 0 else config.batch_size * 4
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate,
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate,
    )

    model = PointerJointGrounder(bert_name=config.model_name).to(device)
    if config.compile_model and hasattr(torch, "compile"):
        print("[train] compiling BERT encoder with torch.compile")
        model.bert = torch.compile(model.bert)
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    steps_per_epoch = max(1, math.ceil(len(train_ds) / config.batch_size))
    total_steps = (
        config.max_steps
        if config.max_steps > 0
        else steps_per_epoch * config.max_epochs
    )
    print(f"[train] steps_per_epoch={steps_per_epoch}  total_steps={total_steps}")

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _linear_warmup_decay(step, config.warmup_steps, total_steps),
    )

    step = 0
    best_metric = -float("inf")
    best_state: Optional[dict] = None
    patience_counter = 0

    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if (
        config.use_bf16 and device.type == "cuda"
    ) else nullcontext()
    if config.use_bf16:
        print(f"[train] using bf16 autocast")

    t0 = time.time()
    done = False
    for epoch in range(config.max_epochs):
        if done:
            break
        model.train()
        for batch in train_loader:
            with amp_ctx:
                out = model.compute_loss(
                    ap_texts=batch["ap_texts"],
                    predicate_lists=batch["predicate_lists"],
                    gold_pred_idx=batch["gold_pred_idx"],
                    slot_candidate_lists=batch["slot_candidate_lists"],
                    slot_type_masks=batch["slot_type_masks"],
                    gold_arg_idx=batch["gold_arg_idx"],
                    lambda_arg=config.lambda_arg,
                )
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % 50 == 0 or step == 1:
                elapsed = time.time() - t0
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  step {step}/{total_steps}  "
                    f"loss={loss.item():.4f}  "
                    f"pred={out['pred_loss'].item():.4f}  "
                    f"arg={out['arg_loss'].item():.4f}  "
                    f"lr={lr_now:.2e}  "
                    f"{step/elapsed:.1f} step/s"
                )
                if use_wandb:
                    _wandb.log({
                        "train/loss": loss.item(),
                        "train/pred_loss": out["pred_loss"].item(),
                        "train/arg_loss": out["arg_loss"].item(),
                        "train/lr": lr_now,
                        "train/step_per_sec": step / elapsed,
                    }, step=step)

            if step % config.eval_every == 0 or step == total_steps:
                metrics = evaluate(model, dev_loader, signature, device, use_bf16=config.use_bf16)
                metrics["step"] = step
                metrics["epoch"] = epoch
                with metrics_path.open("a") as fh:
                    fh.write(json.dumps(metrics) + "\n")
                m = metrics[config.save_best_metric]
                print(
                    f"  eval@{step}: pred_acc={metrics['pred_acc']:.3f}  "
                    f"arg_full={metrics['arg_acc_full_tuple']:.3f}  "
                    f"joint={metrics['joint_acc']:.3f}  "
                    f"(saving={'yes' if m > best_metric else 'no'})"
                )
                if use_wandb:
                    _wandb.log({
                        "eval/pred_acc": metrics["pred_acc"],
                        "eval/pred_f1_macro": metrics["pred_f1_macro"],
                        "eval/arg_acc_full_tuple": metrics["arg_acc_full_tuple"],
                        "eval/joint_acc": metrics["joint_acc"],
                    }, step=step)
                if m > best_metric:
                    best_metric = m
                    patience_counter = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    torch.save({"state_dict": best_state, "metrics": metrics,
                                "config": json.loads(config.to_json())},
                               out_dir / "best.pt")
                elif config.patience >= 0:
                    patience_counter += 1
                    if patience_counter >= config.patience:
                        print(
                            f"  early stopping at step {step} "
                            f"(no improvement for {config.patience} evals)"
                        )
                        done = True
                torch.save({"state_dict": model.state_dict(), "metrics": metrics,
                            "config": json.loads(config.to_json())},
                           out_dir / "last.pt")
                model.train()

            if step >= total_steps or done:
                done = True
                break

    # Final eval (also writes a clean summary)
    final = evaluate(model, dev_loader, signature, device, use_bf16=config.use_bf16)
    final["step"] = step
    (out_dir / "eval.json").write_text(json.dumps(final, indent=2) + "\n")
    print(f"[train] done. best {config.save_best_metric}={best_metric:.3f}")
    if use_wandb:
        _wandb.log({"final/" + k: v for k, v in final.items() if isinstance(v, (int, float))})
        _wandb.finish()
    return final
