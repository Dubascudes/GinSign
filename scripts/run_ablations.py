#!/usr/bin/env python
"""Generate and run the full ablation experiment matrix.

Axes:
  loo       Leave-one-out within VLTL-Bench (3) and priorwork (4)
  joint     Joint multi-domain: all-VLTL, all-priorwork
  cross     Cross-dataset transfer: VLTL→priorwork, priorwork→VLTL
  prefix    Prefix content ablations (shuffle, distractor, partial)

Usage:
  python scripts/run_ablations.py --axes loo joint cross prefix --device cuda
  python scripts/run_ablations.py --axes loo --dry-run   # just list experiments
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ginsign.config import TrainingConfig
from ginsign.eval import evaluate
from ginsign.grounding_dataset import GroundingDataset, make_collate_fn
from ginsign.multi_domain_dataset import (
    DomainDataConfig,
    MultiDomainDataset,
    make_multi_domain_collate_fn,
)
from ginsign.ptr_grounder import PointerJointGrounder
from ginsign.signature_builder import load_cluster_map
from ginsign.signature_io import Signature, load_signature
from ginsign.signature_merge import merge_signatures
from ginsign.signature_transforms import (
    collect_gold_constants,
    inject_distractors,
    partial_signature,
    shuffle_constants,
)
from ginsign.training import _resolve_device, _set_seed, train


# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

@dataclass
class DomainInfo:
    name: str
    family: str          # "vltl" or "priorwork"
    sig_path: str
    train_corpus: str
    test_corpus: str
    cluster_map: Optional[str] = None

    @property
    def signature(self) -> Signature:
        return load_signature(self.sig_path)


DATA = ROOT / "data"
SIG = DATA / "signatures"
VLTL = DATA / "VLTL-Bench"
PW = DATA / "priorwork_cleaned"
NAVI_CM = str(SIG / "navi_clusters.json")

DOMAINS: Dict[str, DomainInfo] = {
    "warehouse": DomainInfo("warehouse", "vltl",
        str(SIG / "warehouse.json"),
        str(VLTL / "train/warehouse.jsonl"), str(VLTL / "test/warehouse.jsonl")),
    "traffic_light": DomainInfo("traffic_light", "vltl",
        str(SIG / "traffic_light.json"),
        str(VLTL / "train/traffic_light.jsonl"), str(VLTL / "test/traffic_light.jsonl")),
    "search_and_rescue": DomainInfo("search_and_rescue", "vltl",
        str(SIG / "search_and_rescue.json"),
        str(VLTL / "train/search_and_rescue.jsonl"), str(VLTL / "test/search_and_rescue.jsonl")),
    "cleanup_world": DomainInfo("cleanup_world", "priorwork",
        str(SIG / "cleanup_world.json"),
        str(PW / "train/cleanup_world.jsonl"), str(PW / "test/cleanup_world.jsonl")),
    "GLTL": DomainInfo("GLTL", "priorwork",
        str(SIG / "GLTL.json"),
        str(PW / "train/GLTL.jsonl"), str(PW / "test/GLTL.jsonl")),
    "conformal": DomainInfo("conformal", "priorwork",
        str(SIG / "conformal.json"),
        str(PW / "train/conformal.jsonl"), str(PW / "test/conformal.jsonl")),
    "navi": DomainInfo("navi", "priorwork",
        str(SIG / "navi.json"),
        str(PW / "train/navi.jsonl"), str(PW / "test/navi.jsonl"),
        cluster_map=NAVI_CM),
}
VLTL_NAMES = [n for n, d in DOMAINS.items() if d.family == "vltl"]
PW_NAMES = [n for n, d in DOMAINS.items() if d.family == "priorwork"]

OUT_ROOT = ROOT / "outputs" / "ablations"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _domain_data_config(name: str) -> DomainDataConfig:
    d = DOMAINS[name]
    cm, ds = None, None
    if d.cluster_map:
        cm_raw, _, ds = load_cluster_map(Path(d.cluster_map))
        cm = cm_raw
    return DomainDataConfig(
        domain_name=name,
        jsonl_path=d.train_corpus,
        signature=d.signature,
        cluster_map=cm,
        drop_set=ds,
    )


def _eval_domain(model, domain_name, device, batch_size=16):
    d = DOMAINS[domain_name]
    sig = d.signature
    cm, ds = None, set()
    if d.cluster_map:
        cm, _, ds = load_cluster_map(Path(d.cluster_map))
    test_ds = GroundingDataset(d.test_corpus, sig, cluster_map=cm, drop_set=ds)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=make_collate_fn(sig))
    return evaluate(model, loader, sig, device)


def _eval_domain_transformed(model, domain_name, sig_transform, device, batch_size=16):
    d = DOMAINS[domain_name]
    sig = sig_transform(d.signature)
    cm, ds = None, set()
    if d.cluster_map:
        cm, _, ds = load_cluster_map(Path(d.cluster_map))
    test_ds = GroundingDataset(d.test_corpus, sig, cluster_map=cm, drop_set=ds)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=make_collate_fn(sig))
    return evaluate(model, loader, sig, device)


def _train_multi_domain(
    train_names: List[str],
    exp_name: str,
    device: torch.device,
    hp: dict,
) -> Path:
    out_dir = OUT_ROOT / exp_name
    if (out_dir / "best.pt").exists():
        print(f"[{exp_name}] checkpoint exists, skipping training")
        return out_dir

    configs = [_domain_data_config(n) for n in train_names]

    train_ds = MultiDomainDataset(configs)
    collate = make_multi_domain_collate_fn(union_predicates=None)

    # Build a dummy dev loader from the first training domain for eval during training.
    # Real per-domain eval happens after training.
    d0 = DOMAINS[train_names[0]]
    sig0 = d0.signature
    cm0, ds0 = None, set()
    if d0.cluster_map:
        cm0, _, ds0 = load_cluster_map(Path(d0.cluster_map))
    dev_ds = GroundingDataset(d0.test_corpus, sig0, cluster_map=cm0, drop_set=ds0)
    dev_collate = make_collate_fn(sig0)

    batch_size = hp.get("batch_size", 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=dev_collate, num_workers=0)

    import math, time, json as json_
    from ginsign.training import _linear_warmup_decay
    from torch.optim import AdamW

    _set_seed(hp.get("seed", 0))
    out_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = hp.get("wandb_project") is not None
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=hp["wandb_project"],
                entity=hp.get("wandb_entity"),
                name=exp_name,
                config={"train_domains": train_names, **hp},
                dir=str(out_dir),
            )
        except Exception:
            use_wandb = False

    model = PointerJointGrounder(bert_name=hp.get("model_name", "bert-base-cased")).to(device)
    optimizer = AdamW(model.parameters(), lr=hp.get("lr", 2e-5),
                      weight_decay=hp.get("weight_decay", 0.01))

    max_epochs = hp.get("max_epochs", 5)
    steps_per_epoch = max(1, math.ceil(len(train_ds) / batch_size))
    total_steps = steps_per_epoch * max_epochs
    eval_every = hp.get("eval_every", 200)
    patience = hp.get("patience", 5)
    warmup = hp.get("warmup_steps", 100)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _linear_warmup_decay(s, warmup, total_steps))

    step = 0
    best_metric = -float("inf")
    patience_counter = 0
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("")
    done = False

    print(f"[{exp_name}] |train|={len(train_ds)} domains={train_names} steps={total_steps}")
    t0 = time.time()
    for epoch in range(max_epochs):
        if done:
            break
        model.train()
        for batch in train_loader:
            out = model.compute_loss(
                ap_texts=batch["ap_texts"],
                predicate_lists=batch["predicate_lists"],
                gold_pred_idx=batch["gold_pred_idx"],
                slot_candidate_lists=batch["slot_candidate_lists"],
                slot_type_masks=batch["slot_type_masks"],
                gold_arg_idx=batch["gold_arg_idx"],
            )
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.get("max_grad_norm", 1.0))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % 50 == 0 or step == 1:
                elapsed = time.time() - t0
                print(f"  step {step}/{total_steps} loss={out['loss'].item():.4f} "
                      f"{step/elapsed:.1f} s/s")
                if use_wandb:
                    import wandb
                    wandb.log({"train/loss": out["loss"].item(),
                               "train/pred_loss": out["pred_loss"].item(),
                               "train/arg_loss": out["arg_loss"].item()}, step=step)

            if step % eval_every == 0 or step == total_steps:
                metrics = evaluate(model, dev_loader, sig0, device)
                metrics["step"] = step
                with metrics_path.open("a") as f:
                    f.write(json_.dumps(metrics) + "\n")
                m = metrics.get("joint_acc", 0)
                improved = m > best_metric
                print(f"  eval@{step}: joint={m:.3f} (saving={'yes' if improved else 'no'})")
                if use_wandb:
                    import wandb
                    wandb.log({"eval/pred_acc": metrics["pred_acc"],
                               "eval/arg_acc_full_tuple": metrics["arg_acc_full_tuple"],
                               "eval/joint_acc": m}, step=step)
                if improved:
                    best_metric = m
                    patience_counter = 0
                    torch.save({"state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                                "metrics": metrics}, out_dir / "best.pt")
                elif patience >= 0:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  early stopping at step {step}")
                        done = True
                torch.save({"state_dict": model.state_dict(), "metrics": metrics},
                           out_dir / "last.pt")
                model.train()

            if step >= total_steps or done:
                done = True
                break

    print(f"[{exp_name}] done. best joint_acc={best_metric:.3f}")
    if use_wandb:
        import wandb
        wandb.finish()
    return out_dir


def _load_checkpoint(ckpt_path, device):
    model = PointerJointGrounder(bert_name="bert-base-cased").to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Experiment generators
# ---------------------------------------------------------------------------

def run_loo(device, hp, batch_size=16):
    """Leave-one-out within each family."""
    for family, names in [("vltl", VLTL_NAMES), ("priorwork", PW_NAMES)]:
        for holdout in names:
            train_names = [n for n in names if n != holdout]
            exp_name = f"loo_{family}__held_{holdout}"
            out_dir = _train_multi_domain(train_names, exp_name, device, hp)
            model = _load_checkpoint(out_dir / "best.pt", device)
            for eval_dom in names:
                metrics = _eval_domain(model, eval_dom, device, batch_size)
                eval_path = out_dir / f"eval_{eval_dom}.json"
                eval_path.write_text(json.dumps(metrics, indent=2) + "\n")
                tag = "HELD" if eval_dom == holdout else "seen"
                print(f"  [{exp_name}] eval {eval_dom} ({tag}): joint={metrics['joint_acc']:.3f}")


def run_joint(device, hp, batch_size=16):
    """Joint training on all domains within each family."""
    for family, names in [("vltl", VLTL_NAMES), ("priorwork", PW_NAMES)]:
        exp_name = f"joint_{family}"
        out_dir = _train_multi_domain(names, exp_name, device, hp)
        model = _load_checkpoint(out_dir / "best.pt", device)
        for eval_dom in names:
            metrics = _eval_domain(model, eval_dom, device, batch_size)
            eval_path = out_dir / f"eval_{eval_dom}.json"
            eval_path.write_text(json.dumps(metrics, indent=2) + "\n")
            print(f"  [{exp_name}] eval {eval_dom}: joint={metrics['joint_acc']:.3f}")


def run_cross(device, hp, batch_size=16):
    """Cross-dataset transfer: train on one family, eval on the other."""
    for train_family, eval_family, train_names, eval_names in [
        ("vltl", "priorwork", VLTL_NAMES, PW_NAMES),
        ("priorwork", "vltl", PW_NAMES, VLTL_NAMES),
    ]:
        exp_name = f"cross_{train_family}_to_{eval_family}"
        out_dir = _train_multi_domain(train_names, exp_name, device, hp)
        model = _load_checkpoint(out_dir / "best.pt", device)
        for eval_dom in eval_names:
            metrics = _eval_domain(model, eval_dom, device, batch_size)
            eval_path = out_dir / f"eval_{eval_dom}.json"
            eval_path.write_text(json.dumps(metrics, indent=2) + "\n")
            print(f"  [{exp_name}] eval {eval_dom}: joint={metrics['joint_acc']:.3f}")


def run_prefix(device, batch_size=16):
    """Prefix content ablations on per-domain checkpoints."""
    all_sigs = [DOMAINS[n].signature for n in DOMAINS]
    for domain_name in DOMAINS:
        ckpt_path = ROOT / "outputs" / domain_name / "best.pt"
        if not ckpt_path.exists():
            print(f"[prefix] skipping {domain_name}: no per-domain checkpoint")
            continue
        model = _load_checkpoint(ckpt_path, device)
        d = DOMAINS[domain_name]
        sig = d.signature
        donor_sigs = [s for s in all_sigs if s.name != sig.name]
        gold_consts = collect_gold_constants(d.test_corpus,
            cluster_map=load_cluster_map(Path(d.cluster_map))[0] if d.cluster_map else None)

        transforms = {
            "shuffle": lambda s: shuffle_constants(s, seed=42),
            "distractor_10": lambda s: inject_distractors(s, donor_sigs, n_per_sort=10, seed=42),
            "distractor_50": lambda s: inject_distractors(s, donor_sigs, n_per_sort=50, seed=42),
            "partial_30": lambda s: partial_signature(s, drop_frac=0.3, seed=42,
                                                      protect_constants=gold_consts),
            "partial_50": lambda s: partial_signature(s, drop_frac=0.5, seed=42,
                                                      protect_constants=gold_consts),
        }
        for t_name, t_fn in transforms.items():
            exp_name = f"prefix_{t_name}__{domain_name}"
            out_dir = OUT_ROOT / exp_name
            out_dir.mkdir(parents=True, exist_ok=True)
            metrics = _eval_domain_transformed(model, domain_name, t_fn, device, batch_size)
            eval_path = out_dir / f"eval_{domain_name}.json"
            eval_path.write_text(json.dumps(metrics, indent=2) + "\n")
            print(f"  [{exp_name}] joint={metrics['joint_acc']:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--axes", nargs="+",
                   choices=["loo", "joint", "cross", "prefix", "all"],
                   default=["all"])
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    device = _resolve_device(args.device)
    hp = {
        "lr": args.lr, "max_epochs": args.epochs, "eval_every": args.eval_every,
        "patience": args.patience, "warmup_steps": args.warmup_steps,
        "seed": args.seed, "batch_size": args.batch_size,
        "wandb_project": args.wandb_project, "wandb_entity": args.wandb_entity,
    }
    axes = set(args.axes)
    if "all" in axes:
        axes = {"loo", "joint", "cross", "prefix"}

    if args.dry_run:
        print("Experiment axes:", sorted(axes))
        print("Hyperparams:", hp)
        print("Device:", device)
        return

    if "loo" in axes:
        print("\n===== Leave-One-Out =====")
        run_loo(device, hp, args.batch_size)
    if "joint" in axes:
        print("\n===== Joint Multi-Domain =====")
        run_joint(device, hp, args.batch_size)
    if "cross" in axes:
        print("\n===== Cross-Dataset Transfer =====")
        run_cross(device, hp, args.batch_size)
    if "prefix" in axes:
        print("\n===== Prefix Ablations =====")
        run_prefix(device, args.batch_size)

    print("\nAll ablations complete. Results in outputs/ablations/")


if __name__ == "__main__":
    main()
