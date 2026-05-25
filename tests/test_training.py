"""Smoke tests for training + eval pipeline.

Tier 2 (need cached bert-base-cased). Tier-1-style sanity checks live
in test_data_layer.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _have_bert():
    try:
        from transformers import BertTokenizerFast
        BertTokenizerFast.from_pretrained("bert-base-cased", local_files_only=True)
        return True
    except Exception:
        return False


needs_bert = pytest.mark.skipif(not _have_bert(), reason="bert-base-cased not in HF cache")


def _tiny_warehouse_setup(tmp_path):
    from ginsign.signature_io import PredicateDef, Signature, save_signature

    sig = Signature(
        name="warehouse",
        sorts=["item", "location"],
        predicates={
            "deliver": PredicateDef("deliver", [["item"], ["location"]]),
            "search":  PredicateDef("search",  [["item"]]),
            "idle":    PredicateDef("idle",    []),
        },
        constants={
            "item": ["backpack", "package"],
            "location": ["loading_dock", "shelf"],
        },
    )
    sig_path = tmp_path / "wh.json"
    save_signature(sig, sig_path)

    rows = [
        {"prop_dict": {"prop_1": {
            "action_canon": "search", "action_ref": "find",
            "args_canon": ["backpack"], "args_ref": ["bookbag"],
        }}},
        {"prop_dict": {"prop_1": {
            "action_canon": "search", "action_ref": "locate",
            "args_canon": ["package"], "args_ref": ["parcel"],
        }}},
        {"prop_dict": {"prop_1": {
            "action_canon": "deliver", "action_ref": "drop off",
            "args_canon": ["package", "shelf"],
            "args_ref": ["parcel", "shelf"],
        }}},
        {"prop_dict": {"prop_1": {
            "action_canon": "deliver", "action_ref": "bring",
            "args_canon": ["backpack", "loading_dock"],
            "args_ref": ["bag", "dock"],
        }}},
        {"prop_dict": {"prop_1": {
            "action_canon": "idle", "action_ref": "stand by",
            "args_canon": [], "args_ref": [],
        }}},
    ]
    corpus = tmp_path / "wh.jsonl"
    corpus.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return sig_path, corpus


@needs_bert
def test_evaluate_returns_well_formed_metrics(tmp_path):
    from ginsign.eval import evaluate
    from ginsign.grounding_dataset import GroundingDataset, make_collate_fn
    from ginsign.ptr_grounder import PointerJointGrounder
    from ginsign.signature_io import load_signature

    sig_path, corpus = _tiny_warehouse_setup(tmp_path)
    sig = load_signature(sig_path)
    ds = GroundingDataset(corpus, sig)
    loader = DataLoader(ds, batch_size=2, collate_fn=make_collate_fn(sig))

    torch.manual_seed(0)
    model = PointerJointGrounder(bert_name="bert-base-cased")
    metrics = evaluate(model, loader, sig, torch.device("cpu"))

    assert metrics["n"] == len(ds)
    assert 0.0 <= metrics["pred_acc"] <= 1.0
    assert 0.0 <= metrics["pred_f1_macro"] <= 1.0
    assert 0.0 <= metrics["arg_acc_full_tuple"] <= 1.0
    assert 0.0 <= metrics["joint_acc"] <= 1.0
    assert isinstance(metrics["arg_acc_per_slot"], dict)


@needs_bert
def test_train_smoke_runs_and_writes_artifacts(tmp_path):
    from ginsign.config import TrainingConfig
    from ginsign.training import train

    sig_path, corpus = _tiny_warehouse_setup(tmp_path)
    out_dir = tmp_path / "run"

    cfg = TrainingConfig(
        signature_path=str(sig_path),
        train_corpus=str(corpus),
        dev_corpus=str(corpus),
        output_dir=str(out_dir),
        max_steps=4,
        max_epochs=10,
        batch_size=2,
        eval_every=2,
        warmup_steps=1,
        lr=1e-4,
        device="cpu",
        seed=0,
    )
    final = train(cfg)
    assert "pred_acc" in final
    assert (out_dir / "config.json").exists()
    assert (out_dir / "metrics.jsonl").exists()
    assert (out_dir / "eval.json").exists()
    assert (out_dir / "best.pt").exists()
    assert (out_dir / "last.pt").exists()
    # metrics.jsonl must contain at least one eval record
    lines = (out_dir / "metrics.jsonl").read_text().splitlines()
    assert len(lines) >= 1
    rec = json.loads(lines[0])
    assert "step" in rec and "pred_acc" in rec


@needs_bert
def test_train_loss_decreases_on_tiny_overfit(tmp_path):
    """Mild overfit check: on a 5-example corpus with high LR + many steps,
    final dev (= train) metrics should beat initial random-init metrics."""
    from ginsign.config import TrainingConfig
    from ginsign.training import train

    sig_path, corpus = _tiny_warehouse_setup(tmp_path)
    out_dir = tmp_path / "overfit"
    cfg = TrainingConfig(
        signature_path=str(sig_path),
        train_corpus=str(corpus),
        dev_corpus=str(corpus),
        output_dir=str(out_dir),
        max_steps=40,
        max_epochs=100,
        batch_size=5,
        eval_every=20,
        warmup_steps=5,
        lr=5e-4,
        device="cpu",
        seed=1,
    )
    final = train(cfg)
    # On 5 examples with 40 steps the model should nail pred_acc; arg
    # may be tougher in 40 steps so we only assert pred_acc here.
    assert final["pred_acc"] >= 0.6, f"pred_acc didn't budge: {final}"


@needs_bert
def test_early_stopping_triggers(tmp_path):
    """With patience=1 and eval_every=2, training should stop well before
    max_steps if the metric plateaus after the first eval."""
    from ginsign.config import TrainingConfig
    from ginsign.training import train

    sig_path, corpus = _tiny_warehouse_setup(tmp_path)
    out_dir = tmp_path / "earlystop"
    cfg = TrainingConfig(
        signature_path=str(sig_path),
        train_corpus=str(corpus),
        dev_corpus=str(corpus),
        output_dir=str(out_dir),
        max_steps=100,
        max_epochs=100,
        batch_size=5,
        eval_every=2,
        patience=1,
        warmup_steps=1,
        lr=5e-4,
        device="cpu",
        seed=0,
    )
    final = train(cfg)
    lines = (out_dir / "metrics.jsonl").read_text().strip().splitlines()
    last_step = json.loads(lines[-1])["step"]
    # With patience=1, should stop much sooner than 100 steps
    assert last_step < 50, f"expected early stop but ran to step {last_step}"
