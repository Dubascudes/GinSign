"""Tests for ablation infrastructure: signature merge, multi-domain
dataset/collator, and signature transforms."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ginsign.signature_io import PredicateDef, Signature
from ginsign.signature_merge import merge_signatures
from ginsign.multi_domain_dataset import (
    DomainDataConfig,
    MultiDomainDataset,
    collate_multi_domain_batch,
    make_multi_domain_collate_fn,
)
from ginsign.signature_transforms import (
    collect_gold_constants,
    inject_distractors,
    partial_signature,
    shuffle_constants,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _wh_sig():
    return Signature(
        name="warehouse",
        sorts=["item", "location"],
        predicates={
            "deliver": PredicateDef("deliver", [["item"], ["location"]]),
            "search":  PredicateDef("search",  [["item"]]),
            "idle":    PredicateDef("idle",    []),
        },
        constants={"item": ["backpack", "package"], "location": ["loading_dock", "shelf"]},
    )

def _cw_sig():
    return Signature(
        name="cleanup_world",
        sorts=["type_at"],
        predicates={"at": PredicateDef("at", [["type_at"]])},
        constants={"type_at": ["blue_room", "green_room", "red_room", "yellow_room"]},
    )


# ---------------------------------------------------------------------------
# signature_merge
# ---------------------------------------------------------------------------

def test_merge_disjoint_signatures():
    merged = merge_signatures([_wh_sig(), _cw_sig()], name="wh_cw")
    assert merged.name == "wh_cw"
    assert set(merged.sorts) == {"item", "location", "type_at"}
    assert set(merged.predicate_names()) == {"at", "deliver", "idle", "search"}
    assert "blue_room" in merged.constants["type_at"]
    assert "backpack" in merged.constants["item"]


def test_merge_overlapping_predicates_same_arity():
    sig_a = Signature(
        name="a", sorts=["room"],
        predicates={"at": PredicateDef("at", [["room"]])},
        constants={"room": ["r1"]},
    )
    sig_b = Signature(
        name="b", sorts=["room"],
        predicates={"at": PredicateDef("at", [["room"]])},
        constants={"room": ["r2"]},
    )
    merged = merge_signatures([sig_a, sig_b])
    assert merged.predicates["at"].arity == 1
    assert set(merged.constants["room"]) == {"r1", "r2"}


def test_merge_conflicting_arity_disambiguates():
    sig_a = Signature(
        name="a", sorts=["x"],
        predicates={"go": PredicateDef("go", [["x"]])},
        constants={"x": ["a"]},
    )
    sig_b = Signature(
        name="b", sorts=["x", "y"],
        predicates={"go": PredicateDef("go", [["x"], ["y"]])},
        constants={"x": ["b"], "y": ["c"]},
    )
    merged = merge_signatures([sig_a, sig_b])
    assert "a/go" in merged.predicates
    assert "b/go" in merged.predicates
    assert merged.predicates["a/go"].arity == 1
    assert merged.predicates["b/go"].arity == 2


def test_merge_conflicting_arity_raises_when_disabled():
    sig_a = Signature(
        name="a", sorts=["x"],
        predicates={"go": PredicateDef("go", [["x"]])},
        constants={"x": ["a"]},
    )
    sig_b = Signature(
        name="b", sorts=["x", "y"],
        predicates={"go": PredicateDef("go", [["x"], ["y"]])},
        constants={"x": ["b"], "y": ["c"]},
    )
    with pytest.raises(ValueError, match="arity"):
        merge_signatures([sig_a, sig_b], disambiguate=False)


# ---------------------------------------------------------------------------
# multi_domain_dataset + collator
# ---------------------------------------------------------------------------

def _write_corpus(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_multi_domain_dataset_combines_domains(tmp_path):
    wh_corpus = tmp_path / "wh.jsonl"
    _write_corpus(wh_corpus, [
        {"prop_dict": {"prop_1": {
            "action_canon": "search", "action_ref": "find",
            "args_canon": ["backpack"], "args_ref": ["bag"],
        }}},
    ])
    cw_corpus = tmp_path / "cw.jsonl"
    _write_corpus(cw_corpus, [
        {"prop_dict": {"prop_1": {
            "action_canon": "at", "action_ref": "go to",
            "args_canon": ["blue_room"], "args_ref": ["blue room"],
        }}},
    ])
    ds = MultiDomainDataset([
        DomainDataConfig("warehouse", str(wh_corpus), _wh_sig()),
        DomainDataConfig("cleanup_world", str(cw_corpus), _cw_sig()),
    ])
    assert len(ds) == 2
    assert {ex.domain for ex in ds.examples} == {"warehouse", "cleanup_world"}


def test_multi_domain_collator_shapes(tmp_path):
    wh_corpus = tmp_path / "wh.jsonl"
    _write_corpus(wh_corpus, [
        {"prop_dict": {"prop_1": {
            "action_canon": "deliver", "action_ref": "drop off",
            "args_canon": ["package", "shelf"], "args_ref": ["parcel", "shelf"],
        }}},
    ])
    cw_corpus = tmp_path / "cw.jsonl"
    _write_corpus(cw_corpus, [
        {"prop_dict": {"prop_1": {
            "action_canon": "at", "action_ref": "go to",
            "args_canon": ["red_room"], "args_ref": ["red room"],
        }}},
    ])
    ds = MultiDomainDataset([
        DomainDataConfig("warehouse", str(wh_corpus), _wh_sig()),
        DomainDataConfig("cleanup_world", str(cw_corpus), _cw_sig()),
    ])
    merged = merge_signatures([_wh_sig(), _cw_sig()])
    union_preds = merged.predicate_names()

    batch = collate_multi_domain_batch(
        [ds[0], ds[1]], union_predicates=union_preds
    )
    assert len(batch["ap_texts"]) == 2
    # Both items should have the union predicate list
    assert batch["predicate_lists"][0] == union_preds
    assert batch["predicate_lists"][1] == union_preds
    # Max arity = 2 (deliver), so 2 slots
    assert len(batch["slot_candidate_lists"]) == 2
    assert batch["gold_arg_idx"].shape == (2, 2)
    # cleanup item has arity 1, so slot 1 is -1
    assert batch["gold_arg_idx"][1, 1].item() == -1


def test_multi_domain_collator_per_item_predicates(tmp_path):
    cw_corpus = tmp_path / "cw.jsonl"
    _write_corpus(cw_corpus, [
        {"prop_dict": {"prop_1": {
            "action_canon": "at", "action_ref": "go to",
            "args_canon": ["green_room"], "args_ref": ["green room"],
        }}},
    ])
    ds = MultiDomainDataset([
        DomainDataConfig("cleanup_world", str(cw_corpus), _cw_sig()),
    ])
    # Without union_predicates -> per-domain predicate list
    batch = collate_multi_domain_batch([ds[0]], union_predicates=None)
    assert batch["predicate_lists"][0] == ["at"]


# ---------------------------------------------------------------------------
# signature_transforms
# ---------------------------------------------------------------------------

def test_shuffle_preserves_set():
    sig = _wh_sig()
    shuffled = shuffle_constants(sig, seed=99)
    assert set(shuffled.constants["item"]) == set(sig.constants["item"])
    assert set(shuffled.constants["location"]) == set(sig.constants["location"])
    # Predicate structure unchanged
    assert shuffled.predicate_names() == sig.predicate_names()


def test_shuffle_different_seed_different_order():
    sig = Signature(
        name="big",
        sorts=["s"],
        predicates={"p": PredicateDef("p", [["s"]])},
        constants={"s": [f"c_{i}" for i in range(20)]},
    )
    a = shuffle_constants(sig, seed=1)
    b = shuffle_constants(sig, seed=2)
    assert a.constants["s"] != b.constants["s"]


def test_inject_distractors_adds_constants():
    sig = _cw_sig()
    donor = _wh_sig()
    augmented = inject_distractors(sig, [donor], n_per_sort=3, seed=0)
    # Original constants still present
    assert all(c in augmented.constants["type_at"] for c in sig.constants["type_at"])
    # Some new ones added (from donor's item or location sorts)
    assert len(augmented.constants["type_at"]) > len(sig.constants["type_at"])


def test_inject_distractors_no_duplicates():
    sig = _cw_sig()
    donor = _cw_sig()  # same signature as donor
    augmented = inject_distractors(sig, [donor], n_per_sort=5, seed=0)
    assert len(augmented.constants["type_at"]) == len(set(augmented.constants["type_at"]))


def test_partial_signature_removes_non_gold():
    sig = _wh_sig()
    protect = {"backpack", "shelf"}
    partial = partial_signature(sig, drop_frac=1.0, seed=0, protect_constants=protect)
    # Protected constants survive even with 100% drop
    assert "backpack" in partial.constants["item"]
    assert "shelf" in partial.constants["location"]
    # Unprotected are removed (package, loading_dock)
    assert "package" not in partial.constants["item"]
    assert "loading_dock" not in partial.constants["location"]


def test_partial_signature_with_zero_drop():
    sig = _wh_sig()
    partial = partial_signature(sig, drop_frac=0.0, seed=0)
    assert partial.constants == sig.constants


def test_collect_gold_constants(tmp_path):
    corpus = tmp_path / "test.jsonl"
    _write_corpus(corpus, [
        {"prop_dict": {"prop_1": {
            "action_canon": "search", "action_ref": "find",
            "args_canon": ["backpack"], "args_ref": ["bag"],
        }}},
        {"prop_dict": {"prop_1": {
            "action_canon": "deliver", "action_ref": "bring",
            "args_canon": ["package", "shelf"], "args_ref": ["pkg", "shelf"],
        }}},
    ])
    golds = collect_gold_constants(str(corpus))
    assert golds == {"backpack", "package", "shelf"}


# ---------------------------------------------------------------------------
# Integration: multi-domain collator -> compute_loss (needs BERT)
# ---------------------------------------------------------------------------

def _have_bert():
    try:
        from transformers import BertTokenizerFast
        BertTokenizerFast.from_pretrained("bert-base-cased", local_files_only=True)
        return True
    except Exception:
        return False


needs_bert = pytest.mark.skipif(not _have_bert(), reason="bert-base-cased not in HF cache")


@needs_bert
def test_multi_domain_compute_loss_finite(tmp_path):
    from ginsign.ptr_grounder import PointerJointGrounder

    wh_corpus = tmp_path / "wh.jsonl"
    _write_corpus(wh_corpus, [
        {"prop_dict": {"prop_1": {
            "action_canon": "search", "action_ref": "find",
            "args_canon": ["backpack"], "args_ref": ["bag"],
        }}},
    ])
    cw_corpus = tmp_path / "cw.jsonl"
    _write_corpus(cw_corpus, [
        {"prop_dict": {"prop_1": {
            "action_canon": "at", "action_ref": "go to",
            "args_canon": ["blue_room"], "args_ref": ["blue room"],
        }}},
    ])
    ds = MultiDomainDataset([
        DomainDataConfig("warehouse", str(wh_corpus), _wh_sig()),
        DomainDataConfig("cleanup_world", str(cw_corpus), _cw_sig()),
    ])
    merged = merge_signatures([_wh_sig(), _cw_sig()])
    batch = collate_multi_domain_batch(
        [ds[0], ds[1]], union_predicates=merged.predicate_names()
    )
    torch.manual_seed(0)
    model = PointerJointGrounder(bert_name="bert-base-cased")
    model.train()
    out = model.compute_loss(
        ap_texts=batch["ap_texts"],
        predicate_lists=batch["predicate_lists"],
        gold_pred_idx=batch["gold_pred_idx"],
        slot_candidate_lists=batch["slot_candidate_lists"],
        slot_type_masks=batch["slot_type_masks"],
        gold_arg_idx=batch["gold_arg_idx"],
    )
    assert torch.isfinite(out["loss"]), f"loss={out['loss']}"
    out["loss"].backward()
