"""Tests for the data layer: signature I/O, signature builder, priorwork
cleaner, navi cluster map, and GroundingDataset integration with the
pointer grounder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ginsign.signature_io import (
    PredicateDef,
    Signature,
    load_signature,
    save_signature,
    signature_from_dict,
    signature_to_dict,
)
from ginsign.signature_builder import (
    build_priorwork_signature,
    build_vltl_signature,
    load_cluster_map,
    verify_corpus_coverage,
)
from ginsign.priorwork_cleaner import _raw_prop_is_bad, clean_file
from ginsign.grounding_dataset import (
    GroundingDataset,
    collate_grounding_batch,
    make_collate_fn,
)


REPO_DATA = ROOT / "data"
SIGNATURES_DIR = REPO_DATA / "signatures"
CLEAN_DIR = REPO_DATA / "priorwork_cleaned"


# ---------------------------------------------------------------------------
# signature_io
# ---------------------------------------------------------------------------


def _toy_signature():
    return Signature(
        name="toy",
        sorts=["item", "loc"],
        predicates={
            "deliver": PredicateDef("deliver", [["item"], ["loc"]]),
            "see": PredicateDef("see", [["item", "loc"]]),  # union slot
            "idle": PredicateDef("idle", []),
        },
        constants={"item": ["apple", "ball"], "loc": ["kitchen"]},
        provenance="test",
    )


def test_signature_roundtrip(tmp_path):
    sig = _toy_signature()
    path = tmp_path / "toy.json"
    save_signature(sig, path)
    loaded = load_signature(path)
    assert loaded.name == sig.name
    assert loaded.sorts == sig.sorts
    assert loaded.predicate_names() == sig.predicate_names()
    assert loaded.predicates["deliver"].arg_sorts == [["item"], ["loc"]]
    assert loaded.predicates["see"].arg_sorts == [["item", "loc"]]
    assert loaded.constants == sig.constants


def test_signature_rejects_unknown_sort_in_predicate():
    with pytest.raises(ValueError, match="unknown sort"):
        Signature(
            name="bad",
            sorts=["item"],
            predicates={"p": PredicateDef("p", [["mystery"]])},
            constants={"item": ["x"]},
        )


def test_signature_rejects_constants_in_two_sorts():
    with pytest.raises(ValueError, match="multiple sorts"):
        Signature(
            name="bad",
            sorts=["a", "b"],
            predicates={},
            constants={"a": ["shared"], "b": ["shared"]},
        )


def test_slot_constants_union():
    sig = _toy_signature()
    assert sig.slot_constants("see", 0) == ["apple", "ball", "kitchen"]
    assert sig.slot_constants("deliver", 0) == ["apple", "ball"]
    assert sig.slot_constants("deliver", 1) == ["kitchen"]


def test_wire_format_collapses_single_sort_slot(tmp_path):
    sig = _toy_signature()
    path = tmp_path / "toy.json"
    save_signature(sig, path)
    blob = json.loads(path.read_text())
    assert blob["predicates"]["deliver"]["arg_sorts"] == ["item", "loc"]
    assert blob["predicates"]["see"]["arg_sorts"] == [["item", "loc"]]


# ---------------------------------------------------------------------------
# signature_builder
# ---------------------------------------------------------------------------


VLTL_TOTAL = REPO_DATA / "VLTL-Bench" / "total"


@pytest.mark.skipif(not VLTL_TOTAL.exists(), reason="VLTL-Bench data missing")
@pytest.mark.parametrize("domain", ["warehouse", "traffic_light", "search_and_rescue"])
def test_vltl_signature_full_coverage(domain):
    sig = build_vltl_signature(domain, VLTL_TOTAL / f"{domain}.jsonl")
    stats = verify_corpus_coverage(sig, VLTL_TOTAL / f"{domain}.jsonl")
    assert stats["n_uncovered"] == 0
    assert stats["n_props"] > 0


@pytest.mark.skipif(not VLTL_TOTAL.exists(), reason="VLTL-Bench data missing")
def test_sandr_photo_is_polymorphic():
    sig = build_vltl_signature("search_and_rescue", VLTL_TOTAL / "search_and_rescue.jsonl")
    assert sig.predicates["photo"].arg_sorts == [["person", "threat"]]
    assert sig.predicates["record"].arg_sorts == [["person", "threat"]]
    # arity-0 predicates have empty arg_sorts
    assert sig.predicates["get_help"].arg_sorts == []


@pytest.mark.skipif(not SIGNATURES_DIR.exists(), reason="signatures not built")
def test_navi_cluster_map_total_and_disjoint():
    path = SIGNATURES_DIR / "navi_clusters.json"
    raw_to_canon, canon_order, drop_set = load_cluster_map(path)
    # Disjoint: each raw -> exactly one canonical (load enforces this)
    assert len(set(raw_to_canon)) == len(raw_to_canon)
    # No overlap between map and drop_set (loader enforces, double-check)
    assert not (set(raw_to_canon) & drop_set)
    assert set(canon_order) == {"go_to", "take", "drop"}


@pytest.mark.skipif(not CLEAN_DIR.exists(), reason="cleaned priorwork missing")
def test_navi_signature_coverage_after_clustering():
    sig, evidence = build_priorwork_signature(
        "navi",
        CLEAN_DIR / "total" / "navi.jsonl",
        cluster_map_path=SIGNATURES_DIR / "navi_clusters.json",
    )
    raw_to_canon, _, drop_set = load_cluster_map(SIGNATURES_DIR / "navi_clusters.json")
    stats = verify_corpus_coverage(
        sig,
        CLEAN_DIR / "total" / "navi.jsonl",
        cluster_map=raw_to_canon,
        drop_set=drop_set,
    )
    assert stats["n_uncovered"] == 0
    assert set(sig.predicate_names()) == {"go_to", "take", "drop"}


# ---------------------------------------------------------------------------
# priorwork_cleaner
# ---------------------------------------------------------------------------


def test_raw_prop_is_bad_keeps_navi_trailing_paren():
    bad, _ = _raw_prop_is_bad({"action_canon": "get_to", "args_canon": ["cup)"]})
    assert not bad


def test_raw_prop_is_bad_drops_whitespace_glitch():
    bad, reason = _raw_prop_is_bad({"action_canon": "at", "args_canon": [" photo"]})
    assert bad and "whitespace" in reason


def test_raw_prop_is_bad_drops_bare_let_go():
    bad, reason = _raw_prop_is_bad({"action_canon": "let_go", "args_canon": []})
    assert bad and "bare let_go" in reason


def test_raw_prop_is_bad_keeps_normal_arg():
    bad, _ = _raw_prop_is_bad({"action_canon": "take", "args_canon": ["apple"]})
    assert not bad


def test_clean_file_strips_paren_and_drops_glitch(tmp_path):
    src = tmp_path / "src.jsonl"
    dst = tmp_path / "dst.jsonl"
    rows = [
        # good navi row (trailing paren that should be stripped)
        {"prop_dict": {"prop_1": {"action_canon": "get_to", "action_ref": "get to",
                                  "args_canon": ["cup)"], "args_ref": ["cup)"]}}},
        # row to drop entirely (only bad prop)
        {"prop_dict": {"prop_1": {"action_canon": "let_go", "action_ref": "let go",
                                  "args_canon": [], "args_ref": []}}},
        # mixed row: keep one prop, drop the other
        {"prop_dict": {
            "prop_1": {"action_canon": "at", "action_ref": "go to",
                       "args_canon": [" photo"], "args_ref": [" photo"]},
            "prop_2": {"action_canon": "pickup", "action_ref": "pickup",
                       "args_canon": ["bottle_3"], "args_ref": ["bottle 3"]},
        }},
    ]
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    result = clean_file(src, dst)
    assert result["stats"]["rows_in"] == 3
    assert result["stats"]["rows_out"] == 2   # one row dropped (all props bad)
    assert result["stats"]["props_dropped"] == 2
    out_rows = [json.loads(l) for l in dst.read_text().splitlines()]
    # Trailing paren stripped
    assert out_rows[0]["prop_dict"]["prop_1"]["args_canon"] == ["cup"]
    # Mixed-row kept only prop_2
    assert list(out_rows[1]["prop_dict"]) == ["prop_2"]


# ---------------------------------------------------------------------------
# GroundingDataset
# ---------------------------------------------------------------------------


def _tiny_warehouse_sig():
    return Signature(
        name="warehouse",
        sorts=["item", "location"],
        predicates={
            "deliver": PredicateDef("deliver", [["item"], ["location"]]),
            "search": PredicateDef("search", [["item"]]),
            "idle": PredicateDef("idle", []),
        },
        constants={
            "item": ["backpack", "package"],
            "location": ["loading_dock", "shelf"],
        },
    )


def _write_tiny_corpus(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_grounding_dataset_flattens_props(tmp_path):
    sig = _tiny_warehouse_sig()
    rows = [
        {"prop_dict": {
            "prop_1": {"action_canon": "search", "action_ref": "find",
                       "args_canon": ["backpack"], "args_ref": ["bookbag"]},
            "prop_2": {"action_canon": "deliver", "action_ref": "drop off",
                       "args_canon": ["package", "shelf"],
                       "args_ref": ["package", "shelf"]},
        }},
        {"prop_dict": {
            "prop_1": {"action_canon": "idle", "action_ref": "stand by",
                       "args_canon": [], "args_ref": []},
        }},
    ]
    src = tmp_path / "wh.jsonl"
    _write_tiny_corpus(src, rows)
    ds = GroundingDataset(src, sig)
    assert len(ds) == 3
    assert {ex.gold_pred for ex in ds.examples} == {"search", "deliver", "idle"}


def test_collator_shapes_and_padding():
    sig = _tiny_warehouse_sig()
    from ginsign.grounding_dataset import GroundingExample as GE
    batch = [
        GE(ap_text="find bookbag", gold_pred="search", gold_args=["backpack"]),
        GE(ap_text="drop off package shelf", gold_pred="deliver",
           gold_args=["package", "shelf"]),
        GE(ap_text="stand by", gold_pred="idle", gold_args=[]),
    ]
    out = collate_grounding_batch(batch, sig)
    assert out["ap_texts"] == [ex.ap_text for ex in batch]
    assert out["gold_pred_idx"].tolist() == [
        sig.predicate_names().index("search"),
        sig.predicate_names().index("deliver"),
        sig.predicate_names().index("idle"),
    ]
    assert len(out["slot_candidate_lists"]) == 2  # max arity = 2 (deliver)
    # slot 0: search has [backpack, package], deliver has [backpack, package], idle pads
    assert out["slot_candidate_lists"][0][0] == ["backpack", "package"]
    assert out["slot_candidate_lists"][0][1] == ["backpack", "package"]
    assert out["slot_candidate_lists"][0][2] == ["[PAD]"]
    assert out["slot_type_masks"][0][2] == [False]
    # slot 1: only deliver in scope
    assert out["slot_candidate_lists"][1][1] == ["loading_dock", "shelf"]
    assert out["slot_candidate_lists"][1][0] == ["[PAD]"]
    # gold_arg_idx shape [B, max_arity]; out-of-scope slots have -1
    assert out["gold_arg_idx"].shape == (3, 2)
    assert out["gold_arg_idx"][0].tolist() == [0, -1]   # search(backpack)
    assert out["gold_arg_idx"][1].tolist() == [1, 1]    # deliver(package, shelf)
    assert out["gold_arg_idx"][2].tolist() == [-1, -1]  # idle


def _have_bert():
    try:
        from transformers import BertTokenizerFast
        BertTokenizerFast.from_pretrained("bert-base-cased", local_files_only=True)
        return True
    except Exception:
        return False


needs_bert = pytest.mark.skipif(not _have_bert(), reason="bert-base-cased not in HF cache")


@needs_bert
def test_dataset_integration_with_grounder(tmp_path):
    """Smoke test: dataset + collator output feeds compute_loss without crash."""
    from ginsign.ptr_grounder import PointerJointGrounder

    sig = _tiny_warehouse_sig()
    rows = [
        {"prop_dict": {
            "prop_1": {"action_canon": "search", "action_ref": "find",
                       "args_canon": ["backpack"], "args_ref": ["bookbag"]},
        }},
        {"prop_dict": {
            "prop_1": {"action_canon": "deliver", "action_ref": "drop off",
                       "args_canon": ["package", "shelf"],
                       "args_ref": ["package", "shelf"]},
        }},
    ]
    src = tmp_path / "wh.jsonl"
    _write_tiny_corpus(src, rows)

    ds = GroundingDataset(src, sig)
    collate = make_collate_fn(sig)
    batch = collate([ds[0], ds[1]])

    torch.manual_seed(0)
    grounder = PointerJointGrounder(bert_name="bert-base-cased")
    grounder.train()
    out = grounder.compute_loss(
        ap_texts=batch["ap_texts"],
        predicate_lists=batch["predicate_lists"],
        gold_pred_idx=batch["gold_pred_idx"],
        slot_candidate_lists=batch["slot_candidate_lists"],
        slot_type_masks=batch["slot_type_masks"],
        gold_arg_idx=batch["gold_arg_idx"],
    )
    assert torch.isfinite(out["loss"])
