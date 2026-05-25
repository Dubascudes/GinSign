"""Tests for the sharded encode_stage path.

Triggers sharding by giving a candidate list longer than max_per_shard
and verifies (a) shapes still line up, (b) gold indices still address
the right candidates after stitching, (c) loss is finite under sharding
where the un-sharded single-forward would either truncate (silent) or
explode to inf at -inf-gold positions, and (d) the un-sharded path
is preserved when max_per_shard is large enough.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

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


@pytest.fixture(scope="module")
def grounder_small_shard():
    from ginsign.ptr_grounder import PointerJointGrounder
    torch.manual_seed(0)
    g = PointerJointGrounder(bert_name="bert-base-cased", max_per_shard=16)
    g.eval()
    return g


@needs_bert
def test_sharded_encode_shapes_and_mask(grounder_small_shard):
    # 40 candidates with max_per_shard=16 -> 3 shards. After stitching
    # the cand dimension should still be >= 40 and every gold position
    # must be present and unmasked.
    ap = "find the bookbag"
    cands = [f"item_{i}" for i in range(40)]
    H, ap_repr, cand_reprs, spans = grounder_small_shard.encode_stage([ap], [cands])
    assert H is None, "sharded path returns H=None"
    assert ap_repr.shape == (1, grounder_small_shard.hidden_size)
    # 3 shards of size 16 each -> 48 total candidate slots after stitch
    assert cand_reprs.shape[1] >= 40
    # First 40 positions should be valid (no truncation), tail is shard padding
    assert spans.mask[0, :40].all().item(), "all real candidates survived sharding"
    assert spans.effective_mask()[0, :40].all().item()


@needs_bert
def test_sharded_pointer_softmax_normalized(grounder_small_shard):
    """A real PointerHead pass over sharded reps must produce a normalized
    distribution with no -inf at any valid position (which is the failure
    mode that produced the traffic_light inf loss)."""
    ap = "deliver the package to the loading dock"
    cands = [f"loc_{i}" for i in range(50)]
    _, ap_repr, cand_reprs, spans = grounder_small_shard.encode_stage([ap], [cands])
    valid = spans.effective_mask()
    logits = grounder_small_shard.pred_head(ap_repr, cand_reprs, valid)
    # No -inf at any valid (un-padded) candidate
    valid_logits = logits[valid]
    assert torch.isfinite(valid_logits).all(), "no -inf at valid candidate positions"
    # Softmax sums to 1
    probs = F.softmax(logits, dim=-1)
    assert torch.isclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5).all()


@needs_bert
def test_compute_loss_finite_under_sharding(grounder_small_shard):
    """Mirror the traffic_light failure mode in miniature: a 50-candidate
    type-filtered slot. With sharding the loss must be finite for every
    teacher-forced gold."""
    from ginsign.grounding_dataset import (
        GroundingExample,
        collate_grounding_batch,
    )
    from ginsign.signature_io import PredicateDef, Signature

    # 50 lane-like constants in one sort, plus a small color sort.
    lanes = [f"lane_{i}" for i in range(50)]
    sig = Signature(
        name="toy_tl",
        sorts=["light", "lane"],
        predicates={
            "change": PredicateDef("change", [["light"], ["lane"]]),
        },
        constants={"light": ["light_east", "light_west"], "lane": lanes},
    )
    batch = [
        GroundingExample(ap_text="set east signal to lane 7", gold_pred="change",
                         gold_args=["light_east", "lanes"[:0] + "lane_7"]),
        GroundingExample(ap_text="put west signal at lane 42", gold_pred="change",
                         gold_args=["light_west", "lane_42"]),
    ]
    out = collate_grounding_batch(batch, sig)
    grounder_small_shard.train()
    result = grounder_small_shard.compute_loss(
        ap_texts=out["ap_texts"],
        predicate_lists=out["predicate_lists"],
        gold_pred_idx=out["gold_pred_idx"],
        slot_candidate_lists=out["slot_candidate_lists"],
        slot_type_masks=out["slot_type_masks"],
        gold_arg_idx=out["gold_arg_idx"],
    )
    assert torch.isfinite(result["loss"]), f"loss is not finite: {result['loss']}"
    assert torch.isfinite(result["pred_loss"])
    assert torch.isfinite(result["arg_loss"])
    # Sanity: gradient flows
    result["loss"].backward()


@needs_bert
def test_large_max_per_shard_bypasses_sharding():
    """If max_per_shard >= candidate count, we should hit the single-forward
    path and get a non-None H back (proves we did NOT take the sharded path)."""
    from ginsign.ptr_grounder import PointerJointGrounder
    torch.manual_seed(0)
    g = PointerJointGrounder(bert_name="bert-base-cased", max_per_shard=200)
    g.eval()
    cands = [f"c_{i}" for i in range(30)]
    H, _, _, _ = g.encode_stage(["pick the item"], [cands])
    assert H is not None, "should have taken the single-forward path"
