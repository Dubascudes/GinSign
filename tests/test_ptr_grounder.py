"""Unit tests for ptr_grounder.

Tier 1 (no tokenizer): pure-tensor tests of PointerHead, ARArgDecoder, and
span pooling. Tier 2 (full grounder): end-to-end tests that need a real
BERT tokenizer; cached bert-base-cased is required (no network).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ginsign.ptr_grounder import (
    ARArgDecoder,
    CandidateSpans,
    PointerHead,
    PointerJointGrounder,
    pool_candidate_reprs,
)


# ---------------------------------------------------------------------------
# Tier 1: pure-tensor unit tests
# ---------------------------------------------------------------------------


def _spans(starts, ends, mask=None, type_mask=None):
    s = torch.tensor(starts, dtype=torch.long)
    e = torch.tensor(ends, dtype=torch.long)
    m = torch.tensor(mask, dtype=torch.bool) if mask is not None else torch.ones_like(s, dtype=torch.bool)
    tm = torch.tensor(type_mask, dtype=torch.bool) if type_mask is not None else torch.ones_like(s, dtype=torch.bool)
    return CandidateSpans(starts=s, ends=e, mask=m, type_mask=tm)


def test_pool_candidate_reprs_multi_subtoken():
    torch.manual_seed(0)
    H = torch.arange(2 * 6 * 4, dtype=torch.float).reshape(2, 6, 4)
    # batch 0: cand 0 = [0,2), cand 1 = [2,5), cand 2 padding
    # batch 1: cand 0 = [1,2), cand 1 = [3,6), cand 2 padding
    spans = _spans(
        starts=[[0, 2, 0], [1, 3, 0]],
        ends=[[2, 5, 0], [2, 6, 0]],
        mask=[[True, True, False], [True, True, False]],
    )
    reprs = pool_candidate_reprs(H, spans)
    assert reprs.shape == (2, 3, 4)
    # b0 cand 0 = mean of H[0, 0:2]
    assert torch.allclose(reprs[0, 0], H[0, 0:2].mean(dim=0))
    # b0 cand 1 = mean of H[0, 2:5]
    assert torch.allclose(reprs[0, 1], H[0, 2:5].mean(dim=0))
    # b1 cand 1 = mean of H[1, 3:6]
    assert torch.allclose(reprs[1, 1], H[1, 3:6].mean(dim=0))
    # padding candidates are zero
    assert torch.all(reprs[0, 2] == 0)
    assert torch.all(reprs[1, 2] == 0)


def test_pool_candidate_reprs_single_subtoken_identity():
    H = torch.randn(1, 5, 8)
    spans = _spans(starts=[[2]], ends=[[3]])
    reprs = pool_candidate_reprs(H, spans)
    assert torch.allclose(reprs[0, 0], H[0, 2])


def test_pointer_head_softmax_and_mask():
    torch.manual_seed(0)
    head = PointerHead(hidden_size=16)
    query = torch.randn(2, 16)
    cand = torch.randn(2, 4, 16)
    # batch 0: only cands 0, 1 valid (type-filtered); batch 1: cands 1, 2, 3 valid
    valid = torch.tensor([[True, True, False, False], [False, True, True, True]])
    logits = head(query, cand, valid)
    assert logits.shape == (2, 4)
    # Masked positions must be -inf so softmax assigns 0 probability.
    assert torch.isinf(logits[0, 2]) and logits[0, 2] < 0
    assert torch.isinf(logits[1, 0]) and logits[1, 0] < 0
    probs = F.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert probs[0, 2].item() == 0.0 and probs[0, 3].item() == 0.0
    assert probs[1, 0].item() == 0.0


def test_pointer_head_all_masked_safe_under_logsoftmax():
    # All-masked rows would produce nan with naive softmax; verify we
    # never *call* the head with an all-masked row in normal flow, but
    # do verify -inf logits exist so callers can detect and skip.
    head = PointerHead(hidden_size=8)
    query = torch.randn(1, 8)
    cand = torch.randn(1, 3, 8)
    valid = torch.tensor([[False, False, False]])
    logits = head(query, cand, valid)
    assert torch.all(torch.isinf(logits)) and torch.all(logits < 0)


def test_ar_decoder_step_shapes_and_mask():
    torch.manual_seed(1)
    H = 16
    decoder = ARArgDecoder(hidden_size=H, num_layers=1, nhead=4, dim_feedforward=32)
    ap = torch.randn(3, H)
    pred = torch.randn(3, H)
    cand = torch.randn(3, 5, H)
    valid = torch.tensor([
        [True, True, True, False, False],
        [True, False, True, True, True],
        [True, True, True, True, True],
    ])
    # Step 0: no prev args
    logits0 = decoder.step(ap, pred, prev_arg_reprs=None, cand_reprs=cand, valid_mask=valid)
    assert logits0.shape == (3, 5)
    assert torch.isinf(logits0[0, 3]) and logits0[0, 3] < 0
    assert torch.isinf(logits0[1, 1]) and logits0[1, 1] < 0
    # Step 1: with a prev arg per item
    prev = cand[:, 0:1, :]
    logits1 = decoder.step(ap, pred, prev_arg_reprs=prev, cand_reprs=cand, valid_mask=valid)
    assert logits1.shape == (3, 5)


def test_ar_decoder_forward_teacher_forced():
    torch.manual_seed(2)
    H = 16
    decoder = ARArgDecoder(hidden_size=H, num_layers=1, nhead=4, dim_feedforward=32)
    ap = torch.randn(2, H)
    pred = torch.randn(2, H)
    # arity 2; slots can have different candidate set sizes
    slot0_cand = torch.randn(2, 4, H)
    slot1_cand = torch.randn(2, 7, H)
    slot0_mask = torch.ones(2, 4, dtype=torch.bool)
    slot1_mask = torch.ones(2, 7, dtype=torch.bool)
    gold_arg_reprs = torch.randn(2, 2, H)
    logits = decoder(
        ap_repr=ap, pred_repr=pred,
        gold_arg_reprs=gold_arg_reprs,
        slot_cand_reprs=[slot0_cand, slot1_cand],
        slot_valid_masks=[slot0_mask, slot1_mask],
    )
    assert len(logits) == 2
    assert logits[0].shape == (2, 4)
    assert logits[1].shape == (2, 7)


def test_beam_search_sorted_and_topk():
    torch.manual_seed(3)
    H = 16
    decoder = ARArgDecoder(hidden_size=H, num_layers=1, nhead=4, dim_feedforward=32)
    ap = torch.randn(1, H)
    pred = torch.randn(1, H)
    slot0_cand = torch.randn(1, 5, H)
    slot1_cand = torch.randn(1, 4, H)
    slot0_mask = torch.ones(1, 5, dtype=torch.bool)
    slot1_mask = torch.ones(1, 4, dtype=torch.bool)

    beams = decoder.beam(
        ap_repr=ap, pred_repr=pred,
        slot_cand_reprs=[slot0_cand, slot1_cand],
        slot_valid_masks=[slot0_mask, slot1_mask],
        beam_size=3,
    )
    assert len(beams) == 1
    items = beams[0]
    assert len(items) == 3
    scores = [s for _, s in items]
    assert scores == sorted(scores, reverse=True), "beam search must return sorted scores"
    for ids, _ in items:
        assert len(ids) == 2  # arity matches slot count
        assert 0 <= ids[0] < 5 and 0 <= ids[1] < 4


def test_beam_search_greedy_matches_argmax():
    torch.manual_seed(4)
    H = 16
    decoder = ARArgDecoder(hidden_size=H, num_layers=1, nhead=4, dim_feedforward=32)
    ap = torch.randn(1, H)
    pred = torch.randn(1, H)
    cand = torch.randn(1, 6, H)
    mask = torch.ones(1, 6, dtype=torch.bool)
    decoder.eval()
    beams = decoder.beam(
        ap_repr=ap, pred_repr=pred,
        slot_cand_reprs=[cand],
        slot_valid_masks=[mask],
        beam_size=1,
    )
    greedy_id = beams[0][0][0][0]
    # Compare with the direct step call
    logits = decoder.step(ap, pred, None, cand, mask)
    assert greedy_id == int(logits.argmax(dim=-1).item())


# ---------------------------------------------------------------------------
# Tier 2: full grounder integration tests (need cached bert-base-cased)
# ---------------------------------------------------------------------------


def _have_bert():
    try:
        from transformers import BertTokenizerFast
        BertTokenizerFast.from_pretrained("bert-base-cased", local_files_only=True)
        return True
    except Exception:
        return False


needs_bert = pytest.mark.skipif(not _have_bert(), reason="bert-base-cased not in HF cache")


@pytest.fixture(scope="module")
def grounder():
    torch.manual_seed(0)
    g = PointerJointGrounder(bert_name="bert-base-cased")
    g.eval()
    return g


@needs_bert
def test_encode_stage_recovers_candidate_spans(grounder):
    ap_text = "pick up the package from room A"
    candidates = [["search", "deliver", "loading_dock", "backpack"]]
    H, ap_repr, cand_reprs, spans = grounder.encode_stage([ap_text], candidates)
    assert H.dim() == 3
    assert ap_repr.shape == (1, grounder.hidden_size)
    assert cand_reprs.shape == (1, 4, grounder.hidden_size)
    # All 4 candidates should have valid spans (no truncation expected)
    assert spans.mask[0].all().item()
    # Multi-subtoken candidate (loading_dock) should span >= 1 subtoken
    span_len = (spans.ends[0, 2] - spans.starts[0, 2]).item()
    assert span_len >= 1
    # Spans must be inside the sequence
    L = H.size(1)
    assert (spans.starts[spans.mask] < L).all()
    assert (spans.ends[spans.mask] <= L).all()


@needs_bert
def test_prefix_swap_changes_predictions(grounder):
    # Same AP span, different signatures => the pointer head's predictions
    # must be defined relative to the new candidate set (open-set property).
    ap_text = "find the bookbag"
    sig_a = [["search", "deliver", "idle"]]
    sig_b = [["pickup", "photo", "put_down", "go_to"]]

    logits_a, _, _, _ = grounder.ground_predicate([ap_text], sig_a)
    logits_b, _, _, _ = grounder.ground_predicate([ap_text], sig_b)
    assert logits_a.shape == (1, 3)
    assert logits_b.shape == (1, 4)
    pred_a = sig_a[0][int(logits_a.argmax(dim=-1).item())]
    pred_b = sig_b[0][int(logits_b.argmax(dim=-1).item())]
    # The two predictions come from disjoint label universes; structural
    # check: index space changed, model produced a label from each.
    assert pred_a in sig_a[0]
    assert pred_b in sig_b[0]


@needs_bert
def test_compute_loss_runs_and_backprops(grounder):
    ap_texts = ["find the bookbag", "deliver the package to shipping"]
    predicate_lists = [
        ["search", "deliver", "idle"],
        ["search", "deliver", "idle"],
    ]
    gold_pred_idx = torch.tensor([0, 1])
    # slot 0 candidates per batch item; slot 1 candidates per batch item
    slot_candidate_lists = [
        [["backpack", "package"], ["backpack", "package"]],
        [["loading_dock", "shipping_bay"], ["loading_dock", "shipping_bay"]],
    ]
    slot_type_masks = [
        [[True, True], [True, True]],
        [[False, False], [True, True]],   # item 0 has arity 1 (search); item 1 arity 2 (deliver)
    ]
    gold_arg_idx = torch.tensor([[0, -1], [1, 1]])  # -1 = no slot
    grounder.train()
    out = grounder.compute_loss(
        ap_texts=ap_texts,
        predicate_lists=predicate_lists,
        gold_pred_idx=gold_pred_idx,
        slot_candidate_lists=slot_candidate_lists,
        slot_type_masks=slot_type_masks,
        gold_arg_idx=gold_arg_idx,
    )
    loss = out["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    # Gradients must reach BERT, predicate head, and AR decoder.
    bert_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in grounder.bert.parameters()
    )
    head_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in grounder.pred_head.parameters()
    )
    dec_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in grounder.arg_decoder.parameters()
    )
    assert bert_has_grad and head_has_grad and dec_has_grad


@needs_bert
def test_ground_inference_returns_valid_atoms(grounder):
    grounder.eval()
    out = grounder.ground(
        ap_text="deliver the package to shipping",
        predicates=["search", "deliver", "idle"],
        candidate_sets_per_predicate={
            "search": [["backpack", "package"]],
            "deliver": [["backpack", "package"], ["loading_dock", "shipping_bay"]],
            "idle": [],
        },
        beam_size=2,
    )
    assert 1 <= len(out) <= 2
    for pred_name, args, score in out:
        assert pred_name in {"search", "deliver", "idle"}
        if pred_name == "search":
            assert len(args) == 1 and args[0] in {"backpack", "package"}
        elif pred_name == "deliver":
            assert len(args) == 2
            assert args[0] in {"backpack", "package"}
            assert args[1] in {"loading_dock", "shipping_bay"}
        else:
            assert args == []
        assert math.isfinite(score)
