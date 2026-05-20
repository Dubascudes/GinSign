"""Pointer-head grounder with autoregressive joint argument decoding.

Replaces the BertForTokenClassification(num_labels=2) head in BERT_Grounder
with: (a) a softmax pointer over candidate spans (mean-pooled subtokens), and
(b) a Transformer decoder that grounds the argument tuple autoregressively,
conditioning each arg on previously chosen args.

The encoder is shared across the predicate and argument stages. The candidate
representations consumed by both heads come from the same forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast


@dataclass
class CandidateSpans:
    """Subtoken offsets for each candidate in an encoded prefix.

    starts/ends index into the BERT sequence dimension; the half-open
    span [starts[b,n], ends[b,n]) is mean-pooled to form candidate n's
    representation. mask masks out padding candidates; type_mask masks
    out candidates that violate the type filter for the current slot.
    """

    starts: torch.LongTensor   # [B, N]
    ends: torch.LongTensor     # [B, N]
    mask: torch.BoolTensor     # [B, N]  False = padding candidate
    type_mask: torch.BoolTensor  # [B, N]  False = filtered out by type

    def effective_mask(self) -> torch.BoolTensor:
        return self.mask & self.type_mask

    def to(self, device) -> "CandidateSpans":
        return CandidateSpans(
            starts=self.starts.to(device),
            ends=self.ends.to(device),
            mask=self.mask.to(device),
            type_mask=self.type_mask.to(device),
        )


def pool_candidate_reprs(H: torch.Tensor, spans: CandidateSpans) -> torch.Tensor:
    """Mean-pool subtoken hidden states over each candidate's span.

    H: [B, L, hidden]. Returns [B, N, hidden]. Pads with zeros for
    invalid candidates (mask=False).
    """
    B, L, Hd = H.shape
    N = spans.starts.size(1)
    arange = torch.arange(L, device=H.device).view(1, 1, L)
    starts = spans.starts.unsqueeze(-1)
    ends = spans.ends.unsqueeze(-1)
    span_mask = (arange >= starts) & (arange < ends)  # [B, N, L]
    span_mask = span_mask & spans.mask.unsqueeze(-1)

    summed = torch.einsum("bnl,blh->bnh", span_mask.float(), H)
    counts = span_mask.sum(dim=2).clamp(min=1).unsqueeze(-1).float()
    return summed / counts


class PointerHead(nn.Module):
    """Softmax pointer over candidates.

    Score per candidate: MLP([h_ap ; h_c ; h_ap * h_c]). Returns raw
    logits with masked positions set to -inf so log_softmax is safe.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        query: torch.Tensor,           # [B, H]
        cand_reprs: torch.Tensor,      # [B, N, H]
        valid_mask: torch.Tensor,      # [B, N]  bool
    ) -> torch.Tensor:                  # [B, N]
        q = query.unsqueeze(1).expand_as(cand_reprs)
        feats = torch.cat([q, cand_reprs, q * cand_reprs], dim=-1)
        logits = self.score(feats).squeeze(-1)
        return logits.masked_fill(~valid_mask, float("-inf"))


class ARArgDecoder(nn.Module):
    """Autoregressive argument decoder.

    State sequence per step r: [bos = ap+pred fusion, arg_1_repr, ...,
    arg_{r-1}_repr]. Transformer decoder layers self-attend over the
    state and cross-attend over the current slot's candidate reprs;
    the last position's hidden state is the query passed to PointerHead.
    """

    MAX_ARITY = 8

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.pointer = PointerHead(hidden_size, dropout=dropout)
        self.slot_emb = nn.Embedding(self.MAX_ARITY + 1, hidden_size)
        self.ap_proj = nn.Linear(hidden_size, hidden_size)
        self.pred_proj = nn.Linear(hidden_size, hidden_size)

    def _bos(self, ap_repr: torch.Tensor, pred_repr: torch.Tensor) -> torch.Tensor:
        return self.ap_proj(ap_repr) + self.pred_proj(pred_repr)

    def _attach_slot_pos(self, tgt: torch.Tensor) -> torch.Tensor:
        B, T, _ = tgt.shape
        ids = torch.arange(T, device=tgt.device).unsqueeze(0).expand(B, -1)
        return tgt + self.slot_emb(ids)

    def step(
        self,
        ap_repr: torch.Tensor,                  # [B, H]
        pred_repr: torch.Tensor,                # [B, H]
        prev_arg_reprs: Optional[torch.Tensor], # [B, r, H] or None
        cand_reprs: torch.Tensor,               # [B, N, H]
        valid_mask: torch.Tensor,               # [B, N]
    ) -> torch.Tensor:                           # [B, N] logits
        bos = self._bos(ap_repr, pred_repr).unsqueeze(1)
        if prev_arg_reprs is None or prev_arg_reprs.size(1) == 0:
            tgt = bos
        else:
            tgt = torch.cat([bos, prev_arg_reprs], dim=1)
        tgt = self._attach_slot_pos(tgt)
        out = self.decoder(
            tgt=tgt,
            memory=cand_reprs,
            memory_key_padding_mask=~valid_mask,
        )
        query = out[:, -1, :]
        return self.pointer(query, cand_reprs, valid_mask)

    def forward(
        self,
        ap_repr: torch.Tensor,
        pred_repr: torch.Tensor,
        gold_arg_reprs: torch.Tensor,           # [B, m, H]  teacher forcing
        slot_cand_reprs: Sequence[torch.Tensor],
        slot_valid_masks: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Teacher-forced training pass; returns per-slot logits."""
        assert len(slot_cand_reprs) == len(slot_valid_masks)
        all_logits = []
        for r, (cand_r, mask_r) in enumerate(zip(slot_cand_reprs, slot_valid_masks)):
            prev = gold_arg_reprs[:, :r, :] if r > 0 else None
            all_logits.append(self.step(ap_repr, pred_repr, prev, cand_r, mask_r))
        return all_logits

    @torch.no_grad()
    def beam(
        self,
        ap_repr: torch.Tensor,                  # [B, H]
        pred_repr: torch.Tensor,                # [B, H]
        slot_cand_reprs: Sequence[torch.Tensor],
        slot_valid_masks: Sequence[torch.Tensor],
        beam_size: int = 1,
    ) -> List[List[Tuple[List[int], float]]]:
        """Beam search per batch item. Returns top-k (arg_ids, logprob) per item."""
        B = ap_repr.size(0)
        m = len(slot_cand_reprs)
        results: List[List[Tuple[List[int], float]]] = []
        for b in range(B):
            ap_b = ap_repr[b : b + 1]
            pred_b = pred_repr[b : b + 1]
            beams: List[Tuple[List[int], float, Optional[torch.Tensor]]] = [([], 0.0, None)]
            for r in range(m):
                cand_r = slot_cand_reprs[r][b : b + 1]
                mask_r = slot_valid_masks[r][b : b + 1]
                n_valid = int(mask_r.sum().item())
                if n_valid == 0:
                    beams = [(ids + [-1], score, prev) for ids, score, prev in beams]
                    continue
                k = min(beam_size, n_valid)
                next_beams: List[Tuple[List[int], float, torch.Tensor]] = []
                for ids, score, prev in beams:
                    logits = self.step(ap_b, pred_b, prev, cand_r, mask_r)
                    logp = F.log_softmax(logits, dim=-1).squeeze(0)
                    top_lp, top_idx = logp.topk(k)
                    for lp, idx in zip(top_lp.tolist(), top_idx.tolist()):
                        arg_repr = cand_r[:, idx : idx + 1, :]
                        new_prev = arg_repr if prev is None else torch.cat([prev, arg_repr], dim=1)
                        next_beams.append((ids + [idx], score + lp, new_prev))
                next_beams.sort(key=lambda x: -x[1])
                beams = next_beams[:beam_size]
            results.append([(ids, score) for ids, score, _ in beams])
        return results


class PointerJointGrounder(nn.Module):
    """End-to-end grounder: BERT encoder -> predicate pointer -> AR arg decoder.

    Encoding format (per stage): [CLS] ap_text [SEP] cand_1 cand_2 ... [SEP]
    where each candidate is one "word" passed via is_split_into_words=True.
    Word ids let us recover the subtoken span of each candidate without
    parsing offsets manually.
    """

    def __init__(self, bert_name: str = "bert-base-cased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_name)
        H = self.bert.config.hidden_size
        self.pred_head = PointerHead(H)
        self.arg_decoder = ARArgDecoder(H)

    @property
    def hidden_size(self) -> int:
        return self.bert.config.hidden_size

    def encode_stage(
        self,
        ap_texts: Sequence[str],
        candidate_lists: Sequence[Sequence[str]],
        type_masks: Optional[Sequence[Sequence[bool]]] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, CandidateSpans]:
        """Encode one stage's batch.

        Returns (H, ap_repr, cand_reprs, spans) where H is the full
        encoder output, ap_repr is the mean-pooled AP text, cand_reprs
        is per-candidate after span pooling, and spans carries offsets +
        masks for downstream use (e.g., the arg decoder's beam search).
        """
        device = device or next(self.parameters()).device
        max_n = max(len(c) for c in candidate_lists)
        padded_cands, mask_rows, type_rows = [], [], []
        for i, cands in enumerate(candidate_lists):
            pad_n = max_n - len(cands)
            padded_cands.append(list(cands) + ["[PAD]"] * pad_n)
            mask_rows.append([True] * len(cands) + [False] * pad_n)
            if type_masks is not None:
                tm = list(type_masks[i]) + [False] * pad_n
            else:
                tm = [True] * len(cands) + [False] * pad_n
            type_rows.append(tm)

        ap_words = [t.split() if isinstance(t, str) else list(t) for t in ap_texts]
        enc = self.tokenizer(
            ap_words,
            padded_cands,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        outputs = self.bert(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")})
        H = outputs.last_hidden_state  # [B, L, hidden]
        B, L, _ = H.shape

        starts = torch.zeros(B, max_n, dtype=torch.long, device=device)
        ends = torch.zeros(B, max_n, dtype=torch.long, device=device)
        ap_masks = torch.zeros(B, L, dtype=torch.bool, device=device)
        valid = torch.tensor(mask_rows, dtype=torch.bool, device=device)
        type_mask = torch.tensor(type_rows, dtype=torch.bool, device=device)

        for b in range(B):
            word_ids = enc.word_ids(batch_index=b)
            seq_ids = enc.sequence_ids(batch_index=b)
            ap_positions = [i for i, (w, s) in enumerate(zip(word_ids, seq_ids)) if s == 0 and w is not None]
            if ap_positions:
                ap_masks[b, ap_positions] = True

            cand_positions: dict = {}
            for i, (w, s) in enumerate(zip(word_ids, seq_ids)):
                if s == 1 and w is not None:
                    cand_positions.setdefault(w, []).append(i)
            for n in range(max_n):
                if not mask_rows[b][n]:
                    continue
                positions = cand_positions.get(n, [])
                if positions:
                    starts[b, n] = positions[0]
                    ends[b, n] = positions[-1] + 1
                else:
                    valid[b, n] = False  # truncated away

        ap_counts = ap_masks.sum(dim=1).clamp(min=1).unsqueeze(-1).float()
        ap_repr = (H * ap_masks.unsqueeze(-1).float()).sum(dim=1) / ap_counts

        spans = CandidateSpans(starts=starts, ends=ends, mask=valid, type_mask=type_mask)
        cand_reprs = pool_candidate_reprs(H, spans)
        return H, ap_repr, cand_reprs, spans

    def ground_predicate(
        self,
        ap_texts: Sequence[str],
        predicate_lists: Sequence[Sequence[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, CandidateSpans]:
        """Returns (logits, ap_repr, cand_reprs, spans) for the predicate stage."""
        H, ap_repr, cand_reprs, spans = self.encode_stage(ap_texts, predicate_lists)
        logits = self.pred_head(ap_repr, cand_reprs, spans.effective_mask())
        return logits, ap_repr, cand_reprs, spans

    def compute_loss(
        self,
        ap_texts: Sequence[str],
        predicate_lists: Sequence[Sequence[str]],
        gold_pred_idx: torch.Tensor,                                  # [B]
        slot_candidate_lists: Sequence[Sequence[Sequence[str]]],      # [m][B][N_r]
        slot_type_masks: Sequence[Sequence[Sequence[bool]]],
        gold_arg_idx: torch.Tensor,                                   # [B, m]   -1 for absent slots
        lambda_arg: float = 1.0,
    ) -> dict:
        """Single forward+loss for joint training; teacher-forced on predicates."""
        device = next(self.parameters()).device
        pred_logits, ap_repr, pred_cand_reprs, _ = self.ground_predicate(ap_texts, predicate_lists)
        L_pred = F.cross_entropy(pred_logits, gold_pred_idx.to(device))

        B = pred_logits.size(0)
        H = self.hidden_size
        # Pred repr = gold predicate's pooled candidate vector (teacher forcing).
        pred_repr = pred_cand_reprs[torch.arange(B), gold_pred_idx.to(device)]

        # Encode each slot. Slots have potentially different candidate sets,
        # so we encode them independently here. (Could be batched if we group
        # by signature, but keep it simple for the first cut.)
        slot_cand_reprs_list = []
        slot_valid_masks_list = []
        gold_arg_reprs = torch.zeros(B, len(slot_candidate_lists), H, device=device)
        for r, (cands_r, tm_r) in enumerate(zip(slot_candidate_lists, slot_type_masks)):
            _, _, cand_reprs_r, spans_r = self.encode_stage(ap_texts, cands_r, tm_r, device=device)
            slot_cand_reprs_list.append(cand_reprs_r)
            slot_valid_masks_list.append(spans_r.effective_mask())
            gold_r = gold_arg_idx[:, r].to(device)
            present = gold_r >= 0
            if present.any():
                safe_idx = gold_r.clamp(min=0)
                gold_arg_reprs[:, r, :] = cand_reprs_r[torch.arange(B), safe_idx] * present.unsqueeze(-1).float()

        arg_logits = self.arg_decoder(
            ap_repr=ap_repr,
            pred_repr=pred_repr,
            gold_arg_reprs=gold_arg_reprs,
            slot_cand_reprs=slot_cand_reprs_list,
            slot_valid_masks=slot_valid_masks_list,
        )

        L_arg = pred_logits.new_zeros(())
        n_arg_slots = 0
        for r, logits_r in enumerate(arg_logits):
            gold_r = gold_arg_idx[:, r].to(device)
            present = gold_r >= 0
            if not present.any():
                continue
            # Score only the present rows: absent rows may have all-masked
            # logits (-inf) which would NaN the loss even when multiplied by 0.
            L_arg = L_arg + F.cross_entropy(logits_r[present], gold_r[present])
            n_arg_slots += 1
        if n_arg_slots > 0:
            L_arg = L_arg / n_arg_slots

        loss = L_pred + lambda_arg * L_arg
        return {
            "loss": loss,
            "pred_loss": L_pred.detach(),
            "arg_loss": L_arg.detach(),
            "pred_logits": pred_logits,
            "arg_logits": arg_logits,
        }

    @torch.no_grad()
    def ground(
        self,
        ap_text: str,
        predicates: Sequence[str],
        candidate_sets_per_predicate: dict,   # {pred_name: [List[str] per slot]}
        type_masks_per_predicate: Optional[dict] = None,
        beam_size: int = 1,
    ) -> List[Tuple[str, List[str], float]]:
        """End-to-end inference for a single AP span.

        Returns a list of (pred_name, [arg_names], score) ordered by score.
        With beam_size=1 returns a single tuple.
        """
        device = next(self.parameters()).device
        pred_logits, ap_repr, pred_cand_reprs, _ = self.ground_predicate(
            [ap_text], [list(predicates)]
        )
        pred_lp = F.log_softmax(pred_logits, dim=-1).squeeze(0)
        pred_topk = pred_lp.topk(min(beam_size, len(predicates)))

        results: List[Tuple[str, List[str], float]] = []
        for p_lp, p_idx in zip(pred_topk.values.tolist(), pred_topk.indices.tolist()):
            pred_name = predicates[p_idx]
            slots = candidate_sets_per_predicate[pred_name]
            type_masks = (type_masks_per_predicate or {}).get(pred_name)
            pred_repr = pred_cand_reprs[:, p_idx, :]

            if not slots:
                results.append((pred_name, [], p_lp))
                continue

            slot_cand_reprs, slot_valid_masks = [], []
            for r, cands_r in enumerate(slots):
                tm_r = [type_masks[r]] if type_masks else None
                _, _, cand_reprs_r, spans_r = self.encode_stage(
                    [ap_text], [cands_r], tm_r, device=device
                )
                slot_cand_reprs.append(cand_reprs_r)
                slot_valid_masks.append(spans_r.effective_mask())

            beams = self.arg_decoder.beam(
                ap_repr=ap_repr,
                pred_repr=pred_repr,
                slot_cand_reprs=slot_cand_reprs,
                slot_valid_masks=slot_valid_masks,
                beam_size=beam_size,
            )[0]
            for arg_ids, arg_score in beams:
                arg_names = [
                    slots[r][i] if 0 <= i < len(slots[r]) else "<MISSING>"
                    for r, i in enumerate(arg_ids)
                ]
                results.append((pred_name, arg_names, p_lp + arg_score))

        results.sort(key=lambda x: -x[2])
        return results[:beam_size]
