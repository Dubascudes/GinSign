#!/usr/bin/env python
"""Fine-tuned seq2seq grounding baseline (rebuttal WLTJ W2).

Fine-tunes t5-base on the identical grounding data GinSign uses:
input  = "ground: <action_ref> <args_ref...> | signature: <serialized signature>"
output = "action_canon ( arg1 , arg2 )"

Reports joint/predicate/argument accuracy and out-of-signature hallucination
rate on the test split.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from transformers import T5ForConditionalGeneration, T5TokenizerFast

from ginsign.grounding_dataset import GroundingDataset
from ginsign.signature_builder import load_cluster_map
from ginsign.signature_io import load_signature


def serialize_signature_generic(sig):
    parts = ["predicates: " + " , ".join(sig.predicates)]
    for sort in sig.sorts:
        consts = sig.constants_of(sort)
        parts.append(f"{sort}: " + " , ".join(consts))
    return " | ".join(parts)


class S2SData(Dataset):
    def __init__(self, examples, sig_str, tok, max_src=512, max_tgt=48):
        self.items = []
        for ex in examples:
            src = f"ground: {ex.ap_text} | signature: {sig_str}"
            tgt = ex.gold_pred + " ( " + " , ".join(ex.gold_args) + " )"
            self.items.append((src, tgt))
        self.tok, self.max_src, self.max_tgt = tok, max_src, max_tgt

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, tok, max_src, max_tgt):
    srcs, tgts = zip(*batch)
    enc = tok(list(srcs), padding=True, truncation=True, max_length=max_src,
              return_tensors="pt")
    lab = tok(list(tgts), padding=True, truncation=True, max_length=max_tgt,
              return_tensors="pt").input_ids
    lab[lab == tok.pad_token_id] = -100
    return enc, lab, list(tgts)


def parse_atom(s):
    s = s.strip()
    if "(" not in s:
        return s.strip(), []
    pred, rest = s.split("(", 1)
    rest = rest.rsplit(")", 1)[0]
    args = [a.strip() for a in rest.split(",") if a.strip()]
    return pred.strip(), args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--signature", required=True)
    ap.add_argument("--cluster-map", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--max-src", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    sig = load_signature(args.signature)
    cm, ds_drop = None, set()
    if args.cluster_map:
        cm, _, ds_drop = load_cluster_map(Path(args.cluster_map))

    train_ds_raw = GroundingDataset(args.train, sig, cluster_map=cm, drop_set=ds_drop)
    test_ds_raw = GroundingDataset(args.test, sig, cluster_map=cm, drop_set=ds_drop)
    sig_str = serialize_signature_generic(sig)
    print(f"signature serialization length (chars): {len(sig_str)}")

    resume_dir = Path(args.out).with_suffix("").parent / f"{args.domain}_model"
    src_model = str(resume_dir) if (resume_dir / "config.json").exists() else "t5-base"
    if src_model != "t5-base":
        print(f"resuming from saved model {src_model} (skipping training)")
    tok = T5TokenizerFast.from_pretrained(src_model)
    model = T5ForConditionalGeneration.from_pretrained(src_model).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"t5-base params: {n_params/1e6:.0f}M, train APs: {len(train_ds_raw)}, "
          f"test APs: {len(test_ds_raw)}")

    train_data = S2SData(train_ds_raw.examples, sig_str, tok)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tok, args.max_src, 48))

    if src_model == "t5-base":
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        opt = AdamW(model.parameters(), lr=args.lr)
        total = len(loader) * args.epochs
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: max(0.0, 1 - s / total))

        model.train()
        step = 0
        recent = []          # last-200-step loss window for early stopping
        converged = False
        for ep in range(args.epochs):
            for enc, lab, _ in loader:
                enc = {k: v.cuda() for k, v in enc.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**enc, labels=lab.cuda()).loss
                loss.backward()
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                recent.append(loss.item())
                if len(recent) > 200:
                    recent.pop(0)
                if step % 100 == 0:
                    print(f"  step {step}/{total} loss={loss.item():.4f}", flush=True)
                if step >= 500 and len(recent) == 200 and sum(recent) / 200 < 1e-3:
                    print(f"  early stop at step {step}: mean loss over last "
                          f"200 steps = {sum(recent)/200:.2e}", flush=True)
                    converged = True
                    break
            if converged:
                break

        # save so eval can be redone without retraining
        save_dir = resume_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)

    # eval
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    torch.cuda.empty_cache()
    model.eval()
    valid_preds = set(sig.predicates)
    all_consts = set()
    for sort in sig.sorts:
        all_consts.update(sig.constants_of(sort))

    test_data = S2SData(test_ds_raw.examples, sig_str, tok)
    tloader = DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False,
                         collate_fn=lambda b: collate(b, tok, args.max_src, 48))
    n = pred_ok = arg_ok = joint_ok = halluc_pred = halluc_arg = 0
    idx = 0
    with torch.no_grad():
        for enc, _, tgts in tloader:
            enc = {k: v.cuda() for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen = model.generate(**enc, max_length=48)
            outs = tok.batch_decode(gen, skip_special_tokens=True)
            for out, tgt in zip(outs, tgts):
                gp, ga = parse_atom(tgt)
                pp, pa = parse_atom(out)
                n += 1
                if pp == gp: pred_ok += 1
                if pa == ga: arg_ok += 1
                if pp == gp and pa == ga: joint_ok += 1
                if pp not in valid_preds: halluc_pred += 1
                if any(a not in all_consts for a in pa): halluc_arg += 1
                idx += 1
    res = {
        "domain": args.domain, "n": n,
        "pred_acc": pred_ok / n, "arg_acc": arg_ok / n, "joint_acc": joint_ok / n,
        "halluc_pred_rate": halluc_pred / n, "halluc_arg_rate": halluc_arg / n,
        "params_millions": n_params / 1e6,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
