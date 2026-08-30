#!/usr/bin/env python
"""Measure GinSign inference latency (rebuttal KUw9 W2).

Reports per-AP latency (batch=1 and batched) on the test split, and latency
as a function of signature size N (constants inflated with distractors from
donor domains, shard size m fixed).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ginsign.eval import evaluate
from ginsign.grounding_dataset import GroundingDataset, make_collate_fn
from ginsign.ptr_grounder import PointerJointGrounder
from ginsign.signature_io import load_signature
from ginsign.signature_transforms import inject_distractors


def load_model(ckpt_path, device, max_per_shard=80):
    model = PointerJointGrounder(bert_name="bert-base-cased",
                                 max_per_shard=max_per_shard).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {k.replace("._orig_mod.", "."): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(sd)
    model.eval()
    return model


def timed_eval(model, corpus, sig, device, batch_size, limit=500):
    ds = GroundingDataset(corpus, sig)
    if limit and len(ds.examples) > limit:
        ds.examples = ds.examples[:limit]
    n = len(ds.examples)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=make_collate_fn(sig))
    evaluate(model, loader, sig, device)  # warmup pass
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    evaluate(model, loader, sig, device)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt / n * 1000, n  # ms per AP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="warehouse")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--scaling-batch", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda")
    dom = args.domain
    corpus = str(ROOT / f"data/VLTL-Bench/test/{dom}.jsonl")
    sig = load_signature(str(ROOT / f"data/signatures/{dom}.json"))
    import os
    ckpt_root = Path(os.environ.get("GINSIGN_CKPT_ROOT", str(ROOT / "outputs")))
    model = load_model(ckpt_root / dom / "best.pt", device)

    results = {"domain": dom, "gpu": torch.cuda.get_device_name(0),
               "params_millions": sum(p.numel() for p in model.parameters()) / 1e6}

    for bs in args.batches:
        ms, n = timed_eval(model, corpus, sig, device, bs)
        results[f"ms_per_ap_batch{bs}"] = ms
        print(f"batch={bs}: {ms:.2f} ms/AP over {n} APs", flush=True)

    # latency vs signature size N (inject distractor constants from other domains)
    donor_names = [d for d in ["warehouse", "traffic_light", "search_and_rescue",
                               "cleanup_world", "GLTL", "conformal", "navi"]
                   if d != dom]
    donors = [load_signature(str(ROOT / f"data/signatures/{n}.json"))
              for n in donor_names]
    scaling = []
    for n_per_sort in (0, 40, 80, 160, 320):
        s = sig if n_per_sort == 0 else inject_distractors(
            sig, donors, n_per_sort=n_per_sort, seed=42)
        n_const = sum(len(s.constants_of(x)) for x in s.sorts)
        ms, _ = timed_eval(model, corpus, s, device, args.scaling_batch)
        scaling.append({"n_distractors_per_sort": n_per_sort,
                        "total_constants": n_const,
                        f"ms_per_ap_batch{args.scaling_batch}": ms})
        print(f"N={n_const} constants: {ms:.2f} ms/AP", flush=True)
    results["latency_vs_signature_size"] = scaling

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
