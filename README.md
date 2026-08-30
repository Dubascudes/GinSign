# GinSign

**Grounding Natural Language Into System Signatures for Temporal Logic Translation.**

GinSign is a modular grounding stage that attaches to any NL-to-TL lifter. It
casts atomic-proposition grounding as signature-conditioned hierarchical
classification: the system signature (sorts, predicates, typed constants) is
enumerated as an input prefix, and a single encoder-only model (BERT-base,
~124M params) selects the predicate and then each typed argument. Because
predictions are selections from the enumerated signature rather than free-form
generations, the out-of-signature rate is 0% by construction.

Paper: *GinSign: Grounding Natural Language Into System Signatures for Temporal
Logic Translation* (ACL ARR May 2026, submission 11103).
Preprint: https://arxiv.org/abs/2512.16770

## Install

```bash
pip install -e .            # core (torch, transformers, datasets)
pip install -e ".[all]"     # + wandb, openai (for LLM-baseline scripts)
pytest                      # sanity-check the install
```

## Repository layout

```
src/ginsign/            Python package: signature schema, pointer grounder,
                        dataset adapters, prefix sharding, priorwork cleaner
scripts/                CLI entry points (training, eval, baselines, tables)
tests/                  pytest suite
data/
  signatures/           System signatures for all 7 domains (+ navi cluster map)
  VLTL-Bench/           Warehouse / Traffic Light / Search & Rescue splits
  priorwork/            Raw Cleanup World, GLTL, Conformal, Navi corpora
  priorwork_cleaned/    Cleaned corpora (regenerable: scripts/clean_priorwork.py)
  rebuttal_splits/      Template- and surface-disjoint re-splits (WH, TL)
eval_data/
  grounding_eval/       Per-AP grounding predictions for every baseline
                        (GPT-3.5/4o/4.1-mini/4o-mini/5, Claude Sonnet,
                        GPT-oss-120B, Llama-3.1-8B, Lang2LTL, few-shot,
                        distractor-50 conditions)
  translation_eval/     Lifted-translation outputs per backbone (LLM direct,
                        NL2TL, GraFT/gt_lifting, nl2spec, nl2ltl) + ground truth
  lifting_eval/         Lifting-stage outputs
  t5_translation_evaluation/  Fine-tuned t5-base translation outputs
results/
  <domain>_predictions.jsonl  GinSign per-AP predictions (feeds the GLE table)
  summary.json                Headline in-domain grounding metrics
  training_summary.json       Best/final metrics per training run
  rebuttal/                   Rebuttal-period experiment outputs (see below)
```

## Training

One domain (~minutes on a single consumer GPU):

```bash
python scripts/train.py \
    --signature data/signatures/warehouse.json \
    --train     data/VLTL-Bench/train/warehouse.jsonl \
    --dev       data/VLTL-Bench/test/warehouse.jsonl \
    --output    outputs/warehouse
```

Navi needs its predicate cluster map (108 paraphrastic verbs -> 3 canonical):

```bash
python scripts/train.py \
    --signature data/signatures/navi.json \
    --train     data/priorwork_cleaned/train/navi.jsonl \
    --dev       data/priorwork_cleaned/test/navi.jsonl \
    --cluster-map data/signatures/navi_clusters.json \
    --output    outputs/navi
```

All seven domains: `scripts/train_all.sh`.

Hyperparameters (paper App. A.8): `bert-base-cased`, lr 2e-5, batch 16,
3 epochs, shard size m=80.

## Reproducing the paper's tables

| Paper table | Command / source |
|---|---|
| Isolated grounding accuracy (Table 2) | `python scripts/summarize_results.py` (reads `eval_data/grounding_eval/` + `results/*_predictions.jsonl`) |
| OOS + unparseable rates (Table 3, App.) | same inputs; OOS = prediction contains a symbol outside `data/signatures/<domain>.json` |
| GLE end-to-end (Table 4) | `python scripts/summarize_results.py` (crosses `eval_data/translation_eval/` backbones with each grounder) |
| Leave-one-out (Table 5) | `python scripts/run_ablations.py --axes loo` (retrains; LOO co-training sets are built on the fly) |
| Held-out signature elements (Table 6) | `python scripts/run_ablations.py --axes joint` |
| Lifting eval | `python scripts/rebuild_tables.py` |
| Runtime / latency (App.) | `python scripts/measure_latency.py`; stored: `results/rebuttal/rebuttal_latency/` |
| Disjoint re-splits (App.) | `python scripts/make_disjoint_splits.py` then `scripts/train.py` on `data/rebuttal_splits/`; stored: `results/rebuttal/rebuttal_disjoint/` |
| Distractor / shuffle / partial-prefix (App.) | `python scripts/run_ablations.py --axes prefix`; stored: `results/rebuttal/ablations/` |
| Seed variance | `results/rebuttal/seed_variance/` (5 seeds x {WH, S&R, CF}) |
| Shard-size sweep m in {10..160} | `results/rebuttal/rebuttal_shard_sweep/` |
| T5-base seq2seq baseline | `python scripts/seq2seq_baseline.py`; stored: `results/rebuttal/rebuttal_seq2seq/` |
| Few-shot GPT-4o (k=5) | `python scripts/fewshot_grounding.py --k 5` (needs `OPENAI_API_KEY`); stored: `eval_data/grounding_eval/gpt-4o-fewshot/` |

Every stored eval JSON ships with the `config.json` / `metrics.jsonl` of the
run that produced it, so numbers can be checked without retraining. Scripts
that call OpenAI/Anthropic APIs read keys from the environment; nothing in
this repository requires an API key to reproduce the paper's GinSign numbers.

## Trained checkpoints

Checkpoints (~500 MB each) exceed GitHub file limits and are not in this
repository. Training from scratch reproduces them in minutes per domain with
the commands above (seed-variance across 5 seeds: max s.d. 0.04 joint-accuracy
points). [TODO: add Hugging Face Hub link if/when uploaded.]

## Datasets and licensing

VLTL-Bench (Warehouse, Traffic Light, Search & Rescue) is from English et al.,
2025. The four prior-work corpora (Cleanup World, GLTL, Conformal, Navi) are
redistributed from their original releases for research use; see the paper's
Section 5.1 for citations. `data/priorwork_cleaned/` is derived from
`data/priorwork/` via `scripts/clean_priorwork.py`.

Code is MIT-licensed (see LICENSE).

## Citation

```bibtex
[TODO: replace with the final ACL BibTeX entry]
@misc{english2025ginsign,
  title  = {GinSign: Grounding Natural Language Into System Signatures
            for Temporal Logic Translation},
  author = {English, William H and Walker, Chase and Simon, Dominic
            and Ewetz, Rickard},
  year   = {2025},
  eprint = {2512.16770},
  archivePrefix = {arXiv}
}
```
