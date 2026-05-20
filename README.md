# GinSign

**Grounding Natural Language into System Signatures for Temporal Logic.**

GinSign translates natural-language specifications into *grounded* Linear
Temporal Logic (LTL) formulas by conditioning on a system signature
(sorts, predicates, constants). It frames grounding as a hierarchical
pointer-classification problem over the signature rather than free-form
generation, so a small encoder model can handle open-set domains without
retraining.

This repository hosts the ACL resubmission codebase, including:

- A pointer-head grounder with autoregressive joint argument decoding
  (`src/ginsign/ptr_grounder.py`).
- A canonical Signature schema with sort, predicate, and union-sort
  support (`src/ginsign/signature_io.py`).
- Corpus-driven signature builders for VLTL-Bench and four prior-work
  datasets — cleanup\_world, GLTL, conformal, navi
  (`src/ginsign/signature_builder.py`).
- A priorwork cleaner that strips parse artifacts and clusters navi's
  108 paraphrastic predicates down to 3 canonical verbs
  (`src/ginsign/priorwork_cleaner.py`, `data/signatures/navi_clusters.json`).
- A PyTorch `GroundingDataset` adapter (`src/ginsign/grounding_dataset.py`).

## Repository layout

```
src/ginsign/        Python package (pip install -e .)
tests/              Unit + integration tests (pytest)
scripts/            Thin CLI entry points
data/
  VLTL-Bench/        primary benchmark (English et al., 2025)
  priorwork/         raw cleanup_world, GLTL, conformal, navi
  priorwork_cleaned/ post-cleaner output (regeneratable)
  signatures/       canonical JSON signatures + navi cluster map
paper/              ICML 2026 submission PDF
docs/               rebuttal strategy + todo notes
notebooks/legacy/   experiment notebooks from prior iterations
```

`archive/` and `deprecated/` hold pre-restructure artifacts and are
gitignored.

## Install

```bash
pip install -e .[dev]
```

## Build the signatures from scratch

```bash
# VLTL-Bench (uses the canonical TYPE_CONSTANTS dict for type lookups)
for d in warehouse traffic_light search_and_rescue; do
  python scripts/build_signatures.py vltl \
      --domain $d \
      --corpus data/VLTL-Bench/total/$d.jsonl \
      --out    data/signatures/$d.json
done

# Priorwork (cleaning first, then signature synthesis)
python scripts/clean_priorwork.py \
    --in-root  data/priorwork \
    --out-root data/priorwork_cleaned

for ds in cleanup_world GLTL conformal; do
  python scripts/build_signatures.py priorwork \
      --name $ds \
      --corpus data/priorwork_cleaned/total/$ds.jsonl \
      --out    data/signatures/$ds.json
done

python scripts/build_signatures.py priorwork \
    --name navi \
    --corpus data/priorwork_cleaned/total/navi.jsonl \
    --out    data/signatures/navi.json \
    --cluster-map data/signatures/navi_clusters.json
```

All 7 builds report 100% corpus coverage by construction.

## Tests

```bash
pytest tests/
```

Tier-1 tests (pure tensor / data) run anywhere. Tier-2 integration tests
that exercise the BERT encoder require `bert-base-cased` in the local
Hugging Face cache; they auto-skip otherwise.

## Status

- Pointer-head + AR decoder grounder: implemented, 12/12 tests passing.
- Data layer (signatures, cleaner, dataset): implemented, 19/19 tests
  passing, 7 signatures generated with full coverage.
- Training loop on the new architecture: **not yet implemented** — next
  pass.
