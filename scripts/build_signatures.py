#!/usr/bin/env python
"""Thin CLI wrapper around ginsign.signature_builder.

Usage:
  python scripts/build_signatures.py vltl --domain warehouse \
      --corpus data/VLTL-Bench/total/warehouse.jsonl \
      --out    data/signatures/warehouse.json

  python scripts/build_signatures.py priorwork --name navi \
      --corpus data/priorwork_cleaned/total/navi.jsonl \
      --out    data/signatures/navi.json \
      --cluster-map data/signatures/navi_clusters.json
"""

from ginsign.signature_builder import main

if __name__ == "__main__":
    main()
