#!/usr/bin/env python
"""Thin CLI wrapper around ginsign.priorwork_cleaner.

Usage:
  python scripts/clean_priorwork.py \
      --in-root  data/priorwork \
      --out-root data/priorwork_cleaned
"""

from ginsign.priorwork_cleaner import main

if __name__ == "__main__":
    main()
