#!/usr/bin/env python3
"""Compatibility wrapper for the packaged OPUS-100 public-corpora demo."""

from __future__ import annotations

from tame_mt.demos.opus100 import main

if __name__ == "__main__":
    raise SystemExit(main())
