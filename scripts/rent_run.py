#!/usr/bin/env python3
"""Thin wrapper: ``python scripts/rent_run.py`` == ``python -m experts4bit_qlora.tools.rent``."""
from __future__ import annotations

from experts4bit_qlora.tools.rent import main

if __name__ == "__main__":
    raise SystemExit(main())
