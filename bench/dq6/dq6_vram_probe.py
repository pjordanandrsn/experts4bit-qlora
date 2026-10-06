"""bench/dq6/dq6_vram_probe.py -- lane DQ6, BOX side: DQ3's host-floor probe (``bench/dq3/dq3_vram_probe.py``, staged beside
this file), unchanged but for the floor. DQ3's 28 GiB is above a 24 GB card's whole memory and would refuse every RTX 4090;
21.5 GiB (3.5 GiB, then 2 GiB blocks) is above DQ6's predicted resident peak at its boundary (bench/dq6/DQ6-PREREG.md) and
below the ~23.5 GiB a healthy 4090 hands out. Exit codes are DQ3's: 0 fits, 3 the GPU refused memory (the runner's rc 18,
machine evidence), anything else a harness error.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "dq3")]     # staged flat on the box; bench/dq3/ in the repo
import dq3_vram_probe as P  # noqa: E402

TOTAL_GIB = 21.5

if __name__ == "__main__":
    P.TOTAL_GIB = TOTAL_GIB
    sys.exit(P.main())
