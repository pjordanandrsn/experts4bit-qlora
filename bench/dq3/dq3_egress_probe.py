"""bench/dq3/dq3_egress_probe.py -- lane DQ3, BOX side: refuse a host whose GitHub egress cannot carry the pinned install,
before the install starts. Exit 0 = fast enough; exit 4 = measured too slow (dq3_run.sh turns it into the registered egress
refusal, rc 14, machine evidence for --exclude-vast-lane-receipt); exit 2 = no transfer at all (DNS, refused connection,
HTTP error): not a measurement of this host's egress, so the runner sends it to the harness rc and excludes nothing.

Why it exists: dq3-5090-2 (machine 147454, Shanghai) passed every check, then cloned grouped-nf4-gemm at ~33 KB/s; the
115 MB experts4bit-qlora clone could never finish inside pip's 30-min alarm. The install is ~135 MB of git plus PyPI wheels.

What is measured is the install's own path: GitHub's codeload tarball of the pinned grouped-nf4-gemm commit, read for at
most WINDOW_S seconds. Floor 1 MB/s: the git part of the install then takes ~2-3 min, as on the hosts that worked.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request

FLOOR_BPS = 1_000_000
WINDOW_S = 30.0
SLOW, NO_TRANSFER = 4, 2


def main() -> int:
    sha = os.environ.get("GNF4_SHA") or "a5edec8789735bff1c0da4708ae5fc93260a1410"
    url = f"https://codeload.github.com/pjordanandrsn/grouped-nf4-gemm/tar.gz/{sha}"
    t0 = time.monotonic()
    got = 0
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                got += len(chunk)
                if time.monotonic() - t0 > WINDOW_S:
                    break
    except (urllib.error.URLError, OSError) as exc:
        if got == 0:
            print(f"EGRESS_PROBE_ERROR no transfer: {exc!r}"[:300])
            return NO_TRANSFER
    dt = max(time.monotonic() - t0, 1e-3)
    bps = got / dt
    line = f"{got / 1e6:.1f} MB in {dt:.1f} s = {bps / 1e6:.2f} MB/s from codeload.github.com (floor {FLOOR_BPS / 1e6:.1f})"
    if bps < FLOOR_BPS:
        print(f"EGRESS_PROBE_SLOW {line}")
        return SLOW
    print(f"EGRESS_PROBE_OK {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
