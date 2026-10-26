"""bench/p237/p237_egress_probe.py -- lane P237, BOX side: refuse a host whose download cannot carry the pinned model, before the
install starts (PREREG "The host floors": a download probe of at least 40 MB/s).

The path measured is the run's own: the Hugging Face resolve URL of the pinned revision's first safetensors shard (its name read from
the Hub's file list at that revision, never typed here), read for at most WINDOW_S seconds. The model is ~61 GB, so at 40 MB/s the
fetch takes about 25 minutes, inside the runner's fetch alarm.

Exit 0 = at or above the floor; 4 = MEASURED below it (p237_run.sh turns it into the launcher's egress code 14, which excludes the
machine); 2 = no transfer at all (DNS, a refused connection, an HTTP error, no shard listed): not a measurement of this host, so the
runner sends it to the harness code 9 and excludes nothing. DQ3's probe (bench/dq3/dq3_egress_probe.py) is the form.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

FLOOR_BPS = 40_000_000
WINDOW_S = 30.0
SLOW, NO_TRANSFER = 4, 2


def first_shard(model, revision, opener=urllib.request.urlopen):
    with opener(f"https://huggingface.co/api/models/{model}/revision/{revision}", timeout=20) as resp:
        files = sorted(s.get("rfilename", "") for s in json.load(resp).get("siblings") or [])
    shards = [f for f in files if f.endswith(".safetensors")]
    return shards[0] if shards else None


def measure(url, opener=urllib.request.urlopen, clock=time.monotonic, window=WINDOW_S):
    """(bytes, seconds) read from ``url`` within ``window``; raises only when nothing at all arrived."""
    t0, got = clock(), 0
    try:
        with opener(url, timeout=20) as resp:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                got += len(chunk)
                if clock() - t0 > window:
                    break
    except (urllib.error.URLError, OSError):
        if got == 0:
            raise
    return got, max(clock() - t0, 1e-3)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: p237_egress_probe.py MODEL REVISION", file=sys.stderr)
        return NO_TRANSFER
    model, revision = argv
    try:
        shard = first_shard(model, revision)
        if shard is None:
            print(f"EGRESS_PROBE_ERROR no safetensors shard listed for {model} @ {revision}")
            return NO_TRANSFER
        got, dt = measure(f"https://huggingface.co/{model}/resolve/{revision}/{shard}")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"EGRESS_PROBE_ERROR no transfer: {exc!r}"[:300])
        return NO_TRANSFER
    bps = got / dt
    line = f"{got / 1e6:.1f} MB in {dt:.1f} s = {bps / 1e6:.2f} MB/s from huggingface.co ({shard}; floor {FLOOR_BPS / 1e6:.0f})"
    if bps < FLOOR_BPS:
        print(f"EGRESS_PROBE_SLOW {line}")
        return SLOW
    print(f"EGRESS_PROBE_OK {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
