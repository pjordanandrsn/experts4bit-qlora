# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The box's path to PyPI's file host, timed before any PyPI-heavy venv is built (#1767; bench/tc1/AXO1767-PREREG.md).

axo1767-5090-2's host fetched from files.pythonhosted.org at 0.88 MB/s while download.pytorch.org gave it 23-66 MB/s, so its
venv-axolotl install took 31 min of its 45-min (2,700 s) alarm, against 37 s for venv-e4b. One fixed, known-size wheel (pinned by size and sha256) is downloaded for at
most --cap-s seconds; the rate is bytes over wall time. Exit 0 when the rate clears --min-mbps (and a complete download matches
its sha256), 3 below the floor, 4 on a sha256 mismatch or a short complete read, 5 when nothing arrives. One JSON line either way.
    python3 pypi_probe.py --min-mbps 10 [--url URL --size N --sha256 HEX] [--cap-s 60]
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request

URL = ("https://files.pythonhosted.org/packages/df/64/ff35eb9e54894cf471ff4716abd3c81eb0a0626869217ce3e6ba4ccf17d7/"
       "scipy-1.18.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl")
SIZE = 35344199
SHA256 = "f55fa87b6c612ecd6b058f167c53231b1d14e412efe361d3d6e38b3631c73218"


def probe(url, size, sha256, min_mbps, cap_s, chunk=1 << 20):
    h, got, t0 = hashlib.sha256(), 0, time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=min(30.0, cap_s)) as r:
            while got < size and time.monotonic() - t0 < cap_s:
                b = r.read(chunk)
                if not b:
                    break
                h.update(b)
                got += len(b)
    except Exception as e:                                  # a host that cannot reach PyPI at all is below any floor
        rec = {"pypi_probe": 1, "url": url, "bytes": got, "seconds": round(time.monotonic() - t0, 2), "error": f"{type(e).__name__}: {e}"[:200]}
        return (5 if got == 0 else 3), rec
    dt = max(time.monotonic() - t0, 1e-6)
    mbps = got / dt / 1e6
    rec = {"pypi_probe": 1, "url": url, "bytes": got, "seconds": round(dt, 2), "mb_per_s": round(mbps, 2), "min_mb_per_s": min_mbps,
           "complete": got >= size}
    if got >= size:
        rec["sha256_ok"] = got == size and h.hexdigest() == sha256
        if not rec["sha256_ok"]:
            return 4, rec
    if got == 0:
        return 5, rec
    return (0 if mbps >= min_mbps else 3), rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mbps", type=float, required=True)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--sha256", default=SHA256)
    ap.add_argument("--cap-s", type=float, default=60.0)
    a = ap.parse_args(argv)
    rc, rec = probe(a.url, a.size, a.sha256, a.min_mbps, a.cap_s)
    rec["verdict"] = {0: "OK", 3: "BELOW_FLOOR", 4: "BAD_BYTES", 5: "NO_BYTES"}[rc]
    print("PYPI PROBE " + json.dumps(rec), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
