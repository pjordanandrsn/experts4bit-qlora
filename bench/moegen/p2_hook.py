#!/usr/bin/env python3
"""MG1 amendment 3: run tp1's arm driver unchanged and record grouped-nf4-gemm's DGRAD_STATS when it exits.

usage: MG1_P2_OUT=<census.json> python p2_hook.py <driver.py> <driver args...>

The driver runs as ``__main__`` through ``runpy``, with ``sys.argv`` as if it had been started directly, so its bytes,
arguments and code path are tp1's. The census holds the driver's sha256, so a reader can check that it ran tp1's file.

The census is written by an ``atexit`` hook. Every exit that runs interpreter shutdown records it: a normal return, or
``sys.exit`` from one of the driver's stubs. ``os._exit``, a signal or the box's alarm does not, and then the census file is
absent. That absence is a row, never a guess. The counters are read from ``sys.modules`` at exit, so the hook imports nothing
before the driver does: grouped-nf4-gemm's ``nf4_qlora`` (``DGRAD_STATS``) and this package's ``experts4bit_qlora.engines.fast``
(``FAST_TRAIN_STATS``) are the modules the driver itself loaded.
"""
import atexit
import hashlib
import json
import os
import runpy
import sys


def _plain(v):
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v if isinstance(v, (int, float, str, bool)) or v is None else repr(v)


def census(driver_sha, argv):
    nq = sys.modules.get("nf4_qlora")
    fast = sys.modules.get("experts4bit_qlora.engines.fast")
    dg = getattr(nq, "DGRAD_STATS", None)
    ft = getattr(fast, "FAST_TRAIN_STATS", None)
    return {"driver_sha256": driver_sha, "argv": argv,
            "dgrad": _plain(dg) if isinstance(dg, dict) else None,
            "fast_train": _plain(ft) if isinstance(ft, dict) else None,
            "reason": None if isinstance(dg, dict) else "nf4_qlora was never imported, or carries no DGRAD_STATS"}


def main():
    out = os.environ["MG1_P2_OUT"]
    driver = sys.argv[1]
    with open(driver, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    argv = [driver] + sys.argv[2:]

    def dump():
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(census(sha, argv), f, indent=1)
        os.replace(tmp, out)

    atexit.register(dump)
    sys.argv = argv
    runpy.run_path(driver, run_name="__main__")


if __name__ == "__main__":
    main()
