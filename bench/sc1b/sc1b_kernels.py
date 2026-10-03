#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc1b_kernels.py -- lane SC1b (#846), read time, DESCRIPTIVE only (no gate reads it): which kernels make up one class's
in-graph time on one arm. It reads a node-mode export and the arm record `sc1b_census.py arm` wrote from it, and lists the
kernels the arm's own name map put mostly in CLASS, by per-step time over the modal-size replays.

  sc1b_kernels.py --node census_<engine>_b<B>_node.sqlite --arm sc1b_arm_<engine>_b<B>.json --cls norm_elem [--top 8]
  sc1b_kernels.py --self-test
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sqlite3
import sys
import tempfile


def breakdown(node_db: str, arm: dict, cls: str, top: int = 8) -> dict:
    nm = arm["node"]["name_map"]
    names = {n for n, c in nm.items() if c.get(cls) and c.get(cls, 0) >= max(c.values())}
    con = sqlite3.connect(f"file:{node_db}?mode=ro", uri=True)
    sid = dict(con.execute("SELECT id, value FROM StringIds"))
    rows = con.execute("SELECT start, end, correlationId, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL "
                       "WHERE graphNodeId IS NOT NULL AND graphNodeId != 0").fetchall()
    con.close()
    dur, cnt = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    for st, en, corr, sn in rows:
        n = sid.get(sn, "?")
        dur[corr][n] += en - st
        cnt[corr][n] += 1
    modal = collections.Counter(sum(c.values()) for c in cnt.values()).most_common(1)[0][0]
    reps = [k for k in dur if sum(cnt[k].values()) == modal]
    tot, calls = collections.Counter(), collections.Counter()
    for k in reps:
        for n in names & set(dur[k]):
            tot[n] += dur[k][n]
            calls[n] += cnt[k][n]
    nrep = max(1, len(reps))
    out = [{"kernel": n, "ms_per_step": round(v / nrep / 1e6, 4), "calls_per_step": round(calls[n] / nrep, 1)} for n, v in tot.most_common(top)]
    return {"engine": arm.get("engine"), "batch": arm.get("batch"), "class": cls, "replays": len(reps),
            "class_term_ms": (arm.get("terms") or {}).get(cls), "listed_ms": round(sum(r["ms_per_step"] for r in out), 4), "kernels": out}


def self_test() -> int:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "n.sqlite")
    con = sqlite3.connect(p)
    con.executescript("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);"
                      "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, correlationId INTEGER, shortName INTEGER, graphNodeId INTEGER);")
    con.executemany("INSERT INTO StringIds VALUES (?, ?)", [(1, "indexSelectSmallIndex"), (2, "gemv"), (3, "rmsnorm")])
    for corr in range(4):
        t = corr * 10_000_000
        con.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)",
                        [(t, t + 6000, corr, 1, 1), (t + 7000, t + 13000, corr, 1, 2), (t + 20000, t + 220000, corr, 2, 3),
                         (t + 230000, t + 240000, corr, 3, 4)])
    con.commit()
    con.close()
    arm = {"engine": "e4b", "batch": 16, "terms": {"norm_elem": 0.022},
           "node": {"name_map": {"indexSelectSmallIndex": {"norm_elem": 8}, "gemv": {"dense_gemm": 4}, "rmsnorm": {"norm_elem": 4}}}}
    r = breakdown(p, arm, "norm_elem")
    assert [k["kernel"] for k in r["kernels"]] == ["indexSelectSmallIndex", "rmsnorm"], r
    assert r["kernels"][0] == {"kernel": "indexSelectSmallIndex", "ms_per_step": 0.012, "calls_per_step": 2.0} and r["replays"] == 4, r
    print("self-test OK (2 checks)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--node")
    ap.add_argument("--arm")
    ap.add_argument("--cls", default="norm_elem")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    with open(a.arm) as f:
        arm = json.load(f)
    print(json.dumps(breakdown(a.node, arm, a.cls, a.top), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
