#!/usr/bin/env python3
"""p236_reduce.py -- lane P236's registered rule (bench/p236/PREREG-p236.md) and the classification of every fetched box.

P236 is DESCRIPTIVE: an op-level head-to-head of grouped-nf4-gemm 0.51.0's decode GEMV (K33, auto, as experts4bit-qlora
serves it: one singleton group per row) against bitsandbytes 0.50.2's ``matmul_4bit`` (its ``bitsandbytes::gemm_4bit``
op: bitsandbytes' fused ``cgemm_4bit`` kernel, or dequantize then ``F.linear``, by bitsandbytes' own per-arch rule), on
the same NF4 bytes, at M = 1, 2, 4, 8 and 16 rows, on six expert projections
(Mixtral-8x7B, Qwen3-30B-A3B and OLMoE-1B-7B, gate_up and down), on one RTX 5090. No license, no default moves.

The verdict is READ or VOID. VOID when any of these holds (each read fail-closed: a missing record is a fault):
  - the record is missing, a dry run, or not ``ok``; another grouped-nf4-gemm or bitsandbytes version, another torch,
    another card class;
  - the RELEASE is not proven (the maintainer's rule for every "at the released pins" row, 2026-10-24): each library's
    installed dist must be its release, by :func:`provenance_faults` -- an index wheel (no ``direct_url.json``, or an
    archive URL on PyPI's hosts) at the registered version, or, for grouped-nf4-gemm only, a git install whose
    ``vcs_info.commit_id`` is the registered release commit. An editable install, a ``file://`` install, another commit
    or a git install of bitsandbytes is refused; the box's tripwire runs the same check before anything is timed;
  - a registered cell (projection x M) is missing, or has another rep count or another ITERS;
  - the premise: bitsandbytes' ``dequantize_4bit`` is not bit-equal to grouped-nf4-gemm's ``dequant_ref`` on a cell's
    bytes (they would not be the same weights);
  - correctness: either arm's output differs from the fp32 reference (x @ dequant(W)^T) by more than TOL of its max |ref|;
  - engagement, counted in an eager pass per cell: the gnf4 arm makes exactly one ``bw_prmt32`` launch per call and no
    other decode-GEMV route; the bitsandbytes arm makes one ``gemm_4bit`` dispatch per call, every one to the same
    route (``custom``, its fused kernel, or ``dequant``, its dequantize-then-linear fallback), which is recorded and
    reported, never required: it is bitsandbytes' own choice at that shape on that card;
  - a timing sample is missing, non-positive or not finite.
READ otherwise. Reported per cell (never a bar): the ratio bnb / gnf4 of the per-call times (> 1: gnf4 is faster), its
median and range over the reps, and both arms' self-pair bands (the second visit over the first, per rep).

    python p236_reduce.py --bench bench_p236.json --out verdict_p236.json
    python p236_reduce.py --classify --work W
    python p236_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
import tempfile

LANE = "P236"
#: (family, projection) -> (N, K): each expert projection's output and input width (gate_up fused, as served)
SHAPES = {"mixtral.gate_up": (28672, 4096), "mixtral.down": (4096, 14336),
          "qwen3.gate_up": (1536, 2048), "qwen3.down": (2048, 768),
          "olmoe.gate_up": (2048, 2048), "olmoe.down": (2048, 1024)}
MS = (1, 2, 4, 8, 16)
ARMS = ("gnf4", "bnb")
#: each rep visits the arms in a palindrome; odd reps swap the two arms (bnb gnf4 gnf4 bnb), so each arm leads half the
#: reps
ORDER = ("gnf4", "bnb", "bnb", "gnf4")
REPS, ITERS, WARM = 20, 50, 10
#: rotate enough weight copies per cell that the copies span at least this many bytes (4x the 5090's 96 MB L2)
ROTATE_BYTES = 384 * 1024 * 1024
MAX_COPIES = 512                  # every registered shape spans 4x L2 under this cap (qwen3.down needs 456)
TOL = 1.5e-2                      # the served tolerance (tests/test_nf4_smallm_bw_gpu.py), max |y - ref| / max |ref|
GNF4_VERSION, BNB_VERSION, TORCH_VERSION = "0.51.0", "0.50.2", "2.8.0+cu128"
GNF4_SHA = "30adb1bfa76f95f9ce4a4c29734b8f36de907eca"   # grouped-nf4-gemm v0.51.0: `git rev-list -n1 v0.51.0` on the canonical repo
GNF4_REPO = "github.com/pjordanandrsn/grouped-nf4-gemm"
PYPI_HOSTS = ("files.pythonhosted.org", "pypi.org", "pypi.python.org")
#: each library's registered release: (version, the one git commit accepted, or None when only an index wheel is)
RELEASES = {"grouped-nf4-gemm": (GNF4_VERSION, GNF4_SHA), "bitsandbytes": (BNB_VERSION, None)}
#: the shared release-proof helper (loggetta bench/release_proof.py at d3d84173), staged byte-identical
RELEASE_PROOF_SHA256 = "341dd130b1cf681a86f3fa1aad2b3e1bf9a2e4e6c22388e0e35b5736fc446713"
BNB_ROUTES = ("custom", "dequant")    # bitsandbytes.backends.cuda.ops: _gemm_4bit_kernel_impl / _dequant_linear_fallback
GEMV_KEYS = ("bw_prmt32", "bw_tree", "bw_splitk", "dotpad", "dotpad_splitk", "scalar", "scalar_splitk")
HOST_CODES = {13: "free disk", 15: "card, compute capability, SM count or VRAM", 16: "host RAM", 18: "CUDA unusable"}
FIT_CODES = {41: "the guard does not fit the deadline (a slow boot)"}
KIT_CODES = {9: "staging, install or pins", 10: "torch does not import (the image)",
             28: "the image's torch is not the pinned build", 78: "run id, environment or frozen knobs", 130: "a signal"}


def dist_record(name):
    """``{"version", "direct_url"}`` of one installed dist, as :func:`provenance_faults` reads it (``direct_url`` None
    when pip wrote none: an install from an index)."""
    import importlib.metadata as md
    d = md.distribution(name)
    raw = d.read_text("direct_url.json")
    return {"version": d.version, "direct_url": json.loads(raw) if raw else None}


def provenance_faults(dists):
    """The release proof, fail closed: every dist in :data:`RELEASES` present, at its version, and installed as the
    release itself (see the module docstring)."""
    from urllib.parse import urlparse
    out = []
    for name, (version, sha) in RELEASES.items():
        d = (dists or {}).get(name) if isinstance(dists, dict) else None
        if not isinstance(d, dict) or "direct_url" not in d:
            out.append(f"{name}: no provenance record ({d!r})")
            continue
        if d.get("version") != version:
            out.append(f"{name}: version {d.get('version')!r}, registered {version}")
        du = d["direct_url"]
        if du is None:
            continue                                          # an index install: the published wheel or sdist
        if not isinstance(du, dict):
            out.append(f"{name}: an unreadable direct_url ({du!r})")
            continue
        url = str(du.get("url") or "")
        if (du.get("dir_info") or {}).get("editable") or url.startswith("file:") or "dir_info" in du:
            out.append(f"{name}: a local or editable install ({url}), not the release")
        elif "vcs_info" in du:
            commit = (du.get("vcs_info") or {}).get("commit_id")
            if sha is None:
                out.append(f"{name}: a git install ({url}@{commit}); registered the published release only")
            elif commit != sha or urlparse(url).netloc + urlparse(url).path.removesuffix(".git") != GNF4_REPO:
                out.append(f"{name}: git {url}@{commit}, registered {GNF4_REPO}@{sha}")
        elif "archive_info" in du:
            if urlparse(url).hostname not in PYPI_HOSTS:
                out.append(f"{name}: an archive from {url}, not the published release")
        else:
            out.append(f"{name}: an install of unknown provenance ({du!r})")
    return out


def release_proof_faults(proofs):
    """bitsandbytes, the index install, must carry release_proof.py's verdict: its installed bytes ARE the wheel PyPI
    publishes for the registered version (grouped-nf4-gemm is git at the release commit: provenance_faults pins it)."""
    p = (proofs or {}).get("bitsandbytes") if isinstance(proofs, dict) else None
    if not isinstance(p, dict):
        return [f"bitsandbytes: no release proof ({p!r})"]
    ok = (p.get("dist") == "bitsandbytes" and p.get("version") == BNB_VERSION and p.get("record") == "RECORD equal"
          and p.get("record_equal") is True and p.get("first_diff") is None
          and str(p.get("wheel", "")).startswith(f"bitsandbytes-{BNB_VERSION}-") and str(p.get("wheel")).endswith(".whl")
          and isinstance(p.get("sha256"), str) and len(p["sha256"]) == 64)
    return [] if ok else [f"bitsandbytes: the release proof does not hold ({p})"]


def cell_key(shape, m):
    return f"{shape}|M{m}"


def order(rep):
    return list(ORDER) if rep % 2 == 0 else [ARMS[1 - ARMS.index(a)] for a in ORDER]


def copies_for(n, k):
    """Weight copies for one projection: packed bytes + fp32 absmax per copy, enough to span ROTATE_BYTES."""
    per = n * k // 2 + n * (k // 64) * 4
    return max(2, min(MAX_COPIES, math.ceil(ROTATE_BYTES / per)))


def _finite_pos(xs):
    return isinstance(xs, list) and xs and all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in xs)


def cell_faults(key, c):
    out = []
    if not isinstance(c, dict):
        return [f"{key}: no record"]
    if c.get("dequant_equal") is not True:
        out.append(f"{key}: bitsandbytes' dequantize_4bit is not bit-equal to dequant_ref on these bytes "
                   f"({c.get('dequant_equal')!r})")
    for arm in ARMS:
        rel = (c.get("rel_err") or {}).get(arm)
        if not isinstance(rel, (int, float)) or not math.isfinite(rel) or rel > TOL:
            out.append(f"{key}: {arm} differs from the fp32 reference by {rel!r} (tolerance {TOL})")
    eng = c.get("engaged") or {}
    g = eng.get("gnf4")
    if not isinstance(g, dict) or any(type(g.get(k_)) is not int for k_ in GEMV_KEYS + ("calls",)):
        out.append(f"{key}: no complete gnf4 dispatch record ({g!r})")
    elif g["bw_prmt32"] != g["calls"] or g["calls"] < 1 or any(g[k_] for k_ in GEMV_KEYS if k_ != "bw_prmt32"):
        out.append(f"{key}: gnf4 engagement {g}, registered one bw_prmt32 launch per call and no other route")
    b = eng.get("bnb")
    if not isinstance(b, dict) or any(type(b.get(k_)) is not int for k_ in BNB_ROUTES + ("calls",)):
        out.append(f"{key}: no complete bitsandbytes route record ({b!r})")
    elif b["calls"] < 1 or sorted(b[k_] for k_ in BNB_ROUTES) != [0, b["calls"]]:
        out.append(f"{key}: bitsandbytes route {b}, registered one gemm_4bit dispatch per call, all to one route")
    reps = c.get("reps")
    if not isinstance(reps, list) or len(reps) != REPS:
        out.append(f"{key}: {len(reps) if isinstance(reps, list) else reps!r} reps, registered {REPS}")
    else:
        for i, r in enumerate(reps):
            if not isinstance(r, dict) or r.get("order") != order(i):
                out.append(f"{key} rep {i}: order {r.get('order') if isinstance(r, dict) else r!r}, "
                           f"registered {order(i)}")
            elif not _finite_pos(r.get("ms_per_call")) or len(r["ms_per_call"]) != len(ORDER):
                out.append(f"{key} rep {i}: timing {r.get('ms_per_call')!r}")
    return out


def cell_read(c):
    """Per rep: ratio = (bnb visits summed) / (gnf4 visits summed); self-pairs = each arm's second visit / its first."""
    ratios, selfp = [], {a: [] for a in ARMS}
    for r in c["reps"]:
        t = {a: [ms for arm, ms in zip(r["order"], r["ms_per_call"]) if arm == a] for a in ARMS}
        ratios.append(sum(t["bnb"]) / sum(t["gnf4"]))
        for a in ARMS:
            selfp[a].append(t[a][1] / t[a][0])
    g_ms = statistics.median(ms for r in c["reps"] for arm, ms in zip(r["order"], r["ms_per_call"]) if arm == "gnf4")
    b_ms = statistics.median(ms for r in c["reps"] for arm, ms in zip(r["order"], r["ms_per_call"]) if arm == "bnb")
    return {"ratio_median": round(statistics.median(ratios), 5), "ratio_range": [round(min(ratios), 5),
                                                                                  round(max(ratios), 5)],
            "gnf4_ms_median": round(g_ms, 6), "bnb_ms_median": round(b_ms, 6),
            "self_pair_band": {a: [round(min(v), 5), round(max(v), 5)] for a, v in selfp.items()},
            "self_pair_median": {a: round(statistics.median(v), 5) for a, v in selfp.items()},
            "bnb_route": next(k_ for k_ in BNB_ROUTES if c["engaged"]["bnb"][k_])}


def reduce(rec):
    out = {"lane": LANE, "verdict": None, "reasons": [], "cells": {}}
    void = []
    if not isinstance(rec, dict):
        out.update(verdict="VOID", reasons=["no bench record"])
        return out
    if rec.get("dry"):
        void.append("a dry run is never a reading")
    if rec.get("status") != "ok":
        void.append(f"the bench status is {rec.get('status')!r}: {rec.get('error')}")
    for k_, want in (("gnf4", GNF4_VERSION), ("bnb", BNB_VERSION), ("torch", TORCH_VERSION)):
        if rec.get(k_) != want:
            void.append(f"{k_} {rec.get(k_)!r}, registered {want}")
    void += provenance_faults(rec.get("provenance"))
    void += release_proof_faults(rec.get("release_proof"))
    if "5090" not in str(rec.get("device")) or rec.get("cc") != [12, 0] or (rec.get("sm_count") or 0) < 160:
        void.append(f"the card {rec.get('device')!r} cc {rec.get('cc')} sms {rec.get('sm_count')}, registered an RTX "
                    "5090 (cc 12.0, >= 160 SMs)")
    if (rec.get("reps"), rec.get("iters")) != (REPS, ITERS):
        void.append(f"reps/iters {rec.get('reps')}/{rec.get('iters')}, registered {REPS}/{ITERS}")
    cells = rec.get("cells") if isinstance(rec.get("cells"), dict) else {}
    for shape in SHAPES:
        for m in MS:
            key = cell_key(shape, m)
            void += cell_faults(key, cells.get(key))
    if void:
        out.update(verdict="VOID", reasons=void)
        return out
    out["cells"] = {key: cell_read(c) for key, c in cells.items()}
    out.update(verdict="READ", reasons=["every cell correct, engaged as registered and timed; descriptive, no bar"])
    return out


def classify(rc, started, run_id=""):
    """The class of one fetched box (P231's table, no premise or download probe in this lane)."""
    if run_id.startswith("p236-shake-"):
        if rc == 0:
            return {"event": "SHAKE_OK", "started": False, "halt": False, "host_refused_cap": False}
        if rc in HOST_CODES or rc in FIT_CODES:
            return {"event": "SHAKE_HOST_REFUSED", "started": False, "halt": False, "host_refused_cap": False}
        return {"event": "SHAKE_FAIL", "started": False, "halt": True, "host_refused_cap": False}
    if not started:
        if rc in HOST_CODES:
            return {"event": "HOST_REFUSED", "started": False, "halt": False, "host_refused_cap": True}
        if rc in FIT_CODES:
            return {"event": "FIT_REFUSED", "started": False, "halt": False, "host_refused_cap": False}
        return {"event": "PRE_INSTALL_HALT", "started": False, "halt": True, "host_refused_cap": False}
    if rc == 0:
        return {"event": "ANSWER", "started": True, "halt": False, "host_refused_cap": False}
    return {"event": "NO_ANSWER", "started": True, "halt": False, "host_refused_cap": False}


def read_box(work):
    codes = sorted(glob.glob(os.path.join(work, "P236_EXIT_CODE.*")))
    if len(codes) != 1 or not os.path.isfile(os.path.join(work, "RUN_ID")):
        raise ValueError(f"{work!r} is not a P236 box work directory: it needs exactly one P236_EXIT_CODE.<nonce> "
                         f"(found {len(codes)}) and RUN_ID; the lane HALTS until the box's files are read by hand")
    return {"run_id": open(os.path.join(work, "RUN_ID"), encoding="utf-8").read().strip(),
            "rc": int(open(codes[0], encoding="utf-8").read().strip()),
            "started": os.path.exists(os.path.join(work, "STARTED"))}


# ------------------------------------------------------------------ self-test --

def fake_record(**over):
    """A READ record in the bench's own shape (every cell, every rep)."""
    cells = {}
    for shape in SHAPES:
        for m in MS:
            reps = []
            for i in range(REPS):
                o = order(i)
                reps.append({"order": o, "ms_per_call": [0.10 if a == "gnf4" else 0.15 for a in o]})
            cells[cell_key(shape, m)] = {
                "dequant_equal": True, "rel_err": {"gnf4": 0.004, "bnb": 0.003},
                "engaged": {"gnf4": {**{k_: 0 for k_ in GEMV_KEYS}, "bw_prmt32": 3, "calls": 3},
                            "bnb": {"custom": 3 if m <= 4 else 0, "dequant": 0 if m <= 4 else 3, "calls": 3}},
                "reps": reps, "copies": 2, "n": SHAPES[shape][0], "k": SHAPES[shape][1], "m": m}
    rec = {"lane": LANE, "status": "ok", "dry": False, "gnf4": GNF4_VERSION, "bnb": BNB_VERSION,
           "provenance": {"grouped-nf4-gemm": {"version": GNF4_VERSION, "direct_url": {
               "url": "https://" + GNF4_REPO + ".git", "vcs_info": {"vcs": "git", "commit_id": GNF4_SHA}}},
               "bitsandbytes": {"version": BNB_VERSION, "direct_url": None}},
           "release_proof": {"bitsandbytes": {"dist": "bitsandbytes", "version": BNB_VERSION, "files": 60,
                                              "wheel": f"bitsandbytes-{BNB_VERSION}-py3-none-manylinux_2_24_x86_64.whl",
                                              "sha256": "0" * 64, "record": "RECORD equal", "record_equal": True,
                                              "first_diff": None}},
           "torch": TORCH_VERSION, "device": "NVIDIA GeForce RTX 5090", "cc": [12, 0], "sm_count": 170, "reps": REPS,
           "iters": ITERS, "cells": cells}
    rec.update(over)
    return rec


def self_test():
    k0 = cell_key("mixtral.gate_up", 1)
    k4 = cell_key("qwen3.down", 4)

    def with_cell(key, **c):
        r = fake_record()
        r["cells"][key] = {**r["cells"][key], **c}
        return r

    def v(rec):
        return reduce(rec)["verdict"]
    good = fake_record()
    r = reduce(good)
    cases = [
        ("read", r["verdict"] == "READ" and r["cells"][k0]["ratio_median"] == 1.5 and r["cells"][k0]["bnb_route"] == "custom"
         and r["cells"][k0]["self_pair_band"] == {"gnf4": [1.0, 1.0], "bnb": [1.0, 1.0]}),
        ("30 cells", len(r["cells"]) == len(SHAPES) * len(MS) == 30),
        ("no record", v(None) == "VOID"),
        ("dry", v(fake_record(dry=True)) == "VOID"),
        ("not ok", v(fake_record(status="error")) == "VOID"),
        ("another gnf4", v(fake_record(gnf4="0.50.0")) == "VOID"),
        ("no provenance", v(fake_record(provenance=None)) == "VOID"),
        ("no release proof", v(fake_record(release_proof=None)) == "VOID"),
        ("a refused release proof", v(fake_record(release_proof={"bitsandbytes": {
            **good["release_proof"]["bitsandbytes"], "record_equal": False, "first_diff": "bitsandbytes/x.py"}})) == "VOID"),
        ("a release proof of another version", v(fake_record(release_proof={"bitsandbytes": {
            **good["release_proof"]["bitsandbytes"], "version": "0.50.1",
            "wheel": "bitsandbytes-0.50.1-py3-none-manylinux_2_24_x86_64.whl"}})) == "VOID"),
        ("gnf4 editable", v(fake_record(provenance={**good["provenance"], "grouped-nf4-gemm": {"version": "0.51.0",
            "direct_url": {"url": "file:///src/gnf4", "dir_info": {"editable": True}}}})) == "VOID"),
        ("gnf4 another commit", v(fake_record(provenance={**good["provenance"], "grouped-nf4-gemm": {"version": "0.51.0",
            "direct_url": {"url": "https://" + GNF4_REPO + ".git", "vcs_info": {"commit_id": "0" * 40}}}})) == "VOID"),
        ("gnf4 another repo", v(fake_record(provenance={**good["provenance"], "grouped-nf4-gemm": {"version": "0.51.0",
            "direct_url": {"url": "https://github.com/x/fork.git", "vcs_info": {"commit_id": GNF4_SHA}}}})) == "VOID"),
        ("gnf4 from a git+file:// repo at the release commit", v(fake_record(provenance={**good["provenance"],
            "grouped-nf4-gemm": {"version": "0.51.0", "direct_url": {"url": "file:///mirror/grouped-nf4-gemm",
                                                                     "vcs_info": {"commit_id": GNF4_SHA}}}})) == "VOID"),
        ("gnf4 a repo whose name only starts the same", v(fake_record(provenance={**good["provenance"],
            "grouped-nf4-gemm": {"version": "0.51.0", "direct_url": {"url": "https://" + GNF4_REPO + "-x.git",
                                                                     "vcs_info": {"commit_id": GNF4_SHA}}}})) == "VOID"),
        ("gnf4 index wheel reads", v(fake_record(provenance={**good["provenance"], "grouped-nf4-gemm": {
            "version": "0.51.0", "direct_url": None}})) == "READ"),
        ("bnb from git", v(fake_record(provenance={**good["provenance"], "bitsandbytes": {"version": "0.50.2",
            "direct_url": {"url": "https://github.com/bitsandbytes-foundation/bitsandbytes.git",
                           "vcs_info": {"commit_id": "a" * 40}}}})) == "VOID"),
        ("bnb a local wheel", v(fake_record(provenance={**good["provenance"], "bitsandbytes": {"version": "0.50.2",
            "direct_url": {"url": "file:///tmp/bnb.whl", "archive_info": {}}}})) == "VOID"),
        ("bnb a PyPI archive reads", v(fake_record(provenance={**good["provenance"], "bitsandbytes": {
            "version": "0.50.2", "direct_url": {"url": "https://files.pythonhosted.org/packages/x/bnb.whl",
                                                "archive_info": {}}}})) == "READ"),
        ("bnb another version", v(fake_record(provenance={**good["provenance"], "bitsandbytes": {"version": "0.50.3",
            "direct_url": None}})) == "VOID"),
        ("another bnb", v(fake_record(bnb="0.49.2")) == "VOID"),
        ("another torch", v(fake_record(torch="2.9.0")) == "VOID"),
        ("another card", v(fake_record(device="NVIDIA GeForce RTX 4090", cc=[8, 9], sm_count=128)) == "VOID"),
        ("few SMs", v(fake_record(sm_count=150)) == "VOID"),
        ("another rep count", v(fake_record(reps=10)) == "VOID"),
        ("a cell missing", v({**good, "cells": {k: c for k, c in good["cells"].items() if k != k4}}) == "VOID"),
        ("cells missing entirely", v(fake_record(cells=None)) == "VOID"),
        ("dequant not equal", v(with_cell(k0, dequant_equal=False)) == "VOID"),
        ("dequant unrecorded", v(with_cell(k0, dequant_equal=None)) == "VOID"),
        ("gnf4 off the reference", v(with_cell(k4, rel_err={"gnf4": 0.02, "bnb": 0.003})) == "VOID"),
        ("bnb off the reference", v(with_cell(k4, rel_err={"gnf4": 0.004, "bnb": float("nan")})) == "VOID"),
        ("rel_err missing", v(with_cell(k4, rel_err={})) == "VOID"),
        ("gnf4 not K33", v(with_cell(k4, engaged={**good["cells"][k4]["engaged"], "gnf4": {
            **{k_: 0 for k_ in GEMV_KEYS}, "dotpad": 3, "calls": 3}})) == "VOID"),
        ("gnf4 partial counters", v(with_cell(k4, engaged={**good["cells"][k4]["engaged"], "gnf4": {
            "bw_prmt32": 3, "calls": 3}})) == "VOID"),
        ("gnf4 two launches a call", v(with_cell(k4, engaged={**good["cells"][k4]["engaged"], "gnf4": {
            **{k_: 0 for k_ in GEMV_KEYS}, "bw_prmt32": 6, "calls": 3}})) == "VOID"),
        ("bnb mixed routes", v(with_cell(k4, engaged={**good["cells"][k4]["engaged"], "bnb": {
            "custom": 2, "dequant": 1, "calls": 3}})) == "VOID"),
        ("bnb fewer dispatches than calls", v(with_cell(k0, engaged={**good["cells"][k0]["engaged"], "bnb": {
            "custom": 2, "dequant": 0, "calls": 3}})) == "VOID"),
        ("bnb either route reads", v(with_cell(k0, engaged={**good["cells"][k0]["engaged"], "bnb": {
            "custom": 0, "dequant": 3, "calls": 3}})) == "READ"),
        ("bnb route unrecorded", v(with_cell(k0, engaged={"gnf4": good["cells"][k0]["engaged"]["gnf4"]})) == "VOID"),
        ("few reps", v(with_cell(k0, reps=good["cells"][k0]["reps"][:5])) == "VOID"),
        ("wrong order", v(with_cell(k0, reps=[{**x, "order": list(ORDER)} for x in good["cells"][k0]["reps"]]))
         == "VOID"),
        ("a zero time", v(with_cell(k0, reps=[{**good["cells"][k0]["reps"][0], "ms_per_call": [0.1, 0, 0.1, 0.1]}]
                                    + good["cells"][k0]["reps"][1:])) == "VOID"),
        ("a missing time", v(with_cell(k0, reps=[{**good["cells"][k0]["reps"][0], "ms_per_call": [0.1, 0.1]}]
                                       + good["cells"][k0]["reps"][1:])) == "VOID"),
        ("palindromes, each arm leading half", order(0) == ["gnf4", "bnb", "bnb", "gnf4"]
         and order(1) == ["bnb", "gnf4", "gnf4", "bnb"] and order(2) == order(0)),
        ("copies span 4x L2 at every shape, under the cap",
         all(copies_for(n, k) * (n * k // 2 + n * (k // 64) * 4) >= ROTATE_BYTES and copies_for(n, k) < MAX_COPIES
             for n, k in SHAPES.values())),
    ]
    # the reported ratio follows the timings: gnf4 twice as fast reads 2
    fast = fake_record()
    for c in fast["cells"].values():
        for rp in c["reps"]:
            rp["ms_per_call"] = [0.05 if a == "gnf4" else 0.10 for a in rp["order"]]
    cases.append(("ratio", reduce(fast)["cells"][k0]["ratio_median"] == 2.0))
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "P236_EXIT_CODE.n"), "w").write("0\n")
        open(os.path.join(d, "RUN_ID"), "w").write("p236-5090-1\n")
        open(os.path.join(d, "STARTED"), "w").close()
        b = read_box(d)
        cases.append(("classify an answer", classify(b["rc"], b["started"], b["run_id"])["event"] == "ANSWER"))
        open(os.path.join(d, "P236_EXIT_CODE.m"), "w").write("0\n")
        try:
            read_box(d)
            cases.append(("two exit files fail closed", False))
        except ValueError:
            cases.append(("two exit files fail closed", True))
    cases += [("host refusal", classify(13, False)["event"] == "HOST_REFUSED"),
              ("fit refusal", classify(41, False)["event"] == "FIT_REFUSED"),
              ("kit halt", classify(9, False)["event"] == "PRE_INSTALL_HALT" and classify(9, False)["halt"]),
              ("a void box is a strike", classify(27, True)["event"] == "NO_ANSWER"),
              ("shake ok", classify(0, False, "p236-shake-1")["event"] == "SHAKE_OK"),
              ("shake fail halts", classify(9, True, "p236-shake-1")["halt"])]
    bad = [n for n, ok in cases if not ok]
    print(f"p236_reduce self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--classify", action="store_true")
    p.add_argument("--work")
    p.add_argument("--bench")
    p.add_argument("--out")
    p.add_argument("--check-args", action="store_true", help="parse and validate the arguments, then exit (tests)")
    a = p.parse_args(argv)
    if a.check_args:
        ok = bool(a.work) if a.classify else bool(a.self_test or (a.bench and a.out))
        print("ARGS OK" if ok else "ARGS BAD")
        return 0 if ok else 2
    if a.self_test:
        return self_test()
    if a.classify:
        try:
            box = read_box(a.work or "")
        except ValueError as e:
            print(f"P236_CLASSIFY_REFUSED {e}")
            return 2
        print("P236_CLASS " + json.dumps(dict(classify(box["rc"], box["started"], box["run_id"]), **box)))
        return 0
    rec = json.load(open(a.bench, encoding="utf-8")) if a.bench and os.path.exists(a.bench) else None
    v = reduce(rec)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(v, f, indent=1)
    print(f"P236_VERDICT {v['verdict']} {json.dumps(v['reasons'])[:2000]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
