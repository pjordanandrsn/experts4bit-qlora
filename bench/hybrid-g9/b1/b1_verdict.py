# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-b1 decision calculator: B scaling curve summary + the H/R0/R1
gap decomposition + the preregistered branch decision. No optimization
claims -- the output IS the deliverable. --self-test first, always.

Kernel occupancy is parsed from --torch-profile-out tables under the
#216 convention: sum Self-CUDA over DEVICE-EVENT rows only (name not
starting with aten::/ProfilerStep/cuda), so aten-op rows never
double-count their kernels."""

import argparse
import json
import re
import sys
from pathlib import Path

AA_SPREAD_MAX = 7.5
GAP_LL = 25.0                 # the preregistered “much less than”
FAST_HOST_OVER_DEV = 2.0      # R1 “fast” anchor: host ≤ 2× kernel occupancy
GS_STEP_LO, GS_STEP_HI = 115.0, 165.0
GS_ATTN_MAX = 55.0
GS_DRAM_MAX = 25.0


def _step(rep):
    return float(rep["decode_median_ms"]["step"])


def _tokens(rep):
    return rep["generated_tokens"]


def parse_kernel_occupancy_ms(table_text, active_steps):
    """Per-step device occupancy from a key_averages table: Self-CUDA of
    device-event rows only (never aten::/cuda-runtime/ProfilerStep)."""
    total_ms = 0.0
    for line in table_text.splitlines():
        cols = re.split(r"\s{2,}", line.strip())
        if len(cols) < 10:
            continue
        name = cols[0]
        if (name.startswith("aten::") or name.startswith("ProfilerStep")
                or name.startswith("cuda") or name in ("Name", "")):
            continue
        m = re.match(r"^([0-9.]+)(us|ms|s)$", cols[6])
        if not m:
            continue
        v = float(m.group(1))
        unit = {"us": 1e-3, "ms": 1.0, "s": 1e3}[m.group(2)]
        total_ms += v * unit
    return total_ms / max(1, active_steps)


def _agree_len(ta, tb):
    n = 0
    for ra in sorted(ta):
        a, b = ta[ra], tb.get(ra, [])
        for x, y in zip(a, b):
            if x != y:
                return n
            n += 1
    return n


def verdict(curve, h1, h2, r0_1, r0_2, r1_1, r1_2, r1_kernel_occ_ms):
    """curve: {B(str): rep}; arms: timed rep pairs. The fast/slow anchor
    compares the CLEAN timed R1 step against kernel occupancy from the
    profiled run -- the profiled run's own step is host-inflated by the
    brackets and profiler and would bias the gate toward HOST (Bugbot,
    e4b#222; the same instrument-tax class AMENDMENT-t5b-h-a hit)."""
    out = {"curve": {}}
    for b, rep in sorted(curve.items(), key=lambda kv: int(kv[0])):
        st = _step(rep)
        out["curve"][b] = {"step_ms": st,
                           "aggregate_tok_s": int(b) * 1000.0 / st,
                           "per_stream_tok_s": 1000.0 / st}
    if "16" not in curve:
        out["verdict"] = ("NO-VERDICT (the B=16 anchor rep is required: "
                          "without it the GS shape-gate cannot certify "
                          "the operating point, and no decomposition may "
                          "be read)")
        return out
    if "16" in curve:
        d = curve["16"]["decode_median_ms"]
        gs = (GS_STEP_LO <= float(d["step"]) <= GS_STEP_HI
              and float(d["attention_host"]) <= GS_ATTN_MAX
              and float(d["dram_experts_host"]) <= GS_DRAM_MAX)
        out["gs_b16_anchor"] = {"pass": bool(gs),
                                "step_ms": float(d["step"])}
        if not gs:
            out["verdict"] = ("NO-VERDICT (B=16 anchor failed the "
                              "certified shape-gate: box/config invalid)")
            return out

    arms = {}
    for name, (x, y) in (("H", (h1, h2)), ("R0", (r0_1, r0_2)),
                         ("R1", (r1_1, r1_2))):
        m1, m2 = _step(x), _step(y)
        spread = abs(m1 - m2) / min(m1, m2) * 100.0
        arms[name] = {"step_ms": min(m1, m2), "aa_spread_pct": spread,
                      "aa_pass": spread < AA_SPREAD_MAX}
        if not arms[name]["aa_pass"]:
            out["arms"] = arms
            out["verdict"] = f"NO-VERDICT (G0 fail on arm {name})"
            return out
    out["arms"] = arms

    ident = {"r0_r1_bitwise": _tokens(r0_1) == _tokens(r1_1),
             "h_vs_r0_agree_tokens": _agree_len(_tokens(h1), _tokens(r0_1)),
             "h_total_tokens": sum(len(v) for v in _tokens(h1).values())}
    out["identity"] = ident
    if not ident["r0_r1_bitwise"]:
        out["verdict"] = ("NO-VERDICT (G1: R0 and R1 are both all-GPU and "
                          "must decode identical tokens; a mismatch means "
                          "the arms are not computing the same function)")
        return out

    sH, s0, s1 = (arms["H"]["step_ms"], arms["R0"]["step_ms"],
                  arms["R1"]["step_ms"])
    g_h_r0 = (sH - s0) / sH * 100.0        # residency tax share
    g_r0_r1 = (s0 - s1) / s0 * 100.0       # abstraction tax share
    fast = s1 <= FAST_HOST_OVER_DEV * r1_kernel_occ_ms
    out["gaps"] = {"H_ms": sH, "R0_ms": s0, "R1_ms": s1,
                   "residency_tax_pct_of_H": g_h_r0,
                   "abstraction_tax_pct_of_R0": g_r0_r1,
                   "r1_kernel_occupancy_ms": r1_kernel_occ_ms,
                   "r1_host_over_device": s1 / max(1e-9, r1_kernel_occ_ms),
                   "r1_fast": bool(fast)}

    branches = []
    if g_h_r0 >= GAP_LL:
        branches.append(
            ("residency", g_h_r0,
             "BRANCH-1 residency tax: hybrid B=1 becomes a latency-hiding "
             "problem -- placement for hit probability, earlier dispatch, "
             "GPU/CPU overlap; prefetch only if the locality trace "
             "licenses it"))
    if g_r0_r1 >= GAP_LL:
        branches.append(
            ("abstraction", g_r0_r1,
             "BRANCH-2 abstraction tax: implement the all-resident "
             "collapse fast path (no tier split / bookkeeping / joins / "
             "placement decisions when the placement is all-VRAM)"))
    if not branches:
        if fast:
            out["verdict"] = (
                "BRANCH-3-DEVICE (arms agree within 25% and R1 host ≤ 2× "
                "kernel occupancy: the B=1 cost is device-side M=1 kernel "
                "work -- open the resident-B1 kernel ladder: M=1/GEMV-"
                "specialized packed kernel, launch collapse)")
        else:
            out["verdict"] = (
                "BRANCH-3-HOST (arms agree within 25% and R1 spends "
                f"{out['gaps']['r1_host_over_device']:.1f}x its kernel "
                "occupancy in host work: M=1 executor problem -- device-"
                "resident routing metadata, op fusion/launch collapse; "
                "native boundary only after structural collapse fails)")
        return out
    branches.sort(key=lambda x: -x[1])
    parts = [b[2] for b in branches]
    if g_h_r0 >= GAP_LL and fast and g_r0_r1 < GAP_LL:
        parts.append("(R1 fast + H slow = two operating regimes, both "
                     "recorded: competitive resident B=1 when the model "
                     "fits; measured degradation beyond VRAM)")
    out["verdict"] = " | ".join(parts)
    return out


def _rep(step, toks, attn=40.0, dram=14.0):
    return {"decode_median_ms": {"step": step, "attention_host": attn,
                                 "dram_experts_host": dram},
            "generated_tokens": toks}


def self_test():
    tok = {"0": [1, 2, 3]}
    tok_h = {"0": [1, 2, 9]}
    curve = {"16": _rep(131.0, tok), "1": _rep(65.0, tok)}
    # case 1: residency dominant (H 65, R0 40, R1 38), R1 fast
    v = verdict(curve, _rep(65.0, tok_h), _rep(66.0, tok_h),
                _rep(40.0, tok), _rep(40.5, tok),
                _rep(38.0, tok), _rep(38.2, tok), 30.0)
    assert v["verdict"].startswith("BRANCH-1"), v["verdict"]
    assert "two operating regimes" in v["verdict"], v["verdict"]
    # case 2: abstraction dominant (H 41, R0 40, R1 25)
    v = verdict(curve, _rep(41.0, tok_h), _rep(41.5, tok_h),
                _rep(40.0, tok), _rep(40.5, tok),
                _rep(25.0, tok), _rep(25.2, tok), 20.0)
    assert v["verdict"].startswith("BRANCH-2"), v["verdict"]
    # case 3-host: all close, host-heavy R1
    v = verdict(curve, _rep(62.0, tok_h), _rep(63.0, tok_h),
                _rep(60.0, tok), _rep(60.5, tok),
                _rep(58.0, tok), _rep(58.4, tok), 8.0)
    assert v["verdict"].startswith("BRANCH-3-HOST"), v["verdict"]
    # case 3-device: all close, device-bound R1
    v = verdict(curve, _rep(62.0, tok_h), _rep(63.0, tok_h),
                _rep(60.0, tok), _rep(60.5, tok),
                _rep(58.0, tok), _rep(58.4, tok), 40.0)
    assert v["verdict"].startswith("BRANCH-3-DEVICE"), v["verdict"]
    # composition: both gaps
    v = verdict(curve, _rep(100.0, tok_h), _rep(101.0, tok_h),
                _rep(60.0, tok), _rep(60.5, tok),
                _rep(30.0, tok), _rep(30.2, tok), 20.0)
    assert "BRANCH-1" in v["verdict"] and "BRANCH-2" in v["verdict"], v
    # G1: R0/R1 token mismatch blocks
    v = verdict(curve, _rep(65.0, tok_h), _rep(66.0, tok_h),
                _rep(40.0, tok), _rep(40.5, tok),
                _rep(38.0, {"0": [7]}), _rep(38.2, {"0": [7]}), 30.0)
    assert v["verdict"].startswith("NO-VERDICT (G1"), v["verdict"]
    # G0: arm spread blocks
    v = verdict(curve, _rep(65.0, tok_h), _rep(80.0, tok_h),
                _rep(40.0, tok), _rep(40.5, tok),
                _rep(38.0, tok), _rep(38.2, tok), 30.0)
    assert v["verdict"].startswith("NO-VERDICT (G0 fail on arm H"), v
    # GS anchor blocks
    bad = dict(curve)
    bad["16"] = _rep(210.0, tok, attn=120.0)
    v = verdict(bad, _rep(65.0, tok_h), _rep(66.0, tok_h),
                _rep(40.0, tok), _rep(40.5, tok),
                _rep(38.0, tok), _rep(38.2, tok), 30.0)
    assert v["verdict"].startswith("NO-VERDICT (B=16 anchor"), v
    # missing B=16 anchor blocks (Bugbot, e4b#222)
    v = verdict({"1": _rep(65.0, tok)}, _rep(65.0, tok_h),
                _rep(66.0, tok_h), _rep(40.0, tok), _rep(40.5, tok),
                _rep(38.0, tok), _rep(38.2, tok), 30.0)
    assert v["verdict"].startswith("NO-VERDICT (the B=16 anchor"), v
    # kernel-table parser: aten and cuda rows excluded, units handled
    tbl = ("Name  x  x  x  x  x  Self CUDA  x  x  Calls\n"
           "aten::mm  1%  1ms  1%  1ms  1us  10.000ms  5%  10ms  10\n"
           "my_kernel  0%  0us  0%  0us  0us  24.000ms  10%  24ms  96\n"
           "cudaLaunchKernel  1%  1ms  1%  1ms  1us  2.000ms  1%  2ms  9\n"
           "Memcpy DtoD  0%  0us  0%  0us  0us  6.000ms  2%  6ms  30\n")
    occ = parse_kernel_occupancy_ms(tbl, 12)
    assert abs(occ - (24.0 + 6.0) / 12) < 1e-6, occ
    print("self-test OK: 10/10 branches exercised")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--curve", nargs="*", default=[],
                    help="B=rep.json pairs, e.g. 1=b1.json 16=b16.json")
    ap.add_argument("--h1"), ap.add_argument("--h2")
    ap.add_argument("--r0-1"), ap.add_argument("--r0-2")
    ap.add_argument("--r1-1"), ap.add_argument("--r1-2")
    ap.add_argument("--r1-kernel-table", help="R1 --torch-profile-out")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        sys.exit(0)
    load = lambda f: json.loads(Path(f).read_text())
    curve = {}
    for pair in a.curve:
        b, f = pair.split("=", 1)
        curve[b] = load(f)
    tab = Path(a.r1_kernel_table).read_text()
    m = re.search(r"profiled decode steps: \d+ \(active window: (\d+)/12\)",
                  tab)
    active = int(m.group(1)) if m else 12
    occ = parse_kernel_occupancy_ms(tab, active)
    print(json.dumps(verdict(curve, load(a.h1), load(a.h2),
                             load(a.r0_1), load(a.r0_2),
                             load(a.r1_1), load(a.r1_2),
                             occ), indent=2))
