#!/usr/bin/env python3
"""The 64-slot decode-step census (#846): where SC2e's bucket-64 decode step goes, and the most each lever can save.
$0, from committed receipts only; it registers nothing.

Inputs (all committed):
- SC2e ``sc2e-5090-1``: the s64a step traces (both draws) and ``capacity_check.json``;
- P119 ``p119-5090-1``: the d64 eager-twin profile (``verdict.json`` classes, ``box.json`` kernels);
- P122 ``p122-5090-1``, P124 ``p124-5090-4``, P126 ``p126-5090-2``: the levers already read at 64 rows;
- #1469's ``loc-a4000-1`` decode routing trace (Qwen3-30B-A3B ids, 4 sequences x 512 steps);
- ``bench/sc2/sc2f_basis.py``'s model, for the top lever's req/s.

**The closure.** A served bucket-64 decode step (one piece, decode only) splits exactly, by its own CUDA events:
host before the replay (``gpu.dec_prep``) + the device replay (``gpu.dec_issue - gpu.dec_prep``) + the host tail
(``step_ms - gpu.dec_issue``). The device replay splits by P119's d64 classes plus an explicit residual (served replay
minus P119's eager-twin total). Means over the steps, so the parts add.

**Ceilings** (the most a lever can save per bucket-64 step, from receipts):
- (a) ``Int4Linear`` above 16 rows: P124 replaced the cuBLAS projections with K16's int4 tile; what is left is K16's
  time minus the int4 bytes at the copy bandwidth (gnf4 K20's floor convention);
- (b) the tile table above 256 routed rows: P122 then P126 took it to the 4-program table; what is left is that table;
- (c) the exposed host gap: everything outside the device replay (a decode lookahead hides it while host < device);
- (d) K19 at 512 rows: K19's time minus its byte floor, bounded by K20's best-plan efficiency at 16 rows.

    python decode64_census.py            # writes decode64_census.json beside this file
    python decode64_census.py --check    # recompute and compare with the committed JSON
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SC2E = REPO / "bench" / "h2h-2026-10-02" / "sc2e" / "receipts" / "sc2e-5090-1"
P119 = REPO / "bench" / "p119" / "receipts" / "p119-5090-1"
P122 = REPO / "bench" / "p122" / "receipts" / "p122-5090-1" / "verdict.json"
P124 = REPO / "bench" / "p124" / "receipts" / "p124-5090-4" / "verdict.json"
P126 = REPO / "bench" / "p126" / "receipts" / "p126-5090-2" / "verdict.json"
LOC = REPO / "bench" / "locality-1469" / "receipts" / "loc-a4000-1" / "decode.npz"
OUT = HERE / "decode64_census.json"
DRAWS = (1, 2)

# Qwen3-30B-A3B shapes; int4-b32 bytes as grouped-nf4-gemm K20's floor counts them (packed int4 + fp16 scale per 32)
HIDDEN, HEADS, KV_HEADS, HEAD_DIM, LAYERS = 2048, 32, 4, 128, 48
EXPERT_SHAPES = ((1536, 2048), (2048, 768))                     # gate_up (N, K), down (N, K)
ATTN_SHAPES = ((HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM, HIDDEN), (HIDDEN, HEADS * HEAD_DIM))   # fused q/k/v, o
COPY_GBPS = 1532.0              # K20's measured 5090 device-to-device copy (gnf4 kernel/RESULTS-k20-k19-plan-sweep-5090.md)
K20_EFFICIENCY_B16 = 0.89       # K20: K19's best plan at 16 rows, against that floor (same file)
SEEDS = 40
RATES = (12, 13, 14, 15, 16, 17)

# What reached the 64-row decode step after SC2e's commit (3bbcb01b, gnf4 0.42.0 b4f93f1c), to main
CHANGES = [
    {"commit": "0c4da09a", "what": "E4B_INT4_WIDE_TILES=auto: the one-launch tile table above 256 routed rows (P122)",
     "reaches_64": True, "read_at_64": "P122 p122-5090-1: served 64-row step ON/OFF 0.9568"},
    {"commit": "a3527276", "what": "E4B_ATTN_INT4_WIDE=auto: attention projections of 17-64 rows on K16's int4 tile (P124)",
     "reaches_64": True, "read_at_64": "P124 p124-5090-4: served 64-row step 0.9648 / 0.9670"},
    {"commit": "2384d2c3", "what": "E4B_INT4_TILE_PROGRAMS=auto (4): the one-launch table over 4 programs (P126)",
     "reaches_64": True, "read_at_64": "P126 p126-5090-2: served 64-row step 0.8442 / 0.8456"},
    {"commit": "dc2221a3", "what": "E4B_PREFILL_LEAN_DISPATCH=auto (P130's P2): K19's lean dispatch above 256 routed rows",
     "reaches_64": True, "read_at_64": None,
     "note": "not prefill-only: bucket 64 routes 512 rows (the smokeflips record shows one_launch|lean|gt256 on a "
             "512-row call); bitwise by construction; P130 timed the prefill replay only"},
    {"commit": "8ea97296", "what": "P127 Phase 2: the MoE residual add into the experts' combine, licensed per decode bucket",
     "reaches_64": True, "read_at_64": None, "note": "bitwise; no speed claim; P127 read T = 1"},
    {"commit": "1ddcb0ae", "what": "P127 Phase 2 wiring: the router's cast in-kernel and rope_norm_qk in the attention folds",
     "reaches_64": True, "read_at_64": None, "note": "the decode folds run up to 64 rows; bitwise; no speed claim"},
    {"commit": "451a3703", "what": "E4B_FUSE_PREFILL_GLUE=1 for qwen3_moe (P130's P1)", "reaches_64": False,
     "read_at_64": None, "note": "adds prefill rows to the folds; decode rows were already folded"},
]


def _jsonl_gz(p):
    with gzip.open(p, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def _mean(xs):
    return statistics.fmean(xs)


def _bpe(shapes) -> int:
    return sum(n * (k // 2) + n * (k // 32) * 2 for n, k in shapes)


def _ols(xs, ys):
    mx, my = _mean(xs), _mean(ys)
    s = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - s * mx, s


def closure(rows: list, classes: dict) -> dict:
    """The served bucket-64 step's exact split, and the device replay by P119's classes plus a residual."""
    s = [r for r in rows if r.get("bucket") == 64 and r.get("dec_pieces") == 1 and not r.get("prefill_tokens")
         and "dec_issue" in (r.get("gpu") or {})]
    step = _mean(r["step_ms"] for r in s)
    pre = _mean(r["gpu"]["dec_prep"] for r in s)
    dev = _mean(r["gpu"]["dec_issue"] - r["gpu"]["dec_prep"] for r in s)
    tail = _mean(r["step_ms"] - r["gpu"]["dec_issue"] for r in s)
    segs = sorted({k for r in s for k in r["seg"]})
    p119_total = sum(classes.values())
    out = {"steps": len(s), "rows_mean": round(_mean(r["decode_rows"] for r in s), 2),
           "step_ms": round(step, 3), "step_ms_p50": round(statistics.median(r["step_ms"] for r in s), 3),
           "host_before_replay_ms": round(pre, 3), "device_replay_ms": round(dev, 3), "host_tail_ms": round(tail, 3),
           "closes_to_ms": round(step - (pre + dev + tail), 6),
           "host_segments_ms": {k: round(_mean(r["seg"].get(k, 0.0) for r in s), 3) for k in segs},
           "device_by_class_ms": {k: round(v, 3) for k, v in sorted(classes.items(), key=lambda kv: -kv[1])},
           "device_residual_ms": round(dev - p119_total, 3)}
    return out


def host_fit(rows: list) -> dict:
    """Least squares on the bucket over every decode-only step, as serve_capacity.StepCosts.from_step_trace fits the
    step: the exposed host gap and the per-stream dispatch, so a lever's saving subtracts from the model's own fit."""
    dec = [r for r in rows if r.get("decode_rows") and not r.get("prefill_tokens") and r.get("bucket")]
    xs = [float(r["bucket"]) for r in dec]
    step = _ols(xs, [r["step_ms"] for r in dec])
    host = _ols(xs, [r["gpu"]["dec_prep"] + r["step_ms"] - r["gpu"]["dec_issue"] for r in dec])
    disp = _ols(xs, [r["seg"].get("dispatch", 0.0) for r in dec])
    by_bucket = {}
    for b in sorted({r["bucket"] for r in dec if r.get("dec_pieces") == 1}):
        rs = [r for r in dec if r["bucket"] == b and r.get("dec_pieces") == 1]
        by_bucket[str(b)] = {"steps": len(rs), "step_ms": round(_mean(r["step_ms"] for r in rs), 3),
                             "host_gap_ms": round(_mean(r["gpu"]["dec_prep"] + r["step_ms"] - r["gpu"]["dec_issue"]
                                                        for r in rs), 3),
                             "dispatch_ms": round(_mean(r["seg"].get("dispatch", 0.0) for r in rs), 3),
                             "device_ms": round(_mean(r["gpu"]["dec_issue"] - r["gpu"]["dec_prep"] for r in rs), 3)}
    return {"decode_steps": len(dec), "step_fit": [round(v, 4) for v in step], "host_gap_fit": [round(v, 4) for v in host],
            "dispatch_fit": [round(v, 4) for v in disp], "by_bucket": by_bucket}


def expert_union() -> dict:
    import numpy as np
    z = np.load(LOC)
    ids, offs, E = z["expert_ids"].astype(np.int64), z["seq_offsets"], int(z["n_experts"])
    T, L, K = ids.shape
    rng = np.random.default_rng(1469)

    def union(idx):
        sub = ids[idx]
        return float(np.mean([len(np.unique(sub[:, layer, :])) for layer in range(L)]))
    cons = [union(np.arange(st, st + 64)) for s in range(len(offs) - 1) for st in range(offs[s], offs[s + 1] - 63, 32)]
    mixed = [union(rng.choice(T, 64, replace=False)) for _ in range(400)]
    return {"experts": E, "top_k": K, "layers": L, "tokens": T,
            "one_sequence_64_mean": round(statistics.fmean(cons), 1),
            "across_sequences_64_mean": round(statistics.fmean(mixed), 1),
            "across_sequences_64_p10_p90": [round(float(np.percentile(mixed, 10)), 1), round(float(np.percentile(mixed, 90)), 1)],
            "uniform_64": round(E * (1 - (1 - K / E) ** 64), 1)}


def levers(p119: dict, p119_box: dict, clo: dict) -> list:
    v122, v124, v126 = (json.loads(p.read_text())["tables"] for p in (P122, P124, P126))
    d64 = p119["decode_classes"]["d64"]
    # (a) Int4Linear above 16 rows
    cublas = v124["profile"]["64"]["attn_cublas_ms"]
    k16 = cublas * v124["int4_over_cublas_64"]
    attn_floor = _bpe(ATTN_SHAPES) * LAYERS / (COPY_GBPS * 1e9) * 1e3
    # (b) the tile table
    t1, t4, t8 = (v126["profile"][str(p)]["table_one_ms" if p == 1 else "table_mp_ms"] for p in (1, 4, 8))
    # (c) the host gap
    host = [clo[f"d{d}"]["host_before_replay_ms"] + clo[f"d{d}"]["host_tail_ms"] for d in DRAWS]
    disp = [clo[f"d{d}"]["host_segments_ms"]["dispatch"] for d in DRAWS]
    # (d) K19 at 512 rows
    k19 = d64["expert_int4"]
    k19_bound = k19 * (1 - K20_EFFICIENCY_B16)
    rows = [
        {"lever": "c", "name": "the exposed host gap (decode lookahead; one stream wakeup a step is a subset)",
         "at_sc2e_ms": [round(x, 3) for x in host], "realized_since_ms": 0.0,
         "ceiling_ms": round(min(host), 3), "subset_dispatch_ms": [round(x, 3) for x in disp],
         "basis": "SC2e s64a step traces: host before the replay + host tail; P118 read the lookahead at 1 request only",
         "needs_fresh_read": "the host path has no default-on change since SC2e; its share of the step grows as the device "
                             "replay falls (P122, P124, P126, P2)"},
        {"lever": "a", "name": "Int4Linear above 16 rows (attention projections)",
         "at_sc2e_ms": round(cublas, 3), "realized_since_ms": round(cublas - k16, 3),
         "ceiling_ms": round(k16 - attn_floor, 3), "int4_floor_ms": round(attn_floor, 3),
         "basis": "P124 p124-5090-4 profile at 64 rows; int4-b32 bytes of fused q/k/v + o at the K20 copy bandwidth",
         "needs_fresh_read": "read at 64 rows by P124 (DEFAULT_ON); K34 found no better K16 plan of 48"},
        {"lever": "d", "name": "K19 at 512 routed rows",
         "at_sc2e_ms": round(k19, 3), "realized_since_ms": None, "ceiling_ms": round(k19_bound, 3),
         "basis": "P119 d64 expert_int4; bound = K19 x (1 - K20's best-plan efficiency at 16 rows)",
         "needs_fresh_read": "P2's lean dispatch (dc2221a3) and P127 Phase 2's combine (8ea97296) reach this call "
                             "since SC2e, bitwise and unread at 64 rows"},
        {"lever": "b", "name": "the tile table above 256 routed rows",
         "at_sc2e_ms": round(t1, 3), "realized_since_ms": round(t1 - t4, 3), "ceiling_ms": round(t4, 3),
         "p8_table_ms": round(t8, 3),
         "basis": "P126 p126-5090-2 profile: one program, then 4 (the default), and 8 (ineligible on block noise)",
         "needs_fresh_read": "read at 64 rows by P122 and P126 (DEFAULT_ON)"},
    ]
    rows.sort(key=lambda r: -r["ceiling_ms"])
    k19_union_note = {"k19_ms": round(k19, 3), "bytes_per_expert": _bpe(EXPERT_SHAPES),
                      "floor_ms_at_union": None}
    tile_r1 = {}
    for b in ("d16", "d32"):
        for name, calls, ms in p119_box["decode"][b]["kernels"]:
            if name == "_tile_table_r1":
                tile_r1[b] = {"calls_per_step": calls, "ms_per_step": round(ms, 4), "us_per_launch": round(ms / calls * 1e3, 2)}
    return rows, k19_union_note, tile_r1


def model(cc: dict, fits: dict) -> dict:
    spec = importlib.util.spec_from_file_location("sc2f_basis", REPO / "bench" / "sc2" / "sc2f_basis.py")
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    sc = b._cap()
    out = {"p130_saving_ms": b.P130_SAVING_MS, "seeds": SEEDS, "rates": list(RATES)}
    for d in DRAWS:
        c = cc[b.SRC.format(d)]["costs_ms"]
        hb, hr = fits[f"d{d}"]["host_gap_fit"]

        def mk(db=0.0, dr=0.0):
            return sc.StepCosts(prefill_step_s=(c["prefill_step"] - b.P130_SAVING_MS) / 1e3,
                                first_decode_s=c["first_decode"] / 1e3, decode_base_s=(c["decode_base"] - db) / 1e3,
                                decode_per_row_s=(c["decode_per_row"] - dr) / 1e3, buckets=b.BUCKETS)
        arms = {}
        for name, k in (("on", mk()), ("on_plus_c", mk(hb, hr))):
            m = {r: statistics.fmean(sc.simulate(k, sc.Workload(n=b.N, rate=r, seed=1000 * d + s),
                                                 max_seqs=b.MAX_SEQS).attainment for s in range(SEEDS)) for r in RATES}
            cross = None
            for lo, hi in zip(RATES, RATES[1:]):
                if m[lo] >= 0.95 > m[hi]:
                    cross = round(lo + (m[lo] - 0.95) / (m[lo] - m[hi]) * (hi - lo), 2)
            arms[name] = {"step_at_64_ms": round((k.decode_base_s + 64 * k.decode_per_row_s) * 1e3, 2),
                          "knee_mean": b.knee(k, d), "mean_attainment": {str(r): round(v, 3) for r, v in m.items()},
                          "rate_at_0.95": cross}
        out[f"d{d}"] = arms
    return out


def census() -> dict:
    p119 = json.loads((P119 / "verdict.json").read_text())["tables"]
    p119_box = json.loads((P119 / "box.json").read_text())
    cc = json.loads((SC2E / "capacity_check.json").read_text())
    clo, fits = {}, {}
    for d in DRAWS:
        rows = _jsonl_gz(SC2E / "sc2" / f"steps_e4b_s64a_d{d}.jsonl.gz")
        clo[f"d{d}"] = closure(rows, p119["decode_classes"]["d64"])
        fits[f"d{d}"] = host_fit(rows)
    lv, k19_note, tile_r1 = levers(p119, p119_box, clo)
    un = expert_union()
    k19_note["floor_ms_at_union"] = round(un["across_sequences_64_mean"] * LAYERS * k19_note["bytes_per_expert"]
                                          / (COPY_GBPS * 1e9) * 1e3, 3)
    return {"schema": "e4b-decode64-census/1",
            "sources": {"sc2e": "sc2e-5090-1 (e4b 3bbcb01b, grouped-nf4-gemm b4f93f1c = 0.42.0)",
                        "p119": "p119-5090-1 (e4b caac7f08, grouped-nf4-gemm b4f93f1c)",
                        "p122": "p122-5090-1", "p124": "p124-5090-4", "p126": "p126-5090-2", "locality": "loc-a4000-1"},
            "capacity_model_step_at_64_ms": {f"d{d}": cc[f"e4b_s64a_d{d}"]["costs_ms"]["decode_step_at_max_seqs"] for d in DRAWS},
            "closure": clo, "host_fit": fits, "levers": lv, "k19": k19_note, "expert_union_per_layer": un,
            "tile_table_r1_per_launch": tile_r1, "changes_since_sc2e": CHANGES, "model": model(cc, fits)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    got = census()
    text = json.dumps(got, indent=1, sort_keys=True) + "\n"
    if a.check:
        want = OUT.read_text(encoding="utf-8")
        if want != text:
            print("DECODE64_CENSUS differs from the committed JSON", file=sys.stderr)
            return 1
        print("DECODE64_CENSUS reproduces the committed JSON")
        return 0
    OUT.write_bytes(text.encode("utf-8"))
    print(f"wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
