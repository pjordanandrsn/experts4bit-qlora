# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P236's op-level head-to-head (bench/p236/PREREG-p236.md): grouped-nf4-gemm 0.51.0's decode GEMV against
bitsandbytes 0.50.2's ``matmul_4bit``, on the SAME NF4 bytes, at M = 1, 2, 4, 8 and 16 rows of one expert projection.

- **gnf4**: ``nf4_grouped.gemm_4bit_grouped(x, B[None], absmax[None], [1] * M, eids)``, one singleton group per row,
  every knob at its default: the call experts4bit-qlora's all-resident collapse makes for a decode step (at one row by
  default; at 2..8 rows under ``E4B_NF4_SMALLM_BW``). At these shapes on a >= 160-SM card grouped-nf4-gemm's ``auto``
  routes it to K33, the bandwidth GEMV: one launch per call, counted in ``bw_prmt32``.
- **bnb**: ``bitsandbytes.matmul_4bit(x, qW.t(), quant_state=qs)``, which dispatches ``bitsandbytes::gemm_4bit``: its fused
  ``cgemm_4bit`` kernel or dequantize then ``F.linear``, by its own per-arch rule. The route is counted, not chosen.

The weights: per projection, a seeded random bf16 matrix quantized ONCE by ``bitsandbytes.functional.quantize_4bit``
(NF4, block 64, no double quant). grouped-nf4-gemm reads the same bytes as ``[N, K/2]`` and the same fp32 absmax as
``[N, K/64]``; bitsandbytes' ``dequantize_4bit`` must equal ``nf4_grouped.dequant_ref`` rounded to the weight dtype on
those bytes (the premise, per cell). Timing rotates over enough distinct weight copies to span 4x the card's L2, so no
arm reads a cached weight. Per cell: the premise, correctness against the fp32 reference and an eager engagement pass,
gated by the reducer's own cell rules BEFORE any timing (a failed cell is recorded with its reason and never timed);
then a warm-up and REPS reps of the palindrome (odd reps swap the arms), each visit ITERS eager calls between CUDA
events.

    python p236_bench.py --out bench_p236.json
    python p236_bench.py --out dry.json --dry        (CPU, no CUDA: the control flow and the record, never a reading)
    python p236_bench.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p236_reduce as R  # noqa: E402  (the registered constants and rules: the staged sibling)

KNOBS = ("GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE", "GNF4_GEMV_DOTPAD", "GNF4_DECODE_PLAN",
         "GNF4_GEMV_SPLITK", "GNF4_PDL", "BNB_CUDA_VERSION")


class Real:
    """The two libraries on CUDA, with the route counters each arm's engagement reads."""

    dry = False

    def __init__(self, torch):
        import bitsandbytes as bnb
        import bitsandbytes.functional as BF
        import bitsandbytes.backends.cuda.ops as BO
        import nf4_grouped as ng
        self.torch, self.bnb, self.BF, self.BO, self.ng = torch, bnb, BF, BO, ng
        self.dev = torch.device("cuda")
        self.route = {k: 0 for k in R.BNB_ROUTES}

    def quantize(self, W):
        qW, qs = self.BF.quantize_4bit(W, blocksize=64, quant_type="nf4", compress_statistics=False)
        n, k = W.shape
        return {"qW": qW, "qs": qs, "B": qW.view(1, n, k // 2), "absmax": qs.absmax.view(1, n, k // 64).float()}

    def dequant_bnb(self, w):
        return self.BF.dequantize_4bit(w["qW"], w["qs"])

    def gnf4(self, x, w, eids):
        return self.ng.gemm_4bit_grouped(x, w["B"], w["absmax"], [1] * x.shape[0], eids)

    def bnb_call(self, x, w):
        return self.bnb.matmul_4bit(x, w["qW"].t(), quant_state=w["qs"])

    def gemv_counts(self):
        return dict(self.ng.dispatch_counts())

    def count_routes(self, on):
        """Wrap (on) or restore (off) bitsandbytes' two gemm_4bit routes, module globals its CUDA kernel looks up at
        call time."""
        if on:
            self._orig = (self.BO._gemm_4bit_kernel_impl, self.BO._dequant_linear_fallback)
            ki, df = self._orig

            def k_(*a, **kw):
                self.route["custom"] += 1
                return ki(*a, **kw)

            def d_(*a, **kw):
                self.route["dequant"] += 1
                return df(*a, **kw)
            self.BO._gemm_4bit_kernel_impl, self.BO._dequant_linear_fallback = k_, d_
        else:
            self.BO._gemm_4bit_kernel_impl, self.BO._dequant_linear_fallback = self._orig

    def timer(self):
        t = self.torch

        class T:
            def __enter__(s):
                s.e0, s.e1 = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
                s.e0.record()
                return s

            def __exit__(s, *exc):
                s.e1.record()
                s.e1.synchronize()
                s.ms = s.e0.elapsed_time(s.e1)
        return T()

    def sync(self):
        self.torch.cuda.synchronize()


class Dry(Real):
    """CPU stand-ins with the same record (CI: the control flow, the record and the reducer's read; never a reading).
    Both arms are the fp32 reference rounded to bf16; the counters advance as the real routes would."""

    dry = True

    def __init__(self, torch):
        import nf4_grouped as ng
        from nf4_pack_ref import quantize_pack_nf4
        self.torch, self.ng, self.qp = torch, ng, quantize_pack_nf4
        self.dev = torch.device("cpu")
        self.route = {k: 0 for k in R.BNB_ROUTES}
        self.gemv = {k: 0 for k in R.GEMV_KEYS}

    def quantize(self, W):
        packed, absmax = self.qp(W.float())
        n, k = W.shape
        return {"B": packed.view(1, n, k // 2), "absmax": absmax.view(1, n, k // 64).float(), "n": n, "k": k}

    def dequant_bnb(self, w):
        return self.ng.dequant_ref(w["B"][0], w["absmax"][0], w["n"], w["k"]).to(self.torch.bfloat16)

    def _mm(self, x, w):
        return (x.float() @ self.ng.dequant_ref(w["B"][0], w["absmax"][0], w["n"], w["k"]).t()).to(self.torch.bfloat16)

    def gnf4(self, x, w, eids):
        self.gemv["bw_prmt32"] += 1
        return self._mm(x, w)

    def bnb_call(self, x, w):
        self.route["custom" if x.shape[0] <= 4 else "dequant"] += 1
        return self._mm(x, w)

    def gemv_counts(self):
        return dict(self.gemv)

    def count_routes(self, on):
        pass

    def timer(self):
        class T:
            def __enter__(s):
                s.t0 = time.perf_counter()
                return s

            def __exit__(s, *exc):
                s.ms = (time.perf_counter() - s.t0) * 1e3
        return T()

    def sync(self):
        pass


def run(a):
    import torch
    for k in KNOBS:
        os.environ.pop(k, None)                 # every library knob at its default: the libraries as shipped
    be = Dry(torch) if a.dry else Real(torch)
    shapes = dict(R.SHAPES) if not a.dry else {s: (128, 256) for s in R.SHAPES}   # dry: small, block-aligned
    reps, iters, warm = (R.REPS, R.ITERS, R.WARM) if not a.dry else (4, 3, 1)
    t0 = time.time()
    rec = {"lane": R.LANE, "dry": be.dry, "status": "error", "reps": reps, "iters": iters, "warm": warm,
           "shapes": {s: list(v) for s, v in shapes.items()}, "ms": list(R.MS), "cells": {}, "torch": torch.__version__}
    try:
        import importlib.metadata as md
        rec["gnf4"] = md.version("grouped-nf4-gemm")
        try:
            rec["gnf4_sha"] = json.loads(md.distribution("grouped-nf4-gemm").read_text("direct_url.json") or "{}").get(
                "vcs_info", {}).get("commit_id")
        except Exception:                        # noqa: BLE001  (an editable or unrecorded install: no commit)
            rec["gnf4_sha"] = None
        rec["bnb"] = md.version("bitsandbytes")
        rec["provenance"] = {name: R.dist_record(name) for name in R.RELEASES}   # the reducer's release proof
    except Exception as e:                       # noqa: BLE001
        rec["versions_error"] = f"{type(e).__name__}: {e}"
    if not be.dry:
        p = torch.cuda.get_device_properties(0)
        rec.update(device=torch.cuda.get_device_name(0), cc=list(torch.cuda.get_device_capability()),
                   sm_count=p.multi_processor_count, l2_bytes=getattr(p, "L2_cache_size", None))

    def out(rc):
        rec["wall_s"] = round(time.time() - t0, 1)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)
        print(f"P236_BENCH rc={rc} status={rec['status']} cells={len(rec['cells'])} in {rec['wall_s']} s", flush=True)
        return rc
    try:
        g = torch.Generator(device=be.dev).manual_seed(236)
        for s_i, (shape, (n, k)) in enumerate(shapes.items()):
            n_copies = R.copies_for(n, k) if not be.dry else 2
            W = [be.quantize((torch.randn(n, k, generator=g, device=be.dev) * 0.02).to(torch.bfloat16))
                 for _ in range(n_copies)]
            w0 = W[0]
            ref_w = be.ng.dequant_ref(w0["B"][0], w0["absmax"][0], n, k)                       # fp32 [N, K]
            deq_equal = bool(torch.equal(be.dequant_bnb(w0), ref_w.to(torch.bfloat16)))
            for m in R.MS:
                key = R.cell_key(shape, m)
                x = torch.randn(m, k, generator=g, device=be.dev).to(torch.bfloat16)
                eids = torch.zeros(m, dtype=torch.int32, device=be.dev)
                ref = x.float() @ ref_w.t()
                scale = ref.abs().max().item()
                rel = {"gnf4": ((be.gnf4(x, w0, eids).float() - ref).abs().max().item()) / scale,
                       "bnb": ((be.bnb_call(x, w0).float() - ref).abs().max().item()) / scale}
                # engagement: one eager pass of 3 calls per arm, counted
                be.sync()
                c0 = be.gemv_counts()
                for i in range(3):
                    be.gnf4(x, W[i % n_copies], eids)
                be.sync()
                c1 = be.gemv_counts()
                g_eng = {key_: c1.get(key_, 0) - c0.get(key_, 0) for key_ in R.GEMV_KEYS}
                g_eng["calls"] = 3
                for key_ in be.route:
                    be.route[key_] = 0
                be.count_routes(True)
                try:
                    for i in range(3):
                        be.bnb_call(x, W[i % n_copies])
                    be.sync()
                finally:
                    be.count_routes(False)
                b_eng = dict(be.route, calls=3)
                cell = {"n": n, "k": k, "m": m, "copies": n_copies, "dequant_equal": deq_equal, "rel_err": rel,
                        "engaged": {"gnf4": g_eng, "bnb": b_eng}, "reps": []}
                rec["cells"][key] = cell
                gate = R.cell_faults(key, {**cell, "reps": [{"order": R.order(i), "ms_per_call": [1.0] * len(R.ORDER)}
                                                             for i in range(R.REPS)]})
                if gate:                                 # the gates come before any timing: a failed cell is never timed
                    cell["gate_failed"] = gate
                    print(f"P236_CELL {key} GATE FAILED, not timed: {gate}", flush=True)
                    continue
                fns = {"gnf4": lambda i: be.gnf4(x, W[i % n_copies], eids), "bnb": lambda i: be.bnb_call(x, W[i % n_copies])}
                for arm in R.ARMS:                                           # warm, untimed
                    for i in range(warm):
                        fns[arm](i)
                be.sync()
                reps_out = []
                for r in range(reps):
                    o = R.order(r)
                    ms = []
                    for arm in o:
                        with be.timer() as tm:
                            for i in range(iters):
                                fns[arm](i)
                        ms.append(tm.ms / iters)
                    reps_out.append({"order": o, "ms_per_call": ms})
                cell["reps"] = reps_out
                print(f"P236_CELL {key} copies={n_copies} deq_eq={deq_equal} rel={rel['gnf4']:.4f}/{rel['bnb']:.4f} "
                      f"gnf4={g_eng['bw_prmt32']}/3 bnb={b_eng} med_ms g={sorted(x_['ms_per_call'][0] for x_ in reps_out)[len(reps_out) // 2]:.4f}",
                      flush=True)
            W = w0 = ref_w = None
            if not be.dry:
                torch.cuda.empty_cache()
        rec["status"] = "ok"
    except Exception as e:                       # noqa: BLE001  (recorded; the reducer VOIDs a record that is not ok)
        rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        return out(34)
    return out(0)


def self_test():
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    src = src[:src.index("def " + "self_test():")]
    checks = [
        ("the gnf4 call is e4b's singleton decode call",
         'self.ng.gemm_4bit_grouped(x, w["B"], w["absmax"], [1] * x.shape[0], eids)' in src),
        ("the bnb call is matmul_4bit on the transposed packed weight",
         'self.bnb.matmul_4bit(x, w["qW"].t(), quant_state=w["qs"])' in src),
        ("one quantization, the same bytes for both",
         'quantize_4bit(W, blocksize=64, quant_type="nf4", compress_statistics=False)' in src
         and '"B": qW.view(1, n, k // 2)' in src),
        ("the premise", "torch.equal(be.dequant_bnb(w0), ref_w.to(torch.bfloat16))" in src),
        ("the reference is fp32", "ref = x.float() @ ref_w.t()" in src),
        ("the order is the reducer's", "o = R.order(r)" in src),
        ("the gates before any timing", src.index("gate = R.cell_faults(key,") < src.index("for r in range(reps):")
         and "continue" in src[src.index("if gate:"):src.index("fns = {")]),
        ("weights rotate", "W[i % n_copies]" in src and "R.copies_for(n, k)" in src),
        ("knobs default", "os.environ.pop(k, None)" in src),
        ("the reducer's shape", R.REPS == 20 and R.ITERS == 50 and len(R.SHAPES) == 6 and R.MS == (1, 2, 4, 8, 16)),
    ]
    bad = [n for n, ok in checks if not ok]
    print(f"p236_bench self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(checks)} checks)")
    return 0 if not bad else 1


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--check-args", action="store_true", help="parse and validate the arguments, then exit (tests)")
    a = p.parse_args(argv)
    if a.check_args:
        ok = bool(a.self_test or a.out)
        print("ARGS OK" if ok else "ARGS BAD")
        return 0 if ok else 2
    if a.self_test:
        return self_test()
    if not a.out:
        p.error("--out is required")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
