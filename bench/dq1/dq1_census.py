"""bench/dq1/dq1_census.py -- lane DQ1, BOX side (bench/dq1/DQ1-PREREG.md).

Is there a dense low-bit execution primitive worth building for dense QLoRA? This instrument reads the two quantities that
bound the answer, on one GPU, from the linear shapes of a real dense model (Qwen3-32B, read from its config; no checkpoint):

1. SPEED HEADROOM. Per frozen-base linear and per row count M: the time of the forward and of the input-gradient (dgrad)
   of every arm, on IDENTICAL NF4 weights:
     bf16   the NF4 weight decoded once to bf16 and kept resident; F.linear / autograd. The floor: no 4-bit route can beat
            cuBLAS on a resident bf16 weight at training M, so (bnb - bf16) / bnb bounds what ANY low-bit kernel can save.
     bnb    bitsandbytes ``matmul_4bit`` (what ``Linear4bit`` calls), nf4 / blocksize 64 / double-quant: the incumbent,
            whatever its own dispatch picks.
     dq     ``dequantize_4bit`` + ``F.linear`` / ``g @ W`` spelled out: the dequant-then-cuBLAS pattern.
     gnf4a  grouped-nf4-gemm at G=1 (one group = the dense linear) under ``GNF4_TRAIN_GEMM=auto``: the G=1 prototype.
     gnf4f  the same under ``GNF4_TRAIN_GEMM=fused``: the packed kernel 2026-08-15 measured on an A6000, re-read here.
   Plus each decoder alone (bnb ``dequantize_4bit``, gnf4 ``dequant_groups``) and a PEFT-shaped LoRA delta (r 16).
2. STREAMING FEASIBILITY. Pinned host->device bandwidth for one layer's NF4 bytes, alone and concurrent with that layer's
   GEMMs, so a frozen dense model larger than VRAM could be judged streamable (compute per layer vs transfer per layer).

Writes one JSON receipt (``--out``). The verdict is ``dq1_reduce.py``'s, never this file's.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time

import torch
import torch.nn.functional as F

# Qwen3-32B (config.json @ the registered revision): hidden 5120, intermediate 25600, 64 q heads, 8 kv heads, head_dim 128.
# name -> (N = out_features, K = in_features, count per decoder layer)
SHAPES = {
    "q_proj": (8192, 5120, 1),
    "kv_proj": (1024, 5120, 2),      # k_proj and v_proj share a shape
    "o_proj": (5120, 8192, 1),
    "gate_up_proj": (25600, 5120, 2),  # gate_proj and up_proj share a shape
    "down_proj": (5120, 25600, 1),
}
ARMS = ("bf16", "bnb", "dq", "gnf4a", "gnf4f")
ORDER = ARMS + tuple(reversed(ARMS))       # palindrome: every arm twice, mirrored, for a self-pair per arm


def warm_until_steady(min_s: float, max_s: float, tol: float = 0.01) -> dict:
    """Amendment 1. Sustained bf16 GEMM load until the card's throughput is STEADY, not for a fixed time. Run 1 (dq1-5090-1)
    read a boost transient: the first ~3 s of load after a 1.5 s warm-up ran ~10 % above the sustained clock, so the first
    arm of every cell (bf16, position 1) timed fast. A block is ~0.25 s of 4096^3 bf16 matmuls; stop once ``min_s`` has
    elapsed and the last four blocks' rates agree within ``tol`` (max/min - 1), or at ``max_s``. Returns what it saw."""
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    flop = 2 * 4096 ** 3
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    rates, t0 = [], time.time()
    while True:
        s.record()
        n = 0
        while True:
            torch.mm(a, b)
            n += 1
            if n % 16 == 0:
                e.record()
                e.synchronize()
                if s.elapsed_time(e) >= 250:
                    break
        rates.append(flop * n / (s.elapsed_time(e) * 1e-3) / 1e12)
        el = time.time() - t0
        last = rates[-4:]
        if (el >= min_s and len(last) == 4 and max(last) / min(last) - 1 <= tol) or el >= max_s:
            return {"seconds": round(el, 2), "blocks": len(rates), "first_tflops": round(rates[0], 1),
                    "steady_tflops": round(sum(last) / len(last), 1), "steady": max(last) / min(last) - 1 <= tol}


def rate_now() -> float:
    """One ~0.25 s block of the warm-up GEMM, TF/s: the clock proxy recorded at the end of each cell (Amendment 1)."""
    return warm_until_steady(0.0, 0.0)["first_tflops"]


def timed(fn, target_ms: float, draws: int) -> list:
    """ms per call: reps sized so one draw lasts ~target_ms, median taken by the reducer over ``draws`` draws."""
    fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    one = max(s.elapsed_time(e), 1e-3)
    reps = max(1, min(5000, int(target_ms / one)))
    out = []
    for _ in range(draws):
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / reps)
    return out


def rel_err(a: torch.Tensor, ref: torch.Tensor) -> float:
    return float((a.float() - ref).norm() / ref.norm())


class Weights:
    """One shape's NF4 weight in both encodings, decoded identically: bnb's (nested absmax, what Linear4bit stores) and
    grouped-nf4-gemm's [1, N, K//2] / fp32 absmax, repacked FROM the bnb tensors (gnf4's ``repack_from_bnb``), so every arm
    multiplies by the same decoded matrix."""

    def __init__(self, N: int, K: int, seed: int):
        import bitsandbytes.functional as BF
        from nf4_grouped import repack_from_bnb

        g = torch.Generator(device="cuda").manual_seed(seed)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g) * (1.0 / K ** 0.5)
        self.q, self.st = BF.quantize_4bit(w, blocksize=64, quant_type="nf4", compress_statistics=True)
        del w
        self.B, self.absmax = repack_from_bnb([self.q], [self.st], N, K)
        self.eids = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.w_dq = BF.dequantize_4bit(self.q, self.st).to(torch.bfloat16)      # [N, K], the decoded weight every arm uses
        self.N, self.K = N, K
        self.packed_bytes = self.q.numel() * self.q.element_size()
        self.bnb_state_bytes = sum(t.numel() * t.element_size() for t in
                                   (self.st.absmax, self.st.state2.absmax, self.st.state2.code, self.st.code))


def make_arm(arm: str, W: Weights, x: torch.Tensor):
    """-> (forward callable, the autograd output to differentiate, the leaf it is differentiated against)."""
    import bitsandbytes as bnb
    import bitsandbytes.functional as BF
    from nf4_qlora import gemm_4bit_grouped_train

    M = x.shape[0]
    xg = x.detach().requires_grad_(True)
    if arm == "bf16":
        def f(a):
            return F.linear(a, W.w_dq)
    elif arm == "bnb":
        def f(a):
            return bnb.matmul_4bit(a, W.q.t(), quant_state=W.st)
    elif arm == "dq":
        class _DQ(torch.autograd.Function):          # bnb MatMul4Bit's backward, with its forward spelled the same way
            @staticmethod
            def forward(ctx, a):
                return F.linear(a, BF.dequantize_4bit(W.q, W.st).to(a.dtype))

            @staticmethod
            def backward(ctx, go):
                return go @ BF.dequantize_4bit(W.q, W.st).to(go.dtype)
        f = _DQ.apply
    elif arm in ("gnf4a", "gnf4f"):
        def f(a):
            return gemm_4bit_grouped_train(a, W.B, W.absmax, [M], W.eids)
    else:
        raise ValueError(arm)
    return f, xg


def route_env(arm: str):
    os.environ["GNF4_TRAIN_GEMM"] = "fused" if arm == "gnf4f" else "auto"


def bnb_dispatch(M: int, N: int, K: int):
    """bitsandbytes' own forward decision at this shape on this card -- "custom" (its fused 4-bit GEMM) or "dequant"
    (dequantize_4bit + F.linear) -- read from bnb's heuristic, mirroring its gemm_4bit op's guards. None when this bnb has no
    such heuristic (the op is then whatever matmul_4bit does; recorded as unknown, never inferred)."""
    try:
        from bitsandbytes.backends.cuda import ops as bops
    except Exception:
        return None
    fn = getattr(bops, "_gemm_4bit_use_custom_cuda", None) or getattr(bops, "_gemm_4bit_use_custom", None)
    if fn is None:
        return None
    if M > 1536:
        return "dequant"
    if M <= 4:
        return "custom"
    try:
        return "custom" if fn(torch.cuda.current_device(), torch.bfloat16, M, N, K) else "dequant"
    except Exception as exc:  # recorded
        return f"unknown: {type(exc).__name__}"


def measure_cell(name, W: Weights, M: int, args) -> dict:
    """Every arm's forward and dgrad at one (shape, M), palindromic order, plus parity against an fp32 reference."""
    import bitsandbytes.functional as BF
    import nf4_qlora
    import nf4_route
    from nf4_route import dequant_groups

    torch.cuda.reset_peak_memory_stats()
    g = torch.Generator(device="cuda").manual_seed(1000 + M)
    x = torch.randn(M, W.K, device="cuda", dtype=torch.bfloat16, generator=g)
    go = torch.randn(M, W.N, device="cuda", dtype=torch.bfloat16, generator=g)
    y_ref = x.float() @ W.w_dq.float().t()
    dx_ref = go.float() @ W.w_dq.float()
    cell = {"shape": name, "N": W.N, "K": W.K, "M": M, "arms": {}, "bnb_dispatch": bnb_dispatch(M, W.N, W.K)}
    cell["warm"] = warm_until_steady(args.warm_s, args.warm_max_s)
    for pos, arm in enumerate(ORDER):
        route_env(arm)
        rec = cell["arms"].setdefault(arm, {"fwd_ms": [], "dgrad_ms": []})
        try:
            f, xg = make_arm(arm, W, x)
            before = dict(nf4_qlora.DGRAD_STATS, loop_reasons=dict(nf4_qlora.DGRAD_STATS["loop_reasons"]))
            rs0 = dict(nf4_route.ROUTE_STATS)
            y = f(xg)
            dx, = torch.autograd.grad(y, xg, go, retain_graph=True)
            if "route" not in rec:          # engagement, read from the libraries' own state on the first pass
                # the route the call TOOK: FusedGroupedNf4 stores it on its ctx, which is y.grad_fn
                rec["route"] = getattr(y.grad_fn, "route", None) if arm.startswith("gnf4") else arm
                rec["dgrad_stats_delta"] = {k: nf4_qlora.DGRAD_STATS[k] - before[k]
                                            for k in ("kernel", "grouped_mm", "dense", "loop")}
                rec["route_stats_delta"] = {k: nf4_route.ROUTE_STATS.get(k, 0) - rs0.get(k, 0)
                                            for k in nf4_route.ROUTE_STATS}
                rec["rel_err_fwd"] = rel_err(y, y_ref)
                rec["rel_err_dgrad"] = rel_err(dx, dx_ref)
                rec["finite"] = bool(torch.isfinite(y).all() and torch.isfinite(dx).all())
            rec["fwd_ms"].append(timed(lambda: f(xg), args.target_ms, args.draws))
            rec["dgrad_ms"].append(timed(lambda: torch.autograd.grad(y, xg, go, retain_graph=True),
                                         args.target_ms, args.draws))
        except Exception as exc:  # recorded, never fatal: an arm that cannot run at a shape is a reading
            rec.setdefault("errors", []).append(f"pos{pos}: {type(exc).__name__}: {str(exc)[:300]}")
        torch.cuda.empty_cache()
    os.environ["GNF4_TRAIN_GEMM"] = "auto"

    # decoders alone, and the LoRA delta (PEFT shape: dropout 0, (x A^T) B^T * alpha/r, A and B trainable bf16)
    try:
        cell["dequant_bnb_ms"] = timed(lambda: BF.dequantize_4bit(W.q, W.st), args.target_ms, args.draws)
        cell["dequant_gnf4_ms"] = timed(lambda: dequant_groups(W.B, W.absmax, W.eids, W.N, W.K), args.target_ms,
                                        args.draws)
        r, alpha = args.lora_r, args.lora_alpha
        A = (torch.randn(r, W.K, device="cuda", dtype=torch.bfloat16) / W.K ** 0.5).requires_grad_(True)
        Bm = (torch.randn(W.N, r, device="cuda", dtype=torch.bfloat16) * 1e-2).requires_grad_(True)
        xg = x.detach().requires_grad_(True)

        def lora(a):
            return F.linear(F.linear(a, A), Bm) * (alpha / r)
        yl = lora(xg)
        cell["lora_fwd_ms"] = timed(lambda: lora(xg), args.target_ms, args.draws)
        cell["lora_bwd_ms"] = timed(lambda: torch.autograd.grad(yl, (xg, A, Bm), go, retain_graph=True),
                                    args.target_ms, args.draws)
    except Exception as exc:  # recorded, never fatal
        cell["extras_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    cell["mem_peak_bytes"] = torch.cuda.max_memory_allocated()
    cell["warm"]["end_tflops"] = round(rate_now(), 1)
    torch.cuda.empty_cache()
    return cell


def layer_bytes(weights: dict) -> int:
    """One decoder layer's frozen NF4 bytes as Linear4bit stores them (packed nibbles + nested absmax state)."""
    return sum((W.packed_bytes + W.bnb_state_bytes) * SHAPES[n][2] for n, W in weights.items())


def h2d_probe(weights: dict, M: int, args) -> dict:
    """Pinned->device copies of one layer's NF4 bytes against one layer's bnb forward GEMMs, as two FULLY-LOADED readings:

    - copy under load: the GEMM window is enqueued first and sized to >= 2x the copy window, so every copy runs while GEMMs run;
    - GEMMs under load: the copy window is enqueued first and sized to >= 2x the GEMM window, so every GEMM runs during DMA.

    Each draw records whether the loaded stream really was covered end to end (its start after the other's start, its end
    before the other's end, on the device timeline); the reducer refuses a stream verdict from an uncovered draw."""
    import bitsandbytes as bnb

    nbytes = layer_bytes(weights)
    src = torch.empty(nbytes, dtype=torch.uint8).pin_memory()
    dst = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    cs = torch.cuda.Stream()
    xs = {n: torch.randn(M, W.K, device="cuda", dtype=torch.bfloat16) for n, W in weights.items()}

    def layer_fwd(k):
        for _ in range(k):
            for n, W in weights.items():
                for _ in range(SHAPES[n][2]):
                    bnb.matmul_4bit(xs[n], W.q.t(), quant_state=W.st)

    def copies(k):
        with torch.cuda.stream(cs):
            for _ in range(k):
                dst.copy_(src, non_blocking=True)

    def ev():
        return torch.cuda.Event(enable_timing=True)

    def window(first, second):
        """Enqueue `first` then `second` on their streams (no sync between). -> (first_ms, second_ms, second covered)."""
        a, b, c, d = ev(), ev(), ev(), ev()
        torch.cuda.synchronize()
        s1, f1 = first
        s2, f2 = second
        a.record(s1)
        f1()
        b.record(s1)
        c.record(s2)
        f2()
        d.record(s2)
        torch.cuda.synchronize()
        covered = a.elapsed_time(c) >= 0 and d.elapsed_time(b) >= 0
        return a.elapsed_time(b), c.elapsed_time(d), covered

    out = {"layer_bytes": nbytes, "M": M, "draws": [], "warm": warm_until_steady(args.warm_s, args.warm_max_s)}
    main = torch.cuda.current_stream()
    for _ in range(args.draws):
        a, b = ev(), ev()
        a.record(cs)
        copies(args.h2d_reps)
        b.record(cs)
        torch.cuda.synchronize()
        copy_alone = a.elapsed_time(b) / args.h2d_reps
        c, d = ev(), ev()
        c.record()
        layer_fwd(args.h2d_reps)
        d.record()
        torch.cuda.synchronize()
        fwd_alone = max(c.elapsed_time(d) / args.h2d_reps, 1e-3)     # one layer's forward, same rep count as the loaded read
        # copy under load: GEMMs first, >= 2x the copy window
        k_fwd = max(2, int(2 * copy_alone * args.h2d_reps / fwd_alone) + 1)
        _, copy_ms, copy_cov = window((main, lambda: layer_fwd(k_fwd)), (cs, lambda: copies(args.h2d_reps)))
        # GEMMs under load: copies first, >= 2x the GEMM window
        k_cp = max(2, int(2 * fwd_alone * args.h2d_reps / copy_alone) + 1)
        _, gemm_ms, gemm_cov = window((cs, lambda: copies(k_cp)), (main, lambda: layer_fwd(args.h2d_reps)))
        out["draws"].append({
            "copy_alone_ms": copy_alone, "copy_alone_gbs": nbytes / copy_alone / 1e6,
            "copy_loaded_gbs": nbytes * args.h2d_reps / copy_ms / 1e6, "copy_covered": bool(copy_cov),
            "gemm_alone_ms": fwd_alone, "gemm_loaded_ms": gemm_ms / args.h2d_reps, "gemm_covered": bool(gemm_cov),
            "layer_fwds_under_copy": k_fwd, "copies_under_gemm": k_cp,
        })
    torch.cuda.empty_cache()
    return out


def forensics() -> dict:
    q = "name,memory.total,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,clocks.max.sm,power.limit"
    try:
        smi = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=30).stdout.strip()
    except Exception as exc:  # recorded
        smi = f"unavailable: {exc}"
    import importlib.metadata as md
    vers = {}
    for d in ("torch", "triton", "bitsandbytes", "grouped-nf4-gemm", "experts4bit-qlora", "transformers"):
        try:
            vers[d] = md.version(d)
        except Exception:
            vers[d] = None
    commits = {}
    for d in ("grouped-nf4-gemm", "experts4bit-qlora"):
        try:
            commits[d] = json.loads(md.distribution(d).read_text("direct_url.json")).get("vcs_info", {}).get("commit_id")
        except Exception:
            commits[d] = None
    cpu = ""
    try:
        cpu = next(ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name"))
    except Exception:
        pass
    return {"nvidia_smi": smi, "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
            "cuda": torch.version.cuda, "versions": vers, "commits": commits, "cpu": cpu, "python": platform.python_version()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--rows", default="512,1024,2048,4096,8192")
    p.add_argument("--shapes", default=",".join(SHAPES))
    p.add_argument("--draws", type=int, default=5)
    p.add_argument("--target-ms", type=float, default=60.0)
    p.add_argument("--warm-s", type=float, default=4.0, help="Amendment 1: minimum warm-up before steadiness is tested")
    p.add_argument("--warm-max-s", type=float, default=30.0, help="Amendment 1: warm-up cap")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--h2d-rows", default="1024,2048,4096")
    p.add_argument("--h2d-reps", type=int, default=4)
    p.add_argument("--rehearsal", action="store_true", help="A2000 correctness rehearsal: a receipt that can never be graded")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True   # torch's default, stated
    rows = [int(v) for v in args.rows.split(",") if v]
    names = [n for n in args.shapes.split(",") if n]
    receipt = {"schema": "dq1-census/1", "rehearsal": bool(args.rehearsal), "started_at": time.strftime("%FT%TZ", time.gmtime()),
               "forensics": forensics(), "shapes": {n: SHAPES[n] for n in names}, "rows": rows, "order": list(ORDER),
               "config": {k: getattr(args, k) for k in ("draws", "target_ms", "warm_s", "warm_max_s", "lora_r", "lora_alpha",
                                                        "h2d_reps")},
               "amendment": 1,
               "cells": [], "h2d": []}

    def flush():
        tmp = args.out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(receipt, fh, indent=1)
        os.replace(tmp, args.out)

    weights = {}
    for i, n in enumerate(names):
        N, K, _ = SHAPES[n]
        weights[n] = Weights(N, K, seed=i)
        for M in rows:
            t0 = time.time()
            receipt["cells"].append(measure_cell(n, weights[n], M, args))
            print(f"cell {n} M={M} {time.time() - t0:.1f}s", flush=True)
            flush()
    if set(names) == set(SHAPES):
        for M in [int(v) for v in args.h2d_rows.split(",") if v]:
            receipt["h2d"].append(h2d_probe(weights, M, args))
            print(f"h2d M={M} done", flush=True)
            flush()
    receipt["finished_at"] = time.strftime("%FT%TZ", time.gmtime())
    flush()
    print("DQ1 census done", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
