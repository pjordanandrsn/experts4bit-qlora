#!/usr/bin/env python3
"""Measured decode and training throughput for every e4b execution config.

The dtype sweep answers "does precision change the answer". It says nothing
about the configs that actually exist here: e4b exposes sixteen ``enable_*``
entry points — resident, pipelined, cold-engine, offloaded, NVMe-backed, fused
— and each is a different execution path through the same weights. A user
choosing between them has no basis to choose without numbers.

So this varies the CONFIG, holding the weights and the shapes fixed:

* **decode** at ``T=1`` — the shape that dominates interactive serving, where
  per-call overhead is the whole cost and a streaming tier hurts most.
* **prefill** at ``T=512`` — the shape where bandwidth dominates instead, and
  the ranking often inverts.
* **training** through ``ExpertsLoRA``, across the three training lanes.

Every row also records peak VRAM, because the reason to accept a slower config
is almost always that it fits when the faster one does not. A throughput table
without the memory column cannot answer the question anyone is really asking.

Configs that cannot run on this host are reported as ``unavailable`` with the
reason rather than omitted — a missing row reads as "not measured", but a
silently absent config reads as "does not exist".
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import traceback

import torch

import experts4bit_qlora as E
from experts4bit_qlora import ExpertsLoRA, ExpertsNbit


def geometry(name):
    """Expert-stack shapes taken from real released models."""
    return {
        # num_experts, hidden, intermediate, top_k
        "olmoe": (64, 2048, 1024, 8),
        "qwen1.5-moe": (60, 2048, 1408, 4),
        "mixtral": (8, 4096, 14336, 2),
        "tiny": (8, 512, 256, 2),
    }[name]


def build(geo, dtype, quant_type="nf4", blocksize=64, device="cuda", seed=0):
    n_e, hid, inter, _ = geo
    torch.manual_seed(seed)
    gate_up = (torch.randn(n_e, 2 * inter, hid) * 0.02).to(dtype)
    down = (torch.randn(n_e, hid, inter) * 0.02).to(dtype)
    base = ExpertsNbit.from_float(gate_up, down, compute_dtype=dtype,
                                  quant_type=quant_type, blocksize=blocksize)
    del gate_up, down
    return base.to(device)


def inputs(geo, T, dtype, device):
    n_e, hid, _, k = geo
    hs = (torch.randn(T, hid) * 1.5).to(dtype).to(device)
    idx = torch.stack([torch.randperm(n_e)[:k] for _ in range(T)]).to(device)
    w = torch.softmax(torch.randn(T, k), dim=-1).to(dtype).to(device)
    return hs, idx, w


def timed(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


# --- configs ---------------------------------------------------------------
# Each entry: name -> (setup(base, geo) -> teardown_or_None, requires)
def _hot_sets(geo, frac):
    """Which experts stay resident. frac=1.0 all-hot, 0.0 all-cold."""
    n_e = geo[0]
    return [list(range(int(n_e * frac)))]


INFER_CONFIGS = [
    ("baseline", lambda b, g: None, None),
    ("fast", lambda b, g: (E.enable_fast(b), E.disable_fast)[1], "fast"),
    ("hot_residency_all_hot",
     lambda b, g: (E.enable_hot_residency(b, _hot_sets(g, 1.0)),
                   E.disable_hot_residency)[1], "hot"),
    ("hot_residency_half",
     lambda b, g: (E.enable_hot_residency(b, _hot_sets(g, 0.5)),
                   E.disable_hot_residency)[1], "hot"),
    ("pipelined_residency_half",
     lambda b, g: (E.enable_pipelined_residency(b, _hot_sets(g, 0.5)),
                   E.disable_pipelined_residency)[1], "hot"),
    ("cold_engine_half",
     lambda b, g: (E.enable_cold_engine(b, _hot_sets(g, 0.5)),
                   E.disable_cold_engine)[1], "cold"),
    ("cold_engine_all_cold",
     lambda b, g: (E.enable_cold_engine(b, _hot_sets(g, 0.0)),
                   E.disable_cold_engine)[1], "cold"),
]

TRAIN_CONFIGS = [
    ("lora_baseline", lambda m: None, None),
    ("fast_train_dgrad_off", lambda m: E.enable_fast_train(m, dgrad=False), "fast"),
    ("fast_train_dgrad_on", lambda m: E.enable_fast_train(m, dgrad=True), "fast"),
    ("batched_train", lambda m: E.enable_batched_train(m), "batched"),
]


def availability():
    return {
        "fast": (E.fast_available(), "fused grouped kernel (needs CUDA+triton)"),
        "hot": (E.hot_residency_available(), "hot residency (needs CUDA)"),
        "cold": (E.cold_engine_available(), "cold engine"),
        "batched": (E.batched_train_available(), "batched training lane"),
    }


def run_infer(geo, dtype, device, quant_type, blocksize, emit):
    avail = availability()
    for name, setup, req in INFER_CONFIGS:
        row = {"kind": "infer", "config": name, "dtype": str(dtype).split(".")[-1],
               "quant": quant_type, "blocksize": blocksize}
        if req and not avail[req][0]:
            row["status"] = f"unavailable: {avail[req][1]}"
            emit(row)
            continue
        base = None
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            base = build(geo, dtype, quant_type, blocksize, device)
            teardown = setup(base, geo)
            for label, T in (("decode", 1), ("prefill", 512)):
                hs, idx, w = inputs(geo, T, dtype, device)
                with torch.no_grad():
                    # Bound as defaults, not captured: the names are rebound each
                    # iteration and released below.
                    dt = timed(lambda b=base, h=hs, i=idx, ww=w: b(h, i, ww))
                row[f"{label}_tok_s"] = round(T / dt, 1)
                row[f"{label}_ms"] = round(dt * 1e3, 3)
                hs = idx = w = None
            row["peak_gb"] = round(
                torch.cuda.max_memory_allocated() / 1e9, 3) if device == "cuda" else 0.0
            if callable(teardown):
                teardown(base)
            row["status"] = "OK"
        except Exception as e:
            row["status"] = f"FAIL {type(e).__name__}: {str(e)[:80]}"
            traceback.print_exc()
        finally:
            base = None
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        emit(row)


def run_train(geo, dtype, device, quant_type, blocksize, emit, steps=8, T=256):
    avail = availability()
    for name, setup, req in TRAIN_CONFIGS:
        row = {"kind": "train", "config": name, "dtype": str(dtype).split(".")[-1],
               "quant": quant_type, "blocksize": blocksize}
        if req and not avail[req][0]:
            row["status"] = f"unavailable: {avail[req][1]}"
            emit(row)
            continue
        mod = None
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            base = build(geo, dtype, quant_type, blocksize, device)
            # Adapters in fp32 regardless of base dtype: fp16 adapters produce
            # non-finite grads on every architecture measured (this is what peft
            # does, and the reason the earlier matrix showed 14/21 rather than
            # 21/21).
            mod = ExpertsLoRA(base, r=8, alpha=16, dtype=torch.float32).to(device)
            setup(mod)
            ps = [p for p in mod.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(ps, lr=1e-4)
            hs, idx, w = inputs(geo, T, dtype, device)
            hs.requires_grad_(False)

            def step(_m=mod, _h=hs, _i=idx, _w=w):
                out = _m(_h, _i, _w)
                out.float().pow(2).mean().backward()
                opt.step()
                opt.zero_grad(set_to_none=True)

            step()          # warm up allocator + any lazy compile
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(steps):
                step()
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            row["train_tok_s"] = round(steps * T / dt, 1)
            row["step_ms"] = round(dt / steps * 1e3, 2)
            row["trainable"] = sum(p.numel() for p in ps)
            row["peak_gb"] = round(
                torch.cuda.max_memory_allocated() / 1e9, 3) if device == "cuda" else 0.0
            row["status"] = "OK"
        except Exception as e:
            row["status"] = f"FAIL {type(e).__name__}: {str(e)[:80]}"
            traceback.print_exc()
        finally:
            mod = None
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        emit(row)


COLS = ["kind", "config", "dtype", "quant", "blocksize", "decode_tok_s",
        "decode_ms", "prefill_tok_s", "train_tok_s", "step_ms", "peak_gb", "status"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="olmoe",
                    choices=["olmoe", "qwen1.5-moe", "mixtral", "tiny"])
    ap.add_argument("--dtypes", default="bfloat16")
    ap.add_argument("--quants", default="nf4,fp4")
    ap.add_argument("--blocksizes", default="64")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="append JSONL here as rows finish")
    a = ap.parse_args()

    geo = geometry(a.geometry)
    print(f"# geometry {a.geometry}: E={geo[0]} hid={geo[1]} inter={geo[2]} k={geo[3]}")
    for k, (ok, why) in availability().items():
        print(f"#   {k}: {'available' if ok else 'UNAVAILABLE'} ({why})")
    print("| " + " | ".join(COLS) + " |", flush=True)
    print("|" + "|".join("---" for _ in COLS) + "|", flush=True)

    def emit(row):
        # Flush per row: a config that OOMs must not cost the rows before it.
        print("| " + " | ".join(str(row.get(c, "-")) for c in COLS) + " |", flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")

    for d in a.dtypes.split(","):
        dtype = getattr(torch, d.strip())
        for q in a.quants.split(","):
            for bs in a.blocksizes.split(","):
                run_infer(geo, dtype, a.device, q.strip(), int(bs), emit)
                run_train(geo, dtype, a.device, q.strip(), int(bs), emit)
    print("CONFIG_MATRIX_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
