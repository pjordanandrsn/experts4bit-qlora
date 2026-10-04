#!/usr/bin/env python
"""Per-family training-step ladder: where a family's step time goes, and which of the Qwen optimisations engaged on it.

One process = one load of one family, then each requested rung in turn. A rung is a set of environment switches (every one is
read at call time by e4b / grouped-nf4-gemm, or at enable_fast_train time, which is re-run per rung after the previous rung's
patches are unwound), so rungs are A/B-able on one box and one load:

  reference   no enable_fast_train: the ExpertsLoRA reference forward and its recompute backward
  fused_pre   enable_fast_train(dgrad=True) with every Qwen-campaign default turned back off (legacy 5-read grouping, no
              pinned ring, no host reuse, untrimmed padded delta, max-group tile, composite RMSNorm, composite RoPE)
  fused       enable_fast_train(dgrad=True) at the package defaults -- the shipped path
  keep        fused + E4B_MOE_KEEP_LAYERS=all + NF4_QLORA_COMPACT_DELTA=1 (the memory-for-speed configuration)

Per rung it records s/step (median of the timed steps, synced), tokens/s, peak allocated/reserved VRAM, host RSS, every
engagement counter the two packages expose (patched/skipped expert modules, dgrad route, LoRA route, RMSNorm variants and
fallbacks, RoPE, MoE-keep layers, host reuse, pinned-ring staged/waits/overflow, train-GEMM route), the host->device syncs of
one step (torch.cuda.set_sync_debug_mode), and -- with --profile -- one step's device time by kernel family plus the device-busy
fraction. Not a parity harness: loss is printed for sanity only; parity verdicts come from the registered lanes.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
import warnings

RUNGS = {
    "reference": {},
    "fused_pre": {"E4B_GROUPING": "legacy", "GNF4_PINNED_RING": "0", "GNF4_HOST_REUSE": "0", "NF4_QLORA_LEAN_DELTA": "0",
                  "GNF4_PREFILL_TILE_RULE": "max", "E4B_FUSED_RMSNORM": "0", "E4B_FUSED_ROPE": "0"},
    "fused": {},
    "keep": {"E4B_MOE_KEEP_LAYERS": "all", "NF4_QLORA_COMPACT_DELTA": "1"},
}


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--rungs", required=True, help="comma list of rungs, run in ONE process on one load, interleaved "
                    "A..Z Z..A (reference once, last)")
    ap.add_argument("--offload", type=int, default=0)
    ap.add_argument("--attn4", type=int, default=1)
    ap.add_argument("--attn-lora", type=int, default=1)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--adapter-dtype", choices=["fp32", "bf16"], default="bf16",
                    help="bf16 = the SHIPPED configuration (train.py DTYPE; TC1 native); fp32 = TC1's matched-comparator arm")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--mb", type=int, default=1)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--profile", type=int, default=1)
    ap.add_argument("--env", default="", help="extra KEY=VAL,KEY=VAL applied after the rung's switches")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def kernel_family(name: str) -> str:
    n = name.lower()
    if any(t in n for t in ("nf4", "gnf4", "_grouped", "dgrad", "lora_delta", "dequant", "grouped_mm", "scatter_combine")):
        return "moe_expert"
    if any(t in n for t in ("flash", "fmha", "sdpa", "attention", "efficient_attention", "cutlass_fmha")):
        return "attention_core"
    if "rms" in n:
        return "rmsnorm"
    if "rope" in n:
        return "rope"
    if any(t in n for t in ("chunk_", "fused_recurrent", "gated_delta", "causal_conv", "selective_scan", "mamba", "ssd_")):
        return "recurrent_ssm"
    if any(t in n for t in ("gemm", "gemv", "cutlass", "sm80_xmma", "sm90_xmma", "ampere_", "cublas", "matmul", "magma")):
        return "dense_gemm"
    if any(t in n for t in ("memcpy", "memset")):
        return "memcpy"
    if any(t in n for t in ("reduce", "sum", "softmax", "topk", "sort", "argsort", "radix", "scan", "index", "gather",
                            "scatter", "cat", "elementwise", "vectorized", "unrolled", "copy", "fill", "where", "mul", "add")):
        return "elementwise_reduce"
    return "other"


def counters():
    out = {}
    try:
        from experts4bit_qlora.engines import fast as ef
        out["fast_train"] = {"patched": ef.FAST_TRAIN_STATS["patched"], "skipped": dict(ef.FAST_TRAIN_STATS["skipped"]),
                             "recurrent_fallbacks": list(ef.FAST_TRAIN_STATS.get("recurrent_fallbacks", []))}
    except Exception as e:  # noqa: BLE001
        out["fast_train"] = repr(e)
    try:
        import nf4_qlora as nq
        out["dgrad"] = {k: (dict(v) if isinstance(v, dict) else int(v)) for k, v in getattr(nq, "DGRAD_STATS", {}).items()}
        out["lora_path"] = {k: int(v) for k, v in nq.LORA_PATH_STATS.items()}
        out["lora_pad"] = {k: float(v) for k, v in nq.LORA_PAD_WASTE.items()}
    except Exception as e:  # noqa: BLE001
        out["dgrad"] = repr(e)
    try:
        import nf4_grouped as ng
        rings = list(getattr(ng, "_RINGS", {}).values())
        out["ring"] = {"staged": sum(r.staged for r in rings), "waits": sum(r.waits for r in rings),
                       "overflow": sum(getattr(r, "overflow", 0) for r in rings)}
        out["host_reuse"] = {k: int(v) for k, v in ng.HOST_REUSE_STATS.items()}
        out["prefill_bm"] = {str(k): int(v) for k, v in (getattr(ng, "PREFILL_BM_STATS", None) or {}).items()}
    except Exception as e:  # noqa: BLE001
        out["ring"] = repr(e)
    try:
        import nf4_route as nr
        out["route"] = {"train_gemm": nr.train_gemm_route(), "stats": {k: int(v) for k, v in nr.ROUTE_STATS.items()}}
    except Exception as e:  # noqa: BLE001
        out["route"] = repr(e)
    try:
        from experts4bit_qlora.engines import rmsnorm_train as rt, rope_train as rp, moe_keep as mk
        out["rmsnorm"] = dict(rt.RMSNORM_TRAIN_STATS)
        out["rope"] = dict(rp.ROPE_TRAIN_STATS)
        out["rope_refused"] = sorted(set(getattr(rp, "ROPE_TRAIN_STATS_REFUSED", [])))
        out["moe_keep"] = dict(mk.MOE_KEEP_STATS)
    except Exception as e:  # noqa: BLE001
        out["rmsnorm"] = repr(e)
    return out


def alpaca_batches(tok, seq, mb, n):
    """Deterministic packed Alpaca text (real routing distribution, not random ids)."""
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ids, i = [], 0
    need = seq * mb * n + 1
    while len(ids) < need:
        r = ds[i]
        ids += tok(f"{r['instruction']}\n{r['input']}\n{r['output']}", add_special_tokens=False)["input_ids"] + [tok.eos_token_id or 0]
        i += 1
    import torch
    t = torch.tensor(ids[:seq * mb * n], dtype=torch.long).view(n, mb, seq)
    return [t[j] for j in range(n)]


def prewarm(model_id, revision):
    """Read the checkpoint shards sequentially into the page cache: the loader's mmap faults read this host's pool at ~3 MB/s,
    a sequential pass at ~30-180 MB/s."""
    try:
        from huggingface_hub import snapshot_download
        d = snapshot_download(model_id, revision=revision, local_files_only=True)
    except Exception:
        d = model_id if os.path.isdir(model_id) else None
    if not d:
        return 0
    n = 0
    t = time.time()
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            with open(os.path.join(d, f), "rb", buffering=0) as fh:
                while True:
                    b = fh.read(64 << 20)
                    if not b:
                        break
                    n += len(b)
    print(f"[ladder] prewarmed {n / 2**30:.1f} GiB in {time.time() - t:.0f}s", flush=True)
    return n


def unpatch_rmsnorm(model):
    n = 0
    for m in model.modules():
        if getattr(m, "_e4b_rmsnorm_train", False):
            del m.forward                                # the instance attribute; the class forward is the composite
            del m._e4b_rmsnorm_train
            if hasattr(m, "_e4b_rmsnorm_variant"):
                del m._e4b_rmsnorm_variant
            n += 1
    return n


def main():
    a = parse()
    rungs = [r.strip() for r in a.rungs.split(",") if r.strip()]
    for r in rungs:
        if r.split("@")[0] not in RUNGS:
            raise SystemExit(f"unknown rung {r!r} (a rung is NAME or NAME@KEY=VAL;KEY=VAL -- a geometry/dispatch probe)")
    base_env = {k: os.environ.get(k) for r in RUNGS.values() for k in r}
    for kv in filter(None, a.env.split(",")):
        k, v = kv.split("=", 1)
        os.environ[k] = v
    import torch
    from transformers import AutoTokenizer
    from experts4bit_qlora import load_moe_4bit_streaming, verify_moe_4bit
    from experts4bit_qlora.lora import (add_attention_lora, quantize_attention_projections_4bit, detect_attention_projections,
                                        trainable_lora_param_ids, ExpertsLoRA)
    from experts4bit_qlora.engines.fast import enable_fast_train, disable_fast_train

    dev = "cuda"
    adt = torch.float32 if a.adapter_dtype == "fp32" else torch.bfloat16
    prewarm(a.model, a.revision)
    t0 = time.time()
    model, cfg = load_moe_4bit_streaming(a.model, dev, torch.bfloat16, a.r, a.alpha, offload=bool(a.offload), revision=a.revision)
    if not a.offload:
        model.to(dev)
    load_s = time.time() - t0
    verify_moe_4bit(model, strict=True)
    for m in model.modules():
        if isinstance(m, ExpertsLoRA):
            for n, p in m.named_parameters(recurse=False):
                if "lora" in n:
                    p.data = p.data.to(adt)
    n_attn_expected = detect_attention_projections(model, exact_linear=True).expected_count
    n_attn4 = quantize_attention_projections_4bit(model) if a.attn4 else 0
    n_attn_lora = add_attention_lora(model, a.r, a.alpha, adt) if a.attn_lora else 0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    ex_ids, at_ids = trainable_lora_param_ids(model)
    train = []
    for n, p in model.named_parameters():
        on = id(p) in ex_ids or id(p) in at_ids
        p.requires_grad_(on)
        if on:
            train.append(p)
    n_train = sum(p.numel() for p in train)
    init = [p.detach().clone() for p in train]          # every rung starts from the same adapters
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    per = a.warmup + a.steps + (1 if a.profile else 0) + 1
    batches = alpaca_batches(tok, a.seq, a.mb, per)
    out = {"model": a.model, "revision": a.revision, "model_type": getattr(cfg, "model_type", None), "offload": a.offload,
           "attn4": a.attn4, "n_attn4": n_attn4, "n_attn_expected": n_attn_expected, "n_attn_lora": n_attn_lora, "r": a.r,
           "adapter_dtype": a.adapter_dtype, "seq": a.seq, "mb": a.mb, "steps": a.steps, "warmup": a.warmup,
           "n_trainable": n_train, "n_expert_modules": sum(1 for m in model.modules() if isinstance(m, ExpertsLoRA)),
           "load_s": round(load_s, 1), "extra_env": a.env, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
           "rungs": []}
    import transformers
    import importlib.metadata as md
    out["versions"] = {"transformers": transformers.__version__, "e4b": md.version("experts4bit-qlora"),
                       "gnf4": md.version("grouped-nf4-gemm")}
    # Interleaved A..Z Z..A (the reference rung once, it is slow and never the question): on a shared host a single pass
    # drifted 20 % between its first and last rung (OLMoE, fused 1.777 -> 1.447 s/step), so every rung is read twice in
    # mirrored positions and reported as the mean of the two (see summarize()).
    fast = [r for r in rungs if r != "reference"]
    order = fast + [r + "#2" for r in reversed(fast)] + (["reference"] if "reference" in rungs else [])
    for rung in order:
        name, _, probe = rung.split("#")[0].partition("@")
        probe_env = dict(kv.split("=", 1) for kv in probe.split(";") if kv)
        for k in probe_env:
            base_env.setdefault(k, os.environ.get(k))
        for k, v in base_env.items():                    # back to the process's own environment, then this rung's switches
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for k, v in {**RUNGS[name], **probe_env}.items():
            os.environ[k] = v
        disable_fast_train(model)
        unpatch_rmsnorm(model)
        with torch.no_grad():
            for p, q in zip(train, init):
                p.copy_(q)
        c0 = counters()
        n_patched = enable_fast_train(model, verbose=True, dgrad=True) if name != "reference" else 0
        rec = run_rung(a, model, train, batches, torch)
        rec.update({"rung": rung, "rung_env": {**RUNGS[name], **probe_env}, "n_patched": n_patched,
                    "counters_before": c0, "counters": counters()})
        out["rungs"].append(rec)
        print(f"[ladder] {rung}: {rec['s_per_step']:.3f} s/step, {rec['tokens_per_s']} tok/s, peak {rec['peak_alloc_gb']:.2f} GB, "
              f"syncs {rec['syncs_one_step']}", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
    out["summary"] = summarize(out["rungs"])
    json.dump(out, open(a.out, "w"), indent=1)
    for k, v in out["summary"].items():
        print(f"[ladder] SUMMARY {k}: {v}", flush=True)


def summarize(recs):
    """Per rung: the mean s/step of its two mirrored readings, their spread, and the ratio to `fused` (the shipped path)."""
    by = {}
    for r in recs:
        by.setdefault(r["rung"].split("#")[0], []).append(r)
    out = {}
    for k, rs in by.items():
        v = [r["s_per_step"] for r in rs]
        out[k] = {"s_per_step_mean": round(sum(v) / len(v), 4), "readings": v,
                  "spread": round((max(v) - min(v)) / min(v), 3) if len(v) > 1 else None,
                  "peak_alloc_gb": max(r["peak_alloc_gb"] for r in rs),
                  "syncs_one_step": rs[0]["syncs_one_step"],
                  "device_busy_frac": [(r["profile"] or {}).get("device_busy_frac") for r in rs]}
        dv = [(r["profile"] or {}).get("device_busy_s") for r in rs]
        dv = [x for x in dv if x]
        # Device-busy seconds of one profiled step: the GPU work itself, which a contended host does not inflate. On a shared
        # seat (2-core CPU quota) the wall clock of a host-bound step swung 2.3x between mirrored readings while this stayed
        # within ~8 %, so it is the within-box measure to compare rungs by; wall is kept beside it.
        out[k]["device_s_mean"] = round(sum(dv) / len(dv), 4) if dv else None
    if "fused" in out:
        f = out["fused"]["s_per_step_mean"]
        fd = out["fused"].get("device_s_mean")
        for k in out:
            out[k]["ratio_to_fused"] = round(out[k]["s_per_step_mean"] / f, 3)
            if fd and out[k].get("device_s_mean"):
                out[k]["device_ratio_to_fused"] = round(out[k]["device_s_mean"] / fd, 3)
    return out


def run_rung(a, model, train, batches, torch):
    dev = "cuda"
    opt = torch.optim.AdamW(train, lr=1e-5)

    def step(b):
        ids = b.to(dev)
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss.detach()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    losses, times = [], []
    bi = 0
    for i in range(a.warmup + a.steps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        lo = step(batches[bi]); bi += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        losses.append(float(lo))
        if i >= a.warmup:
            times.append(dt)
        print(f"[ladder] step {i} {dt:.3f}s loss {float(lo):.4f}", flush=True)
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_res = torch.cuda.max_memory_reserved() / 2**30
    syncs = None
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode(1)
            step(batches[bi]); bi += 1
            torch.cuda.set_sync_debug_mode(0)
        syncs = sum(1 for x in w if "synchroniz" in str(x.message).lower())
    except Exception as e:  # noqa: BLE001
        torch.cuda.set_sync_debug_mode(0)
        syncs = repr(e)
    prof = profile_step(step, batches[bi], torch) if a.profile else None
    med = statistics.median(times)
    return {"s_per_step": round(med, 4), "s_per_step_all": [round(x, 4) for x in times],
            "tokens_per_s": round(a.seq * a.mb / med, 1), "peak_alloc_gb": round(peak_alloc, 3),
            "peak_reserved_gb": round(peak_res, 3),
            "host_maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
            "losses": [round(x, 4) for x in losses], "syncs_one_step": syncs, "profile": prof}


def profile_step(step, b, torch):
    from torch.profiler import profile, ProfilerActivity
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        t = time.perf_counter()
        step(b)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
    fam, kernels, n_launch, busy, cpu_ops = {}, {}, 0, [], 0
    for ev in p.events():
        dt = str(getattr(ev, "device_type", ""))
        if "CUDA" in dt:
            d = ev.time_range.elapsed_us()
            f = kernel_family(ev.name)
            fam[f] = fam.get(f, 0.0) + d
            kernels[ev.name] = kernels.get(ev.name, 0.0) + d
            n_launch += 1
            busy.append((ev.time_range.start, ev.time_range.end))
        elif "CPU" in dt:
            cpu_ops += 1
    busy.sort()
    merged, cur = 0.0, None
    for s0, e0 in busy:
        if cur is None or s0 > cur[1]:
            if cur:
                merged += cur[1] - cur[0]
            cur = [s0, e0]
        else:
            cur[1] = max(cur[1], e0)
    if cur:
        merged += cur[1] - cur[0]
    top = sorted(kernels.items(), key=lambda kv: -kv[1])[:25]
    return {"wall_s": round(wall, 4), "device_busy_s": round(merged / 1e6, 4), "device_busy_frac": round(merged / 1e6 / wall, 3),
            "device_launches": n_launch, "cpu_events": cpu_ops,
            "family_ms": {k: round(v / 1e3, 2) for k, v in sorted(fam.items(), key=lambda kv: -kv[1])},
            "top_kernels_ms": [[k[:120], round(v / 1e3, 2)] for k, v in top]}


if __name__ == "__main__":
    sys.exit(main())
