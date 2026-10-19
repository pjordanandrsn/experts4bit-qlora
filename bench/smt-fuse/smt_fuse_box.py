#!/usr/bin/env python3
"""smt_fuse_box.py -- the BOX side of the SMT fuse measurement (bench/smt-fuse/PREREG-smt-fuse.md).

One Qwen3-235B-A22B MoE layer's experts as the streaming loader builds them under expert offload:
128 experts, gate/up [1536, 4096] and down [4096, 1536] in bf16. Each rep runs the loader's own path, every step of
q235-pro6000-7's one timed block ("fusing + quantizing ... offloaded to pinned CPU RAM"):
  read     each expert's three tensors from a safetensors file on the box's disk, through
           safe_open(..., device="cpu").get_tensor, as the loader's host staging reads them (page cache warm after the
           first rep: the cold read of a real checkpoint is a different term and is not measured here);
  fuse     experts4bit_qlora.arch.moe_conventions.fuse_experts on the host (the op torch's intra-op pool runs);
  h2d      the fused stacks to the GPU;
  quant    Experts4bit.from_float(..., quant_type="nf4") on the GPU;
  offload  ExpertsLoRA around it and engines.offload.enable_expert_offload(pin=True): the pinned host home and the
           device-to-host copy (the cached pinned blocks are released after each rep, so every rep pays a fresh pin);
and is timed wall-clock from the first read to the offloaded layer, with the pool set explicitly per rep.
The reps interleave POOLS = (16, 32, 32, 16) for CYCLES cycles, so each (16, 32) pair runs back to back.

Recorded per rep: the pool torch reports, the host load average at its start, the segment walls, and the
sha256 of the fused bytes and of the nf4 bytes, which must be identical across every rep (the pool is speed-only).
Recorded once: the CPU model, physical and logical counts, the cgroup quota, the affinity, torch's default pool.

    python smt_fuse_box.py --out result.json [--experts N]     (N < 128 only for the $0 dry run)
    python smt_fuse_box.py --summarize result.json
    python smt_fuse_box.py --self-test
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

POOLS = (16, 32, 32, 16)
CYCLES = 4
E_FULL, INTER, HIDDEN = 128, 1536, 4096
SEED = 0
KEY = "model.layers.0.mlp.experts.{e}.{p}.weight"
SEGMENTS = ("read_s", "fuse_s", "h2d_s", "quant_s", "offload_s", "total_s", "block_s", "residual_s")
CAP_FLOOR_S = 600                 # the cap never drops below this: a short warm rep on a busy host is noisy


def lscpu():
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    keep = ("Model name", "CPU(s)", "Thread(s) per core", "Core(s) per socket", "Socket(s)", "On-line CPU(s) list")
    return {k.strip(): v.strip() for k, v in (line.split(":", 1) for line in out.splitlines() if ":" in line)
            if k.strip() in keep}


def host_facts():
    import torch

    from experts4bit_qlora.util import container_cpu_quota

    cpu = lscpu()
    tpc, cps, sock = (int(cpu.get(k) or 0) for k in ("Thread(s) per core", "Core(s) per socket", "Socket(s)"))
    return {"cpu_model": cpu.get("Model name"), "logical": int(cpu.get("CPU(s)") or 0), "physical": cps * sock,
            "threads_per_core": tpc, "affinity": len(os.sched_getaffinity(0)), "quota": container_cpu_quota()[0],
            "torch_default_pool": torch.get_num_threads(), "torch": torch.__version__}


def sha(t):
    return hashlib.sha256(t.detach().contiguous().view(-1).view(__import__("torch").uint8).cpu().numpy().tobytes()).hexdigest()


def slab(n_experts):
    import torch

    g = torch.Generator().manual_seed(SEED)
    mk = (lambda a, b: torch.randn(a, b, generator=g).to(torch.bfloat16))  # noqa: E731
    return ([mk(INTER, HIDDEN) for _ in range(n_experts)], [mk(INTER, HIDDEN) for _ in range(n_experts)],
            [mk(HIDDEN, INTER) for _ in range(n_experts)])


def write_slab(path, gate, up, down):
    """The slab once, as a checkpoint lays it out: one tensor per expert and projection."""
    from safetensors.torch import save_file

    t = {}
    for e in range(len(gate)):
        t[KEY.format(e=e, p="gate_proj")], t[KEY.format(e=e, p="up_proj")] = gate[e], up[e]
        t[KEY.format(e=e, p="down_proj")] = down[e]
    save_file({k: v.contiguous() for k, v in t.items()}, path)


def one_rep(pool, slab_path, n_experts):
    import gc

    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    from experts4bit_qlora._vendor.experts import Experts4bit
    from experts4bit_qlora.arch.moe_conventions import fuse_experts
    from experts4bit_qlora.engines.offload import enable_expert_offload
    from experts4bit_qlora.lora import ExpertsLoRA

    torch.set_num_threads(pool)
    load = os.getloadavg()[0]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with safe_open(slab_path, framework="pt", device="cpu") as f:
        gate = [f.get_tensor(KEY.format(e=e, p="gate_proj")) for e in range(n_experts)]
        up = [f.get_tensor(KEY.format(e=e, p="up_proj")) for e in range(n_experts)]
        down = [f.get_tensor(KEY.format(e=e, p="down_proj")) for e in range(n_experts)]
    t1 = time.perf_counter()
    gu, dn = fuse_experts(gate, up, down)
    t2 = time.perf_counter()
    gu_d, dn_d = gu.to("cuda"), dn.to("cuda")
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    base = Experts4bit.from_float(gu_d, dn_d, has_gate=True, activation=F.silu, compute_dtype=torch.bfloat16,
                                  quant_type="nf4", blocksize=64)
    torch.cuda.synchronize()
    t4 = time.perf_counter()
    nf4 = hashlib.sha256((sha(base.gate_up_proj) + sha(base.down_proj)).encode()).hexdigest()
    t4b = time.perf_counter()                            # the hash is not the loader's work: excluded from the walls
    experts = ExpertsLoRA(base, r=16, alpha=16, dtype=torch.bfloat16).to("cuda")
    handle = enable_expert_offload(experts, "cuda", pin=True)
    torch.cuda.synchronize()
    t5 = time.perf_counter()
    rec = {"pool": pool, "torch_pool": torch.get_num_threads(), "load_avg_1m": load,
           "read_s": t1 - t0, "fuse_s": t2 - t1, "h2d_s": t3 - t2, "quant_s": t4 - t3, "offload_s": t5 - t4b,
           "total_s": (t4 - t0) + (t5 - t4b), "nf4_sha256": nf4}
    t5b = time.perf_counter()
    rec["fused_sha256"] = hashlib.sha256((sha(gu) + sha(dn)).encode()).hexdigest()
    t5c = time.perf_counter()                            # this hash is excluded from the block too
    del gate, up, down, gu, dn, gu_d, dn_d, base, experts, handle
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    try:                                                 # every rep pays a fresh pin, as each real layer does
        from experts4bit_qlora.engines.host_heap import release_cached_pinned_memory
        release_cached_pinned_memory()
    except Exception:                                    # noqa: BLE001 -- recorded below when it is unavailable
        rec["pinned_release"] = "unavailable"
    t6 = time.perf_counter()
    # the WHOLE block, first read to the layer's memory freed (the real loader frees each layer's GPU copies and
    # staging too), less the two hashes; the residual is what no segment timed (the free, and any gap between them)
    rec["block_s"] = (t6 - t0) - (t4b - t4) - (t5c - t5b)
    rec["residual_s"] = rec["block_s"] - rec["total_s"]
    return rec


def phase_cap(warm_by_pool, plan, time_left_s):
    """The probe's on-box cap (lane_record.phase_caps.probe): 1.5 x the planned reps' walls, each taken from its own
    pool's warm rep on this box, at least CAP_FLOOR_S, never beyond the time the box has left."""
    return min(time_left_s, max(CAP_FLOOR_S, 1.5 * sum(warm_by_pool[pool] for pool in plan)))


def summarize(res):
    """Per-pool medians (seconds for this slab, and scaled to a 128-expert layer), the (16, 32) pairs' 32/16 ratios
    with their range, and whether every rep's fused and nf4 bytes agree."""
    reps, scale = res["reps"], E_FULL / res["experts"]
    out = {"per_pool": {}, "pairs": [], "bytes_identical": None}
    for pool in sorted({r["pool"] for r in reps}):
        rs = [r for r in reps if r["pool"] == pool]
        out["per_pool"][str(pool)] = {k: statistics.median(r[k] for r in rs) for k in SEGMENTS if k in rs[0]}
        for k in list(out["per_pool"][str(pool)]):
            out["per_pool"][str(pool)][k.replace("_s", "_s_per_layer")] = out["per_pool"][str(pool)][k] * scale
        out["per_pool"][str(pool)]["n"] = len(rs)
    for i in range(0, len(reps) - 1, 2):                 # (16, 32), (32, 16), ... -> each pair's 32/16 ratio
        a, b = reps[i], reps[i + 1]
        if {a["pool"], b["pool"]} == {16, 32}:
            r16, r32 = (a, b) if a["pool"] == 16 else (b, a)
            out["pairs"].append({k.replace("_s", "_32_over_16"): r32[k] / r16[k] for k in SEGMENTS if k in r16})
    for k in [s.replace("_s", "_32_over_16") for s in SEGMENTS]:
        v = [p[k] for p in out["pairs"] if k in p]
        out[k] = {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v),
                  "range_includes_1": min(v) <= 1.0 <= max(v)} if v else None
    out["bytes_identical"] = (len({r["fused_sha256"] for r in reps}) == 1 and len({r["nf4_sha256"] for r in reps}) == 1)
    return out


def run(a):
    import torch

    facts = host_facts()
    cache = os.path.join(os.path.dirname(os.path.abspath(a.out)), ".cache")     # .cache stays on the box (the fetch skips it)
    os.makedirs(cache, exist_ok=True)
    slab_path = os.path.join(cache, "slab.safetensors")
    write_slab(slab_path, *slab(a.experts))
    # warm each pool once (allocator, CUDA context, bnb kernels): the cap is sized from each pool's own warm wall, since
    # a slower pool is exactly what this measures
    warm = {pool: one_rep(pool, slab_path, a.experts)["block_s"] for pool in sorted(set(POOLS))}
    plan = [pool for _ in range(CYCLES) for pool in POOLS]
    cap = phase_cap(warm, plan, float(os.environ.get("SMT_FUSE_TIME_LEFT_S", "inf")))
    print("SMT_FUSE_CAP " + json.dumps({"warm_total_s": warm, "reps": len(plan), "cap_s": cap}), flush=True)
    t_start = time.perf_counter()
    reps = []
    for c in range(CYCLES):
        for pool in POOLS:
            if time.perf_counter() - t_start > cap:
                print(f"SMT_FUSE_CAP_HIT after {len(reps)} reps", flush=True)
                return 4
            rec = one_rep(pool, slab_path, a.experts)
            rec["cycle"] = c
            reps.append(rec)
            print("SMT_FUSE_REP " + json.dumps(rec), flush=True)
    res = {"facts": facts, "experts": a.experts, "shape": [E_FULL, INTER, HIDDEN], "pools": POOLS, "cycles": CYCLES,
           "gpu": torch.cuda.get_device_name(0), "e4b_sha": os.environ.get("E4B_SHA"), "reps": reps}
    res["summary"] = summarize(res)
    json.dump(res, open(a.out, "w"), indent=1)
    print("SMT_FUSE_SUMMARY " + json.dumps({"facts": facts, **res["summary"]}), flush=True)
    return 0 if res["summary"]["bytes_identical"] else 3


def self_test():
    n = 0
    def rep(pool, total, fuse, f="a", q="b"):
        return {"pool": pool, "total_s": total, "fuse_s": fuse, "read_s": 0.5, "h2d_s": 0.1, "quant_s": 0.2,
                "offload_s": 0.3, "block_s": total + 0.4, "residual_s": 0.4, "fused_sha256": f, "nf4_sha256": q}
    reps = [rep(16, 10.0, 4.0), rep(32, 12.0, 6.0), rep(32, 11.0, 5.0), rep(16, 10.0, 4.0)]
    s = summarize({"reps": reps, "experts": 128})
    assert s["per_pool"]["16"]["total_s"] == 10.0 and s["per_pool"]["32"]["total_s"] == 11.5
    assert len(s["pairs"]) == 2 and abs(s["pairs"][0]["total_32_over_16"] - 1.2) < 1e-12
    assert abs(s["pairs"][1]["total_32_over_16"] - 1.1) < 1e-12 and s["total_32_over_16"]["range_includes_1"] is False
    n += 1
    s = summarize({"reps": reps, "experts": 32})
    assert s["per_pool"]["16"]["total_s_per_layer"] == 40.0                   # scaled to 128 experts
    assert s["per_pool"]["16"]["read_s"] == 0.5 and s["per_pool"]["32"]["offload_s"] == 0.3     # every segment reported
    assert s["read_32_over_16"]["median"] == 1.0 and s["offload_32_over_16"]["n"] == 2
    assert s["per_pool"]["16"]["residual_s"] == 0.4 and s["per_pool"]["32"]["block_s"] == 11.9   # the block and residual
    n += 1
    reps2 = [rep(16, 10.0, 4.0), rep(32, 9.0, 3.0), rep(32, 11.0, 5.0), rep(16, 10.0, 4.0)]
    assert summarize({"reps": reps2, "experts": 128})["total_32_over_16"]["range_includes_1"] is True
    n += 1
    assert summarize({"reps": reps, "experts": 128})["bytes_identical"] is True
    assert summarize({"reps": reps[:3] + [rep(16, 10.0, 4.0, q="c")], "experts": 128})["bytes_identical"] is False
    n += 1
    assert POOLS == (16, 32, 32, 16) and CYCLES >= 3 and (E_FULL, INTER, HIDDEN) == (128, 1536, 4096)
    n += 1
    plan = [pool for _ in range(CYCLES) for pool in POOLS]
    assert phase_cap({16: 50.0, 32: 100.0}, plan, float("inf")) == 1.5 * CYCLES * (2 * 50.0 + 2 * 100.0)
    assert phase_cap({16: 0.2, 32: 0.1}, plan, float("inf")) == CAP_FLOOR_S          # the floor on a tiny slab
    assert phase_cap({16: 10.0, 32: 30.0}, plan, 100.0) == 100.0                       # never beyond the time left
    n += 1
    print(f"smt_fuse_box self-test OK ({n}/6 cases)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--out")
    p.add_argument("--experts", type=int, default=E_FULL)
    p.add_argument("--summarize")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.summarize:
        print(json.dumps(summarize(json.load(open(a.summarize))), indent=1))
        return 0
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
