#!/usr/bin/env python3
"""smt_fuse_box.py -- the BOX side of the SMT fuse measurement (bench/smt-fuse/PREREG-smt-fuse.md).

One Qwen3-235B-A22B MoE layer's experts as the streaming loader builds them under expert offload:
128 experts, gate/up [1536, 4096] and down [4096, 1536] in bf16. Each rep runs the loader's own path:
  fuse   experts4bit_qlora.arch.moe_conventions.fuse_experts on the host (the op torch's intra-op pool runs);
  h2d    the fused stacks to the GPU;
  quant  Experts4bit.from_float(..., quant_type="nf4") on the GPU;
and is timed wall-clock from slab-in to nf4-out, with the pool set explicitly (torch.set_num_threads) per rep.
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


def one_rep(pool, gate, up, down):
    import torch
    import torch.nn.functional as F

    from experts4bit_qlora._vendor.experts import Experts4bit
    from experts4bit_qlora.arch.moe_conventions import fuse_experts

    torch.set_num_threads(pool)
    load = os.getloadavg()[0]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gu, dn = fuse_experts(gate, up, down)
    t1 = time.perf_counter()
    gu_d, dn_d = gu.to("cuda"), dn.to("cuda")
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    base = Experts4bit.from_float(gu_d, dn_d, has_gate=True, activation=F.silu, compute_dtype=torch.bfloat16,
                                  quant_type="nf4", blocksize=64)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    rec = {"pool": pool, "torch_pool": torch.get_num_threads(), "load_avg_1m": load,
           "fuse_s": t1 - t0, "h2d_s": t2 - t1, "quant_s": t3 - t2, "total_s": t3 - t0,
           "fused_sha256": hashlib.sha256((sha(gu) + sha(dn)).encode()).hexdigest(),
           "nf4_sha256": hashlib.sha256((sha(base.gate_up_proj) + sha(base.down_proj)).encode()).hexdigest()}
    del gu, dn, gu_d, dn_d, base
    torch.cuda.empty_cache()
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
        out["per_pool"][str(pool)] = {k: statistics.median(r[k] for r in rs) for k in ("fuse_s", "h2d_s", "quant_s", "total_s")}
        out["per_pool"][str(pool)]["total_s_per_layer"] = out["per_pool"][str(pool)]["total_s"] * scale
        out["per_pool"][str(pool)]["fuse_s_per_layer"] = out["per_pool"][str(pool)]["fuse_s"] * scale
        out["per_pool"][str(pool)]["n"] = len(rs)
    for i in range(0, len(reps) - 1, 2):                 # (16, 32), (32, 16), ... -> each pair's 32/16 ratio
        a, b = reps[i], reps[i + 1]
        if {a["pool"], b["pool"]} == {16, 32}:
            r16, r32 = (a, b) if a["pool"] == 16 else (b, a)
            out["pairs"].append({"total_32_over_16": r32["total_s"] / r16["total_s"],
                                 "fuse_32_over_16": r32["fuse_s"] / r16["fuse_s"]})
    for k in ("total_32_over_16", "fuse_32_over_16"):
        v = [p[k] for p in out["pairs"]]
        out[k] = {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v),
                  "range_includes_1": min(v) <= 1.0 <= max(v)} if v else None
    out["bytes_identical"] = (len({r["fused_sha256"] for r in reps}) == 1 and len({r["nf4_sha256"] for r in reps}) == 1)
    return out


def run(a):
    import torch

    facts = host_facts()
    gate, up, down = slab(a.experts)
    # warm each pool once (allocator, CUDA context, bnb kernels): the cap is sized from each pool's own warm wall, since
    # a slower pool is exactly what this measures
    warm = {pool: one_rep(pool, gate, up, down)["total_s"] for pool in sorted(set(POOLS))}
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
            rec = one_rep(pool, gate, up, down)
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
        return {"pool": pool, "total_s": total, "fuse_s": fuse, "h2d_s": 0.1, "quant_s": 0.2,
                "fused_sha256": f, "nf4_sha256": q}
    reps = [rep(16, 10.0, 4.0), rep(32, 12.0, 6.0), rep(32, 11.0, 5.0), rep(16, 10.0, 4.0)]
    s = summarize({"reps": reps, "experts": 128})
    assert s["per_pool"]["16"]["total_s"] == 10.0 and s["per_pool"]["32"]["total_s"] == 11.5
    assert len(s["pairs"]) == 2 and abs(s["pairs"][0]["total_32_over_16"] - 1.2) < 1e-12
    assert abs(s["pairs"][1]["total_32_over_16"] - 1.1) < 1e-12 and s["total_32_over_16"]["range_includes_1"] is False
    n += 1
    s = summarize({"reps": reps, "experts": 32})
    assert s["per_pool"]["16"]["total_s_per_layer"] == 40.0                   # scaled to 128 experts
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
