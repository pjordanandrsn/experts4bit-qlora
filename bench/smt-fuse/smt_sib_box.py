#!/usr/bin/env python3
"""smt_sib_box.py -- the BOX side of the SMT fuse measurement's Amendment 1 (bench/smt-fuse/PREREG-smt-fuse.md).

smt-fuse-2 read the pool 16 vs 31 under a 30.72-CPU quota on a 128-core host with no pinning, so its 31 threads could
each run on its own physical core: it measured the pool under a quota, not the second hardware thread. This probe asks
the second-thread question directly. Each rep runs in a FRESH process whose CPU set is fixed with sched_setaffinity
BEFORE torch is imported, so every intra-op worker inherits it (a running process's OpenMP workers keep the mask they
were created with). The three CPU sets come from the host's own sibling topology:
  PHYS  K physical cores, one hardware thread each                pool K
  SIB   the SAME K cores, both hardware threads                   pool 2K
  WIDE  2K physical cores, one hardware thread each               pool 2K
SIB/PHYS on the FUSE is what the second hardware thread adds on the same cores and caches (the reading; the block is
reported, since offload dominates it). SIB/WIDE is what it is
worth against a real core; WIDE also spans more cache, so that ratio is reported, never read.
K = min(16, floor(quota / 2.5)): the 2K-thread arms run at no more than 80 % of the quota, so the cgroup does not bind
before the cores do; the cgroup's throttling is recorded per rep, as smt-fuse-2 recorded it.

Each rep's process runs smt_fuse_box.one_rep -- the registered block, unchanged -- once untimed (allocator, CUDA context,
bnb kernels), then REPEATS = 15 times timed; the rep is their per-segment median (the maintainer's pass: one timed block
per process could not resolve the effect at smt-fuse-2's spread, and 5 resolved only a fuse SIB/PHYS of about 0.65). Around the timed rep it reads /proc/stat for the arm's CPUs and for the COMPANION threads
(the siblings of the arm's cores that the arm does not use): a co-tenant busy on a PHYS arm's companions turns that arm
into a shared-core arm, so the rep is flagged.

    python smt_sib_box.py --plan --quota Q            (stdlib only: the pre-install refusal, exit 5 when refused)
    python smt_sib_box.py --out result.json [--experts N]
    python smt_sib_box.py --summarize result.json
    python smt_sib_box.py --self-test
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("PHYS", "SIB", "WIDE")
ORDER = ("PHYS", "SIB", "WIDE", "WIDE", "SIB", "PHYS")   # two triples per cycle, each arm once per triple
CYCLES = 4                                              # 24 reps, 8 triples
K_MAX, MIN_K = 16, 8                                    # under 8 cores the arms are too small to read (refused, $0)
QUOTA_PER_K = 2.5                                       # 2K threads at <= 80 % of the quota
SEGMENTS = ("read_s", "fuse_s", "h2d_s", "quant_s", "offload_s", "total_s", "block_s", "residual_s")
READ = ("fuse_s",)                                      # the op the pool drives: the reading (the maintainer's pass)
REPORTED = ("block_s",)                                 # what the loader pays: offload dominates it, so reported only
REPEATS = 15                                            # timed one_reps per process; the rep is their median
# The time fit (the maintainer's pass: a guard is a fit and is tested against what it guards). Every term is smt-fuse-2's
# measured wall ROUNDED UP: guard armed -> probe start 209 s; its probe phase 217 s for the slab + 18 one_reps, charged
# whole to the reps (one_rep hashes 4.8 GB outside its timed block, so a block's wall is ~3x its block_s); the per-process
# start (interpreter, torch, CUDA context) is not measured yet and is taken high. The reserve is smt_fuse_run.sh's 600 s.
FIT = {"setup_s": 240, "slab_s": 60, "block_wall_s": 12.5, "process_start_s": 30, "reserve_s": 600}
GUARD_S = 9000                                          # the launch's guard: 2.5 h


def time_fit(repeats, fit=FIT):
    """Seconds the run needs at `repeats`: setup, slab, the 3 sizing + 24 timed processes, each 1 untimed + `repeats`
    timed blocks, and the reserve."""
    per_process = fit["process_start_s"] + (1 + repeats) * fit["block_wall_s"]
    return fit["setup_s"] + fit["slab_s"] + (len(ARMS) + CYCLES * len(ORDER)) * per_process + fit["reserve_s"]
THROTTLE_FLAG = 0.05                                    # smt-fuse-2's flag, unchanged
COMPANION_FLAG = 0.25                                   # a PHYS/WIDE rep whose companions are > 25 % busy is flagged
MIN_TRIPLES = 6                                         # fewer readable triples than this: VOID_BY_HOST


def parse_cpulist(s):
    out = []
    for part in s.strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def topology(cpus, read=None):
    """The host's two-thread cores whose BOTH threads are in `cpus`, as (first, second) pairs, sorted."""
    def sysfs(n):
        return open(f"/sys/devices/system/cpu/cpu{n}/topology/thread_siblings_list").read()
    read = read or sysfs
    cpus, cores = set(cpus), set()
    for n in cpus:
        try:
            sib = tuple(sorted(parse_cpulist(read(n))))
        except (OSError, ValueError):
            continue
        if len(sib) == 2 and set(sib) <= cpus:
            cores.add(sib)
    return sorted(cores)


def choose_arms(cores, quota):
    """K and the three CPU sets, or (None, why) when the host cannot carry the arms."""
    k = min(K_MAX, math.floor(quota / QUOTA_PER_K)) if quota else K_MAX
    if k < MIN_K:
        return None, f"quota {quota} gives K={k} < {MIN_K}"
    if len(cores) < 2 * k:
        return None, f"{len(cores)} two-thread cores in the affinity < 2K = {2 * k}"
    arms = {"PHYS": [a for a, _b in cores[:k]], "SIB": sorted(x for c in cores[:k] for x in c),
            "WIDE": [a for a, _b in cores[:2 * k]]}
    return {"k": k, "arms": arms, "pool": {"PHYS": k, "SIB": 2 * k, "WIDE": 2 * k}}, None


def companions(arm_cpus, cores):
    """The siblings of the arm's cores that the arm does not use."""
    s = set(arm_cpus)
    return sorted(x for c in cores if s & set(c) for x in c if x not in s)


def proc_stat():
    """Per-CPU (busy, total) jiffies from /proc/stat, or {} where it cannot be read."""
    out = {}
    try:
        for line in open("/proc/stat"):
            if line.startswith("cpu") and line[3:4].isdigit():
                f = line.split()
                v = [int(x) for x in f[1:]]
                idle = v[3] + (v[4] if len(v) > 4 else 0)
                out[int(f[0][3:])] = (sum(v[:8]) - idle, sum(v[:8]))
    except OSError:
        pass
    return out


def busy_share(s0, s1, cpus):
    db = sum(s1[c][0] - s0[c][0] for c in cpus if c in s0 and c in s1)
    dt = sum(s1[c][1] - s0[c][1] for c in cpus if c in s0 and c in s1)
    return db / dt if dt > 0 else None


def median_of(recs):
    """One rep from REPEATS timed blocks in one process: each segment's median; the worst throttling; the bytes only if
    every block agrees (else MISMATCH, which fails bytes_identical); the raw blocks kept beside it."""
    rec = {k: statistics.median(r[k] for r in recs) for k in SEGMENTS if k in recs[0]}
    rec["throttled_period_share"] = max((r.get("throttled_period_share") or 0.0) for r in recs)
    rec["throttled_cpu_s"] = sum((r.get("throttled_cpu_s") or 0.0) for r in recs)
    for h in ("fused_sha256", "nf4_sha256"):
        vals = {r[h] for r in recs}
        rec[h] = vals.pop() if len(vals) == 1 else "MISMATCH"
    for k in ("pool", "torch_pool", "load_avg_1m"):
        if k in recs[0]:
            rec[k] = recs[0][k]
    rec["repeats"] = [{k: r[k] for k in (*SEGMENTS, "throttled_period_share") if k in r} for r in recs]
    return rec


def child(a):
    """One rep in its own process: pin first, then import torch; the registered block once untimed, then REPEATS times."""
    cpus = [int(x) for x in a.cpus.split(",")]
    os.sched_setaffinity(0, cpus)
    sys.path.insert(0, HERE)
    import smt_fuse_box as B

    B.one_rep(a.pool, a.slab, a.experts)                                   # untimed: allocator, context, kernels
    comp = [int(x) for x in a.companions.split(",") if x]
    s0 = proc_stat()
    rec = median_of([B.one_rep(a.pool, a.slab, a.experts) for _ in range(REPEATS)])
    s1 = proc_stat()
    rec.update(arm=a.rep, cpus=len(cpus), affinity_seen=len(os.sched_getaffinity(0)),
               arm_busy=busy_share(s0, s1, cpus), companion_busy=busy_share(s0, s1, comp) if comp else None)
    print("SMT_SIB_REP " + json.dumps(rec), flush=True)
    return 0


def flagged(r):
    return bool((r.get("throttled_period_share") or 0) > THROTTLE_FLAG
                or (r.get("arm") != "SIB" and (r.get("companion_busy") or 0) > COMPANION_FLAG))


def summarize(res):
    """Per-arm medians (scaled to a 128-expert layer), each triple's SIB/PHYS and SIB/WIDE ratios, their median and
    range over the unflagged triples, the reading on fuse and block, and whether every rep's bytes agree."""
    reps, scale = res["reps"], 128 / res["experts"]
    out = {"per_arm": {}, "triples": [], "k": res.get("k")}
    for arm in ARMS:
        rs = [r for r in reps if r["arm"] == arm]
        if rs:
            m = {k: statistics.median(r[k] for r in rs) for k in SEGMENTS if k in rs[0]}
            m.update({k.replace("_s", "_s_per_layer"): v * scale for k, v in list(m.items())})
            out["per_arm"][arm] = dict(m, n=len(rs))
    for i in range(0, len(reps) - 2, 3):
        t = {r["arm"]: r for r in reps[i:i + 3]}
        if set(t) != set(ARMS):
            continue
        row = {"flagged": any(flagged(r) for r in t.values())}
        for k in SEGMENTS:
            if k in t["SIB"]:
                for other in ("PHYS", "WIDE"):
                    if t[other].get(k):
                        row[f"sib_over_{other.lower()}_{k[:-2]}"] = t["SIB"][k] / t[other][k]
        out["triples"].append(row)
    out["flagged_triples"] = sum(t["flagged"] for t in out["triples"])
    readable = [t for t in out["triples"] if not t["flagged"]]
    for key in sorted({k for t in out["triples"] for k in t if k.startswith("sib_over_")}):
        v = [t[key] for t in readable if key in t]
        out[key] = ({"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v),
                     "reading": "below_1" if max(v) < 1 else "above_1" if min(v) > 1 else "includes_1"} if v else None)
    out["void_by_host"] = len(readable) < MIN_TRIPLES
    out["reading"] = None if out["void_by_host"] else {k: out[f"sib_over_phys_{k[:-2]}"]["reading"] for k in READ}
    out["reported"] = None if out["void_by_host"] else {k: out[f"sib_over_phys_{k[:-2]}"]["reading"] for k in REPORTED}
    fused, nf4 = {r["fused_sha256"] for r in reps}, {r["nf4_sha256"] for r in reps}
    out["bytes_identical"] = len(fused) == 1 and len(nf4) == 1 and "MISMATCH" not in fused | nf4   # within-process too
    return out


def run(a):
    sys.path.insert(0, HERE)
    import torch

    import smt_fuse_box as B

    facts = B.host_facts()
    dry = os.environ.get("SMT_FUSE_DRY") == "1"
    affinity = sorted(os.sched_getaffinity(0))
    cores = topology(affinity)
    if dry:                                              # the $0 dry run checks the wiring on any box, never a rate
        c = affinity + affinity                          # (a box without SMT repeats its CPUs: correctness only)
        plan = {"k": 1, "arms": {"PHYS": [c[0]], "SIB": sorted({c[0], c[1]}), "WIDE": sorted({c[0], c[2]})},
                "pool": {"PHYS": 1, "SIB": 2, "WIDE": 2}}
    else:
        plan, why = choose_arms(cores, facts["quota"])
        if plan is None:
            print("SMT_SIB_REFUSED " + json.dumps({"why": why}), flush=True)
            return 5
    cache = os.path.join(os.path.dirname(os.path.abspath(a.out)), ".cache")
    os.makedirs(cache, exist_ok=True)
    slab_path = os.path.join(cache, "slab.safetensors")
    B.write_slab(slab_path, *B.slab(a.experts))
    print("SMT_SIB_PLAN " + json.dumps({"k": plan["k"], "arms": plan["arms"], "two_thread_cores": len(cores)}),
          flush=True)
    time_left = float(os.environ.get("SMT_FUSE_TIME_LEFT_S", "inf"))
    t_start = time.perf_counter()

    def rep(arm, limit):
        cmd = [sys.executable, os.path.abspath(__file__), "--rep", arm, "--cpus", ",".join(map(str, plan["arms"][arm])),
               "--companions", ",".join(map(str, companions(plan["arms"][arm], cores))),
               "--pool", str(plan["pool"][arm]), "--slab", slab_path, "--experts", str(a.experts)]
        t0 = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=max(1.0, limit))
        except subprocess.TimeoutExpired:
            return None, time.perf_counter() - t0, "timeout"
        line = next((x for x in p.stdout.splitlines() if x.startswith("SMT_SIB_REP ")), None)
        if p.returncode != 0 or line is None:
            return None, time.perf_counter() - t0, (p.stdout + p.stderr)[-800:]
        return json.loads(line[len("SMT_SIB_REP "):]), time.perf_counter() - t0, None

    warm = {}
    for arm in ARMS:                                     # the cap is sized from each arm's own process wall on this box
        rec, wall, err = rep(arm, time_left - (time.perf_counter() - t_start))
        if rec is None:
            print(f"SMT_SIB_CHILD_FAILED warm {arm}: {err}", flush=True)
            return 6
        warm[arm] = wall
    order = [arm for _ in range(CYCLES) for arm in ORDER]
    cap = B.phase_cap(warm, order, time_left - (time.perf_counter() - t_start))
    print("SMT_SIB_CAP " + json.dumps({"warm_process_s": warm, "reps": len(order), "cap_s": cap}), flush=True)
    t_cap, reps = time.perf_counter(), []
    for i, arm in enumerate(order):
        left = cap - (time.perf_counter() - t_cap)
        if left <= 0:
            print(f"SMT_SIB_CAP_HIT after {len(reps)} reps", flush=True)
            return 4
        rec, _wall, err = rep(arm, left)
        if rec is None:
            print(f"SMT_SIB_CHILD_FAILED rep {i} {arm}: {err}", flush=True)
            return 4 if err == "timeout" else 6
        rec["cycle"] = i // len(ORDER)
        reps.append(rec)
        print("SMT_SIB_REP " + json.dumps(rec), flush=True)
    res = {"facts": facts, "experts": a.experts, "k": plan["k"], "arms": plan["arms"], "pool": plan["pool"],
           "two_thread_cores": len(cores), "order": list(ORDER), "cycles": CYCLES, "gpu": torch.cuda.get_device_name(0),
           "e4b_sha": os.environ.get("E4B_SHA"), "reps": reps}
    res["summary"] = summarize(res)
    json.dump(res, open(a.out, "w"), indent=1)
    print("SMT_SIB_SUMMARY " + json.dumps({"facts": facts, **res["summary"]}), flush=True)
    return 0 if res["summary"]["bytes_identical"] else 3


def self_test():
    n = 0
    assert parse_cpulist("0,128") == [0, 128] and parse_cpulist("4-5") == [4, 5] and parse_cpulist("7") == [7]
    sysfs = {c: f"{c % 64},{c % 64 + 64}" for c in range(128)}               # 64 cores, siblings n and n+64
    cores = topology(range(128), read=lambda c: sysfs[c])
    assert len(cores) == 64 and cores[0] == (0, 64) and cores[-1] == (63, 127)
    assert len(topology(range(70), read=lambda c: sysfs[c])) == 6        # only cores with BOTH threads in the affinity
    assert topology(range(8), read=lambda c: str(c)) == []                # no SMT: no two-thread cores
    n += 1
    plan, why = choose_arms(cores, 30.72)                                 # smt-fuse-2's host quota
    assert why is None and plan["k"] == 12 and plan["pool"] == {"PHYS": 12, "SIB": 24, "WIDE": 24}
    assert plan["arms"]["PHYS"] == list(range(12)) and plan["arms"]["SIB"] == list(range(12)) + list(range(64, 76))
    assert plan["arms"]["WIDE"] == list(range(24)) and 24 <= 0.8 * 30.72 + 1e-9
    assert choose_arms(cores, None)[0]["k"] == 16 and choose_arms(cores, 64.0)[0]["k"] == 16
    assert choose_arms(cores, 16.0)[0] is None and "K=6" in choose_arms(cores, 16.0)[1]
    assert choose_arms(cores[:20], 30.72)[0] is None                      # 20 cores < 2K = 24
    n += 1
    assert companions([0, 1, 2], cores) == [64, 65, 66] and companions(list(range(12)) + list(range(64, 76)), cores) == []
    s0, s1 = {0: (10, 100), 1: (0, 100)}, {0: (60, 200), 1: (0, 200)}
    assert busy_share(s0, s1, [0]) == 0.5 and busy_share(s0, s1, [0, 1]) == 0.25 and busy_share(s0, s1, [9]) is None
    n += 1

    def r(arm, fuse, block, comp=0.0, thr=0.0, f="a", q="b"):
        return {"arm": arm, "fuse_s": fuse, "block_s": block, "total_s": block - 0.5, "read_s": 0.1, "h2d_s": 0.2,
                "quant_s": 0.1, "offload_s": 1.0, "residual_s": 0.5, "companion_busy": comp,
                "throttled_period_share": thr, "fused_sha256": f, "nf4_sha256": q}
    trip = [r("PHYS", 1.0, 4.0), r("SIB", 0.8, 3.8), r("WIDE", 0.6, 3.6), r("WIDE", 0.6, 3.6), r("SIB", 0.9, 3.9),
            r("PHYS", 1.0, 4.0)]
    s = summarize({"reps": trip * 4, "experts": 128, "k": 12})
    assert len(s["triples"]) == 8 and s["flagged_triples"] == 0 and s["void_by_host"] is False
    assert abs(s["sib_over_phys_fuse"]["median"] - 0.85) < 1e-12 and s["sib_over_phys_fuse"]["reading"] == "below_1"
    assert s["sib_over_wide_fuse"]["reading"] == "above_1" and s["reading"] == {"fuse_s": "below_1"}
    assert s["reported"] == {"block_s": "below_1"}
    assert s["per_arm"]["SIB"]["n"] == 8 and s["per_arm"]["PHYS"]["block_s_per_layer"] == 4.0
    assert summarize({"reps": trip * 4, "experts": 32, "k": 12})["per_arm"]["PHYS"]["block_s_per_layer"] == 16.0
    mixed = [r("PHYS", 1.0, 4.0), r("SIB", 1.1, 4.1), r("WIDE", 0.6, 3.6)] + trip[3:]
    assert summarize({"reps": mixed * 4, "experts": 128})["sib_over_phys_fuse"]["reading"] == "includes_1"
    n += 1
    noisy = [r("PHYS", 1.0, 4.0, comp=0.4)] + trip[1:]                     # a co-tenant on the PHYS arm's companions
    s = summarize({"reps": noisy * 4, "experts": 128})
    assert s["flagged_triples"] == 4 and s["sib_over_phys_fuse"]["n"] == 4 and s["void_by_host"] is True
    assert s["reading"] is None
    assert not flagged(r("SIB", 1, 1, comp=0.9)) and flagged(r("SIB", 1, 1, thr=0.06))    # SIB's companions are its own
    n += 1
    assert summarize({"reps": trip * 4, "experts": 128})["bytes_identical"] is True
    assert summarize({"reps": trip * 3 + trip[:5] + [r("PHYS", 1.0, 4.0, q="c")], "experts": 128})["bytes_identical"] is False
    assert ORDER == ("PHYS", "SIB", "WIDE", "WIDE", "SIB", "PHYS") and CYCLES == 4 and READ == ("fuse_s",)
    assert REPORTED == ("block_s",) and REPEATS == 15
    assert time_fit(REPEATS) <= GUARD_S and time_fit(REPEATS) == 240 + 60 + 27 * (30 + 16 * 12.5) + 600   # 7110 s
    assert time_fit(15) > 3600 and time_fit(5) > 3600 and time_fit(REPEATS) <= GUARD_S - 1800   # 1.0 h never fit
    n += 1
    blocks = [dict(r("PHYS", f, 4.0 + f), throttled_period_share=t, throttled_cpu_s=0.1, pool=12)
              for f, t in ((0.5, 0.0), (0.3, 0.02), (0.9, 0.0), (0.4, 0.07), (0.35, 0.0))]
    m = median_of(blocks)
    assert m["fuse_s"] == 0.4 and m["block_s"] == 4.4 and m["throttled_period_share"] == 0.07   # medians; worst throttle
    assert abs(m["throttled_cpu_s"] - 0.5) < 1e-12 and m["pool"] == 12 and len(m["repeats"]) == 5
    assert m["fused_sha256"] == "a" and m["nf4_sha256"] == "b" and flagged(dict(m, arm="SIB"))
    assert median_of(blocks[:4] + [dict(blocks[4], nf4_sha256="z")])["nf4_sha256"] == "MISMATCH"
    allbad = [dict(x, nf4_sha256="MISMATCH") for x in trip * 4]          # nondeterminism inside EVERY process
    assert summarize({"reps": allbad, "experts": 128})["bytes_identical"] is False
    n += 1
    print(f"smt_sib_box self-test OK ({n}/7 cases)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--quota", default="none")
    p.add_argument("--out")
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--summarize")
    p.add_argument("--rep", choices=ARMS)
    p.add_argument("--cpus")
    p.add_argument("--companions", default="")
    p.add_argument("--pool", type=int)
    p.add_argument("--slab")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.plan:
        quota = None if a.quota == "none" else float(a.quota)
        cores = topology(sorted(os.sched_getaffinity(0)))
        plan, why = choose_arms(cores, quota)
        print("SMT_SIB_PLAN " + json.dumps({"two_thread_cores": len(cores), "quota": quota,
                                            "k": plan["k"] if plan else None, "why": why}))
        return 0 if plan else 5
    if a.summarize:
        print(json.dumps(summarize(json.load(open(a.summarize))), indent=1))
        return 0
    if a.rep:
        return child(a)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
