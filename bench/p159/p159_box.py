#!/usr/bin/env python3
"""p159_box.py -- the BOX side of P159 (bench/p159/PREREG-p159.md): what makes the loader's host-side expert fuse
run 17x slower in some host windows?

Every block is the registered SMT fuse block (smt_fuse_box.one_rep's read, fuse, h2d, quant and offload calls, in that
order, on the same seeded slab). `one_block` below carries the same calls plus counter reads at the segment boundaries;
the counter reads sit outside every segment's timed span, so they land in the residual. tests/test_p159.py checks
that the calls match one_rep's. Every block's fused and nf4 sha256 must equal the registered cross-host constants.

Processes (each fresh, one untimed warm block, then 15 timed blocks):
  M  5 triples of (M, R, M'), rotated one position per triple; M and M' are identical (the null); R fuses from the
     slab read fully into anonymous memory at process start (untimed), with no file mapping.
  N  15 blocks with THP disabled for the process (prctl PR_SET_THP_DISABLE, set before torch is imported).
  P  15 blocks on a pre-faulted heap (GLIBC_TUNABLES: mmap_max=0, no trim; the warm block faults it once).
  B  15 M blocks with a balloon held inside the box's own cgroup, to the 1.3 x registered RAM floor; run only under a
     finite memory.max and a balloon of at most 96 GiB.
Order: 4 cycles of M, X, M, Y, M, Z and a closing M; (X, Y, Z) rotate (N, P, B).

    python p159_box.py --plan                         (stdlib only: the pre-install refusal, exit 5 when refused)
    python p159_box.py --fit                          (stdlib only: the registered fit, in seconds)
    python p159_box.py --out result.json [--experts N]
    python p159_box.py --self-test
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "..", "smt-fuse")):  # staged flat on the box; side by side in the repo
    if os.path.exists(os.path.join(_p, "smt_fuse_box.py")) and _p not in sys.path:
        sys.path.insert(0, _p)

REPEATS = 15
TRIPLE = ("M", "R", "M2")                                # M2 is M' (the null's identical arm)
ROTATIONS = (("M", "R", "M2"), ("R", "M2", "M"), ("M2", "M", "R"))
CYCLE_SLOTS = (("N", "P", "B"), ("P", "B", "N"), ("B", "N", "P"), ("N", "P", "B"))
FAST_S, SLOW_S = 1.5, 3.0                                # fuse seconds per 128-expert layer: FAST <=, SLOW >=
# the registered bytes at 128 experts: smt-sib-2 and smt-fuse-2 produced these on two hosts
FUSED_SHA = "099fa1099154a74f10f6f65af01d94af18d9b29949f210250599589f7c944d3d"
NF4_SHA = "78784a7094e4d9fe3f43c9e74e4f9f734323aca515496497f12bd7694b5956b9"
E_FULL = 128
GIB = 1 << 30
SLAB_BYTES = 3 * 1536 * 4096 * 2 * E_FULL               # 4.83 GB at 128 experts
PR_SET_THP_DISABLE, PR_GET_THP_DISABLE = 41, 42
P_TUNABLES = "glibc.malloc.mmap_max=0:glibc.malloc.trim_threshold=4611686018427387904"
BALLOON_MAX = 96 * GIB
DRY_BALLOON_MAX = 1 * GIB                                # a dry run only exercises the path: never pressure a lab host
FLOOR = 1.3                                              # the registered RAM floor: 1.3 x the measured need
MIN_MEMTOTAL, MIN_AVAILABLE = 64 * GIB, 32 * GIB         # refusals before the install (dry: 8 / 4 GiB)
K_MAX, MIN_K = 16, 8                                     # the pool: K physical cores, one thread each
VMSTAT_KEYS = ("thp_fault_alloc", "thp_fault_fallback", "thp_fault_fallback_charge", "compact_stall", "compact_fail",
               "compact_success", "pgfault", "pgmajfault", "pgscan_direct", "pgsteal_direct", "allocstall_normal",
               "allocstall_movable", "allocstall_dma32", "numa_hint_faults", "numa_pages_migrated")
# The fit (bench/common/README.md, "Guard fit"): each term the max over hosts measured so far plus 25 %, or taken high.
FIT = {"install_to_gate_s": 345, "process_start_s": 3, "block_wall_s": 26.5, "resident_s": 15, "balloon_s": 120,
       "reserve_s": 600}
BOOT_S, GUARD_S = 300, 18000                             # the static fit: (boot + fit) x 1.25 <= the 5.0 h guard
# Closed forms (the PREREG's arithmetic): fresh anonymous bytes of one 128-expert fuse, and its first-touch faults.
FUSE_FRESH_BYTES = E_FULL * (2 * 1536 * 4096 * 2) * 2 + E_FULL * (4096 * 1536 * 2)   # temporaries + gate_up + down
FAULTS_4K, FAULTS_2M = FUSE_FRESH_BYTES // 4096, FUSE_FRESH_BYTES // (2 << 20)


# ---------------------------------------------------------------- the plan (stdlib)

def process_plan():
    """The 25 processes in order: 4 cycles of M, X, M, Y, M, Z, then a closing M."""
    out = []
    for slots in CYCLE_SLOTS:
        for s in slots:
            out += ["M", s]
    return out + ["M"]


def block_arms(kind):
    """The 15 timed blocks' arms for one process kind."""
    if kind == "M":
        return [arm for k in range(REPEATS // 3) for arm in ROTATIONS[k % 3]]
    return {"N": ["N"], "P": ["P"], "B": ["M"]}[kind] * REPEATS


def time_fit(fit=FIT, b_runs=True):
    """NEED (the probe) and FIT, in seconds, for the registered plan."""
    plan = process_plan()
    per = fit["process_start_s"] + (1 + REPEATS) * fit["block_wall_s"]
    need = sum(per + (fit["resident_s"] if k in ("M", "B") else 0) + (fit["balloon_s"] if k == "B" else 0)
               for k in plan if b_runs or k != "B")
    return {"need_s": need, "fit_s": fit["install_to_gate_s"] + need + fit["reserve_s"]}


def balloon_bytes(memory_max, idle_current, peak_delta, floor=FLOOR, dry=False):
    """The balloon that leaves (floor - 1) x peak_delta free at a block's peak, or None when B must not run. A dry run
    clamps it to DRY_BALLOON_MAX: it checks the wiring, so it never holds more than that on a shared lab host."""
    if memory_max is None or peak_delta is None or idle_current is None:
        return None
    b = int(memory_max - idle_current - floor * peak_delta)
    if dry:
        return min(b, DRY_BALLOON_MAX) if b > 0 else None
    return b if 0 < b <= BALLOON_MAX else None


def state_of(fuse_s, experts):
    v = fuse_s * E_FULL / experts
    return "FAST" if v <= FAST_S else "SLOW" if v >= SLOW_S else "MID"


# ---------------------------------------------------------------- host readers (stdlib; missing sources -> None)

def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def meminfo():
    t = _read("/proc/meminfo") or ""
    kv = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in t.splitlines() if len(line.split()) >= 2}
    return {k: kv.get(k) for k in ("MemTotal", "MemAvailable", "MemFree", "Cached", "AnonHugePages")}


def vmstat():
    t = _read("/proc/vmstat") or ""
    kv = dict(line.split() for line in t.splitlines() if len(line.split()) == 2)
    return {k: int(kv[k]) for k in VMSTAT_KEYS if k in kv}


def rusage():
    import resource

    ru = resource.getrusage(resource.RUSAGE_SELF)
    return {"minflt": ru.ru_minflt, "majflt": ru.ru_majflt}


def smaps_rollup():
    t = _read("/proc/self/smaps_rollup") or ""
    kv = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in t.splitlines() if line.endswith("kB")}
    return {"anon_huge_bytes": kv.get("AnonHugePages"), "anon_bytes": kv.get("Anonymous")}


def parse_buddyinfo(text, order_min=9):
    """Free blocks of order >= order_min, summed per node (fragmentation: what huge pages can come from)."""
    out = {}
    for line in (text or "").splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 5 or parts[0] != "Node":
            continue
        counts = [int(x) for x in parts[4:]]
        out[parts[1]] = out.get(parts[1], 0) + sum(counts[order_min:])
    return out


def psi_memory():
    t = _read("/proc/pressure/memory")
    if not t:
        return None
    out = {}
    for line in t.splitlines():
        kind, *fields = line.split()
        out[kind] = int(dict(f.split("=") for f in fields)["total"])
    return out


def steal_ticks():
    t = _read("/proc/stat") or ""
    first = t.splitlines()[0].split() if t else []
    return int(first[8]) if len(first) > 8 else None


def cgroup_memory(root="/sys/fs/cgroup"):
    def num(name):
        v = (_read(os.path.join(root, name)) or "").strip()
        return None if v in ("", "max") else int(v)
    events = _read(os.path.join(root, "memory.events"))
    stat = _read(os.path.join(root, "memory.stat"))
    keep = ("anon", "file", "anon_thp", "pgfault", "pgmajfault", "thp_fault_alloc")
    return {"memory_max": num("memory.max"), "memory_current": num("memory.current"),
            "memory_peak": num("memory.peak"),
            "events": dict((ln.split()[0], int(ln.split()[1])) for ln in events.splitlines()) if events else None,
            "stat": {ln.split()[0]: int(ln.split()[1]) for ln in stat.splitlines() if ln.split()[0] in keep}
            if stat else None}


def slab_mapped(path):
    return path in (_read("/proc/self/maps") or "")


def host_once():
    def first_word_in_brackets(p):
        t = _read(p) or ""
        return t[t.find("[") + 1:t.find("]")] if "[" in t else (t.strip() or None)
    nodes = sorted(d for d in os.listdir("/sys/devices/system/node") if d.startswith("node")) \
        if os.path.isdir("/sys/devices/system/node") else []
    return {"thp_enabled": first_word_in_brackets("/sys/kernel/mm/transparent_hugepage/enabled"),
            "thp_defrag": first_word_in_brackets("/sys/kernel/mm/transparent_hugepage/defrag"),
            "khugepaged_defrag": (_read("/sys/kernel/mm/transparent_hugepage/khugepaged/defrag") or "").strip()
            or None,
            "numa_balancing": (_read("/proc/sys/kernel/numa_balancing") or "").strip() or None,
            "numa_nodes": len(nodes), "kernel": os.uname().release, "mem_total": meminfo()["MemTotal"],
            "cgroup_memory_max": cgroup_memory()["memory_max"]}


def cpu_quota(root="/sys/fs/cgroup"):
    """The cgroup's CPU quota in CPUs (v2 cpu.max, else v1 cfs), or None when unlimited or unreadable."""
    v2 = (_read(os.path.join(root, "cpu.max")) or "").split()
    if len(v2) == 2 and v2[0] != "max":
        return int(v2[0]) / int(v2[1])
    q, per = (_read(os.path.join(root, "cpu", f)) for f in ("cpu.cfs_quota_us", "cpu.cfs_period_us"))
    if q and per and int(q) > 0:
        return int(q) / int(per)
    return None


def pool_size(n_physical, quota, k_max=K_MAX):
    """K: at most k_max physical cores, and no more threads than 80 % of the quota (so the cgroup never binds)."""
    return min(k_max, n_physical, math.floor(quota / 1.25) if quota else k_max)


def physical_pool(affinity, k_max=K_MAX):
    """K physical cores' first threads (one hardware thread each), from the host's sibling topology."""
    seen, out = set(), []
    for cpu in affinity:
        sib = (_read(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") or str(cpu)).strip()
        if sib not in seen:
            seen.add(sib)
            out.append(cpu)
    return out[:k_max]


def plan_check(dry=False):
    """The refusal before the install: MemTotal and MemAvailable floors, K physical cores. Returns (ok, why, facts)."""
    m = meminfo()
    quota = cpu_quota()
    k = pool_size(len(physical_pool(sorted(os.sched_getaffinity(0)), k_max=10**6)), quota)
    total_min, avail_min, k_min = (8 * GIB, 4 * GIB, 1) if dry else (MIN_MEMTOTAL, MIN_AVAILABLE, MIN_K)
    facts = {"mem_total": m["MemTotal"], "mem_available": m["MemAvailable"], "k": k, "quota": quota,
             "cgroup_memory_max": cgroup_memory()["memory_max"]}
    if (m["MemTotal"] or 0) < total_min:
        return False, f"MemTotal {m['MemTotal']} < {total_min}", facts
    if (m["MemAvailable"] or 0) < avail_min:
        return False, f"MemAvailable {m['MemAvailable']} < {avail_min}", facts
    if k < k_min:
        return False, f"pool K {k} < {k_min} (physical cores, or 80 % of the quota)", facts
    return True, None, facts


# ---------------------------------------------------------------- the block (torch; runs in the child)

def boundary():
    return {"ru": rusage(), "vm": vmstat()}


def delta(a, b):
    return {"minflt": b["ru"]["minflt"] - a["ru"]["minflt"], "majflt": b["ru"]["majflt"] - a["ru"]["majflt"],
            **{k: b["vm"][k] - a["vm"][k] for k in a["vm"] if k in b["vm"]}}


def one_block(pool, slab_path, n_experts, resident=None):
    """smt_fuse_box.one_rep's calls, in its order, with counters at the segment boundaries (outside the timed spans).
    resident: the R arm's (gate, up, down) lists, read into anonymous memory at process start; None reads the file."""
    import gc
    import hashlib

    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    import smt_fuse_box as SF
    from experts4bit_qlora._vendor.experts import Experts4bit
    from experts4bit_qlora.arch.moe_conventions import fuse_experts
    from experts4bit_qlora.engines.offload import enable_expert_offload
    from experts4bit_qlora.lora import ExpertsLoRA

    KEY = SF.KEY
    torch.set_num_threads(pool)
    pre = {"buddy_order9": parse_buddyinfo(_read("/proc/buddyinfo")), "mem": meminfo(), "psi": psi_memory(),
           "steal": steal_ticks(), "cgroup": cgroup_memory(), "smaps": smaps_rollup()}
    cs0 = SF.cpu_stat()
    b0 = boundary()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if resident is None:
        with safe_open(slab_path, framework="pt", device="cpu") as f:
            gate = [f.get_tensor(KEY.format(e=e, p="gate_proj")) for e in range(n_experts)]
            up = [f.get_tensor(KEY.format(e=e, p="up_proj")) for e in range(n_experts)]
            down = [f.get_tensor(KEY.format(e=e, p="down_proj")) for e in range(n_experts)]
    else:
        gate, up, down = (list(x) for x in resident)
    t1 = time.perf_counter()
    b1 = boundary()
    mapped_during_read = slab_mapped(slab_path)
    t1b = time.perf_counter()
    gu, dn = fuse_experts(gate, up, down)
    t2 = time.perf_counter()
    b2 = boundary()
    sm = smaps_rollup()
    t2b = time.perf_counter()
    gu_d, dn_d = gu.to("cuda"), dn.to("cuda")
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    base = Experts4bit.from_float(gu_d, dn_d, has_gate=True, activation=F.silu, compute_dtype=torch.bfloat16,
                                  quant_type="nf4", blocksize=64)
    torch.cuda.synchronize()
    t4 = time.perf_counter()
    nf4 = hashlib.sha256((SF.sha(base.gate_up_proj) + SF.sha(base.down_proj)).encode()).hexdigest()
    b4 = boundary()
    t4b = time.perf_counter()                            # the hash and the counter read are excluded from the walls
    experts = ExpertsLoRA(base, r=16, alpha=16, dtype=torch.bfloat16).to("cuda")
    handle = enable_expert_offload(experts, "cuda", pin=True)
    torch.cuda.synchronize()
    t5 = time.perf_counter()
    b5 = boundary()
    t5b = time.perf_counter()
    fused = hashlib.sha256((SF.sha(gu) + SF.sha(dn)).encode()).hexdigest()
    t5c = time.perf_counter()
    del gate, up, down, gu, dn, gu_d, dn_d, base, experts, handle
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    try:
        from experts4bit_qlora.engines.host_heap import release_cached_pinned_memory
        release_cached_pinned_memory()
    except Exception:                                    # noqa: BLE001 -- recorded below
        pre["pinned_release"] = "unavailable"
    t6 = time.perf_counter()
    rec = {"read_s": t1 - t0, "fuse_s": t2 - t1b, "h2d_s": t3 - t2b, "quant_s": t4 - t3, "offload_s": t5 - t4b,
           "nf4_sha256": nf4, "fused_sha256": fused, "torch_pool": torch.get_num_threads(),
           "slab_mapped_during_read": mapped_during_read, "smaps_after_fuse": sm,
           "read": delta(b0, b1), "fuse": delta(b1, b2), "offload": delta(b4, b5), "pre": pre,
           "psi_after": psi_memory(), "steal_after": steal_ticks()}
    rec["total_s"] = rec["read_s"] + rec["fuse_s"] + rec["h2d_s"] + rec["quant_s"] + rec["offload_s"]
    rec["block_s"] = (t6 - t0) - (t4b - t4) - (t5c - t5b)
    rec["residual_s"] = rec["block_s"] - rec["total_s"]
    cs1 = SF.cpu_stat()
    if cs0 and cs1:
        dp = cs1["nr_periods"] - cs0["nr_periods"]
        rec["throttled_period_share"] = (cs1["nr_throttled"] - cs0["nr_throttled"]) / dp if dp > 0 else 0.0
    return rec


def vm_hwm():
    t = _read("/proc/self/status") or ""
    return next((int(ln.split()[1]) * 1024 for ln in t.splitlines() if ln.startswith("VmHWM:")), None)


def child(a):
    """One process: the arm's setup before torch is imported, the warm block untimed, then 15 timed blocks."""
    os.sched_setaffinity(0, [int(x) for x in a.cpus.split(",")])
    engaged = {}
    if a.kind == "N":
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        rc = libc.prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0)
        engaged["thp_disabled"] = rc == 0 and libc.prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0) == 1
    if a.kind == "P":
        engaged["tunables"] = os.environ.get("GLIBC_TUNABLES") == P_TUNABLES
    import torch

    import smt_fuse_box as SF  # noqa: F401 -- the slab's keys and helpers

    torch.cuda.init()
    process_start_s = time.time() - a.spawned if a.spawned else None
    resident, resident_s = None, None
    if a.kind in ("M", "B"):                             # the R arm's slab, in anonymous memory, untimed
        from safetensors import safe_open
        t = time.perf_counter()
        with safe_open(a.slab, framework="pt", device="cpu") as f:
            resident = tuple([f.get_tensor(SF.KEY.format(e=e, p=p)).clone() for e in range(a.experts)]
                             for p in ("gate_proj", "up_proj", "down_proj"))
        resident_s = time.perf_counter() - t
    t = time.perf_counter()
    one_block(a.pool, a.slab, a.experts, resident=None)  # untimed warm block (P: faults the heap once)
    warm_wall_s = time.perf_counter() - t
    blocks = []
    for arm in block_arms(a.kind):
        t = time.perf_counter()
        rec = one_block(a.pool, a.slab, a.experts, resident=resident if arm == "R" else None)
        rec.update(arm=arm, wall_s=time.perf_counter() - t)
        blocks.append(rec)
    out = {"kind": a.kind, "engaged": engaged, "process_start_s": process_start_s, "resident_s": resident_s,
           "warm_wall_s": warm_wall_s, "vm_hwm": vm_hwm(), "blocks": blocks}
    print("P159_PROCESS " + json.dumps(out), flush=True)
    return 0


# ---------------------------------------------------------------- the parent

def inflate_balloon(nbytes):
    """A separate process that touches nbytes of anonymous memory and holds it until killed."""
    code = ("import mmap,sys,time\nn=int(sys.argv[1]);m=mmap.mmap(-1,n)\n"
            "for i in range(0,n,4096): m[i]=1\nprint('BALLOON_READY',flush=True)\ntime.sleep(10**7)\n")
    p = subprocess.Popen([sys.executable, "-c", code, str(nbytes)], stdout=subprocess.PIPE, text=True)
    line = p.stdout.readline()
    if not line.startswith("BALLOON_READY"):
        p.kill()
        return None
    return p


def run(a):
    sys.path.insert(0, HERE)
    import torch

    import smt_fuse_box as SF

    dry = os.environ.get("P159_DRY") == "1"
    ok, why, pfacts = plan_check(dry)
    if not ok:
        print("P159_REFUSED " + json.dumps({"why": why, **pfacts}), flush=True)
        return 5
    facts = {**SF.host_facts(), **host_once()}
    phys = physical_pool(sorted(os.sched_getaffinity(0)), k_max=10**6)
    pool_cpus = phys[:max(1, pool_size(len(phys), pfacts["quota"]))]
    cache = os.path.join(os.path.dirname(os.path.abspath(a.out)), ".cache")
    os.makedirs(cache, exist_ok=True)
    slab_path = os.path.join(cache, "slab.safetensors")
    SF.write_slab(slab_path, *SF.slab(a.experts))
    time_left = float(os.environ.get("P159_TIME_LEFT_S", "inf"))
    t_start = time.perf_counter()
    plan = process_plan()
    print("P159_PLAN " + json.dumps({"processes": plan, "pool": len(pool_cpus), "facts": facts}), flush=True)

    def proc(kind, limit):
        env = dict(os.environ)
        if kind == "P":
            env["GLIBC_TUNABLES"] = P_TUNABLES
        cmd = [sys.executable, os.path.abspath(__file__), "--child", kind, "--cpus", ",".join(map(str, pool_cpus)),
               "--pool", str(len(pool_cpus)), "--slab", slab_path, "--experts", str(a.experts),
               "--spawned", repr(time.time())]
        t0 = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=max(1.0, limit), env=env)
        except subprocess.TimeoutExpired:
            return None, time.perf_counter() - t0, "timeout"
        line = next((x for x in p.stdout.splitlines() if x.startswith("P159_PROCESS ")), None)
        if p.returncode != 0 or line is None:
            return None, time.perf_counter() - t0, (p.stdout + p.stderr)[-800:]
        return json.loads(line[len("P159_PROCESS "):]), time.perf_counter() - t0, None

    idle = cgroup_memory()
    processes, b_status, first_wall = [], None, None
    for i, kind in enumerate(plan):
        left = time_left - (time.perf_counter() - t_start)
        cap = left if first_wall is None else min(left, max(600.0, 2.0 * first_wall))
        if left <= 0:
            print(f"P159_CAP_HIT after {len(processes)} processes", flush=True)
            return 4
        balloon = None
        if kind == "B":
            if b_status is None:
                # the need: the cgroup's peak above its idle level so far (memory.peak), else the first M process's
                # VmHWM plus the slab's page cache; the balloon leaves (FLOOR - 1) x need free at a block's peak
                now = cgroup_memory()
                first_m = next(p for p in processes if p.get("kind") == "M")
                if now["memory_peak"] is not None and idle["memory_current"] is not None:
                    need, source = now["memory_peak"] - idle["memory_current"], "memory.peak"
                else:
                    need, source = (first_m.get("vm_hwm") or 0) + SLAB_BYTES * a.experts // E_FULL, "VmHWM+slab"
                b_status = {"bytes": balloon_bytes(idle["memory_max"], now["memory_current"], need, dry=dry),
                            "need": need, "need_source": source, "memory_max": idle["memory_max"]}
                print("P159_BALLOON " + json.dumps(b_status), flush=True)
            if not b_status["bytes"]:
                processes.append({"kind": "B", "not_run": "balloon refused (no finite memory.max, or over 96 GiB)"})
                continue
            tb = time.perf_counter()
            balloon = inflate_balloon(b_status["bytes"])
            if balloon is None:
                processes.append({"kind": "B", "not_run": "balloon failed to inflate"})
                continue
            b_status.setdefault("inflate_s", []).append(time.perf_counter() - tb)
        try:
            rec, wall, err = proc(kind, cap)
        finally:
            if balloon is not None:
                balloon.kill()
                balloon.wait()
        if rec is None:
            print(f"P159_CHILD_FAILED process {i} {kind}: {err}", flush=True)
            return 4 if err == "timeout" else 6
        if first_wall is None:
            first_wall = wall
        rec.update(index=i, process_wall_s=wall)
        processes.append(rec)
        print(f"P159_DONE {i} {kind} wall={wall:.1f}s", flush=True)
    res = {"facts": facts, "experts": a.experts, "pool": len(pool_cpus), "gpu": torch.cuda.get_device_name(0),
           "e4b_sha": os.environ.get("E4B_SHA"), "plan": plan, "balloon": b_status, "processes": processes}
    shas = {(b["fused_sha256"], b["nf4_sha256"]) for p in processes for b in p.get("blocks", [])}
    res["bytes_ok"] = (shas == {(FUSED_SHA, NF4_SHA)}) if a.experts == E_FULL else (len(shas) == 1)
    json.dump(res, open(a.out, "w"), indent=1)
    print("P159_RESULT " + json.dumps({"bytes_ok": res["bytes_ok"], "processes": len(processes)}), flush=True)
    return 0 if res["bytes_ok"] else 3


# ---------------------------------------------------------------- self-test (stdlib)

def self_test():
    plan = process_plan()
    assert len(plan) == 25 and plan.count("M") == 13 and all(plan.count(k) == 4 for k in "NPB"), plan
    assert all(plan[i - 1] == "M" and plan[i + 1] == "M" for i, k in enumerate(plan) if k != "M")
    m = block_arms("M")
    assert len(m) == REPEATS and all(m.count(x) == 5 for x in TRIPLE)
    for x in TRIPLE:                                     # each arm takes each triple position at least once
        assert {m[3 * k:3 * k + 3].index(x) for k in range(5)} == {0, 1, 2}, x
    assert block_arms("B") == ["M"] * REPEATS and block_arms("N") == ["N"] * REPEATS
    f = time_fit()
    assert f == {"need_s": 11410.0, "fit_s": 12355.0}, f
    assert int(f["fit_s"]) * 125 // 100 == 15443
    assert (BOOT_S + f["fit_s"]) * 1.25 <= GUARD_S and (BOOT_S + f["fit_s"]) * 1.25 == 15818.75
    assert FUSE_FRESH_BYTES == int(7.5 * GIB) and FAULTS_4K == 1966080 and FAULTS_2M == 3840
    assert balloon_bytes(None, 10, 10) is None
    assert balloon_bytes(100 * GIB, 10 * GIB, 20 * GIB) == 64 * GIB
    assert balloon_bytes(400 * GIB, 10 * GIB, 20 * GIB) is None          # over 96 GiB: B not run
    assert balloon_bytes(30 * GIB, 10 * GIB, 20 * GIB) is None           # no room: B not run
    assert balloon_bytes(400 * GIB, 10 * GIB, 20 * GIB, dry=True) == GIB  # a dry run holds at most 1 GiB
    assert balloon_bytes(30 * GIB, 10 * GIB, 20 * GIB, dry=True) is None
    assert state_of(0.4, 128) == "FAST" and state_of(7.2, 128) == "SLOW" and state_of(2.0, 128) == "MID"
    assert state_of(0.4 * 8 / 128, 8) == "FAST"
    bi = "Node 0, zone   Normal  1 2 3 4 5 6 7 8 9 10 11\nNode 1, zone   Normal  0 0 0 0 0 0 0 0 0 5 6\n"
    assert parse_buddyinfo(bi) == {"0": 21, "1": 11}
    assert pool_size(128, 61.2) == 16 and pool_size(128, 12.0) == 9 and pool_size(6, None) == 6
    print("p159_box self-test OK (15 cases)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--fit", action="store_true")
    p.add_argument("--out")
    p.add_argument("--experts", type=int, default=E_FULL)
    p.add_argument("--child")
    p.add_argument("--cpus")
    p.add_argument("--pool", type=int)
    p.add_argument("--slab")
    p.add_argument("--spawned", type=float)
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.plan:
        ok, why, facts = plan_check(os.environ.get("P159_DRY") == "1")
        print("P159_PLAN_CHECK " + json.dumps({"ok": ok, "why": why, **facts}), flush=True)
        return 0 if ok else 5
    if a.fit:
        print(int(time_fit()["fit_s"]))
        return 0
    if a.child:
        a.kind = a.child
        return child(a)
    if a.out:
        return run(a)
    p.error("one of --self-test, --plan, --fit, --out, --child")


if __name__ == "__main__":
    raise SystemExit(main())
