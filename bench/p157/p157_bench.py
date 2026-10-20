# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P157's microbench (bench/p157/PREREG-p157.md): K25's stages knocked out one at a time, at today's plan on
Qwen3-30B-A3B's W16 shapes, on S0's real router picks.

A0 is K25 (grouped-nf4-gemm ``nf4_smallm.gemm_nf4_grouped_smallm``) called exactly as P155 called it, which is how e4b's
served route calls it on the NF4 store at 16 rows: gate_up gathers the 16 token rows through the tile table's ``order``
(``gather_div=8``) and down stores into the caller's order (``scatter=order``), over
``int4_b32.build_group_tiles_fused(ids, 128, 16, lean=True, sorted_ids=True)``. The other arms are
``p157_arms.launch`` in the same call form. Full-size random NF4 stores for all 48 MoE layers (16.3 GB), so no replay
is served from L2 across layers.

1. PINS. Each arm's compiled kernel, per projection: registers, local memory, shared memory and carveout, read from the
   driver. Where the driver's occupancy query gives an arm more CTAs per SM than A0, dynamic shared memory is added
   (128-byte steps) until it gives A0's, and the pad is verified by the same query.
2. STATIC. The SASS of every kernel that runs, counted with sass_count (gnf4 ``bench/sass_count``, staged byte for
   byte): the inner loop, loads per weight, the whole function's loads, an instruction-stream hash.
3. REPS. On each of 70 router-replay steps (58-127), one paired rep: every arm's graph (both GEMMs of all 48 layers), captured
   and replayed in the registered order (a Williams design), replays interleaved arm by arm, each timed replay right after an untimed
   replay of the spacer S, a graph of K25's calls with its own output buffers (every arm follows the same work, so no
   arm inherits a light neighbour's power and clock headroom, and none finds its own outputs warm). While each capture
   is open, its kernel nodes' ``sharedMemBytes`` are read back. NVML samples clocks while each replay runs. After the
   replays, T2's, A0p's and ACX0's (the ladder's cubin unrotated) outputs are compared with A0's, bit for bit. Each rung's DRAM dose comes from the step's routing: the
   distinct ``eid & emask`` experts of every call.

The record holds the raw replay times; ``p157_reduce.py`` derives every read from them.

    python p157_bench.py --rr rr.json.pt --out bench_p157.json
    python p157_bench.py --rr rr.json.pt --out shake.json --shakedown --layers 4 --reps 8
    python p157_bench.py --rr rr.json.pt --ncu-target ACX --estimate-out est.json   # under ncu: one arm's calls
    python p157_bench.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p157_reduce as R  # noqa: E402  (the registered rules: the staged sibling)

NCU_LAYERS = 4                  # the ncu pass: layers 0-3 of the first held-out step, one arm per process
#: The K25-named and ablation-named launches before the measured round, which the runner passes to ncu's --launch-skip
#: with --kernel-name: the pins compile every arm once per projection, then three warm rounds of the NCU_LAYERS layers.
NCU_SKIP = 2 * (len(R.ARMS) - 1) + 3 * 2 * NCU_LAYERS
NCU_COUNT = 2 * NCU_LAYERS
BLOCK = R.TODAY["warps"] * 32


class Poisoned(RuntimeError):
    """A failure left the CUDA context unusable; nothing after it can be timed."""


def s0_picks(rr_path):
    """S0's picks from P153's router replay, checked before any data (P155's checks, unchanged)."""
    import torch
    raw = open(rr_path, "rb").read()
    if hashlib.sha256(raw).hexdigest() != R.RR_SHA256:
        raise SystemExit(f"P157_ROUTING_FAULT rr.json.pt sha256 is not the pinned {R.RR_SHA256[:8]}")
    d = torch.load(rr_path, weights_only=True)
    picks = d["picks"]
    if tuple(picks.shape) != (R.LAYERS, 2 * R.TOKENS, R.STEPS, R.TOPK) or picks.dtype != torch.int16:
        raise SystemExit(f"P157_ROUTING_FAULT picks {tuple(picks.shape)} {picks.dtype}")
    s0 = picks[:, 0:R.TOKENS].contiguous()
    sha = hashlib.sha256(s0.numpy().astype("<i2").tobytes()).hexdigest()
    if sha != R.S0_SHA256:
        raise SystemExit(f"P157_ROUTING_FAULT S0's picks hash {sha[:8]}, pinned {R.S0_SHA256[:8]}")
    if int(s0.min()) < 0 or int(s0.max()) >= R.E:
        raise SystemExit("P157_ROUTING_FAULT an expert id outside [0, 128)")
    srt = s0.sort(dim=-1).values
    if bool((srt[..., 1:] == srt[..., :-1]).any()):
        raise SystemExit("P157_ROUTING_FAULT a row picks one expert twice")
    return s0, sha


def slabs(s0, step, layers):
    """Each rung's distinct masked experts per call: {str(mask): {"gu": [per layer], "dn": [per layer]}} (both
    projections route the same ids)."""
    out = {}
    for mask in (0, 3, 15, 127):
        per = [len({int(e) & mask for e in s0[li, :, step, :].reshape(-1).tolist()}) for li in range(layers)]
        out[str(mask)] = {"gu": per, "dn": list(per)}
    return out


def variants(fn):
    """{key: CompiledKernel} in a Triton JIT function's per-device caches (P155's reader)."""
    out = {}
    for dev in getattr(fn, "device_caches", {}).values():
        cache = dev[0] if isinstance(dev, tuple) else dev
        if isinstance(cache, dict):
            out.update(cache)
    return out


_ADDR = re.compile(r"^/\*[0-9a-f]+\*/\s*")


def static_record(kernel, sc, *, trip, bytes_per_iteration=None):
    """sass_count's record of a kernel that ran, from its own cubin, disassembled (:func:`static_from_sass`)."""
    from triton.tools.disasm import path_to_cuobjdump
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(kernel.asm["cubin"])
        path = f.name
    try:
        raw = subprocess.run([path_to_cuobjdump(), "-sass", path], capture_output=True, text=True, check=True).stdout
    finally:
        os.remove(path)
    return static_from_sass(raw, sc, arch=kernel.metadata.target.arch, trip=trip,
                            bytes_per_iteration=bytes_per_iteration, n_regs=kernel.n_regs, n_spills=kernel.n_spills,
                            shared=kernel.metadata.shared)


def static_from_sass(raw, sc, *, arch, trip, bytes_per_iteration=None, **facts):
    """The record of one disassembled kernel. Whether it holds an MMA is read from its SASS (``has_mma``), never
    assumed from the arm: with one, the inner loop is found by the MMA and unroll-checked by MMA count; without one, by
    the load stream, unroll-checked by bytes against A0's (``bytes_per_iteration``). Also the whole function's loads
    and an instruction-stream hash (addresses stripped, so a kernel's name does not enter it)."""
    rows = sc.parse_sass(raw)
    has_mma = any(sc.classify(op) == "mma" for _, op, _ in rows)
    rec = sc.loop_record(rows, num_warps=R.TODAY["warps"], weights_per_iteration=R.TODAY["block_n"] * R.TODAY["kc"],
                         tile=(16, R.TODAY["block_n"], R.TODAY["kc"]), loop_key="mma" if has_mma else "load",
                         bytes_per_iteration=None if has_mma else bytes_per_iteration)
    loop = rec.get("inner_loop")
    out = {**sc.toolchain(), "arch": f"sm_{arch}", "total_instructions": len(rows), "has_mma": has_mma,
           "sass_sha256": hashlib.sha256("\n".join(_ADDR.sub("", t) for _, _, t in rows).encode()).hexdigest(),
           "loads_in_function": rec["loads_in_function"], "trip": trip, "unrolled_away": loop is None, **facts}
    if loop is not None:
        out.update(loop_instructions=loop["instructions"], loop_opcodes=loop["opcodes"], unroll=rec.get("unroll"),
                   unroll_guard=rec.get("unroll_guard"), lane_instr_per_weight=rec["lane_instr_per_weight"],
                   loads_per_weight=rec["loads_per_weight"], shared_per_weight=rec["shared_per_weight"],
                   loads_outside_loop=rec.get("loads_outside_loop"),
                   loop_loads={op: n for op, n in loop["opcodes"].items() if sc.classify(op) == "load_global"},
                   stream_bytes=sc.stream_bytes(rec) if rec.get("unroll") else None)
    return out


class Box:
    """The stores, the tables and the calls, shared by the reading and the ncu target."""

    def __init__(self, a, steps, layers):
        import torch
        import nf4_smallm
        import p157_arms as A
        from int4_b32 import build_group_tiles_fused
        os.environ["P157_ABLATION"] = "1"
        self.torch, self.A, self.k25 = torch, A, nf4_smallm.gemm_nf4_grouped_smallm
        self.kfn = nf4_smallm._gemm_nf4_grouped_smallm
        self.s0, self.s0_sha = s0_picks(a.rr)
        self.L = layers
        dev = "cuda"
        g = torch.Generator(device=dev).manual_seed(157)
        self.W = []
        for _ in range(layers):
            self.W.append({proj: (torch.randint(0, 256, (R.E, N, K // 2), dtype=torch.uint8, generator=g, device=dev),
                                  torch.rand(R.E, N, K // 64, generator=g, device=dev) * 0.05 + 0.01)
                           for proj, (N, K) in R.SHAPES.items()})
        self.X = [(torch.randn(R.TOKENS, R.SHAPES["gu"][1], generator=g, device=dev) / 4).to(torch.bfloat16)
                  for _ in range(layers)]
        self.H = [(torch.randn(R.TOKENS * R.TOPK, R.SHAPES["dn"][1], generator=g, device=dev) / 4).to(torch.bfloat16)
                  for _ in range(layers)]
        self.tabs = {}
        for s in steps:
            self.tabs[s] = []
            for li in range(layers):
                ids = self.s0[li, :, s, :].reshape(-1).to(torch.long).to(dev)
                row0, rows, grp, order, _c, _s = build_group_tiles_fused(ids, R.E, 16, lean=True, sorted_ids=True)
                self.tabs[s].append((row0, rows, grp, order))
        torch.cuda.synchronize()
        self.pad = {arm: {"gu": 0, "dn": 0} for arm in R.ARMS}

    def call(self, arm, proj, li, tab):
        row0, rows, grp, order = tab
        p, am = self.W[li][proj]
        if arm in ("A0", "T2", R.SPACER):
            if proj == "gu":
                return self.k25(self.X[li], p, am, row0, rows, grp, order, gather_div=R.TOPK, lut="tree",
                                dot_bf16=False, **R.TODAY)
            return self.k25(self.H[li], p, am, row0, rows, grp, None, scatter=order, lut="tree", dot_bf16=False,
                            **R.TODAY)
        plan = dict(block_n=R.TODAY["block_n"], kc=R.TODAY["kc"], warps=R.TODAY["warps"], stages=R.TODAY["stages"],
                    pad=self.pad[arm][proj])
        if proj == "gu":
            return self.A.launch(arm, self.X[li], p, am, row0, rows, grp, order, gather_div=R.TOPK, **plan)
        return self.A.launch(arm, self.H[li], p, am, row0, rows, grp, None, scatter=order, **plan)

    def graph_fn(self, arm, s):
        return lambda: [(self.call(arm, "gu", li, self.tabs[s][li]), self.call(arm, "dn", li, self.tabs[s][li]))
                        for li in range(self.L)]

    def pin(self, step):
        """Compile every arm once per projection, read its kernel from the driver, and size its pad (module
        docstring, step 1). Returns {arm: {proj: facts}} and {arm: {proj: kernel}}."""
        from p157_cuda import func_attrs, occupancy
        torch = self.torch
        kernels, facts = {}, {}
        tab = self.tabs[step][0]
        for arm in R.ARMS:
            if arm == "T2":
                continue
            kernels[arm] = {}
            fn = self.kfn if arm == "A0" else self.A._k25_ablate
            for proj in ("gu", "dn"):
                before = set(variants(fn))
                self.call(arm, proj, 0, tab)
                torch.cuda.synchronize()
                new = [k for key, k in variants(fn).items() if key not in before]
                if new:
                    kernels[arm][proj] = new[0]
                elif arm in self.A.SHARES:                # one cubin by construction (the ladder; A3 with AL)
                    kernels[arm][proj] = kernels[self.A.SHARES[arm]][proj]
                else:
                    raise RuntimeError(f"{arm} {proj}: no new compiled variant")
        for proj in ("gu", "dn"):
            a0 = kernels["A0"][proj]
            a0._init_handles()
            want = occupancy(a0.function, BLOCK, a0.metadata.shared)
            for arm, ks in kernels.items():
                k = ks[proj]
                k._init_handles()
                base = occupancy(k.function, BLOCK, k.metadata.shared)
                pad = 0
                if arm != "A0" and base > want:
                    room = 48 * 1024 - k.metadata.shared
                    pad = next((p for p in range(128, room + 1, 128)
                                if occupancy(k.function, BLOCK, k.metadata.shared + p) <= want), 0)
                    if pad and arm in self.A.SHARES:
                        pad = self.pad[self.A.SHARES[arm]][proj]   # one cubin, one pad
                    if pad:
                        self.A.set_pad(k, pad)
                self.pad.setdefault(arm, {})[proj] = pad
                launch_shared = k.metadata.shared + pad
                facts.setdefault(arm, {})[proj] = {
                    "shared": k.metadata.shared, "pad": pad, "launch_shared": launch_shared,
                    "ctas_per_sm_unpinned": base, "ctas_per_sm": occupancy(k.function, BLOCK, launch_shared),
                    "n_regs": k.n_regs, "n_spills": k.n_spills, **func_attrs(k.function)}
        self.pad["T2"] = dict(self.pad["A0"])
        self.pad["ACX0"] = dict(self.pad["AC"])
        self.pad[R.SPACER] = dict(self.pad["A0"])
        return facts, kernels


def main(a):
    import torch
    from p157_cuda import ClockSampler, capture_kernel_nodes, device_attrs
    t0 = time.time()
    walls = {}
    steps = list(R.CONFIRM_STEPS[:a.reps])
    if not a.shakedown and (a.layers != R.LAYERS or steps != list(R.CONFIRM_STEPS)):
        raise SystemExit("reading knobs are frozen: --layers and --reps are shakedown-only")
    t = time.time()
    box = Box(a, steps, a.layers)
    L = box.L
    walls["setup_s"] = round(time.time() - t, 1)
    cap = torch.cuda.get_device_capability()
    rec = {"lane": "P157", "shakedown": bool(a.shakedown), "device": torch.cuda.get_device_name(), "cc": list(cap),
           "torch": torch.__version__, "layers": L, "rr_sha256": R.RR_SHA256, "s0_sha256": box.s0_sha,
           "today": R.TODAY, "arms": list(R.ARMS), "confirm_steps": steps, "iters": R.ITERS, "spacer": R.SPACER,
           "device_attrs": device_attrs(), "store_bytes": sum(p.numel() + q.numel() * 4 for st in box.W
                                                              for p, q in st.values())}
    import triton
    rec["triton"] = triton.__version__
    rec["n_real"] = [[len(set(box.s0[li, :, s, :].reshape(-1).tolist())) for li in range(L)]
                     for s in R.CONFIRM_STEPS]

    # ---- 1. PINS and 2. STATIC
    t = time.time()
    occ, kernels = box.pin(steps[0])
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import sass_count as sc
    static = {}
    trips = {p: -(-K // R.TODAY["kc"]) for p, (_N, K) in R.SHAPES.items()}
    for proj in ("gu", "dn"):
        a0 = static_record(kernels["A0"][proj], sc, trip=trips[proj])
        static.setdefault("A0", {})[proj] = a0
        for arm, ks in kernels.items():
            if arm == "A0":
                continue
            try:                                      # an arm without an MMA is unroll-checked by A0's stream bytes
                static.setdefault(arm, {})[proj] = static_record(ks[proj], sc, trip=trips[proj],
                                                                 bytes_per_iteration=a0["stream_bytes"])
            except Exception as e:                    # noqa: BLE001  (recorded; the reducer's gate fails the arm)
                static.setdefault(arm, {})[proj] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    rec["static"], rec["occupancy"] = static, occ
    walls["pin_static_s"] = round(time.time() - t, 1)
    print("P157_PINS " + json.dumps({arm: {p: (o["ctas_per_sm_unpinned"], o["pad"], o["ctas_per_sm"])
                                           for p, o in v.items()} for arm, v in occ.items()}), flush=True)

    def check_context():
        try:
            torch.cuda.synchronize()
        except Exception as e:
            raise Poisoned(f"{type(e).__name__}: {str(e)[:200]}") from e

    def capture(fn, warm=2):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            outs = fn()
            nodes = capture_kernel_nodes(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return gr, outs, nodes

    def replay_ms(gr):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        gr.replay()
        e1.record()
        e1.synchronize()
        return e0.elapsed_time(e1)

    # ---- 3. REPS
    t = time.time()
    grid_y = {p: -(-N // R.TODAY["block_n"]) for p, (N, _K) in R.SHAPES.items()}
    proj_of = {v: k for k, v in grid_y.items()}
    sampler = ClockSampler()
    sampler.start()
    reps, node_shared = [], {arm: {"gu": set(), "dn": set()} for arm in R.ARMS}
    try:
        for r, s in enumerate(steps):
            rot = R.order(R.ARMS, r)
            graphs = {R.SPACER: capture(box.graph_fn(R.SPACER, s))}   # its own buffers: never timed
            for arm in rot:
                try:
                    graphs[arm] = capture(box.graph_fn(arm, s))
                except Exception:
                    check_context()
                    raise
                for _f, gd, _b, sh in graphs[arm][2]:
                    node_shared[arm].setdefault(proj_of.get(gd[1], f"grid_y={gd[1]}"), set()).add(sh)
            rp = {arm: [] for arm in rot}
            for _ in range(R.ITERS):
                for arm in rot:
                    graphs[R.SPACER][0].replay()      # untimed: every timed replay follows the same work
                    sampler.tag = arm
                    rp[arm].append(replay_ms(graphs[arm][0]))
                    sampler.tag = None
            ref = graphs["A0"][1]
            ident = {arm: all(torch.equal(o[0], q[0]) and torch.equal(o[1], q[1]) for o, q in zip(graphs[arm][1], ref))
                     for arm in ("T2", "A0p", "ACX0")}
            reps.append({"step": s, "order": rot, "replays_ms": rp, "identical": ident,
                         "slabs": slabs(box.s0, s, L)})
            del graphs, ref
            torch.cuda.empty_cache()
            print(f"P157_REP {r} step {s} " + " ".join(f"{a_}={statistics.median(rp[a_]):.3f}" for a_ in R.ARMS)
                  + f" identical={all(ident.values())}", flush=True)
    except Poisoned as e:
        rec["poisoned"] = str(e)
    finally:
        sampler.stop()
    for arm, v in node_shared.items():
        for proj, shs in v.items():
            if arm in occ:
                occ[arm][proj]["node_shared"] = sorted(shs)
    rec["reps"], rec["clocks"] = reps, sampler.summary()
    walls["reps_s"] = round(time.time() - t, 1)
    rec["walls"], rec["wall_s"] = walls, round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    if rec.get("poisoned"):
        print(f"P157_POISONED {rec['poisoned']}", flush=True)
        return 32
    print(f"P157_BENCH done in {rec['wall_s']} s; walls {json.dumps(walls)}", flush=True)
    return 0


def ncu_target(a):
    """What Nsight Compute measures (one process per arm): layers 0-3 of the first held-out step at the arm's pins.
    Three warm rounds of (gate_up, down) per layer, then the round the runner's ``ncu --launch-skip`` measures. Writes
    the router-replay estimate of the same calls' DRAM bytes to ``--estimate-out``. Prints nothing (ncu's CSV stays
    clean)."""
    s = R.CONFIRM_STEPS[0]
    box = Box(a, [s], NCU_LAYERS)
    box.pin(s)
    for _ in range(4):
        for li in range(NCU_LAYERS):
            box.call(a.ncu_target, "gu", li, box.tabs[s][li])
            box.call(a.ncu_target, "dn", li, box.tabs[s][li])
    box.torch.cuda.synchronize()
    if a.estimate_out:
        sl = slabs(box.s0, s, NCU_LAYERS)
        mask = R.MASKS.get(a.ncu_target, 127)
        with open(a.estimate_out, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"arm": a.ncu_target, "step": s, "layers": NCU_LAYERS, "estimate_bytes": R.dose(sl, mask)}, f)
    return 0


def ncu_collect():
    """The runner's Nsight pass, gathered: for each arm in ``NCU_ARMS``, the summed ``dram__bytes_read.sum`` of its
    ``ncu_<arm>.csv`` (exactly ``NCU_COUNT`` kernels, else refused) and the router-replay estimate its target wrote."""
    import csv
    out = {"dram_read_bytes": {}, "estimate_bytes": {}, "kernels": {}}
    want = int(os.environ["NCU_COUNT"])
    for arm in os.environ["NCU_ARMS"].split():
        with open(f"ncu_{arm}.csv", encoding="utf-8", errors="replace") as f:
            rows = [r for r in csv.DictReader(ln for ln in f if ln.startswith('"'))
                    if r.get("Metric Name") == "dram__bytes_read.sum"]
        if len(rows) != want:
            raise SystemExit(f"{arm}: {len(rows)} kernels measured, expected {want}")
        out["kernels"][arm] = len(rows)
        out["dram_read_bytes"][arm] = sum(float(r["Metric Value"].replace(",", "")) for r in rows)
        with open(f"est_{arm}.json", encoding="utf-8") as f:
            out["estimate_bytes"][arm] = json.load(f)["estimate_bytes"]
    json.dump(out, sys.stdout, indent=1)
    return 0


def self_test():
    """The CPU-checkable wiring: the rules come from the reducer, the call form is the served route's, the steps are
    58-127 in whole Williams squares (every arm takes every position, and every ordered neighbour pair
    recurs, equally often), and each rung's dose
    counts distinct masked experts."""
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    src = src[:src.index("\ndef self_test")]          # the code under test, never these asserts' own literals
    assert "gather_div=R.TOPK, lut=\"tree\",\n                                dot_bf16=False, **R.TODAY" in src
    assert "None, scatter=order, lut=\"tree\", dot_bf16=False,\n                            **R.TODAY" in src
    assert "build_group_tiles_fused(ids, R.E, 16, lean=True, sorted_ids=True)" in src
    assert "self.A.launch(arm, self.X[li], p, am, row0, rows, grp, order, " "gather_div=R.TOPK, **plan)" in src
    assert "self.A.launch(arm, self.H[li], p, am, row0, rows, grp, None, " "scatter=order, **plan)" in src
    assert "graphs[R.SPACER][0].replay()      # untimed" in src and "graphs = {R.SPACER: capture(" in src
    assert "nodes = capture_kernel_nodes(torch.cuda.current_stream().cuda_stream)" in src
    assert "rot = R.order(R.ARMS, r)" in src and 'for arm in ("T2", "A0p", "ACX0")}' in src
    assert "self.A.set_pad(k, pad)" in src and "occupancy(k.function, BLOCK, k.metadata.shared + p) <= want" in src
    assert len(R.CONFIRM_STEPS) == 70 and R.CONFIRM_STEPS[-1] == 127 and len(R.CONFIRM_STEPS) % len(R.ARMS) == 0
    pos = {a_: [0] * len(R.ARMS) for a_ in R.ARMS}
    pairs = {}
    for r in range(len(R.CONFIRM_STEPS)):
        o = R.order(R.ARMS, r)
        for i, a_ in enumerate(o):
            pos[a_][i] += 1
        for x, y in zip(o, o[1:]):
            pairs[(x, y)] = pairs.get((x, y), 0) + 1
    reps = len(R.CONFIRM_STEPS) // len(R.ARMS)
    assert all(set(v) == {reps} for v in pos.values()), pos
    assert len(pairs) == len(R.ARMS) * (len(R.ARMS) - 1) and set(pairs.values()) == {reps}, pairs

    class _Picks:                                   # a stand-in for the [L, 16, steps, 8] tensor's slicing
        def __init__(self, ids):
            self.ids = ids

        def __getitem__(self, ix):
            return self

        def reshape(self, *_):
            return self

        def tolist(self):
            return self.ids
    sl = slabs(_Picks([1, 2, 3, 5, 17, 33, 127, 64]), 64, 2)
    assert sl["127"]["gu"] == [8, 8] and sl["0"]["gu"] == [1, 1] and sl["3"]["dn"] == [4, 4], sl
    assert sl["15"]["gu"] == [6, 6], sl                   # {1, 2, 3, 5, 1, 1, 15, 0}
    import contextlib
    import io
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            hdr = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            for arm, v in (("ACX", "1,000,000"), ("AC", "20,000")):
                with open(f"ncu_{arm}.csv", "w", encoding="utf-8") as f:
                    f.write("==PROF== noise\n" + hdr + "".join(
                        f'"{i}","_k25_ablate","dram__bytes_read.sum","byte","{v}"\n' for i in range(NCU_COUNT)))
                with open(f"est_{arm}.json", "w", encoding="utf-8") as f:
                    json.dump({"estimate_bytes": 7}, f)
            os.environ.update(NCU_ARMS="ACX AC", NCU_COUNT=str(NCU_COUNT))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ncu_collect()
            got = json.loads(buf.getvalue())
            assert got["dram_read_bytes"] == {"ACX": 1e6 * NCU_COUNT, "AC": 2e4 * NCU_COUNT}, got
            os.environ["NCU_COUNT"] = str(NCU_COUNT + 1)
            try:
                ncu_collect()
                raise AssertionError("a short capture was not refused")
            except SystemExit:
                pass
        finally:
            os.chdir(here)
    # has_mma is read from the SASS, never from the arm: one loop with 8 HMMA (one iteration of today's tile on 4
    # warps), one without an MMA whose load stream is two of A0's iterations (ptxas unrolled it 2x)
    import sass_count as sc
    sc.toolchain = lambda: {"triton": "3.4.0", "ptxas": "12.8.93"}      # no triton needed to count text
    head = "        /*0000*/                   MOV R1, c[0x0][0x28] ;\n"
    mma = head + "".join(f"        /*{0x10 * (i + 1):04x}*/                   HMMA.1688.F32.TF32 R4, R8, R12, R4 ;\n"
                         for i in range(8))
    mma += ("        /*0090*/                   LDGSTS.E.BYPASS.128 [R2], desc[UR4][R6.64] ;\n"
            "        /*00a0*/              @P0 BRA 0x10 ;\n        /*00b0*/                   EXIT ;\n")
    rec = static_from_sass(mma, sc, arch=120, trip=12)
    assert rec["has_mma"] and rec["unroll"] == 1 and rec["unroll_guard"] == "mma" and rec["stream_bytes"] == 2048, rec
    light = (head + "        /*0010*/                   LDGSTS.E.BYPASS.128 [R2], desc[UR4][R6.64] ;\n"
             "        /*0020*/                   LDGSTS.E.BYPASS.128 [R3], desc[UR4][R8.64] ;\n"
             "        /*0030*/                   LOP3.LUT R4, R4, R5, RZ, 0x96, !PT ;\n"
             "        /*0040*/              @P0 BRA 0x10 ;\n        /*0050*/                   EXIT ;\n")
    rec = static_from_sass(light, sc, arch=120, trip=12, bytes_per_iteration=2048)
    assert not rec["has_mma"] and rec["unroll"] == 2 and rec["unroll_guard"] == "bytes", rec
    # NVML: a failed read is UNREAD (None), never 0; a sample straddling a tag change is dropped and counted; the
    # field-value union is read by its type
    import ctypes
    import p157_cuda as cu

    class _NV:
        fail = set()

        def __getattr__(self, name):
            def call(*args):
                if name in self.fail:
                    return 999
                ref = args[-1] if args else None
                obj = getattr(ref, "_obj", None)
                if name == "nvmlDeviceGetClockInfo":
                    obj.value = 2400 if args[1] == 1 else 14001
                elif name == "nvmlDeviceGetPowerUsage":
                    obj.value = 480000
                elif name == "nvmlDeviceGetEnforcedPowerLimit":
                    obj.value = 575000
                elif name == "nvmlDeviceGetFieldValues":
                    obj.valueType, obj.value, obj.nvmlReturn = 1, 470123, 0
                elif name.startswith("nvmlDeviceGetCurrentClocks"):
                    obj.value = 4
                return 0
            return call
    nv = _NV()
    smp = cu.ClockSampler(nv=nv)
    assert smp.error is None, smp.error
    smp.tag = "A1"
    smp.sample_once()
    nv.fail = {"nvmlDeviceGetPowerUsage", "nvmlDeviceGetEnforcedPowerLimit"}
    smp.sample_once()
    read = smp._read

    def flip():
        v = read()
        smp.tag = "A2b"
        return v
    smp._read = flip
    smp.sample_once()
    smm = smp.summary()
    a1 = smm["A1"]
    assert smm["dropped_straddling"] == 1 and a1["n"] == 2 and "A2b" not in smm, smm
    assert a1["sm_mhz_median"] == 2400 and a1["power_w_median"] == 480.0 and a1["enforced_limit_w"] == [575.0], a1
    assert a1["unread"]["power"] == 1 and a1["unread"]["limit"] == 1 and a1["power_instant_w_median"] == 470.123, a1
    assert cu.field_value(0, int.from_bytes(ctypes.c_double(1.5), "little")) == 1.5
    assert cu.field_value(5, 0xFFFFFFFF) == -1 and cu.field_value(3, 1 << 40) == 1 << 40 and cu.field_value(9, 1) is None
    print("p157_bench self-test OK (10 cases)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rr")
    ap.add_argument("--out")
    ap.add_argument("--layers", type=int, default=R.LAYERS)
    ap.add_argument("--reps", type=int, default=len(R.CONFIRM_STEPS))
    ap.add_argument("--shakedown", action="store_true")
    ap.add_argument("--ncu-target", choices=[x for x in R.ARMS if x != "T2"])
    ap.add_argument("--estimate-out")
    ap.add_argument("--ncu-collect", action="store_true", help="gather the runner's Nsight CSVs (NCU_ARMS, NCU_COUNT)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        try:
            self_test()
        except AssertionError as e:
            print(f"p157_bench self-test FAILED: {e}")
            sys.exit(1)
        sys.exit(0)
    if args.ncu_collect:
        sys.exit(ncu_collect())
    if not args.rr:
        ap.error("--rr is required")
    if args.ncu_target:
        sys.exit(ncu_target(args))
    if not args.out:
        ap.error("--out is required")
    sys.exit(main(args))
