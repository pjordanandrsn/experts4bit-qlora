#!/usr/bin/env python3
"""p121_box.py -- lane P121 (bench/p121/PREREG-p121.md; e4b#846): ONE process per arm or phase. Does K25
(``E4B_NF4_GROUPED_SMALLM=auto``, the default since P96) cost quality or speed against the served NF4 M-tile (``0``) on
Qwen3-30B-A3B at the served W16 step?

K25 has no family gate: it takes every device-grouped NF4 decode step with T > 1 and at most 256 routed rows, so on
Qwen3-30B-A3B it runs at W16 (16 rows x top-8 = 128 routed rows). Its licence (P93's speed, P96's quality) read
Granite and OLMoE only. ``T == 1`` stays on the decode GEMV under ``auto``, so W1 runs the same kernels in both arms.

The engine is the shipped default server (``PagedServeConfig.from_env()`` + ``build_engine(cfg)``), ``max_seqs`` 16
named by the runner. The arms differ ONLY in ``E4B_NF4_GROUPED_SMALLM``:
  K0  ``0``     the NF4 M-tile (``nf4_grouped.gemm_4bit_grouped_captured``, TF32 on the fp32 dequant);
  K1  ``auto``  K25 (``nf4_smallm.gemm_nf4_grouped_smallm``) at <= 256 routed rows, the shipped default.

``RouteCounter`` wraps both kernels in their modules BEFORE ``build_engine`` (e4b reads them from the modules at call
time), so every record carries which kernel ran: K25 calls, and M-tile calls at <= 256 and > 256 rows.

``--mode speed``: P109's workloads, timing and token records at P109's registered bytes (W16: 16 prompts at once; W1:
row 0 alone), the route tally after the build (the bucket graphs' captures) and over each workload.

Every arm and phase records its peak allocated and reserved memory (``_peak``), after the build and at its end: reported,
never gated (the maintainer's addition; K25 and the M-tile can differ in workspace).

``--mode quality``: P115 Phase B's instrument (``p115_quality.measure_phase`` at its registered bytes) on the default
server built eager, at ``--group`` 16 windows a pass: every decode step 16 rows, 128 routed rows, device grouping, lean
glue and padding as served. K0 scores R, its floor (``rep``, ``half``, ``chunk``) and ``mutant_scale``, and saves R's
log-probs; K1 scores ON against them. Both texts (wikitext, c4val1).

The continuity row (reported, no bar; the maintainer's ruling): P96's two arms through the same instrument at
``--group`` 1 (T == 1, bucket 1) on 12 wikitext windows -- Cm ``0`` with ``E4B_NF4_T1_DEVICE_GROUPING=1`` (the M-tile at
T == 1) scores R, Ct ``1`` with T == 1 device grouping ``0`` (K25 at T == 1) scores ON against it.

    python p121_box.py --mode speed --prompts prompts.json --out arm_K0a.json --tag K0a      (arm from P121_ARM)
    python p121_box.py --mode quality --out quality_off.json --ref-dir work/ref               (P121_ARM=K0 / K1)
    python p121_box.py --prompts-only --model ID --revision SHA --out prompts.json
    python p121_box.py --self-test
"""
import argparse
import functools
import importlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p109_box  # noqa: E402  (staged at P109's registered bytes: prompts, run_pass, slope, digest, _mem, _smi)

ARMS = ("K0", "K1", "Cm", "Ct")
KNOB = "E4B_NF4_GROUPED_SMALLM"
T1 = "E4B_NF4_T1_DEVICE_GROUPING"
ARM_VALUE = {"K0": "0", "K1": "auto"}
# each arm's knobs, named explicitly; Cm / Ct are P96's m and t arms (its run.sh ARM_M / ARM_T), the continuity row
ARM_ENV = {"K0": {KNOB: "0"}, "K1": {KNOB: "auto"}, "Cm": {KNOB: "0", T1: "1"}, "Ct": {KNOB: "1", T1: "0"}}
# held at their defaults: the fusion knobs (0 since P115 Phase C), the decode GEMV, K25's T == 1 opt-in, the lean glue
UNSET = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI", "GNF4_GEMV_BW",
         "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE", "E4B_NF4_T1_DEVICE_GROUPING", "E4B_INT4_LEAN_GLUE")
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")
QUALITY_ARMS = {"K0": ("R", "rep", "half", "chunk", "mutant_scale"), "K1": ("ON",), "Cm": ("R",), "Ct": ("ON",)}
OFF_PHASE = ("K0", "Cm")
ROUTES = (("nf4_smallm", "gemm_nf4_grouped_smallm", "k25"), ("nf4_grouped", "gemm_4bit_grouped_captured", "mtile"))
SMALL = 256           # K25's row cap: an M-tile call at <= 256 rows is one K25 would take under auto


def arm_env_ok(arm: str, env):
    """``(ok, why)``: the arm's knobs carry its values, named explicitly; every other route knob is unset."""
    if arm not in ARMS:
        return False, f"unknown arm {arm!r}"
    want = ARM_ENV[arm]
    bad = {k: env.get(k) for k, v in want.items() if (env.get(k) or "").strip().lower() != v}
    bad.update({k: env.get(k) for k in UNSET if k not in want and (env.get(k) or "").strip()})
    return (not bad), f"{arm} with {bad}"


def _arm():
    arm = os.environ.get("P121_ARM", "")
    ok, why = arm_env_ok(arm, os.environ)
    if not ok:
        raise SystemExit(f"REFUSED: P121_ARM={arm!r}: {why}")
    return arm


class RouteCounter:
    """Counts K25 and M-tile calls by wrapping them in their modules (``functools.wraps`` keeps their signatures).
    M-tile calls split at ``SMALL`` routed rows (the first argument's rows): prefill chunks route more and take the
    M-tile in both arms."""

    def __init__(self):
        self.counts = {"k25": 0, "mtile_small": 0, "mtile_large": 0}
        self.wrapped = []

    def install(self, modules=None):
        for modname, fname, key in ROUTES:
            mod = (modules or {}).get(modname) or importlib.import_module(modname)
            orig = getattr(mod, fname, None)
            if orig is None or getattr(orig, "_p121_counted", False):
                continue

            def make(_k, _o):
                @functools.wraps(_o)
                def wrapper(*a, **kw):
                    if _k == "k25":
                        self.counts["k25"] += 1
                    else:
                        rows = int(a[0].shape[0]) if a and hasattr(a[0], "shape") else 0
                        self.counts["mtile_small" if rows <= SMALL else "mtile_large"] += 1
                    return _o(*a, **kw)
                return wrapper

            w = make(key, orig)
            w._p121_counted = True
            setattr(mod, fname, w)
            self.wrapped.append(f"{modname}.{fname}")
        return self

    def snapshot(self) -> dict:
        return dict(self.counts)


def _delta(after, before):
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


def _gemv_counts():
    try:
        import nf4_grouped
        return dict(nf4_grouped.dispatch_counts())
    except (ImportError, AttributeError):
        return {}


def _peak(torch) -> dict:
    """Peak and current allocated / reserved bytes on the device, since the process started."""
    return {"max_allocated": int(torch.cuda.max_memory_allocated()), "max_reserved": int(torch.cuda.max_memory_reserved()),
            "allocated": int(torch.cuda.memory_allocated()), "reserved": int(torch.cuda.memory_reserved())}


def _gib(b) -> float:
    return round(b / 2**30, 3)


def _common(cfg, parts, arm, load_s):
    import torch
    info = parts.info
    return {"arm": arm, "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
            "model": cfg.model, "revision": cfg.revision, "knobs": {k: os.environ.get(k) for k in (KNOB,) + UNSET},
            "torch": torch.__version__, "load_s": round(load_s, 2), "fusions": {k: info.get(k) for k in CENSUS_KEYS},
            "model_type": info.get("model_type"), "moe_layers": info.get("moe_layers"), "top_k": info.get("top_k"),
            "graph_status": ({str(k): v for k, v in info["graph_status"].items()} if info.get("graph_status") else None),
            "grouping": info.get("grouping"), "max_seqs": cfg.max_seqs, "buckets": list(cfg.buckets),
            "prove": os.environ.get("P121_PROVE", "0") == "1"}


def speed_main(a) -> int:
    arm = _arm()
    if arm not in ARM_VALUE:
        raise SystemExit(f"REFUSED: the speed arms are K0 and K1, not {arm}")
    pf = json.load(open(a.prompts))
    rows = pf["rows"]
    if p109_box.digest(rows) != pf["prompts_sha256"] or len(rows) != p109_box.ROWS:
        raise SystemExit("REFUSED: prompts.json does not match its own digest")
    import torch
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if not cfg.graphs or (cfg.max_seqs, cfg.placement, tuple(cfg.buckets)) != (16, "all-vram", (1, 2, 4, 8, 16)):
        raise SystemExit(f"REFUSED: not the default graph server at 16 slots: graphs={cfg.graphs} max_seqs={cfg.max_seqs} "
                         f"placement={cfg.placement} buckets={cfg.buckets}")
    rc = RouteCounter().install()                    # before build_engine: the bucket graphs capture their kernels
    r0, g0 = rc.snapshot(), _gemv_counts()
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    load_s = time.perf_counter() - t0
    rec = {"mode": "speed", "tag": a.tag, **_common(cfg, parts, arm, load_s), "routes_wrapped": rc.wrapped,
           "route_build": _delta(rc.snapshot(), r0), "gemv_build": _delta(_gemv_counts(), g0), "peak_build": _peak(torch),
           "prompts_sha256": pf["prompts_sha256"], "short": a.short, "long": a.long, "reps": a.reps,
           "mem_after_load": p109_box._mem(torch), "workloads": {}, "status": "ok"}
    for wname, b in p109_box.WORKLOADS.items():
        wrows = rows[:b]
        walls, digests, last = {}, {}, {}
        rw, gw = rc.snapshot(), _gemv_counts()
        for n in (a.short, a.long):
            p109_box.run_pass(parts, torch, wrows, n)                      # warm, untimed
            ws, ds = [], []
            for _ in range(a.reps):
                wall, _steps, toks = p109_box.run_pass(parts, torch, wrows, n)
                ws.append(round(wall, 5))
                ds.append(p109_box.digest(toks))
                last[str(n)] = toks
            walls[str(n)], digests[str(n)] = ws, ds
        s = p109_box.slope(walls[str(a.short)], walls[str(a.long)], b, a.short, a.long)
        rec["workloads"][wname] = {"batch": b, "walls": walls, "rep_digests": digests, "tokens": last,
                                   "routes": _delta(rc.snapshot(), rw), "gemv": _delta(_gemv_counts(), gw), **s}
        print(f"P121_W {arm}/{a.tag} {wname} B={b} decode_tok_s={s['decode_tok_s']} walls={walls}", flush=True)
    gs = getattr(parts.runner, "graph_stats", None)
    rec["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None
    rec["route_total"] = _delta(rc.snapshot(), r0)
    rec["mem_after_runs"] = p109_box._mem(torch)
    rec["peak"] = _peak(torch)
    rec["nvidia_smi"] = p109_box._smi()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P121_ARM " + json.dumps({"arm": arm, "tag": a.tag, "route_build": rec["route_build"], "load_s": rec["load_s"],
                                    "peak_gib": [_gib(rec["peak"]["max_allocated"]), _gib(rec["peak"]["max_reserved"])],
                                    "W16": rec["workloads"]["W16"]["decode_tok_s"],
                                    "W1": rec["workloads"]["W1"]["decode_tok_s"]}, default=str), flush=True)
    return 0


def quality_main(a) -> int:
    arm = _arm()
    import torch
    import transformers

    import p115_quality as q
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if cfg.graphs or cfg.placement != "all-vram":
        raise SystemExit(f"REFUSED: the quality phase builds the default server eager at all-vram (graphs={cfg.graphs})")
    rc = RouteCounter().install()
    t0 = time.time()
    parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    load_s = time.time() - t0
    peak_build = _peak(torch)
    fwd = q.ForwardCounter(model)
    texts = tuple(t for t in a.texts.split(",") if t)
    windows = {t: q.LOADERS[t](parts.tokenizer, a.windows, a.prompt, a.cont) for t in texts}
    r1 = rc.snapshot()
    rec = q.measure_phase(model, windows, phase="off" if arm in OFF_PHASE else "on", prompt=a.prompt, cont=a.cont,
                          chunk=a.chunk, floor_chunk=a.floor_chunk, group=a.group, device=cfg.device,
                          ref_dir=a.ref_dir, counters=rc, fwd=fwd, arms=QUALITY_ARMS[arm])
    rec = {"mode": "quality", **_common(cfg, parts, arm, load_s), "transformers": transformers.__version__,
           "routes_wrapped": rc.wrapped, "route_measure": _delta(rc.snapshot(), r1), **rec, "peak_build": peak_build,
           "peak": _peak(torch), "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "status": "ok"}
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print(f"P121_QUALITY {arm} windows={rec['windows']} group={rec['group']} routes={rec['route_measure']} "
          f"peak {_gib(rec['peak']['max_allocated'])} / {_gib(rec['peak']['max_reserved'])} GiB "
          f"load {load_s:.0f}s {rec['seconds']} s", flush=True)
    return 0


def self_test() -> int:
    import types

    k1 = {KNOB: "auto"}
    ok = [ARMS == ("K0", "K1", "Cm", "Ct"), p109_box.WORKLOADS == {"W16": 16, "W1": 1}, p109_box.self_test() == 0,
          arm_env_ok("K0", {KNOB: "0"})[0], arm_env_ok("K1", k1)[0], arm_env_ok("K1", {KNOB: " AUTO "})[0],
          not arm_env_ok("K1", {})[0],                                    # unset is the default, but named explicitly
          not arm_env_ok("K1", {KNOB: "1"})[0],                           # 1 would send T == 1 to K25 too
          not arm_env_ok("K1", {**k1, "E4B_NF4_T1_DEVICE_GROUPING": "1"})[0],
          not arm_env_ok("K0", {KNOB: "0", "E4B_FUSE_T1_GLUE": "0"})[0],  # every other route knob unset
          not arm_env_ok("K2", k1)[0], QUALITY_ARMS["K1"] == ("ON",),
          arm_env_ok("Cm", {KNOB: "0", T1: "1"})[0], arm_env_ok("Ct", {KNOB: "1", T1: "0"})[0],
          not arm_env_ok("Ct", {KNOB: "1"})[0], not arm_env_ok("Cm", {KNOB: "0"})[0],   # P96's arms, both knobs named
          QUALITY_ARMS["Cm"] == ("R",) and QUALITY_ARMS["Ct"] == ("ON",)]

    class _T:
        def __init__(self, n):
            self.shape = (n, 8)
    sm, ng = types.ModuleType("nf4_smallm_st"), types.ModuleType("nf4_grouped_st")

    def gemm_nf4_grouped_smallm(x, packed, absmax, *a, block_n=32, **kw):
        return "k25"

    def gemm_4bit_grouped_captured(x, packed, absmax, *a):
        return "mtile"
    sm.gemm_nf4_grouped_smallm, ng.gemm_4bit_grouped_captured = gemm_nf4_grouped_smallm, gemm_4bit_grouped_captured
    rc = RouteCounter().install({"nf4_smallm": sm, "nf4_grouped": ng})
    out = [sm.gemm_nf4_grouped_smallm(_T(16), 0, 0), ng.gemm_4bit_grouped_captured(_T(128), 0, 0),
           ng.gemm_4bit_grouped_captured(_T(4096), 0, 0)]
    ok.append(out == ["k25", "mtile", "mtile"] and rc.snapshot() == {"k25": 1, "mtile_small": 1, "mtile_large": 1})
    ok.append("block_n" in __import__("inspect").signature(sm.gemm_nf4_grouped_smallm).parameters)
    RouteCounter().install({"nf4_smallm": sm, "nf4_grouped": ng})                   # never wrapped twice
    sm.gemm_nf4_grouped_smallm(_T(16), 0, 0)
    ok.append(rc.snapshot()["k25"] == 2)
    ok.append(_gib(3 * 2**30 + 2**29) == 3.5)
    print(f"p121_box self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--prompts-only", action="store_true")
    p.add_argument("--mode", choices=("speed", "quality"))
    p.add_argument("--model")
    p.add_argument("--revision", default="")
    p.add_argument("--prompts")
    p.add_argument("--out")
    p.add_argument("--tag", default="")
    p.add_argument("--short", type=int, default=32)
    p.add_argument("--long", type=int, default=160)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--ref-dir")
    p.add_argument("--texts", default="wikitext,c4val1")
    p.add_argument("--windows", type=int, default=48)
    p.add_argument("--group", type=int, default=16)
    p.add_argument("--prompt", type=int, default=512)
    p.add_argument("--cont", type=int, default=128)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--floor-chunk", type=int, default=256)
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.prompts_only:
        return p109_box.prompts_main(a)
    if a.mode == "speed":
        return speed_main(a)
    if a.mode == "quality":
        if not (a.out and a.ref_dir):
            raise SystemExit("REFUSED: --out and --ref-dir are required")
        return quality_main(a)
    raise SystemExit("REFUSED: --mode speed|quality, --prompts-only or --self-test")


if __name__ == "__main__":
    sys.exit(main())
