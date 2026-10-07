#!/usr/bin/env python3
"""p116_box.py -- lane P116 (bench/p116/PREREG-p116.md; e4b#1313): ONE process per arm or quality phase. Does the
bandwidth-targeted NF4 decode GEMV (grouped-nf4-gemm K33, ``GNF4_GEMV_BW=1``) decode the default ``serve_paged`` server
faster at one request, at no measurable quality cost?

The engine is the shipped default server: ``PagedServeConfig.from_env()`` + ``build_engine(cfg)``, with only the model,
arena and calibration set. The arms differ ONLY in grouped-nf4-gemm's decode GEMV switch, read at call time inside
``gemm_4bit_grouped``'s single-row decode branch:
  B0  ``GNF4_GEMV_BW`` unset (the shipped default: dot-pad at Qwen3's shapes on >= 160-SM parts, the scalar GEMV
      elsewhere) and no plan override;
  B1  ``GNF4_GEMV_BW=1`` with ``GNF4_GEMV_BW_PLAN`` = K33's selected plan per shape (:data:`BW_PLAN`, from
      grouped-nf4-gemm ``kernel/receipts-k33/5090/k33.json``).
The served graph server reaches that branch only at ``T == 1`` (``hot_residency._collapsed_grouping``: singleton groups at
one token, device grouping -- the M-tile GEMM -- above it), so W1 is the subject and W16 a control.

``--mode speed`` (arm from ``P116_ARM``): P109's workloads, timing and token records, imported from ``p109_box.py`` at its
registered bytes -- W16: 16 distinct 512-token prompts at once; W1: row 0 alone; SHORT and LONG new tokens; one warm
pass, then REPS timed passes; p37's slope -- plus ``nf4_grouped.dispatch_counts()`` after the build and after the runs.

``--mode quality`` (phase from ``P116_ARM``: ``B0`` = off, ``B1`` = on): Phase B's teacher-forced instrument
(``p115_quality.measure_phase``, P110's padded bucket arithmetic) on the default server built eager, at ONE window per
pass (``--group 1``): bucket 1, ``T == 1``, the served W1 arithmetic. Off scores R, ``rep``, ``chunk`` and
``mutant_scale``; on scores ON against R's saved log-probs. ``half`` has no meaning at one window and is not run.

    python p116_box.py --mode speed --prompts prompts.json --out arm_B0a.json --tag B0a   (arm from P116_ARM)
    python p116_box.py --mode quality --out quality_off.json --ref-dir work/ref --windows 48 (P116_ARM=B0 / B1)
    python p116_box.py --prompts-only --model ID --revision SHA --out prompts.json
    python p116_box.py --self-test
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p109_box  # noqa: E402  (staged at P109's registered bytes: prompts, run_pass, slope, digest, _mem, _smi)

ARMS = ("B0", "B1")
#: K33's selected plan (BLOCK_N, KC, warps, split-K) per (N, K), from grouped-nf4-gemm kernel/receipts-k33/5090/k33.json:
#: Qwen3-30B-A3B gate_up / down, Granite-3.1-3b-a800m gate_up / down, OLMoE-1B-7B gate_up / down
BW_PLANS = {(1536, 2048): (16, 1024, 4, 1), (2048, 768): (16, 256, 4, 1),
            (1024, 1536): (16, 512, 8, 1), (1536, 512): (16, 256, 8, 1),
            (2048, 2048): (16, 1024, 4, 1), (2048, 1024): (16, 256, 4, 1)}
BW_PLAN = ";".join(f"{n},{k}={','.join(str(x) for x in p)}" for (n, k), p in BW_PLANS.items())
KNOBS = ("GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE")
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")
QUALITY_ARMS = {"B0": ("R", "rep", "chunk", "mutant_scale"), "B1": ("ON",)}


def arm_env_ok(arm: str, env):
    """``(ok, why)``: B0 has the switch unset (or ``0``) and no plan or decode override; B1 has ``GNF4_GEMV_BW=1``,
    ``GNF4_GEMV_BW_PLAN`` = :data:`BW_PLAN` and no decode override (``prmt32``, the compiled default)."""
    val = {k: (env.get(k) or "").strip() for k in KNOBS}
    if arm == "B0":
        want = {"GNF4_GEMV_BW": ("", "0"), "GNF4_GEMV_BW_PLAN": ("",), "GNF4_GEMV_BW_DECODE": ("",)}
    elif arm == "B1":
        want = {"GNF4_GEMV_BW": ("1",), "GNF4_GEMV_BW_PLAN": (BW_PLAN,), "GNF4_GEMV_BW_DECODE": ("",)}
    else:
        return False, f"unknown arm {arm!r}"
    bad = {k: v for k, v in val.items() if v not in want[k]}
    return (not bad), f"{arm} with {bad}"


def _arm():
    arm = os.environ.get("P116_ARM", "")
    if arm not in ARMS:
        raise SystemExit(f"REFUSED: P116_ARM={arm!r}, expected one of {ARMS}")
    ok, why = arm_env_ok(arm, os.environ)
    if not ok:
        raise SystemExit(f"REFUSED: arm {arm}: {why}")
    return arm


def _counts():
    import nf4_grouped
    return dict(nf4_grouped.dispatch_counts())


def _delta(after, before):
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


def speed_main(a) -> int:
    arm = _arm()
    pf = json.load(open(a.prompts))
    rows = pf["rows"]
    if p109_box.digest(rows) != pf["prompts_sha256"] or len(rows) != p109_box.ROWS:
        raise SystemExit("REFUSED: prompts.json does not match its own digest")
    import torch
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if not cfg.graphs or (cfg.max_seqs, cfg.placement, tuple(cfg.buckets)) != (16, "all-vram", (1, 2, 4, 8, 16)):
        raise SystemExit(f"REFUSED: not the default graph server: graphs={cfg.graphs} max_seqs={cfg.max_seqs} "
                         f"placement={cfg.placement} buckets={cfg.buckets}")
    c0 = _counts()
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    load_s = time.perf_counter() - t0
    c_build = _counts()
    info = parts.info
    rec = {"mode": "speed", "arm": arm, "tag": a.tag, "e4b_sha": os.environ.get("E4B_SHA"),
           "gnf4_sha": os.environ.get("GNF4_SHA"), "model": cfg.model, "revision": cfg.revision,
           "knobs": {k: os.environ.get(k) for k in KNOBS}, "torch": torch.__version__, "load_s": round(load_s, 2),
           "fusions": {k: info.get(k) for k in CENSUS_KEYS}, "fusion_modes": info.get("fusion_modes"),
           "levers_env": info.get("levers_env"),
           "graph_status": ({str(k): v for k, v in info["graph_status"].items()} if info.get("graph_status") else None),
           "grouping": info.get("grouping"), "dispatch_build": _delta(c_build, c0),
           "prompts_sha256": pf["prompts_sha256"], "short": a.short, "long": a.long, "reps": a.reps,
           "mem_after_load": p109_box._mem(torch), "workloads": {}, "status": "ok"}
    for wname, b in p109_box.WORKLOADS.items():
        wrows = rows[:b]
        walls, digests, last = {}, {}, {}
        cw = _counts()
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
                                   "dispatch": _delta(_counts(), cw), **s}
        print(f"P116_W {arm}/{a.tag} {wname} B={b} decode_tok_s={s['decode_tok_s']} median={s['decode_tok_s_median']} "
              f"walls={walls}", flush=True)
    gs = getattr(parts.runner, "graph_stats", None)
    rec["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None
    rec["dispatch_total"] = _delta(_counts(), c0)
    rec["mem_after_runs"] = p109_box._mem(torch)
    rec["nvidia_smi"] = p109_box._smi()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P116_ARM " + json.dumps({"arm": arm, "tag": a.tag, "dispatch_build": rec["dispatch_build"],
                                    "load_s": rec["load_s"], "graph_status": rec["graph_status"],
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
        raise SystemExit(f"REFUSED: the quality read builds the default server eager at all-vram (graphs={cfg.graphs})")
    if a.group != 1:
        raise SystemExit(f"REFUSED: --group {a.group}; P116 reads quality at one window per pass (T == 1)")
    counters = q.KernelCounters().install()                          # BEFORE build_engine, as Phase B
    c0 = _counts()
    t0 = time.time()
    parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    load_s = time.time() - t0
    fwd = q.ForwardCounter(model)
    windows = {t: q.LOADERS[t](parts.tokenizer, a.windows, a.prompt, a.cont) for t in q.TEXTS}
    c1 = _counts()
    phase = "off" if arm == "B0" else "on"
    rec = q.measure_phase(model, windows, phase=phase, prompt=a.prompt, cont=a.cont, chunk=a.chunk,
                          floor_chunk=a.floor_chunk, group=1, device=cfg.device, ref_dir=a.ref_dir,
                          counters=counters, fwd=fwd, arms=QUALITY_ARMS[arm])
    info = parts.info
    rec = {"mode": "quality", "arm": arm, "model": cfg.model, "revision": cfg.revision,
           "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
           "knobs": {k: os.environ.get(k) for k in KNOBS}, "transformers": transformers.__version__,
           "load_s": round(load_s, 1), "fusions": {k: info.get(k) for k in CENSUS_KEYS},
           "fusion_modes": info.get("fusion_modes"), "dispatch_build": _delta(c1, c0),
           "dispatch_measure": _delta(_counts(), c1), **rec, "gpu": torch.cuda.get_device_name(0),
           "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "status": "ok"}
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print(f"P116_QUALITY {arm} windows={rec['windows']} dispatch={rec['dispatch_measure']} load {load_s:.0f}s "
          f"mem {rec['max_mem_gb']} GB {rec['seconds']} s", flush=True)
    return 0


def self_test() -> int:
    b1 = {"GNF4_GEMV_BW": "1", "GNF4_GEMV_BW_PLAN": BW_PLAN}
    ok = [ARMS == ("B0", "B1"), p109_box.WORKLOADS == {"W16": 16, "W1": 1}, p109_box.self_test() == 0,
          arm_env_ok("B0", {})[0], arm_env_ok("B0", {"GNF4_GEMV_BW": "0"})[0],
          not arm_env_ok("B0", {"GNF4_GEMV_BW": "1"})[0], not arm_env_ok("B0", {"GNF4_GEMV_BW_PLAN": BW_PLAN})[0],
          arm_env_ok("B1", b1)[0], not arm_env_ok("B1", {"GNF4_GEMV_BW": "1"})[0],
          not arm_env_ok("B1", {**b1, "GNF4_GEMV_BW_DECODE": "tree"})[0], not arm_env_ok("B1", {**b1, "GNF4_GEMV_BW": "auto"})[0],
          not arm_env_ok("B2", b1)[0],
          BW_PLAN.startswith("1536,2048=16,1024,4,1;2048,768=16,256,4,1;") and BW_PLAN.count(";") == 5,
          QUALITY_ARMS == {"B0": ("R", "rep", "chunk", "mutant_scale"), "B1": ("ON",)}]
    print(f"p116_box self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
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
    p.add_argument("--windows", type=int, default=48)
    p.add_argument("--group", type=int, default=1)
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
        return quality_main(a)
    raise SystemExit("REFUSED: --mode speed|quality, --prompts-only or --self-test")


if __name__ == "__main__":
    sys.exit(main())
