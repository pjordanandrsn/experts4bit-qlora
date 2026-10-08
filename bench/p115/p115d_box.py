#!/usr/bin/env python3
"""p115d_box.py -- lane P115 Phase D (bench/p115/PREREG-p115.md, Amendment 3; e4b#1313): ONE process per arm or phase.
Do the B=1 fused stack and P116's bandwidth GEMV, each read alone, compute nothing grossly wrong together, and what does
the stack gain on top of the GEMV?

The engine is the shipped default server (``PagedServeConfig.from_env()`` + ``build_engine(cfg)``) at grouped-nf4-gemm
0.43.0, where ``GNF4_GEMV_BW=auto`` is the default: at Qwen3-30B-A3B's two expert shapes on >= 160-SM parts the decode
GEMV is the bandwidth route (P116). ``max_seqs`` 16 is named by the runner. The arms differ ONLY in the four fusion knobs:
  D0  all four ``0`` (the unfused server);
  D1  all four ``auto`` (the candidate default on an allowlisted family; on Qwen3-30B-A3B every knob engages).

``--mode sane`` (the gate the maintainer asked for in #1318's review): Phase B's instrument (``p115_quality.measure_phase``)
on the default server built eager, at ONE window per pass (``group`` 1): bucket 1, ``T == 1``, the served W1 arithmetic,
the only place the bandwidth GEMV and the folds meet (``hot_residency._collapsed_grouping`` sends ``T > 1`` to the
M-tile). Phase C's SANE windows and gate (12 wikitext windows x 128 positions; |bias| <= 0.02 nats, argmax agreement >=
0.95); D0 scores R and saves its log-probs, D1 scores ON against them. grouped-nf4-gemm's dispatch tally is recorded over
the measurement, so the read proves the bandwidth GEMV ran in both phases.

``--mode speed`` (reported, not ruled): P109's workloads, timing and token records at P109's registered bytes, with the
tally after the build and after the runs, as P116's box records them.

    python p115d_box.py --mode speed --prompts prompts.json --out arm_D0a.json --tag D0a   (arm from P115D_ARM)
    python p115d_box.py --mode sane --out sane_off.json --ref-dir work/ref                  (P115D_ARM=D0 / D1)
    python p115d_box.py --prompts-only --model ID --revision SHA --out prompts.json
    python p115d_box.py --self-test
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p109_box  # noqa: E402  (staged at P109's registered bytes: prompts, run_pass, slope, digest, _mem, _smi)

ARMS = ("D0", "D1")
KNOBS = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
ARM_VALUE = {"D0": "0", "D1": "auto"}
GEMV_KNOBS = ("GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE")
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")
SANE_PHASE_ARMS = {"D0": ("R",), "D1": ("ON",)}


def arm_env_ok(arm: str, env):
    """``(ok, why)``: the four fusion knobs carry the arm's value; the decode GEMV knobs are unset (its default)."""
    if arm not in ARMS:
        return False, f"unknown arm {arm!r}"
    bad = {k: env.get(k) for k in KNOBS if (env.get(k) or "").strip().lower() != ARM_VALUE[arm]}
    bad.update({k: env.get(k) for k in GEMV_KNOBS if (env.get(k) or "").strip()})
    return (not bad), f"{arm} with {bad}"


def _arm():
    arm = os.environ.get("P115D_ARM", "")
    ok, why = arm_env_ok(arm, os.environ)
    if not ok:
        raise SystemExit(f"REFUSED: P115D_ARM={arm!r}: {why}")
    return arm


def _counts():
    import nf4_grouped
    return dict(nf4_grouped.dispatch_counts())


def _delta(after, before):
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


def _common(cfg, parts, arm, load_s):
    import torch
    info = parts.info
    return {"arm": arm, "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
            "model": cfg.model, "revision": cfg.revision, "knobs": {k: os.environ.get(k) for k in KNOBS + GEMV_KNOBS},
            "torch": torch.__version__, "load_s": round(load_s, 2), "fusions": {k: info.get(k) for k in CENSUS_KEYS},
            "fusion_modes": info.get("fusion_modes"), "model_type": info.get("model_type"),
            "graph_status": ({str(k): v for k, v in info["graph_status"].items()} if info.get("graph_status") else None),
            "grouping": info.get("grouping"), "max_seqs": cfg.max_seqs, "buckets": list(cfg.buckets)}


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
        raise SystemExit(f"REFUSED: not the default graph server at 16 slots: graphs={cfg.graphs} max_seqs={cfg.max_seqs} "
                         f"placement={cfg.placement} buckets={cfg.buckets}")
    c0 = _counts()
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    load_s = time.perf_counter() - t0
    c_build = _counts()
    rec = {"mode": "speed", "tag": a.tag, **_common(cfg, parts, arm, load_s), "dispatch_build": _delta(c_build, c0),
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
        print(f"P115D_W {arm}/{a.tag} {wname} B={b} decode_tok_s={s['decode_tok_s']} walls={walls}", flush=True)
    gs = getattr(parts.runner, "graph_stats", None)
    rec["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None
    rec["dispatch_total"] = _delta(_counts(), c0)
    rec["mem_after_runs"] = p109_box._mem(torch)
    rec["nvidia_smi"] = p109_box._smi()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P115D_ARM " + json.dumps({"arm": arm, "tag": a.tag, "fusions": rec["fusions"], "load_s": rec["load_s"],
                                     "W16": rec["workloads"]["W16"]["decode_tok_s"],
                                     "W1": rec["workloads"]["W1"]["decode_tok_s"]}, default=str), flush=True)
    return 0


def sane_main(a) -> int:
    arm = _arm()
    import torch
    import transformers

    import p115_quality as q
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if cfg.graphs or cfg.placement != "all-vram":
        raise SystemExit(f"REFUSED: SANE builds the default server eager at all-vram (graphs={cfg.graphs})")
    counters = q.KernelCounters().install()                          # before build_engine, as Phase B
    c0 = _counts()
    t0 = time.time()
    parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    load_s = time.time() - t0
    fwd = q.ForwardCounter(model)
    windows = {"wikitext": q.LOADERS["wikitext"](parts.tokenizer, a.windows, a.prompt, a.cont)}
    c1 = _counts()
    rec = q.measure_phase(model, windows, phase="off" if arm == "D0" else "on", prompt=a.prompt, cont=a.cont,
                          chunk=a.chunk, floor_chunk=a.chunk, group=1, device=cfg.device, ref_dir=a.ref_dir,
                          counters=counters, fwd=fwd, arms=SANE_PHASE_ARMS[arm])
    rec = {"mode": "sane", **_common(cfg, parts, arm, load_s), "transformers": transformers.__version__,
           "dispatch_build": _delta(c1, c0), "dispatch_measure": _delta(_counts(), c1), **rec,
           "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "status": "ok"}
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print(f"P115D_SANE {arm} windows={rec['windows']} census={rec['fusions']} dispatch={rec['dispatch_measure']} "
          f"{rec['seconds']} s", flush=True)
    return 0


def self_test() -> int:
    d1 = {k: "auto" for k in KNOBS}
    ok = [ARMS == ("D0", "D1"), p109_box.WORKLOADS == {"W16": 16, "W1": 1}, p109_box.self_test() == 0,
          arm_env_ok("D0", {k: "0" for k in KNOBS})[0], arm_env_ok("D1", d1)[0], arm_env_ok("D1", {k: " AUTO " for k in KNOBS})[0],
          not arm_env_ok("D0", {})[0],                                    # unset is not the registered 0: named explicitly
          not arm_env_ok("D1", {**d1, "E4B_FUSE_T1_GLUE": "1"})[0],
          not arm_env_ok("D1", {**d1, "GNF4_GEMV_BW": "1"})[0],          # the GEMV at its default, never forced
          not arm_env_ok("D1", {**d1, "GNF4_GEMV_BW_PLAN": "1536,2048=16,1024,4,1"})[0],
          not arm_env_ok("D2", d1)[0],
          SANE_PHASE_ARMS == {"D0": ("R",), "D1": ("ON",)}]
    print(f"p115d_box self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--prompts-only", action="store_true")
    p.add_argument("--mode", choices=("speed", "sane"))
    p.add_argument("--model")
    p.add_argument("--revision", default="")
    p.add_argument("--prompts")
    p.add_argument("--out")
    p.add_argument("--tag", default="")
    p.add_argument("--short", type=int, default=32)
    p.add_argument("--long", type=int, default=160)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--ref-dir")
    p.add_argument("--windows", type=int, default=12)
    p.add_argument("--prompt", type=int, default=512)
    p.add_argument("--cont", type=int, default=128)
    p.add_argument("--chunk", type=int, default=512)
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.prompts_only:
        return p109_box.prompts_main(a)
    if a.mode == "speed":
        return speed_main(a)
    if a.mode == "sane":
        return sane_main(a)
    raise SystemExit("REFUSED: --mode speed|sane, --prompts-only or --self-test")


if __name__ == "__main__":
    sys.exit(main())
