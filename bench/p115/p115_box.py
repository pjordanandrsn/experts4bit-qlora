#!/usr/bin/env python3
"""p115_box.py -- lane P115 (bench/p115/PREREG-p115.md; e4b#1313), the SPEED phase: ONE arm in its own process. Does the
registered B=1 fused stack decode the default ``serve_paged`` server faster?

The engine is the shipped default server: ``PagedServeConfig.from_env()`` + ``build_engine(cfg)``, with only the model,
arena and calibration set; it captures bucketed decode graphs (``E4B_PAGED_GRAPHS=auto``). The arms differ ONLY in the
four fusion knobs, which ``build_engine`` reads at one assembly point (``serve_paged._apply_fusions``):
  F0  all four unset (the shipped default);
  F1  ``E4B_PAGED_FUSE_QKV=1 E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1`` -- the registered B=1 fused
      stack (P54 / P58 / P88). Under the proving run (``P115_PROVE=1``, Granite) the three folds only: Granite has no
      Qwen3-MoE attention, so ``E4B_PAGED_FUSE_QKV=1`` would refuse a vacuous fusion there.

The workloads, timing and token records are P109's, imported from ``p109_box.py`` at its registered bytes:
- W16: 16 distinct 512-token prompts added at once; W1: row 0 alone.
- SHORT and LONG new tokens; one warm pass, then REPS timed passes; p37's slope.
- Every timed pass's token digest; the last pass's tokens per row.

    python p115_box.py --prompts prompts.json --out arm_F0a.json --tag F0a       (arm from P115_ARM; engine from env)
    python p115_box.py --prompts-only --model ID --revision SHA --out prompts.json
    python p115_box.py --self-test
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p109_box  # noqa: E402  (staged at P109's registered bytes: prompts, run_pass, slope, digest, _mem, _smi)

ARMS = ("F0", "F1")
FOLDS = ("E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
KNOBS = ("E4B_PAGED_FUSE_QKV",) + FOLDS
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")


def arm_env_ok(arm: str, env, prove: bool):
    """``(ok, why)``: F0 has every knob unset (or ``0``); F1 has the three folds at ``1`` and ``E4B_PAGED_FUSE_QKV=1``
    (unset under the proving run)."""
    val = {k: (env.get(k) or "").strip() for k in KNOBS}
    if arm == "F0":
        bad = {k: v for k, v in val.items() if v not in ("", "0")}
        return (not bad), f"F0 with {bad}"
    if arm == "F1":
        want = {k: "1" for k in FOLDS}
        want["E4B_PAGED_FUSE_QKV"] = "" if prove else "1"
        bad = {k: v for k, v in val.items() if v != want[k]}
        return (not bad), f"F1 with {bad} (want {want})"
    return False, f"unknown arm {arm!r}"


def arm_main(a) -> int:
    arm = os.environ.get("P115_ARM", "")
    if arm not in ARMS:
        raise SystemExit(f"REFUSED: P115_ARM={arm!r}, expected one of {ARMS}")
    prove = os.environ.get("P115_PROVE", "0") == "1"
    ok, why = arm_env_ok(arm, os.environ, prove)
    if not ok:
        raise SystemExit(f"REFUSED: arm {arm}: {why}")
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
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    load_s = time.perf_counter() - t0
    info = parts.info
    rec = {"arm": arm, "tag": a.tag, "prove": prove, "e4b_sha": os.environ.get("E4B_SHA"),
           "gnf4_sha": os.environ.get("GNF4_SHA"), "model": cfg.model, "revision": cfg.revision,
           "torch": torch.__version__, "load_s": round(load_s, 2), "fuse_qkv": bool(cfg.fuse_qkv),
           "fusions": {k: info.get(k) for k in CENSUS_KEYS}, "levers_env": info.get("levers_env"),
           "graph_status": ({str(k): v for k, v in info["graph_status"].items()} if info.get("graph_status") else None),
           "grouping": info.get("grouping"),
           "prompts_sha256": pf["prompts_sha256"], "short": a.short, "long": a.long, "reps": a.reps,
           "mem_after_load": p109_box._mem(torch), "workloads": {}, "status": "ok"}
    for wname, b in p109_box.WORKLOADS.items():
        wrows = rows[:b]
        walls, digests, last = {}, {}, {}
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
        rec["workloads"][wname] = {"batch": b, "walls": walls, "rep_digests": digests, "tokens": last, **s}
        print(f"P115_W {arm}/{a.tag} {wname} B={b} decode_tok_s={s['decode_tok_s']} median={s['decode_tok_s_median']} "
              f"walls={walls}", flush=True)
    gs = getattr(parts.runner, "graph_stats", None)
    rec["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None
    rec["mem_after_runs"] = p109_box._mem(torch)
    rec["nvidia_smi"] = p109_box._smi()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P115_ARM " + json.dumps({"arm": arm, "tag": a.tag, "fuse_qkv": rec["fuse_qkv"], "fusions": rec["fusions"],
                                    "load_s": rec["load_s"], "graph_status": rec["graph_status"],
                                    "W16": rec["workloads"]["W16"]["decode_tok_s"],
                                    "W1": rec["workloads"]["W1"]["decode_tok_s"]}, default=str), flush=True)
    return 0


def self_test() -> int:
    on = {k: "1" for k in KNOBS}
    folds_only = {k: "1" for k in FOLDS}
    ok = [ARMS == ("F0", "F1"), p109_box.WORKLOADS == {"W16": 16, "W1": 1}, p109_box.self_test() == 0,
          arm_env_ok("F0", {}, False)[0], arm_env_ok("F0", {k: "0" for k in KNOBS}, False)[0],
          not arm_env_ok("F0", {"E4B_FUSE_T1_GLUE": "1"}, False)[0],
          arm_env_ok("F1", on, False)[0], not arm_env_ok("F1", folds_only, False)[0],
          arm_env_ok("F1", folds_only, True)[0], not arm_env_ok("F1", on, True)[0],
          not arm_env_ok("F1", {**on, "E4B_FUSE_ROUTER_EPI": "0"}, False)[0], not arm_env_ok("F2", on, False)[0]]
    print(f"p115_box self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--prompts-only", action="store_true")
    p.add_argument("--model")
    p.add_argument("--revision", default="")
    p.add_argument("--prompts")
    p.add_argument("--out")
    p.add_argument("--tag", default="")
    p.add_argument("--short", type=int, default=32)
    p.add_argument("--long", type=int, default=160)
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.prompts_only:
        return p109_box.prompts_main(a)
    return arm_main(a)


if __name__ == "__main__":
    sys.exit(main())
