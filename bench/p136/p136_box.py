#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""P136 unprofiled default-server reference arms, adapted from P123.

P109 supplies fixed W1/W16 prompts and 32/160-token slopes. gpt-oss records
W16 only; the two independent arms provide a noise check, not a speed claim.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p109_box  # staged at P109's registered bytes

KNOBS = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
GEMV_KNOBS = ("GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE")
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")


def default_env_ok(env) -> tuple:
    """``(ok, why)``: the subject is the shipped default -- the fusion and decode-GEMV knobs are unset."""
    bad = {k: env.get(k) for k in KNOBS + GEMV_KNOBS if (env.get(k) or "").strip()}
    return (not bad), f"set: {bad}"


def _counts():
    import nf4_grouped
    return dict(nf4_grouped.dispatch_counts())


def _delta(after, before):
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


def speed_main(a) -> int:
    ok, why = default_env_ok(os.environ)
    if not ok:
        raise SystemExit(f"REFUSED: not the shipped default ({why})")
    with open(a.prompts) as f:
        pf = json.load(f)
    rows = pf["rows"]
    if p109_box.digest(rows) != pf["prompts_sha256"] or len(rows) != p109_box.ROWS:
        raise SystemExit("REFUSED: prompts.json does not match its own digest")
    import torch

    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if not cfg.graphs or (cfg.max_seqs, cfg.placement) != (16, "all-vram"):
        raise SystemExit(f"REFUSED: not the default graph server at 16 slots: graphs={cfg.graphs} "
                         f"max_seqs={cfg.max_seqs} placement={cfg.placement}")
    c0 = _counts()
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    load_s = time.perf_counter() - t0
    c_build = _counts()
    info = parts.info
    rec = {"mode": "speed", "tag": a.tag, "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
           "model": cfg.model, "revision": cfg.revision, "torch": torch.__version__, "load_s": round(load_s, 2),
           "knobs": {k: os.environ.get(k) for k in KNOBS + GEMV_KNOBS},
           "fusions": {k: info.get(k) for k in CENSUS_KEYS}, "fusion_modes": info.get("fusion_modes"),
           "fusion_sources": info.get("fusion_sources"), "model_type": info.get("model_type"), "moe_layers": info.get("moe_layers"),
           "int4_store_kinds": info.get("int4_store_kinds"), "int4_expert_layers": info.get("int4_expert_layers"),
           "grouping": info.get("grouping"),
           "max_tokens_per_seq": cfg.max_tokens_per_seq,
           "graph_status": ({str(k): v for k, v in info["graph_status"].items()} if info.get("graph_status") else None),
           "max_seqs": cfg.max_seqs, "buckets": list(cfg.buckets), "dispatch_build": _delta(c_build, c0),
           "prompts_sha256": pf["prompts_sha256"], "short": a.short, "long": a.long, "reps": a.reps,
           "mem_after_load": p109_box._mem(torch), "workloads": {}, "status": "ok"}
    for wname, b in p109_box.WORKLOADS.items():
        if os.environ.get("P136_FAMILY") == "gptoss" and wname != "W16":
            continue
        wrows = rows[:b]
        walls = {}
        for n in (a.short, a.long):
            p109_box.run_pass(parts, torch, wrows, n)                      # warm, untimed
            ws = []
            for _ in range(a.reps):
                wall, _steps, _toks = p109_box.run_pass(parts, torch, wrows, n)
                ws.append(round(wall, 5))
            walls[str(n)] = ws
        s = p109_box.slope(walls[str(a.short)], walls[str(a.long)], b, a.short, a.long)
        rec["workloads"][wname] = {"batch": b, "walls": walls, **s}
        print(f"P136_W {a.tag} {wname} B={b} decode_ms_per_step={s.get('decode_ms_per_step')} walls={walls}", flush=True)
    gs = getattr(parts.runner, "graph_stats", None)
    rec["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None
    rec["dispatch_total"] = _delta(_counts(), c0)
    rec["mem_after_runs"] = p109_box._mem(torch)
    rec["nvidia_smi"] = p109_box._smi()
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print("P136_ARM " + json.dumps({"tag": a.tag, "fusions": rec["fusions"], "load_s": rec["load_s"],
                                    "W16_ms": rec["workloads"]["W16"].get("decode_ms_per_step"),
                                    "W1_ms": (rec["workloads"].get("W1") or {}).get("decode_ms_per_step")}, default=str), flush=True)
    return 0


def self_test() -> int:
    cases = [
        ("the default passes", default_env_ok({"E4B_PAGED_MAX_SEQS": "16"})[0]),
        ("an empty knob is unset", default_env_ok({"E4B_FUSE_T1_GLUE": " "})[0]),
        ("a set fusion knob refuses", not default_env_ok({"E4B_PAGED_FUSE_QKV": "auto"})[0]),
        ("a set GEMV knob refuses", not default_env_ok({"GNF4_GEMV_BW": "1"})[0]),
        ("P109's workloads", p109_box.WORKLOADS == {"W16": 16, "W1": 1}),
    ]
    bad = [n for n, ok in cases if not ok]
    print(f"p136_box self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--prompts")
    p.add_argument("--out")
    p.add_argument("--tag", default="")
    p.add_argument("--short", type=int, default=32)
    p.add_argument("--long", type=int, default=160)
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    return speed_main(a)


if __name__ == "__main__":
    sys.exit(main())
