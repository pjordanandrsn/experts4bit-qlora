#!/usr/bin/env python3
"""p111_box.py -- lane P111 (bench/p111/PREREG-p111.md), ONE arm in its own process: does ``E4B_KV_STEP_SELECT=1``
decode the default ``serve_paged`` server's tokens exactly, and faster?

The engine is the shipped default server: ``PagedServeConfig.from_env()`` + ``build_engine(cfg)``, with only the model,
arena and calibration set. Since 0.43.0 that server captures bucketed decode graphs (``E4B_PAGED_GRAPHS=auto``). The
arms differ ONLY in ``E4B_KV_STEP_SELECT``, which the KV pool reads at construction:
  S0  the per-layer selection (switch off; the shipped default);
  S1  one selection per step (switch on).

The workloads, timing and token records are P109's, imported from ``p109_box.py`` at its registered bytes:
- W16: 16 distinct 512-token prompts added at once; W1: row 0 alone.
- SHORT and LONG new tokens; one warm pass, then REPS timed passes; p37's slope.
- Every timed pass's token digest; the last pass's tokens per row.

    python p111_box.py --prompts prompts.json --out arm_S0a.json --tag S0a       (arm from P111_ARM; engine from env)
    python p111_box.py --prompts-only --model ID --revision SHA --out prompts.json
    python p111_box.py --self-test
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p109_box  # noqa: E402  (staged at P109's registered bytes: prompts, run_pass, slope, digest, _mem, _smi)

ARMS = ("S0", "S1")


def arm_main(a) -> int:
    arm = os.environ.get("P111_ARM", "")
    if arm not in ARMS:
        raise SystemExit(f"REFUSED: P111_ARM={arm!r}, expected one of {ARMS}")
    want = arm == "S1"
    if (os.environ.get("E4B_KV_STEP_SELECT", "0").strip() == "1") != want:
        raise SystemExit(f"REFUSED: arm {arm} with E4B_KV_STEP_SELECT={os.environ.get('E4B_KV_STEP_SELECT')!r}")
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
    info, kv = parts.info, parts.runner.kv
    rec = {"arm": arm, "tag": a.tag, "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
           "model": cfg.model, "revision": cfg.revision, "torch": torch.__version__, "load_s": round(load_s, 2),
           "step_select": bool(getattr(kv, "_step_select", False)),
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
        print(f"P111_W {arm}/{a.tag} {wname} B={b} decode_tok_s={s['decode_tok_s']} median={s['decode_tok_s_median']} "
              f"walls={walls}", flush=True)
    gs = getattr(parts.runner, "graph_stats", None)
    rec["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None
    rec["mem_after_runs"] = p109_box._mem(torch)
    rec["nvidia_smi"] = p109_box._smi()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P111_ARM " + json.dumps({"arm": arm, "tag": a.tag, "step_select": rec["step_select"], "load_s": rec["load_s"],
                                    "graph_status": rec["graph_status"], "W16": rec["workloads"]["W16"]["decode_tok_s"],
                                    "W1": rec["workloads"]["W1"]["decode_tok_s"]}), flush=True)
    return 0


def self_test() -> int:
    ok = [ARMS == ("S0", "S1"), p109_box.WORKLOADS == {"W16": 16, "W1": 1}, p109_box.self_test() == 0]
    print(f"p111_box self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
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
