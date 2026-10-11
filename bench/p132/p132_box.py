#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""p132_box.py -- lane P132 (#1313; bench/p132/PREREG-p132.md): split SD2's 1 -> n row verify increment by kernel class.

The server is SD2's stage V server, built by sd2_box.py's own ``_build_v`` (staged at SD2's bytes and imported, never
edited): the read target e4b 88ae1cfd + grouped-nf4-gemm 0.45.0, E4B_PAGED_SPEC=eagle3 at k = 3 with the pinned head,
16 slots, graphs on, so decode bucket 1 and verify buckets 2, 3, 4 are captured graphs.

  --mode timing   SD2's V1 timed terms re-run here, unprofiled (sd2_box.v1_terms, unchanged): the A/A anchors and, per
                  k = n - 1, verify_ms / full_ms. The reducer checks them against sd2-5090-3 and the desk target.
  --mode census   ONE condition per process, bracketed for Nsight Systems (cudaProfilerApi): n = 1 is the plain decode
                  bucket-1 replay (runner._run_decode_bucketed, the anchor's call); n = 2..4 is the verify bucket-n replay
                  (runner._spec_verify, the call V1's verify_ms times) at a fixed base fed the T == 1 oracle's own next
                  tokens, rewound to the base between replays (sd2_box.Gate.rewind, V0's construction).
  --mode canary   before the census: buckets 1..4 captured as graphs, and one plain + one verify replay per n, each
                  checked to return; the short census capture around it is the runner's (p132_run.sh), read by SC1b's
                  classifier before the full phase.
  --self-test     the pieces that need no GPU.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BUCKETS = (1, 2, 3, 4)
NS = (2, 3, 4)
ORACLE_WARM = 4            # plain steps after the prompt before the base is taken (as capture_bitwise does)


def _server():
    import sd2_box
    cfg, parts, rec = sd2_box._build_v(None)
    gate = sd2_box.Gate(parts, None)                  # begin / end / rewind only: no logits tap is installed
    return sd2_box, cfg, parts, rec, gate


def _record(cfg, rec):
    gs = (rec.get("census") or {}).get("graph_status") or {}
    return {"e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"), "model": cfg.model,
            "revision": cfg.revision, "load_s": rec.get("load_s"), "graph_status": {str(b): gs.get(str(b)) for b in BUCKETS},
            "spec_build": (rec.get("census") or {}).get("spec_build")}


def setup_verify(runner, gate, rid, slot, n):
    """The base, and the verify inputs at it: row 0 is the token at the base, rows 1..n-1 the T == 1 oracle's own next
    tokens (a fully accepted verify). Leaves the slot rewound to the base."""
    import torch
    for _ in range(ORACLE_WARM):
        runner._run_decode_bucketed([rid])
    base = runner.pos_of[rid] - 1
    oracle = [runner._run_decode_bucketed([rid])[rid] for _ in range(n - 1)]
    gate.rewind(rid, slot, base)
    ids = torch.tensor([runner.tokens[rid][base]] + [int(t) for t in oracle], dtype=torch.long, device="cuda")
    pos = torch.arange(base, base + n, device="cuda")
    return base, ids, pos


def replay_fn(runner, gate, rid, slot, n):
    """One replay of condition n: the plain bucket-1 decode (n = 1) or one verify of n rows at a fixed base."""
    if n == 1:
        return lambda: runner._run_decode_bucketed([rid])
    base, ids, pos = setup_verify(runner, gate, rid, slot, n)

    def one():
        tok = runner._spec_verify(slot, n, ids, pos)
        gate.rewind(rid, slot, base)
        return tok
    return one


def census_main(a) -> int:
    import torch
    sd2_box, cfg, parts, rec, gate = _server()
    row = sd2_box.load_rows(a.prompts_r)[0]
    out = {"mode": "census", "n": a.n, "capture": a.capture, **_record(cfg, rec)}
    rid, slot = gate.begin(row)
    try:
        one = replay_fn(parts.runner, gate, rid, slot, a.n)
        for _ in range(a.skip):
            one()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        t0 = time.perf_counter()
        for _ in range(a.steps):
            one()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        torch.cuda.profiler.stop()
    finally:
        gate.end(rid)
    gs = getattr(parts.runner, "graph_stats", None)
    out.update({"steps_bracketed": a.steps, "skip": a.skip, "profiled_ms_per_replay": round(wall * 1e3 / a.steps, 4),
                "graph_stats": {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None,
                "max_memory_allocated": int(torch.cuda.max_memory_allocated()), "status": "ok"})
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    print("P132_CENSUS " + json.dumps({k: out[k] for k in ("n", "capture", "steps_bracketed", "profiled_ms_per_replay")}),
          flush=True)
    return 0


def timing_main(a) -> int:
    import torch
    sd2_box, cfg, parts, rec, gate = _server()
    row = sd2_box.load_rows(a.prompts_r)[0]
    v1 = sd2_box.v1_terms(parts.runner, gate, row, sd2_box.TIMED, sd2_box.WARM)
    out = {"mode": "timing", **_record(cfg, rec), **v1, "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
           "status": "ok"}
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    print("P132_TIMING " + json.dumps({"anchors": [v1["anchor_a_ms"], v1["anchor_b_ms"]],
                                      **{f"k{k}": v1[f"k{k}"]["verify_ms"] for k in (1, 2, 3)}}), flush=True)
    return 0


def canary_main(a) -> int:
    """Fail in minutes, not the hour: every bucket captured, and each condition's replay returns once."""
    import torch
    sd2_box, cfg, parts, rec, gate = _server()
    row = sd2_box.load_rows(a.prompts_r)[0]
    base = _record(cfg, rec)
    missing = [b for b in BUCKETS if base["graph_status"].get(str(b)) != "graph"]
    out = {"mode": "canary", **base, "missing_buckets": missing, "replays": {}}
    if not missing:
        for n in (1,) + NS:
            rid, slot = gate.begin(row)
            try:
                tok = replay_fn(parts.runner, gate, rid, slot, n)()
                torch.cuda.synchronize()
                out["replays"][str(n)] = "ok" if tok is not None else "none"
            except Exception as e:                    # noqa: BLE001 -- the canary records and fails closed
                torch.cuda.synchronize()
                out["replays"][str(n)] = f"{type(e).__name__}: {str(e)[:200]}"
            finally:
                gate.end(rid)
    ok = not missing and all(out["replays"].get(str(n)) == "ok" for n in (1,) + NS)
    out["status"] = "ok" if ok else "failed"
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    print("P132_CANARY " + json.dumps({"status": out["status"], "missing_buckets": missing, "replays": out["replays"]}),
          flush=True)
    return 0 if ok else 26


def self_test() -> int:
    cases = [("buckets 1..4", BUCKETS == (1, 2, 3, 4)), ("n = 2..4", NS == (2, 3, 4)),
             ("the oracle warm-up is capture_bitwise's", ORACLE_WARM == 4)]
    try:
        import sd2_box
        cases += [("SD2's server is k = 3", sd2_box.K_SERVER == 3),
                  ("SD2's V1 sizes", (sd2_box.TIMED, sd2_box.WARM) == (64, 8)),
                  ("SD2's builder and V1 are importable", callable(sd2_box._build_v) and callable(sd2_box.v1_terms))]
    except Exception as e:                            # noqa: BLE001
        cases.append((f"sd2_box imports ({type(e).__name__}: {e})", False))
    bad = [n for n, ok in cases if not ok]
    print(f"p132_box self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--mode", choices=("timing", "census", "canary"))
    p.add_argument("--n", type=int, choices=(1, 2, 3, 4))
    p.add_argument("--capture", default="", help="graph or node: recorded only (the runner picks nsys's trace mode)")
    p.add_argument("--skip", type=int, default=32)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--prompts-r")
    p.add_argument("--out")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    return {"timing": timing_main, "census": census_main, "canary": canary_main}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
