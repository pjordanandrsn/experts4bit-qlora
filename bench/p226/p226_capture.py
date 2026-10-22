#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P226's routing capture (bench/p226/PREREG-p226.md, "The data"): 32 DISTINCT requests decoded together at bucket 32 on
the shipped default server, with every MoE router's top-8 picks recorded for every row over 128 decode steps. It is
captured on the reading box BEFORE any timing, and its hash is pinned in the receipt.

It reuses P154's routing pass (``p154_route.py``, staged byte-identical) for the recorder, the step grouping, the store
and route records and their faults. What differs from P154:
- **32 rows**, the prompts ``wikitext_rows(tok, n=48)[16:48]`` (P109's builder; rows 0-15 are P109's and P153's prompts,
  so the capture uses rows no lane has decoded). Each row is 512 prompt tokens with a 256-token budget: at B = 32 the
  first-admitted row is up to 31 tokens ahead, so P154's 160 would retire rows inside the 128-step window.
- **The server:** the shipped default except ``E4B_PAGED_MAX_SEQS=32`` (so a bucket-32 decode exists) and graphs off (so
  the router hooks fire); the int8 head (qwen3_moe's default since e4b#1838) is RECORDED, never refused, since the
  capture stands for the served default's routing.
- **The route, asserted over the window:** device-grouped K25 on the NF4 store, 2 K25 calls a layer a step, the
  library's route record only ``nf4_k25|le256`` (32 rows x top-8 = 256 routed rows), and its plan record only
  ``c1|le256`` (the served default plan on qwen3_moe since P161).
- **The output:** the picks as one ``[48, 32, 128, 8]`` int16 tensor (``capture.pt``, key ``picks``), its sha256 over the
  ``<i2`` bytes, and the distinct-experts table the byte floor reads.

Eager decode at B = 32 may choose tokens a graphed server would not (e4b's served tokens depend on the schedule). The
bench needs the ROUTING DISTRIBUTION of 32 independent requests, not bit-exact served tokens; the PREREG says so.

    python p226_capture.py --prompts-only --model M --revision R --out prompts_p226.json
    python p226_capture.py --prompts prompts_p226.json --out capture.json --picks capture.pt   (engine from E4B_PAGED_*)
    python p226_capture.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p154_route as P  # noqa: E402  (staged beside this file, byte-identical to P154's)

BATCH, SKIP, STEPS, TOKENS = 32, 32, 128, 256
FIRST_ROW, N_ROWS = 16, 32                    # wikitext rows 16..47 (P109's builder; rows 0-15 are P109's own)
LAYERS, EXPERTS, TOP_K = 48, 128, 8
ROUTE_KEY, PLAN_KEY = "nf4_k25|le256", "c1|le256"


def picks_tensor(steps, torch):
    """``steps`` from :func:`p154_route.group_steps`: each step's 48 calls (layer, rows, k, idx). Returns the
    ``[layers, rows, steps, k]`` int16 tensor and the sha256 of its little-endian bytes."""
    t = torch.tensor([[call[3] for call in chunk] for chunk in steps], dtype=torch.int16)   # [steps, layers, rows, k]
    t = t.permute(1, 2, 0, 3).contiguous()
    return t, hashlib.sha256(t.numpy().astype("<i2").tobytes()).hexdigest()


def plan_seen_faults(delta, steps, n_layers):
    want = {PLAN_KEY: n_layers * steps}            # counted once a MoE layer a step, beside ROUTE_SEEN
    return [] if delta == want else [f"plan: the library's K25 plan record over the window is {delta}, {want} registered"]


def route_faults_32(delta, steps, n_layers):
    want = {ROUTE_KEY: n_layers * steps}
    return [] if delta == want else [f"route: the library's route record over the window is {delta}, {want} registered"]


def prompts_main(a) -> int:
    import p109_box
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    rows = p109_box.wikitext_rows(tok, n=FIRST_ROW + N_ROWS)[FIRST_ROW:]
    rec = {"rows": rows, "prompts_sha256": p109_box.digest(rows), "first_row": FIRST_ROW, "n_rows": len(rows)}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f)
    print("P226_PROMPTS " + json.dumps({"rows": len(rows), "prompt_tokens": len(rows[0]), "sha256": rec["prompts_sha256"]}),
          flush=True)
    return 0


def capture_main(a) -> int:
    import p109_box
    from p136_box import default_env_ok
    from sc1b_e4b_census import census_window

    env = {k: v for k, v in os.environ.items() if k not in ("E4B_PAGED_GRAPHS", "E4B_PAGED_MAX_SEQS")}
    ok, why = default_env_ok(env)
    if not ok or os.environ.get("E4B_PAGED_GRAPHS") != "0" or os.environ.get("E4B_PAGED_MAX_SEQS") != str(BATCH):
        raise SystemExit(f"REFUSED: the capture runs the default server with graphs off, max_seqs {BATCH} and nothing "
                         f"else set ({why})")
    started_at = float(os.environ["P226_STARTED_AT"])
    with open(a.prompts, encoding="utf-8") as f:
        pf = json.load(f)
    rows = pf["rows"]
    if p109_box.digest(rows) != pf["prompts_sha256"] or len(rows) != BATCH or len({tuple(r) for r in rows}) != BATCH:
        raise SystemExit("REFUSED: prompts_p226.json does not match its digest, or is not 32 distinct rows")
    import torch

    from experts4bit_qlora.engines import hot_residency
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if cfg.graphs or (cfg.max_seqs, cfg.placement) != (BATCH, "all-vram"):
        raise SystemExit(f"REFUSED: graphs={cfg.graphs} max_seqs={cfg.max_seqs} placement={cfg.placement}")
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    info, model = parts.info, parts.runner.model
    hot_residency.DEVICE_GROUPING[0] = True          # the switches serve_paged sets for a server that captures decode
    hot_residency.FORCE_SINGLETON_GROUPS[0] = False
    rec = P.Recorder()
    layers = P.attach(model, rec)
    try:
        import nf4_smallm
        rec.count_k25(nf4_smallm)
        k25_ok = True
    except (ImportError, AttributeError):
        k25_ok = False
    n_layers = int(info.get("moe_layers") or 0)
    n_experts = int(getattr(model.config, "num_experts", 0) or 0)
    top_k = int(getattr(model.config, "num_experts_per_tok", 0) or 0)
    out = {"lane": "P226", "batch": BATCH, "mode": "routing", "graphs": False, "e4b_sha": os.environ.get("E4B_SHA"),
           "gnf4_sha": os.environ.get("GNF4_SHA"), "model": cfg.model, "revision": cfg.revision,
           "load_s": round(time.perf_counter() - t0, 1), "prompts_sha256": pf["prompts_sha256"],
           "first_row": pf.get("first_row"), "moe_layers": n_layers, "num_experts": n_experts, "top_k": top_k,
           "routers_hooked": len(layers), "skip": SKIP, "steps_bracketed": STEPS, "tokens": TOKENS}
    faults = [] if layers == list(range(n_layers)) else [f"{len(layers)} routers hooked, {n_layers} MoE layers"]
    if (n_layers, n_experts, top_k) != (LAYERS, EXPERTS, TOP_K):
        faults.append(f"model: {n_layers} MoE layers, {n_experts} experts, top-{top_k}; registered "
                      f"{LAYERS} / {EXPERTS} / top-{TOP_K}")
    out["store"], store_faults = P.store_record(model, n_layers, os.environ.get("E4B_PAGED_ARENA"), started_at)
    out["route"], route_faults = P.route_record(hot_residency, BATCH, out["store"].get("kind"), k25_ok)
    faults += store_faults + route_faults
    try:
        from experts4bit_qlora.engines.int8_head_bw import Int8HeadLinear
        out["head"] = {"int8_head_modules": sum(1 for m in model.modules() if isinstance(m, Int8HeadLinear)),
                       "E4B_INT8_HEAD_BW": os.environ.get("E4B_INT8_HEAD_BW")}
    except ImportError:
        out["head"] = {"int8_head_modules": 0, "E4B_INT8_HEAD_BW": os.environ.get("E4B_INT8_HEAD_BW")}
    p109_box.run_pass(parts, torch, rows, 32)                      # warm (untimed, hooks off)
    sched = parts.scheduler
    for r in rows:
        sched.add_request(list(r), max_new_tokens=TOKENS)
    seen = {}

    def _start():
        seen["route"] = dict(hot_residency.ROUTE_SEEN)
        seen["plan"] = dict(hot_residency.K25_PLAN_SEEN)
        rec.start()

    def _stop():
        rec.stop()
        seen["route_after"] = dict(hot_residency.ROUTE_SEEN)
        seen["plan_after"] = dict(hot_residency.K25_PLAN_SEEN)
    out.update(census_window(sched, BATCH, SKIP, STEPS, _start, _stop, torch.cuda.synchronize, tokens=TOKENS))
    sched.run_until_idle()
    steps, f2 = P.group_steps(rec.calls, n_layers)
    s = P.summarize(steps, n_experts, BATCH, top_k)
    out.update(s)
    route_delta = P.route_seen_delta(seen.get("route", {}), seen.get("route_after", {}))
    plan_delta = P.route_seen_delta(seen.get("plan", {}), seen.get("plan_after", {}))
    out["route"].update(k25_calls_in_window=rec.k25_calls, route_seen_in_window=route_delta,
                        k25_plan_seen_in_window=plan_delta)
    try:
        out["copy_rate"] = P.copy_rate(torch)
    except Exception as ex:  # noqa: BLE001 -- reported only
        out["copy_rate"] = {"error": f"{type(ex).__name__}: {ex}"}
    out["faults"] = (faults + f2 + s["faults"] + P.k25_faults(rec.k25_calls, len(steps), n_layers)
                     + route_faults_32(route_delta, len(steps), n_layers)
                     + plan_seen_faults(plan_delta, len(steps), n_layers)
                     + ([] if len(steps) == STEPS else [f"{len(steps)} steps recorded, {STEPS} bracketed"]))
    if not out["faults"]:
        t, sha = picks_tensor(steps, torch)
        torch.save({"picks": t}, a.picks)
        out["picks"] = {"shape": list(t.shape), "dtype": "int16", "sha256_le_i2": sha,
                        "file_sha256": hashlib.sha256(open(a.picks, "rb").read()).hexdigest()}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f)
    print("P226_CAPTURE " + json.dumps({"steps": len(steps), "summary": s["summary"], "k25_calls": rec.k25_calls,
                                        "picks": out.get("picks"), "faults": out["faults"][:4]}), flush=True)
    return 0 if not out["faults"] else 29


def self_test() -> int:
    import random

    import torch
    n = 0

    def ok(c, msg):
        nonlocal n
        assert c, msg
        n += 1
    rng = random.Random(226)
    layers, rows, k, n_steps = 3, 4, 2, 5
    steps = [[(li, rows, k, [rng.sample(range(16), k) for _ in range(rows)]) for li in range(layers)]
             for _ in range(n_steps)]
    t, sha = picks_tensor(steps, torch)
    ok(tuple(t.shape) == (layers, rows, n_steps, k) and t.dtype == torch.int16, f"picks are [layers, rows, steps, k]: {t.shape}")
    ok(all(t[li, r, s].tolist() == steps[s][li][3][r] for li in range(layers) for r in range(rows) for s in range(n_steps)),
       "each pick lands at its own layer, row and step")
    ok(sha == hashlib.sha256(t.numpy().astype("<i2").tobytes()).hexdigest() and len(sha) == 64, "the hash is over <i2 bytes")
    ok(not route_faults_32({ROUTE_KEY: layers * n_steps}, n_steps, layers)
       and route_faults_32({ROUTE_KEY: layers * n_steps, "nf4_mtile_captured|gt256": 1}, n_steps, layers),
       "only nf4_k25|le256 in the window")
    ok(not plan_seen_faults({PLAN_KEY: layers * n_steps}, n_steps, layers)
       and plan_seen_faults({"today|le256": layers * n_steps}, n_steps, layers)
       and plan_seen_faults({PLAN_KEY: 2 * layers * n_steps}, n_steps, layers), "only c1, once a layer a step")
    ok(BATCH * TOP_K == 256 and ROUTE_KEY.endswith("le256"), "32 rows x top-8 is K25's 256-row bound, not over it")
    ok(FIRST_ROW == 16 and N_ROWS == BATCH and SKIP + STEPS + BATCH - 1 < TOKENS,
       "rows 16..47, and the first-admitted row cannot retire inside the window")
    ok(P.K25_PER_LAYER_STEP == 2, "two K25 calls a layer a step (gate_up and down)")
    print(f"p226_capture self-test OK ({n} checks)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompts-only", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--revision")
    ap.add_argument("--prompts")
    ap.add_argument("--out")
    ap.add_argument("--picks")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--check-args", action="store_true", help="apply the argument checks a real run applies, then stop")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.prompts_only:
        if not (a.model and a.revision and a.out):
            ap.error("--prompts-only needs --model, --revision and --out")
    elif not (a.prompts and a.out and a.picks):
        ap.error("--prompts, --out and --picks (or --prompts-only, or --self-test)")
    if a.check_args:
        print("ARGS OK")
        return 0
    return prompts_main(a) if a.prompts_only else capture_main(a)


if __name__ == "__main__":
    sys.exit(main())
