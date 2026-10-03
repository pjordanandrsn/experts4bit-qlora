#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc1b_e4b_census.py -- lane SC1b (#846): bracket exactly N steady decode steps of e4b's serve_paged scheduler for an
Nsight Systems capture (`nsys profile --capture-range=cudaProfilerApi ...`).

SC1's own driver (sc1_e4b_sched.py) is imported, never edited: lanes P100 and P102 pin its bytes. This driver builds the
engine exactly as SC1's arm does (the same env, the same refusals), adds the B rows, steps the scheduler with `step()`
until a plan is decode-only with all B rows, takes K more steps, calls cudaProfilerStart, takes exactly N `step()`s --
each must be decode-only with all B rows, else it refuses -- calls cudaProfilerStop and drains. The profiled wall is
recorded and is NEVER a speed reading: every speed in SC1b is SC1's own unprofiled arm.

  env: SC1_BATCH, SC1_ARM, SC1_PROMPTS, SC1_OUT, and SC1's engine env (E4B_PAGED_*, the levers); PYTHONPATH= (A4)
  sc1b_e4b_census.py --skip K --steps N
  sc1b_e4b_census.py --write-prompts PATH --vocab-from CONFIG.json [--batch 16]   (the proof's Granite rows, amendment A2)
  sc1b_e4b_census.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Refusal(SystemExit):
    pass


TOKENS = 160          # SC1b's registered generation length per row (round 2 B1: the window must not outlive the first row)


def census_window(sched, batch: int, skip: int, steps: int, start, stop, sync, clock=time.perf_counter, max_ramp: int = 4096,
                  tokens: int = TOKENS):
    """Ramp to full decode, skip K steps, bracket exactly N decode-only full-batch steps. Returns the record. Refuses BEFORE
    opening the range if any row would retire inside the window (at B=16 serve_paged admits one 512-token prefill chunk per
    step, so the first-admitted row is up to B-1 tokens ahead of the last)."""
    ramp = 0
    while True:
        plan = sched.step()
        ramp += 1
        if plan.is_empty:
            raise Refusal("REFUSED: the scheduler went idle before every row reached decode (max_new_tokens too small?)")
        if plan.decode and not plan.prefill and len(plan.decode) == batch:
            break
        if ramp >= max_ramp:
            raise Refusal(f"REFUSED: no full decode-only step within {max_ramp} steps")
    for i in range(skip):
        plan = sched.step()
        if plan.prefill or len(plan.decode) != batch:
            raise Refusal(f"REFUSED: warm step {i} was not a full decode-only step")
    outs = [len(r.out) for r in getattr(sched, "active", {}).values()]
    if outs and max(outs) + steps >= tokens:
        raise Refusal(f"REFUSED: the first-admitted row holds {max(outs)} of {tokens} tokens; {steps} more would retire it inside "
                      "the window (raise the row budget)")
    positions = [min(outs) + 1, max(outs) + steps] if outs else None
    sync()
    start()
    t0 = clock()
    for i in range(steps):
        plan = sched.step()
        if plan.prefill or len(plan.decode) != batch:
            stop()
            raise Refusal(f"REFUSED: bracketed step {i} was not a full decode-only step (prefill {len(plan.prefill)}, "
                          f"decode {len(plan.decode)} of {batch}) -- the window would not be N steady steps")
    sync()
    wall = clock() - t0
    stop()
    return {"ramp_steps": ramp, "skip": skip, "steps_bracketed": steps, "row_tokens": tokens, "window_decode_positions": positions,
            "profiled_wall_s": round(wall, 6),
            "profiled_ms_per_step": round(wall / steps * 1e3, 4) if steps else None,
            "note": "profiled wall: never a speed reading (SC1's unprofiled arm is the speed)"}


def selftest() -> int:
    class Plan:
        def __init__(self, prefill, decode):
            self.prefill, self.decode = prefill, decode

        @property
        def is_empty(self):
            return not self.prefill and not self.decode

    class FakeSched:
        """B rows: two prefill steps, then decode for `left` steps per row."""
        def __init__(self, batch, left, prefill_steps=2):
            self.batch, self.left, self.pf, self.log = batch, left, prefill_steps, []

        def step(self):
            if self.pf:
                self.pf -= 1
                p = Plan([(r, 0, 256) for r in range(self.batch)], [])
            elif self.left > 0:
                self.left -= 1
                p = Plan([], list(range(self.batch)))
            else:
                p = Plan([], [])
            self.log.append(("pf" if p.prefill else "dec" if p.decode else "idle", self._prof))
            return p
        _prof = False

    events = []
    s = FakeSched(4, 20)

    def start():
        s._prof = True
        events.append("start")

    def stop():
        s._prof = False
        events.append("stop")
    r = census_window(s, 4, 3, 8, start, stop, lambda: events.append("sync"), clock=iter(range(0, 100, 5)).__next__)
    prof = [k for k, p in s.log if p]
    assert prof == ["dec"] * 8, prof                                     # exactly N decode steps inside the bracket
    assert events == ["sync", "start", "sync", "stop"], events
    assert r["ramp_steps"] == 3 and r["steps_bracketed"] == 8, r          # 2 prefill + the first full decode
    assert [k for k, _ in s.log[:3]] == ["pf", "pf", "dec"] and len(s.log) == 3 + 3 + 8, s.log
    # too few tokens: the bracket would run into idle -> refuse, and stop the profiler first
    s2, ev2 = FakeSched(4, 6), []
    try:
        census_window(s2, 4, 3, 8, lambda: ev2.append("start"), lambda: ev2.append("stop"), lambda: None)
        raise AssertionError("expected a refusal")
    except Refusal as e:
        assert "bracketed step" in str(e) and ev2 == ["start", "stop"], (e, ev2)
    # a partial batch never reaches a full decode step
    s3 = FakeSched(4, 0)
    try:
        census_window(s3, 4, 0, 1, lambda: None, lambda: None, lambda: None)
        raise AssertionError("expected a refusal")
    except Refusal as e:
        assert "idle" in str(e), e
    print("selftest OK (3 cases)")
    return 0


def write_prompts(path: str, vocab: int, batch: int, n: int | None = None, seed: int = 0) -> str:
    """Amendment A2: B distinct rows of seeded random token ids in [16, vocab), written in SC1's prompt-file format with
    SC1's own digest and checked by SC1's own `load_prompts`. Only the proof's Granite capture reads it: the lane's
    prompts_b{B}.json carries Qwen3 ids, which overflow Granite's vocabulary. The instrument proof needs real decode
    steps, not meaningful text."""
    import random

    import sc1_e4b_sched as sc1
    n = sc1.PROMPT_LEN if n is None else n
    if vocab <= 16 + n:
        raise Refusal(f"REFUSED: vocab {vocab} is too small for {n}-token rows")
    rng = random.Random(seed)
    rows: list = []
    while len(rows) < batch:
        r = [rng.randrange(16, vocab) for _ in range(n)]
        if r not in rows:
            rows.append(r)
    pf = {"batch": batch, "prompts": rows, "prompts_sha256": sc1.prompts_digest(rows),
          "source": f"seeded random token ids in [16, {vocab}), seed {seed}: SC1b proof only (A2)"}
    with open(path, "w") as f:
        json.dump(pf, f)
    return sc1.load_prompts(path, batch, prompt_len=n)["info"]["prompts_sha256"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip", type=int, default=32)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--write-prompts")
    ap.add_argument("--vocab-from")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.write_prompts:
        with open(a.vocab_from) as f:
            vocab = int(json.load(f)["vocab_size"])
        sha = write_prompts(a.write_prompts, vocab, a.batch)
        print("SC1B_PROMPTS " + json.dumps({"path": a.write_prompts, "batch": a.batch, "vocab": vocab, "prompts_sha256": sha}), flush=True)
        return 0
    import sc1_e4b_sched as sc1                                      # staged beside this file on the box, SC1's bytes
    hook = sc1.harness_hook_loaded()
    if hook:
        raise Refusal(f"REFUSED: the P42 harness hook is loaded ({hook}); run with PYTHONPATH= (SC1 A4)")
    batch = int(os.environ["SC1_BATCH"])
    arm = os.environ["SC1_ARM"]
    out_path = os.environ["SC1_OUT"]
    import torch

    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if cfg.max_seqs != batch or cfg.placement != "all-vram" or not cfg.graphs:
        raise Refusal(f"REFUSED: max_seqs {cfg.max_seqs} / placement {cfg.placement} / graphs {cfg.graphs}: SC1's arm is "
                      f"max_seqs = B = {batch}, all-vram, graphs on")
    env = dict(os.environ)
    rec = {"engine": "e4b", "arm": arm, "batch": batch, "mode": "census", "e4b_sha": env.get("E4B_SHA"),
           "route_env": {k: env.get(k) for k in sc1.ROUTE_ENV + ("E4B_INT4_PREFILL",)},
           "engine_kwargs": {k: env.get(k) for k in sc1.PAGED_ENV if env.get(k) is not None}, "torch": torch.__version__}
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    rec["load_s"] = round(time.perf_counter() - t0, 1)
    rec["census"] = parts.info
    pf = sc1.load_prompts(os.environ["SC1_PROMPTS"], batch, prompt_len=sc1.PROMPT_LEN)
    rows = pf["prompts"]
    rec.update(prompts_sha256=pf["info"]["prompts_sha256"])
    sc1.run_batch(parts, torch, rows, sc1.SHORT)                       # warm (untimed): every graph bucket replayed once
    sched = parts.scheduler
    for r in rows:
        sched.add_request(list(r), max_new_tokens=TOKENS)
    rec.update(census_window(sched, batch, a.skip, a.steps, torch.cuda.profiler.start, torch.cuda.profiler.stop,
                             torch.cuda.synchronize))
    sched.run_until_idle()
    rec["graph_stats"] = sc1._graph_stats(parts.runner)
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print("SC1B_E4B " + json.dumps({k: rec[k] for k in ("arm", "batch", "steps_bracketed", "profiled_ms_per_step")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
