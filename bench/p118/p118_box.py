#!/usr/bin/env python3
"""p118_box.py -- lane P118 (bench/p118/PREREG-p118.md; e4b#1313): ONE process per arm. Does the decode lookahead
(``E4B_PAGED_DECODE_LOOKAHEAD=1``: each decode step issued before the previous one is read back, its input ids taken on
the device) decode the default ``serve_paged`` server faster at one request, with identical tokens?

The engine is the shipped default server: ``PagedServeConfig.from_env()`` + ``build_engine(cfg)``, with the model, arena
and calibration set, and ``max_seqs`` 16 named (the W16 workload's width). The arms differ ONLY in the switch:
  L0  ``E4B_PAGED_DECODE_LOOKAHEAD`` unset (the shipped default: every decode step read back before the next is issued);
  L1  ``E4B_PAGED_DECODE_LOOKAHEAD=1`` (``ContinuousScheduler(lookahead=True)``: ``PagedModelRunner.issue_decode`` for
      step t+1, then ``collect_decode`` for step t).

P109's workloads, timing and token records, imported from ``p109_box.py`` at its registered bytes -- W16: 16 distinct
512-token prompts at once; W1: row 0 alone; SHORT and LONG new tokens; one warm pass, then REPS timed passes; p37's slope
(``stop_ids`` None: every sequence runs to its length, so the lookahead issues the synchronous path's rows step for step).

Engagement, counted in-process on the runner's two entry points: issues, collects, and collects with a newer step already
queued behind them (the overlap). L1 must overlap; L0 must never call them.

The mechanism, priced after the timed passes (never inside them): one more pass per workload at LONG tokens under
``engines.step_trace.StepTrace``, driven step by step. In L0 the GPU is idle at ``dec_prep``, so a decode step's device
time is ``gpu.dec_issue - gpu.dec_prep`` and the rest of ``step_ms`` is the host gap the lookahead removes; in L1 the
decode step's period should fall to about that device time. Reported, not ruled.

    python p118_box.py --prompts prompts.json --out arm_L0a.json --tag L0a --trace-dir work   (arm from P118_ARM)
    python p118_box.py --prompts-only --model ID --revision SHA --out prompts.json
    python p118_box.py --self-test
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p109_box  # noqa: E402  (staged at P109's registered bytes: prompts, run_pass, slope, digest, _mem, _smi)

ARMS = ("L0", "L1")
KNOB = "E4B_PAGED_DECODE_LOOKAHEAD"
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")


def arm_env_ok(arm: str, env):
    """``(ok, why)``: L0 has the switch unset or ``0``; L1 has it ``1``."""
    v = (env.get(KNOB) or "").strip()
    want = {"L0": ("", "0"), "L1": ("1",)}.get(arm)
    if want is None:
        return False, f"unknown arm {arm!r}"
    return v in want, f"{arm} with {KNOB}={v!r}"


def _arm():
    arm = os.environ.get("P118_ARM", "")
    if arm not in ARMS:
        raise SystemExit(f"REFUSED: P118_ARM={arm!r}, expected one of {ARMS}")
    ok, why = arm_env_ok(arm, os.environ)
    if not ok:
        raise SystemExit(f"REFUSED: arm {arm}: {why}")
    return arm


class Overlap:
    """Counts the runner's lookahead calls: issues, collects, and collects made with a newer step queued behind them."""

    def __init__(self, runner):
        self.issues = self.collects = self.overlapped = 0
        self._out = 0
        issue, collect = runner.issue_decode, runner.collect_decode

        def issue_counted(rids):
            self.issues += 1
            self._out += 1
            return issue(rids)

        def collect_counted(handle):
            self.collects += 1
            self.overlapped += int(self._out == 2)
            self._out -= 1
            return collect(handle)

        runner.issue_decode, runner.collect_decode = issue_counted, collect_counted

    def snap(self):
        return {"issues": self.issues, "collects": self.collects, "overlapped": self.overlapped}


def trace_summary(rows) -> dict:
    """Decode-only steps of a traced pass (decode rows, no prefill): the median step period, its device time where the
    GPU was idle at ``dec_prep`` (``gpu.dec_issue - gpu.dec_prep``; meaningful in L0), and the host remainder."""
    dec = [r for r in rows if r.get("decode_rows") and not r.get("prefill_tokens")]
    out = {"steps": len(rows), "decode_steps": len(dec)}
    if not dec:
        return out
    step = [r["step_ms"] for r in dec]
    dev = [r["gpu"]["dec_issue"] - r["gpu"]["dec_prep"] for r in dec
           if isinstance(r.get("gpu"), dict) and "dec_issue" in r["gpu"] and "dec_prep" in r["gpu"]]
    out["step_ms_p50"] = round(statistics.median(step), 4)
    if dev and len(dev) == len(dec):
        out["device_ms_p50"] = round(statistics.median(dev), 4)
        out["host_gap_ms_p50"] = round(statistics.median(s - d for s, d in zip(step, dev)), 4)
    segs = {}
    for r in dec:
        for k, v in r.get("seg", {}).items():
            segs.setdefault(k, []).append(v)
    out["seg_ms_p50"] = {k: round(statistics.median(v), 4) for k, v in sorted(segs.items())}
    return out


def traced_pass(parts, torch, rows, n_tokens, path) -> dict:
    """One pass of ``rows`` at ``n_tokens`` under a step trace written to ``path``, stepped here as the server's engine
    loop steps (begin, step, end; an empty step discarded). Outside every timed pass."""
    from experts4bit_qlora.engines.step_trace import StepTrace
    sched, runner = parts.scheduler, parts.runner
    if os.path.exists(path):
        os.remove(path)
    tr = StepTrace(path, cuda=True)
    sched.tracer = runner.tracer = tr
    try:
        before = len(sched.done)
        for r in rows:
            sched.add_request(list(r), max_new_tokens=n_tokens)
        while sched.queue or sched.active:
            tr.begin(ops=0)
            plan = sched.step()
            if plan.is_empty:
                tr.discard()
                break
            tr.end()
        torch.cuda.synchronize()
        tr.close()
        if len(sched.done) - before != len(rows):
            raise SystemExit(f"REFUSED: the traced pass finished {len(sched.done) - before} of {len(rows)} requests")
    finally:
        sched.tracer = runner.tracer = None
    with open(path, encoding="utf-8") as f:
        return trace_summary([json.loads(line) for line in f if line.strip()])


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
        raise SystemExit(f"REFUSED: not the default graph server at 16 slots: graphs={cfg.graphs} "
                         f"max_seqs={cfg.max_seqs} placement={cfg.placement} buckets={cfg.buckets}")
    if cfg.decode_lookahead != (arm == "L1"):
        raise SystemExit(f"REFUSED: arm {arm} built decode_lookahead={cfg.decode_lookahead}")
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    load_s = time.perf_counter() - t0
    if parts.scheduler.lookahead != (arm == "L1"):
        raise SystemExit(f"REFUSED: arm {arm} built a scheduler with lookahead={parts.scheduler.lookahead}")
    ov = Overlap(parts.runner)
    info = parts.info
    rec = {"mode": "speed", "arm": arm, "tag": a.tag, "e4b_sha": os.environ.get("E4B_SHA"),
           "gnf4_sha": os.environ.get("GNF4_SHA"), "model": cfg.model, "revision": cfg.revision,
           "knob": os.environ.get(KNOB), "lookahead": parts.scheduler.lookahead, "torch": torch.__version__,
           "load_s": round(load_s, 2), "fusions": {k: info.get(k) for k in CENSUS_KEYS},
           "fusion_modes": info.get("fusion_modes"), "levers_env": info.get("levers_env"),
           "graph_status": ({str(k): v for k, v in info["graph_status"].items()} if info.get("graph_status") else None),
           "grouping": info.get("grouping"), "prompts_sha256": pf["prompts_sha256"], "short": a.short,
           "long": a.long, "reps": a.reps, "mem_after_load": p109_box._mem(torch), "workloads": {}, "status": "ok"}
    for wname, b in p109_box.WORKLOADS.items():
        wrows = rows[:b]
        walls, digests, last = {}, {}, {}
        o0 = ov.snap()
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
        o1 = ov.snap()
        rec["workloads"][wname] = {"batch": b, "walls": walls, "rep_digests": digests, "tokens": last,
                                   "lookahead_calls": {k: o1[k] - o0[k] for k in o1}, **s}
        print(f"P118_W {arm}/{a.tag} {wname} B={b} decode_tok_s={s['decode_tok_s']} median={s['decode_tok_s_median']} "
              f"walls={walls} calls={rec['workloads'][wname]['lookahead_calls']}", flush=True)
    gs = getattr(parts.runner, "graph_stats", None)
    rec["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None   # timed passes only
    rec["lookahead_calls"] = ov.snap()
    rec["lookahead_discarded"] = parts.scheduler.stats().get("lookahead_discarded", 0)
    rec["trace"] = {}
    if a.trace_dir:
        for wname, b in p109_box.WORKLOADS.items():
            rec["trace"][wname] = traced_pass(parts, torch, rows[:b], a.long,
                                              os.path.join(a.trace_dir, f"trace_{a.tag}_{wname}.jsonl"))
            print(f"P118_TRACE {arm}/{a.tag} {wname} {json.dumps(rec['trace'][wname])[:400]}", flush=True)
    rec["mem_after_runs"] = p109_box._mem(torch)
    rec["nvidia_smi"] = p109_box._smi()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P118_ARM " + json.dumps({"arm": arm, "tag": a.tag, "lookahead": rec["lookahead"], "load_s": rec["load_s"],
                                    "graph_status": rec["graph_status"], "calls": rec["lookahead_calls"],
                                    "W16": rec["workloads"]["W16"]["decode_tok_s"],
                                    "W1": rec["workloads"]["W1"]["decode_tok_s"]}, default=str), flush=True)
    return 0


def _self_test_trace() -> bool:
    def row(step, prep, issue, pf=0, rows_=1):
        return {"step_ms": step, "decode_rows": rows_, "prefill_tokens": pf, "gpu": {"dec_prep": prep, "dec_issue": issue},
                "seg": {"plan": 0.1, "dec_sync": step - 0.4}}
    s = trace_summary([row(9.0, 0.3, 8.6), row(9.2, 0.3, 8.8), row(9.1, 0.3, 8.7), row(40.0, 30.0, 38.0, pf=512)])
    return (s["decode_steps"] == 3 and s["step_ms_p50"] == 9.1 and s["device_ms_p50"] == 8.4
            and s["host_gap_ms_p50"] == 0.7 and s["seg_ms_p50"]["plan"] == 0.1
            and trace_summary([])["decode_steps"] == 0)


def _self_test_overlap() -> bool:
    class R:
        def issue_decode(self, rids):
            return list(rids)

        def collect_decode(self, h):
            return {r: 0 for r in h}
    r = R()
    ov = Overlap(r)
    h1 = r.issue_decode([0])
    h2 = r.issue_decode([0])
    r.collect_decode(h1)          # a newer step queued behind it
    r.collect_decode(h2)          # nothing behind it (the drain)
    return ov.snap() == {"issues": 2, "collects": 2, "overlapped": 1}


def self_test() -> int:
    ok = [ARMS == ("L0", "L1"), p109_box.WORKLOADS == {"W16": 16, "W1": 1}, p109_box.self_test() == 0,
          arm_env_ok("L0", {})[0], arm_env_ok("L0", {KNOB: "0"})[0], not arm_env_ok("L0", {KNOB: "1"})[0],
          arm_env_ok("L1", {KNOB: "1"})[0], not arm_env_ok("L1", {})[0], not arm_env_ok("L1", {KNOB: "auto"})[0],
          not arm_env_ok("L2", {KNOB: "1"})[0], _self_test_trace(), _self_test_overlap()]
    print(f"p118_box self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
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
    p.add_argument("--trace-dir", default="")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.prompts_only:
        return p109_box.prompts_main(a)
    if a.prompts and a.out:
        return speed_main(a)
    raise SystemExit("REFUSED: --prompts and --out, --prompts-only or --self-test")


if __name__ == "__main__":
    sys.exit(main())
