#!/usr/bin/env python3
"""p161_box.py -- lane P161, the served read of K25's plan c1 under K39's decode (bench/p161/PREREG-p161.md; e4b#1313):
ONE process per arm. P158's box (bench/p158/p158_box.py) by counted edits: the K25 PLAN knob in place of the decode's
(the decode stays at qwen3_moe's default, prmt, in both arms), plus a capture-time spy on K25's block shapes; no SANE
mode (the arms are bitwise-identical by contract, so the token gate replaces a quality read); a clock sampler beside
every timed block.

The engine is the shipped default server (``PagedServeConfig.from_env()`` + ``build_engine(cfg)``) at grouped-nf4-gemm
v0.50.0: the B=1 fused stack on (auto, qwen3_moe allowlisted), the bandwidth decode GEMV on (P116), the attention split
cap at its qwen3_moe default (16), ``max_seqs`` 16 named by the runner, 4096 tokens per sequence (the default), the int8
lm_head at its qwen3_moe default (on, P153's FLIP), the dense attention GEMV off, K25 at its default (auto), its decode
at qwen3_moe's default (prmt, P158's FLIP). The arms differ ONLY in ``E4B_K25_PLAN``:
  S0  ``today``  (named: today's server, K25 at its single plan);
  S1  ``c1``     (P156's first carried pair, read by P160; bit-identical to today's plan by contract).

``--mode speed``: P109's workloads, timing and token records at P109's registered bytes, with K25's resolutions
(``info["k25_lut"]``, ``info["k25_plan"]``) and THREE engagement records read after the build and after the runs:
``hot_residency.K25_LUT_SEEN`` (per call, by decode), ``hot_residency.K25_PLAN_SEEN`` (per call, by plan) and a spy on
the kernel's own block-shape arguments (per kernel call; two per K25 call). Each counts eager calls and graph captures,
never a replay, so the spy costs no timed block. The cap's plans, the
GEMV tally, the card's telemetry read either side of every timed block, and a clock sampler DURING each timed block:
``nvidia-smi ... -lms 500`` in its own process (never a thread of the timed one), its lines kept per block.

    python p161_box.py --mode speed --prompts prompts.json --out arm_S0a.json --tag S0a   (arm from P161_ARM)
    python p161_box.py --prompts-only --model ID --revision SHA --out prompts.json
    python p161_box.py --self-test
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p109_box  # noqa: E402  (staged at P109's registered bytes: prompts, run_pass, slope, digest, _mem, _smi)

ARMS = ("S0", "S1")
KNOB = "E4B_K25_PLAN"
ARM_VALUE = {"S0": "today", "S1": "c1"}
LUT_KNOB = "E4B_K25_LUT"                  # unset in both arms: K25's decode at its qwen3_moe default (prmt, P158)
K25_KNOB = "E4B_NF4_GROUPED_SMALLM"       # unset in both arms: K25 at its default (auto)
HEAD_KNOB = "E4B_INT8_HEAD_BW"            # unset in both arms: the int8 head at its qwen3_moe default, ON (P153)
DENSE_KNOB = "E4B_DENSE_GEMV_BW"          # unset in both arms: the dense attention GEMV off (P151's)
CAP_KNOB = "E4B_ATTN_SPLIT_CAP"           # unset in both arms: the cap at its qwen3_moe default (16, e4b#1792)
FUSION_KNOBS = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
GEMV_KNOBS = ("GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE")
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")
TELEMETRY = ("power.limit", "enforced.power.limit", "clocks.max.sm", "temperature.gpu", "clocks.sm", "power.draw")
SAMPLE = ("clocks.sm", "enforced.power.limit", "power.draw.instant", "clocks_event_reasons.active")
SAMPLE_MS = 500                                  # 100 ms cost +0.25 % on the A2000 smoke (reproduced); 500 ms is inside the A/A noise


def arm_env_ok(arm: str, env):
    """``(ok, why)``: the knob carries the arm's value; K25's mode and decode, the int8 head, the split cap, the dense
    attention GEMV, fusion and decode GEMV knobs are unset (their defaults)."""
    if arm not in ARMS:
        return False, f"unknown arm {arm!r}"
    bad = {}
    if (env.get(KNOB) or "").strip() != ARM_VALUE[arm]:
        bad[KNOB] = env.get(KNOB)
    bad.update({k: env.get(k) for k in (K25_KNOB, LUT_KNOB, HEAD_KNOB, CAP_KNOB, DENSE_KNOB) + FUSION_KNOBS + GEMV_KNOBS
                if (env.get(k) or "").strip()})
    return (not bad), f"{arm} with {bad}"


def _arm():
    arm = os.environ.get("P161_ARM", "")
    ok, why = arm_env_ok(arm, os.environ)
    if not ok:
        raise SystemExit(f"REFUSED: P161_ARM={arm!r}: {why}")
    return arm


def _counts():
    import nf4_grouped
    return dict(nf4_grouped.dispatch_counts())


def _delta(after, before):
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


def _seen():
    """K25's decode record now: ``{"<lut>|le256": calls}`` (eager calls and graph captures, never a replay)."""
    from experts4bit_qlora.engines import hot_residency
    return dict(sorted(hot_residency.K25_LUT_SEEN.items()))


def _plan_seen():
    """K25's plan record now: ``{"<plan>|le256": calls}`` (e4b#1877; eager calls and graph captures, never a replay)."""
    from experts4bit_qlora.engines import hot_residency
    return dict(sorted(hot_residency.K25_PLAN_SEEN.items()))


SHAPES: dict = {}


def shape_key(kw: dict) -> str:
    """One K25 kernel call's plan as ``"<block_n>/<kc>/<warps>/<stages>/<lut>"``."""
    return "/".join(str(kw.get(k)) for k in ("block_n", "kc", "warps", "stages", "lut"))


def install_shape_spy() -> None:
    """The second witness: count every K25 kernel call by the block shape it receives, passing it through unchanged.
    Installed before the build. It keeps K25's signature (``functools.wraps``), because the route reads it to decide the
    lean glue (``_lean_glue_supported``). Python runs it on eager calls and graph captures only, never on a replay."""
    import functools
    import nf4_smallm
    k25 = nf4_smallm.gemm_nf4_grouped_smallm

    @functools.wraps(k25)
    def spy(*args, **kw):
        key = shape_key(kw)
        SHAPES[key] = SHAPES.get(key, 0) + 1
        return k25(*args, **kw)
    nf4_smallm.gemm_nf4_grouped_smallm = spy


def _shapes():
    return dict(sorted(SHAPES.items()))


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_telemetry(line: str) -> dict:
    """One ``nvidia-smi --query-gpu=<TELEMETRY> --format=csv,noheader,nounits`` line as floats; an unreadable field
    (``[N/A]``, missing) is None."""
    vals = [v.strip() for v in (line or "").split(",")]
    return {k: (_num(vals[i]) if i < len(vals) else None) for i, k in enumerate(TELEMETRY)}


def _telemetry() -> dict:
    """The card now, read between timed blocks."""
    try:
        line = subprocess.check_output(["nvidia-smi", "--query-gpu=" + ",".join(TELEMETRY), "--format=csv,noheader,nounits"],
                                       text=True, timeout=20).splitlines()[0]
    except Exception:  # noqa: BLE001 -- an unreadable card reads None in every field; the reducer VOIDs on the limit
        line = ""
    return parse_telemetry(line)


def parse_samples(text: str) -> list:
    """The sampler's lines as dicts over :data:`SAMPLE`: numbers as floats, the event-reason bitmask as an int (hex or
    decimal), an unreadable field None. A line with the wrong field count is dropped."""
    out = []
    for ln in (text or "").splitlines():
        vals = [v.strip() for v in ln.split(",")]
        if len(vals) != len(SAMPLE):
            continue
        rec = {k: _num(v) for k, v in zip(SAMPLE[:3], vals[:3])}
        try:
            rec[SAMPLE[3]] = int(vals[3], 0)
        except ValueError:
            rec[SAMPLE[3]] = None
        out.append(rec)
    return out


class Sampler:
    """``nvidia-smi --query-gpu=<SAMPLE> -lms 500`` in its own process for the life of one timed block. Its lines are
    the block's clock record; a sampler that cannot start, or that prints nothing, leaves ``[]`` (the reducer reads that
    as UNREAD, never as 0)."""

    def __enter__(self):
        import tempfile
        self.samples, self.error, self.p = [], None, None
        self.out, self.err = tempfile.TemporaryFile("w+"), tempfile.TemporaryFile("w+")   # files, never a pipe that fills
        try:
            self.p = subprocess.Popen(["nvidia-smi", "--query-gpu=" + ",".join(SAMPLE), "--format=csv,noheader,nounits",
                                       "-lms", str(SAMPLE_MS)], stdout=self.out, stderr=self.err, text=True)
        except Exception as e:  # noqa: BLE001
            self.error = f"sampler did not start: {type(e).__name__}"
        return self

    def __exit__(self, *exc):
        if self.p is not None:
            self.p.terminate()
            try:
                self.p.wait(timeout=20)
            except Exception:  # noqa: BLE001
                self.p.kill()
                self.p.wait()
            self.out.seek(0)
            self.err.seek(0)
            self.samples = parse_samples(self.out.read())
            self.error = self.err.read().strip()[:300] or None
        self.out.close()
        self.err.close()
        return False


def _sms():
    import torch
    return torch.cuda.get_device_properties(0).multi_processor_count


def _common(cfg, parts, arm, load_s):
    import torch
    info = parts.info
    return {"arm": arm, "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
            "model": cfg.model, "revision": cfg.revision,
            "knobs": {k: os.environ.get(k) for k in (KNOB, K25_KNOB, LUT_KNOB, HEAD_KNOB, CAP_KNOB, DENSE_KNOB) + FUSION_KNOBS
                      + GEMV_KNOBS},
            "torch": torch.__version__, "sm_count": _sms(), "load_s": round(load_s, 2),
            "fusions": {k: info.get(k) for k in CENSUS_KEYS}, "fusion_modes": info.get("fusion_modes"),
            "fusion_sources": info.get("fusion_sources"), "attn_split_cap": info.get("attn_split_cap"),
            "dense_gemv_bw": info.get("dense_gemv_bw"), "int8_head_bw": info.get("int8_head_bw"),
            "k25_lut": info.get("k25_lut"), "k25_plan": info.get("k25_plan"),
            "model_type": info.get("model_type"), "max_tokens_per_seq": cfg.max_tokens_per_seq,
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
    if not cfg.graphs or (cfg.max_seqs, cfg.placement, tuple(cfg.buckets), cfg.max_tokens_per_seq) != (16, "all-vram", (1, 2, 4, 8, 16), 4096):
        raise SystemExit(f"REFUSED: not the default graph server at 16 slots of 4096 tokens: graphs={cfg.graphs} "
                         f"max_seqs={cfg.max_seqs} placement={cfg.placement} buckets={cfg.buckets} "
                         f"max_tokens_per_seq={cfg.max_tokens_per_seq}")
    install_shape_spy()
    c0 = _counts()
    seen0, plan0, shapes0 = _seen(), _plan_seen(), _shapes()
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    load_s = time.perf_counter() - t0
    c_build = _counts()
    rec = {"mode": "speed", "tag": a.tag, **_common(cfg, parts, arm, load_s), "dispatch_build": _delta(c_build, c0),
           "k25_seen_before_build": seen0, "k25_seen_build": _seen(),
           "k25_plan_seen_before_build": plan0, "k25_plan_seen_build": _plan_seen(),
           "k25_shapes_before_build": shapes0, "k25_shapes_build": _shapes(),
           "prompts_sha256": pf["prompts_sha256"], "short": a.short, "long": a.long, "reps": a.reps,
           "mem_after_load": p109_box._mem(torch), "telemetry_after_load": _telemetry(), "workloads": {}, "status": "ok"}
    for wname, b in p109_box.WORKLOADS.items():
        wrows = rows[:b]
        walls, digests, last, telem, samples = {}, {}, {}, {}, {}
        cw = _counts()
        for n in (a.short, a.long):
            p109_box.run_pass(parts, torch, wrows, n)                      # warm, untimed
            ws, ds = [], []
            before = _telemetry()                                          # either side of the timed block
            with Sampler() as smp:                                         # and during it, in its own process
                for _ in range(a.reps):
                    wall, _steps, toks = p109_box.run_pass(parts, torch, wrows, n)
                    ws.append(round(wall, 5))
                    ds.append(p109_box.digest(toks))
                    last[str(n)] = toks
            walls[str(n)], digests[str(n)] = ws, ds
            telem[str(n)] = {"before": before, "after": _telemetry()}
            samples[str(n)] = {"samples": smp.samples, "error": smp.error}
        s = p109_box.slope(walls[str(a.short)], walls[str(a.long)], b, a.short, a.long)
        rec["workloads"][wname] = {"batch": b, "walls": walls, "rep_digests": digests, "tokens": last,
                                   "dispatch": _delta(_counts(), cw), "telemetry": telem, "clock_samples": samples, **s}
        print(f"P161_W {arm}/{a.tag} {wname} B={b} decode_tok_s={s['decode_tok_s']} walls={walls} "
              f"samples={ {n: len(v['samples']) for n, v in samples.items()} }", flush=True)
    gs = getattr(parts.runner, "graph_stats", None)
    rec["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None
    rec["attn_split_cap_after_runs"] = {"plans": {str(b): dict(r) for b, r in sorted(parts.runner.kv.split_plans.items())}} \
        if getattr(getattr(parts, "runner", None), "kv", None) is not None else None
    rec["k25_seen_after_runs"] = _seen()
    rec["k25_plan_seen_after_runs"] = _plan_seen()
    rec["k25_shapes_after_runs"] = _shapes()
    rec["dispatch_total"] = _delta(_counts(), c0)
    rec["mem_after_runs"] = p109_box._mem(torch)
    rec["nvidia_smi"] = p109_box._smi()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P161_ARM " + json.dumps({"arm": arm, "tag": a.tag, "k25_lut": rec["k25_lut"], "k25_plan": rec["k25_plan"],
                                      "k25_seen_build": rec["k25_seen_build"], "k25_plan_seen_build": rec["k25_plan_seen_build"],
                                      "k25_shapes_build": rec["k25_shapes_build"],
                                      "load_s": rec["load_s"], "W16": rec["workloads"]["W16"]["decode_tok_s"],
                                      "W1": rec["workloads"]["W1"]["decode_tok_s"]}, default=str), flush=True)
    return 0


def self_test() -> int:
    try:
        ok = _cases()
    except Exception as e:  # noqa: BLE001 -- a box that crashes on a registered case fails its self-test
        print(f"p161_box self-test FAILED [{type(e).__name__}: {e}]")
        return 1
    bad = [n for n, x in ok if not x]
    print(f"p161_box self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(ok) - len(bad)}/{len(ok)} cases)")
    return 0 if not bad else 1


def _spy_case() -> bool:
    """The spy on a stand-in kernel module: it counts by shape, returns the kernel's value unchanged, and keeps the
    kernel's signature (what ``_lean_glue_supported`` reads)."""
    import inspect
    import types
    saved, saved_shapes = sys.modules.get("nf4_smallm"), dict(SHAPES)
    try:
        def gemm_nf4_grouped_smallm(x, packed, absmax, row0, rows, grp, order=None, *, scatter=None, gather_div=1,
                                    block_n=32, kc=256, warps=4, stages=2, lut="pair", dot_bf16=None):
            return ("ran", block_n, kc)
        stand_in = types.ModuleType("nf4_smallm")
        stand_in.gemm_nf4_grouped_smallm = gemm_nf4_grouped_smallm
        sys.modules["nf4_smallm"] = stand_in
        SHAPES.clear()
        install_shape_spy()
        k = stand_in.gemm_nf4_grouped_smallm
        out = [k(0, 0, 0, 0, 0, 0, block_n=64, kc=256, warps=4, stages=2, lut="prmt"),
               k(0, 0, 0, 0, 0, 0, block_n=128, kc=128, warps=8, stages=3, lut="prmt"),
               k(0, 0, 0, 0, 0, 0, block_n=64, kc=256, warps=4, stages=2, lut="prmt")]
        return (out[0] == ("ran", 64, 256) and _shapes() == {"128/128/8/3/prmt": 1, "64/256/4/2/prmt": 2}
                and "gather_div" in inspect.signature(k).parameters and k is not gemm_nf4_grouped_smallm)
    finally:
        SHAPES.clear()
        SHAPES.update(saved_shapes)
        if saved is None:
            sys.modules.pop("nf4_smallm", None)
        else:
            sys.modules["nf4_smallm"] = saved


def _cases() -> list:
    s1 = {KNOB: "c1"}
    ok = [("arms", ARMS == ("S0", "S1")), ("P109's workloads", p109_box.WORKLOADS == {"W16": 16, "W1": 1}),
          ("P109's self-test", p109_box.self_test() == 0),
          ("S0 named", arm_env_ok("S0", {KNOB: "today"})[0]), ("S1", arm_env_ok("S1", s1)[0]),
          ("S1 padded", arm_env_ok("S1", {KNOB: " c1 "})[0]),
          ("S0 unset is refused (named explicitly)", not arm_env_ok("S0", {})[0]),
          ("S1 at another plan is refused", not arm_env_ok("S1", {KNOB: "c2"})[0]),
          ("S0 at c1 is refused", not arm_env_ok("S0", s1)[0]),
          ("the decode set is refused (qwen3_moe's default, prmt, is the subject)",
           not arm_env_ok("S1", {**s1, LUT_KNOB: "prmt"})[0]),
          ("the decode at the tree is refused too", not arm_env_ok("S0", {KNOB: "today", LUT_KNOB: "tree"})[0]),
          ("a shape key", shape_key({"block_n": 64, "kc": 256, "warps": 4, "stages": 2, "lut": "prmt"}) == "64/256/4/2/prmt"),
          ("a missing shape field reads None", shape_key({"block_n": 64}) == "64/None/None/None/None"),
          ("the spy counts, passes through and keeps the signature", _spy_case()),
          ("K25's mode set is refused (auto is the subject)", not arm_env_ok("S1", {**s1, K25_KNOB: "1"})[0]),
          ("the int8 head set is refused", not arm_env_ok("S1", {**s1, HEAD_KNOB: "1"})[0]),
          ("the split cap set is refused (its default is the subject's)", not arm_env_ok("S1", {**s1, CAP_KNOB: "16"})[0]),
          ("the split cap set on S0 too", not arm_env_ok("S0", {KNOB: "today", CAP_KNOB: "0"})[0]),
          ("the dense attention GEMV set is refused", not arm_env_ok("S1", {**s1, DENSE_KNOB: "1"})[0]),
          ("a fusion knob set is refused (the default is the subject)", not arm_env_ok("S1", {**s1, "E4B_FUSE_T1_GLUE": "auto"})[0]),
          ("the GEMV at its default, never forced", not arm_env_ok("S1", {**s1, "GNF4_GEMV_BW": "1"})[0]),
          ("an unknown arm", not arm_env_ok("S2", s1)[0]),
          ("telemetry parses", parse_telemetry("575.00, 575.00, 3090, 41, 2842, 80.12") == {
              "power.limit": 575.0, "enforced.power.limit": 575.0, "clocks.max.sm": 3090.0, "temperature.gpu": 41.0,
              "clocks.sm": 2842.0, "power.draw": 80.12}),
          ("an unreadable field is None", parse_telemetry("[N/A], [N/A], 3090, 41")["power.limit"] is None
           and parse_telemetry("[N/A], [N/A], 3090, 41")["enforced.power.limit"] is None
           and parse_telemetry("[N/A], [N/A], 3090, 41")["power.draw"] is None),
          ("no line reads None everywhere", set(parse_telemetry("").values()) == {None}),
          ("samples parse, the reasons as a bitmask", parse_samples("2842, 575.00, 412.3, 0x0000000000000004\n") == [
              {"clocks.sm": 2842.0, "enforced.power.limit": 575.0, "power.draw.instant": 412.3,
               "clocks_event_reasons.active": 4}]),
          ("an unreadable sample field is None, never 0", parse_samples("[N/A], 575, [N/A], [N/A]\n") == [
              {"clocks.sm": None, "enforced.power.limit": 575.0, "power.draw.instant": None,
               "clocks_event_reasons.active": None}]),
          ("a malformed sample line is dropped", parse_samples("2842, 575\nnot a line\n") == []),
          ("no sampler output is no samples", parse_samples("") == [])]
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--prompts-only", action="store_true")
    p.add_argument("--mode", choices=("speed",))
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
    if a.mode == "speed":
        return speed_main(a)
    raise SystemExit("REFUSED: --mode speed, --prompts-only or --self-test")


if __name__ == "__main__":
    sys.exit(main())
