#!/usr/bin/env python3
"""p223_box.py -- lane P223 (bench/p223/PREREG-p223.md): ONE process per arm, on Mixtral-8x7B's server. P149's box Q+S
box (bench/p149/p149_box.py) by substitution: the arms, no Q, no token-pinned blocks, no SANE phases.

The engine is the shipped server (``PagedServeConfig.from_env()`` + ``build_engine(cfg)``) at 16 slots of 768 tokens,
graphs on, every fusion knob at its default. The arms differ ONLY in three knobs:
  D0  none (the default server: bf16 attention projections, the library's split plan, buckets ``auto`` = 1,2,4,8,16);
  I0  ``E4B_ATTN_SPLIT_CAP=1`` and ``E4B_PAGED_BUCKETS=2,4,8,16`` (the schedule-invariant mode);
  D1  ``E4B_SERVE_ATTN_INT8=1`` (P149's S1);
  I1  ``E4B_SERVE_ATTN_INT8=1`` with both invariance knobs.

``--mode speed`` (P143 Amendment 3's form, as P149): P109's W16 and W1 workloads at P109's registered bytes, SHORT 32 and
LONG 160, one untimed warm pass and five timed reps each. Every timed rep's tokens are kept (the schedule-invariance
read compares each row's 32-token stream with its own 160-token stream), and every rep is digested.

ENGAGEMENT (asserted by the reducer from what this records): the split cap and the plan each captured bucket ran
(``info["attn_split_cap"]``, the KV's own record) beside the library's plan per bucket (grouped-nf4-gemm's
``decode_split_plan``, reported); the runner's buckets; per bucket, the runner's own replay, eager-step, row and pad-row
counters; per decode call, its live rows and the bucket ``bucket_for`` gives them (``DecodeCensus``, which wraps
``run_decode`` in every arm alike); and, per captured bucket, the int8 projections' calls during that bucket's warm-up
and capture.

    python p223_box.py --mode speed --prompts prompts.json --out arm_D0a.json --tag D0a
    python p223_box.py --prompts-only --model ID --revision SHA --out prompts.json
    python p223_box.py --self-test
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

INT8, SPLIT, BUCKETS = "E4B_SERVE_ATTN_INT8", "E4B_ATTN_SPLIT_CAP", "E4B_PAGED_BUCKETS"
ARM_KNOBS = {"D0": {}, "I0": {SPLIT: "1", BUCKETS: "2,4,8,16"},
             "D1": {INT8: "1"}, "I1": {INT8: "1", SPLIT: "1", BUCKETS: "2,4,8,16"}}
ARMS = tuple(ARM_KNOBS)
TAGS = ("D0a", "I0a", "I1a", "D1a", "D1b", "I1b", "I0b", "D0b")      # two ABBA blocks, eight fresh processes
DEFAULT_KNOBS = ("E4B_SERVE_ATTN_INT4", "E4B_SERVE_ATTN_INT4_CALIB", "E4B_DENSE_GEMV_BW", "E4B_INT8_HEAD_BW",
                 "E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI",
                 "GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE", "E4B_PAGED_DECODE_LOOKAHEAD",
                 "E4B_NF4_GROUPED_SMALLM", "E4B_NF4_T1_DEVICE_GROUPING")
SERVER = {"max_seqs": 16, "max_tokens_per_seq": 768, "placement": "all-vram"}
ARM_BUCKETS = {"D": (1, 2, 4, 8, 16), "I": (2, 4, 8, 16)}
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")


def arm_env_ok(arm: str, env):
    """``(ok, why)``: each of the three knobs carries the arm's value (unset where the arm has none); every other serving
    knob is unset (its default)."""
    if arm not in ARMS:
        return False, f"unknown arm {arm!r}"
    bad = {k: env.get(k) for k in (INT8, SPLIT, BUCKETS) if (env.get(k) or "").strip() != ARM_KNOBS[arm].get(k, "")}
    bad.update({k: env.get(k) for k in DEFAULT_KNOBS if (env.get(k) or "").strip()})
    return (not bad), f"{arm} with {bad}"


def _arm():
    arm = os.environ.get("P223_ARM", "")
    ok, why = arm_env_ok(arm, os.environ)
    if not ok:
        raise SystemExit(f"REFUSED: P223_ARM={arm!r}: {why}")
    return arm


def server_refusal(cfg, arm):
    """None for the registered server at ``arm``'s buckets, else the reason (16 slots x 768 tokens, all-vram, graphs on;
    D arms ``auto`` = 1,2,4,8,16, I arms 2,4,8,16 with the split cap at 1)."""
    got = {"max_seqs": cfg.max_seqs, "max_tokens_per_seq": cfg.max_tokens_per_seq, "placement": cfg.placement}
    want_b = ARM_BUCKETS[arm[0]]
    if not cfg.graphs or got != SERVER or tuple(cfg.buckets) != want_b:
        return f"not the registered server: graphs={cfg.graphs} {got} buckets={tuple(cfg.buckets)} (want {SERVER} {want_b})"
    split = getattr(cfg, "attn_split_cap", None)
    if (arm[0] == "I") != (split == "1"):
        return f"{arm}: the split cap reads {split!r}"
    return None


def heat():
    """The GPU's temperature (C), SM clock (MHz), power draw (W) and active clock-event reasons (a bitmask; 0x4 is the
    software power cap), read outside every timed window: the arms run in separate processes, so a clock share of a
    ratio shows here. Reported, never ruled."""
    import subprocess
    out = {"t": round(time.time(), 3)}
    for fields in (("temperature.gpu", "clocks.sm", "power.draw", "clocks_event_reasons.active"),
                   ("temperature.gpu", "clocks.sm")):           # a driver without the newer fields: the two kept
        try:
            row = subprocess.check_output(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                                          text=True, timeout=20).strip().splitlines()[0].split(",")
        except Exception as e:  # noqa: BLE001 -- reported only
            out["error"] = f"{type(e).__name__}: {str(e)[:120]}"
            continue
        for f, v in zip(fields, (x.strip() for x in row)):  # each field on its own: an [N/A] drops only itself
            key = {"temperature.gpu": "temp_c", "clocks.sm": "sm_mhz", "power.draw": "power_w",
                   "clocks_event_reasons.active": "clock_events"}[f]
            try:
                out[key] = v if f == "clocks_event_reasons.active" else float(v)
            except ValueError:
                out[key] = None
        out.pop("error", None)
        break
    return out


def card():
    """The card facts the rule VOIDs on (name, compute capability, enforced power limit), read in every process."""
    import subprocess
    out = {}
    fields = ("name", "compute_cap", "power.limit", "enforced.power.limit")
    try:
        row = subprocess.check_output(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                                      text=True, timeout=20).strip().splitlines()[0].split(",")
    except Exception as e:  # noqa: BLE001 -- a missing fact VOIDs in the reducer
        return {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    for f, v in zip(fields, (x.strip() for x in row)):
        key = f.replace(".", "_")
        if f in ("power.limit", "enforced.power.limit"):
            try:
                v = float(v)
            except ValueError:
                v = None
        out[key] = v
    return out


# ---------------------------------------------------------------------------------------------- the int8 records --
def _int8_mods(model):
    try:
        from experts4bit_qlora.engines.int8_attn import Int8Linear
    except ImportError:
        return []
    return [m for m in model.modules() if isinstance(m, Int8Linear)]


def path_counts(mods) -> dict:
    """``{"routed": {rows: calls}, "passed": {rows: calls}}`` summed over the int8 projections (calls, not modules)."""
    out = {"routed": {}, "passed": {}}
    for m in mods:
        for path in ("routed", "passed"):
            for r, c in getattr(m, path).items():
                out[path][str(r)] = out[path].get(str(r), 0) + c
    return out


def counts_delta(after, before) -> dict:
    return {p: {r: c - before[p].get(r, 0) for r, c in after[p].items() if c - before[p].get(r, 0)} for p in after}


class CaptureCensus:
    """Per captured bucket, the int8 projections' calls during its warm-up and capture (Python runs there only). Wraps
    ``PagedModelRunner._capture_bucket`` on the class, before ``build_engine`` captures."""

    def __init__(self):
        self.by_bucket = {}

    def __enter__(self):
        from experts4bit_qlora.engines import paged_runner as pr
        self.cls, self.orig = pr.PagedModelRunner, pr.PagedModelRunner._capture_bucket
        outer = self

        def cap(runner, b, warmup):
            mods = _int8_mods(runner.model)
            before = path_counts(mods)
            try:
                return outer.orig(runner, b, warmup)
            finally:
                outer.by_bucket[str(b)] = counts_delta(path_counts(mods), before)
        self.cls._capture_bucket = cap
        return self

    def __exit__(self, *exc):
        self.cls._capture_bucket = self.orig
        return False


# ------------------------------------------------------------------------------------------- the schedule records --
class DecodeCensus:
    """Per decode call, its live rows ``n`` and the bucket the runner's own ``bucket_for`` gives them (``"chunked"`` above
    the largest): ``{"n:b": calls}``. Wraps the runner's ``run_decode`` (the scheduler's one decode entry with the
    lookahead at its default, off) in every arm alike, so the dict increment it adds cancels in each I/D pair."""

    def __init__(self, runner):
        from experts4bit_qlora.engines.paged_runner import bucket_for
        self.r, self.bucket_for, self.calls = runner, bucket_for, {}
        self.orig = runner.run_decode
        runner.run_decode = self.decode

    def decode(self, rids):
        n = len(rids)
        b = self.bucket_for(n, self.r._buckets) if n <= self.r._buckets[-1] else "chunked"
        k = f"{n}:{b}"
        self.calls[k] = self.calls.get(k, 0) + 1
        return self.orig(rids)

    def take(self) -> dict:
        out, self.calls = dict(sorted(self.calls.items())), {}
        return out

    def restore(self):
        self.r.run_decode = self.orig


def graph_stats(runner) -> dict:
    gs = getattr(runner, "graph_stats", None)
    return {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else {}


def stats_delta(after, before) -> dict:
    return {b: {k: v - (before.get(b) or {}).get(k, 0) for k, v in s.items()} for b, s in after.items()}


def library_split_plans(kv, buckets, sms) -> dict:
    """REPORTED: the split count grouped-nf4-gemm's ``decode_split_plan`` gives each bucket with no cap, by the same
    arguments the KV's own engagement record passes it (``Fp8PagedKV._record_split``)."""
    try:
        import fp8_paged_attn
        plan_fn = fp8_paged_attn.decode_split_plan
    except (ImportError, AttributeError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    cols = int(kv.block_table[0].shape[-1])
    return {str(b): plan_fn(int(b), kv.Hs[0], cols, block_tokens=kv.bt, ktile=64, compute="fp8", pack_heads=False, sms=sms)
            for b in buckets}


def _common(cfg, parts, arm, load_s, census):
    import torch
    info = parts.info
    runner = parts.runner
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    return {"arm": arm, "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
            "model": cfg.model, "revision": cfg.revision,
            "knobs": {k: os.environ.get(k) for k in (INT8, SPLIT, BUCKETS) + DEFAULT_KNOBS},
            "torch": torch.__version__, "load_s": round(load_s, 2), "card": card(),
            "cc": ".".join(str(x) for x in torch.cuda.get_device_capability()), "sm_count": sms,
            "fusions": {k: info.get(k) for k in CENSUS_KEYS}, "fusion_modes": info.get("fusion_modes"),
            "model_type": info.get("model_type"), "attn_int8": info.get("attn_int8"),
            "attn_int8_projections": info.get("attn_int8_projections"), "levers_env": info.get("levers_env"),
            "graph_status": ({str(k): v for k, v in info["graph_status"].items()} if info.get("graph_status") else None),
            "capture_paths": census.by_bucket, "max_seqs": cfg.max_seqs, "max_tokens_per_seq": cfg.max_tokens_per_seq,
            "buckets": list(cfg.buckets), "buckets_requested": getattr(cfg, "buckets_requested", None),
            "runner_buckets": list(runner._buckets), "attn_split_cap": info.get("attn_split_cap"),
            "split_cap_requested": getattr(cfg, "attn_split_cap", None),
            "library_split_plans": library_split_plans(runner.kv, runner._buckets, sms),
            "int8_modules": len(_int8_mods(runner.model))}


# -------------------------------------------------------------------------------------------------------- speed --
def speed_main(a) -> int:
    import p109_box  # noqa: E402  (staged at P109's registered bytes: prompts, run_pass, slope, digest, _mem, _smi)
    arm = _arm()
    if a.tag not in TAGS or a.tag[:2] != arm:
        raise SystemExit(f"REFUSED: tag {a.tag} is not a registered process of arm {arm}")
    pf = json.load(open(a.prompts))
    rows = pf["rows"]
    if p109_box.digest(rows) != pf["prompts_sha256"] or len(rows) != p109_box.ROWS:
        raise SystemExit("REFUSED: prompts.json does not match its own digest")
    import torch
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    why = server_refusal(cfg, arm)
    if why:
        raise SystemExit(f"REFUSED: {why}")
    t0 = time.perf_counter()
    with CaptureCensus() as census:
        parts = build_engine(cfg)
    load_s = time.perf_counter() - t0
    rec = {"mode": "speed", "tag": a.tag, **_common(cfg, parts, arm, load_s, census),
           "prompts_sha256": pf["prompts_sha256"], "short": a.short, "long": a.long, "reps": a.reps,
           "mem_after_load": p109_box._mem(torch), "workloads": {}, "phases": {}, "status": "ok"}
    rec["phases"]["load"] = [round(t0, 3), round(time.perf_counter(), 3)]
    if (arm[1] == "1") != bool(rec["int8_modules"]):
        raise SystemExit(f"REFUSED: arm {arm} built {rec['int8_modules']} int8 projections")
    runner = parts.runner
    dc = DecodeCensus(runner)
    gs_build = graph_stats(runner)
    for wname, b in p109_box.WORKLOADS.items():
        wrows = rows[:b]
        walls, digests, rep_tokens, warm_tokens, calls, gstats = {}, {}, {}, {}, {}, {}
        tw = time.time()
        heats = {}
        for n in (a.short, a.long):
            g0 = graph_stats(runner)
            dc.take()
            _w, _s, warm_tokens[str(n)] = p109_box.run_pass(parts, torch, wrows, n)       # warm, untimed
            ws, ds, ts = [], [], []
            heats[str(n)] = {"start": heat()}
            for _ in range(a.reps):
                wall, _steps, toks = p109_box.run_pass(parts, torch, wrows, n)
                ws.append(round(wall, 5))
                ds.append(p109_box.digest(toks))
                ts.append(toks)
            heats[str(n)]["end"] = heat()
            walls[str(n)], digests[str(n)], rep_tokens[str(n)] = ws, ds, ts
            calls[str(n)] = dc.take()                       # the warm pass and the five timed reps
            gstats[str(n)] = stats_delta(graph_stats(runner), g0)
        s = p109_box.slope(walls[str(a.short)], walls[str(a.long)], b, a.short, a.long)
        rec["workloads"][wname] = {"batch": b, "walls": walls, "rep_digests": digests, "rep_tokens": rep_tokens,
                                   "warm_tokens": warm_tokens, "tokens": {k: v[-1] for k, v in rep_tokens.items()},
                                   "decode_calls": calls, "graph_stats": gstats, "heat": heats, **s}
        rec["phases"][wname] = [round(tw, 3), round(time.time(), 3)]
        print(f"P223_W {arm}/{a.tag} {wname} B={b} decode_tok_s={s['decode_tok_s']} walls={walls}", flush=True)
    dc.restore()
    rec["graph_stats_build"] = gs_build
    rec["graph_stats"] = graph_stats(runner)
    rec["paths_after_reps"] = path_counts(_int8_mods(runner.model))
    rec["mem_after_runs"] = p109_box._mem(torch)
    rec["nvidia_smi"] = p109_box._smi()
    rec["card_after"] = card()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P223_ARM " + json.dumps({"arm": arm, "tag": a.tag, "load_s": rec["load_s"], "int8": rec["int8_modules"],
                                    "buckets": rec["runner_buckets"], "split": rec["attn_split_cap"],
                                    "W16": rec["workloads"]["W16"]["decode_tok_s"],
                                    "W1": rec["workloads"]["W1"]["decode_tok_s"]}, default=str), flush=True)
    return 0


def prompts_main(a) -> int:
    import p109_box
    return p109_box.prompts_main(a)


# -------------------------------------------------------------------------------------------------------- tests --
def _census_cases(torch) -> list:
    """DecodeCensus on a stand-in runner: it records each call's rows and bucket and hands the call through unchanged."""
    try:
        import experts4bit_qlora.engines.paged_runner  # noqa: F401  (DecodeCensus reads its bucket_for)
    except ImportError:
        return []

    class R:
        def __init__(self, buckets):
            self._buckets, self.seen = buckets, []

        def run_decode(self, rids):
            self.seen.append(list(rids))
            return {r: 7 for r in rids}
    out = []
    r = R((2, 4, 8, 16))
    dc = DecodeCensus(r)
    got = [r.run_decode(x) for x in ([1], [1, 2], [1, 2, 3], list(range(16)), [5])]
    out.append(("a lone row is bucket 2's under the I buckets; the calls pass through",
                dc.take() == {"16:16": 1, "1:2": 2, "2:2": 1, "3:4": 1} and got[0] == {1: 7} and len(r.seen) == 5))
    dc.restore()
    r.run_decode([1])
    out.append(("restore puts the runner's own method back", dc.calls == {} and len(r.seen) == 6))
    r = R((1, 2, 4, 8, 16))
    dc = DecodeCensus(r)
    r.run_decode([1])
    out.append(("a lone row is bucket 1's under the D buckets", dc.take() == {"1:1": 1}))
    return out


def self_test() -> int:
    cases = []
    i0 = {SPLIT: "1", BUCKETS: "2,4,8,16"}
    i1 = {**i0, INT8: "1"}
    cases.append(("D0 is the default server", arm_env_ok("D0", {})[0] and not arm_env_ok("D0", {SPLIT: "1"})[0]))
    cases.append(("I0 is the two knobs", arm_env_ok("I0", i0)[0] and not arm_env_ok("I0", {SPLIT: "1"})[0]
                  and not arm_env_ok("I0", {**i0, BUCKETS: "1,2,4,8,16"})[0]))
    cases.append(("D1 is the int8 lever alone", arm_env_ok("D1", {INT8: "1"})[0] and not arm_env_ok("D1", {**i0, INT8: "1"})[0]))
    cases.append(("I1 is the lever and both knobs", arm_env_ok("I1", i1)[0] and not arm_env_ok("I1", i0)[0]))
    cases.append(("another serving knob refuses", not arm_env_ok("I1", {**i1, "E4B_DENSE_GEMV_BW": "1"})[0]
                  and not arm_env_ok("D0", {"E4B_NF4_GROUPED_SMALLM": "1"})[0]))
    cases.append(("the lookahead stays at its default", not arm_env_ok("D0", {"E4B_PAGED_DECODE_LOOKAHEAD": "1"})[0]))
    cases.append(("an unknown arm refuses", not arm_env_ok("S1", {INT8: "1"})[0]))
    cases.append(("the process order is two ABBA blocks", TAGS == ("D0a", "I0a", "I1a", "D1a", "D1b", "I1b", "I0b", "D0b")))

    class C:
        graphs, max_seqs, max_tokens_per_seq, buckets, placement = True, 16, 768, (1, 2, 4, 8, 16), "all-vram"
        attn_split_cap = "default"
    cases.append(("the registered D server passes", server_refusal(C, "D0") is None and server_refusal(C, "D1") is None))
    cases.append(("the D server refuses as an I arm", server_refusal(C, "I0") is not None))
    C.buckets, C.attn_split_cap = (2, 4, 8, 16), "1"
    cases.append(("the registered I server passes", server_refusal(C, "I1") is None))
    C.attn_split_cap = "default"
    cases.append(("an I server without the cap refuses", server_refusal(C, "I0") is not None))
    C.attn_split_cap, C.max_tokens_per_seq = "1", 4096
    cases.append(("4096 tokens per sequence refuses", server_refusal(C, "I0") is not None))
    C.max_tokens_per_seq, C.graphs = 768, False
    cases.append(("an eager server refuses", server_refusal(C, "I0") is not None))
    a = {"routed": {"1": 4, "16": 2}, "passed": {"511": 8}}
    b = {"routed": {"1": 4, "16": 1}, "passed": {}}
    cases.append(("deltas keep only what moved", counts_delta(a, b) == {"routed": {"16": 1}, "passed": {"511": 8}}))
    cases.append(("graph-stat deltas per bucket", stats_delta({"2": {"replays": 9, "pad_rows": 4}}, {"2": {"replays": 2}})
                  == {"2": {"replays": 7, "pad_rows": 4}}))
    try:
        import torch
        cases += _census_cases(torch)
    except ImportError:
        pass
    bad = [c for c, ok in cases if not ok]
    print(f"p223_box self-test: {len(cases) - len(bad)}/{len(cases)} OK" + (f"; FAILED {bad}" if bad else ""))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("speed",))
    ap.add_argument("--prompts")
    ap.add_argument("--out")
    ap.add_argument("--tag")
    ap.add_argument("--short", type=int, default=32)
    ap.add_argument("--long", type=int, default=160)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--prompts-only", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--revision")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.prompts_only:
        return prompts_main(a)
    if a.mode == "speed":
        return speed_main(a)
    ap.error("nothing to do")
    return 2


if __name__ == "__main__":
    sys.exit(main())
