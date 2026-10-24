#!/usr/bin/env python3
"""p232_box.py -- lane P232, the served read of the int8 lm_head at 2, 4 and 8 rows (bench/p232/PREREG-p232.md; lever K,
K2b): ONE process per arm or phase. P153's box (bench/p153/p153_box.py) by counted edits: this lane's arms and knob, P109's
workloads at W2, W4 and W8 in place of W16 and W1, and the quality read at group T = 2, 4 and 8 in place of SANE's T == 1.

The engine is the shipped default server (``PagedServeConfig.from_env()`` + ``build_engine(cfg)``) at grouped-nf4-gemm
v0.50.0 (K36 and K37 as released): the B=1 fused stack on (auto, qwen3_moe allowlisted), the bandwidth decode GEMV on
(P116), the attention split cap at its qwen3_moe default (16), the int8 head on by its qwen3_moe default (P153),
``max_seqs`` 16 named by the runner, 4096 tokens per sequence (the default), the dense attention GEMV knob off. The arms
differ ONLY in:
  R    ``E4B_INT8_HEAD_ROWS`` unset (today: 2..16 head rows run ``F.linear`` on the rounded grid; one row K36);
  K    ``E4B_INT8_HEAD_ROWS=8`` (2..8 head rows run K37's ``dense_i8.linear_i8`` over the same codes and scales);
  B16  ``E4B_INT8_HEAD_BW=0`` (the bf16 head, nothing swapped), quality only: the reported bf16 comparison.

``--mode quality``: Phase B's instrument (``p115_quality.measure_phase``) on the default server built eager, at group
T = 2, 4 and 8 in turn in ONE process (T windows decoded together per pass, so every decode step routes T rows through
the head), 24 wikitext windows x 128 positions, every KV built inside the measurement at the server's 4096 tokens with
the server's cap (A2's ``ServedKV``). R scores R with measure_phase's ``rep`` floor draw and saves its log-probs per T
(``ref_dir/T<T>``); K and B16 score ON against them. Per T, beside the measurement: the head's calls during it (read on
the head itself, by row count), its engagement, the cap's plans. After every T's records: per-row bucket stability
(the head's output for the saved T-row decode inputs at bucket T against the same rows at bucket 2), reported only.

``--mode speed`` (the rule's g_T): P109's run_pass, slope, digest and token records at W2, W4 and W8 (the first 2, 4 or 8
of P109's prompt rows added at once, so every decode step runs bucket T), with the head's engagement at the build
(``info["int8_head_bw"]``), per workload the graph replays and the head's eager calls, the cap's plans, the GEMV tally,
and the card's telemetry read either side of every timed block, never during one.

    python p232_box.py --mode speed --prompts prompts.json --out arm_Ra.json --tag Ra   (arm from P232_ARM)
    python p232_box.py --mode quality --out quality_R.json --ref-dir work/ref            (P232_ARM=R / K / B16)
    python p232_box.py --prompts-only --model ID --revision SHA --out prompts.json
    python p232_box.py --self-test
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p109_box  # noqa: E402  (staged at P109's registered bytes: prompts, run_pass, slope, digest, _mem, _smi)

ARMS = ("R", "K", "B16")
SPEED_ARMS = ("R", "K")
ROWS_KNOB = "E4B_INT8_HEAD_ROWS"
HEAD_KNOB = "E4B_INT8_HEAD_BW"           # unset in R and K (on by its qwen3_moe default, P153); B16 sets 0
#: per arm, the value each of the two knobs must carry (None: unset)
ARM_VALUE = {"R": {ROWS_KNOB: None, HEAD_KNOB: None}, "K": {ROWS_KNOB: "8", HEAD_KNOB: None},
             "B16": {ROWS_KNOB: None, HEAD_KNOB: "0"}}
DENSE_KNOB = "E4B_DENSE_GEMV_BW"         # unset in every arm: the dense attention GEMV off (its own read is P151)
CAP_KNOB = "E4B_ATTN_SPLIT_CAP"           # unset in every arm: the cap at its qwen3_moe default (16, e4b#1792)
SPEC_KNOB = "E4B_PAGED_SPEC"              # unset: K2b is refused with speculative decode
FUSION_KNOBS = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
GEMV_KNOBS = ("GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE")
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")
WORKLOADS = {"W2": 2, "W4": 4, "W8": 8}   # the rows the knob routes; W1 and W16 are not touched by it
GROUPS = (2, 4, 8)                        # the quality read's T, each on its own
QUALITY_PHASE_ARMS = {"R": ("R", "rep"), "K": ("ON",), "B16": ("ON",)}
KEEP_STEPS = 2                            # T-row decode inputs saved per T for the bucket-stability report
TELEMETRY = ("power.limit", "clocks.max.sm", "temperature.gpu", "clocks.sm", "power.draw")


def arm_env_ok(arm: str, env, *, speed=False):
    """``(ok, why)``: the two head knobs carry the arm's values (unset where None); the split cap, the dense attention
    GEMV, speculative decode, fusion and decode GEMV knobs are unset (their defaults). B16 runs in the quality phases
    only."""
    if arm not in (SPEED_ARMS if speed else ARMS):
        return False, f"unknown arm {arm!r}" + (" (B16 is quality-only)" if arm == "B16" else "")
    bad = {}
    for k, want in ARM_VALUE[arm].items():
        got = (env.get(k) or "").strip()
        if got != (want or ""):
            bad[k] = env.get(k)
    bad.update({k: env.get(k) for k in (CAP_KNOB, DENSE_KNOB, SPEC_KNOB) + FUSION_KNOBS + GEMV_KNOBS
                if (env.get(k) or "").strip()})
    return (not bad), f"{arm} with {bad}"


def _arm(speed=False):
    arm = os.environ.get("P232_ARM", "")
    ok, why = arm_env_ok(arm, os.environ, speed=speed)
    if not ok:
        raise SystemExit(f"REFUSED: P232_ARM={arm!r}: {why}")
    return arm


def _counts():
    import nf4_grouped
    return dict(nf4_grouped.dispatch_counts())


def _delta(after, before):
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


def head_calls(head) -> dict:
    """The head's own call counts by row count (``Int8HeadLinear.routed`` / ``.passed``); a bf16 head has none."""
    return {"routed": {str(k): v for k, v in sorted((getattr(head, "routed", None) or {}).items())},
            "passed": {str(k): v for k, v in sorted((getattr(head, "passed", None) or {}).items())}}


def calls_delta(after: dict, before: dict) -> dict:
    """The calls between two :func:`head_calls` reads, zeros dropped."""
    out = {}
    for side in ("routed", "passed"):
        a, b = after.get(side) or {}, before.get(side) or {}
        out[side] = {k: a[k] - b.get(k, 0) for k in a if a[k] - b.get(k, 0)}
    return out


def graph_snapshot(parts) -> dict:
    gs = getattr(getattr(parts, "runner", None), "graph_stats", None)
    return {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else {}


def graph_delta(after: dict, before: dict) -> dict:
    """Per bucket, the replays and eager steps between two :func:`graph_snapshot` reads."""
    return {b: {k: v.get(k, 0) - (before.get(b) or {}).get(k, 0) for k in ("replays", "eager_steps")}
            for b, v in after.items()}


def parse_telemetry(line: str) -> dict:
    """One ``nvidia-smi --query-gpu=<TELEMETRY> --format=csv,noheader,nounits`` line as floats; an unreadable field
    (``[N/A]``, missing) is None."""
    vals = [v.strip() for v in (line or "").split(",")]
    out = {}
    for i, k in enumerate(TELEMETRY):
        try:
            out[k] = float(vals[i])
        except (IndexError, ValueError):
            out[k] = None
    return out


def _telemetry() -> dict:
    """The card now: power limit, max SM clock, temperature, SM clock and draw. Read between timed blocks, never during
    one (a sampler would share the host with the timed loop)."""
    try:
        line = subprocess.check_output(["nvidia-smi", "--query-gpu=" + ",".join(TELEMETRY), "--format=csv,noheader,nounits"],
                                       text=True, timeout=20).splitlines()[0]
    except Exception:  # noqa: BLE001 -- an unreadable card reads None in every field; the reducer VOIDs on the limit
        line = ""
    return parse_telemetry(line)


def _engagement(parts) -> dict:
    """The knob's engagement now (engines/int8_head_bw.py): per row count, whether the swapped head routed or passed,
    and with ``E4B_INT8_HEAD_ROWS`` above 1 which kernel each routed row count took."""
    from experts4bit_qlora.engines.int8_head_bw import engagement
    return engagement(parts.runner.model)


def _sms():
    import torch
    return torch.cuda.get_device_properties(0).multi_processor_count


def _common(cfg, parts, arm, load_s, shake):
    import torch
    info = parts.info
    return {"arm": arm, "shakedown": bool(shake), "e4b_sha": os.environ.get("E4B_SHA"),
            "gnf4_sha": os.environ.get("GNF4_SHA"), "model": cfg.model, "revision": cfg.revision,
            "knobs": {k: os.environ.get(k) for k in (ROWS_KNOB, HEAD_KNOB, CAP_KNOB, DENSE_KNOB, SPEC_KNOB) + FUSION_KNOBS
                      + GEMV_KNOBS},
            "torch": torch.__version__, "sm_count": _sms(), "gpu": torch.cuda.get_device_name(0), "load_s": round(load_s, 2),
            "fusions": {k: info.get(k) for k in CENSUS_KEYS}, "fusion_modes": info.get("fusion_modes"),
            "fusion_sources": info.get("fusion_sources"), "attn_split_cap": info.get("attn_split_cap"),
            "dense_gemv_bw": info.get("dense_gemv_bw"), "int8_head_bw": info.get("int8_head_bw"),
            "model_type": info.get("model_type"), "max_tokens_per_seq": cfg.max_tokens_per_seq,
            "graph_status": ({str(k): v for k, v in info["graph_status"].items()} if info.get("graph_status") else None),
            "grouping": info.get("grouping"), "max_seqs": cfg.max_seqs, "buckets": list(cfg.buckets)}


class ServedKV:
    """Inside the block, every ``Fp8PagedKV`` is built at ``tokens`` per sequence and carries ``cap`` (None = the
    library's plan), as ``build_engine`` builds and sets the server's own. ``built`` keeps one entry per KV: the
    capacity its caller asked for, the one it got, and its ``split_plans`` (the dict itself, which the KV fills in
    place) -- never the KV, whose pool is freed with its pass."""

    def __init__(self, tokens: int, cap):
        self.tokens, self.cap, self.built = int(tokens), cap, []

    def __enter__(self):
        from experts4bit_qlora.engines import fp8_paged_kv as m
        self.cls, self.orig = m.Fp8PagedKV, m.Fp8PagedKV.__init__
        outer = self

        def init(kv, *a, **kw):
            if "max_tokens_per_seq" not in kw:
                raise SystemExit("REFUSED: a KV built without a keyword max_tokens_per_seq; its capacity cannot be served")
            asked = kw["max_tokens_per_seq"]
            if asked > outer.tokens:
                raise SystemExit(f"REFUSED: a KV asked for {asked} tokens, above the server's {outer.tokens}")
            kw["max_tokens_per_seq"] = outer.tokens
            outer.orig(kv, *a, **kw)
            kv.split_cap = outer.cap
            outer.built.append({"asked": asked, "max_tokens_per_seq": outer.tokens, "blocks_per_seq": kv.blocks_per_seq,
                                "split_cap": outer.cap, "plans": kv.split_plans})
        self.cls.__init__ = init
        return self

    def __exit__(self, *exc):
        self.cls.__init__ = self.orig
        return False

    def report(self) -> dict:
        """``(plans, summary)``: the plans merged over every KV (calls summed; plan and split count must agree), and
        the capacities."""
        plans, clash = {}, []
        for b in self.built:
            for B, r in b["plans"].items():
                m = plans.setdefault(str(B), {"calls": 0, "plan": r.get("plan"), "n_split": r.get("n_split")})
                m["calls"] += r.get("calls", 0)
                if (m["plan"], m["n_split"]) != (r.get("plan"), r.get("n_split")):
                    clash.append({"B": B, "first": [m["plan"], m["n_split"]], "then": [r.get("plan"), r.get("n_split")]})
        return {"passes": len(self.built), "asked": sorted({b["asked"] for b in self.built}),
                "max_tokens_per_seq": sorted({b["max_tokens_per_seq"] for b in self.built}),
                "blocks_per_seq": sorted({b["blocks_per_seq"] for b in self.built}),
                "split_cap": self.cap, "plans": plans, "plan_clash": clash}


class HeadRowsKeep:
    """While open at ``rows``, saves (on the host) the first :data:`KEEP_STEPS` inputs of exactly ``rows`` rows that the
    output head sees: the eager decode's T-row steps, for :func:`bucket_stability` after every record is written."""

    def __init__(self, head, keep=KEEP_STEPS):
        self.head, self.keep, self.saved, self.rows, self.hook = head, keep, {}, None, None

    def at(self, rows):
        self.rows = int(rows)
        return self

    def __enter__(self):
        def pre(mod, args):
            x = args[0]
            n = x.numel() // mod.in_features if mod.in_features else 0
            got = self.saved.setdefault(self.rows, [])
            if n == self.rows and len(got) < self.keep:
                got.append(x.detach().reshape(n, mod.in_features).to("cpu", copy=True))
        self.hook = self.head.register_forward_pre_hook(pre)
        return self

    def __exit__(self, *exc):
        self.hook.remove()
        return False


def bucket_stability(head, saved: dict) -> dict:
    """Reported, never ruled (DESIGN-K2.md section 4: a property, not a contract): per T, each saved T-row input through
    the head at once (bucket T) against the same rows two at a time (bucket 2), row by row -- bitwise identical rows
    out of the rows, and the largest absolute difference. Run after the measurement's records: these calls add to the
    head's counts."""
    import torch
    out = {}
    dev = head.weight.device
    with torch.no_grad():
        for T, xs in sorted(saved.items()):
            same, rows, worst = 0, 0, 0.0
            for x in xs:
                x = x.to(dev)
                y_t = head(x)
                y_2 = torch.cat([head(x[i:i + 2]) for i in range(0, T, 2)])
                for i in range(T):
                    rows += 1
                    same += int(torch.equal(y_t[i], y_2[i]))
                worst = max(worst, float((y_t.float() - y_2.float()).abs().max()))
            out[str(T)] = {"rows_identical": same, "rows": rows, "max_abs_diff": worst, "steps": len(xs)}
    return out


def speed_main(a) -> int:
    arm = _arm(speed=True)
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
    c0 = _counts()
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    load_s = time.perf_counter() - t0
    c_build = _counts()
    head = parts.runner.model.get_output_embeddings()
    rec = {"mode": "speed", "tag": a.tag, **_common(cfg, parts, arm, load_s, a.shakedown), "dispatch_build": _delta(c_build, c0),
           "prompts_sha256": pf["prompts_sha256"], "short": a.short, "long": a.long, "reps": a.reps,
           "mem_after_load": p109_box._mem(torch), "telemetry_after_load": _telemetry(), "workloads": {}, "status": "ok"}
    for wname, b in WORKLOADS.items():
        wrows = rows[:b]
        walls, digests, last, telem = {}, {}, {}, {}
        cw, gw, hw = _counts(), graph_snapshot(parts), head_calls(head)
        for n in (a.short, a.long):
            p109_box.run_pass(parts, torch, wrows, n)                      # warm, untimed
            ws, ds = [], []
            before = _telemetry()                                          # either side of the timed block
            for _ in range(a.reps):
                wall, _steps, toks = p109_box.run_pass(parts, torch, wrows, n)
                ws.append(round(wall, 5))
                ds.append(p109_box.digest(toks))
                last[str(n)] = toks
            walls[str(n)], digests[str(n)] = ws, ds
            telem[str(n)] = {"before": before, "after": _telemetry()}
        s = p109_box.slope(walls[str(a.short)], walls[str(a.long)], b, a.short, a.long)
        rec["workloads"][wname] = {"batch": b, "walls": walls, "rep_digests": digests, "tokens": last,
                                   "dispatch": _delta(_counts(), cw), "graph_delta": graph_delta(graph_snapshot(parts), gw),
                                   "head_calls": calls_delta(head_calls(head), hw), "telemetry": telem, **s}
        print(f"P232_W {arm}/{a.tag} {wname} B={b} decode_tok_s={s['decode_tok_s']} walls={walls} "
              f"telemetry={telem}", flush=True)
    rec["graph_stats"] = graph_snapshot(parts) or None
    rec["attn_split_cap_after_runs"] = {"plans": {str(b): dict(r) for b, r in sorted(parts.runner.kv.split_plans.items())}} \
        if getattr(getattr(parts, "runner", None), "kv", None) is not None else None
    rec["int8_head_after_runs"] = _engagement(parts)
    rec["dispatch_total"] = _delta(_counts(), c0)
    rec["mem_after_runs"] = p109_box._mem(torch)
    rec["nvidia_smi"] = p109_box._smi()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P232_ARM " + json.dumps({"arm": arm, "tag": a.tag, "rows": (rec["int8_head_bw"] or {}).get("rows"),
                                      "load_s": rec["load_s"],
                                      **{w: rec["workloads"][w]["decode_tok_s"] for w in WORKLOADS}}, default=str), flush=True)
    return 0


def quality_main(a) -> int:
    arm = _arm()
    if any(a.windows % T for T in GROUPS):
        raise SystemExit(f"REFUSED: --windows {a.windows} does not divide by every T in {GROUPS}")
    import torch
    import transformers

    import fp8_paged_attn
    import p115_quality as q
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if cfg.graphs or cfg.placement != "all-vram" or cfg.max_tokens_per_seq != 4096:
        raise SystemExit(f"REFUSED: the quality read builds the default server eager at all-vram and 4096 tokens per "
                         f"sequence (graphs={cfg.graphs}, max_tokens_per_seq={cfg.max_tokens_per_seq})")
    counters = q.KernelCounters().install()                          # before build_engine, as Phase B
    c0 = _counts()
    t0 = time.time()
    parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    load_s = time.time() - t0
    fwd = q.ForwardCounter(model)
    windows = {"wikitext": q.LOADERS["wikitext"](parts.tokenizer, a.windows, a.prompt, a.cont)}
    common = _common(cfg, parts, arm, load_s, a.shakedown)
    cap = (common["attn_split_cap"] or {}).get("cap")
    blocks = -(-cfg.max_tokens_per_seq // 16)
    head = model.get_output_embeddings()
    keep = HeadRowsKeep(head)
    phases = {}
    c_build = _counts()
    for T in GROUPS:
        plan_t = fp8_paged_attn.decode_split_plan(T, 4, blocks, block_tokens=16, ktile=64, compute="fp8", pack_heads=False,
                                                  sms=_sms())
        c1, h1 = _counts(), head_calls(head)
        with ServedKV(cfg.max_tokens_per_seq, cap) as served, keep.at(T):
            rec = q.measure_phase(model, windows, phase="off" if arm == "R" else "on", prompt=a.prompt, cont=a.cont,
                                  chunk=a.chunk, floor_chunk=a.chunk, group=T, device=cfg.device,
                                  ref_dir=os.path.join(a.ref_dir, f"T{T}"), counters=counters, fwd=fwd,
                                  arms=QUALITY_PHASE_ARMS[arm])
        sk = served.report()
        phases[str(T)] = {**rec, "attn_split_cap": {"cap": cap, "source": (common["attn_split_cap"] or {}).get("source"),
                                                    "plans": sk["plans"]},
                          "quality_kv": {**{k: v for k, v in sk.items() if k != "plans"}, "plan_bT": plan_t},
                          "head_calls": calls_delta(head_calls(head), h1), "int8_head_bw_measure": _engagement(parts),
                          "dispatch_measure": _delta(_counts(), c1)}
        print(f"P232_QT {arm} T={T} windows={rec['windows']} cap={cap} head_calls={phases[str(T)]['head_calls']} "
              f"quality_kv={phases[str(T)]['quality_kv']} plans={sk['plans']} {rec['seconds']} s", flush=True)
    out = {"mode": "quality", **common, "transformers": transformers.__version__, "dispatch_build": _delta(c_build, c0),
           "phases": phases, "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "status": "ok"}
    out["bucket_stability"] = bucket_stability(head, keep.saved)       # after every T's records: reported only
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    print(f"P232_QUALITY {arm} groups={list(phases)} stability={out['bucket_stability']} "
          f"head={(common['int8_head_bw'] or {}).get('rows')}", flush=True)
    return 0


def _self_test_served_kv() -> list:
    """The patch on a stand-in class: the capacity raised to the server's, the cap set, the plans dict held."""
    import types
    from contextlib import contextmanager

    class FakeKV:
        def __init__(self, layers, h, d, *, batch, max_tokens_per_seq, device="cpu", scratch_slots=0):
            self.blocks_per_seq = -(-max_tokens_per_seq // 16)
            self.split_cap = None
            self.split_plans = {}

    @contextmanager
    def fake_module():
        mods = {n: sys.modules.get(n) for n in ("experts4bit_qlora", "experts4bit_qlora.engines",
                                                 "experts4bit_qlora.engines.fp8_paged_kv")}
        root, eng, mod = (types.ModuleType(n) for n in mods)
        mod.Fp8PagedKV = FakeKV
        eng.fp8_paged_kv = mod
        root.engines = eng
        sys.modules.update(dict(zip(mods, (root, eng, mod))))
        try:
            yield
        finally:
            for n, m in mods.items():
                if m is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = m
    out = []
    with fake_module():
        orig = FakeKV.__init__
        with ServedKV(4096, 16) as s:
            kv = FakeKV(48, 4, 128, batch=4, max_tokens_per_seq=656)
            kv.split_plans.setdefault(4, {"calls": 0, "plan": 23, "n_split": 16})["calls"] += 48
            kv2 = FakeKV(48, 4, 128, batch=4, max_tokens_per_seq=656)
            kv2.split_plans[4] = {"calls": 48, "plan": 23, "n_split": 16}
            try:
                FakeKV(48, 4, 128, batch=4, max_tokens_per_seq=8192)
                refused = False
            except SystemExit:
                refused = True
        r = s.report()
        out += [("the pass's KV built at the server's capacity, with the cap", (kv.blocks_per_seq, kv.split_cap) == (256, 16)),
                ("the plans merged over passes", r["plans"] == {"4": {"calls": 96, "plan": 23, "n_split": 16}}
                 and r["passes"] == 2 and r["asked"] == [656] and r["max_tokens_per_seq"] == [4096] and not r["plan_clash"]),
                ("a KV above the server's capacity is refused", refused),
                ("the class is restored on exit", FakeKV.__init__ is orig)]
    return out


def _self_test_head() -> list:
    """The head's call reads and the bucket-stability report on CPU stand-ins (torch CPU only)."""
    import torch
    import torch.nn as nn

    class Counted(nn.Linear):
        def __init__(self, k, n):
            super().__init__(k, n, bias=False)
            self.routed, self.passed = {}, {}

        def forward(self, x):
            rows = x.numel() // self.in_features
            d = self.routed if rows <= 8 else self.passed
            d[rows] = d.get(rows, 0) + 1
            return super().forward(x)

    class RowShift(nn.Linear):
        """A head whose row 0 reads differently at 8 rows than at 2: what the report must name."""

        def forward(self, x):
            y = super().forward(x)
            if x.shape[0] == 8:
                y = y.clone()
                y[0] += 1.0
            return y
    torch.manual_seed(232)

    def ints(*shape):                                    # small integers: exact in fp32 in any accumulation order
        return torch.randint(-3, 4, shape).float()
    h = Counted(16, 32)
    shift = RowShift(16, 32, bias=False)
    with torch.no_grad():
        h.weight.copy_(ints(32, 16))
        shift.weight.copy_(ints(32, 16))
    with torch.no_grad():
        h(ints(2, 16))                                   # a row count read before, never after: its zero is dropped
    before = head_calls(h)
    keep = HeadRowsKeep(h)
    with torch.no_grad():
        with keep.at(4):
            for _ in range(3):
                h(ints(4, 16))
            h(ints(512, 16))
        with keep.at(8):
            h(ints(1, 8, 16))
    d = calls_delta(head_calls(h), before)
    st = bucket_stability(h, keep.saved)
    st2 = bucket_stability(shift, {8: [ints(8, 16)]})
    return [("the head's calls by row count, zeros dropped", d == {"routed": {"4": 3, "8": 1}, "passed": {"512": 1}}),
            ("a bf16 head has no calls", head_calls(nn.Linear(4, 4)) == {"routed": {}, "passed": {}}),
            ("the hook keeps two T-row steps per T, nothing else",
             sorted(keep.saved) == [4, 8] and len(keep.saved[4]) == KEEP_STEPS and keep.saved[8][0].shape == (8, 16)),
            ("the hook is removed on exit", not h._forward_pre_hooks),
            ("a per-row head is bucket-stable", st["4"]["rows_identical"] == 8 and st["4"]["rows"] == 8
             and st["8"]["rows_identical"] == 8),
            ("a bucket-dependent row is named", st2["8"]["rows_identical"] == 7 and st2["8"]["max_abs_diff"] >= 1.0),
            ("graph replays per bucket between two reads",
             graph_delta({"2": {"replays": 900, "eager_steps": 1}}, {"2": {"replays": 100, "eager_steps": 1}})
             == {"2": {"replays": 800, "eager_steps": 0}})]


def self_test() -> int:
    k = {ROWS_KNOB: "8"}
    ok = [("arms", ARMS == ("R", "K", "B16") and SPEED_ARMS == ("R", "K")),
          ("the workloads", WORKLOADS == {"W2": 2, "W4": 4, "W8": 8} and GROUPS == (2, 4, 8)),
          ("P109's self-test", p109_box.self_test() == 0),
          ("R: both knobs unset", arm_env_ok("R", {})[0] and arm_env_ok("R", {}, speed=True)[0]),
          ("K", arm_env_ok("K", k, speed=True)[0]), ("K padded", arm_env_ok("K", {ROWS_KNOB: " 8 "})[0]),
          ("B16", arm_env_ok("B16", {HEAD_KNOB: "0"})[0]),
          ("B16 never in a speed arm", not arm_env_ok("B16", {HEAD_KNOB: "0"}, speed=True)[0]),
          ("R with the knob set is refused", not arm_env_ok("R", k)[0]),
          ("R with the row knob at 1 is refused (unset is the arm)", not arm_env_ok("R", {ROWS_KNOB: "1"})[0]),
          ("K at 4 is refused", not arm_env_ok("K", {ROWS_KNOB: "4"})[0]),
          ("K unset is refused", not arm_env_ok("K", {})[0]),
          ("K with the head named on is refused (the default is the subject)", not arm_env_ok("K", {**k, HEAD_KNOB: "1"})[0]),
          ("R with the head named on is refused", not arm_env_ok("R", {HEAD_KNOB: "1"})[0]),
          ("B16 with the row knob is refused", not arm_env_ok("B16", {HEAD_KNOB: "0", ROWS_KNOB: "8"})[0]),
          ("B16 unset is refused", not arm_env_ok("B16", {})[0]),
          ("the split cap set is refused (its default is the subject's)", not arm_env_ok("K", {**k, CAP_KNOB: "16"})[0]),
          ("the split cap set on R too", not arm_env_ok("R", {CAP_KNOB: "0"})[0]),
          ("the dense attention GEMV set is refused", not arm_env_ok("K", {**k, DENSE_KNOB: "1"})[0]),
          ("speculative decode set is refused", not arm_env_ok("R", {SPEC_KNOB: "1"})[0]),
          ("a fusion knob set is refused (the default is the subject)", not arm_env_ok("K", {**k, "E4B_FUSE_T1_GLUE": "auto"})[0]),
          ("the GEMV at its default, never forced", not arm_env_ok("K", {**k, "GNF4_GEMV_BW": "1"})[0]),
          ("an unknown arm", not arm_env_ok("S1", k)[0]),
          ("the quality phases", QUALITY_PHASE_ARMS == {"R": ("R", "rep"), "K": ("ON",), "B16": ("ON",)}),
          ("telemetry parses", parse_telemetry("575.00, 3090, 41, 2842, 80.12") == {
              "power.limit": 575.0, "clocks.max.sm": 3090.0, "temperature.gpu": 41.0, "clocks.sm": 2842.0, "power.draw": 80.12}),
          ("an unreadable field is None", parse_telemetry("[N/A], 3090, 41")["power.limit"] is None
           and parse_telemetry("[N/A], 3090, 41")["power.draw"] is None),
          ("no line reads None everywhere", set(parse_telemetry("").values()) == {None})]
    ok += _self_test_served_kv()
    ok += _self_test_head()
    bad = [n for n, x in ok if not x]
    print(f"p232_box self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(ok) - len(bad)}/{len(ok)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--prompts-only", action="store_true")
    p.add_argument("--mode", choices=("speed", "quality"))
    p.add_argument("--shakedown", action="store_true")
    p.add_argument("--model")
    p.add_argument("--revision", default="")
    p.add_argument("--prompts")
    p.add_argument("--out")
    p.add_argument("--tag", default="")
    p.add_argument("--short", type=int, default=32)
    p.add_argument("--long", type=int, default=160)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--ref-dir")
    p.add_argument("--windows", type=int, default=24)
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
    if a.mode == "quality":
        return quality_main(a)
    raise SystemExit("REFUSED: --mode speed|quality, --prompts-only or --self-test")


if __name__ == "__main__":
    sys.exit(main())
