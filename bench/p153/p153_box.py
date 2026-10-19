#!/usr/bin/env python3
"""p153_box.py -- lane P153, D2's served read (bench/p153/PREREG-p153.md; e4b#1313): ONE process per arm or phase. P151's
box (bench/p151/p151_box.py, itself P143 Amendment 2's by counted edits) by counted edits.

The engine is the shipped default server (``PagedServeConfig.from_env()`` + ``build_engine(cfg)``) at grouped-nf4-gemm
K36's merge (0.47.1 plus ``dense_gemv``'s bf16 and int8 GEMVs): the B=1 fused stack on (auto, qwen3_moe allowlisted),
the bandwidth decode GEMV on (P116), the attention split cap at its qwen3_moe default (16), ``max_seqs`` 16 named by the
runner, 4096 tokens per sequence (the default), the dense attention GEMV knob off. The arms differ ONLY in
``E4B_INT8_HEAD_BW``:
  S0  ``0``  (named: today's server, cuBLAS on the bf16 lm_head);
  S1  ``1``  (the lm_head rounded in place to its per-row int8 grid, P150's I8HEAD, so EVERY row count reads the grid;
            a single bf16 row goes to K36's ``gemv_i8_bw``, everything else to ``F.linear`` on the rounded weight).

``--mode sane``: P143 A2's gate, unchanged in its windows, positions, grouping and bars, as P151 ran it: Phase B's
instrument (``p115_quality.measure_phase``) on the default server built eager at ONE window per pass (bucket 1,
``T == 1``), 12 wikitext windows x 128 positions, every KV built inside the measurement at the server's 4096 tokens with
the server's cap (A2's ``ServedKV``); S0 scores R and saves its log-probs, S1 scores ON against them. Recorded beside it:
the knob's record and the head's engagement after the measurement.

``--mode speed`` (the rule's g1): P109's workloads, timing and token records at P109's registered bytes, with the
head's engagement (``info["int8_head_bw"]``), the cap's plans, the GEMV tally, and the card's telemetry read either side
of every timed block, never during one.

    python p153_box.py --mode speed --prompts prompts.json --out arm_S0a.json --tag S0a   (arm from P153_ARM)
    python p153_box.py --mode sane --out sane_off.json --ref-dir work/ref                  (P153_ARM=S0 / S1)
    python p153_box.py --prompts-only --model ID --revision SHA --out prompts.json
    python p153_box.py --self-test
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
KNOB = "E4B_INT8_HEAD_BW"
DENSE_KNOB = "E4B_DENSE_GEMV_BW"         # unset in both arms: the dense attention GEMV off (its own read is P151)
ARM_VALUE = {"S0": "0", "S1": "1"}
CAP_KNOB = "E4B_ATTN_SPLIT_CAP"           # unset in both arms: the cap at its qwen3_moe default (16, e4b#1792)
FUSION_KNOBS = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
GEMV_KNOBS = ("GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE")
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")
SANE_PHASE_ARMS = {"S0": ("R",), "S1": ("ON",)}
TELEMETRY = ("power.limit", "clocks.max.sm", "temperature.gpu", "clocks.sm", "power.draw")


def arm_env_ok(arm: str, env):
    """``(ok, why)``: the knob carries the arm's value; the split cap, the dense attention GEMV, fusion and decode
    GEMV knobs are unset (their defaults)."""
    if arm not in ARMS:
        return False, f"unknown arm {arm!r}"
    bad = {}
    if (env.get(KNOB) or "").strip() != ARM_VALUE[arm]:
        bad[KNOB] = env.get(KNOB)
    bad.update({k: env.get(k) for k in (CAP_KNOB, DENSE_KNOB) + FUSION_KNOBS + GEMV_KNOBS if (env.get(k) or "").strip()})
    return (not bad), f"{arm} with {bad}"


def _arm():
    arm = os.environ.get("P153_ARM", "")
    ok, why = arm_env_ok(arm, os.environ)
    if not ok:
        raise SystemExit(f"REFUSED: P153_ARM={arm!r}: {why}")
    return arm


def _counts():
    import nf4_grouped
    return dict(nf4_grouped.dispatch_counts())


def _delta(after, before):
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


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
    """The knob's engagement now (engines/int8_head_bw.py): per row count, whether the swapped head routed or passed."""
    from experts4bit_qlora.engines.int8_head_bw import engagement
    return engagement(parts.runner.model)


def _sms():
    import torch
    return torch.cuda.get_device_properties(0).multi_processor_count


def _common(cfg, parts, arm, load_s):
    import torch
    info = parts.info
    return {"arm": arm, "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
            "model": cfg.model, "revision": cfg.revision,
            "knobs": {k: os.environ.get(k) for k in (KNOB, CAP_KNOB, DENSE_KNOB) + FUSION_KNOBS + GEMV_KNOBS},
            "torch": torch.__version__, "sm_count": _sms(), "load_s": round(load_s, 2),
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
    c0 = _counts()
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    load_s = time.perf_counter() - t0
    c_build = _counts()
    rec = {"mode": "speed", "tag": a.tag, **_common(cfg, parts, arm, load_s), "dispatch_build": _delta(c_build, c0),
           "prompts_sha256": pf["prompts_sha256"], "short": a.short, "long": a.long, "reps": a.reps,
           "mem_after_load": p109_box._mem(torch), "telemetry_after_load": _telemetry(), "workloads": {}, "status": "ok"}
    for wname, b in p109_box.WORKLOADS.items():
        wrows = rows[:b]
        walls, digests, last, telem = {}, {}, {}, {}
        cw = _counts()
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
                                   "dispatch": _delta(_counts(), cw), "telemetry": telem, **s}
        print(f"P153_W {arm}/{a.tag} {wname} B={b} decode_tok_s={s['decode_tok_s']} walls={walls} "
              f"telemetry={telem}", flush=True)
    gs = getattr(parts.runner, "graph_stats", None)
    rec["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None
    rec["attn_split_cap_after_runs"] = {"plans": {str(b): dict(r) for b, r in sorted(parts.runner.kv.split_plans.items())}} \
        if getattr(getattr(parts, "runner", None), "kv", None) is not None else None
    rec["int8_head_after_runs"] = _engagement(parts)
    rec["dispatch_total"] = _delta(_counts(), c0)
    rec["mem_after_runs"] = p109_box._mem(torch)
    rec["nvidia_smi"] = p109_box._smi()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P153_ARM " + json.dumps({"arm": arm, "tag": a.tag, "int8_head_bw": rec["int8_head_bw"], "load_s": rec["load_s"],
                                      "W16": rec["workloads"]["W16"]["decode_tok_s"],
                                      "W1": rec["workloads"]["W1"]["decode_tok_s"]}, default=str), flush=True)
    return 0


def sane_main(a) -> int:
    arm = _arm()
    import torch
    import transformers

    import fp8_paged_attn
    import p115_quality as q
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if cfg.graphs or cfg.placement != "all-vram" or cfg.max_tokens_per_seq != 4096:
        raise SystemExit(f"REFUSED: SANE builds the default server eager at all-vram and 4096 tokens per sequence "
                         f"(graphs={cfg.graphs}, max_tokens_per_seq={cfg.max_tokens_per_seq})")
    counters = q.KernelCounters().install()                          # before build_engine, as Phase B
    c0 = _counts()
    t0 = time.time()
    parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    load_s = time.time() - t0
    fwd = q.ForwardCounter(model)
    windows = {"wikitext": q.LOADERS["wikitext"](parts.tokenizer, a.windows, a.prompt, a.cont)}
    common = _common(cfg, parts, arm, load_s)
    cap = (common["attn_split_cap"] or {}).get("cap")
    blocks = -(-cfg.max_tokens_per_seq // 16)
    plan_b1 = fp8_paged_attn.decode_split_plan(1, 4, blocks, block_tokens=16, ktile=64, compute="fp8", pack_heads=False,
                                               sms=_sms())
    c1 = _counts()
    with ServedKV(cfg.max_tokens_per_seq, cap) as served:
        rec = q.measure_phase(model, windows, phase="off" if arm == "S0" else "on", prompt=a.prompt, cont=a.cont,
                              chunk=a.chunk, floor_chunk=a.chunk, group=1, device=cfg.device, ref_dir=a.ref_dir,
                              counters=counters, fwd=fwd, arms=SANE_PHASE_ARMS[arm])
    sk = served.report()
    common["attn_split_cap_server"] = common["attn_split_cap"]
    common["attn_split_cap"] = {"cap": cap, "source": (common["attn_split_cap_server"] or {}).get("source"),
                                "plans": sk["plans"]}
    rec = {"mode": "sane", **common, "transformers": transformers.__version__,
           "sane_kv": {**{k: v for k, v in sk.items() if k != "plans"}, "plan_b1": plan_b1},
           "dispatch_build": _delta(c1, c0), "dispatch_measure": _delta(_counts(), c1),
           "int8_head_bw_measure": _engagement(parts), **rec,
           "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "status": "ok"}
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print(f"P153_SANE {arm} windows={rec['windows']} cap={cap} head={rec['int8_head_bw_measure']} "
          f"sane_kv={rec['sane_kv']} plans={sk['plans']} "
          f"dispatch={rec['dispatch_measure']} {rec['seconds']} s", flush=True)
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
            kv = FakeKV(48, 4, 128, batch=1, max_tokens_per_seq=656)
            kv.split_plans.setdefault(1, {"calls": 0, "plan": 32, "n_split": 16})["calls"] += 48
            kv2 = FakeKV(48, 4, 128, batch=1, max_tokens_per_seq=656)
            kv2.split_plans[1] = {"calls": 48, "plan": 32, "n_split": 16}
            try:
                FakeKV(48, 4, 128, batch=1, max_tokens_per_seq=8192)
                refused = False
            except SystemExit:
                refused = True
        r = s.report()
        out += [("the pass's KV built at the server's capacity, with the cap", (kv.blocks_per_seq, kv.split_cap) == (256, 16)),
                ("the plans merged over passes", r["plans"] == {"1": {"calls": 96, "plan": 32, "n_split": 16}}
                 and r["passes"] == 2 and r["asked"] == [656] and r["max_tokens_per_seq"] == [4096] and not r["plan_clash"]),
                ("a KV above the server's capacity is refused", refused),
                ("the class is restored on exit", FakeKV.__init__ is orig)]
        with ServedKV(4096, None) as s0:
            kv0 = FakeKV(48, 4, 128, batch=1, max_tokens_per_seq=656)
        out.append(("S0: the capacity served, no cap, nothing recorded",
                    (kv0.blocks_per_seq, kv0.split_cap) == (256, None) and s0.report()["plans"] == {}))
    return out


def self_test() -> int:
    s1 = {KNOB: "1"}
    ok = [("arms", ARMS == ("S0", "S1")), ("P109's workloads", p109_box.WORKLOADS == {"W16": 16, "W1": 1}),
          ("P109's self-test", p109_box.self_test() == 0),
          ("S0 named", arm_env_ok("S0", {KNOB: "0"})[0]), ("S1", arm_env_ok("S1", s1)[0]),
          ("S1 padded", arm_env_ok("S1", {KNOB: " 1 "})[0]),
          ("S0 unset is refused (named explicitly)", not arm_env_ok("S0", {})[0]),
          ("S1 at another value is refused", not arm_env_ok("S1", {KNOB: "2"})[0]),
          ("the split cap set is refused (its default is the subject's)", not arm_env_ok("S1", {**s1, CAP_KNOB: "16"})[0]),
          ("the split cap set on S0 too", not arm_env_ok("S0", {KNOB: "0", CAP_KNOB: "0"})[0]),
          ("the dense attention GEMV set is refused", not arm_env_ok("S1", {**s1, DENSE_KNOB: "1"})[0]),
          ("a fusion knob set is refused (the default is the subject)", not arm_env_ok("S1", {**s1, "E4B_FUSE_T1_GLUE": "auto"})[0]),
          ("the GEMV at its default, never forced", not arm_env_ok("S1", {**s1, "GNF4_GEMV_BW": "1"})[0]),
          ("an unknown arm", not arm_env_ok("S2", s1)[0]),
          ("the SANE phases", SANE_PHASE_ARMS == {"S0": ("R",), "S1": ("ON",)}),
          ("telemetry parses", parse_telemetry("575.00, 3090, 41, 2842, 80.12") == {
              "power.limit": 575.0, "clocks.max.sm": 3090.0, "temperature.gpu": 41.0, "clocks.sm": 2842.0, "power.draw": 80.12}),
          ("an unreadable field is None", parse_telemetry("[N/A], 3090, 41")["power.limit"] is None
           and parse_telemetry("[N/A], 3090, 41")["power.draw"] is None),
          ("no line reads None everywhere", set(parse_telemetry("").values()) == {None})]
    ok += _self_test_served_kv()
    bad = [n for n, x in ok if not x]
    print(f"p153_box self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(ok) - len(bad)}/{len(ok)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--prompts-only", action="store_true")
    p.add_argument("--mode", choices=("speed", "sane"))
    p.add_argument("--model")
    p.add_argument("--revision", default="")
    p.add_argument("--prompts")
    p.add_argument("--out")
    p.add_argument("--tag", default="")
    p.add_argument("--short", type=int, default=32)
    p.add_argument("--long", type=int, default=160)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--ref-dir")
    p.add_argument("--windows", type=int, default=12)
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
    if a.mode == "sane":
        return sane_main(a)
    raise SystemExit("REFUSED: --mode speed|sane, --prompts-only or --self-test")


if __name__ == "__main__":
    sys.exit(main())
