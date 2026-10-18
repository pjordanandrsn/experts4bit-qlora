#!/usr/bin/env python3
"""p143a2_box.py -- lane P143 Amendment 2, the served read (bench/p143/PREREG-p143.md, "Amendment 2"; e4b#1313): ONE
process per arm or phase. P115 Phase D's box (bench/p115/p115d_box.py) by counted edits.

The engine is the shipped default server (``PagedServeConfig.from_env()`` + ``build_engine(cfg)``) at grouped-nf4-gemm
0.47.1: the B=1 fused stack on (auto, qwen3_moe allowlisted), the bandwidth decode GEMV on (P116), ``max_seqs`` 16 named
by the runner, 4096 tokens per sequence (the default). The arms differ ONLY in ``E4B_ATTN_SPLIT_CAP``:
  S0  ``0``  (named: the library's plan, today's server);
  S1  ``16`` (the paged attention decode at ``min(16, plan)`` splits: the cap e4b#1768 hands grouped-nf4-gemm's kernel).

``--mode sane``: Phase D's gate, unchanged in its windows, positions, grouping and bars -- Phase B's instrument
(``p115_quality.measure_phase``) on the default server built eager at ONE window per pass (bucket 1, ``T == 1``), 12
wikitext windows x 128 positions; S0 scores R and saves its log-probs, S1 scores ON against them. ONE change, without
which the gate reads nothing: the instrument's pass (``p110_box.paged_pass``) builds its own KV sized ``P + C + 16`` =
656 tokens, where the library plans 11 splits at B = 1 and a cap of 16 cannot bind. Here every KV built inside the
measurement is built at the server's own capacity (``cfg.max_tokens_per_seq``, 4096: plan 32) and carries the server's
resolved cap (``info["attn_split_cap"]["cap"]``), exactly as ``build_engine`` sets its own. Recorded: each pass's KV
capacity, its cap and what the cap met (``Fp8PagedKV.split_plans``), and the library's B = 1 plan at that capacity.

``--mode speed`` (the rule's g1): P109's workloads, timing and token records at P109's registered bytes, with the cap's
engagement (``info["attn_split_cap"]``: every captured bucket's plan and split count) and the GEMV tally.

    python p143a2_box.py --mode speed --prompts prompts.json --out arm_S0a.json --tag S0a   (arm from P143A2_ARM)
    python p143a2_box.py --mode sane --out sane_off.json --ref-dir work/ref                  (P143A2_ARM=S0 / S1)
    python p143a2_box.py --prompts-only --model ID --revision SHA --out prompts.json
    python p143a2_box.py --self-test
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p109_box  # noqa: E402  (staged at P109's registered bytes: prompts, run_pass, slope, digest, _mem, _smi)

ARMS = ("S0", "S1")
CAP_KNOB = "E4B_ATTN_SPLIT_CAP"
ARM_VALUE = {"S0": "0", "S1": "16"}
FUSION_KNOBS = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
GEMV_KNOBS = ("GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE")
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")
SANE_PHASE_ARMS = {"S0": ("R",), "S1": ("ON",)}


def arm_env_ok(arm: str, env):
    """``(ok, why)``: the cap knob carries the arm's value; the fusion and decode GEMV knobs are unset (their defaults)."""
    if arm not in ARMS:
        return False, f"unknown arm {arm!r}"
    bad = {}
    if (env.get(CAP_KNOB) or "").strip() != ARM_VALUE[arm]:
        bad[CAP_KNOB] = env.get(CAP_KNOB)
    bad.update({k: env.get(k) for k in FUSION_KNOBS + GEMV_KNOBS if (env.get(k) or "").strip()})
    return (not bad), f"{arm} with {bad}"


def _arm():
    arm = os.environ.get("P143A2_ARM", "")
    ok, why = arm_env_ok(arm, os.environ)
    if not ok:
        raise SystemExit(f"REFUSED: P143A2_ARM={arm!r}: {why}")
    return arm


def _counts():
    import nf4_grouped
    return dict(nf4_grouped.dispatch_counts())


def _delta(after, before):
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


def _sms():
    import torch
    return torch.cuda.get_device_properties(0).multi_processor_count


def _common(cfg, parts, arm, load_s):
    import torch
    info = parts.info
    return {"arm": arm, "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
            "model": cfg.model, "revision": cfg.revision,
            "knobs": {k: os.environ.get(k) for k in (CAP_KNOB,) + FUSION_KNOBS + GEMV_KNOBS},
            "torch": torch.__version__, "sm_count": _sms(), "load_s": round(load_s, 2),
            "fusions": {k: info.get(k) for k in CENSUS_KEYS}, "fusion_modes": info.get("fusion_modes"),
            "fusion_sources": info.get("fusion_sources"), "attn_split_cap": info.get("attn_split_cap"),
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
           "mem_after_load": p109_box._mem(torch), "workloads": {}, "status": "ok"}
    for wname, b in p109_box.WORKLOADS.items():
        wrows = rows[:b]
        walls, digests, last = {}, {}, {}
        cw = _counts()
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
        rec["workloads"][wname] = {"batch": b, "walls": walls, "rep_digests": digests, "tokens": last,
                                   "dispatch": _delta(_counts(), cw), **s}
        print(f"P143A2_W {arm}/{a.tag} {wname} B={b} decode_tok_s={s['decode_tok_s']} walls={walls}", flush=True)
    gs = getattr(parts.runner, "graph_stats", None)
    rec["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None
    rec["attn_split_cap_after_runs"] = {"plans": {str(b): dict(r) for b, r in sorted(parts.runner.kv.split_plans.items())}} \
        if getattr(getattr(parts, "runner", None), "kv", None) is not None else None
    rec["dispatch_total"] = _delta(_counts(), c0)
    rec["mem_after_runs"] = p109_box._mem(torch)
    rec["nvidia_smi"] = p109_box._smi()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P143A2_ARM " + json.dumps({"arm": arm, "tag": a.tag, "attn_split_cap": rec["attn_split_cap"], "load_s": rec["load_s"],
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
           "dispatch_build": _delta(c1, c0), "dispatch_measure": _delta(_counts(), c1), **rec,
           "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "status": "ok"}
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print(f"P143A2_SANE {arm} windows={rec['windows']} cap={cap} sane_kv={rec['sane_kv']} plans={sk['plans']} "
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
    s1 = {CAP_KNOB: "16"}
    ok = [("arms", ARMS == ("S0", "S1")), ("P109's workloads", p109_box.WORKLOADS == {"W16": 16, "W1": 1}),
          ("P109's self-test", p109_box.self_test() == 0),
          ("S0 named", arm_env_ok("S0", {CAP_KNOB: "0"})[0]), ("S1", arm_env_ok("S1", s1)[0]),
          ("S1 padded", arm_env_ok("S1", {CAP_KNOB: " 16 "})[0]),
          ("S0 unset is refused (named explicitly)", not arm_env_ok("S0", {})[0]),
          ("S1 at another cap is refused", not arm_env_ok("S1", {CAP_KNOB: "8"})[0]),
          ("a fusion knob set is refused (the default is the subject)", not arm_env_ok("S1", {**s1, "E4B_FUSE_T1_GLUE": "auto"})[0]),
          ("the GEMV at its default, never forced", not arm_env_ok("S1", {**s1, "GNF4_GEMV_BW": "1"})[0]),
          ("an unknown arm", not arm_env_ok("S2", s1)[0]),
          ("the SANE phases", SANE_PHASE_ARMS == {"S0": ("R",), "S1": ("ON",)})]
    ok += _self_test_served_kv()
    bad = [n for n, x in ok if not x]
    print(f"p143a2_box self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(ok) - len(bad)}/{len(ok)} cases)")
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
