#!/usr/bin/env python3
"""p133_box.py -- lane P133 (bench/p133/PREREG-p133.md): ONE process per arm, fit check or quality phase. Does the
bandwidth-targeted NF4 decode GEMV (grouped-nf4-gemm K33, ``GNF4_GEMV_BW=1``) decode the default ``serve_paged`` server
faster at one request on an RTX 4090 (sm_89, 128 SMs), where ``GNF4_GEMV_BW=auto``'s >= 160-SM gate keeps it off?

Derived from ``bench/p116/p116_box.py`` by named substitutions:
  * the arms: B0 = ``GNF4_GEMV_BW`` unset, the shipped default. At 128 SMs that is the scalar GEMV: auto engages only
    at >= 160 SMs, and dot-pad carries the same gate. B1 = ``GNF4_GEMV_BW=1`` at the shipped plan table
    (grouped-nf4-gemm 0.46.0's ``_BW_PLANS``, K33's selected plans), with no plan override;
  * the control: P116's W16 becomes ``Wn`` at the box's own width, n = min(max_seqs, 16). ``max_seqs`` is
    ``serve_paged``'s auto choice, the widest of 64 / 32 / 16 that the footprint estimate fits, so a 24 GB card may
    differ from the 5090. Below 2 the control is void and reported, and W1 reads alone;
  * ``--mode fit``: before any arm, one default-server build and one W1 pass. If W1 does not fit, the run is VOID with no
    rerun (the registration's rule), before any arm runs;
  * ``P133_DRY=1``: the $0 dry run on a tiny fixture, on any CUDA card. The server is whatever the card builds: on sm_86
    that is eager, with host grouping and no graphs. Recorded, never a reading.

    python p133_box.py --mode fit --prompts prompts.json --out fit.json             (P133_ARM=B0)
    python p133_box.py --mode speed --prompts prompts.json --out arm_B0a.json --tag B0a   (arm from P133_ARM)
    python p133_box.py --mode quality --out quality_off.json --ref-dir work/ref --windows 24 (P133_ARM=B0 / B1)
    python p133_box.py --prompts-only --model ID --revision SHA --out prompts.json
    python p133_box.py --premise junit.xml --expect kernel|served          (P133_DRY=1: the sm_86 served list)
    python p133_box.py --self-test
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p109_box  # noqa: E402  (staged at P109's registered bytes: prompts, run_pass, slope, digest, _mem, _smi)

ARMS = ("B0", "B1")
KNOBS = ("GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE")
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")
QUALITY_ARMS = {"B0": ("R", "rep", "chunk", "mutant_scale"), "B1": ("ON",)}
CONTROL_MAX = p109_box.ROWS                  # P109's 16 prompts: the control is never wider than that
PDL = "griddepcontrol needs sm_90+ (the switch is inert below)"
FP8 = "the fp8 paged KV needs native e4m3 (sm_89+)"
_FP8_IDS = ("test_decode_graph_buckets::test_a_bucket_replay_decodes_exactly_as_the_padded_eager_step",
            "test_decode_graph_buckets::test_every_bucket_step_advances_its_rows_own_kv_length[graph]",
            "test_decode_graph_buckets::test_every_bucket_step_advances_its_rows_own_kv_length[padded-eager]",
            "test_kv_step_select::test_the_step_selection_decodes_exactly_as_the_per_layer_selection[graph]",
            "test_kv_step_select::test_the_step_selection_decodes_exactly_as_the_per_layer_selection[padded-eager]")
#: The premise, per suite: rc 0, exactly this many passed, and exactly these test ids skipped with these reasons, so a new
#: skip reads as a failure. The kernel's two PDL checks skip below sm_90 (the 4090 is sm_89); the served route's five
#: fp8-KV cases skip below sm_89, so only the dry run (sm_86) skips them.
PREMISE = {"kernel": (33, {"test_nf4_gemv_bw::test_the_pdl_preamble_precedes_every_global_load": PDL,
                           "test_nf4_gemv_bw::test_pdl_on_is_bitwise_pdl_off": PDL}),
           "served": (18, {}),
           "served_dry": (13, {i: FP8 for i in _FP8_IDS})}


def dry() -> bool:
    return os.environ.get("P133_DRY", "") == "1"


def arm_env_ok(arm: str, env):
    """``(ok, why)``: B0 leaves all three knobs unset (the shipped default); B1 has ``GNF4_GEMV_BW=1`` and no plan or
    decode override (the shipped table, ``prmt32``)."""
    val = {k: (env.get(k) or "").strip() for k in KNOBS}
    if arm == "B0":
        want = {"GNF4_GEMV_BW": ("",), "GNF4_GEMV_BW_PLAN": ("",), "GNF4_GEMV_BW_DECODE": ("",)}
    elif arm == "B1":
        want = {"GNF4_GEMV_BW": ("1",), "GNF4_GEMV_BW_PLAN": ("",), "GNF4_GEMV_BW_DECODE": ("",)}
    else:
        return False, f"unknown arm {arm!r}"
    bad = {k: v for k, v in val.items() if v not in want[k]}
    return (not bad), f"{arm} with {bad}"


def premise_faults(xml_text: str, expect: str) -> list:
    """A pytest junit report against :data:`PREMISE`: no failure or error, the registered pass count, and exactly the
    registered skipped ids, each with its registered reason."""
    import xml.etree.ElementTree as ET
    want_passed, want_skips = PREMISE[expect]
    passed, skips, bad = 0, {}, []
    for tc in ET.fromstring(xml_text).iter("testcase"):
        tid = f"{tc.get('classname', '').rsplit('.', 1)[-1]}::{tc.get('name')}"
        kids = {c.tag: c for c in tc}
        if "failure" in kids or "error" in kids:
            bad.append(f"{tid} failed")
        elif "skipped" in kids:
            skips[tid] = kids["skipped"].get("message", "")
        else:
            passed += 1
    if passed != want_passed:
        bad.append(f"{passed} passed, registered {want_passed}")
    if set(skips) != set(want_skips):
        bad.append(f"skipped {sorted(skips)}, registered {sorted(want_skips)}")
    bad += [f"{k} skipped for {v!r}, registered {want_skips[k]!r}" for k, v in skips.items()
            if k in want_skips and v != want_skips[k]]
    return bad


def premise_main(a) -> int:
    expect = "served_dry" if a.expect == "served" and dry() else a.expect
    bad = premise_faults(open(a.premise).read(), expect)
    print(f"P133_PREMISE {expect} {'ok' if not bad else 'FAILED ' + json.dumps(bad)[:1500]}", flush=True)
    return 0 if not bad else 1


def control_width(max_seqs: int) -> int:
    """The control's width: the box's own ``max_seqs``, capped at P109's 16 prompts; 0 when below 2 (no control)."""
    n = min(int(max_seqs), CONTROL_MAX)
    return n if n >= 2 else 0


def workloads(max_seqs: int) -> dict:
    n = control_width(max_seqs)
    return {"W1": 1, **({f"W{n}": n} if n else {})}


def _arm():
    arm = os.environ.get("P133_ARM", "")
    if arm not in ARMS:
        raise SystemExit(f"REFUSED: P133_ARM={arm!r}, expected one of {ARMS}")
    ok, why = arm_env_ok(arm, os.environ)
    if not ok:
        raise SystemExit(f"REFUSED: arm {arm}: {why}")
    return arm


def _counts():
    import nf4_grouped
    return dict(nf4_grouped.dispatch_counts())


def _delta(after, before):
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


def _served_cfg():
    from experts4bit_qlora.serve_paged import PagedServeConfig
    cfg = PagedServeConfig.from_env()
    if not dry() and (not cfg.graphs or cfg.placement != "all-vram"):
        raise SystemExit(f"REFUSED: not the default graph server: graphs={cfg.graphs} placement={cfg.placement}")
    return cfg


def _base(cfg, mode, arm, a, torch, info, load_s, dispatch_build):
    return {"mode": mode, "arm": arm, "tag": getattr(a, "tag", ""), "dry": dry(), "e4b_sha": os.environ.get("E4B_SHA"),
            "gnf4_sha": os.environ.get("GNF4_SHA"), "model": cfg.model, "revision": cfg.revision,
            "knobs": {k: os.environ.get(k) for k in KNOBS}, "torch": torch.__version__, "load_s": round(load_s, 2),
            "max_seqs": cfg.max_seqs, "buckets": list(cfg.buckets), "graphs": bool(cfg.graphs),
            "placement": cfg.placement, "fusions": {k: info.get(k) for k in CENSUS_KEYS},
            "fusion_modes": info.get("fusion_modes"), "levers_env": info.get("levers_env"),
            "graph_status": ({str(k): v for k, v in info["graph_status"].items()} if info.get("graph_status") else None),
            "grouping": info.get("grouping"), "dispatch_build": dispatch_build,
            "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability()),
            "sm_count": torch.cuda.get_device_properties(0).multi_processor_count}


def fit_main(a) -> int:
    """One default-server build and one W1 pass at the short length. A failure is the fit verdict, never retried."""
    arm = _arm()
    if arm != "B0":
        raise SystemExit("REFUSED: the fit check runs the shipped default (B0)")
    pf = json.load(open(a.prompts))
    import torch
    from experts4bit_qlora.serve_paged import build_engine
    cfg = _served_cfg()
    rec = {"mode": "fit", "status": "error", "dry": dry(), "model": cfg.model, "max_seqs": cfg.max_seqs,
           "buckets": list(cfg.buckets), "graphs": bool(cfg.graphs)}
    try:
        c0 = _counts()
        t0 = time.perf_counter()
        parts = build_engine(cfg)
        rec.update(_base(cfg, "fit", arm, a, torch, parts.info, time.perf_counter() - t0, _delta(_counts(), c0)))
        wall, steps, toks = p109_box.run_pass(parts, torch, pf["rows"][:1], a.short)
        rec.update(status="ok", w1_wall_s=round(wall, 4), w1_steps=steps, mem=p109_box._mem(torch),
                   control_width=control_width(cfg.max_seqs))
    except Exception as e:  # noqa: BLE001 -- not fitting is a finding, recorded with its reason
        rec["error"] = f"{type(e).__name__}: {e}"[:600]
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P133_FIT " + json.dumps({k: rec.get(k) for k in ("status", "max_seqs", "buckets", "graphs", "control_width",
                                                             "w1_wall_s", "error")}, default=str), flush=True)
    return 0 if rec["status"] == "ok" else 31


def speed_main(a) -> int:
    arm = _arm()
    pf = json.load(open(a.prompts))
    rows = pf["rows"]
    if p109_box.digest(rows) != pf["prompts_sha256"] or len(rows) != p109_box.ROWS:
        raise SystemExit("REFUSED: prompts.json does not match its own digest")
    import torch
    from experts4bit_qlora.serve_paged import build_engine
    cfg = _served_cfg()
    c0 = _counts()
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    load_s = time.perf_counter() - t0
    c_build = _counts()
    rec = _base(cfg, "speed", arm, a, torch, parts.info, load_s, _delta(c_build, c0))
    wl = workloads(cfg.max_seqs)
    rec.update(prompts_sha256=pf["prompts_sha256"], short=a.short, long=a.long, reps=a.reps,
               control=next((w for w in wl if w != "W1"), None), mem_after_load=p109_box._mem(torch), workloads={},
               status="ok")
    for wname, b in wl.items():
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
        print(f"P133_W {arm}/{a.tag} {wname} B={b} decode_tok_s={s['decode_tok_s']} median={s['decode_tok_s_median']} "
              f"walls={walls}", flush=True)
    gs = getattr(parts.runner, "graph_stats", None)
    rec["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None
    rec["dispatch_total"] = _delta(_counts(), c0)
    rec["mem_after_runs"] = p109_box._mem(torch)
    rec["nvidia_smi"] = p109_box._smi()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P133_ARM " + json.dumps({"arm": arm, "tag": a.tag, "dispatch_build": rec["dispatch_build"],
                                    "load_s": rec["load_s"], "graph_status": rec["graph_status"],
                                    "control": rec["control"], "max_seqs": rec["max_seqs"],
                                    **{w: rec["workloads"][w]["decode_tok_s"] for w in rec["workloads"]}},
                                   default=str), flush=True)
    return 0


def quality_main(a) -> int:
    arm = _arm()
    import torch
    import transformers

    import p115_quality as q
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if cfg.graphs or cfg.placement != "all-vram":
        raise SystemExit(f"REFUSED: the quality read builds the default server eager at all-vram (graphs={cfg.graphs})")
    if a.group != 1:
        raise SystemExit(f"REFUSED: --group {a.group}; P133 reads quality at one window per pass (T == 1)")
    counters = q.KernelCounters().install()                          # BEFORE build_engine, as Phase B
    c0 = _counts()
    t0 = time.time()
    parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    load_s = time.time() - t0
    fwd = q.ForwardCounter(model)
    windows = {t: q.LOADERS[t](parts.tokenizer, a.windows, a.prompt, a.cont) for t in q.TEXTS}
    c1 = _counts()
    phase = "off" if arm == "B0" else "on"
    rec = q.measure_phase(model, windows, phase=phase, prompt=a.prompt, cont=a.cont, chunk=a.chunk,
                          floor_chunk=a.floor_chunk, group=1, device=cfg.device, ref_dir=a.ref_dir,
                          counters=counters, fwd=fwd, arms=QUALITY_ARMS[arm])
    info = parts.info
    rec = {"mode": "quality", "arm": arm, "dry": dry(), "model": cfg.model, "revision": cfg.revision,
           "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
           "knobs": {k: os.environ.get(k) for k in KNOBS}, "transformers": transformers.__version__,
           "load_s": round(load_s, 1), "buckets": list(cfg.buckets), "fusions": {k: info.get(k) for k in CENSUS_KEYS},
           "fusion_modes": info.get("fusion_modes"), "dispatch_build": _delta(c1, c0),
           "dispatch_measure": _delta(_counts(), c1), **rec, "gpu": torch.cuda.get_device_name(0),
           "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "status": "ok"}
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print(f"P133_QUALITY {arm} windows={rec['windows']} dispatch={rec['dispatch_measure']} load {load_s:.0f}s "
          f"mem {rec['max_mem_gb']} GB {rec['seconds']} s", flush=True)
    return 0


def self_test() -> int:
    b1 = {"GNF4_GEMV_BW": "1"}

    def junit(ok=0, skips=(), fail=()):
        cases = [f'<testcase classname="test_x" name="t{i}"/>' for i in range(ok)]
        cases += [f'<testcase classname="{c}" name="{n}"><skipped message="{m}"/></testcase>'
                  for (c, n), m in ((i.split("::"), m) for i, m in skips)]
        cases += [f'<testcase classname="test_x" name="f{i}"><failure message="x"/></testcase>' for i in fail]
        return f'<testsuites><testsuite name="pytest">{"".join(cases)}</testsuite></testsuites>'
    k_skips = list(PREMISE["kernel"][1].items())
    d_skips = list(PREMISE["served_dry"][1].items())
    ok = [ARMS == ("B0", "B1"), p109_box.self_test() == 0, CONTROL_MAX == 16,
          arm_env_ok("B0", {})[0], not arm_env_ok("B0", {"GNF4_GEMV_BW": "0"})[0],
          not arm_env_ok("B0", {"GNF4_GEMV_BW": "1"})[0], not arm_env_ok("B0", {"GNF4_GEMV_BW": "auto"})[0],
          arm_env_ok("B1", b1)[0], not arm_env_ok("B1", {})[0],
          not arm_env_ok("B1", {**b1, "GNF4_GEMV_BW_PLAN": "1536,2048=16,256,4,1"})[0],
          not arm_env_ok("B1", {**b1, "GNF4_GEMV_BW_DECODE": "tree"})[0], not arm_env_ok("B2", b1)[0],
          workloads(16) == {"W1": 1, "W16": 16}, workloads(64) == {"W1": 1, "W16": 16},
          workloads(8) == {"W1": 1, "W8": 8}, workloads(1) == {"W1": 1}, control_width(2) == 2, control_width(1) == 0,
          QUALITY_ARMS == {"B0": ("R", "rep", "chunk", "mutant_scale"), "B1": ("ON",)},
          premise_faults(junit(33, k_skips), "kernel") == [], premise_faults(junit(18), "served") == [],
          premise_faults(junit(13, d_skips), "served_dry") == [], bool(premise_faults(junit(32, k_skips), "kernel")),
          bool(premise_faults(junit(33, k_skips[:1]), "kernel")), bool(premise_faults(junit(33, k_skips, [1]), "kernel")),
          bool(premise_faults(junit(18, d_skips[:1]), "served")),
          bool(premise_faults(junit(33, [(k_skips[0][0], "another reason"), k_skips[1]]), "kernel"))]
    print(f"p133_box self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--prompts-only", action="store_true")
    p.add_argument("--mode", choices=("fit", "speed", "quality"))
    p.add_argument("--premise")
    p.add_argument("--expect", choices=("kernel", "served"))
    p.add_argument("--model")
    p.add_argument("--revision", default="")
    p.add_argument("--prompts")
    p.add_argument("--out")
    p.add_argument("--tag", default="")
    p.add_argument("--short", type=int, default=32)
    p.add_argument("--long", type=int, default=160)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--ref-dir")
    p.add_argument("--windows", type=int, default=24)
    p.add_argument("--group", type=int, default=1)
    p.add_argument("--prompt", type=int, default=512)
    p.add_argument("--cont", type=int, default=128)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--floor-chunk", type=int, default=256)
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.prompts_only:
        return p109_box.prompts_main(a)
    if a.premise:
        return premise_main(a)
    if a.mode == "fit":
        return fit_main(a)
    if a.mode == "speed":
        return speed_main(a)
    if a.mode == "quality":
        return quality_main(a)
    raise SystemExit("REFUSED: --mode fit|speed|quality, --prompts-only, --premise or --self-test")


if __name__ == "__main__":
    sys.exit(main())
