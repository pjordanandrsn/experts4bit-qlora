#!/usr/bin/env python3
"""p231_box.py -- lane P231, lever B3 (bench/p231/PREREG-p231.md): ONE process per arm, fit check, quality phase or
premise read. Does ``E4B_NF4_SMALLM_BW=1`` (the per-row route: every (token, slot) row its own group, so grouped-nf4-gemm's
K33 bandwidth GEMV, one row per program) decode the default ``serve_paged`` server faster at 2 and 4 requests on
Mixtral-8x7B on one RTX 5090, at no measurable quality cost? P141's box (``bench/p141/p141_box.py``) by named edits:

  * the switch is ``E4B_NF4_SMALLM_BW`` (e4b#2009): B0 unset (the shipped default: the device tile table and K25 at
    2..8 tokens), B1 ``1``. grouped-nf4-gemm's own K33 knobs stay unset in both: Mixtral's shapes are in
    ``_BW_SHAPES`` with shipped plans since P141, so K33 is what the shipped T == 1 decode already runs;
  * the workloads are P109's prompts at W1, W2, W4, W8 and W16. W2 and W4 are ruled; W1 and W16 are controls where
    the switch is inert (T == 1 is singleton already; 16 is above the switch's 8); W8 is reported only;
  * timing is decode-only (P113's: each pass's wall less its largest ttft);
  * engagement is read PER BUCKET at capture: ``PagedModelRunner._capture_bucket`` is wrapped for the build only,
    and each bucket's e4b route tally (``hot_residency.ROUTE_SEEN``) and gnf4 dispatch counts are recorded;
  * the quality read is at ``--group 4`` (P115's padded bucket arithmetic, T == 4: the route B1 changes), behind the
    premise ``--mode rows``;
  * ``--mode rows``, the premise the maintainer set for reading quality at group 4 only: on the card, before the
    fetch, B1's expert call (``hot_residency._fused_over_stack`` with the token rows, exactly as the collapse makes it)
    on random NF4 stores at Mixtral's two shapes returns BITWISE the same output for each (token, slot) row at
    T = 1, 2 and 4.

The server is P136's Mixtral size, 16 slots x 768 tokens (P141's).

    python p231_box.py --mode rows --out rows.json                                       (no model; before the fetch)
    python p231_box.py --mode fit --prompts prompts.json --out fit.json                  (P231_ARM=B0)
    python p231_box.py --mode speed --prompts prompts.json --out arm_B0a.json --tag B0a  (arm from P231_ARM)
    python p231_box.py --mode quality --out quality_off.json --ref-dir work/ref --windows 24 --group 4 (P231_ARM=B0/B1)
    python p231_box.py --prompts-only --model ID --revision SHA --out prompts.json
    python p231_box.py --premise junit.xml --expect served
    python p231_box.py --self-test
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p109_box  # noqa: E402  (staged at P109's registered bytes: prompts, slope, digest, _mem, _smi)

ARMS = ("B0", "B1")
KNOB = "E4B_NF4_SMALLM_BW"
GNF4_KNOBS = ("GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE", "GNF4_GEMV_DOTPAD")
WORKLOADS = {"W1": 1, "W2": 2, "W4": 4, "W8": 8, "W16": 16}
RULED, CONTROLS, REPORTED = ("W2", "W4"), ("W1", "W16"), ("W8",)
QUALITY_GROUP = 4
QUALITY_ARMS = {"B0": ("R", "rep", "half", "chunk", "mutant_scale"), "B1": ("ON",)}
SERVER_SIZE = (16, 768)           # (max_seqs, max_tokens_per_seq): P136's Mixtral server (P141's)
#: the premise's pytest suite on the card: rc 0, exactly this many passed, nothing skipped (P141's served list less
#: its K33 GPU file, which tested the GNF4_GEMV_BW route P141 read)
PREMISE = {"served": (7, {})}
#: Mixtral-8x7B's expert stack (P227's spec): 8 experts, top-2, gate_up (N1, K1) and down (N2, K2)
MIXTRAL = {"E": 8, "topk": 2, "shapes": (28672, 4096, 4096, 14336)}
#: the premise's routings over four tokens (top-2 of 8), each a list of per-token expert pairs: all distinct, a shared
#: expert, all four on one pair, then three seeded draws (built in rows_main)
ROW_ROUTINGS = {"distinct": [[0, 1], [2, 3], [4, 5], [6, 7]], "shared": [[0, 1], [1, 2], [2, 3], [3, 0]],
                "one_pair": [[5, 6], [5, 6], [5, 6], [5, 6]]}
ROW_SEEDS = (2311, 2312, 2313)


def arm_env_ok(arm: str, env):
    """``(ok, why)``: B0 leaves the switch unset; B1 sets it to ``1``. Both leave grouped-nf4-gemm's K33 knobs unset."""
    if arm not in ARMS:
        return False, f"unknown arm {arm!r}"
    bad = {k: env.get(k) for k in GNF4_KNOBS if (env.get(k) or "").strip()}
    want = "" if arm == "B0" else "1"
    if (env.get(KNOB) or "").strip() != want:
        bad[KNOB] = env.get(KNOB)
    return (not bad), f"{arm} with {bad}"


def premise_faults(xml_text: str, expect: str) -> list:
    """A pytest junit report against :data:`PREMISE`: no failure or error, the registered pass count, nothing skipped."""
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
    return bad


def premise_main(a) -> int:
    bad = premise_faults(open(a.premise).read(), a.expect)
    print(f"P231_PREMISE {a.expect} {'ok' if not bad else 'FAILED ' + json.dumps(bad)[:1500]}", flush=True)
    return 0 if not bad else 1


def row_mismatches(outs: dict, k: int) -> list:
    """``outs[T]`` is the list of per-call outputs at T tokens per call over the SAME four token rows (T = 1: four calls;
    T = 2: two; T = 4: one), each ``[T * k, H]`` in input row order. Every (token, slot) row must be bitwise equal across
    T. Returns the mismatching ``(token, slot, T)`` triples against T = 1."""
    import torch
    rows = {T: torch.cat(v, dim=0) for T, v in outs.items()}
    ref = rows[1]
    bad = []
    for T, r in rows.items():
        if r.shape != ref.shape:
            bad.append(("shape", T, tuple(r.shape)))
            continue
        for i in range(r.shape[0]):
            if not torch.equal(r[i], ref[i]):
                bad.append((i // k, i % k, T))
    return bad


def rows_main(a) -> int:
    """The premise: B1's expert call returns each (token, slot) row bitwise identically at T = 1, 2 and 4."""
    import torch
    import torch.nn.functional as F
    import nf4_grouped as ng
    from experts4bit_qlora.engines import hot_residency as hr
    for k in GNF4_KNOBS:
        os.environ.pop(k, None)
    os.environ[KNOB] = "1"
    dev = torch.device("cuda")
    E, k = MIXTRAL["E"], MIXTRAL["topk"]
    n1, k1, n2, k2 = MIXTRAL["shapes"]
    g = torch.Generator(device=dev).manual_seed(231)
    Wt = (torch.randint(0, 256, (E, n1, k1 // 2), dtype=torch.uint8, generator=g, device=dev),
          torch.rand(E, n1, k1 // 64, generator=g, device=dev) * 0.05 + 0.01,
          torch.randint(0, 256, (E, n2, k2 // 2), dtype=torch.uint8, generator=g, device=dev),
          torch.rand(E, n2, k2 // 64, generator=g, device=dev) * 0.05 + 0.01)
    routings = dict(ROW_ROUTINGS)
    for s in ROW_SEEDS:
        gs = torch.Generator().manual_seed(s)
        routings[f"seed{s}"] = [torch.randperm(E, generator=gs)[:k].tolist() for _ in range(4)]
    X = (torch.randn(4, k1, generator=g, device=dev) * 0.3).to(torch.bfloat16)
    rec = {"mode": "rows", "shapes": list(MIXTRAL["shapes"]), "topk": k, "routings": routings, "cases": {},
           "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
           "gpu": torch.cuda.get_device_name(0), "status": "ok"}

    def call(tokens, pairs):
        T = len(tokens)
        ids = torch.tensor([e for t in tokens for e in pairs[t]], dtype=torch.long, device=dev)
        rt = torch.arange(T * k, device=dev) // k
        return hr._fused_over_stack(None, ids, *Wt, MIXTRAL["shapes"], True, F.silu, singleton_groups=True,
                                    device_grouping=False, int4_stores=None, x_tokens=(X[tokens], rt, k))
    all_bad = []
    for name, pairs in routings.items():
        c0 = dict(ng.dispatch_counts())
        outs = {1: [call([t], pairs) for t in range(4)], 2: [call([0, 1], pairs), call([2, 3], pairs)],
                4: [call([0, 1, 2, 3], pairs)]}
        torch.cuda.synchronize()
        d = {key: ng.dispatch_counts().get(key, 0) - c0.get(key, 0) for key in ng.dispatch_counts()}
        bad = row_mismatches(outs, k)
        bw = sum(v for key, v in d.items() if key.startswith("bw_"))
        rec["cases"][name] = {"mismatches": [list(b) for b in bad[:20]], "n_mismatch": len(bad), "dispatch": d,
                              "bw_launches": bw}
        if bw == 0:
            bad = bad + [("no K33 launch", name, 0)]
        all_bad += [(name, *b) for b in bad]
    rec["ok"] = not all_bad
    rec["status"] = "ok" if not all_bad else "failed"
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print(f"P231_ROWS {'ok' if not all_bad else 'FAILED'} cases={len(routings)} "
          f"mismatches={sum(c['n_mismatch'] for c in rec['cases'].values())} "
          f"bw_launches={[c['bw_launches'] for c in rec['cases'].values()]}", flush=True)
    return 0 if not all_bad else 1


def _arm():
    arm = os.environ.get("P231_ARM", "")
    ok, why = arm_env_ok(arm, os.environ)
    if not ok:
        raise SystemExit(f"REFUSED: arm {arm}: {why}")
    return arm


def _counts():
    import nf4_grouped
    return dict(nf4_grouped.dispatch_counts())


def _routes():
    from experts4bit_qlora.engines import hot_residency as hr
    return dict(hr.ROUTE_SEEN)


def _delta(after, before):
    return {k: after.get(k, 0) - before.get(k, 0) for k in after if after.get(k, 0) - before.get(k, 0)}


def _served_cfg(graphs=True):
    from experts4bit_qlora.serve_paged import PagedServeConfig
    cfg = PagedServeConfig.from_env()
    if bool(cfg.graphs) != graphs or cfg.placement != "all-vram":
        raise SystemExit(f"REFUSED: graphs={cfg.graphs} placement={cfg.placement}, registered graphs={graphs} all-vram")
    if (cfg.max_seqs, cfg.max_tokens_per_seq) != SERVER_SIZE:
        raise SystemExit(f"REFUSED: the server is {cfg.max_seqs} x {cfg.max_tokens_per_seq}, registered "
                         f"{SERVER_SIZE[0]} x {SERVER_SIZE[1]} (P136's Mixtral size)")
    return cfg


def build_tagged(cfg):
    """``build_engine(cfg)`` with each decode bucket's capture tagged: ``{bucket: {"route": ..., "dispatch": ...}}``
    from e4b's route tally and gnf4's dispatch counts around every ``_capture_bucket`` call (warm-up and capture). The
    wrap is restored before the build returns, so nothing timed runs under it."""
    from experts4bit_qlora.engines import paged_runner
    from experts4bit_qlora.serve_paged import build_engine
    by_bucket, orig = {}, paged_runner.PagedModelRunner._capture_bucket

    def tagged(self, b, warmup):
        r0, c0 = _routes(), _counts()
        try:
            return orig(self, b, warmup)
        finally:
            by_bucket[str(int(b))] = {"route": _delta(_routes(), r0), "dispatch": _delta(_counts(), c0)}
    paged_runner.PagedModelRunner._capture_bucket = tagged
    try:
        parts = build_engine(cfg)
    finally:
        paged_runner.PagedModelRunner._capture_bucket = orig
    return parts, by_bucket


def run_pass_ttft(parts, torch, rows, n_tokens):
    """P109's ``run_pass`` that also returns every row's ``Request.ttft`` (P113's): ``(wall, steps, tokens, ttfts)``."""
    sched = parts.scheduler
    before = len(sched.done)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    rids = [sched.add_request(list(r), max_new_tokens=n_tokens) for r in rows]
    steps = sched.run_until_idle()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    new = {r.rid: r for r in sched.done[before:]}
    if sorted(new) != sorted(rids) or sched.active or sched.queue:
        raise AssertionError(f"scheduler did not drain: done {sorted(new)} vs added {sorted(rids)}")
    for rid, row in zip(rids, rows):
        q = new[rid]
        if len(q.out) != n_tokens or q.prompt_len != len(row):
            raise AssertionError(f"request {rid}: {len(q.out)} tokens from {q.prompt_len} prompt tokens, "
                                 f"expected {n_tokens} from {len(row)}")
    return wall, steps, [[int(t) for t in new[r].out] for r in rids], [new[r].ttft for r in rids]


def decode_only(walls, ttfts):
    """Each pass's wall minus its largest ttft, or None when a pass lacks a row's ttft."""
    out = []
    for w, ts in zip(walls, ttfts):
        if not ts or any(t is None or t < 0 for t in ts) or max(ts) >= w:
            return None
        out.append(round(w - max(ts), 5))
    return out


def _base(cfg, mode, arm, a, torch, info, load_s):
    return {"mode": mode, "arm": arm, "tag": getattr(a, "tag", ""), "e4b_sha": os.environ.get("E4B_SHA"),
            "gnf4_sha": os.environ.get("GNF4_SHA"), "model": cfg.model, "revision": cfg.revision,
            "knob": os.environ.get(KNOB), "gnf4_knobs": {k: os.environ.get(k) for k in GNF4_KNOBS},
            "torch": torch.__version__, "load_s": round(load_s, 2), "max_seqs": cfg.max_seqs,
            "max_tokens_per_seq": cfg.max_tokens_per_seq, "buckets": list(cfg.buckets), "graphs": bool(cfg.graphs),
            "placement": cfg.placement,
            "graph_status": ({str(k): v for k, v in info["graph_status"].items()} if info.get("graph_status") else None),
            "grouping": info.get("grouping"), "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability()),
            "sm_count": torch.cuda.get_device_properties(0).multi_processor_count}


def fit_main(a) -> int:
    """One default-server build and one W16 pass at the short length (the widest workload). A failure is the fit
    verdict, never retried."""
    arm = _arm()
    if arm != "B0":
        raise SystemExit("REFUSED: the fit check runs the shipped default (B0)")
    pf = json.load(open(a.prompts))
    import torch
    cfg = _served_cfg()
    rec = {"mode": "fit", "status": "error", "model": cfg.model, "max_seqs": cfg.max_seqs,
           "max_tokens_per_seq": cfg.max_tokens_per_seq, "buckets": list(cfg.buckets), "graphs": bool(cfg.graphs)}
    try:
        t0 = time.perf_counter()
        parts, by_bucket = build_tagged(cfg)
        rec.update(_base(cfg, "fit", arm, a, torch, parts.info, time.perf_counter() - t0))
        wall, _steps, _toks, _tt = run_pass_ttft(parts, torch, pf["rows"][:WORKLOADS["W16"]], a.short)
        rec.update(status="ok", w16_wall_s=round(wall, 4), mem=p109_box._mem(torch), by_bucket=by_bucket)
    except Exception as e:  # noqa: BLE001 -- not fitting is a finding, recorded with its reason
        rec["error"] = f"{type(e).__name__}: {e}"[:600]
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P231_FIT " + json.dumps({k: rec.get(k) for k in ("status", "max_seqs", "buckets", "graphs", "w16_wall_s",
                                                             "error")}, default=str), flush=True)
    return 0 if rec["status"] == "ok" else 31


def speed_main(a) -> int:
    arm = _arm()
    pf = json.load(open(a.prompts))
    rows = pf["rows"]
    if p109_box.digest(rows) != pf["prompts_sha256"] or len(rows) != p109_box.ROWS:
        raise SystemExit("REFUSED: prompts.json does not match its own digest")
    import torch
    cfg = _served_cfg()
    c0, r0 = _counts(), _routes()
    t0 = time.perf_counter()
    parts, by_bucket = build_tagged(cfg)
    load_s = time.perf_counter() - t0
    rec = _base(cfg, "speed", arm, a, torch, parts.info, load_s)
    rec.update(by_bucket=by_bucket, dispatch_build=_delta(_counts(), c0), route_build=_delta(_routes(), r0),
               prompts_sha256=pf["prompts_sha256"], short=a.short, long=a.long, reps=a.reps,
               mem_after_load=p109_box._mem(torch), workloads={}, status="ok")
    for wname, b in WORKLOADS.items():
        wrows = rows[:b]
        walls, ttfts, digests, last = {}, {}, {}, {}
        for n in (a.short, a.long):
            run_pass_ttft(parts, torch, wrows, n)                         # warm, untimed
            ws, ts, ds = [], [], []
            for _ in range(a.reps):
                wall, _steps, toks, tt = run_pass_ttft(parts, torch, wrows, n)
                ws.append(round(wall, 5))
                ts.append([round(x, 5) if x is not None else None for x in tt])
                ds.append(p109_box.digest(toks))
                last[str(n)] = toks
            walls[str(n)], ttfts[str(n)], digests[str(n)] = ws, ts, ds
        dec = {n: decode_only(walls[n], ttfts[n]) for n in walls}
        whole = p109_box.slope(walls[str(a.short)], walls[str(a.long)], b, a.short, a.long)
        if dec[str(a.short)] is None or dec[str(a.long)] is None:
            s = {"decode_tok_s": None, "decode_tok_s_median": None, "status": "void: a pass lacks a row's ttft"}
        else:
            s = p109_box.slope(dec[str(a.short)], dec[str(a.long)], b, a.short, a.long)
        rec["workloads"][wname] = {"batch": b, "walls": walls, "ttfts": ttfts, "decode_only_walls": dec,
                                   "rep_digests": digests, "tokens": last, **s, "whole_pass": dict(whole)}
        print(f"P231_W {arm}/{a.tag} {wname} B={b} decode_tok_s={s['decode_tok_s']} whole_pass={whole['decode_tok_s']}",
              flush=True)
    gs = getattr(parts.runner, "graph_stats", None)
    rec["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None
    rec["mem_after_runs"] = p109_box._mem(torch)
    rec["nvidia_smi"] = p109_box._smi()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P231_ARM " + json.dumps({"arm": arm, "tag": a.tag, "load_s": rec["load_s"], "graph_status": rec["graph_status"],
                                    "by_bucket_routes": {b: v["route"] for b, v in by_bucket.items()},
                                    **{w: rec["workloads"][w]["decode_tok_s"] for w in rec["workloads"]}},
                                   default=str), flush=True)
    return 0


def quality_main(a) -> int:
    arm = _arm()
    if a.group != QUALITY_GROUP:
        raise SystemExit(f"REFUSED: --group {a.group}; P231 reads quality at group {QUALITY_GROUP} (T == 4)")
    import torch
    import transformers

    import p115_quality as q
    from experts4bit_qlora.serve_paged import build_engine
    cfg = _served_cfg(graphs=False)
    counters = q.KernelCounters().install()                          # BEFORE build_engine, as Phase B
    c0 = _counts()
    t0 = time.time()
    parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    load_s = time.time() - t0
    fwd = q.ForwardCounter(model)
    windows = {t: q.LOADERS[t](parts.tokenizer, a.windows, a.prompt, a.cont) for t in q.TEXTS}
    c1, r1 = _counts(), _routes()
    phase = "off" if arm == "B0" else "on"
    rec = q.measure_phase(model, windows, phase=phase, prompt=a.prompt, cont=a.cont, chunk=a.chunk,
                          floor_chunk=a.floor_chunk, group=QUALITY_GROUP, device=cfg.device, ref_dir=a.ref_dir,
                          counters=counters, fwd=fwd, arms=QUALITY_ARMS[arm])
    rec = {"mode": "quality", "arm": arm, "model": cfg.model, "revision": cfg.revision,
           "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"), "knob": os.environ.get(KNOB),
           "gnf4_knobs": {k: os.environ.get(k) for k in GNF4_KNOBS}, "max_seqs": cfg.max_seqs,
           "max_tokens_per_seq": cfg.max_tokens_per_seq, "transformers": transformers.__version__,
           "load_s": round(load_s, 1), "buckets": list(cfg.buckets), "dispatch_build": _delta(c1, c0),
           "dispatch_measure": _delta(_counts(), c1), "route_measure": _delta(_routes(), r1), **rec,
           "gpu": torch.cuda.get_device_name(0), "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "status": "ok"}
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print(f"P231_QUALITY {arm} windows={rec['windows']} dispatch={rec['dispatch_measure']} "
          f"routes={rec['route_measure']} load {load_s:.0f}s mem {rec['max_mem_gb']} GB {rec['seconds']} s", flush=True)
    return 0


def self_test() -> int:
    import torch

    def junit(ok=0, skips=(), fail=()):
        cases = [f'<testcase classname="test_x" name="t{i}"/>' for i in range(ok)]
        cases += [f'<testcase classname="{c}" name="{n}"><skipped message="{m}"/></testcase>'
                  for (c, n), m in ((i.split("::"), m) for i, m in skips)]
        cases += [f'<testcase classname="test_x" name="f{i}"><failure message="x"/></testcase>' for i in fail]
        return f'<testsuites><testsuite name="pytest">{"".join(cases)}</testsuite></testsuites>'
    same = {1: [torch.tensor([[1.0, 2.0], [3.0, 4.0]]) for _ in range(4)],
            2: [torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]]) for _ in range(2)],
            4: [torch.tensor([[1.0, 2.0], [3.0, 4.0]] * 4)]}
    moved = {**same, 4: [torch.tensor([[1.0, 2.0], [3.0, 4.0]] * 3 + [[1.0, 2.0], [3.0, 4.0 + 2 ** -20]])]}
    ok = [ARMS == ("B0", "B1"), p109_box.self_test() == 0, KNOB == "E4B_NF4_SMALLM_BW",
          arm_env_ok("B0", {})[0], not arm_env_ok("B0", {KNOB: "1"})[0], not arm_env_ok("B0", {KNOB: "0"})[0],
          arm_env_ok("B1", {KNOB: "1"})[0], not arm_env_ok("B1", {})[0],
          not arm_env_ok("B1", {KNOB: "1", "GNF4_GEMV_BW": "1"})[0], not arm_env_ok("B0", {"GNF4_GEMV_BW_PLAN": "x"})[0],
          not arm_env_ok("B2", {KNOB: "1"})[0],
          WORKLOADS == {"W1": 1, "W2": 2, "W4": 4, "W8": 8, "W16": 16},
          set(RULED) | set(CONTROLS) | set(REPORTED) == set(WORKLOADS) and not set(RULED) & set(CONTROLS),
          SERVER_SIZE == (16, 768) and QUALITY_GROUP == 4,
          QUALITY_ARMS == {"B0": ("R", "rep", "half", "chunk", "mutant_scale"), "B1": ("ON",)},
          premise_faults(junit(7), "served") == [], bool(premise_faults(junit(6), "served")),
          bool(premise_faults(junit(7, [("test_x::s", "why")]), "served")),
          bool(premise_faults(junit(7, [], [1]), "served")),
          row_mismatches(same, 2) == [], row_mismatches(moved, 2) == [(3, 1, 4)],
          len(ROW_ROUTINGS) == 3 and len(ROW_SEEDS) == 3 and all(len(p) == 4 for p in ROW_ROUTINGS.values()),
          decode_only([6.0, 7.4], [[0.5, 5.0], [0.4, 6.1]]) == [1.0, 1.3], decode_only([6.0], [[0.5, None]]) is None]
    print(f"p231_box self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--prompts-only", action="store_true")
    p.add_argument("--mode", choices=("rows", "fit", "speed", "quality"))
    p.add_argument("--premise")
    p.add_argument("--expect", choices=("served",))
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
    p.add_argument("--group", type=int, default=QUALITY_GROUP)
    p.add_argument("--prompt", type=int, default=512)
    p.add_argument("--cont", type=int, default=128)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--floor-chunk", type=int, default=256)
    p.add_argument("--check-args", action="store_true", help="parse and validate the arguments, then exit (tests)")
    a = p.parse_args(argv)
    if a.check_args:
        if a.prompts_only:
            ok = bool(a.model and a.revision and a.out)
        elif a.premise:
            ok = bool(a.expect)
        elif a.mode == "rows":
            ok = bool(a.out)
        elif a.mode in ("fit", "speed"):
            ok = bool(a.prompts and a.out) and (a.mode == "fit" or bool(a.tag)) and 0 < a.short < a.long and a.reps > 0
        elif a.mode == "quality":
            ok = bool(a.out and a.ref_dir) and a.group == QUALITY_GROUP and a.windows > 0
        else:
            ok = False
        print("ARGS OK" if ok else "ARGS BAD")
        return 0 if ok else 2
    if a.self_test:
        return self_test()
    if a.prompts_only:
        return p109_box.prompts_main(a)
    if a.premise:
        return premise_main(a)
    if a.mode == "rows":
        return rows_main(a)
    if a.mode == "fit":
        return fit_main(a)
    if a.mode == "speed":
        return speed_main(a)
    if a.mode == "quality":
        return quality_main(a)
    raise SystemExit("REFUSED: --mode rows|fit|speed|quality, --prompts-only, --premise or --self-test")


if __name__ == "__main__":
    sys.exit(main())
