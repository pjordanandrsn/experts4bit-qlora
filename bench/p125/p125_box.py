#!/usr/bin/env python3
"""p125_box.py -- lane P125 (bench/p125/PREREG-p125.md; e4b#1313): ONE process per arm. Does calibrated int4 attention
(and the calibrated int4 lm_head on top) license on the shipped default's single-stream decode, and what does it buy?

The engine is the shipped default server (``PagedServeConfig.from_env()`` + ``build_engine(cfg)``) with every fusion and
decode-GEMV knob unset (the B=1 fused stack resolves ``auto`` on ``qwen3_moe``; grouped-nf4-gemm 0.43.0's bandwidth GEMV
at its default). The arms differ ONLY in the int4 levers, calibrated on the box exactly as a user's build runs them
(32 x 512 tokens of C4 validation shard 0, ``serve_paged._calib_batches``):
  A  the default (no int4 lever);
  B  ``E4B_SERVE_ATTN_INT4_CALIB=1``: the 192 attention projections on calibrated int4 (fused q/k/v stays on: 96 modules);
  C  B + ``E4B_SERVE_LMHEAD_INT4_CALIB=1``: the lm_head too;
  M  ``E4B_SERVE_ATTN_INT4=1``: RTN int4 attention, the reported sensitivity check (predicted to FAIL the gate);
  K  B with every Int4Linear's scales rolled one 32-block along K after the build: the sure-fail mutant (the VOID rung).

``--mode speed`` (reported, not ruled; A, B and C): P109's workloads at P109's registered bytes on the default graph
server at 16 slots, with the int4 census, every Int4Linear's digest (calibration determinism across builds), memory
(free before load, peak after the build, after the first prefill, over the runs; the bf16 copies resident after) and
each arm's ``E4B_PAGED_MAX_SEQS=auto`` slot count at 2048 and 4096 tokens a slot on the measured free memory.

``--mode quality`` (the gate; every arm): Phase D's instrument (``p115_quality.measure_phase``) on the server built
eager, wikitext, at ONE window per pass (``T == 1``: the activation-quantised ``gemv_int4_b32`` path) and in 16-row
pieces (``group`` 16: K16), plus c4val1 in 16-row pieces (the K8-style ppl, reported). A scores R and saves its
log-probs per gate; every other arm scores ON against them. A count-only Route records each Int4Linear call by route and
row count over each gate's passes, so the read proves which path ran.

    python p125_box.py --mode speed --prompts prompts.json --out arm_B1.json --tag B1     (arm from P125_ARM)
    python p125_box.py --mode quality --out quality_B.json --ref-dir work/ref              (P125_ARM=A writes R)
    python p125_box.py --prompts-only --model ID --revision SHA --out prompts.json
    python p125_box.py --self-test
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p109_box  # noqa: E402  (staged at P109's registered bytes: prompts, run_pass, slope, digest, _mem, _smi)

ARMS = ("A", "B", "C", "M", "K")
SPEED_ARMS = ("A", "B", "C")
#: every lever that changes the int4 path or its calibration: each arm names its own, the rest stay unset (the
#: calibration's own knobs at their defaults, exactly as a user's build runs it)
LEVERS = ("E4B_SERVE_EXP_INT4", "E4B_SERVE_EXP_INT4_CALIB", "E4B_SERVE_ATTN_INT4", "E4B_SERVE_ATTN_INT4_CALIB",
          "E4B_SERVE_LMHEAD_INT4_CALIB", "E4B_SERVE_DENSE_INT4_CALIB", "E4B_ATTN_INT4_WIDE", "E4B_ATTN_INT4_SMALLM",
          "E4B_CALIB_SOURCE", "E4B_CALIB_NSEQ", "E4B_CALIB_LAYERS_PER_PASS", "E4B_INT4_ARTIFACT_DIR",
          "E4B_INT4_EXPECTED_FINGERPRINT", "E4B_INT4_DUMP_ARTIFACT_DIR", "E4B_INT4_ASSIGNMENT")
ARM_ENV = {"A": {}, "B": {"E4B_SERVE_ATTN_INT4_CALIB": "1"},
           "C": {"E4B_SERVE_ATTN_INT4_CALIB": "1", "E4B_SERVE_LMHEAD_INT4_CALIB": "1"},
           "M": {"E4B_SERVE_ATTN_INT4": "1"}, "K": {"E4B_SERVE_ATTN_INT4_CALIB": "1"}}
KNOBS = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
GEMV_KNOBS = ("GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE")
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")
INT4_KEYS = ("int4_attn_projections", "attn_int4_calib_projections", "attn_int4_rtn_projections")
#: the quality gates: (name, text, group, windows for every arm but K, windows for K)
GATES = (("t1", "wikitext", 1, 108, 12), ("k16", "wikitext", 16, 112, 16), ("c4", "c4val1", 16, 16, 16))
SLOT_TOKENS = (2048, 4096)
#: Amendment 3: the corpora's sizes under Qwen3-30B-A3B's tokenizer (measured on p125-5090-1's box and offline with the
#: pinned tokenizer) and each text's window stride. P97's loader starts a window every 4096 tokens, so wikitext-2-raw
#: test holds 73 windows of 640, fewer than the 108 and 112 the gates registered; P125's own wikitext loader takes P97's
#: corpus, join and tokenisation byte for byte and starts a window every 2048 tokens (capacity 146). c4val1 keeps
#: P115's loader (4096; capacity 212).
CORPUS_TOKENS = {"wikitext": 298938, "c4val1": 866460}
STRIDE = {"wikitext": 2048, "c4val1": 4096}
WINDOW = 512 + 128


def capacity(text: str, tokens=None) -> int:
    """Windows of 640 tokens that fit ``text`` at its stride."""
    tokens = CORPUS_TOKENS[text] if tokens is None else tokens
    return (tokens - WINDOW) // STRIDE[text] + 1 if tokens >= WINDOW else 0


def wikitext_windows(tok, n, prompt, cont, stride=STRIDE["wikitext"]):
    """P97's wikitext-2-raw test corpus, joined and tokenised as P97 does; window k starts at token k * ``stride``."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    out = []
    for k in range(n):
        w = ids[k * stride:k * stride + prompt + cont]
        assert w.numel() == prompt + cont, f"wikitext window {k} has {w.numel()} tokens (stride {stride}, {ids.numel()} tokens)"
        out.append(w.tolist())
    return out


def loaders():
    import p115_quality as q
    return {"wikitext": wikitext_windows, "c4val1": q.LOADERS["c4val1"]}


def windows_check(a) -> int:
    """Amendment 3's preflight, before any arm: every gate's windows load, full, from the real corpora and tokenizer."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision or None)
    out, bad = {}, []
    for name, text, _group, n, n_k in GATES:
        want = max(n, n_k)
        want = min(want, a.max_windows) if a.max_windows else want
        try:
            ws = loaders()[text](tok, want, a.prompt, a.cont)
            out[name] = {"text": text, "windows": len(ws), "full": all(len(w) == a.prompt + a.cont for w in ws)}
        except AssertionError as e:
            bad.append(f"{name}: {e}")
            out[name] = {"text": text, "windows": 0, "error": str(e)[:200]}
    print("P125_WINDOWS " + json.dumps({"gates": out, "refused": bad}), flush=True)
    return 1 if bad else 0


def arm_env_ok(arm: str, env):
    """``(ok, why)``: the arm's levers set to ``1``, every other lever, fusion knob and decode-GEMV knob unset."""
    if arm not in ARMS:
        return False, f"unknown arm {arm!r}"
    want = ARM_ENV[arm]
    bad = {k: env.get(k) for k in LEVERS if (env.get(k) or "").strip() != want.get(k, "")}
    bad.update({k: env.get(k) for k in KNOBS + GEMV_KNOBS if (env.get(k) or "").strip()})
    return (not bad), f"{arm} with {bad}"


def _arm(modes):
    arm = os.environ.get("P125_ARM", "")
    ok, why = arm_env_ok(arm, os.environ)
    if not ok or arm not in modes:
        raise SystemExit(f"REFUSED: P125_ARM={arm!r}: {why if not ok else 'not an arm of this mode'}")
    return arm


def _counts():
    import nf4_grouped
    return dict(nf4_grouped.dispatch_counts())


def _delta(after, before):
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


def int4_linears(model):
    from experts4bit_qlora.engines.int4_attn import Int4Linear
    return [(n, m) for n, m in model.named_modules() if isinstance(m, Int4Linear)]


def digest(m) -> str:
    h = hashlib.sha256()
    for t in (m.packed, m.scales):
        h.update(t.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


class Route:
    """Counts each Int4Linear call by route and row count, ``"<route>:<rows>" -> calls`` (P124's Route, counting only:
    every module keeps its own flags). Graph replays never reach Python; the quality passes are eager."""

    def __init__(self, mods):
        self.mods, self.counts, self.saved = list(mods), {}, []

    def _add(self, route, rows):
        k = f"{route}:{int(rows)}"
        self.counts[k] = self.counts.get(k, 0) + 1

    def __enter__(self):
        for m in self.mods:
            self.saved.append((m, m._smallm, m._gemv))
            rows_now = [0]
            real_fwd, real_bf16, real_sm, real_gemv = m.forward, m._bf16_weight, m._smallm, m._gemv

            def forward(x, _real=real_fwd, _rows=rows_now):
                _rows[0] = x.numel() // x.shape[-1]
                return _real(x)

            def bf16_weight(_real=real_bf16, _rows=rows_now):
                self._add("bf16", _rows[0])
                return _real()

            def gemv(*a, _real=real_gemv, _rows=rows_now, **kw):
                self._add("gemv", _rows[0])
                return _real(*a, **kw)

            m.forward, m._bf16_weight, m._gemv = forward, bf16_weight, gemv
            if real_sm is not None:
                def smallm(x, *a, _real=real_sm, **kw):
                    self._add("k16" if int(x.shape[0]) <= 16 else "wide", int(x.shape[0]))
                    return _real(x, *a, **kw)
                m._smallm = smallm
        return self

    def __exit__(self, *exc):
        for m, sm, gemv in self.saved:
            m._smallm, m._gemv = sm, gemv
            for name in ("forward", "_bf16_weight"):
                m.__dict__.pop(name, None)
        return False


def roll_scales(mods) -> int:
    """The sure-fail mutant K: every Int4Linear's scales rolled one 32-block along K, in place, before any call (so the
    bf16 copy, the GEMV and K16 all read every weight on its neighbour's scale)."""
    import torch
    with torch.no_grad():
        for _n, m in mods:
            m.scales.copy_(m.scales.roll(-1, dims=-1))
    return len(mods)


def slots(cfg, model_config, free_bytes, arm, head_numel) -> dict:
    """``E4B_PAGED_MAX_SEQS=auto``'s slot count for this arm on ``free_bytes``, at 2048 and 4096 tokens a slot, by e4b's
    own resolver. Its estimate knows ``attn_int4``; it has no lm_head flag, so C is also corrected by hand: the head's
    int4 grid on top (its bf16 copy equals the bf16 head the estimate already prices)."""
    from dataclasses import replace

    from experts4bit_qlora.arch.topology import describe_moe
    from experts4bit_qlora.serve_paged import _buckets_env
    from experts4bit_qlora.serve_recipe import (MAX_SEQS_AUTO_MARGIN_BYTES, MAX_SEQS_AUTO_WIDTHS,
                                                MAX_SEQS_AUTO_WIDTHS_AUTO_BUCKETS, ServeSetup, choose_max_seqs,
                                                estimate_serve_footprint, int4_store_bytes,
                                                prefill_graph_reserve_bytes)
    topo = describe_moe(model_config)
    b = _buckets_env("")
    widths = MAX_SEQS_AUTO_WIDTHS_AUTO_BUCKETS if str(b).strip().lower() == "auto" else MAX_SEQS_AUTO_WIDTHS
    attn = arm in ("B", "C")
    head = int4_store_bytes(*head_numel) if arm == "C" else 0
    out = {"free_bytes": int(free_bytes), "widths": list(widths), "head_int4_bytes": head}
    for tokens in SLOT_TOKENS:
        setup = ServeSetup(placement="all-vram", max_seqs=16, max_tokens_per_seq=tokens, chunk_tokens=cfg.chunk_tokens,
                           graphs=True, buckets=b, kv_groups=cfg.kv_groups, prefill_graph=cfg.prefill_graph,
                           vram_gb=cfg.vram_gb, dram_gb=cfg.dram_gb, hot_rows=cfg.hot_rows, bulk_kv=cfg.bulk_kv,
                           exp_int4=False, attn_int4=attn)
        res = choose_max_seqs(topo, setup, free_bytes, widths=widths)
        needs = {}
        for w in sorted(set(widths), reverse=True):
            s = replace(setup, max_seqs=w)
            fp = estimate_serve_footprint(topo, s)
            needs[w] = None if fp.refusals else fp.device_bytes + prefill_graph_reserve_bytes(topo, s)
        fit = [w for w, n in needs.items() if n is not None and n + head + MAX_SEQS_AUTO_MARGIN_BYTES <= free_bytes]
        out[str(tokens)] = {"estimate_slots": int(res["max_seqs"]), "need_bytes": {str(w): n for w, n in needs.items()},
                            "corrected_slots": max(fit) if fit else min(widths)}
    return out


def _common(cfg, parts, arm, load_s, model):
    import torch
    info = parts.info
    mods = int4_linears(model)
    head = getattr(model, "lm_head", None)
    return {"arm": arm, "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
            "model": cfg.model, "revision": cfg.revision,
            "env": {k: os.environ.get(k) for k in LEVERS + KNOBS + GEMV_KNOBS}, "torch": torch.__version__,
            "load_s": round(load_s, 2), "fusions": {k: info.get(k) for k in CENSUS_KEYS},
            "fusion_sources": info.get("fusion_sources"), "fusion_modes": info.get("fusion_modes"),
            "model_type": info.get("model_type"), "int4": {k: info.get(k) for k in INT4_KEYS},
            "int4_modules": [[n, m.N, m.K, digest(m)] for n, m in mods], "head_type": type(head).__name__,
            "levers_env": info.get("levers_env"), "grouping": info.get("grouping"), "max_seqs": cfg.max_seqs,
            "buckets": list(cfg.buckets), "graphs": cfg.graphs}


def _resident(model):
    out = []
    for n, m in int4_linears(model):
        c = getattr(m, "_bf16_cache", None)
        out.append([n, c is not None, int(c.numel() * c.element_size()) if c is not None else 0])
    return out


def speed_main(a) -> int:
    arm = _arm(SPEED_ARMS)
    pf = json.load(open(a.prompts))
    rows = pf["rows"]
    if p109_box.digest(rows) != pf["prompts_sha256"] or len(rows) != p109_box.ROWS:
        raise SystemExit("REFUSED: prompts.json does not match its own digest")
    import torch
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if not cfg.graphs or (cfg.max_seqs, cfg.placement, tuple(cfg.buckets)) != (16, "all-vram", (1, 2, 4, 8, 16)):
        raise SystemExit(f"REFUSED: not the default graph server at 16 slots: graphs={cfg.graphs} max_seqs={cfg.max_seqs} "
                         f"placement={cfg.placement} buckets={cfg.buckets}")
    torch.cuda.init()
    free_before, total = torch.cuda.mem_get_info()
    c0 = _counts()
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    load_s = time.perf_counter() - t0
    c_build = _counts()
    model = parts.runner.model
    peak_build = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    rec = {"mode": "speed", "tag": a.tag, **_common(cfg, parts, arm, load_s, model), "dispatch_build": _delta(c_build, c0),
           "prompts_sha256": pf["prompts_sha256"], "short": a.short, "long": a.long, "reps": a.reps,
           "memory": {"free_before_load": int(free_before), "total": int(total), "peak_after_build": int(peak_build)},
           "workloads": {}, "status": "ok"}
    first = True
    for wname, b in p109_box.WORKLOADS.items():
        wrows = rows[:b]
        walls, digests, last = {}, {}, {}
        cw = _counts()
        for n in (a.short, a.long):
            p109_box.run_pass(parts, torch, wrows, n)                      # warm, untimed
            if first:                                                     # the first prefill: the bf16 copies are built
                rec["memory"]["peak_first_prefill"] = int(torch.cuda.max_memory_allocated())
                torch.cuda.reset_peak_memory_stats()
                first = False
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
        print(f"P125_W {arm}/{a.tag} {wname} B={b} decode_tok_s={s['decode_tok_s']} walls={walls}", flush=True)
    rec["memory"]["peak_runs"] = int(torch.cuda.max_memory_allocated())
    rec["memory"]["allocated_after_runs"] = int(torch.cuda.memory_allocated())
    rec["memory"]["reserved_after_runs"] = int(torch.cuda.memory_reserved())
    rec["bf16_resident"] = _resident(model)
    head = model.lm_head
    head_numel = (int(head.N), int(head.K)) if hasattr(head, "N") else tuple(int(x) for x in head.weight.shape)
    rec["slots"] = slots(cfg, model.config, free_before, arm, head_numel)
    gs = getattr(parts.runner, "graph_stats", None)
    rec["graph_stats"] = {str(k): dict(v) for k, v in gs.items()} if isinstance(gs, dict) else None
    rec["graph_status"] = ({str(k): v for k, v in parts.info["graph_status"].items()}
                           if parts.info.get("graph_status") else None)
    rec["dispatch_total"] = _delta(_counts(), c0)
    rec["nvidia_smi"] = p109_box._smi()
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P125_ARM " + json.dumps({"arm": arm, "tag": a.tag, "fusions": rec["fusions"], "int4": rec["int4"],
                                    "load_s": rec["load_s"], "W16": rec["workloads"]["W16"]["decode_tok_s"],
                                    "W1": rec["workloads"]["W1"]["decode_tok_s"],
                                    "slots": {t: rec["slots"][str(t)]["corrected_slots"] for t in SLOT_TOKENS}},
                                   default=str), flush=True)
    return 0


def gate_record(text, group, n, routes, dispatch, measured) -> dict:
    """One gate's record: ``measure_phase``'s result with the box's own keys ON TOP. ``measure_phase`` returns a
    ``windows`` of its own ({text: count}); merged last, it overwrote this gate's count (Amendment 2, ``p125-prove-2``)."""
    return {**measured, "text": text, "group": group, "windows": int(n), "routes": dict(sorted(routes.items())),
            "dispatch": dispatch}


def quality_main(a) -> int:
    arm = _arm(ARMS)
    import torch
    import transformers

    import p115_quality as q
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if cfg.graphs or cfg.placement != "all-vram":
        raise SystemExit(f"REFUSED: the quality gates build the default server eager at all-vram (graphs={cfg.graphs})")
    counters = q.KernelCounters().install()                          # before build_engine, as Phase B
    c0 = _counts()
    t0 = time.time()
    parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    load_s = time.time() - t0
    mods = int4_linears(model)
    rolled = roll_scales(mods) if arm == "K" else 0
    fwd = q.ForwardCounter(model)
    rec = {"mode": "quality", **_common(cfg, parts, arm, load_s, model), "transformers": transformers.__version__,
           "dispatch_build": _delta(_counts(), c0), "rolled_modules": rolled, "gates": {}, "status": "ok"}
    for name, text, group, n, n_k in GATES[:a.gates]:
        n = n_k if arm == "K" else n
        n = min(n, a.max_windows) if a.max_windows else n
        windows = {text: loaders()[text](parts.tokenizer, n, a.prompt, a.cont)}
        c1 = _counts()
        with Route([m for _n, m in mods]) as rt:
            r = q.measure_phase(model, windows, phase="off" if arm == "A" else "on", prompt=a.prompt, cont=a.cont,
                                chunk=a.chunk, floor_chunk=a.chunk, group=group, device=cfg.device,
                                ref_dir=os.path.join(a.ref_dir, name), counters=counters, fwd=fwd,
                                arms=("R",) if arm == "A" else ("ON",))
        rec["gates"][name] = gate_record(text, group, n, rt.counts, _delta(_counts(), c1), r)
        print(f"P125_GATE {arm} {name} {text} group={group} windows={n} routes={rec['gates'][name]['routes']} "
              f"{r.get('seconds')} s", flush=True)
    rec["n_int4_modules"] = len(mods)
    rec["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("P125_QUALITY " + json.dumps({"arm": arm, "fusions": rec["fusions"], "int4": rec["int4"],
                                        "n_int4_modules": len(mods), "load_s": rec["load_s"]}, default=str), flush=True)
    return 0


def self_test() -> int:
    b = {"E4B_SERVE_ATTN_INT4_CALIB": "1"}
    c = {**b, "E4B_SERVE_LMHEAD_INT4_CALIB": "1"}
    cases = [
        ("the arms", ARMS == ("A", "B", "C", "M", "K") and SPEED_ARMS == ("A", "B", "C")),
        ("P109's workloads", p109_box.WORKLOADS == {"W16": 16, "W1": 1} and p109_box.self_test() == 0),
        ("A: nothing set", arm_env_ok("A", {})[0]),
        ("B: the calibrated attention alone", arm_env_ok("B", b)[0]),
        ("C: attention and head", arm_env_ok("C", c)[0]),
        ("M: RTN attention", arm_env_ok("M", {"E4B_SERVE_ATTN_INT4": "1"})[0]),
        ("K: B's build", arm_env_ok("K", b)[0]),
        ("A refuses a lever", not arm_env_ok("A", b)[0]),
        ("B refuses the head", not arm_env_ok("B", c)[0]),
        ("B refuses a calibration knob", not arm_env_ok("B", {**b, "E4B_CALIB_NSEQ": "8"})[0]),
        ("B refuses the wide route", not arm_env_ok("B", {**b, "E4B_ATTN_INT4_WIDE": "1"})[0]),
        ("B refuses a fusion knob set", not arm_env_ok("B", {**b, "E4B_PAGED_FUSE_QKV": "auto"})[0]),
        ("B refuses a forced GEMV", not arm_env_ok("B", {**b, "GNF4_GEMV_BW": "1"})[0]),
        ("an unknown arm", not arm_env_ok("D", {})[0]),
        ("the gates", [g[:3] for g in GATES] == [("t1", "wikitext", 1), ("k16", "wikitext", 16), ("c4", "c4val1", 16)]
         and all(n % g == 0 and k % g == 0 for _, _, g, n, k in GATES)),
        ("the window counts (the registered arithmetic)", [g[3] for g in GATES] == [108, 112, 16]),
        ("Amendment 3: every gate's windows fit its corpus", all(max(n, k) <= capacity(t) for _, t, _, n, k in GATES)),
        ("Amendment 3: P97's 4096 stride would not (73 windows)",
         (CORPUS_TOKENS["wikitext"] - WINDOW) // 4096 + 1 == 73 and capacity("wikitext") == 146),
    ]
    bad = [name for name, ok in cases if not ok]
    print(f"p125_box self-test {'OK' if not bad else 'FAILED: ' + '; '.join(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--prompts-only", action="store_true")
    p.add_argument("--windows-check", action="store_true", help="Amendment 3's preflight: every gate's windows fit")
    p.add_argument("--mode", choices=("speed", "quality"))
    p.add_argument("--model")
    p.add_argument("--revision", default="")
    p.add_argument("--prompts")
    p.add_argument("--out")
    p.add_argument("--tag", default="")
    p.add_argument("--short", type=int, default=32)
    p.add_argument("--long", type=int, default=160)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--ref-dir")
    p.add_argument("--prompt", type=int, default=512)
    p.add_argument("--cont", type=int, default=128)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--gates", type=int, default=len(GATES), help="the first N gates (the proof runs them all)")
    p.add_argument("--max-windows", type=int, default=0, help="the proof's window cap per gate (0: the registered counts)")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.prompts_only:
        return p109_box.prompts_main(a)
    if a.windows_check:
        return windows_check(a)
    if a.mode == "speed":
        return speed_main(a)
    if a.mode == "quality":
        return quality_main(a)
    raise SystemExit("REFUSED: --mode speed|quality, --prompts-only or --self-test")


if __name__ == "__main__":
    sys.exit(main())
