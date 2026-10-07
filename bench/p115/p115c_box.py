#!/usr/bin/env python3
"""p115c_box.py -- lane P115 Phase C (bench/p115/PREREG-p115.md, Amendment 2; e4b#1313): does ``auto`` on the four
fusion knobs engage by structure on two other families, refuse nothing it should not, and compute nothing grossly wrong?

One process per (model, build), each building the shipped default server with ``PagedServeConfig.from_env()`` +
``build_engine``; only the four fusion knobs differ:

- ``serve`` mode, arm from ``P115C_ARM``:
  - ``off``: all four ``0``. Census; the 16 prompts x ``--new`` tokens, twice (the determinism control).
  - ``on``: all four ``auto``. Census; the same prompts once.
  - ``explicit``: all four ``1``. ``build_engine`` must refuse (a vacuous enable); the record carries the refusal.
- ``sane`` mode (the maintainer's gross-error gate, pre-data, 59b79fb7), arm ``off`` or ``on``: the default server built
  eager (graphs off), Phase B's instrument (``p115_quality.measure_phase``) at reduced size: ``--windows`` wikitext
  windows of 512 prompt tokens and ``--cont`` teacher-forced positions, R (the graph server's arithmetic) under ``off``,
  ON under ``on`` against R's saved log-probs.

The prompts are P109's wikitext rows in the model's own tokenizer (``p109_box.prompts_main``).

    python p115c_box.py --mode serve --prompts prompts.json --out serve_off.json --model-tag gptoss  (arm from P115C_ARM)
    python p115c_box.py --mode sane --out sane_off.json --ref-dir work/ref_gptoss --model-tag gptoss
    python p115c_box.py --prompts-only --model ID --revision SHA --out prompts.json
    python p115c_box.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p109_box  # noqa: E402  (staged at P109's registered bytes: prompts, run_pass, digest, _mem)

ARMS = ("off", "on", "explicit")
KNOBS = ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")
ARM_VALUE = {"off": "0", "on": "auto", "explicit": "1"}
CENSUS_KEYS = ("fuse_qkv_n", "fuse_t1_glue_n", "fuse_t1_glue_r2_n", "fuse_router_epilogue_n")


def arm_env_ok(arm, env):
    want = ARM_VALUE.get(arm)
    if want is None:
        return False, f"unknown arm {arm!r}"
    bad = {k: env.get(k) for k in KNOBS if (env.get(k) or "").strip().lower() != want}
    return (not bad), f"arm {arm} wants every knob {want!r}, got {bad}"


def serve_main(a, arm) -> int:
    import torch

    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    rec = {"mode": "serve", "arm": arm, "model_tag": a.model_tag, "e4b_sha": os.environ.get("E4B_SHA"),
           "gnf4_sha": os.environ.get("GNF4_SHA"), "torch": torch.__version__, "status": "ok"}
    cfg = PagedServeConfig.from_env()
    rec.update(model=cfg.model, revision=cfg.revision, fusion_modes=dict(cfg.fusion_modes), graphs=bool(cfg.graphs),
               max_seqs=cfg.max_seqs, placement=cfg.placement)
    t0 = time.perf_counter()
    try:
        parts = build_engine(cfg)
    except Exception as exc:                                         # noqa: BLE001 -- the explicit arm's refusal is the record
        rec.update(raised=True, exc_type=type(exc).__name__, exc_message=str(exc)[:800],
                   load_s=round(time.perf_counter() - t0, 2), status="raised" if arm == "explicit" else "error",
                   traceback=traceback.format_exc()[-2000:])
        json.dump(rec, open(a.out, "w"), indent=1)
        print(f"P115C_SERVE {a.model_tag}/{arm} raised {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        return 0 if arm == "explicit" else 1
    rec.update(raised=False, load_s=round(time.perf_counter() - t0, 2))
    info = parts.info
    rec["census"] = {k: info.get(k) for k in CENSUS_KEYS}
    rec["fusion_report"] = info.get("fusion_report")
    rec["graph_status"] = {str(k): v for k, v in (info.get("graph_status") or {}).items()}
    if arm == "explicit":
        rec["status"] = "built"                                      # the reducer reads this as the gate failing
        json.dump(rec, open(a.out, "w"), indent=1, default=str)
        print(f"P115C_SERVE {a.model_tag}/explicit BUILT (it should have refused) census={rec['census']}", flush=True)
        return 0
    pf = json.load(open(a.prompts))
    rows = pf["rows"]
    if p109_box.digest(rows) != pf["prompts_sha256"]:
        raise SystemExit("REFUSED: prompts.json does not match its own digest")
    rec["prompts_sha256"] = pf["prompts_sha256"]
    passes = 2 if arm == "off" else 1
    rec["tokens"] = []
    for _ in range(passes):
        _wall, _steps, toks = p109_box.run_pass(parts, torch, rows, a.new)
        rec["tokens"].append(toks)
    rec["new"] = a.new
    rec["mem_after_runs"] = p109_box._mem(torch)
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print(f"P115C_SERVE {a.model_tag}/{arm} census={rec['census']} graphs={rec['graph_status']} passes={passes}", flush=True)
    return 0


def sane_main(a, arm) -> int:
    import torch
    import transformers

    import p115_quality as q
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if cfg.graphs or cfg.placement != "all-vram":
        raise SystemExit(f"REFUSED: SANE builds the default server eager at all-vram (graphs={cfg.graphs})")
    counters = q.KernelCounters().install()                          # before build_engine, as Phase B
    t0 = time.time()
    parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    fwd = q.ForwardCounter(model)
    windows = {"wikitext": q.LOADERS["wikitext"](parts.tokenizer, a.windows, a.prompt, a.cont)}
    rec = q.measure_phase(model, windows, phase=arm, prompt=a.prompt, cont=a.cont, chunk=a.chunk,
                          floor_chunk=a.chunk, group=a.windows, device=cfg.device, ref_dir=a.ref_dir,
                          counters=counters, fwd=fwd, arms=("R",) if arm == "off" else ("ON",))
    rec = {"mode": "sane", "arm": arm, "model_tag": a.model_tag, "model": cfg.model, "revision": cfg.revision,
           "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"),
           "transformers": transformers.__version__, "load_s": round(time.time() - t0, 1),
           "census": {k: parts.info.get(k) for k in CENSUS_KEYS}, "fusion_modes": dict(cfg.fusion_modes),
           **rec, "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "status": "ok"}
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print(f"P115C_SANE {a.model_tag}/{arm} windows={rec['windows']} census={rec['census']} {rec['seconds']} s", flush=True)
    return 0


def self_test() -> int:
    on = {k: "auto" for k in KNOBS}
    ok = [ARMS == ("off", "on", "explicit"), arm_env_ok("on", on)[0], not arm_env_ok("off", on)[0],
          arm_env_ok("off", {k: "0" for k in KNOBS})[0], arm_env_ok("explicit", {k: "1" for k in KNOBS})[0],
          not arm_env_ok("on", {**on, "E4B_FUSE_ROUTER_EPI": "1"})[0], not arm_env_ok("x", on)[0],
          p109_box.self_test() == 0]
    print(f"p115c_box self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--prompts-only", action="store_true")
    p.add_argument("--mode", choices=("serve", "sane"))
    p.add_argument("--model")
    p.add_argument("--revision", default="")
    p.add_argument("--model-tag", default="")
    p.add_argument("--prompts")
    p.add_argument("--out")
    p.add_argument("--ref-dir")
    p.add_argument("--new", type=int, default=32)
    p.add_argument("--windows", type=int, default=12)
    p.add_argument("--prompt", type=int, default=512)
    p.add_argument("--cont", type=int, default=128)
    p.add_argument("--chunk", type=int, default=512)
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.prompts_only:
        return p109_box.prompts_main(a)
    arm = os.environ.get("P115C_ARM", "")
    ok, why = arm_env_ok(arm, os.environ)
    if not ok:
        raise SystemExit(f"REFUSED: {why}")
    if a.mode == "sane" and arm == "explicit":
        raise SystemExit("REFUSED: SANE has no explicit arm")
    return serve_main(a, arm) if a.mode == "serve" else sane_main(a, arm)


if __name__ == "__main__":
    sys.exit(main())
