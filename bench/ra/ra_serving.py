#!/usr/bin/env python3
"""Real RA decode/quality wrappers over frozen helpers and release-default models.

One invocation per component in the selected release venv. No speed switch is
set here. The supervisor must verify wheel/input provenance before invocation.
"""
from __future__ import annotations

import argparse
import copy
import functools
import json
import os
import sys
import time
from pathlib import Path

import ra_env
import ra_fallback
import ra_quality
import ra_routes
import ra_stage


def delta(after, before):
    return {k: after[k] - before.get(k, 0) for k in after}


def feature(mode, patched, calls, *, scope, fallback_calls=None):
    if mode not in ("0", "1", "auto") or type(patched) is not int or patched < 0 or type(calls) is not int or calls < 0:
        raise ValueError("unresolved feature/counter")
    state = "off" if mode == "0" else "on" if patched else "inapplicable"
    if (mode == "0" and patched) or (mode == "1" and not patched) or (state == "on" and not calls) or (state != "on" and calls):
        raise ValueError("vacuous or contradictory feature engagement")
    if fallback_calls is not None and (type(fallback_calls) is not int or fallback_calls < 0):
        raise ValueError("unresolved fallback counter")
    return {"mode": state, "patched": patched, "calls": calls, "fallback_calls": fallback_calls,
            "fallback_coverage": "OBSERVED" if fallback_calls is not None else "UNVERIFIED", "calls_scope": scope}


def fusion_features(info, kernels, qkv_calls, *, scope, fallback, defaults):
    ra_fallback.check_fold_reports(info)
    modes = info["fusion_modes"]
    if modes != defaults["fusion_modes"]:
        raise ValueError("assembly modes differ from independent release defaults")
    layers, attention = info["fuse_t1_glue_r2_n"]
    fields = {
        "qkv": ("E4B_PAGED_FUSE_QKV", info["fuse_qkv_n"], qkv_calls),
        "rms_glue": ("E4B_FUSE_T1_GLUE", info["fuse_t1_glue_n"], kernels["rmsnorm_rows"]),
        "residual_glue": ("E4B_FUSE_T1_GLUE_R2", layers, kernels["rmsnorm_resid_rows"]),
        "rope_glue": ("E4B_FUSE_T1_GLUE_R2", attention, kernels["rope_norm_heads"] + kernels["rope_heads"]),
        "router_epilogue": ("E4B_FUSE_ROUTER_EPI", info["fuse_router_epilogue_n"], kernels["router_epilogue"]),
    }
    return {name: feature(modes[key], patched, calls, scope=scope,
                          fallback_calls=fallback.get(name, {}).get("fallback_calls"))
            for name, (key, patched, calls) in fields.items()}


def glue_census(info):
    layers, attention = info["fuse_t1_glue_r2_n"]
    return {"rms_glue": info["fuse_t1_glue_n"], "residual_glue": layers, "rope_glue": attention,
            "router_epilogue": info["fuse_router_epilogue_n"]}


def build_instrumented(server, instrument, cfg, *, routes=None):
    """Install kernel counters before folds bind them; QKV hooks before capture."""
    counters = instrument.KernelCounters().install()
    forwards = []
    original = server._apply_fusions

    @functools.wraps(original)
    def folds(model, *args, **kwargs):
        candidates = routes.candidates(model) if routes else []
        result = original(model, *args, **kwargs)
        if routes:
            routes.assemble(model, candidates, cfg.fusion_modes)
        forward = instrument.ForwardCounter(model)
        forward.ra_glue = ra_fallback.GlueObserver().install(model)
        forwards.append(forward)
        return result

    server._apply_fusions = folds
    try:
        parts = server.build_engine(cfg)
    finally:
        server._apply_fusions = original
    if len(forwards) != 1:
        raise ValueError("one instrumented fusion assembly required")
    return parts, counters, forwards[0]


@ra_routes.observed
def decode(spec, helper, instrument, server, torch, nf4, *, routes=None):
    pf = json.loads(Path(spec["prompts"]).read_bytes())
    rows = pf["rows"]
    if len(rows) != 16 or any(len(row) != 512 or any(type(t) is not int or t < 0 for t in row) for row in rows) or \
            helper.digest(rows) != pf["prompts_sha256"]:
        raise ValueError("decode prompt bytes/shape")
    cfg = server.PagedServeConfig.from_env()
    defaults = ra_fallback.resolved_defaults(server, cfg)
    start_dispatch = nf4.dispatch_counts()
    start = time.perf_counter()
    parts, counters, fwd = build_instrumented(server, instrument, cfg, routes=routes)
    build_dispatch = delta(nf4.dispatch_counts(), start_dispatch)
    record = {"status": "ok", "model": cfg.model, "revision": cfg.revision,
              "load_s": time.perf_counter() - start, "config": {k: v for k, v in vars(cfg).items() if k != "token"},
              "resolved_buckets": list(cfg.buckets), "graphs": cfg.graphs,
              "graph_status": {str(k): v for k, v in (parts.info.get("graph_status") or {}).items()},
              "prompts_sha256": pf["prompts_sha256"], "dispatch_build": build_dispatch,
              "short": spec["short"], "long": spec["long"], "reps": spec["reps"], "workloads": {}}
    for name, batch in helper.WORKLOADS.items():
        walls, digests, last = {}, {}, {}
        before = nf4.dispatch_counts()
        for n in (spec["short"], spec["long"]):
            helper.run_pass(parts, torch, rows[:batch], n)
            walls[str(n)], digests[str(n)] = [], []
            for _ in range(spec["reps"]):
                wall, _, tokens = helper.run_pass(parts, torch, rows[:batch], n)
                walls[str(n)].append(wall)
                digests[str(n)].append(helper.digest(tokens))
                last[str(n)] = tokens
        record["workloads"][name] = {"batch": batch, "walls": walls, "rep_digests": digests, "tokens": last,
                                       "dispatch": delta(nf4.dispatch_counts(), before),
                                       **helper.slope(walls[str(spec["short"])], walls[str(spec["long"])], batch,
                                                      spec["short"], spec["long"])}
    record["graph_stats"] = {str(k): dict(v) for k, v in parts.runner.graph_stats.items()}
    record["kernels"] = counters.snapshot()
    record["forward_counts"] = fwd.snapshot()
    record["dispatch_total"] = delta(nf4.dispatch_counts(), start_dispatch)
    record["fusions"] = {k: copy.deepcopy(parts.info[k]) for k in instrument.CENSUS_KEYS}
    record["fusion_modes"] = copy.deepcopy(parts.info["fusion_modes"])
    record["resolved_defaults"] = defaults
    record["glue_modules"] = fwd.ra_glue.snapshot()
    record["route_evidence"] = routes.evidence(parts.info, fwd.qkv_calls)
    record["route_modules"] = routes.snapshot()
    record["unscoped_native_calls"] = routes.unscoped_calls
    record["fallback_evidence"] = {**fwd.ra_glue.evidence(glue_census(parts.info)), **record["route_evidence"]}
    record["features"] = fusion_features(parts.info, record["kernels"], fwd.qkv_calls, scope="build-capture+warm+timed",
                                          fallback=record["fallback_evidence"], defaults=defaults)
    calls = record["route_evidence"]["decode_gemv"]["calls"]
    if calls <= 0 or (cfg.graphs and record["graph_stats"].get("1", {}).get("replays", 0) <= 0):
        raise ValueError("W1 GEMV capture/replay not engaged")
    record["features"]["decode_gemv"] = feature("auto", parts.info["moe_layers"], calls,
                                                  scope="build-capture+warm+timed",
                                                  fallback_calls=record["route_evidence"]["decode_gemv"]["fallback_calls"])
    record["proves_gpu_engagement"] = False
    return record


@ra_routes.observed
def quality(spec, instrument, server, torch, nf4, *, routes=None):
    cfg = server.PagedServeConfig.from_env()
    defaults = ra_fallback.resolved_defaults(server, cfg)
    if cfg.graphs:
        raise ValueError("quality requires named eager fixture")
    parts, counters, fwd = build_instrumented(server, instrument, cfg, routes=routes)
    parts.runner.model.eval()
    windows = json.loads(Path(spec["windows"]).read_bytes())
    if set(windows) != {"wikitext"} or len(windows["wikitext"]) != 12 or any(
            len(w) != 512 + spec["cont"] or any(type(t) is not int or t < 0 for t in w) for w in windows["wikitext"]):
        raise ValueError("quality window shape")
    original = instrument.p110_box.paged_pass

    @functools.wraps(original)
    def observed_pass(*args, **kwargs):
        before = nf4.dispatch_counts()
        glue_before = fwd.ra_glue.snapshot()
        route_before = routes.snapshot()
        qkv_before = fwd.qkv_calls
        result, engagement = original(*args, **kwargs)
        engagement["gemv_dispatch"] = delta(nf4.dispatch_counts(), before)
        engagement["route_evidence"] = routes.evidence(parts.info, fwd.qkv_calls - qkv_before, route_before)
        engagement["fallback_evidence"] = {**fwd.ra_glue.evidence(glue_census(parts.info), glue_before),
                                           **engagement["route_evidence"]}
        return result, engagement

    instrument.p110_box.paged_pass = observed_pass
    try:
        record = ra_quality.measure_r(instrument, parts.runner.model, windows, phase="off", prompt=512,
                                      cont=spec["cont"], chunk=512, floor_chunk=512, group=spec["group"],
                                      device=cfg.device, ref_dir=spec["ref_dir"], counters=counters, fwd=fwd)
    finally:
        instrument.p110_box.paged_pass = original
    for engagement in record["engagement"]["wikitext"]["R"]:
        engagement["features"] = fusion_features(parts.info, engagement["kernels"], engagement["qkv_calls"], scope="R-pass",
                                                       fallback=engagement["fallback_evidence"], defaults=defaults)
        calls = engagement["route_evidence"]["decode_gemv"]["calls"]
        if spec["group"] == 1 and calls <= 0:
            raise ValueError("group-1 GEMV did not dispatch")
        if spec["group"] == 12 and calls != 0:
            raise ValueError("group-12 unexpectedly dispatched singleton GEMV")
        engagement["features"]["decode_gemv"] = feature("auto", parts.info["moe_layers"] if spec["group"] == 1 else 0,
                                                           calls, scope="R-pass",
                                                           fallback_calls=engagement["route_evidence"]["decode_gemv"]["fallback_calls"])
    record.update(status="ok", model=cfg.model, revision=cfg.revision, resolved_defaults=defaults,
                  glue_modules=fwd.ra_glue.snapshot(), proves_gpu_engagement=False,
                  fusions={k: copy.deepcopy(parts.info[k]) for k in instrument.CENSUS_KEYS},
                  fusion_modes=copy.deepcopy(parts.info["fusion_modes"]))
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("decode", "quality"))
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--instruments", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if not all(p.is_absolute() for p in (args.spec, args.instruments, args.out)):
        raise ValueError("component paths must be absolute")
    ra_env.check_current(args.mode, os.environ)
    ra_stage.verify(args.instruments)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.instruments))
    import torch
    import nf4_grouped
    import p109_box
    import p115_quality
    from experts4bit_qlora import serve_paged
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("one usable CUDA GPU required")
    prop = torch.cuda.get_device_properties(0)
    if "RTX 5090" not in prop.name or (prop.major, prop.minor) != (12, 0):
        raise ValueError("registered RTX 5090/sm_120 required")
    spec = json.loads(args.spec.read_bytes())
    if any(k in spec and not Path(spec[k]).is_absolute() for k in ("prompts", "windows", "ref_dir")):
        raise ValueError("input/ref paths must be absolute")
    proof = spec["kind"] == "PROOF"
    if spec["kind"] not in ("PROOF", "READING"):
        raise ValueError("unknown run kind")
    if args.mode == "decode":
        if (spec["short"], spec["long"], spec["reps"]) != ((8, 24, 1) if proof else (32, 160, 3)):
            raise ValueError("decode battery changed")
        record = decode(spec, p109_box, p115_quality, serve_paged, torch, nf4_grouped)
    else:
        if spec["cont"] != (32 if proof else 128) or spec["group"] not in (1, 12):
            raise ValueError("quality battery changed")
        record = quality(spec, p115_quality, serve_paged, torch, nf4_grouped)
    args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
