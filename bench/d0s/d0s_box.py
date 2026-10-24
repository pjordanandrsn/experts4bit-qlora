# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""D0s on the box (PREREG-d0s.md): the base bake, the writer's merges, then each arm served and read, one process each.

    python bench/d0s/d0s_box.py prepare --mode read|shake --out RUN [--snapshot DIR --adapters DIR]
    python bench/d0s/d0s_box.py arm --spec RUN/arms/<fixture>-<arm>.json

``prepare`` (one process):
- bakes the base arena (grouped-nf4-gemm's ``bake_nf4``, its default quantizer on the card);
- per fixture, derives the attention-only adapter (the fixture with its expert ``lora_B`` zeroed, a new directory whose
  manifest records what changed) and runs the merged-arena writer twice (``source="arena"`` and ``"checkpoint"``). A
  writer refusal is recorded as WRITER_REFUSED for that cell, never retried;
- writes one arm spec per arm.

``arm`` (one fresh process per arm): builds the server through ``serve_paged.build_engine`` from the spec's
environment, makes one teacher-forced pass over the rows on a fresh runner over the served model (per-row NLL sums and
the per-position argmax, from fp32 log-probs), then the cost runs (W1 and W16, each run twice and the second timed) and
the peak VRAM, and writes the arm's record.

``--mode shake`` does all of it on the GPU serve smoke's tiny random Qwen3-MoE (no download), with a tiny adapter of
each absmax form and synthetic rows, so every phase that needs no checkpoint runs before the reading.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
READ_MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
READ_REVISION = "7f1c97f440f06ce36705e4f2b843edb5925f4498"
FIXTURES = ("nodq", "dq")
ARMS = ("BASE-ATTN-1", "BASE-ATTN-2", "MERGED-Q", "MERGED-W")
COST_PROMPT, COST_NEW, COST_WIDTHS = 512, 64, (1, 16)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path, obj) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


# ---- the attention-only adapter --------------------------------------------------------------------------------


def attention_only(src: str, dst: str) -> dict:
    """A NEW adapter directory: ``src``'s files with every expert ``lora_B`` zeroed in ``adapter.safetensors`` (so the
    adapter carries no live expert LoRA and serves on the base arena, attention only), its manifest's file and tensor
    records updated and a ``derived_from`` note naming the change. Everything else, the fingerprints included, is
    ``src``'s: the base is the same."""
    import torch
    from safetensors.torch import load_file, save_file

    from experts4bit_qlora.adapter_dir import verify_adapter_dir

    ad = verify_adapter_dir(src)
    if os.path.lexists(dst):
        raise SystemExit(f"{dst} exists")
    tensors = load_file(os.path.join(src, "adapter.safetensors"))
    zeroed = sorted(n for n in ad.expert_tensors if n.endswith("_lora_B"))
    if not zeroed:
        raise SystemExit(f"{src} has no expert lora_B to zero")
    for n in zeroed:
        tensors[n] = torch.zeros_like(tensors[n])
    os.mkdir(dst)
    for name in ad.manifest["files"]:
        if name != "adapter.safetensors":
            shutil.copyfile(os.path.join(src, name), os.path.join(dst, name))
    save_file(tensors, os.path.join(dst, "adapter.safetensors"))
    man = json.loads(json.dumps(ad.manifest))
    body = os.path.join(dst, "adapter.safetensors")
    man["files"]["adapter.safetensors"] = {"sha256": sha256_file(body), "bytes": os.path.getsize(body)}
    man["derived_from_attention_only"] = {"source_adapter_id": ad.adapter_id, "zeroed_expert_lora_B": len(zeroed),
                                          "why": "the BASE-ATTN arm: attention LoRA only, on the base arena"}
    with open(os.path.join(dst, "adapter_manifest.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(man, indent=1, sort_keys=True) + "\n")
    out = verify_adapter_dir(dst)
    if any(bool(torch.any(out.tensors[n] != 0)) for n in out.expert_tensors if n.endswith("_lora_B")):
        raise SystemExit("an expert lora_B is still live in the attention-only adapter")
    return {"adapter_id": out.adapter_id, "source_adapter_id": ad.adapter_id, "zeroed": len(zeroed)}


# ---- prepare ---------------------------------------------------------------------------------------------------


def _smoke():
    spec = importlib.util.spec_from_file_location("gpu_serve_smoke", ROOT / "bench" / "smoke" / "gpu_serve_smoke.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _shake_model(out: Path) -> dict:
    """The smoke's tiny random Qwen3-MoE, staged as an offline HF cache entry (no download)."""
    smoke = _smoke()
    model = smoke.random_model("qwen3_moe")
    raw = out / "model"
    smoke.write_checkpoint(model, raw)
    model_id, snapshot, revision = smoke.stage_offline_snapshot(raw, out, "qwen3_moe")
    return {"model": model_id, "revision": revision, "snapshot": str(snapshot), "hf_cache": str(out / "hf-cache"),
            "_hf_model": model}


def _shake_adapters(out: Path, base_arena: Path, inputs: dict) -> dict:
    """For each absmax form a tiny /2 adapter (attention and expert LoRA) whose recorded base is the base arena baked
    on this card, plus synthetic rows."""
    import torch
    from safetensors.torch import save_file
    from torch import nn

    from experts4bit_qlora import adapter_schema as S
    from experts4bit_qlora import merged_arena as MA
    from experts4bit_qlora.adapter_dir import arena_source

    model = inputs.pop("_hf_model")
    idx = json.load(open(str(base_arena) + ".index.json"))
    dims, n_layers = idx["model_dims"], int(idx["n_layers"])
    inter, hidden, n_exp, r, alpha = int(dims["I"]), int(dims["H"]), int(dims["E"]), 2, 4
    names = {li: f"model.layers.{li}.mlp.experts" for li in range(n_layers)}
    fp = dict(S.arena_expert_fingerprint(str(base_arena), MA._stand_ins(names, idx)), seconds=0.0)
    cfg = model.config
    g = torch.Generator().manual_seed(7)
    tensors = {}
    for li in range(n_layers):
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = model.get_submodule(f"model.layers.{li}.self_attn.{p}")
            tensors[f"model.layers.{li}.self_attn.{p}.lora_A"] = torch.randn(r, lin.in_features, generator=g)
            tensors[f"model.layers.{li}.self_attn.{p}.lora_B"] = torch.randn(lin.out_features, r, generator=g) * 0.01
        for w, (n, k) in (("gate_up", (2 * inter, hidden)), ("down", (hidden, inter))):
            tensors[f"{names[li]}.{w}_lora_A"] = torch.randn(n_exp, r, k, generator=g)
            tensors[f"{names[li]}.{w}_lora_B"] = torch.randn(n_exp, n, r, generator=g) * 0.05
    attn_root = nn.Module()
    attn_root.model = model.model
    attn_fp = dict(S.quant_fingerprint(attn_root, "attention"), seconds=0.0)
    adapters = {}
    for fx in FIXTURES:
        d = out / "adapters" / fx
        d.mkdir(parents=True)
        save_file(tensors, str(d / "adapter.safetensors"))
        body = (d / "adapter.safetensors").read_bytes()
        efp = dict(fp)
        if fx == "dq":                      # the recorded trained form differs; its fp32 source is the bake's
            efp = {**fp, "sha256": hashlib.sha256(b"shake-dq-trained-form" + fp["sha256"].encode()).hexdigest(),
                   "fp32_source": {"sha256": fp["sha256"], "absmax_by_stack": fp["absmax_by_stack"]}}
        man = {"schema": "backend-adapter/2", "backend": "experts4bit", "weights": "adapter.safetensors",
               "files": {"adapter.safetensors": {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}},
               "resolved_model_revision": arena_source(idx)[1],
               "setup": {"quant_type": "nf4", "blocksize": 64, "r": r, "alpha": alpha, "attn_4bit": False,
                         "train_experts": True, "train_attention": True},
               "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()},
               "expert_absmax": {"form": "fp32" if fx == "nodq" else "double-quantized"}, "train_fused_qkv": 0,
               "attn_quant": {"form": "bf16"}, "versions": {},
               "lora_scale_applied": {"attention": alpha / r, "experts": alpha / r},
               "fingerprints": {"experts": efp, "attention": attn_fp}, "run_id": f"d0s-shake-{fx}",
               "created_at": "2026-10-24T00:00:00Z"}
        (d / "adapter_manifest.json").write_text(json.dumps(man))
        adapters[fx] = str(d)
    g2 = torch.Generator().manual_seed(11)
    rows = [torch.randint(4, int(cfg.vocab_size), (int(n),), generator=g2).tolist()
            for n in torch.randint(12, 40, (16,), generator=g2)]
    for fx in FIXTURES:
        _write_json(out / f"rows-{fx}.json", {"rows": [{"ids": r} for r in rows]})
    return {**inputs, "adapters": adapters, "rows": {fx: str(out / f"rows-{fx}.json") for fx in FIXTURES}}


def _read_inputs(out: Path, snapshot: str, adapters_dir: str) -> dict:
    return {"model": READ_MODEL, "revision": READ_REVISION, "snapshot": snapshot,
            "adapters": {fx: os.path.join(adapters_dir, f"fixture-{fx}-v2") for fx in FIXTURES},
            "rows": {fx: str(ROOT / "bench" / "d0" / "receipts" / "d0-p224s3" / fx / "heldout.json") for fx in FIXTURES},
            "hf_cache": os.environ.get("HF_HUB_CACHE")}


def prepare(mode: str, out: Path, snapshot: str | None, adapters_dir: str | None) -> dict:
    import nvme_bake_nf4 as nb

    from experts4bit_qlora import merged_arena as MA

    out.mkdir(parents=True, exist_ok=True)
    inputs = _shake_model(out) if mode == "shake" else _read_inputs(out, snapshot, adapters_dir)
    base = out / "base.arena"
    t0 = time.time()
    nb.bake_nf4(inputs["snapshot"], str(base), log=lambda *a, **k: None)       # the bake's default quantizer, on the card
    if mode == "shake":
        inputs = _shake_adapters(out, base, inputs)
    record = {"mode": mode, "inputs": inputs, "bake_s": round(time.time() - t0, 1),
              "base_index_sha256": sha256_file(str(base) + ".index.json"), "fixtures": {}}
    calib = out / "calib.json"
    calib.write_text(json.dumps({"cpu_bench": {"scatter_best": {"gbs": 1}},
                                 "gpu_bench": {"devices": [{"b_vram_triad_gbs": 1}]}}))
    (out / "arms").mkdir(exist_ok=True)
    for fx in FIXTURES:
        ad = inputs["adapters"][fx]
        attn = out / "adapters-derived" / f"{fx}-attention-only"
        attn.parent.mkdir(parents=True, exist_ok=True)
        fxrec = {"adapter": ad, "adapter_manifest_sha256": sha256_file(os.path.join(ad, "adapter_manifest.json")),
                 "attention_only": attention_only(ad, str(attn)), "cells": {}}
        cells = {"MERGED-Q": ("arena", None), "MERGED-W": ("checkpoint", inputs["snapshot"])}
        for cell, (source, ckpt) in cells.items():
            dst = out / f"{fx}-{cell}.arena"
            t0 = time.time()
            try:
                block = MA.write(str(base), ad, str(dst), source=source, checkpoint=ckpt, log=lambda *a: None)
                fxrec["cells"][cell] = {"status": "WRITTEN", "arena": str(dst), "merge": block,
                                        "write_s": round(time.time() - t0, 1)}
            except MA.MergeRefused as e:
                fxrec["cells"][cell] = {"status": "WRITER_REFUSED", "why": str(e), "write_s": round(time.time() - t0, 1)}
        for arm in ARMS:
            if arm.startswith("BASE-ATTN"):
                arena, adapter, merged = str(base), str(attn), "0"
            elif fxrec["cells"][arm]["status"] == "WRITTEN":
                arena, adapter, merged = fxrec["cells"][arm]["arena"], ad, "1"
            else:
                continue
            env = {"E4B_PAGED_MODEL": inputs["model"], "E4B_PAGED_REVISION": inputs["revision"],
                   "E4B_PAGED_ARENA": arena, "E4B_PAGED_CALIB": str(calib), "E4B_PAGED_ADAPTER": adapter,
                   "E4B_PAGED_MERGED_ARENA": merged}
            _write_json(out / "arms" / f"{fx}-{arm}.json", {"fixture": fx, "arm": arm, "env": env,
                                                            "rows": inputs["rows"][fx], "hf_cache": inputs["hf_cache"],
                                                            "out": str(out / "results" / f"{fx}-{arm}.json")})
        record["fixtures"][fx] = fxrec
    (out / "results").mkdir(exist_ok=True)
    _write_json(out / "prepare.json", record)
    return record


# ---- one arm -----------------------------------------------------------------------------------------------------


def teacher_forced(model, rows, device) -> list:
    """Per row: ``(nll sum, n, argmax list)`` over positions 0..len-2 predicting tokens 1..len-1 (D0's scoring), on a
    fresh runner over the served model: the first token prefilled, every later one fed teacher-forced through the
    paged decode path, fp32 log-probs from the served logits. The decode runs eagerly, one forward per step for every
    live row: a decode-graph replay never returns logits to the caller (P117's pattern); the served graphs run in the
    cost passes."""
    import torch

    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner, kv_layers
    from experts4bit_qlora.serve_paged import _kv_geometry

    cfg = getattr(model.config, "text_config", None) or model.config
    hkv, hd = _kv_geometry(model.config)
    n = len(rows)
    kv = Fp8PagedKV(kv_layers(model, int(cfg.num_hidden_layers)), hkv, hd, batch=n,
                    max_tokens_per_seq=max(len(r) for r in rows) + 16, device=device)
    runner = PagedModelRunner(model, kv, device=device)
    fwd, inner = [], model.forward

    def keep(*a, **kw):
        o = inner(*a, **kw)
        fwd.append(o.logits)
        return o

    nll, am = [0.0] * n, [[] for _ in rows]

    def score(rid, lp, target):
        nll[rid] -= float(lp[target])
        am[rid].append(int(lp.argmax()))

    model.forward = keep
    try:
        with torch.no_grad():
            for rid, r in enumerate(rows):
                runner.bind(rid, rid, r[:1])
            for rid, r in enumerate(rows):
                fwd.clear()
                runner.run_prefill([(rid, 0, 1)])
                score(rid, fwd[-1][0, -1].float().log_softmax(-1), r[1])
                runner.tokens[rid][-1] = r[1]
            t = 1
            while True:
                active = [rid for rid, r in enumerate(rows) if t < len(r) - 1]
                if not active:
                    break
                fwd.clear()
                runner.run_decode(active)
                if len(fwd) != 1 or fwd[0].shape[0] != len(active):
                    raise RuntimeError(f"step {t}: {[tuple(x.shape) for x in fwd]} logits for {len(active)} rows; "
                                       f"the eager decode is one forward over every live row")
                lps = fwd[0][:, -1].float().log_softmax(-1)
                for j, rid in enumerate(active):
                    score(rid, lps[j], rows[rid][t + 1])
                    runner.tokens[rid][-1] = rows[rid][t + 1]
                t += 1
    finally:
        model.forward = inner
    del runner, kv
    return [(nll[i], len(rows[i]) - 1, am[i]) for i in range(n)]


def cost(parts, prompt) -> dict:
    """W1 and W16: each run twice through the built scheduler, the second timed. TTFT and ITL from the requests' own
    records, tokens/s from the wall clock."""
    import torch

    out = {}
    for width in COST_WIDTHS:
        for attempt in range(2):
            sched = parts.scheduler
            before = len(sched.done)
            t0 = time.time()
            rids = [sched.add_request(prompt, max_new_tokens=COST_NEW, stop_ids=()) for _ in range(width)]
            while len(sched.done) - before < width:
                sched.step()
            torch.cuda.synchronize()
            wall = time.time() - t0
            reqs = [r for r in sched.done if r.rid in set(rids)]
        ttft = [r.ttft for r in reqs]
        itl = [(r.finished_at - r.first_token_at) / max(len(r.out) - 1, 1) for r in reqs]
        out[f"W{width}"] = {"ttft_s": sum(ttft) / len(ttft), "itl_s": sum(itl) / len(itl),
                            "tok_s": sum(len(r.out) for r in reqs) / wall, "requests": width}
    return out


def arm(spec_path: str) -> dict:
    spec = json.load(open(spec_path))
    os.environ.update({k: v for k, v in spec["env"].items()})
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    if spec.get("hf_cache"):
        os.environ["HF_HUB_CACHE"] = spec["hf_cache"]
    import torch

    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine

    torch.cuda.reset_peak_memory_stats()
    cfg = PagedServeConfig.from_env()
    t0 = time.time()
    parts = build_engine(cfg)
    build_s = time.time() - t0
    rows = [r["ids"] for r in json.load(open(spec["rows"]))["rows"]]
    model = parts.runner.model
    t0 = time.time()
    scored = teacher_forced(model, rows, cfg.device)
    pass_s = time.time() - t0
    prompt = [t for r in rows for t in r][:COST_PROMPT]
    prompt = (prompt * (COST_PROMPT // max(len(prompt), 1) + 1))[:COST_PROMPT]
    costs = cost(parts, prompt)
    info = parts.info
    adapter = info.get("adapter") or {}
    rec = {"fixture": spec["fixture"], "arm": spec["arm"], "env": spec["env"],
           "rows": [{"row": i, "nll": s[0], "n": s[1], "argmax": s[2]} for i, s in enumerate(scored)],
           "cost": costs, "peak_vram_bytes": torch.cuda.max_memory_allocated(),
           "build_s": round(build_s, 1), "pass_s": round(pass_s, 1),
           "served": {"adapter_id": adapter.get("adapter_id"), "fingerprints": adapter.get("fingerprints"),
                      "base_match": adapter.get("base_match"), "merged_arena": adapter.get("merged_arena"),
                      "graph_status": info.get("graph_status"), "placement": cfg.placement},
           "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability())}
    _write_json(spec["out"], rec)
    return rec


def run_arms(out: Path) -> dict:
    """Every arm spec in its own process (a fresh server each); a failed arm is recorded, the others still run."""
    status = {}
    for spec in sorted((out / "arms").glob("*.json")):
        log = out / "results" / (spec.stem + ".log")
        with open(log, "w") as f:
            r = subprocess.run([sys.executable, __file__, "arm", "--spec", str(spec)], stdout=f, stderr=subprocess.STDOUT,
                               timeout=3600)
        status[spec.stem] = r.returncode
    _write_json(out / "arms-status.json", status)
    return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--mode", choices=("read", "shake"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--snapshot")
    p.add_argument("--adapters")
    a1 = sub.add_parser("arm")
    a1.add_argument("--spec", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        if a.mode == "read" and not (a.snapshot and a.adapters):
            ap.error("--mode read needs --snapshot and --adapters")
        rec = prepare(a.mode, Path(a.out), a.snapshot, a.adapters)
        print(json.dumps({fx: {c: v["status"] for c, v in r["cells"].items()} for fx, r in rec["fixtures"].items()}))
        return 0
    if a.cmd == "arm":
        arm(a.spec)
        return 0
    status = run_arms(Path(a.out))
    print(json.dumps(status))
    return 0 if all(v == 0 for v in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
