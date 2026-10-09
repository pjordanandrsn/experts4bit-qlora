# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Exercise the real CUDA paged-server build with local random MoE checkpoints.

This is a correctness smoke, not a quality or speed benchmark. No model or
tokenizer downloads, mocks, or replacement residency/attention implementations.
Each family/stack runs in a fresh process, with serving knobs unset except the
local assets and the explicitly requested int4 levers.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

FAMILIES = ("qwen3_moe", "granitemoe", "mixtral", "qwen3_5_moe")
STACKS = ("default", "int4", "folds")
CELLS = tuple((f, s) for f in FAMILIES for s in STACKS
              if s != "folds" or f in ("qwen3_moe", "mixtral"))
PROMPT_TOKENS, NEW_TOKENS = 512, 4


def random_model(family):
    import torch
    from transformers import AutoModelForCausalLM
    from transformers import (GraniteMoeConfig, MixtralConfig, Qwen3MoeConfig,
                              Qwen3_5MoeTextConfig)

    common = dict(vocab_size=256, hidden_size=128, num_hidden_layers=2,
                  num_attention_heads=2, num_key_value_heads=2,
                  max_position_embeddings=8192, tie_word_embeddings=False,
                  bos_token_id=1, eos_token_id=2, pad_token_id=0)
    if family == "qwen3_moe":
        cfg = Qwen3MoeConfig(**common, intermediate_size=128, moe_intermediate_size=128,
                             head_dim=64, num_experts=4, num_experts_per_tok=2,
                             decoder_sparse_step=1, mlp_only_layers=[])
    elif family == "granitemoe":
        cfg = GraniteMoeConfig(**common, intermediate_size=128,
                               num_local_experts=4, num_experts_per_tok=2)
    elif family == "mixtral":
        cfg = MixtralConfig(**common, intermediate_size=128,
                            num_local_experts=4, num_experts_per_tok=2)
    elif family == "qwen3_5_moe":
        cfg = Qwen3_5MoeTextConfig(**common, head_dim=64, moe_intermediate_size=128,
                                   shared_expert_intermediate_size=128,
                                   num_experts=4, num_experts_per_tok=2,
                                   layer_types=["linear_attention", "full_attention"],
                                   linear_num_key_heads=2, linear_num_value_heads=2,
                                   linear_key_head_dim=64, linear_value_head_dim=64,
                                   linear_conv_kernel_dim=4)
    else:
        raise ValueError(f"unknown family: {family}")
    torch.manual_seed(1689)
    return AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).eval()


def write_checkpoint(model, path):
    """Write per-expert Qwen3/Mixtral and fused Granite/Qwen3.5 layouts."""
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    path.mkdir()
    tensors = {}
    family = model.config.model_type
    for name, value in model.state_dict().items():
        if family == "mixtral":
            name = name.replace(".mlp.", ".block_sparse_moe.")
        elif family == "qwen3_5_moe_text" and name.startswith("model."):
            name = "model.language_model." + name.removeprefix("model.")
        # The Qwen/Mixtral checkpoints use per-expert linears; Transformers
        # stores their random model as fused gate-first stacks in memory.
        if family in ("qwen3_moe", "mixtral") and name.endswith("experts.gate_up_proj"):
            base = name.removesuffix("gate_up_proj")
            for e in range(value.shape[0]):
                gate, up = value[e].chunk(2, dim=0)
                if family == "mixtral":
                    names = ("w1", "w3")
                else:
                    names = ("gate_proj", "up_proj")
                for role, tensor in zip(names, (gate, up), strict=True):
                    tensors[f"{base}{e}.{role}.weight"] = tensor.contiguous().clone()
        elif family in ("qwen3_moe", "mixtral") and name.endswith("experts.down_proj"):
            base = name.removesuffix("down_proj")
            role = "w2" if family == "mixtral" else "down_proj"
            for e in range(value.shape[0]):
                tensors[f"{base}{e}.{role}.weight"] = value[e].contiguous().clone()
        else:
            tensors[name] = value.contiguous().clone()
    if family == "qwen3_5_moe_text":
        import torch
        tensors["model.visual.patch_embed.proj.weight"] = torch.full((4, 3, 2, 2, 2), 42, dtype=torch.bfloat16)
        # Distracting MTP experts and a full decoder block must never join the
        # text tower's two MoE layers or its arena.
        for name, value in model.state_dict().items():
            if name.startswith("model.layers.1."):
                tensors["mtp.layers.0." + name.removeprefix("model.layers.1.")] = value.contiguous().clone()
    save_file(tensors, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": dict.fromkeys(tensors, "model.safetensors")}))
    if family == "qwen3_5_moe_text":
        from transformers import Qwen3_5MoeConfig
        Qwen3_5MoeConfig(text_config=model.config.to_dict()).save_pretrained(path)
    else:
        model.config.save_pretrained(path)
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocab.update({f"t{i}": i for i in range(4, model.config.vocab_size)})
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", bos_token="[BOS]",
                            eos_token="[EOS]", unk_token="[UNK]").save_pretrained(path)


def bake_arena(snapshot, root):
    """Quantize with the real loader, then relocate its NF4 bytes into an arena."""
    import torch
    from nvme_arena import bake_expert_tensors
    from safetensors.torch import save_file
    from experts4bit_qlora import load_moe_4bit_streaming
    from experts4bit_qlora.engines.hot_residency import target_modules
    from experts4bit_qlora.engines.nvme_experts import NF4_SEGMENTS

    model, _ = load_moe_4bit_streaming(str(snapshot), "cuda", torch.bfloat16, 8, 16)
    tensors = {}
    mods = target_modules(model)
    if len(mods) != 2:
        raise RuntimeError(f"random checkpoint loaded {len(mods)} expert layers, expected the two text layers")
    for layer, mod in enumerate(mods):
        e = mod.num_experts
        n1, k1 = mod._gate_up_shape
        n2, k2 = mod._down_shape
        stacks = {
            NF4_SEGMENTS["c_gu_p"]: mod.gate_up_proj.view(e, n1, k1 // 2),
            NF4_SEGMENTS["c_gu_a"]: mod.gate_up_absmax.view(e, n1, k1 // 64).float(),
            NF4_SEGMENTS["c_dn_p"]: mod.down_proj.view(e, n2, k2 // 2),
            NF4_SEGMENTS["c_dn_a"]: mod.down_absmax.view(e, n2, k2 // 64).float(),
        }
        for kind, stack in stacks.items():
            for expert in range(e):
                tensors[f"model.layers.{layer}.mlp.experts.{expert}.{kind}"] = stack[expert].cpu().contiguous().clone()
    snap = root / "nf4"
    snap.mkdir()
    save_file(tensors, str(snap / "model.safetensors"))
    del model, mods, mod, stacks, stack, tensors
    gc.collect()
    torch.cuda.empty_cache()
    arena = root / "nf4.arena"
    bake_expert_tensors(str(snap), str(arena),
                        name_template="model.layers.{layer}.mlp.experts.{expert}.{kind}",
                        kinds=tuple(NF4_SEGMENTS.values()), align=4096, log=lambda *a: None)
    return arena


def stage_offline_snapshot(snapshot, root, family):
    """A synthetic, content-addressed HF cache fixture; no upstream repository.

    The server's int4 source reader calls snapshot_download even for local
    paths. A complete offline cache entry exercises that unmodified reader.
    HF_HUB_CACHE must be set before the child imports huggingface_hub.
    """
    digest = hashlib.sha256()
    for path in sorted(snapshot.iterdir()):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    revision = digest.hexdigest()[:40]
    model_id = f"e4b-smoke-random/{family}"
    cache = root / "hf-cache" / ("models--" + model_id.replace("/", "--"))
    dest = cache / "snapshots" / revision
    dest.parent.mkdir(parents=True)
    snapshot.rename(dest)
    (cache / "refs").mkdir()
    (cache / "refs" / "main").write_text(revision)
    return model_id, dest, revision


def source_identity():
    import importlib.metadata
    import experts4bit_qlora

    root = Path(experts4bit_qlora.__file__).resolve().parents[1]
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    tree = hashlib.sha256()
    for path in sorted((root / "experts4bit_qlora").rglob("*.py")):
        tree.update(str(path.relative_to(root)).encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return {"source": str(root), "git_head": head.stdout.strip() if head.returncode == 0 else None,
            "package_sha256": tree.hexdigest(),
            "installed_e4b": importlib.metadata.version("experts4bit-qlora"),
            "gnf4": importlib.metadata.version("grouped-nf4-gemm")}


def check_graphs(capability, enabled, status, stats):
    """A supported GPU must capture and replay; a refused graph is not PASS."""
    if tuple(capability) >= (8, 9) and not enabled:
        raise RuntimeError("sm_89+ default decode graphs unexpectedly resolved to eager")
    if enabled:
        if not status or any(v != "graph" for v in status.values()):
            raise RuntimeError(f"default decode graphs failed to capture: {status}")
        if not stats or sum(v["replays"] for v in stats.values()) < 1:
            raise RuntimeError("default decode graphs never replayed")


def check_residual(info):
    residual = info.get("moe_residual") or {}
    if residual.get("licensed") != 2 or residual.get("partial") != 0 or residual.get("probe_errors") != []:
        raise RuntimeError(f"MoE residual was not fully licensed on both served layers: {residual}")
    if not info.get("fuse_t1_glue_n") or not all(info.get("fuse_t1_glue_r2_n") or (0,)):
        raise RuntimeError(f"residual folds did not engage: {info}")


def run_cell(family, stack, root):
    import torch
    from experts4bit_qlora.engines.hot_residency import target_modules
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this smoke never skips or falls back to CPU")
    # Clear inherited levers only in this child. The default stack must really
    # resolve from the shipped defaults, rather than the caller's bench flags.
    for name in list(os.environ):
        if name.startswith("E4B_"):
            del os.environ[name]
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    snapshot = root / "model"
    model = random_model(family)
    write_checkpoint(model, snapshot)
    del model
    model_id, snapshot, fixture_revision = stage_offline_snapshot(snapshot, root, family)
    arena = bake_arena(snapshot, root)
    calib = root / "calib.json"
    # Synthetic solver inputs, not measured bandwidth. The production build
    # immediately overrides the resulting placement to all-VRAM.
    calib.write_text(json.dumps({"cpu_bench": {"scatter_best": {"gbs": 1}},
                                 "gpu_bench": {"devices": [{"b_vram_triad_gbs": 1}]}}))
    os.environ.update(E4B_PAGED_MODEL=model_id, E4B_PAGED_ARENA=str(arena),
                      E4B_PAGED_CALIB=str(calib))
    expert_int4_unsupported = None
    if stack == "int4":
        # The shipped server serves the Qwen3.5 text tower. Its expert-int4
        # convention is currently absent (#1485), an explicit applicability
        # boundary agreed by the maintainer, not a caught execution failure.
        # Once that convention exists, this same cell requires expert int4 too.
        if family == "qwen3_5_moe":
            from experts4bit_qlora.arch.moe_conventions import MoEConventionError, convention_for
            try:
                convention_for("qwen3_5_moe_text")
            except MoEConventionError as exc:
                expert_int4_unsupported = f"{type(exc).__name__}: {exc}"
        os.environ["E4B_SERVE_ATTN_INT4"] = "1"
        if expert_int4_unsupported is None:
            os.environ["E4B_SERVE_EXP_INT4"] = "1"
    elif stack == "folds":
        os.environ.update({k: "auto" for k in
                           ("E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI")})
    cfg = PagedServeConfig.from_env()
    parts = build_engine(cfg)
    states = [type(m._hot_residency).__name__ for m in target_modules(parts.runner.model)]
    if len(states) != 2 or set(states) != {"_HybridTier"}:
        raise RuntimeError(f"real hybrid residency was not installed on both layers: {states}")
    if cfg.placement != "all-vram":
        raise RuntimeError(f"unexpected default placement: {cfg.placement}")
    expected_int4_layers = 0 if expert_int4_unsupported is not None else 2
    if stack == "int4" and (parts.info["int4_expert_layers"] != expected_int4_layers or parts.info["int4_attn_projections"] < 1):
        raise RuntimeError(f"int4 stack did not engage: {parts.info}")
    if family == "qwen3_moe" or stack == "folds":
        check_residual(parts.info)
    # One complete default-size prefill and three decode steps. A second
    # request also checks that slot reset survives a real hybrid state reuse.
    prompt = [4 + i % 252 for i in range(PROMPT_TOKENS)]
    for _ in range(2):
        rid = parts.scheduler.add_request(prompt, max_new_tokens=NEW_TOKENS, stop_ids=())
        for _ in range(PROMPT_TOKENS + NEW_TOKENS + 2):
            parts.scheduler.step()
            done = next((r for r in parts.scheduler.done if r.rid == rid), None)
            if done is not None:
                break
        else:
            raise RuntimeError("scheduler did not finish the request")
        if done.prompt_pos != PROMPT_TOKENS or len(done.out) != NEW_TOKENS or done.finish_reason != "length":
            raise RuntimeError(f"incomplete generation: prompt={done.prompt_pos}, output={done.out}, "
                               f"finish={done.finish_reason}")
        if not all(0 <= t < 256 for t in done.out):
            raise RuntimeError(f"out-of-vocabulary output: {done.out}")
    torch.cuda.synchronize()
    graph_status = parts.info["graph_status"]
    graph_stats = getattr(parts.runner, "graph_stats", None)
    check_graphs(torch.cuda.get_device_capability(), cfg.graphs, graph_status, graph_stats)
    return {"device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__, "source": source_identity(), "placement": cfg.placement,
            "synthetic_snapshot": {"id": model_id, "content_revision": fixture_revision},
            "residency_states": states, "graph_default_enabled": cfg.graphs,
            "decode_coverage": "graph capture and replay" if cfg.graphs else "eager default; graphs not exercised",
            "graph_status": graph_status, "graph_stats": graph_stats,
            "prefill_graph": parts.runner.prefill_graph_stats(), "int4_expert_layers": parts.info["int4_expert_layers"],
            "expert_int4_unsupported": expert_int4_unsupported,
            "moe_residual": parts.info["moe_residual"], "fusion_report": parts.info["fusion_report"],
            "int4_attn_projections": parts.info["int4_attn_projections"],
            "generated": [r.out for r in parts.scheduler.done], "requests_completed": len(parts.scheduler.done)}


def worker(args):
    row = {"family": args.family, "stack": args.stack, "status": "FAIL"}
    try:
        row["source"] = source_identity()
        row.update(run_cell(args.family, args.stack, Path(args.output_dir)))
        row["status"] = "PASS"
    except Exception as exc:
        row.update(exception=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    Path(args.result).write_text(json.dumps(row, indent=2, default=str) + "\n")
    return 0 if row["status"] == "PASS" else 1


def run_suite(root, timeout):
    deadline = time.monotonic() + timeout
    rows = []
    for family in FAMILIES:
        for stack in STACKS:
            if (family, stack) not in CELLS:
                continue
            cell = root / f"{family}-{stack}"
            cell.mkdir()
            result = cell / "result.json"
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--family", family,
                   "--stack", stack, "--output-dir", str(cell), "--result", str(result)]
            row = {"family": family, "stack": stack, "status": "FAIL"}
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(cmd, timeout)
                env = {k: v for k, v in os.environ.items() if not k.startswith("E4B_")}
                env.update(HF_HUB_CACHE=str(cell / "hf-cache"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           TOKENIZERS_PARALLELISM="false")
                with (cell / "worker.log").open("w") as log:
                    proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=remaining, env=env)
                if result.exists():
                    row = json.loads(result.read_text())
                    if proc.returncode != 0 and row.get("status") == "PASS":
                        row.update(status="FAIL", exception=f"worker exited {proc.returncode} after reporting PASS")
                else:
                    row["exception"] = f"worker exited {proc.returncode} without a result"
            except subprocess.TimeoutExpired:
                row["exception"] = f"suite exceeded its {timeout:g}s correctness-smoke deadline"
            rows.append(row)
            print(f"{row['status']} {family}/{stack}" +
                  (f": {row['exception']}" if row.get("exception") else ""), flush=True)
            if row.get("expert_int4_unsupported"):
                print(f"  expert-int4 UNSUPPORTED; attention-int4 exercised, experts NF4: "
                      f"{row['expert_int4_unsupported']}", flush=True)
    summary = {"schema": "e4b-gpu-serve-smoke/1", "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL",
               "random_weights": True, "downloads": False, "cells": rows}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, default=str), flush=True)
    return 0 if summary["status"] == "PASS" else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", help="new directory for local assets, worker logs and summary.json")
    p.add_argument("--timeout", type=float, default=300, help="whole-suite deadline in seconds (default: 300)")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--family", choices=FAMILIES, help=argparse.SUPPRESS)
    p.add_argument("--stack", choices=STACKS, help=argparse.SUPPRESS)
    p.add_argument("--result", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        return worker(args)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        p.error("--timeout must be positive")
    if args.output_dir:
        root = Path(args.output_dir).resolve()
        root.mkdir(parents=True, exist_ok=False)
        return run_suite(root, args.timeout)
    with tempfile.TemporaryDirectory(prefix="e4b-gpu-serve-smoke-") as tmp:
        return run_suite(Path(tmp), args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
