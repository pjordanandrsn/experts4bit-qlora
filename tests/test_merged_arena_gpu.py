# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""On CUDA, through the real paged-server build (bench/d0/ARENA-DESIGN.md; run on a rented card by bench/gpu-contract).

Each case runs ``bench/smoke/gpu_serve_smoke.py``'s own ``run_cell`` (a tiny seeded random Qwen3-MoE, its arena,
``serve_paged.build_engine``, two full requests) in a fresh process. ``build_engine`` is wrapped only to keep the
``info`` it returns, the dict ``/health`` serves; nothing is replaced.

- **The default build** (no adapter, a base arena) carries no adapter and no merge record. Its residency states, graph
  status, fusion report and generated ids are recorded as junit properties, to compare with ``main`` on the same class
  of card.
- **A merged arena with its own adapter** (``E4B_PAGED_MERGED_ARENA=1``), merged by the writer from a ``bake_nf4`` base of
  the same staged checkpoint, builds and serves. ``/health``'s adapter
  record carries the arena's merge block, and its experts read ``merged``: the served fingerprint equals
  ``merge.merged_qfp_v1``. This case skips where ``experts4bit_qlora.merged_arena`` does not exist (a baseline tree).
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (bench/gpu-contract runs it)")

ROOT = Path(__file__).resolve().parents[1]

CHILD = r'''
import hashlib, importlib.util, json, os, sys
from pathlib import Path

root, mode, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
spec = importlib.util.spec_from_file_location("gpu_serve_smoke", "bench/smoke/gpu_serve_smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)
import experts4bit_qlora.serve_paged as sp

seen = {}
real_build = sp.build_engine


def keep_info(cfg):
    parts = real_build(cfg)
    seen["info"] = json.loads(json.dumps(parts.info, default=str))
    return parts


sp.build_engine = keep_info
if mode == "merged":
    import torch
    from safetensors.torch import save_file

    from experts4bit_qlora import merged_arena as MA
    from experts4bit_qlora.adapter_dir import arena_source
    from experts4bit_qlora.adapter_schema import arena_expert_fingerprint

    import nvme_bake_nf4

    def bake_then_merge(snapshot, where):
        """The writer's base: grouped-nf4-gemm's quantize bake of the smoke's staged snapshot (the smoke's own bake
        relocates the loader's bytes, a bake_mode the writer does not take). Then an expert adapter for the model and
        the merged arena, which is served instead."""
        arena = where / "nf4q.arena"
        nvme_bake_nf4.bake_nf4(str(snapshot), str(arena), log=lambda *a, **k: None)
        idx = json.load(open(str(arena) + ".index.json"))
        dims, n_layers = idx["model_dims"], int(idx["n_layers"])
        inter, hidden, n_exp, r, alpha = int(dims["I"]), int(dims["H"]), int(dims["E"]), 2, 4
        names = {li: f"model.layers.{li}.mlp.experts" for li in range(n_layers)}
        g = torch.Generator().manual_seed(7)
        tensors = {}
        for li, stack in names.items():
            for which, (n, k) in (("gate_up", (2 * inter, hidden)), ("down", (hidden, inter))):
                tensors[f"{stack}.{which}_lora_A"] = torch.randn(n_exp, r, k, generator=g)
                tensors[f"{stack}.{which}_lora_B"] = torch.randn(n_exp, n, r, generator=g) * 0.05
        fp = arena_expert_fingerprint(str(arena), MA._stand_ins(names, idx))
        fp.update(seconds=0.0)
        d = where / "adapter"
        d.mkdir()
        save_file(tensors, str(d / "adapter.safetensors"))
        body = (d / "adapter.safetensors").read_bytes()
        man = {"schema": "backend-adapter/2", "backend": "experts4bit", "weights": "adapter.safetensors",
               "files": {"adapter.safetensors": {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}},
               "resolved_model_revision": arena_source(idx)[1],
               "setup": {"quant_type": "nf4", "blocksize": 64, "r": r, "alpha": alpha, "attn_4bit": False,
                         "train_experts": True, "train_attention": False},
               "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()},
               "expert_absmax": {"form": "fp32"}, "train_fused_qkv": 0, "attn_quant": {"form": "bf16"},
               "versions": {}, "lora_scale_applied": {"attention": None, "experts": alpha / r},
               "fingerprints": {"experts": fp}, "run_id": "merged-arena-gpu", "created_at": "2026-10-23T00:00:00Z"}
        (d / "adapter_manifest.json").write_text(json.dumps(man))
        merged = where / "merged.arena"
        seen["merge"] = MA.write(str(arena), str(d), str(merged), log=lambda *a: None)
        os.environ.update(E4B_PAGED_ADAPTER=str(d), E4B_PAGED_MERGED_ARENA="1")
        return merged

    smoke.bake_arena = bake_then_merge
row = smoke.run_cell("qwen3_moe", "default", root)
out.write_text(json.dumps({"row": row, "seen": seen}, default=str))
'''


def _cell(tmp_path, mode):
    out = tmp_path / "cell.json"
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-c", CHILD, str(tmp_path), mode, str(out)], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-6000:])
    return json.loads(out.read_text())


def _record(record_property, prefix, row):
    for key in ("residency_states", "graph_status", "graph_default_enabled", "fusion_report", "moe_residual",
                "generated", "device", "capability"):
        record_property(f"{prefix}.{key}", json.dumps(row.get(key), sort_keys=True, default=str))


def test_the_default_build_carries_no_adapter_and_no_merge_record(tmp_path, record_property):
    got = _cell(tmp_path, "default")
    info, row = got["seen"]["info"], got["row"]
    assert info.get("adapter") is None, info.get("adapter")
    assert "merge" not in json.dumps(info), "a merge record on the default build"
    assert row["residency_states"] == ["_HybridTier", "_HybridTier"] and row["requests_completed"] == 2
    _record(record_property, "default", row)


def test_a_merged_arena_serves_with_its_own_adapter_and_reports_the_merge(tmp_path, record_property):
    pytest.importorskip("experts4bit_qlora.merged_arena")
    got = _cell(tmp_path, "merged")
    info, row, merge = got["seen"]["info"], got["row"], got["seen"]["merge"]
    adapter = info.get("adapter") or {}
    assert adapter.get("merged_arena") == merge, adapter
    assert adapter["fingerprints"]["experts"]["status"] == "verified (merged)"
    assert adapter["fingerprints"]["experts"]["served"] == merge["merged_qfp_v1"] != merge["base_qfp_v1"]
    assert adapter["base_match"]["experts"].startswith("merged (qfp-v1 ")
    assert row["residency_states"] == ["_HybridTier", "_HybridTier"] and row["requests_completed"] == 2
    _record(record_property, "merged", row)
