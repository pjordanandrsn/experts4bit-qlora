# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P121's box (``bench/p121/p121_box.py``) on CPU, before any GPU is rented.

No local card runs the fp8 paged KV, the bucket graphs or grouped-nf4-gemm's kernels, so what is checked here is the
wiring the reading depends on:
- ``RouteCounter`` counts K25 and M-tile calls made the way ``hot_residency`` makes them -- K25 through
  ``import nf4_smallm as m; m.gemm_nf4_grouped_smallm``, the M-tile through ``from nf4_grouped import
  gemm_4bit_grouped_captured`` inside the call -- and splits M-tile calls at 256 routed rows;
- the wrappers keep the kernels' signatures and are never stacked;
- the arms refuse any knob but their own, and an unnamed knob;
- ``p115_quality.measure_phase`` (P115 Phase B's instrument, at its registered bytes) records the counter's deltas per
  pass under ``kernels``, the key the reducer reads, at 16 windows a pass.
Real engagement (K25 calls 2 x layers per decode step under ``auto``, none under ``0``) is the proof's to show.
"""
import importlib.util
import inspect
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
LANES = [ROOT / "bench" / d for d in ("p121", "p115", "p110", "p108", "p97", "p109")]


def _load(name, lane="p121"):
    for d in LANES:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench" / lane / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _T:
    def __init__(self, rows):
        self.shape = (rows, 64)


def _stubs():
    sm, ng = types.ModuleType("nf4_smallm"), types.ModuleType("nf4_grouped")

    def gemm_nf4_grouped_smallm(x, packed, absmax, t_row0, t_rows, t_grp, order=None, *, block_n=32, kc=64, warps=4,
                                stages=3, lut="tree", dot_bf16=False, gather_div=None, scatter=None):
        return "k25"

    def gemm_4bit_grouped_captured(x, packed, absmax, t_row0, t_rows, t_grp, block_m):
        return "mtile"
    sm.gemm_nf4_grouped_smallm, ng.gemm_4bit_grouped_captured = gemm_nf4_grouped_smallm, gemm_4bit_grouped_captured
    return sm, ng


def test_the_counter_sees_calls_made_the_way_hot_residency_makes_them(monkeypatch):
    box = _load("p121_box")
    sm, ng = _stubs()
    monkeypatch.setitem(sys.modules, "nf4_smallm", sm)
    monkeypatch.setitem(sys.modules, "nf4_grouped", ng)
    rc = box.RouteCounter().install()
    import nf4_smallm as k25_mod                                   # hot_residency: K25 from the module, per call
    from nf4_grouped import gemm_4bit_grouped_captured             # hot_residency: the M-tile imported inside the call
    assert k25_mod.gemm_nf4_grouped_smallm(_T(16), 0, 0, 0, 0, 0) == "k25"
    assert gemm_4bit_grouped_captured(_T(128), 0, 0, 0, 0, 0, 16) == "mtile"       # a W16 step: 128 routed rows
    assert gemm_4bit_grouped_captured(_T(4096), 0, 0, 0, 0, 0, 16) == "mtile"      # a 512-token prefill chunk
    assert rc.snapshot() == {"k25": 1, "mtile_small": 1, "mtile_large": 1}
    assert rc.wrapped == ["nf4_smallm.gemm_nf4_grouped_smallm", "nf4_grouped.gemm_4bit_grouped_captured"]
    assert "gather_div" in inspect.signature(sm.gemm_nf4_grouped_smallm).parameters      # the signature survives
    box.RouteCounter().install()                                   # a second install wraps nothing
    k25_mod.gemm_nf4_grouped_smallm(_T(16), 0, 0, 0, 0, 0)
    assert rc.snapshot()["k25"] == 2


@pytest.mark.parametrize("arm,env,ok", [
    ("K0", {"E4B_NF4_GROUPED_SMALLM": "0"}, True),
    ("K1", {"E4B_NF4_GROUPED_SMALLM": "auto"}, True),
    ("K1", {}, False),                                             # the default, but not named
    ("K1", {"E4B_NF4_GROUPED_SMALLM": "1"}, False),                # 1 also sends T == 1 to K25
    ("K0", {"E4B_NF4_GROUPED_SMALLM": "0", "E4B_PAGED_FUSE_QKV": "auto"}, False),
    ("K1", {"E4B_NF4_GROUPED_SMALLM": "auto", "E4B_INT4_LEAN_GLUE": "0"}, False),
    ("K1", {"E4B_NF4_GROUPED_SMALLM": "auto", "GNF4_GEMV_BW": "0"}, False),
    ("Cm", {"E4B_NF4_GROUPED_SMALLM": "0", "E4B_NF4_T1_DEVICE_GROUPING": "1"}, True),     # P96's m
    ("Ct", {"E4B_NF4_GROUPED_SMALLM": "1", "E4B_NF4_T1_DEVICE_GROUPING": "0"}, True),     # P96's t
    ("Ct", {"E4B_NF4_GROUPED_SMALLM": "1"}, False),                                       # P96 named both knobs
])
def test_each_arm_runs_only_its_own_knob(arm, env, ok):
    box = _load("p121_box")
    assert box.arm_env_ok(arm, env)[0] is ok


def test_an_arm_without_its_knob_is_refused_before_the_engine_is_built(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E4B_", "GNF4_", "P121_"))}
    sep = ";" if sys.platform.startswith("win") else ":"
    env["PYTHONPATH"] = sep.join(str(d) for d in LANES)
    env["P121_ARM"] = "K1"
    out = subprocess.run([sys.executable, str(ROOT / "bench" / "p121" / "p121_box.py"), "--mode", "quality",
                          "--out", str(tmp_path / "q.json"), "--ref-dir", str(tmp_path / "ref")],
                         capture_output=True, text=True, env=env)
    assert out.returncode != 0 and "REFUSED: P121_ARM='K1'" in (out.stdout + out.stderr)


def _tiny():
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(vocab_size=256, hidden_size=128, intermediate_size=128, moe_intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32, num_experts=4,
                         num_experts_per_tok=2, max_position_embeddings=512, decoder_sparse_step=1, mlp_only_layers=[])
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(cfg).eval()


def test_the_quality_phase_records_the_route_deltas_per_pass(tmp_path, monkeypatch):
    pytest.importorskip("transformers.models.qwen3_moe", reason="needs transformers with Qwen3-MoE")
    box = _load("p121_box")
    q = _load("p115_quality", lane="p115")
    sm, ng = _stubs()
    monkeypatch.setitem(sys.modules, "nf4_smallm", sm)
    monkeypatch.setitem(sys.modules, "nf4_grouped", ng)
    rc = box.RouteCounter().install()
    P, C, G = 24, 4, 16
    g = torch.Generator().manual_seed(1)
    windows = {"wikitext": [torch.randint(0, 256, (P + C,), generator=g).tolist() for _ in range(G)]}
    rec = q.measure_phase(_tiny(), windows, phase="off", prompt=P, cont=C, chunk=P, floor_chunk=8, group=G, device="cpu",
                          ref_dir=str(tmp_path / "ref"), ref_kw={"device_grouping": True}, stand_in=True, counters=rc,
                          arms=box.QUALITY_ARMS["K0"])
    assert rec["group"] == G and rec["arms"] == list(box.QUALITY_ARMS["K0"])
    for arm in box.QUALITY_ARMS["K0"]:
        passes = rec["engagement"]["wikitext"][arm]
        assert len(passes) == 1 and passes[0]["kernels"] == {"k25": 0, "mtile_small": 0, "mtile_large": 0}, arm
    assert rec["counters_wrapped"] == rc.wrapped
