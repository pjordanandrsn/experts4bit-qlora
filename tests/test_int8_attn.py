# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""``E4B_SERVE_ATTN_INT8`` (engines/int8_attn.py; lane P149, grouped-nf4-gemm K37) on a tiny Mixtral, on CPU: the
census, the three paths, the engagement record, the refusals and serve_paged's lever.

A stand-in computes the dequantized product where K37's ``linear_i8`` would, so these run on any grouped-nf4-gemm. With
one that carries K37, tests/test_int8_attn_interp.py runs the real kernel under Triton's interpreter, in its own process
(the last test). The kernel's own contract (relative RMS at most 1.05 x cuBLAS's, one-hot exact, bitwise) is
grouped-nf4-gemm's ``kernel/test_dense_i8.py``, checked on a CUDA card."""
import os
import pathlib
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from experts4bit_qlora import serve_paged as sp  # noqa: E402
from experts4bit_qlora.engines import int8_attn as ia  # noqa: E402

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")


def _quantize(w):
    """K36's grid (``dense_gemv.quantize_rows_int8``), which K37 re-exports: per output row, ``s = absmax / 127``."""
    wf = w.float()
    amax = wf.abs().amax(dim=1)
    s = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    q = torch.clamp(torch.round(wf / s[:, None]), -127, 127).to(torch.int8)
    return q.contiguous(), s.contiguous()


def _ref_linear(x, q, s):
    """A stand-in for ``linear_i8``: the dequantized product in fp32, rounded once to bf16."""
    assert x.dim() == 2 and x.shape[0] <= ia.ROWS_MAX and x.is_contiguous()
    return (x.float() @ (q.float() * s[:, None]).T).to(torch.bfloat16)


def _mixtral(layers=2, seed=0):
    torch.manual_seed(seed)
    cfg = transformers.MixtralConfig(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=layers,
                                     num_attention_heads=4, num_key_value_heads=2, num_local_experts=4,
                                     num_experts_per_tok=2, max_position_embeddings=64)
    return transformers.MixtralForCausalLM(cfg).to(torch.bfloat16).eval()


def _enable(m):
    return ia.enable_serve_attn_int8(m, linear=_ref_linear, quantize=_quantize)


def _rounded(m):
    """The same model with each attention weight replaced by ``bf16(q * s)``: P147's fake-quant round trip."""
    with torch.no_grad():
        for layer in m.model.layers:
            for name in PROJ:
                lin = getattr(layer.self_attn, name)
                q, s = _quantize(lin.weight)
                lin.weight.copy_((q.float() * s[:, None]).to(torch.bfloat16))
    return m


@pytest.fixture
def routed(monkeypatch):
    monkeypatch.setattr(ia, "_on_device", lambda x: True)   # CPU calls take the kernel path, as CUDA calls do


def test_the_census_swaps_every_attention_projection_keeps_its_name_and_leaves_the_head():
    m = _mixtral(layers=3)
    head = m.get_output_embeddings()
    assert _enable(m) == 3 * 4                                   # Mixtral-8x7B's 32 layers: 128
    for layer in m.model.layers:
        kids = dict(layer.self_attn.named_children())
        assert set(kids) == set(PROJ) and all(isinstance(kids[k], ia.Int8Linear) for k in PROJ)
    assert m.get_output_embeddings() is head and type(head) is torch.nn.Linear
    assert ia.engagement(m) == {"modules": 12, "routed_modules_by_rows": {}, "passed_modules_by_rows": {}}


def test_the_swap_keeps_only_codes_and_scales_on_the_grid():
    m = _mixtral()
    w = m.model.layers[0].self_attn.q_proj.weight.detach().clone()
    _enable(m)
    mod = m.model.layers[0].self_attn.q_proj
    assert list(mod.parameters()) == [] and mod.bias is None and not hasattr(mod, "weight")
    assert mod.state_dict() == {}                               # non-persistent: nothing new in a checkpoint
    q, s = _quantize(w)
    assert mod.q.dtype == torch.int8 and mod.s.dtype == torch.float32
    assert torch.equal(mod.q, q) and torch.equal(mod.s, s)


def test_the_rope_fold_still_matches_every_swapped_attention():
    """R2's norm-less rope fold licenses a module on its children's names and its own params / buffers: the swap
    keeps the names and puts the codes on the children, so the fold engages exactly as it does on S0."""
    from experts4bit_qlora.engines import glue_r2
    m = _mixtral()
    _enable(m)
    kernel_side = SimpleNamespace(rope_heads=lambda *a, **k: None)
    assert all(glue_r2._patch_attention_rope_only(layer.self_attn, kernel_side) for layer in m.model.layers)


def test_a_count_off_the_census_refuses_and_swaps_nothing(monkeypatch):
    m = _mixtral()
    real = ia.attention_projections
    monkeypatch.setattr(ia, "attention_projections", lambda model: list(real(model))[:-1])   # one projection missed
    with pytest.raises(RuntimeError, match="exactly 4 per layer"):
        _enable(m)
    assert not any(isinstance(x, ia.Int8Linear) for x in m.modules())


@pytest.mark.parametrize("rows,path", [(1, "routed"), (2, "routed"), (16, "routed"), (17, "passed"), (40, "passed")])
def test_each_call_takes_its_path_and_is_counted(routed, rows, path):
    m = _mixtral()
    _enable(m)
    mod = m.model.layers[0].self_attn.o_proj
    x = torch.randn(rows, mod.in_features).to(torch.bfloat16)
    out = mod(x)
    want = torch.nn.functional.linear(x, (mod.q.float() * mod.s[:, None]).to(torch.bfloat16))
    assert out.shape == (rows, mod.out_features) and out.dtype == torch.bfloat16
    if path == "routed":
        assert mod.routed == {rows: 1} and mod.passed == {}
        assert torch.allclose(out.float(), want.float(), rtol=2 ** -7, atol=1e-3)
    else:
        assert mod.passed == {rows: 1} and mod.routed == {}
        assert torch.equal(out, want)                           # the transient round trip, then F.linear


def test_a_strided_activation_is_copied_never_refused(routed):
    m = _mixtral()
    _enable(m)
    mod = m.model.layers[0].self_attn.q_proj
    x = torch.randn(mod.in_features, 3).to(torch.bfloat16).T     # 3 rows, not contiguous
    assert not x.is_contiguous()
    assert mod(x).shape == (3, mod.out_features) and mod.routed == {3: 1}


def test_a_cpu_call_without_the_seam_dequantizes_and_is_counted_as_passed():
    m = _mixtral()
    _enable(m)
    mod = m.model.layers[1].self_attn.k_proj
    mod(torch.randn(1, mod.in_features).to(torch.bfloat16))
    assert mod.passed == {1: 1} and mod.routed == {}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_a_non_bf16_call_dequantizes_on_the_bf16_grid_and_is_counted_as_passed(routed, dtype):
    """The docstring's promise: a call that is not bf16 takes the transient path even on the device, and reads the
    weight P147 rounded (bf16(q * s)), cast to the call's dtype."""
    m = _mixtral()
    _enable(m)
    mod = m.model.layers[0].self_attn.v_proj
    x = torch.randn(1, mod.in_features, dtype=dtype)
    out = mod(x)
    want = torch.nn.functional.linear(x, (mod.q.float() * mod.s[:, None]).to(torch.bfloat16).to(dtype))
    assert mod.passed == {1: 1} and mod.routed == {} and out.dtype == dtype
    assert torch.equal(out, want)


def test_a_prefill_is_the_rounded_model_bit_for_bit_and_every_call_is_counted(routed):
    m, ref = _mixtral(), _rounded(_mixtral())
    _enable(m)
    with torch.no_grad():
        m(torch.randint(0, 128, (1, 5)))                        # a decode-sized call: the kernels
        ids = torch.randint(0, 128, (1, 20))                    # a prefill: transient dequantization
        assert torch.equal(m(ids).logits, ref(ids).logits)
    assert ia.engagement(m) == {"modules": 8, "routed_modules_by_rows": {"5": 8}, "passed_modules_by_rows": {"20": 8}}


def test_the_refusals():
    with pytest.raises(RuntimeError, match="biased"):
        ia.Int8Linear(torch.nn.Linear(8, 4, bias=True).to(torch.bfloat16), _ref_linear, _quantize)
    with pytest.raises(RuntimeError, match="not a bf16"):
        ia.Int8Linear(torch.nn.Linear(8, 4, bias=False), _ref_linear, _quantize)
    with pytest.raises(RuntimeError, match="matched no attention projection"):
        ia.enable_serve_attn_int8(torch.nn.Sequential(torch.nn.Linear(4, 4)), linear=_ref_linear, quantize=_quantize)
    m = _mixtral()
    _enable(m)
    with pytest.raises(RuntimeError, match="matched no attention projection"):   # nothing plain is left to swap
        _enable(m)


def test_a_grouped_nf4_gemm_without_k37_refuses_with_a_sentence(monkeypatch):
    import builtins
    real = builtins.__import__

    def no_dense_i8(name, *a, **k):
        if name == "dense_i8":
            raise ImportError("No module named 'dense_i8'")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_dense_i8)
    with pytest.raises(RuntimeError, match="needs grouped-nf4-gemm with dense_i8.linear_i8"):
        ia.enable_serve_attn_int8(_mixtral())


def test_fuse_qkv_refuses_int8_projections_rather_than_crash():
    from experts4bit_qlora.engines.qkv_fuse import fuse_qkv
    torch.manual_seed(0)
    cfg = transformers.Qwen3MoeConfig(vocab_size=128, hidden_size=64, intermediate_size=96, moe_intermediate_size=32,
                                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                      num_experts=4, num_experts_per_tok=2, max_position_embeddings=64)
    m = transformers.Qwen3MoeForCausalLM(cfg).to(torch.bfloat16).eval()
    assert _enable(m) == 4                                      # no census on qwen3_moe: its four, q/k norms left
    with pytest.raises(RuntimeError, match="Int8Linear"):
        fuse_qkv(m)


# --- serve_paged: the lever and the census -------------------------------------------------------------------------

@pytest.fixture
def lever_env(monkeypatch):
    for k in sp.LEVER_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(ia.ENV, "1")
    monkeypatch.setattr(ia, "kernel", lambda: (_ref_linear, _quantize))
    return monkeypatch


def test_the_lever_is_snapshotted_and_swaps_through_apply_levers(lever_env):
    assert ia.ENV in sp.LEVER_ENV
    m = _mixtral()
    out = sp._apply_levers(m, None, None)
    assert out["attn_int8_projections"] == 8 and ia.engagement(m)["modules"] == 8


@pytest.mark.parametrize("other,cfg,match", [
    ("E4B_SERVE_ATTN_INT4", None, "refuses to combine with int4 attention"),
    ("E4B_SERVE_ATTN_INT4_CALIB", None, "refuses to combine with int4 attention"),
    (None, SimpleNamespace(dense_gemv_bw="1"), "refuses to combine with E4B_DENSE_GEMV_BW=1"),
])
def test_the_lever_refuses_to_combine_before_anything_is_swapped(lever_env, other, cfg, match):
    if other:
        lever_env.setenv(other, "1")
    m = _mixtral()
    with pytest.raises(RuntimeError, match=match):
        sp._apply_levers(m, cfg, None)
    assert not any(isinstance(x, ia.Int8Linear) for x in m.modules())


@pytest.mark.parametrize("value", ["2", "true", "on"])
def test_the_lever_reads_0_or_1_only(lever_env, value):
    lever_env.setenv(ia.ENV, value)
    with pytest.raises(ValueError, match="0 or 1"):
        sp._apply_levers(_mixtral(), None, None)


def test_the_lever_off_reports_zero_and_the_census_is_empty(lever_env):
    lever_env.setenv(ia.ENV, "0")
    m = _mixtral()
    out = sp._apply_levers(m, None, None)
    assert out["attn_int8_projections"] == 0
    assert sp._attn_int8_census(m, out, {}, {}) == {"projections": 0}


def _r2(attns, mode="auto", **note):
    return ({"fuse_t1_glue_r2_n": [2, attns]},
            {"modes": {"E4B_FUSE_T1_GLUE_R2": mode}, "folds": {"E4B_FUSE_T1_GLUE_R2": {"patched": [2, attns], **note}}})


def test_the_census_records_the_rope_fold_on_every_int8_attention(lever_env):
    m = _mixtral()
    levers = sp._apply_levers(m, None, None)
    assert sp._attn_int8_census(m, levers, *_r2(2, kernel_gaps=0)) == {
        "projections": 8, "attention_modules": 2, "r2_mode": "auto", "r2_rope_fold_attentions": 2}


@pytest.mark.parametrize("attns,mode,note,raises", [
    (1, "auto", {}, True),                                      # the fold missed a swapped module: the arms differ
    (1, "1", {}, True),
    (0, "0", {}, False),                                        # the fold off: nothing to compare
    (0, "auto", {"skipped": "no kernel"}, False),               # skipped in both arms alike
    (1, "auto", {"kernel_gaps": 1}, False),
])
def test_the_census_refuses_a_rope_fold_that_missed_an_int8_attention(lever_env, attns, mode, note, raises):
    m = _mixtral()
    levers = sp._apply_levers(m, None, None)
    if raises:
        with pytest.raises(RuntimeError, match="R2's rope fold patched 1 attention modules, not the 2"):
            sp._attn_int8_census(m, levers, *_r2(attns, mode, **note))
    else:
        assert sp._attn_int8_census(m, levers, *_r2(attns, mode, **note))["r2_rope_fold_attentions"] == attns


def test_the_census_refuses_a_projection_the_fusions_replaced(lever_env):
    m = _mixtral()
    levers = sp._apply_levers(m, None, None)
    m.model.layers[0].self_attn.o_proj = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="7 are Int8Linear after the fusions"):
        sp._attn_int8_census(m, levers, *_r2(2))


# --- the real kernel: tests/test_int8_attn_interp.py, interpreted, in its own process ---------------------------------

def test_the_kernel_path_passes_interpreted_in_its_own_process():
    """CI's full suite imports triton compiled before test_int8_attn_interp.py can set TRITON_INTERPRET, so that file
    skips itself in-process; here it runs in a fresh process. Without K37 (``dense_i8``) it skips, and so does this."""
    pytest.importorskip("triton")
    here = pathlib.Path(__file__).resolve().parent
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider",
                          str(here / "test_int8_attn_interp.py")], capture_output=True, text=True, cwd=here.parent,
                         env={**os.environ, "TRITON_INTERPRET": "1"}, timeout=900)
    last = (out.stdout.strip().splitlines() or [""])[-1]
    # a module-level skip collects nothing: pytest exits 5 (no tests collected)
    if out.returncode in (0, 5) and re.match(r"1 skipped(, \d+ warnings?)? in ", last) and "without dense_i8" in out.stdout:
        pytest.skip("grouped-nf4-gemm without dense_i8 (K37)")
    ok = out.returncode == 0 and re.match(r"4 passed(, \d+ warnings?)? in ", last)
    assert ok, out.stdout[-3000:] + out.stderr[-2000:]
