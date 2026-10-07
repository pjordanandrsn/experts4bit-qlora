# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P115's quality phase (``bench/p115/p115_quality.py``) on CPU, before any GPU is rented.

Two things are checked here, because no local card runs the fp8 paged KV or the bucketed path:

1. **The two-phase driver** (``measure_phase``) on a tiny Qwen3-MoE with the decode attention stood in and the reference
   unpadded (device grouping on, as the subject's R is, minus the bucket padding the card adds):
   - the OFF phase scores R, rep, half, chunk and mutant_scale on every window of both texts and writes R's log-probs;
   - the ON phase reads them back and scores ON against them; on the same model ON is R, so its NLL equals R's;
   - the decode attention is counted exactly; the mutant moves the scores.
2. **The per-step kernel counts the reducer registers**, on the same tiny model in bf16 with ``int4_b32`` stood in by
   torch functions: ``serve_paged._apply_fusions`` with the registered fused stack (fused q/k/v + the three folds)
   patches ``{L, 4L + 1, [L, L], L}`` modules, and ONE decode-shaped forward calls ``rmsnorm_rows`` L + 1 times,
   ``rmsnorm_resid_rows`` L, ``rope_norm_heads`` 2L, ``router_epilogue`` L and every ``qkv_proj`` once -- the
   ``PER_STEP`` table in ``p115_reduce.py``. A prefill-shaped forward calls none of them.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
LANES = [ROOT / "bench" / d for d in ("p115", "p110", "p108", "p97", "p109")]


def _load(name):
    for d in LANES:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench" / "p115" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny(dtype=torch.float32, layers=2):
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(vocab_size=256, hidden_size=128, intermediate_size=128, moe_intermediate_size=64,
                         num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                         num_experts=4, num_experts_per_tok=2, max_position_embeddings=512, decoder_sparse_step=1,
                         mlp_only_layers=[])
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(cfg).to(dtype).eval()


P, C, GROUP = 24, 6, 4
UNPADDED = {"device_grouping": True}


@pytest.fixture(scope="module")
def phases(tmp_path_factory):
    pytest.importorskip("transformers.models.qwen3_moe", reason="needs transformers with Qwen3-MoE")
    q = _load("p115_quality")
    g = torch.Generator().manual_seed(1)
    windows = {t: [torch.randint(0, 256, (P + C,), generator=g).tolist() for _ in range(2 * GROUP)] for t in q.TEXTS}
    model = _tiny()
    ref = tmp_path_factory.mktemp("ref")
    kw = dict(prompt=P, cont=C, chunk=P, floor_chunk=8, group=GROUP, device="cpu", ref_dir=str(ref), ref_kw=UNPADDED,
              stand_in=True)
    off = q.measure_phase(model, windows, phase="off", **kw)
    on = q.measure_phase(model, windows, phase="on", **kw)
    return q, windows, off, on, ref


def test_the_off_phase_scores_every_arm_and_writes_the_reference(phases):
    q, windows, off, _on, ref = phases
    for t in q.TEXTS:
        per = off["per_window"][t]
        assert [len(per[a]) for a in q.OFF_ARMS] == [8, GROUP, 8, 8, 8], t
        assert off["rep_identical"][t] is True
        assert sorted(p.name for p in ref.iterdir() if p.name.startswith(f"R_{t}_")) == [f"R_{t}_g0.pt", f"R_{t}_g1.pt"]
    assert off["windows_sha256"]["wikitext"] != off["windows_sha256"]["c4val1"]


def test_the_on_phase_scores_against_the_saved_reference(phases):
    q, _w, off, on, _ref = phases
    for t in q.TEXTS:
        r = {x["window"]: x["nll"] for x in off["per_window"][t]["R"]}
        assert len(on["per_window"][t]["ON"]) == 8
        for x in on["per_window"][t]["ON"]:              # the same model and arithmetic: ON is R
            assert x["nll"] == r[x["window"]] and x["argmax_agree"] == 1.0 and abs(x["kl"]) < 1e-6, (t, x)


def test_decode_attention_is_counted_and_the_mutant_moves_the_scores(phases):
    q, _w, off, on, _ref = phases
    for rec in (off, on):
        for t in q.TEXTS:
            for arm, passes in rec["engagement"][t].items():
                for e in passes:
                    assert e["decode_calls"] == (C - 1) * 2 * (2 if arm == "half" else 1), (arm, e)
                    assert e["grouping_flags_in_pass"]["device_grouping"] is True
    for t in q.TEXTS:
        r = {x["window"]: x["nll"] for x in off["per_window"][t]["R"]}
        moved = [abs(x["nll"] - r[x["window"]]) for x in off["per_window"][t]["mutant_scale"]]
        assert max(moved) > 1e-3, moved


# ------------------------------------------------------------ kernel counts --

def _kernel_stub():
    stub = types.ModuleType("int4_b32")

    def _norm(x, w, eps):
        xf = x.float()
        return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()

    def rmsnorm_rows(x, w, eps):
        return _norm(x, w, eps).to(torch.bfloat16)

    def rmsnorm_resid_rows(x, resid, w, eps, scale=1.0):
        if scale != 1.0:
            x = x * scale
        s = (x.float() + resid.float()).to(torch.bfloat16)
        return _norm(s, w, eps).to(torch.bfloat16), s

    def scaled_resid_add_rows(x, resid, scale):
        return resid + x * scale

    def _rope(xn, cos, sin):
        half = xn.shape[-1] // 2
        rot = torch.cat([-xn[..., half:], xn[..., :half]], dim=-1)
        return (xn * cos.float().unsqueeze(1) + rot * sin.float().unsqueeze(1)).to(torch.bfloat16)

    def rope_norm_heads(x, w, cos, sin, eps):
        return _rope(_norm(x, w, eps).to(torch.bfloat16).float(), cos, sin)

    def rope_heads(x, cos, sin):
        return _rope(x.float(), cos, sin)

    def router_epilogue(logits, k, norm, *, select_on_logits=False, bias=None):
        if select_on_logits:
            x = logits.float() + (bias.float() if bias is not None else 0.0)
            top, i = torch.topk(x, k, dim=-1)
            return x, torch.softmax(top, dim=-1), i
        probs = torch.softmax(logits.float(), dim=-1)
        v, i = torch.topk(probs, k, dim=-1)
        if norm:
            v = v / v.sum(dim=-1, keepdim=True)
        return probs, v, i

    for f in (rmsnorm_rows, rmsnorm_resid_rows, scaled_resid_add_rows, rope_norm_heads, rope_heads, router_epilogue):
        setattr(stub, f.__name__, f)
    return stub


def _tiny_granite(layers):
    from transformers import GraniteMoeConfig
    from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeForCausalLM
    cfg = GraniteMoeConfig(vocab_size=256, hidden_size=128, intermediate_size=64, num_hidden_layers=layers,
                           num_attention_heads=4, num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2,
                           max_position_embeddings=512, residual_multiplier=0.22, embedding_multiplier=12.0,
                           attention_multiplier=0.015625, logits_scaling=6.0)
    torch.manual_seed(0)
    return GraniteMoeForCausalLM(cfg).to(torch.bfloat16).eval()


@pytest.mark.parametrize("family,layers", [("qwen3", 2), ("qwen3", 3), ("granite", 2)])
def test_the_registered_fused_stack_patches_and_calls_the_per_step_counts(monkeypatch, family, layers):
    """The reading's stack (fused q/k/v + the three folds) on Qwen3-MoE; the proving run's (the three folds) on
    GraniteMoe. The census and the per-step glue-kernel calls must be the reducer's ``census_for`` / ``per_step``."""
    pytest.importorskip("transformers.models.qwen3_moe", reason="needs transformers with Qwen3-MoE")
    if family == "granite":
        pytest.importorskip("transformers.models.granitemoe", reason="needs transformers with GraniteMoe")
    q = _load("p115_quality")
    red = _load("p115_reduce")
    stub = _kernel_stub()
    monkeypatch.setitem(sys.modules, "int4_b32", stub)
    for k in ("E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI"):
        monkeypatch.setenv(k, "1")
    fused = family == "qwen3"
    counters = q.KernelCounters().install()            # before the fusions, as the box does
    from experts4bit_qlora.serve_paged import PagedServeConfig, _apply_fusions
    model = _tiny(torch.bfloat16, layers) if family == "qwen3" else _tiny_granite(layers)
    census = _apply_fusions(model, PagedServeConfig(fuse_qkv=fused))
    assert {k: census[k] for k in q.CENSUS_KEYS} == red.census_for(family, layers, fused=fused), census
    fwd = q.ForwardCounter(model)
    assert fwd.qkv_modules == (layers if fused else 0)
    with torch.no_grad():
        before = counters.snapshot()
        model(input_ids=torch.randint(0, 256, (4, 32)), use_cache=False)       # prefill-shaped: 128 rows, none fused
        assert counters.snapshot() == before, counters.snapshot()
        model(input_ids=torch.randint(0, 256, (4, 1)), position_ids=torch.full((4, 1), 32), use_cache=False)
    got = {k: v - before[k] for k, v in counters.snapshot().items()}
    assert {k: v for k, v in got.items() if v} == red.per_step(family, layers), got
    assert fwd.snapshot() == {"forwards": 2, "qkv_calls": 2 * layers if fused else 0}
