# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane FAM, Amendment 2 (bench/fam/PREREG-fam.md; e4b#1362): the box runs the largest shape first, on CPU.

``fam-qw36-1`` VOIDed: on the hybrid Qwen3.6 the box's first runner (shape 1, padded: 1 + 16 scratch slots) sized and
froze the model's linear-state pool at 17 slots, so the first shape-12 runner (12 + 16) was refused. The fix runs shape 12
first. It rests on three claims, each checked here on a tiny Qwen3.5-MoE hybrid:

1. **Order.** ``run_cells`` scores every shape-12 cell of a text before its shape-1 cells.
2. **The refusal and the fix at the box's slot counts.** ``enable_decode_graphs(capture=False)`` (the padded R's setup,
   which runs on CPU) freezes a 17-slot pool that a 28-slot runner cannot grow; sized 28 first, the 17-slot runner binds.
3. **No arithmetic change.** A shape-1 R pass is bit-identical on a 28- and a 17-slot pool, and bit-identical after a
   shape-12 pass on the same pool to one on a fresh pool: a reused slot is reset at bind. The decode-graph bucket
   selector reads the same rows from either pool size, bit for bit.

What CPU cannot run: the padded decode itself (the fused fp8 KV append). Shape 1 decodes in bucket 1, so its pass has no
padding rows and reads only its own slot.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.cache_utils", reason="needs transformers")
pytest.importorskip("transformers.models.qwen3_5_moe.configuration_qwen3_5_moe",
                    reason="needs transformers with Qwen3.5-MoE")

ROOT = Path(__file__).resolve().parents[1]
LANES = [ROOT / "bench" / d for d in ("fam", "p115", "p110", "p108", "p97")]
LIN, ATT = "linear_attention", "full_attention"
P, C = 24, 4


def _load(name, lane):
    for d in LANES:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench" / lane / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def hybrid():
    """Builds a tiny Qwen3.6-shaped hybrid (three linear layers to one attention layer, by default), seeded, with paged
    attention bound: the same weights on every call."""
    from hybrid_reference import reference_modeling
    from transformers import Qwen3_5MoeTextConfig
    q35 = reference_modeling("qwen3_5_moe")

    from experts4bit_qlora.engines import paged_attention

    def make(layer_types=(LIN, LIN, LIN, ATT)):
        cfg = Qwen3_5MoeTextConfig(vocab_size=128, hidden_size=64, num_hidden_layers=len(layer_types),
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=32, moe_intermediate_size=32,
                                   shared_expert_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
                                   linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                                   linear_value_head_dim=16, linear_conv_kernel_dim=4, layer_types=list(layer_types),
                                   max_position_embeddings=256)
        torch.manual_seed(0)
        m = q35.Qwen3_5MoeForCausalLM(cfg).to(torch.bfloat16).eval()
        paged_attention.register(m)
        return m
    return make


def _windows(n=12):
    g = torch.Generator().manual_seed(3)
    return [torch.randint(0, 128, (P + C,), generator=g).tolist() for _ in range(n)]


def _r_pass(box110, model, ws):
    """The R pass as the CPU instrument runs it: unpadded, device grouping on, stand-in decode attention."""
    lps, _ = box110.paged_pass(model, ws, P, C, P, "cpu", device_grouping=True, stand_in=True)
    return lps


def test_the_box_scores_the_largest_shape_first(monkeypatch):
    box = _load("fam_box", "fam")
    seen = []

    def measure_phase(model, ws, *, phase, group, ref_dir, arms, **kw):
        seen.append(group)
        return {"phase": phase, "arms": list(arms)}
    monkeypatch.setitem(sys.modules, "p115_quality", types.SimpleNamespace(measure_phase=measure_phase))
    windows = {t: [[0] * 4] * 36 for t in box.TEXTS}
    for config in ("OFF", "ON_auto"):
        seen.clear()
        cells = box.run_cells(None, windows, config=config, texts=box.TEXTS, shapes=(1, 12), sets=tuple(box.SETS),
                              prompt=2, cont=2, chunk=2, floor_chunk=1, device="cpu", ref_root="/nonexistent")
        per_text = len(seen) // len(box.TEXTS)
        for i in range(len(box.TEXTS)):
            groups = seen[i * per_text:(i + 1) * per_text]
            assert groups == sorted(groups, reverse=True) and set(groups) == {1, 12}, (config, groups)
        assert sorted(cells) == sorted(box.cell_key(t, s, n) for t in box.TEXTS for s in (1, 12) for n in box.SETS)


def _runner(model, n, scratch):
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner, kv_layers
    from experts4bit_qlora.serve_paged import _kv_geometry
    hkv, hd = _kv_geometry(model.config)
    kv = Fp8PagedKV(kv_layers(model, model.config.num_hidden_layers), hkv, hd, batch=n, max_tokens_per_seq=P + C + 16,
                    device="cpu", scratch_slots=scratch)
    return PagedModelRunner(model, kv, device="cpu")


def test_the_refusal_and_the_fix_at_the_box_slot_counts(hybrid):
    pytest.importorskip("row_pool", reason="needs grouped-nf4-gemm N-series")
    pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm N-series")
    box110 = _load("p110_box", "p110")
    scratch = max(box110.BUCKETS)
    old = hybrid()                                               # the registered order: shape 1, then shape 12
    first = _runner(old, 1, scratch)
    first.enable_decode_graphs(box110.BUCKETS, capture=False, verbose=False)
    assert first.linear_state.n_slots == 17 and first.linear_state.frozen
    with pytest.raises(RuntimeError, match="17 slots and a captured decode graph holds"):
        _runner(old, 12, scratch)
    new = hybrid()                                               # Amendment 2: shape 12, then shape 1
    big = _runner(new, 12, scratch)
    big.enable_decode_graphs(box110.BUCKETS, capture=False, verbose=False)
    assert big.linear_state.n_slots == 28 and big.linear_state.frozen
    small = _runner(new, 1, scratch)
    small.enable_decode_graphs(box110.BUCKETS, capture=False, verbose=False)
    assert small.linear_state is big.linear_state and small.linear_state.n_slots == 28


def test_a_shape_1_pass_is_bit_identical_on_a_28_and_a_17_slot_pool(hybrid):
    pytest.importorskip("row_pool", reason="needs grouped-nf4-gemm N-series")
    pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm N-series")
    from experts4bit_qlora.engines import linear_state
    box110 = _load("p110_box", "p110")
    ws = _windows(1)
    got = {}
    for n_slots in (28, 17):
        model = hybrid()
        linear_state.install(model, n_slots)
        got[n_slots] = _r_pass(box110, model, ws)
        assert model._e4b_linear_state.n_slots == n_slots
    assert all(torch.equal(a, b) for a, b in zip(got[28][0], got[17][0], strict=True))


def test_a_shape_1_pass_after_a_shape_12_pass_matches_a_fresh_pool(hybrid, monkeypatch):
    """The reset is ``PagedModelRunner.bind`` -> ``LinearStatePool.reset``: the one call ``paged_pass`` makes for every
    window, padded on the card or unpadded here."""
    pytest.importorskip("row_pool", reason="needs grouped-nf4-gemm N-series")
    pytest.importorskip("fp8_kv", reason="needs grouped-nf4-gemm N-series")
    from experts4bit_qlora.engines import linear_state
    box110 = _load("p110_box", "p110")
    ws = _windows(12)
    used = hybrid()
    _r_pass(box110, used, ws)                                    # slots 0-11 now carry the shape-12 windows' state
    assert all(used._e4b_linear_state.has[:12])
    resets, reset = [], linear_state.LinearStatePool.reset

    def spy(self, slot):
        resets.append((slot, self.has[slot]))
        return reset(self, slot)
    monkeypatch.setattr(linear_state.LinearStatePool, "reset", spy)
    after = _r_pass(box110, used, ws[:1])
    assert resets[0] == (0, True)                                # bind reset slot 0, which carried a window's state
    fresh = _r_pass(box110, hybrid(), ws[:1])
    assert all(torch.equal(a, b) for a, b in zip(after[0], fresh[0], strict=True))


def test_the_bucket_selector_reads_the_same_rows_from_either_pool_size(hybrid):
    """The padded decode gathers and scatters a bucket's rows through its device selector; the pool's size must not
    change what it reads."""
    from experts4bit_qlora.engines import linear_state, paged_attention
    prompt, steps = [5, 9, 2, 7, 1], 3
    runs = {}
    for n_slots in (28, 17):
        model = hybrid((LIN, LIN, LIN))                           # all linear: no K/V to bind
        pool = linear_state.install(model, n_slots)
        out = []
        with torch.no_grad():
            prev = paged_attention.set_context(paged_attention.PagedAttentionContext(kv=None, slots=[0], mode="decode"))
            try:
                model(input_ids=torch.tensor([prompt]), position_ids=torch.arange(len(prompt))[None], use_cache=False,
                      attention_mask={LIN: None, ATT: None})
            finally:
                paged_attention.set_context(prev)
            pool.mark([0])
            for step in range(steps):
                ctx = paged_attention.PagedAttentionContext(kv=None, slots=[0], mode="decode")
                ctx.kv = types.SimpleNamespace(_g_sel=torch.tensor([0]))
                assert linear_state._bucket_selector(ctx) is not None
                prev = paged_attention.set_context(ctx)
                try:
                    o = model(input_ids=torch.tensor([[11 + step]]), position_ids=torch.tensor([[len(prompt) + step]]),
                              use_cache=False, attention_mask={LIN: None, ATT: None})
                finally:
                    paged_attention.set_context(prev)
                out.append(o.logits[:, -1])
        runs[n_slots] = (out, {k: (pool.conv[k][0].clone(), pool.rec[k][0].clone()) for k in pool.conv})
    (lg_a, st_a), (lg_b, st_b) = runs[28], runs[17]
    assert all(torch.equal(a, b) for a, b in zip(lg_a, lg_b, strict=True))
    assert st_a.keys() == st_b.keys() and all(torch.equal(st_a[k][0], st_b[k][0]) and torch.equal(st_a[k][1], st_b[k][1])
                                              for k in st_a)
