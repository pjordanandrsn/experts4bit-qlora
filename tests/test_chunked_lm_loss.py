"""The opt-in chunked causal-LM loss (``E4B_CHUNKED_LM_LOSS``, engines/chunked_lm_loss.py) against Hugging Face's stock loss.

What is pinned, on tiny configs of every family in ``SUPPORTED`` (CPU, plus CUDA when present):

* the loss equals the stock loss to fp32 rounding and EVERY gradient (the head included -- these tiny models train all of their
  parameters) equals the stock gradient to fp32 rounding, with the router auxiliary loss on, a chunk size that does not divide
  the supervised token count, ignored labels inside rows, and micro-batch 2 with right padding and an attention mask;
* ``num_items_in_batch``, ``shift_labels`` and an all-ignored batch (stock's nan, zero gradients) follow Hugging Face;
* bf16 and CPU autocast stay within a bf16 rounding of the stock gradients;
* the switch off is the stock path byte for byte: unset, nothing is patched; ``torch.no_grad`` forwards and generation run the
  stock forward even when patched; ``disable_chunked_lm_loss`` restores it exactly;
* refusals: a class outside the table, a replaced ``loss_function``, a hooked head; and the run-time probe catches a change made
  after ``lm_head`` the table does not describe, re-running that call stock;
* ``auto``'s size gate: a forward under it is the stock forward exactly, one at it chunks, and the 1 GiB gate separates the TC1
  shapes it was set between;
* ``enable_fast_train`` applies it from the environment variable (``auto`` with its gate) and ``disable_fast_train`` unwinds it.

Why the tolerances: the chunked path changes only the ORDER of fp32 summations -- the cross-entropy summed per chunk and then
across chunks, the head's matmul blocked over a chunk's rows instead of all of them, a trainable head's weight gradient summed
over chunks. So the loss is held to 8 fp32 ulps of its magnitude and each gradient tensor to 1e-5 of its own largest element
(measured: at most 1 ulp on the loss and 1.4e-6 on a gradient, over the eleven families) -- or of 1e-3 of the largest
gradient anywhere in the model, whichever is larger. The floor is for tensors whose whole gradient is tiny next to the
model's: the reorder's rounding reaches them from upstream at the scale of the LARGE gradients, not theirs. On
qwen3_5_moe the Gated DeltaNet's dt_bias / A_log gradients peak near 2e-5 against a model maximum near 0.2, and a CI
runner read dt_bias at 1.3e-5 of its own maximum (1.79e-10 absolute; #1159's run 37323695463) where the Mac reads
2e-6. The floor is 2e-9 absolute there, still 1/7000 of dt_bias's own gradient, so a wrong scale or a dropped term
in it stays far outside the bound.
"""
import math
import warnings

import pytest
import torch

from experts4bit_qlora.engines import chunked_lm_loss as C

tr = pytest.importorskip("transformers")

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
V = 97
EPS32 = torch.finfo(torch.float32).eps
LOSS_ULPS = 8
GRAD_REL = 1e-5
GRAD_FLOOR = 1e-3        # of the model's largest gradient; see the module docstring


def _common():
    return dict(vocab_size=V, hidden_size=64, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)


def _config(family):
    c = _common()
    if family == "qwen3_moe":
        return tr.Qwen3MoeForCausalLM, tr.Qwen3MoeConfig(**c, intermediate_size=128, moe_intermediate_size=32, num_experts=8,
                                                         num_experts_per_tok=2, num_hidden_layers=2, head_dim=16,
                                                         router_aux_loss_coef=0.01)
    if family == "mixtral":
        return tr.MixtralForCausalLM, tr.MixtralConfig(**c, intermediate_size=64, num_local_experts=4, num_experts_per_tok=2,
                                                       num_hidden_layers=2, router_aux_loss_coef=0.02)
    if family == "olmoe":
        return tr.OlmoeForCausalLM, tr.OlmoeConfig(**c, intermediate_size=64, num_experts=8, num_experts_per_tok=2,
                                                   num_hidden_layers=2, router_aux_loss_coef=0.01)
    if family == "granitemoe":
        return tr.GraniteMoeForCausalLM, tr.GraniteMoeConfig(**c, intermediate_size=64, num_local_experts=4, num_experts_per_tok=2,
                                                             num_hidden_layers=2, logits_scaling=8.0, router_aux_loss_coef=0.01)
    if family == "granitemoeshared":
        return tr.GraniteMoeSharedForCausalLM, tr.GraniteMoeSharedConfig(
            **c, intermediate_size=64, shared_intermediate_size=32, num_local_experts=4, num_experts_per_tok=2,
            num_hidden_layers=2, logits_scaling=6.0, router_aux_loss_coef=0.01)
    if family == "granitemoehybrid":
        return tr.GraniteMoeHybridForCausalLM, tr.GraniteMoeHybridConfig(
            **c, intermediate_size=64, shared_intermediate_size=32, num_local_experts=4, num_experts_per_tok=2,
            num_hidden_layers=2, logits_scaling=6.0, router_aux_loss_coef=0.01, layer_types=["mamba", "attention"],
            mamba_n_heads=4, mamba_d_head=32, mamba_d_state=16, mamba_n_groups=1, mamba_chunk_size=8)
    if family == "gpt_oss":
        return tr.GptOssForCausalLM, tr.GptOssConfig(**c, intermediate_size=64, num_local_experts=4, num_experts_per_tok=2,
                                                     num_hidden_layers=2, head_dim=16, router_aux_loss_coef=0.01,
                                                     layer_types=["sliding_attention", "full_attention"], sliding_window=8)
    if family == "ernie4_5_moe":
        return tr.Ernie4_5_MoeForCausalLM, tr.Ernie4_5_MoeConfig(**c, intermediate_size=64, moe_intermediate_size=32,
                                                                 moe_num_experts=8, moe_k=2, num_hidden_layers=2,
                                                                 moe_layer_start_index=0, router_aux_loss_coef=0.01)
    if family == "lfm2_moe":
        return tr.Lfm2MoeForCausalLM, tr.Lfm2MoeConfig(**c, intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=4,
                                                       num_experts=8, num_experts_per_tok=2, num_dense_layers=1,
                                                       layer_types=["conv", "full_attention", "conv", "full_attention"])
    if family == "qwen3_5_moe":
        return tr.Qwen3_5MoeForCausalLM, tr.Qwen3_5MoeTextConfig(
            **c, moe_intermediate_size=32, shared_expert_intermediate_size=32, num_hidden_layers=2, head_dim=16, num_experts=8,
            num_experts_per_tok=2, linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=16,
            linear_value_head_dim=16, layer_types=["linear_attention", "full_attention"], router_aux_loss_coef=0.01)
    if family == "nemotron_h":
        return tr.NemotronHForCausalLM, tr.NemotronHConfig(
            **c, intermediate_size=64, num_hidden_layers=3, head_dim=16, layers_block_type=["mamba", "moe", "attention"],
            n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=32, moe_shared_expert_intermediate_size=32,
            n_groups=1, n_group=1, topk_group=1, mamba_num_heads=4, mamba_head_dim=16, ssm_state_size=16, chunk_size=8)
    raise KeyError(family)


FAMILIES = ["qwen3_moe", "mixtral", "olmoe", "granitemoe", "granitemoeshared", "granitemoehybrid", "gpt_oss", "ernie4_5_moe",
            "lfm2_moe", "qwen3_5_moe", "nemotron_h"]
NO_AUX = {"lfm2_moe", "nemotron_h"}


def _model(family, dev="cpu", dtype=torch.float32, seed=0):
    cls, cfg = _config(family)
    torch.manual_seed(seed)
    m = cls(cfg).to(dtype).to(dev)
    m.train()
    return m


def _batch(dev, mb=2, seq=24, pad=5, seed=3):
    """``mb`` rows of ``seq`` tokens: row 0 has a masked prompt (5 ignored labels); the last row is right-padded by ``pad``
    (labels -100 and attention 0 on the pads)."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, V, (mb, seq), generator=g)
    labels = ids.clone()
    labels[0, :5] = -100
    att = torch.ones_like(ids)
    if mb > 1 and pad:
        labels[-1, seq - pad:] = -100
        att[-1, seq - pad:] = 0
    return ids.to(dev), labels.to(dev), att.to(dev)


def _step(m, ids, labels, att=None, **kw):
    m.zero_grad(set_to_none=True)
    out = m(input_ids=ids, labels=labels, attention_mask=att, **kw)
    out.loss.backward()
    return out, {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}


def _assert_loss_close(a, b):
    a, b = float(torch.as_tensor(a).detach()), float(torch.as_tensor(b).detach())
    assert abs(a - b) <= LOSS_ULPS * EPS32 * max(abs(a), 1.0), (a, b)


def _assert_grads_close(ga, gb, rel=GRAD_REL):
    assert ga.keys() == gb.keys()
    top = max((float(g.float().abs().max()) for g in ga.values()), default=0.0)
    for n in ga:
        x, y = ga[n].float(), gb[n].float()
        scale = float(x.abs().max())
        err = float((x - y).abs().max())
        assert err <= rel * max(scale, GRAD_FLOOR * top) + 1e-30, \
            f"{n}: max |diff| {err:.3e} vs max |grad| {scale:.3e} (model max {top:.3e})"


# --------------------------------------------------------------------------------------------------------------- semantics --

def test_env_parsing(monkeypatch):
    for v, want in (("", None), ("0", None), ("off", None), ("1", C.DEFAULT_CHUNK), ("on", C.DEFAULT_CHUNK),
                    ("512", 512), ("2048", 2048)):
        monkeypatch.setenv("E4B_CHUNKED_LM_LOSS", v)
        assert C.chunked_lm_loss_requested() == want, v
    monkeypatch.delenv("E4B_CHUNKED_LM_LOSS")
    assert C.chunked_lm_loss_requested() is None and C.chunked_lm_loss_min_bytes() is None
    for v in ("auto", " AUTO "):                            # chunks of DEFAULT_CHUNK, behind the size gate
        monkeypatch.setenv("E4B_CHUNKED_LM_LOSS", v)
        assert C.chunked_lm_loss_requested() == C.DEFAULT_CHUNK and C.chunked_lm_loss_min_bytes() == C.AUTO_MIN_LOGITS_BYTES == 1 << 30
    for v in ("1", "512"):
        monkeypatch.setenv("E4B_CHUNKED_LM_LOSS", v)
        assert C.chunked_lm_loss_min_bytes() is None
    for bad in ("-4", "abc"):
        monkeypatch.setenv("E4B_CHUNKED_LM_LOSS", bad)
        with pytest.raises(ValueError):
            C.chunked_lm_loss_requested()


@pytest.mark.parametrize("family", FAMILIES)
def test_loss_and_every_gradient_match_stock(family):
    """Micro-batch 2 with right padding and an attention mask, a masked prompt, the router auxiliary loss on, 7-token chunks
    (the 37 supervised tokens leave a 2-token tail)."""
    ids, labels, att = _batch("cpu")
    kw = {} if family in NO_AUX else {"output_router_logits": True}
    ref_out, ref = _step(_model(family), ids, labels, att, **kw)
    m = _model(family)
    assert C.enable_chunked_lm_loss(m, 7) == 1
    out, got = _step(m, ids, labels, att, **kw)
    assert out.logits is None and ref_out.logits is not None
    if family not in NO_AUX:
        assert out.aux_loss is not None and torch.equal(out.aux_loss, ref_out.aux_loss)
    _assert_loss_close(ref_out.loss, out.loss)
    _assert_grads_close(ref, got)
    assert m.lm_head.weight.grad is not None              # the head's gradient is compared too (tied heads under the embedding's name)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("chunk", [1, 5, 16, 37, 38, 4096])
def test_chunk_sizes_including_ones_that_do_not_divide(dev, chunk):
    ids, labels, att = _batch(dev)
    ref_out, ref = _step(_model("qwen3_moe", dev), ids, labels, att)
    m = _model("qwen3_moe", dev)
    C.enable_chunked_lm_loss(m, chunk)
    out, got = _step(m, ids, labels, att)
    _assert_loss_close(ref_out.loss, out.loss)
    _assert_grads_close(ref, got)


def test_auto_gate_runs_small_forwards_stock_and_large_ones_chunked():
    """``auto``'s gate is the stock fp32 logits' size, positions x vocabulary x 4 bytes, against ``min_logits_bytes``: one byte
    over it the call is the stock forward exactly (logits returned, loss and every gradient ``torch.equal`` to an unpatched
    model's, counted in ``small_calls``); at it the call chunks. Re-enabling changes the gate in place."""
    ids, labels, att = _batch("cpu")
    need = labels.numel() * V * 4                          # 2 x 24 positions at V = 97: what the stock path would upcast
    ref_out, ref = _step(_model("qwen3_moe"), ids, labels, att)
    m = _model("qwen3_moe")
    assert C.enable_chunked_lm_loss(m, 7, min_logits_bytes=need + 1) == 1
    small, chunked = C.CHUNKED_LM_LOSS_STATS["small_calls"], C.CHUNKED_LM_LOSS_STATS["chunked_calls"]
    out, got = _step(m, ids, labels, att)
    assert out.logits is not None and torch.equal(out.loss, ref_out.loss)
    assert ref.keys() == got.keys() and all(torch.equal(ref[k], got[k]) for k in ref)
    assert C.CHUNKED_LM_LOSS_STATS["small_calls"] == small + 1 and C.CHUNKED_LM_LOSS_STATS["chunked_calls"] == chunked
    assert C.enable_chunked_lm_loss(m, 7, min_logits_bytes=need) == 0 and m._e4b_chunked_lm_loss.min_bytes == need
    out, got = _step(m, ids, labels, att)
    assert out.logits is None and C.CHUNKED_LM_LOSS_STATS["chunked_calls"] == chunked + 1
    _assert_loss_close(ref_out.loss, out.loss)
    _assert_grads_close(ref, got)


def test_auto_gate_separates_the_tc1_shapes_it_was_set_on():
    """The 1 GiB gate against the shapes it was chosen between (Qwen3's 151,936-token vocabulary): TC1's field recipe at its
    largest -- the two longest of the 1,200 Alpaca rows, 755 tokens each, padded together -- stays stock; one packed 4,096-token
    row chunks; Mixtral's 32,000-token vocabulary stays stock at 4,096."""
    def fp32(positions, vocab):
        return positions * vocab * 4
    assert fp32(2 * 755, 151936) < C.AUTO_MIN_LOGITS_BYTES <= fp32(4096, 151936)
    assert fp32(4096, 32000) < C.AUTO_MIN_LOGITS_BYTES


def test_ignored_rows_tail_padding_and_tied_head():
    """Whole ignored stretches, a row with a single supervised token, and a head tied to the input embeddings (the head's
    gradient and the embedding's land in one tensor)."""
    cls, cfg = _config("qwen3_moe")
    cfg.tie_word_embeddings = True
    ids, labels, att = _batch("cpu", mb=3, seq=20, pad=6)
    labels[1, :] = -100
    labels[1, 7] = ids[1, 7]
    labels[2, 3:9] = -100

    def mk():
        torch.manual_seed(0)
        m = cls(cfg).float()
        m.train()
        return m
    ref_out, ref = _step(mk(), ids, labels, att)
    m = mk()
    assert m.lm_head.weight is m.model.embed_tokens.weight
    C.enable_chunked_lm_loss(m, 4)
    out, got = _step(m, ids, labels, att)
    _assert_loss_close(ref_out.loss, out.loss)
    _assert_grads_close(ref, got)


def test_all_ignored_is_nan_with_zero_gradients_like_stock():
    ids, labels, att = _batch("cpu")
    labels[:] = -100
    ref_out, ref = _step(_model("qwen3_moe"), ids, labels, att)
    m = _model("qwen3_moe")
    C.enable_chunked_lm_loss(m, 7)
    out, got = _step(m, ids, labels, att)
    assert math.isnan(float(ref_out.loss.detach())) and math.isnan(float(out.loss.detach()))
    assert ref.keys() == got.keys()
    for n in ref:
        assert torch.equal(ref[n], got[n]), n


def test_num_items_in_batch_and_shift_labels_follow_hugging_face():
    ids, labels, att = _batch("cpu")
    n_items = torch.tensor(53)                 # the Trainer's count across accumulation steps: the loss is a SUM over it
    ref_out, ref = _step(_model("olmoe"), ids, labels, att, num_items_in_batch=n_items)
    m = _model("olmoe")
    C.enable_chunked_lm_loss(m, 6)
    out, got = _step(m, ids, labels, att, num_items_in_batch=n_items)
    _assert_loss_close(ref_out.loss, out.loss)
    _assert_grads_close(ref, got)
    shifted = torch.roll(labels, -1, dims=1)    # a caller's own shift (context parallelism passes one): used as given
    shifted[:, -1] = -100
    shifted[0, 10] = shifted[1, 2] = -100       # two positions the plain shift would supervise
    ref_out, ref = _step(_model("olmoe"), ids, labels, att, shift_labels=shifted)
    out, got = _step(m, ids, labels, att, shift_labels=shifted)
    _assert_loss_close(ref_out.loss, out.loss)
    _assert_grads_close(ref, got)


def test_the_comparison_detects_a_dropped_shift():
    """The tolerance is tight enough to see the most likely bug: the loss without the shift by one is a different number."""
    ids, labels, _ = _batch("cpu")
    m = _model("qwen3_moe")
    with torch.no_grad():
        out = m(input_ids=ids, labels=labels)
        h = m.model(input_ids=ids).last_hidden_state
        right = C.chunked_causal_lm_loss(h, labels, m.lm_head, chunk=7)
        unshifted = C.chunked_causal_lm_loss(h, labels, m.lm_head, chunk=7, shift_labels=labels)
    _assert_loss_close(out.loss, right)
    assert abs(float(out.loss) - float(unshifted)) > 1e3 * LOSS_ULPS * EPS32 * float(out.loss)


@pytest.mark.parametrize("mode", ["bf16", "autocast"])
def test_bf16_and_autocast_stay_within_bf16_rounding(mode):
    """A bf16 model, and an fp32 model under CPU bf16 autocast. The hidden-state gradient goes through the same per-row bf16
    matmuls as stock; a trainable bf16 head's weight gradient is summed over chunks in bf16 (one rounding per chunk), so the
    bound here is a couple of bf16 ulps of each tensor's largest element, not the fp32 bound."""
    ids, labels, _ = _batch("cpu", mb=1, seq=40)
    dtype = torch.bfloat16 if mode == "bf16" else torch.float32

    def run(m):
        m.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=(mode == "autocast")):
            out = m(input_ids=ids, labels=labels)
        out.loss.backward()
        return out, {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
    ref_out, ref = run(_model("qwen3_moe", dtype=dtype))
    m = _model("qwen3_moe", dtype=dtype)
    C.enable_chunked_lm_loss(m, 7)
    out, got = run(m)
    assert out.loss.dtype == ref_out.loss.dtype == torch.float32
    _assert_loss_close(ref_out.loss, out.loss)
    _assert_grads_close(ref, got, rel=2 * torch.finfo(torch.bfloat16).eps)


# --------------------------------------------------------------------------------------------------- off, eval, unwinding --

def test_switch_off_is_the_stock_path_byte_for_byte(monkeypatch):
    monkeypatch.delenv("E4B_CHUNKED_LM_LOSS", raising=False)
    ids, labels, att = _batch("cpu")
    ref_out, ref = _step(_model("qwen3_moe"), ids, labels, att)
    m = _model("qwen3_moe")
    assert C.chunked_lm_loss_requested() is None
    assert "forward" not in vars(m) and "forward" not in vars(m.lm_head)
    out, got = _step(m, ids, labels, att)
    assert torch.equal(ref_out.loss, out.loss) and torch.equal(ref_out.logits, out.logits)
    for n in ref:
        assert torch.equal(ref[n], got[n]), n
    assert C.enable_chunked_lm_loss(m, 7) == 1
    assert C.disable_chunked_lm_loss(m) == 1
    assert "forward" not in vars(m) and "forward" not in vars(m.lm_head) and not hasattr(m, "_e4b_chunked_lm_loss")
    out, got = _step(m, ids, labels, att)
    assert torch.equal(ref_out.loss, out.loss) and torch.equal(ref_out.logits, out.logits)
    for n in ref:
        assert torch.equal(ref[n], got[n]), n


def test_no_grad_eval_and_generation_run_stock_when_patched():
    ids, labels, _ = _batch("cpu", mb=1)
    ref = _model("qwen3_moe")
    m = _model("qwen3_moe")
    C.enable_chunked_lm_loss(m, 7)
    calls = dict(C.CHUNKED_LM_LOSS_STATS)
    with torch.no_grad():
        a, b = ref(input_ids=ids, labels=labels), m(input_ids=ids, labels=labels)
    assert torch.equal(a.loss, b.loss) and torch.equal(a.logits, b.logits)
    assert C.CHUNKED_LM_LOSS_STATS["chunked_calls"] == calls["chunked_calls"]
    a = ref.generate(ids[:, :6], max_new_tokens=4, do_sample=False)
    b = m.generate(ids[:, :6], max_new_tokens=4, do_sample=False)
    assert torch.equal(a, b)
    out = m(input_ids=ids)                     # no labels, gradients on: stock, logits present
    assert out.logits is not None and out.loss is None
    out = m(input_ids=ids, labels=labels, return_dict=False)
    assert isinstance(out, tuple) and torch.is_tensor(out[1])


def test_positional_labels_are_found():
    ids, labels, att = _batch("cpu")
    ref_out, ref = _step(_model("mixtral"), ids, labels, att)
    m = _model("mixtral")
    C.enable_chunked_lm_loss(m, 9)
    m.zero_grad(set_to_none=True)
    # forward(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, ...)
    out = m(ids, att, None, None, None, labels)
    out.loss.backward()
    assert out.logits is None
    _assert_loss_close(ref_out.loss, out.loss)
    _assert_grads_close(ref, {n: p.grad for n, p in m.named_parameters() if p.grad is not None})


def test_a_wrapper_reaches_the_causal_lm_inside():
    """A PEFT-style wrapper (whose class name may itself end in ``ForCausalLM``) is searched for the supported causal LM."""
    class WrapForCausalLM(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.base_model = m

        def forward(self, **kw):
            return self.base_model(**kw)
    ids, labels, att = _batch("cpu")
    ref_out, ref = _step(_model("olmoe"), ids, labels, att)
    w = WrapForCausalLM(_model("olmoe"))
    assert C.enable_chunked_lm_loss(w, 5) == 1 and hasattr(w.base_model, "_e4b_chunked_lm_loss")
    out, got = _step(w, ids, labels, att)
    assert out.logits is None
    _assert_loss_close(ref_out.loss, out.loss)
    _assert_grads_close(ref, {n.removeprefix("base_model."): g for n, g in got.items()})
    assert C.disable_chunked_lm_loss(w) == 1


# ---------------------------------------------------------------------------------------------------------------- refusals --

def test_refuses_a_class_outside_the_table():
    cfg = tr.Qwen3Config(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16)
    m = tr.Qwen3ForCausalLM(cfg)
    with pytest.warns(RuntimeWarning, match="not in the chunked-loss table"):
        assert C.enable_chunked_lm_loss(m) == 0
    assert "forward" not in vars(m) and "forward" not in vars(m.lm_head)
    assert "Qwen3ForCausalLM" in C.CHUNKED_LM_LOSS_STATS["refused"]


def test_refuses_a_replaced_loss_function_and_a_hooked_head():
    m = _model("qwen3_moe")
    m.loss_function = lambda *a, **k: None
    assert "not ForCausalLMLoss" in C.chunked_lm_loss_refusal(m)
    m = _model("qwen3_moe")
    h = m.lm_head.register_forward_hook(lambda mod, i, o: o)
    assert "hooks" in C.chunked_lm_loss_refusal(m)
    with pytest.warns(RuntimeWarning):
        assert C.enable_chunked_lm_loss(m) == 0
    h.remove()
    assert C.chunked_lm_loss_refusal(m) is None


def test_run_time_probe_catches_an_unverified_change_after_the_head():
    """A forward hook added AFTER enabling changes what ``lm_head`` returns: the probe comes back changed, the chunked loss
    turns itself off for the model, and the call re-runs stock -- the result is the stock loss under the same hook."""
    ids, labels, att = _batch("cpu")
    ref = _model("qwen3_moe")
    ref.lm_head.register_forward_hook(lambda mod, i, o: o * 0.5)
    ref_out, ref_g = _step(ref, ids, labels, att)
    m = _model("qwen3_moe")
    C.enable_chunked_lm_loss(m, 7)
    m.lm_head.register_forward_hook(lambda mod, i, o: o * 0.5)
    before = C.CHUNKED_LM_LOSS_STATS["runtime_refusals"]
    with pytest.warns(RuntimeWarning, match="disabled on Qwen3MoeForCausalLM"):
        out, got = _step(m, ids, labels, att)
    assert C.CHUNKED_LM_LOSS_STATS["runtime_refusals"] == before + 1
    assert out.logits is not None
    assert torch.equal(ref_out.loss, out.loss)
    for n in ref_g:
        assert torch.equal(ref_g[n], got[n]), n
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _step(m, ids, labels, att)             # refused once, stock from then on, silently


# ------------------------------------------------------------------------------------------------------- enable_fast_train --

def test_enable_fast_train_applies_it_from_the_environment_and_disable_unwinds(monkeypatch):
    pytest.importorskip("nf4_qlora", reason="enable_fast_train returns 0 before its loop without the kernel")
    pytest.importorskip("bitsandbytes")
    from quant_guard import require_quantize
    require_quantize("cpu")
    from experts4bit_qlora import Experts4bit, ExpertsLoRA, disable_fast_train, enable_fast_train

    def model_with_a_patchable_stack():
        m = _model("qwen3_moe")
        g = torch.Generator().manual_seed(3)
        base = Experts4bit.from_float(torch.randn(2, 128, 64, generator=g) * 0.1, torch.randn(2, 64, 64, generator=g) * 0.1,
                                      has_gate=True, quant_type="nf4", compute_dtype=torch.float32)
        m.model.e4b_probe_stack = ExpertsLoRA(base, r=4, alpha=8)    # enable_fast_train's count must be > 0 to apply switches
        return m
    for k in ("E4B_FUSED_ROPE", "E4B_FUSED_RMSNORM", "E4B_MOE_KEEP_LAYERS"):
        monkeypatch.setenv(k, "0")
    monkeypatch.delenv("E4B_CHUNKED_LM_LOSS", raising=False)
    m = model_with_a_patchable_stack()
    assert enable_fast_train(m) >= 1
    assert not hasattr(m, "_e4b_chunked_lm_loss") and "forward" not in vars(m)
    disable_fast_train(m)
    monkeypatch.setenv("E4B_CHUNKED_LM_LOSS", "7")
    m = model_with_a_patchable_stack()
    assert enable_fast_train(m) >= 1
    assert m._e4b_chunked_lm_loss.chunk == 7 and m._e4b_chunked_lm_loss.min_bytes is None and "forward" in vars(m)
    ids, labels, att = _batch("cpu")
    out = m(input_ids=ids, labels=labels, attention_mask=att)
    assert out.logits is None
    assert disable_fast_train(m) >= 1
    monkeypatch.setenv("E4B_CHUNKED_LM_LOSS", "auto")      # the gate rides along; this tiny batch's logits are far under it
    m = model_with_a_patchable_stack()
    assert enable_fast_train(m) >= 1
    assert m._e4b_chunked_lm_loss.min_bytes == C.AUTO_MIN_LOGITS_BYTES
    assert m(input_ids=ids, labels=labels, attention_mask=att).logits is not None
    assert disable_fast_train(m) >= 1
    assert not hasattr(m, "_e4b_chunked_lm_loss") and "forward" not in vars(m) and "forward" not in vars(m.lm_head)
