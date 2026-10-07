"""E4B_CKPT_OFFLOAD=1: each checkpointed decoder layer keeps its input in pinned host memory (reentrant checkpoint inside
save_on_cpu). Gradients equal Hugging Face's non-reentrant checkpointing; a frozen embedding still lets gradients reach the first
layer's trainable weights; on CUDA the device memory held between forward and backward falls by about one hidden-state tensor per
checkpointed layer."""
import warnings

import pytest

torch = pytest.importorskip("torch")
tr = pytest.importorskip("transformers")

from experts4bit_qlora.engines import ckpt_offload  # noqa: E402


def _tiny(seed=0, device="cpu", layers=3, hidden=64):
    cfg = tr.Qwen3MoeConfig(hidden_size=hidden, intermediate_size=128, moe_intermediate_size=32, num_experts=8, num_experts_per_tok=2,
                            num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=97,
                            max_position_embeddings=64, decoder_sparse_step=1, norm_topk_prob=True)
    torch.manual_seed(seed)
    m = tr.Qwen3MoeForCausalLM(cfg).float().to(device)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.config.use_cache = False
    m.train()
    return m


def _grads(m, device="cpu"):
    ids = torch.randint(0, 97, (2, 24), generator=torch.Generator().manual_seed(3)).to(device)
    m.zero_grad(set_to_none=True)
    m(input_ids=ids, labels=ids).loss.backward()
    return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


def test_gradients_equal_the_default_checkpointing():
    ref = _grads(_tiny())
    m = _tiny()
    assert ckpt_offload.enable_checkpoint_offload(m) == 3
    got = _grads(m)
    assert ref.keys() == got.keys()
    for n in ref:
        torch.testing.assert_close(got[n], ref[n], rtol=0, atol=0, msg=n)


def test_frozen_embedding_still_reaches_the_first_layer():
    """LoRA-style training freezes the embedding, so layer 0's input carries no grad; the reentrant checkpoint needs one, which
    enable_input_require_grads provides. Every trainable weight gets its gradient, equal to the default checkpointing's."""
    def frozen(m):
        for n, p in m.named_parameters():
            p.requires_grad_("self_attn.q_proj" in n)
        return m
    ref = _grads(frozen(_tiny()))
    m = frozen(_tiny())
    assert ckpt_offload.enable_checkpoint_offload(m) == 3
    got = _grads(m)
    assert sorted(ref) == sorted(got) and any(".layers.0." in n for n in got)
    for n in ref:
        torch.testing.assert_close(got[n], ref[n], rtol=0, atol=0, msg=n)


def test_disable_restores_and_enable_is_idempotent():
    m = _tiny()
    before = [lay._gradient_checkpointing_func for lay in m.model.layers]
    assert ckpt_offload.enable_checkpoint_offload(m) == 3
    assert ckpt_offload.enable_checkpoint_offload(m) == 0
    assert all(lay._gradient_checkpointing_func is ckpt_offload.offloaded_checkpoint for lay in m.model.layers)
    assert ckpt_offload.disable_checkpoint_offload(m) == 3
    assert [lay._gradient_checkpointing_func for lay in m.model.layers] == before


def test_no_checkpointing_is_left_alone_with_a_warning():
    m = _tiny()
    m.gradient_checkpointing_disable()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert ckpt_offload.enable_checkpoint_offload(m) == 0
    assert any("no layer has gradient checkpointing enabled" in str(x.message) for x in w)


def test_env_switch(monkeypatch):
    for v, want in (("", False), ("0", False), ("1", True)):
        monkeypatch.setenv("E4B_CKPT_OFFLOAD", v)
        assert ckpt_offload.checkpoint_offload_requested() is want


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device memory is a CUDA statistic")
def test_device_memory_held_across_the_step_falls():
    """Between forward and backward the default keeps every layer's input on the device; with the switch they live in pinned
    host memory, so the device memory held after the forward falls by at least (layers - 1) hidden-state tensors."""
    layers, hidden, seq, batch = 6, 256, 512, 2

    def held(offload):
        m = _tiny(device="cuda", layers=layers, hidden=hidden)
        if offload:
            assert ckpt_offload.enable_checkpoint_offload(m) == layers
        ids = torch.randint(0, 97, (batch, seq), generator=torch.Generator().manual_seed(5)).cuda()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        loss = m(input_ids=ids, labels=ids).loss
        torch.cuda.synchronize()
        h = torch.cuda.memory_allocated() - base
        loss.backward()
        g = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
        del m, loss
        torch.cuda.empty_cache()
        return h, g
    h0, g0 = held(False)
    h1, g1 = held(True)
    tensor = batch * seq * hidden * 4
    print(f"\nheld after forward: {h0 / 1e6:.1f} MB -> {h1 / 1e6:.1f} MB ({(h0 - h1) / tensor:.2f} hidden-state tensors)")
    assert h0 - h1 >= (layers - 1) * tensor, (h0, h1, tensor)
    for n in g0:
        torch.testing.assert_close(g1[n], g0[n], rtol=1e-5, atol=1e-6, msg=n)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused training path is CUDA-only")
def test_fused_experts_under_the_offloaded_checkpoint():
    """e4b's fused ExpertsLoRA forward + backward re-run by the reentrant checkpoint, its input in pinned host memory: every
    gradient equals the plain fused path's to the fused kernels' own run-to-run tolerance."""
    pytest.importorskip("nf4_qlora", reason="needs grouped-nf4-gemm")
    pytest.importorskip("bitsandbytes")
    from quant_guard import require_quantize
    require_quantize("cuda")
    from experts4bit_qlora import Experts4bit, ExpertsLoRA, enable_fast_train
    E, H, I, K, T = 8, 256, 128, 2, 64

    def build():
        torch.manual_seed(0)
        base = Experts4bit.from_float((torch.randn(E, 2 * I, H) * 0.1).cuda(), (torch.randn(E, H, I) * 0.1).cuda(),
                                      quant_type="nf4", compute_dtype=torch.bfloat16)
        mod = ExpertsLoRA(base, r=8, alpha=16, dtype=torch.float32).cuda()
        with torch.no_grad():
            for p in (mod.gate_up_lora_B, mod.down_lora_B):
                p.normal_(0, 0.02)
        assert enable_fast_train(mod, absmax_dq=False) == 1
        return mod.train()
    torch.manual_seed(1)
    hs0 = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
    idx = torch.randint(0, E, (T, K), device="cuda")
    wts = torch.rand(T, K, dtype=torch.bfloat16, device="cuda")
    loss_w = torch.randn(T, H, device="cuda")

    def run(mod, offload):
        hs = hs0.clone().requires_grad_(True)
        out = ckpt_offload.offloaded_checkpoint(lambda x: mod(x, idx, wts), hs) if offload else mod(hs, idx, wts)
        (out.float() * loss_w).sum().backward()
        return hs.grad.clone(), {n: p.grad.clone() for n, p in mod.named_parameters() if p.grad is not None}
    gx0, g0 = run(build(), False)
    gx1, g1 = run(build(), True)
    assert g0.keys() == g1.keys() and g0
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()
    errs = {"dx": rel(gx1, gx0), **{n: rel(g1[n], g0[n]) for n in g0}}
    print("\n" + "  ".join(f"{k}={v:.1e}" for k, v in errs.items()))
    assert all(v < 1e-2 for v in errs.values()), errs


def test_enable_fast_train_applies_it_from_the_environment(monkeypatch):
    pytest.importorskip("nf4_qlora", reason="enable_fast_train returns 0 before its loop without the kernel")
    pytest.importorskip("bitsandbytes")
    from quant_guard import require_quantize
    require_quantize("cpu")
    from experts4bit_qlora import Experts4bit, ExpertsLoRA, disable_fast_train, enable_fast_train
    for k in ("E4B_FUSED_ROPE", "E4B_FUSED_RMSNORM", "E4B_MOE_KEEP_LAYERS", "E4B_CHUNKED_LM_LOSS"):
        monkeypatch.setenv(k, "0")

    def model():
        m = _tiny()
        g = torch.Generator().manual_seed(3)
        base = Experts4bit.from_float(torch.randn(2, 128, 64, generator=g) * 0.1, torch.randn(2, 64, 64, generator=g) * 0.1,
                                      has_gate=True, quant_type="nf4", compute_dtype=torch.float32)
        m.model.e4b_probe_stack = ExpertsLoRA(base, r=4, alpha=8)
        return m
    monkeypatch.delenv("E4B_CKPT_OFFLOAD", raising=False)
    m = model()
    assert enable_fast_train(m) >= 1
    assert not any(getattr(lay, "_e4b_ckpt_offload_ref", None) for lay in m.model.layers)
    monkeypatch.setenv("E4B_CKPT_OFFLOAD", "1")
    m = model()
    assert enable_fast_train(m) >= 1
    assert all(lay._gradient_checkpointing_func is ckpt_offload.offloaded_checkpoint for lay in m.model.layers)
    disable_fast_train(m)
    assert not any(getattr(lay, "_e4b_ckpt_offload_ref", None) for lay in m.model.layers)
