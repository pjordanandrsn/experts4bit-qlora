"""Structural tests: the generalized streaming loader handles each supported fused-MoE architecture.

For each architecture, build a tiny model, write it as a checkpoint in the on-disk expert layout that
architecture's real checkpoints use (per-expert for OLMoE/Qwen3, fused for Gemma-4), run
``load_moe_4bit_streaming`` end-to-end, and assert experts were quantized to ``Experts4bit`` +
``ExpertsLoRA``, attention LoRA attached, no meta tensors remain, and a forward pass runs.

Nothing in the loader is CUDA-specific: on a host whose bitsandbytes can 4-bit quantize on CPU these
tests run there too (so CPU-only CI actually exercises the loader); with a GPU they run on it, and if
bnb has no working 4-bit backend on the test device they skip cleanly.
"""

import json
import os

import pytest
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# bf16 on the GPU (faithful to real checkpoints); fp32 on CPU. The training test recomputes each
# layer under gradient checkpointing, and MoE top-k routing is tie-fragile: in bf16 (~2^-8 relative
# resolution) a near-tied router logit can round differently between the forward and the recompute
# (CPU kernel heuristics vary within a process), changing the routed token sets' *shapes* and
# crashing with a CheckpointError. fp32 shrinks the tie window by ~2^15, making CPU CI deterministic.
# The same hazard exists for any checkpointed data-dependent-routing MoE (stock transformers
# included) — this is a property of recomputing routing, not of this package's quantization.
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
# bnb signals a missing/broken 4-bit backend in several ways depending on the build; catch them all
# so a host without a working bnb 4-bit path SKIPS cleanly (matches the other test modules).
_QUANTIZE_UNAVAILABLE = (RuntimeError, NotImplementedError, AssertionError, ImportError, OSError)


def _olmoe():
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

    return OlmoeForCausalLM(
        OlmoeConfig(
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_experts=8,
            num_experts_per_tok=2,
            vocab_size=128,
        )
    )


def _qwen3_moe():
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    return Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=8,
            num_experts_per_tok=2,
            vocab_size=128,
            decoder_sparse_step=1,
            mlp_only_layers=[],
            head_dim=16,
        )
    )


def _gemma4():
    pytest.importorskip("transformers.models.gemma4", reason="this transformers has no gemma4")
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    return Gemma4ForCausalLM(
        Gemma4TextConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
            head_dim=16,
            num_experts=8,
            top_k_experts=2,
            moe_intermediate_size=64,
            enable_moe_block=True,
        )
    )


def _write_ckpt(model, d, per_expert):
    """Save a checkpoint. per_expert=True splits fused experts back to per-expert Linears (OLMoE/Qwen3
    on-disk layout); per_expert=False keeps them fused (Gemma-4 on-disk layout)."""
    from safetensors.torch import save_file

    new = {}
    for k, v in model.state_dict().items():
        if per_expert and k.endswith("experts.gate_up_proj"):  # [n_exp, 2*inter, hidden] -> per-expert
            base = k[: -len("gate_up_proj")]
            for e in range(v.shape[0]):
                g, u = v[e].chunk(2, dim=0)
                new[f"{base}{e}.gate_proj.weight"] = g.contiguous()
                new[f"{base}{e}.up_proj.weight"] = u.contiguous()
        elif per_expert and k.endswith("experts.down_proj"):  # [n_exp, hidden, inter] -> per-expert
            base = k[: -len("down_proj")]
            for e in range(v.shape[0]):
                new[f"{base}{e}.down_proj.weight"] = v[e].contiguous()
        else:
            new[k] = v  # keep fused (Gemma-4) or non-expert tensors as-is
    # DTYPE: bf16 on GPU, fp32 on CPU (see top). .clone() breaks shared storage (e.g. Gemma-4 ties
    # lm_head to embed_tokens) — safetensors refuses tensors that share memory. Under bf16 the .to()
    # already copied; under fp32 it's a no-op, so the clone is load-bearing (matches the
    # test_reference_parity.py writer).
    new = {k: v.to(DTYPE).contiguous().clone() for k, v in new.items()}
    save_file(new, os.path.join(d, "model.safetensors"))
    json.dump(
        {"weight_map": {k: "model.safetensors" for k in new}},
        open(os.path.join(d, "model.safetensors.index.json"), "w"),
    )
    model.config.save_pretrained(d)


@pytest.mark.parametrize(
    "build,per_expert",
    [(_olmoe, True), (_qwen3_moe, True), (_gemma4, False)],
    ids=["olmoe", "qwen3_moe", "gemma4"],
)
def test_loader_handles_architecture(build, per_expert, tmp_path):
    from experts4bit_qlora import ExpertsLoRA
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    from experts4bit_qlora.lora import add_attention_lora

    torch.manual_seed(0)
    _write_ckpt(build(), str(tmp_path), per_expert=per_expert)
    try:
        model, cfg = load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8)
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bitsandbytes 4-bit quantize unavailable on {DEVICE}: {e}")
    n_attn = add_attention_lora(model, 4, 8, DTYPE)

    n_expert_mods = sum(isinstance(m, ExpertsLoRA) for m in model.modules())
    assert 1 <= n_expert_mods <= cfg.num_hidden_layers  # experts replaced on the MoE layers
    assert n_attn == cfg.num_hidden_layers * 4  # q/k/v/o per layer
    assert not [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]

    model.config.use_cache = False
    out = model(input_ids=torch.randint(0, cfg.vocab_size, (1, 8), device=DEVICE))
    assert tuple(out.logits.shape) == (1, 8, cfg.vocab_size)


@pytest.mark.parametrize("quant_type", ["int8", "bf16"])
def test_loader_quant_type_threads_through(quant_type, tmp_path):
    """The loader's ``quant_type`` knob reaches the fused-expert quantizer: an OLMoE checkpoint
    streamed with a non-nf4 scheme builds ExpertsNbit bases of that scheme and runs a forward.
    (int8 = 8-bit blockwise; bf16 = 16-bit passthrough — spans both non-4-bit storage families.)"""
    from experts4bit_qlora import ExpertsLoRA
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    torch.manual_seed(0)
    _write_ckpt(_olmoe(), str(tmp_path), per_expert=True)
    try:
        model, cfg = load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8, quant_type=quant_type)
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bitsandbytes {quant_type} quantize unavailable on {DEVICE}: {e}")

    experts = [m.base for m in model.modules() if isinstance(m, ExpertsLoRA)]
    assert experts and all(b.quant_type == quant_type for b in experts)
    model.config.use_cache = False
    out = model(input_ids=torch.randint(0, cfg.vocab_size, (1, 8), device=DEVICE))
    assert tuple(out.logits.shape) == (1, 8, cfg.vocab_size)


def test_unsupported_model_type_errors(tmp_path):
    """A non-fused-MoE architecture (e.g. a dense model) fails fast with a clear message. Resolved
    from a local config dir so the test never touches the Hub (it used to fetch "gpt2" and errored
    on any host without Hub reachability instead of exercising the fail-fast path)."""
    from experts4bit_qlora.loader import SUPPORTED_MODEL_TYPES, load_moe_4bit_streaming

    assert {"olmoe", "qwen3_moe", "gemma4"} <= SUPPORTED_MODEL_TYPES
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt2"}))
    with pytest.raises(NotImplementedError, match="Unsupported model_type"):
        load_moe_4bit_streaming(str(tmp_path), "cpu", torch.bfloat16, r=4, alpha=8)


def test_loader_handles_multimodal_gemma4_checkpoint(tmp_path):
    """The multimodal `gemma4` top-level-config path — previously untested end to end: the text
    tower sits under `text_config`, its tensors under `model.language_model.*` (vision tensors
    alongside, `lm_head` absent/tied). The loader must build the text CausalLM from the sub-config,
    strip the prefix, drop the vision weights, tie lm_head — and compute the same function as the
    text tower it came from (ExpertsLoRA is zero-delta at init, so logits isolate NF4 error)."""
    pytest.importorskip("transformers.models.gemma4", reason="this transformers has no gemma4")
    from safetensors.torch import save_file
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config

    from experts4bit_qlora import ExpertsLoRA
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    torch.manual_seed(0)
    ref = _gemma4().to(DEVICE, dtype=torch.bfloat16).eval()
    ref.config.use_cache = False

    # Write the text tower the way a multimodal Gemma-4 checkpoint stores it.
    sd = {}
    for k, v in ref.state_dict().items():
        if k == "lm_head.weight":
            continue  # tied to embed_tokens on disk
        sd["model.language_model." + k[len("model.") :]] = v
    sd["model.vision_tower.patch_embedding.weight"] = torch.randn(8, 8)  # must be ignored
    sd = {k: v.to(torch.bfloat16).contiguous().clone() for k, v in sd.items()}
    save_file(sd, os.path.join(tmp_path, "model.safetensors"))
    json.dump(
        {"weight_map": {k: "model.safetensors" for k in sd}},
        open(os.path.join(tmp_path, "model.safetensors.index.json"), "w"),
    )
    Gemma4Config(text_config=ref.config.to_dict()).save_pretrained(tmp_path)

    try:
        model, cfg = load_moe_4bit_streaming(str(tmp_path), DEVICE, torch.bfloat16, r=4, alpha=8)
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bitsandbytes 4-bit quantize unavailable on {DEVICE}: {e}")

    assert cfg.model_type == "gemma4"  # the loader was really on the multimodal branch
    assert sum(isinstance(m, ExpertsLoRA) for m in model.modules()) >= 1
    assert not model.lm_head.weight.is_meta  # tied, not left on meta
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    assert not [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]

    model.eval()
    model.config.use_cache = False
    ids = torch.randint(0, ref.config.vocab_size, (1, 8), device=DEVICE)
    with torch.no_grad():
        got = model(input_ids=ids).logits
        want = ref(input_ids=ids).logits
    cos = torch.nn.functional.cosine_similarity(got.flatten(0, 1).float(), want.flatten(0, 1).float(), dim=-1)
    assert cos.mean() > 0.9  # same function as the text tower, within NF4-on-experts error


@pytest.mark.parametrize(
    "build,per_expert",
    [(_olmoe, True), (_qwen3_moe, True), (_gemma4, False)],
    ids=["olmoe", "qwen3_moe", "gemma4"],
)
def test_loaded_model_trains_with_frozen_experts(build, per_expert, tmp_path):
    """Full code-path test: load 4-bit, add LoRA, run real training steps with gradient checkpointing.

    Asserts the whole training path works for each architecture: the held-out loss decreases (the LoRA
    adapters learn), the frozen 4-bit expert packed weights never receive a gradient and stay
    bit-identical, and nothing goes NaN.
    """
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    from experts4bit_qlora.lora import add_attention_lora

    torch.manual_seed(0)
    _write_ckpt(build(), str(tmp_path), per_expert=per_expert)
    try:
        model, cfg = load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8)
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bitsandbytes 4-bit quantize unavailable on {DEVICE}: {e}")
    add_attention_lora(model, 4, 8, DTYPE)

    trainable = []
    for n, p in model.named_parameters():
        p.requires_grad_("lora" in n)  # only LoRA adapters train
        if p.requires_grad:
            trainable.append(p)
    assert trainable
    # The Experts4bit packed weights are Parameters named `...gate_up_proj` / `...down_proj` (no `.weight`),
    # which distinguishes them from the dense-MLP Linears (`...mlp.down_proj.weight`).
    packed_before = {
        n: p.detach().clone() for n, p in model.named_parameters() if n.endswith(("gate_up_proj", "down_proj"))
    }
    assert packed_before  # experts were quantized to frozen 4-bit
    lora0 = trainable[0].detach().clone()

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    opt = torch.optim.Adam(trainable, lr=3e-3)

    torch.manual_seed(1)
    ids = torch.randint(0, cfg.vocab_size, (2, 16), device=DEVICE)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        for n, p in model.named_parameters():
            if not p.requires_grad:
                assert p.grad is None  # frozen params (incl. the 4-bit experts) never get a gradient
        opt.step()
        losses.append(out.loss.item())

    assert all(x == x for x in losses)  # no NaN
    assert losses[-1] < losses[0]  # the LoRA adapters learned (overfit the fixed batch)
    assert not torch.equal(trainable[0].detach(), lora0)  # a LoRA parameter actually moved
    for n, p in model.named_parameters():
        if n in packed_before:
            assert torch.equal(p.detach(), packed_before[n])  # frozen 4-bit experts unchanged
