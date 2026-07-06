"""Structural tests: the generalized streaming loader handles each supported fused-MoE architecture.

For each architecture, build a tiny model, write it as a checkpoint in the on-disk expert layout that
architecture's real checkpoints use (per-expert for OLMoE/Qwen3, fused for Gemma-4/GraniteMoe), run
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


def _granitemoe(tie_word_embeddings=False):
    from transformers.models.granitemoe.configuration_granitemoe import GraniteMoeConfig
    from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeForCausalLM

    return GraniteMoeForCausalLM(
        GraniteMoeConfig(
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_local_experts=8,
            num_experts_per_tok=2,
            vocab_size=128,
            tie_word_embeddings=tie_word_embeddings,
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
    [(_olmoe, True), (_qwen3_moe, True), (_gemma4, False), (_granitemoe, False)],
    ids=["olmoe", "qwen3_moe", "gemma4", "granitemoe"],
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


@pytest.mark.parametrize(
    "quant_type,expected",
    [("int8", "int8"), ("bf16", "bf16"), ("BFLOAT16", "bf16")],
    ids=["int8", "bf16", "alias-BFLOAT16"],
)
def test_loader_quant_type_threads_through(quant_type, expected, tmp_path):
    """The loader's ``quant_type`` knob reaches the fused-expert quantizer: an OLMoE checkpoint
    streamed with a non-nf4 scheme builds ExpertsNbit bases of that scheme and runs a forward.
    (int8 = 8-bit blockwise; bf16 = 16-bit passthrough — spans both non-4-bit storage families;
    the alias spelling proves normalization happens before the class dispatch, not after.)"""
    from experts4bit_qlora import ExpertsLoRA
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    torch.manual_seed(0)
    _write_ckpt(_olmoe(), str(tmp_path), per_expert=True)
    try:
        model, cfg = load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8, quant_type=quant_type)
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bitsandbytes {quant_type} quantize unavailable on {DEVICE}: {e}")

    experts = [m.base for m in model.modules() if isinstance(m, ExpertsLoRA)]
    assert experts and all(b.quant_type == expected for b in experts)
    model.config.use_cache = False
    out = model(input_ids=torch.randint(0, cfg.vocab_size, (1, 8), device=DEVICE))
    assert tuple(out.logits.shape) == (1, 8, cfg.vocab_size)


def test_loader_rejects_bad_quant_type_before_any_io(tmp_path):
    """A bad quant_type fails BEFORE any config read, download, or shard streaming: the target
    directory is empty, so getting the quant_type ValueError (and not a file-not-found error)
    proves validation runs first."""
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    with pytest.raises(ValueError, match="quant_type must be one of"):
        load_moe_4bit_streaming(str(tmp_path), "cpu", torch.bfloat16, r=4, alpha=8, quant_type="int4")


def test_loader_rejects_checkpoint_with_no_experts(tmp_path):
    """A supported model_type whose checkpoint contains zero expert tensors must fail loudly, not
    return a model that silently skipped quantization (the bnb#1849 failure class this loader
    exists to prevent). No quantize happens before the guard, so this runs on any host."""
    from safetensors.torch import save_file

    from experts4bit_qlora.loader import load_moe_4bit_streaming

    torch.manual_seed(0)
    model = _olmoe()
    new = {k: v.to(DTYPE).contiguous().clone() for k, v in model.state_dict().items() if "experts." not in k}
    save_file(new, os.path.join(tmp_path, "model.safetensors"))
    json.dump(
        {"weight_map": {k: "model.safetensors" for k in new}},
        open(os.path.join(tmp_path, "model.safetensors.index.json"), "w"),
    )
    model.config.save_pretrained(tmp_path)

    with pytest.raises(RuntimeError, match="no fused expert stacks found"):
        load_moe_4bit_streaming(str(tmp_path), "cpu", torch.float32, r=4, alpha=8)


def test_unsupported_model_type_errors(tmp_path):
    """A non-fused-MoE architecture (e.g. a dense model) fails fast with a clear message. Resolved
    from a local config dir so the test never touches the Hub (it used to fetch "gpt2" and errored
    on any host without Hub reachability instead of exercising the fail-fast path)."""
    from experts4bit_qlora.loader import SUPPORTED_MODEL_TYPES, load_moe_4bit_streaming

    assert {"olmoe", "qwen3_moe", "gemma4", "granitemoe"} <= SUPPORTED_MODEL_TYPES
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
        try:
            want = ref(input_ids=ids).logits
        except RuntimeError as e:
            # Oracle limitation, not a library defect: stock transformers routes fused MoE through
            # torch._grouped_mm, hard-gated to cc 9.0 — the REFERENCE dies on sm_120 (Blackwell)
            # while this package's own path runs (see the same guard in test_reference_parity.py).
            if "_grouped_mm" in str(e):
                pytest.skip(f"transformers reference (the oracle) cannot run on this device: {e}")
            raise
    cos = torch.nn.functional.cosine_similarity(got.flatten(0, 1).float(), want.flatten(0, 1).float(), dim=-1)
    assert cos.mean() > 0.9  # same function as the text tower, within NF4-on-experts error


def test_loader_handles_legacy_granitemoe_checkpoint(tmp_path):
    """The real GraniteMoe on-disk layout — legacy tensor spellings AND no index file: Hub Granite
    checkpoints (e.g. ibm-granite/granite-3.0-1b-a400m-instruct) store the fused expert stacks as
    `block_sparse_moe.input_linear.weight` [E, 2*inter, hidden] / `output_linear.weight`
    [E, hidden, inter], the router one module deeper at `router.layer.weight`, drop `lm_head.weight`
    (tied) — and, being small, ship as a single `model.safetensors` with no
    `model.safetensors.index.json`. The loader must synthesize the weight map from the file's own
    header, apply the legacy renames, tie lm_head — and compute the same function as the reference
    it came from (ExpertsLoRA is zero-delta at init, so the logits isolate NF4 error)."""
    from safetensors.torch import save_file

    from experts4bit_qlora import ExpertsLoRA
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    torch.manual_seed(0)
    ref = _granitemoe(tie_word_embeddings=True).to(DEVICE, dtype=torch.bfloat16).eval()
    ref.config.use_cache = False

    # Write the checkpoint the way the Hub GraniteMoe checkpoints store it (verified against the
    # safetensors header of granite-3.0-1b-a400m-instruct).
    sd = {}
    for k, v in ref.state_dict().items():
        if k == "lm_head.weight":
            continue  # tied to embed_tokens on disk
        k = k.replace("block_sparse_moe.experts.gate_up_proj", "block_sparse_moe.input_linear.weight")
        k = k.replace("block_sparse_moe.experts.down_proj", "block_sparse_moe.output_linear.weight")
        k = k.replace("block_sparse_moe.router.weight", "block_sparse_moe.router.layer.weight")
        sd[k] = v.to(torch.bfloat16).contiguous().clone()
    save_file(sd, os.path.join(tmp_path, "model.safetensors"))  # single file — deliberately no index.json
    ref.config.save_pretrained(tmp_path)

    try:
        model, cfg = load_moe_4bit_streaming(str(tmp_path), DEVICE, torch.bfloat16, r=4, alpha=8)
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bitsandbytes 4-bit quantize unavailable on {DEVICE}: {e}")

    assert cfg.model_type == "granitemoe"  # the loader really took the granitemoe + legacy-rename path
    assert sum(isinstance(m, ExpertsLoRA) for m in model.modules()) == cfg.num_hidden_layers
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
    assert cos.mean() > 0.9  # same function as the reference, within NF4-on-experts error


@pytest.mark.parametrize(
    "build,per_expert",
    [(_olmoe, True), (_qwen3_moe, True), (_gemma4, False), (_granitemoe, False)],
    ids=["olmoe", "qwen3_moe", "gemma4", "granitemoe"],
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
