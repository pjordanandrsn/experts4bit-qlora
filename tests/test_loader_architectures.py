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

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

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


def _gemma4(tie_word_embeddings=True):
    pytest.importorskip("transformers.models.gemma4", reason="this transformers has no gemma4")
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    return Gemma4ForCausalLM(
        Gemma4TextConfig(
            tie_word_embeddings=tie_word_embeddings,
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


def _write_multimodal_gemma4(ref, tmp_path, with_head=True):
    """Write `ref` the way a multimodal Gemma-4 checkpoint stores it: the text tower under
    `model.language_model.*`, a vision tensor alongside — and the output head at TOP level,
    outside the text-tower prefix, which is where an untied multimodal checkpoint keeps it."""
    from safetensors.torch import save_file
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config

    sd = {}
    for k, v in ref.state_dict().items():
        if k == "lm_head.weight":
            if with_head:
                sd["lm_head.weight"] = v
            continue
        sd["model.language_model." + k[len("model.") :]] = v
    sd["model.vision_tower.patch_embedding.weight"] = torch.randn(8, 8)  # must be ignored
    sd = {k: v.to(torch.bfloat16).contiguous().clone() for k, v in sd.items()}
    save_file(sd, os.path.join(tmp_path, "model.safetensors"))
    json.dump(
        {"weight_map": {k: "model.safetensors" for k in sd}},
        open(os.path.join(tmp_path, "model.safetensors.index.json"), "w"),
    )
    Gemma4Config(text_config=ref.config.to_dict()).save_pretrained(tmp_path)


def test_multimodal_untied_lm_head_is_loaded_not_tied(tmp_path):
    """An UNTIED multimodal checkpoint must get its real output head — the regression path.

    The text-tower prefix filter drops every key that does not start with
    `model.language_model.`, and `lm_head.weight` sits at top level, so it used to be dropped
    and the tie fallback then assigned embed_tokens as the head unconditionally. Nothing raised:
    the model loaded, generated plausibly-shaped tokens, and computed every logit through the
    wrong matrix. Assert the head is a DISTINCT tensor carrying the checkpoint's own values."""
    pytest.importorskip("transformers.models.gemma4", reason="this transformers has no gemma4")
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    torch.manual_seed(0)
    ref = _gemma4(tie_word_embeddings=False).to(DEVICE, dtype=torch.bfloat16).eval()
    ref.config.use_cache = False
    assert ref.lm_head.weight.data_ptr() != ref.model.embed_tokens.weight.data_ptr()
    _write_multimodal_gemma4(ref, tmp_path, with_head=True)

    try:
        model, _ = load_moe_4bit_streaming(str(tmp_path), DEVICE, torch.bfloat16, r=4, alpha=8)
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bitsandbytes 4-bit quantize unavailable on {DEVICE}: {e}")

    assert not model.lm_head.weight.is_meta
    # The bug, stated as the test would have caught it: the head is not the embedding matrix...
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
    assert model.lm_head.weight.shape == ref.lm_head.weight.shape
    # ...and it is the checkpoint's head, bit-for-bit (non-expert weights are not quantized).
    assert torch.equal(model.lm_head.weight.to(ref.lm_head.weight.dtype).cpu(),
                       ref.lm_head.weight.cpu())


def test_multimodal_untied_lm_head_missing_raises(tmp_path):
    """Untied config + no head anywhere in the checkpoint = refuse, loudly.

    Silently tying here is the failure mode issue #37 describes: no error, plausible outputs,
    and a LoRA that "converges" by steering hidden states into embed_tokens, then collapses on
    an inference stack that maps lm_head correctly. A load-time raise is the cheap end of that."""
    pytest.importorskip("transformers.models.gemma4", reason="this transformers has no gemma4")
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    torch.manual_seed(0)
    ref = _gemma4(tie_word_embeddings=False).to(DEVICE, dtype=torch.bfloat16).eval()
    _write_multimodal_gemma4(ref, tmp_path, with_head=False)

    try:
        with pytest.raises(RuntimeError, match="tie_word_embeddings=False"):
            load_moe_4bit_streaming(str(tmp_path), DEVICE, torch.bfloat16, r=4, alpha=8)
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bitsandbytes 4-bit quantize unavailable on {DEVICE}: {e}")


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


@pytest.mark.parametrize(
    "build,per_expert",
    [(_olmoe, True), (_qwen3_moe, True), (_gemma4, False), (_granitemoe, False)],
    ids=["olmoe", "qwen3_moe", "gemma4", "granitemoe"],
)
def test_loaded_model_trains_with_offload(build, per_expert, tmp_path):
    """The same training path as above, but with ``offload=True`` -- the matrix's actual fixture.

    Every rented-pod training run this project reports uses ``offload=True`` + gradient
    checkpointing, and until now **no test enabled offload**. That intersection is exactly
    where the autograd-pins-staged-experts bug lived: two earlier fixes for it were inert and
    both passed the suite, because the suite never staged an offloaded expert through a
    backward. This closes it per architecture, so a loader change that breaks staging for one
    model family fails here instead of on a billed GPU.

    Asserts the stage pre-hook is not merely registered but **fires** (a patch count is not a
    call count), that gradients flow to the adapters and never to the frozen experts, and that
    the packed bytes are bit-identical afterwards -- read from the CPU homes, since under
    offload the module's registered tensors are 0-element placeholders.
    """
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    from experts4bit_qlora.lora import add_attention_lora
    from experts4bit_qlora.engines.offload import _ExpertOffload

    _ExpertOffload._resident = None  # class-level single-slot; isolate this test
    _ExpertOffload._staged_now = set()
    try:
        torch.manual_seed(0)
        _write_ckpt(build(), str(tmp_path), per_expert=per_expert)
        try:
            model, cfg = load_moe_4bit_streaming(
                str(tmp_path), DEVICE, DTYPE, r=4, alpha=8, offload=True, pin=False
            )
        except _QUANTIZE_UNAVAILABLE as e:
            pytest.skip(f"bitsandbytes 4-bit quantize unavailable on {DEVICE}: {e}")
        add_attention_lora(model, 4, 8, DTYPE)

        # The loader hangs the handle on the module it hooked; the packed bytes now live in
        # handle.home (CPU) and the module holds 0-element placeholders while evicted.
        hooked = [m for m in model.modules() if getattr(m, "_offload", None) is not None]
        assert hooked, "offload=True attached no handles -- the loader offloaded no experts"
        packed_before = {}
        for li, mod in enumerate(hooked):
            h = mod._offload
            for name in h._param_names + h._buffer_names:
                t = h.home[name]
                assert t.numel() > 0, f"layer {li} {name}: home is empty -- nothing to compare"
                assert getattr(h.base, name).numel() == 0, f"layer {li} {name}: not evicted after load"
                packed_before[(li, name)] = t.detach().clone()
        assert packed_before

        # Staging is observed by its EFFECT, not by counting hooks: our pre-hook registers after
        # the loader's, so if staging works the base tensors are materialized by the time it runs.
        live = []
        staged, evicted = [], []
        for mod in hooked:
            h = mod._offload
            name = h._param_names[0]
            mod.register_forward_pre_hook(
                lambda *_a, _h=h, _n=name: live.append(getattr(_h.base, _n).numel())
            )
            # Count stage() itself, so a failure can say whether the pre-hook never ran or ran
            # and something re-evicted afterwards -- those have different fixes.
            _real_stage, _real_evict = h.stage, h.evict

            def _counting_stage(_h=h, _r=_real_stage):
                staged.append(_h)
                return _r()

            def _counting_evict(_h=h, _r=_real_evict):
                evicted.append(_h)
                return _r()

            h.stage, h.evict = _counting_stage, _counting_evict

        for n, p in model.named_parameters():
            p.requires_grad_("lora" in n)  # only LoRA adapters train
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert trainable
        lora0 = trainable[0].detach().clone()

        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.train()
        opt = torch.optim.Adam(trainable, lr=3e-3)

        torch.manual_seed(1)
        ids = torch.randint(0, cfg.vocab_size, (2, 16), device=DEVICE)
        losses = []
        for _ in range(8):
            opt.zero_grad()
            out = model(input_ids=ids, labels=ids)
            n_call, n_stage, n_evict = len(live), len(staged), len(evicted)
            try:
                out.loss.backward()
            except RuntimeError as e:  # re-raise carrying the staging counts the message needs
                raise AssertionError(
                    f"backward failed under offload for {len(hooked)} offloaded layer(s). "
                    f"forward: {n_call} calls / {n_stage} stage / {n_evict} evict; "
                    f"backward so far: {len(live) - n_call} calls / {len(staged) - n_stage} stage "
                    f"/ {len(evicted) - n_evict} evict. A backward stage of 0 means the pre-hook "
                    "never ran; stage>0 with evict>0 means the checkpoint recompute ran PAST the "
                    "expert module, firing the evict POST-hook, so the layer was un-staged before "
                    f"its own backward read it. Original: {e}"
                ) from e
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    assert p.grad is None  # frozen params (incl. the offloaded experts) get no grad
            opt.step()
            losses.append(out.loss.item())

        assert len(live) >= len(hooked), f"expert modules ran {len(live)}x for {len(hooked)} layers"
        assert all(n > 0 for n in live), "experts were NOT staged: still 0-element inside the forward"
        assert all(x == x for x in losses)  # no NaN
        assert losses[-1] < losses[0]  # the adapters learned through the staged experts
        assert not torch.equal(trainable[0].detach(), lora0)
        for (li, name), before in packed_before.items():
            now = hooked[li]._offload.home[name]
            assert torch.equal(now.detach(), before), f"layer {li} {name} changed under offload"
    finally:
        _ExpertOffload._resident = None
        _ExpertOffload._staged_now = set()


@pytest.mark.parametrize(
    "build,per_expert",
    [(_olmoe, True), (_qwen3_moe, True), (_gemma4, False), (_granitemoe, False)],
    ids=["olmoe", "qwen3_moe", "gemma4", "granitemoe"],
)
def test_gradient_checkpointing_recomputes_the_expert_layer(build, per_expert, tmp_path):
    """Gradient checkpointing must RECOMPUTE the expert module, not merely be enabled on it.

    ``gradient_checkpointing_enable()`` setting a flag on every submodule is a *blind* metric for
    the property offloaded training depends on: that each expert layer's forward RUNS AGAIN during
    backward, so the offload pre-hook re-stages its 4-bit weights before the re-dequantization
    reads them. A model can report the flag set on 100 % of its modules and still never route the
    expert layer through ``_gradient_checkpointing_func``.

    So count the recompute instead of reading the flag: hook every ``ExpertsLoRA``, run one
    forward, then one backward, and require the module to be called again during the backward.
    Offload is deliberately OFF here -- this isolates the recompute question from staging, and
    keeps the failure a clean count rather than a crash inside the backward.
    """
    from experts4bit_qlora import ExpertsLoRA
    from experts4bit_qlora.loader import load_moe_4bit_streaming
    from experts4bit_qlora.lora import add_attention_lora

    torch.manual_seed(0)
    _write_ckpt(build(), str(tmp_path), per_expert=per_expert)
    try:
        model, cfg = load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8)
    except _QUANTIZE_UNAVAILABLE as e:
        pytest.skip(f"bitsandbytes 4-bit quantize unavailable on {DEVICE}: {e}")
    add_attention_lora(model, 4, 8, DTYPE)
    for n, p in model.named_parameters():
        p.requires_grad_("lora" in n)

    expert_mods = [m for m in model.modules() if isinstance(m, ExpertsLoRA)]
    assert expert_mods
    calls = []
    for mod in expert_mods:
        mod.register_forward_pre_hook(lambda *_a: calls.append(1))

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()

    # Positive control on the instrument: the flag reads set on the modules that carry it, which
    # is the measurement this test exists to distrust. If it were ever False the count below
    # would be trivially explained, so pin it.
    flagged = [m for m in model.modules() if getattr(m, "gradient_checkpointing", False)]
    assert flagged, "gradient_checkpointing flag set nowhere -- the count below would be vacuous"

    torch.manual_seed(1)
    ids = torch.randint(0, cfg.vocab_size, (2, 16), device=DEVICE)
    out = model(input_ids=ids, labels=ids)
    n_forward = len(calls)
    assert n_forward >= len(expert_mods), f"{n_forward} expert calls for {len(expert_mods)} layers"
    out.loss.backward()
    n_recompute = len(calls) - n_forward

    assert n_recompute > 0, (
        f"gradient checkpointing did NOT recompute any expert layer: {n_forward} calls in the "
        f"forward, {n_recompute} in the backward, across {len(expert_mods)} ExpertsLoRA modules "
        f"with the flag set on {len(flagged)}. Offloaded training is unsupported for this "
        "architecture until the expert layer is inside the checkpointed region -- the offload "
        "pre-hook only re-stages the 4-bit weights when the layer forward runs again."
    )


# ------------------------- arena serving: experts on meta -------------------
def _bake_arena_for(model, arena_path):
    """Relocate a loaded model's own quantized expert stacks into an NF4 arena."""
    import struct

    from nvme_arena import bake_expert_tensors

    from experts4bit_qlora import ExpertsNbit
    from experts4bit_qlora.engines.nvme_experts import NF4_SEGMENTS

    mods = [m for m in model.modules() if isinstance(m, ExpertsNbit)]
    tensors, dt = {}, {torch.uint8: "U8", torch.float32: "F32"}
    for lay, mod in enumerate(mods):
        e = mod.num_experts
        n1, k1 = mod._gate_up_shape
        n2, k2 = mod._down_shape
        stacks = {
            NF4_SEGMENTS["c_gu_p"]: mod.gate_up_proj.view(e, n1, k1 // 2),
            NF4_SEGMENTS["c_gu_a"]: mod.gate_up_absmax.view(e, n1, k1 // 64).float(),
            NF4_SEGMENTS["c_dn_p"]: mod.down_proj.view(e, n2, k2 // 2),
            NF4_SEGMENTS["c_dn_a"]: mod.down_absmax.view(e, n2, k2 // 64).float(),
        }
        for kind, st in stacks.items():
            for i in range(e):
                x = st[i].contiguous().cpu()
                tensors[f"model.layers.{lay}.mlp.experts.{i}.{kind}"] = (
                    tuple(x.shape), dt[x.dtype], x.numpy().tobytes())
    hdr, blobs, off = {}, [], 0
    for name, (shape, dtype, data) in tensors.items():
        hdr[name] = {"dtype": dtype, "shape": list(shape),
                     "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hj = json.dumps(hdr).encode()
    snap = os.path.join(os.path.dirname(arena_path), "arena_snap")
    os.makedirs(snap, exist_ok=True)
    with open(os.path.join(snap, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hj)) + hj + b"".join(blobs))
    bake_expert_tensors(
        snap, arena_path,
        name_template="model.layers.{layer}.mlp.experts.{expert}.{kind}",
        kinds=tuple(NF4_SEGMENTS.values()), align=4096, log=lambda *a: None)
    return len(mods)


def test_arena_train_mode_is_reachable_end_to_end(tmp_path):
    """The path a CALLER takes, which is what nothing else here covered.

    `enable_nvme_train_residency` requires ExpertsLoRA-wrapped modules, and the
    arena loader built bare meta experts — so its own documented usage refused
    every module with "not ExpertsLoRA-wrapped" and the feature could not be
    reached at all. 29 CPU tests missed it because every one of them constructs
    `ExpertsLoRA` by hand: they exercised the mechanism, never the route to it.
    Found on a rented A5000, which is an expensive place to learn it.

    `arena_train=True` is what asks for the adapter. The base stays on `meta` —
    that is the whole point — so only the adapter is materialized.
    """
    pytest.importorskip("nvme_arena")
    pytest.importorskip("nvme_residency")
    from experts4bit_qlora import enable_nvme_train_residency
    from experts4bit_qlora.lora import ExpertsLoRA
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    _write_ckpt(_olmoe(), str(tmp_path), per_expert=True)
    ref, _cfg = load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8)
    arena_path = str(tmp_path / "experts.arena")
    n_layers = _bake_arena_for(ref, arena_path)

    model, _cfg = load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8,
                                          arena=arena_path, arena_train=True)
    loras = [m for m in model.modules() if isinstance(m, ExpertsLoRA)]
    assert len(loras) == n_layers, "arena_train must wrap every MoE layer's experts"
    for w in loras:
        assert w.base.gate_up_proj.is_meta, "the base must stay on meta"
        assert not w.gate_up_lora_A.is_meta, "the adapter must be real and trainable"
        assert w.gate_up_lora_A.requires_grad

    # ...and the enabler that could not previously be reached now attaches.
    n = enable_nvme_train_residency(model, arena_path, hot_rows=8, device=DEVICE,
                                    pinned=False)
    assert n == n_layers
    for w in loras:
        assert hasattr(w.base, "_e4b_arena_offload")
    next(iter(loras)).base._e4b_cold_tier.close()


def test_arena_serving_mode_stays_unwrapped(tmp_path):
    """The regression this fix nearly introduced, pinned.

    `r` is a REQUIRED positional, and the documented serving example passes `r=8`
    before calling `enable_mxfp4_nvme_residency`, which refuses wrapped modules.
    Wrapping whenever `r` is truthy would have fixed training by breaking serving.
    The wrap is gated on an explicit `arena_train`; the default must stay bare.
    """
    pytest.importorskip("nvme_arena")
    from experts4bit_qlora.lora import ExpertsLoRA
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    _write_ckpt(_olmoe(), str(tmp_path), per_expert=True)
    ref, _cfg = load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8)
    arena_path = str(tmp_path / "experts.arena")
    _bake_arena_for(ref, arena_path)

    model, _cfg = load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=8, alpha=16,
                                          arena=arena_path)          # no arena_train
    assert not [m for m in model.modules() if isinstance(m, ExpertsLoRA)], \
        "serving loads must stay unwrapped even with a nonzero r"


def test_loader_arena_mode_builds_meta_experts(tmp_path):
    """`arena=` must produce expert modules that hold SHAPES ONLY.

    The non-expert weights still load normally; only the experts are left
    unmaterialized, which is what lets a checkpoint whose experts exceed host RAM
    be opened at all. The model is not runnable until `enable_nvme_residency`
    attaches the arena — asserted here by checking the buffers really are meta.
    """
    pytest.importorskip("nvme_arena")
    from experts4bit_qlora import ExpertsNbit
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    _write_ckpt(_olmoe(), str(tmp_path), per_expert=True)
    ref, _cfg = load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8)
    arena_path = str(tmp_path / "experts.arena")
    n_layers = _bake_arena_for(ref, arena_path)
    assert n_layers > 0

    model, _cfg = load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8,
                                          arena=arena_path)
    mods = [m for m in model.modules() if isinstance(m, ExpertsNbit)]
    assert len(mods) == n_layers, "arena mode must still build one module per MoE layer"
    for m in mods:
        assert m.gate_up_proj.is_meta and m.down_proj.is_meta, "experts materialized"
        assert m.gate_up_absmax.is_meta and m.down_absmax.is_meta
    # non-expert weights are real, not meta — the point is that ONLY experts defer
    assert not model.model.embed_tokens.weight.is_meta
    for name, buf in model.named_buffers():
        if "experts" not in name:
            assert not buf.is_meta, f"non-expert buffer left on meta: {name}"


def test_loader_arena_mode_refuses_per_expert_biases(tmp_path, monkeypatch):
    """gpt-oss keeps per-expert biases beside the packed stacks. Arena serving does
    not carry them yet, so it must REFUSE rather than drop them — dropping would
    silently change the epilogue."""
    pytest.importorskip("nvme_arena")
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    _write_ckpt(_olmoe(), str(tmp_path), per_expert=True)
    ref, _ = load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8)
    arena_path = str(tmp_path / "e.arena")
    _bake_arena_for(ref, arena_path)

    # splice a bias key into the index so the guard sees gpt-oss-shaped input
    ipath = os.path.join(str(tmp_path), "model.safetensors.index.json")
    idx = json.load(open(ipath))
    lay0 = next(k for k in idx["weight_map"] if ".mlp.experts." in k)
    pfx = lay0.split(".mlp.experts.")[0] + ".mlp.experts."
    idx["weight_map"][pfx + "gate_up_proj_bias"] = "model.safetensors"
    json.dump(idx, open(ipath, "w"))

    with pytest.raises(NotImplementedError, match="per-expert biases"):
        load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8,
                                arena=arena_path)


# The tiny fixture has 2 layers, so {0} is the only subset that leaves one to skip;
# {0, 1} would be the whole model and would test nothing (CI caught exactly that).
@pytest.mark.parametrize("keep", [{0}])
def test_quantize_layers_restricts_which_layers_are_quantized(keep, tmp_path):
    """`quantize_layers` leaves the excluded MoE layers in the base dtype.

    Asserted structurally rather than by output error: a numeric check could pass simply
    because a tiny random model is insensitive to quantization, which would make the test
    green while the subset logic did nothing.
    """
    from experts4bit_qlora import ExpertsNbit
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    model = _olmoe()
    n_layers = model.config.num_hidden_layers
    assert n_layers > max(keep) + 1, "fixture must have a layer outside `keep` to skip"
    _write_ckpt(model, tmp_path, per_expert=False)

    m, _ = load_moe_4bit_streaming(str(tmp_path), "cpu", torch.float32, r=4, alpha=8,
                                   quantize_layers=keep)
    quantized = {
        i for i in range(n_layers)
        if any(isinstance(sub, ExpertsNbit)
               for sub in m.model.layers[i].mlp.modules())
    }
    assert quantized == keep, f"expected only {sorted(keep)} quantized, got {sorted(quantized)}"


def test_quantize_layers_default_quantizes_every_layer(tmp_path):
    """Default (None) is unchanged behaviour — the regression this feature could cause."""
    from experts4bit_qlora import ExpertsNbit
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    model = _olmoe()
    n_layers = model.config.num_hidden_layers
    _write_ckpt(model, tmp_path, per_expert=False)

    m, _ = load_moe_4bit_streaming(str(tmp_path), "cpu", torch.float32, r=4, alpha=8)
    quantized = {
        i for i in range(n_layers)
        if any(isinstance(sub, ExpertsNbit)
               for sub in m.model.layers[i].mlp.modules())
    }
    assert quantized == set(range(n_layers))
