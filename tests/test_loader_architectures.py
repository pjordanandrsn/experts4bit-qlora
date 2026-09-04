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

from quant_guard import load_or_skip

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
# The "is this bnb missing, or is this the LOADER refusing?" distinction lives in
# tests/quant_guard.py, shared with test_reference_parity.py — see that module's docstring for the
# two mutants that proved a broad `except` turns this whole file into an instrument that cannot fail.


def _load_or_skip(path, dtype=DTYPE, *, what="4-bit", **kw):
    """`quant_guard.load_or_skip` bound to this module's DEVICE/DTYPE. Every arm loads through here.

    ``dtype`` stays a positional default because several arms pin bf16 instead of the module default,
    and ``what`` because the quant_type arms name the scheme they actually asked for in the skip
    reason."""
    return load_or_skip(path, DEVICE, dtype, what=what, **kw)


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
    from experts4bit_qlora.lora import add_attention_lora

    torch.manual_seed(0)
    _write_ckpt(build(), str(tmp_path), per_expert=per_expert)
    model, cfg = _load_or_skip(str(tmp_path), r=4, alpha=8)
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

    torch.manual_seed(0)
    _write_ckpt(_olmoe(), str(tmp_path), per_expert=True)
    model, cfg = _load_or_skip(str(tmp_path), r=4, alpha=8, quant_type=quant_type, what=quant_type)

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

    model, cfg = _load_or_skip(str(tmp_path), torch.bfloat16, r=4, alpha=8)

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

    torch.manual_seed(0)
    ref = _gemma4(tie_word_embeddings=False).to(DEVICE, dtype=torch.bfloat16).eval()
    ref.config.use_cache = False
    assert ref.lm_head.weight.data_ptr() != ref.model.embed_tokens.weight.data_ptr()
    _write_multimodal_gemma4(ref, tmp_path, with_head=True)

    model, _ = _load_or_skip(str(tmp_path), torch.bfloat16, r=4, alpha=8)

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

    torch.manual_seed(0)
    ref = _gemma4(tie_word_embeddings=False).to(DEVICE, dtype=torch.bfloat16).eval()
    _write_multimodal_gemma4(ref, tmp_path, with_head=False)

    with pytest.raises(RuntimeError, match="tie_word_embeddings=False"):
        _load_or_skip(str(tmp_path), torch.bfloat16, r=4, alpha=8)


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

    model, cfg = _load_or_skip(str(tmp_path), torch.bfloat16, r=4, alpha=8)

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
    from experts4bit_qlora.lora import add_attention_lora

    torch.manual_seed(0)
    _write_ckpt(build(), str(tmp_path), per_expert=per_expert)
    model, cfg = _load_or_skip(str(tmp_path), r=4, alpha=8)
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
    from experts4bit_qlora.lora import add_attention_lora
    from experts4bit_qlora.engines.offload import _ExpertOffload

    _ExpertOffload._resident = None  # class-level single-slot; isolate this test
    _ExpertOffload._staged_now = set()
    try:
        torch.manual_seed(0)
        _write_ckpt(build(), str(tmp_path), per_expert=per_expert)
        model, cfg = _load_or_skip(str(tmp_path), r=4, alpha=8, offload=True, pin=False)
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
    from experts4bit_qlora.lora import add_attention_lora

    torch.manual_seed(0)
    _write_ckpt(build(), str(tmp_path), per_expert=per_expert)
    model, cfg = _load_or_skip(str(tmp_path), r=4, alpha=8)
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


def test_loader_arena_mode_refuses_unknown_per_expert_biases(tmp_path, monkeypatch):
    """A per-expert bias spelling the arena path does not model must REFUSE
    rather than be dropped -- dropping silently changes the epilogue. Only
    the gpt-oss spelling is carried (see the carry test below)."""
    pytest.importorskip("nvme_arena")
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    _write_ckpt(_olmoe(), str(tmp_path), per_expert=True)
    ref, _ = load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8)
    arena_path = str(tmp_path / "e.arena")
    _bake_arena_for(ref, arena_path)

    ipath = os.path.join(str(tmp_path), "model.safetensors.index.json")
    idx = json.load(open(ipath))
    lay0 = next(k for k in idx["weight_map"] if ".mlp.experts." in k)
    pfx = lay0.split(".mlp.experts.")[0] + ".mlp.experts."
    idx["weight_map"][pfx + "some_other_bias"] = "model.safetensors"
    json.dump(idx, open(ipath, "w"))

    with pytest.raises(NotImplementedError, match="per-expert biases"):
        load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8,
                                arena=arena_path)


def test_loader_arena_mode_carries_gptoss_biases(tmp_path):
    """The gpt-oss spelling IS carried: weights stream from the arena while
    the two small bias stacks stay resident, de-interleaved to match the
    baked gate-block-then-up-block weight layout."""
    pytest.importorskip("nvme_arena")
    import torch
    from safetensors.torch import load_file, save_file

    from experts4bit_qlora.arch.gptoss import GPTOSS_ALPHA, GPTOSS_LIMIT
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    _write_ckpt(_olmoe(), str(tmp_path), per_expert=True)
    ref, _ = load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8)
    arena_path = str(tmp_path / "e.arena")
    _bake_arena_for(ref, arena_path)

    # add REAL gpt-oss-spelled bias tensors (index + shard) for every MoE layer
    fp = os.path.join(str(tmp_path), "model.safetensors")
    tensors = load_file(fp)
    ipath = os.path.join(str(tmp_path), "model.safetensors.index.json")
    idx = json.load(open(ipath))
    prefixes = sorted({k.split(".mlp.experts.")[0] + ".mlp.experts."
                       for k in idx["weight_map"] if ".mlp.experts." in k})
    assert prefixes, "fixture has no expert layers"
    mods = [m for m in ref.modules() if hasattr(m, "gate_up_proj")]
    E = mods[0].num_experts
    n1, _k1 = mods[0]._gate_up_shape
    _n2, hidden = mods[0]._down_shape[0], mods[0]._down_shape[0]
    for pfx in prefixes:
        gub = torch.arange(E * n1, dtype=DTYPE).reshape(E, n1)
        dnb = torch.zeros(E, hidden, dtype=DTYPE)
        tensors[pfx + "gate_up_proj_bias"] = gub
        tensors[pfx + "down_proj_bias"] = dnb
        idx["weight_map"][pfx + "gate_up_proj_bias"] = "model.safetensors"
        idx["weight_map"][pfx + "down_proj_bias"] = "model.safetensors"
    save_file({k: v.contiguous().clone() for k, v in tensors.items()}, fp)
    json.dump(idx, open(ipath, "w"))

    model, _ = load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4,
                                       alpha=8, arena=arena_path)
    carried = [m for m in model.modules() if hasattr(m, "gate_up_bias")]
    assert len(carried) == len(prefixes), "every MoE layer must carry biases"
    for m in carried:
        assert not m.gate_up_bias.is_meta and not m.down_bias.is_meta
        assert m.gate_up_bias.shape == (E, n1)
        assert m.alpha == GPTOSS_ALPHA and m.limit == GPTOSS_LIMIT
        # de-interleaved: first half is the EVEN source rows
        src = torch.arange(E * n1, dtype=DTYPE).reshape(E, n1)
        want = torch.cat([src[:, 0::2], src[:, 1::2]], dim=1)
        assert torch.equal(m.gate_up_bias.cpu().to(DTYPE), want)


# Both on-disk expert layouts, on every `quantize_layers` test. Only `per_expert=False`
# used to be covered, and that was the one layout the feature actually worked on: a
# skipped layer was `continue`d before any read branch, so its per-expert keys never
# reached `expert_keys` and the non-expert `_assign` pass walked them into
# `get_submodule("...experts.0")` — which transformers >= 5 does not build, since the
# experts are one fused module. Every per-expert checkpoint died on
# "OlmoeExperts has no attribute `0`", naming neither the flag nor the layout.
_LAYOUTS = pytest.mark.parametrize("per_expert", [False, True],
                                   ids=["fused_on_disk", "per_expert_on_disk"])


# The tiny fixture has 2 layers, so {0} is the only subset that leaves one to skip;
# {0, 1} would be the whole model and would test nothing (CI caught exactly that).
@_LAYOUTS
@pytest.mark.parametrize("keep", [{0}])
def test_quantize_layers_restricts_which_layers_are_quantized(keep, per_expert, tmp_path):
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
    _write_ckpt(model, tmp_path, per_expert=per_expert)

    m, _ = load_moe_4bit_streaming(str(tmp_path), "cpu", torch.float32, r=4, alpha=8,
                                   quantize_layers=keep)
    quantized = {
        i for i in range(n_layers)
        if any(isinstance(sub, ExpertsNbit)
               for sub in m.model.layers[i].mlp.modules())
    }
    assert quantized == keep, f"expected only {sorted(keep)} quantized, got {sorted(quantized)}"


@_LAYOUTS
def test_quantize_layers_default_quantizes_every_layer(per_expert, tmp_path):
    """Default (None) is unchanged behaviour — the regression this feature could cause."""
    from experts4bit_qlora import ExpertsNbit
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    model = _olmoe()
    n_layers = model.config.num_hidden_layers
    _write_ckpt(model, tmp_path, per_expert=per_expert)

    m, _ = load_moe_4bit_streaming(str(tmp_path), "cpu", torch.float32, r=4, alpha=8)
    quantized = {
        i for i in range(n_layers)
        if any(isinstance(sub, ExpertsNbit)
               for sub in m.model.layers[i].mlp.modules())
    }
    assert quantized == set(range(n_layers))


# ---------------------------------------------------------------------------
# The mixtral convention: block_sparse_moe / w1,w3,w2 on disk -> mlp.experts in
# the tree. The family that proved the loader was conflating the two prefixes.
# ---------------------------------------------------------------------------


def _mixtral():
    pytest.importorskip("transformers.models.mixtral", reason="this transformers has no mixtral")
    from transformers.models.mixtral.configuration_mixtral import MixtralConfig
    from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM

    return MixtralForCausalLM(
        MixtralConfig(
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_local_experts=8,
            num_experts_per_tok=2,
            vocab_size=128,
        )
    )


def _phimoe():
    pytest.importorskip("transformers.models.phimoe", reason="this transformers has no phimoe")
    from transformers.models.phimoe.configuration_phimoe import PhimoeConfig
    from transformers.models.phimoe.modeling_phimoe import PhimoeForCausalLM

    return PhimoeForCausalLM(
        PhimoeConfig(
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_local_experts=8,
            num_experts_per_tok=2,
            vocab_size=128,
        )
    )


#: How the built tree spells the router, per family. transformers renames
#: `.block_sparse_moe.` -> `.mlp.` for both, and phimoe renames `.gate.weight`
#: -> `.router.weight` on top of that — so the two families diverge on exactly
#: one key and the checkpoint spells both of them `block_sparse_moe.gate.weight`.
_ROUTER_IN_TREE = {"mixtral": "mlp.gate.weight", "phimoe": "mlp.router.weight"}


def _write_block_sparse_ckpt(model, d):
    """Save `model` the way a RELEASED mixtral-convention checkpoint is spelled.

    Two divergences from the built tree, and the loader has to bridge both:

    * experts are PER-EXPERT under a different container —
      ``block_sparse_moe.experts.{e}.{w1,w3,w2}.weight`` rather than the tree's
      fused ``mlp.experts.{gate_up,down}_proj``;
    * the router is ``block_sparse_moe.gate.weight`` whatever the tree calls it.

    w1 = gate and w3 = up (pinned in arch/mixtral.py against upstream's converter
    and the expert forward). This writer splits the FUSED stack back the same way,
    so a loader that swapped them would round-trip a different function — which is
    what the logit comparison downstream detects.
    """
    from safetensors.torch import save_file

    router = _ROUTER_IN_TREE[model.config.model_type]
    new = {}
    for k, v in model.state_dict().items():
        if k.endswith("mlp.experts.gate_up_proj"):
            base = k[: -len("mlp.experts.gate_up_proj")] + "block_sparse_moe.experts."
            for e in range(v.shape[0]):
                gate, up = v[e].chunk(2, dim=0)
                new[f"{base}{e}.w1.weight"] = gate.contiguous()
                new[f"{base}{e}.w3.weight"] = up.contiguous()
        elif k.endswith("mlp.experts.down_proj"):
            base = k[: -len("mlp.experts.down_proj")] + "block_sparse_moe.experts."
            for e in range(v.shape[0]):
                new[f"{base}{e}.w2.weight"] = v[e].contiguous()
        elif k.endswith(router):
            new[k[: -len(router)] + "block_sparse_moe.gate.weight"] = v
        else:
            new[k] = v
    new = {k: v.to(DTYPE).contiguous().clone() for k, v in new.items()}
    save_file(new, os.path.join(d, "model.safetensors"))
    json.dump(
        {"weight_map": {k: "model.safetensors" for k in new}},
        open(os.path.join(d, "model.safetensors.index.json"), "w"),
    )
    model.config.save_pretrained(d)
    return new


@pytest.mark.parametrize("build", [_mixtral, _phimoe], ids=["mixtral", "phimoe"])
def test_loader_handles_block_sparse_moe_checkpoint(build, tmp_path):
    """A mixtral-convention checkpoint loads, quantizes, and computes the reference function.

    This is the regression the change exists for. The loader built ONE prefix from the
    convention's `fused_prefix` — the MODULE path, `mlp.experts` — and used it to look
    for checkpoint keys, which for this family live under `block_sparse_moe.experts`.
    Nothing matched, every MoE layer fell through the dense `continue`, and the load
    ended in the zero-expert-stacks guard.

    Orientation is checked at the STACK, not at the logits. w1 and w3 are shape-identical,
    so a loader that read w3 as the gate produces `act(up) * gate` with every shape
    agreeing and nothing raising — and on a tiny random model that barely moves the
    output, because SiLU is near-linear around zero and the logits are dominated by the
    embedding and attention path. A half-swap was measured at cosine 0.99+ here, so an
    end-to-end similarity bar loose enough to tolerate NF4 error cannot also catch it.
    Comparing the dequantized stack against the reference — and against its own swapped
    halves — separates the two by ~15x instead. The forward is still exercised below, for
    what it does prove: that routing runs and the model computes the reference function.
    """
    from experts4bit_qlora import ExpertsLoRA

    torch.manual_seed(0)
    ref = build().to(DEVICE, dtype=DTYPE).eval()
    ref.config.use_cache = False
    _write_block_sparse_ckpt(ref, str(tmp_path))

    model, cfg = _load_or_skip(str(tmp_path), r=4, alpha=8)

    assert sum(isinstance(m, ExpertsLoRA) for m in model.modules()) == cfg.num_hidden_layers
    assert not [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]

    # w1 = gate and it lands in the FIRST half of the fused stack.
    for layer in range(cfg.num_hidden_layers):
        base = model.model.layers[layer].mlp.experts.base
        want = ref.model.layers[layer].mlp.experts.gate_up_proj[0].float().cpu()
        got = base._dequantize_expert(
            base.gate_up_proj, base.gate_up_absmax, base._gate_up_shape, 0, torch.float32
        ).detach().float().cpu()
        swapped = torch.cat(want.chunk(2, dim=0)[::-1], dim=0)
        err, err_swapped = (got - want).abs().mean(), (got - swapped).abs().mean()
        assert err < 0.2 * err_swapped, f"layer {layer}: gate/up look swapped ({err} vs {err_swapped})"

    model.eval()
    model.config.use_cache = False
    ids = torch.randint(0, cfg.vocab_size, (1, 8), device=DEVICE)
    with torch.no_grad():
        got = model(input_ids=ids).logits
        try:
            want = ref(input_ids=ids).logits
        except RuntimeError as e:
            # Same oracle limitation the gemma4 arm documents: stock transformers routes
            # fused MoE through torch._grouped_mm, hard-gated to cc 9.0.
            if "_grouped_mm" in str(e):
                pytest.skip(f"transformers reference (the oracle) cannot run on this device: {e}")
            raise
    cos = torch.nn.functional.cosine_similarity(got.flatten(0, 1).float(), want.flatten(0, 1).float(), dim=-1)
    assert cos.mean() > 0.9


@pytest.mark.parametrize("build", [_mixtral, _phimoe], ids=["mixtral", "phimoe"])
def test_block_sparse_router_reaches_the_tree_spelling(build, tmp_path):
    """The NON-expert half of the same divergence: the router.

    Both families store it as `block_sparse_moe.gate.weight`; the tree wants
    `mlp.gate.weight` (mixtral) or `mlp.router.weight` (phimoe). Those are SUBSTRING
    renames, which the loader's suffix-anchored renamer cannot express — so without
    the convention's own rename table the key lands nowhere and `_assign` dies on a
    submodule that does not exist. Assert the checkpoint's values arrive, bit-for-bit:
    routers are not quantized, so an exact comparison is available and a near-miss
    would mean the wrong tensor got placed.
    """
    torch.manual_seed(0)
    ref = build().to(DEVICE, dtype=DTYPE).eval()
    ref.config.use_cache = False
    on_disk = _write_block_sparse_ckpt(ref, str(tmp_path))
    router = _ROUTER_IN_TREE[ref.config.model_type]
    assert not any(k.endswith(router) for k in on_disk), "fixture must not pre-apply the rename"

    model, cfg = _load_or_skip(str(tmp_path), r=4, alpha=8)

    for i in range(cfg.num_hidden_layers):
        got = model.get_parameter(f"model.layers.{i}.{router}")
        want = on_disk[f"model.layers.{i}.block_sparse_moe.gate.weight"]
        assert torch.equal(got.cpu(), want.cpu())


def test_checkpoint_prefix_is_not_the_module_prefix():
    """State the decoupling directly, so a future refactor that re-merges the two fails here.

    `expert_layout_for` answers the MODULE question (where the fused stack is placed);
    `_index_per_expert_keys` answers the CHECKPOINT question (which keys feed it). For the
    mixtral family the two strings differ, which is the whole point — a test that only
    loaded the model would still pass if someone re-derived one from the other for a
    family where they happen to coincide.
    """
    from experts4bit_qlora.loader import _convention_or_none, _index_per_expert_keys, expert_layout_for

    module_rel, has_gate = expert_layout_for("mixtral")
    assert (module_rel, has_gate) == ("mlp.experts", True)

    conv = _convention_or_none("mixtral")
    keys = [f"model.layers.0.block_sparse_moe.experts.{e}.{p}.weight" for e in range(2) for p in ("w1", "w3", "w2")]
    keys += ["model.layers.0.block_sparse_moe.gate.weight", "model.embed_tokens.weight"]
    index = _index_per_expert_keys(conv, keys)

    assert set(index) == {0}
    assert index[0]["gate"][1] == "model.layers.0.block_sparse_moe.experts.1.w1.weight"
    assert index[0]["up"][1] == "model.layers.0.block_sparse_moe.experts.1.w3.weight"
    assert index[0]["down"][1] == "model.layers.0.block_sparse_moe.experts.1.w2.weight"
    # The router and the embeddings are NOT expert keys — they must reach the passthrough pass.
    indexed = {k for byrole in index[0].values() for k in byrole.values()}
    assert "model.layers.0.block_sparse_moe.gate.weight" not in indexed
    assert "model.embed_tokens.weight" not in indexed
    # ...and no expert key starts with the module prefix, which is what used to be searched for.
    assert not any(k.startswith(f"model.layers.0.{module_rel}.") for k in indexed)


def test_pre_fused_families_are_never_indexed_as_per_expert():
    """Conventions with an empty `roles` (gpt-oss, granitemoe, dbrx, qwen3_vl_moe...) ship
    already-fused stacks. The index must stay empty for them, so their branches keep
    addressing the checkpoint by the module prefix exactly as before — and a model_type with
    no convention at all (the dedicated-quant specials) must not raise on the lookup."""
    from experts4bit_qlora.loader import _convention_or_none, _index_per_expert_keys

    # gemma4 joined this group on 2026-09-04: adjudicated pre-fused, empty roles,
    # so the loader's dedicated path is untouched and the int4 lane can plan it
    for model_type in ("gpt_oss", "granitemoe", "dbrx", "qwen3_vl_moe", "gemma4", "gemma4_text"):
        conv = _convention_or_none(model_type)
        assert conv is not None and not conv.roles
        assert _index_per_expert_keys(conv, ["model.layers.0.mlp.experts.gate_up_proj"]) == {}
        assert _index_per_expert_keys(conv, ["model.language_model.layers.0.experts.gate_up_proj"]) == {}
    for model_type in ("kimi_k3", "deepseek_v4"):
        assert _convention_or_none(model_type) is None
        assert _index_per_expert_keys(None, ["model.layers.0.mlp.experts.0.gate_proj.weight"]) == {}


def test_admission_gate_covers_the_family_and_its_aliases():
    """Which STORAGE layouts the streaming read handles is one convention-membership
    test, not a per-model list — so the minimax aliases come along with mixtral. The
    negative arm matters as much: admitting a family whose read this loop has NOT been
    shown to handle (flat stacks, non-gated experts, hybrid Mamba towers) is the mistake
    the narrow gate prevents."""
    from experts4bit_qlora.loader import _read_compatible_convention

    for model_type in ("mixtral", "phimoe", "minimax", "minimax_m2", "minimax_m3_vl",
                       "olmoe", "qwen3_moe", "qwen2_moe", "deepseek_v3"):
        assert _read_compatible_convention(model_type), model_type
    for model_type in ("dbrx", "nemotron_h", "jamba", "lfm2_moe", "gpt2", "llama"):
        assert not _read_compatible_convention(model_type), model_type


def test_clamped_swiglu_epilogue_is_refused_despite_a_readable_layout(tmp_path):
    """Sharing a STORAGE convention is not sharing an EPILOGUE, and the loader must
    refuse rather than substitute.

    `minimax_m3_vl` is on the mixtral convention — the read above handles its keys
    perfectly — but its experts run a clamped SwiGLU-OAI (`gate.clamp(max=limit)`,
    `up.clamp(±limit)`, `gate * sigmoid(gate * alpha)`), not the plain `act(gate) * up`
    the generic Experts4bit computes. Every weight would land, every shape would agree,
    nothing would raise, and the model would compute the wrong function — so admitting
    it on convention membership alone was a real defect in the first cut of this change.

    `hidden_act` cannot catch it: these families compute the gate inline from
    `swiglu_alpha`/`swiglu_limit` and leave `hidden_act` reading like an ordinary
    activation, so the loader's unknown-activation warning never fires. The refusal
    therefore keys on the epilogue's own config fields.
    """
    from experts4bit_qlora.loader import _declares_clamped_swiglu, load_moe_4bit_streaming

    ref = _mixtral()
    assert not _declares_clamped_swiglu(ref.config)          # plain SwiGLU: untouched
    _write_block_sparse_ckpt(ref, str(tmp_path))

    # Same readable layout, plus the epilogue fields. Written onto the saved config so
    # the refusal is exercised through the real entry point, not by calling the predicate.
    cfg = json.load(open(os.path.join(tmp_path, "config.json")))
    cfg["swiglu_alpha"], cfg["swiglu_limit"] = 1.702, 7.0
    json.dump(cfg, open(os.path.join(tmp_path, "config.json"), "w"))

    with pytest.raises(NotImplementedError, match="CLAMPED SwiGLU"):
        load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8)


def test_families_with_their_own_clamped_epilogue_are_not_refused():
    """The refusal must not catch gpt_oss or deepseek_v4. They clamp too, but each is
    named in SUPPORTED_ARCHITECTURES and carries an Experts4bit subclass that reproduces
    its epilogue — so the gate is scoped to the convention-admitted path, and this is the
    arm that says so. A refusal broad enough to hit them would break shipped families."""
    from experts4bit_qlora.loader import SUPPORTED_ARCHITECTURES, _declares_clamped_swiglu

    class _Cfg:
        swiglu_limit = 7.0

    assert _declares_clamped_swiglu(_Cfg())                  # they would trip the predicate
    for model_type in ("gpt_oss", "deepseek_v4"):            # ...but the gate never asks
        assert model_type in SUPPORTED_ARCHITECTURES, model_type


@pytest.mark.parametrize("build", [_olmoe, _qwen3_moe], ids=["olmoe", "qwen3_moe"])
def test_existing_per_expert_families_load_unchanged(build, tmp_path, monkeypatch):
    """The families that already worked must be BIT-IDENTICAL after the rewrite.

    Their per-expert read used to be a literal `epfx + "{e}.gate_proj.weight"` probe; it is
    now driven by the convention index. Both paths still exist — the literal one is the
    fallback for model_types that have no convention — so the two can be run against the same
    checkpoint and compared directly. Forcing `_convention_or_none` to None is exactly what a
    conventionless family sees, which makes this an equivalence proof rather than a smoke test:
    every packed 4-bit byte, every absmax scale and every non-expert weight must match.
    """
    from experts4bit_qlora import loader as loader_mod
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    torch.manual_seed(0)          # deterministic fixture weights
    _write_ckpt(build(), str(tmp_path), per_expert=True)

    def _tensors(model):
        return {n: t for n, t in list(model.named_parameters()) + list(model.named_buffers())}

    # Reseed before EACH load: ExpertsLoRA draws its A matrices at construction, so two
    # loads in one process differ there for a reason that has nothing to do with the read
    # path. Seeding makes the whole comparison exact instead of forcing an exemption that
    # would also excuse a real difference.
    torch.manual_seed(1234)
    via_convention, _ = _load_or_skip(str(tmp_path), r=4, alpha=8)
    monkeypatch.setattr(loader_mod, "_convention_or_none", lambda model_type: None)
    torch.manual_seed(1234)
    via_literal, _ = load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8)

    got, want = _tensors(via_convention), _tensors(via_literal)
    assert set(got) == set(want)
    packed = [n for n in got if n.endswith(".experts.base.gate_up_proj")]
    # Guard the guard: without a real packed 4-bit stack in the comparison, the loop below
    # would be comparing attention and norms only and would pass no matter what the read did.
    assert packed, "no packed 4-bit expert stack in the comparison"
    assert all(got[n].dtype == torch.uint8 for n in packed), packed
    for n in sorted(got):
        assert got[n].dtype == want[n].dtype, n
        assert torch.equal(got[n].cpu(), want[n].cpu()), n


@_LAYOUTS
def test_quantize_layers_excluded_layer_holds_the_source_weights(per_expert, tmp_path):
    """The excluded layer is not merely unquantized — it holds the RIGHT tensors.

    Loading without raising is too weak a bar for the per-expert layout: fusing
    gate_up[e] = cat([gate, up]) in the wrong order, or stacking on the wrong axis,
    produces plausible shapes and a silently wrong model on exactly the layers the caller
    chose to protect. So compare against an oracle outside the loader — the source
    module's own fused parameters — and require equality, not closeness.
    """
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    torch.manual_seed(0)
    model = _olmoe()
    gold = {k: v.clone() for k, v in model.state_dict().items() if ".experts." in k}
    _write_ckpt(model, tmp_path, per_expert=per_expert)

    # Write, load and compare all at DTYPE. Pinning fp32 here while `_write_ckpt` stores
    # DTYPE passed only because DTYPE *is* fp32 on a CPU host: on a GPU one the checkpoint
    # is bf16, and demanding bit-equality with the fp32 original fails on both layouts —
    # the fused arm because `_assign` keeps the checkpoint dtype, the per-expert arm
    # because the values have been through a bf16 round-trip.
    m, _ = load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8,
                                   quantize_layers={0})
    skipped = m.model.layers[1].mlp.experts
    for proj in ("gate_up_proj", "down_proj"):
        got = getattr(skipped, proj)
        want = gold[f"model.layers.1.mlp.experts.{proj}"].to(DTYPE)
        assert not got.is_meta, f"{proj} never reached the excluded layer"
        # The checkpoint's dtype, which is what `_assign` gives every other weight in the
        # model — so both on-disk layouts land the excluded layer in the same state.
        assert got.dtype == DTYPE, f"{proj} is {got.dtype}, not the checkpoint dtype"
        assert torch.equal(got.cpu(), want.cpu()), f"{proj} does not match the source weights"


@pytest.mark.parametrize("build", [_olmoe, _qwen3_moe], ids=["olmoe", "qwen3_moe"])
def test_quantize_layers_excluded_layer_identical_through_both_reads(build, tmp_path, monkeypatch):
    """The excluded layer comes out the same whichever per-expert read located it.

    `_place_unquantized_experts` mirrors the quantizing branch's dispatch, so it inherits
    that branch's two per-expert arms: one driven by the convention index, one a literal
    `epfx + "{e}.gate_proj.weight"` probe kept for the model_types that have no convention
    entry. Every family with a CPU fixture now has a convention, so the literal arm is
    unreachable from this suite unless it is forced — and an unreachable arm is an
    unchecked one: a gate/up swap in it passes every other test here.

    Forcing `_convention_or_none` to None is exactly what a conventionless family sees
    (same mechanism as `test_existing_per_expert_families_load_unchanged`), which makes
    this an equivalence proof for the excluded layer rather than a smoke test.
    """
    from experts4bit_qlora import loader as loader_mod
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    torch.manual_seed(0)
    _write_ckpt(build(), str(tmp_path), per_expert=True)

    def _excluded(model):
        e = model.model.layers[1].mlp.experts
        return {n: t for n, t in list(e.named_parameters()) + list(e.named_buffers())}

    torch.manual_seed(1234)
    via_convention, _ = load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8,
                                                quantize_layers={0})
    monkeypatch.setattr(loader_mod, "_convention_or_none", lambda model_type: None)
    torch.manual_seed(1234)
    via_literal, _ = load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8,
                                             quantize_layers={0})

    got, want = _excluded(via_convention), _excluded(via_literal)
    assert set(got) == set(want)
    # Guard the guard: with no expert stack in the comparison this would be asserting
    # nothing at all, and would stay green whatever either read did.
    assert "gate_up_proj" in got, f"no expert stack on the excluded layer: {sorted(got)}"
    for n in sorted(got):
        assert not got[n].is_meta and not want[n].is_meta, n
        assert got[n].dtype == want[n].dtype, n
        assert torch.equal(got[n].cpu(), want[n].cpu()), n


@pytest.mark.parametrize("build", [_mixtral, _phimoe], ids=["mixtral", "phimoe"])
def test_quantize_layers_excluded_layer_from_a_block_sparse_checkpoint(build, tmp_path):
    """An excluded layer is filled from keys the MODULE prefix does not even match.

    This family stores experts under `block_sparse_moe.experts.<e>.{w1,w3,w2}` and builds
    them at `mlp.experts`, so an excluded layer's checkpoint keys do not start with the
    module prefix at all — the literal per-expert arm cannot see them, and the generic
    weight walk died on "MixtralExperts has no attribute `0`". The convention index
    locates them, and its fuser keeps w1=gate / w3=up where that was adjudicated: gate
    and up are shape-identical, so a swap here would round-trip a different function on
    exactly the layers the caller chose to leave alone, with nothing raised.
    """
    from experts4bit_qlora import ExpertsNbit
    from experts4bit_qlora.loader import load_moe_4bit_streaming

    torch.manual_seed(0)
    model = build()
    gold = {k: v.clone() for k, v in model.state_dict().items() if ".experts." in k}
    _write_block_sparse_ckpt(model, str(tmp_path))

    m, _ = load_moe_4bit_streaming(str(tmp_path), DEVICE, DTYPE, r=4, alpha=8,
                                   quantize_layers={0})
    quantized = {
        i for i in range(model.config.num_hidden_layers)
        if any(isinstance(sub, ExpertsNbit) for sub in m.model.layers[i].mlp.modules())
    }
    assert quantized == {0}, f"expected only layer 0 quantized, got {sorted(quantized)}"

    skipped = m.model.layers[1].mlp.experts
    for proj in ("gate_up_proj", "down_proj"):
        got = getattr(skipped, proj)
        want = gold[f"model.layers.1.mlp.experts.{proj}"].to(DTYPE)
        assert not got.is_meta, f"{proj} never reached the excluded layer"
        assert got.dtype == DTYPE, f"{proj} is {got.dtype}, not the base dtype"
        assert torch.equal(got.cpu(), want.cpu()), f"{proj} does not match the source weights"


# gpt-oss and Kimi-K3/DeepSeek-V4 keep their experts PACKED (MXFP4 blocks/scales), and this
# loader can only read that layout by dequantizing it on the way into a quantized module —
# so "leave this layer in the base dtype" has no implementation for them. Their real
# checkpoints need CUDA and tens of GB, so the dispatch is exercised directly here: the
# point under test is which spelling routes where, and that is decidable from the key names
# alone. What must NOT happen is the pre-fix behaviour — falling through to the generic
# weight walk and dying on `get_submodule` or a stray meta tensor, naming neither the flag
# nor the layout.
@pytest.mark.parametrize("model_type,keys", [
    ("gpt_oss", ["gate_up_proj_blocks", "gate_up_proj_scales", "down_proj_blocks"]),
    ("kimi_k3", ["0.w1.weight_packed", "0.w1.weight_scale", "0.w2.weight_packed"]),
])
def test_quantize_layers_refuses_packed_expert_layouts(model_type, keys):
    from experts4bit_qlora.loader import _place_unquantized_experts

    epfx = "model.layers.1.mlp.experts."
    weight_map = {epfx + k: "model.safetensors" for k in keys}
    with pytest.raises(NotImplementedError) as e:
        # `model`/`get` are unused on this path — it refuses before reading anything.
        _place_unquantized_experts(None, epfx, 1, weight_map, None, 8, model_type, {}, True)
    msg = str(e.value)
    assert "quantize_layers" in msg, "the refusal must name the flag the caller passed"
    assert epfx in msg, "the refusal must name the layout it found"
    # And it must read as a REFUSAL to the shared guard, not as an absent bnb backend: this
    # is a NotImplementedError, which `QUANTIZE_UNAVAILABLE` catches, so an unregistered
    # message would turn a declined load into a green skip. `test_quant_guard.py` enforces
    # this across the whole loader; asserted here too so the refusal's own arm says it.
    from quant_guard import is_loader_refusal

    assert is_loader_refusal(e.value), f"unregistered loader refusal: {msg!r}"


@pytest.mark.parametrize("keys", [
    [],                                          # dense layer: no experts here at all
    ["gate_up_proj", "down_proj"],               # already fused on disk
])
def test_quantize_layers_leaves_placeable_layouts_to_the_weight_walk(keys):
    """Dense and fused-on-disk layers are placed by the generic `_assign` pass, unchanged.

    Claiming those keys here would be the mirror-image bug: the walk would skip them and
    the layer would keep meta parameters.
    """
    from experts4bit_qlora.loader import _place_unquantized_experts

    epfx = "model.layers.1.mlp.experts."
    weight_map = {epfx + k: "model.safetensors" for k in keys}
    consumed, narrowed = _place_unquantized_experts(
        None, epfx, 1, weight_map, None, 8, "olmoe", {}, True)
    assert consumed == set() and narrowed == []
