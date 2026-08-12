"""Streaming 4-bit loader for fused-MoE checkpoints (OLMoE, Qwen3-MoE / Qwen3.5-MoE, Gemma-4, GraniteMoe).

Streams the checkpoint tensor-by-tensor straight onto the GPU, quantizing each fused expert stack
to NF4 on the way and dropping the bf16 source immediately, so the full bf16 model is never
materialized in CPU *or* GPU memory. Each layer's fused ``experts`` module is swapped for a frozen
4-bit :class:`Experts4bit` base wrapped in trainable per-expert :class:`ExpertsLoRA` adapters.

Supports SwiGLU fused-MoE architectures. Experts may be stored on disk either **per-expert**
(``...experts.{e}.{gate,up,down}_proj.weight`` — OLMoE, Qwen3-MoE) or already **fused**
(``...experts.{gate_up,down}_proj`` — Gemma-4, GraniteMoe); both are handled. The experts module
sits under the MLP for OLMoE/Qwen3, directly on the layer (beside a parallel dense MLP) for
Gemma-4, and under a ``block_sparse_moe`` block for GraniteMoe — whose Hub checkpoints use legacy
tensor spellings (``input_linear``/``output_linear``, see :data:`LEGACY_KEY_RENAMES`). Requires
transformers>=5.0.
"""

import json
import os
import struct

from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from safetensors import safe_open
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.activations import ACT2FN

from . import Experts4bit, ExpertsNbit, normalize_quant_type
from .deepseek_v4 import DEFAULT_SWIGLU_LIMIT, DeepseekV4Experts4bit
from .deepseek_v4 import rename_checkpoint_key as rename_deepseek_v4_key
from .fp8_blocks import convert_to_fp8_blocks
from .gptoss import GptOssExperts4bit
from .lora import ExpertsLoRA
from .mxfp4 import dequantize_mxfp4
from .offload import enable_expert_offload, enable_inference_prefetch
from .util import log

# model_type -> experts submodule path relative to `model.layers.{i}`.
# OLMoE/Qwen3 nest experts under the MLP; Gemma-4 puts them beside a parallel dense MLP; GraniteMoe
# nests them under a `block_sparse_moe` block (router + experts; no parallel dense branch).
SUPPORTED_ARCHITECTURES = {
    "olmoe": "mlp.experts",
    "qwen3_moe": "mlp.experts",
    "qwen3_5_moe": "mlp.experts",
    "gpt_oss": "mlp.experts",  # GPT-OSS: on-disk MXFP4 blocks/scales + per-proj biases (see gptoss.py)
    "gemma4": "experts",  # multimodal top-level config
    "gemma4_text": "experts",  # the text tower (what a text-only QLoRA loads)
    "granitemoe": "block_sparse_moe.experts",  # IBM Granite MoE (granite-3.0-*-a*m, PowerMoE-3b)
    # Kimi K3 (`KimiK3ForConditionalGeneration`, DeepSeek-V3 lineage): per-expert
    # MXFP4 under `block_sparse_moe.experts.{e}.w{1,3,2}` — see K3_PER_EXPERT_MXFP4.
    "kimi_k3": "block_sparse_moe.experts",
    # DeepSeek-V4 (Flash/Pro): per-expert MXFP4 under `mlp.experts.{e}.w{1,3,2}`, same
    # lineage as K3 but with honest dtype labels and a `weight`/`scale` suffix pair.
    # Its DENSE half is block-scaled FP8, not bf16 — see DEEPSEEK_V4_FP8_DENSE.
    "deepseek_v4": "mlp.experts",
}
# model_type -> checkpoint prefix for the text tower of a MULTIMODAL config. Gemma-4
# nests the language model as `model.language_model.`; Kimi K3 reverses the order
# (`language_model.model.`), so the prefix cannot be derived from the presence of
# `text_config` alone — it is per-family.
MULTIMODAL_CKPT_PREFIX = {"kimi_k3": "language_model.model."}
# model_type -> (gate, up, down) on-disk projection spellings for per-expert MXFP4
# checkpoints, plus the packed/scale suffixes. K3: w1=gate, w3=up, w2=down —
# confirmed by SHAPES, not convention (w1/w3 are [inter, latent]; w2 is
# [latent, inter]).
K3_PER_EXPERT_MXFP4 = {
    "kimi_k3": (("w1", "w3", "w2"), "weight_packed", "weight_scale"),
    # V4 keeps K3's w1/w3/w2 spelling and shapes but names the tensors `weight`/`scale`
    # and labels their dtypes honestly (`I8` / `F8_E8M0` rather than K3's `U8`/`U8`).
    # `dequantize_mxfp4` reinterprets both, so the same branch reads them unchanged.
    "deepseek_v4": (("w1", "w3", "w2"), "weight", "scale"),
}
# model_type -> the checkpoint is published in a NON-transformers spelling and every key
# must be rewritten before it can be matched against the module tree. Unlike
# MULTIMODAL_CKPT_PREFIX (which strips a prefix), these checkpoints need a full rename —
# including ADDING the `model.` prefix, which a prefix-strip cannot express.
CKPT_KEY_REWRITERS = {"deepseek_v4": rename_deepseek_v4_key}
# model_type -> the non-expert ("dense") weights are block-scaled FP8 rather than bf16,
# and each `X.weight` carries a companion `X.scale`. Those pairs become Fp8BlockLinear.
DEEPSEEK_V4_FP8_DENSE = {"deepseek_v4"}
SUPPORTED_MODEL_TYPES = set(SUPPORTED_ARCHITECTURES)


def _read_compatible_convention(model_type):
    """True if this model_type loads through the EXISTING per-expert read path.

    olmoe and qwen3_moe are already supported and are the qwen2_moe convention;
    every other qwen2_moe-convention model_type stores experts identically
    (``mlp.experts.{e}.{gate,up,down}_proj.weight``), so the same streaming read
    handles them with no new code. Validated on a rented A6000: Qwen1.5-MoE-A2.7B
    (qwen2_moe, not previously listed) loaded, nf4-quantized, and generated.

    Deliberately NARROW: mixtral (w1/w3/w2 spelling), dbrx (flat stacks) and
    nemotron_h (no gate) need their own read handling and are NOT admitted here."""
    from .moe_conventions import MoEConventionError, convention_for
    try:
        return convention_for(model_type).name == "qwen2_moe"
    except MoEConventionError:
        return False


def expert_layout_for(model_type):
    """``(expert_submodule_path, has_gate)`` for the quantized loader.

    The MoE convention system (:mod:`experts4bit_qlora.moe_conventions`) is the
    broad, adjudicated source of truth for expert layout — 41 model_types and
    counting — and its ``fused_prefix`` is exactly the submodule path this loader
    calls ``expert_rel``. Where a convention exists, defer to it, and take
    ``has_gate`` from it too (so a non-gated family like nemotron_h is handled
    correctly rather than assumed SwiGLU). Fall back to this loader's own map for
    the dedicated-quant specials (gemma4, kimi_k3, deepseek_v4) that predate the
    convention system and carry MXFP4/FP8 handling a plain convention does not.

    Verified: for every model_type both systems cover, the paths already agree
    (see tests) — this makes that agreement the mechanism, not a coincidence.
    """
    from .moe_conventions import MoEConventionError, convention_for
    try:
        conv = convention_for(model_type)
        return conv.fused_prefix, conv.gated
    except MoEConventionError:
        if model_type in SUPPORTED_ARCHITECTURES:
            return SUPPORTED_ARCHITECTURES[model_type], True
        raise

# model_type -> ((legacy on-disk spelling, name in the transformers>=5 module tree), ...).
# GraniteMoe checkpoints on the Hub predate the standardized fused-experts interface: the fused
# stacks are stored as `block_sparse_moe.input_linear.weight` [num_experts, 2*inter, hidden]
# (gate+up pre-fused, gate first — the module chunks the projection in half and activates the first
# half, matching Experts4bit's has_gate convention) and `block_sparse_moe.output_linear.weight`
# [num_experts, hidden, inter]; the router weight sits one module deeper, at `router.layer.weight`.
# transformers' own from_pretrained applies exactly these renames (conversion_mapping.py,
# "granitemoe"); this loader reads shards directly, so it must apply them itself. Substring
# renames over the whole key set: a checkpoint already saved with the current names matches
# nothing and passes through unchanged.
LEGACY_KEY_RENAMES = {
    "granitemoe": (
        ("block_sparse_moe.input_linear.weight", "block_sparse_moe.experts.gate_up_proj"),
        ("block_sparse_moe.output_linear.weight", "block_sparse_moe.experts.down_proj"),
        ("block_sparse_moe.router.layer.weight", "block_sparse_moe.router.weight"),
    ),
}


def _fit(name, tensor, want):
    """Reconcile a checkpoint tensor with the shape the built module declares.

    This loader used to place whatever the shard held, unchecked — and `_assign`
    REPLACES the meta parameter, so a wrong-shaped tensor silently becomes the model's
    shape. That is the failure mode with no symptom: nothing raises at load, and the
    disagreement surfaces (if at all) as a broadcast that happens to work.

    Kimi K3 makes it concrete. Its 69 KDA layers ship ``A_log`` per-head — 96 values —
    ZERO-PADDED to 128 lanes, while the released modeling code builds ``(96,)`` and
    never slices. vLLM's ``a_log_weight_loader`` narrows to the head count and the
    discarded tail is exact zeros, so narrowing is the reference behaviour rather than
    a guess about intent.

    Permit exactly that shape of disagreement: 1-D, LONGER than the parameter, and
    provably zero past its end. Everything else raises. A blanket narrow would
    reintroduce the silent wrongness this check exists to catch, because the padding is
    not inert if used: ``-exp(0) = -1`` would give 32 phantom heads a full-strength
    decay.

    Dtype is deliberately NOT coerced. The checkpoint's dtype is the authority (K3
    keeps ``A_log``/``dt_bias``/``o_norm`` in fp32 under a bf16 model), and quietly
    casting weights is the thing this package exists not to do.
    """
    if want is None or tuple(tensor.shape) == tuple(want.shape):
        return tensor, False
    if (tensor.dim() == want.dim() == 1 and tensor.numel() > want.numel()
            and bool((tensor[want.numel():] == 0).all())):
        return tensor[:want.numel()].clone(), True
    raise ValueError(
        f"{name}: checkpoint holds {tuple(tensor.shape)} but the model declares "
        f"{tuple(want.shape)}. Only 1-D trailing padding that is provably zero is "
        "narrowed; this is not that, so placing it would give the model the "
        "checkpoint's shape and no error until much later. Fix the config, the key "
        "mapping, or the checkpoint."
    )


def _assign(model, name, tensor):
    """Place a real (GPU) tensor into a meta-initialized module by dotted name.

    Returns True if the tensor was narrowed to fit — see :func:`_fit`.
    """
    *path, attr = name.split(".")
    mod = model.get_submodule(".".join(path)) if path else model
    if attr in mod._parameters:
        tensor, cut = _fit(name, tensor, mod._parameters[attr])
        mod._parameters[attr] = torch.nn.Parameter(tensor, requires_grad=False)
        return cut
    if attr in mod._buffers:
        tensor, cut = _fit(name, tensor, mod._buffers[attr])
        mod._buffers[attr] = tensor
        return cut
    # No declared parameter or buffer to check against: nothing to compare, and
    # nothing the model already believes about the shape to contradict.
    setattr(mod, attr, tensor)
    return False


class _RawShardReader:
    """Read a safetensors tensor as raw ``uint8``, bypassing torch's dtype table.

    ``safe_open(...).get_tensor`` materializes through that table, so a shard declaring a
    dtype this torch build does not have cannot be read *at all* — it raises
    ``AttributeError`` from inside ``torch.__getattr__``, not from safetensors. DeepSeek-V4
    hits this immediately: its MXFP4 expert scales and its FP8 dense scales are both
    labelled ``F8_E8M0``, which torch only grew in 2.7.

    The bytes were never the problem, only the label. And both consumers want the biased
    exponent *byte* anyway — ``dequantize_mxfp4`` feeds it to ``ldexp``, ``Fp8BlockLinear``
    normalizes it to ``uint8`` on construction — so reading it as ``uint8`` here is the
    more direct path, not a workaround for old torch.
    """

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.header.pop("__metadata__", None)
        self.base = 8 + n

    def u8(self, key):
        meta = self.header[key]
        a, b = meta["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.base + a)
            buf = bytearray(f.read(b - a))
        return torch.frombuffer(buf, dtype=torch.uint8).reshape(meta["shape"])


def _install_fp8_block_linears(model, weight_map, get, expert_keys, dtype, device):
    """Swap every dense ``nn.Linear`` whose checkpoint weight is block-scaled FP8.

    DeepSeek-V4 keeps its non-expert half in ``float8_e4m3fn`` with one ``e8m0`` scale per
    ``[128, 128]`` tile. Those weights have no home in a bf16 ``nn.Linear`` and their
    ``.scale`` companions have no module at all, so `_assign` cannot place either; the pair
    becomes an :class:`Fp8BlockLinear`, which keeps the FP8 bytes resident and decodes on
    use (~1 byte/param instead of 2 — the difference between fitting a 12 GB card and not).

    Pairing is by **sibling presence**, never by the ``.scale`` suffix. V4's
    hyper-connections ship a standalone parameter literally *named* ``scale``
    (``attn_hc.scale``, ``ffn_hc.scale``, ``hc_head.hc_scale``) with no ``weight`` beside
    it. Keying off the suffix would swallow those three per layer and leave the HC modules
    silently unloaded — caught by the reverse arm of ``test_deepseek_v4_keys``.

    Returns the set of checkpoint keys it consumed.
    """
    consumed = set()
    for name in list(weight_map):
        if not name.endswith(".weight") or name in expert_keys:
            continue
        stem = name[: -len(".weight")]
        scale_key = stem + ".scale"
        if scale_key not in weight_map or scale_key in expert_keys:
            continue
        w = get(name)
        if w.dtype not in (torch.float8_e4m3fn, torch.uint8, torch.int8):
            continue                       # a `.scale` sibling that is not FP8 storage
        target = model.get_submodule(stem)
        # Convert IN PLACE rather than substituting an Fp8BlockLinear. V4's
        # `self_attn.o_a_proj` is a `DeepseekV4GroupedLinear` — an `nn.Linear` SUBCLASS
        # whose forward is block-diagonal — so swapping in a real Linear quietly turns a
        # grouped projection into a dense one. Rebinding the weight keeps whatever
        # forward the module already has.
        convert_to_fp8_blocks(target, w, get(scale_key), compute_dtype=dtype)
        target.to(device)
        consumed |= {name, scale_key}
    return consumed


def load_moe_4bit_streaming(
    model_id, device, dtype, r, alpha, offload=False, pin=True, prefetch=False, quant_type="nf4",
    trust_remote_code=None, arena=None, quantize_layers=None,
):
    """Stream the checkpoint onto the GPU, quantizing fused experts to Experts4bit on the way.

    Peak memory stays low: the model is built on ``meta`` (no allocation), then each tensor is read
    one at a time directly to the GPU. The big fused-expert stacks are quantized to NF4 (~3.5x
    smaller) and their bf16 source is dropped immediately, so the full bf16 model never exists.

    When ``offload`` is set, each layer's frozen 4-bit experts are moved to (pinned, if ``pin``) CPU
    RAM *immediately after that layer is built* — inside the per-layer loop, never in a post-load
    pass (which would require every layer's experts GPU-resident first, defeating the purpose). A
    forward pre-hook streams a layer's experts back to the GPU just-in-time and a post-hook evicts
    them, so only one layer's experts are GPU-resident at a time (see :mod:`experts4bit_qlora.offload`).

    ``prefetch=True`` (with ``offload``) additionally links the layers for inference prefetch: during
    ``no_grad`` forwards each layer starts the next layer's H2D copy on a side stream, overlapping
    transfer with compute at a bounded cost of two layers resident instead of one. Training forwards
    are unaffected. See :func:`experts4bit_qlora.offload.enable_inference_prefetch`.
    """
    # Validate + canonicalize the scheme FIRST: a bad quant_type must fail here, before any config
    # fetch, snapshot download, or shard read — and the Experts4bit-vs-ExpertsNbit class dispatch
    # below must only ever see canonical names (an unnormalized alias would silently pick the
    # wrong class).
    quant_type = normalize_quant_type(quant_type)
    # `arena`: serve experts from a baked NVMe arena instead of reading them out of
    # the checkpoint. The expert modules are then built on `meta` (shapes only), so
    # expert storage stops scaling with model size — the difference between needing
    # 1.446 TB of host RAM for Kimi K3's experts and needing none. Pair with
    # `nvme_experts.enable_nvme_residency` on the SAME arena to make them reachable;
    # the model returned here cannot run until you do.
    arena_index = None
    if arena is not None:
        from nvme_arena import load_index
        arena_index = load_index(arena)
    if prefetch and not offload:
        raise ValueError(
            "prefetch=True requires offload=True: prefetch overlaps the H2D copy of offloaded "
            "experts; without offload there is nothing to prefetch."
        )
    # Architectures that live in the checkpoint's own repo (Kimi K3's Kimi-Linear
    # attention + SiTU activation are not in any released transformers) cannot be
    # read at all without this, and AutoConfig raises BEFORE the architecture gate
    # below, so the failure is opaque: "The repository ... contains custom code".
    # Opt-in only — executing repo code is the caller's decision, never a default.
    if trust_remote_code is None:
        trust_remote_code = os.environ.get("E4B_TRUST_REMOTE_CODE", "0") == "1"
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model_type = getattr(config, "model_type", None)
    if model_type not in SUPPORTED_ARCHITECTURES and not _read_compatible_convention(model_type):
        raise NotImplementedError(
            f"Unsupported model_type={model_type!r}. This streaming loader handles SwiGLU fused-MoE "
            f"checkpoints: {sorted(SUPPORTED_ARCHITECTURES)}, plus every model_type on the qwen2_moe "
            f"convention (same per-expert mlp.experts.N.{{gate,up,down}}_proj read path). The Experts4bit "
            "primitive itself is model-agnostic — see the README 'Scope' note to adapt another architecture."
        )
    # Identity ("zero-computation") experts: the router indexes a space LARGER than the
    # set of real experts, and the surplus indices route the token through nn.Identity
    # scaled by its router weight instead of a SwiGLU. LongCat-Flash ships 512 routed +
    # 256 identity by default. Experts4bit has no identity slot, so a load would build
    # only the routed experts while the router keeps emitting indices past the end.
    #
    # Refusing here rather than at the read: the surplus experts carry gate_up rows the
    # forward never reads and NO down_proj at all, so the per-expert reader consumes
    # 0..n_routed-1, leaves the rest orphaned, and the generic weight walk then dies on
    # `get_submodule(".../experts.10")` with `ExpertsLoRA has no attribute '10'` — which
    # says nothing about what is actually unsupported.
    _gate_cfg = getattr(config, "text_config", None) or config   # same unwrap as lm_config below
    n_zero = int(getattr(_gate_cfg, "zero_expert_num", 0) or 0)
    if n_zero > 0:
        n_routed = int(getattr(_gate_cfg, "n_routed_experts", 0) or 0)
        raise NotImplementedError(
            f"{model_type!r} uses {n_zero} identity ('zero-computation') experts on top of "
            f"{n_routed} routed experts. The router selects over all {n_routed + n_zero}, and "
            "indices at or above the routed count pass the token through unchanged rather "
            "than through a SwiGLU expert. Experts4bit represents SwiGLU experts only, so "
            "loading just the routed ones would leave the router addressing experts that do "
            "not exist. Supporting this needs an identity slot in the expert primitive, not "
            "a loader change."
        )
    # Source the expert path and gate from the convention when one exists (the
    # broad source of truth), else this loader's own map. Both agree today; this
    # makes the convention authoritative so a non-gated family loads correctly.
    expert_rel, has_gate = expert_layout_for(model_type)
    # Multimodal configs (e.g. Gemma-4's `gemma4`) nest the language model under `text_config` and
    # prefix its checkpoint tensors with `model.language_model.` (vision lives under `model.vision_tower.`).
    # Build + size the text tower from that sub-config, and strip the prefix so keys match the text
    # CausalLM we build (dropping the vision weights we don't need for a text-only QLoRA).
    lm_config = getattr(config, "text_config", None) or config
    ckpt_prefix = ""
    if lm_config is not config:
        ckpt_prefix = MULTIMODAL_CKPT_PREFIX.get(model_type, "model.language_model.")
    act_name = getattr(lm_config, "hidden_activation", None) or getattr(lm_config, "hidden_act", "silu")
    try:
        activation = ACT2FN[act_name]
    except KeyError:
        # gpt_oss uses its own clamped GLU inside GptOssExperts4bit; the generic
        # activation is unused on that path, so a missing ACT2FN entry is fine.
        # Kimi K3 declares hidden_act="situ", defined only in the checkpoint's own
        # modeling code -- surface it rather than silently substituting SiLU, which
        # would load a model that runs but is quietly WRONG.
        activation = None
        if act_name not in ("silu", None) and model_type not in ("gpt_oss",):
            log(f"  NOTE: activation {act_name!r} is not in transformers' ACT2FN; "
                f"the fused-expert GLU will use the module default. Verify numerics "
                f"before trusting outputs from this checkpoint.")

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            lm_config, dtype=dtype, trust_remote_code=trust_remote_code)

    snap = (
        model_id
        if os.path.isdir(model_id)
        else snapshot_download(
            model_id,
            allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt"],
        )
    )
    index_path = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(index_path):
        raw_map = json.load(open(index_path))["weight_map"]
    else:
        # Small checkpoints ship unsharded with no index (e.g. granite-3.0-1b-a400m-instruct is a
        # single model.safetensors): synthesize the weight map from the file's own key list so the
        # streaming path below works unchanged. Anything else is unstreamable — fail loudly.
        single_path = os.path.join(snap, "model.safetensors")
        if not os.path.exists(single_path):
            raise FileNotFoundError(
                f"{model_id!r}: found neither 'model.safetensors.index.json' (sharded) nor "
                "'model.safetensors' (single-file) in the snapshot — nothing this loader can stream."
            )
        with safe_open(single_path, framework="pt", device="cpu") as f:
            raw_map = dict.fromkeys(f.keys(), "model.safetensors")
    rewrite = CKPT_KEY_REWRITERS.get(model_type)
    if rewrite is not None:
        # This checkpoint ships in the model's OWN reference spelling. transformers can
        # convert it — via the central `conversion_mapping.py`, not a class attribute —
        # but only inside `from_pretrained`, which this streaming loader never calls.
        # So rewrite every key here, including adding the `model.` prefix, which the
        # ckpt_prefix branch below cannot express (it only ever strips).
        weight_map, orig_key, dropped = {}, {}, []
        for k, f in raw_map.items():
            new = rewrite(k)
            if new is None:
                dropped.append(k)     # no module to receive it (V4: the MTP block)
                continue
            weight_map[new] = f
            orig_key[new] = k
        if dropped:
            log(f"  skipped {len(dropped)} checkpoint tensor(s) the text model does not "
                f"build, e.g. {dropped[0]}")
        if not weight_map:
            raise RuntimeError(
                f"{model_id!r}: the {model_type!r} key rewriter mapped every one of "
                f"{len(raw_map)} checkpoint tensors to nothing. That is a rewriter/"
                "checkpoint mismatch, not an empty checkpoint — refusing rather than "
                "proceeding to build a model with no weights."
            )
    elif ckpt_prefix:
        weight_map = {"model." + k[len(ckpt_prefix) :]: f for k, f in raw_map.items() if k.startswith(ckpt_prefix)}
        orig_key = {"model." + k[len(ckpt_prefix) :]: k for k in raw_map if k.startswith(ckpt_prefix)}
        # The output head does NOT carry the text-tower prefix — Gemma-4 keeps it at top level
        # (`lm_head.weight`), K3 one level up from its own (`language_model.lm_head.weight`) — so
        # the filter above drops it along with the vision weights. For a tied checkpoint that is
        # right (there is no head on disk); for an UNTIED one the tie fallback further down would
        # then install embed_tokens as the output head with no error and plausible-shaped logits.
        # Recover it here so the ordinary `_assign` pass places the real head.
        heads = [k for k in raw_map if k.endswith("lm_head.weight") and not k.startswith(ckpt_prefix)]
        if len(heads) == 1:
            weight_map["lm_head.weight"] = raw_map[heads[0]]
            orig_key["lm_head.weight"] = heads[0]
            log(f"  untied output head recovered from {heads[0]!r} (outside the "
                f"{ckpt_prefix!r} text-tower prefix, which would otherwise drop it)")
        elif len(heads) > 1:
            # Ambiguous rather than absent: picking one would be a guess about which tower's
            # head this text model wants. Say so here; the tie gate below decides the outcome.
            log(f"  NOTE: {len(heads)} candidate output heads outside {ckpt_prefix!r} "
                f"({heads[:3]}) — none assigned automatically")
    else:
        weight_map, orig_key = raw_map, {k: k for k in raw_map}
    # Normalize legacy tensor spellings (see LEGACY_KEY_RENAMES) so the expert-fusing loop and the
    # non-expert `_assign` pass below only ever see the module names the built model actually has;
    # `orig_key` keeps pointing at the on-disk spelling the shard must be read with.
    for old, new in LEGACY_KEY_RENAMES.get(model_type, ()):
        for key in [k for k in weight_map if old in k]:
            renamed = key.replace(old, new)
            weight_map[renamed] = weight_map.pop(key)
            orig_key[renamed] = orig_key.pop(key)
    handles = {f: safe_open(os.path.join(snap, f), framework="pt", device=device) for f in set(weight_map.values())}

    raw_readers = {}

    def get(name):
        fn = weight_map[name]
        try:
            return handles[fn].get_tensor(orig_key[name])
        except AttributeError:
            # The shard declares a dtype this torch build has no name for (see
            # _RawShardReader). Fall back to the bytes. Deliberately narrow: only this
            # one failure mode reaches here, and anything else still raises.
            if fn not in raw_readers:
                raw_readers[fn] = _RawShardReader(os.path.join(snap, fn))
                log(f"  {fn}: holds a dtype this torch ({torch.__version__}) cannot "
                    f"name; reading those tensors as raw uint8")
            return raw_readers[fn].u8(orig_key[name]).to(device)

    n_layers = lm_config.num_hidden_layers
    n_exp = getattr(lm_config, "num_local_experts", None) or getattr(lm_config, "num_experts", None)
    log(f"  fusing + quantizing experts (up to {n_layers}x{n_exp}) to {quant_type} (streaming)...")
    expert_keys = set()
    meta_expert_prefixes = []      # arena mode: modules whose buffers stay on meta
    offload_handles = []
    n_moe = 0
    for i in range(n_layers):
        epfx = f"model.layers.{i}.{expert_rel}."  # "...mlp.experts." / "...experts." / "...block_sparse_moe.experts."
        if quantize_layers is not None and i not in quantize_layers:
            # Deliberately left in the base dtype. Same effect as the dense-layer
            # `continue` below: the original module stays in place, unquantized.
            continue
        if arena_index is not None:
            # Every checkpoint key under the experts submodule is an expert tensor,
            # so they can be marked read-and-skipped WITHOUT reading them — which is
            # the whole point: at K3 scale the read is what does not fit.
            keys = {k for k in weight_map if k.startswith(epfx)}
            if not keys:
                continue                                  # dense layer
            bias = sorted(k for k in keys if k.endswith("_bias"))
            if bias:
                raise NotImplementedError(
                    f"layer {i}: arena serving does not yet carry per-expert biases "
                    f"({bias[0]!r} and {len(bias) - 1} more). gpt-oss keeps its "
                    "biases resident alongside the packed stacks; an arena path for "
                    "them is a separate change. Refusing rather than dropping them, "
                    "which would silently change the epilogue.")
            expert_keys.update(keys)
            n_moe += 1
            from .nvme_experts import build_meta_experts
            v4 = model_type == "deepseek_v4"
            experts = build_meta_experts(
                arena_index, n_exp, has_gate=True, activation=activation,
                compute_dtype=dtype, quant_type=quant_type,
                cls=DeepseekV4Experts4bit if v4 else None)
            if v4:
                # The epilogue has to survive the arena path too — see the bare-build
                # note on the resident branch below.
                experts.limit = float(
                    getattr(lm_config, "swiglu_limit", DEFAULT_SWIGLU_LIMIT))
            parent, leaf = epfx.rstrip(".").rsplit(".", 1)
            setattr(model.get_submodule(parent), leaf, experts)
            meta_expert_prefixes.append(epfx)
            continue
        if f"{epfx}gate_up_proj_blocks" in weight_map:
            # GPT-OSS: experts on disk as MXFP4 (blocks/scales) + per-projection biases.
            # Dequantize the exact released bytes, then build a faithful NF4 expert
            # (biases + clamped GLU) — see gptoss.py. Built bare (no ExpertsLoRA):
            # GPT-OSS-aware training LoRA is a separate change.
            gate_up = dequantize_mxfp4(get(f"{epfx}gate_up_proj_blocks"), get(f"{epfx}gate_up_proj_scales"), dtype=dtype)
            down = dequantize_mxfp4(get(f"{epfx}down_proj_blocks"), get(f"{epfx}down_proj_scales"), dtype=dtype)
            gu_bias = get(f"{epfx}gate_up_proj_bias").to(dtype)
            dn_bias = get(f"{epfx}down_proj_bias").to(dtype)
            expert_keys.update({
                f"{epfx}{k}" for k in (
                    "gate_up_proj_blocks", "gate_up_proj_scales", "gate_up_proj_bias",
                    "down_proj_blocks", "down_proj_scales", "down_proj_bias",
                )
            })
            n_moe += 1
            experts = GptOssExperts4bit.from_gptoss(
                gate_up, gu_bias, down, dn_bias, quant_type=quant_type, compute_dtype=dtype
            ).to(device)
            if offload:
                # Bare-module offload: packed experts stream from (pinned) CPU one layer at a
                # time; the small biases stay resident. Lets gpt-oss-20b (~11 GB NF4) load and
                # run on a 12 GB card.
                offload_handles.append(enable_expert_offload(experts, device, pin=pin))
            parent, leaf = epfx.rstrip(".").rsplit(".", 1)
            setattr(model.get_submodule(parent), leaf, experts)
            del gate_up, down
            continue
        if f"{epfx}gate_up_proj" in weight_map:
            # Already fused on disk (Gemma-4; GraniteMoe after the legacy rename above):
            # [num_experts, 2*inter, hidden] / [num_experts, hidden, inter].
            gate_up = get(f"{epfx}gate_up_proj").to(dtype)
            down = get(f"{epfx}down_proj").to(dtype)
            expert_keys.update({f"{epfx}gate_up_proj", f"{epfx}down_proj"})
        elif model_type in K3_PER_EXPERT_MXFP4 and \
                f"{epfx}0.{K3_PER_EXPERT_MXFP4[model_type][0][0]}.{K3_PER_EXPERT_MXFP4[model_type][1]}" in weight_map:
            # Per-expert MXFP4 on disk (Kimi K3): each projection is a packed uint8
            # `[rows, K//2]` plus an e8m0 `[rows, K//32]` scale, two fp4 nibbles per
            # byte. Same numeric format gpt-oss uses, different NAMING and shape rank
            # -- so it reuses `dequantize_mxfp4` unchanged.
            #
            # `dequantize_mxfp4` returns the TRANSPOSE of the stored matrix
            # (`[K, rows]`, the layout GptOssExperts wants), but `from_float` below
            # expects gate_up `[2*inter, hidden]` / down `[hidden, inter]` in STORED
            # orientation -- hence the `.T`. Getting this wrong yields a model that
            # loads with plausible shapes and computes garbage.
            (p_gate, p_up, p_down), packed_kind, scale_kind = K3_PER_EXPERT_MXFP4[model_type]

            def _mxfp4_expert(e, proj):
                blocks = get(f"{epfx}{e}.{proj}.{packed_kind}")
                scales = get(f"{epfx}{e}.{proj}.{scale_kind}")
                rows, kh = blocks.shape
                groups = scales.shape[1]
                w = dequantize_mxfp4(
                    blocks.reshape(rows, groups, kh // groups), scales, dtype=dtype)
                return w.T.contiguous()          # [K, rows] -> stored [rows, K]

            gate_up_rows, down_rows = [], []
            for e in range(n_exp):
                gate_up_rows.append(torch.cat(
                    [_mxfp4_expert(e, p_gate), _mxfp4_expert(e, p_up)], dim=0))
                down_rows.append(_mxfp4_expert(e, p_down))
                expert_keys.update({
                    f"{epfx}{e}.{proj}.{kind}"
                    for proj in (p_gate, p_up, p_down)
                    for kind in (packed_kind, scale_kind)
                })
            gate_up = torch.stack(gate_up_rows).to(dtype)
            down = torch.stack(down_rows).to(dtype)
        elif f"{epfx}0.gate_proj.weight" in weight_map:
            # Per-expert Linears on disk (OLMoE, Qwen3): fuse gate_up[e] = cat([gate, up]).
            gate_up_rows, down_rows = [], []
            for e in range(n_exp):
                g, u, d = (
                    get(f"{epfx}{e}.gate_proj.weight"),
                    get(f"{epfx}{e}.up_proj.weight"),
                    get(f"{epfx}{e}.down_proj.weight"),
                )
                gate_up_rows.append(torch.cat([g, u], dim=0))  # [2*inter, hidden]
                down_rows.append(d)  # [hidden, inter]
                expert_keys.update({f"{epfx}{e}.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")})
            gate_up = torch.stack(gate_up_rows).to(dtype)
            down = torch.stack(down_rows).to(dtype)
        else:
            continue  # dense layer (no experts here — e.g. Qwen3 mlp_only_layers, or a dense Gemma-4 layer)
        n_moe += 1
        if model_type == "deepseek_v4":
            # V4's epilogue is a CLAMPED SwiGLU, and `ExpertsLoRA` re-implements the expert
            # math inline (to inject the delta before the nonlinearity) rather than calling
            # `base.forward`. It used to hardcode a plain SwiGLU there, so wrapping V4 would
            # have dropped the clamps on every training forward with nothing raised — which
            # is why this was built bare. The adapter now takes the epilogue from the base's
            # `_apply_gate` (see `lora._epilogue`), so wrapping is faithful and V4 is
            # trainable like any other supported architecture.
            base = DeepseekV4Experts4bit.from_deepseek_v4(
                gate_up, down,
                limit=float(getattr(lm_config, "swiglu_limit", DEFAULT_SWIGLU_LIMIT)),
                quant_type=quant_type, compute_dtype=dtype,
            )
            experts = ExpertsLoRA(base, r=r, alpha=alpha, dtype=dtype).to(device)
        else:
            # Instantiate the most-specific class for the scheme: 4-bit loads stay `Experts4bit`
            # instances, so downstream `isinstance(x, Experts4bit)` checks keep working exactly as
            # they did before the ExpertsNbit fold.
            base_cls = Experts4bit if quant_type in ("nf4", "fp4") else ExpertsNbit
            base = base_cls.from_float(
                gate_up, down, has_gate=has_gate, activation=activation, quant_type=quant_type, compute_dtype=dtype
            )
            experts = ExpertsLoRA(base, r=r, alpha=alpha, dtype=dtype).to(device)
        if offload:
            # Move this layer's packed experts to (pinned) CPU now, before the next layer is built,
            # so the GPU never holds more than one layer's experts at a time during load.
            offload_handles.append(enable_expert_offload(experts, device, pin=pin))
        parent, leaf = epfx.rstrip(".").rsplit(
            ".", 1
        )  # ("model.layers.i.mlp","experts") or ("model.layers.i","experts")
        setattr(model.get_submodule(parent), leaf, experts)
        del gate_up, down
    if n_moe == 0:
        raise RuntimeError(
            f"no fused expert stacks found in {model_id!r} (model_type={model_type!r}): expected "
            f"'model.layers.<i>.{expert_rel}.gate_up_proj' (fused) or "
            f"'model.layers.<i>.{expert_rel}.0.gate_proj.weight' (per-expert) tensors in the "
            "checkpoint. Refusing to return a model with zero quantized expert layers — silently "
            "skipping the experts is the exact failure this loader exists to prevent."
        )
    log(f"  quantized experts on {n_moe}/{n_layers} MoE layers ({n_exp} experts each)")

    if offload_handles:
        pinned = all(h.pinned for h in offload_handles)
        log(
            f"  offloaded {len(offload_handles)} layers' 4-bit experts to {'pinned ' if pinned else ''}CPU RAM "
            "(streamed to GPU one layer at a time during train/eval)"
        )
        if prefetch:
            # Handles were appended in layer order above, which is what the circular linking needs.
            enable_inference_prefetch(offload_handles)
            log("  inference prefetch ON: next layer's experts copy on a side stream during no_grad forwards")
        from .offload import _arena_enabled, _stats_enabled, report_offload_environment

        if _arena_enabled():
            log("  offload arena ON (E4B_OFFLOAD_ARENA): experts staged as consolidated per-dtype copies")
        if _stats_enabled():  # A2: name the PCIe bus + H2D ceiling these per-layer figures ride
            report_offload_environment(device, log)

    if model_type in DEEPSEEK_V4_FP8_DENSE:
        fp8_keys = _install_fp8_block_linears(
            model, weight_map, get, expert_keys, dtype, device)
        expert_keys |= fp8_keys          # placed as modules; not for the `_assign` pass
        log(f"  installed {len(fp8_keys) // 2} block-scaled FP8 linear(s) "
            f"(dense side stays FP8-resident, decoded on use)")

    log("  loading non-expert weights (attention/embeddings/router/norms/dense-mlp)...")
    narrowed = []
    for name in weight_map:
        if name not in expert_keys:
            if _assign(model, name, get(name)):
                narrowed.append(name)
    if narrowed:
        # Loud on purpose: this is a checkpoint/modeling-code disagreement that the
        # loader worked around, not a routine step. K3 hits it 69 times (A_log).
        log(f"  narrowed {len(narrowed)} zero-padded tensor(s) to the shape the model "
            f"declares, e.g. {narrowed[0]} — the discarded tail was verified zero "
            f"(see loader._fit)")

    # Non-persistent buffers (rotary inv_freq) aren't in the checkpoint — recompute every rotary module
    # the model has (some architectures, e.g. Gemma, use more than one). Generic; no per-model import.
    # Use `lm_config` (the text tower's config): a multimodal top-level config (Gemma-4's `Gemma4Config`)
    # lacks the rotary fields (`max_position_embeddings`, rope_theta) that live on `text_config`.
    for name, module in list(model.named_modules()):
        if type(module).__name__.endswith("RotaryEmbedding"):
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            setattr(parent, name.rsplit(".", 1)[-1], type(module)(lm_config).to(device))
    # Tie lm_head if the checkpoint relied on weight tying — and ONLY then. This used to be
    # unconditional, which is correct for a tied checkpoint (Gemma-4 ships no `lm_head.weight`)
    # and silently wrong for an untied one whose head never reached the model: every forward
    # then computes logits through the embedding matrix. Nothing raises, generations are
    # plausibly shaped, initial train loss sits at ln(vocab), and LoRA "converges" by learning
    # to steer hidden states into embed_tokens — then collapses when the adapter is served on a
    # stack that maps lm_head correctly. Gate on the config and fail loud instead.
    if model.lm_head.weight.is_meta:
        if getattr(lm_config, "tie_word_embeddings", True):
            model.lm_head.weight = model.model.embed_tokens.weight
        else:
            raise RuntimeError(
                f"{model_id!r}: config declares tie_word_embeddings=False, but no "
                "'lm_head.weight' reached the model — refusing to tie the output head to "
                "embed_tokens, which would load and generate without ever erroring. The head "
                "is in the checkpoint under a spelling this loader did not map; check the "
                "index against the multimodal prefix filter and the per-family key rewriters."
            )

    # This guard exists to catch a silently-incomplete load, so arena mode narrows
    # it rather than disabling it: expert buffers under a meta expert module are
    # INTENTIONALLY unmaterialized (they are served from the arena), but anything
    # else still on meta is the bug this check was written for.
    named = list(model.named_parameters()) + list(model.named_buffers())
    stray = [n for n, x in named
             if x.is_meta and not any(n.startswith(pfx) for pfx in meta_expert_prefixes)]
    if stray:
        raise RuntimeError(f"unmaterialized meta tensors remain: {stray[:8]}")
    if meta_expert_prefixes:
        deferred = sum(1 for n, x in named
                       if x.is_meta and any(n.startswith(p) for p in meta_expert_prefixes))
        # Name the enabler that matches THIS arena. The two are not interchangeable:
        # pointing the NF4 tier at an MXFP4 bake reads the wrong segments, so a caller
        # following the log gets a failure or a silent non-binding rather than a model.
        _nf4_arena = any("nf4." in str(seg)
                         for seg in (arena_index.get("segments") or ()))
        _enabler = ("enable_nvme_residency" if _nf4_arena
                    else "enable_mxfp4_nvme_residency")
        log(f"  experts deferred to the arena: {len(meta_expert_prefixes)} module(s), "
            f"{deferred} unmaterialized buffer(s) — call "
            f"nvme_experts.{_enabler}() before running this model")
    return model, config


# Backwards-compatible alias (was OLMoE-only).
load_olmoe_4bit_streaming = load_moe_4bit_streaming
