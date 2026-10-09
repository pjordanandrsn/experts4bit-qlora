# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Composite text projection and actual CPU int4 bytes must agree with plain text."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")
from safetensors.torch import load_file, save_file  # noqa: E402

from experts4bit_qlora.arch.moe_conventions import MoEConventionError  # noqa: E402
from experts4bit_qlora.arch.moe_plan import plan_moe_checkpoint  # noqa: E402
from experts4bit_qlora.engines.int4_experts import enable_serve_experts_int4, safetensors_reader  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "gpu_serve_smoke", Path(__file__).resolve().parents[1] / "bench/smoke/gpu_serve_smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


@pytest.fixture
def checkpoints(tmp_path):
    model = smoke.random_model("qwen3_5_moe")
    composite = tmp_path / "composite"
    smoke.write_checkpoint(model, composite)  # native text + visual and full MTP-with-experts distractors
    plain = tmp_path / "plain"
    plain.mkdir()
    save_file({k: v.contiguous().clone() for k, v in model.state_dict().items()}, str(plain / "model.safetensors"))
    model.config.save_pretrained(plain)
    return model, composite, plain


def test_composite_plan_claims_only_text_and_head_by_structure(checkpoints):
    model, composite, _ = checkpoints
    keys, _ = safetensors_reader(str(composite))
    # A differently named auxiliary block is also outside the text root:
    # this check prevents a visual/MTP blacklist from masquerading as structure.
    keys.append("auxiliary_decoder.layers.0.mlp.experts.gate_up_proj")
    plan = plan_moe_checkpoint(keys, model, model.config.model_type, skip_extra_layers=True)
    assert len(plan.passthrough) == len(model.state_dict())
    assert set(plan.passthrough.values()) == set(model.state_dict())
    for k, target in plan.passthrough.items():
        assert k == "lm_head.weight" or k.startswith("model.language_model.")
        assert target == "lm_head.weight" or target.startswith("model.")
    assert any(k.startswith("model.visual.") for k in plan.skipped_keys)
    assert any(k.startswith("mtp.") and "experts" in k for k in plan.skipped_keys)
    assert "auxiliary_decoder.layers.0.mlp.experts.gate_up_proj" in plan.skipped_keys
    assert len([v for v in plan.passthrough.values() if v.endswith("experts.gate_up_proj")]) == 2


def test_native_text_alias_is_admitted_and_loads_plain_checkpoint_on_cpu(checkpoints):
    from experts4bit_qlora import ExpertsNbit
    from experts4bit_qlora.loader import check_admission, load_moe_4bit_streaming
    from quant_guard import require_quantize

    model, _, plain = checkpoints
    assert model.config.model_type == "qwen3_5_moe_text"
    check_admission(model.config)
    require_quantize("cpu")
    # One layer quantized and one left plain exercises both native fused reads.
    # A missing CPU quantizer may skip here, after the admission assertion;
    # refusal by the actual loader remains a hard failure.
    loaded, config = load_moe_4bit_streaming(
        str(plain), "cpu", torch.float32, r=4, alpha=8, quantize_layers={0})
    assert config.model_type == "qwen3_5_moe_text"
    assert all(not parameter.is_meta for parameter in loaded.parameters())
    assert any(isinstance(m, ExpertsNbit) for m in loaded.model.layers[0].mlp.experts.modules())
    for name in ("gate_up_proj", "down_proj"):
        assert torch.equal(getattr(loaded.model.layers[1].mlp.experts, name), getattr(model.model.layers[1].mlp.experts, name).float())


@pytest.mark.parametrize("fault", ("unknown_text", "missing_expert", "mixed_roots"))
def test_bad_text_mapping_refuses_before_repacking(checkpoints, fault):
    model, composite, _ = checkpoints
    keys, _ = safetensors_reader(str(composite))
    if fault == "unknown_text":
        keys.append("model.language_model.layers.0.mlp.unadjudicated.weight")
    elif fault == "missing_expert":
        keys.remove("model.language_model.layers.0.mlp.experts.down_proj")
    else:
        keys.append("model.layers.0.mlp.experts.gate_up_proj")
    with pytest.raises(MoEConventionError):
        plan_moe_checkpoint(keys, model, model.config.model_type, skip_extra_layers=True)


def _cpu_residency_tree(config):
    root = torch.nn.Module()
    root.config = config
    states = []
    for layer in range(config.num_hidden_layers):
        node = root
        for part in f"model.layers.{layer}.mlp.experts".split("."):
            if part not in node._modules:
                node.add_module(part, torch.nn.Module())
            node = node._modules[part]
        state = SimpleNamespace(_all_hot=lambda: True)
        for name in ("h_gu_p", "h_gu_a", "h_dn_p", "h_dn_a"):
            setattr(state, name, torch.zeros(3, dtype=torch.uint8))
        node._hot_residency = state
        states.append(state)
    return root, states


def test_actual_cpu_repack_is_bitwise_equal_to_plain_text(checkpoints, monkeypatch):
    pytest.importorskip("int4_b32")
    monkeypatch.setenv("E4B_INT4_KEEP_NF4", "1")
    model, composite, plain = checkpoints
    actual, a = _cpu_residency_tree(model.config)
    reference, b = _cpu_residency_tree(model.config)
    assert enable_serve_experts_int4(actual, str(composite), plan_model=model) == 2
    # The existing plain-text path explicitly uses the original native family
    # convention; no composite projection or text-type alias is needed there.
    assert enable_serve_experts_int4(reference, str(plain), model_type="qwen3_5_moe", plan_model=model) == 2
    for left, right in zip(a, b, strict=True):
        for role in ("gu", "dn"):
            x, y = left._int4_stores[role], right._int4_stores[role]
            assert (x["N"], x["K"]) == (y["N"], y["K"])
            for field in ("packed", "scales"):
                assert x[field].device.type == y[field].device.type == "cpu"
                assert torch.equal(x[field], y[field]), (role, field)



def test_upstream_prefix_conversion_preserves_native_tensors_and_packed_bytes(checkpoints, tmp_path, monkeypatch):
    """Execute upstream's rename and tensor conversion before actual CPU packing."""
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping

    pytest.importorskip("int4_b32")
    monkeypatch.setenv("E4B_INT4_KEEP_NF4", "1")
    model, composite, plain = checkpoints
    mapping = get_checkpoint_conversion_mapping("qwen3_5_moe_text")
    native = {}
    for source, tensor in load_file(str(composite / "model.safetensors")).items():
        if source != "lm_head.weight" and not source.startswith("model.language_model."):
            continue
        rename = next(c for c in mapping if type(c).__name__ == "PrefixChange")
        target, pattern = rename.rename_source_key(source)
        if pattern is not None:
            rename.add_tensor(target, source, pattern, tensor)
            converted = rename.convert(target, model=model, config=model.config)
            values = list(converted.values())
            assert len(values) == 1 and len(values[0]) == 1
            assert values[0][0] is tensor  # actual upstream conversion does not transpose/copy
            tensor = values[0][0]
        else:
            assert source == "lm_head.weight" and target == source
        assert torch.equal(tensor, model.state_dict()[target])
        for converter in mapping:
            assert converter.rename_source_key(target) == (target, None)
        native[target] = tensor
    assert set(native) == set(model.state_dict())
    converted_path = tmp_path / "upstream_native"
    converted_path.mkdir()
    save_file(native, str(converted_path / "model.safetensors"))
    model.config.save_pretrained(converted_path)
    trees = [_cpu_residency_tree(model.config) for _ in range(3)]
    for (tree, _), checkpoint in zip(trees, (composite, plain, converted_path), strict=True):
        assert enable_serve_experts_int4(tree, str(checkpoint), plan_model=model) == 2
    for layer in range(model.config.num_hidden_layers):
        for role in ("gu", "dn"):
            stores = [states[layer]._int4_stores[role] for _, states in trees]
            assert len({(x["N"], x["K"]) for x in stores}) == 1
            for field in ("packed", "scales"):
                assert all(torch.equal(stores[0][field], x[field]) for x in stores[1:])


@pytest.mark.parametrize("composite_root", (False, True))
def test_legacy_per_expert_checkpoint_refuses_before_packing(checkpoints, tmp_path, composite_root):
    """Legacy gate/up/down weights are real tensors, but are outside native support."""
    model, _, _ = checkpoints
    tensors = {}
    for key, value in model.state_dict().items():
        if composite_root and key.startswith("model."):
            key = "model.language_model." + key.removeprefix("model.")
        if key.endswith("experts.gate_up_proj"):
            prefix = key.removesuffix("gate_up_proj")
            for expert, stack in enumerate(value):
                for role, tensor in zip(("gate_proj", "up_proj"), stack.chunk(2, dim=0), strict=True):
                    tensors[f"{prefix}{expert}.{role}.weight"] = tensor.contiguous().clone()
        elif key.endswith("experts.down_proj"):
            prefix = key.removesuffix("down_proj")
            for expert, tensor in enumerate(value):
                tensors[f"{prefix}{expert}.down_proj.weight"] = tensor.contiguous().clone()
        else:
            tensors[key] = value.contiguous().clone()
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    save_file(tensors, str(legacy / "model.safetensors"))
    model.config.save_pretrained(legacy)
    keys, _ = safetensors_reader(str(legacy))
    with pytest.raises(MoEConventionError, match="do not map|missing|unclaimed|unknown"):
        plan_moe_checkpoint(keys, model, model.config.model_type, skip_extra_layers=True)
    # Planning refusal above always executes, even without the optional packer.
    pytest.importorskip("int4_b32")
    tree, states = _cpu_residency_tree(model.config)
    with pytest.raises(MoEConventionError, match="do not map|missing|unclaimed|unknown"):
        enable_serve_experts_int4(tree, str(legacy), plan_model=model)
    assert all(not hasattr(state, "_int4_stores") for state in states)
