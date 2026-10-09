# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Composite text projection and actual CPU int4 bytes must agree with plain text."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402

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
