"""experts4bit_qlora.adapter_peft: the PEFT layout SGLang's MoE LoRA loader reads, checked against the trained forward.

The parity test runs ``ExpertsLoRA``'s own forward (a real NF4 stack quantized on the CPU, nonzero adapters) and
recomputes the expert output from the EXPORTED tensors, read back from disk, the way SGLang v0.5.23's loader
normalizes them. A stacked gate_up ``lora_A`` is repeated to ``2r`` (``srt/lora/lora.py:490-498``), and each of the
two output slices uses its own ``r`` rows (stacked multiply 2, ``srt/lora/utils.py:342-343``), gate rows first. The
base weights on both sides are e4b's own dequantization, so the comparison isolates the adapter mapping. It checks the
export against SGLang's documented layout; a served check against SGLang itself needs a GPU and is not part of this.
"""
import hashlib
import importlib
import json
import os

import pytest

#: under CI (GitHub Actions sets CI=true) every dependency is a HARD import, so a green check proves these tests ran
IN_CI = os.environ.get("CI") == "true"


def _need(name):
    return importlib.import_module(name) if IN_CI else pytest.importorskip(name)


torch = _need("torch")
_need("bitsandbytes")
st = _need("safetensors.torch")

E, H, INTER, R, ALPHA = 4, 128, 64, 2, 4
REV = "c" * 40
STACK = "model.layers.0.mlp.experts"
ATTN_LAYER = "model.layers.0.self_attn"
#: projection -> output width (k and v narrower, as under grouped-query attention)
ATTN_OUT = {"q_proj": H, "k_proj": 64, "v_proj": 64}


def _experts():
    """A real ``ExpertsLoRA`` over an NF4 stack quantized on the CPU, with nonzero adapters (B is zero at init)."""
    from experts4bit_qlora._vendor.experts import Experts4bit
    from experts4bit_qlora.lora import ExpertsLoRA
    g = torch.Generator().manual_seed(0)
    gu = torch.randn(E, 2 * INTER, H, generator=g) * 0.05
    dn = torch.randn(E, H, INTER, generator=g) * 0.05
    lora = ExpertsLoRA(Experts4bit.from_float(gu, dn, quant_type="nf4", blocksize=64), r=R, alpha=ALPHA,
                       dtype=torch.float32)
    with torch.no_grad():
        for p in (lora.gate_up_lora_A, lora.gate_up_lora_B, lora.down_lora_A, lora.down_lora_B):
            p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    return lora.eval()


def _write_adapter(root, tensors, *, revision=REV, extra=None):
    """An adapter directory in the shape Loggetta's ``--adapter-out`` writes (``backend-adapter/1``)."""
    root.mkdir(parents=True, exist_ok=True)
    st.save_file(tensors, str(root / "adapter.safetensors"))
    data = (root / "adapter.safetensors").read_bytes()
    manifest = {"schema": "backend-adapter/1", "backend": "experts4bit", "weights": "adapter.safetensors",
                "files": {"adapter.safetensors": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}},
                "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()},
                "setup": {"r": R, "alpha": ALPHA}, "model": {"model": "org/tiny-moe"}}
    if revision:
        manifest["resolved_model_revision"] = revision
    manifest.update(extra or {})
    (root / "adapter_manifest.json").write_bytes(json.dumps(manifest).encode())
    return root


def _trained_tensors(lora):
    g = torch.Generator().manual_seed(1)
    t = {f"{STACK}.{n}": getattr(lora, n).detach().clone()
         for n in ("gate_up_lora_A", "gate_up_lora_B", "down_lora_A", "down_lora_B")}
    for proj, out in ATTN_OUT.items():
        t[f"{ATTN_LAYER}.{proj}.lora_A"] = torch.randn(R, H, generator=g)
        t[f"{ATTN_LAYER}.{proj}.lora_B"] = torch.randn(out, R, generator=g)
    return t


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    from experts4bit_qlora.adapter_peft import export_peft
    lora = _experts()
    tensors = _trained_tensors(lora)
    root = tmp_path_factory.mktemp("peft")
    record = export_peft(_write_adapter(root / "adapter", tensors), root / "out")
    peft = st.load_file(str(root / "out" / "adapter_model.safetensors"))
    config = json.loads((root / "out" / "adapter_config.json").read_text(encoding="utf-8"))
    return lora, tensors, peft, config, record, root


def _sglang_reference(stack, peft, config, x, idx, w):
    """The expert output from the exported tensors, as SGLang's loader normalizes them (see the module docstring)."""
    import torch.nn.functional as F

    from experts4bit_qlora.absmax_dq import expert_absmax_rows
    p = f"base_model.model.{STACK}"
    a = peft[f"{p}.gate_up_proj.lora_A.weight"].repeat(1, 2, 1)       # lora.py:497, [E, 2r, H]
    b = peft[f"{p}.gate_up_proj.lora_B.weight"]                       # kept as stacked, [E, 2*inter, r]
    ad, bd = peft[f"{p}.down_proj.lora_A.weight"], peft[f"{p}.down_proj.lora_B.weight"]
    s, r, inter = config["lora_alpha"] / config["r"], config["r"], ad.shape[2]
    gu_abs, dn_abs = expert_absmax_rows(stack, "gate_up"), expert_absmax_rows(stack, "down")
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        xt = x[t:t + 1]
        for j in range(idx.shape[1]):
            e = int(idx[t, j])
            w13 = stack._dequantize_expert(stack.gate_up_proj, gu_abs, stack._gate_up_shape, e, torch.float32)
            w2 = stack._dequantize_expert(stack.down_proj, dn_abs, stack._down_shape, e, torch.float32)
            proj = xt @ w13.T
            gate = proj[:, :inter] + s * (xt @ a[e, :r].T) @ b[e, :inter].T
            up = proj[:, inter:] + s * (xt @ a[e, r:].T) @ b[e, inter:].T
            h = F.silu(gate) * up
            out[t] += w[t, j] * (h @ w2.T + s * (h @ ad[e].T) @ bd[e].T)[0]
    return out


def _inputs():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(5, H, generator=g)
    idx = torch.stack([torch.randperm(E, generator=g)[:2] for _ in range(5)])
    w = torch.softmax(torch.randn(5, 2, generator=g), dim=-1)
    return x, idx, w


def test_the_keys_shapes_values_and_config_are_the_trained_adapters(exported):
    lora, tensors, peft, config, record, _ = exported
    p = "base_model.model."
    want = {f"{p}{STACK}.gate_up_proj.lora_A.weight": f"{STACK}.gate_up_lora_A",
            f"{p}{STACK}.gate_up_proj.lora_B.weight": f"{STACK}.gate_up_lora_B",
            f"{p}{STACK}.down_proj.lora_A.weight": f"{STACK}.down_lora_A",
            f"{p}{STACK}.down_proj.lora_B.weight": f"{STACK}.down_lora_B"}
    for proj in ATTN_OUT:
        for ab in "AB":
            want[f"{p}{ATTN_LAYER}.{proj}.lora_{ab}.weight"] = f"{ATTN_LAYER}.{proj}.lora_{ab}"
    assert set(peft) == set(want)
    for key, src in want.items():
        assert peft[key].dtype == tensors[src].dtype and torch.equal(peft[key], tensors[src]), key
    assert list(peft[f"{p}{STACK}.gate_up_proj.lora_A.weight"].shape) == [E, R, H]      # never pre-repeated to 2r
    assert config["r"] == R and config["lora_alpha"] == ALPHA and config["revision"] == REV
    assert config["base_model_name_or_path"] == "org/tiny-moe" and config["peft_type"] == "LORA"
    assert config["use_rslora"] is False
    assert config["target_modules"] == ["down_proj", "gate_up_proj", "k_proj", "q_proj", "v_proj"]
    assert record["expert_stacks"] == 1 and record["attention_modules"] == 3 and record["scale"] == ALPHA / R
    root = exported[5]
    assert record["peft_weights_sha256"] == hashlib.sha256((root / "out" / "adapter_model.safetensors").read_bytes()
                                                           ).hexdigest()
    assert record["source_adapter_sha256"] == hashlib.sha256((root / "adapter" / "adapter.safetensors").read_bytes()
                                                             ).hexdigest()


def test_sglang_reads_the_expert_stacks_as_per_expert_never_shared_outer(exported):
    """``_detect_shared_outer_loras`` (``srt/lora/lora_manager.py:764-798``): a 3-D gate_up lora_A is shared-outer
    exactly when its first dimension is 1."""
    _, _, peft, _, _, _ = exported
    gate_up_a = [t for k, t in peft.items() if "gate_up_proj" in k and "lora_A" in k]
    assert gate_up_a and all(t.dim() == 3 and t.shape[0] == E != 1 for t in gate_up_a)


def test_the_exported_adapter_reproduces_the_trained_expert_output(exported):
    lora, _, peft, config, _, _ = exported
    x, idx, w = _inputs()
    with torch.no_grad():
        trained = lora(x, idx, w)
        served = _sglang_reference(lora.base, peft, config, x, idx, w)
    assert torch.allclose(trained, served, rtol=1e-4, atol=1e-5), (trained - served).abs().max()


def _swap_gate_up(peft, config):
    k = f"base_model.model.{STACK}.gate_up_proj.lora_B.weight"
    peft[k] = torch.cat([peft[k][:, INTER:], peft[k][:, :INTER]], dim=1)


def _roll_experts(peft, config):
    for k in list(peft):
        if ".experts." in k:
            peft[k] = peft[k].roll(1, dims=0)


def _double_alpha(peft, config):
    config["lora_alpha"] = 2 * config["lora_alpha"]


def _drop_down_delta(peft, config):
    k = f"base_model.model.{STACK}.down_proj.lora_A.weight"
    peft[k] = torch.zeros_like(peft[k])


@pytest.mark.parametrize("mutate", [_swap_gate_up, _roll_experts, _double_alpha, _drop_down_delta])
def test_the_parity_check_catches_each_wrong_layout(exported, mutate):
    """Each mutant is a layout an exporter could get wrong. The parity comparison must reject every one, so a green
    parity test is evidence about the mapping, not a comparison that cannot fail."""
    lora, _, peft, config, _, _ = exported
    peft, config = dict(peft), dict(config)
    mutate(peft, config)
    x, idx, w = _inputs()
    with torch.no_grad():
        trained = lora(x, idx, w)
        served = _sglang_reference(lora.base, peft, config, x, idx, w)
    assert not torch.allclose(trained, served, rtol=1e-4, atol=1e-5)


def _refusal(tmp_path, tensors, **kw):
    from experts4bit_qlora.adapter_peft import ExportRefused, export_peft
    with pytest.raises(ExportRefused) as e:
        export_peft(_write_adapter(tmp_path / "adapter", tensors, **kw), tmp_path / "out")
    assert not (tmp_path / "out").exists()
    return e.value.reason


def test_refuses_an_adapter_with_no_base_revision(exported, tmp_path):
    assert "resolved_model_revision" in _refusal(tmp_path, exported[1], revision=None)


def test_refuses_an_incomplete_expert_stack(exported, tmp_path):
    t = {k: v for k, v in exported[1].items() if not k.endswith("down_lora_B")}
    assert "missing ['down_B']" in _refusal(tmp_path, t)


def test_refuses_a_non_gated_stack(exported, tmp_path):
    t = dict(exported[1])
    t[f"{STACK}.gate_up_lora_B"] = t[f"{STACK}.gate_up_lora_B"][:, :INTER].contiguous()
    assert "non-gated" in _refusal(tmp_path, t)


def test_refuses_a_one_expert_stack_sglang_would_read_as_shared_outer(exported, tmp_path):
    t = {k: (v[:1].contiguous() if ".experts." in k else v) for k, v in exported[1].items()}
    assert "shared-outer" in _refusal(tmp_path, t)


def test_refuses_a_stack_whose_factors_disagree_on_the_rank(exported, tmp_path):
    t = dict(exported[1])
    t[f"{STACK}.down_lora_A"] = torch.zeros(E, R + 1, INTER)
    t[f"{STACK}.down_lora_B"] = torch.zeros(E, H, R + 1)
    assert f"of rank {R}" in _refusal(tmp_path, t)


def test_refuses_a_stack_whose_down_output_is_not_its_hidden_size(exported, tmp_path):
    t = dict(exported[1])
    t[f"{STACK}.down_lora_B"] = torch.zeros(E, 2 * H, R)
    assert f"of rank {R}" in _refusal(tmp_path, t)


@pytest.mark.parametrize("drop", [("v_proj",), ("q_proj",), ("q_proj", "v_proj")])
def test_refuses_an_attention_layer_without_both_q_and_v(exported, tmp_path, drop):
    """SGLang reads v whenever q is present and stacks nothing for a layer without q (srt/lora/lora.py:281-306)."""
    t = {k: v for k, v in exported[1].items() if not any(f".{d}." in k for d in drop)}
    assert "needs q and v together" in _refusal(tmp_path, t)


def test_exports_q_and_v_without_k(exported, tmp_path):
    """SGLang zero-fills a missing k (srt/lora/lora.py:289-294), so q + v alone is servable."""
    from experts4bit_qlora.adapter_peft import export_peft
    t = {k: v for k, v in exported[1].items() if ".k_proj." not in k}
    record = export_peft(_write_adapter(tmp_path / "adapter", t), tmp_path / "out")
    assert record["attention_modules"] == 2


def test_refuses_an_adapter_with_no_lora_tensors(tmp_path):
    assert "no LoRA tensors" in _refusal(tmp_path, {})


@pytest.mark.parametrize("applied", [2.0, {"attention": ALPHA / R, "experts": True}, {"experts": "2.0"}])
def test_refuses_a_malformed_scale_record(exported, tmp_path, applied):
    assert "lora_scale_applied" in _refusal(tmp_path, exported[1], extra={"lora_scale_applied": applied})


def test_a_scope_recorded_as_none_holds_no_modules_and_exports(exported, tmp_path):
    from experts4bit_qlora.adapter_peft import export_peft
    root = _write_adapter(tmp_path / "adapter", exported[1], extra={"lora_scale_applied": {"attention": None,
                                                                                         "experts": ALPHA / R}})
    assert export_peft(root, tmp_path / "out")["expert_stacks"] == 1


def test_refuses_tensors_that_are_neither_attention_nor_expert_lora(exported, tmp_path):
    t = dict(exported[1])
    t["model.layers.0.mlp.gate.lora_A"] = torch.zeros(R, H)
    assert "neither attention nor expert" in _refusal(tmp_path, t)


def test_refuses_a_recorded_scale_sglang_cannot_apply(exported, tmp_path):
    reason = _refusal(tmp_path, exported[1], extra={"lora_scale_applied": {"attention": ALPHA / R, "experts": 1.0}})
    assert "lora_scale_applied.experts" in reason


def test_refuses_a_directory_that_does_not_verify(exported, tmp_path):
    from experts4bit_qlora.adapter_dir import AdapterRefused
    from experts4bit_qlora.adapter_peft import export_peft
    root = _write_adapter(tmp_path / "adapter", exported[1])
    (root / "adapter.safetensors").write_bytes((root / "adapter.safetensors").read_bytes() + b"x")
    with pytest.raises(AdapterRefused) as e:
        export_peft(root, tmp_path / "out")
    assert e.value.code == 7


def test_never_overwrites_a_non_empty_directory(exported, tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "keep.txt").write_bytes(b"keep")
    from experts4bit_qlora.adapter_peft import ExportRefused, export_peft
    with pytest.raises(ExportRefused, match="never overwrites"):
        export_peft(_write_adapter(tmp_path / "adapter", exported[1]), tmp_path / "out")
    assert (tmp_path / "out" / "keep.txt").read_bytes() == b"keep"


def test_the_cli_exits_0_on_export_and_2_on_a_refusal(exported, tmp_path, capsys):
    from experts4bit_qlora.adapter_peft import main
    src = _write_adapter(tmp_path / "adapter", exported[1])
    assert main([str(src), str(tmp_path / "out")]) == 0
    assert json.loads(capsys.readouterr().out)["expert_stacks"] == 1
    assert main([str(src), str(tmp_path / "out")]) == 2                # the second export would overwrite
    assert "never overwrites" in capsys.readouterr().err
    assert main([str(src)]) == 2
