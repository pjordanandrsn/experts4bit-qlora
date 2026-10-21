"""PA-S: backend-adapter/2 (experts4bit_qlora/adapter_schema.py, docs/ADAPTER_PORTABILITY.md section 6).

CPU only. bitsandbytes' 4-bit quantization runs on the CPU here, so the canonical fingerprint is checked on real
``Linear4bit`` and ``Experts4bit`` storage: the same base stored nested and expanded fingerprints alike, and a
double-quantized expert absmax fingerprints differently from an fp32 one (F1).
"""
import hashlib
import json

import pytest

torch = pytest.importorskip("torch")
bnb = pytest.importorskip("bitsandbytes")
pytest.importorskip("safetensors")
tr = pytest.importorskip("transformers")

from experts4bit_qlora import adapter_dir as AD  # noqa: E402
from experts4bit_qlora import adapter_schema as S  # noqa: E402

LAYER = "model.layers.0.self_attn"


def _model(seed=0):
    cfg = tr.Qwen3MoeConfig(hidden_size=64, intermediate_size=64, moe_intermediate_size=32, num_experts=4,
                            num_experts_per_tok=2, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=16, vocab_size=128, max_position_embeddings=64)
    torch.manual_seed(seed)
    return tr.Qwen3MoeForCausalLM(cfg).to(torch.bfloat16).eval()


# ------------------------------------------------------------------------------------------------- the fingerprint

def test_the_attention_fingerprint_is_pinned_to_qfp_v1_and_deterministic():
    m = _model()
    a, b = S.quant_fingerprint(m, "attention"), S.quant_fingerprint(m, "attention")
    assert a["scheme_id"] == "qfp-v1" and a["scope"] == "attention" and a["modules"] == 4
    assert a["sha256"] == b["sha256"] and len(a["sha256"]) == 64
    with torch.no_grad():
        m.get_submodule(f"{LAYER}.k_proj").weight.view(torch.int16)[0, 0] ^= 1      # one bit of one weight
    assert S.quant_fingerprint(m, "attention")["sha256"] != a["sha256"]


def test_qfp_v1_is_the_documented_canonicalization():
    """Recompute the attention fingerprint from the docstring's rule, independently of the implementation."""
    m = _model()
    lines = []
    for p in ("k_proj", "o_proj", "q_proj", "v_proj"):
        w = m.get_submodule(f"{LAYER}.{p}").weight.detach().contiguous()
        data = w.view(torch.uint8).numpy().tobytes()
        e = hashlib.sha256(b"").hexdigest()
        lines.append("\t".join((f"{LAYER}.{p}", "dense:bfloat16", "x".join(map(str, w.shape)), "0",
                                hashlib.sha256(data).hexdigest(), e, e)))
    expect = hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode()).hexdigest()
    assert S.quant_fingerprint(m, "attention")["sha256"] == expect


def _cpu_quantized(m, compress_statistics):
    """Quantize every attention projection to NF4 on the CPU (bitsandbytes quantizes a Params4bit when it moves)."""
    from bitsandbytes.functional import quantize_4bit

    attn = m.get_submodule(LAYER)
    for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
        lin = getattr(attn, p)
        packed, qs = quantize_4bit(lin.weight.data, blocksize=64, quant_type="nf4",
                                   compress_statistics=compress_statistics)
        q = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=False, compute_dtype=torch.bfloat16,
                              quant_type="nf4")
        q.weight = bnb.nn.Params4bit(packed, requires_grad=False, quant_state=qs, quant_type="nf4",
                                     bnb_quantized=True)
        setattr(attn, p, q)
    return m


def test_the_same_nf4_base_fingerprints_alike_nested_or_expanded():
    """Training's fused q/k/v stores the absmax expanded to fp32; the unfused path keeps it nested. Both dequantize the
    same, so qfp-v1 hashes them the same."""
    from bitsandbytes.functional import QuantState, dequantize_blockwise

    nested = _cpu_quantized(_model(), compress_statistics=True)
    expanded = _cpu_quantized(_model(), compress_statistics=True)
    for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
        w = expanded.get_submodule(f"{LAYER}.{p}").weight
        qs = w.quant_state
        a = dequantize_blockwise(qs.absmax, qs.state2) + qs.offset
        w.quant_state = QuantState(absmax=a.float(), shape=qs.shape, code=qs.code, blocksize=qs.blocksize,
                                   quant_type=qs.quant_type, dtype=qs.dtype)
        assert not w.quant_state.nested
    fn, fe = S.quant_fingerprint(nested, "attention"), S.quant_fingerprint(expanded, "attention")
    assert fn["sha256"] == fe["sha256"] and fn["modules"] == 4


def test_a_nested_absmax_is_not_the_fp32_one():
    """The F1 case in miniature: the same weights with fp32 statistics and with nested (lossy) statistics dequantize
    differently, and qfp-v1 tells them apart."""
    fp32 = _cpu_quantized(_model(), compress_statistics=False)
    nested = _cpu_quantized(_model(), compress_statistics=True)
    assert S.quant_fingerprint(fp32, "attention")["sha256"] != S.quant_fingerprint(nested, "attention")["sha256"]


def test_a_lora_wrapper_fingerprints_under_its_projection_name():
    from experts4bit_qlora.lora import LoRALinear

    m = _model()
    plain = S.quant_fingerprint(m, "attention")
    attn = m.get_submodule(LAYER)
    for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(attn, p, LoRALinear(getattr(attn, p), r=2, alpha=4))
    assert S.quant_fingerprint(m, "attention")["sha256"] == plain["sha256"]      # the base, under the same names
    assert S.lora_scale_applied(m) == {"attention": 2.0, "experts": None}


def test_the_expert_fingerprint_reads_the_canonical_absmax_and_strips_the_wrapper():
    from experts4bit_qlora._vendor.experts import Experts4bit
    from experts4bit_qlora.absmax_dq import compress_expert_absmax_
    from experts4bit_qlora.lora import ExpertsLoRA

    torch.manual_seed(3)
    E, H, inter = 4, 64, 64
    w_gu, w_dn = torch.randn(E, 2 * inter, H, dtype=torch.bfloat16), torch.randn(E, H, inter, dtype=torch.bfloat16)

    def model_with_stack(wrap):
        m = torch.nn.Module()
        stack = Experts4bit.from_float(w_gu, w_dn, quant_type="nf4", blocksize=64)
        m.experts = ExpertsLoRA(stack, r=2, alpha=4) if wrap else stack
        return m

    bare, wrapped = model_with_stack(False), model_with_stack(True)
    fb, fw = S.quant_fingerprint(bare, "experts"), S.quant_fingerprint(wrapped, "experts")
    assert fb["sha256"] == fw["sha256"] and fb["modules"] == 2                     # gate_up and down, named alike
    compress_expert_absmax_(wrapped)                                               # training's default (F1)
    assert S.quant_fingerprint(wrapped, "experts")["sha256"] != fb["sha256"]


def test_the_expert_fingerprint_is_a_recipe_check_not_a_content_proof(monkeypatch):
    """qfp-v1 samples the packed bytes at both ends of each stack: a byte between them is not covered (the documented
    limit), while a byte inside the sample, or any absmax value, is."""
    from experts4bit_qlora._vendor.experts import Experts4bit

    torch.manual_seed(4)
    m = torch.nn.Module()
    m.experts = Experts4bit.from_float(torch.randn(4, 128, 64, dtype=torch.bfloat16),
                                       torch.randn(4, 64, 64, dtype=torch.bfloat16), quant_type="nf4", blocksize=64)
    monkeypatch.setattr(S, "EXPERT_SUBSET_BYTES", 1024)
    base = S.quant_fingerprint(m, "experts")["sha256"]
    flat = m.experts.gate_up_proj.data.view(-1)
    with torch.no_grad():
        flat[flat.numel() // 2] ^= 0x11                          # the middle of the stack: outside the sample
    assert S.quant_fingerprint(m, "experts")["sha256"] == base
    with torch.no_grad():
        flat[3] ^= 0x11                                          # inside the first 1024 bytes
    assert S.quant_fingerprint(m, "experts")["sha256"] != base


def _offloaded_by_hand(m):
    """Evict an ExpertsLoRA's stack the way engines/offload does: the base's tensors become 0-element placeholders and
    the real ones live in ``handle.home`` on the CPU."""
    import types

    lora = m.experts
    base = lora.base
    home = {n: getattr(base, n).detach().clone() for n in ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax")}
    for n in ("gate_up_proj", "down_proj"):
        base._parameters[n].data = torch.empty(0, dtype=home[n].dtype)
    for n in ("gate_up_absmax", "down_absmax"):
        base._buffers[n] = torch.empty(0, dtype=home[n].dtype)
    lora._offload = types.SimpleNamespace(base=base, home=home)
    return m


def _lora_stack(seed=5):
    from experts4bit_qlora._vendor.experts import Experts4bit
    from experts4bit_qlora.lora import ExpertsLoRA

    torch.manual_seed(seed)
    m = torch.nn.Module()
    m.experts = ExpertsLoRA(Experts4bit.from_float(torch.randn(4, 128, 64, dtype=torch.bfloat16),
                                                   torch.randn(4, 64, 64, dtype=torch.bfloat16), quant_type="nf4",
                                                   blocksize=64), r=2, alpha=4)
    return m


def test_an_offloaded_stack_fingerprints_as_the_resident_one():
    """cerin-maintainer's case: offload (CHOOSING's mode on 12 and 24 GB cards) evicts each stack to a host home; the
    canonical form is the same wherever the bytes live."""
    resident = S.quant_fingerprint(_lora_stack(), "experts")
    evicted = S.quant_fingerprint(_offloaded_by_hand(_lora_stack()), "experts", source="offload_homes")
    assert evicted["sha256"] == resident["sha256"] and evicted["modules"] == resident["modules"] == 2
    assert S.quant_fingerprint(_lora_stack(), "experts", source="offload_homes")["sha256"] == resident["sha256"]


def test_each_source_reads_where_it_says():
    with pytest.raises(ValueError, match="fingerprint it with source='offload_homes'"):
        S.quant_fingerprint(_offloaded_by_hand(_lora_stack()), "experts", source="resident")
    with pytest.raises(NotImplementedError, match="PA-1"):
        S.quant_fingerprint(_lora_stack(), "experts", source="arena")
    with pytest.raises(ValueError, match="source must be one of"):
        S.quant_fingerprint(_lora_stack(), "experts", source="disk")


def test_an_evicted_stack_with_no_host_copy_is_refused_by_name():
    m = _offloaded_by_hand(_lora_stack())
    del m.experts._offload.home["down_proj"]
    with pytest.raises(ValueError, match="expert stack experts is evicted and its host copy of down_proj is unavailable"):
        S.quant_fingerprint(m, "experts", source="offload_homes")
    m2 = _offloaded_by_hand(_lora_stack())
    del m2.experts._offload                                              # a placeholder and no handle at all
    with pytest.raises(ValueError, match="no offload handle"):
        S.quant_fingerprint(m2, "experts", source="offload_homes")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="REGISTERED for PA-4's box: real expert offload needs CUDA")
def test_real_offload_fingerprints_as_resident_on_cuda():
    """A PA-4 gate: on a CUDA box, a stack under engines.offload's real eviction fingerprints as the same stack resident
    on the device."""
    from experts4bit_qlora.engines.offload import enable_expert_offload

    resident = _lora_stack()
    resident.experts.to("cuda")
    off = _lora_stack()
    off.experts.to("cuda")
    enable_expert_offload(off.experts, torch.device("cuda"), pin=False)
    assert off.experts.base.gate_up_proj.numel() == 0                   # really evicted
    assert (S.quant_fingerprint(off, "experts", source="offload_homes")["sha256"]
            == S.quant_fingerprint(resident, "experts")["sha256"])


def test_the_canonical_line_functions_are_the_one_path():
    """quant_fingerprint builds every expert line through expert_line, so a reader that finds the same tensors elsewhere
    (an arena's rows) gets the same line."""
    from experts4bit_qlora.absmax_dq import expert_absmax_fp32

    m = _lora_stack()
    st = m.experts.base
    lines = [S.expert_line("experts", w, bits=st.bits, quant_type=st.quant_type, num_experts=st.num_experts,
                           shape=getattr(st, f"_{w}_shape"), blocksize=st.blocksize,
                           packed=getattr(st, f"{w}_proj"), absmax=expert_absmax_fp32(st, w), code=st.code)[0]
             for w in ("gate_up", "down")]
    expect = hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode()).hexdigest()
    assert S.quant_fingerprint(m, "experts")["sha256"] == expect


def test_an_empty_or_unknown_scope_is_refused():
    with pytest.raises(ValueError, match="scope must be one of"):
        S.quant_fingerprint(_model(), "router")
    with pytest.raises(ValueError, match="no experts module"):
        S.quant_fingerprint(torch.nn.Linear(4, 4), "experts")


def test_lora_scale_applied_refuses_disagreeing_modules():
    from experts4bit_qlora.lora import LoRALinear

    m = _model()
    attn = m.get_submodule(LAYER)
    attn.q_proj = LoRALinear(attn.q_proj, r=2, alpha=4)
    attn.k_proj = LoRALinear(attn.k_proj, r=2, alpha=8)
    with pytest.raises(ValueError, match="different scales"):
        S.lora_scale_applied(m)


# -------------------------------------------------------------------------------------- resolved training facts

def test_the_resolved_facts_come_from_the_runtime_records():
    m = _model()
    facts = S.resolved_training_facts(m, None, fast_stats={"absmax_dq": {"compressed": 48, "requested": True}},
                                      qkv_stats={"fused": 48, "refused": {}})
    assert facts["expert_absmax"]["form"] == "double-quantized" and facts["expert_absmax"]["stacks"] == 48
    assert facts["train_fused_qkv"] == 48
    assert facts["attn_quant"]["form"] == "bf16" and set(facts["versions"]) >= {"torch", "bitsandbytes"}
    off = S.resolved_training_facts(m, None, fast_stats={"absmax_dq": {"compressed": 0, "kept_fp32": "off"}},
                                    qkv_stats={"fused": 0})
    assert off["expert_absmax"]["form"] == "fp32"
    none = S.resolved_training_facts(m, None, fast_stats=None, qkv_stats=None)
    assert none["expert_absmax"]["form"] == "fp32" and none["train_fused_qkv"] == 0      # no stale record is read
    assert S.resolved_training_facts(m, None, fast_stats=None, qkv_stats={"fused": 7})["train_fused_qkv"] == 7
    nf4 = S.resolved_training_facts(_cpu_quantized(_model(), True), None, fast_stats=None, qkv_stats=None)
    assert nf4["attn_quant"] == {"form": "bnb4", "quant_type": "nf4", "blocksize": 64, "nested": True}


def test_a_mixed_attention_recipe_is_refused_not_summarized():
    m = _cpu_quantized(_model(), True)
    attn = m.get_submodule(LAYER)
    attn.o_proj = torch.nn.Linear(attn.o_proj.in_features, attn.o_proj.out_features, bias=False, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="quantized differently"):
        S.resolved_training_facts(m, None, fast_stats=None, qkv_stats=None)


def test_prepare_qlora_training_records_the_facts_after_enable_fast_train():
    """C1: the facts are added after enable_fast_train in prepare_qlora_training's body (a report built earlier would
    lack them), and before the trainable parameters are collected."""
    from pathlib import Path

    src = (Path(S.__file__).parent / "recipe.py").read_text(encoding="utf-8")
    body = src[src.index("def prepare_qlora_training("):]
    i_fast = body.index("n = enable_fast_train(model, dgrad=setup.dgrad)")
    i_facts = body.index("report.update(resolved_training_facts(model, setup, fast_stats=FAST_TRAIN_STATS")
    i_train = body.index("    trainable = []")
    assert i_fast < i_facts < i_train


# -------------------------------------------------------------------------------------------- the v2 verifier

def _v2(tmp_path, *, backend="experts4bit", scheme="qfp-v1", drop=None, scale=2.0, name="ad", fp_edit=None):
    from safetensors.torch import save_file

    m = _model()
    g = torch.Generator().manual_seed(1)
    t = {}
    for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
        lin = m.get_submodule(f"{LAYER}.{p}")
        t[f"{LAYER}.{p}.lora_A"] = torch.randn(2, lin.in_features, generator=g).to(torch.bfloat16)
        t[f"{LAYER}.{p}.lora_B"] = torch.randn(lin.out_features, 2, generator=g).to(torch.bfloat16)
    d = tmp_path / name
    d.mkdir()
    save_file(t, str(d / "adapter.safetensors"))
    raw = (d / "adapter.safetensors").read_bytes()
    man = {"schema": "backend-adapter/2", "backend": backend, "weights": "adapter.safetensors",
           "files": {"adapter.safetensors": {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}},
           "resolved_model_revision": "a" * 40,
           "setup": {"quant_type": "nf4", "blocksize": 64, "r": 2, "alpha": 4, "attn_4bit": False},
           "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in t.items()},
           "expert_absmax": {"form": "fp32", "stacks": 0}, "train_fused_qkv": 0,
           "attn_quant": {"form": "bf16"}, "versions": {"torch": "x"},
           "lora_scale_applied": {"attention": scale, "experts": None},
           "fingerprints": {"attention": {"scheme_id": scheme, "sha256": "b" * 64, "modules": 4, "seconds": 0.5},
                            "experts": {"scheme_id": scheme, "sha256": "c" * 64, "modules": 2, "seconds": 1.5}},
           "run_id": "r", "created_at": "2026-10-21T00:00:00Z"}
    if drop:
        man.pop(drop)
    if fp_edit:
        fp_edit(man["fingerprints"])
    (d / "adapter_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return d


def test_v2_verifies_and_keeps_the_expert_refusal_until_pa1(tmp_path):
    ad = AD.verify_adapter_dir(_v2(tmp_path))
    assert ad.manifest["schema"] == "backend-adapter/2"
    r = AD.servable_refusals(ad, served_revision="a" * 40, requested_revision=None,
                             arena_index={"snapshot": "/c/models--x--y/snapshots/" + "a" * 40,
                                          "quantizer": {"quant_type": "nf4", "blocksize": 64}},
                             levers={}, served_model="x/y")
    assert [c for c, _ in r] == [9] and "backend-adapter/2 records the expert absmax form (fp32)" in r[0][1]
    assert "PA-1" in AD.base_match(ad)["experts"]


@pytest.mark.parametrize("kw,needle", [
    ({"backend": "dense"}, "defined for backend=experts4bit only"),
    ({"scheme": "qfp-v9"}, "is not one this server knows"),
    ({"drop": "fingerprints"}, "lacks ['fingerprints']"),
    ({"drop": "expert_absmax"}, "lacks ['expert_absmax']"),
    ({"scale": 1.5}, "is not the setup's alpha/r"),
    ({"fp_edit": lambda f: f.pop("attention")}, "lacks the ['attention'] scope"),
    ({"fp_edit": lambda f: f.pop("experts")}, "lacks the ['experts'] scope"),
    ({"fp_edit": lambda f: f["experts"].update(modules=0)}, "is not a positive count"),
    ({"fp_edit": lambda f: f["attention"].pop("seconds")}, "is not a measured duration"),
    ({"fp_edit": lambda f: f["attention"].update(sha256="xyz")}, "is not a sha256"),
])
def test_v2_refuses_each_fault_by_name(tmp_path, kw, needle):
    with pytest.raises(AD.AdapterRefused) as e:
        AD.verify_adapter_dir(_v2(tmp_path, **kw))
    assert e.value.code == 7 and needle in str(e.value), str(e.value)


def test_the_v2_field_list_is_one_table_with_reasons():
    assert set(S.V2_FIELDS) == {"expert_absmax", "train_fused_qkv", "attn_quant", "versions", "lora_scale_applied",
                                "fingerprints", "run_id", "created_at"}
    assert all(len(why) > 20 for why in S.V2_FIELDS.values())
