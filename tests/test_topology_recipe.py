"""Config-only topology (arch/topology.describe_moe) and the QLoRA recipe's footprint (recipe.estimate_qlora_footprint).

The topology is checked against a REAL model built from the same config (parameter counts must agree exactly); the
footprint's derived items against the modules the load builds; refusals by structure, never by family name. The
GPU test builds a tiny checkpoint, runs prepare_qlora_training on both expert kernels and checks the trainable
count the estimate priced is the count that trains.
"""
import pytest
import torch

tr = pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from experts4bit_qlora.arch.topology import ROUTED_TOP_K_KEYS, describe_moe, routed_top_k  # noqa: E402
from experts4bit_qlora.loader import admission_refusal  # noqa: E402
from experts4bit_qlora.recipe import QLoRASetup, estimate_qlora_footprint, setup_refusals  # noqa: E402

H, INTER, E, K, L = 128, 64, 8, 2, 3


def _qwen3(**kw):
    base = dict(hidden_size=H, intermediate_size=256, moe_intermediate_size=INTER, num_experts=E, num_experts_per_tok=K,
                num_hidden_layers=L, num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=192,
                max_position_embeddings=64, decoder_sparse_step=1, norm_topk_prob=True, tie_word_embeddings=False)
    base.update(kw)
    return tr.Qwen3MoeConfig(**base)


def _olmoe():
    return tr.OlmoeConfig(hidden_size=H, intermediate_size=INTER, num_experts=E, num_experts_per_tok=K, num_hidden_layers=L,
                          num_attention_heads=4, num_key_value_heads=4, vocab_size=192, max_position_embeddings=64)


def _granite():
    return tr.GraniteMoeConfig(hidden_size=H, intermediate_size=INTER, num_local_experts=E, num_experts_per_tok=K,
                               num_hidden_layers=L, num_attention_heads=4, num_key_value_heads=4, vocab_size=192,
                               max_position_embeddings=64, tie_word_embeddings=True)


@pytest.mark.parametrize("key", ROUTED_TOP_K_KEYS)
def test_routed_top_k_every_spelling(key):
    from types import SimpleNamespace

    assert routed_top_k(SimpleNamespace(**{key: 6})) == 6
    assert routed_top_k(SimpleNamespace(text_config=SimpleNamespace(**{key: 3}))) == 3
    assert routed_top_k(SimpleNamespace(num_experts=128)) is None


def test_both_former_alias_lists_now_resolve_the_union():
    from types import SimpleNamespace

    from experts4bit_qlora.engines.int4_experts import _top_k
    from experts4bit_qlora.serve_paged import _routed_topk

    # each spelling was missing from one of the two lists before they were unified
    assert _routed_topk(SimpleNamespace(n_routed_experts_per_tok=4)) == 4
    assert _top_k(SimpleNamespace(config=SimpleNamespace(num_experts_per_token=4))) == 4
    with pytest.raises(ValueError):
        _routed_topk(SimpleNamespace(num_experts=8))


def _real_split(cfg):
    """(expert numel, non-expert numel) of the model transformers builds for real from ``cfg``."""
    torch.manual_seed(0)
    m = tr.AutoModelForCausalLM.from_config(cfg)
    expert = sum(p.numel() for n, p in m.named_parameters() if ".experts." in n)
    return expert, sum(p.numel() for p in m.parameters()) - expert


@pytest.mark.parametrize("make,model_type,convention", [(_qwen3, "qwen3_moe", "qwen2_moe"),
                                                       (_olmoe, "olmoe", "qwen2_moe"),
                                                       (_granite, "granitemoe", "granitemoe")])
def test_describe_moe_agrees_with_the_real_model(make, model_type, convention):
    cfg = make()
    topo = describe_moe(cfg)
    assert topo.loader_refusal is None and topo.model_type == model_type and topo.convention == convention
    assert topo.moe_layers == tuple(range(L)) and topo.n_experts == E and topo.top_k == K
    assert {(s.hidden, s.intermediate, s.first_name) for s in topo.expert_stacks} == {(H, INTER, "gate_up_proj")}
    expert, dense = _real_split(cfg)
    assert topo.expert_numel == expert
    assert topo.dense_numel == dense
    assert topo.attention.count == 4 * L and topo.attention.layers == L
    assert topo.to_dict()["expert_numel"] == expert            # JSON-ready


def test_dense_layers_are_found_by_structure():
    topo = describe_moe(_qwen3(mlp_only_layers=[0]))
    assert topo.moe_layers == (1, 2)


def test_tied_head_is_counted_once():
    topo = describe_moe(_granite())
    assert topo.tied_embeddings and topo.lm_head_numel == 0
    assert topo.dense_numel == _real_split(_granite())[1]


def test_a_refused_config_is_described_not_raised():
    cfg = tr.LlamaConfig(hidden_size=H, intermediate_size=INTER, num_hidden_layers=2, num_attention_heads=4, vocab_size=192)
    topo = describe_moe(cfg)
    assert topo.loader_refusal and "Unsupported model_type='llama'" in topo.loader_refusal
    assert topo.expert_stacks == () and admission_refusal(cfg) == topo.loader_refusal
    fp = estimate_qlora_footprint(topo, QLoRASetup(), tokens_per_microbatch=64)
    assert fp.items == () and fp.refusals


def test_a_biased_expert_stack_refuses_expert_adapters_by_structure():
    cfg = tr.GptOssConfig(hidden_size=H, intermediate_size=INTER, num_local_experts=E, num_experts_per_tok=K,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                          vocab_size=192, max_position_embeddings=64, layer_types=["full_attention"] * 2)
    topo = describe_moe(cfg)
    assert topo.expert_bias_tensors
    assert any("ExpertsLoRA cannot represent" in r for r in setup_refusals(topo, QLoRASetup()))
    assert not any("ExpertsLoRA" in r for r in setup_refusals(topo, QLoRASetup(train_experts=False,
                                                                                   expert_kernel="reference")))


def test_footprint_derived_items_are_the_built_modules():
    from experts4bit_qlora import Experts4bit
    from experts4bit_qlora.lora import ExpertsLoRA

    topo = describe_moe(_qwen3())
    fp = estimate_qlora_footprint(topo, QLoRASetup(), tokens_per_microbatch=256)
    items = {i.name: i for i in fp.items}
    base = Experts4bit(E, H, INTER, quant_type="nf4", blocksize=64)
    one = sum(t.numel() * t.element_size() for t in list(base.parameters()) + list(base.buffers()))
    assert items["frozen expert stacks"].bytes == L * one and items["frozen expert stacks"].basis == "derived"
    lora = sum(p.numel() for n, p in ExpertsLoRA(base, r=8, alpha=16, dtype=torch.bfloat16).named_parameters() if "lora" in n)
    assert items["expert LoRA adapters"].bytes == L * lora * 2
    attn = 8 * topo.attention.in_plus_out
    assert items["attention LoRA adapters"].bytes == attn * 2
    assert items["optimizer state (adamw)"].bytes == 2 * items["adapter gradients"].bytes == 2 * 2 * (L * lora + attn)
    eight = {i.name: i for i in estimate_qlora_footprint(topo, QLoRASetup(adapter_dtype="fp32"), tokens_per_microbatch=256,
                                                          optimizer="adamw_8bit").items}
    assert eight["optimizer state (adamw_8bit)"].bytes == int((2 + 8 / 256) * (L * lora + attn))
    assert items["activations"].basis == "heuristic"
    assert fp.unmodelled and fp.device_bytes == sum(i.bytes for i in fp.items if i.where == "device")


def test_host_residency_moves_the_slab_and_rounds_pinned_requests():
    topo = describe_moe(_qwen3())
    dev = estimate_qlora_footprint(topo, QLoRASetup(), tokens_per_microbatch=256)
    host = estimate_qlora_footprint(topo, QLoRASetup(expert_residency="host"), tokens_per_microbatch=256)
    slab = {i.name: i for i in dev.items}["frozen expert stacks"].bytes
    h = {i.name: i for i in host.items}
    assert h["frozen expert stacks, one layer staged"].bytes == slab // L
    assert h["frozen expert stacks, host homes (pinned)"].bytes >= slab - L * 16 * 4   # code buffers stay on device
    unpinned = estimate_qlora_footprint(topo, QLoRASetup(expert_residency="host", pin=False), tokens_per_microbatch=256)
    assert unpinned.host_bytes <= host.host_bytes
    assert host.device_bytes < dev.device_bytes
    assert h["expert staging, host to device per micro-batch"].bytes == 2 * slab       # forward + recompute
    assert not any(i.where == "link" for i in dev.items)


def test_estimate_grows_with_tokens_and_kept_layers():
    topo = describe_moe(_qwen3())
    a = estimate_qlora_footprint(topo, QLoRASetup(), tokens_per_microbatch=128).device_bytes
    b = estimate_qlora_footprint(topo, QLoRASetup(), tokens_per_microbatch=1024).device_bytes
    c = estimate_qlora_footprint(topo, QLoRASetup(keep_moe_layers=2), tokens_per_microbatch=1024).device_bytes
    assert a < b < c


def test_setup_refusals_are_words_not_exceptions():
    topo = describe_moe(_qwen3())
    assert setup_refusals(topo, QLoRASetup()) == ()
    assert setup_refusals(topo, QLoRASetup(quant_type="int8"))            # the grouped kernel reads NF4 only
    assert setup_refusals(topo, QLoRASetup(expert_residency="nvme"))
    assert setup_refusals(topo, QLoRASetup(train_experts=False, train_attention=False, expert_kernel="reference"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the loader quantizes on a CUDA device")
@pytest.mark.parametrize("kernel", ["reference", "grouped_nf4"])
def test_prepare_trains_exactly_what_the_estimate_priced(tmp_path, kernel):
    if kernel == "grouped_nf4":
        pytest.importorskip("nf4_qlora")
    from experts4bit_qlora.recipe import prepare_qlora_training

    torch.manual_seed(0)
    tr.AutoModelForCausalLM.from_config(_qwen3()).to(torch.bfloat16).save_pretrained(tmp_path)
    setup = QLoRASetup(expert_kernel=kernel)
    topo = describe_moe(str(tmp_path))
    fp = {i.name: i for i in estimate_qlora_footprint(topo, setup, tokens_per_microbatch=64).items}
    prep = prepare_qlora_training(str(tmp_path), setup)
    priced = (fp["expert LoRA adapters"].bytes + fp["attention LoRA adapters"].bytes) // 2
    assert prep.report["trainable_numel"] == priced
    assert all(p.requires_grad for p in prep.trainable)
    assert sum(p.numel() for p in prep.model.parameters() if p.requires_grad) == priced
    if kernel == "grouped_nf4":
        assert prep.report["fused_expert_modules"] == L
    ids = torch.randint(0, 192, (1, 64), device="cuda")
    loss = prep.model(input_ids=ids, labels=ids).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in prep.trainable)


def test_attention_that_cannot_be_wrapped_is_refused_not_silently_skipped():
    import dataclasses

    topo = describe_moe(_qwen3())
    undescribed = dataclasses.replace(topo, attention=None, provenance={**topo.provenance, "attention": "not described: x"})
    assert any("could not be described" in r for r in setup_refusals(undescribed, QLoRASetup()))
    assert not any("attention" in r for r in setup_refusals(undescribed, QLoRASetup(train_attention=False)))
    none = dataclasses.replace(topo, attention=dataclasses.replace(topo.attention, count=0, layers=0))
    assert any("no attention projection" in r for r in setup_refusals(none, QLoRASetup()))
    fp = estimate_qlora_footprint(none, QLoRASetup(train_attention=False), tokens_per_microbatch=64)
    assert any("mix tokens without q/k/v attention" in u for u in fp.unmodelled)


def test_logits_and_loss_are_priced_at_three_fp32_tensors_per_logit():
    """The loss branch of the activation item is three fp32 logits-sized tensors, 12 B per logit: an allocator replay
    of granite-3.1-3b-a800m training on an RTX A2000 found them live together at the loss peak (loggetta#49), where 10 B
    left the peak short by 2 x T x V bytes. A large vocabulary makes the loss the larger branch."""
    from experts4bit_qlora.recipe import LOGITS_LOSS_BYTES

    T, V = 256, 50_000
    topo = describe_moe(_qwen3(vocab_size=V))
    act = {i.name: i for i in estimate_qlora_footprint(topo, QLoRASetup(), tokens_per_microbatch=T).items}["activations"]
    assert LOGITS_LOSS_BYTES == 12
    assert act.bytes - L * T * H * 2 == T * V * 12            # saved layer inputs + the loss branch
    assert "three fp32 logits-sized tensors" in act.detail


def _activations(topo, T, **setup):
    return {i.name: i for i in estimate_qlora_footprint(topo, QLoRASetup(**setup), tokens_per_microbatch=T).items}[
        "activations"]


def test_the_loss_branch_is_chunked_exactly_when_the_run_chunks_it(monkeypatch):
    """``enable_fast_train`` routes a supported architecture's training loss through the chunked LM loss when the stock
    fp32 logits reach ``AUTO_MIN_LOGITS_BYTES`` (``auto``, the default); the estimate prices that branch with the
    engine's own ``chunked_loss_bytes``, and the stock branch everywhere else."""
    from experts4bit_qlora.engines.chunked_lm_loss import AUTO_MIN_LOGITS_BYTES, DEFAULT_CHUNK, chunked_loss_bytes
    from experts4bit_qlora.recipe import LOGITS_LOSS_BYTES

    V = 50_000
    topo = describe_moe(_qwen3(vocab_size=V, architectures=["Qwen3MoeForCausalLM"]))
    gate_T = -(-AUTO_MIN_LOGITS_BYTES // (V * 4))                    # the first T whose stock fp32 logits reach the gate
    boundaries = lambda T: L * T * H * 2                            # noqa: E731
    monkeypatch.delenv("E4B_CHUNKED_LM_LOSS", raising=False)          # auto, the default
    above, below = gate_T, gate_T - 1
    assert _activations(topo, above).bytes - boundaries(above) == chunked_loss_bytes(above, V, hidden=H)
    assert "chunked LM loss in %d-token chunks" % DEFAULT_CHUNK in _activations(topo, above).detail
    assert _activations(topo, below).bytes - boundaries(below) == below * V * LOGITS_LOSS_BYTES
    # the reference loop never calls enable_fast_train: stock, whatever T
    assert _activations(topo, above, expert_kernel="reference").bytes - boundaries(above) == above * V * LOGITS_LOSS_BYTES
    monkeypatch.setenv("E4B_CHUNKED_LM_LOSS", "0")                   # off
    assert _activations(topo, above).bytes - boundaries(above) == above * V * LOGITS_LOSS_BYTES
    monkeypatch.setenv("E4B_CHUNKED_LM_LOSS", "256")                 # on at any size, in 256-token chunks
    assert _activations(topo, 600).bytes - boundaries(600) == chunked_loss_bytes(600, V, hidden=H, chunk=256)


def test_an_architecture_outside_the_chunked_table_keeps_the_stock_loss(monkeypatch):
    from experts4bit_qlora.engines.chunked_lm_loss import AUTO_MIN_LOGITS_BYTES
    from experts4bit_qlora.recipe import LOGITS_LOSS_BYTES

    V = 50_000
    T = -(-AUTO_MIN_LOGITS_BYTES // (V * 4))
    monkeypatch.delenv("E4B_CHUNKED_LM_LOSS", raising=False)
    topo = describe_moe(_qwen3(vocab_size=V, architectures=["SomeOtherMoeForCausalLM"]))
    assert _activations(topo, T).bytes - L * T * H * 2 == T * V * LOGITS_LOSS_BYTES


def test_estimate_env_reports_the_switches_the_estimate_reads(monkeypatch):
    """Every switch ``estimate_env`` reports changes the estimate somewhere, and it reports this process's value."""
    from experts4bit_qlora import estimate_env
    from experts4bit_qlora.engines.chunked_lm_loss import AUTO_MIN_LOGITS_BYTES

    from experts4bit_qlora.engines.chunked_lm_loss import DEFAULT_CHUNK

    monkeypatch.delenv("E4B_CHUNKED_LM_LOSS", raising=False)
    unset = estimate_env()
    assert unset == {"E4B_CHUNKED_LM_LOSS": {"chunk": DEFAULT_CHUNK, "auto_gate_bytes": AUTO_MIN_LOGITS_BYTES}}
    for same in ("", "auto", "AUTO"):                      # the engine treats these as unset: no spurious difference
        monkeypatch.setenv("E4B_CHUNKED_LM_LOSS", same)
        assert estimate_env() == unset, same
    monkeypatch.setenv("E4B_CHUNKED_LM_LOSS", "1")        # every forward chunks: the gate is off
    assert estimate_env() == {"E4B_CHUNKED_LM_LOSS": {"chunk": DEFAULT_CHUNK, "auto_gate_bytes": None}}
    monkeypatch.setenv("E4B_CHUNKED_LM_LOSS", "0")
    assert estimate_env() == {"E4B_CHUNKED_LM_LOSS": {"chunk": None, "auto_gate_bytes": None}}
    V = 50_000
    topo = describe_moe(_qwen3(vocab_size=V, architectures=["Qwen3MoeForCausalLM"]))
    T = -(-AUTO_MIN_LOGITS_BYTES // (V * 4))
    for switch in estimate_env():
        monkeypatch.setenv(switch, "auto")
        on = _activations(topo, T).bytes
        monkeypatch.setenv(switch, "0")
        assert _activations(topo, T).bytes != on, switch
