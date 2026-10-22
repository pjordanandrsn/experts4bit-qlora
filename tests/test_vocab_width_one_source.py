# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""One source for the vocabulary width, on the plan's side and the run's.

The plan prices the LM loss from ``topology.vocab_size`` (``recipe._loss_chunk`` and ``_eval_loss_bytes``). The chunked loss's
run-time gate reads ``lm_head.out_features`` (``engines.chunked_lm_loss``: ``labels.numel() x lm_head.out_features x 4`` against
1 GiB). ``describe_moe`` now takes the width from the meta tree's LM head, and ``prepare_qlora_training`` refuses, by name, a
loaded model whose head is another width. A head resized after the topology was described (``resize_token_embeddings``) is the
case the refusal exists for: near the gate, the plan and the run would otherwise pick different loss branches without saying so.
"""
import pytest

torch = pytest.importorskip("torch")
tr = pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from experts4bit_qlora import loader  # noqa: E402
from experts4bit_qlora.arch import topology as topo_mod  # noqa: E402
from experts4bit_qlora.recipe import QLoRASetup, prepare_qlora_training, vocab_width_refusal  # noqa: E402

V, RESIZED = 97, 128


def _cfg():
    return tr.Qwen3MoeConfig(hidden_size=64, intermediate_size=128, moe_intermediate_size=64, num_experts=8, num_experts_per_tok=2,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=V,
                             max_position_embeddings=64, decoder_sparse_step=1, norm_topk_prob=True, tie_word_embeddings=False)


def test_describe_reads_the_width_from_the_lm_head():
    topo = topo_mod.describe_moe(_cfg())
    assert topo.vocab_size == V
    assert topo.provenance["vocab_size"].startswith("lm_head.out_features")


def test_describe_takes_the_head_width_when_it_differs_from_the_config(monkeypatch):
    """The tree describe_moe walks has an LM head 128 wide while the config says 97: the topology carries the head's width, the one the
    chunked loss's gate reads, never the config's."""
    build = tr.AutoModelForCausalLM.from_config

    def wide_head(config, *a, **k):
        tree = build(config, *a, **k)
        tree.lm_head = torch.nn.Linear(config.hidden_size, RESIZED, bias=False)
        return tree

    monkeypatch.setattr(tr.AutoModelForCausalLM, "from_config", wide_head)
    cfg = _cfg()
    assert cfg.vocab_size == V
    assert topo_mod.describe_moe(cfg).vocab_size == RESIZED


def test_a_resized_head_is_refused_by_name():
    cfg = _cfg()
    topo = topo_mod.describe_moe(cfg)
    torch.manual_seed(0)
    model = tr.Qwen3MoeForCausalLM(cfg)
    assert vocab_width_refusal(topo, model) is None
    model.resize_token_embeddings(RESIZED)
    assert model.get_output_embeddings().out_features == RESIZED and model.config.vocab_size == RESIZED
    why = vocab_width_refusal(topo, model)
    assert why is not None and f"{V}-wide vocabulary" in why and f"lm_head is {RESIZED} wide" in why, why


def test_prepare_refuses_a_head_of_another_width_before_any_engine(tmp_path, monkeypatch):
    """prepare_qlora_training describes the checkpoint, then loads it. A loaded head of another width than the one described
    raises ValueError naming both widths, before gradient checkpointing, LoRA or any engine is touched."""
    cfg = _cfg()
    cfg.save_pretrained(tmp_path)
    torch.manual_seed(0)
    model = tr.Qwen3MoeForCausalLM(cfg)
    model.resize_token_embeddings(RESIZED)
    touched = []
    monkeypatch.setattr(loader, "load_moe_4bit_streaming", lambda *a, **k: (model, model.config))
    monkeypatch.setattr(model, "gradient_checkpointing_enable", lambda *a, **k: touched.append("checkpointing"))
    setup = QLoRASetup(r=4, alpha=4, adapter_dtype="fp32", expert_kernel="reference", expert_residency="device", pin=False)
    with pytest.raises(ValueError, match=f"lm_head is {RESIZED} wide"):
        prepare_qlora_training(str(tmp_path), setup, device="cpu")
    assert touched == []


def test_the_plan_prices_stock_when_the_chunked_loss_refuses_the_head(monkeypatch):
    """chunked_lm_loss_refusal refuses an lm_head whose width differs from the config's vocabulary (quietly under auto), so the run
    takes the stock loss. The plan must price that: with the chunked loss forced on (E4B_CHUNKED_LM_LOSS=1), a head 128 wide on a
    97-token vocabulary reads the stock loss, while the same model with a matching head reads the chunked one."""
    from experts4bit_qlora import recipe

    monkeypatch.setenv("E4B_CHUNKED_LM_LOSS", "1")
    setup = QLoRASetup(expert_kernel="grouped_nf4")
    matched = topo_mod.describe_moe(_cfg())
    assert recipe._head_refusal(matched) is None and recipe._loss_chunk(matched, setup, 2048, matched.vocab_size)
    build = tr.AutoModelForCausalLM.from_config

    def wide_head(config, *a, **k):
        tree = build(config, *a, **k)
        tree.lm_head = torch.nn.Linear(config.hidden_size, RESIZED, bias=False)
        return tree

    monkeypatch.setattr(tr.AutoModelForCausalLM, "from_config", wide_head)
    wide = topo_mod.describe_moe(_cfg())
    assert (wide.vocab_size, wide.config_vocab_size) == (RESIZED, V)
    assert f"{RESIZED}-wide lm_head on a {V}-token vocabulary" in recipe._head_refusal(wide)
    assert recipe._loss_chunk(wide, setup, 2048, wide.vocab_size) is None

    def activations(topo):
        fp = recipe.estimate_qlora_footprint(topo, setup, tokens_per_microbatch=2048)
        return {i.name: i.detail for i in fp.items}

    assert "chunked LM loss" in activations(matched)["activations"]
    assert "chunked LM loss" not in activations(wide)["activations"]
    assert any("the model is not patched" in d for d in activations(wide).values())
