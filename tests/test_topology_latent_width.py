"""describe_moe reads a latent MoE's expert width from the stack itself, and nothing else moves (#1597).

The controls are the released ``config.json`` of every tested family at the revision its support row read, plus the
Qwen3-235B anchor, under ``fixtures/topology_controls`` (``SOURCES.json``: repo, revision, sha256, licence; each file
stays under its own licence, see ``NOTICE``). The latent subject is Nemotron-3-Super-120B-A12B's. Everything builds on
meta: no weights, no network.
"""
import hashlib
import json
import pathlib

import pytest
import torch

tr = pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from experts4bit_qlora.arch import topology  # noqa: E402
from experts4bit_qlora.arch.topology import describe_moe  # noqa: E402
from experts4bit_qlora.recipe import QLoRASetup, estimate_qlora_footprint  # noqa: E402

FIX = pathlib.Path(__file__).parent / "fixtures" / "topology_controls"
SOURCES = json.loads((FIX / "SOURCES.json").read_text())
CONTROLS = sorted(k for k, v in SOURCES.items() if v["role"] == "control")
SUPER = "nemotron-3-super-120b-a12b"


def _config(slug):
    return tr.AutoConfig.from_pretrained(FIX / slug)


def _legacy_widths(first, down, hidden_size, gated):
    """The rule before #1597: the input width is always ``config.hidden_size``."""
    E = int(down.shape[0])
    return hidden_size, (down.numel() // (E * hidden_size) if E and hidden_size else 0), False


def _as_json(topo):
    return json.dumps(topo.to_dict(), sort_keys=True, default=str)


def _tiny_latent():
    # hidden 512, latent 128, intermediate 256: the pre-#1597 rule reads (512, 64), still block-aligned, so both build
    return tr.NemotronHConfig(hidden_size=512, moe_latent_size=128, moe_intermediate_size=256, n_routed_experts=4,
                              num_experts_per_tok=2, layers_block_type=["moe", "attention", "moe"], num_hidden_layers=3,
                              num_attention_heads=4, num_key_value_heads=2, head_dim=128, vocab_size=192,
                              max_position_embeddings=64, moe_shared_expert_intermediate_size=128, mamba_num_heads=8,
                              mamba_head_dim=16, n_groups=2, ssm_state_size=16)


def test_the_fixtures_are_the_released_bytes():
    for slug, src in SOURCES.items():
        assert hashlib.sha256((FIX / slug / "config.json").read_bytes()).hexdigest() == src["sha256"], slug
        assert src["license"] and (FIX / src["license_file"]).is_file(), slug        # see fixtures/.../NOTICE


@pytest.mark.parametrize("slug", CONTROLS)
def test_every_tested_family_is_described_byte_for_byte_as_before(slug, monkeypatch):
    cfg = _config(slug)
    after = describe_moe(cfg)
    assert after.expert_stacks and "expert_width" not in after.provenance
    monkeypatch.setattr(topology, "_stack_widths", _legacy_widths)
    assert _as_json(after) == _as_json(describe_moe(cfg))


@pytest.mark.parametrize("slug", CONTROLS + [SUPER])
def test_the_widths_are_the_ones_from_float_reads(slug):
    """Experts4bit.from_float takes [E, out, in] stacks: input width gate_up.shape[2], intermediate down.shape[2].
    gpt-oss is stored [E, in, out] and built by its own loader path, so it only keeps its hidden_size here."""
    topo = describe_moe(_config(slug))
    for s in topo.expert_stacks:
        if topo.model_type == "gpt_oss":
            assert s.hidden == topo.hidden_size
        else:
            assert (s.hidden, s.intermediate) == (s.first_shape[2], s.down_shape[2]), (slug, s)


def test_a_latent_stack_is_described_at_its_own_width(monkeypatch):
    cfg = _config(SUPER)
    lm = getattr(cfg, "text_config", None) or cfg
    W, inter = lm.moe_latent_size, lm.moe_intermediate_size
    topo = describe_moe(cfg)
    assert (topo.hidden_size, W, inter, len(topo.expert_stacks)) == (4096, 1024, 2688, 40)
    assert {(s.hidden, s.intermediate, s.first_shape) for s in topo.expert_stacks} == {(W, inter, (512, inter, W))}
    assert "1024-wide latent" in topo.provenance["expert_width"]
    assert estimate_qlora_footprint(topo, QLoRASetup(expert_residency="host"), tokens_per_microbatch=2048).device_bytes
    monkeypatch.setattr(topology, "_stack_widths", _legacy_widths)
    legacy = describe_moe(cfg)
    assert {(s.hidden, s.intermediate) for s in legacy.expert_stacks} == {(4096, 672)}   # what #1597 read
    with pytest.raises(ValueError, match="intermediate_dim \\(672\\)"):                # ...which the estimate refused
        estimate_qlora_footprint(legacy, QLoRASetup(expert_residency="host"), tokens_per_microbatch=2048)


def test_the_estimate_prices_the_latent_modules_the_load_builds(monkeypatch):
    from experts4bit_qlora import Experts4bit
    from experts4bit_qlora.lora import ExpertsLoRA

    topo = describe_moe(_tiny_latent())
    n = len(topo.expert_stacks)
    assert n == 2 and {(s.hidden, s.intermediate) for s in topo.expert_stacks} == {(128, 256)}
    items = {i.name: i for i in estimate_qlora_footprint(topo, QLoRASetup(), tokens_per_microbatch=256).items}
    base = Experts4bit(4, 128, 256, has_gate=False, quant_type="nf4", blocksize=64)
    one = sum(t.numel() * t.element_size() for t in list(base.parameters()) + list(base.buffers()))
    lora = sum(p.numel() for k, p in ExpertsLoRA(base, r=8, alpha=16, dtype=torch.bfloat16).named_parameters() if "lora" in k)
    assert items["frozen expert stacks"].bytes == n * one
    assert items["expert LoRA adapters"].bytes == n * lora * 2
    assert "128 + 256 + 256" in items["activations"].detail        # H + first_out + I of the grouped-NF4 backward
    monkeypatch.setattr(topology, "_stack_widths", _legacy_widths)
    legacy = {i.name: i for i in estimate_qlora_footprint(describe_moe(_tiny_latent()), QLoRASetup(),
                                                          tokens_per_microbatch=256).items}
    assert legacy["expert LoRA adapters"].bytes != items["expert LoRA adapters"].bytes


def test_the_int4_stores_are_priced_at_the_latent_widths():
    from experts4bit_qlora.serve_recipe import _int4_expert_stores, _int4_split_k, int4_store_bytes

    topo = describe_moe(_tiny_latent())
    total, _, _ = _int4_expert_stores(topo)
    want = 0
    for _ in topo.expert_stacks:                        # up [E, I, W], down [E, W, I], top-2 partials
        want += int4_store_bytes(256, 128, 4) + int4_store_bytes(128, 256, 4)
        want += 4 * 2 * (_int4_split_k(256, 128)[0] * 256 + _int4_split_k(128, 256)[0] * 128)
    assert total == want


def test_a_stack_that_fits_no_rule_raises():
    first, down = torch.empty(4, 96, 64, device="meta"), torch.empty(4, 32, 48, device="meta")
    with pytest.raises(ValueError, match="neither hidden_size"):
        topology._stack_widths(first, down, 256, False)
    assert topology._stack_widths(torch.empty(4, 96, 32, device="meta"), torch.empty(4, 32, 48, device="meta"),
                                  256, True) == (32, 48, True)
    assert topology._stack_widths(first, torch.empty(4, 256, 48, device="meta"), 256, False) == (256, 48, False)
