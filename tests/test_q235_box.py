# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235's box script (``bench/q235/instrument/q235_box.py``) on the CPU: the mechanics of the registered step on a small 4-layer
Qwen3-MoE, on the reference path (the fused kernels need CUDA; the RTX A2000 rehearsal runs those).

- The steps (Amendment 2): at step 0 every LoRA-A gradient is exactly zero (B starts at zero) and every B gradient nonzero and every B
  changed; at step 1 every A gradient is nonzero and every A changed, and B's rules repeat; the frozen base's hashes equal after both
  steps; the in-process repeat recorded.
- The fresh-process repeat (``--repeat``) reproduces step 0's loss.
- Each planted error (Amendment 1: routing, epilogue) changes the planted layer's output and leaves the layers before it untouched,
  so an inert plant cannot pass for a band result.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("bitsandbytes")

ROOT = Path(__file__).resolve().parents[1]
BOX = ROOT / "bench" / "q235" / "instrument" / "q235_box.py"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(vocab_size=256, hidden_size=128, intermediate_size=256, moe_intermediate_size=64, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=32, num_experts=8, num_experts_per_tok=2,
                         decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=True, tie_word_embeddings=False,
                         max_position_embeddings=128)
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("q235box")
    Qwen3MoeForCausalLM(cfg).to(torch.bfloat16).save_pretrained(root / "model")
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(3, 256, (2, 24), generator=g)
    mask = torch.ones(2, 24, dtype=torch.long)
    mask[1, 17:] = 0
    ids = ids.masked_fill(mask == 0, 0)
    torch.save({"input_ids": ids, "attention_mask": mask, "labels": ids.masked_fill(mask == 0, -100)}, root / "tokens.pt")
    return root


def _box(root, name, *extra):
    out = root / name
    r = subprocess.run([sys.executable, str(BOX), str(root / "model"), str(root / "tokens.pt"), "--out", str(out), "--device", "cpu",
                        "--enabler", "reference", *extra], capture_output=True, text=True, cwd=ROOT,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    if r.returncode != 0:
        pytest.skip(f"box could not run here: {r.stderr[-400:]}") if "quantize" in r.stderr and "backend" in r.stderr else None
        raise AssertionError(r.stderr[-2000:])
    return json.load(open(out / "box.json")), out


@pytest.fixture(scope="module")
def step(tiny):
    return _box(tiny, "step")


def test_the_step_records_every_registered_check(step):
    rec, out = step
    assert all(rec["rules"].values()), rec["rules"]                 # Amendment 2: by factor, step 0 then step 1
    g0 = rec["step"]["gradients"]
    assert not g0["missing"] and g0["lora_A"] == g0["lora_B"] > 0 and g0["every_A_zero"]   # B starts at zero
    assert rec["step1"]["gradients"]["every_A_nonzero"] and rec["step1"]["optimizer_delta"]["every_A_changed"]
    assert rec["frozen_sha256_after_both_steps_equal"] and rec["frozen_tensors"] > 0
    assert rec["in_process"]["loss_equal"]                          # the CPU reference path is deterministic
    assert rec["step"]["optimizer"].startswith("torch AdamW") and rec["step"]["lr"] == 2e-4
    assert len(rec["layer_sha256"]) == 4 and (out / "nf4.pt").exists()
    prof = rec["load"]["profile"]                                   # gate 2: where the load's time went
    assert prof["calls"]["read"] > 0 and prof["read_bytes"] > 0 and prof["calls"]["quantize"] == 4 and prof["calls"]["assemble"] == 4
    assert prof["total_s"] >= prof["read"] + prof["assemble"] + prof["quantize"] + prof["offload"] - 1e-6


def test_the_fresh_process_repeat_reproduces_step_0(tiny, step):
    rec, _ = _box(tiny, "repeat", "--repeat")
    assert rec["repeat"]["loss"] == step[0]["in_process"]["loss_0"]
    assert rec["layer_sha256"] == step[0]["layer_sha256"]


@pytest.mark.parametrize("kind", ["routing", "epilogue"])
def test_each_planted_error_changes_its_layer_and_nothing_before_it(tiny, step, kind):
    rec, _ = _box(tiny, f"plant_{kind}", "--plant", kind)
    i = rec["plant"]["layer"]
    assert i == 2
    clean = step[0]["layer_sha256"]
    assert rec["layer_sha256"][:i] == clean[:i], "a plant must not touch the layers before it"
    assert rec["layer_sha256"][i] != clean[i], f"the {kind} plant was inert at its layer"


def _box_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("q235_box", BOX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_frozen_hashes_refuse_a_placeholder_and_a_missing_stack(tiny):
    """STRICT: the frozen-base hashes cover every 4-bit expert stack's packed storage and absmax, and a frozen tensor that
    reads as 0 elements (an offload placeholder the state-dict hook did not replace) is an error, never a hash of nothing."""
    from experts4bit_qlora import load_moe_4bit_streaming
    box = _box_module()
    model, _ = load_moe_4bit_streaming(str(tiny / "model"), "cpu", torch.bfloat16, 4, 8, quant_type="nf4")
    h = box.frozen_hashes(model)
    stacks = [n for n, m in model.named_modules() if type(m).__name__ in ("Experts4bit", "ExpertsNbit")]
    assert len(stacks) == 4 and all(f"{n}.{k}" in h for n in stacks for k in box.EXPERT_STACK_TENSORS), sorted(h)[:8]
    base = model.get_submodule(stacks[1])
    keep = base.down_absmax
    base.down_absmax = torch.empty(0)                                   # what an evicted stack holds without its hook
    with pytest.raises(RuntimeError, match="0 elements"):
        box.frozen_hashes(model)
    del base.down_absmax                                                 # gone from the state dict altogether
    with pytest.raises(RuntimeError, match="miss expert storage"):
        box.frozen_hashes(model)
    base.register_buffer("down_absmax", keep)
    assert box.frozen_hashes(model) == h


def test_a_strict_failure_names_itself():
    box = _box_module()
    ok = {"rules": {k: True for k in box.STRICT_RULES}, "frozen_sha256_after_both_steps_equal": True}
    assert box.strict_failures(ok) == []
    bad = {"rules": {**ok["rules"], "step1_every_A_changed": False}, "frozen_sha256_after_both_steps_equal": False}
    assert box.strict_failures(bad) == ["step1_every_A_changed", "frozen_sha256_after_both_steps_equal"]
    assert box.strict_failures({"repeat": {"loss": 1.0}}, repeat=True) == []   # the repeat runs step 0's forward only
    # fail CLOSED outside --repeat: a record that stopped early (no rules, no frozen equality) is not clean
    assert set(box.strict_failures({})) == set(box.STRICT_RULES) | {"frozen_sha256_after_both_steps_equal"}
    assert "step1_every_B_changed" in box.strict_failures({"rules": {k: True for k in box.STRICT_RULES[:-1]},
                                                          "frozen_sha256_after_both_steps_equal": True})
