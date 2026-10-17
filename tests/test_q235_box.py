# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235's box script (``bench/q235/instrument/q235_box.py``) on the CPU: the mechanics of the registered step on a small 4-layer
Qwen3-MoE, on the reference path (the fused kernels need CUDA; the RTX A2000 rehearsal runs those).

- The steps (Amendment 2): at step 0 every LoRA-A gradient is exactly zero (B starts at zero) and every B gradient nonzero and every B
  changed; at step 1 every A gradient is nonzero and every A changed, and B's rules repeat; the frozen base's hashes equal after both
  steps; the in-process repeat recorded.
- The fresh-process repeat (``--repeat``) reproduces step 0's loss.
- Each planted error (Amendment 1: routing, epilogue) changes the planted layer's output and leaves the layers before it untouched,
  so an inert plant cannot pass for a band result.
- The same holds on a small Mixtral saved in its released per-expert spelling (``block_sparse_moe`` w1/w3/w2), in separate runs
  and as Amendment 3's in-process plants, and its router's top-k is captured at every layer: the family ladder's R2 subject
  (#1630) runs this script, and the plants' hooks assume a router that returns its top-k indices third and an expert call that
  takes them second.
- Amendment 3's in-process plants (``--repeat --plants``): the repeat's clean forward is the null; each plant run after it in the same
  process reproduces the separate-process plant bit for bit, and an unplant that leaves anything behind raises.
- Amendment 4: the fp32 copies behind the gradient and delta checks are host copies, never views, and none is made on the device
  inside the step.
- Amendment 6 (``--micro-batch``): shape B's token-weighted accumulation matches shape A's gradient at both steps, its two planted
  accumulation bugs do not, and the comparison's views read as registered.
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
    assert all(0 <= rec[k]["grads_host_copy_s"] <= rec[k]["elapsed_s"] for k in ("step", "step1"))   # Amendment 4, timed apart
    for hm in (rec["load"]["host_memory"], rec["step"]["host_memory"], rec["step1"]["host_memory"]):   # Amendment 4, reported
        assert hm == {} if not Path("/proc/self/status").exists() else (
            set(hm) == {"VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem"} and hm["RssAnon"] > 0 and hm["VmHWM"] >= hm["VmRSS"])
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


def test_the_in_process_plants_reproduce_the_separate_process_plants(tiny, step):
    rec, out = _box(tiny, "repeat_plants", "--repeat", "--plants")
    assert rec["repeat"]["loss"] == step[0]["in_process"]["loss_0"]
    assert rec["layer_sha256"] == step[0]["layer_sha256"], "the null must be the box's step-0 forward"
    for kind in ("routing", "epilogue"):
        p = rec["plants"][kind]
        assert p["layer"] == 2 and p["pre_plant_equal_null"] and p["planted_layer_differs"], p
        assert p["pre_plant_max_rel_diff"] == 0.0 and len(p["pre_plant_rel_diff"]) == 2       # the tolerance's input, recorded
        sep, _ = _box(tiny, f"sep_{kind}", "--repeat", "--plant", kind)
        assert p["layer_sha256"] == sep["layer_sha256"], f"the in-process {kind} plant differs from the separate-process one"
        assert p["loss"] == sep["repeat"]["loss"]
        d = out / f"plant_{kind}"
        on_disk = json.load(open(d / "box.json"))
        assert on_disk["plant"] == {"kind": kind, "layer": 2} and on_disk["layer_sha256"] == p["layer_sha256"]
        assert len(list(d.glob("layer*.pt"))) == 4 and (d / "nf4.pt").exists() and json.load(open(d / "nf4.json"))["loss"] == p["loss"]


def test_plants_need_the_repeat(tiny):
    for extra in (["--plants"], ["--repeat", "--plants", "--plant", "routing"]):
        r = subprocess.run([sys.executable, str(BOX), str(tiny / "model"), str(tiny / "tokens.pt"), "--out", str(tiny / "x"),
                            "--device", "cpu", "--enabler", "reference", *extra], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 2 and "--plants rides the fresh-process repeat" in r.stderr, r.stderr[-300:]


def test_an_unplant_that_leaves_anything_behind_raises(tiny, monkeypatch):
    from experts4bit_qlora import load_moe_4bit_streaming
    box = _box_module()
    model, _ = load_moe_4bit_streaming(str(tiny / "model"), "cpu", torch.bfloat16, 4, 8, quant_type="nf4")
    tok = torch.load(tiny / "tokens.pt", weights_only=True)

    def forward():
        with torch.no_grad():
            return float(model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], labels=tok["labels"]).loss)
    null, _, hooks = box.capture(model)
    forward()
    for h in hooks:
        h.remove()
    for kind in ("routing", "epilogue"):                           # the real unplant restores the surface exactly
        before = box.plant_surface(model, 2)
        _, undo = box.plant(model, kind)
        assert box.plant_surface(model, 2) != before
        undo()
        assert box.plant_surface(model, 2) == before
    assert set(box.in_process_plants(model, forward, null)) == {"routing", "epilogue"}
    real = box.plant
    monkeypatch.setattr(box, "plant", lambda m, kind: (real(m, kind)[0], lambda: None))   # an unplant that removes nothing
    with pytest.raises(RuntimeError, match="did not come off layer 2"):
        box.in_process_plants(model, forward, null)


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


@pytest.fixture(scope="module")
def tiny8(tiny):
    """Amendment 6's shape: 8 right-padded rows of unequal length, so micro-batches of 2 carry unequal counts of valid labels."""
    g = torch.Generator().manual_seed(2)
    lengths = [7, 6, 13, 12, 24, 15, 8, 7]
    ids = torch.randint(3, 256, (8, 24), generator=g)
    mask = (torch.arange(24)[None, :] < torch.tensor(lengths)[:, None]).long()
    ids = ids.masked_fill(mask == 0, 0)
    torch.save({"input_ids": ids, "attention_mask": mask, "labels": ids.masked_fill(mask == 0, -100)}, tiny / "tokens8.pt")
    return tiny


def _box8(root, name, *extra):
    out = root / name
    r = subprocess.run([sys.executable, str(BOX), str(root / "model"), str(root / "tokens8.pt"), "--out", str(out), "--device", "cpu",
                        "--enabler", "reference", *extra], capture_output=True, text=True, cwd=ROOT,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    if r.returncode != 0:
        raise AssertionError(r.stderr[-2000:])
    return json.load(open(out / "box.json"))


@pytest.fixture(scope="module")
def shape_b(tiny8):
    return _box8(tiny8, "shape_b", "--micro-batch", "2")


def test_shape_b_accumulates_to_shape_a_and_its_plants_do_not(shape_b):
    """Amendment 6 on the CPU: the token-weighted accumulation over 4 micro-batches of 2 rows matches shape A's gradient at the same
    parameters at both steps, every registered step rule still holds, and both planted accumulation bugs sit far from it."""
    rec = shape_b
    assert all(rec["rules"].values()), rec["rules"]
    sb = rec["shape_b"]
    assert sb["micro_batch"] == 2 and sb["accumulation"] == 4 and sb["widths"] == [7, 13, 24, 8]
    assert sb["valid_per_micro_batch"] == [11, 23, 37, 13] and sb["valid_total"] == 84
    for k in ("step", "step1"):                              # shape A vs shape B: REPORTED, and bf16 batch-shape numerics vary by
        ga = rec[k]["ga_check"]                              # host (threads, library versions), so only its views and a loose bound
        assert ga["compared"] > 0 and not ga["zero_mismatch"], ga
        assert ga["max_rel"] < 0.1 and ga["cosine"] > 0.999, ga
        assert ga["rel_p50"] <= ga["rel_p99"] <= ga["max_rel"] and len(ga["rel"]) == ga["compared"]
    assert rec["step"]["ga_check"]["compared"] < rec["step1"]["ga_check"]["compared"]   # step 0's A gradients are exactly zero
    for k in ("step", "step1"):                              # rule 8: deterministic check passes; the registered step on the default
        r8 = rec[k]["rule8"]
        assert (r8["algorithms"], r8["restored"], rec[k]["algorithms"]) == ("deterministic", "default", "default"), (k, r8)
        assert r8["check"]["compared"] == rec[k]["ga_check"]["compared"] and not r8["check"]["zero_mismatch"]
        assert r8["check"]["max_rel"] < 1e-2 and r8["floor"]["max_rel"] == 0.0, r8     # two deterministic accumulations agree
        assert rec[k]["ga_mb_check"]["max_rel"] < 1e-2                                 # the step's own accumulation: reported
    floor = max(rec[k]["rule8"]["check"]["max_rel"] for k in ("step", "step1"))
    for kind in ("mean", "overwrite"):
        assert rec["ga_plants"][kind]["algorithms"] == "deterministic"
        assert rec["ga_plants"][kind]["vs_micro_batches"]["max_rel"] > 10 * max(floor, 1e-3), (kind, rec["ga_plants"][kind], floor)
    assert rec["ga_plants"]["overwrite"]["vs_micro_batches"]["norm_ratio"] < 0.9
    rf = rec["route_flips"]                                  # reported: shape A vs B router top-k sets, per layer, real tokens only
    assert len(rf["per_layer"]) == 4 and all(x["real"] == 84 + 8 for x in rf["per_layer"]) and 0 <= rf["flipped"] <= rf["token_layers"]


def test_the_deterministic_check_mode_is_always_restored():
    """Amendment 6: rule 8's check passes run under deterministic algorithms and the previous setting comes back however the block
    ends, so the registered steps never inherit it."""
    box = _box_module()
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    with box.deterministic():
        assert box.algorithms() == "deterministic" and torch.is_deterministic_algorithms_warn_only_enabled()
    assert (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()) == before
    with pytest.raises(RuntimeError):
        with box.deterministic():
            raise RuntimeError("a check pass failed")
    assert box.algorithms() == "default" and (torch.are_deterministic_algorithms_enabled(),
                                              torch.is_deterministic_algorithms_warn_only_enabled()) == before


def test_shape_b_needs_whole_micro_batches_and_stays_off_the_repeat(tiny8):
    for extra in (["--micro-batch", "3"], ["--micro-batch", "2", "--repeat"]):
        r = subprocess.run([sys.executable, str(BOX), str(tiny8 / "model"), str(tiny8 / "tokens8.pt"), "--out", str(tiny8 / "x"),
                            "--device", "cpu", "--enabler", "reference", *extra], capture_output=True, text=True, cwd=ROOT,
                           env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
        assert r.returncode != 0, extra


def test_the_gradient_comparison_reads_each_view():
    """compare_grads: identical gradients read zero everywhere; a missing gradient counts as zero (relative difference 1); a gradient
    where the reference is exactly zero is a zero mismatch; cosine and norm ratio cover the whole adapter gradient."""
    cg = _box_module().compare_grads
    ref = {"model.layers.0.a": torch.tensor([3.0, 4.0]), "model.layers.1.b": torch.tensor([1.0, 0.0]), "model.layers.1.z": torch.zeros(2)}
    same = cg(ref, {k: v.clone() for k, v in ref.items()})
    assert same["max_rel"] == 0.0 and same["cosine"] == pytest.approx(1.0) and same["norm_ratio"] == pytest.approx(1.0)
    assert same["compared"] == 2 and not same["zero_mismatch"] and same["per_layer_max_rel"] == {"0": 0.0, "1": 0.0}
    half = cg(ref, {"model.layers.0.a": torch.tensor([3.0, 4.0]), "model.layers.1.z": torch.tensor([0.0, 1.0])})
    assert half["max_rel"] == 1.0 and half["max_rel_tensor"] == "model.layers.1.b" and half["zero_mismatch"] == ["model.layers.1.z"]
    assert half["per_layer_max_rel"] == {"0": 0.0, "1": 1.0}
    q = cg({f"t{i}": torch.tensor([1.0]) for i in range(100)}, {f"t{i}": torch.tensor([1.0 + i / 100]) for i in range(100)})
    assert q["rel_p50"] == pytest.approx(0.49) and q["rel_p99"] == pytest.approx(0.98) and q["max_rel"] == pytest.approx(0.99)


def test_the_adapter_copies_are_fp32_host_copies_never_views():
    """Amendment 4: the gradient and delta checks' copies are fp32, on the host, and never share storage with what they copy."""
    host_copy = _box_module().host_copy
    src = torch.randn(3, 5, dtype=torch.bfloat16)
    c = host_copy(src)
    assert c.dtype == torch.float32 and c.device.type == "cpu" and torch.equal(c, src.float())   # bf16 -> fp32 is exact
    w = torch.nn.Parameter(torch.randn(4, 2))                       # already fp32 on the host: still a copy, not the same tensor
    c = host_copy(w)
    assert not c.requires_grad and c.data_ptr() != w.data_ptr()
    with torch.no_grad():
        w.add_(1.0)
    assert not torch.equal(c, w.detach())                           # the "before" snapshot does not follow the parameter


def test_no_adapter_copy_stays_on_the_device():
    """Amendment 4: inside the step (``fwd_bwd``, ``one_step``) every copy of an adapter or its gradient goes through ``host_copy``;
    a ``.clone()`` or ``.float()`` there would put an fp32 copy back on the device and into the step's memory record."""
    import ast
    tree = ast.parse(BOX.read_text())
    funcs = {f.name: f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name in ("fwd_bwd", "one_step")}
    assert set(funcs) == {"fwd_bwd", "one_step"}
    for name, f in funcs.items():
        calls = [c for c in ast.walk(f) if isinstance(c, ast.Call)]
        on_device = [ast.unparse(c) for c in calls if isinstance(c.func, ast.Attribute) and c.func.attr in ("clone", "float")]
        assert not on_device, (name, on_device)
        assert any(isinstance(c.func, ast.Name) and c.func.id == "host_copy" for c in calls), name


@pytest.fixture(scope="module")
def tiny_mixtral(tmp_path_factory):
    from transformers import MixtralConfig, MixtralForCausalLM
    cfg = MixtralConfig(vocab_size=256, hidden_size=128, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=32, num_local_experts=8, num_experts_per_tok=2, tie_word_embeddings=False,
                        max_position_embeddings=128)
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("q235box_mixtral")
    MixtralForCausalLM(cfg).to(torch.bfloat16).save_pretrained(root / "model")
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(3, 256, (2, 24), generator=g)
    mask = torch.ones(2, 24, dtype=torch.long)
    mask[1, 17:] = 0
    ids = ids.masked_fill(mask == 0, 0)
    torch.save({"input_ids": ids, "attention_mask": mask, "labels": ids.masked_fill(mask == 0, -100)}, root / "tokens.pt")
    return root


def _box_inproc(root, plant=None, plants=False):
    """The box's own ``run()`` in this process, forward-only (``--repeat``), each layer hashed exactly as its ``main()`` hashes it.
    One import and one small load per call instead of a subprocess each: these are engagement checks, not a second process's
    reproducibility (the subprocess tests above cover that)."""
    import argparse
    import hashlib
    a = argparse.Namespace(checkpoint=str(root / "model"), tokens=str(root / "tokens.pt"), out=None, device="cpu",
                           enabler="reference", adapter_dtype="bf16", r=16, alpha=16, lr=2e-4, weight_decay=0.001,
                           plant=plant, repeat=True, plants=plants)
    out, layer_out = _box_module().run(a)
    sha = [hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest() for t in layer_out if t is not None]
    return out, sha


@pytest.fixture(scope="module")
def mixtral_clean(tiny_mixtral):
    return _box_inproc(tiny_mixtral)


def test_mixtral_is_saved_in_its_released_spelling(tiny_mixtral):
    from safetensors import safe_open
    keys = set(safe_open(str(tiny_mixtral / "model" / "model.safetensors"), "pt").keys())
    assert "model.layers.0.block_sparse_moe.experts.0.w1.weight" in keys and not any("gate_up_proj" in k for k in keys)


def test_mixtral_router_top_k_is_captured_at_every_layer(mixtral_clean):
    out, sha = mixtral_clean
    topk = out["_topk"]
    assert len(topk) == len(sha) == 4
    for t in topk:
        assert t is not None and t.shape[-1] == 2 and int(t.min()) >= 0 and int(t.max()) < 8


@pytest.fixture(scope="module")
def mixtral_plants(tiny_mixtral):
    return {kind: _box_inproc(tiny_mixtral, plant=kind) for kind in ("routing", "epilogue")}


@pytest.mark.parametrize("kind", ["routing", "epilogue"])
def test_each_planted_error_engages_on_mixtral(mixtral_clean, mixtral_plants, kind):
    out, sha = mixtral_plants[kind]
    i = out["plant"]["layer"]
    assert i == 2
    clean = mixtral_clean[1]
    assert sha[:i] == clean[:i], "a plant must not touch the layers before it"
    assert sha[i] != clean[i], f"the {kind} plant was inert at its layer on Mixtral"


def test_the_in_process_plants_reproduce_the_separate_plants_on_mixtral(tiny_mixtral, mixtral_clean, mixtral_plants):
    """Amendment 3's ``--repeat --plants`` on R2's subject family: the null is the clean forward, each unplant comes off the
    Mixtral layer (``in_process_plants`` raises otherwise), and each in-process plant equals its own run bit for bit."""
    out, null = _box_inproc(tiny_mixtral, plants=True)
    assert null == mixtral_clean[1], "the null must be the clean forward"
    for kind in ("routing", "epilogue"):
        p = out["plants"][kind]
        assert p["layer"] == 2 and p["pre_plant_equal_null"] and p["planted_layer_differs"], p
        assert p["pre_plant_max_rel_diff"] == 0.0 and len(p["pre_plant_rel_diff"]) == 2
        sep_out, sep = mixtral_plants[kind]
        assert p["layer_sha256"] == sep, f"the in-process {kind} plant differs from its own run on Mixtral"
        assert p["loss"] == sep_out["repeat"]["loss"]


def test_the_moe_timing_is_reported_only_and_leaves_the_registered_record_alone(tiny, step):
    """``--time-moe``: one extra timed forward-backward after both steps, under ``moe_timing``. The registered fields are the
    untimed run's (the CPU reference path is deterministic, so equal bit for bit), and off, the record carries no trace of it."""
    rec, _ = _box(tiny, "timed", "--time-moe", "2", "--time-moe-bound")
    base = step[0]
    assert not any(k.startswith("time_moe") for k in base["argv"]) and "moe_timing" not in base
    assert set(rec) - set(base) == {"moe_timing"} and rec["argv"]["time_moe"] == 2 and rec["argv"]["time_moe_bound"]
    assert rec["rules"] == base["rules"] and rec["in_process"] == base["in_process"]
    assert rec["layer_sha256"] == base["layer_sha256"] and rec["frozen_sha256_after_load"] == base["frozen_sha256_after_load"]
    for k in ("step", "step1"):
        assert rec[k]["loss"] == base[k]["loss"] and rec[k]["gradients"] == base[k]["gradients"]
        assert rec[k]["optimizer_delta"] == base[k]["optimizer_delta"]
    t = rec["moe_timing"]
    assert "error" not in t, t.get("error")
    assert t["reported_only"] and t["read_by_verdict"] is False and t["sampled_layers"] == [0, 3]
    assert t["pass_off_s"] > 0 and t["pass_on_s"] > 0 and "overhead" in t
    assert 0 <= t["pass_on_grads_copy_s"] <= t["pass_on_s"] and 0 <= t["pass_off_grads_copy_s"] <= t["pass_off_s"]
    (mb,) = t["micro_batches"]                                     # shape A: the one micro-batch, every row
    assert (mb["rows"], mb["width"], mb["positions"]) == (2, 24, 48) and mb["valid"] > 0
    for i in ("0", "3"):
        row = t["layers"][i]
        assert row["staged_by_offload"] is False                     # the CPU rehearsal has no offload
        assert row["forward"]["calls"] == 1 and row["recompute"]["calls"] == 1 and row["backward"]["calls"] == 1, row
        # early stop off: the non-reentrant recompute reaches the module's return, so its compute is observed too
        assert all(v is not None and v >= 0 for ph in ("forward", "recompute") for f in ("stage_ms", "compute_ms") for v in row[ph][f])
        bw = row["backward"]
        assert bw["recompute_inside"] == [True], bw                 # Hugging Face's checkpoint recomputes inside the window
        assert 0 <= bw["own_ms"][0] <= bw["ms"][0]


def test_the_moe_timing_times_shape_bs_accumulation_one_pair_per_micro_batch(tiny8, shape_b):
    """Under ``--micro-batch`` the timed pass is the registered step's own forward-backward, shape B's token accumulation, so each
    sampled layer has one stage/compute pair and one backward per micro-batch, each beside its micro-batch's shape. The registered
    fields are the untimed shape-B run's."""
    rec = _box8(tiny8, "shape_b_timed", "--micro-batch", "2", "--time-moe", "2", "--time-moe-bound")
    base = shape_b
    assert set(rec) - set(base) == {"moe_timing"}
    assert rec["rules"] == base["rules"] and rec["shape_b"] == base["shape_b"] and rec["ga_plants"] == base["ga_plants"]
    for k in ("step", "step1"):
        assert rec[k]["loss"] == base[k]["loss"] and rec[k]["gradients"] == base[k]["gradients"]
        assert rec[k]["ga_check"] == base[k]["ga_check"] and rec[k]["ga_mb_check"] == base[k]["ga_mb_check"]
    t = rec["moe_timing"]
    assert "error" not in t, t.get("error")
    assert [m["width"] for m in t["micro_batches"]] == base["shape_b"]["widths"]
    assert [m["valid"] for m in t["micro_batches"]] == base["shape_b"]["valid_per_micro_batch"]
    assert all(m["rows"] == 2 and m["positions"] == 2 * m["width"] for m in t["micro_batches"])
    for i in ("0", "3"):
        row = t["layers"][i]
        assert row["forward"]["calls"] == row["recompute"]["calls"] == row["backward"]["calls"] == 4, row
        assert all(v is not None and v >= 0 for ph in ("forward", "recompute") for v in row[ph]["compute_ms"])
        bw = row["backward"]
        assert bw["recompute_inside"] == [True] * 4 and all(0 <= o <= w for o, w in zip(bw["own_ms"], bw["ms"])), bw


def test_the_moe_timing_needs_the_stepped_run(tiny):
    for extra, msg in ((["--time-moe", "2", "--repeat"], "--time-moe N (N >= 1) rides the stepped run"),
                       (["--time-moe", "-1"], "--time-moe N (N >= 1) rides the stepped run"),
                       (["--time-moe-bound"], "--time-moe-bound needs --time-moe")):
        r = subprocess.run([sys.executable, str(BOX), str(tiny / "model"), str(tiny / "tokens.pt"), "--out", str(tiny / "x"),
                            "--device", "cpu", "--enabler", "reference", *extra], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 2 and msg in r.stderr, (extra, r.stderr[-300:])


class _Experts(torch.nn.Module):
    """A stand-in experts module carrying hooks named and placed as e4b's offload places its own: the staging pre-hook
    (``_stage_pre_hook``) and the evict post-hook, which skips inside a backward."""

    def __init__(self, log):
        super().__init__()
        from experts4bit_qlora.engines.offload import _in_backward
        self.w, self.log = torch.nn.Parameter(torch.randn(8)), log

        def _stage_pre_hook(module, args):
            log.append("stage")

        def _evict_post_hook(module, args, output):
            if not _in_backward():
                log.append("evict")
        self.register_forward_pre_hook(_stage_pre_hook)
        self.register_forward_hook(_evict_post_hook)

    def forward(self, x):
        self.log.append("compute")
        return (x * self.w).sin() + 1


def _fake_model(n_layers, log):
    from types import SimpleNamespace
    layers = [SimpleNamespace(mlp=SimpleNamespace(experts=_Experts(log))) for _ in range(n_layers)]
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


def test_the_moe_timer_brackets_e4bs_staging_and_compute_and_changes_nothing():
    """The timer's marks sit around e4b's hooks: its stage start before ``_stage_pre_hook``, its stage end after it, its compute
    end before the evict post-hook. Its hooks change no output or gradient, and they are gone afterwards. With the checkpoint's
    early stop on (as the registered steps run), a non-reentrant recompute stops inside the module: its compute is not observed,
    and neither is the module's own backward after it, rather than either being guessed."""
    from torch.utils.checkpoint import checkpoint
    box = _box_module()
    log = []
    model = _fake_model(5, log)
    mods = [lay.mlp.experts for lay in model.model.layers]
    x = torch.randn(8, requires_grad=True)

    def fwd_bwd():
        for m in mods:
            m.w.grad = None
        h = x
        for m in mods:
            h = checkpoint(m, h, use_reentrant=False)
        h.sum().backward()
        return h.detach(), [m.w.grad.clone() for m in mods]
    plain = fwd_bwd()
    hooks_before = [(dict(m._forward_pre_hooks), dict(m._forward_hooks)) for m in mods]

    timer = box.MoETimer(model, 3, cuda=False)
    assert timer.layers == [0, 2, 4] and all(timer.staged.values())
    log.clear()
    real = timer._stamp
    timer._stamp = lambda: log.append("mark") or real()
    with torch.no_grad():
        mods[0](torch.randn(8))
    assert log == ["mark", "stage", "mark", "compute", "mark", "evict"], log
    timer._stamp = real
    timer.marks[0].clear()
    timed = fwd_bwd()
    rec = timer.record()
    timer.remove()
    assert torch.equal(timed[0], plain[0]) and all(torch.equal(a, b) for a, b in zip(timed[1], plain[1]))
    assert [(dict(m._forward_pre_hooks), dict(m._forward_hooks)) for m in mods] == hooks_before
    assert not any(m._backward_hooks or m._backward_pre_hooks for m in mods)
    assert set(rec) == {"0", "2", "4"}
    for row in rec.values():
        assert row["staged_by_offload"] and row["forward"]["calls"] == row["recompute"]["calls"] == row["backward"]["calls"] == 1
        assert row["forward"]["compute_ms"][0] >= 0
        assert row["recompute"]["compute_ms"] == [None]             # stopped at the last saved tensor, before the return
        assert row["backward"]["recompute_inside"] == [True] and row["backward"]["own_ms"] == [None]


def test_the_timed_pass_observes_the_recompute_under_either_checkpoint_and_a_failure_never_fails_the_box():
    """``time_moe`` turns the early stop off, so the non-reentrant recompute returns and the module's own backward is the window
    after it; the reentrant checkpoint (the fast path's default) recomputes before the window, which is then the module's own."""
    from torch.utils.checkpoint import checkpoint
    box = _box_module()
    model = _fake_model(2, [])
    mods = [lay.mlp.experts for lay in model.model.layers]
    for reentrant in (False, True):
        def fwd_bwd():
            h = torch.randn(8, requires_grad=True)
            for m in mods:
                h = checkpoint(m, h, use_reentrant=reentrant)
            h.sum().backward()
            return 0.0, {}, 0.0, 0.0, 0.0                              # the box's fwd_bwd: its last is the grads copy's time
        rec = box.time_moe(model, fwd_bwd, 2, cuda=False, sync=lambda: None, bound=True,
                           micro_batches=[{"rows": 1, "width": 8, "valid": 7}])
        assert "error" not in rec and rec["sampled_layers"] == [0, 1] and rec["pass_off_s"] >= 0 and "overhead" in rec
        assert rec["pass_on_grads_copy_s"] == rec["pass_off_grads_copy_s"] == 0.0
        assert rec["micro_batches"] == [{"rows": 1, "width": 8, "valid": 7, "positions": 8}]
        for row in rec["layers"].values():
            assert row["recompute"]["calls"] == 1 and row["recompute"]["compute_ms"][0] >= 0, (reentrant, row)
            bw = row["backward"]
            assert bw["recompute_inside"] == [not reentrant], (reentrant, bw)
            assert 0 <= bw["own_ms"][0] <= bw["ms"][0] and (bw["own_ms"] == bw["ms"]) is reentrant

    def boom():
        raise RuntimeError("out of memory")
    rec = box.time_moe(model, boom, 2, cuda=False, sync=lambda: None)
    assert rec["error"] == "RuntimeError: out of memory" and "layers" not in rec
    assert all(len(m._forward_pre_hooks) == 1 and len(m._forward_hooks) == 1 and not m._backward_hooks for m in mods)
