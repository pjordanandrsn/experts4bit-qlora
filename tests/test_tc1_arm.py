"""CI wrapper for ``bench/tc1/tc1_arm.py --selftest`` (lane TC1, TC1-PREREG.md), mirroring tests/test_tp4_arm.py.

The selftest drives the three framework branches through the one ``run_arm`` on CPU with mocked kernels (tp4's T8), plus
TC1's additions: T19 (``--adapter-dtype`` casts EVERY arm's adapters to fp32, e4b included; ``native`` leaves the shipped
bf16), T20 (``--lora-init matched:<seed>``: the per-slot deterministic LoRA A over the two real adapter layouts -- e4b's and
PEFT 0.21.2's -- on one seeded frozen base, so the matched arms report IDENTICAL step-0 losses and trajectories that agree
to a MEASURED tolerance while the native inits do not), and T21 (the frozen-base probe with its byte-flip control on every
row, and on genuine CPU-quantised e4b / bitsandbytes storage). A green suite that never executes is not a gate, so CI runs
it end to end and asserts the receipt-level outcomes -- including that a real run without ``--prereg`` is refused before any
receipt exists. The structural slot mapping is also exercised against the REAL peft when it is importable (skipped with a
reason otherwise: pyproject's ``[test]`` extra does not pull peft).
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ARM = REPO / "bench" / "tc1" / "tc1_arm.py"
RUN_SH = REPO / "bench" / "tc1" / "tc1_run.sh"
DRIVE_SH = REPO / "bench" / "tc1" / "tc1_drive.sh"


def _run(*args, script=ARM):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=900, cwd=REPO)


def _load_arm_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tc1_arm_under_test", ARM)
    m = importlib.util.module_from_spec(spec)
    sys.modules["tc1_arm_under_test"] = m
    spec.loader.exec_module(m)
    return m


def _selftest_receipts(extra=()):
    p = _run("--selftest", *extra)
    assert p.returncode == 0, (p.stdout + p.stderr)[-3000:]
    d = Path(re.search(r"SELFTEST OK dir=(\S+)", p.stdout).group(1))
    return p, d


def test_tc1_arm_selftest():
    p, d = _selftest_receipts()
    assert "SELFTEST OK" in p.stdout
    # tp4's dry-runs still pass through this copy (the T10 detector, the #542 structural expert selection)
    assert "'tiny_keqv30': 115" in p.stdout and "'tiny_plain4': 16" in p.stdout and "'tiny_missing_k': ['layers.1']" in p.stdout
    assert "'granite_on_disk': {'n': 6, 'substring': 0, 'param_re': 0}" in p.stdout, p.stdout[-1500:]
    assert "'refusals': ['dense', 'not_per_layer', 'per_expert_2d', 'ragged_declared_matches_nothing', 'ragged_no_config']" in p.stdout
    # receipts name THIS lane's pre-registration
    ref = json.loads((d / "tiny_e4b_reference_attn4.json").read_text())
    assert ref["prereg"] == "tc1/TC1-PREREG.md" and ref["harness"].startswith("tc1_arm.py (copy of tp4_arm.py @ 10ce711d")
    # TC1b: --note lands verbatim on exactly the arm it was given to (the p38 anchor arms record how they differ from tp4's)
    assert ref["note"] == "selftest note (TC1b --note)" and json.loads((d / "tiny_e4b_fused_attn4.json").read_text())["note"] is None
    # T19: the default cast reaches e4b's expert adapters (bf16 as the loader builds them -> fp32), recorded
    assert ref["adapter_dtype"] == "fp32" and list(ref["adapter_dtypes_after"]) == ["torch.float32"] and ref["lora_cast_to_fp32"] > 0, ref["adapter_dtypes_before"]
    sh = json.loads((d / "tiny_e4b_fused_attn4_shipped.json").read_text())
    assert sh["adapter_dtype"] == "native" and "torch.bfloat16" in sh["adapter_dtypes_after"] and sh["lora_cast_to_fp32"] == 0 and sh["matched_init"] is None
    # T20: the matched arms -- identical step 0, complete slot maps, the measured tolerance printed
    M = {t: json.loads((d / f"tiny_{fw}_{t}.json").read_text()) for fw, t in (("e4b", "fused_attn4_m"), ("e4b", "reference_attn4_m"), ("hf", "hf_peft_m"), ("unsloth", "ckpt_unsloth_m"))}
    assert len({r["eval_loss_step0"] for r in M.values()}) == 1, {t: r["eval_loss_step0"] for t, r in M.items()}
    for t, r in M.items():
        mi = r["matched_init"]
        assert r["lora_init"] == "matched:3407" and mi["complete"] is True and mi["n_slots_set"] == mi["n_slots_expected"] == 24 and not mi["unmapped"], (t, mi)
    assert re.search(r"matched_tolerance=\{'hf_vs_e4b': \{'train_max': [0-9.e-]+", p.stdout), "the selftest must print the measured matched-trajectory tolerance"
    ex = json.loads((d / "tiny_hf_hf_peft_m_extra.json").read_text())
    assert ex["matched_init"]["complete"] is False and ex["matched_init"]["n_slots_set"] == 26, ex["matched_init"]
    bnz = json.loads((d / "tiny_e4b_fused_attn4_m_bnz.json").read_text())
    assert bnz["status"] == "harness_error" and bnz["matched_init"]["b_nonzero"] == ["model.layers.0.mlp.experts.down_lora_B"]
    # T21: the probe on every OK row, three slots, every control detecting its flip
    for name in ("tiny_e4b_fused_attn4", "tiny_unsloth_ckpt_unsloth", "tiny_hf_hf_peft", "tiny_e4b_reference_attn4_m"):
        r = json.loads((d / f"{name}.json").read_text())
        fp = r["frozen_base_probe"]
        assert set(fp["slots"]) == {"gate_up", "down", "q_proj"} and fp["control_detects_flip"] is True and not fp["errors"], (name, fp)
    # T21 on genuine storage ran (a skip is reported, never silent)
    m = re.search(r"frozen_probe_real=(\{.*?\}) phase2=", p.stdout, re.S)
    assert m, p.stdout[-800:]
    if "'skipped'" in m.group(1):
        pytest.skip("the real-storage probe skipped: " + m.group(1))
    assert "'expert_slots_same_bytes': {'gate_up': True, 'down': True}" in m.group(1) and "'attention_dq_on_e4b': True" in m.group(1), m.group(1)
    # phase 2: the backend counters route by the requested backend and the axolotl arm ran its census, slot map and probe
    p2 = re.search(r"phase2=(\{.*)$", p.stdout, re.S).group(1)
    assert "'ckpt_unsloth_gmm': {'unsloth_grouped_mm': 8, 'unsloth_triton': 0, 'unsloth_loop': 0, 'moe_bnb4bit_backend': 8}" in p2 and "'ckpt_unsloth_loop': {'unsloth_grouped_mm': 0, 'unsloth_triton': 0, 'unsloth_loop': 8" in p2, p2[:600]
    if "'axolotl': {'skipped'" in p2:
        pytest.skip("the axolotl tiny arm skipped: " + p2[p2.index("'axolotl'"):][:200])
    assert "'quantized_moe_experts_n': 4, 'n_bnb4bit_unwrapped': 2, 'slots': 24, 'probe': {'gate_up': 'nf4/64', 'down': 'nf4/64'" in p2, p2[-600:]
    # phase 3: the controls that could not fail now have a failing case each, printed by the selftest, and the receipts carry the new fields
    for needle in ("FAILING-CASE A:", "FAILING-CASE B:", "FAILING-CASE D:", "'blind_hasher_detects': False", "'control_tensor': 'model.layers.0.mlp.experts.base.down_absmax'"):
        assert needle in p.stdout, needle
    # TC3: the lever helpers, the per-micro-batch step context (entered steps x accum times) and the lever refusal's failing case
    assert "FAILING-CASE TC3-lever: hf_offload: NotImplementedError at step 1" in p.stdout and "'lever_refusal': 'NotImplementedError'" in p.stdout
    tc3 = re.search(r"tc3=(\{.*\})$", p.stdout, re.S).group(1)
    assert "'step_ctx': {'entered': 48, 'steps_x_accum': 48}" in tc3 or "'step_ctx': 'skipped" in tc3, tc3
    rf = json.loads((d / "tiny_hf_hf_peft_m_offload.json").read_text())
    assert rf["status"] == "refused" and rf["memory_lever"] == "hf_offload" and rf["exception_type"] == "NotImplementedError" and rf["host_ram_high_water_gb"] > 0 and rf["hf_offload"]["device_map_summary"]["any_cpu_or_disk"]
    res = json.loads((d / "tiny_e4b_fused_attn4.json").read_text())        # a resident arm: no lever, the host-RAM fields on the row all the same
    assert res["memory_lever"] is None and res["host_ram"]["high_water_gb"] == res["host_ram_high_water_gb"] > 0 and "host_ram_total_gb" in res
    fu = json.loads((d / "tiny_e4b_fused_attn4.json").read_text())
    assert fu["C1_control_tensor"] and fu["C1_control_detects_flipped_byte"] is True and fu["C1_regime_by_tensor"] == {"u8-packed": 4, "fp32": 4}
    assert fu["lora_path_present"] is True and fu["lora_path_loop_steps"] == [] and fu["matched_init_sha"] and fu["matched_init_sha_slots"] == 24 and fu["loss_step2"] == fu["losses"][2]
    assert set(fu["dynamo_counters"]) == {"step10", "step12"} and len(fu["eval_rows"]) == len(fu["eval_curve"]) and len(fu["microbatch_padded_len"]) == fu["steps"]
    assert fu["arm_facts"]["torch_num_threads"] >= 1 and fu["arm_facts"]["lora_delta_dtype"].startswith("A.dtype")
    lp = json.loads((d / "tiny_e4b_fused_attn4_loop.json").read_text())
    assert lp["lora_path_loop_steps"] == list(range(1, lp["steps"] + 1))
    gm = json.loads((d / "tiny_unsloth_ckpt_unsloth_gmm.json").read_text())
    assert gm["unsloth_grouped_mm_calls_per_step_min"] == 6 * 2 * 4 and gm["unsloth_manual_grouped_mm_calls_per_step_max"] == 0
    hf = json.loads((d / "tiny_hf_hf_peft.json").read_text())
    assert hf["hf_double_quant"] == {"requested": True, "loaded_attention_nested": None}   # the HF arm's default: double-quant ON, matching e4b's attention
    # T17 (P43) still holds: every step printed, every micro-batch timed, the CELL line never carries the per-step lists
    assert ref["log_every"] == 1 and ref["microbatch_timing"] is True
    assert '"microbatch_ms"' not in "".join(line for line in p.stdout.splitlines() if line.startswith("CELL "))
    # TC2 (T23-T27): the 16-bit Unsloth load's receipt (packed class, counted GEMM, the probe's dequantize() regime), the suffixed attn_only stubs with a
    # matched init over the trainable slots, the HF arm's dispatch record, and the failing cases printed
    mx = json.loads((d / "tiny_unsloth_ckpt_unsloth_mxfp4.json").read_text())
    assert mx["unsloth_load_in_4bit"] is False and mx["census"]["expert_param_classes"] == {"Mxfp4ExpertParam": 4} and mx["unsloth_packed_calls_per_step_min"] == {"unsloth_mxfp4_grouped_mm": 8}
    assert mx["frozen_base_probe"]["slots"]["gate_up"]["regime"] == "Mxfp4ExpertParam-packed/dequantize()" and mx["frozen_base_probe"]["control_detects_flip"] is True
    assert mx["unsloth_double_quant"]["how"].startswith("not applicable") and mx["moe_backend_selected"] == "grouped_mm" and mx["unsloth_backend_calls_per_step_min"]["moe_bnb4bit_backend"] == 0
    ao = json.loads((d / "tinygo_e4b_attn_only_m.json").read_text())
    ao2 = json.loads((d / "tinygo_e4b_attn_only_m_d2.json").read_text())
    assert ao["matched_init"]["complete"] is True and ao["matched_init"]["n_slots_expected"] == 8 and ao["matched_init"]["expected_parts"]["experts"].startswith("excluded") and ao["matched_init_sha"] == ao2["matched_init_sha"]
    st = json.loads((d / "tinygo_e4b_fused_attn4_m.json").read_text())
    assert st["status"] == "refused" and st["cited"] == "tp1,tp2" and st["probed_by"] == "attn_only" and not (d / "tinygo_e4b_fused_attn4_m_d2.json").exists()
    t214 = json.loads((d / "tiny_hf_hf_peft_m_t214.json").read_text())
    assert t214["hf_experts_dispatch"]["requested"] == "grouped_mm" and t214["hf_experts_dispatch"]["reached_grouped_mm"] is False and t214["hf_experts_dispatch"]["torch_grouped_mm_calls_per_step_min"] == 0
    for needle in ("FAILING-CASE TC2-T24 (arm):", "FAILING-CASE TC2-T26 (arm):", "FAILING-CASE TC2-dispatch (arm):", "tc2={'mxfp4': {'classes': {'Mxfp4ExpertParam': 4}, 'packed_calls': {'unsloth_mxfp4_grouped_mm': 8}"):
        assert needle in p.stdout, needle


def test_real_run_without_prereg_refuses():
    p = _run("--framework", "hf", "--arm", "hf")   # no --selftest, no --prereg: refuse before any cell or stub
    assert p.returncode == 2, (p.returncode, (p.stdout + p.stderr)[-2000:])
    assert "--prereg is required" in p.stderr and "tc1/TC1-PREREG.md" in p.stderr


def test_a_malformed_lora_init_is_refused_at_the_command_line():
    p = _run("--framework", "hf", "--arm", "hf", "--prereg", "tc1/TC1-PREREG.md", "--lora-init", "matched:x")
    assert p.returncode == 2 and "--lora-init must be" in p.stderr, (p.returncode, p.stderr[-500:])


def test_matched_lora_A_is_the_registered_generator():
    """T20: the slot tensor is a pure function of (seed, layer, kind, expert, r, fan_in) with PEFT's bound."""
    import hashlib
    import math

    import torch
    arm = _load_arm_module()
    A = arm.matched_lora_A(3407, 5, "down", 17, 16, 768)
    assert tuple(A.shape) == (16, 768) and A.dtype == torch.float32 and A.device.type == "cpu"
    assert float(A.abs().max()) <= 1.0 / math.sqrt(768) and float(A.abs().max()) > 0.5 / math.sqrt(768)
    assert torch.equal(A, arm.matched_lora_A(3407, 5, "down", 17, 16, 768))
    for other in ((3408, 5, "down", 17), (3407, 6, "down", 17), (3407, 5, "gate_up", 17), (3407, 5, "down", 18)):
        assert not torch.equal(A, arm.matched_lora_A(*other, 16, 768))
    assert arm.slot_seed(3407, 5, "down", 17) == int.from_bytes(hashlib.sha256(b"3407|5|down|17").digest()[:8], "big")
    assert arm.parse_lora_init("native") is None and arm.parse_lora_init("matched:3407") == 3407
    with pytest.raises(ValueError):
        arm.parse_lora_init("matched:")


def test_slot_mapping_on_the_real_peft_param_wrapper():
    """T20 against the INSTALLED peft: a transformers-v5-shaped tiny MoE through get_peft_model(target_modules +
    target_parameters) -- the slot map must find every expert through ParamWrapper.parameter_name, write the registered
    tensor into rows e*r:(e+1)*r of the [r*E, in] A, and PEFT's own get_delta_factors must then see expert e's A as
    exactly that tensor. Skips (with the reason) where peft is not installed: pyproject's [test] extra does not pull it."""
    peft = pytest.importorskip("peft")
    import types

    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model
    arm = _load_arm_module()
    E, H, inter, L, r = 4, 8, 6, 2, 2

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * inter, H))   # transformers v5 layout [E, out, in]
            self.down_proj = nn.Parameter(torch.randn(E, H, inter))

        def forward(self, x):
            return x

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(self.self_attn, p, nn.Linear(H, H, bias=False))
            self.mlp = nn.Module()
            self.mlp.experts = Experts()

        def forward(self, x):
            return self.mlp.experts(self.self_attn.q_proj(x))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Layer() for _ in range(L)])
            self.config = types.SimpleNamespace(model_type="tiny", num_hidden_layers=L, hidden_size=H, num_experts=E, to_dict=lambda: {})

        def forward(self, x):
            for layer in self.model.layers:
                x = layer(x)
            return x
    mods = [f"model.layers.{i}.self_attn.{p}" for i in range(L) for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
    params = [f"model.layers.{i}.mlp.experts.{n}" for i in range(L) for n in ("gate_up_proj", "down_proj")]
    pm = get_peft_model(M(), LoraConfig(r=r, lora_alpha=4, lora_dropout=0.0, bias="none", target_modules=mods, target_parameters=params))
    mi = arm.apply_matched_init(pm, "matched:3407", L, pm.config if hasattr(pm, "config") else None)
    assert mi["complete"] is True and mi["n_slots_set"] == mi["n_slots_expected"] == 8 + 2 * L * E, mi
    assert mi["kinds"] == {"q": L, "k": L, "v": L, "o": L, "gate_up": L * E, "down": L * E}, mi["kinds"]
    assert all("structure" in v for k, v in mi["mapping_rules"].items() if "experts" in k), mi["mapping_rules"]
    wrappers = {(arm.layer_index_of(n), m.parameter_name): m for n, m in pm.named_modules() if type(m).__name__ == "ParamWrapper"}
    assert len(wrappers) == 2 * L
    for (layer, pname), w in wrappers.items():
        kind = "gate_up" if pname == "gate_up_proj" else "down"
        A = w.lora_A["default"].weight
        fan_in = A.shape[1]
        lhs, rhs, _ = w.get_delta_factors("default")                        # peft 0.21.2: lhs = B [E, out, r], rhs = A [E, r, in]
        for e in range(E):
            want = arm.matched_lora_A(3407, layer, kind, e, r, fan_in)
            assert torch.equal(A[e * r:(e + 1) * r], want), (layer, pname, e)
            assert torch.equal(rhs[e].to(torch.float32), want), (layer, pname, e, "get_delta_factors does not see the slot tensor")
        assert not w.lora_B["default"].weight.any()
    assert mi["expected_parts"] == {"n_attention_projections": 8, "n_layers": L, "n_experts": E}
    assert peft.__version__


def test_frozen_base_probe_on_genuine_storage():
    """T21 on real bitsandbytes NF4 storage (CPU): e4b's per-expert dequant bytes equal bnb's whole-stack slice, every
    control detects its flip, and e4b's attention Params4bit carries bnb's default double-quant (`+dq`)."""
    pytest.importorskip("bitsandbytes")
    arm = _load_arm_module()
    out = arm._selftest_frozen_probe_real()
    if "skipped" in out:
        pytest.skip(out["skipped"])
    assert out["expert_slots_same_bytes"] == {"gate_up": True, "down": True} and out["attention_dq_on_e4b"] is True, out
    assert out["e4b_regimes"] == {"gate_up": "nf4/64", "down": "nf4/64", "q_proj": "nf4/64+dq"} and out["hf_regimes"]["q_proj"] == "bf16", out


# ----------------------------------------------------------------------------- e4b#548: where the time before step 1 goes (tp4's tests, on this copy)
def test_phase_seconds_account_for_the_whole_prologue():
    p, d = _selftest_receipts()
    assert "PROLOGUE " in p.stdout
    for name in ("tiny_e4b_fused_attn4", "tiny_unsloth_ckpt_unsloth", "tiny_hf_hf_peft", "tiny_hf_hf_peft_m"):
        r = json.loads((d / f"{name}.json").read_text())
        ph, tot = r["phase_seconds"], r["prologue_s"]
        for k in ("preamble", "load_weights", "census", "frozen_probe", "trainable_sha", "counters", "c1_before", "eval0", "optimizer"):   # TC1: + frozen_probe
            assert k in ph, (name, k, sorted(ph))
        assert "c1_after" in ph and "adapter_save" in ph, (name, sorted(ph))
        named = sum(v for k, v in ph.items() if k not in ("c1_after", "adapter_save", "eval_final"))
        assert abs(named + r["prologue_unattributed_s"] - tot) < 0.05, (name, named, r["prologue_unattributed_s"], tot)
        assert 0 <= r["prologue_unattributed_s"] <= 0.5 * tot + 0.05, (name, r["prologue_unattributed_s"], tot)
        assert all(v >= 0 for v in ph.values()), ph


def test_a_refused_row_also_says_where_its_time_went():
    _, d = _selftest_receipts()
    r = json.loads((d / "tiny_unsloth_ckpt_unsloth_refuse.json").read_text())
    assert r["status"] == "refused" and r["phase_seconds"] and "preamble" in r["phase_seconds"], r
    bad = json.loads((d / "tiny_unsloth_ckpt_unsloth_badsha.json").read_text())
    assert bad["status"] == "tokens_mismatch" and bad["phase_seconds"] == {} and bad["phase_in_flight"] is None, bad


def test_an_over_budget_prologue_refuses_itself_and_names_the_phase():
    p = _run("--selftest", "--phase-budget-s", "0.001")
    out = p.stdout + p.stderr
    assert p.returncode == 16, (p.returncode, out[-3000:])
    cell = [ln for ln in p.stdout.splitlines() if ln.startswith("CELL PHASE_ALARM ")]
    assert len(cell) == 1, p.stdout[-2000:]
    r = json.loads(cell[0][len("CELL PHASE_ALARM "):])
    assert r["status"] == "phase_alarm" and r["phase_in_flight"] and r["phase_in_flight"] in r["phase"], r
    assert any(k.startswith(r["phase_in_flight"]) for k in r["phase_seconds"]), r["phase_seconds"]


def test_the_budget_defaults_to_a_share_of_the_arms_own_alarm():
    arm = _load_arm_module()

    def ns(**kw):
        return type("A", (), dict({"phase_budget_s": None}, **kw))()
    old = {k: os.environ.get(k) for k in ("TC1_PHASE_BUDGET_S", "TC1_ARM_ALARM_S")}
    try:
        for k in old:
            os.environ.pop(k, None)
        assert arm.phase_budget_for(ns()) == 0.0
        os.environ["TC1_ARM_ALARM_S"] = "3600"
        assert arm.phase_budget_for(ns()) == 3600 * arm.PROLOGUE_BUDGET_SHARE
        os.environ["TC1_PHASE_BUDGET_S"] = "900"
        assert arm.phase_budget_for(ns()) == 900.0
        assert arm.phase_budget_for(ns(phase_budget_s=42.0)) == 42.0
        assert arm.phase_budget_for(ns(phase_budget_s=0.0)) == 0.0
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_tc1_run_sh_tells_the_arm_the_alarm_it_runs_under():
    sh = RUN_SH.read_text()
    assert "TC1_ARM_ALARM_S=$A perl -e \"alarm $A; exec @ARGV\"" in sh, "the arm must be given the same A perl alarms on"


def test_tc1_run_sh_runs_the_registered_arm_order_with_the_matched_flags():
    """Phase 3 I [F4]: the judged family's invocations, in order, with the matched / profiled flags; the labelled rows live in
    tc1_native_family (its own box, its own e4b fused_m first) with the venv selectors and flags the instruction names."""
    body = RUN_SH.read_text()
    fam = body[body.index("tc1_family(){"):body.index("# tc1_native_family")]
    nat = body[body.index("tc1_native_family(){"):body.index("# ---------------------------------------------------------------- the plan")]
    calls = re.findall(r"(?:arm|draw2|todo_arm)\s+\$FAM\s+(e4b|unsloth|hf|axolotl)\s+(\S+)", fam)
    want = [("e4b", "fused_attn4_m"), ("unsloth", "ckpt_unsloth_m"), ("e4b", "reference_attn4_m"), ("e4b", "fused_attn4_m"), ("unsloth", "ckpt_unsloth_m"),
            ("hf", "hf_peft_m"), ("axolotl", "ckpt_axolotl_m"), ("e4b", "fused_attn4_m_prof"), ("unsloth", "ckpt_unsloth_prof"),
            ("e4b", "fused_attn4_m_mb1"), ("unsloth", "ckpt_unsloth_m_mb1"), ("hf", "hf_peft_m_mb1")]
    assert calls == want, calls
    assert fam.count("draw2 $FAM") == 2 and "todo_arm $FAM" not in fam
    assert 'MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"' in fam and re.search(r"reference_attn4_m reference .* \$MATCH", fam)
    assert re.search(r"ckpt_unsloth_m unsloth .* --unsloth-moe-backend grouped_mm \$MATCH", fam)
    assert re.search(r"fused_attn4_m_prof fused \$PAL .* \$MATCH \$PROF", fam) and re.search(r"ckpt_unsloth_prof unsloth \$PAL .* --unsloth-moe-backend grouped_mm \$MATCH \$PROF", fam)
    assert fam.count("dmon_start") == 2 and 'PROF="--log-every 1 --microbatch-timing 1 --profile-steps $PROFILE_STEPS --profile-warm $PROFILE_WARM"' in fam
    ncalls = re.findall(r"(?:arm|draw2)\s+\$FAM\s+(e4b|unsloth|hf|axolotl)\s+(\S+)", nat)
    assert ncalls == [("e4b", "fused_attn4_m"), ("unsloth", "ckpt_unsloth_best"), ("unsloth", "ckpt_unsloth_t28"), ("unsloth", "ckpt_unsloth_triton"), ("e4b", "fused_attn4_shipped"),
                      ("e4b", "fused_attn4_m_nodgrad"), ("e4b", "fused_attn4_m_t212"), ("axolotl", "ckpt_axolotl_best"), ("hf", "hf_peft_m_mb1_t214")], ncalls
    assert re.search(r"UNS_VENV=t28 arm \$FAM unsloth ckpt_unsloth_t28 unsloth .* --unsloth-moe-backend default \$MATCH", nat)
    assert re.search(r"ckpt_unsloth_best unsloth .* --unsloth-moe-backend grouped_mm --unsloth-speed-tilt 1 --adapter-dtype fp32 --lora-init native", nat)
    assert re.search(r"ckpt_unsloth_triton unsloth .* --unsloth-moe-backend unsloth_triton \$MATCH", nat)
    assert re.search(r"fused_attn4_shipped fused .* \$NATIVE", nat) and re.search(r"fused_attn4_m_nodgrad fused .* --dgrad 0 \$MATCH", nat)
    assert re.search(r"E4B_VENV=t212 arm \$FAM e4b fused_attn4_m_t212 fused .* \$MATCH", nat)
    assert re.search(r"ckpt_axolotl_best axolotl \$AAL .* --axolotl-best 1 --adapter-dtype fp32 --lora-init native", nat)
    assert re.search(r"HF_VENV=t214 arm \$FAM hf hf_peft_m_mb1_t214 hf .* mb1 .* --hf-experts-implementation grouped_mm \$MATCH", nat) and 'status_of qwen3 hf hf_peft_m' in nat
    assert 'draw2(){ local FAM=$1 FW=$2 TAG=$3; shift 3; arm "$FAM" "$FW" "${TAG}_d2" "$@"; }' in body
    assert 'qwen3native) tc1_native_family qwen3native' in body and 'qwen3)       tc1_family        qwen3' in body
    # the driver >= 580 gate precedes every cu130 install and the gated arms refuse by name, never fall back to the t28 venv
    assert '[ "$DRIVER_MAJOR" -ge 580 ]' in body and body.index("CU130_OK=1; case") < body.index("venv-unsloth-t28:") < body.index("uv venv --python 3.12")
    assert 'refused "$CU130_REASON"' in body and "unsloth[cu130-torch2121]" in body and "unsloth[cu128-torch280]" in body
    assert '--extra-index-url https://download.pytorch.org/whl/cu130' in body and '"axolotl==$AX_VER"' in body
    # J: the t212 install into venv-unsloth has its own tripwire and its own rows; F: OMP_NUM_THREADS = the physical core count on every arm
    assert "pip_e4b_t212.log" in body and 'T212_OK=1' in body and "OMP_NUM_THREADS=$PHYS" in body and "lscpu -p=CORE,SOCKET" in body


def test_gpu_class_check_accepts_h100_spellings_and_labels_the_box():
    """K: TC1_GPU_CLASS=H100 must pass the class check for every H100 spelling (lane TC1c), 5090 still passes, a 4090 is refused,
    and the recorded box class is 'RTX <n>' for a numeric class and the class string otherwise."""
    body = RUN_SH.read_text()
    check = re.search(r'^case "\$GPU_NAME" in \*"\$GPU_CLASS"\*\) ;; \*\) .*?;; esac$', body, re.M)
    label = re.search(r'^case "\$GPU_CLASS" in \[0-9\]\*\) BOX_CLASS="RTX \$GPU_CLASS";; \*\) BOX_CLASS="\$GPU_CLASS";; esac$', body, re.M)
    assert check and label, "the class check / label lines are not in the shape this test drives"
    for name, cls, ok, want_label in (("NVIDIA H100 NVL", "H100", True, "H100"), ("NVIDIA H100 80GB HBM3", "H100", True, "H100"), ("NVIDIA H100 PCIe", "H100", True, "H100"),
                                      ("NVIDIA GeForce RTX 5090", "5090", True, "RTX 5090"), ("NVIDIA GeForce RTX 4090", "5090", False, None),
                                      ("NVIDIA GeForce RTX 4090", "4090", True, "RTX 4090"), ("NVIDIA RTX A2000 12GB", "RTX A2000", True, "RTX A2000"),      # TC3's two boxes
                                      ("NVIDIA GeForce RTX 5090", "RTX A2000", False, None)):
        script = "\n".join(['say(){ echo "$*"; }; finish(){ echo "FINISH $1"; exit $1; }', f'GPU_NAME="{name}"; GPU_CLASS="{cls}"', check.group(0), label.group(0), 'echo "PASS label=$BOX_CLASS"'])
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        if ok:
            assert r.returncode == 0 and f"PASS label={want_label}" in r.stdout, (name, cls, r.stdout, r.stderr)
        else:
            assert r.returncode == 12 and "BOX REFUSED" in r.stdout, (name, cls, r.stdout)


def test_every_env_knob_the_box_reads_is_forwarded_by_the_driver():
    """tp4's lesson (TP4-PREREG amendment 2): a knob the box script reads from the environment that the controller does not
    forward silently never reaches the box. Every TC1_* read by tc1_run.sh or tc1_arm.py must be in tc1_drive.sh's forwarded
    list or its handshake string, unless the box script sets it itself."""
    run, arm, drive = RUN_SH.read_text(), ARM.read_text(), DRIVE_SH.read_text()
    read = set(re.findall(r"\$\{(TC1_[A-Z0-9_]+)", run)) | set(re.findall(r"\$\{!v", run) and re.findall(r"for v in ((?:TC1_[A-Z_]+ )+)", run)[0].split())
    read |= set(re.findall(r'os\.environ\.get\("(TC1_[A-Z0-9_]+)"', arm)) | set(re.findall(r'\("(TC1_[A-Z0-9_]+)", [0-9.A-Z_]+\)', arm))
    set_by_box = set(re.findall(r"(?:^|\s)(TC1_[A-Z0-9_]+)=", run))
    forwarded = set(re.findall(r"TC1_[A-Z0-9_]+", drive[drive.index("PASS="):drive.index("TC1_DRIVE_DRYRUN")]))
    # TC3: TC1_LOCAL_* are the HAND-RUN knobs of the owned 12 GB box (TC1_LOCAL_BOX=1 skips the launcher's nonce / instance / deadline) and are
    # deliberately never forwarded by the launcher's driver -- a rental must never run without its nonce.
    missing = sorted(k for k in read - set_by_box - forwarded if not k.startswith("TC1_LOCAL_"))
    assert not missing, f"read by the box/arm but never forwarded by tc1_drive.sh: {missing}"
    assert not any(k.startswith("TC1_LOCAL_") for k in forwarded) and "TC1_LOCAL" not in drive, "tc1_drive.sh must never forward the hand-run knobs"
    assert {"TC1_LOCAL_BOX", "TC1_LOCAL_OUT", "TC1_LOCAL_SNAPSHOT", "TC1_LOCAL_PYTHON"} <= read
    for must in ("TC1_STEPS", "TC1_EVAL_N", "TC1_EVAL_EVERY", "TC1_MATCHED_SEED", "TC1_PHASE_BUDGET_S", "TC1_PROFILE_STEPS"):
        assert must in forwarded, must


TC1B_KNOBS = ("TC1_CURVE_STEPS", "TC1_CURVE_EVAL_EVERY", "TC1_CURVE_EVAL_N", "TC1_T1_MB", "TC1_T1_ACCUM", "TC1_R64_R", "TC1_R64_ALPHA")


def test_tc1b_knobs_are_read_by_the_box_and_forwarded_by_the_driver():
    """TC1b: every knob the curve / t1 / r64 sub-fixtures read from the environment is in tc1_drive.sh's forwarded list (the grep the
    instruction asks for), and the box script reads each one with its registered default."""
    run, drive = RUN_SH.read_text(), DRIVE_SH.read_text()
    forwarded_block = drive[drive.index("for v in TC1_FAMILIES"):drive.index("; do", drive.index("for v in TC1_FAMILIES"))]
    for knob, default in zip(TC1B_KNOBS, ("200", "40", "16", "1", "1", "64", "64")):
        assert re.search(rf"\${{{knob}:-{default}}}", run), (knob, default, "the box script does not read it with the registered default")
        assert knob in forwarded_block.split(), (knob, "read by tc1_run.sh but not forwarded by tc1_drive.sh")
    # the anchor fixture is tp4's literals (byte-for-byte), not knobs; its eval instrument is what tp4 RAN (8 rows), stated in the script
    assert "A_STEPS=60; A_SEQ=512; A_MB=1; A_ACCUM=1; A_R=8; A_ALPHA=16; A_LR=1e-4; A_WD=0.01; A_WARMUP=0; A_SCHED=constant; A_OPTIM=adamw_torch; A_SEED=0; A_TEMPLATE=clinical" in run
    assert "A_EVAL_N=8; A_EVAL_EVERY=20" in run and "amendment 3" in run
    assert not re.search(r"\$\{TC1_A_", run), "the anchor literals must not become knobs"


def test_tc1_drive_stages_the_clinical_builder_and_manifest_for_tc1b():
    """TC1b: tc1_drive.sh's STAGE list carries n9_datasets.py and ds_manifest.json from bench/flagship-matrix (referenced, never copied into
    bench/tc1), and tc1_run.sh refuses to start without them and builds/verifies the clinical dataset exactly as tp4_run.sh did."""
    drive, run, tp4 = DRIVE_SH.read_text(), RUN_SH.read_text(), (REPO / "bench" / "tp4" / "tp4_run.sh").read_text()
    stage = re.search(r'^STAGE="(.*)"$', drive, re.M).group(1).split()
    assert "$REPO/bench/flagship-matrix/drivers/n9_datasets.py" in stage and "$REPO/bench/flagship-matrix/ds_manifest.json" in stage, stage
    assert "$REPO/bench/tp4/tp4_alpaca.py" in stage and not (REPO / "bench" / "tc1" / "n9_datasets.py").exists() and not (REPO / "bench" / "tc1" / "ds_manifest.json").exists()
    assert (REPO / "bench" / "flagship-matrix" / "drivers" / "n9_datasets.py").is_file() and (REPO / "bench" / "flagship-matrix" / "ds_manifest.json").is_file()
    assert "for f in tc1_arm.py tc1_reduce.py tp4_alpaca.py n9_datasets.py ds_manifest.json; do [ -s $W/$f ]" in run
    # the clinical build + sha check are tp4_run.sh's lines, gated on the qwen3curve token
    for line in ("(cd $W/data && $PY_E4B $W/n9_datasets.py $W/data > $W/logs/dataset_clinical.log 2>&1); tail -1 logs/dataset_clinical.log",
                 "CLIN_SHA=$($PY_E4B -c \"import json; print(json.load(open('$W/ds_manifest.json'))['clinical']['sha256'])\")",
                 "GOT=$(sha256sum $W/data/ds_clinical.json | awk '{print $1}'); [ \"$GOT\" = \"$CLIN_SHA\" ] || { say \"DATASET MISMATCH clinical: $GOT != $CLIN_SHA\"; finish 13; }"):
        assert line in run and line in tp4, line
    gate = run[run.index('case " $FAMILIES " in *" qwen3curve "*)      # TC1b: the anchor pair'):]
    assert gate.index("n9_datasets.py $W/data") < gate.index("esac")


def test_tc1_run_sh_runs_the_curve_family_in_the_registered_order():
    """TC1b (TC1B-PREREG 'Arms, in this order'): tc1_curve_family's invocations in order with the registered recipes, flags, venvs and
    alarms; the matched pair first; the anchor pair byte-for-byte tp4's arms (native e4b precision, tp4's fp32 cast on Unsloth, native
    init, the loader's double-quant, tp4's targets and loader) with the venv difference recorded by --note; the plan line's alarms."""
    body = RUN_SH.read_text()
    cf = body[body.index("tc1_curve_family(){"):body.index("# tc1_family FAM MID REV")]
    calls = re.findall(r"(?:arm|draw2|todo_arm)\s+\$FAM\s+(e4b|unsloth|hf|axolotl)\s+(\S+)", cf)
    assert calls == [("e4b", "fused_attn4_m_200"), ("unsloth", "ckpt_unsloth_m_200"), ("e4b", "fused_attn4_shipped_200"),
                     ("e4b", "fused_attn4_p38"), ("unsloth", "ckpt_unsloth_p38"), ("unsloth", "ckpt_unsloth_p38_t28"),
                     ("e4b", "fused_attn4_m_t1"), ("unsloth", "ckpt_unsloth_m_t1"), ("e4b", "fused_attn4_m_r64"), ("unsloth", "ckpt_unsloth_m_r64")], calls
    assert "draw2 $FAM" not in cf and "todo_arm $FAM" not in cf
    assert 'tc1_prepare $FAM "$MID" $REV $FAL "$ALL" $CURVE_EVAL_N' in cf          # the family's tokens file carries the 16 held-out rows
    assert 'MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"' in cf and 'NATIVE="--adapter-dtype native --lora-init native"' in cf
    assert 'ANCHOR_E4B="--adapter-dtype native --lora-init native" ANCHOR_UNS="--adapter-dtype fp32 --lora-init native --unsloth-double-quant default"' in cf
    assert re.search(r"fused_attn4_m_200 fused \$EAL .* 0 curve \$TOK \$TS --attn-4bit 1 \$MATCH$", cf, re.M)
    assert re.search(r"ckpt_unsloth_m_200 unsloth \$UAL .* 0 curve \$TOK \$TS \$UNS --unsloth-moe-backend grouped_mm \$MATCH$", cf, re.M)
    assert re.search(r"fused_attn4_shipped_200 fused \$EAL .* 0 curve \$TOK \$TS --attn-4bit 1 \$NATIVE$", cf, re.M)
    assert re.search(r"fused_attn4_p38 fused \$AAL .* 0 anchor \$ATOK \$ATS --attn-4bit 1 \$ANCHOR_E4B$", cf, re.M)
    assert re.search(r"ckpt_unsloth_p38 unsloth \$AAL .* 0 anchor \$ATOK \$ATS \$UNS --unsloth-moe-backend grouped_mm \$ANCHOR_UNS \\\n\s+--note \"tp4's anchor arm EXCEPT the venv", cf)
    assert re.search(r"UNS_VENV=t28 arm \$FAM unsloth ckpt_unsloth_p38_t28 unsloth \$AAL .* 0 anchor \$ATOK \$ATS \$UNS --unsloth-moe-backend default \$ANCHOR_UNS \\\n\s+--note \"byte-for-byte tp4's anchor arm", cf)
    assert re.search(r"fused_attn4_m_t1 fused \$SAL .* 0 t1 \$TOK \$TS --attn-4bit 1 \$MATCH$", cf, re.M) and re.search(r"ckpt_unsloth_m_t1 unsloth \$SAL .* 0 t1 .* --unsloth-moe-backend grouped_mm \$MATCH$", cf, re.M)
    assert re.search(r"fused_attn4_m_r64 fused \$SAL .* 0 r64 \$TOK \$TS --attn-4bit 1 \$MATCH$", cf, re.M) and re.search(r"ckpt_unsloth_m_r64 unsloth \$SAL .* 0 r64 .* --unsloth-moe-backend grouped_mm \$MATCH$", cf, re.M)
    # the anchor's tokens: tp2's text through tp4's tokenise call (clinical, A_SEQ, A_EVAL_N rows), harness_error stubs for all three if it fails
    assert 'tokenise $FAM "$MID" $REV clinical $A_SEQ $W/data/ds_clinical.json $CLIN_SHA $ATOK $A_EVAL_N' in cf
    assert cf.count("harness_error") == 3 and "ckpt_unsloth_p38_t28 unsloth harness_error" in cf
    # the plan line: the token, the pin, and the registered alarms (fetch 5400, e4b 200-step 4800, Unsloth 200-step 9000, anchor 1800, t1/r64 3600)
    assert "qwen3curve)  tc1_curve_family  qwen3curve  Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 4800 9000 1800 3600;;" in body
    assert "PREREG=tc1/TC1B-PREREG.md" in body and 'TC1_PREREG:-}" ] || PREREG=tc1/TC1B-PREREG.md' in body
    # the existing families' slices are untouched by the new function (it sits before tc1_family's comment block)
    assert body.index("tc1_curve_family(){") < body.index("# tc1_family FAM MID REV") < body.index("tc1_family(){") < body.index("tc1_native_family(){")


def test_curve_recipe_overrides_execute_through_bash():
    """The RECIPE case lines out of the real file, executed: each sub-fixture overrides exactly the knobs the registration names and
    names its own e4b arm for --expect-trainable (a fixture-scoped override that does not cover a knob is not a fixture: tp4 amendment 3)."""
    body = RUN_SH.read_text()
    fx = body[body.index("STEPS=${TC1_STEPS:-20}"):body.index("A_EVAL_N=8; A_EVAL_EVERY=20\n") + len("A_EVAL_N=8; A_EVAL_EVERY=20\n")]
    fx = "\n".join(ln for ln in fx.splitlines() if ln and not ln.startswith("#"))
    init = re.search(r"^\s*local (s=\$STEPS q=\$SEQ .*ex_tag=fused_attn4_m)$", body, re.M).group(1)
    case = body[body.index('  case "$RECIPE" in'):]
    case = case[:case.index("  esac\n") + len("  esac\n")]
    want = {"field": "s=20 q=2048 m=2 ac=4 r=16 al=16 lr=2e-4 wd=0.001 wu=5 sc=linear op=adamw_8bit sd=3407 en=8 ee=20 ex_tag=fused_attn4_m",
            "curve": "s=200 q=2048 m=2 ac=4 r=16 al=16 lr=2e-4 wd=0.001 wu=5 sc=linear op=adamw_8bit sd=3407 en=16 ee=40 ex_tag=fused_attn4_m_200",
            "anchor": "s=60 q=512 m=1 ac=1 r=8 al=16 lr=1e-4 wd=0.01 wu=0 sc=constant op=adamw_torch sd=0 en=8 ee=20 ex_tag=fused_attn4_p38",
            "t1": "s=20 q=2048 m=1 ac=1 r=16 al=16 lr=2e-4 wd=0.001 wu=5 sc=linear op=adamw_8bit sd=3407 en=8 ee=20 ex_tag=fused_attn4_m_t1",
            "r64": "s=20 q=2048 m=2 ac=4 r=64 al=64 lr=2e-4 wd=0.001 wu=5 sc=linear op=adamw_8bit sd=3407 en=8 ee=20 ex_tag=fused_attn4_m_r64",
            "mb1": "s=20 q=2048 m=1 ac=8 r=16 al=16 lr=2e-4 wd=0.001 wu=5 sc=linear op=adamw_8bit sd=3407 en=8 ee=20 ex_tag=fused_attn4_m_mb1"}
    env = {k: v for k, v in os.environ.items() if not k.startswith("TC1_")}
    for recipe, exp in want.items():
        script = "\n".join([fx, f"RECIPE={recipe}", init, case, 'echo "s=$s q=$q m=$m ac=$ac r=$r al=$al lr=$lr wd=$wd wu=$wu sc=$sc op=$op sd=$sd en=$en ee=$ee ex_tag=$ex_tag"'])
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
        assert r.returncode == 0 and r.stdout.strip() == exp, (recipe, r.stdout, r.stderr)


def test_unsloth_knobs_and_double_quant_decision(monkeypatch):
    """P2-1: the knobs reach the environment only for an Unsloth arm, `default` leaves it alone, and the double-quant kwarg
    follows the from_pretrained signature read at runtime."""
    import types
    arm = _load_arm_module()
    for k in ("UNSLOTH_MOE_BACKEND", "UNSLOTH_MOE_RECOMPUTE", "UNSLOTH_MOE_GC_REPLAY_PIN"):
        monkeypatch.delenv(k, raising=False)
    out = arm.apply_unsloth_knobs(types.SimpleNamespace(framework="unsloth", unsloth_moe_backend="unsloth_triton", unsloth_speed_tilt=1, unsloth_double_quant="off"))
    assert os.environ["UNSLOTH_MOE_BACKEND"] == "unsloth_triton" and os.environ["UNSLOTH_MOE_RECOMPUTE"] == "0" and os.environ["UNSLOTH_MOE_GC_REPLAY_PIN"] == "1"
    assert out["env_set"] == {"UNSLOTH_MOE_BACKEND": "unsloth_triton", "UNSLOTH_MOE_RECOMPUTE": "0", "UNSLOTH_MOE_GC_REPLAY_PIN": "1"} and out["speed_tilt"] is True
    for k in ("UNSLOTH_MOE_BACKEND", "UNSLOTH_MOE_RECOMPUTE", "UNSLOTH_MOE_GC_REPLAY_PIN"):
        monkeypatch.delenv(k, raising=False)
    out = arm.apply_unsloth_knobs(types.SimpleNamespace(framework="unsloth", unsloth_moe_backend="default", unsloth_speed_tilt=0, unsloth_double_quant="off"))
    assert out["env_set"] == {} and "UNSLOTH_MOE_BACKEND" not in os.environ
    out = arm.apply_unsloth_knobs(types.SimpleNamespace(framework="e4b", unsloth_moe_backend="grouped_mm", unsloth_speed_tilt=1))
    assert out["env_set"] == {} and "UNSLOTH_MOE_BACKEND" not in os.environ                 # not an Unsloth arm: nothing touched

    def named(model_name, max_seq_length, dtype, load_in_4bit, bnb_4bit_use_double_quant=True):
        pass

    def neither(model_name, **kwargs):
        pass
    kw, dq = arm.unsloth_double_quant_kwargs(named, True)
    assert kw == {"bnb_4bit_use_double_quant": False} and dq["requested"] is False
    kw, dq = arm.unsloth_double_quant_kwargs(neither, True)
    assert kw == {} and dq["how"].startswith("unknown-default")
    assert arm.unsloth_double_quant_kwargs(named, False)[0] == {}


def test_axolotl_config_dict_and_census():
    """P2-2: the config builder is pure and carries the spec's keys; the census counts real bnb parametrizations (CPU)."""
    import types

    import torch
    import torch.nn as nn
    pytest.importorskip("bitsandbytes")
    arm = _load_arm_module()
    a = types.SimpleNamespace(r=16, alpha=16, seq=2048, seed=3407, micro_batch=2, accum=4, steps=20, lr=2e-4, weight_decay=0.001, warmup_steps=5)
    mods = ["model.layers.0.self_attn.q_proj"]
    params = ["model.layers.0.mlp.experts.gate_up_proj", "model.layers.0.mlp.experts.down_proj"]
    c = arm.axolotl_config_dict(a, "/snap/qwen3", mods, params, best=False, double_quant=False, dataset_path="/root/tc1/data/ds_alpaca.json")
    for k, v in (("load_in_4bit", True), ("adapter", "qlora"), ("quantize_moe_experts", True), ("bnb_4bit_use_double_quant", False), ("lora_r", 16), ("lora_alpha", 16),
                 ("lora_dropout", 0.0), ("gradient_checkpointing", True), ("bf16", True), ("optimizer", "adamw_bnb_8bit"), ("lr_scheduler", "linear"),
                 ("micro_batch_size", 2), ("gradient_accumulation_steps", 4), ("max_steps", 20), ("learning_rate", 2e-4), ("weight_decay", 0.001), ("warmup_steps", 5), ("seed", 3407), ("sequence_len", 2048)):
        assert c[k] == v, (k, c.get(k))
    assert c["lora_target_modules"] == mods and c["lora_target_parameters"] == params and c["datasets"][0]["path"] == "/root/tc1/data/ds_alpaca.json" and "plugins" not in c
    assert c["gradient_checkpointing_kwargs"] == {"use_reentrant": False} and c["bnb_config_kwargs"] == {"bnb_4bit_use_double_quant": False}
    b = arm.axolotl_config_dict(a, "/snap/qwen3", mods, params, best=True)
    assert b["plugins"] == ["axolotl.integrations.kernels.KernelsPlugin"] and b["expert_backend"] == "scattermoe" and b["moe_bnb_fast"] is True
    from bitsandbytes.nn.parametrize import replace_parameter_4bit

    class Stack(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(4, 24, 16, dtype=torch.bfloat16))
            self.down_proj = nn.Parameter(torch.randn(4, 16, 12, dtype=torch.bfloat16))
    m = nn.Module()
    m.layers = nn.ModuleList([nn.Module(), nn.Module()])
    for i, blk in enumerate(m.layers):
        blk.mlp = nn.Module()
        blk.mlp.experts = Stack()
        try:
            replace_parameter_4bit(blk.mlp.experts, "gate_up_proj", compress_statistics=False, quant_type="nf4", blocksize=64)
        except Exception as e:
            pytest.skip(f"bnb replace_parameter_4bit unavailable on CPU here: {e}")
        if i == 0:
            replace_parameter_4bit(blk.mlp.experts, "down_proj", compress_statistics=False, quant_type="nf4", blocksize=64)
    c = arm.axolotl_expert_census(m)
    assert c == {"quantized_moe_experts_n": 3, "parametrized_params": 3, "n_experts_modules": 2, "n_bnb4bit_unwrapped": 1, "samples": ["layers.0.mlp.experts.gate_up_proj", "layers.0.mlp.experts.down_proj", "layers.1.mlp.experts.gate_up_proj"]}, c
    assert arm.logical_stack_shape(m.layers[0].mlp.experts, "gate_up_proj") == (4, 24, 16)      # from the parametrization's quant_state
    assert arm.quant_state_of(m.layers[0].mlp.experts, "gate_up_proj").blocksize == 64


def test_load_e4b_records_its_own_phases(monkeypatch):
    import types

    import torch.nn as nn
    arm = _load_arm_module()
    calls = []

    class _Proj(nn.Linear):
        pass

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(self, p, _Proj(4, 4, bias=False))

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Layer(), _Layer()])
            self.config = types.SimpleNamespace(use_cache=True, num_hidden_layers=2, model_type="fake_moe")

        def gradient_checkpointing_enable(self, **kw):
            calls.append("ckpt")

    model = _Model()

    def _slow(name, secs, ret=None):
        def _fn(*a, **kw):
            calls.append(name)
            time.sleep(secs)
            return ret
        return _fn

    e4b = types.ModuleType("experts4bit_qlora")
    e4b.__version__ = "0.0-test"
    e4b.load_moe_4bit_streaming = _slow("load", 0.05, (model, model.config))
    e4b.verify_moe_4bit = _slow("verify", 0.05, {"n_quantized": 2, "n_unquantized": 0})
    e4b.enable_fast_train = _slow("enable", 0.05, 2)
    e4b.disable_fast_train = _slow("disable_fast", 0.0)
    e4b.enable_batched_train = _slow("enable_batched", 0.0, 0)
    e4b.disable_batched_train = _slow("disable_batched", 0.0)
    lora = types.ModuleType("experts4bit_qlora.lora")
    lora.DETECTOR_VERSION = "test"
    lora.add_attention_lora = _slow("lora", 0.05)
    lora.detect_attention_projections = _slow("detect", 0.0)
    lora.quantize_attention_projections_4bit = _slow("quantize", 0.0, 0)
    e4b.lora = lora
    tf = types.ModuleType("transformers")
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=_slow("tokenizer", 0.05, object()))
    for name, mod in (("experts4bit_qlora", e4b), ("experts4bit_qlora.lora", lora), ("transformers", tf)):
        monkeypatch.setitem(sys.modules, name, mod)
    a = types.SimpleNamespace(model="fake/model", revision="0" * 40, r=8, alpha=16, offload=1, attn_4bit=0, arm="fused")
    arm.PH.reset()
    arm.PH.begin(time.perf_counter())
    _model, x = arm.load_e4b(a)
    ph = arm.PH.report()["phase_seconds"]
    assert list(ph) == ["load_weights", "verify", "attn4", "lora", "enable", "tokenizer"], ph
    for k in ("load_weights", "verify", "lora", "enable", "tokenizer"):
        assert ph[k] >= 0.04, (k, ph)
    assert x["n_patched"] == 2 and calls.count("enable") == 1


def test_arm_env_prefix_actually_executes():
    """The arm's env prefix must RUN (tp4's P56 draw-1 lesson): the real prefix out of the real file, executed."""
    sh = RUN_SH.read_text()
    m = re.search(r"^(\s*env \$ARM_ENV .*?TC1_ARM_ALARM_S=\$A) perl", sh, re.M)
    assert m, "the arm's env prefix is not in the shape this test knows how to drive"
    prefix = m.group(1).strip()
    for arm, want in (("fused", ""), ("batched", "64")):
        script = f"""
        ARM={arm}; GPU_CLASS=5090; A=3600
        ARM_ENV=""; [ "$ARM" = batched ] && ARM_ENV="E4B_BATCHED_PAD_WASTE_LIMIT=64"
        PHYS=4; BOX_CLASS="RTX 5090"
        {prefix} /bin/sh -c 'echo RAN box="$TC1_BOX_CLASS" alarm="$TC1_ARM_ALARM_S" pad="$E4B_BATCHED_PAD_WASTE_LIMIT" omp="$OMP_NUM_THREADS"'
        """
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert r.returncode == 0 and "RAN" in r.stdout and 'box=RTX 5090' in r.stdout and "alarm=3600" in r.stdout and f"pad={want}" in r.stdout and "omp=4" in r.stdout, (arm, r.stdout, r.stderr)


# ----------------------------------------------------------------------------- lane TC3 (the frontier tokens, the hand-run block, the lever helpers)
def test_tc1_run_sh_runs_the_frontier_families_in_the_registered_order():
    """TC3 (TC3-PREREG-draft 'Arms'): the 24 GB token's eleven arms in order with their lever flags and the draft's alarms; the 12 GB token's seven with the
    second offload draw, the t28 Unsloth venv and the loader-default backend; the deepspeed extra only under the 24 GB token and as a separate non-fatal step;
    the axolotl venv install no longer the uv first-index line that failed on both TC1 boxes; the local-snapshot path; TC1's slices untouched."""
    body = RUN_SH.read_text()
    ff = body[body.index("tc1_frontier_family(){"):body.index("# tc1_frontier12_family FAM MID REV")]
    f12 = body[body.index("tc1_frontier12_family(){"):body.index("# tc2_small_family FAM MID REV")]
    pat = r"(?:arm|draw2|todo_arm)\s+\$FAM\s+(e4b|unsloth|hf|axolotl)\s+(\S+)"
    assert re.findall(pat, ff) == [("e4b", "fused_attn4_m"), ("e4b", "fused_attn4_m_offload"), ("e4b", "fused_attn4_m_mb1"), ("e4b", "fused_attn4_shipped"), ("unsloth", "ckpt_unsloth_m"), ("unsloth", "ckpt_unsloth_m_mb1"),
                                   ("hf", "hf_peft_m"), ("hf", "hf_peft_m_offload"), ("axolotl", "ckpt_axolotl_m"), ("axolotl", "ckpt_axolotl_m_layeroffload"),
                                   ("axolotl", "ckpt_axolotl_m_zero3"), ("e4b", "reference_attn4_m_offload")], re.findall(pat, ff)
    assert "draw2 $FAM" not in ff and "todo_arm" not in ff and 'MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"' in ff
    assert re.search(r"fused_attn4_m fused \$ERAL .* 0 field \$TOK \$TS --attn-4bit 1 \$MATCH", ff) and re.search(r"fused_attn4_m_offload fused \$EOAL .* 1 field \$TOK \$TS --attn-4bit 1 \$MATCH", ff)
    assert re.search(r"fused_attn4_m_mb1 fused \$MBAL .* 0 mb1 \$TOK \$TS --attn-4bit 1 \$MATCH", ff)
    assert re.search(r"ckpt_unsloth_m unsloth \$UAL .* 0 field .* --unsloth-moe-backend grouped_mm \$MATCH", ff) and re.search(r"ckpt_unsloth_m_mb1 unsloth \$UAL .* 0 mb1 .* --unsloth-moe-backend grouped_mm \$MATCH", ff)
    assert re.search(r"hf_peft_m hf \$HAL .* 0 field \$TOK \$TS \$MATCH", ff) and re.search(r"hf_peft_m_offload hf \$HOAL .* 0 field \$TOK \$TS --hf-offload 1 \$MATCH", ff)
    assert re.search(r"ckpt_axolotl_m axolotl \$AAL .* 0 field .* --axolotl-dataset \$W/data/ds_alpaca.json \$MATCH", ff)
    assert re.search(r"ckpt_axolotl_m_layeroffload axolotl \$ALAL .* --axolotl-layer-offload 1 \$MATCH", ff) and re.search(r"ckpt_axolotl_m_zero3 axolotl \$AZAL .* --axolotl-zero3 1 \$MATCH", ff)
    assert re.search(r"reference_attn4_m_offload reference \$ROAL .* 1 field \$TOK \$TS --attn-4bit 1 \$MATCH", ff) and "can_run 900 $FAM/e4b/reference_m_offload" in ff
    assert re.findall(pat, f12) == [("e4b", "fused_attn4_m_offload"), ("e4b", "fused_attn4_m_offload"), ("e4b", "reference_attn4_m_offload"), ("e4b", "fused_attn4_m"),
                                    ("e4b", "fused_attn4_m_offload_mb1"), ("e4b", "reference_attn4_m_offload_mb1"), ("e4b", "fused_attn4_shipped_offload"),
                                    ("unsloth", "ckpt_unsloth_m_mb1"), ("hf", "hf_peft_m_mb1"), ("axolotl", "ckpt_axolotl_m")], re.findall(pat, f12)
    # the 12 GB secondary runs only on a field-recipe OOM (one draw each), the as-shipped offload row always; the plan line's alarms and the 12 h hand-run deadline
    assert 'so=$(status_of $FAM e4b fused_attn4_m_offload)' in f12 and 'if [ "$so" = oom ]; then' in f12 and f12.count("_mb1 fused $EOAL") == 1 and f12.count("_mb1 reference $ROAL") == 1
    assert re.search(r"fused_attn4_m_offload_mb1 fused \$EOAL .* 1 mb1 \$TOK \$TS --attn-4bit 1 \$MATCH", f12) and re.search(r"fused_attn4_shipped_offload fused \$EOAL .* 1 field \$TOK \$TS --attn-4bit 1 \$NATIVE", f12)
    assert 'TC1_DEADLINE_EPOCH=${TC1_DEADLINE_EPOCH:-$(( $(date +%s) + ${TC1_LOCAL_HOURS:-12} * 3600 ))}' in body
    assert f12.count("draw2 $FAM e4b fused_attn4_m_offload fused $EOAL") == 1
    assert re.search(r"UNS_VENV=t28 arm \$FAM unsloth ckpt_unsloth_m_mb1 unsloth \$UAL .* 0 mb1 .* --unsloth-moe-backend default \$MATCH", f12)
    assert re.search(r"hf_peft_m_mb1 hf \$HAL .* 0 mb1 \$TOK \$TS \$MATCH", f12) and re.search(r"ckpt_axolotl_m axolotl \$AAL .* 0 field .* \$MATCH", f12)
    assert '${NOTE:+--note} ${NOTE:+"$NOTE"}' in ff and '${NOTE:+--note} ${NOTE:+"$NOTE"}' in f12 and 'NOTE="TC1_LOCAL_BOX hand run: snapshot ${TC1_LOCAL_SNAPSHOT:-?} (pin_proof: ${PIN_PROOF:-?})"' in f12
    # the plan lines: the tokens, the pin, the draft's alarms (e4b resident 1200, offload 3600, mb1 3600, Unsloth 3600, HF 1800 / 3600, axolotl 2700 / 3600 / 3600, reference offload 5400)
    assert "qwen3frontier)   tc1_frontier_family   qwen3frontier   Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 1200 3600 3600 3600 1800 3600 2700 3600 3600 5400;;" in body
    assert "qwen3frontier12) tc1_frontier12_family qwen3frontier12 Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 1200 7200 3600 1800 2700 14400;;" in body
    assert 'PREREG=tc1/TC3-PREREG.md' in body and '*" qwen3frontier "*|*" qwen3frontier12 "*)' in body
    # the axolotl venv: torch alone from the cu130 index, axolotl from PyPI under unsafe-best-match (the TC1 boxes' uv first-index failure), the deepspeed extra separate and non-fatal
    # the axolotl venv's install is TC1 amendment 3's registered single line (dry-resolved; torch 2.14.0+cu130 by axolotl's own pin) under the base interpreter;
    # TC3 keeps AX_PKG / AX_INDEX for its separate, non-fatal deepspeed step
    assert 'AX_PKG="axolotl==$AX_VER"; AX_INDEX="https://download.pytorch.org/whl/cu130"' in body and "AX_TORCH_PIN" not in body
    assert re.search(r'\$PY_BASE -m uv pip install --python \$PY_AX "axolotl==\$AX_VER" --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match > logs/pip_axolotl.log', body)
    assert 'AX_PKG="axolotl==$AX_VER"' in body and body.count("--index-strategy unsafe-best-match") >= 2      # the registered install line and the deepspeed step
    assert '"axolotl==$AX_VER" --extra-index-url https://download.pytorch.org/whl/cu130 > logs/pip_axolotl.log' not in body, "the uv line that failed on both TC1 boxes must not survive"
    assert '--no-build-isolation "axolotl[deepspeed]==$AX_VER"' in body and body.index('case " $FAMILIES " in *" qwen3frontier "*)\n    say "venv-axolotl + axolotl[deepspeed]') > body.index('pip(axolotl) rc=$rc')
    # the local snapshot replaces the fetch, the pin proof links the directory into the private cache, free_family keeps it; the frontier venvs from PY_BASE
    assert 'if [ -n "${TC1_LOCAL_SNAPSHOT:-}" ]; then local_snapshot $FAM $MID $REV; frc=$?; else fetch $FAM $MID $REV $FAL; frc=$?; fi' in body
    assert "local_snapshot(){" in body and 'ln -sfn "$SNAP" $RDIR/snapshots/$REV' in body and 'out["pin_proof"], out["offline_reason"] = "offline"' in body and "PIN_PROOF=$PROOF" in body
    assert "$PY_BASE -m venv --system-site-packages $W/venv-e4b" in body and "$PY_BASE -m venv $W/venv-unsloth-t28" in body and 'free_family(){ [ "$TC1_LOCAL_BOX" = 1 ]' in body
    assert 'for t in (tag, "attn_only_m", "reference_attn4_m", "fused_attn4_m_offload", "reference_attn4_m_offload"):' in body
    assert '[ "$CU130_OK" != 1 ] && [ "$TC1_LOCAL_BOX" != 1 ]' in body
    # the function block sits outside every TC1 / TC1b slice
    assert body.index("tc1_frontier_family(){") < body.index("# tc1_curve_family FAM MID REV") < body.index("tc1_curve_family(){") < body.index("tc1_family(){") < body.index("tc1_native_family(){")


def test_local_box_block_executes_through_bash(tmp_path):
    """TC3: the TC1_LOCAL_BOX block out of the real file, executed -- a hand run defaults the box id, the run id, the instance id, the deadline (now + 6 h),
    the nonce and the private HF cache under TC1_LOCAL_OUT; the launcher's path leaves everything exactly as before; TC1_LOCAL_BOX=1 without TC1_LOCAL_OUT refuses."""
    body = RUN_SH.read_text()
    blk = body[body.index("TC1_LOCAL_BOX=${TC1_LOCAL_BOX:-0}"):body.index("PY_BASE=${TC1_LOCAL_PYTHON:-python}")]
    env = {k: v for k, v in os.environ.items() if not k.startswith("TC1_")}
    script = "set -u\nLANE=tc1\n" + blk + 'echo "W=$W BOX=$TC1_BOX RUN=$TC1_RUN_ID IID=$TC1_INSTANCE_ID DL=$TC1_DEADLINE_EPOCH NONCE=$TC1_RUN_NONCE HF=$HF_HUB_CACHE TW=$TC1_W"'
    t0 = int(time.time())
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={**env, "TC1_LOCAL_BOX": "1", "TC1_LOCAL_OUT": str(tmp_path)})
    assert r.returncode == 0, (r.stdout, r.stderr)
    kv = dict(tok.split("=", 1) for tok in r.stdout.split())
    assert kv["W"] == str(tmp_path) and kv["BOX"] == "A" and kv["RUN"].startswith("local-") and kv["IID"].startswith("local:") and kv["NONCE"].startswith("local-"), kv
    assert t0 + 12 * 3600 - 5 <= int(kv["DL"]) <= t0 + 12 * 3600 + 60 and kv["HF"] == str(tmp_path) + "/hf-cache" and kv["TW"] == str(tmp_path), kv
    r = subprocess.run(["bash", "-c", "set -u\nLANE=tc1\n" + blk + 'echo "W=$W TW=$TC1_W BOX=${TC1_BOX:-unset} DL=${TC1_DEADLINE_EPOCH:-unset} HF=${HF_HUB_CACHE:-unset}"'], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and r.stdout.split() == ["W=/root/tc1", "TW=/root/tc1", "BOX=unset", "DL=unset", "HF=unset"], (r.stdout, r.stderr)
    r = subprocess.run(["bash", "-c", "set -u\nLANE=tc1\n" + blk], capture_output=True, text=True, env={**env, "TC1_LOCAL_BOX": "1"})
    assert r.returncode == 78 and "TC1_LOCAL_OUT" in r.stdout, (r.returncode, r.stdout)


def test_tc3_arm_helpers_and_flags():
    """TC3: the pure helpers the lever arms use (the max_memory cap, the device-map summary, the config dict's lever keys, the ZeRO-3 dict, lever_of), the
    host-RAM report's shape, and the five flags with their defaults."""
    import types
    arm = _load_arm_module()
    assert arm.hf_max_memory(24 * (1 << 30), 125 * (1 << 30), 2) == {0: "22GiB", "cpu": "125GiB"} and arm.hf_max_memory(1 << 30, 1 << 29, 2) == {0: "1GiB", "cpu": "1GiB"}
    dm = arm.device_map_summary({"model.layers.0": 0, "model.layers.1.mlp.experts": "cpu", "lm_head": "disk"})
    assert dm == {"n_entries": 3, "by_device": {"cuda:0": 1, "cpu": 1, "disk": 1}, "experts_entries_by_device": {"cpu": 1}, "cpu_or_disk_sample": ["model.layers.1.mlp.experts", "lm_head"], "any_cpu_or_disk": True}
    assert arm.device_map_summary(None) is None
    a = types.SimpleNamespace(r=16, alpha=16, seq=2048, seed=3407, micro_batch=2, accum=4, steps=20, lr=2e-4, weight_decay=0.001, warmup_steps=5)
    z = arm.axolotl_zero3_config(a)
    assert z == {"zero_optimization": {"stage": 3, "offload_param": {"device": "cpu", "pin_memory": True}, "offload_optimizer": {"device": "cpu", "pin_memory": True}},
                 "bf16": {"enabled": True}, "train_micro_batch_size_per_gpu": 2, "gradient_accumulation_steps": 4}
    c = arm.axolotl_config_dict(a, "/snap", ["m"], ["p"], layer_offload=True, deepspeed=z, quantize_moe_experts=False)
    assert c["layer_offloading"] is True and c["deepspeed"] is z and c["quantize_moe_experts"] is False and c["load_in_4bit"] is True
    c0 = arm.axolotl_config_dict(a, "/snap", ["m"], ["p"])
    assert "layer_offloading" not in c0 and "deepspeed" not in c0 and c0["quantize_moe_experts"] is True
    assert arm.lever_of(types.SimpleNamespace(framework="hf", hf_offload=1)) == "hf_offload" and arm.lever_of(types.SimpleNamespace(framework="e4b", offload=1)) == "e4b_offload"
    assert arm.lever_of(types.SimpleNamespace(framework="axolotl", axolotl_zero3=1)) == "axolotl_zero3" and arm.lever_of(types.SimpleNamespace(framework="axolotl", axolotl_layer_offload=1)) == "axolotl_layer_offload"
    assert arm.lever_of(types.SimpleNamespace(framework="e4b", offload=0)) is None and set(arm.LEVER_LABELS) == {"e4b_offload", "hf_offload", "axolotl_layer_offload", "axolotl_zero3"}
    h = arm.host_ram_report()
    assert h["ru_maxrss_gb"] is not None and h["high_water_gb"] >= h["ru_maxrss_gb"] > 0 and set(h) >= {"rss_hwm_gb", "cgroup_peak_gb", "cgroup_limit_gb", "total_gb", "high_water_gb"}
    p = _run("--help")
    for flag in ("--hf-offload", "--hf-offload-gpu-margin-gib", "--hf-offload-fp32-cpu", "--axolotl-layer-offload", "--axolotl-zero3"):
        assert flag in p.stdout, flag


def test_run_and_drive_scripts_parse():
    for sh in (RUN_SH, DRIVE_SH):
        r = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
        assert r.returncode == 0, (sh, r.stderr)
    drive = DRIVE_SH.read_text()
    assert "GNF4_SHA=${GNF4_SHA:-846b512b905468c08f5748943d08769b572affa2}" in drive      # the v0.34.0 COMMIT, not the tag object
    assert '$REPO/bench/tp4/tp4_alpaca.py' in drive                                     # tp4's Alpaca builder is referenced, not copied
    assert "$REPO/bench/flagship-matrix/drivers/n9_datasets.py" in drive and "$HERE/n9_datasets.py" not in drive   # TC1b: the clinical builder likewise (referenced)


# ----------------------------------------------------------------------------- lane TC2 (the tc2small / tc2big tokens)
TC2_KNOBS = ("TC1_SMALL_STEPS", "TC1_SMALL_EVAL_N", "TC1_SMALL_EVAL_EVERY")


def test_tc2_knobs_are_read_by_the_box_and_forwarded_by_the_driver():
    """TC2: the small families' instrument knobs (N 60, 48 rows every 20 -- tp4's registration for them) are read by tc1_run.sh with those
    defaults and forwarded by tc1_drive.sh; box B exists on both sides and defaults to the tc2big token; the tokens default their PREREG."""
    run, drive = RUN_SH.read_text(), DRIVE_SH.read_text()
    forwarded_block = drive[drive.index("for v in TC1_FAMILIES"):drive.index("; do", drive.index("for v in TC1_FAMILIES"))]
    for knob, default in zip(TC2_KNOBS, ("60", "48", "20")):
        assert re.search(rf"\${{{knob}:-{default}}}", run), (knob, default, "the box script does not read it with the registered default")
        assert knob in forwarded_block.split(), (knob, "read by tc1_run.sh but not forwarded by tc1_drive.sh")
    assert 'case "$TC1_BOX" in A|B)' in run and 'case "$TC1_BOX" in A|B)' in drive
    assert 'A) FAMILIES=${TC1_FAMILIES:-"qwen3"};;' in run and 'B) FAMILIES=${TC1_FAMILIES:-"tc2big"};;' in run
    assert '[ -n "${TC1_PREREG:-}" ] || PREREG=tc1/TC2-PREREG.md' in run and 'case " $FAMILIES " in *" tc2small "*|*" tc2big "*|*" tc2mixtral "*|*" tc2qwen35off "*)' in run
    assert "small)  s=$SMALL_STEPS; en=$SMALL_EVAL_N; ee=$SMALL_EVAL_EVERY; ex_tag=fused_attn4_m;;" in run


def test_tc2_run_sh_runs_the_registered_arm_order_with_the_flags():
    """TC2 (TC2-PREREG-draft 'Arms per family'): tc2_small_family's two modes and tc2_big_family's invocations in order with the registered
    recipes, flags, venvs and stubs; the plan tables carry the pins, tp4's ceilings (axolotl = hf + 900) and the target lists; TC1's own
    function is untouched."""
    body = RUN_SH.read_text()
    small = body[body.index("tc2_small_family(){"):body.index("# tc2_big_family")]
    big = body[body.index("tc2_big_family(){"):body.index("# tc2_small_box / tc2_big_box")]
    go = small[small.index('if [ "$MODE" = gptoss ]; then\n    stubw'):small.index("  else\n    can_run 600 $FAM/e4b/fused_m")]
    normal = small[small.index("  else\n    can_run 600 $FAM/e4b/fused_m"):]
    calls = r"(?:arm|draw2)\s+\$FAM\s+(e4b|unsloth|hf|axolotl)\s+(\S+)"
    assert re.findall(calls, normal) == [
        ("e4b", "fused_attn4_m"), ("hf", "hf_peft_m"), ("e4b", "reference_attn4_m"), ("e4b", "fused_attn4_m"), ("hf", "hf_peft_m"), ("unsloth", "ckpt_unsloth_m"),
        ("unsloth", "ckpt_unsloth_m_experts"), ("hf", "hf_peft_m_t214"), ("axolotl", "ckpt_axolotl_m"), ("axolotl", "ckpt_axolotl_best"), ("e4b", "fused_attn4_shipped")]
    assert normal.count("draw2 $FAM") == 2 and "todo_arm" not in small
    assert re.findall(calls, go) == [("e4b", "attn_only_m"), ("e4b", "attn_only_m"), ("unsloth", "ckpt_unsloth_m"), ("unsloth", "ckpt_unsloth_mxfp4"),
                                     ("unsloth", "ckpt_unsloth_mxfp4"), ("hf", "hf_peft_m"), ("axolotl", "ckpt_axolotl_m")]
    assert re.findall(r"stubw \$FAM e4b (\S+) (\S+) refused", go) == [("fused_attn4_m", "fused"), ("reference_attn4_m", "reference")]
    assert '\'{"cited": "tp1,tp2", "n_patched": 0}\'' in go and '\'{"cited": "tp4", "n_patched": 0}\'' in go and go.index("stubw") < go.index("arm   $FAM e4b attn_only_m")
    assert re.search(r"arm   \$FAM e4b attn_only_m attn_only \$EAL \"\$MID\" \$REV 0 small \$TOK \$TS --attn-4bit 0 \$MATCH", go)
    assert re.search(r"draw2 \$FAM e4b attn_only_m attn_only \$EAL .* 0 small .* --attn-4bit 0 \$MATCH", go)
    assert re.search(r"ckpt_unsloth_m unsloth \$UAL .* 0 small .* \$UNS --unsloth-moe-backend grouped_mm --unsloth-load-in-4bit 1 \$MATCH", go)
    assert re.search(r"arm   \$FAM unsloth ckpt_unsloth_mxfp4 unsloth \$UAL .* 0 small .* \$UNS --unsloth-moe-backend grouped_mm --unsloth-load-in-4bit 0 \$MATCH", go)
    assert re.search(r"draw2 \$FAM unsloth ckpt_unsloth_mxfp4 unsloth \$UAL .* --unsloth-load-in-4bit 0 \$MATCH", go)
    assert re.search(r"fused_attn4_m fused \$EAL \"\$MID\" \$REV 0 small \$TOK \$TS --attn-4bit 1 \$MATCH", normal) and re.search(r"reference_attn4_m reference \$RAL .* 0 small .* --attn-4bit 1 \$MATCH", normal)
    assert re.search(r'ckpt_unsloth_m unsloth \$UAL .* 0 small .* \$UNS --unsloth-moe-backend grouped_mm \$MATCH', normal)
    assert re.search(r'\[ -n "\$UT2" \] && can_run 600 \$FAM/unsloth/m_experts && arm \$FAM unsloth ckpt_unsloth_m_experts unsloth \$UAL .* --grad-ckpt unsloth --unsloth-targets "\$UT2" --unsloth-moe-backend grouped_mm \$MATCH', normal)
    assert re.search(r"HF_VENV=t214 arm \$FAM hf hf_peft_m_t214 hf \$HAL .* 0 small .* --hf-experts-implementation grouped_mm \$MATCH", normal)
    assert re.search(r"ckpt_axolotl_best axolotl \$AAL .* --axolotl-best 1 --adapter-dtype fp32 --lora-init native", normal) and re.search(r"fused_attn4_shipped fused \$EAL .* 0 small .* \$NATIVE", normal)
    assert 'tc1_prepare $FAM "$MID" $REV $FAL "$ALL" $SMALL_EVAL_N' in small and 'MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"' in small
    assert re.findall(calls, big) == [
        ("e4b", "fused_attn4_m"), ("unsloth", "ckpt_unsloth_m"), ("e4b", "fused_attn4_m"), ("unsloth", "ckpt_unsloth_m"), ("unsloth", "ckpt_unsloth_m_experts"),
        ("hf", "hf_peft_m"), ("axolotl", "ckpt_axolotl_m"), ("axolotl", "ckpt_axolotl_best"), ("e4b", "fused_attn4_shipped"), ("e4b", "reference_attn4_m"),
        ("e4b", "fused_attn4_m_mb1"), ("unsloth", "ckpt_unsloth_m_mb1"), ("hf", "hf_peft_m_mb1")]
    assert re.search(r"fused_attn4_m fused \$EAL \"\$MID\" \$REV \$OFF field \$TOK \$TS --attn-4bit 1 \$MATCH", big) and big.count("draw2 $FAM") == 2
    assert re.search(r"ckpt_unsloth_m unsloth \$UAL \"\$MID\" \$REV 0 field \$TOK \$TS \$UNS --unsloth-moe-backend grouped_mm \$MATCH", big)
    assert re.search(r'ckpt_unsloth_m_experts unsloth \$UAL .* 0 field .* --grad-ckpt unsloth --unsloth-targets "\$UT2" \$UP2ARG --unsloth-moe-backend grouped_mm \$MATCH', big)
    assert 'local UP2ARG=""; [ -n "$UP2" ] && UP2ARG="--unsloth-target-parameters $UP2"' in big
    assert re.search(r"reference_attn4_m reference \$RAL \"\$MID\" \$REV \$OFF field .* --attn-4bit 1 \$MATCH", big) and big.index("reference_attn4_m reference") > big.index("fused_attn4_shipped fused")
    assert re.search(r"fused_attn4_shipped fused \$EAL \"\$MID\" \$REV \$OFF field .* \$NATIVE", big) and re.search(r"fused_attn4_m_mb1 fused \$EAL \"\$MID\" \$REV \$OFF mb1", big)
    assert 'UNS="--grad-ckpt unsloth --unsloth-targets $UT${TC2_UNS_TARGET_PARAMS:+ --unsloth-target-parameters $TC2_UNS_TARGET_PARAMS}"' in big and 'UNS="--grad-ckpt unsloth --unsloth-targets $UT7"' in small
    # the plan tables: the registered pins, tp4's ceilings (FETCH E4B UNS HF AX=HF+900 REF), gptoss MODE, granite's second list, qwen3_5's UT4 + explicit target_parameters, mixtral's offload
    assert 'tc2_small_family granite  ibm-granite/granite-3.1-3b-a800m-instruct a02780686e08a03fe0d2679a293b5c74a90efa89 1800 1800 1800 1800 2700 2400 normal "$UT_GRANITE2"' in body
    assert 'tc2_small_family olmoe    allenai/OLMoE-1B-7B-0924-Instruct         7f1c97f440f06ce36705e4f2b843edb5925f4498 2400 2400 2400 2400 3300 3000 normal ""' in body
    assert 'tc2_small_family gptoss   openai/gpt-oss-20b                        6cee5e81ee83917806bbde320786a8fb61efebee 3000 3600 2400 2400 3300 3600 gptoss ""' in body
    assert 'tc2_big_family   qwen3_5  Qwen/Qwen3.6-35B-A3B                      995ad96eacd98c81ed38be0c5b274b04031597b0 6000 3600 3600 1800 2700 5400 0 "$UT4" "$UT4" "$UP_QWEN3_5"' in body
    assert 'tc2_big_family   mixtral  mistralai/Mixtral-8x7B-Instruct-v0.1      eba92302a2861cdc0098cc54bc9f17cb2c47eb61 7200 5400 2400 1800 2700 6000 1 "$UT7" ""     ""' in body
    assert 'UT4="q_proj,k_proj,v_proj,o_proj"' in body and 'UT_GRANITE2="q_proj,k_proj,v_proj,o_proj,input_linear,output_linear"' in body
    assert 'UP_QWEN3_5="mlp.experts.gate_up_proj,mlp.experts.down_proj"' in body and 'UT7="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"' in body
    assert "tc2small)    tc2_small_box;;" in body and "tc2big)      tc2_big_box;;" in body and "tc2mixtral)  tc2_mixtral_redraw;;" in body and "tc2qwen35off) tc2_qwen35_offload;;" in body and 'for t in (tag, "attn_only_m", "reference_attn4_m", "fused_attn4_m_offload", "reference_attn4_m_offload"):' in body
    fam = body[body.index("tc1_family(){"):body.index("# tc1_native_family")]
    assert re.findall(calls, fam)[:3] == [("e4b", "fused_attn4_m"), ("unsloth", "ckpt_unsloth_m"), ("e4b", "reference_attn4_m")]   # TC1 untouched


def test_tc2_pure_helpers_and_knobs():
    """TC2 T23-T27: the target_parameters and experts_implementation decisions follow signatures / exceptions read at runtime, the knobs reach
    the receipt block, the attn_only stub tags carry the lane's suffix, and the self-decoding test keys on a class override (every tensor
    carries Tensor.dequantize())."""
    import types
    import torch
    arm = _load_arm_module()

    def gpm_with(model, r=16, target_modules=None, target_parameters=None):
        pass

    def gpm_without(model, r=16, target_modules=None):
        pass
    want = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    kw, info = arm.unsloth_target_parameters_kwargs(gpm_with, want)
    assert kw == {"target_parameters": want} and info == {"requested": want, "passed": True, "how": "get_peft_model(target_parameters=[...])"}
    kw, info = arm.unsloth_target_parameters_kwargs(gpm_without, want)
    assert kw == {} and info["passed"] is False and "does not name target_parameters" in info["how"]
    assert arm.unsloth_target_parameters_kwargs(gpm_without, [])[1]["how"] == "not requested"
    assert arm.unsloth_target_parameters_of(types.SimpleNamespace(unsloth_target_parameters="a, b,")) == ["a", "b"] and arm.unsloth_target_parameters_of(types.SimpleNamespace()) == []
    assert arm.unsloth_load_in_4bit_of(types.SimpleNamespace(unsloth_load_in_4bit=0)) is False and arm.unsloth_load_in_4bit_of(types.SimpleNamespace()) is True

    def fp_rejects(path, **kw):
        if "experts_implementation" in kw:
            raise TypeError("__init__() got an unexpected keyword argument 'experts_implementation'")
        return ("model", kw)
    m, info = arm.hf_from_pretrained_experts_impl(fp_rejects, "grouped_mm", "p", dtype=1)
    assert info["accepted"] is False and "experts_implementation" in info["error"] and m[1] == {"dtype": 1}
    m, info = arm.hf_from_pretrained_experts_impl(lambda path, **kw: ("model", kw), "grouped_mm", "p")
    assert info["accepted"] is True and m[1] == {"experts_implementation": "grouped_mm"}
    assert arm.attn_only_stub_tags("attn_only_m_d2") == ("fused_attn4_m", "reference_attn4_m") and arm.attn_only_stub_tags("attn_only") == ("fused_attn4", "reference_attn4")
    out = arm.apply_unsloth_knobs(types.SimpleNamespace(framework="e4b", unsloth_moe_backend="default", unsloth_speed_tilt=0, unsloth_double_quant="off", unsloth_load_in_4bit=0, unsloth_target_parameters="x,y"))
    assert out["load_in_4bit_requested"] is False and out["target_parameters_requested"] == ["x", "y"] and out["env_set"] == {}
    assert arm.self_decoding(torch.nn.Parameter(torch.zeros(2, dtype=torch.uint8), requires_grad=False)) is False

    class Packed(torch.nn.Parameter):
        def dequantize(self):
            return self.float()
    assert arm.self_decoding(Packed(torch.zeros(2, dtype=torch.uint8), requires_grad=False)) is True
    assert "unsloth_mxfp4_grouped_mm" in arm.UNSLOTH_PACKED_KEYS and arm.UNSLOTH_PACKED_FUNCS[0][:3] == ("unsloth_zoo.mxfp4_gemm", "Mxfp4GroupedMM", "apply")
    p = _run("--help")
    assert "--unsloth-load-in-4bit" in p.stdout and "--unsloth-target-parameters" in p.stdout


def test_amendment_3_axolotl_family_uv_index_strategy_and_no_unsloth_venv_on_that_token():
    """TC1-PREREG amendment 3: the axolotl rows re-asked on their own box (two draws of the matched pair, the native-best row, the HF t214 mb1
    row unconditionally), the uv install that reads PyPI past the cu130 index, and the Unsloth venvs skipped on that token alone."""
    body = RUN_SH.read_text()
    ax = body[body.index("tc1_axolotl_family(){"):body.index("# tc1_curve_family FAM MID REV")]
    calls = re.findall(r"(?:arm|draw2)\s+\$FAM\s+(e4b|unsloth|hf|axolotl)\s+(\S+)", ax)
    assert calls == [("e4b", "fused_attn4_m"), ("axolotl", "ckpt_axolotl_m"), ("e4b", "fused_attn4_m"), ("axolotl", "ckpt_axolotl_m"),
                     ("axolotl", "ckpt_axolotl_best"), ("hf", "hf_peft_m_mb1_t214")], calls
    assert ax.count("draw2 $FAM") == 2 and re.search(r"ckpt_axolotl_m axolotl \$AAL .* --axolotl-dataset \$W/data/ds_alpaca.json \$MATCH", ax)
    assert re.search(r"ckpt_axolotl_best axolotl \$AAL .* --axolotl-best 1 --adapter-dtype fp32 --lora-init native", ax)
    assert re.search(r"HF_VENV=t214 arm \$FAM hf hf_peft_m_mb1_t214 hf .* mb1 .* --hf-experts-implementation grouped_mm \$MATCH", ax) and "status_of" not in ax
    assert "qwen3axolotl) tc1_axolotl_family qwen3axolotl Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600 1800 2700;;" in body
    # the curve, matched and native families keep their registered call lists: the new function sits between the TC2 box tables and the curve family, outside every test slice
    assert body.index("tc2_big_box(){") < body.index("tc1_axolotl_family(){") < body.index("tc1_curve_family(){") < body.index("tc1_family(){") < body.index("tc1_native_family(){")
    # the uv install reads PyPI past the cu130 index: uv's first-index strategy left axolotl's packaging==26.0 unsatisfiable on both TC1 boxes
    assert re.search(r'uv pip install --python \$PY_AX "axolotl==\$AX_VER" --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match > logs/pip_axolotl.log', body)
    # the token alone builds no Unsloth venv; every other token still builds both, behind the same driver gate
    assert 'NEED_UNSLOTH=1; case " $FAMILIES " in " qwen3axolotl "|" qwen3nativebest200 "|" qwen3syncab "|" qwen3prof945 "|" qwen3leanab "|" qwen3tileab "|" qwen3rmsab ") NEED_UNSLOTH=0;; esac' in body   # amendments 8, 10, 12-15 add their tokens
    assert 'if [ "$NEED_UNSLOTH" = 1 ]; then\nUNS_T28_OK=1' in body and 'if [ "$CU130_OK" = 1 ] && [ "$NEED_UNSLOTH" = 1 ]; then' in body
    assert body.index("NEED_UNSLOTH=1; case") < body.index("venv-unsloth-t28:") and body.count("runs no Unsloth arm (TC1-PREREG amendment 3)") == 2


def test_tc3_amendment_1_every_venv_the_box_makes_upgrades_pip_first():
    """TC3-PREREG amendment 1: the owned box's `python -m venv` carries ensurepip's pip 22.0.2, which reads e4b / grouped-nf4-gemm as "unknown 0.0.0";
    the script upgrades pip in each venv it makes (logged, never fatal) before the install that needs it."""
    body = RUN_SH.read_text()
    assert 'pip_fresh(){ "$1" -m pip install -q --no-input -U pip > "logs/pip_upgrade_$2.log" 2>&1 ||' in body
    assert "venv-e4b || { say \"VENV FAIL (e4b)\"; finish 9; }" in body and body.index("pip_fresh $PY_E4B e4b") < body.index("pip_e4b.log")
    assert "venv-unsloth-t28 && pip_fresh $PY_UNS_T28 unsloth-t28 && perl" in body and "venv-unsloth && pip_fresh $PY_UNS unsloth && perl" in body
    assert body.index("pip_fresh(){") < body.index("pip_fresh $PY_E4B e4b")


def test_tc3_amendment_2_c1_hashes_the_offload_home_not_the_placeholder():
    """TC3 amendment 2: under e4b's expert offload the base's packed stacks and absmax buffers are 0-element GPU placeholders and the bytes
    live in `experts_lora._offload.home`; the hasher must read the home (no empties, the control flips the home copy) and the resident
    case is untouched."""
    import torch
    from torch import nn
    arm = _load_arm_module()

    class Base(nn.Module):
        def __init__(self, empty):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.empty(0, dtype=torch.uint8) if empty else torch.randint(0, 255, (4, 8), dtype=torch.uint8), requires_grad=False)
            self.down_proj = nn.Parameter(torch.empty(0, dtype=torch.uint8) if empty else torch.randint(0, 255, (4, 8), dtype=torch.uint8), requires_grad=False)
            self.register_buffer("gate_up_absmax", torch.empty(0) if empty else torch.rand(4))
            self.register_buffer("down_absmax", torch.empty(0) if empty else torch.rand(4))

    class Handle:
        def __init__(self, base):
            self.base = base
            self.home = {"gate_up_proj": torch.randint(0, 255, (4, 8), dtype=torch.uint8), "down_proj": torch.randint(0, 255, (4, 8), dtype=torch.uint8),
                         "gate_up_absmax": torch.rand(4), "down_absmax": torch.rand(4)}

    class ExpertsLoRA(nn.Module):
        def __init__(self, empty, offload):
            super().__init__()
            self.base = Base(empty)
            self.lora_A = nn.Parameter(torch.zeros(2, 2))
            if offload:
                self._offload = Handle(self.base)

    class Layer(nn.Module):
        def __init__(self, empty, offload):
            super().__init__()
            self.experts = ExpertsLoRA(empty, offload)

    class Model(nn.Module):
        def __init__(self, empty, offload):
            super().__init__()
            self.layers = nn.ModuleList([Layer(empty, offload), Layer(empty, offload)])

    off = Model(empty=True, offload=True)
    homes = arm.offload_homes(off)
    assert set(homes) == {f"layers.{i}.experts.base.{k}" for i in (0, 1) for k in ("gate_up_proj", "down_proj", "gate_up_absmax", "down_absmax")}, sorted(homes)
    ft = {n: t for n, t, _ in arm.frozen_tensors(off)}
    assert all(ft[n] is homes[n] for n in homes) and all(ft[n].numel() > 0 for n in homes), "the hasher must yield the home tensors"
    h, nbytes, empties, regimes = arm.hashes_frozen(off)
    assert empties == 0 and nbytes == 2 * (2 * 32 + 2 * 16) and regimes == {"u8-packed": 4, "fp32": 4}, (empties, nbytes, regimes)
    ctl = arm.c1_control(off, h)
    assert ctl["detects"] and ctl["tensor"] in homes, ctl
    # the same hasher on a RESIDENT model (no handle) reads the parameters themselves, as before
    res = Model(empty=False, offload=False)
    assert arm.offload_homes(res) == {}
    h2, nbytes2, empties2, _ = arm.hashes_frozen(res)
    assert empties2 == 0 and nbytes2 == nbytes and len(h2) == 8 and arm.c1_control(res, h2)["detects"]
    # an evicted model WITHOUT a handle is what the first hand run saw: every placeholder counts as an empty (the assertion in run_arm fires)
    _, _, empties3, _ = arm.hashes_frozen(Model(empty=True, offload=False))
    assert empties3 == 8


def test_tc3_amendment_4_an_exception_after_load_is_a_row():
    """TC3 amendment 4: a framework exception after load (tc3-4090-1's axolotl arm: a dtype mismatch in transformers' Qwen3-MoE router) is a
    `refused` row with its phase and traceback; a harness AssertionError (the first 12 GB arm's C1) is a `harness_error` row; a stub's own
    SystemExit propagates; nothing ever reads "rc=1 and no receipt" again."""
    import argparse
    import json
    import tempfile
    arm = _load_arm_module()
    d = tempfile.mkdtemp()
    a = argparse.Namespace(framework="axolotl", fam="qwen3frontier", model="Qwen/Qwen3-30B-A3B", revision="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
                           arm="axolotl", tag="ckpt_axolotl_m", steps=20, seq=2048, accum=4, micro_batch=2, offload=0, prereg="tc1/TC3-PREREG.md", out=d, note=None)

    def boom(a_, loader, sampler=True):
        raise RuntimeError("expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float")
    with pytest.raises(SystemExit) as ex:
        arm.run_arm_or_row(a, None, run=boom)
    assert ex.value.code == 3
    r = json.load(open(f"{d}/qwen3frontier_axolotl_ckpt_axolotl_m.json"))
    assert r["status"] == "refused" and r["exception_type"] == "RuntimeError" and "same dtype" in r["reason"] and "in phase" in r["reason"] and "boom" in r["traceback_tail"], r

    def c1(a_, loader, sampler=True):
        raise AssertionError("C1 saw 97 empty frozen tensors")
    a.framework, a.tag, a.arm = "e4b", "fused_attn4_m_offload", "fused"
    with pytest.raises(SystemExit) as ex:
        arm.run_arm_or_row(a, None, run=c1)
    assert ex.value.code == 10
    r = json.load(open(f"{d}/qwen3frontier_e4b_fused_attn4_m_offload.json"))
    assert r["status"] == "harness_error" and r["exception_type"] == "AssertionError" and "97 empty" in r["reason"], r

    def oom(a_, loader, sampler=True):
        raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
    a.tag = "fused_attn4_m"
    with pytest.raises(SystemExit) as ex:
        arm.run_arm_or_row(a, None, run=oom)
    assert ex.value.code == 5 and json.load(open(f"{d}/qwen3frontier_e4b_fused_attn4_m.json"))["status"] == "oom"

    def done(a_, loader, sampler=True):
        raise SystemExit(7)
    with pytest.raises(SystemExit) as ex:
        arm.run_arm_or_row(a, None, run=done)
    assert ex.value.code == 7
    assert arm.run_arm_or_row(a, None, run=lambda a_, loader_, sampler=True: "ok") == "ok"


def test_tc2_amendment_4_qwen35_offload_token():
    """TC2 amendment 4: `tc2qwen35off` runs Qwen3.6's matched set with every e4b arm under expert offload and the matched Unsloth arm
    (both draws) given the family's expert target parameters; HF, both axolotl arms and e4b as shipped are skipped as not_run stubs;
    the target-parameter knob is reset after the family so no other token inherits it."""
    run = RUN_SH.read_text()
    m = re.search(r"^tc2_qwen35_offload\(\)\{\n(.*?)^\}\n", run, re.DOTALL | re.MULTILINE)
    assert m, "tc2_qwen35_offload is gone"
    body = m.group(1)
    assert 'TC2_UNS_TARGET_PARAMS="$UP_QWEN3_5"' in body and body.rstrip().endswith('TC2_UNS_TARGET_PARAMS=""')
    call = re.search(r"tc2_big_family +qwen3_5 +Qwen/Qwen3\.6-35B-A3B +995ad96eacd98c81ed38be0c5b274b04031597b0 +(\d+) +(\d+) +(\d+) +(\d+) +(\d+) +(\d+) +(\d) +\"\$UT4\" +\"\" +\"\"", body)
    assert call, body
    _fetch, e4b, _uns, _hf, _ax, ref, off = call.groups()
    assert off == "1" and int(e4b) >= 3600 and int(ref) >= int(e4b), "every e4b arm under offload, with an alarm for the slower stream"
    for arm in ("qwen3_5/hf/hf_peft_m", "qwen3_5/axolotl/ckpt_axolotl_m", "qwen3_5/axolotl/ckpt_axolotl_best", "qwen3_5/e4b/fused_attn4_shipped"):
        assert arm in body, arm
    assert "qwen3_5/unsloth/ckpt_unsloth_m " not in body and "qwen3_5/e4b/reference_attn4_m" not in body, "the pair and the parity control run"
    assert 'tc2qwen35off) tc2_qwen35_offload;;' in run
    assert "TC2_UNS_TARGET_PARAMS" not in run.split("tc2_qwen35_offload(){")[0].split("tc2_big_family(){")[0], "no other token sets it"


def test_tc1_amendment_4_axolotl_router_recast_is_what_autocast_computes():
    """TC1 amendment 4: axolotl's loader leaves frozen `.gate` routers in fp32 and its trainer autocasts them to bf16 per call;
    the harness runs no autocast, so it casts them once after load. Only frozen fp32 `.gate` weights move; a trainable one, a
    non-gate module and a quantised (non-fp32) one are left alone; the bf16 result equals autocast's per-call cast; an fp32
    value that is not a bf16 round trip is still cast (autocast would round it the same way) and is counted, not refused."""
    torch = pytest.importorskip("torch")
    arm = _load_arm_module()

    class Router(torch.nn.Module):
        def __init__(self, w):
            super().__init__()
            self.weight = torch.nn.Parameter(w, requires_grad=False)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = Router(torch.randn(4, 8).to(torch.bfloat16).to(torch.float32))       # an upcast of a bf16 load
            self.up = torch.nn.Linear(8, 8).to(torch.float32)                                 # not a gate
            self.other = torch.nn.Module()
            self.other.gate = Router(torch.tensor([[1.0 + 2 ** -12] * 8]))                    # not a bf16 round trip
            self.trained = torch.nn.Module()
            self.trained.gate = torch.nn.Linear(8, 2)                                         # trainable: left alone

    m = torch.nn.Module()
    m.layers = torch.nn.ModuleList([Block(), Block()])
    x = torch.randn(3, 8).to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        want = torch.nn.functional.linear(x, m.layers[0].gate.weight)                        # what axolotl's trainer computes
    rep = arm.axolotl_router_recast(m)
    assert rep["n_recast"] == 4 and rep["n_not_exact_round_trip"] == 2 and rep["to"] == "bfloat16", rep
    assert m.layers[0].gate.weight.dtype == torch.bfloat16 and m.layers[0].other.gate.weight.dtype == torch.bfloat16
    assert m.layers[0].up.weight.dtype == torch.float32 and m.layers[0].trained.gate.weight.dtype == torch.float32
    got = torch.nn.functional.linear(x, m.layers[0].gate.weight)                              # the harness: no autocast
    assert torch.equal(got, want)


def test_tc1_amendment_4_only_the_scattermoe_arm_reaches_the_hub():
    """TC1 amendment 4: every arm runs with HF_HUB_OFFLINE=1 except axolotl's scattermoe native-best (its KernelsPlugin fetches
    kernels-community kernels at load), and that arm records the kernel commits it fetched; the receipt carries both records."""
    run = RUN_SH.read_text()
    assert 'local OFFL=1; case "$FW/$TAG" in axolotl/ckpt_axolotl_best*) OFFL=0;; esac' in run      # amendments 6 and 8: every scattermoe tag
    assert "env $ARM_ENV HF_HUB_OFFLINE=$OFFL UNSLOTH_ENABLE_LOGGING=1" in run and "HF_HUB_OFFLINE=1 UNSLOTH" not in run
    src = (REPO / "bench" / "tc1" / "tc1_arm.py").read_text()
    assert 'x["axolotl_router_recast"] = axolotl_router_recast(model)' in src
    assert '"axolotl_router_recast": x.get("axolotl_router_recast"), "hub_kernels_cached": x.get("hub_kernels_cached")' in src


_TC2_AM5 = {}   # module global, as the harness's wrappers are reached through a module attribute (torch._grouped_mm, Unsloth's backends)


def test_tc2_amendment_5_traced_counts_are_exact_and_free_under_torch_compile():
    """TC2 amendment 5: a counter bumped by a harness-style wrapper inside compiled code -- an autograd.Function under activation
    checkpointing, the shape of Unsloth's compiled Mixtral MoE path -- counts every call (the checkpoint replay included, as an eager
    run does), breaks no graph and recompiles nothing as the count grows. Eager calls keep the plain increment; dict() and
    setdefault behave as on a dict."""
    torch = pytest.importorskip("torch")
    from torch._dynamo.utils import counters as dyn
    arm = _load_arm_module()
    c = arm.TracedCounts(("calls",))
    c.setdefault("other", 0)

    def wrapped_mm(a, b, _slot=c.slot("calls"), _bump=arm._tc1_bump):     # the harness's wrapper shape: an int slot by default
        _bump(_slot)
        return torch.mm(a, b)
    _TC2_AM5["mm"] = wrapped_mm

    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            return _TC2_AM5["mm"](a, b)

        @staticmethod
        def backward(ctx, g):
            a, b = ctx.saved_tensors
            return g @ b.t(), a.t() @ g

    def model(x, w):
        return torch.utils.checkpoint.checkpoint(lambda x_, w_: torch.nn.functional.silu(F.apply(x_, w_)), x, w, use_reentrant=False)

    torch._dynamo.reset()
    dyn.clear()
    f = torch.compile(model, backend="aot_eager")
    w = torch.randn(16, 16, requires_grad=True)
    for i in range(9):
        f(torch.randn(4 + 4 * (i % 3), 16), w).sum().backward()
    # 9 real forwards; whether the checkpoint's recomputed forward re-runs the counter's side effect is torch's policy, not the counter's
    # (torch 2.8 replays it: 18; newer torch skips side effects in the recomputed backward: 9). The reducer's engagement floors are
    # forward-only minimums (L*A experts calls, 6*L*A grouped_mm calls per step), so either count clears them; what matters is that
    # every executed call counts, with no graph break and no per-call recompile.
    assert c["calls"] in (9, 18), dict(c)
    assert sum(dyn["graph_break"].values()) == 0, dict(dyn["graph_break"])
    assert dyn["frames"]["total"] <= 6, dict(dyn["frames"])  # a handful of compiles for three shapes, never one per call
    c.bump("other")                                       # eager: the plain increment
    n = c["calls"]
    assert dict(c) == {"calls": n, "other": 1} and len(c) == 2



def test_reducer_regime_sees_axolotls_parametrized_4bit_experts():
    """Corrected 2026-10-02: axolotl's quantize_moe_experts keeps expert stacks as parametrizations, invisible to the census's
    Params4bit_expert_stacks; the reducer read them as bf16. The real Qwen3 (96 stacks) and Granite (64) receipts are NF4."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("tc1_reduce_regime", REPO / "bench" / "tc1" / "tc1_reduce.py")
    red = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(red)
    base = {"framework": "axolotl", "census": {"Params4bit_expert_stacks": 0, "Linear4bit": 384},
            "frozen_base_probe": {"slots": {"down": {"regime": "nf4/64+dq"}}}}
    q = dict(base, axolotl_bnb4bit_modules={"quantized_moe_experts_n": 96})
    assert red.regime_of("qwen3", q).startswith("4-bit expert stacks (axolotl quantize_moe_experts: 96 parametrized stacks, nf4/64+dq)")
    g = dict(base, n_layers=32, axolotl_bnb4bit_modules={"quantized_moe_experts_n": 64})
    assert red.regime_of("granite", g).startswith("4-bit expert stacks")
    none = dict(base, axolotl_bnb4bit_modules={"quantized_moe_experts_n": 0})
    assert "NOT the 4-bit MoE regime" in red.regime_of("qwen3", none)


def test_tc1_amendment_5_native_best_token():
    """TC1 amendment 5: `qwen3nativebest` runs the three native-best configurations interleaved, two draws each, then e4b's matched
    anchor; every native arm carries its own init and adapter precision, and axolotl's is the scattermoe native-best."""
    run = RUN_SH.read_text()
    m = re.search(r"^tc1_nativebest_family\(\)\{.*?^  free_family", run, re.DOTALL | re.MULTILINE)
    assert m, "tc1_nativebest_family is gone"
    body = m.group(0)
    order = re.findall(r"&& (?:arm|draw2) +\$FAM (e4b|axolotl|unsloth) (\S+)", body)
    assert order == [("e4b", "fused_attn4_shipped"), ("axolotl", "ckpt_axolotl_best"), ("unsloth", "ckpt_unsloth_best"),
                     ("e4b", "fused_attn4_shipped"), ("axolotl", "ckpt_axolotl_best"), ("unsloth", "ckpt_unsloth_best"),
                     ("e4b", "fused_attn4_m")], order
    assert body.count("draw2") == 3 and "--axolotl-best 1" in body and "--unsloth-speed-tilt 1" in body
    assert 'local NATIVE="--adapter-dtype native --lora-init native"' in body
    assert "qwen3nativebest) tc1_nativebest_family qwen3nativebest Qwen/Qwen3-30B-A3B" in run


def test_tc1_amendment_8_native_best_200_token():
    """TC1 amendment 8: `qwen3nativebest200` runs e4b as shipped and axolotl's scattermoe native-best over the 200-step curve recipe,
    two interleaved draws each, then e4b's matched fused arm over the same 200 steps; every scattermoe tag reaches the Hub (a prefix
    rule), and the box skips the Unsloth install it has no arm for."""
    import subprocess
    run = RUN_SH.read_text()
    m = re.search(r"^tc1_nativebest200_family\(\)\{.*?^  free_family", run, re.DOTALL | re.MULTILINE)
    assert m, "tc1_nativebest200_family is gone"
    body = m.group(0)
    order = re.findall(r"&& (?:arm|draw2) +\$FAM (e4b|axolotl|unsloth) (\S+)", body)
    assert order == [("e4b", "fused_attn4_shipped_200"), ("axolotl", "ckpt_axolotl_best_200"), ("e4b", "fused_attn4_shipped_200"),
                     ("axolotl", "ckpt_axolotl_best_200"), ("e4b", "fused_attn4_m_200")], order
    assert body.count("draw2") == 2 and body.count(" curve $TOK $TS") == 5 and "--axolotl-best 1" in body and "tc1_prepare $FAM \"$MID\" $REV $FAL \"$ALL\" $CURVE_EVAL_N" in body
    assert "qwen3nativebest200) tc1_nativebest200_family qwen3nativebest200 Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in run
    assert 'case " $FAMILIES " in " qwen3axolotl "|" qwen3nativebest200 "|" qwen3syncab "|" qwen3prof945 "|" qwen3leanab "|" qwen3tileab "|" qwen3rmsab ") NEED_UNSLOTH=0;; esac' in run
    rule = re.search(r'^  local OFFL=1; case .*?esac$', run, re.MULTILINE).group(0)
    for tag, want in (("ckpt_axolotl_best", "0"), ("ckpt_axolotl_best_d2", "0"), ("ckpt_axolotl_best_200", "0"), ("ckpt_axolotl_best_200_d2", "0"),
                      ("ckpt_axolotl_m", "1"), ("ckpt_axolotl_m_d2", "1")):
        got = subprocess.run(["bash", "-c", f'f(){{ FW=axolotl; TAG={tag}\n{rule}\necho $OFFL; }}; f'], capture_output=True, text=True, check=True).stdout.strip()   # `local` needs a function
        assert got == want, (tag, got)
    got = subprocess.run(["bash", "-c", f'f(){{ FW=e4b; TAG=ckpt_axolotl_best\n{rule}\necho $OFFL; }}; f'], capture_output=True, text=True, check=True).stdout.strip()
    assert got == "1"


def test_tc1_amendment_10_sync_ab_token():
    """TC1 amendment 10 (#945): `qwen3syncab` runs e4b against itself -- legacy grouping + pageable copies vs single-read grouping +
    pinned ring -- on the shipped and matched arms, two draws each in ABBA order, each arm handed its path through TC1_ARM_EXTRA_ENV,
    which `arm` appends to the `env` word list; the arm records the path it ran (sync_ab)."""
    import subprocess
    run = RUN_SH.read_text()
    m = re.search(r"^tc1_syncab_family\(\)\{.*?^  free_family", run, re.DOTALL | re.MULTILINE)
    assert m, "tc1_syncab_family is gone"
    calls = re.findall(r'TC1_ARM_EXTRA_ENV="\$(LEG|NEW)" (arm|draw2) +\$FAM e4b (\S+) fused .* \$(NATIVE|MATCH)$', m.group(0), re.MULTILINE)
    assert calls == [("LEG", "arm", "fused_attn4_shipped_legacy", "NATIVE"), ("NEW", "arm", "fused_attn4_shipped_sync1", "NATIVE"),
                     ("LEG", "arm", "fused_attn4_m_legacy", "MATCH"), ("NEW", "arm", "fused_attn4_m_sync1", "MATCH"),
                     ("NEW", "draw2", "fused_attn4_m_sync1", "MATCH"), ("LEG", "draw2", "fused_attn4_m_legacy", "MATCH"),
                     ("NEW", "draw2", "fused_attn4_shipped_sync1", "NATIVE"), ("LEG", "draw2", "fused_attn4_shipped_legacy", "NATIVE")], calls
    assert 'local LEG="E4B_GROUPING=legacy GNF4_PINNED_RING=0" NEW="E4B_GROUPING=single GNF4_PINNED_RING=1"' in m.group(0)
    assert "qwen3syncab) tc1_syncab_family qwen3syncab Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in run
    hook = re.search(r'^  \[ -n "\$\{TC1_ARM_EXTRA_ENV:-\}" \] && ARM_ENV=.*$', run, re.MULTILINE).group(0)
    script = 'f(){ local ARM="fused"; local ARM_ENV=""\n' + hook + '\necho "[$ARM_ENV]"; }\nTC1_ARM_EXTRA_ENV="A=1 B=2" f; f'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["[", "A=1", "B=2]", "[]"], out
    src = (REPO / "bench" / "tc1" / "tc1_arm.py").read_text()
    assert '"sync_ab": sync_ab,' in src and '"ring_staged": int(sum(r.staged for r in _rings))' in src


def test_tc1_amendment_13_lean_delta_token():
    """TC1 amendment 13 (#945): `qwen3leanab` runs e4b against itself -- grouped-nf4-gemm's previous padded LoRA delta
    (NF4_QLORA_LEAN_DELTA=0) vs its trimmed body (=1), both on the post-#945 sync path -- on the shipped and matched arms, two draws
    each in ABBA order; the arm records the body it ran (lean_ab) and the delta's per-path call counts."""
    run = RUN_SH.read_text()
    m = re.search(r"^tc1_leanab_family\(\)\{.*?^  free_family", run, re.DOTALL | re.MULTILINE)
    assert m, "tc1_leanab_family is gone"
    calls = re.findall(r'TC1_ARM_EXTRA_ENV="\$(OLD|NEW)" (arm|draw2) +\$FAM e4b (\S+) fused .* \$(NATIVE|MATCH)$', m.group(0), re.MULTILINE)
    assert calls == [("OLD", "arm", "fused_attn4_shipped_lean0", "NATIVE"), ("NEW", "arm", "fused_attn4_shipped_lean1", "NATIVE"),
                     ("OLD", "arm", "fused_attn4_m_lean0", "MATCH"), ("NEW", "arm", "fused_attn4_m_lean1", "MATCH"),
                     ("NEW", "draw2", "fused_attn4_m_lean1", "MATCH"), ("OLD", "draw2", "fused_attn4_m_lean0", "MATCH"),
                     ("NEW", "draw2", "fused_attn4_shipped_lean1", "NATIVE"), ("OLD", "draw2", "fused_attn4_shipped_lean0", "NATIVE")], calls
    assert 'local OLD="NF4_QLORA_LEAN_DELTA=0" NEW="NF4_QLORA_LEAN_DELTA=1"' in m.group(0) and "NF4_QLORA_LORA_PATH" not in m.group(0)
    assert "qwen3leanab) tc1_leanab_family qwen3leanab Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in run
    src = (REPO / "bench" / "tc1" / "tc1_arm.py").read_text()
    assert '"lean_ab": lean_ab,' in src and "_lean_on = bool(_nq._lean_delta_enabled())" in src and '"lora_path_calls":' in src


def test_tc1_amendment_14_tile_rule_token():
    """TC1 amendment 14 (#945): `qwen3tileab` runs e4b against itself -- grouped-nf4-gemm's max-keyed prefill M-tile
    (GNF4_PREFILL_TILE_RULE=max) vs the cost rule (=cost) -- on the shipped and matched arms, two draws each in ABBA order; the arm
    records the rule it ran and the tile heights it launched (tile_ab)."""
    run = RUN_SH.read_text()
    m = re.search(r"^tc1_tileab_family\(\)\{.*?^  free_family", run, re.DOTALL | re.MULTILINE)
    assert m, "tc1_tileab_family is gone"
    calls = re.findall(r'TC1_ARM_EXTRA_ENV="\$(OLD|NEW)" (arm|draw2) +\$FAM e4b (\S+) fused .* \$(NATIVE|MATCH)$', m.group(0), re.MULTILINE)
    assert calls == [("OLD", "arm", "fused_attn4_shipped_tilemax", "NATIVE"), ("NEW", "arm", "fused_attn4_shipped_tilecost", "NATIVE"),
                     ("OLD", "arm", "fused_attn4_m_tilemax", "MATCH"), ("NEW", "arm", "fused_attn4_m_tilecost", "MATCH"),
                     ("NEW", "draw2", "fused_attn4_m_tilecost", "MATCH"), ("OLD", "draw2", "fused_attn4_m_tilemax", "MATCH"),
                     ("NEW", "draw2", "fused_attn4_shipped_tilecost", "NATIVE"), ("OLD", "draw2", "fused_attn4_shipped_tilemax", "NATIVE")], calls
    assert 'local OLD="GNF4_PREFILL_TILE_RULE=max" NEW="GNF4_PREFILL_TILE_RULE=cost"' in m.group(0)
    assert "qwen3tileab) tc1_tileab_family qwen3tileab Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in run
    src = (REPO / "bench" / "tc1" / "tc1_arm.py").read_text()
    assert '"tile_ab": tile_ab,' in src and "_rule = _ngt._prefill_tile_rule()" in src and '"prefill_bm_launches":' in src


def test_tc1_amendment_15_fused_rmsnorm_token():
    """TC1 amendment 15 (#945): `qwen3rmsab` runs e4b against itself -- the Hugging Face RMSNorm composite (E4B_FUSED_RMSNORM=0) vs e4b's
    fused training kernel (=1) -- on the shipped and matched arms, two draws each in ABBA order; the arm records rms_ab."""
    run = RUN_SH.read_text()
    m = re.search(r"^tc1_rmsab_family\(\)\{.*?^  free_family", run, re.DOTALL | re.MULTILINE)
    assert m, "tc1_rmsab_family is gone"
    calls = re.findall(r'TC1_ARM_EXTRA_ENV="\$(OLD|NEW)" (arm|draw2) +\$FAM e4b (\S+) fused .* \$(NATIVE|MATCH)$', m.group(0), re.MULTILINE)
    assert calls == [("OLD", "arm", "fused_attn4_shipped_rms0", "NATIVE"), ("NEW", "arm", "fused_attn4_shipped_rms1", "NATIVE"),
                     ("OLD", "arm", "fused_attn4_m_rms0", "MATCH"), ("NEW", "arm", "fused_attn4_m_rms1", "MATCH"),
                     ("NEW", "draw2", "fused_attn4_m_rms1", "MATCH"), ("OLD", "draw2", "fused_attn4_m_rms0", "MATCH"),
                     ("NEW", "draw2", "fused_attn4_shipped_rms1", "NATIVE"), ("OLD", "draw2", "fused_attn4_shipped_rms0", "NATIVE")], calls
    assert 'local OLD="E4B_FUSED_RMSNORM=0" NEW="E4B_FUSED_RMSNORM=1"' in m.group(0)
    assert "qwen3rmsab) tc1_rmsab_family qwen3rmsab Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in run
    src = (REPO / "bench" / "tc1" / "tc1_arm.py").read_text()
    assert '"rms_ab": rms_ab,' in src and "from experts4bit_qlora.engines import rmsnorm_train as _rt" in src


def test_tc1_amendment_12_profile_token():
    """TC1 amendment 12 (#945): `qwen3prof945` profiles e4b shipped and matched on the new path and matched on the legacy path, each arm
    handed its path through TC1_ARM_EXTRA_ENV with the profile flags and dmon beside; the arm records the ring's ACTUAL state."""
    run = RUN_SH.read_text()
    m = re.search(r"^tc1_prof945_family\(\)\{.*?^  free_family", run, re.DOTALL | re.MULTILINE)
    assert m, "tc1_prof945_family is gone"
    calls = re.findall(r'TC1_ARM_EXTRA_ENV="\$(LEG|NEW)" arm \$FAM e4b (\S+) fused .* \$(NATIVE|MATCH) \$PROF$', m.group(0), re.MULTILINE)
    assert calls == [("NEW", "fused_attn4_shipped_prof", "NATIVE"), ("NEW", "fused_attn4_m_prof", "MATCH"), ("LEG", "fused_attn4_m_prof_legacy", "MATCH")], calls
    assert m.group(0).count("dmon_start") == 3 and m.group(0).count("dmon_stop $dp") == 3
    assert "qwen3prof945) tc1_prof945_family qwen3prof945 Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in run
    src = (REPO / "bench" / "tc1" / "tc1_arm.py").read_text()
    assert "_ring_on = bool(_ng._pinned_ring_enabled())" in src and '"gnf4_pinned_ring_env": os.environ.get("GNF4_PINNED_RING"),' in src

