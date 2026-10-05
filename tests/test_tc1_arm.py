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
import tempfile
import time
import types
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
    # TC1 amendment 23: the census arms' static classes on both tiny layouts, its failing case, and no mem_census on an arm without the flag
    cen = json.loads((d / "tiny_e4b_fused_attn4_census.json").read_text())["mem_census"]
    assert cen["static_after_setup"]["expert_absmax"] == 256 and cen["static_after_setup"]["expert_params"] == 2304 and cen["trace"].startswith("not recorded: no CUDA")
    assert json.loads((d / "tiny_unsloth_ckpt_unsloth_census.json").read_text())["mem_census"]["static_after_setup"]["frozen_expert_weights"] == 2304
    assert "FAILING-CASE A23-census (arm): mem_census = static_after_setup: RuntimeError: selftest: census failure injected -> the arm's status ok" in p.stdout
    assert "mem_census" not in ref and "--mem-census" in ref["harness"]
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


def test_gpu_class_check_accepts_h100_spellings_and_labels_the_box(tmp_path):
    """K: TC1_GPU_CLASS=H100 must pass the class check for every H100 spelling (lane TC1c), 5090 still passes, a 4090 is refused,
    and the recorded box class is 'RTX <n>' for a numeric class and the class string otherwise. The refusal appends BOX_REFUSED to
    a cwd-relative summary.txt, so each case runs in its own temp dir: run from the repo root, it left summary.txt behind there."""
    body = RUN_SH.read_text()
    check = re.search(r'^case "\$GPU_NAME" in \*"\$GPU_CLASS"\*\) ;; \*\) .*?;; esac$', body, re.M)
    label = re.search(r'^case "\$GPU_CLASS" in \[0-9\]\*\) BOX_CLASS="RTX \$GPU_CLASS";; \*\) BOX_CLASS="\$GPU_CLASS";; esac$', body, re.M)
    assert check and label, "the class check / label lines are not in the shape this test drives"
    for i, (name, cls, ok, want_label) in enumerate((("NVIDIA H100 NVL", "H100", True, "H100"), ("NVIDIA H100 80GB HBM3", "H100", True, "H100"), ("NVIDIA H100 PCIe", "H100", True, "H100"),
                                                     ("NVIDIA GeForce RTX 5090", "5090", True, "RTX 5090"), ("NVIDIA GeForce RTX 4090", "5090", False, None),
                                                     ("NVIDIA GeForce RTX 4090", "4090", True, "RTX 4090"), ("NVIDIA RTX A2000 12GB", "RTX A2000", True, "RTX A2000"),      # TC3's two boxes
                                                     ("NVIDIA GeForce RTX 5090", "RTX A2000", False, None))):
        script = "\n".join(['say(){ echo "$*"; }; finish(){ echo "FINISH $1"; exit $1; }', f'GPU_NAME="{name}"; GPU_CLASS="{cls}"', check.group(0), label.group(0), 'echo "PASS label=$BOX_CLASS"'])
        cwd = tmp_path / f"case{i}"
        cwd.mkdir()
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=cwd)
        summary = cwd / "summary.txt"
        if ok:
            assert r.returncode == 0 and f"PASS label={want_label}" in r.stdout and not summary.exists(), (name, cls, r.stdout, r.stderr)
        else:
            assert r.returncode == 12 and "BOX REFUSED" in r.stdout, (name, cls, r.stdout)
            assert summary.read_text() == f"BOX_REFUSED gpu={name}\n", (name, cls, summary.read_text())


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
    assert '[ -n "${TC1_PREREG:-}" ] || PREREG=tc1/TC2-PREREG.md' in run and 'case " $FAMILIES " in *" tc2small "*|*" tc2big "*|*" tc2mixtral "*|*" tc2qwen35off "*|*" tc2resident "*|*" tc2mixtralres "*|*" tc2qwen35mb1 "*)' in run
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
    assert re.search(r"fused_attn4_m fused \$EAL \"\$MID\" \$REV \$OFF \$PRIM \$TOK \$TS --attn-4bit 1 \$MATCH", big) and big.count("draw2 $FAM") == 2
    assert re.search(r"ckpt_unsloth_m unsloth \$UAL \"\$MID\" \$REV 0 \$PRIM \$TOK \$TS \$UNS --unsloth-moe-backend grouped_mm \$MATCH", big)
    assert "local PRIM=${TC2_PRIMARY_RECIPE:-field} EENV=${TC2_E4B_ARM_ENV:-}" in big      # TC2 amendment 8: unset on every other token -> field, nothing
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
    assert 'NEED_UNSLOTH=1; case " $FAMILIES " in " qwen3axolotl "|" qwen3nativebest200 "|" qwen3syncab "|" qwen3prof945 "|" qwen3leanab "|" qwen3tileab "|" qwen3rmsab "|" qwen3reuseab "|" qwen3keepab "|" routebench "|" fusedsweep ") NEED_UNSLOTH=0;; esac' in body   # amendments 8, 10, 12-15 add their tokens
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


def test_tc2_amendment_6_resident_token():
    """TC2 amendment 6: `tc2resident` runs Mixtral, then Qwen3.6, with EVERY e4b arm resident (--offload 0); the Qwen3.6 matched Unsloth
    arms get the family's expert target parameters and the knob is reset after; HF, both axolotl arms and e4b as shipped are skipped as
    not_run stubs on both families; the pair and the parity control run."""
    run = RUN_SH.read_text()
    m = re.search(r"^tc2_resident\(\)\{\n(.*?)^\}\n", run, re.DOTALL | re.MULTILINE)
    assert m, "tc2_resident is gone"
    body = m.group(1)
    mx = re.search(r"tc2_big_family +mixtral +mistralai/Mixtral-8x7B-Instruct-v0\.1 +eba92302a2861cdc0098cc54bc9f17cb2c47eb61 +(\d+) +(\d+) +(\d+) +(\d+) +(\d+) +(\d+) +(\d) +\"\$UT7\"", body)
    qw = re.search(r"tc2_big_family +qwen3_5 +Qwen/Qwen3\.6-35B-A3B +995ad96eacd98c81ed38be0c5b274b04031597b0 +(\d+) +(\d+) +(\d+) +(\d+) +(\d+) +(\d+) +(\d) +\"\$UT4\" +\"\" +\"\"", body)
    assert mx and qw, body
    assert mx.group(7) == "0" and qw.group(7) == "0", "every e4b arm resident"
    assert body.index("mixtral") < body.index("qwen3_5  Qwen"), "Mixtral first"
    assert body.index('TC2_UNS_TARGET_PARAMS="$UP_QWEN3_5"') < body.index("tc2_big_family   qwen3_5") and body.rstrip().endswith('TC2_UNS_TARGET_PARAMS=""')
    for fam in ("mixtral", "qwen3_5"):
        for arm in ("hf/hf_peft_m", "axolotl/ckpt_axolotl_m", "axolotl/ckpt_axolotl_best", "e4b/fused_attn4_shipped"):
            assert f"{fam}/{arm}" in body, (fam, arm)
        assert f"{fam}/unsloth/ckpt_unsloth_m " not in body and f"{fam}/e4b/reference_attn4_m" not in body, "the pair and the parity control run"
    assert "tc2resident) tc2_resident;;" in run


def _tc2_big_family_calls(run, token, tmp_path, stat=""):
    """Runs `token` with tc2_big_family's REAL body (and skip / draw2 / the target lists from tc1_run.sh) against a stub `arm` that prints
    one line per arm: fw/tag, then `skip`, or the recipe, the offload and the arm's TC1_ARM_EXTRA_ENV. `stat` = "fam/fw/tag=oom ..." for
    status_of. Prints the three TC2 knobs after the token returns."""
    big = re.search(r"^tc2_big_family\(\)\{.*?^  free_family .*?\n", run, re.DOTALL | re.MULTILINE).group(0)
    tok = re.search(rf"^{token}\(\)\{{\n.*?^\}}\n", run, re.DOTALL | re.MULTILINE).group(0)
    lines = [re.search(rf"^{pat}.*$", run, re.MULTILINE).group(0) for pat in (r"skip\(\)\{", r"draw2\(\)\{", r"UT7=", r"UT4=", r"UP_QWEN3_5=")]
    script = "\n".join([
        "set -uo pipefail", 'say(){ :; }; MATCHED_SEED=3407; W=/w; SKIP=""; STAT="' + stat + '"', *lines,
        'tc1_prepare(){ TOK=tok; TS=sha; return 0; }; can_run(){ return 0; }; free_family(){ :; }',
        'status_of(){ case " $STAT " in *" $1/$2/$3=oom "*) echo oom;; *) echo ok;; esac; }',
        'arm(){ if skip $1 || skip $1/$2/$3; then echo "$2/$3 skip"; else echo "$2/$3 $9 off=$8 [${TC1_ARM_EXTRA_ENV:-}] $*"; fi; }',
        big, tok, token,
        'echo "knobs [${TC2_UNS_TARGET_PARAMS:-}] [${TC2_PRIMARY_RECIPE:-}] [${TC2_E4B_ARM_ENV:-}]"'])
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0 and not out.stderr, out.stderr
    return [ln for ln in out.stdout.splitlines() if ln and not ln.endswith(" DONE") and not ln.startswith(("PRIMARY ", "SECONDARY "))]   # those: summary.txt


def test_tc2_amendment_8_tokens(tmp_path):
    """TC2 amendment 8: `tc2mixtralres` (box M) runs Mixtral alone with every e4b arm resident at e4b's DEFAULT settings -- no environment
    from the token -- Unsloth x2 and e4b x2 at the field recipe, then the e4b reference; `tc2qwen35mb1` (box Q) runs Qwen3.6 alone, resident,
    with the PRIMARY pair at micro-batch 1 (TC2_PRIMARY_RECIPE=mb1 on the four primary arms, under their primary tags), E4B_ABSMAX_DQ=1
    TRAIN_FROZEN_4BIT=1 on the e4b arms only (TC2_E4B_ARM_ENV -> TC1_ARM_EXTRA_ENV), Unsloth with the family's expert target parameters,
    the reference skipped as well, and no _mb1 secondary even after an OOM; every knob reset after. HF, both axolotl arms and e4b as shipped
    are not_run stubs on both. Every other token runs exactly as before (field recipe, no extra env)."""
    run = RUN_SH.read_text()
    bodies = {}
    for fn in ("tc2_mixtral_resident", "tc2_qwen35_mb1"):
        m = re.search(rf"^{fn}\(\)\{{\n(.*?)^\}}\n", run, re.DOTALL | re.MULTILINE)
        assert m, f"{fn} is gone"
        bodies[fn] = m.group(1)
    mx, qw = bodies["tc2_mixtral_resident"], bodies["tc2_qwen35_mb1"]
    assert re.search(r'tc2_big_family +mixtral +mistralai/Mixtral-8x7B-Instruct-v0\.1 +eba92302a2861cdc0098cc54bc9f17cb2c47eb61 +7200 3600 3600 1800 2700 5400 0 "\$UT7" +"" +""', mx), mx
    assert re.search(r'tc2_big_family +qwen3_5 +Qwen/Qwen3\.6-35B-A3B +995ad96eacd98c81ed38be0c5b274b04031597b0 +6000 3600 3600 1800 2700 5400 0 "\$UT4" +"" +""', qw), qw
    for knob in ("TC2_PRIMARY_RECIPE", "TC2_E4B_ARM_ENV", "TC2_UNS_TARGET_PARAMS", "TC1_E4B_ENV", "TC1_ARM_EXTRA_ENV"):
        assert knob not in mx, (knob, "box M runs e4b at its defaults")
    for k, v in (("TC2_UNS_TARGET_PARAMS", '"$UP_QWEN3_5"'), ("TC2_PRIMARY_RECIPE", "mb1"), ("TC2_E4B_ARM_ENV", '"E4B_ABSMAX_DQ=1 TRAIN_FROZEN_4BIT=1"')):
        assert qw.index(f"{k}={v}") < qw.index("tc2_big_family") < qw.index(f'{k}=""'), k
    assert qw.rstrip().split("\n")[-3:] == ['  TC2_E4B_ARM_ENV=""', '  TC2_PRIMARY_RECIPE=""', '  TC2_UNS_TARGET_PARAMS=""']
    for knob in ("TC2_PRIMARY_RECIPE=", "TC2_E4B_ARM_ENV="):            # assigned in tc2_qwen35_mb1 alone (set, then reset)
        assert run.count(knob) == qw.count(knob) == 2, knob
    assert "tc2mixtralres) tc2_mixtral_resident;;" in run and "tc2qwen35mb1) tc2_qwen35_mb1;;" in run
    # tc2_big_family: the four primary arms take the recipe, every e4b arm (and only an e4b arm) the extra env, and an mb1 primary has no secondary
    big = re.search(r"^tc2_big_family\(\)\{.*?^  free_family", run, re.DOTALL | re.MULTILINE).group(0)
    assert len(re.findall(r" \$PRIM \$TOK", big)) == 4 and big.count('TC1_ARM_EXTRA_ENV="$EENV"') == 5
    assert all(re.search(r'TC1_ARM_EXTRA_ENV="\$EENV" (arm|draw2) +\$FAM e4b ', ln) for ln in big.split("\n") if "TC1_ARM_EXTRA_ENV" in ln and "&&" in ln)
    assert big.index('if [ "$PRIM" != field ]; then') < big.index('elif [ "$se" = oom ] || [ "$su" = oom ] || [ "$sh" = oom ]; then')
    stubbed = ["hf/hf_peft_m skip", "axolotl/ckpt_axolotl_m skip", "axolotl/ckpt_axolotl_best skip", "e4b/fused_attn4_shipped skip"]
    # box M: the field recipe, no extra env anywhere, the reference runs
    calls = _tc2_big_family_calls(run, "tc2_mixtral_resident", tmp_path)
    head = [" ".join(c.split()[:4]) for c in calls[:4]]
    assert head == ["e4b/fused_attn4_m field off=0 []", "unsloth/ckpt_unsloth_m field off=0 []", "e4b/fused_attn4_m_d2 field off=0 []", "unsloth/ckpt_unsloth_m_d2 field off=0 []"], calls
    assert calls[4:8] == stubbed and calls[8].startswith("e4b/reference_attn4_m field off=0 [] ") and calls[9] == "knobs [] [] []", calls
    assert "--unsloth-target-parameters" not in calls[1] and "--unsloth-targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" in calls[1]
    # box Q: mb1 on the four primary arms, the e4b env on e4b arms only, the expert target parameters on both Unsloth draws, the reference skipped
    for stat in ("", "qwen3_5/e4b/fused_attn4_m=oom qwen3_5/unsloth/ckpt_unsloth_m=oom"):
        calls = _tc2_big_family_calls(run, "tc2_qwen35_mb1", tmp_path, stat=stat)
        head = [" ".join(c.split()[:5]) for c in calls[:4]]
        assert head == ["e4b/fused_attn4_m mb1 off=0 [E4B_ABSMAX_DQ=1 TRAIN_FROZEN_4BIT=1]", "unsloth/ckpt_unsloth_m mb1 off=0 [] qwen3_5",
                        "e4b/fused_attn4_m_d2 mb1 off=0 [E4B_ABSMAX_DQ=1 TRAIN_FROZEN_4BIT=1]", "unsloth/ckpt_unsloth_m_d2 mb1 off=0 [] qwen3_5"], (stat, calls)
        for c in (calls[1], calls[3]):
            assert "--unsloth-targets q_proj,k_proj,v_proj,o_proj --unsloth-target-parameters mlp.experts.gate_up_proj,mlp.experts.down_proj" in c, c
        assert calls[4:] == stubbed + ["e4b/reference_attn4_m skip", "knobs [] [] []"], (stat, calls)      # no _mb1 secondary, every knob reset
    summ = (tmp_path / "summary.txt").read_text()
    assert summ.count("SECONDARY qwen3_5: none -- the primary pair ran recipe mb1 already") == 2 and "PRIMARY mixtral" not in summ
    assert summ.count("PRIMARY qwen3_5: the four primary arms run recipe mb1 (TC2_PRIMARY_RECIPE); e4b arms' extra env: E4B_ABSMAX_DQ=1 TRAIN_FROZEN_4BIT=1") == 2
    # an existing token is untouched: tc2_resident's eight arm lines run the field recipe with no extra env (Qwen3.6's OOM still falls to _mb1)
    calls = _tc2_big_family_calls(run, "tc2_resident", tmp_path, stat="qwen3_5/e4b/fused_attn4_m=oom")
    ran = [c for c in calls if not c.endswith(" skip") and not c.startswith("knobs")]
    assert all(" off=0 [] " in c for c in ran) and [c.split()[1] for c in ran].count("field") == 10 and [c.split()[0] for c in ran][-2:] == ["e4b/fused_attn4_m_mb1", "unsloth/ckpt_unsloth_m_mb1"], ran


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
    assert 'case " $FAMILIES " in " qwen3axolotl "|" qwen3nativebest200 "|" qwen3syncab "|" qwen3prof945 "|" qwen3leanab "|" qwen3tileab "|" qwen3rmsab "|" qwen3reuseab "|" qwen3keepab "|" routebench "|" fusedsweep ") NEED_UNSLOTH=0;; esac' in run
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


def test_tc1c_amendment_2_e4b_env_reaches_only_e4b_arms():
    """TC1c amendment 2: TC1_E4B_ENV is forwarded by tc1_drive.sh and lands on every e4b arm's env word list -- never on another
    framework's -- after the per-arm TC1_ARM_EXTRA_ENV; an unset value adds nothing."""
    run, drive = RUN_SH.read_text(), DRIVE_SH.read_text()
    forwarded_block = drive[drive.index("for v in TC1_FAMILIES"):drive.index("; do", drive.index("for v in TC1_FAMILIES"))]
    assert "TC1_E4B_ENV" in forwarded_block.split()
    extra = re.search(r'^  \[ -n "\$\{TC1_ARM_EXTRA_ENV:-\}" \] && ARM_ENV=.*$', run, re.MULTILINE).group(0)
    hook = re.search(r'^  \[ "\$FW" = e4b \] && \[ -n "\$\{TC1_E4B_ENV:-\}" \] && ARM_ENV=.*$', run, re.MULTILINE).group(0)
    assert run.index(extra) < run.index(hook)
    script = ('f(){ local FW=$1 ARM="fused"; local ARM_ENV=""\n' + extra + "\n" + hook + '\necho "[$ARM_ENV]"; }\n'
              'TC1_E4B_ENV="K=1 C=2" f e4b; TC1_E4B_ENV="K=1" f unsloth; f e4b; TC1_ARM_EXTRA_ENV="A=1" TC1_E4B_ENV="K=1" f e4b')
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.split("\n")
    assert [o.split() for o in out if o] == [["[", "K=1", "C=2]"], ["[]"], ["[]"], ["[", "A=1", "K=1]"]], out


def test_tc1c_amendment_3_routebench_token():
    """TC1c amendment 3: `routebench` runs bench/tc1/route_bench.py on the staged recorded calls under an alarm, writes ROUTEBENCH.json,
    builds no Unsloth venv and downloads no model; neither staged nor written file matches the reducer's *_*_*.json receipt glob."""
    import fnmatch
    run = RUN_SH.read_text()
    m = re.search(r"^tc1_routebench_family\(\)\{.*?DONE\" \| tee -a summary.txt; \}", run, re.DOTALL | re.MULTILINE)
    assert m, "tc1_routebench_family is gone"
    body = m.group(0)
    assert "$PY_E4B -u $W/route_bench.py $W/routecalls-qwen3.json $W/ROUTEBENCH.json --reps 10" in body and 'perl -e "alarm $AL' in body
    assert "tc1_prepare" not in body                                   # no model fetch, no tokens
    assert "routebench)  tc1_routebench_family routebench 1800;;" in run
    for name in ("routecalls-qwen3.json", "ROUTEBENCH.json"):
        assert not fnmatch.fnmatch(name, "*_*_*.json"), name
    bench = REPO / "bench" / "tc1"
    assert (bench / "route_bench.py").is_file() and (bench / "routecalls-qwen3.json").is_file()
    calls = json.loads((bench / "routecalls-qwen3.json").read_text())["calls"]
    assert {c["op"] for c in calls} == {"fwd", "dgrad"} and all(c["eids"] == sorted(c["eids"]) for c in calls)
    assert {(c["N"], c["K"]) for c in calls} == {(1536, 2048), (2048, 768)}


def test_tc1c_amendment_4_route_record():
    """TC1c amendment 4: every e4b arm records the training GEMM route grouped-nf4-gemm took (GNF4_TRAIN_GEMM) and its call counts."""
    src = (REPO / "bench" / "tc1" / "tc1_arm.py").read_text()
    assert '"route_ab": route_ab,' in src and "_nr.train_gemm_route()" in src and '"gnf4_train_gemm_env": os.environ.get("GNF4_TRAIN_GEMM")' in src


def test_tc1c_amendment_5_fusedsweep_token():
    """TC1c amendment 5: `fusedsweep` runs bench/tc1/fused_sweep.py on the staged recorded calls under an alarm, writes FUSEDSWEEP.json
    (outside the reducer's receipt glob), builds no Unsloth venv and downloads no model."""
    import fnmatch
    run = RUN_SH.read_text()
    m = re.search(r"^tc1_fusedsweep_family\(\)\{.*?DONE\" \| tee -a summary.txt; \}", run, re.DOTALL | re.MULTILINE)
    assert m, "tc1_fusedsweep_family is gone"
    assert "$PY_E4B -u $W/fused_sweep.py $W/routecalls-qwen3.json $W/FUSEDSWEEP.json --reps 5" in m.group(0) and "tc1_prepare" not in m.group(0)
    assert "fusedsweep)  tc1_fusedsweep_family fusedsweep 2400;;" in run
    assert not fnmatch.fnmatch("FUSEDSWEEP.json", "*_*_*.json") and (REPO / "bench" / "tc1" / "fused_sweep.py").is_file()


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


def test_tc1_amendment_20_host_reuse_token():
    """TC1 amendment 20 (#945): `qwen3reuseab` runs e4b against itself -- grouped-nf4-gemm's per-pass host reuse off (GNF4_HOST_REUSE=0)
    vs on (=1) -- on the shipped and matched arms, two draws each in ABBA order; the arm records the flag in force and the hit counts."""
    run = RUN_SH.read_text()
    m = re.search(r"^tc1_reuseab_family\(\)\{.*?^  free_family", run, re.DOTALL | re.MULTILINE)
    assert m, "tc1_reuseab_family is gone"
    calls = re.findall(r'TC1_ARM_EXTRA_ENV="\$(OLD|NEW)" (arm|draw2) +\$FAM e4b (\S+) fused .* \$(NATIVE|MATCH)$', m.group(0), re.MULTILINE)
    assert calls == [("OLD", "arm", "fused_attn4_shipped_reuse0", "NATIVE"), ("NEW", "arm", "fused_attn4_shipped_reuse1", "NATIVE"),
                     ("OLD", "arm", "fused_attn4_m_reuse0", "MATCH"), ("NEW", "arm", "fused_attn4_m_reuse1", "MATCH"),
                     ("NEW", "draw2", "fused_attn4_m_reuse1", "MATCH"), ("OLD", "draw2", "fused_attn4_m_reuse0", "MATCH"),
                     ("NEW", "draw2", "fused_attn4_shipped_reuse1", "NATIVE"), ("OLD", "draw2", "fused_attn4_shipped_reuse0", "NATIVE")], calls
    assert 'local OLD="GNF4_HOST_REUSE=0" NEW="GNF4_HOST_REUSE=1"' in m.group(0)
    assert "qwen3reuseab) tc1_reuseab_family qwen3reuseab Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in run
    src = (REPO / "bench" / "tc1" / "tc1_arm.py").read_text()
    assert '"reuse_ab": reuse_ab,' in src and "_ron = bool(_ngr._host_reuse_enabled())" in src and '"HOST_REUSE_STATS"' in src


def test_tc1_amendment_21_moe_keep_token():
    """TC1 amendment 21 (#945): `qwen3keepab` runs e4b against itself -- whole-layer checkpointing vs keeping the last n layers' MoE
    activations with the compact delta (32 on the shipped arm, 16 on the matched arm) -- two draws each in ABBA order; the arm records
    the layers it kept and the compact delta's state."""
    run = RUN_SH.read_text()
    m = re.search(r"^tc1_keepab_family\(\)\{.*?^  free_family", run, re.DOTALL | re.MULTILINE)
    assert m, "tc1_keepab_family is gone"
    calls = re.findall(r'TC1_ARM_EXTRA_ENV="\$(OLD|SHIP_NEW|M_NEW)" +(arm|draw2) +\$FAM e4b (\S+) fused .* \$(NATIVE|MATCH)$', m.group(0), re.MULTILINE)
    assert calls == [("OLD", "arm", "fused_attn4_shipped_keep0", "NATIVE"), ("SHIP_NEW", "arm", "fused_attn4_shipped_keep1", "NATIVE"),
                     ("OLD", "arm", "fused_attn4_m_keep0", "MATCH"), ("M_NEW", "arm", "fused_attn4_m_keep1", "MATCH"),
                     ("M_NEW", "draw2", "fused_attn4_m_keep1", "MATCH"), ("OLD", "draw2", "fused_attn4_m_keep0", "MATCH"),
                     ("SHIP_NEW", "draw2", "fused_attn4_shipped_keep1", "NATIVE"), ("OLD", "draw2", "fused_attn4_shipped_keep0", "NATIVE")], calls
    assert ('local OLD="E4B_MOE_KEEP_LAYERS=0" SHIP_NEW="E4B_MOE_KEEP_LAYERS=32 NF4_QLORA_COMPACT_DELTA=1" '
            'M_NEW="E4B_MOE_KEEP_LAYERS=16 NF4_QLORA_COMPACT_DELTA=1"') in m.group(0)
    assert "qwen3keepab) tc1_keepab_family qwen3keepab Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in run
    src = (REPO / "bench" / "tc1" / "tc1_arm.py").read_text()
    assert '"keep_ab": keep_ab,' in src and '"layers_kept": _kept' in src and "_nqk._compact_delta_enabled()" in src


def test_tc1_amendment_22_dense_route_tokens():
    """TC1 amendment 22: `qwen3denseab` and `mixtraldenseab` run e4b against itself on the matched arm -- grouped-nf4-gemm's fused 4-bit
    kernels (GNF4_TRAIN_GEMM=fused) vs its dense route (=dense, gnf4#459) -- two draws a side in ABBA order, every arm resident; Mixtral at
    TC2's pin (fetch 7200, e4b 3600), prepared as tc2_big_family prepares it, with E4B_ABSMAX_DQ=1 on BOTH sides and never on Qwen3; neither
    token, alone or both on the one box, builds an Unsloth venv; the arm records route_ab and absmax_dq."""
    run = RUN_SH.read_text()
    order = [("OLD", "arm", "fused_attn4_m_dense0"), ("NEW", "arm", "fused_attn4_m_dense1"),
             ("NEW", "draw2", "fused_attn4_m_dense1"), ("OLD", "draw2", "fused_attn4_m_dense0")]
    envs = {"tc1_denseab_family": 'local OLD="GNF4_TRAIN_GEMM=fused" NEW="GNF4_TRAIN_GEMM=dense"',
            "tc1_mixtral_denseab_family": 'local OLD="GNF4_TRAIN_GEMM=fused E4B_ABSMAX_DQ=1" NEW="GNF4_TRAIN_GEMM=dense E4B_ABSMAX_DQ=1"'}
    for fn, env in envs.items():
        m = re.search(rf"^{fn}\(\)\{{.*?^  free_family", run, re.DOTALL | re.MULTILINE)
        assert m, f"{fn} is gone"
        body = m.group(0)
        calls = re.findall(r'TC1_ARM_EXTRA_ENV="\$(OLD|NEW)" (arm|draw2) +\$FAM e4b (\S+) fused \$EAL "\$MID" \$REV (\d) field \$TOK \$TS --attn-4bit 1 \$MATCH$',
                           body, re.MULTILINE)
        assert [c[:3] for c in calls] == order, (fn, calls)
        assert [c[3] for c in calls] == ["0"] * 4, "every arm resident (--offload 0)"
        assert body.count("TC1_ARM_EXTRA_ENV=") == 4 and env in body
        assert 'local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"' in body and "NATIVE" not in body
        assert 'local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0' in body      # tc2_big_family's preparation: alpaca, seq, EVAL_N rows
        assert ('local ALL="e4b:fused_attn4_m_dense0:fused e4b:fused_attn4_m_dense1:fused e4b:fused_attn4_m_dense1_d2:fused '
                'e4b:fused_attn4_m_dense0_d2:fused"') in body
        assert ("E4B_ABSMAX_DQ=1" in body) == (fn == "tc1_mixtral_denseab_family")
    tc2 = re.search(r"^tc2_big_family\(\)\{.*?^  free_family", run, re.DOTALL | re.MULTILINE).group(0)
    assert 'local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0' in tc2
    assert re.search(r"tc2_big_family +mixtral +mistralai/Mixtral-8x7B-Instruct-v0\.1 +eba92302a2861cdc0098cc54bc9f17cb2c47eb61 +7200 ", run)
    assert "  qwen3denseab) tc1_denseab_family qwen3denseab Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600;;" in run
    assert ("  mixtraldenseab) tc1_mixtral_denseab_family mixtraldenseab mistralai/Mixtral-8x7B-Instruct-v0.1 "
            "eba92302a2861cdc0098cc54bc9f17cb2c47eb61 7200 3600;;") in run
    # no Unsloth venv for these tokens, alone or together; the shared line for the other tokens is untouched and still runs first
    line = 'case " $FAMILIES " in " qwen3denseab "|" mixtraldenseab "|" qwen3denseab mixtraldenseab "|" mixtraldenseab qwen3denseab ") NEED_UNSLOTH=0;; esac'
    shared = re.search(r"^NEED_UNSLOTH=1; case .*$", run, re.MULTILINE).group(0)
    assert line in run and run.index(shared) < run.index(line) < run.index("venv-unsloth-t28:")
    for fams, want in (("qwen3denseab", "0"), ("mixtraldenseab", "0"), ("qwen3denseab mixtraldenseab", "0"), ("mixtraldenseab qwen3denseab", "0"),
                       ("qwen3reuseab", "0"), ("qwen3", "1"), ("tc2big", "1"), ("qwen3denseab qwen3", "1")):
        out = subprocess.run(["bash", "-c", f'FAMILIES="{fams}"\n{shared}\n{line}\necho "$NEED_UNSLOTH"'], capture_output=True, text=True, check=True).stdout.strip()
        assert out == want, (fams, out)
    # the family's env composes with a box's TC1_E4B_ENV as on every family: both reach the arm's process, the box's words after the family's
    extra = re.search(r'^  \[ -n "\$\{TC1_ARM_EXTRA_ENV:-\}" \] && ARM_ENV=.*$', run, re.MULTILINE).group(0)
    hook = re.search(r'^  \[ "\$FW" = e4b \] && \[ -n "\$\{TC1_E4B_ENV:-\}" \] && ARM_ENV=.*$', run, re.MULTILINE).group(0)
    script = ('f(){ local FW=e4b; local ARM_ENV=""\n' + extra + "\n" + hook + "\n"
              'env $ARM_ENV /bin/sh -c \'echo "route=$GNF4_TRAIN_GEMM dq=$E4B_ABSMAX_DQ box=$K"\'; }\n'
              'TC1_ARM_EXTRA_ENV="GNF4_TRAIN_GEMM=dense E4B_ABSMAX_DQ=1" TC1_E4B_ENV="K=1" f\nTC1_ARM_EXTRA_ENV="GNF4_TRAIN_GEMM=fused" f')
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.split("\n")
    assert out[:2] == ["route=dense dq=1 box=1", "route=fused dq= box="], out
    src = (REPO / "bench" / "tc1" / "tc1_arm.py").read_text()
    assert '"route_ab": route_ab,' in src and '"stats": {k: int(v) for k, v in _rstats.items()}' in src
    assert '"absmax_dq": bool(getattr(a, "absmax_dq", 0)),' in src


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



def test_frozen_4bit_flag_defaults_from_the_env_and_runs_between_attn4_and_the_lora():
    """TRAIN_FROZEN_4BIT: the e4b arm's --frozen-4bit takes its default from the variable (TC1_E4B_ENV hands it to e4b arms only),
    converts after the attention 4-bit census and BEFORE the attention LoRA wraps (so no adapter's base is ever a candidate),
    and the receipt records the switch and the count."""
    src = ARM.read_text()
    assert 'ap.add_argument("--frozen-4bit", type=int, default=int(os.environ.get("TRAIN_FROZEN_4BIT", "0") == "1"),' in src
    i = src.index("attn4_census_check(a, model, x, detect_attention_projections, quantize_attention_projections_4bit)   # T10")
    j = src.index('x["n_frozen4"] = quantize_frozen_linears_4bit(model)')
    k = src.index("add_attention_lora(model, a.r, a.alpha, torch.float32)", i)
    assert i < j < k
    assert '"frozen_4bit": bool(getattr(a, "frozen_4bit", 0)), "n_frozen4": x.get("n_frozen4", 0),' in src


# ----------------------------------------------------------------------------- ABSMAX-DQ: e4b's double-quantized expert absmax
def _run_env(env_extra, *args):
    env = dict(os.environ)
    env.pop("E4B_ABSMAX_DQ", None)
    env.update(env_extra)
    return subprocess.run([sys.executable, str(ARM), *args], capture_output=True, text=True, timeout=300, cwd=REPO, env=env)


def test_absmax_dq_reaches_e4b_arms_only_and_refuses_offload_before_load(tmp_path):
    """--absmax-dq defaults from E4B_ABSMAX_DQ=1 (what TC1_E4B_ENV hands the e4b arms), is a harness_error on any other
    framework (exit 19) and a refused row with --offload 1 (exit 3) -- both before anything loads, so CI drives them."""
    common = ["--prereg", "tc1/TC1-PREREG.md", "--out", str(tmp_path), "--adapter-dir", str(tmp_path / "ad")]
    p = _run_env({"E4B_ABSMAX_DQ": "1"}, "--framework", "unsloth", "--arm", "unsloth", "--tag", "u", *common)
    assert p.returncode == 19, (p.stdout + p.stderr)[-2000:]
    rec = json.loads((tmp_path / "qwen3_unsloth_u.json").read_text())
    assert rec["status"] == "harness_error" and rec["absmax_dq"] is True and "e4b switch only" in rec["reason"], rec
    for env, flag in (({"E4B_ABSMAX_DQ": "1"}, []), ({}, ["--absmax-dq", "1"])):
        p = _run_env(env, "--framework", "e4b", "--arm", "fused", "--offload", "1", "--tag", "f", *flag, *common)
        assert p.returncode == 3, (p.stdout + p.stderr)[-2000:]
        rec = json.loads((tmp_path / "qwen3_e4b_f.json").read_text())
        assert rec["status"] == "refused" and rec["absmax_dq"] is True and "resident-only" in rec["reason"], rec
    src = ARM.read_text()
    assert 'ap.add_argument("--absmax-dq", type=int, default=int(os.environ.get("E4B_ABSMAX_DQ", "0") == "1"),' in src


def _tiny_e4b_model(E=4, H=128, inter=64, n_layers=2):
    """A real e4b model on CPU: Experts4bit stacks under ExpertsLoRA at layers.<i>.mlp.experts, fp32 q/k/v/o."""
    import types

    import torch
    import torch.nn as nn

    from experts4bit_qlora import Experts4bit, ExpertsLoRA

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            for i in range(n_layers):
                g = torch.Generator().manual_seed(i)
                layer = nn.Module()
                layer.self_attn = nn.Module()
                for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    setattr(layer.self_attn, p, nn.Linear(H, H, bias=False))
                layer.mlp = nn.Module()
                base = Experts4bit.from_float(torch.randn(E, 2 * inter, H, generator=g, dtype=torch.bfloat16),
                                              torch.randn(E, H, inter, generator=g, dtype=torch.bfloat16), blocksize=64, quant_type="nf4")
                layer.mlp.experts = ExpertsLoRA(base, r=2, alpha=4, dtype=torch.bfloat16)
                self.layers.append(layer)
            self.config = types.SimpleNamespace(use_cache=True, num_hidden_layers=n_layers, model_type="tiny_e4b", hidden_size=H,
                                                num_experts=E)

        def gradient_checkpointing_enable(self, **kw):
            pass

        def to(self, *a, **k):                # the arm's resident `.to("cuda")`, a no-op on a CPU test
            return self

    return _M()


def test_load_e4b_compresses_after_attn4_and_before_the_lora_wrap(monkeypatch):
    """On REAL e4b storage: --absmax-dq 1 runs compress_expert_absmax_ in its own phase between attn4 and lora, records
    modules / bytes before / after / ratio, and the frozen-base probe and C1 then read the compressed stack."""
    pytest.importorskip("bitsandbytes")
    import types

    import experts4bit_qlora
    arm = _load_arm_module()
    try:
        model = _tiny_e4b_model()
    except Exception as e:                    # the bnb CPU 4-bit path; skip only when it is genuinely absent
        pytest.skip(f"bitsandbytes CPU 4-bit quantisation unavailable: {type(e).__name__}: {e}")
    monkeypatch.setattr(experts4bit_qlora, "load_moe_4bit_streaming", lambda *a, **k: (model, model.config), raising=False)
    monkeypatch.setattr(experts4bit_qlora, "verify_moe_4bit", lambda m, strict=True: {"n_quantized": 2, "n_unquantized": 0})
    tf = types.ModuleType("transformers")
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: object())
    monkeypatch.setitem(sys.modules, "transformers", tf)
    a = types.SimpleNamespace(model="fake/model", revision="0" * 40, r=2, alpha=4, offload=0, attn_4bit=0, arm="reference", absmax_dq=1)
    arm.PH.reset()
    arm.PH.begin(time.perf_counter())
    _model, x = arm.load_e4b(a)
    assert list(arm.PH.report()["phase_seconds"]) == ["load_weights", "verify", "attn4", "absmax_dq", "lora", "enable", "tokenizer"]
    rep = x["absmax_dq"]
    E, H, inter = 4, 128, 64
    m_gu, m_dn = E * 2 * inter * H // 64, E * H * inter // 64
    nested = sum(m + 4 * -(-m // 256) + 4 + 1024 for m in (m_gu, m_dn))
    assert rep == {"modules": 2, "bytes_before": 2 * 4 * (m_gu + m_dn), "bytes_after": 2 * nested,
                   "ratio": round(4 * (m_gu + m_dn) / nested, 4)}, rep
    probe = arm.frozen_base_probe(model, "e4b")
    assert not probe["errors"] and probe["control_detects_flip"] is True, probe
    assert probe["slots"]["gate_up"]["regime"] == "nf4/64+dq" and probe["slots"]["down"]["regime"] == "nf4/64+dq"
    # C1 hashes the compressed buffers (they sit under `experts`), and nothing it hashes is empty
    h, nbytes, empties, regimes = arm.hashes_frozen(model)
    assert empties == 0 and any(k.endswith("gate_up_absmax_q") for k in h) and not any(k.endswith(".gate_up_absmax") for k in h), sorted(h)
    assert arm.c1_control(model, h)["detects"]


def test_frozen_base_probe_reads_a_compressed_stack_as_bnbs_nested_path_does():
    """SAME-BYTES with double-quant on both sides: e4b's compressed stack and bitsandbytes' whole-stack Params4bit with
    compress_statistics=True (what a double-quant bnb load holds) dequantise expert 0 to the same bf16 bytes."""
    bnb = pytest.importorskip("bitsandbytes")
    import bitsandbytes.functional as BF
    import torch
    import torch.nn as nn

    from experts4bit_qlora import compress_expert_absmax_
    arm = _load_arm_module()
    E, H, inter = 4, 128, 64
    try:
        e4b_m = _tiny_e4b_model(E, H, inter, n_layers=1)
    except Exception as e:
        pytest.skip(f"bitsandbytes CPU 4-bit quantisation unavailable: {type(e).__name__}: {e}")
    assert compress_expert_absmax_(e4b_m) == 1
    g = torch.Generator().manual_seed(0)                                  # layer 0's stacks, regenerated
    gu = torch.randn(E, 2 * inter, H, generator=g, dtype=torch.bfloat16)
    dn = torch.randn(E, H, inter, generator=g, dtype=torch.bfloat16)

    class _Stack(nn.Module):
        pass
    st = _Stack()
    for name, w in (("gate_up_proj", gu), ("down_proj", dn)):
        q, s = BF.quantize_4bit(w.contiguous(), blocksize=64, compress_statistics=True, quant_type="nf4")
        setattr(st, name, bnb.nn.Params4bit(q, requires_grad=False, quant_state=s, quant_type="nf4", blocksize=64, bnb_quantized=True))
    hf_m = nn.Module()
    hf_m.layers = nn.ModuleList([nn.Module()])
    hf_m.layers[0].mlp = nn.Module()
    w1 = arm._ParamWrapper(st, "gate_up_proj", E, H, 2 * inter, 2, 4, 1, torch.bfloat16, "hf")
    hf_m.layers[0].mlp.experts = arm._ParamWrapper(w1, "down_proj", E, inter, H, 2, 4, 2, torch.bfloat16, "hf")
    hf_m.config = e4b_m.config
    pe, ph = arm.frozen_base_probe(e4b_m, "e4b"), arm.frozen_base_probe(hf_m, "hf")
    for k in ("gate_up", "down"):
        assert pe["slots"][k]["regime"] == ph["slots"][k]["regime"] == "nf4/64+dq", (k, pe["slots"][k]["regime"], ph["slots"][k]["regime"])
        assert pe["slots"][k]["sha"] == ph["slots"][k]["sha"], f"{k}: e4b's double-quantized expert 0 != bnb's nested whole-stack slice"


# ----------------------------------------------------------------------------- TC1 amendment 23: the memory census
_F_TORCH = [{"filename": "venv/site-packages/torch/autograd/graph.py", "line": 824, "name": "_engine_run_backward"}]
_F_E4B = [{"filename": "venv/site-packages/torch/nn/functional.py", "line": 1, "name": "linear"},
          {"filename": "venv/site-packages/experts4bit_qlora/lora.py", "line": 500, "name": "forward"},
          {"filename": "work/tc1_arm.py", "line": 3000, "name": "run_arm"}]
_F_GNF4 = [{"filename": "venv/site-packages/torch/_tensor.py", "line": 9, "name": "__torch_function__"},
           {"filename": "venv/site-packages/nf4_qlora.py", "line": 77, "name": "lora_delta_grouped"},
           {"filename": "venv/site-packages/experts4bit_qlora/engines/fast.py", "line": 300, "name": "forward"}]
_F_HF = [{"filename": "venv/site-packages/torch/nn/functional.py", "line": 2, "name": "softmax"},
         {"filename": "venv/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py", "line": 200, "name": "forward"}]
_F_OPT = [{"filename": "venv/site-packages/torch/optim/adamw.py", "line": 100, "name": "_init_group"}]


def _synthetic_snapshot():
    """A torch.cuda.memory._snapshot()-shaped dict: a frozen weight resident from the load (W, no frame), a pre-window block (P), a gradient
    (G, a C++ allocation: no frame), an e4b activation freed after the peak (A1) at an address the optimizer state reuses after the peak (O),
    a grouped-nf4-gemm buffer still live (A6, under an outer e4b frame), a backward C++ temporary (A3), a torch-only allocation (A4), and an
    address reused before the peak (A2 freed, A5 allocated at it -- the peak)."""
    blk = lambda a, s, fr, st="active_allocated": {"address": a, "size": s + 12, "requested_size": s, "state": st, "frames": fr}   # noqa: E731
    ev = lambda act, a, s, fr, t: {"action": act, "addr": a, "size": s, "stream": 0, "frames": fr, "time_us": t}                # noqa: E731
    return {"segments": [{"device": 0, "address": 0, "blocks": [blk(4096, 1000, []), blk(36864, 50, []), blk(20480, 200, []), blk(8192, 400, _F_OPT),
                                                               blk(32768, 20, _F_GNF4), blk(12288, 512, [], "inactive")]}],
            "device_traces": [[ev("alloc", 8192, 300, _F_E4B, 10.0), ev("alloc", 12288, 100, _F_GNF4, 11.0), ev("alloc", 32768, 20, _F_GNF4, 12.0),
                               ev("alloc", 20480, 200, [], 20.0), ev("alloc", 16384, 80, [], 21.0), ev("alloc", 28672, 60, _F_TORCH, 22.0),
                               ev("free_requested", 12288, 100, _F_GNF4, 23.0), ev("free_completed", 12288, 100, _F_GNF4, 23.5),
                               ev("alloc", 12288, 150, _F_HF, 24.0),
                               ev("free_requested", 8192, 300, _F_E4B, 30.0), ev("free_requested", 16384, 80, [], 31.0), ev("free_requested", 28672, 60, _F_TORCH, 32.0),
                               ev("free_requested", 12288, 150, _F_HF, 33.0), ev("alloc", 8192, 400, _F_OPT, 40.0),
                               {"action": "segment_alloc", "addr": 1 << 30, "size": 2 << 20, "stream": 0, "frames": [], "time_us": 41.0}]]}


_STATIC = [(4096, 1000, "frozen_expert_weights"), (20480, 200, "adapter_grads"), (8192, 400, "optimizer_state")]
_MARKS = [(5.0, "s1.mb1.forward"), (20.0, "s1.mb1.backward"), (35.0, "s1.optimizer")]
_GROUPS = [("static:frozen_expert_weights", 1000, 1), ("site:experts4bit_qlora/lora.py:500 forward", 300, 1), ("static:adapter_grads", 200, 1),
           ("site:transformers/models/qwen3_moe/modeling_qwen3_moe.py:200 forward", 150, 1),
           ("unattributed:backward, no Python frame (an autograd C++ op or allocator-internal)", 80, 1), ("unattributed:backward, torch frames only", 60, 1),
           ("unattributed:allocated before the census window, no frame", 50, 1), ("site:nf4_qlora.py:77 lora_delta_grouped", 20, 1)]


def test_mem_census_reducer_finds_the_peak_and_groups_the_live_set():
    """The pure reducer on a synthetic snapshot: the backward replay's peak (an alloc, the latest when tied) and window start, the live set
    at the peak (a block freed after it comes back with its allocation stack; a block that is not the same allocation at the snapshot never
    takes a static label from an address reused later), the registered grouping (static class, the innermost e4b / nf4_ / Unsloth / bnb /
    torch.optim frame, the first non-torch frame, else unattributed with the harness phase), attributed_fraction, and the same reading on
    a ring that wrapped before the peak."""
    arm = _load_arm_module()
    snap = _synthetic_snapshot()
    red = arm.reduce_memory_snapshot(snap, device=0, static_ranges=_STATIC, marks=_MARKS)
    assert (red["peak_bytes"], red["peak_event_index"], red["window_start_bytes"], red["end_bytes"]) == (1860, 8, 1050, 1670), red
    assert red["live_bytes"] == red["peak_bytes"] and red["live_blocks"] == 8 and red["inconsistent_events"] == 0 and red["window_events"] == 15
    assert [(g["group"], g["bytes"], g["count"]) for g in red["live_at_peak_top"]] == _GROUPS, red["live_at_peak_top"]
    assert red["attributed_bytes"] == 1670 and red["attributed_fraction"] == round(1670 / 1860, 4)
    assert red["peak_phase"] == "s1.mb1.backward" and red["static_at_peak"] == {"frozen_expert_weights": 1000, "adapter_grads": 200} and red["n_groups"] == 8
    assert [g["group"] for g in arm.reduce_memory_snapshot(snap, static_ranges=_STATIC, marks=_MARKS, top=3)["live_at_peak_top"]] == [g[0] for g in _GROUPS[:3]]
    wrapped = dict(snap, device_traces=[snap["device_traces"][0][3:]])            # the ring dropped the three oldest allocations
    rw = arm.reduce_memory_snapshot(wrapped, static_ranges=_STATIC, marks=_MARKS)
    assert (rw["peak_bytes"], rw["peak_event_index"], rw["window_start_bytes"]) == (1860, 5, 1470) and rw["live_at_peak_top"] == red["live_at_peak_top"], rw
    # without the address map or the marks every block falls back to its frames; an empty snapshot reads nothing
    bare = arm.reduce_memory_snapshot(snap)
    assert ("unattributed:allocated before the census window, no frame", 1050, 2) in [(g["group"], g["bytes"], g["count"]) for g in bare["live_at_peak_top"]]
    assert ("unattributed:no Python frame (an autograd C++ op or allocator-internal)", 280, 2) in [(g["group"], g["bytes"], g["count"]) for g in bare["live_at_peak_top"]]
    empty = arm.reduce_memory_snapshot({})
    assert empty["peak_bytes"] == 0 and empty["attributed_fraction"] is None and empty["live_at_peak_top"] == []
    # a trace that disagrees with the end state is counted, never fatal
    bad = dict(snap, device_traces=[snap["device_traces"][0] + [{"action": "alloc", "addr": 49152, "size": 10, "frames": [], "time_us": 50.0}]])
    assert arm.reduce_memory_snapshot(bad)["inconsistent_events"] == 1


def test_mem_census_grouping_rules_and_the_recorders_kwargs():
    arm = _load_arm_module()
    assert arm.census_site_of(_F_GNF4) == "nf4_qlora.py:77 lora_delta_grouped"                     # the INNERMOST marker frame, not the outer e4b one
    assert arm.census_site_of(_F_OPT) == "torch/optim/adamw.py:100 _init_group"                   # torch/optim is a marker though it is torch
    assert arm.census_site_of(_F_HF) == "transformers/models/qwen3_moe/modeling_qwen3_moe.py:200 forward"   # no marker: the first non-torch frame
    assert arm.census_site_of(_F_TORCH) is None and arm.census_site_of([]) is None and arm.census_site_of(None) is None
    assert arm.census_site_of([{"filename": "venv/site-packages/unsloth_zoo/temporary_patches/moe_utils.py", "line": 3790, "name": "forward_native_grouped_mm"}]).startswith("unsloth_zoo/")
    assert arm.census_site_of([{"filename": "venv/site-packages/bitsandbytes/optim/optimizer.py", "line": 5, "name": "init_state"}]) == "bitsandbytes/optim/optimizer.py:5 init_state"
    assert arm.census_frame_label({"filename": "/home/someone/x/run.py", "line": 3, "name": "f"}) == "run.py:3 f"             # never a host path
    assert arm.census_frame_label({"filename": "/src/experts4bit_qlora/lora.py", "line": 3, "name": "f"}) == "experts4bit_qlora/lora.py:3 f"
    torch28 = ["enabled", "context", "stacks", "max_entries", "device", "clear_history", "compile_context", "global_record_annotations"]
    want = {"context": "alloc", "stacks": "python", "max_entries": arm.MEM_CENSUS_MAX_ENTRIES}
    assert arm.mem_history_kwargs(torch28) == ("current", "all", want)
    assert arm.mem_history_kwargs(torch28 + ["skip_actions"]) == ("current", "all", want)            # torch 2.13's impl (read locally)
    assert arm.mem_history_kwargs(["enabled", "args", "**"]) == ("current", "all", want)              # a (enabled, *args, **kwargs) wrapper
    assert arm.mem_history_kwargs(None) == ("current", "all", want)
    assert arm.mem_history_kwargs(["enabled", "context", "max_entries"]) == ("current", "all", {"context": "alloc", "max_entries": arm.MEM_CENSUS_MAX_ENTRIES})
    legacy = arm.mem_history_kwargs(["enabled", "record_context", "trace_alloc_max_entries", "trace_alloc_record_context", "device", "record_context_cpp"], 10)
    assert legacy == ("legacy", True, {"record_context": True, "trace_alloc_max_entries": 10, "trace_alloc_record_context": True, "record_context_cpp": False})
    import inspect as _inspect
    rec, snap, names = arm._mem_history_api()                                                           # this torch's real recorder and snapshot
    assert rec is not None and snap is not None and names is not None and ("max_entries" in names or "**" in names), names
    impl = getattr(arm.torch.cuda.memory, "_record_memory_history_impl", None) or rec
    assert set(arm.mem_history_kwargs(names)[2]) <= set(_inspect.signature(impl).parameters), (names, arm.mem_history_kwargs(names))


def test_mem_census_class_drives_snapshot_and_reduction_and_never_raises(monkeypatch):
    """MemCensus's glue on CPU with torch.cuda's statistics and the recorder faked: recording starts with the registered kwargs, setup_done
    reads the baseline, a checkpoint snapshots only after the max grew by MEM_CENSUS_GROW_BYTES (or when forced), the reduction is kept,
    finish() stops recording and returns the receipt block; an exception anywhere is kept as {"error"} and never reaches the arm."""
    import torch
    arm = _load_arm_module()
    state = {"max": 0, "alloc": 0}
    calls, snaps = [], []
    for name, fn in (("is_available", lambda: True), ("max_memory_allocated", lambda *a, **k: state["max"]), ("memory_allocated", lambda *a, **k: state["alloc"]),
                     ("max_memory_reserved", lambda *a, **k: state["max"] + 512), ("memory_reserved", lambda *a, **k: state["alloc"] + 512), ("current_device", lambda: 0)):
        monkeypatch.setattr(torch.cuda, name, fn)

    def rec(enabled, **kw):
        calls.append((enabled, kw))

    def snap():
        snaps.append(1)
        return _synthetic_snapshot()
    monkeypatch.setattr(arm, "_mem_history_api", lambda: (rec, snap, ["enabled", "context", "stacks", "max_entries"]))
    model = torch.nn.Linear(4, 4)
    mc = arm.MemCensus(dev="cuda")
    mc.start()
    assert calls == [("all", {"context": "alloc", "stacks": "python", "max_entries": arm.MEM_CENSUS_MAX_ENTRIES})] and mc.recording
    mc.setup_done(model, None)
    mc.mark("s1.mb1.forward")
    state.update({"max": arm.MEM_CENSUS_GROW_BYTES - 1})
    mc.checkpoint("s1.mb1", model, None)
    assert snaps == [], "no snapshot below the growth threshold"
    state.update({"max": 1860, "alloc": 1670})
    mc.checkpoint("s1.mb2", model, None, force=True)
    assert len(snaps) == 1 and mc.best["checkpoint"] == "s1.mb2" and mc.best["peak_in_window"] is True and mc.best["peak_bytes"] == 1860
    state.update({"max": 1860 + arm.MEM_CENSUS_GROW_BYTES})
    mc.checkpoint("s2.mb1", model, None)
    assert len(snaps) == 2 and mc.best["checkpoint"] == "s2.mb1" and mc.best["peak_in_window"] is False     # the window's peak is far below the run's max
    out = mc.finish(model, None)
    assert calls[-1] == (None, {}) and not mc.recording and len(snaps) == 2                        # finish: no growth since -> no third snapshot; recording stopped
    assert out["peak_allocated_bytes"] == 1860 + arm.MEM_CENSUS_GROW_BYTES and out["snapshots"] == 2 and out["peak_window"]["checkpoint"] == "s2.mb1"
    assert [g["group"] for g in out["live_at_peak_top"]][:2] == ["unattributed:allocated before the census window, no frame", "site:experts4bit_qlora/lora.py:500 forward"]
    assert out["static_after_setup"]["allocated_bytes"] == 0 and out["static_after_train"]["allocated_bytes"] == 1670 and "error" not in out
    # a recorder that raises: kept as the error, every later call a no-op, finish() returns {"error"} -- never an exception
    def bad_rec(enabled, **kw):
        raise RuntimeError("recorder exploded")
    monkeypatch.setattr(arm, "_mem_history_api", lambda: (bad_rec, snap, ["enabled", "context", "stacks", "max_entries"]))
    mc = arm.MemCensus(dev="cuda")
    mc.start()
    mc.setup_done(model, None)
    mc.checkpoint("s1.mb1", model, None, force=True)
    out = mc.finish(model, None)
    assert set(out) == {"error", "torch", "max_entries", "stacks", "context"} and out["error"] == "start: RuntimeError: recorder exploded", out
    # a snapshot that raises mid-run: recording is stopped, the reduction so far is dropped, finish() returns the error
    def bad_snap():
        raise MemoryError("host out of memory converting the ring")
    calls.clear()
    monkeypatch.setattr(arm, "_mem_history_api", lambda: (rec, bad_snap, ["enabled", "context", "stacks", "max_entries"]))
    mc = arm.MemCensus(dev="cuda")
    mc.start()
    mc.setup_done(model, None)
    mc.checkpoint("s3.optimizer", model, None, force=True)
    assert calls[-1] == (None, {}) and mc.error.startswith("checkpoint s3.optimizer: MemoryError: host out of memory")
    assert mc.finish(model, None)["error"] == mc.error


def test_mem_census_static_classes_on_real_e4b_storage():
    """P41's instrument on REAL e4b storage (CPU): the fp32 expert absmax is exactly expert_params / 64 x 4 bytes and equals the library's
    expert_absmax_bytes; after compress_expert_absmax_ the census reads #1040's four nested buffers, again equal to the library's count;
    the packed stacks are half a byte a weight; gradients and optimizer state land in their own classes."""
    pytest.importorskip("bitsandbytes")
    import torch

    from experts4bit_qlora import compress_expert_absmax_, expert_absmax_bytes
    arm = _load_arm_module()
    try:
        model = _tiny_e4b_model()
    except Exception as e:
        pytest.skip(f"bitsandbytes CPU 4-bit quantisation unavailable: {type(e).__name__}: {e}")
    E, H, inter, L = 4, 128, 64, 2
    ep = L * E * (2 * inter * H + H * inter)
    s, ranges = arm.static_mem_census(model, None, "cpu")
    assert s["expert_params"] == ep and s["expert_stacks"] == 2 * L and s["frozen_expert_weights"] == ep // 2, s
    assert s["expert_absmax"] == ep // 64 * 4 == expert_absmax_bytes(model) and s["expert_absmax_parts"] == {"fp32": ep // 64 * 4}, s
    trainable = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    assert s["trainable_adapters"] == trainable > 0 and s["adapter_grads"] == 0 and s["optimizer_state"] == 0 and s["allocated_bytes"] is None, s
    assert {c for _, _, c in ranges} >= {"frozen_expert_weights", "expert_absmax", "trainable_adapters", "other_buffers"}
    assert compress_expert_absmax_(model) == L
    s2, _ = arm.static_mem_census(model, None, "cpu")
    assert s2["expert_absmax"] == expert_absmax_bytes(model) and set(s2["expert_absmax_parts"]) == {"nested_q", "nested_s", "nested_off", "nested_code"}, s2
    m_gu, m_dn = E * 2 * inter * H // 64, E * H * inter // 64                     # one u8 code a block, an fp32 scale per 256, an offset, a 256-entry map
    assert s2["expert_absmax_parts"]["nested_q"] == ep // 64 and s2["expert_absmax"] == L * sum(m + 4 * -(-m // 256) + 4 + 1024 for m in (m_gu, m_dn)), s2
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)
    for p in params:
        p.grad = torch.zeros_like(p)
    opt.step()
    s3, _ = arm.static_mem_census(model, opt, "cpu")
    assert s3["adapter_grads"] == trainable and s3["optimizer_state"] >= 2 * trainable and s3["frozen_expert_weights"] == ep // 2, s3


def test_tc1_amendment_23_memory_census_token():
    """TC1 amendment 23: `qwen3memcensus` runs, IN THIS ORDER, e4b fused_attn4_m_mb1 (defaults), e4b fused_attn4_m_mb1_dq (E4B_ABSMAX_DQ=1)
    and Unsloth ckpt_unsloth_m_mb1 (TC1's qwen3 Unsloth arm, grouped_mm) at the mb1 recipe on Qwen3-30B-A3B at the pin, one draw each, every
    arm with --mem-census 1; the token builds the Unsloth venvs (it is in neither NEED_UNSLOTH=0 list); the arm's flag is off by default."""
    run = RUN_SH.read_text()
    m = re.search(r"^tc1_memcensus_family\(\)\{.*?^  free_family", run, re.DOTALL | re.MULTILINE)
    assert m, "tc1_memcensus_family is gone"
    body = m.group(0)
    calls = re.findall(r'^  can_run 600 \$FAM/\S+ +&& (?:TC1_ARM_EXTRA_ENV="([^"]*)" )?arm \$FAM (e4b|unsloth) (\S+) (fused|unsloth) \$(EAL|UAL) "\$MID" \$REV (\d) (\w+) \$TOK \$TS (.*)$',
                       body, re.MULTILINE)
    assert [c[:7] for c in calls] == [("", "e4b", "fused_attn4_m_mb1", "fused", "EAL", "0", "mb1"),
                                      ("E4B_ABSMAX_DQ=1", "e4b", "fused_attn4_m_mb1_dq", "fused", "EAL", "0", "mb1"),
                                      ("", "unsloth", "ckpt_unsloth_m_mb1", "unsloth", "UAL", "0", "mb1")], calls
    assert [c[7] for c in calls] == ["--attn-4bit 1 $MATCH $CEN", "--attn-4bit 1 $MATCH $CEN", "$UNS --unsloth-moe-backend grouped_mm $MATCH $CEN"], calls
    assert 'local CEN="--mem-census 1"' in body and 'local MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"' in body
    assert 'local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"' in body and body.count("TC1_ARM_EXTRA_ENV=") == 1 and "draw2" not in body
    assert 'local ALL="e4b:fused_attn4_m_mb1:fused e4b:fused_attn4_m_mb1_dq:fused unsloth:ckpt_unsloth_m_mb1:unsloth"' in body
    assert 'local TOK TS; tc1_prepare $FAM "$MID" $REV $FAL "$ALL" || return 0' in body
    fam = re.search(r"^tc1_family\(\)\{.*?^  free_family", run, re.DOTALL | re.MULTILINE).group(0)        # TC1's qwen3 Unsloth mb1 arm: the same flags
    assert 'arm $FAM unsloth ckpt_unsloth_m_mb1 unsloth $UAL "$MID" $REV 0 mb1 $TOK $TS $UNS --unsloth-moe-backend grouped_mm $MATCH' in fam
    assert 'local UNS="--grad-ckpt unsloth --unsloth-targets $UT7"' in fam
    assert "  qwen3memcensus) tc1_memcensus_family qwen3memcensus Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 3600 3600;;" in run
    assert "    mb1)    m=1; ac=$(( MB * ACCUM )); ex_tag=fused_attn4_m_mb1;;" in run and "MB=${TC1_MB:-2}; ACCUM=${TC1_ACCUM:-4}" in run   # 1 x 8
    shared = re.search(r"^NEED_UNSLOTH=1; case .*$", run, re.MULTILINE).group(0)
    dense = re.search(r'^case " \$FAMILIES " in " qwen3denseab ".*NEED_UNSLOTH=0;; esac$', run, re.MULTILINE).group(0)
    for fams, want in (("qwen3memcensus", "1"), ("qwen3keepab", "0"), ("qwen3denseab", "0"), ("qwen3", "1")):
        out = subprocess.run(["bash", "-c", f'FAMILIES="{fams}"\n{shared}\n{dense}\necho "$NEED_UNSLOTH"'], capture_output=True, text=True, check=True).stdout.strip()
        assert out == want, (fams, out)
    # the arm's flag: present, off by default, and E4B_ABSMAX_DQ=1 in the dq arm's environment is what turns its absmax switch on
    src = ARM.read_text()
    assert 'ap.add_argument("--mem-census", type=int, default=0,' in src and '**({"mem_census": mem_census} if mcen is not None else {}),' in src
    assert 'ap.add_argument("--absmax-dq", type=int, default=int(os.environ.get("E4B_ABSMAX_DQ", "0") == "1"),' in src
    p = _run("--help")
    assert p.returncode == 0 and "--mem-census" in p.stdout, p.stdout[-800:]
    # the reducer registers the family without speed positions, and its self-test reads P41-P43
    red = (REPO / "bench" / "tc1" / "tc1_reduce.py").read_text()
    assert 'MEMCENSUS_FAM = "qwen3memcensus"' in red and "NO_SPEED_FAMS = {MEMCENSUS_FAM:" in red and "## Predictions P41 / P42 / P43" in red
    r = _run("--selftest", script=REPO / "bench" / "tc1" / "tc1_reduce.py")
    assert r.returncode == 0 and "REDUCE SELFTEST OK" in r.stdout and "FAILING-CASE A23-P41 (reducer)" in r.stdout and "FAILING-CASE A23-P43 (reducer)" in r.stdout, r.stdout[-1500:]


def _load_gate_shell(tmp_path, loads, gate="6.0", retries=None):
    """Run tc1_run.sh's own arm() (TC1 amendment 33) with arm_once stubbed: each attempt writes a receipt and a gpuclk file whose load1
    column is the next value of `loads`. Returns (summary text, the files in loadvoid/, the attempts made)."""
    body = RUN_SH.read_text()
    m = re.search(r"^arm\(\)\{.*?^  done; \}\n", body, re.S | re.M)
    assert m, "arm() (the load gate) not found in tc1_run.sh"
    w = tmp_path / "w"
    (w / "logs").mkdir(parents=True)
    (w / "loads.txt").write_text("\n".join(str(x) for x in loads) + "\n")
    stub = f"""
set -u
W={w}; PY_E4B={sys.executable}; cd $W
say(){{ :; }}
can_run(){{ return 0; }}
status_of(){{ $PY_E4B -c "import json,sys; print(json.load(open(sys.argv[1])).get('status','missing'))" "$W/${{1}}_${{2}}_${{3}}.json" 2>/dev/null || echo missing; }}
arm_once(){{ local l; l=$(head -1 $W/loads.txt); sed -i.bak 1d $W/loads.txt; echo x >> $W/attempts.txt
  echo '{{"status": "ok"}}' > $W/${{1}}_${{2}}_${{3}}.json
  for i in 1 2 3; do echo "1 2400, 14001, 50, 0x0 | $l 1.0 1.0 | cpu 1 2 3" >> $W/gpuclk_${{1}}_${{2}}_${{3}}.txt; done
  echo log > $W/logs/run_${{1}}_${{2}}_${{3}}.log; : > $W/vram_${{1}}_${{2}}_${{3}}.txt; }}
{m.group(0)}
arm fam e4b tag fused 600 mid rev 0 field tok sha --x 1
"""
    env = dict(os.environ)
    env.pop("TC1_LOAD_GATE", None)
    env.pop("TC1_LOAD_RETRIES", None)
    if gate is not None:
        env["TC1_LOAD_GATE"] = gate
    if retries is not None:
        env["TC1_LOAD_RETRIES"] = str(retries)
    subprocess.run(["bash", "-c", stub], env=env, check=True, capture_output=True, text=True)
    summ = (w / "summary.txt").read_text() if (w / "summary.txt").exists() else ""
    void = sorted(p.name for p in (w / "loadvoid").iterdir()) if (w / "loadvoid").exists() else []
    return summ, void, len((w / "attempts.txt").read_text().split())


def test_load_gate_voids_a_busy_draw_and_reruns_it(tmp_path):
    """TC1 amendment 33: a draw whose median host load1 exceeds TC1_LOAD_GATE is set aside to loadvoid/ (.a1) and run again; the
    quiet re-run stands. Unset, arm is arm_once (one attempt, no LOADGATE line). The retries are capped and the last attempt stands."""
    summ, void, n = _load_gate_shell(tmp_path / "a", [12.0, 3.0])
    assert n == 2 and "attempt 0 VOID (host load1 median 12.0 > 6.0): re-run 1 of 2" in summ and "attempt 1 load1_median 3.0 gate 6.0 status ok over 0" in summ
    assert void == ["fam_e4b_tag.json.a1", "gpuclk_fam_e4b_tag.txt.a1", "run_fam_e4b_tag.log.a1", "vram_fam_e4b_tag.txt.a1"], void
    summ, void, n = _load_gate_shell(tmp_path / "b", [12.0, 3.0], gate=None)
    assert n == 1 and "LOADGATE" not in summ and void == []
    summ, void, n = _load_gate_shell(tmp_path / "c", [20.0, 21.0, 22.0, 23.0], retries=2)
    assert n == 3 and summ.count("VOID") == 2 and "attempt 2 load1_median 22.0 gate 6.0 status ok over 1" in summ
    summ, void, n = _load_gate_shell(tmp_path / "d", [4.0])
    assert n == 1 and "VOID" not in summ and void == []


# ----------------------------------------------------------------------------- TC1 amendment 39: the packed 4,096-token regime
class _EosTok:
    """A byte tokenizer with an EOS: '</s>' at the end of the text is id 0, every other byte 1 + (b % 61); truncates only when asked."""
    eos_token, eos_token_id, pad_token_id = "</s>", 0, None

    def __call__(self, text, truncation=False, max_length=None):
        n = 0
        while text.endswith("</s>"):
            text, n = text[:-4], n + 1
        ids = [1 + (b % 61) for b in text.encode()] + [0] * n
        return types.SimpleNamespace(input_ids=ids[:max_length] if (truncation and max_length) else ids)


def _tiny_rows(n, tag):
    return [{"instruction": f"{tag} instruction {k} " + "x" * (k % 7), "input": ("ctx " * (k % 3)).strip(), "output": f"answer {k} " + "y" * (3 * k % 11)}
            for k in range(n)]


def _tiny_args(d, **kw):
    import hashlib
    d.mkdir(parents=True, exist_ok=True)
    p = d / "ds_tiny.json"
    if not p.exists():
        p.write_text(json.dumps({"train": _tiny_rows(12, "t"), "eval": _tiny_rows(5, "e")}, sort_keys=True))
    a = types.SimpleNamespace(data=str(p), data_sha=hashlib.sha256(p.read_bytes()).hexdigest(), model="tiny/tok", revision="r0", seq=400, eval_n=4,
                              template="alpaca", tokens=str(d / "tokens_tiny.json"), fam="tiny")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# origin/main's prepare (before --pack existed) on _tiny_args with _EosTok: the whole tokens file and its sha256 field
UNPACKED_TINY_FILE_SHA = "1ff1771f1d9b6712df2239bc6418a9f9194a4f838eeffca854cafaa5cac47dca"
UNPACKED_TINY_TOKENS_SHA = "05d712ec4f9d4f075e73fcd3e22d2cbddebd3d2e115539c741aefc05fe402ba4"


def test_tc1_amendment_39_unpacked_tokens_file_is_byte_identical(tmp_path):
    """--pack 0 (the default, and every token but the packed one) writes the tokens file byte-for-byte as before the flag existed: the
    golden shas were produced by origin/main's prepare on the same inputs (the real-tokenizer proof, Qwen3-30B-A3B's field-recipe file,
    is in the PR description). No pack key appears, and the rows are encode_rows' one example per row."""
    import hashlib
    m = _load_arm_module()
    for kw in ({}, {"pack": 0}):
        d = tmp_path / ("a" if not kw else "b")
        rec = m.prepare(_tiny_args(d, **kw), tok=_EosTok())
        assert hashlib.sha256((d / "tokens_tiny.json").read_bytes()).hexdigest() == UNPACKED_TINY_FILE_SHA
        assert rec["sha256"] == UNPACKED_TINY_TOKENS_SHA and not any(k.startswith("pack") for k in rec)
        assert rec["train"] == m.encode_rows(_EosTok(), _tiny_rows(12, "t"), 400, "alpaca", "</s>") and len({len(r) for r in rec["train"]}) > 1


def test_tc1_amendment_39_pack_rows_are_exactly_seq_with_eos_between_examples():
    """pack_rows: every row exactly `seq` tokens, the stream is the examples' own token lists in order with EOS between them (the alpaca
    template's own EOS, never doubled; appended to a template without one), the tail dropped, examples running across row boundaries."""
    m = _load_arm_module()
    tok, rows = _EosTok(), _tiny_rows(40, "t")
    per = [tok(m.render_row(r, "alpaca", "</s>")).input_ids for r in rows]
    assert all(p[-1] == 0 and p.count(0) == 1 for p in per)                          # one EOS per example, at its end
    out, st = m.pack_rows(tok, rows, 128, "alpaca", "</s>")
    stream = [t for p in per for t in p]
    assert {len(r) for r in out} == {128} and len(out) == len(stream) // 128 == st["rows"]
    assert [t for r in out for t in r] == stream[:len(out) * 128] and st["tokens_dropped"] == len(stream) - len(out) * 128
    assert st["examples_used"] == 40 and st["examples_split_across_rows"] > 0 and st["stream_tokens"] == len(stream)
    ends = [sum(len(p) for p in per[:i + 1]) - 1 for i in range(len(per))]         # EOS exactly where each example ends, nowhere else
    flat = [t for r in out for t in r]
    assert [i for i, t in enumerate(flat) if t == 0] == [e for e in ends if e < len(flat)]
    # a template without EOS (clinical) gets the tokenizer's EOS appended between examples; n_rows stops early
    out_c, st_c = m.pack_rows(tok, rows, 64, "clinical", "", n_rows=3)
    per_c = [tok(m.render_row(r, "clinical", "")).input_ids + [0] for r in rows]
    assert len(out_c) == 3 and [t for r in out_c for t in r] == [t for p in per_c for t in p][:192] and st_c["examples_used"] < 40
    # per-example truncation is gone: an example longer than seq spans rows whole
    long = [{"instruction": "z" * 300, "input": "", "output": "w"}]
    out_l, _ = m.pack_rows(tok, long * 3, 128, "alpaca", "</s>")
    assert [t for r in out_l for t in r] == (tok(m.render_row(long[0], "alpaca", "</s>")).input_ids * 3)[:len(out_l) * 128]
    with pytest.raises(ValueError):
        m.pack_rows(types.SimpleNamespace(eos_token_id=None), rows, 64)


def _pack_world(m, tmp_path, n_src=80, train_pool=24, eval_pool=10, monkeypatch=None):
    """A synthetic source laid out as tp4_alpaca.py lays out its pin: shuffled once with PACK_SEED, the registered ds its first 12 + 5 rows."""
    import hashlib
    import random
    src_rows = [{"instruction": f"s{k} " + "q" * (k % 9), "input": "in" if k % 4 == 0 else "", "output": f"o{k} " + "r" * (5 * k % 13), "extra": k} for k in range(n_src)]
    src = tmp_path / "alpaca_data_cleaned.json"
    src.write_text(json.dumps(src_rows))
    idx = list(range(n_src))
    random.Random(m.PACK_SEED).shuffle(idx)
    clean = [{"instruction": src_rows[i]["instruction"], "input": src_rows[i].get("input", ""), "output": src_rows[i]["output"]} for i in idx]
    ds = {"train": clean[:12], "eval": clean[12:17]}
    monkeypatch.setattr(m, "PACK_SRC_SHA256", hashlib.sha256(src.read_bytes()).hexdigest())
    monkeypatch.setattr(m, "PACK_SRC_ROWS", n_src)
    monkeypatch.setattr(m, "PACK_TRAIN_POOL", train_pool)
    monkeypatch.setattr(m, "PACK_EVAL_POOL", eval_pool)
    return src, ds, clean


def test_tc1_amendment_39_pack_pools_extend_the_registered_text_in_its_own_order(tmp_path, monkeypatch):
    """pack_pools: [train 12 | eval 5 | train extension | eval extension] of the shuffled source -- the registered rows first in each pool,
    train and held-out disjoint; refused (exit 13) on a wrong source sha or a prefix that is not the registered rows."""
    m = _load_arm_module()
    src, ds, clean = _pack_world(m, tmp_path, monkeypatch=monkeypatch)
    tr, ev, rec = m.pack_pools(str(src), ds)
    assert tr == clean[:12] + clean[17:29] and ev == clean[12:17] + clean[29:34]
    assert rec["train_examples"] == 24 and rec["eval_examples"] == 10 and rec["registered_prefix"] == [12, 5] and rec["seed"] == m.PACK_SEED
    assert not {json.dumps(r, sort_keys=True) for r in tr} & {json.dumps(r, sort_keys=True) for r in ev}
    with pytest.raises(SystemExit) as e:
        m.pack_pools(str(src), {"train": ds["train"][::-1], "eval": ds["eval"]})
    assert e.value.code == 13
    monkeypatch.setattr(m, "PACK_SRC_SHA256", "0" * 64)
    with pytest.raises(SystemExit) as e:
        m.pack_pools(str(src), ds)
    assert e.value.code == 13


def test_tc1_amendment_39_prepare_pack_writes_packed_rows_and_their_shas(tmp_path, monkeypatch):
    """prepare --pack 1: train rows and eval_n held-out rows all exactly --seq tokens, built from the pools by pack_rows; the sha256 (and the
    train-only sha tc1_run.sh prints) cover the packed rows; pack / pack_sep_id / pack_pools / pack_stats recorded; too few rows refuse."""
    import hashlib
    m = _load_arm_module()
    src, ds, clean = _pack_world(m, tmp_path, monkeypatch=monkeypatch)
    d = tmp_path / "w"
    d.mkdir()
    (d / "ds_tiny.json").write_text(json.dumps(ds, sort_keys=True))
    a = _tiny_args(d, seq=96, eval_n=3, pack=1, pack_src=str(src), pack_min_rows=10, fam="tinypack", tokens=str(d / "tokens_tinypack.json"))
    rec = m.prepare(a, tok=_EosTok())
    tr_pool, ev_pool, _ = m.pack_pools(str(src), ds)
    want_tr, _ = m.pack_rows(_EosTok(), tr_pool, 96, "alpaca", "</s>")
    want_ev, _ = m.pack_rows(_EosTok(), ev_pool, 96, "alpaca", "</s>", n_rows=3)
    assert rec["train"] == want_tr and rec["eval"] == want_ev and len(rec["eval"]) == 3 and len(rec["train"]) >= 10
    assert {len(r) for r in rec["train"] + rec["eval"]} == {96} and rec["train_tokens"] == 96 * len(rec["train"])
    on_disk = json.loads((d / "tokens_tinypack.json").read_text())
    body = json.dumps({"train": on_disk["train"], "eval": on_disk["eval"]}, separators=(",", ":")).encode()
    assert on_disk["sha256"] == hashlib.sha256(body).hexdigest() == rec["sha256"] != UNPACKED_TINY_TOKENS_SHA
    assert on_disk["pack"] is True and on_disk["pack_sep_id"] == 0 and on_disk["seq"] == 96
    assert on_disk["pack_pools"]["train_examples"] == 24 and on_disk["pack_stats"]["eval"]["rows"] == 3
    a.pack_min_rows = 10 ** 6
    with pytest.raises(SystemExit) as e:
        m.prepare(a, tok=_EosTok())
    assert e.value.code == 13
    a.pack_min_rows, a.pack_src = 0, None
    with pytest.raises(SystemExit) as e:
        m.prepare(a, tok=_EosTok())
    assert e.value.code == 13


def test_tc1_amendment_39_pack_constants_are_tp4_alpacas():
    """The packed pools' pin and seed are tp4_alpaca.py's (the shuffled-prefix check in pack_pools proves them at run time as well)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("tp4_alpaca_under_test", REPO / "bench" / "tp4" / "tp4_alpaca.py")
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    m = _load_arm_module()
    assert (m.PACK_SRC_SHA256, m.PACK_SEED, m.PACK_SRC_ROWS) == (t.FILE_SHA256, t.SEED, 51760)
    assert m.PACK_TRAIN_POOL >= t.N_TRAIN and m.PACK_EVAL_POOL >= t.N_EVAL


def test_tc1_amendment_39_packed_token_and_its_box_fixture():
    """tc1_run.sh: the qwen3samestack4k token runs amendment 25's family with its alarms; TC1_PACK is read with default 0, forwarded by
    tc1_drive.sh and recorded on the FIXTURE line; tc1_prepare hands --prepare the pack flags (at least steps x micro-batch x accum rows)
    only on a packing box, through tokenise's pass-through; the box refuses a packed token without TC1_PACK=1 TC1_SEQ=4096 and TC1_PACK=1
    beside any field-recipe token -- executed through bash."""
    run, drive = RUN_SH.read_text(), DRIVE_SH.read_text()
    assert ("  qwen3samestack4k) tc1_samestack_family qwen3samestack4k Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 "
            "5400 3600 5400 7200;;") in run
    assert re.search(r"^PACK=\$\{TC1_PACK:-0\}$", run, re.M) and "matched_seed=$MATCHED_SEED pack=$PACK\" | tee -a summary.txt" in run
    forwarded_block = drive[drive.index("for v in TC1_FAMILIES"):drive.index("; do", drive.index("for v in TC1_FAMILIES"))]
    assert "TC1_PACK" in forwarded_block.split()
    assert '--template $TEMPLATE_ --tokens $TOK "${@:10}" > logs/prepare_' in run
    assert "tokenise $FAM \"$MID\" $REV alpaca $SEQ $W/data/ds_alpaca.json $DS_ALPACA_SHA $TOK $EVN $PK; then" in run
    pk = re.search(r'^  local PK=""; \[ "\$PACK" = 1 \] && PK=.*$', run, re.M).group(0)
    for pack, want in (("1", "--pack 1 --pack-src /root/tc1/data/alpaca_data_cleaned.json --pack-min-rows 120"), ("0", "")):
        out = subprocess.run(["bash", "-c", f"W=/root/tc1 STEPS=30 MB=1 ACCUM=4 PACK={pack}; f(){{\n{pk}\necho \"$PK\"; }}; f"],
                             capture_output=True, text=True, check=True).stdout.strip()
        assert out == want, (pack, out)
    # tokenise passes everything after its 9th argument to --prepare
    tk = re.search(r"^tokenise\(\)\{.*?return 0; \}$", run, re.S | re.M).group(0)
    with tempfile.TemporaryDirectory() as td:
        script = f"cd {td}; mkdir -p logs; W={td}; PY_E4B=echo; say(){{ :; }}\n{tk}\ntokenise fam mid rev alpaca 4096 data sha tok 8 --pack 1 --pack-src s --pack-min-rows 120"
        subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
        log = (Path(td) / "logs" / "prepare_fam_alpaca.log").read_text()
    assert log.strip().endswith("--eval-n 8 --template alpaca --tokens tok --pack 1 --pack-src s --pack-min-rows 120"), log
    # the box-level refusal, executed
    start = run.index('case "$PACK" in 0|1) ;;')
    block = run[start:run.index("\ndone\n", start) + len("\ndone\n")]
    for fams, pack, seq, ok in (("qwen3samestack4k", "1", "4096", True), ("qwen3samestack4k", "0", "4096", False), ("qwen3samestack4k", "1", "2048", False),
                                ("qwen3samestack", "1", "4096", False), ("qwen3samestack", "0", "2048", True), ("qwen3 qwen3samestack", "0", "2048", True),
                                ("qwen3samestack4k qwen3samestack", "1", "4096", False), ("qwen3samestack4k", "2", "4096", False)):
        with tempfile.TemporaryDirectory() as td:
            script = f'cd {td}; say(){{ echo "$*"; }}; finish(){{ exit $1; }}; FAMILIES="{fams}"; PACK={pack}; SEQ={seq}\n{block}echo PASSED'
            r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            summ = (Path(td) / "summary.txt").read_text() if (Path(td) / "summary.txt").exists() else ""
        assert (r.returncode == 0 and "PASSED" in r.stdout) if ok else (r.returncode == 78 and "refusing" in r.stdout and "BOX_REFUSED" in summ), (fams, pack, seq, r.stdout)
    assert run.index(block) > run.index('echo "BOX $TC1_BOX families:') and run.index(block) < run.index("# cu130 wheels")
