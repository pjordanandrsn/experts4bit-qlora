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
    # T17 (P43) still holds: every step printed, every micro-batch timed, the CELL line never carries the per-step lists
    assert ref["log_every"] == 1 and ref["microbatch_timing"] is True
    assert '"microbatch_ms"' not in "".join(line for line in p.stdout.splitlines() if line.startswith("CELL "))


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
    """TC1-PREREG 'Arms, in this order': the box script's invocations, in order, with the matched / native flags and the
    draw2 + TODO hooks exactly where the registration puts them."""
    body = RUN_SH.read_text()
    fam = body[body.index("tc1_family(){"):body.index("# ---------------------------------------------------------------- the plan")]
    calls = re.findall(r"(?:arm|draw2|todo_arm)\s+\$FAM\s+(e4b|unsloth|hf|axolotl)\s+(\S+)", fam)
    want = [("e4b", "fused_attn4_m"), ("unsloth", "ckpt_unsloth_m"), ("e4b", "fused_attn4_m"), ("unsloth", "ckpt_unsloth_m"), ("hf", "hf_peft_m"),
            ("axolotl", "ckpt_axolotl_m"), ("axolotl", "ckpt_axolotl_best"), ("unsloth", "ckpt_unsloth_best"), ("unsloth", "ckpt_unsloth_t28"), ("unsloth", "ckpt_unsloth_triton"),
            ("e4b", "fused_attn4_shipped"), ("e4b", "reference_attn4_m"), ("unsloth", "ckpt_unsloth_prof"), ("e4b", "fused_attn4_m_mb1"), ("unsloth", "ckpt_unsloth_m_mb1"), ("hf", "hf_peft_m_mb1")]
    assert calls == want, calls
    assert fam.count("draw2 $FAM") == 2 and fam.count("todo_arm $FAM") == 0      # phase 2: every registered arm is implemented
    assert 'MATCH="--adapter-dtype fp32 --lora-init matched:$MATCHED_SEED"' in fam and 'NATIVE="--adapter-dtype native --lora-init native"' in fam
    assert re.search(r"fused_attn4_shipped fused .* \$NATIVE", fam) and re.search(r"reference_attn4_m reference .* \$MATCH", fam)
    assert 'draw2(){ local FAM=$1 FW=$2 TAG=$3; shift 3; arm "$FAM" "$FW" "${TAG}_d2" "$@"; }' in body
    assert "--profile-steps $PROFILE_STEPS --profile-warm $PROFILE_WARM" in fam and "dmon_start" in fam
    # phase 2: the Unsloth venv/backend/tilt knobs and the axolotl flags sit on the arms the instruction names
    assert re.search(r"ckpt_unsloth_m unsloth .* --unsloth-moe-backend grouped_mm \$MATCH", fam)
    assert re.search(r"UNS_VENV=t28 arm \$FAM unsloth ckpt_unsloth_t28 unsloth .* --unsloth-moe-backend default \$MATCH", fam)
    assert re.search(r"ckpt_unsloth_best unsloth .* --unsloth-moe-backend grouped_mm --unsloth-speed-tilt 1 --adapter-dtype fp32 --lora-init native", fam)
    assert re.search(r"ckpt_unsloth_triton unsloth .* --unsloth-moe-backend unsloth_triton \$MATCH", fam)
    assert re.search(r"ckpt_unsloth_prof unsloth .* --unsloth-moe-backend grouped_mm \$MATCH .*--profile-steps", fam)
    assert re.search(r"ckpt_axolotl_m axolotl \$AAL .* --axolotl-dataset \$W/data/ds_alpaca.json \$MATCH", fam)
    assert re.search(r"ckpt_axolotl_best axolotl \$AAL .* --axolotl-best 1 --adapter-dtype fp32 --lora-init native", fam)
    # the driver >= 580 gate precedes every cu130 install and the gated arms refuse by name, never fall back to the t28 venv
    assert '[ "$DRIVER_MAJOR" -ge 580 ]' in body and body.index("CU130_OK=1; case") < body.index("venv-unsloth-t28:") < body.index("uv venv --python 3.12")
    assert 'refused "$CU130_REASON"' in body and "unsloth[cu130-torch2121]" in body and "unsloth[cu128-torch280]" in body
    assert '--extra-index-url https://download.pytorch.org/whl/cu130' in body and '"axolotl==$AX_VER"' in body


def test_every_env_knob_the_box_reads_is_forwarded_by_the_driver():
    """tp4's lesson (TP4-PREREG amendment 2): a knob the box script reads from the environment that the controller does not
    forward silently never reaches the box. Every TC1_* read by tc1_run.sh or tc1_arm.py must be in tc1_drive.sh's forwarded
    list or its handshake string, unless the box script sets it itself."""
    run, arm, drive = RUN_SH.read_text(), ARM.read_text(), DRIVE_SH.read_text()
    read = set(re.findall(r"\$\{(TC1_[A-Z0-9_]+)", run)) | set(re.findall(r"\$\{!v", run) and re.findall(r"for v in ((?:TC1_[A-Z_]+ )+)", run)[0].split())
    read |= set(re.findall(r'os\.environ\.get\("(TC1_[A-Z0-9_]+)"', arm)) | set(re.findall(r'\("(TC1_[A-Z0-9_]+)", [0-9.A-Z_]+\)', arm))
    set_by_box = set(re.findall(r"(?:^|\s)(TC1_[A-Z0-9_]+)=", run))
    forwarded = set(re.findall(r"TC1_[A-Z0-9_]+", drive[drive.index("PASS="):drive.index("TC1_DRIVE_DRYRUN")]))
    missing = sorted(read - set_by_box - forwarded)
    assert not missing, f"read by the box/arm but never forwarded by tc1_drive.sh: {missing}"
    for must in ("TC1_STEPS", "TC1_EVAL_N", "TC1_EVAL_EVERY", "TC1_MATCHED_SEED", "TC1_PHASE_BUDGET_S", "TC1_PROFILE_STEPS"):
        assert must in forwarded, must


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
        {prefix} /bin/sh -c 'echo RAN box="$TC1_BOX_CLASS" alarm="$TC1_ARM_ALARM_S" pad="$E4B_BATCHED_PAD_WASTE_LIMIT"'
        """
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert r.returncode == 0 and "RAN" in r.stdout and 'box=RTX 5090' in r.stdout and "alarm=3600" in r.stdout and f"pad={want}" in r.stdout, (arm, r.stdout, r.stderr)


def test_run_and_drive_scripts_parse():
    for sh in (RUN_SH, DRIVE_SH):
        r = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
        assert r.returncode == 0, (sh, r.stderr)
    drive = DRIVE_SH.read_text()
    assert "GNF4_SHA=${GNF4_SHA:-846b512b905468c08f5748943d08769b572affa2}" in drive      # the v0.34.0 COMMIT, not the tag object
    assert '$REPO/bench/tp4/tp4_alpaca.py' in drive and "n9_datasets" not in drive     # tp4's Alpaca builder is referenced, not copied
