"""CI wrapper for ``bench/tp4/tp4_arm.py --selftest`` (lane tp4, TP4-PREREG.md).

The selftest drives all three framework branches (e4b / Unsloth / plain HF+PEFT+bnb) through the one ``run_arm``
on CPU with mocked kernels, plus the T12/T13/T14 additions (Alpaca template, micro-batches of 2 with padding, the
linear-with-warmup schedule) and the tp3 T10 detector dry-runs. A green suite that never executes is not a gate, so
CI runs it end to end and asserts the receipt-level outcomes -- including that a real run without ``--prereg`` is
refused before any receipt exists, and that the pinned Alpaca subset builder reproduces the registered sha when the
source file is present (skipped without it: the test never downloads).
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ARM = REPO / "bench" / "tp4" / "tp4_arm.py"
ALPACA = REPO / "bench" / "tp4" / "tp4_alpaca.py"
REGISTERED_DS_SHA = "5324987afa4042556953026289e8dbdbe8a936b32832ed9e603b9192b706a2fb"   # TP4-PREREG.md "Fixture"


def _run(*args, script=ARM):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=900, cwd=REPO)


def test_tp4_arm_selftest():
    p = _run("--selftest")
    tail = (p.stdout + p.stderr)[-3000:]
    assert p.returncode == 0, tail
    assert "SELFTEST OK" in p.stdout
    # the tp3 T10 dry-runs against the REAL structural detector still pass through this copy
    assert "'tiny_keqv30': 115" in p.stdout and "'tiny_plain4': 16" in p.stdout and "'tiny_missing_k': ['layers.1']" in p.stdout
    # #542: the HF arm's expert selection dry-runs -- the Granite-named stacks the old NAME substring took zero of are
    # selected by structure, and every empty/implausible layout refuses instead of running attention-only
    assert "'granite_on_disk': {'n': 6, 'substring': 0, 'param_re': 0}" in p.stdout, p.stdout[-1500:]
    assert "'refusals': ['dense', 'not_per_layer', 'per_expert_2d', 'ragged_declared_matches_nothing', 'ragged_no_config']" in p.stdout
    d = Path(re.search(r"SELFTEST OK dir=(\S+)", p.stdout).group(1))
    # receipts name THIS lane's pre-registration
    assert json.loads((d / "tiny_e4b_reference_attn4.json").read_text())["prereg"] == "tp4/TP4-PREREG.md"
    # T11: the hf arm wrote a receipt with the module-level counter and the field regime recorded by the census
    hf = json.loads((d / "tiny_hf_hf_peft.json").read_text())
    assert hf["status"] == "ok" and hf["kernel_counter_key"] == "experts_forward" and hf["hf_targets"]["n_target_parameters"] == 4
    # T12/T13/T14: micro-batch 2 on the alpaca template with padding and a linear schedule, every framework
    for fw, tag in (("e4b", "fused_attn4"), ("unsloth", "ckpt_unsloth"), ("hf", "hf_peft")):
        r = json.loads((d / f"tinya_{fw}_{tag}.json").read_text())
        assert r["status"] == "ok" and r["micro_batch"] == 2 and r["template"] == "alpaca" and r["tokens_padded_total"] > 0, (fw, r["status"])
        assert r["lr_per_step"][0] == 0.0 and r["optimizer"].endswith("schedule=linear warmup_steps=3"), (fw, r["optimizer"])
    # T17 (P43): the selftest runs with --log-every 1 --microbatch-timing 1, so every step is printed and every step's
    # receipt carries one timing per micro-batch; the CELL line never carries the per-step lists
    ref = json.loads((d / "tiny_e4b_reference_attn4.json").read_text())
    assert ref["log_every"] == 1 and ref["microbatch_timing"] is True
    assert len(ref["microbatch_ms"]) == ref["steps"] and all(len(mb) == ref["accum"] for mb in ref["microbatch_ms"]), ref["microbatch_ms"][:2]
    assert all(isinstance(v, float) and v >= 0 for mb in ref["microbatch_ms"] for v in mb)
    assert sum(1 for line in p.stdout.splitlines() if line.strip().startswith(f"step {ref['steps']}/{ref['steps']} ")) >= 1
    assert "mb_ms [" in p.stdout
    assert '"microbatch_ms"' not in "".join(line for line in p.stdout.splitlines() if line.startswith("CELL "))


def test_real_run_without_prereg_refuses():
    p = _run("--framework", "hf", "--arm", "hf")   # no --selftest, no --prereg: refuse before any cell or stub
    assert p.returncode == 2, (p.returncode, (p.stdout + p.stderr)[-2000:])
    assert "--prereg is required" in p.stderr


def test_alpaca_builder_refuses_a_wrong_source(tmp_path):
    bad = tmp_path / "alpaca_data_cleaned.json"
    bad.write_text("[]")
    p = _run("--src", str(bad), "--out", str(tmp_path / "ds.json"), script=ALPACA)
    assert p.returncode == 13 and "SOURCE MISMATCH" in p.stdout, (p.returncode, p.stdout[-500:])


@pytest.mark.skipif(not os.environ.get("TP4_ALPACA_SRC"), reason="set TP4_ALPACA_SRC to a local copy of the pinned alpaca_data_cleaned.json; the test never downloads")
def test_alpaca_builder_reproduces_the_registered_sha(tmp_path):
    out = tmp_path / "ds_alpaca.json"
    p = _run("--src", os.environ["TP4_ALPACA_SRC"], "--out", str(out), script=ALPACA)
    assert p.returncode == 0, p.stdout[-500:] + p.stderr[-500:]
    assert hashlib.sha256(out.read_bytes()).hexdigest() == REGISTERED_DS_SHA


# ----------------------------------------------------------------------------- e4b#548: where the time before step 1 goes
def _selftest_receipts(extra=()):
    p = _run("--selftest", *extra)
    assert p.returncode == 0, (p.stdout + p.stderr)[-3000:]
    d = Path(re.search(r"SELFTEST OK dir=(\S+)", p.stdout).group(1))
    return p, d


def test_phase_seconds_account_for_the_whole_prologue():
    """#548: every receipt carries a named breakdown of the window before step 1, and the parts add up to the whole.

    The residual (`prologue_unattributed_s`) is the field that makes this a gate rather than decoration: the issue was
    raised because ~33 min sat in a window the receipt could not describe, so a breakdown that can quietly omit a phase
    would reproduce the same failure with more JSON.
    """
    p, d = _selftest_receipts()
    assert "PROLOGUE " in p.stdout, "the arm must print its phase table as it opens the timed window"
    for name in ("tiny_e4b_fused_attn4", "tiny_unsloth_ckpt_unsloth", "tiny_hf_hf_peft"):
        r = json.loads((d / f"{name}.json").read_text())
        ph, tot = r["phase_seconds"], r["prologue_s"]
        # the phases the issue asked for by name, plus the loader-side ones
        for k in ("preamble", "load_weights", "census", "trainable_sha", "counters", "c1_before", "eval0", "optimizer"):
            assert k in ph, (name, k, sorted(ph))
        assert "c1_after" in ph and "adapter_save" in ph, (name, sorted(ph))     # the epilogue's second full C1 pass
        named = sum(v for k, v in ph.items() if k not in ("c1_after", "adapter_save", "eval_final"))
        assert abs(named + r["prologue_unattributed_s"] - tot) < 0.05, (name, named, r["prologue_unattributed_s"], tot)
        assert 0 <= r["prologue_unattributed_s"] <= 0.5 * tot + 0.05, (name, r["prologue_unattributed_s"], tot)
        assert all(v >= 0 for v in ph.values()), ph


def test_a_refused_row_also_says_where_its_time_went():
    """#548: the row the issue complains about is an ALARMED/refused one. A breakdown only ok-rows carry is useless."""
    _, d = _selftest_receipts()
    r = json.loads((d / "tiny_unsloth_ckpt_unsloth_refuse.json").read_text())
    assert r["status"] == "refused" and r["phase_seconds"], r
    assert "preamble" in r["phase_seconds"], r["phase_seconds"]
    # ... and a row that died before anything was timed still carries the (empty) table, so "nothing was measured"
    # is distinguishable from "this receipt predates the field"
    bad = json.loads((d / "tiny_unsloth_ckpt_unsloth_badsha.json").read_text())
    assert bad["status"] == "tokens_mismatch" and bad["phase_seconds"] == {} and bad["phase_in_flight"] is None, bad


def test_an_over_budget_prologue_refuses_itself_and_names_the_phase():
    """#548 (3): loud, not after the fact.

    What this replaces is a real tp4 row -- `status: alarm`, *"the process could not write its own stub"* -- where
    SIGALRM killed an arm 3570 s into a prologue and no phase could be named. Here the arm refuses itself from inside
    the offending phase: exit 16, status `phase_alarm`, the phase in flight and every phase already closed.
    """
    p = _run("--selftest", "--phase-budget-s", "0.001")
    out = p.stdout + p.stderr
    assert p.returncode == 16, (p.returncode, out[-3000:])
    assert "PHASE ALARM " in p.stdout, out[-2000:]
    cell = [ln for ln in p.stdout.splitlines() if ln.startswith("CELL PHASE_ALARM ")]
    assert len(cell) == 1, p.stdout[-2000:]
    r = json.loads(cell[0][len("CELL PHASE_ALARM "):])
    assert r["status"] == "phase_alarm" and r["phase_in_flight"], r
    assert r["phase_in_flight"] in r["phase"], r                      # the row names the phase it died in
    assert r["phase_seconds"] and r["phase_elapsed_s"] is not None and r["phase_budget_s"] is not None, r
    # the same phase appears as in-flight in the table, so the table is never silently short of the running phase
    assert any(k.startswith(r["phase_in_flight"]) for k in r["phase_seconds"]), r["phase_seconds"]


def test_the_budget_defaults_to_a_share_of_the_arms_own_alarm():
    """#548: the budget has to mean 'this arm can no longer finish', which is a fact about the alarm tp4_run.sh chose."""
    arm = _load_arm_module()
    def ns(**kw):
        return type("A", (), dict({"phase_budget_s": None}, **kw))()
    old = {k: os.environ.get(k) for k in ("TP4_PHASE_BUDGET_S", "TP4_ARM_ALARM_S")}
    try:
        for k in old:
            os.environ.pop(k, None)
        assert arm.phase_budget_for(ns()) == 0.0                       # unset: record only, never fire
        os.environ["TP4_ARM_ALARM_S"] = "3600"
        assert arm.phase_budget_for(ns()) == 3600 * arm.PROLOGUE_BUDGET_SHARE
        os.environ["TP4_PHASE_BUDGET_S"] = "900"
        assert arm.phase_budget_for(ns()) == 900.0                     # the explicit env wins over the share
        assert arm.phase_budget_for(ns(phase_budget_s=42.0)) == 42.0   # the flag wins over both
        assert arm.phase_budget_for(ns(phase_budget_s=0.0)) == 0.0     # ... including an explicit 0, which turns it off
        os.environ["TP4_PHASE_BUDGET_S"] = "0"
        assert arm.phase_budget_for(ns()) == 0.0                       # and the env can turn it off under a run script
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_tp4_run_sh_tells_the_arm_the_alarm_it_runs_under():
    """#548: an unexported budget is an inert watchdog -- the arm cannot derive one from an alarm it is not told."""
    sh = (REPO / "bench" / "tp4" / "tp4_run.sh").read_text()
    assert "TP4_ARM_ALARM_S=$A perl -e \"alarm $A; exec @ARGV\"" in sh, "the arm must be given the same A perl alarms on"


def _load_arm_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tp4_arm_under_test", ARM)
    m = importlib.util.module_from_spec(spec)
    sys.modules["tp4_arm_under_test"] = m
    spec.loader.exec_module(m)
    return m


def test_load_e4b_records_its_own_phases(monkeypatch):
    """#548: the loader's split is exercised, not merely written.

    `load_s` already covered the attention-4-bit conversion, `add_attention_lora` and `enable_fast_train` -- they are
    inside `load_e4b`. Splitting them is what lets a receipt say which of the four the loader's seconds went to, so
    the split has to be driven end to end here; a GPU is not available in CI and is not needed for the bookkeeping.
    """
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
    lora.DETECTOR_VERSION = "test"                  # avoids inspect.getsource on a synthesised module
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
        assert ph[k] >= 0.04, (k, ph)               # each slept 0.05 s: the phase it was billed to is the one it ran in
    assert x["n_patched"] == 2 and calls.count("enable") == 1


def test_the_prologue_residual_cannot_go_negative_from_rounding():
    """e4b#629 CI: `prologue_unattributed_s` read -0.001 and the invariant assert fired.

    Every phase lies inside the prologue window, so in exact arithmetic
    total - sum(parts) >= 0 always. But each part is rounded to 1 ms for the
    table, and ~10 of those roundings can accumulate PAST the total. Computing
    the residual from the rounded parts therefore produced a negative number
    for a quantity that cannot be negative -- an arithmetic artefact presented
    as a measurement.

    This reproduces it from the mechanism rather than waiting for CI to roll it
    again: ten phases of 0.6 ms each round UP to 1 ms apiece (10 ms of rounded
    parts) inside a 7 ms window.
    """
    import importlib.util
    import pathlib
    import time
    spec = importlib.util.spec_from_file_location(
        "tp4_arm_mod", pathlib.Path(__file__).resolve().parents[1] / "bench" / "tp4" / "tp4_arm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ph = mod.Phases()
    t0 = time.perf_counter()
    ph.begin(t0)
    for i in range(10):
        ph.mark(f"p{i}", 0.0006)          # rounds to 0.001 each -> 0.010 rounded, 0.006 exact
    ph._t0 = time.perf_counter() - 0.007  # a 7 ms window: exact parts (6 ms) fit, rounded parts (10 ms) do not
    ph.end_prologue()
    r = ph.report()

    assert r["prologue_unattributed_s"] >= 0, (
        "the residual is a duration and cannot be negative; "
        f"got {r['prologue_unattributed_s']} from parts {r['phase_seconds']}")
    # and it is still the honest remainder, not a clamp: exact total minus exact parts
    assert r["prologue_unattributed_s"] <= r["prologue_s"]
