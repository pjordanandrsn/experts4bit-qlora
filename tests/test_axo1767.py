"""#1767's four-score reading (bench/tc1/AXO1767-PREREG.md): the reducer on hp1k-5090-8's real scores (1) and (2) and on every cell of
the registered table, the rescore switches in tc1_arm.py, and the qwen3axo1767 token's control flow in tc1_run.sh with every harness
call stubbed. No model, no GPU."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ARM = REPO / "bench" / "tc1" / "tc1_arm.py"
RUN_SH = REPO / "bench" / "tc1" / "tc1_run.sh"
DRIVE_SH = REPO / "bench" / "tc1" / "tc1_drive.sh"
REDUCE = REPO / "bench" / "tc1" / "axo1767_reduce.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


R = _load(REDUCE, "axo1767_reduce_under_test")

# hp1k-5090-8's committed receipt, the fields the reducer reads: the two shake arms' eval_curve and eval_shake.json's arms and bases.
REAL_AX = {"tag": "hp1shake_axolotl", "eval_curve": [{"step": 0, "heldout_loss": 10.99624}, {"step": 3, "heldout_loss": 10.94001}]}
REAL_E4B = {"tag": "hp1shake_e4b", "eval_curve": [{"step": 0, "heldout_loss": 11.00224}, {"step": 3, "heldout_loss": 10.98109}]}
REAL_EVAL = {"bases": [{"dir": "/root/tc1/hp1/shake/base_e4b_qwen3hp1shake", "status": "OK", "zero_adapter_vs_inloop_step0": {"hp1shake_e4b": -0.002331}},
                       {"dir": "/root/tc1/hp1/shake/base_axolotl_qwen3hp1shake", "status": "OK",
                        "zero_adapter_vs_inloop_step0": {"hp1shake_axolotl": -0.004365}}],
             "arms": [{"tag": "hp1shake_axolotl", "status": "VOID", "evaluator": {"row_mean": 10.974538}},
                      {"tag": "hp1shake_e4b", "status": "SCORED", "evaluator": {"row_mean": 10.979891}}]}


def _w(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return str(p)


def _rescore(tmp_path, name, loss, zero=10.996, status="ok"):
    return _w(tmp_path, name, {"status": status, "adapter_loss": loss, "zero_adapter_loss": zero, "reason": None if status == "ok" else "x"})


def _real(tmp_path, ax=REAL_AX, ev=REAL_EVAL):
    return _w(tmp_path, "ax.json", ax), _w(tmp_path, "ev.json", ev), _w(tmp_path, "e4b.json", REAL_E4B)


def test_the_real_record_alone_reads_incomplete_with_its_own_numbers(tmp_path):
    ax, ev, ctl = _real(tmp_path)
    out = R.reduce(ax, ev, control=ctl)
    assert out["verdict"] == "INCOMPLETE", out
    assert out["pairs"]["2_minus_1"] == {"delta": 0.034528, "band": "DIFFER"}
    assert out["checks"]["control_round_trip"] == {"tag": "hp1shake_e4b", "delta": -0.001199, "band": "MATCH"}
    assert out["checks"]["evaluator_zero_vs_inloop_step0"] == -0.004365


@pytest.mark.parametrize("on,off,verdict", [
    (10.9420, 10.9750, "KERNELS"),          # A +0.0020, B +0.0005, C +0.0330
    (10.9410, 10.9400, "EXPORT"),           # A +0.0010, B -0.0345
    (10.9750, 10.9760, "PROCESS_STATE"),    # A +0.0350, B +0.0015
    (10.9900, 10.9400, "NEITHER"),          # A +0.0500, B -0.0345
    (10.9550, 10.9750, "INCONCLUSIVE"),     # A +0.0150, between the bands
])
def test_every_cell_of_the_registered_table_on_the_real_scores_1_and_2(tmp_path, on, off, verdict):
    ax, ev, ctl = _real(tmp_path)
    out = R.reduce(ax, ev, on=_rescore(tmp_path, "on.json", on), off=_rescore(tmp_path, "off.json", off), control=ctl)
    assert out["verdict"] == verdict, out


def test_a_chain_of_matches_that_does_not_close_is_inconclusive(tmp_path):
    ax = {"tag": "hp1shake_axolotl", "eval_curve": [{"step": 0, "heldout_loss": 11.0}, {"step": 3, "heldout_loss": 10.000}]}
    ev = {"bases": [{"zero_adapter_vs_inloop_step0": {"hp1shake_axolotl": 0.0}}], "arms": [{"tag": "hp1shake_axolotl", "evaluator": {"row_mean": 10.025}}]}
    out = R.reduce(_w(tmp_path, "ax.json", ax), _w(tmp_path, "ev.json", ev),
                   on=_rescore(tmp_path, "on.json", 10.009, zero=11.0), off=_rescore(tmp_path, "off.json", 10.017, zero=11.0))
    assert out["pairs"]["2_minus_1"]["band"] == "DIFFER" and {out["pairs"][k]["band"] for k in ("A_4_minus_1", "B_3_minus_2", "C_3_minus_4")} == {"MATCH"}
    assert out["verdict"] == "INCONCLUSIVE", out


def test_a_closed_gap_reads_not_reproduced(tmp_path):
    ax = dict(REAL_AX, eval_curve=[{"step": 0, "heldout_loss": 10.99624}, {"step": 3, "heldout_loss": 10.97000}])
    out = R.reduce(*_real(tmp_path, ax=ax)[:2])
    assert out["verdict"] == "NOT_REPRODUCED" and out["pairs"]["2_minus_1"]["band"] == "MATCH", out


@pytest.mark.parametrize("case", ["control", "zero", "status", "noeval"])
def test_an_instrument_failure_voids_the_reading(tmp_path, case):
    ev = json.loads(json.dumps(REAL_EVAL))
    on, off = 10.9420, 10.9750
    kw = {}
    if case == "control":
        ev["arms"][1]["evaluator"]["row_mean"] = 10.98109 + 0.03
    if case == "noeval":
        ev["arms"] = ev["arms"][1:]
    ax, evp, ctl = _real(tmp_path, ev=ev)
    kw["on"] = _rescore(tmp_path, "on.json", on, zero=10.996 + (0.02 if case == "zero" else 0.0))
    kw["off"] = _rescore(tmp_path, "off.json", off, status="refused" if case == "status" else "ok")
    out = R.reduce(ax, evp, control=ctl, **kw)
    assert out["verdict"] == "VOID", out


def test_main_prints_the_verdict_line_and_writes_the_reading(tmp_path):
    ax, ev, ctl = _real(tmp_path)
    o = tmp_path / "fourscore.json"
    p = subprocess.run([sys.executable, str(REDUCE), "--arm", ax, "--eval", ev, "--control", ctl, "--out", str(o)],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and p.stdout.startswith("AXO1767 VERDICT INCOMPLETE: "), p.stdout + p.stderr
    assert json.loads(o.read_text())["verdict"] == "INCOMPLETE"


# ---------------------------------------------------------------- tc1_arm.py: the switches


def test_kernels_off_is_peft_reference_config_and_never_combines_with_best():
    arm = _load(ARM, "tc1_arm_axo1767_under_test")
    import types
    a = types.SimpleNamespace(r=16, alpha=16, seq=2048, seed=3407, micro_batch=2, accum=4, steps=3, lr=2e-4, weight_decay=0.001, warmup_steps=5)
    off = arm.axolotl_config_dict(a, "/snap", ["m"], ["p"], best=False, kernels=False)
    for k in ("lora_mlp_kernel", "lora_qkv_kernel", "lora_o_kernel", "lora_embedding_kernel"):
        assert off[k] is False, k
    assert off["experts_implementation"] == "eager" and "plugins" not in off and "expert_backend" not in off
    on = arm.axolotl_config_dict(a, "/snap", ["m"], ["p"], best=True)
    assert "lora_qkv_kernel" not in on and "experts_implementation" not in on and on["expert_backend"] == "scattermoe"
    with pytest.raises(ValueError):
        arm.axolotl_config_dict(a, "/snap", ["m"], ["p"], best=True, kernels=False)


@pytest.mark.parametrize("args,needle", [
    (["--framework", "axolotl", "--arm", "axolotl", "--axolotl-kernels", "off", "--axolotl-best", "1"], "--axolotl-best 0"),
    (["--framework", "e4b", "--arm", "fused", "--axolotl-kernels", "off"], "axolotl switch"),
])
def test_the_switch_refuses_a_bad_combination_before_anything_loads(tmp_path, args, needle):
    out = tmp_path / "out"
    out.mkdir()
    p = subprocess.run([sys.executable, str(ARM), "--tag", "z", "--prereg", "tc1/AXO1767-PREREG.md", "--out", str(out), "--adapter-dir", str(out / "ad"), *args],
                       capture_output=True, text=True, timeout=300, cwd=REPO)
    assert p.returncode == 3 and "Traceback" not in p.stderr, (p.stdout + p.stderr)[-2000:]
    rec = json.loads(next(out.glob("*_z.json")).read_text())
    assert rec["status"] == "refused" and needle in rec["reason"], rec


# ---------------------------------------------------------------- tc1_run.sh: the token's control flow, stubbed

START = "# ---------------------------------------------------------------- AXO1767 (bench/tc1/AXO1767-PREREG.md"   # through the HP1 functions it uses
END = "tc1_curve_family(){"
FAKEPY = r"""#!/bin/bash
echo "PY: $*" >> "$W/py.log"
case "$*" in
  *hp1_trunc.py*) echo '{"model": "local/cut", "revision": "abc"}';;
  *hp1_eval.py*) echo "EVAL_OFFLINE=${HF_HUB_OFFLINE:-unset}" >> "$W/py.log"; out=$(echo "$*" | sed -n 's/.*--out \([^ ]*\).*/\1/p'); echo '{}' > "$out"; exit 3;;
  *axo1767_reduce.py*) echo "AXO1767 VERDICT STUB: reducer called";;
  *"import json,sys"*) python3 -c "${@:2}";;
esac
exit 0
"""
HARNESS = r"""#!/bin/bash
W=$PWD; HP1_EVAL_N=48; HP1_EVAL_EVERY=200; HP1_SEL_SEEDS="3407 3408 3409"; HP1_CONF_SEEDS="3410 3411 3412"
HP1_SHAKE_LAYERS=2; HP1_SHAKE_STEPS=3; CURVE_STEPS=200; HP1_AX_DROP=0
PY_E4B=$W/fakepy
say(){ echo "SAY: $*" >> "$W/say.log"; }
can_run(){ return 0; }
tc1_prepare(){ TOK=$W/tokens.json; TS=deadbeef; return 0; }
arm(){ echo "$1 $2 $3 ${*:12}" >> "$W/arms.log"; touch "$W/${1}_${2}_${3}.json"
  if [ "$2/$3" = axolotl/hp1shake_axolotl ] && [ -z "${NO_ADAPTER:-}" ]; then mkdir -p "$W/adapters/${1}_$3"; echo x > "$W/adapters/${1}_$3/adapter_model.safetensors"; fi; }
status_of(){ [ "${FAIL:-}" = "$2/$3" ] && echo install_failed || echo ok; }
mkdir -p logs hp1 data; : > summary.txt; : > arms.log
echo '{"fam": "qwen3hp1", "sha256": "deadbeef", "train": [], "eval": []}' > tokens.json
source ./funcs.sh
tc1_axo1767_family qwen3hp1 Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400
"""


def drive(tmp_path, **env):
    src = RUN_SH.read_text()
    (tmp_path / "funcs.sh").write_text(src[src.index(START):src.index(END)])
    (tmp_path / "fakepy").write_text(FAKEPY)
    (tmp_path / "fakepy").chmod(0o755)
    (tmp_path / "harness.sh").write_text(HARNESS)
    p = subprocess.run(["bash", "harness.sh"], cwd=tmp_path, env=dict(os.environ, W=str(tmp_path), **env), capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    arms = [ln.split() for ln in (tmp_path / "arms.log").read_text().splitlines()]
    return arms, (tmp_path / "summary.txt").read_text(), (tmp_path / "py.log").read_text()


def test_the_token_trains_two_arms_rescores_on_and_off_then_reads(tmp_path):
    arms, summary, py = drive(tmp_path)
    assert [a[1:3] for a in arms] == [["e4b", "hp1shake_e4b"], ["axolotl", "hp1shake_axolotl"], ["axolotl", "axo1767_rescore_on"], ["axolotl", "axo1767_rescore_off"]]
    assert all(a[0] == "qwen3hp1shake" for a in arms)
    ax, on, off = arms[1], arms[2], arms[3]
    assert "--hp1-export" in ax and ax[ax.index("--axolotl-best") + 1] == "1" and "--rescore-adapter" not in ax
    ad = str(tmp_path / "adapters" / "qwen3hp1shake_hp1shake_axolotl")
    for r, k, best in ((on, "on", "1"), (off, "off", "0")):
        assert r[r.index("--rescore-adapter") + 1] == ad and r[r.index("--axolotl-kernels") + 1] == k and r[r.index("--axolotl-best") + 1] == best
        assert "--hp1-export" not in r
    ev = [ln for ln in py.splitlines() if "hp1_eval.py" in ln]
    assert len(ev) == 1 and "--require-all-scored" not in ev[0] and "--zero-tol 0.01" in ev[0] and "EVAL_OFFLINE=1" in py
    assert "AXO1767 evaluator rc=3" in summary and "AXO1767 VERDICT STUB" in summary
    assert (tmp_path / "axo1767" / "peft_adapter_axolotl" / "adapter_model.safetensors").exists()


@pytest.mark.parametrize("env", [{"FAIL": "axolotl/hp1shake_axolotl"}, {"NO_ADAPTER": "1"}])
def test_no_trained_axolotl_adapter_voids_without_a_rescore(tmp_path, env):
    arms, summary, py = drive(tmp_path, **env)
    assert [a[2] for a in arms] == ["hp1shake_e4b", "hp1shake_axolotl"] and "AXO1767 VERDICT VOID" in summary and "axo1767_reduce" not in py


def test_the_token_is_dispatched_staged_and_needs_no_unsloth_venv():
    run = RUN_SH.read_text()
    assert "\n  qwen3axo1767) tc1_axo1767_family qwen3hp1 " in run
    assert 'case " $FAMILIES " in " qwen3axo1767 ") NEED_UNSLOTH=0;; esac' in run
    assert "PREREG=tc1/AXO1767-PREREG.md" in run
    drive_sh = DRIVE_SH.read_text()
    line = next(ln for ln in drive_sh.splitlines() if '*" qwen3axo1767 "*)' in ln)
    for f in ("hp1_lib.py", "hp1_eval.py", "hp1_trunc.py", "axo1767_reduce.py"):
        assert f in line, f
