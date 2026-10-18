"""The qwen3hp1 token's control flow (bench/tc1/tc1_run.sh, the HP1 functions) with every harness call stubbed: which runs it
starts, when it ends the draw, what it redraws. The functions are extracted from tc1_run.sh itself, so a change there is what these
tests run. Bash only; no model, no GPU."""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUN_SH = REPO / "bench" / "tc1" / "tc1_run.sh"
START = "# ---------------------------------------------------------------- HP1 (bench/hp1/HP1-PREREG.md)"
END = "tc1_curve_family(){"

# The stub interpreter: answers the driver's python calls from the environment of the scenario.
FAKEPY = r"""#!/bin/bash
echo "PY: $*" >> "$W/py.log"
case "$*" in *hp1_eval.py*) echo "EVAL_OFFLINE=${HF_HUB_OFFLINE:-unset}" >> "$W/py.log";; esac
case "$*" in
  *hp1_trunc.py*) echo '{"model": "local/cut", "revision": "abc"}';;
  *"hp1_reduce.py select"*) [ -n "${SEL:-}" ] && echo "HP1_SELECTED=$SEL";;
  *"hp1_reduce.py voids"*eval_early*) [ "${EARLY:-ok}" = runvoid ] && [ ! -e "$W/early_redrawn" ] && { echo hp1_k00_s3407; touch "$W/early_redrawn"; };;
  *"hp1_reduce.py voids"*eval_K*) [ -n "${VOID_K:-}" ] && [ ! -e "$W/k_redrawn" ] && { echo "$VOID_K"; touch "$W/k_redrawn"; };;
  *"hp1_reduce.py voids"*) ;;
  *hp1_eval.py*)
    out=$(echo "$*" | sed -n 's/.*--out \([^ ]*\).*/\1/p'); st=OK
    case "$out" in *eval_early*) case "${EARLY:-ok}" in basevoid) st=VOID;; runvoid) [ -e "$W/early_redrawn" ] && st=OK || st=UNUSED;; esac;; esac
    echo "{\"bases\":[{\"seconds\":120,\"status\":\"$st\"}],\"arms\":[{\"tag\":\"x\"}]}" > "$out";;
  *"import json,sys"*) python3 -c "${@:2}";;
esac
exit 0
"""

HARNESS = r"""#!/bin/bash
W=$PWD; HP1_EVAL_N=48; HP1_EVAL_EVERY=200; HP1_SEL_SEEDS="3407 3408 3409"; HP1_CONF_SEEDS="3410 3411 3412"
HP1_SHAKE_LAYERS=2; HP1_SHAKE_STEPS=3; CURVE_STEPS=200; UT7="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
PY_E4B=$W/fakepy
say(){ echo "SAY: $*" >> "$W/say.log"; }
can_run(){ case "$2" in ${NO_TIME_FOR:-__none__}) echo "SKIPPED $2" >> summary.txt; return 1;; esac; return 0; }
tc1_prepare(){ TOK=$W/tokens.json; TS=deadbeef; return 0; }
arm(){ echo "$1 $2 $3 ${*:12}" >> "$W/arms.log"; touch "$W/${1}_${2}_${3}.json"; }
status_of(){ [ "${SHAKE_FAIL:-}" = "$2" ] && echo install_failed || echo ok; }
hp1_bulk_manifest(){ echo "manifest $1" >> "$W/manifest.log"; }
mkdir -p logs hp1 data; : > summary.txt; : > arms.log
source ./hp1_funcs.sh
tc1_hp1_family qwen3hp1 Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 2400 3600 3600
"""


def drive(tmp_path, **env):
    src = RUN_SH.read_text()
    (tmp_path / "hp1_funcs.sh").write_text(src[src.index(START):src.index(END)])
    (tmp_path / "fakepy").write_text(FAKEPY)
    (tmp_path / "fakepy").chmod(0o755)
    (tmp_path / "harness.sh").write_text(HARNESS)
    e = dict(os.environ, W=str(tmp_path))
    e.update({"SEL": "k10", **env})
    p = subprocess.run(["bash", "harness.sh"], cwd=tmp_path, env=e, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    arms = [ln.split() for ln in (tmp_path / "arms.log").read_text().splitlines()]
    summary = (tmp_path / "summary.txt").read_text()
    return arms, summary


def tags(arms, fam="qwen3hp1"):
    return [a[2] for a in arms if a[0] == fam]


def test_the_full_path_runs_k_then_p_with_one_base_export_each(tmp_path):
    arms, summary = drive(tmp_path)
    t = tags(arms)
    k = [x for x in t if x.startswith("hp1_k") and x.endswith(("3407", "3408", "3409"))]
    assert len(k) == 12 and t[0] == "hp1_k00_s3407"
    assert "HP1 SHAKEDOWN_OK" in summary and "HP1 EARLY_BASE_CHECK_OK" in summary and "HP1_SELECTED=k10" in summary
    p = [x for x in t if x.endswith(("3410", "3411", "3412"))]
    assert sorted(p) == sorted([f"hp1_{r}_s{s}" for s in (3410, 3411, 3412) for r in ("e_k10", "k00", "unsloth", "axolotl")])
    exporters = [a[2] for a in arms if a[0] == "qwen3hp1" and "--hp1-export-base" in a]
    assert exporters == ["hp1_k00_s3407", "hp1_unsloth_s3410", "hp1_axolotl_s3410"]
    k_args = {a[2]: a for a in arms if a[0] == "qwen3hp1"}
    assert "--order-seed" in k_args["hp1_k10_s3408"] and "matched:3408" in k_args["hp1_k10_s3408"]


def test_a_failed_box_side_shakedown_ends_the_draw_unscored(tmp_path):
    arms, summary = drive(tmp_path, SHAKE_FAIL="axolotl")
    assert "HP1 SHAKEDOWN_FAILED" in summary and tags(arms) == []


def test_a_base_level_early_failure_ends_the_draw_after_one_run(tmp_path):
    arms, summary = drive(tmp_path, EARLY="basevoid")
    assert "HP1 EARLY_BASE_CHECK_FAILED VOID" in summary and tags(arms) == ["hp1_k00_s3407"]
    assert "HP1_REDRAW" not in summary                    # a base-level VOID is never redrawn


def test_a_run_level_early_void_is_redrawn_once_then_the_k_block_runs(tmp_path):
    arms, summary = drive(tmp_path, EARLY="runvoid")
    t = tags(arms)
    assert t.count("hp1_k00_s3407") == 2 and "HP1_REDRAW hp1_k00_s3407" in summary and "HP1 EARLY_BASE_CHECK_OK" in summary
    redraw = [a for a in arms if a[0] == "qwen3hp1" and a[2] == "hp1_k00_s3407"][1]
    assert "--hp1-export-base" in redraw                  # the redraw re-exports its framework's base
    assert (tmp_path / "hp1" / "void" / "qwen3hp1_e4b_hp1_k00_s3407.json.a1").exists()


def test_a_run_level_void_in_the_k_block_is_redrawn_once(tmp_path):
    arms, summary = drive(tmp_path, VOID_K="hp1_k01_s3408")
    assert tags(arms).count("hp1_k01_s3408") == 2 and summary.count("HP1_REDRAW hp1_k01_s3408") == 1


def test_an_empty_selection_skips_the_p_block(tmp_path):
    arms, summary = drive(tmp_path, SEL="")
    assert "HP1 K_INCOMPLETE" in summary
    assert not [x for x in tags(arms) if x.endswith(("3410", "3411", "3412"))]


def test_no_time_for_the_p_pass_loses_only_the_p_readout(tmp_path):
    arms, summary = drive(tmp_path, NO_TIME_FOR="hp1/eval_P*")
    assert "HP1 EVAL_P SKIPPED" in summary and "P readout lost" in summary
    assert len([x for x in tags(arms) if x.endswith(("3410", "3411", "3412"))]) == 12


@pytest.mark.parametrize("knob", ["TC1_HP1_EVAL_N", "TC1_HP1_EVAL_EVERY", "TC1_HP1_SEL_SEEDS", "TC1_HP1_CONF_SEEDS",
                                  "TC1_HP1_SHAKE_LAYERS", "TC1_HP1_SHAKE_STEPS"])
def test_every_hp1_knob_is_read_by_the_box_and_forwarded_by_the_driver(knob):
    assert knob in RUN_SH.read_text()
    assert knob in (REPO / "bench" / "tc1" / "tc1_drive.sh").read_text()


def test_the_k_pass_gate_is_sized_per_arm():
    """The gate after the 1-arm early pass is that pass's per-arm seconds x the next pass's arm count x 1.3 + 600 s."""
    src = RUN_SH.read_text()
    assert "HP1_EVAL_PER_ARM_S * n * 13) / 10 + 600" in src and "len(d['arms'])" in src


def test_every_evaluator_pass_runs_offline(tmp_path):
    """The checkpoint is already on the box; the evaluator must never reach the Hub (the rented image's huggingface_hub 2.x / httpx2
    can break content-encoded responses), so every hp1_eval.py call runs with HF_HUB_OFFLINE=1."""
    drive(tmp_path)
    lines = [ln for ln in (tmp_path / "py.log").read_text().splitlines() if ln.startswith("EVAL_OFFLINE=")]
    assert lines and all(ln == "EVAL_OFFLINE=1" for ln in lines), lines


def test_the_manifest_is_written_before_each_reducer_pass_and_handed_to_it(tmp_path):
    drive(tmp_path)
    py = (tmp_path / "py.log").read_text()
    assert (tmp_path / "manifest.log").read_text().count("manifest") >= 2
    for sub in ("select", "verdict"):
        line = [ln for ln in py.splitlines() if f"hp1_reduce.py {sub}" in ln][0]
        assert "--manifest" in line and "bulk_manifest.json" in line


def _finish_harness(tmp_path, left_s):
    """finish() + hp1_finish_manifest() + hp1_bulk_manifest(), cut from tc1_run.sh, on a fake box with one base export and one adapter."""
    src = RUN_SH.read_text()

    def cut(start, end):
        i = src.index(start)
        return src[i:src.index(end, i) + len(end)] + "\n"
    fns = (cut("hp1_bulk_manifest(){", 'tail -1 "$W/logs/hp1_bulk_manifest.log"; }')
           + cut("hp1_finish_manifest(){", 'bulk_manifest.SKIPPED"; fi; }')
           + cut("finish(){", 'exit "$rc"; }'))
    (tmp_path / "hp1" / "base_e4b_qwen3hp1").mkdir(parents=True, exist_ok=True)
    (tmp_path / "hp1" / "base_e4b_qwen3hp1" / "index.json").write_text("{}")
    (tmp_path / "hp1" / "canonical_qwen3hp1_hp1_k00_s3407.safetensors").write_bytes(b"\x01" * 64)
    (tmp_path / "logs").mkdir(exist_ok=True)
    sh = tmp_path / "finish.sh"
    sh.write_text(f'W={tmp_path}; NONCE=n; cd "$W"; say(){{ echo "SAY: $*"; }}; left(){{ echo {left_s}; }}\n' + fns + "finish 0\n")
    return subprocess.run(["bash", str(sh)], capture_output=True, text=True, timeout=60)


def test_the_finish_path_writes_the_manifest_when_time_allows(tmp_path):
    """A normal end, a refusal or a deadline stop all go through finish(): a stopped run still pins its bulk bytes."""
    import json
    p = _finish_harness(tmp_path, 100000)
    assert p.returncode == 0, p.stderr
    man = json.loads((tmp_path / "hp1" / "bulk_manifest.json").read_text())
    assert {r["path"] for r in man["files"]} == {"hp1/base_e4b_qwen3hp1/index.json", "hp1/canonical_qwen3hp1_hp1_k00_s3407.safetensors"}
    assert (tmp_path / "TP_DONE.n").exists()


def test_the_finish_path_skips_the_manifest_without_the_time_and_says_so(tmp_path):
    p = _finish_harness(tmp_path, 5)
    assert p.returncode == 0, p.stderr
    assert not (tmp_path / "hp1" / "bulk_manifest.json").exists() and (tmp_path / "hp1" / "bulk_manifest.SKIPPED").exists()
    assert (tmp_path / "TP_DONE.n").exists()


def test_a_finish_with_unhashed_bytes_skips_even_when_the_last_pass_was_fast(tmp_path):
    """The size+mtime cache makes the last pass's `seconds` tiny; a deadline stop after new adapters must size its check from the
    UNHASHED bytes x the recorded rate x 1.5 + 60 s, and SKIP (never start hashing past the guard) when that does not fit."""
    import json
    (tmp_path / "hp1").mkdir()
    (tmp_path / "hp1" / "bulk_manifest.json").write_text(json.dumps({"hp1_bulk_manifest": 1, "files": [], "seconds": 0.1,
                                                                    "hash_rate_bps": 1000}))
    (tmp_path / "hp1" / "canonical_qwen3hp1_hp1_e_k10_s3410.safetensors").write_bytes(b"\x02" * 1_000_000)   # 1 MB unhashed
    p = _finish_harness(tmp_path, 1000)            # needs 1e6 / 1000 x 1.5 + 60 = 1560 s > 1000 s left
    assert p.returncode == 0, p.stderr
    assert (tmp_path / "hp1" / "bulk_manifest.SKIPPED").exists()
    assert json.loads((tmp_path / "hp1" / "bulk_manifest.json").read_text())["files"] == []   # it never started hashing
