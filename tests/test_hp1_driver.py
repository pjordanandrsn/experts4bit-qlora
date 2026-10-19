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
  *"hp1_reduce.py select"*) [ -n "${STUB_SEL:-}" ] && echo "HP1_SELECTED=$STUB_SEL";;
  *"hp1_reduce.py voids"*eval_early*) [ "${STUB_EARLY:-ok}" = runvoid ] && [ ! -e "$W/early_redrawn" ] && { echo hp1_k00_s3407; touch "$W/early_redrawn"; };;
  *"hp1_reduce.py voids"*eval_K*) [ -n "${VOID_K:-}" ] && [ ! -e "$W/k_redrawn" ] && { echo "$VOID_K"; touch "$W/k_redrawn"; };;
  *"hp1_reduce.py voids"*) ;;
  *hp1_caltokens.py*) out=$(echo "$*" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
    [ "${STUB_CALTOK:-ok}" = refuse ] && { echo "HP1 CALTOKENS REFUSED: the pool's first 48 rows do not reproduce HP1's held-out rows"; exit 13; }
    tk=$(echo "$*" | sed -n 's/.*--tokens \([^ ]*\).*/\1/p')
    python3 -c "import json,sys; fam = json.load(open(sys.argv[2]))['fam']; json.dump({'fam': fam, 'sha256': 'c' * 64, 'train': [[1, 2]], 'eval': [[i] for i in range(48, 96)]}, open(sys.argv[1], 'w'))" "$out" "$tk"
    echo "HP1 CALTOKENS rows 49-96 of the held-out pool; prefix reproduces HP1's 48; sha=$(printf 'c%.0s' $(seq 64))";;
  *"hp2_reduce.py g1"*) out=$(echo "$*" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
    if [ "${STUB_G1:-pass}" = close ]; then echo '{"verdict": "CLOSE", "tau": 0.03}' > "$out"; echo "HP2_G1=CLOSE"; exit 4; fi
    echo '{"verdict": "PASS", "roundtrip_bounds": {"e4b": [-0.0013, 0.0158], "unsloth": [0.0021, 0.0162]}}' > "$out"; echo "HP2_G1=PASS";;
  *"hp2_reduce.py verdict"*) echo "HP2_VERDICT=PARITY: no measurable difference at margin m";;
  *hp1_calib.py*) out=$(echo "$*" | sed -n 's/.*--out \([^ ]*\).*/\1/p'); echo '{"verdict": "PASS", "tol": 0.0123}' > "$out"; echo "HP1_CALIB=PASS tol=0.0123";;
  *hp1_eval.py*)
    out=$(echo "$*" | sed -n 's/.*--out \([^ ]*\).*/\1/p'); st=OK
    case "$out:${STUB_SHAKE_EVAL:-ok}" in
      *eval_shake*:axvoid) echo '{"bases":[{"dir":"/x/base_e4b_s","status":"OK"},{"dir":"/x/base_unsloth_s","status":"OK"},{"dir":"/x/base_axolotl_s","status":"OK"}],"arms":[{"tag":"hp1shake_e4b","status":"SCORED"},{"tag":"hp1shake_unsloth","status":"SCORED"},{"tag":"hp1shake_axolotl","status":"VOID","reason":"STRICT round-trip 6(b)"}]}' > "$out"; exit 1;;
      *eval_shake*:e4bvoid) echo '{"bases":[{"dir":"/x/base_e4b_s","status":"OK"}],"arms":[{"tag":"hp1shake_e4b","status":"VOID","reason":"x"},{"tag":"hp1shake_unsloth","status":"SCORED"},{"tag":"hp1shake_axolotl","status":"SCORED"}]}' > "$out"; exit 1;;
      *eval_shake*:bothvoid) echo '{"bases":[{"dir":"/x/base_e4b_s","status":"OK"},{"dir":"/x/base_unsloth_s","status":"OK"},{"dir":"/x/base_axolotl_s","status":"OK"}],"arms":[{"tag":"hp1shake_e4b","status":"VOID","reason":"x"},{"tag":"hp1shake_unsloth","status":"SCORED"},{"tag":"hp1shake_axolotl","status":"VOID","reason":"y"}]}' > "$out"; exit 1;;
      *eval_shake*:basevoid) echo '{"bases":[{"dir":"/x/base_unsloth_s","status":"VOID"}],"arms":[{"tag":"hp1shake_e4b","status":"SCORED"},{"tag":"hp1shake_unsloth","status":"SCORED"},{"tag":"hp1shake_axolotl","status":"VOID"}]}' > "$out"; exit 1;;
    esac
    case "$out" in *eval_early*) case "${STUB_EARLY:-ok}" in basevoid) st=VOID;; runvoid) [ -e "$W/early_redrawn" ] && st=OK || st=UNUSED;; esac;; esac
    echo "{\"bases\":[{\"seconds\":120,\"status\":\"$st\"}],\"arms\":[{\"tag\":\"x\"}]}" > "$out";;
  *"import json,sys"*) python3 -c "${@:2}";;
esac
exit 0
"""

HARNESS = r"""#!/bin/bash
W=$PWD; HP1_EVAL_N=48; HP1_EVAL_EVERY=200; HP1_SEL_SEEDS="3407 3408 3409"; HP1_CONF_SEEDS="3410 3411 3412"
HP1_SHAKE_LAYERS=2; HP1_SHAKE_STEPS=3; CURVE_STEPS=200; UT7="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
HP1_AX_DROP=${TC1_HP1_DROP_AXOLOTL:-0}
HP1_ZERO_TOL=${TC1_HP1_ZERO_TOL:-}; HP1_CAL_CEILING=0.01
PY_E4B=$W/fakepy
say(){ echo "SAY: $*" >> "$W/say.log"; }
can_run(){ case "$2" in ${NO_TIME_FOR:-__none__}) echo "SKIPPED $2" >> summary.txt; return 1;; esac; return 0; }
tc1_prepare(){ TOK=$W/tokens.json; TS=deadbeef
  # tokenised for the family being prepared, as tc1_arm --prepare labels it (HP1's own family keeps the harness file as written)
  [ "$1" != qwen3hp1 ] && python3 -c "import json,sys; json.dump({'fam': sys.argv[1], 'sha256': 'deadbeef', 'train': [[1, 2]], 'eval': []}, open(sys.argv[2], 'w'))" "$1" "$TOK"
  return 0; }
# tc1_arm's own rule (tc1_arm.py, the tokens check): a tokens file labelled for another fam is refused, rc=13 TOKENS_MISMATCH.
arm(){ local tf; tf=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('fam') or '')" "${10}")
  if [ -n "$tf" ] && [ "$tf" != "$1" ]; then echo "$1 $2 $3" >> "$W/mismatch.log"; return 0; fi
  echo "$1 $2 $3 ${*:12}" >> "$W/arms.log"; touch "$W/${1}_${2}_${3}.json"; }
status_of(){ grep -qx "$1 $2 $3" "$W/mismatch.log" 2>/dev/null && { echo tokens_mismatch; return; }
  [ "${SHAKE_FAIL:-}" = "$2" ] && echo install_failed || echo ok; }
hp1_bulk_manifest(){ echo "manifest $1" >> "$W/manifest.log"; }
mkdir -p logs hp1 data; : > summary.txt; : > arms.log
echo '{"fam": "qwen3hp1", "sha256": "deadbeef", "train": [], "eval": []}' > tokens.json
source ./hp1_funcs.sh
${FAMILY_FN:-tc1_hp1_family} ${FAMILY_FAM:-qwen3hp1} Qwen/Qwen3-30B-A3B ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 5400 2400 3600 3600
"""


def drive(tmp_path, mutate=None, **env):
    src = RUN_SH.read_text()
    if mutate:
        src = mutate(src)
    (tmp_path / "hp1_funcs.sh").write_text(src[src.index(START):src.index(END)])
    (tmp_path / "fakepy").write_text(FAKEPY)
    (tmp_path / "fakepy").chmod(0o755)
    (tmp_path / "harness.sh").write_text(HARNESS)
    e = dict(os.environ, W=str(tmp_path))
    e.update({"STUB_SEL": "k10", **env})
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


@pytest.mark.parametrize("fw", ["e4b", "unsloth"])
def test_a_failed_box_side_shakedown_ends_the_draw_unscored(tmp_path, fw):
    arms, summary = drive(tmp_path, SHAKE_FAIL=fw)
    assert "HP1 SHAKEDOWN_FAILED" in summary and tags(arms) == []


def test_an_axolotl_shakedown_failure_drops_axolotl_and_the_e4b_k_block_runs(tmp_path):
    """HP1-PREREG Amendment 3 (hp1k-5090-3's refusal): the axolotl shake arm failing drops axolotl; e4b + Unsloth still gate the draw,
    the evaluator reads only their receipts and bases, the K block runs, and the P block runs no axolotl arm."""
    arms, summary = drive(tmp_path, SHAKE_FAIL="axolotl")
    assert "HP1 AXOLOTL_DROPPED shakedown status=install_failed" in summary and "HP1 SHAKEDOWN_OK" in summary
    t = tags(arms)
    assert len([x for x in t if x.startswith("hp1_k")]) >= 12 and not [x for x in t if "axolotl" in x]
    assert [x for x in t if x.startswith("hp1_unsloth_s")] == ["hp1_unsloth_s3410", "hp1_unsloth_s3411", "hp1_unsloth_s3412"]
    ev = next(ln for ln in (tmp_path / "py.log").read_text().splitlines() if "hp1_eval.py" in ln and "eval_shake" in ln)
    assert "hp1shake_e4b.json" in ev and "hp1shake_unsloth.json" in ev and "hp1shake_axolotl" not in ev and "base_axolotl" not in ev


def test_box_2_with_axolotl_already_dropped_runs_no_axolotl_arm_anywhere(tmp_path):
    arms, summary = drive(tmp_path, TC1_HP1_DROP_AXOLOTL="1")
    assert "HP1 SHAKEDOWN_OK" in summary and "AXOLOTL_DROPPED" not in summary
    assert not [a for a in arms if a[1] == "axolotl"]
    assert sorted(tags(arms, "qwen3hp1shake")) == ["hp1shake_e4b", "hp1shake_unsloth"]


def test_a_base_level_early_failure_ends_the_draw_after_one_run(tmp_path):
    arms, summary = drive(tmp_path, STUB_EARLY="basevoid")
    assert "HP1 EARLY_BASE_CHECK_FAILED VOID" in summary and tags(arms) == ["hp1_k00_s3407"]
    assert "HP1_REDRAW" not in summary                    # a base-level VOID is never redrawn


def test_a_run_level_early_void_is_redrawn_once_then_the_k_block_runs(tmp_path):
    arms, summary = drive(tmp_path, STUB_EARLY="runvoid")
    t = tags(arms)
    assert t.count("hp1_k00_s3407") == 2 and "HP1_REDRAW hp1_k00_s3407" in summary and "HP1 EARLY_BASE_CHECK_OK" in summary
    redraw = [a for a in arms if a[0] == "qwen3hp1" and a[2] == "hp1_k00_s3407"][1]
    assert "--hp1-export-base" in redraw                  # the redraw re-exports its framework's base
    assert (tmp_path / "hp1" / "void" / "qwen3hp1_e4b_hp1_k00_s3407.json.a1").exists()


def test_a_run_level_void_in_the_k_block_is_redrawn_once(tmp_path):
    arms, summary = drive(tmp_path, VOID_K="hp1_k01_s3408")
    assert tags(arms).count("hp1_k01_s3408") == 2 and summary.count("HP1_REDRAW hp1_k01_s3408") == 1


def test_an_empty_selection_skips_the_p_block(tmp_path):
    arms, summary = drive(tmp_path, STUB_SEL="")
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


# ----------------------------------------------------------------------------- the split path (Amendment 2: box 1 / box 2)
def _stage_box1_outputs(tmp_path, selected="k10"):
    """What box 1 carries out, staged flat beside box 2's runner, and the sha256s its receipt pins."""
    import hashlib
    import json
    (tmp_path / "eval_K.json").write_text(json.dumps({"tokens": {"eval_n": 48}, "arms": []}))
    (tmp_path / "select.json").write_text(json.dumps({"verdict": "OK", "k": {"selected": selected}}))
    (tmp_path / "bulk_manifest.json").write_text(json.dumps({"hp1_bulk_manifest": 1, "files": []}))
    return {f"TC1_HP1_K_{k}_SHA256": hashlib.sha256((tmp_path / f).read_bytes()).hexdigest()
            for k, f in (("EVAL", "eval_K.json"), ("SELECT", "select.json"), ("MANIFEST", "bulk_manifest.json"))}


def test_split_box1_ends_at_the_selection_and_prints_what_it_carries(tmp_path):
    arms, summary = drive(tmp_path, FAMILY_FN="tc1_hp1k_family")
    t = tags(arms)
    assert len([x for x in t if x.startswith("hp1_k")]) == 12 and not [x for x in t if x.endswith(("3410", "3411", "3412"))]
    assert "HP1_SELECTED=k10" in summary and all(f"HP1_CARRY {f} sha256" in summary for f in ("eval_K.json", "select.json", "bulk_manifest.json"))


def test_split_box2_refuses_without_box1s_files(tmp_path):
    arms, summary = drive(tmp_path, FAMILY_FN="tc1_hp1p_family")
    assert "HP1 P_REFUSED carry eval_K.json" in summary and tags(arms) == []


def test_split_box2_refuses_a_carried_file_that_does_not_match_box1s_hash(tmp_path):
    pins = _stage_box1_outputs(tmp_path)
    pins["TC1_HP1_K_SELECT_SHA256"] = "0" * 64
    arms, summary = drive(tmp_path, FAMILY_FN="tc1_hp1p_family", **pins)
    assert "HP1 P_REFUSED carry select.json" in summary and tags(arms) == []


def test_split_box2_refuses_a_selection_that_does_not_re_derive(tmp_path):
    pins = _stage_box1_outputs(tmp_path, selected="k01")     # select.json says k01; the staged eval_K.json re-derives k10
    arms, summary = drive(tmp_path, FAMILY_FN="tc1_hp1p_family", **pins)
    assert "HP1 P_REFUSED selection" in summary and tags(arms) == []


def test_split_box2_runs_its_own_shakedown_and_early_check_before_p(tmp_path):
    pins = _stage_box1_outputs(tmp_path)
    arms, summary = drive(tmp_path, FAMILY_FN="tc1_hp1p_family", **pins)
    assert "HP1 SHAKEDOWN_OK" in summary and "HP1 EARLY_BASE_CHECK_OK" in summary
    t = tags(arms)
    assert t[0] == "hp1_e_k10_s3410" and not [x for x in t if x.endswith(("3407", "3408", "3409"))]
    assert len(t) == 12
    first = [a for a in arms if a[0] == "qwen3hp1"][0]
    assert "--hp1-export-base" in first                    # box 2's first E run exports the e4b base for its early check
    py = (tmp_path / "py.log").read_text()
    line = [ln for ln in py.splitlines() if "hp1_reduce.py verdict" in ln][0]
    assert "eval_K_box1.json" in line and "bulk_manifest_box1.json" in line and line.count("--manifest") == 2


def test_split_box2_ends_on_a_base_level_early_failure(tmp_path):
    pins = _stage_box1_outputs(tmp_path)
    arms, summary = drive(tmp_path, FAMILY_FN="tc1_hp1p_family", STUB_EARLY="basevoid", **pins)
    assert "HP1 EARLY_BASE_CHECK_FAILED" in summary and tags(arms) == ["hp1_e_k10_s3410"]



def test_split_box2_refuses_a_box1_manifest_that_does_not_match(tmp_path):
    pins = _stage_box1_outputs(tmp_path)
    pins["TC1_HP1_K_MANIFEST_SHA256"] = "f" * 64
    arms, summary = drive(tmp_path, FAMILY_FN="tc1_hp1p_family", **pins)
    assert "HP1 P_REFUSED carry bulk_manifest.json" in summary and tags(arms) == []


def _controller_pin_check(tmp_path, pins, stage):
    src = (REPO / "bench" / "tc1" / "tc1_drive.sh").read_text()
    i = src.index("# HP1 split box 2 (qwen3hp1p")
    block = src[i:src.index("# end HP1 split box 2 pins", i)]
    sh = tmp_path / "pins.sh"
    sh.write_text('say(){ echo "SAY: $*"; }\n' + block + "echo PINS_OK\n")
    env = dict(os.environ, TC1_FAMILIES="qwen3hp1p", TC1_EXTRA_STAGE=" ".join(str(p) for p in stage), **pins)
    return subprocess.run(["bash", str(sh)], env=env, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("case", ["ok", "short_pin", "non_hex_pin", "unset_pin", "file_missing", "hash_mismatch"])
def test_the_controller_checks_box1s_carry_pins_before_renting(tmp_path, case):
    """Amendment 2: a mis-pinned box 2 is a $0 refusal on the controller (rc 78), never a paid box that spends the run id."""
    stage_dir = tmp_path / "carry"
    stage_dir.mkdir()
    pins = _stage_box1_outputs(stage_dir)
    stage = [stage_dir / f for f in ("eval_K.json", "select.json", "bulk_manifest.json")]
    if case == "short_pin":
        pins["TC1_HP1_K_EVAL_SHA256"] = pins["TC1_HP1_K_EVAL_SHA256"][:40]
    elif case == "non_hex_pin":
        pins["TC1_HP1_K_SELECT_SHA256"] = "Z" * 64
    elif case == "unset_pin":
        pins.pop("TC1_HP1_K_MANIFEST_SHA256")
    elif case == "file_missing":
        stage = stage[:2]
    elif case == "hash_mismatch":
        (stage_dir / "select.json").write_text("{}")
    p = _controller_pin_check(tmp_path, pins, stage)
    if case == "ok":
        assert p.returncode == 0 and "PINS_OK" in p.stdout, p.stdout + p.stderr
    else:
        assert p.returncode == 78 and "refusing" in p.stdout and "PINS_OK" not in p.stdout, p.stdout + p.stderr


def test_the_shakedown_hands_each_arm_a_tokens_file_labelled_for_its_own_fam(tmp_path):
    """hp1k-5090-1: the shakedown arms run as fam qwen3hp1shake and were handed the qwen3hp1 tokens file, so tc1_arm refused all
    three (rc=13 TOKENS_MISMATCH) and the draw ended SHAKEDOWN_FAILED. The stub arm applies tc1_arm's fam rule."""
    arms, summary = drive(tmp_path)
    assert not (tmp_path / "mismatch.log").exists()
    assert "HP1 SHAKEDOWN_OK" in summary
    assert sorted(tags(arms, "qwen3hp1shake")) == ["hp1shake_axolotl", "hp1shake_e4b", "hp1shake_unsloth"]
    import json
    relabelled = json.loads((tmp_path / "hp1" / "tokens_qwen3hp1shake.json").read_text())
    original = json.loads((tmp_path / "tokens.json").read_text())
    assert relabelled["fam"] == "qwen3hp1shake" and {k: v for k, v in relabelled.items() if k != "fam"} == \
        {k: v for k, v in original.items() if k != "fam"}


def test_without_the_relabel_the_shakedown_fails_as_hp1k_5090_1_did(tmp_path):
    arms, summary = drive(tmp_path, mutate=lambda s: s.replace("hp1shake $STOK $TS", "hp1shake $TOK $TS"))
    assert sorted((tmp_path / "mismatch.log").read_text().split("\n")[:-1]) == [
        "qwen3hp1shake axolotl hp1shake_axolotl", "qwen3hp1shake e4b hp1shake_e4b", "qwen3hp1shake unsloth hp1shake_unsloth"]
    assert "HP1 SHAKEDOWN_FAILED" in summary and not [t for t in tags(arms) if t.startswith("hp1_k")]


def _offline_for(fw, tag, *args):
    """arm_once's Hub-offline rule, extracted from tc1_run.sh and run on its own."""
    src = RUN_SH.read_text()
    a = src.index('  local OFFL=1; case "$FW/$TAG"')
    b = src.index("\n", src.index('[ "$FW" = axolotl ] && case " $* "', a))
    fn = "f(){ local FW=$1 TAG=$2; shift 2\n" + src[a:b] + "\n  echo $OFFL; }\nf \"$@\"\n"
    p = subprocess.run(["bash", "-c", fn, "x", fw, tag, *args], capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


def test_every_axolotl_best_arm_reaches_the_hub_for_its_kernels_whatever_its_tag():
    """hp1k-5090-3: hp1shake_axolotl ran --axolotl-best 1 offline and KernelsPlugin refused at load. The scored P arms
    (hp1_axolotl_s3410..3412) carry the same flag, so they would have refused the same way on box 2."""
    for tag in ("hp1shake_axolotl", "hp1_axolotl_s3410", "hp1_axolotl_s3412", "ckpt_axolotl_best", "ckpt_axolotl_best_200_d2"):
        assert _offline_for("axolotl", tag, "--axolotl-dataset", "d.json", "--axolotl-best", "1", "--seed", "3410") == "0", tag
    assert _offline_for("axolotl", "ckpt_axolotl_best") == "0"                 # TC1's tag rule still holds
    assert _offline_for("axolotl", "ckpt_axolotl_m", "--axolotl-best", "0") == "1"
    assert _offline_for("axolotl", "ckpt_axolotl_m") == "1"
    for fw in ("e4b", "unsloth"):                                              # the flag is axolotl's; every other arm stays offline
        assert _offline_for(fw, "hp1_x", "--axolotl-best", "1") == "1"


def test_the_hp1_axolotl_calls_carry_the_flag_the_rule_reads():
    src = RUN_SH.read_text()
    hp1 = src[src.index(START):src.index(END)]
    import re
    joined = re.sub(r"\\\n\s*", " ", hp1)                                     # backslash continuations joined
    calls = [ln for ln in joined.splitlines() if re.search(r"(?:arm|hp1_run) \S+ axolotl ", ln)]
    assert calls, "no HP1 axolotl call found"
    for ln in calls:
        assert "--axolotl-best 1" in ln, ln


def test_an_axolotl_void_at_the_shakedown_evaluator_drops_axolotl_and_the_k_block_runs(tmp_path):
    """hp1k-5090-8: the axolotl shake arm trained, then the evaluator VOIDed it (STRICT round-trip 6(b)) and --require-all-scored ended
    the draw. Amendment 3 drops axolotl for that too, when e4b and Unsloth scored and their bases are OK."""
    arms, summary = drive(tmp_path, STUB_SHAKE_EVAL="axvoid")
    assert "HP1 AXOLOTL_DROPPED shakedown evaluator VOID: STRICT round-trip 6(b)" in summary and "HP1 SHAKEDOWN_OK" in summary
    t = tags(arms)
    assert len([x for x in t if x.startswith("hp1_k")]) >= 12 and not [x for x in t if "axolotl" in x]


@pytest.mark.parametrize("case", ["e4bvoid", "basevoid", "bothvoid"])   # bothvoid: an axolotl VOID never masks an e4b VOID
def test_a_non_axolotl_evaluator_failure_still_ends_the_draw(tmp_path, case):
    arms, summary = drive(tmp_path, STUB_SHAKE_EVAL=case)
    assert "HP1 SHAKEDOWN_FAILED" in summary and "AXOLOTL_DROPPED" not in summary and tags(arms) == []


def test_the_calibration_box_runs_one_e4b_k00_step_on_held_out_pool_rows_49_to_96(tmp_path):
    """HP1-PREREG Amendment 4: the registered dataset carries only HP1's 48 held-out rows, so hp1_caltokens.py builds rows 49-96 from
    tc1_arm's held-out pool (tests/test_hp1_caltokens.py runs it on the real prepare / pack_pools); one e4b K00 run (1 step, its base
    exported) on them, then the evaluator and hp1_calib.py; calib.json is carried out with its sha256."""
    arms, summary = drive(tmp_path, FAMILY_FN="tc1_hp1cal_family", FAMILY_FAM="qwen3hp1cal")
    assert [a[:3] for a in arms] == [["qwen3hp1cal", "e4b", "hp1cal_k00_s3407"]]
    a = arms[0]
    assert "--hp1-export-base" in a and "native" in a and "--order-seed" in a
    py = (tmp_path / "py.log").read_text()
    ct = next(ln for ln in py.splitlines() if "hp1_caltokens.py" in ln)
    assert "--pack-src" in ct and "alpaca_data_cleaned.json" in ct and "--n 48" in ct and "--tokens" in ct
    assert "c" * 64 in summary and "held-out pool rows 49-96" in summary
    ev = next(ln for ln in py.splitlines() if "hp1_eval.py" in ln and "eval_cal" in ln)
    assert "tokens_qwen3hp1cal_calib.json" in ev and "--zero-tol" not in ev
    cal = next(ln for ln in py.splitlines() if "hp1_calib.py" in ln)
    assert "--ceiling 0.01" in cal and "--tag hp1cal_k00_s3407" in cal
    assert "HP1_CALIB=PASS tol=0.0123" in summary and "HP1_CARRY calib.json sha256" in summary


def test_a_calibration_tokens_refusal_ends_the_draw_unscored(tmp_path):
    arms, summary = drive(tmp_path, FAMILY_FN="tc1_hp1cal_family", FAMILY_FAM="qwen3hp1cal", STUB_CALTOK="refuse")
    assert arms == [] and "HP1 CALIB_REFUSED tokens: HP1 CALTOKENS REFUSED" in summary and "HP1_CALIB=" not in summary


def test_the_calibrated_tolerance_reaches_every_full_depth_evaluator_pass_and_not_the_shakedown(tmp_path):
    drive(tmp_path, TC1_HP1_ZERO_TOL="0.0123")
    evs = [ln for ln in (tmp_path / "py.log").read_text().splitlines() if "hp1_eval.py" in ln and "--out" in ln]
    shake = [ln for ln in evs if "eval_shake" in ln]
    deep = [ln for ln in evs if "eval_shake" not in ln]
    assert shake and deep and all("--zero-tol" not in ln for ln in shake)
    assert all("--zero-tol 0.0123" in ln for ln in deep)                  # early, K and P passes


def test_without_a_calibrated_tolerance_no_pass_overrides_the_default(tmp_path):
    drive(tmp_path)
    assert "--zero-tol" not in (tmp_path / "py.log").read_text()


def test_hp2_runs_shakedown_calibration_g1_then_e4b_k00_vs_unsloth_at_the_calibrated_bound(tmp_path):
    import json
    """HP2-PREREG: A (shakedown, e4b + Unsloth, no axolotl), B (one e4b K00 step + one Unsloth step on 148 calibration rows, both
    bases exported, the evaluator, hp2_reduce g1), C (e4b K00 + Unsloth on the P seeds), D (the P pass at tau, the verdict)."""
    arms, summary = drive(tmp_path, FAMILY_FN="tc1_hp2_family", FAMILY_FAM="qwen3hp2")
    assert not [a for a in arms if a[1] == "axolotl"]
    assert sorted(tags(arms, "qwen3hp2shake")) == ["hp1shake_e4b", "hp1shake_unsloth"]     # the shake fam is ${FAM}shake
    t = tags(arms, "qwen3hp2")
    assert t[:2] == ["hp2cal_e4b_k00_s3407", "hp2cal_unsloth_s3407"]
    assert t[2:] == [f"hp2_{fw}_s{s}" for s in (3410, 3411, 3412) for fw in ("k00", "unsloth")]
    cal = {a[2]: a for a in arms if a[2].startswith("hp2cal_")}
    assert all("--hp1-export-base" in a for a in cal.values())
    exporters = [a[2] for a in arms if a[0] == "qwen3hp2" and a[2].startswith("hp2_") and "--hp1-export-base" in a]
    assert exporters == ["hp2_k00_s3410", "hp2_unsloth_s3410"]
    py = (tmp_path / "py.log").read_text()
    ct = next(ln for ln in py.splitlines() if "hp1_caltokens.py" in ln)
    assert "--start 96" in ct and "--count 300" in ct
    evc = next(ln for ln in py.splitlines() if "hp1_eval.py" in ln and "eval_cal" in ln)
    assert "--eval-n 300" in evc and "base_e4b_qwen3hp2" in evc and "base_unsloth_qwen3hp2" in evc
    evp = next(ln for ln in py.splitlines() if "hp1_eval.py" in ln and "eval_P" in ln)
    assert "--zero-report-only 1" in evp and "--roundtrip-bounds @" in evp and "--zero-tol" not in evp
    assert json.loads((tmp_path / "hp2" / "roundtrip_bounds.json").read_text())["unsloth"] == [0.0021, 0.0162]
    assert "hp2_reduce.py verdict" in py and "--g1" in py and "HP2 G1 PASS" in summary and "HP2_VERDICT=PARITY" in summary
    assert py.index("hp2_reduce.py g1") < py.index("eval_P")              # G1 is read before the confirmation pass


def test_hp2_g1_closure_ends_the_draw_before_any_confirmation_arm(tmp_path):
    arms, summary = drive(tmp_path, FAMILY_FN="tc1_hp2_family", FAMILY_FAM="qwen3hp2", STUB_G1="close")
    t = tags(arms, "qwen3hp2")
    assert t == ["hp2cal_e4b_k00_s3407", "hp2cal_unsloth_s3407"] and "HP2 G1_CLOSED" in summary and "not a strike" in summary
    assert "hp2_reduce.py verdict" not in (tmp_path / "py.log").read_text()
