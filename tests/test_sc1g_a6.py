"""SC1g amendment A6 (bench/sc2/SC1g-PREREG.md): box J under A5's instrument, graded within the box by the per-window
median of per-position full KL. These tests are A6's wiring proof (the maintainer's review: box J's A5 helpers are new
wiring on a different box): the arm list is driven through the real box script with only the engine calls stubbed, and the
reducer's A6 reading is self-tested on synthetic rows (sc1g_reduce.py --self-test)."""
from __future__ import annotations

import math
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
BOX = (REPO / "bench" / "sc2" / "sc1g_box_i.sh").read_text()
RUN = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()


def test_box_j_runs_a6_within_the_box_in_priority_order():
    """box_j is A6's flow: fetches, the bake, four windows, box R's rows staged, then the arms -- conv1's (b), (a), (b') first,
    then (b) + (a) on conv2-conv4, then (c) -- and the A6 reading. No comparator, no GGUF, no proof (guard <= 1 h)."""
    assert "J) box_j;; K) box_k;; esac" in RUN and 'J) PROVE_NEEDS="";;' in RUN
    body = BOX[BOX.index("\nbox_j(){"):]
    body = body[:body.index("; }\n") + 4]
    assert "install_" not in body and "fetch_gptoss_gguf" not in body
    assert "SC1G_NCONV=4 i_windows || finish 19; i_ref_full_stage" in body and "i_arms_a6" in body
    assert "--out $W/sc1g/verdict_sc1g_a6.json" in body
    arms = BOX[BOX.index("i_arms_a6(){"):BOX.index("\nbox_j(){")]
    order = [arms.index(s) for s in ("i_e4b_full e4b_serve_served_conv1 ", "i_e4b_full e4b_a6mx_served_conv1 ",
                                     "i_e4b_full e4b_a6rep_served_conv1 ", "for SRC in conv2 conv3 conv4",
                                     "i_e4b_full e4b_a6g0_served_$SRC")]
    assert order == sorted(order), order
    # A3's box J is kept for the record under its own name; the old body is not what box J runs
    assert "box_j_a3(){" in BOX and "i_j k1 e4b_serve_v1_conv1" in BOX[BOX.index("box_j_a3(){"):BOX.index("i_arms_a6(){")]


def test_arm_a_changes_exactly_one_variable_from_the_baseline():
    """(a)'s stack is the served stack with E4B_INT4_KEEP_NF4=0 and nothing else changed; (c) adds only E4B_MXFP4_GEMV=0."""
    def val(name):
        line = next(ln for ln in BOX.splitlines() if ln.startswith(f'{name}="'))
        return set(line.split('"')[1].split())
    serve, mxpre = val("SC1G_E4B_SERVE"), val("SC1G_E4B_MXPRE")
    assert serve ^ mxpre == {"E4B_INT4_KEEP_NF4=1", "E4B_INT4_KEEP_NF4=0"}
    arms = BOX[BOX.index("i_arms_a6(){"):BOX.index("\nbox_j(){")]
    assert '"$SC1G_E4B_MXPRE"' in arms and '"$SC1G_E4B_SERVE E4B_MXFP4_GEMV=0"' in arms


def _drive(tmp_path, func, norows=False):
    """Source the real box script, stub only what touches an engine or the box, run `func`; -> the recorded calls."""
    calls = tmp_path / "calls"
    py = tmp_path / "py.sh"
    py.write_text(f'#!/bin/sh\necho "PY $*" >> {calls}\n')
    py.chmod(0o755)
    script = f"""set -u
W={tmp_path}; PY={py}; BOX=J; cd {tmp_path}
. {REPO / 'bench' / 'sc2' / 'sc1g_box_i.sh'}
phase(){{ :; }}; say(){{ :; }}; line(){{ echo "LINE $*" >> {calls}; }}; can_run(){{ return 0; }}; i_pin_ok(){{ return 0; }}
gpu_free(){{ :; }}; quiesce(){{ :; }}; finish(){{ echo "FINISH $1" >> {calls}; exit "$1"; }}
fetch_gptoss(){{ :; }}; bake_gptoss(){{ :; }}; i_windows(){{ :; }}; i_ref_full_stage(){{ echo "STAGE" >> {calls}; }}
i_e4b(){{ echo "ARM $1|$2|$3" >> {calls}; }}
i_ref_full(){{ [ -n "{'1' if norows else ''}" ] && return 1; echo "sha_$1"; }}
stub(){{ echo "STUB ${{1##*/}} $5" >> {calls}; }}
{func}
"""
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    return calls.read_text().splitlines()


def test_a6_arm_list_drives_every_arm_through_the_a5_helpers(tmp_path):
    """The real i_arms_a6 + i_e4b_full + i_ref_full chain: 13 arms in the registered order, each its own i_e4b call (one
    process each), each carrying box R's re-hashed row file, its sha and its own KL record path."""
    got = [ln for ln in _drive(tmp_path, "i_arms_a6") if ln.startswith("ARM ")]
    names = [g.split("|")[0][4:] for g in got]
    want = (["e4b_serve_served_conv1", "e4b_a6mx_served_conv1", "e4b_a6rep_served_conv1"]
            + [n for s in ("conv2", "conv3", "conv4") for n in (f"e4b_serve_served_{s}", f"e4b_a6mx_served_{s}")]
            + [f"e4b_a6g0_served_{s}" for s in ("conv1", "conv2", "conv3", "conv4")])
    assert names == want, names
    for g in got:
        name, stack, src = g[4:].split("|")
        assert f"SC1G_REF_FULL_FILE={tmp_path}/sc1g_ref_full/ref_full_{src}.npy" in stack
        assert f"SC1G_REF_FULL_SHA=sha_{src}" in stack and f"SC1G_KL_OUT={tmp_path}/sc1g/kl_{name}.npz" in stack
        assert ("E4B_INT4_KEEP_NF4=0" in stack) == name.startswith("e4b_a6mx_")
        assert ("E4B_MXFP4_GEMV=0" in stack) == name.startswith("e4b_a6g0_")


def test_a6_arms_are_refused_without_box_r_rows(tmp_path):
    """No registered rows (or a sha mismatch): every arm is a refusal stub, and no engine runs."""
    calls = _drive(tmp_path, "i_arms_a6", norows=True)
    assert not [ln for ln in calls if ln.startswith("ARM ")]
    stubs = [ln for ln in calls if ln.startswith("STUB ")]
    assert len(stubs) == 13 and all(ln.endswith(" refused") for ln in stubs)


def test_box_j_stages_the_rows_runs_the_arms_then_the_a6_reading(tmp_path):
    calls = _drive(tmp_path, "box_j")
    stage = calls.index("STAGE")
    arms = [i for i, ln in enumerate(calls) if ln.startswith("ARM ")]
    reduce_ = [i for i, ln in enumerate(calls) if ln.startswith("PY ") and "verdict_sc1g_a6.json" in ln]
    assert arms and reduce_ and stage < arms[0] and arms[-1] < reduce_[0]


def test_the_vllm_a5_medians_are_pinned_from_the_committed_reading():
    """The descriptive bar's numbers come from sc1g-5090-a5-1's committed vLLM KL records (1e-12 relative: floats are
    compared with a tolerance, never ==)."""
    import sys

    import numpy as np
    sys.path.insert(0, str(REPO / "bench" / "sc2"))
    import sc1g_reduce as red
    d = REPO / "bench" / "h2h-2026-10-02" / "sc1g" / "receipts" / "sc1g-5090-a5-1" / "sc1g"
    for s, pinned in red.A6_VLLM_A5_MEDIAN.items():
        got = float(np.median(np.load(d / f"kl_nll_vllm_served_{s}.npz")["eng_kl"]))
        assert math.isclose(got, pinned, rel_tol=1e-12), (s, got, pinned)
