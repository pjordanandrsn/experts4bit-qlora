"""tc1_run.sh's amendment 76 family (qwen3pos76co): the pre-check and the ABBA switch, run from the script's own functions.

The family runs grouped-nf4-gemm's kernel/test_compact_delta.py at the AFTER commit on the box's GPU before any arm, and refuses at
cents on a failure or on a skipped CUDA case. It then runs cd1cl1 at BEFORE, switches venv-unsloth to AFTER for two draws, and switches
back for BEFORE's second draw. These tests cut the functions from tc1_run.sh and run them against a fake python and git on PATH, so no
GPU, network or model is needed.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = (REPO / "bench" / "tc1" / "tc1_run.sh").read_text()
BEFORE, AFTER = "c62159642cbae5c1bd281f04c8b1e4101a5e557a", "75421b30a13332c12611f428c66531a7bbb13cbd"


def _cut(pattern: str) -> str:
    m = re.search(pattern, RUN, re.DOTALL | re.MULTILINE)
    assert m, f"pattern no longer matches tc1_run.sh: {pattern}"
    return m.group(0)


FUNCS = (_cut(r"^POS76_GNF4_BEFORE=.*?\n") + _cut(r"^POS76_GNF4_AFTER=.*?\n") + _cut(r"^gnf4_t212_at\(\)\{.*?^PYG\n\}\n")
         + _cut(r"^tc1_pos76_family\(\)\{.*?free_family \$FAM \$\{MID//\\//--\}; \}\n"))

FAKE_PY = r"""#!/usr/bin/env bash
echo "PY $*" >> "$CALLS"
case "$1 ${2:-}" in
  "-m pip") case "$*" in
      *" freeze"*) echo "torch==2.12.1+cu130"; [ -f "$CALLS.dirty" ] && echo "pytest==9.0.0"; exit 0;;
      *grouped-nf4-gemm.git@*) [ -n "${FAKE_SWITCH_FAIL:-}" ] && [[ "$*" == *"$FAKE_SWITCH_FAIL"* ]] && exit 1;;
    esac; exit 0;;
  "-m pytest") echo "PYTEST_PYTHONPATH=${PYTHONPATH:-}" >> "$CALLS"; [ -n "${FAKE_PRECHECK_DIRTIES:-}" ] && touch "$CALLS.dirty"
               printf '%s\n' "${FAKE_PYTEST_OUT:-3 passed}"; exit "${FAKE_PYTEST_RC:-0}";;
esac
cat > /dev/null; exit 0
"""
FAKE_GIT = r"""#!/usr/bin/env bash
echo "GIT $*" >> "$CALLS"
if [ "$1" = clone ]; then d="${@: -1}"; mkdir -p "$d/kernel"; touch "$d/kernel/test_compact_delta.py"; fi
exit 0
"""
STUBS = r"""
say(){ echo "SAY $*"; }
stubw(){ echo "STUB $2/$3 $5 $6" >> "$CALLS"; }
tc1_prepare(){ echo "PREPARE" >> "$CALLS"; TOK=tok; TS=ts; return 0; }
can_run(){ return 0; }
arm(){ echo "ARM $3 [$TC1_ARM_EXTRA_ENV]" >> "$CALLS"; }
draw2(){ echo "DRAW2 $3 [$TC1_ARM_EXTRA_ENV]" >> "$CALLS"; }
free_family(){ :; }
quota_floor_threads(){ [ -n "${FAKE_THREADS_EMPTY:-}" ] || echo 11; }
"""


def _run(tmp: Path, gnf4_b: str = AFTER, **env) -> tuple[subprocess.CompletedProcess, list[str], str]:
    bindir = tmp / "bin"
    bindir.mkdir(exist_ok=True)
    for name, body in (("py", FAKE_PY), ("git", FAKE_GIT)):
        (bindir / name).write_text(body)
        (bindir / name).chmod(0o755)
    (tmp / "logs").mkdir(exist_ok=True)
    calls = tmp / "calls.txt"
    script = (f'W="{tmp}"; STEPS=60; EVAL_EVERY=20; EVAL_N=8; GNF4_SHA={BEFORE}; GNF4_SHA_B={gnf4_b}; T212_OK=1; T212_REASON=""\n'
              f'PY_UNS="{bindir}/py"; MATCHED_SEED=0; PHYS=48; ALLOT=12\n' + STUBS + FUNCS + "tc1_pos76_family qwen3pos76co M rev 1 1\n")
    r = subprocess.run(["bash", "-c", script], cwd=tmp, check=False, capture_output=True, text=True, timeout=60,
                       env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "CALLS": str(calls), **env})
    summary = (tmp / "summary.txt").read_text() if (tmp / "summary.txt").exists() else ""
    return r, calls.read_text().splitlines() if calls.exists() else [], summary


def _runs(calls):
    return [c for c in calls if c.startswith(("ARM ", "DRAW2 "))]


def test_a_clean_pre_check_runs_the_four_arms_in_abba_order(tmp_path: Path):
    r, calls, summary = _run(tmp_path)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "PRECHECK qwen3pos76co OK 3 passed" in summary and "QWEN3POS76CO DONE" in summary, summary
    pytest_at = next(i for i, c in enumerate(calls) if "-m pytest" in c)
    assert pytest_at < calls.index("PREPARE")                       # the pre-check runs before the model is fetched
    assert any(c == f"GIT -C {tmp_path}/gnf4-after checkout -q {AFTER}" for c in calls), calls
    x = ("[NF4_QLORA_COMPACT_DELTA=1 E4B_CHUNKED_LM_LOSS=1 OMP_NUM_THREADS=11 MKL_NUM_THREADS=11 OPENBLAS_NUM_THREADS=11 "
         "TC1_EXPECT_THREADS=11 TC1_PHYS=48]")
    assert "THREADS qwen3pos76co 11 (floor of the quota; allotment 12, physical cores 48; amendment 77)" in summary, summary
    assert _runs(calls) == [f"ARM fused_attn4_m_cd1cl1_gb {x}", f"ARM fused_attn4_m_cd1cl1_ga {x}",
                            f"DRAW2 fused_attn4_m_cd1cl1_ga {x}", f"DRAW2 fused_attn4_m_cd1cl1_gb {x}"]
    switches = [c for c in calls if "grouped-nf4-gemm.git@" in c]
    assert [s.rsplit("@", 1)[1] for s in switches] == [AFTER, BEFORE]
    order = [c for c in calls if c.startswith(("ARM ", "DRAW2 ")) or "grouped-nf4-gemm.git@" in c]
    assert [o.split()[1] if not o.startswith("PY") else o.rsplit("@", 1)[1] for o in order] == [
        "fused_attn4_m_cd1cl1_gb", AFTER, "fused_attn4_m_cd1cl1_ga", "fused_attn4_m_cd1cl1_ga", BEFORE, "fused_attn4_m_cd1cl1_gb"]


def test_the_pre_check_installs_pytest_beside_the_measured_venv_and_refuses_if_the_venv_changes(tmp_path: Path):
    r, calls, summary = _run(tmp_path)
    assert any("-m pip install" in c and f"--target {tmp_path}/gnf4-after-site pytest" in c for c in calls), calls
    assert f"PYTEST_PYTHONPATH={tmp_path}/gnf4-after-site" in calls, calls
    assert [c for c in calls if "pip freeze" in c] == ["PY -m pip freeze --all"] * 2
    d = tmp_path / "dirty"
    d.mkdir()
    r, calls, summary = _run(d, FAKE_PRECHECK_DIRTIES="1")
    assert "QWEN3POS76CO REFUSED the pre-check changed venv-unsloth's package set" in summary, summary
    assert "PREPARE" not in calls and not _runs(calls)


def test_a_failing_pre_check_refuses_the_family_before_the_fetch(tmp_path: Path):
    r, calls, summary = _run(tmp_path, FAKE_PYTEST_RC="1", FAKE_PYTEST_OUT="1 failed, 2 passed")
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "QWEN3POS76CO REFUSED pre-check FAILED" in summary, summary
    assert "PREPARE" not in calls and not _runs(calls)
    assert len([c for c in calls if c.startswith("STUB ")]) == 4 and all(" not_run pre-check FAILED" in c for c in calls if c.startswith("STUB "))


def test_a_skipped_cuda_case_refuses_the_family(tmp_path: Path):
    out = "SKIPPED [1] test_compact_delta.py:88: the peak is a CUDA allocator statistic\n2 passed, 1 skipped"
    r, calls, summary = _run(tmp_path, FAKE_PYTEST_OUT=out)
    assert "QWEN3POS76CO REFUSED pre-check incomplete: a CUDA case skipped" in summary, summary
    assert "PREPARE" not in calls and not _runs(calls)


def test_a_wrong_after_commit_refuses_without_running_the_pre_check(tmp_path: Path):
    r, calls, summary = _run(tmp_path, gnf4_b="a83f6bb3" + "0" * 32)
    assert "amendment 76's AFTER side is grouped-nf4-gemm " + AFTER in summary, summary
    assert not any("-m pytest" in c for c in calls) and not any(c.startswith("GIT ") for c in calls)
    assert "PREPARE" not in calls and not _runs(calls)


def test_a_failed_switch_leaves_only_that_side_not_run(tmp_path: Path):
    r, calls, summary = _run(tmp_path, FAKE_SWITCH_FAIL=AFTER)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert [c.split()[1] for c in _runs(calls)] == ["fused_attn4_m_cd1cl1_gb", "fused_attn4_m_cd1cl1_gb"]
    stubs = [c for c in calls if c.startswith("STUB ")]
    assert [s.split()[1] for s in stubs] == ["e4b/fused_attn4_m_cd1cl1_ga", "e4b/fused_attn4_m_cd1cl1_ga_d2"], stubs
    assert all("the switch to grouped-nf4-gemm " + AFTER + " failed" in s for s in stubs)


FLOOR = _cut(r"^POS76_MIN_ALLOT=.*?^esac\n")


def _floor(tmp: Path, allot: int, families: str) -> str:
    script = ('say(){ echo "SAY $*"; }\nfinish(){ echo "FINISH $1"; exit "$1"; }\n'
              f'ALLOT={allot}; FAMILIES="{families}"\n' + FLOOR + 'echo PASSED\n')
    r = subprocess.run(["bash", "-c", script], cwd=tmp, capture_output=True, text=True, timeout=30)
    return r.stdout + ((tmp / "summary.txt").read_text() if (tmp / "summary.txt").exists() else "")


def test_the_cpu_floor_refuses_a_slow_box_before_any_venv(tmp_path: Path):
    """Amendment 76's pre-flight floor: an allotment under 42 refuses the box (finish 18) when the family is on it; 42 and up pass, and
    other families are not affected."""
    assert "POS76_MIN_ALLOT=42" in FLOOR
    out = _floor(tmp_path, 41, "qwen3pos76co")
    assert "FINISH 18" in out and "BOX_REFUSED cpu_allotment=41 floor=42 (amendment 76)" in out and "PASSED" not in out, out
    for allot, fam in ((42, "qwen3pos76co"), (47, "qwen3pos76co"), (16, "qwen3pos75cl")):
        d = tmp_path / f"{allot}-{fam}"
        d.mkdir()
        assert "PASSED" in _floor(d, allot, fam) and "FINISH" not in _floor(d, allot, fam), (allot, fam)


def test_amendment_77_threads_fall_back_to_the_allotment_and_the_real_helper_reads_a_count(tmp_path: Path):
    """With no readable quota count the family's pools take ALLOT; the real quota_floor_threads prints a positive count no larger than
    the affinity set."""
    r, calls, summary = _run(tmp_path, FAKE_THREADS_EMPTY="1")
    assert "THREADS qwen3pos76co 12 (floor of the quota" in summary and all("OMP_NUM_THREADS=12 " in c for c in _runs(calls)), summary
    real = _cut(r"^quota_floor_threads\(\)\{.*?2>/dev/null; \}\n")
    assert "max(1, min(n, math.floor(q)))" in real                     # the quota rounded DOWN, never above it
    out = subprocess.run(["bash", "-c", real + "quota_floor_threads"], capture_output=True, text=True, timeout=30).stdout.strip()
    assert out.isdigit() and 1 <= int(out) <= len(os.sched_getaffinity(0)), out
