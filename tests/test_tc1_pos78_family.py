"""tc1_run.sh's amendment 78 family (qwen3pos78fit): the card band, the pre-check, the pad-buckets probe and the token check, run from the
script's own functions.

Before its one arm (pb1cl256, 60 steps), the family refuses at cents on: a card whose memory.total is not a 24 GiB card's (rc 18); a failing
or CUDA-skipping run of grouped-nf4-gemm's kernel/test_compact_delta.py and test_compact_buckets.py on the box's GPU; an installed nf4_qlora
that does not read NF4_QLORA_PAD_BUCKETS=1 as bucketed; and tokens that are not tc1-5090-155's. The pre-check clone sits OUTSIDE the
fetched work dir and is removed after. These tests cut the functions from tc1_run.sh and run them against a fake python, git and
nvidia-smi on PATH, so no GPU, network or model is needed.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
#: bench/common rule 6: amendment 78 recorded its verdict (FIT, its read PR), so this closed lane's tests read their files at its frozen
#: launch commit (experts4bit-qlora#2093's merge, the commit both boxes ran), never the live tree. A later change to a shared file
#: (tc1_run.sh, TC1-PREREG.md) then neither re-pins nor fails it.
TC1_78_FROZEN = "576d99e9fef92c406642e995d9945fc5e4cf3292"


def _at_frozen(rel: str) -> str:
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{TC1_78_FROZEN}:{rel}"], capture_output=True, text=True)
    if r.returncode != 0:
        # CI checks out full history (fetch-depth: 0), so a miss there is a failure, never a skip
        if os.environ.get("CI") == "true":
            raise AssertionError(f"git show {TC1_78_FROZEN}:{rel} failed: {r.stderr.strip()}")
        pytest.skip(f"the frozen commit {TC1_78_FROZEN[:8]} is not in this clone (shallow?)", allow_module_level=True)
    return r.stdout


RUN = _at_frozen("bench/tc1/tc1_run.sh")
GNF4 = "efdda943d5a310343ed7fef2e23d1cbf2df27e51"
TOKENS, TRAIN = "bfc742f67e376ecfee191e2c6fd34170ab2ed310260a8c89bc79c46e3e7521ac", "604f314033450d4d4b5953bb13b3651d919d9b05c46a317e6267a3c2f6c007d7"
ENV = "[NF4_QLORA_COMPACT_DELTA=1 NF4_QLORA_PAD_BUCKETS=1 E4B_CHUNKED_LM_LOSS=256 TC1_STEP_MEM=1]"


def _cut(pattern: str) -> str:
    m = re.search(pattern, RUN, re.DOTALL | re.MULTILINE)
    assert m, f"pattern no longer matches tc1_run.sh: {pattern}"
    return m.group(0)


FUNCS = (_cut(r"^POS78_GNF4=.*?\n") + _cut(r"^POS78_TOKENS_SHA=.*?\n") + _cut(r"^POS78_TRAIN_SHA=.*?\n")
         + _cut(r"^tc1_pos78_family\(\)\{.*?free_family \$FAM \$\{MID//\\//--\}; \}\n"))
BAND = _cut(r"^POS78_MIB_MIN=.*?\n") + _cut(r"^case \" \$FAMILIES \" in \*\" qwen3pos78fit \"\*\)\n.*?^esac\n")

FAKE_PY = r"""#!/usr/bin/env bash
echo "PY $*" >> "$CALLS"
case "$1 ${2:-}" in
  "-m pip") case "$*" in *" freeze"*) echo "torch==2.12.1+cu130"; [ -f "$CALLS.dirty" ] && echo "pytest==9.0.0";; esac; exit 0;;
  "-m pytest") echo "PYTEST_PYTHONPATH=${PYTHONPATH:-} CWD=$PWD" >> "$CALLS"; [ -n "${FAKE_PRECHECK_DIRTIES:-}" ] && touch "$CALLS.dirty"
               printf '%s\n' "${FAKE_PYTEST_OUT:-201 passed, 60 skipped}"; exit "${FAKE_PYTEST_RC:-0}";;
esac
if [ "$1" = -c ]; then echo "PROBE_CWD=$PWD PROBE_PYTHONPATH=${PYTHONPATH:-} PAD_BUCKETS=${NF4_QLORA_PAD_BUCKETS:-}" >> "$CALLS"
  printf '%s\n' "${FAKE_PROBE_OUT-1 True True True}"; exit 0; fi
cat > /dev/null; exit 0
"""
FAKE_GIT = r"""#!/usr/bin/env bash
echo "GIT $*" >> "$CALLS"
if [ "$1" = clone ]; then d="${@: -1}"; mkdir -p "$d/kernel"; touch "$d/kernel/test_compact_delta.py" "$d/kernel/test_compact_buckets.py"; fi
exit 0
"""
STUBS = r"""
say(){ echo "SAY $*"; }
stubw(){ echo "STUB $2/$3 $5 $6" >> "$CALLS"; }
tc1_prepare(){ echo "PREPARE" >> "$CALLS"; TOK=tok; TS=${FAKE_TS:-__TOKENS__}; return 0; }
train_sha(){ echo "${FAKE_TRAIN:-__TRAIN__}"; }
can_run(){ return 0; }
arm(){ echo "ARM $3 [$TC1_ARM_EXTRA_ENV]" >> "$CALLS"; }
free_family(){ echo "FREE $1" >> "$CALLS"; }
""".replace("__TOKENS__", TOKENS).replace("__TRAIN__", TRAIN)


def _bins(tmp: Path) -> Path:
    bindir = tmp / "bin"
    bindir.mkdir(exist_ok=True)
    for name, body in (("py", FAKE_PY), ("git", FAKE_GIT)):
        (bindir / name).write_text(body)
        (bindir / name).chmod(0o755)
    return bindir


def _run(tmp: Path, gnf4: str = GNF4, steps: int = 60, **env) -> tuple[subprocess.CompletedProcess, list[str], str]:
    bindir = _bins(tmp)
    w = tmp / "tc1"                                    # the fetched work dir; the pre-check clone must land beside it, never in it
    (w / "logs").mkdir(parents=True, exist_ok=True)
    calls = tmp / "calls.txt"
    script = (f'W="{w}"; STEPS={steps}; EVAL_EVERY=20; EVAL_N=8; GNF4_SHA={gnf4}; T212_OK=1; T212_REASON=""\n'
              f'PY_UNS="{bindir}/py"; MATCHED_SEED=0\n' + STUBS + FUNCS + "tc1_pos78_family qwen3pos78fit M rev 1 1\n")
    r = subprocess.run(["bash", "-c", script], cwd=w, check=False, capture_output=True, text=True, timeout=60,
                       env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "CALLS": str(calls), **env})
    summary = (w / "summary.txt").read_text() if (w / "summary.txt").exists() else ""
    return r, calls.read_text().splitlines() if calls.exists() else [], summary


def _arms(calls):
    return [c for c in calls if c.startswith("ARM ")]


def test_a_clean_run_checks_the_card_code_and_tokens_then_trains_the_one_arm(tmp_path: Path):
    r, calls, summary = _run(tmp_path)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "PRECHECK qwen3pos78fit OK 201 passed, 60 skipped" in summary and "PADBUCKETS_PROBE qwen3pos78fit 1 True True True" in summary, summary
    assert "TOKENS MATCH tc1-5090-155" in summary and "QWEN3POS78FIT DONE" in summary, summary
    assert _arms(calls) == [f"ARM fused_attn4_m_fd_pb1cl256 {ENV}"], calls
    pc = tmp_path / "gnf4-pos78"                       # beside the work dir, not in it
    assert any(c == f"GIT -C {pc} checkout -q {GNF4}" for c in calls), calls
    assert not pc.exists() and not (tmp_path / "gnf4-pos78-site").exists()      # removed after the pre-check
    assert not any("gnf4" in p.name for p in (tmp_path / "tc1").iterdir())
    pytest_call = next(c for c in calls if "-m pytest" in c)
    assert "test_compact_delta.py test_compact_buckets.py" in pytest_call, pytest_call
    assert calls.index(pytest_call) < calls.index("PREPARE")                     # the pre-check and the probe run before the fetch
    probe = next(c for c in calls if c.startswith("PROBE_CWD="))
    # the installed module, as the arm imports it: the arm's environment (never the pre-check's --target dir) and the work dir, not the clone
    assert probe == f"PROBE_CWD={tmp_path / 'tc1'} PROBE_PYTHONPATH={os.environ.get('PYTHONPATH', '')} PAD_BUCKETS=1", probe
    assert "gnf4-pos78" not in probe
    assert calls.index(probe) < calls.index("PREPARE")


def test_a_failing_pre_check_refuses_before_the_fetch(tmp_path: Path):
    r, calls, summary = _run(tmp_path, FAKE_PYTEST_RC="1", FAKE_PYTEST_OUT="2 failed, 199 passed, 60 skipped")
    assert "QWEN3POS78FIT REFUSED pre-check FAILED" in summary, summary
    assert "PREPARE" not in calls and not _arms(calls) and not any(c.startswith("PROBE_CWD=") for c in calls)
    assert [c for c in calls if c.startswith("STUB ")] == [c for c in calls if c.startswith("STUB e4b/fused_attn4_m_fd_pb1cl256 not_run pre-check FAILED")]
    assert len([c for c in calls if c.startswith("STUB ")]) == 1


def test_a_skipped_cuda_case_refuses_but_a_failing_tests_printed_source_does_not(tmp_path: Path):
    for reason in ("max_memory_allocated is a CUDA allocator statistic", "device memory is a CUDA statistic"):
        d = tmp_path / reason.split()[0]
        d.mkdir()
        r, calls, summary = _run(d, FAKE_PYTEST_OUT=f"SKIPPED [2] test_compact_buckets.py:75: {reason}\n199 passed, 62 skipped")
        assert "QWEN3POS78FIT REFUSED pre-check incomplete: a CUDA case skipped" in summary, summary
        assert not _arms(calls)
    d = tmp_path / "source"
    d.mkdir()
    out = '    @pytest.mark.skipif(not CUDA, reason="max_memory_allocated is a CUDA allocator statistic")\n201 passed, 60 skipped'
    r, calls, summary = _run(d, FAKE_PYTEST_OUT=out)
    assert "PRECHECK qwen3pos78fit OK" in summary and _arms(calls), summary


def test_a_probe_that_does_not_read_buckets_refuses_before_the_fetch(tmp_path: Path):
    for i, out in enumerate(("auto True True False", "", "Traceback (most recent call last):")):
        d = tmp_path / f"p{i}"
        d.mkdir()
        r, calls, summary = _run(d, FAKE_PROBE_OUT=out)
        assert "QWEN3POS78FIT REFUSED the pad-buckets probe read" in summary, summary
        assert "PREPARE" not in calls and not _arms(calls)


def test_tokens_that_are_not_tc1_5090_155s_refuse_the_arm(tmp_path: Path):
    for i, kw in enumerate(({"FAKE_TS": "0" * 64}, {"FAKE_TRAIN": "1" * 64})):
        d = tmp_path / f"t{i}"
        d.mkdir()
        r, calls, summary = _run(d, **kw)
        assert "QWEN3POS78FIT REFUSED the tokens sha" in summary, summary
        assert "PREPARE" in calls and not _arms(calls) and "FREE qwen3pos78fit" in calls


def test_the_wrong_pin_window_or_a_changed_venv_refuses_without_an_arm(tmp_path: Path):
    for i, kw in enumerate(({"gnf4": "7bcddcca157493188ed5e5c4bb4bcb1af7a5ab07"}, {"steps": 20})):
        d = tmp_path / f"w{i}"
        d.mkdir()
        r, calls, summary = _run(d, **kw)
        assert "QWEN3POS78FIT REFUSED amendment 78" in summary, summary
        assert not any(c.startswith("GIT clone") for c in calls) and not _arms(calls)
    d = tmp_path / "dirty"
    d.mkdir()
    r, calls, summary = _run(d, FAKE_PRECHECK_DIRTIES="1")
    assert "QWEN3POS78FIT REFUSED the pre-check changed venv-unsloth's package set" in summary, summary
    assert not _arms(calls)


def _band(tmp: Path, mib: str, families: str = "qwen3pos78fit") -> tuple[int, str]:
    bindir = tmp / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    (bindir / "nvidia-smi").write_text(f'#!/bin/sh\n[ -n "{mib}" ] && echo "{mib}"\nexit 0\n')
    (bindir / "nvidia-smi").chmod(0o755)
    script = f'FAMILIES="{families}"\nsay(){{ :; }}\nfinish(){{ exit $1; }}\n' + BAND + "echo PASSED\n"
    r = subprocess.run(["bash", "-c", script], cwd=tmp, check=False, capture_output=True, text=True, timeout=30,
                       env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"})
    summary = (tmp / "summary.txt").read_text() if (tmp / "summary.txt").exists() else ""
    return r.returncode, summary + r.stdout


def test_the_card_band_passes_a_24_gib_card_and_refuses_anything_else_as_a_host_floor(tmp_path: Path):
    for i, mib in enumerate(("24564", "23500", "24600")):
        rc, out = _band(tmp_path / f"ok{i}", mib)
        assert rc == 0 and "PASSED" in out and f"GPU_MEMORY_TOTAL {mib} MiB" in out, (mib, out)
    for i, mib in enumerate(("49140", "23499", "24601", "", "N/A")):
        rc, out = _band(tmp_path / f"no{i}", mib)
        assert rc == 18 and "PASSED" not in out and "BOX_REFUSED gpu_memory_total=" in out, (mib, rc, out)
    rc, out = _band(tmp_path / "other", "49140", families="qwen3pos76co")      # other families are untouched
    assert rc == 0 and "PASSED" in out and "GPU_MEMORY_TOTAL" not in out
