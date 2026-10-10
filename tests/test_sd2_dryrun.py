"""SD2's proof and read: the WHOLE box path of bench/sd2/sd2_run.sh in each mode, run under bash with the heavy steps
stubbed, must reach SD2_SUCCESS with no command-not-found and no unexpected exit (SD1's pattern, tests/test_sd1_dryrun.py, after SC5 lost three
proofs to harness defects a dry run catches).

The test runs the script's OWN text, relocated from /root/sd2 to a temporary directory and with /proc/meminfo faked.
External commands are stubbed on PATH:
- nvidia-smi;
- python (the torch probes, pip, the tripwire and GPU-test heredocs, the GPU tests, and every script: the fetch, head
  fetch, bake, prompts, proof and reducer write the files the runner checks; each self-test passes);
- perl's alarm wrapper;
- git (clones and worktrees made, rev-parse answers each worktree's registered SHA);
- df, free, lscpu and sha256sum (the head's registered digest).

The head arrives as the Hugging Face cache lays it out: a snapshot SYMLINK to a sparse blob of the registered size, read
by the real `stat`. sd1-5090-1 died there (SD1's Amendment 1): `stat -c %s` without -L measured the link, 76 B.

Every shell function and every check in the runner runs as written. It runs nothing on a GPU and fetches nothing; what it
proves is that the control flow reaches success."""
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "sd2"
SD1 = REPO / "bench" / "sd1"
RUN = (LANE / "sd2_run.sh").read_text(encoding="utf-8")
HEAD_SHA = re.search(r"HEAD_SHA256=([0-9a-f]{64})", RUN).group(1)
HEAD_BYTES = re.search(r"HEAD_BYTES=(\d+)", RUN).group(1)
E4B_T = re.search(r"^E4B_T_PROVE=([0-9a-f]{40})", RUN, re.M).group(1)
_READ = re.search(r"^E4B_T_READ=(\S+)", RUN, re.M).group(1)
E4B_T_READ = _READ if re.fullmatch(r"[0-9a-f]{40}", _READ) else "2" * 40
RUN_READ = RUN if _READ == E4B_T_READ else RUN.replace("E4B_T_READ=" + _READ, "E4B_T_READ=" + E4B_T_READ)
GNF4_T = re.search(r"^GNF4_T=([0-9a-f]{40})", RUN, re.M).group(1)
HARNESS = "1" * 40

STUBS = {
    "nvidia-smi": r'''case "$*" in
  *"query-gpu=name,memory.total"*) echo "NVIDIA GeForce RTX 5090, 32607 MiB, 595.91.07, GPU-dryrun, 12.0";;
  *"query-gpu=name"*) echo "NVIDIA GeForce RTX 5090";;
  *) echo "nvidia-smi dryrun";;
esac''',
    "perl": r'''shift 2; exec "$@"''',
    "lscpu": r'''printf 'Vendor ID:  AuthenticAMD\nModel name:  AMD Ryzen 9 9950X (dryrun)\nCPU(s):  32\n' ''',
    "free": r'''printf '              total        used        free\nMem:            123           4         100\n' ''',
    "df": r'''case "$*" in *--output=avail*) printf 'Avail\n500G\n';; *) printf 'Filesystem Size Used Avail Use%% Mounted\noverlay 500G 1G 499G 1%% /\n';; esac''',
    "sha256sum": r'''case "${1:-}" in -c) exit 0;; esac
for f; do case "$(basename "$f")" in model.safetensors) echo "''' + HEAD_SHA + r'''  $f";; *) echo "0000000000000000000000000000000000000000000000000000000000000000  $f";; esac; done''',
    "git": r'''dir=""; [ "${1:-}" = -C ] && { dir=$2; shift 2; }
case "${1:-}" in
  clone) for a; do last=$a; done; mkdir -p "$last/.git";;
  worktree) for a; do args="$args $a"; done; set -- $args; path=$5; mkdir -p "$path"
            case "$path" in *e4b_H) mkdir -p "$path/bench/common"; echo "# dryrun" > "$path/bench/common/hf_fetch_watchdog.py";; esac;;
  rev-parse) case "$dir" in *e4b_T) echo "$DRYRUN_E4B_T";; *gnf4_T) echo ''' + GNF4_T + r''';; *e4b_H) echo "$E4B_SHA";; *) echo 0;; esac;;
esac
exit 0''',
    "python": r'''echo "python $*" >> "$PWD/dryrun.log"
case "${1:-}" in
  -m) [ "${2:-}" = pytest ] && exit "${DRYRUN_PYTEST_RC:-0}"; exit 0;;
  -c) case "$2" in
        *"print(torch.__version__"*) echo "2.8.0+cu128";;
        *hf_hub_download*) mkdir -p "$PWD/hf/blobs" "$PWD/hf/head"
          dd if=/dev/zero of="$PWD/hf/blobs/headblob" bs=1 count=0 seek=''' + HEAD_BYTES + r''' 2>/dev/null
          ln -sf ../blobs/headblob "$PWD/hf/head/model.safetensors"; echo "{}" > "$PWD/hf/head/config.json"
          echo "$PWD/hf/head/model.safetensors";;
      esac; exit 0;;
  -) cat > /dev/null
     if [ -n "${2:-}" ]; then printf '{"rc": %s, "passed": 1}\n' "$2" > "$PWD/gpu_tests.json"; echo "SD2_GPU_TESTS {\"rc\": $2}"
     else echo "e4b target dryrun" >> "$PWD/versions.txt"; fi; exit 0;;
esac
script=$(basename "${1:-x}"); shift
out=""; outdir="."; prev=""
for a; do case "$prev" in --out) out=$a;; --outdir) outdir=$a;; esac; prev=$a; done
case " $* " in *" --self-test "*) echo "${script%.py} self-test OK (dryrun)"; exit 0;; esac
case "$script" in
  hf_fetch_watchdog.py) mkdir -p "$PWD/hf/snap"; echo "$PWD/hf/snap";;
  k8_bake.py) mkdir -p "$K8_WORK"; : > "$K8_WORK/nf4.arena"; echo "BAKE OK dryrun";;
  p109_box.py) printf '{"rows": [], "prompts_sha256": "x"}\n' > "$out"; echo "P109_PROMPTS dryrun";;
  sd1_box.py) case " $* " in
      *" --chat-prompts "*) for w in C-think C-nothink; do printf '{"rows": []}\n' > "$outdir/prompts_$w.json"; done; echo "SD1_PROMPTS dryrun";;
    esac;;
  sd2_box.py) printf '{"dryrun": true}\n' > "$out"
    case " $* " in *" --prove "*) echo "SD2_PROVE done";; *" --read-v "*) echo "SD2_V done";; *" --read-e "*) echo "SD2_E done";;
                   *" --read-q "*) echo "SD2_Q done";; esac;;
  sd2_audit.py) [ -n "$out" ] && printf '{"ok": true}\n' > "$out"; echo "SD2_AUDIT dryrun"; exit "${DRYRUN_AUDIT_RC:-0}";;
  sd2_reduce.py) case " $* " in
      *" --gate-v "*) echo "SD2_GATE ${DRYRUN_GATE:-ALL}";;
      *" --read "*) printf '{"verdict": "DRYRUN"}\n' > "$out"; echo "SD2_READ_VERDICT {\"verdict\": \"DRYRUN\"}";;
      *) printf '{"verdict": "DRYRUN"}\n' > "$out"; echo "SD2_PROVE_VERDICT {\"verdict\": \"DRYRUN\"}";;
    esac;;
esac
exit 0''',
}

needs_bash = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="the box script needs a POSIX bash with executable stubs")


def test_ci_runs_the_dry_run_rather_than_skipping_it():
    if not (os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"):
        pytest.skip("checked only in CI")
    assert sys.platform != "win32" and shutil.which("bash"), "CI found no bash: the SD2 dry run would be skipped"


def _run(tmp_path, run_text, nonce="a" * 64, deadline="4102444800", extra_env=None):
    w = tmp_path / "w"
    w.mkdir(exist_ok=True)
    (tmp_path / "meminfo").write_text("MemTotal:       129000000 kB\n")
    text = run_text.replace("/root/sd2", str(w)).replace("/proc/meminfo", str(tmp_path / "meminfo"))
    assert text != run_text
    for name in ("sd2_box.py", "sd2_reduce.py", "sd2_audit.py", "audit_read.tsv", "staged.sha256"):
        shutil.copyfile(LANE / name, w / name)
    for name in ("sd1_box.py", "sd1_eagle3.py", "chat_prompts.json", "expect_w1.json"):
        shutil.copyfile(SD1 / name, w / name)
    for rel in ("p109/p109_box.py", "p115/p115_quality.py", "p110/p110_box.py", "p108/p108_box.py", "p97/p97_box.py",
                "p39/k8_bake.py", "p39/calib.json"):
        src = REPO / "bench" / rel
        shutil.copyfile(src, w / src.name)
    (w / "sd2_run.sh").write_text(text)
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    for name, body in STUBS.items():
        p = b / name
        p.write_text("#!/bin/bash\n" + body + "\n")
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    env = {"PATH": f"{b}:{os.environ.get('PATH', '/usr/bin:/bin')}", "HOME": str(tmp_path), "LANG": "C",
           "SD2_RUN_NONCE": nonce, "SD2_RUN_ID": "sd2-dryrun", "SD2_DEADLINE_EPOCH": deadline, "SD2_INSTANCE_ID": "0",
           "E4B_SHA": HARNESS, "DRYRUN_E4B_T": E4B_T_READ if (extra_env or {}).get("SD2_MODE") == "read" else E4B_T,
           **(extra_env or {})}
    out = subprocess.run(["bash", str(w / "sd2_run.sh")], capture_output=True, text=True, env=env, timeout=300, cwd=w)
    return w, out


@needs_bash
def test_the_box_path_reaches_success_with_every_heavy_step_stubbed(tmp_path):
    w, out = _run(tmp_path, RUN)
    log = out.stdout + out.stderr
    assert "command not found" not in log, log[-3000:]
    assert out.returncode == 0, log[-3000:]
    assert (w / ("SD2_SUCCESS." + "a" * 64)).exists() and (w / ("TP_DONE." + "a" * 64)).exists(), log[-3000:]
    summary = (w / "summary.txt").read_text()
    for line in ("KNOBS e4b_target=" + E4B_T, "FETCH Qwen/Qwen3-30B-A3B@", "HEAD RedHatAI/Qwen3-30B-A3B-speculator.eagle3@",
                 "BAKE OK", "SD2_GPU_TESTS", "prove rc=0", "SD2_PROVE done", "SD2_PROVE_VERDICT"):
        assert line in summary, (line, summary[-2000:])
    assert "REHEARSAL" not in summary and not (w / "REFUSAL").exists()
    assert (w / "verdict_prove.json").is_file() and (w / "gpu_tests.json").is_file()
    calls = (w / "dryrun.log").read_text()
    tests_log = w / "src" / "e4b_T" / "dryrun.log"           # the GPU tests run from the target's checkout
    calls += tests_log.read_text() if tests_log.exists() else ""
    for step in ("sd2_box.py --prove ", "--prompts-c ", "--head-dir ", "--chat-prompts ", "--prompts-only ",
                 "sd2_reduce.py --prove ", "-m pytest "):
        assert step in calls, (step, calls[-2000:])


@needs_bash
def test_the_dry_run_catches_a_misnamed_function(tmp_path):
    anchor = "can_run $NEED_BAKE bake || finish 40"
    assert RUN.count(anchor) == 1
    w, out = _run(tmp_path, RUN.replace(anchor, "can_runx $NEED_BAKE bake || finish 40"), nonce="b" * 64)
    assert out.returncode != 0 and not (w / ("SD2_SUCCESS." + "b" * 64)).exists()
    assert "can_runx: command not found" in out.stdout + out.stderr


@needs_bash
def test_the_dry_run_catches_a_head_digest_mismatch(tmp_path):
    w, out = _run(tmp_path, RUN.replace(HEAD_SHA, "f" * 64), nonce="c" * 64)
    assert out.returncode == 11 and "HEAD MISMATCH" in out.stdout + out.stderr


@needs_bash
def test_the_dry_run_catches_a_size_check_that_does_not_follow_the_cache_symlink(tmp_path):
    """sd1-5090-1's defect (SD1's Amendment 1), kept as a mutant: stat without -L reads the snapshot symlink."""
    anchor = '[ "$(stat -L -c %s "$HEAD")" = "$HEAD_BYTES" ]'
    assert RUN.count(anchor) == 1
    w, out = _run(tmp_path, RUN.replace(anchor, '[ "$(stat -c %s "$HEAD")" = "$HEAD_BYTES" ]'), nonce="d" * 64)
    assert out.returncode == 11 and "HEAD MISMATCH" in out.stdout + out.stderr


@needs_bash
def test_a_deadline_shorter_than_the_registered_guard_is_refused_at_once(tmp_path):
    """Amendment 1b: sd2-prove-1 launched under a 0.75 h guard and stopped before the fetch. A launch whose deadline
    leaves less than the registered guard now refuses before anything is installed (rc 17)."""
    import time
    w, out = _run(tmp_path, RUN, nonce="e" * 64, deadline=str(int(time.time()) + 2700))
    assert out.returncode == 17 and "REFUSED: the deadline leaves" in out.stdout + out.stderr
    calls = (w / "dryrun.log").read_text() if (w / "dryrun.log").exists() else ""
    assert "pip install" not in calls and "hf_fetch_watchdog" not in calls


@needs_bash
def test_failing_gpu_tests_stop_the_lane_before_the_fetch(tmp_path):
    """Amendment 1b: the GPU tests need no checkpoint, so they run first; a failure stops the lane (rc 24) with their
    record kept and no 62 GB download."""
    w, out = _run(tmp_path, RUN, nonce="f" * 64, extra_env={"DRYRUN_PYTEST_RC": "1"})
    assert out.returncode == 24 and "GPU TESTS FAILED" in out.stdout + out.stderr
    assert (w / "gpu_tests.json").is_file()
    assert "--repo Qwen/Qwen3-30B-A3B" not in (w / "dryrun.log").read_text()      # no checkpoint fetch


READ = {"SD2_MODE": "read"}
ARMS = ("OFF-a", "ON1-a", "ON2-a", "ON2-b", "ON1-b", "OFF-b")


def _calls(w):
    calls = (w / "dryrun.log").read_text()
    tests_log = w / "src" / "e4b_T" / "dryrun.log"
    return calls + (tests_log.read_text() if tests_log.exists() else "")


@needs_bash
def test_the_read_path_reaches_success_with_every_heavy_step_stubbed(tmp_path):
    """Amendment 3: the read's audit, stage V, the gate, E's six arms in their registered order, Q, and the verdict."""
    w, out = _run(tmp_path, RUN_READ, nonce="1" * 64, extra_env=READ)
    log = out.stdout + out.stderr
    assert "command not found" not in log, log[-3000:]
    assert out.returncode == 0, log[-3000:]
    assert (w / ("SD2_SUCCESS." + "1" * 64)).exists(), log[-3000:]
    summary = (w / "summary.txt").read_text()
    for line in ("KNOBS e4b_target=" + E4B_T_READ, "SD2_AUDIT", "v rc=0", "SD2_V done", "SD2_GATE ALL", "q rc=0",
                 "SD2_READ_VERDICT"):
        assert line in summary, (line, summary[-2000:])
    assert "prove rc=" not in summary
    for arm in ARMS:
        assert f"e_{arm} rc=0" in summary and (w / f"e_{arm}.json").is_file(), arm
    calls = _calls(w)
    order = [calls.index(f"--read-e {arm} ") for arm in ARMS]
    assert order == sorted(order), "E's arms ran out of the registered palindrome"
    for step in ("sd2_audit.py --repo ", "sd2_box.py --read-v ", "--prompts-cn ", "sd2_box.py --read-q --windows 48 --prompt 512 --cont 128 ",
                 "sd2_reduce.py --gate-v ", "sd2_reduce.py --read ", "--expect-e4b " + E4B_T_READ, "-m pytest "):
        assert step in calls, (step, calls[-2000:])
    assert calls.index("sd2_audit.py --repo ") < calls.index("hf_fetch_watchdog"), "the audit runs before the fetch"


@needs_bash
def test_a_void_gate_runs_neither_e_nor_q(tmp_path):
    """V0 fails closed: the gate's VOID skips E and Q, and the reducer still writes the (VOID) verdict."""
    w, out = _run(tmp_path, RUN_READ, nonce="2" * 64, extra_env={**READ, "DRYRUN_GATE": "VOID"})
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]
    calls = _calls(w)
    assert "--read-e " not in calls and "--read-q " not in calls and "sd2_reduce.py --read " in calls


@needs_bash
def test_the_stop_rule_runs_e_and_skips_q(tmp_path):
    w, out = _run(tmp_path, RUN_READ, nonce="3" * 64, extra_env={**READ, "DRYRUN_GATE": "E_ONLY"})
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]
    calls = _calls(w)
    assert all(f"--read-e {arm} " in calls for arm in ARMS) and "--read-q " not in calls
    assert "stage Q skipped: the stop rule" in (w / "summary.txt").read_text()


@needs_bash
def test_a_failed_audit_stops_the_read_before_the_fetch(tmp_path):
    w, out = _run(tmp_path, RUN_READ, nonce="4" * 64, extra_env={**READ, "DRYRUN_AUDIT_RC": "31"})
    assert out.returncode == 31 and "AUDIT FAIL" in out.stdout + out.stderr
    assert "hf_fetch_watchdog" not in (w / "dryrun.log").read_text()


@needs_bash
def test_the_read_refuses_a_deadline_shorter_than_its_own_guard(tmp_path):
    """The proof's guard would admit this deadline; the read's 3.5 h does not (rc 17, before any install)."""
    import time
    w, out = _run(tmp_path, RUN_READ, nonce="5" * 64, deadline=str(int(time.time()) + 6000), extra_env=READ)
    assert out.returncode == 17 and "REFUSED: the deadline leaves" in out.stdout + out.stderr


@needs_bash
def test_the_read_refuses_an_unpinned_target(tmp_path):
    w, out = _run(tmp_path, RUN.replace("E4B_T_READ=" + _READ, "E4B_T_READ=__READ_TARGET__"), nonce="6" * 64,
                  extra_env=READ)
    assert out.returncode == 78 and "the read target is not a sha" in out.stdout + out.stderr


@needs_bash
def test_the_dry_run_catches_a_misnamed_function_on_the_read_path(tmp_path):
    anchor = "stage q $? $W/q.json"
    assert RUN_READ.count(anchor) == 1
    w, out = _run(tmp_path, RUN_READ.replace(anchor, "stagex q $? $W/q.json"), nonce="7" * 64, extra_env=READ)
    assert out.returncode != 0 and not (w / ("SD2_SUCCESS." + "7" * 64)).exists()
    assert "stagex: command not found" in out.stdout + out.stderr
