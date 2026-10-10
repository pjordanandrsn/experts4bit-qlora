"""SD1: the WHOLE box path of bench/sd1/sd1_run.sh, run under bash with the heavy steps stubbed, must reach SD1_SUCCESS with
no command-not-found and no unexpected exit (the maintainer's ask, after SC5 lost three proofs to harness defects a dry run
catches; tests/test_sc5_box_m_dryrun.py is the pattern).

The test runs the script's OWN text, relocated from /root/sd1 to a temporary directory and with /proc/meminfo faked.
External commands are stubbed on PATH:
- nvidia-smi;
- python (the torch probes, pip, the tripwire heredoc, and every script: the fetch, head fetch, bake, prompts, captures and
  reducer write the files the runner checks; each self-test passes);
- perl's alarm wrapper;
- git (clones and worktrees made, rev-parse answers each worktree's registered SHA);
- df, free, lscpu, stat and sha256sum (the head's registered size and digest).

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
LANE = REPO / "bench" / "sd1"
RUN = (LANE / "sd1_run.sh").read_text(encoding="utf-8")
HEAD_SHA = re.search(r"HEAD_SHA256=([0-9a-f]{64})", RUN).group(1)
HEAD_BYTES = re.search(r"HEAD_BYTES=(\d+)", RUN).group(1)
E4B_T = re.search(r"^E4B_T=([0-9a-f]{40})", RUN, re.M).group(1)
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
    "stat": "echo " + HEAD_BYTES,
    "sha256sum": r'''case "${1:-}" in -c) exit 0;; esac
for f; do case "$(basename "$f")" in model.safetensors) echo "''' + HEAD_SHA + r'''  $f";; *) echo "0000000000000000000000000000000000000000000000000000000000000000  $f";; esac; done''',
    "git": r'''dir=""; [ "${1:-}" = -C ] && { dir=$2; shift 2; }
case "${1:-}" in
  clone) for a; do last=$a; done; mkdir -p "$last/.git";;
  worktree) for a; do args="$args $a"; done; set -- $args; path=$5; mkdir -p "$path"
            case "$path" in *e4b_H) mkdir -p "$path/bench/common"; echo "# dryrun" > "$path/bench/common/hf_fetch_watchdog.py";; esac;;
  rev-parse) case "$dir" in *e4b_T) echo ''' + E4B_T + r''';; *gnf4_T) echo ''' + GNF4_T + r''';; *e4b_H) echo "$E4B_SHA";; *) echo 0;; esac;;
esac
exit 0''',
    "python": r'''echo "python $*" >> "$PWD/dryrun.log"
case "${1:-}" in
  -m) exit 0;;
  -c) case "$2" in
        *"print(torch.__version__"*) echo "2.8.0+cu128";;
        *hf_hub_download*) mkdir -p "$PWD/hf/head"; : > "$PWD/hf/head/model.safetensors"; echo "$PWD/hf/head/model.safetensors";;
      esac; exit 0;;
  -) cat > /dev/null; echo "e4b target dryrun" >> "$PWD/versions.txt"; exit 0;;
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
      *" --capture "*) printf '{"rows": []}\n' > "$out"; echo "SD1_CAPTURE {\"dryrun\": true}";;
    esac;;
  sd1_reduce.py) printf '{"verdict": "DRYRUN"}\n' > "$out"; echo "SD1_VERDICT {\"verdict\": \"DRYRUN\"}";;
esac
exit 0''',
}

needs_bash = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="the box script needs a POSIX bash with executable stubs")


def test_ci_runs_the_dry_run_rather_than_skipping_it():
    if not (os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"):
        pytest.skip("checked only in CI")
    assert sys.platform != "win32" and shutil.which("bash"), "CI found no bash: the SD1 dry run would be skipped"


def _run(tmp_path, run_text, nonce="a" * 64):
    w = tmp_path / "w"
    w.mkdir(exist_ok=True)
    (tmp_path / "meminfo").write_text("MemTotal:       129000000 kB\n")
    text = run_text.replace("/root/sd1", str(w)).replace("/proc/meminfo", str(tmp_path / "meminfo"))
    assert text != run_text
    for name in ("sd1_box.py", "sd1_eagle3.py", "sd1_reduce.py", "chat_prompts.json", "expect_w1.json", "staged.sha256"):
        shutil.copyfile(LANE / name, w / name)
    for src in (REPO / "bench" / "p109" / "p109_box.py", REPO / "bench" / "p39" / "k8_bake.py", REPO / "bench" / "p39" / "calib.json"):
        shutil.copyfile(src, w / src.name)
    (w / "sd1_run.sh").write_text(text)
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    for name, body in STUBS.items():
        p = b / name
        p.write_text("#!/bin/bash\n" + body + "\n")
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    env = {"PATH": f"{b}:{os.environ.get('PATH', '/usr/bin:/bin')}", "HOME": str(tmp_path), "LANG": "C",
           "SD1_RUN_NONCE": nonce, "SD1_RUN_ID": "sd1-dryrun", "SD1_DEADLINE_EPOCH": "4102444800", "SD1_INSTANCE_ID": "0",
           "E4B_SHA": HARNESS}
    out = subprocess.run(["bash", str(w / "sd1_run.sh")], capture_output=True, text=True, env=env, timeout=300, cwd=w)
    return w, out


@needs_bash
def test_the_box_path_reaches_success_with_every_heavy_step_stubbed(tmp_path):
    w, out = _run(tmp_path, RUN)
    log = out.stdout + out.stderr
    assert "command not found" not in log, log[-3000:]
    assert out.returncode == 0, log[-3000:]
    assert (w / ("SD1_SUCCESS." + "a" * 64)).exists() and (w / ("TP_DONE." + "a" * 64)).exists(), log[-3000:]
    summary = (w / "summary.txt").read_text()
    for line in ("KNOBS e4b_target=" + E4B_T, "FETCH Qwen/Qwen3-30B-A3B@", "HEAD RedHatAI/Qwen3-30B-A3B-speculator.eagle3@",
                 "BAKE OK", "capture R rc=0", "capture C-think rc=0", "capture C-nothink rc=0", "SD1_VERDICT"):
        assert line in summary, (line, summary[-2000:])
    assert "REHEARSAL" not in summary and not (w / "REFUSAL").exists()
    assert (w / "verdict.json").is_file()
    calls = (w / "dryrun.log").read_text()
    for step in ("--capture R ", "--capture C-think ", "--capture C-nothink ", "--expect-w1 ", "--chat-prompts ",
                 "--prompts-only ", "sd1_reduce.py --caps "):
        assert step in calls, (step, calls[-2000:])


@needs_bash
def test_the_dry_run_catches_a_misnamed_function(tmp_path):
    anchor = "can_run $NEED_BAKE bake || finish 40"
    assert RUN.count(anchor) == 1
    w, out = _run(tmp_path, RUN.replace(anchor, "can_runx $NEED_BAKE bake || finish 40"), nonce="b" * 64)
    assert out.returncode != 0 and not (w / ("SD1_SUCCESS." + "b" * 64)).exists()
    assert "can_runx: command not found" in out.stdout + out.stderr


@needs_bash
def test_the_dry_run_catches_a_head_digest_mismatch(tmp_path):
    w, out = _run(tmp_path, RUN.replace(HEAD_SHA, "f" * 64), nonce="c" * 64)
    assert out.returncode == 11 and "HEAD MISMATCH" in out.stdout + out.stderr
