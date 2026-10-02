"""tc1_drive.sh's incremental receipt fetch (e4b#835, box B).

Box B of lane TC2 (`tc1-5090-22`) ran 4 h 40 min, finished about twenty arms, and was then stopped by its host; the
driver fetched receipts only after TP_DONE or at the deadline, so every finished arm died with the instance's disk.
The driver now copies the box's tree into `$RUN_DIR/tc1.partial` each time the box's summary line changes, and the
final fetch falls back to that copy when it cannot reach the box. These tests run the driver's own code -- the
function and the final-fetch block, cut from the script -- against a fake `rsync` on PATH, so no box is needed.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DRIVE = (REPO / "bench" / "tc1" / "tc1_drive.sh").read_text()

FAKE_RSYNC = r"""#!/usr/bin/env bash
# a stand-in for rsync: records its argv, copies $FAKE_BOX/ to the last argument unless FAKE_RC is non-zero
printf '%s\n' "$@" > "$FAKE_ARGV"
[ "${FAKE_RC:-0}" = 0 ] || exit "$FAKE_RC"
dest="${@: -1}"
mkdir -p "$dest" && cp -R "$FAKE_BOX/." "$dest"
"""


def _cut(pattern: str) -> str:
    m = re.search(pattern, DRIVE, re.DOTALL | re.MULTILINE)
    assert m, f"pattern no longer matches tc1_drive.sh: {pattern}"
    return m.group(0)


EXCLUDES = _cut(r"^TC1_RSYNC_EXCLUDES=\(.*?\)\n")
FUNC = _cut(r"^tc1_partial_fetch\(\) \{.*?^\}\n")
FINAL = _cut(r"^if ! rsync -az .*?^rm -rf \"\$RUN_DIR/tc1\.partial\"\n")


def _env(tmp: Path, rc: int = 0) -> dict:
    bindir = tmp / "bin"
    bindir.mkdir(exist_ok=True)
    fake = bindir / "rsync"
    fake.write_text(FAKE_RSYNC)
    fake.chmod(0o755)
    return {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "FAKE_BOX": str(tmp / "box"),
            "FAKE_ARGV": str(tmp / "argv"), "FAKE_RC": str(rc)}


def _bash(script: str, tmp: Path, env: dict) -> subprocess.CompletedProcess:
    prelude = ('say(){ echo "SAY $*"; }\nHOST=box.example; PORT=2222; W=/root/tc1; E4B_RENT_SSH_OPTS="-o UserKnownHostsFile=/x"\n'
               f'RUN_DIR="{tmp}/run"; PARTIAL_AT=2026-10-02T11:30:00Z\n')
    return subprocess.run(["bash", "-c", prelude + EXCLUDES + FUNC + script], cwd=tmp, env=env, check=False,
                          capture_output=True, text=True, timeout=60)


def _box(tmp: Path) -> None:
    (tmp / "box" / "logs").mkdir(parents=True)
    (tmp / "box" / "mixtral_e4b_fused_attn4_m.json").write_text('{"status": "ok"}')
    (tmp / "box" / "logs" / "run_mixtral_e4b_fused_attn4_m.log").write_text("step 20/20\n")


def test_partial_fetch_copies_the_box_and_passes_the_excludes_unexpanded(tmp_path: Path):
    _box(tmp_path)
    (tmp_path / "venv-should-not-glob").write_text("")   # an unquoted venv* would expand to this name
    r = _bash('tc1_partial_fetch "$RUN_DIR/tc1.partial" && echo OK', tmp_path, _env(tmp_path))
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr
    assert (tmp_path / "run" / "tc1.partial" / "mixtral_e4b_fused_attn4_m.json").exists()
    argv = (tmp_path / "argv").read_text().splitlines()
    assert "venv*" in argv and "venv-should-not-glob" not in argv
    assert "--timeout=120" in argv and argv[-2] == "root@box.example:/root/tc1/"
    assert any("-o UserKnownHostsFile=/x" in a for a in argv), "the launcher's ssh options must reach rsync's -e"


def test_a_failed_partial_fetch_returns_nonzero_and_never_ends_the_poll(tmp_path: Path):
    _box(tmp_path)
    r = _bash('tc1_partial_fetch "$RUN_DIR/tc1.partial"; echo "rc=$?"; echo STILL-POLLING', tmp_path, _env(tmp_path, rc=255))
    assert "rc=255" in r.stdout and "STILL-POLLING" in r.stdout, (r.stdout, r.stderr)


def test_the_final_fetch_keeps_the_partial_copy_when_the_box_is_gone(tmp_path: Path):
    _box(tmp_path)
    partial = tmp_path / "run" / "tc1.partial"
    partial.mkdir(parents=True)
    (partial / "qwen3_5_unsloth_ckpt_unsloth_m.json").write_text('{"status": "ok", "from": "partial"}')
    (partial / "summary.txt").write_text("partial summary\n")
    final = tmp_path / "run" / "tc1"
    final.mkdir()
    (final / "summary.txt").write_text("the final rsync's newer summary\n")   # a partially successful final rsync
    r = _bash('mkdir -p "$RUN_DIR/tc1"\n' + FINAL + 'echo FINISHED', tmp_path, _env(tmp_path, rc=255))
    assert r.returncode == 22 and "FINISHED" not in r.stdout, (r.stdout, r.stderr)
    assert "kept the incremental copy from 2026-10-02T11:30:00Z" in r.stdout
    assert (final / "qwen3_5_unsloth_ckpt_unsloth_m.json").read_text().endswith('"partial"}')
    assert (final / "summary.txt").read_text() == "the final rsync's newer summary\n", "the fallback must never overwrite"
    assert not partial.exists()


def test_the_final_fetch_with_nothing_to_keep_says_so(tmp_path: Path):
    _box(tmp_path)
    r = _bash('mkdir -p "$RUN_DIR/tc1"\n' + FINAL, tmp_path, _env(tmp_path, rc=255))
    assert r.returncode == 22 and "no incremental copy to keep" in r.stdout, (r.stdout, r.stderr)


def test_a_successful_final_fetch_removes_the_partial_copy(tmp_path: Path):
    _box(tmp_path)
    partial = tmp_path / "run" / "tc1.partial"
    partial.mkdir(parents=True)
    (partial / "stale.json").write_text("{}")
    r = _bash('mkdir -p "$RUN_DIR/tc1"\n' + FINAL + 'echo FINISHED', tmp_path, _env(tmp_path))
    assert r.returncode == 0 and "FINISHED" in r.stdout, (r.stdout, r.stderr)
    assert (tmp_path / "run" / "tc1" / "mixtral_e4b_fused_attn4_m.json").exists() and not partial.exists()


def test_the_poll_loop_calls_the_partial_fetch_on_a_summary_change():
    loop = _cut(r'^  if \[ "\$line" != "\$LAST" \]; then\n.*?^  fi\n')
    assert 'tc1_partial_fetch "$RUN_DIR/tc1.partial"' in loop and "PARTIAL_AT=$(date -u" in loop
    assert "exit" not in loop, "a failed partial fetch must never end the poll"
