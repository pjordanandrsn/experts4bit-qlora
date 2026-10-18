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

import pytest

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


def _bash(script: str, tmp: Path, env: dict, *, shell: str = "bash") -> subprocess.CompletedProcess:
    prelude = ('say(){ echo "SAY $*"; }\nHOST=box.example; PORT=2222; W=/root/tc1; E4B_RENT_SSH_OPTS="-o UserKnownHostsFile=/x"\n'
               f'RUN_DIR="{tmp}/run"; PARTIAL_AT=2026-10-02T11:30:00Z\n')
    return subprocess.run([shell, "-c", prelude + EXCLUDES + FUNC + script], cwd=tmp, env=env, check=False,
                          capture_output=True, text=True, timeout=60)


def test_partial_fetch_bash32_empty_and_populated_optional_excludes(tmp_path: Path):
    _box(tmp_path)
    for values, wanted in (("()", []), ("(--exclude 'receipts/capture-trace.json')", ["--exclude", "receipts/capture-trace.json"])):
        r = _bash(f'set -uo pipefail; TC1_PARTIAL_EXCLUDES={values}; tc1_partial_fetch "$RUN_DIR/tc1.partial" && echo OK',
                  tmp_path, _env(tmp_path), shell="/bin/bash")
        assert r.returncode == 0 and "OK" in r.stdout, r.stderr
        argv = (tmp_path / "argv").read_text().splitlines()
        assert all(item in argv for item in wanted)
        assert ("receipts/capture-trace.json" in argv) == bool(wanted)


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


def test_hp1_bulk_files_stay_on_the_box_and_its_small_outputs_come_back(tmp_path: Path):
    """HP1's base exports (~17 GB each) and canonical adapters (~2.6 GB each), and the shakedown cut, are excluded from every fetch
    by the driver's own TC1_RSYNC_EXCLUDES, run through the REAL rsync; the evaluator JSONs, select.json and the receipts come back."""
    import shutil
    rsync = shutil.which("rsync")
    if rsync is None:
        pytest.skip("no rsync on PATH")
    box = tmp_path / "box"
    keep = ["qwen3hp1_e4b_hp1_k00_s3407.json", "summary.txt", "hp1/eval_K.json", "hp1/select.json", "hp1/eval_early.json",
            "hp1/eval_shake.json"]
    drop = ["hp1/base_e4b_qwen3hp1/layer000.safetensors", "hp1/base_e4b_qwen3hp1/index.json",
            "hp1/shake/base_e4b_qwen3hp1shake/global.safetensors", "hp1/canonical_qwen3hp1_hp1_k00_s3407.safetensors",
            "hp1/shake/canonical_qwen3hp1shake_hp1shake_e4b.safetensors", "hp1/shake-cache/models--local--x/config.json"]
    for rel in keep + drop:
        (box / rel).parent.mkdir(parents=True, exist_ok=True)
        (box / rel).write_text("x")
    for i, rel in enumerate(drop):                      # distinct bytes, so a wrong sha256 cannot pass by coincidence
        (box / rel).write_bytes(bytes([i]) * (100 + i))
    (box / "logs").mkdir()
    run_sh = (REPO / "bench" / "tc1" / "tc1_run.sh").read_text()
    manifest_fn = re.search(r"^hp1_bulk_manifest\(\)\{.*?^  tail -1 .*?; \}\n", run_sh, re.DOTALL | re.MULTILINE).group(0)
    out = tmp_path / "out"
    script = ("set -uo pipefail\nsay(){ echo \"$*\"; }\n" + manifest_fn + f'hp1_bulk_manifest "{box}"\n' + EXCLUDES
              + f'rsync -a "${{TC1_RSYNC_EXCLUDES[@]}}" "{box}/" "{out}/"\n')
    r = subprocess.run([shutil.which("bash") or "bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    for rel in keep:
        assert (out / rel).exists(), f"{rel} must come back"
    for rel in drop:
        assert not (out / rel).exists(), f"{rel} must stay on the box"
    # the bulk files keep their identity: the manifest comes back and lists each excluded file with its true sha256 and size
    import hashlib
    import json
    man = json.loads((out / "hp1" / "bulk_manifest.json").read_text())   # fetched: the manifest itself is not excluded
    rows = {row["path"]: row for row in man["files"]}
    left_on_box = sorted(str(f.relative_to(box)) for f in box.rglob("*") if f.is_file() and not (out / f.relative_to(box)).exists())
    assert sorted(rows) == left_on_box == sorted(drop), "the manifest must list exactly what the excludes left on the box"
    for rel in drop:
        data = (box / rel).read_bytes()
        assert rows[rel]["sha256"] == hashlib.sha256(data).hexdigest() and rows[rel]["bytes"] == len(data)
