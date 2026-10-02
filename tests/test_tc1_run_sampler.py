"""tc1_run.sh's per-arm GPU sampler (e4b#835, TC1 amendment 7).

tc1-5090-33's two e4b-shipped draws differed by 6.1 % on every step at lower power, and the box recorded no clock, temperature or host
load to say why. The sampler now writes a second file per arm (`gpuclk_<arm>.txt`) beside the unchanged `vram_<arm>.txt`. These tests run
the script's own functions, cut from it, against a fake `nvidia-smi` on PATH, so no GPU is needed.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = (REPO / "bench" / "tc1" / "tc1_run.sh").read_text()

FAKE_SMI = r"""#!/usr/bin/env bash
case "$*" in
  *clocks.sm*) [ -n "$FAKE_CLK_FAIL" ] && { echo 'Field "clocks_event_reasons.active" is not a valid field to query.' >&2; exit 2; }
               echo "2407, 14001, 61, 0x0000000000000004";;
  *memory.used*) echo "24581, 97, 288.4";;
esac
"""


def _cut(pattern: str) -> str:
    m = re.search(pattern, RUN, re.DOTALL | re.MULTILINE)
    assert m, f"pattern no longer matches tc1_run.sh: {pattern}"
    return m.group(0)


FUNCS = _cut(r"^vram_start\(\)\{.*?echo \$!; \}\n") + _cut(r"^vram_stop\(\)\{.*?\}\n")


def _run(tmp: Path, **env) -> subprocess.CompletedProcess:
    bindir = tmp / "bin"
    bindir.mkdir(exist_ok=True)
    fake = bindir / "nvidia-smi"
    fake.write_text(FAKE_SMI)
    fake.chmod(0o755)
    script = f'W="{tmp}"\n' + FUNCS + 'sp=$(vram_start fam_e4b_arm); echo "PID $sp"; sleep 2.5; vram_stop $sp; echo STOPPED\n'
    return subprocess.run(["bash", "-c", script], cwd=tmp, check=False, capture_output=True, text=True, timeout=20,
                          env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", **env})


def test_the_sampler_returns_at_once_and_writes_both_files(tmp_path: Path):
    r = _run(tmp_path)                       # a loop whose stdout stayed on the $(...) pipe would hang here until the timeout
    assert r.returncode == 0 and "STOPPED" in r.stdout and re.search(r"PID \d+", r.stdout), (r.stdout, r.stderr)
    vram = (tmp_path / "vram_fam_e4b_arm.txt").read_text().splitlines()
    clk = (tmp_path / "gpuclk_fam_e4b_arm.txt").read_text().splitlines()
    assert len(vram) >= 2 and all(re.fullmatch(r"\d+ 24581, 97, 288\.4", ln) for ln in vram), vram   # the columns every reader knows, unchanged
    assert len(clk) >= 2 and all(re.match(r"\d+ 2407, 14001, 61, 0x0000000000000004 \| ", ln) for ln in clk), clk
    assert all(ln.count(" | ") == 2 for ln in clk), clk


def test_an_unknown_clock_field_fails_only_its_own_line(tmp_path: Path):
    r = _run(tmp_path, FAKE_CLK_FAIL="1")
    assert r.returncode == 0 and "STOPPED" in r.stdout, (r.stdout, r.stderr)
    assert all(re.fullmatch(r"\d+ 24581, 97, 288\.4", ln) for ln in (tmp_path / "vram_fam_e4b_arm.txt").read_text().splitlines())
    assert "not a valid field" in (tmp_path / "gpuclk_fam_e4b_arm.txt").read_text()


def test_a_second_start_for_the_same_arm_truncates_both_files(tmp_path: Path):
    (tmp_path / "vram_fam_e4b_arm.txt").write_text("stale\n")
    (tmp_path / "gpuclk_fam_e4b_arm.txt").write_text("stale\n")
    _run(tmp_path)
    assert "stale" not in (tmp_path / "vram_fam_e4b_arm.txt").read_text() and "stale" not in (tmp_path / "gpuclk_fam_e4b_arm.txt").read_text()
