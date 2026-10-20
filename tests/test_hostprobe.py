"""bench/hostprobe: the host-facts probe (PREREG-hostprobe.md). The box script reads files and its environment only."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "bench/hostprobe"
_spec = importlib.util.spec_from_file_location("hostprobe_box", HERE / "hostprobe_box.py")
hb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hb)
STDLIB_ONLY = {"__future__", "json", "os", "re", "sys", "time", "pathlib"}


def test_the_box_script_imports_nothing_heavy():
    tree = ast.parse((HERE / "hostprobe_box.py").read_text())
    names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert names <= STDLIB_ONLY, names - STDLIB_ONLY


def _box(tmp: Path, *, enabled="always [madvise] never", defrag="[always] defer defer+madvise madvise never",
         memmax="367224946688", v1=None) -> Path:
    files = {"sys/kernel/mm/transparent_hugepage/enabled": enabled,
             "sys/kernel/mm/transparent_hugepage/defrag": defrag,
             "sys/kernel/mm/transparent_hugepage/khugepaged/defrag": "1",
             "proc/sys/kernel/osrelease": "7.0.0-38-generic", "proc/sys/kernel/numa_balancing": "1",
             "proc/meminfo": "MemTotal:       1056591104 kB\nMemFree:  1 kB", "sys/devices/system/cpu/online": "0-255"}
    if memmax is not None:
        files["sys/fs/cgroup/memory.max"] = memmax
    if v1 is not None:
        files["sys/fs/cgroup/memory/memory.limit_in_bytes"] = v1
    for rel, text in files.items():
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp / rel).write_text(text + "\n")
    for n in (0, 1):
        (tmp / f"sys/devices/system/node/node{n}").mkdir(parents=True)
    return tmp


def test_the_facts_it_reads_and_their_values(tmp_path, monkeypatch):
    monkeypatch.setenv("THP_MEM_ALLOC_ENABLE", "1")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    f = hb.facts(_box(tmp_path))
    assert (f["thp_enabled"], f["thp_defrag"], f["khugepaged_defrag"]) == ("madvise", "always", 1)
    assert (f["kernel"], f["numa_balancing"], f["mem_total_bytes"]) == ("7.0.0-38-generic", 1, 1056591104 * 1024)
    assert (f["cgroup_memory_max"], f["online_cpus"], f["numa_nodes"]) == (367224946688, 256, 2)
    assert f["thp_mem_alloc_enable"] == 1 and "omp_num_threads" not in f and f["measured_at"].endswith("Z")
    assert "affinity_cpus" not in f                          # only read on the real root


@pytest.mark.parametrize("enabled,want", [("[always] madvise never", "always"), ("always madvise [never]", "never"),
                                          ("always [weird] never", None), ("garbage", None)])
def test_an_unknown_or_unselected_thp_mode_is_left_out(tmp_path, enabled, want):
    assert hb.facts(_box(tmp_path, enabled=enabled)).get("thp_enabled") == want


def test_cgroup_limits_unlimited_and_v1(tmp_path):
    assert hb.facts(_box(tmp_path / "a", memmax="max"))["cgroup_memory_max"] == "max"
    assert hb.facts(_box(tmp_path / "b", memmax=None, v1="9223372036854771712"))["cgroup_memory_max"] == "max"
    assert hb.facts(_box(tmp_path / "c", memmax=None, v1="68719476736"))["cgroup_memory_max"] == 68719476736
    assert "cgroup_memory_max" not in hb.facts(_box(tmp_path / "d", memmax=None))


def test_main_prints_one_line_and_never_fails(capsys, monkeypatch, tmp_path):
    assert hb.main(["--root", str(_box(tmp_path))]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and json.loads(out[0].split(" ", 1)[1])["thp_enabled"] == "madvise"
    monkeypatch.setattr(hb, "facts", lambda root: 1 / 0)
    assert hb.main([]) == 0 and json.loads(capsys.readouterr().out.split(" ", 1)[1]) == {"error": "ZeroDivisionError"}


def _drive(tmp_path, run_id, ssh_body=None, unset=()):
    env = {k: v for k, v in os.environ.items() if not k.startswith("E4B_RENT_")}
    env.update(E4B_RENT_SSH_HOST="h.invalid", E4B_RENT_SSH_PORT="22", E4B_RENT_SSH_OPTS="-o X=Y",
               E4B_RENT_RUN_DIR=str(tmp_path / "run"), E4B_RENT_RUN_ID=run_id)
    for k in unset:
        env.pop(k)
    if ssh_body is not None:                                  # a stand-in ssh first on PATH: runs the streamed script locally
        bindir = tmp_path / "bin"
        bindir.mkdir()
        (bindir / "ssh").write_text("#!/bin/sh\n" + ssh_body + "\n")
        (bindir / "ssh").chmod(0o755)
        env["PATH"] = f"{bindir}:{env['PATH']}"
    return subprocess.run(["bash", str(HERE / "hostprobe_drive.sh")], env=env, capture_output=True, text=True,
                          timeout=60)


@pytest.mark.parametrize("bad", ["p159-1", "hostprobe-11541", "hostprobe-x-1", "hostprobe-11541-1;x"])
def test_the_drive_refuses_an_undeclared_run_id(tmp_path, bad):
    assert _drive(tmp_path, bad).returncode == 78


def test_the_drive_refuses_without_the_launchers_environment(tmp_path):
    assert _drive(tmp_path, "hostprobe-11541-1", unset=("E4B_RENT_SSH_HOST",)).returncode == 78


def test_the_drive_writes_the_facts_line_to_the_receipt(tmp_path):
    r = _drive(tmp_path, "hostprobe-11541-1", ssh_body="exec python3 -")
    facts = json.loads((tmp_path / "run/hostprobe/facts.json").read_text())
    assert r.returncode == 0 and "measured_at" in facts, r.stdout + r.stderr


def test_no_facts_line_is_rc_22_and_writes_no_facts(tmp_path):
    r = _drive(tmp_path, "hostprobe-11541-1", ssh_body="cat >/dev/null; echo nothing here")
    assert r.returncode == 22 and not (tmp_path / "run/hostprobe/facts.json").exists()
