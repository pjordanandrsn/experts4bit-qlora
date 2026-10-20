"""bench/common/thp_window.py: THP records around a timed window, read from synthetic /proc and /sys trees."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "bench/common/thp_window.py"
_spec = importlib.util.spec_from_file_location("thp_window", SRC)
tw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tw)

ROLLUP = """55a0c0000000-7ffc12345000 ---p 00000000 00:00 0                          [rollup]
Rss:             4194304 kB
Pss:             4100000 kB
Anonymous:       3145728 kB
AnonHugePages:   2097152 kB
Swap:                  0 kB
"""
VMSTAT = "nr_free_pages 123\nthp_fault_alloc 10\nthp_fault_fallback 3\nthp_split_page 1\nthp_bogus 99\n"


def _tree(tmp: Path, *, enabled="always [madvise] never", defrag="[always] defer defer+madvise madvise never",
          rollup=ROLLUP, vmstat=VMSTAT, thp=True) -> Path:
    files = {"proc/self/smaps_rollup": rollup, "proc/vmstat": vmstat}
    if thp:
        files.update({"sys/kernel/mm/transparent_hugepage/enabled": enabled,
                      "sys/kernel/mm/transparent_hugepage/defrag": defrag})
    for rel, text in files.items():
        if text is None:
            continue
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp / rel).write_text(text)
    return tmp


def test_it_imports_the_standard_library_only():
    tree = ast.parse(SRC.read_text())
    names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert names <= {"__future__", "os", "re", "time", "pathlib"}, names


def test_a_snapshot_reads_the_mode_the_process_pages_and_the_counters(tmp_path, monkeypatch):
    monkeypatch.setenv("THP_MEM_ALLOC_ENABLE", "1")
    s = tw.snapshot(root=_tree(tmp_path))
    assert (s["thp_enabled"], s["thp_defrag"], s["thp_mem_alloc_enable"]) == ("madvise", "always", "1")
    assert (s["anon_huge_kb"], s["anon_kb"], s["rss_kb"]) == (2097152, 3145728, 4194304)
    assert s["vmstat"] == {"thp_fault_alloc": 10, "thp_fault_fallback": 3, "thp_split_page": 1}   # unknown keys ignored
    assert isinstance(s["t"], float)


def test_an_unset_switch_is_left_out_not_zero(tmp_path, monkeypatch):
    monkeypatch.delenv("THP_MEM_ALLOC_ENABLE", raising=False)
    assert "thp_mem_alloc_enable" not in tw.snapshot(root=_tree(tmp_path))


def test_a_kernel_without_thp_reads_absent_and_failed_reads_are_left_out(tmp_path):
    s = tw.snapshot(root=_tree(tmp_path, thp=False, rollup=None, vmstat=None))
    assert (s["thp_enabled"], s["thp_defrag"]) == ("absent", "absent")
    assert not any(k in s for k in ("anon_huge_kb", "anon_kb", "rss_kb", "vmstat"))


@pytest.mark.parametrize("enabled,want", [("[always] madvise never", "always"), ("always madvise [never]", "never"),
                                          ("always [weird] never", None), ("garbage", None)])
def test_an_unknown_or_unselected_mode_is_left_out(tmp_path, enabled, want):
    assert tw.snapshot(root=_tree(tmp_path, enabled=enabled)).get("thp_enabled") == want


def test_a_window_carries_both_snapshots_the_seconds_and_the_counter_deltas(tmp_path):
    root = _tree(tmp_path)
    before = tw.snapshot(root=root)
    (root / "proc/vmstat").write_text("thp_fault_alloc 25\nthp_fault_fallback 3\nthp_collapse_alloc 2\n")
    (root / "proc/self/smaps_rollup").write_text(ROLLUP.replace("2097152", "4096"))
    rec = tw.window(before, root=root)
    assert rec["before"] is before and rec["after"]["anon_huge_kb"] == 4096
    assert rec["vmstat_delta"] == {"thp_fault_alloc": 15, "thp_fault_fallback": 0}     # keys read in both snapshots only
    assert rec["anon_huge_kb_max"] == 2097152 and rec["seconds"] >= 0


def test_another_process_is_read_by_pid(tmp_path):
    root = _tree(tmp_path)
    (root / "proc/4242").mkdir(parents=True)
    (root / "proc/4242/smaps_rollup").write_text(ROLLUP.replace("2097152", "0"))
    assert tw.snapshot(pid=4242, root=root)["anon_huge_kb"] == 0


def test_the_real_host_reads_without_error():
    s = tw.snapshot()
    assert "t" in s and tw.window(s)["seconds"] >= 0
