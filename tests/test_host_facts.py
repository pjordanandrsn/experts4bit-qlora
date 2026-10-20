"""bench/common/host_facts.py: the box's own measured facts for the host registry, numeric or a small enum, with a
source per fact and no identifiers. Every reading here comes from a synthetic /proc and /sys tree."""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("host_facts", ROOT / "bench/common/host_facts.py")
hf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hf)

# what a box must never write into host_facts.json (the citation rule: no identifiers)
FORBIDDEN = ("host", "hostname", "container", "uuid", "serial", "model", "name", "ip", "mac", "id")


def _box(tmp: Path, *, sockets=2, cores=2, threads=2, nodes=2, thp="always [madvise] never",
         defrag="[always] defer madvise never", balancing="1", cpu_max="6120000 100000", max_khz="3500000") -> Path:
    blocks = []
    n = 0
    for s in range(sockets):
        for c in range(cores):
            for _t in range(threads):
                blocks.append(f"processor\t: {n}\nvendor_id\t: AuthenticAMD\n"
                              "model name\t: AMD EPYC 7B13 64-Core Processor\n"
                              f"physical id\t: {s}\ncore id\t\t: {c}\nflags\t\t: fpu sse\n")
                n += 1
    (tmp / "proc/sys/kernel").mkdir(parents=True)
    (tmp / "proc/cpuinfo").write_text("\n".join(blocks))
    if balancing is not None:
        (tmp / "proc/sys/kernel/numa_balancing").write_text(balancing + "\n")
    for i in range(nodes):
        (tmp / f"sys/devices/system/node/node{i}").mkdir(parents=True)
    (tmp / "sys/devices/system/node/possible").parent.mkdir(parents=True, exist_ok=True)
    (tmp / "sys/devices/system/node/possible").write_text("0-1\n")          # not a node directory: never counted
    thp_dir = tmp / "sys/kernel/mm/transparent_hugepage"
    thp_dir.mkdir(parents=True)
    if thp is not None:
        (thp_dir / "enabled").write_text(thp + "\n")
    if defrag is not None:
        (thp_dir / "defrag").write_text(defrag + "\n")
    for c, khz in enumerate([max_khz] if isinstance(max_khz, str) else (max_khz or [])):
        (tmp / f"sys/devices/system/cpu/cpu{c}/cpufreq").mkdir(parents=True)
        (tmp / f"sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq").write_text(khz + "\n")
    if cpu_max is not None:
        (tmp / "sys/fs/cgroup").mkdir(parents=True)
        (tmp / "sys/fs/cgroup/cpu.max").write_text(cpu_max + "\n")
    return tmp


def _smi(tmp: Path, line: str = "575.00, 3090, 32607", rc: int = 0) -> str:
    p = tmp / "fake-smi"
    p.write_text(f"#!/bin/sh\necho '{line}'\nexit {rc}\n")
    p.chmod(0o755)
    return str(p)


@pytest.fixture(autouse=True)
def _fast_loop(monkeypatch):
    monkeypatch.setattr(hf, "cpu_loop_mops", lambda: 42.5)        # the real loop has its own test below
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)


def test_the_facts_a_box_measures_and_where_each_came_from(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "affinity", lambda: set(range(6)))
    monkeypatch.setenv("OMP_NUM_THREADS", "24")
    root = _box(tmp_path / "box")
    f = hf.gather(root, _smi(tmp_path))
    assert f["host_logical_cpus"] == 8 and f["host_physical_cores"] == 4 and f["host_threads_per_core"] == 2.0
    assert f["host_numa_nodes"] == 2 and f["numa_balancing"] == 1
    assert f["thp_enabled"] == "madvise" and f["thp_defrag"] == "always"
    assert f["cpu_quota_cpus"] == 61.2 and "cpu_quota_unlimited" not in f
    assert f["affinity_cpus"] == 6                                         # the run's share, beside the host's 8
    assert (f["gpu_power_limit_w"], f["gpu_max_sm_mhz"], f["gpu_mem_total_mib"]) == (575.0, 3090.0, 32607.0)
    assert f["cpu_max_mhz"] == f["cpu_max_mhz_lo"] == 3500.0 and f["cpu_loop_mops"] == 42.5 and f["omp_num_threads"] == 24
    assert f["cpu_loop_python"] == sys.version_info[0] * 100 + sys.version_info[1]
    measured = {k for k in f if k not in ("measured_at", "sources")}
    assert set(f["sources"]) == measured                                   # every fact names its source
    assert dt.datetime.strptime(f["measured_at"], "%Y-%m-%dT%H:%M:%SZ")


def test_no_identifier_and_no_free_text_is_ever_written(tmp_path):
    f = hf.gather(_box(tmp_path / "box"), _smi(tmp_path))
    for k, v in f.items():
        if k in ("measured_at", "sources"):
            continue
        parts = k.split("_")[1:] if k.startswith("host_") else k.split("_")       # host_ marks a host-wide fact
        assert not any(bad == part for part in parts for bad in FORBIDDEN), k
        assert isinstance(v, (int, float)) or v in hf.THP_MODES, (k, v)   # a number or the small enum
    assert "EPYC" not in json.dumps(f)                                     # the CPU model string stays out


def test_a_fact_that_cannot_be_read_is_left_out(tmp_path):
    root = _box(tmp_path / "box", nodes=0, thp=None, defrag="garbage", balancing=None, cpu_max="max 100000",
                max_khz=None)
    f = hf.gather(root, str(tmp_path / "no-such-smi"))
    for k in ("host_numa_nodes", "thp_enabled", "thp_defrag", "numa_balancing", "cpu_quota_cpus", "gpu_power_limit_w",
              "cpu_max_mhz", "cpu_max_mhz_lo", "omp_num_threads"):
        assert k not in f and k not in f["sources"], k
    assert f["host_physical_cores"] == 4                                        # the rest still reads
    assert "gpu_max_sm_mhz" not in hf.gather(root, _smi(tmp_path, rc=9))   # a failing nvidia-smi adds nothing
    assert "gpu_mem_total_mib" not in hf.gather(root, _smi(tmp_path, line="[N/A], 3090"))


def test_the_quota_reads_cgroup_v1_and_tells_unlimited_from_unreadable(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "affinity", lambda: set(range(4)))
    v2max = hf.gather(_box(tmp_path / "a", cpu_max="max 100000"), str(tmp_path / "no-smi"))
    assert v2max["cpu_quota_unlimited"] == 1 and "cpu_quota_cpus" not in v2max
    v1 = _box(tmp_path / "b", cpu_max=None)
    (v1 / "sys/fs/cgroup/cpu").mkdir(parents=True)
    (v1 / "sys/fs/cgroup/cpu/cpu.cfs_quota_us").write_text("400000\n")
    (v1 / "sys/fs/cgroup/cpu/cpu.cfs_period_us").write_text("100000\n")
    f = hf.gather(v1, str(tmp_path / "no-smi"))
    assert f["cpu_quota_cpus"] == 4.0 and "cgroup v1" in f["sources"]["cpu_quota_cpus"]
    (v1 / "sys/fs/cgroup/cpu/cpu.cfs_quota_us").write_text("-1\n")
    assert hf.gather(v1, str(tmp_path / "no-smi"))["cpu_quota_unlimited"] == 1
    none = hf.gather(_box(tmp_path / "c", cpu_max=None), str(tmp_path / "no-smi"))
    assert "cpu_quota_cpus" not in none and "cpu_quota_unlimited" not in none     # unreadable: neither key


def test_no_affinity_call_leaves_the_fact_out(tmp_path, monkeypatch):
    def gone():
        raise AttributeError("sched_getaffinity")                     # macOS has no affinity call
    monkeypatch.setattr(hf, "affinity", gone)
    assert "affinity_cpus" not in hf.gather(_box(tmp_path / "box"), str(tmp_path / "no-smi"))


def test_cpuinfo_without_topology_gives_logical_cpus_only(tmp_path):
    root = tmp_path / "box"
    (root / "proc").mkdir(parents=True)
    (root / "proc/cpuinfo").write_text("processor\t: 0\n\nprocessor\t: 1\n")   # e.g. an ARM box: no physical/core id
    f = hf.gather(root, str(tmp_path / "no-smi"))
    assert f["host_logical_cpus"] == 2 and "host_physical_cores" not in f and "host_threads_per_core" not in f


def test_main_writes_the_file_and_never_fails_the_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOST_FACTS_ROOT", str(_box(tmp_path / "box")))
    monkeypatch.setenv("HOST_FACTS_SMI", _smi(tmp_path))
    out = tmp_path / "host_facts.json"
    assert hf.main([str(out)]) == 0
    assert json.loads(out.read_text())["host_physical_cores"] == 4 and "HOST_FACTS" in capsys.readouterr().out
    assert hf.main([str(tmp_path / "no-such-dir" / "host_facts.json")]) == 0   # unwritable: says so, exits 0
    assert "not written" in capsys.readouterr().out
    assert hf.main([]) == 0


def test_omp_is_left_out_unless_a_positive_integer(tmp_path, monkeypatch):
    root = _box(tmp_path / "box")
    for v in ("", "0", "auto", "8,4"):                                     # unset, zero or a list: absent, never 0
        monkeypatch.setenv("OMP_NUM_THREADS", v)
        assert "omp_num_threads" not in hf.gather(root, str(tmp_path / "no-smi")), v


def test_a_failing_loop_leaves_both_loop_facts_out(tmp_path, monkeypatch):
    def broken():
        raise RuntimeError("no timer")
    monkeypatch.setattr(hf, "cpu_loop_mops", broken)
    f = hf.gather(_box(tmp_path / "box"), str(tmp_path / "no-smi"))
    assert "cpu_loop_mops" not in f and "cpu_loop_python" not in f and "cpu_loop_mops" not in f["sources"]


def test_the_real_loop_is_fast_and_positive(monkeypatch):
    monkeypatch.undo()                                                      # the real function, not the fixture's
    monkeypatch.setattr(hf, "LOOP_N", 100_000)
    import time as _t
    t0 = _t.perf_counter()
    v = hf.cpu_loop_mops()
    assert v > 0 and _t.perf_counter() - t0 < 5


def test_a_hybrid_part_records_its_highest_and_lowest_core_maximum(tmp_path, monkeypatch):
    """A P+E part (e.g. 8 P-cores at 5.7 GHz, E-cores at 4.6): cpu0 alone would report only the P clock."""
    root = _box(tmp_path / "box", max_khz=["5700000", "5700000", "4600000", "4600000"])
    monkeypatch.setattr(hf, "affinity", lambda: {0, 1, 2, 3})
    f = hf.gather(root, str(tmp_path / "no-smi"))
    assert (f["cpu_max_mhz"], f["cpu_max_mhz_lo"]) == (5700.0, 4600.0)
    monkeypatch.setattr(hf, "affinity", lambda: {2, 3})                    # the run's share: E-cores only
    f = hf.gather(root, str(tmp_path / "no-smi"))
    assert (f["cpu_max_mhz"], f["cpu_max_mhz_lo"]) == (4600.0, 4600.0)


def test_no_affinity_call_reads_cpu0(tmp_path, monkeypatch):
    def gone():
        raise AttributeError("sched_getaffinity")
    monkeypatch.setattr(hf, "affinity", gone)
    root = _box(tmp_path / "box", max_khz=["3000000", "2000000"])
    f = hf.gather(root, str(tmp_path / "no-smi"))
    assert (f["cpu_max_mhz"], f["cpu_max_mhz_lo"]) == (3000.0, 3000.0)
