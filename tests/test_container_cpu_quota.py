"""container_cpu_quota / host_cpu_report (experts4bit_qlora/util.py): the loader logs what a rented container may run on,
so a slow host-side load can be read from its receipt. CPU only, against fake cgroup trees."""
import os

from experts4bit_qlora.util import container_cpu_quota, host_cpu_report


def _tree(tmp_path, files):
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return str(tmp_path)


def test_v2_quota_in_cpus(tmp_path):
    q, d = container_cpu_quota(_tree(tmp_path, {"cpu.max": "3071999 100000\n"}))
    assert abs(q - 30.71999) < 1e-9 and d["cgroup"] == "v2"


def test_v2_unlimited_is_none(tmp_path):
    q, d = container_cpu_quota(_tree(tmp_path, {"cpu.max": "max 100000\n"}))
    assert q is None and "unlimited" in d["cpu.max"]


def test_v1_quota_and_unlimited(tmp_path):
    q, d = container_cpu_quota(_tree(tmp_path / "a", {"cpu/cpu.cfs_quota_us": "200000\n", "cpu/cpu.cfs_period_us": "100000\n"}))
    assert q == 2.0 and d["cgroup"] == "v1"
    q, d = container_cpu_quota(_tree(tmp_path / "b", {"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"}))
    assert q is None and d["cfs_quota_us"] == -1


def test_nothing_readable_is_none(tmp_path):
    q, d = container_cpu_quota(str(tmp_path))
    assert q is None and "no cgroup CPU controller" in d["error"]
    q, d = container_cpu_quota(_tree(tmp_path / "c", {"cpu.max": "garbage\n"}))
    assert q is None and "error" in d


def test_the_report_names_what_the_load_log_prints(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    r = host_cpu_report(_tree(tmp_path, {"cpu.max": "200000 100000\n"}))
    assert r["cpu_quota"] == 2.0 and r["omp_num_threads"] == "7" and r["torch_threads"] >= 1
    assert r["visible_cpus"] == (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())
