"""Q235 Amendment 9 (reported only): the box's per-copy staging record and the host-rate probe's by-buffer link.

The probe's helpers are pure and run anywhere: the sysfs CPU list, the PCI bus id, the NUMA topology and the card's NUMA node and
IOMMU mode read from a fake sysfs tree, and the numa_maps lookup. The staging record is checked on fake CUDA events: its rate and
spread, the per-layer attribution under the registered schedule, and its refusal to attribute when the call count is not that
schedule's.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "bench" / "q235" / "q235_hostrate.py"
BOX = ROOT / "bench" / "q235" / "instrument" / "q235_box.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def probe():
    return _load(PROBE, "q235_hostrate_a9")


def test_cpu_lists_and_bus_ids_parse(probe):
    assert probe._cpulist("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]
    assert probe._cpulist("5") == [5]
    assert probe.sysfs_bdf("00000000:41:00.0") == "0000:41:00.0"
    assert probe.sysfs_bdf("0000000A:C1:00.0\n") == "000a:c1:00.0"


def _sysfs(tmp_path, nodes, numa_node="1", iommu_type="DMA-FQ"):
    root = tmp_path / "sys"
    for n, cpus in nodes.items():
        d = root / "devices" / "system" / "node" / f"node{n}"
        d.mkdir(parents=True)
        (d / "cpulist").write_text(cpus + "\n")
        (d / "meminfo").write_text(f"Node {n} MemTotal:       1048576 kB\nNode {n} MemFree: 1 kB\n")
    dev = root / "bus" / "pci" / "devices" / "0000:41:00.0"
    dev.mkdir(parents=True)
    (dev / "numa_node").write_text(numa_node + "\n")
    if iommu_type is not None:
        g = root / "kernel" / "iommu_groups" / "37"
        g.mkdir(parents=True)
        (g / "type").write_text(iommu_type + "\n")
        os.symlink(g, dev / "iommu_group")
    cmd = tmp_path / "cmdline"
    cmd.write_text("BOOT_IMAGE=/vmlinuz root=/dev/sda1 amd_iommu=on iommu=pt quiet\n")
    return root, cmd


def test_numa_topology_reads_nodes_with_cpus_only(probe, tmp_path):
    root, _ = _sysfs(tmp_path, {0: "0-11,48-59", 1: "12-23", 2: ""})
    topo = probe.numa_topology(root)
    assert sorted(topo) == [0, 1]                       # a node with no CPUs (memory only) is not a binding target
    assert len(topo[0]["cpus"]) == 24 and topo[1]["mem_bytes"] == 1024 ** 3
    assert probe.numa_topology(tmp_path / "absent") == {}


def test_card_placement_reads_numa_node_iommu_mode_and_kernel_args(probe, tmp_path):
    root, cmd = _sysfs(tmp_path, {0: "0-3"})
    c = probe.card_placement("00000000:41:00.0", root, cmd)
    assert c["gpu_numa_node"] == 1 and c["iommu_group"] == "37" and c["iommu_type"] == "DMA-FQ"
    assert c["kernel_iommu_args"] == ["amd_iommu=on", "iommu=pt"]
    root3, cmd3 = _sysfs(tmp_path / "c", {0: "0-3"})
    (root3 / "bus" / "pci" / "devices" / "0000:41:00.0" / "numa_node").unlink()      # a kernel without NUMA has no such file
    c4 = probe.card_placement("00000000:41:00.0", root3, cmd3)
    assert c4["gpu_numa_node"] is None and "without NUMA" in c4["gpu_numa_why"]
    root2, cmd2 = _sysfs(tmp_path / "b", {0: "0-3"}, numa_node="-1", iommu_type=None)
    c2 = probe.card_placement("00000000:41:00.0", root2, cmd2)
    assert c2["gpu_numa_node"] == -1 and c2["iommu_type"] is None and "no IOMMU group" in c2["iommu_why"]
    c3 = probe.card_placement("00000000:99:00.0", root, cmd)
    assert "not exposed" in c3["why"]
    assert "no PCI bus id" in probe.card_placement(None, root, cmd)["why"]


def test_numa_maps_lookup_finds_the_mapping_holding_the_pointer(probe):
    maps = ("7f0000000000-7f0040000000 rw-s 00000000 00:05 12 /dev/nvidiactl\n"
            "7f1000000000-7f1040000000 rw-p 00000000 00:00 0\n")
    numa = ("7f0000000000 default file=/dev/nvidiactl\n"
            "7f1000000000 bind:1 anon=262144 dirty=262144 N1=262144 kernelpagesize_kB=4\n")
    assert probe.numa_maps_nodes(0x7f1000001000, maps, numa) == {"policy": "bind:1", "pages": {"1": 262144}}
    assert probe.numa_maps_nodes(0x7f0000001000, maps, numa) is None            # a driver mapping that lists no node counts
    assert probe.numa_maps_nodes(0x10, maps, numa) is None                      # no mapping holds it


def test_page_nodes_answers_or_says_why(probe):
    import ctypes
    buf = ctypes.create_string_buffer(1 << 20)
    got = probe.page_nodes(ctypes.addressof(buf), 1 << 20, samples=4)
    if sys.platform == "linux":
        assert got["method"] in ("move_pages", "numa_maps", None)
        assert got["method"] is not None or got["why"]
        if got["method"] == "move_pages":
            assert sum(got["nodes"].values()) >= 1
    else:
        assert got["method"] is None and "not Linux" in got["why"]


def test_link_by_buffer_without_cuda_reports_why(probe, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert probe.link_by_buffer(8, 5) == {"h2d_by_buffer": None, "h2d_by_buffer_why": "no CUDA device"}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert probe.link_by_buffer(0, 5)["h2d_by_buffer_why"].startswith("disabled")


class _Ev:
    def __init__(self, t):
        self.t = t

    def elapsed_time(self, other):
        return other.t - self.t


def _copies(ms, nbytes=1000):
    out, t = [], 0.0
    for m in ms:
        out.append((_Ev(t), _Ev(t + m), nbytes, 4, "sync"))
        t += m + 1.0
    return out


def test_staging_record_rates_spread_and_the_registered_schedule():
    pytest.importorskip("torch")
    box = _load(BOX, "q235_box_a9")
    # 3 layers, 2 passes: per pass layers 1, 2 forward, then 2, 1, 0 recompute (2L - 1 = 5 calls)
    sched = [10.0, 20.0, 20.0, 10.0, 40.0]
    rec = box.staging_record(_copies(sched * 2), n_layers=3, passes=2)
    assert rec["n"] == 10 and rec["ms"][:5] == sched and rec["nbytes"] == [1000] and rec["policies"] == ["sync"]
    assert rec["total_ms"] == 200.0 and rec["gb_s"] == round(10 * 1000 / 0.2 / 1e9, 3)
    assert rec["gb_s_min"] == round(1000 / 0.04 / 1e9, 3) and rec["gb_s_max"] == round(1000 / 0.01 / 1e9, 3)
    by = rec["by_layer"]
    assert set(by) == {"0", "1", "2"}
    assert by["0"]["calls"] == 2 and by["0"]["ms_mean"] == 40.0             # recompute only: layer 0 stays resident for the forward
    assert by["1"]["calls"] == 4 and by["1"]["ms_mean"] == 10.0
    assert by["2"]["calls"] == 4 and by["2"]["ms_mean"] == 20.0


def test_staging_record_does_not_attribute_an_unregistered_schedule():
    pytest.importorskip("torch")
    box = _load(BOX, "q235_box_a9b")
    rec = box.staging_record(_copies([5.0] * 11), n_layers=3, passes=2)
    assert "by_layer" not in rec and "11 calls, not 2 pass(es) x 5" in rec["by_layer_why"]
    assert box.staging_record([], n_layers=3, passes=1) == {"n": 0}


def test_a_failing_staging_record_is_recorded_not_raised():
    pytest.importorskip("torch")
    box = _load(BOX, "q235_box_a9c")

    class _Broken:
        def elapsed_time(self, other):
            raise RuntimeError("event not recorded")

    got = box.staging_or_error([(_Broken(), _Broken(), 1, 1, "sync")], n_layers=3, passes=1)
    assert got == {"error": "RuntimeError: event not recorded"}


def test_the_single_buffer_read_runs_before_the_by_buffer_read(probe, monkeypatch, tmp_path):
    """The floors read the single-buffer figure: it must be timed before any by-buffer block exists for it to reuse."""
    order = []
    monkeypatch.setattr(probe, "link", lambda copies: order.append("link") or {"h2d_gb_s": 1.0})
    monkeypatch.setattr(probe, "link_by_buffer", lambda k, c: order.append("by_buffer") or {"h2d_by_buffer": None})
    monkeypatch.setattr(probe, "host_copy", lambda: order.append("host_copy") or {"host_copy_gb_s": 1.0})
    monkeypatch.setattr(probe, "disk", lambda d, g, r: order.append("disk") or {"disk_read_gb_s": 1.0, "disk_method": "x"})
    monkeypatch.setattr(sys, "argv", ["q235_hostrate.py", "--out", str(tmp_path / "h.json"), "--dir", str(tmp_path)])
    probe.main()
    assert order == ["link", "by_buffer", "host_copy", "disk"]


def test_a_failing_by_buffer_read_is_recorded_and_the_phase_still_passes(probe, monkeypatch, tmp_path):
    def boom(k, c):
        raise RuntimeError("pin failed")
    monkeypatch.setattr(probe, "link", lambda copies: {"h2d_gb_s": 9.0})
    monkeypatch.setattr(probe, "link_by_buffer", boom)
    monkeypatch.setattr(probe, "host_copy", lambda: {"host_copy_gb_s": 9.0})
    monkeypatch.setattr(probe, "disk", lambda d, g, r: {"disk_read_gb_s": 9.0, "disk_method": "x"})
    monkeypatch.setattr(sys, "argv", ["q235_hostrate.py", "--out", str(tmp_path / "h.json"), "--dir", str(tmp_path),
                                      "--min-h2d-gbs", "1"])
    probe.main()                                                           # no SystemExit: the floors pass, the failure is recorded
    import json
    rec = json.loads((tmp_path / "h.json").read_text())
    assert rec["h2d_by_buffer"] is None and "RuntimeError: pin failed" in rec["h2d_by_buffer_why"] and rec["h2d_gb_s"] == 9.0
