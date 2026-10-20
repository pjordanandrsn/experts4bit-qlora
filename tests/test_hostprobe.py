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
STDLIB_ONLY = {"__future__", "json", "os", "re", "subprocess", "sys", "time", "pathlib"}


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
    monkeypatch.setattr(hb, "facts", lambda *a, **k: 1 / 0)
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


def test_a_kernel_without_thp_reads_absent_and_a_failed_read_is_left_out(tmp_path):
    root = _box(tmp_path / "a")
    import shutil
    shutil.rmtree(root / "sys/kernel/mm/transparent_hugepage")             # a kernel built without THP
    f = hb.facts(root, smi=None)
    assert (f["thp_enabled"], f["thp_defrag"]) == ("absent", "absent") and "khugepaged_defrag" not in f
    root = _box(tmp_path / "b")
    (root / "sys/kernel/mm/transparent_hugepage/enabled").unlink()          # the directory is there, the read fails
    f = hb.facts(root, smi=None)
    assert "thp_enabled" not in f and f["thp_defrag"] == "always"


def _smi(tmp_path, body: str) -> str:
    p = tmp_path / "nvidia-smi"
    p.write_text("#!/bin/sh\n" + body + "\n")
    p.chmod(0o755)
    return str(p)


@pytest.mark.parametrize("body,want", [
    ('echo "575.00, 600.00, 3090"', {"gpu_power_limit_w": 575.0, "gpu_enforced_power_limit_w": 600.0, "gpu_max_sm_mhz": 3090.0}),
    ('echo "[N/A], 575.00, 3090"', {"gpu_enforced_power_limit_w": 575.0, "gpu_max_sm_mhz": 3090.0}),
    ("echo nope; exit 9", {}),
])
def test_gpu_facts_from_the_query_and_what_is_left_out(tmp_path, body, want):
    assert hb.gpu_facts(_smi(tmp_path, body)) == want


def test_a_missing_nvidia_smi_adds_nothing(tmp_path):
    assert hb.gpu_facts(str(tmp_path / "no-such-smi")) == {}
    f = hb.facts(_box(tmp_path / "box"), smi=str(tmp_path / "no-such-smi"))
    assert not any(k.startswith("gpu_") for k in f)


# Amendment 1: GPU 0's health facts. The stand-in nvidia-smi answers each call by its flag, logs every call, and fails
# a --query-gpu that names a field its help does not list, as the real one does. Its --help-query-gpu and --version
# answers are verbatim captures (tests/fixtures/hostprobe/PROVENANCE.md); the query VALUES are made up for the test.
FIX = ROOT / "tests/fixtures/hostprobe"
HELP = (FIX / "nvidia-smi-help-query-gpu-575.64.05.excerpt.txt").read_text()
VERSION = (FIX / "nvidia-smi-version-575.64.05.txt").read_text()
# A constructed older-driver listing (not a capture): no clocks_event_reasons name and no retired_pages.pending.
OLDER = '"driver_version"\n"vbios_version"\n"clocks_throttle_reasons.active"\n"ecc.mode.current"\n'
STAND_IN = """echo "$@" >> "@T@/calls.log"
sleep @SLEEP@
case "$1" in
  --help-query-gpu) cat "@T@/help.txt" ;;
  --version) cat "@T@/version.txt" ;;
  --id=0) case "$2" in
    --query-gpu=power.limit*) echo "450.00, 450.00, 3120" ;;
    --query-gpu=*) for f in $(echo "${2#--query-gpu=}" | tr , " "); do
                     grep -q -e "^\\"$f\\"" -e "or \\"$f\\"" "@T@/help.txt" || { echo "Field \\"$f\\" is not a valid field to query."; exit 2; }
                   done; cat "@T@/query.txt" ;;
    --query-remapped-rows=*) cat "@T@/remap.txt"; exit @REMAP_RC@ ;;
  esac ;;
esac"""


def _health_smi(tmp_path, *, help_text=HELP, version=VERSION, sleep="0",
                query="575.64.05, 94.06.19.00.1F, Enabled, 0, 3, 0, 0, [N/A], [N/A], No, [N/A]",
                remap="0, 0, No, No", remap_rc=0) -> str:
    for name, text in (("help.txt", help_text), ("version.txt", version), ("query.txt", query + "\n"),
                       ("remap.txt", remap + "\n")):
        (tmp_path / name).write_text(text)
    body = STAND_IN.replace("@T@", str(tmp_path)).replace("@SLEEP@", sleep).replace("@REMAP_RC@", str(remap_rc))
    return _smi(tmp_path, body)


def _sent(tmp_path) -> list[str]:
    calls = (tmp_path / "calls.log").read_text().splitlines()
    q = [c for c in calls if "--query-gpu=" in c and "power.limit" not in c][0]
    return q.split("--query-gpu=", 1)[1].split(" ")[0].split(",")


def test_the_fixture_is_the_capture_it_names():
    import hashlib
    sums = {"nvidia-smi-help-query-gpu-575.64.05.excerpt.txt": "c15ce933eaebf627d924aa5caa739ed2668aa4670588b79f117416061e8d5f2a",
            "nvidia-smi-version-575.64.05.txt": "65422cc56db1df2224a9d40b87a2792319201563bd5d417fe8f07f6276d1278e"}
    for name, want in sums.items():
        assert hashlib.sha256((FIX / name).read_bytes()).hexdigest() == want
        assert want in (FIX / "PROVENANCE.md").read_text()


def test_gpu_health_reads_versions_ecc_pages_rows_and_reasons(tmp_path):
    h = hb.gpu_health(_health_smi(tmp_path))
    # on the real 575.64.05 listing every field is listed; event reasons by its new name (the old one is its alias)
    assert _sent(tmp_path) == ["driver_version", "vbios_version", "ecc.mode.current",
                               "ecc.errors.corrected.volatile.total", "ecc.errors.corrected.aggregate.total",
                               "ecc.errors.uncorrected.volatile.total", "ecc.errors.uncorrected.aggregate.total",
                               "retired_pages.single_bit_ecc.count", "retired_pages.double_bit.count",
                               "retired_pages.pending", "clocks_event_reasons.active"]
    assert h == {"gpu_driver_version": "575.64.05", "gpu_vbios_version": "94.06.19.00.1F", "gpu_ecc_mode": "enabled",
                 "gpu_ecc_corrected_volatile": 0, "gpu_ecc_corrected_aggregate": 3, "gpu_ecc_uncorrected_volatile": 0,
                 "gpu_ecc_uncorrected_aggregate": 0, "gpu_retired_pages_pending": "no",
                 "gpu_remapped_rows_correctable": 0, "gpu_remapped_rows_uncorrectable": 0,
                 "gpu_remapped_rows_pending": "no", "gpu_remapped_rows_failure": "no", "gpu_cuda_version": "12.9"}


def test_a_field_the_driver_does_not_list_is_never_queried_and_the_older_reason_name_is_used(tmp_path):
    q = "535.104.05, 95.02.3C.40.C3, [N/A], 0x0000000000000004"
    h = hb.gpu_health(_health_smi(tmp_path, help_text=OLDER, query=q, remap="[N/A]", version="garbage"))
    assert _sent(tmp_path) == ["driver_version", "vbios_version", "ecc.mode.current", "clocks_throttle_reasons.active"]
    assert h == {"gpu_driver_version": "535.104.05", "gpu_vbios_version": "95.02.3C.40.C3", "gpu_clock_event_reasons": 4}


def test_unsupported_remapped_rows_and_a_missing_version_flag_add_nothing(tmp_path):
    h = hb.gpu_health(_health_smi(tmp_path, remap="Remapped rows not supported", remap_rc=3, version=""))
    assert not any(k.startswith("gpu_remapped_rows_") for k in h) and "gpu_cuda_version" not in h
    assert h["gpu_driver_version"] == "575.64.05"


def test_a_call_that_times_out_stops_every_later_call(tmp_path, monkeypatch):
    monkeypatch.setattr(hb, "SMI_TIMEOUT", 1)
    f = hb.facts(_box(tmp_path / "box"), smi=_health_smi(tmp_path, sleep="3"))
    assert f["gpu_smi_stopped"] == "timeout" and not any(k.startswith("gpu_") and k != "gpu_smi_stopped" for k in f)
    log = tmp_path / "calls.log"                # on a loaded host the killed call may not have logged itself yet
    assert len(log.read_text().splitlines() if log.exists() else []) <= 1     # no call after the one that timed out


def test_a_slow_driver_runs_out_the_shared_budget_and_keeps_what_it_read(tmp_path, monkeypatch):
    # A clock that advances 1.5 s per read stands in for a slow driver, so a loaded host cannot move the result:
    # calls start at 0 and 1.5 s, and at 3.0 s less than 1 s of the 3.5 s budget is left.
    import time as real_time
    from types import SimpleNamespace
    ticks = iter(i * 1.5 for i in range(100))
    monkeypatch.setattr(hb, "time", SimpleNamespace(monotonic=lambda: next(ticks), strftime=real_time.strftime,
                                                    gmtime=real_time.gmtime))        # the probe's clock only
    monkeypatch.setattr(hb, "SMI_TIMEOUT", 5)
    monkeypatch.setattr(hb, "SMI_BUDGET", 3.5)
    f = hb.facts(_box(tmp_path / "box"), smi=_health_smi(tmp_path))
    assert f["gpu_smi_stopped"] == "budget" and f["gpu_power_limit_w"] == 450.0
    assert "gpu_driver_version" not in f and len((tmp_path / "calls.log").read_text().splitlines()) == 2


def test_facts_carry_the_power_and_health_rows_together(tmp_path):
    f = hb.facts(_box(tmp_path / "box"), smi=_health_smi(tmp_path))
    assert (f["gpu_power_limit_w"], f["gpu_driver_version"], f["gpu_ecc_mode"]) == (450.0, "575.64.05", "enabled")
    assert "gpu_smi_stopped" not in f
    calls = (tmp_path / "calls.log").read_text().splitlines()
    assert len(calls) == 5 and calls[1] == "--help-query-gpu"            # power, field list, query, rows, --version


@pytest.mark.parametrize("raw,kind,want", [("Enabled", "mode", "enabled"), ("[N/A]", "mode", None), ("12", "int", 12),
                                           ("-1", "int", None), ("Yes", "yesno", "yes"), ("1", "yesno", None),
                                           ("0x00000000000000A0", "hex", 160), ("0xZZ", "hex", None),
                                           ("575.64.05", "version", "575.64.05"), ("575", "version", None),
                                           ("GPU-1234 serial", "vbios", None)])
def test_a_printed_value_becomes_a_fact_or_is_left_out(raw, kind, want):
    assert hb._value(raw, kind) == want


def test_the_health_read_prints_no_identifier():
    names = {f for f, _, _ in hb.HEALTH_QUERY + hb.REMAP_QUERY}
    assert not any(s in n for n in names for s in ("uuid", "serial", "pci", "name", "board_id"))


def test_a_name_listed_only_as_an_alias_still_counts(tmp_path):
    aliased = '"driver_version"\n"clocks_throttle_reasons.active" or "clocks_event_reasons.active"\n'
    hb.gpu_health(_health_smi(tmp_path, help_text=aliased, query="575.64.05, 0x1"))
    assert _sent(tmp_path) == ["driver_version", "clocks_event_reasons.active"]


def test_a_call_that_times_out_is_killed_and_waited_on_for_one_second_at_most(monkeypatch):
    seen = []

    class Unkillable:                       # nvidia-smi in uninterruptible sleep: SIGKILL lands, the process stays
        def __init__(self, *a, **k):
            self.returncode = None

        def communicate(self, timeout=None):
            raise hb.subprocess.TimeoutExpired("nvidia-smi", timeout)

        def kill(self):
            seen.append("kill")

        stdout = None

        def wait(self, timeout=None):
            if timeout is None:
                raise AssertionError("an unbounded wait on a process that may never exit")
            seen.append(f"wait {timeout}")
            raise hb.subprocess.TimeoutExpired("nvidia-smi", timeout)

    monkeypatch.setattr(hb.subprocess, "Popen", Unkillable)
    state: dict = {}
    assert hb._smi("nvidia-smi", ["--version"], state) is None
    assert seen == ["kill", "wait 1"] and state["stopped"] == "timeout"
