#!/usr/bin/env python3
"""HOST-FACTS PROBE, box side (bench/hostprobe/PREREG-hostprobe.md): print the facts that bound other lanes' reads, and
nothing else. Standard library only: no torch, no model, no measurement of any kind.

    python3 hostprobe_box.py            -> one line, ``HOSTPROBE_JSON {...}``, on stdout; always exits 0
    python3 hostprobe_box.py --root R   -> read /proc and /sys under R instead of / (tests)

A fact that cannot be read is left out, never guessed. A kernel built without THP (no transparent_hugepage directory)
reads ``thp_enabled``/``thp_defrag`` = ``absent``, which is a fact, not a failed read. Every value is a number or a short
enum except ``kernel`` (the release string). ``thp_mem_alloc_enable`` and ``omp_num_threads`` are read from THIS
process's environment, i.e. the ssh session's, not any lane's. GPU 0's power limit, enforced limit and maximum SM clock
come from ``nvidia-smi``'s query, bounded at 20 s. Amendment 1 adds GPU 0's health facts (driver, CUDA and VBIOS
versions, ECC mode and error counts, retired pages, remapped rows, the clock-event reasons at the probe's moment) from
at most four more calls. Every ``nvidia-smi`` call shares one 40 s budget (each call at most 20 s), and a call that
times out stops every later one."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

THP_ENABLED = ("always", "madvise", "never")
THP_DEFRAG = ("always", "defer", "defer+madvise", "madvise", "never")


def _read(root: Path, rel: str) -> str | None:
    try:
        return (root / rel).read_text().strip()
    except OSError:
        return None


def _bracketed(text: str | None, allowed: tuple[str, ...]) -> str | None:
    """The selected mode of a sysfs choice line (``always [madvise] never`` -> ``madvise``), only if it is a known one."""
    m = re.search(r"\[([^\]]+)\]", text or "")
    return m.group(1) if m and m.group(1) in allowed else None


def _env_int(name: str) -> int | None:
    v = os.environ.get(name, "").strip()
    return int(v) if v.isdigit() else None


GPU_QUERY = ("power.limit", "enforced.power.limit", "clocks.max.sm")
GPU_KEYS = ("gpu_power_limit_w", "gpu_enforced_power_limit_w", "gpu_max_sm_mhz")


SMI_TIMEOUT = 20      # one call
SMI_BUDGET = 40       # every nvidia-smi call of one probe together


def _smi(smi: str, args: list[str], state: dict) -> str | None:
    """One bounded ``nvidia-smi`` call: its stdout when it exits 0 with output, else None. Every call of a probe shares
    one budget (``SMI_BUDGET``), and each is bounded at ``min(SMI_TIMEOUT, what is left)``. A call that times out, or a
    budget that runs out, stops every later call (``state["stopped"]`` = ``timeout`` or ``budget``). A slow or hung
    driver therefore holds the box for ``SMI_BUDGET`` at most."""
    if state.get("stopped"):
        return None
    now = time.monotonic()
    left = min(SMI_TIMEOUT, state.setdefault("end", now + SMI_BUDGET) - now)
    if left < 1:
        state["stopped"] = "budget"
        return None
    try:
        r = subprocess.run([smi, *args], capture_output=True, text=True, timeout=left)
    except subprocess.TimeoutExpired:
        state["stopped"] = "timeout" if left >= SMI_TIMEOUT else "budget"
        return None
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 and r.stdout.strip() else None


def gpu_facts(smi: str = "nvidia-smi", state: dict | None = None) -> dict:
    """GPU 0's power limit, enforced power limit and maximum SM clock (numbers only; '[N/A]' or a failed call is left
    out). Bounded at 20 s: a hung driver never holds the box."""
    text = _smi(smi, ["--id=0", "--query-gpu=" + ",".join(GPU_QUERY), "--format=csv,noheader,nounits"],
                {} if state is None else state)
    if text is None:
        return {}
    out = {}
    for key, raw in zip(GPU_KEYS, text.strip().splitlines()[0].split(",")):
        try:
            out[key] = float(raw.strip())
        except ValueError:
            continue
    return out


# Amendment 1: GPU 0's health facts. (query field, key, kind); a field the driver does not list is never queried,
# because one unknown name fails the whole query. Where two names give one key, the first the driver lists is used.
HEALTH_QUERY = (
    ("driver_version", "gpu_driver_version", "version"),
    ("vbios_version", "gpu_vbios_version", "vbios"),
    ("ecc.mode.current", "gpu_ecc_mode", "mode"),
    ("ecc.errors.corrected.volatile.total", "gpu_ecc_corrected_volatile", "int"),
    ("ecc.errors.corrected.aggregate.total", "gpu_ecc_corrected_aggregate", "int"),
    ("ecc.errors.uncorrected.volatile.total", "gpu_ecc_uncorrected_volatile", "int"),
    ("ecc.errors.uncorrected.aggregate.total", "gpu_ecc_uncorrected_aggregate", "int"),
    ("retired_pages.single_bit_ecc.count", "gpu_retired_pages_sbe", "int"),
    ("retired_pages.double_bit.count", "gpu_retired_pages_dbe", "int"),
    ("retired_pages.pending", "gpu_retired_pages_pending", "yesno"),
    ("clocks_event_reasons.active", "gpu_clock_event_reasons", "hex"),
    ("clocks_throttle_reasons.active", "gpu_clock_event_reasons", "hex"),    # the older drivers' name for the same mask
)
REMAP_QUERY = (
    ("remapped_rows.correctable", "gpu_remapped_rows_correctable", "int"),
    ("remapped_rows.uncorrectable", "gpu_remapped_rows_uncorrectable", "int"),
    ("remapped_rows.pending", "gpu_remapped_rows_pending", "yesno"),
    ("remapped_rows.failure", "gpu_remapped_rows_failure", "yesno"),
)


def _value(raw: str, kind: str):
    """A printed value as its fact, or None when it is not one ('[N/A]', '[Not Supported]', anything unexpected)."""
    raw = raw.strip()
    if kind == "int":
        return int(raw) if raw.isdigit() else None
    if kind == "version":
        return raw if re.fullmatch(r"\d{1,4}(\.\d{1,4}){1,3}", raw) else None
    if kind == "vbios":
        return raw if re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z.]{0,31}", raw) else None
    if kind == "mode":
        return raw.lower() if raw.lower() in ("enabled", "disabled") else None
    if kind == "yesno":
        return raw.lower() if raw.lower() in ("yes", "no") else None
    if kind == "hex":
        return int(raw, 16) if re.fullmatch(r"0x[0-9A-Fa-f]{1,16}", raw) else None
    return None


def _query(smi: str, flag: str, fields: tuple, state: dict) -> dict:
    text = _smi(smi, ["--id=0", f"{flag}=" + ",".join(f for f, _, _ in fields), "--format=csv,noheader,nounits"], state)
    if text is None:
        return {}
    out = {}
    for (_, key, kind), raw in zip(fields, text.strip().splitlines()[0].split(",")):
        v = _value(raw, kind)
        if v is not None:
            out[key] = v
    return out


def gpu_health(smi: str = "nvidia-smi", state: dict | None = None) -> dict:
    """Amendment 1: GPU 0's driver, CUDA and VBIOS versions, ECC mode and error counts, retired pages, remapped rows and
    the active clock-event reasons (a bitmask, at the probe's moment). At most four calls, inside the shared budget: the
    field list, the query, the remapped-rows query and ``--version``. A value that is not a fact is left out. A field
    counts as listed under any of the names on its name line (``"new.name" or "old.name"``)."""
    state = {} if state is None else state
    listed = {name for line in (_smi(smi, ["--help-query-gpu"], state) or "").splitlines() if line.startswith('"')
              for name in re.findall(r'"([a-z0-9_.]+)"', line)}
    fields, keys = [], set()
    for f, key, kind in HEALTH_QUERY:
        if f in listed and key not in keys:
            fields.append((f, key, kind))
            keys.add(key)
    out = _query(smi, "--query-gpu", tuple(fields), state) if fields else {}
    out.update(_query(smi, "--query-remapped-rows", REMAP_QUERY, state))
    m = re.search(r"^CUDA Version\s*:\s*(\d{1,3}\.\d{1,3})\s*$", _smi(smi, ["--version"], state) or "", re.M)
    if m:
        out["gpu_cuda_version"] = m.group(1)
    return out


def facts(root: Path = Path("/"), smi: str | None = "nvidia-smi") -> dict:
    out: dict = {}
    if not (root / "sys/kernel/mm/transparent_hugepage").is_dir():
        out["thp_enabled"] = out["thp_defrag"] = "absent"      # a kernel without THP: a fact, not a failed read
    for key, rel, allowed in (("thp_enabled", "sys/kernel/mm/transparent_hugepage/enabled", THP_ENABLED),
                              ("thp_defrag", "sys/kernel/mm/transparent_hugepage/defrag", THP_DEFRAG),
                              ("khugepaged_defrag", "sys/kernel/mm/transparent_hugepage/khugepaged/defrag", None)):
        if key in out:
            continue
        raw = _read(root, rel)
        if allowed is None:
            if raw is not None and raw.isdigit():
                out[key] = int(raw)
        else:
            mode = _bracketed(raw, allowed)
            if mode:
                out[key] = mode
    kernel = _read(root, "proc/sys/kernel/osrelease")
    if kernel and re.fullmatch(r"[0-9A-Za-z._+~-]{1,64}", kernel):
        out["kernel"] = kernel
    bal = _read(root, "proc/sys/kernel/numa_balancing")
    if bal is not None and bal.isdigit():
        out["numa_balancing"] = int(bal)
    meminfo = _read(root, "proc/meminfo") or ""
    m = re.search(r"^MemTotal:\s+(\d+) kB", meminfo, re.M)
    if m:
        out["mem_total_bytes"] = int(m.group(1)) * 1024
    mx = _read(root, "sys/fs/cgroup/memory.max")                                  # cgroup v2
    if mx == "max":
        out["cgroup_memory_max"] = "max"
    elif mx is not None and mx.isdigit():
        out["cgroup_memory_max"] = int(mx)
    else:
        v1 = _read(root, "sys/fs/cgroup/memory/memory.limit_in_bytes")           # cgroup v1
        if v1 is not None and v1.isdigit():
            out["cgroup_memory_max"] = "max" if int(v1) >= 1 << 62 else int(v1)
    cpus = _read(root, "sys/devices/system/cpu/online")
    if cpus and re.fullmatch(r"[0-9,\-]+", cpus):
        n = 0
        for part in cpus.split(","):
            a, _, b = part.partition("-")
            n += (int(b) - int(a) + 1) if b else 1
        out["online_cpus"] = n
    nodes = [p for p in (root / "sys/devices/system/node").glob("node[0-9]*")] if (root / "sys/devices/system/node").is_dir() else []
    if nodes:
        out["numa_nodes"] = len(nodes)
    if str(root) == "/" and hasattr(os, "sched_getaffinity"):
        out["affinity_cpus"] = len(os.sched_getaffinity(0))
    for key, name in (("thp_mem_alloc_enable", "THP_MEM_ALLOC_ENABLE"), ("omp_num_threads", "OMP_NUM_THREADS")):
        v = _env_int(name)
        if v is not None:
            out[key] = v
    if smi:
        state: dict = {}
        out.update(gpu_facts(smi, state))
        out.update(gpu_health(smi, state))
        if state.get("stopped"):
            out["gpu_smi_stopped"] = state["stopped"]      # a fact about the driver: timeout, or the budget ran out
    out["measured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[argv.index("--root") + 1]) if "--root" in argv else Path("/")
    smi = argv[argv.index("--smi") + 1] if "--smi" in argv else "nvidia-smi"
    try:
        f = facts(root, smi)
    except Exception as e:  # noqa: BLE001 -- a probe never fails its box; the drive records what came back
        f = {"error": f"{type(e).__name__}"}
    print("HOSTPROBE_JSON " + json.dumps(f, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
