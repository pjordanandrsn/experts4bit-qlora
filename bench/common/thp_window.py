"""Transparent-huge-page records around a timed window (shared by lanes; standard library only).

The launcher's host-facts probe (``host_facts.py``) reads the host's THP mode once per run. A lane whose reading can
depend on huge pages needs more, read inside the measured process at the timed window: whether huge pages were
possible there, and whether this process actually held any.

    import thp_window
    before = thp_window.snapshot()          # just before the timed window
    ...                                     # the timed work
    rec = thp_window.window(before)         # just after: both snapshots and the counter deltas

A snapshot holds:
- ``thp_enabled`` / ``thp_defrag``: the selected mode in /sys/kernel/mm/transparent_hugepage, ``absent`` on a kernel built
  without THP. The mode is host-wide.
- ``thp_mem_alloc_enable``: the read process's ``THP_MEM_ALLOC_ENABLE``, the switch with which torch asks for huge pages on
  its CPU allocations under ``madvise``: this process's environment for ``pid="self"``, else /proc/<pid>/environ (left
  out if unreadable). It is left out when unset: absent is not 0.
- ``anon_huge_kb``, ``anon_kb``, ``rss_kb``: this process's AnonHugePages, Anonymous and Rss from /proc/<pid>/smaps_rollup.
  ``anon_huge_kb`` > 0 means the process held huge pages at that moment.
- ``vmstat``: the host's ``thp_*`` event counters from /proc/vmstat. They count every process on the host, so a window's
  delta includes co-tenants' faults; read them as the host's, not the lane's.
- ``t``: ``time.monotonic()`` at the read.

A fact that cannot be read is left out, never guessed. Reading costs a few small /proc and /sys reads (well under a
millisecond), but a lane should still take snapshots outside the interval it times."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

THP_MODES = ("always", "madvise", "never", "defer", "defer+madvise")
VMSTAT_KEYS = ("thp_fault_alloc", "thp_fault_fallback", "thp_fault_fallback_charge", "thp_collapse_alloc",
               "thp_collapse_alloc_failed", "thp_split_page", "thp_deferred_split_page")
SMAPS_KEYS = (("AnonHugePages", "anon_huge_kb"), ("Anonymous", "anon_kb"), ("Rss", "rss_kb"))


def _read(root: Path, rel: str) -> str | None:
    try:
        return (root / rel).read_text()
    except OSError:
        return None


def _mode(text: str | None) -> str | None:
    """The selected mode of a sysfs choice line (``always [madvise] never`` -> ``madvise``), only if it is a known one."""
    m = re.search(r"\[([^\]]+)\]", text or "")
    return m.group(1) if m and m.group(1) in THP_MODES else None


def _env(pid: int | str, root: Path) -> dict:
    """The read process's environment: os.environ for ``self``, else /proc/<pid>/environ ({} if unreadable). Another
    process's switch is never reported from the caller's environment."""
    if str(pid) == "self":
        return dict(os.environ)
    try:
        raw = (root / f"proc/{pid}/environ").read_bytes()
    except OSError:
        return {}
    pairs = (item.decode("utf-8", "replace").partition("=") for item in raw.split(b"\0") if item)
    return {k: v for k, _, v in pairs if k}


def snapshot(pid: int | str = "self", root: Path | str = "/") -> dict:
    root = Path(root)
    out: dict = {"t": time.monotonic()}
    if not (root / "sys/kernel/mm/transparent_hugepage").is_dir():
        out["thp_enabled"] = out["thp_defrag"] = "absent"          # a kernel without THP: a fact, not a failed read
    else:
        for key in ("enabled", "defrag"):
            mode = _mode(_read(root, f"sys/kernel/mm/transparent_hugepage/{key}"))
            if mode:
                out[f"thp_{key}"] = mode
    env = _env(pid, root).get("THP_MEM_ALLOC_ENABLE", "").strip()
    if env:
        out["thp_mem_alloc_enable"] = env                             # as set: torch reads only "1" as on
    rollup = _read(root, f"proc/{pid}/smaps_rollup") or ""
    for field, key in SMAPS_KEYS:
        m = re.search(rf"^{field}:\s+(\d+) kB$", rollup, re.M)
        if m:
            out[key] = int(m.group(1))
    vmstat = {}
    for line in (_read(root, "proc/vmstat") or "").splitlines():
        name, _, value = line.partition(" ")
        if name in VMSTAT_KEYS and value.strip().isdigit():
            vmstat[name] = int(value)
    if vmstat:
        out["vmstat"] = vmstat
    return out


def window(before: dict, after: dict | None = None, pid: int | str = "self", root: Path | str = "/") -> dict:
    """Both snapshots of a timed window, the seconds between them, and the host's thp_* counter deltas (keys read in both
    snapshots only). ``anon_huge_kb_ends_max`` is the larger of the two endpoint readings: it can miss a peak inside the
    window, which only a sampler during the window would see."""
    after = snapshot(pid, root) if after is None else after
    b, a = before.get("vmstat") or {}, after.get("vmstat") or {}
    rec = {"before": before, "after": after, "seconds": round(after["t"] - before["t"], 6),
           "vmstat_delta": {k: a[k] - b[k] for k in VMSTAT_KEYS if k in a and k in b}}
    if "anon_huge_kb" in before and "anon_huge_kb" in after:
        rec["anon_huge_kb_ends_max"] = max(before["anon_huge_kb"], after["anon_huge_kb"])
    return rec
