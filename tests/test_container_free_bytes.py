"""`container_free_bytes` must read BOTH cgroup layouts, and count cache as free.

Written against two failures that each wasted a rented run:

  * a v2-only reader measured nothing on a RunPod pod, because those pods are
    cgroup **v1** — it fell back to a guess and reported the fallback as if it
    were a measurement;
  * subtracting nothing for the page cache read **18.3 MB free** straight after a
    138 GiB write, because both cgroup versions count reclaimable cache as used.

`root` is a parameter precisely so both layouts can be exercised here rather than
only on whichever host happens to run the suite.
"""
from __future__ import annotations

import os

from experts4bit_qlora.util import container_free_bytes

GB = 1 << 30


def _write(p, text):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)


def _v2(root, limit, current, file_cache):
    _write(os.path.join(root, "memory.max"), f"{limit}\n")
    _write(os.path.join(root, "memory.current"), f"{current}\n")
    _write(os.path.join(root, "memory.stat"),
           f"anon {current - file_cache}\nfile {file_cache}\nkernel 4096\n")


def _v1(root, limit, usage, cache):
    _write(os.path.join(root, "memory/memory.limit_in_bytes"), f"{limit}\n")
    _write(os.path.join(root, "memory/memory.usage_in_bytes"), f"{usage}\n")
    _write(os.path.join(root, "memory/memory.stat"),
           f"cache {cache}\nrss {usage - cache}\ntotal_cache {cache}\n")


def test_v2_counts_page_cache_as_free(tmp_path):
    root = str(tmp_path / "v2")
    _v2(root, limit=125 * GB, current=120 * GB, file_cache=118 * GB)
    free, detail = container_free_bytes(root)
    assert detail["cgroup"] == "v2"
    # 120 GiB "used" is almost all reclaimable cache: 2 GiB really in use.
    assert detail["in_use"] == 2 * GB
    assert free == 123 * GB, f"got {free / GB:.1f} GiB"


def test_v1_is_read_too(tmp_path):
    """The layout RunPod pods and QNAP Container Station actually present."""
    root = str(tmp_path / "v1")
    _v1(root, limit=125 * GB, usage=120 * GB, cache=118 * GB)
    free, detail = container_free_bytes(root)
    assert detail["cgroup"] == "v1"
    assert detail["in_use"] == 2 * GB
    assert free == 123 * GB


def test_v2_wins_when_both_are_present(tmp_path):
    root = str(tmp_path / "both")
    _v2(root, limit=100 * GB, current=10 * GB, file_cache=0)
    _v1(root, limit=999 * GB, usage=0, cache=0)
    free, detail = container_free_bytes(root)
    assert detail["cgroup"] == "v2" and free == 90 * GB


def test_unlimited_returns_none_not_a_number(tmp_path):
    """`None` means "could not measure" and must not be mistaken for zero free.

    v2 spells it `max`; v1 spells it a near-2**63 sentinel. Treating that integer
    as a real limit would report ~8 exabytes free.
    """
    r2 = str(tmp_path / "u2")
    _v2(r2, limit=0, current=0, file_cache=0)
    _write(os.path.join(r2, "memory.max"), "max\n")
    free, detail = container_free_bytes(r2)
    assert free is None and "unlimited" in str(detail["limit"])

    r1 = str(tmp_path / "u1")
    _v1(r1, limit=9223372036854771712, usage=1 * GB, cache=0)
    free, detail = container_free_bytes(r1)
    assert free is None, f"v1 unlimited sentinel reported {free}"
    assert detail["note"] == "unlimited sentinel"


def test_absent_controller_reports_why(tmp_path):
    free, detail = container_free_bytes(str(tmp_path / "nothing"))
    assert free is None
    assert "no cgroup" in detail["error"]


def test_never_returns_a_negative(tmp_path):
    """Usage above the limit happens transiently under pressure; clamp, do not sign-flip."""
    root = str(tmp_path / "over")
    _v2(root, limit=10 * GB, current=12 * GB, file_cache=0)
    free, _ = container_free_bytes(root)
    assert free == 0


def test_the_real_host_is_readable_or_says_why():
    """A smoke check on whatever actually runs this — no assertion on the value.

    The CPU gate runs in a cgroup v1 container, so this exercises the v1 branch
    against a real kernel rather than a fixture.
    """
    free, detail = container_free_bytes()
    assert isinstance(detail, dict) and detail, "must always explain itself"
    if free is None:
        assert "error" in detail or "limit" in detail, f"unexplained None: {detail}"
    else:
        assert free >= 0 and "cgroup" in detail
