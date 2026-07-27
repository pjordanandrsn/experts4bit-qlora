# SPDX-License-Identifier: Apache-2.0
"""`gbps` must be wall-based, and must not be able to exceed the link.

Finding #52: `routed_gbps` reported 22.83 GB/s on a link independently probed at
12.65-18.46 GB/s -- an implied transfer rate faster than the hardware. The cause
was the denominator: bytes over the summed per-stage *copy window*, which does
not bound the transfer. The bytes were always right (7.984 GB/token, matching an
independent model); only the divisor was wrong.

That number is load-bearing -- R4 of PREREG-routed-residual, and #22's whole
residual analysis, are stated against it -- so it gets a test.

Source-contract tests: they read the file, so they run on CPU CI with no
torch/GPU, which is where a refactor would quietly restore the old divisor.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "experts4bit_qlora" / "offload.py"


def _report_src():
    tree = ast.parse(_SRC.read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "offload_stats_report":
            return ast.unparse(n)
    raise AssertionError("offload_stats_report not found")


def test_gbps_divides_by_wall_not_the_copy_window():
    src = _report_src()
    assert re.search(r"gbps\s*=\s*gb\s*/\s*wall_s", src), (
        "`gbps` is no longer bytes-over-wall-time. Dividing by the summed "
        "per-stage copy window produced a rate ABOVE the physical link "
        "(22.83 GB/s on a 12.65-18.46 GB/s link, finding #52)."
    )


def test_the_window_rate_is_kept_but_renamed():
    src = _report_src()
    assert "gbps_copy_window" in src, (
        "the copy-window rate should stay available for diagnosis -- but under a "
        "name that does not read as transfer efficiency."
    )


def test_window_rate_carries_its_warning():
    """A caveat only in a commit message protects nobody reading the code."""
    body = _SRC.read_text()
    assert "does not bound the transfer" in body or "does **not** bound the transfer" in body, (
        "the gbps_copy_window caveat was removed; without it the next reader "
        "will quote a faster-than-light number as a result."
    )


def test_stats_stamp_a_window_start():
    body = _SRC.read_text()
    assert "self.t0 = time.perf_counter()" in body, (
        "wall time needs a well-defined window start; reset_offload_stats() "
        "drops the stats object so the next stage restarts the clock."
    )
    assert re.search(r"wall_s\s*=\s*max\(time\.perf_counter\(\)\s*-\s*stats\.t0", body), (
        "wall_s is no longer measured from the stats window start."
    )
