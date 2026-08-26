# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""PREREG-k13's census instrument, exercised for real without a GPU.

K12 proved the MoE tier cannot be traced as one graph but not WHICH
host read breaks it: its log carries a frame stack, and a stack is
not an attribution. So the instrument's whole value is that dynamo
NAMES the site. These tests drive it against functions whose break
site is known by construction, and check it reports that site --
rather than checking the source contains the right call, which is
what a stack-reading version would have passed too.
"""
import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch._dynamo")

_SRC = (pathlib.Path(__file__).resolve().parents[1]
        / "bench" / "hybrid-g9" / "step_decomp.py").read_text()


def _load():
    """Lift the helper out of the harness; it needs no CUDA or model."""
    start = _SRC.index("def _graph_break_census(")
    end = _SRC.index("def _mech_reset(")
    ns = {}
    exec(compile(_SRC[start:end], "census", "exec"), ns)
    return ns["_graph_break_census"]


def _breaking_step():
    x = torch.zeros(4)

    @torch.compile(dynamic=False)
    def step():
        y = x + 1
        n = int(y.sum().item())     # <-- the break, line is asserted below
        return y * n
    return step


def _clean_step():
    x = torch.zeros(4)

    @torch.compile(dynamic=False)
    def step():
        return (x + 1) * 2
    return step


def test_it_names_the_break_site_dynamo_reported():
    census = _load()
    out = census(_breaking_step())
    assert out["error"] is None, out["error"]
    assert out["breaks"], "a step with an .item() must yield a break"
    b = out["breaks"][0]
    # named by dynamo: this file, the enclosing function, a real line
    assert b["file"].endswith("test_graph_break_census.py"), b
    assert b["func"] == "step", b
    assert b["line"] > 0, b
    assert "item()" in b["reason"], b["reason"]


def test_a_clean_step_yields_an_EMPTY_census():
    """The k13 verdict REFUSES an empty census, so empty must be
    reachable -- otherwise that refusal could never fire."""
    census = _load()
    out = census(_clean_step())
    assert out["error"] is None, out["error"]
    assert out["breaks"] == [], out["breaks"]


def test_phase_is_measured_and_is_one_of_two_values():
    census = _load()
    out = census(_breaking_step())
    assert out["breaks"][0]["phase"] in ("trace", "step")
    assert "re-ran the step after compilation" in out["phase_basis"]
    # the empirical basis must actually be recorded, not asserted
    assert out["counter_total_recompiled"] is not None


def test_identical_sites_MERGE_and_sum_rather_than_rank_as_two():
    """Two entries for one site would rank as two smaller breaks and
    the top-break pick would be wrong."""
    census = _load()
    out = census(_breaking_step())
    sites = [(b["file"], b["line"], b["func"], b["reason"])
             for b in out["breaks"]]
    assert len(sites) == len(set(sites)), sites


def test_a_failing_explain_returns_a_STRUCTURED_error():
    """A census that raised would abort the arm; the verdict needs to
    see the failure instead."""
    census = _load()

    def boom():
        raise RuntimeError("nope")
    out = census(boom)
    assert out["error"] and "nope" in out["error"], out
    assert out["breaks"] == []


def test_the_flag_is_registered_and_defaults_off():
    import re
    m = re.search(r'add_argument\("--graph-break-census"[^)]*\)', _SRC, re.S)
    assert m, "flag not registered"
    assert "store_true" in m.group(0), "must default OFF"
