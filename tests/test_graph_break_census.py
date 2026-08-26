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


def test_phase_CAN_return_trace_not_only_step():
    """The bug review caught (e4b#290): a probe with one outcome.

    The first version cleared the counters and re-ran the step right
    after `explain()` -- but explain() RESETS dynamo, so that re-run
    recompiled and re-counted every break. It could only ever say
    "step". A measurement no input can make say "trace" is not a
    measurement.

    A `.item()` break is re-traced on each compile but does NOT
    re-fire once the artifact is warm, so with the warm call in place
    this must come back "trace".
    """
    census = _load()
    out = census(_breaking_step())
    assert out["error"] is None, out["error"]
    assert out["breaks"][0]["phase"] == "trace", (
        "with the warm call absorbing the post-explain recompile, a "
        "break that does not re-fire must read as trace; 'step' here "
        "means the probe is counting the recompile again")
    assert out["counter_total_recompiled"] == 0, out


def test_phase_basis_records_the_warm_call():
    census = _load()
    out = census(_breaking_step())
    assert "WARMED" in out["phase_basis"], out["phase_basis"]
    assert out["counter_total_recompiled"] is not None


def test_a_phase_probe_failure_keeps_the_census():
    """Losing a good census to a failure in the phase probe is the
    worse outcome. The verdict REFUSES an unknown phase -- correctly --
    but the receipt must say why rather than the arm aborting."""
    census = _load()
    state = {"n": 0}

    def flaky():
        # the break is INLINE here: a pre-compiled inner is opaque to
        # explain(), which found no breaks at all and made this test
        # fail for a reason unrelated to what it checks
        state["n"] += 1
        if state["n"] > 1:          # survives explain, dies in the probe
            raise RuntimeError("probe boom")
        x = torch.zeros(4)
        y = x + 1
        return y * int(y.sum().item())
    out = census(flaky)
    assert out["error"] is None, "the census itself succeeded"
    assert out["breaks"], "the named breaks must survive"
    assert out["phase_error"] and "probe boom" in out["phase_error"]
    assert out["breaks"][0]["phase"] is None


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


def test_it_REFUSES_the_flag_combinations_that_would_silently_skip_it():
    """The K8 failure class: an arm that runs, exits 0, measures nothing.

    The ppl block returns before the census site, so
    `--graph-break-census --ppl-steps` would write a perplexity report
    and silently skip the census. Likewise the census lives on the b1d
    stage, and censusing an untraced region reports zero breaks for a
    reason unrelated to the MoE tier -- which `k13_verdict` would then
    REFUSE as an empty census, blaming the wrong thing.
    """
    i = _SRC.index('if getattr(a, "graph_break_census", False):',
                   _SRC.index("is a B=1 instrument"))
    # anchored on the end of the guard block, not a byte count -- a
    # fixed window has silently truncated two tests in this repo
    # already, both times reporting the wrong thing
    guard = _SRC[i:_SRC.index("\n    if a.b1d_loop:", i)]
    for needle, why in (
            ("a.ppl_steps", "ppl would return first and skip the census"),
            ("a.b1d_loop", "the census lives on the b1d stage"),
            ("a.compile_layers", "an untraced region has no breaks to find"),
            ("a.batch > 1", "batch>1 routes to BV3 and skips the census "
                            "-- and --batch DEFAULTS to 4")):
        assert needle in guard, f"unguarded: {why}"
    assert guard.count("raise SystemExit") >= 4, guard[:200]
