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

import logging
import re

torch = pytest.importorskip("torch")
pytest.importorskip("torch._dynamo")

_SRC = (pathlib.Path(__file__).resolve().parents[1]
        / "bench" / "hybrid-g9" / "step_decomp.py").read_text()


def _load():
    """Lift the helper out of the harness; it needs no CUDA or model."""
    # start at the module-level helpers, not the function -- the site
    # matcher and its regex live above _graph_break_census and lifting
    # the function alone left them undefined at runtime
    start = _SRC.index("_BREAK_FRAME = re.compile(")
    end = _SRC.index("def _mech_reset(")
    # seed the namespace the lifted block expects; exec() gets a fresh
    # globals dict, so imports in THIS file do not reach it
    ns = {"re": re, "logging": logging}
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


def test_the_REAL_MODULE_imports():
    """The gap that let a NameError reach CI.

    Every other test here lifts the census block into an exec
    namespace that this file SEEDS with `re` and `logging`. That made
    them pass while the real module was broken: `_BREAK_FRAME =
    re.compile(...)` sat at module scope with no `import re`, and CI
    caught it, not the suite.

    A test that constructs its own environment cannot vouch for the
    module's own. This imports the file for real.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "step_decomp_import_check",
        pathlib.Path(__file__).resolve().parents[1]
        / "bench" / "hybrid-g9" / "step_decomp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # raises on a missing import
    assert getattr(mod, "_BREAK_FRAME", None) is not None
    assert getattr(mod, "_BREAK_HEAD", None) is not None
    # _match_site is GONE by design: it paired counter reasons to
    # separately-captured sites by word overlap, and each dynamo
    # record already carries its reason and its stack together
    assert not hasattr(mod, "_match_site")
    assert callable(getattr(mod, "_graph_break_census", None))


def test_it_names_breaks_from_the_REAL_compile_not_a_wrapper():
    """The redesign: trace the compile, not a wrapper around it.

    The explain()-based version named exactly one break on the box --
    the paged-attention disable boundary -- IDENTICALLY in both cells,
    with the MoE tier absent. It traced the outer step and never
    descended into layer bodies already wrapped in torch.compile,
    which is where the work lives. Two counter reasons for a step
    with 48 MoE layers was the tell.
    """
    census = _load()
    out = census(_breaking_step())
    assert out["error"] is None, out["error"]
    assert out["source"] == "dynamo-log/real-compile"
    assert out["breaks"], "the real compile breaks; it must be named"
    b = out["breaks"][0]
    assert b["line"] > 0 and b["file"] != "<no-frame>", b
    assert b["func"] == "step", b
    assert "item" in b["reason"].lower() or "graph break" in b["reason"].lower()


def test_each_record_carries_its_OWN_reason_and_site():
    """No cross-matching. The previous version paired counter reasons
    to separately-captured sites by word overlap; with 1016 captured
    sites every reason collapsed onto one location."""
    census = _load()
    out = census(_breaking_step())
    for b in out["breaks"]:
        assert b["reason"], b
        # a site and a reason from the SAME record, or an explicit
        # no-frame marker -- never a guessed pairing
        assert (b["file"] != "<no-frame>") == (b["line"] > 0), b


def test_phase_is_recorded_per_break_and_is_one_of_two():
    census = _load()
    out = census(_breaking_step())
    assert all(b["phase"] in ("trace", "step") for b in out["breaks"])
    assert "run FOR REAL" in out["phase_basis"]


def test_a_clean_step_yields_no_breaks():
    """k13_verdict REFUSES an empty census, so empty must stay
    reachable or that refusal could never fire."""
    census = _load()
    out = census(_clean_step())
    assert out["error"] is None
    assert out["breaks"] == [], out["breaks"]


def test_identical_records_aggregate_rather_than_repeat():
    census = _load()
    out = census(_breaking_step())
    keys = [(b["file"], b["line"], b["func"], b["reason"]) for b in out["breaks"]]
    assert len(keys) == len(set(keys)), keys
    assert any(b["count"] >= 1 for b in out["breaks"])


def _step_with_precompiled_inner():
    """The harness shape: layer bodies ALREADY wrapped in
    torch.compile, called by an outer step. This is what the toy
    fixtures lack, and it is the only place the redesign differs."""
    x = torch.zeros(4)

    @torch.compile(dynamic=False)
    def inner_layer(t):
        u = t + 1
        return u * int(u.sum().item())     # the break, INSIDE the layer

    def step():
        return inner_layer(x) + 1
    return step, "inner_layer"


def test_it_sees_inside_already_compiled_layer_bodies():
    """A smoke test, NOT a pin on the redesign -- read the caveat.

    The census must name breaks that live inside layer bodies already
    wrapped in torch.compile, because that is where the MoE work is.
    This checks it does.

    What it does NOT do is distinguish the redesign from the
    explain()-wrapper it replaced. Swapping the real compile back to
    `explain(one_step)()` leaves EVERY test in this file green,
    including this one: on CPU both paths surface the inner break.
    The difference only appeared on the box -- 2 counter reasons for
    a step with 48 MoE layers, and the same single outer disable
    boundary named in both cells.

    So the redesign rests on box evidence and on being strictly
    closer to what K12 measured (observing the real compile rather
    than a wrapper around it), NOT on a passing test here. Saying so
    is better than a green check that proves something else.
    """
    census = _load()
    step, inner_name = _step_with_precompiled_inner()
    out = census(step)
    assert out["error"] is None, out["error"]
    funcs = {b["func"] for b in out["breaks"]}
    assert inner_name in funcs, (
        f"census named {funcs or 'nothing'}; a break inside a "
        "pre-compiled layer must be visible")


def test_phase_field_is_WELL_FORMED_and_nothing_more():
    """Phase is recorded, not verified — and this test says so.

    I tried five fixtures to pin the trace-vs-step classification and
    every one passed under a mutation that broke the logic it named:
      - assertions inside `if count > count_at_compile`, a branch that
        never runs (a recompile logs under a NEW key rather than
        incrementing an old one: `existing-and-grew: 0`)
      - an `expected` computed from the same fields the code uses,
        which therefore cannot disagree with it
    A green check that survives breaking the thing it tests is worse
    than no check, because it reads as coverage.

    So this asserts only what it can: the field exists, and its value
    is one of the three legal ones. PREREG-k13 RECORDS phase and never
    scores it, and `k13_verdict` refuses a blank one — so a
    misclassification cannot silently change a verdict. That is the
    actual protection; this test is not.
    """
    census = _load()
    out = census(_breaking_step())
    assert out["error"] is None, out["error"]
    assert out["breaks"], "fixture must produce breaks"
    for b in out["breaks"]:
        assert "phase" in b and "count_at_compile" in b, b
        assert b["phase"] in ("trace", "step", None), b["phase"]

def test_a_phase_probe_failure_KEEPS_the_captured_census():
    """The shared-try bug (review, e4b#292), which I had fixed in the
    previous design and reintroduced in the rewrite.

    An exception after the real compile must not discard a census
    that was already captured -- k13_verdict would then see an empty
    instrument and blame the wrong thing.
    """
    census = _load()
    state = {"n": 0}

    # the inner MUST be compiled: this census observes real
    # compilation, so an uncompiled callable produces no dynamo
    # tracing and therefore no breaks to preserve. (The old
    # explain()-based design compiled whatever it was handed, which
    # is why this fixture needed changing with the redesign.)
    @torch.compile(dynamic=False)
    def inner(t_):
        u = t_ + 1
        return u * int(u.sum().item())

    def flaky():
        state["n"] += 1
        if state["n"] > 1:              # survives the compile, dies in phase
            raise RuntimeError("phase boom")
        return inner(torch.zeros(4))
    out = census(flaky)
    assert out["error"] is None, "the compile itself succeeded"
    assert out["breaks"], "the captured breaks must survive"
    assert out["phase_error"] and "phase boom" in out["phase_error"]
    assert all(b["phase"] is None for b in out["breaks"])
