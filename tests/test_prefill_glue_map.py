# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The prefill-glue design note (bench/prefill-glue/DESIGN.md) is exactly glue_map.py's output over P119's committed
profile, and the glue map closes: every elementwise, routing and device-copy kernel is assigned, and the groups sum to
the total."""
import importlib.util
import pathlib

LANE = pathlib.Path(__file__).resolve().parents[1] / "bench" / "prefill-glue"


def _mod():
    spec = importlib.util.spec_from_file_location("glue_map", LANE / "glue_map.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_note_is_the_scripts_output():
    assert _mod().main(["--check"]) == 0


def test_the_glue_map_closes():
    g = _mod().gather()
    assert not g["unassigned"], g["unassigned"]
    assert abs(sum(g["ms"].values()) - g["scope"]) < 1e-6
    assert abs(g["scope"] - (g["elementwise"] + g["route"] + g["dtod"])) < 1e-3      # P119's three classes, in full


def test_a_shared_kernel_is_split_in_full():
    m = _mod()
    for _subs, _count, frac, _dagger in m.RULES:
        assert abs(sum(frac.values()) - 1) < 1e-9 and set(frac) <= set(m.GROUPS)
