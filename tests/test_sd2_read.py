# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane SD2's read (``bench/sd2/PREREG-sd2.md``, Amendment 3), its CPU checks: the verify pass against a T == 1 pass on
a toy target (the maintainer's condition, before the box), Q's buckets and statistics against P110's and P115's, and
SD1's constants against their receipt. The read's verdicts are ``sd2_reduce.py``'s self-test (``tests/test_sd2.py``)."""
import importlib.util
import json
import pathlib
import random
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "sd2"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("P, C, k", [(9, 13, 1), (9, 13, 2), (9, 13, 3), (5, 7, 2), (512, 128, 1), (512, 128, 2)])
def test_the_verify_pass_equals_a_t1_pass_on_a_toy_target(P, C, k):
    """On a causal toy target whose arithmetic is row-count independent, the verify pass's per-position log-probs equal
    a T == 1 teacher-forced pass, and the construction's own mutants (row i reading one short; the steps fed one token
    late) do not."""
    pytest.importorskip("torch")
    box = _load("sd2_box_t", LANE / "sd2_box.py")
    assert box.verify_assembly_check(P, C, k)
    assert not box.verify_assembly_check(P, C, k, stagger=-1)
    assert not box.verify_assembly_check(P, C, k, plan_shift=1)
    plan = box.verify_plan(P, C, k)
    scored = [b + 1 + i for b, use in plan for i in range(use)]
    assert scored == list(range(P + 1, P + C)), "every position after P scored once, in order"
    assert all(use == k + 1 for _b, use in plan[:-1]) and 1 <= plan[-1][1] <= k + 1
    assert all(len(box.step_ids(list(range(P + C)), b, k)) == k + 1 for b, _u in plan), "every step feeds k + 1 rows"


def test_q_buckets_are_p110s():
    """Q's ON runner pads to P110's buckets, the R arm's (read from P110's source: it imports its staged siblings)."""
    box = _load("sd2_box_b", LANE / "sd2_box.py")
    src = (REPO / "bench" / "p110" / "p110_box.py").read_text(encoding="utf-8")
    want = tuple(int(x) for x in re.search(r"^BUCKETS = \(([^)]*)\)", src, re.M).group(1).split(","))
    assert tuple(box.Q_BUCKETS) == want


def test_q_statistics_and_bar_are_p115s():
    """The read's Q statistics are P115's ``_stats`` / ``_passes`` (P110's bar), summed with math.fsum."""
    red = _load("sd2_reduce_q", LANE / "sd2_reduce.py")
    p115 = _load("p115_reduce_q", REPO / "bench" / "p115" / "p115_reduce.py")
    rng = random.Random(1313)
    for _ in range(20):
        ref = {w: 2.0 + rng.random() for w in range(48)}
        per = {"X": [{"window": w, "nll": ref[w] + rng.gauss(0, 0.01), "argmax_agree": rng.random(), "kl": rng.random() / 100}
                     for w in range(48)]}
        a, b = red._q_stats(per, "X", ref), p115._stats(per, "X", ref)
        for key in ("n", "bias", "spread", "max_abs", "mean_kl", "argmax_agree"):
            assert abs(a[key] - b[key]) < 1e-12, key
        fl = {"B_floor": rng.random() / 100, "S_floor": rng.random() / 100}
        assert red._q_passes(a, fl) == p115._passes(b, fl)
    assert (red.TOL, red.SPREAD_X, red.SPREAD_MIN) == (p115.TOL, p115.SPREAD_X, p115.SPREAD_MIN)


def test_sd1s_constants_are_its_receipt():
    red = _load("sd2_reduce_c", LANE / "sd2_reduce.py")
    v = json.loads((REPO / "bench" / "sd1" / "receipts" / "sd1-5090-2" / "verdict.json").read_text(encoding="utf-8"))
    for w in red.WORKLOADS:
        for k in (1, 2, 3):
            row = v["routes"]["eagle3"][w][str(k)]
            assert red.TAU_SD1[w][k] == row["tau"] and red.S_SD1[w][k] == row["S_independent"], (w, k)
    run = (LANE / "sd2_run.sh").read_text(encoding="utf-8")
    assert f"HEAD_SHA256={red.HEAD_SHA256};" in run
