"""PGX2 Amendment 2 at $0 (bench/pgx/PREREG-pgx2.md): a part record's findings verify in any order.

- the record first: box 1 (pgx2-4090-r2-2-1) wrote its part record on the box, and the controller re-derived the same
  record from the fetched read/. Both are sha-pinned in tests/fixtures/pgx2_box1_part/. They hold the same findings in
  different orders, so the old list-for-list comparison refuses them, and the amended comparison verifies them. Every
  other field is equal exactly;
- a synthetic box with four token-invariance findings, reduced box-side under PYTHONHASHSEED 0..3, each in its own process:
  the four records do not all list their findings in one order, and every one of them verifies when classed under each
  seed;
- what must still refuse: a finding whose content differs, one missing, one duplicated (a set comparison would pass
  it), findings that are not a list, and any other field that differs, is added or is missing;
- the page carries the amendment.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PGX = REPO / "bench" / "pgx"
R1 = PGX / "receipts" / "pgx-shake-r1"
FIX = REPO / "tests" / "fixtures" / "pgx2_box1_part"
PAGE = PGX / "PREREG-pgx2.md"
sys.path.insert(0, str(PGX))

import pgx_box_end as be  # noqa: E402
import pgx_schedule as sch  # noqa: E402

PINS = {"part_verdict.json": "c901adedccd25dcf9aba2a517ed52a5f55eb4fac490647ecc4e49decf20c64f7",
        "rederived_seed0.json": "2859e55582e4e771ec7ae007f0957b73798d5144eab5e7387b8115f368f4f1cb"}
RID = "pgx2-4090-1-2-1"
NONCE = "0123456789abcdef"
SEEDS = ("0", "1", "2", "3")
FLIPPED = ("S4", "S8", "V8", "E-hyb")       # the arms whose P2 is made to differ from P1: four findings


def _fixture(name):
    b = (FIX / name).read_bytes()
    assert hashlib.sha256(b).hexdigest() == PINS[name], name
    return json.loads(b)


# ---------------------------------------------------------------- the real record
def test_box_1s_real_record_refused_by_the_old_comparison_verifies_under_the_amendment():
    part, want = _fixture("part_verdict.json"), _fixture("rederived_seed0.json")
    assert part != want                                              # the old comparison: a strike
    assert be.differs(part, want) == []                              # the amended one: it verifies
    assert part["findings"] != want["findings"]
    assert be.canonical_findings(part["findings"]) == be.canonical_findings(want["findings"])
    assert {k: part[k] for k in part if k != "findings"} == {k: want[k] for k in want if k != "findings"}
    assert part["verdict"] == "PART_FAULTS" and part["draws"] == [1, 2] and part["void_lifetimes"] == [["S4", 4, 1]]
    assert len(part["findings"]) == 6
    assert all(f["kind"] == "token_invariance" and f["arm"] == "S4" and f["rules"] is False for f in part["findings"])


# ---------------------------------------------------------------- a synthetic box, reduced under four hash seeds
def _box(root: Path, seed: str) -> Path:
    """A fetched PGX2 box 1 directory whose read/ is r1's real 8 GB lifetimes with P2 made to differ from P1 at four arms,
    and whose part record the box's own reducer wrote in a fresh process under PYTHONHASHSEED=seed."""
    d = root / seed / RID / "pgx"
    d.mkdir(parents=True)
    for name, text in (("STARTED", ""), ("RUN_ID", RID + "\n"), ("PGX_RUN_NONCE", NONCE + "\n"),
                       (f"PGX_EXIT_CODE.{NONCE}", "27\n")):
        (d / name).write_text(text)
    (d / "box_schedule.txt").write_text("\n".join(be.schedule_lines("2-1")) + "\n")
    man = json.loads((PGX / "prompts" / "manifest.json").read_text())
    reg = {"profile_sha": hashlib.sha256((PGX / "fit_profile.jsonl").read_bytes()).hexdigest(),
           "budget_bytes": {str(b): b * 10**9 for b in (4, 8, 12)},
           "sets": {k: v["prompts_sha"] for k, v in man["sets"].items()}}
    (d / "registration.json").write_text(json.dumps(reg))
    shutil.copytree(R1 / "shake", d / "read")
    for arm in FLIPPED:
        p = d / "read" / arm / "8" / "d1" / "P2.json"
        rec = json.loads(p.read_text())
        rec["requests"][0]["text_sha"] = "0" * 64
        p.write_text(json.dumps(rec))
    r = subprocess.run([sys.executable, str(PGX / "pgx_reduce.py"), "--work", str(d / "read"), "--registration",
                        str(d / "registration.json"), "--mode", "part", "--draws", "1", "2", "--out",
                        str(d / "part_verdict.json")], env={**os.environ, "PYTHONHASHSEED": seed},
                       capture_output=True, text=True)
    assert r.returncode == 27, r.stdout + r.stderr
    return d


def _order(d: Path) -> tuple:
    return tuple((f["arm"], f["budget"], f["draw"]) for f in json.loads((d / "part_verdict.json").read_text())["findings"])


@pytest.fixture(scope="module")
def boxes(tmp_path_factory):
    root = tmp_path_factory.mktemp("seeds")
    return {s: _box(root, s) for s in SEEDS}


def test_the_seeds_do_list_the_findings_in_different_orders(boxes):
    orders = {s: _order(d) for s, d in boxes.items()}
    assert all(sorted(o) == sorted((a, 8, 1) for a in FLIPPED) for o in orders.values()), orders
    assert len(set(orders.values())) > 1, orders                    # else this test would not exercise the amendment


@pytest.mark.parametrize("box_seed", SEEDS)
@pytest.mark.parametrize("ctl_seed", SEEDS)
def test_every_seeds_record_verifies_under_every_seed(boxes, box_seed, ctl_seed):
    d = boxes[box_seed]
    r = subprocess.run([sys.executable, str(PGX / "pgx_box_end.py"), str(d), "--run-id", RID, "--rc", "27"],
                       env={**os.environ, "PYTHONHASHSEED": ctl_seed}, capture_output=True, text=True)
    assert r.returncode == be.EXIT[be.PART_FAULTS], r.stdout + r.stderr
    assert r.stdout.startswith(f"BOX_END PART_FAULTS {RID} rc 27")


# ---------------------------------------------------------------- what must still refuse
def _edit(fn):
    def go(rec):
        fn(rec)
        return rec
    return go


def _first(key, value):
    return lambda r: r["findings"][0].update({key: value})


REFUSE = {
    "a finding's differing_requests differs": (_first("differing_requests", 99), "findings"),
    "a finding's rules differs": (lambda r: r["findings"][0].update(rules=not r["findings"][0]["rules"]), "findings"),
    "a finding's arm differs": (_first("arm", "S8x"), "findings"),
    "one finding missing": (lambda r: r["findings"].pop(), "findings"),
    "one finding duplicated (a set comparison would pass it)": (lambda r: r["findings"].append(dict(r["findings"][0])),
                                                                "findings"),
    "one finding added": (lambda r: r["findings"].append({"kind": "token_invariance", "arm": "S4", "budget": 12,
                                                          "draw": 2, "differing_requests": 1, "rules": False}),
                          "findings"),
    "findings is not a list": (lambda r: r.update(findings={str(i): f for i, f in enumerate(r["findings"])}), "findings"),
    "no findings at all": (lambda r: r.update(findings=[]), "findings"),
    "the verdict differs": (lambda r: r.update(verdict="PART_OK"), "verdict"),
    "a VOID lifetime hidden": (lambda r: r["void_lifetimes"].pop(), "void_lifetimes"),
    "a faulted row hidden": (lambda r: r["faults"].pop(sorted(r["faults"])[0]), "faults"),
    "a fault reworded": (lambda r: r["faults"][sorted(r["faults"])[0]].__setitem__(0, "x"), "faults"),
    "the draws differ": (lambda r: r.update(draws=[1]), "draws"),
    "a field added": (lambda r: r.update(extra=None), "extra"),
    "a field missing": (lambda r: r.pop("faults"), "faults"),
}


@pytest.mark.parametrize("name", sorted(REFUSE))
def test_a_record_that_differs_other_than_in_order_is_a_strike(boxes, name):
    d = boxes["0"]
    path = d / "part_verdict.json"
    good = path.read_text()
    try:
        assert be.classify(str(d), RID, 27)["class"] == be.PART_FAULTS          # the unmutated box is (b)
        fn, field = REFUSE[name]
        path.write_text(json.dumps(_edit(fn)(json.loads(good))))
        res = be.classify(str(d), RID, 27)
        assert res["class"] == be.STRIKE, (name, res)
        assert any(w.startswith("part_verdict.json does not verify") and field in w for w in res["why"]), res["why"]
    finally:
        path.write_text(good)


def test_the_same_record_reversed_verifies(boxes):
    d = boxes["0"]
    path = d / "part_verdict.json"
    good = path.read_text()
    try:
        rec = json.loads(good)
        rec["findings"].reverse()
        path.write_text(json.dumps(rec))
        assert be.classify(str(d), RID, 27)["class"] == be.PART_FAULTS
    finally:
        path.write_text(good)


# ---------------------------------------------------------------- the comparison itself
A = {"kind": "token_invariance", "arm": "S4", "budget": 4, "draw": 1, "differing_requests": 7, "rules": False}
B = {**A, "differing_requests": 5}          # equal to A in every field but one


def _rec(findings):
    return {"verdict": "PART_FAULTS", "draws": [1, 2], "findings": findings, "void_lifetimes": [], "faults": {}}


@pytest.mark.parametrize("got,want,ok", [
    ([A, B], [B, A], True),                 # entries tied on every field but one, in either order
    ([A, B], [A, B], True),
    ([A], [B], False),                      # a form that drops a field would pass this
    ([A, A], [A, B], False),
    ([A, A], [A], False),                   # a set comparison would pass this
    ([], [A], False),
    ([{**A, "x": 1}], [A], False),
    ([dict(reversed(list(A.items())))], [A], True),                # key order inside an entry carries no meaning
])
def test_differs_compares_findings_as_a_multiset(got, want, ok):
    assert (be.differs(_rec(got), _rec(want)) == []) is ok
    assert be.differs(_rec(got), _rec(want)) in ([], ["findings"])


def test_differs_on_a_record_that_is_not_an_object():
    assert be.differs([1], _rec([A])) == ["<not an object>"]


def test_the_reducer_still_lists_findings_by_set_order():
    """The amendment changes the controller only: the staged reducer is the launch commit's byte for byte."""
    pinned = {ln.split()[1]: ln.split()[0] for ln in (PGX / "staged-pgx.sha256").read_text().splitlines() if ln.strip()}
    assert hashlib.sha256((PGX / "pgx_reduce.py").read_bytes()).hexdigest() == pinned["pgx_reduce.py"]
    assert pinned["pgx_reduce.py"] == "0b0adc4fc8092e0436d4d57513a6abb2a1f3e12ceeaf891763687fc6052aa835"
    assert "pgx_box_end.py" not in pinned


# ---------------------------------------------------------------- the page
def test_the_page_registers_the_amendment():
    text = PAGE.read_text(encoding="utf-8")
    heading = "## Amendment 2 (2026-11-08): a part record's findings verify in any order, registered before box 2"
    assert text.count(heading) == 1
    amend = " ".join(text[text.index(heading):].split())
    for term in ("`PYTHONHASHSEED`", "as a multiset", "Every other field is compared exactly", "`pgx2-4090-r2-2-1`",
                 "`PART_FAULTS` (b)", "not a strike", "VOID", "No box-side byte changes", "`pgx2-4090-1-2-2`",
                 "a duplicated"):
        assert term in amend, term
    assert sch.parse_run_id("pgx2-4090-1-2-2")[:2] == (2, 2)
