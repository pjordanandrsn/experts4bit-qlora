"""The SC2f/SC2h/SC2i kit hold (SC2f Amendment 5, SC2h Amendment 3, SC2i Amendment 2; bench/sc2/sc2_kit_hold.py).

The check reads REGISTERED commits only: the base hold (`1a25b209`) and the new hold (e4b#2041's merge). It never reads the
live tree, so a later kit change on main cannot turn this test red. The launch-time HOLD gate (launch commit against the
hold) runs on the controller before the create and is recorded with the run.
"""
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench" / "sc2"))
import sc2_kit_hold as kh  # noqa: E402


def test_self_test_refuses_each_gate_with_its_evidence_removed():
    r = subprocess.run([sys.executable, str(REPO / "bench" / "sc2" / "sc2_kit_hold.py"), "--self-test"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SC2_KIT_HOLD self-test PASS" in r.stdout


def test_the_hold_is_registered():
    assert kh.HOLD and len(kh.HOLD) == 40, "the amendments register e4b#2041's merge commit as HOLD"


def test_the_registered_delta_is_exactly_the_declared_set():
    rec = kh.check(kh.Git(str(REPO)), kh.HOLD, at=kh.HOLD)
    assert rec["verdict"] == "OK", rec["why"]
    kinds = [x["kind"] for x in rec["run_hunks"]]
    assert kinds.count("brotli") == 1 and kinds.count("fetch-retry") == 4
    assert kinds.count("q-line") == 2 and kinds.count("q-wiring") == kh.Q_REPLACED
    assert rec["changed"] == ["sc1_run.sh"] and rec["added"] == sorted(kh.NEW) and not rec["removed"]
    assert rec["inert_hits"] and all(x["allowed"] for x in rec["inert_hits"])


def test_the_check_itself_is_not_staged():
    """A controller-side tool: staging it would change the kit it holds."""
    pin = (REPO / "bench" / "sc1" / "staged.sha256").read_text().split()
    assert "sc2_kit_hold.py" not in pin
