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
    rec = kh.check(kh.Git(str(REPO)), kh.HOLD, at=kh.HOLD, e4b_sha=kh.STACK)
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


def test_the_check_refuses_a_launch_without_the_stack():
    assert kh.check(kh.Git(str(REPO)), kh.HOLD, at=kh.HOLD)["verdict"] == "REFUSE"
    assert kh.check(kh.Git(str(REPO)), kh.HOLD, at=kh.HOLD, e4b_sha=kh.HOLD)["verdict"] == "REFUSE"


FIX = REPO / "tests" / "fixtures" / "sc2_verdict_stack"   # a real fetched run's records (sc2j-r3: box Q at 7b8f1f69)


def test_verdict_stack_on_a_real_run(tmp_path):
    """The real records name 7b8f1f69, so against the SC2f/h/i stack they read VOID; the same bytes with the commit
    replaced read OK, and each record altered alone reads VOID."""
    v, s = (FIX / "versions.txt").read_text(), (FIX / "summary.txt").read_text()
    assert kh.HOLD in v and kh.HOLD in s
    (tmp_path / "versions.txt").write_text(v)
    (tmp_path / "summary.txt").write_text(s)
    assert kh.verdict_stack(tmp_path)["verdict"] == "VOID"
    for vv, ss, want in ((v.replace(kh.HOLD, kh.STACK), s.replace(kh.HOLD, kh.STACK), "OK"),
                         (v, s.replace(kh.HOLD, kh.STACK), "VOID"), (v.replace(kh.HOLD, kh.STACK), s, "VOID")):
        (tmp_path / "versions.txt").write_text(vv)
        (tmp_path / "summary.txt").write_text(ss)
        assert kh.verdict_stack(tmp_path)["verdict"] == want


def test_rule_8_driver_delta_on_the_held_driver():
    """bench/common rule 8 on 7b8f1f69's own sc1_drive.sh: exactly one ' --no-links' on pull_box's rsync line passes;
    anything more, or elsewhere, refuses."""
    held = subprocess.run(["git", "-C", str(REPO), "show", f"{kh.HOLD}:bench/sc1/sc1_drive.sh"], capture_output=True,
                          text=True, check=True).stdout
    assert held.count(kh.PULL_BOX_RSYNC) == 1 and kh.NO_LINKS not in held
    declared = held.replace("  rsync -az --timeout", "  rsync -az --no-links --timeout", 1)
    assert kh.driver_delta_ok(held, held) and kh.driver_delta_ok(declared, held)
    assert not kh.driver_delta_ok(declared.replace("ConnectTimeout=20", "ConnectTimeout=21", 1), held)
    assert not kh.driver_delta_ok(declared.replace(kh.NO_LINKS, kh.NO_LINKS * 2, 1), held)
    in_ssh = held.replace("-o ConnectTimeout=20 ", "-o ConnectTimeout=20 --no-links ", 1)
    assert in_ssh.count(kh.NO_LINKS) == 1 and not kh.driver_delta_ok(in_ssh, held)   # inside -e "ssh ...", not an rsync arg
    at_end = held.replace(kh.PULL_BOX_RSYNC, kh.PULL_BOX_RSYNC + kh.NO_LINKS, 1)
    assert not kh.driver_delta_ok(at_end, held)                                       # same line, another position
    elsewhere = held.replace('SC2="$REPO/bench/sc2";', 'SC2="$REPO/bench/sc2"; # --no-links', 1)
    assert elsewhere != held and not kh.driver_delta_ok(elsewhere, held)


def test_the_live_driver_fetches_without_links():
    """bench/common rule 8: pull_box's fetch never imports links."""
    drive = (REPO / "bench" / "sc1" / "sc1_drive.sh").read_text()
    line = next(x for x in drive.splitlines() if x.startswith("  rsync -az"))
    assert " --no-links " in line


def test_package_hold_on_registered_commits():
    """SC2j Amendment 2's package hold reads OK on a commit against itself, and refuses across a staged-kit change
    (1a25b209 -> 7b8f1f69 changed sc1_run.sh, and staged.sha256 with it)."""
    g = kh.Git(str(REPO))
    assert kh.package_hold(g, kh.HOLD, kh.HOLD)["verdict"] == "OK"
    r = kh.package_hold(g, kh.BASE, kh.HOLD)
    assert r["verdict"] == "REFUSE" and any("staged.sha256" in w for w in r["why"])
