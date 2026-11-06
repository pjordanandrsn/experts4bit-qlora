"""PGX Amendment 4 at $0: the end class of a reading box (bench/pgx/pgx_box_end.py).

- (b) on the shakedown's REAL lifetimes (r1, committed), reduced in part mode as box 2-1 would: PART_FAULTS, verified,
  not a strike, no arm without a serving lifetime;
- (a) on a whole-draw fixture built from r1's real records: PART_OK;
- (c) every way a box can end without a verifiable record, each a mutant of a (b) or (a) directory that the classifier
  must catch, and named by the check that caught it;
- the second rule: an (arm, budget) at which no lifetime served stops for a person; one serving lifetime, or a lifetime
  skipped for time, does not;
- the box's reduced lifetimes are exactly the reducer's schedule for its draws, so "every scheduled lifetime" is one set;
- the page carries the amendment.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PGX = REPO / "bench" / "pgx"
R1 = PGX / "receipts" / "pgx-shake-r1"
sys.path.insert(0, str(PGX))

import pgx_box_end as be  # noqa: E402
import pgx_reduce as red  # noqa: E402
import pgx_schedule as sch  # noqa: E402

NONCE = "0123456789abcdef"
K = {"S4": {4: 33, 8: 67, 12: 101}, "S8": {4: 17, 8: 35, 12: 52}, "V8": {4: 17, 8: 35, 12: 52}}


def _reg():
    man = json.loads((PGX / "prompts" / "manifest.json").read_text())
    return {"profile_sha": hashlib.sha256((PGX / "fit_profile.jsonl").read_bytes()).hexdigest(),
            "budget_bytes": {str(b): b * 10**9 for b in (4, 8, 12)},
            "sets": {k: v["prompts_sha"] for k, v in man["sets"].items()},
            "preview": {k: v["vram_mass_share"] for k, v in json.loads((R1 / "place_preview.json").read_text()).items()
                        if k.startswith("E-hyb")}}


def _r1(rel):
    return json.loads((R1 / rel).read_text())


def _box(tmp_path, rid, read_from, rc, success=None):
    """A fetched reading box directory, as pgx_drive.sh leaves it: STARTED, RUN_ID, the nonce's exit code (and its success
    marker on 0), box_schedule.txt, registration.json, read/, and the part record the box's own reduction wrote."""
    k, i, _ = sch.parse_run_id(rid)
    spec = f"{k}-{i}"
    d = tmp_path / rid / "pgx"
    d.mkdir(parents=True)
    (d / "STARTED").write_text("")
    (d / "RUN_ID").write_text(rid + "\n")
    (d / "PGX_RUN_NONCE").write_text(NONCE + "\n")
    (d / f"PGX_EXIT_CODE.{NONCE}").write_text(f"{rc}\n")
    if success if success is not None else rc == 0:
        (d / f"PGX_SUCCESS.{NONCE}").write_text("")
    (d / "box_schedule.txt").write_text("\n".join(be.schedule_lines(spec)) + "\n")
    (d / "registration.json").write_text(json.dumps(_reg()))
    read_from(d / "read")
    draws = sorted({r["draw"] for r in sch.box_rows(spec)})
    got = red.main(["--work", str(d / "read"), "--registration", str(d / "registration.json"), "--mode", "part",
                    "--draws", *map(str, draws), "--out", str(d / "part_verdict.json")])
    assert got == rc or rc not in (0, 27), (got, rc)
    return d


def _r1_read(dst):
    shutil.copytree(R1 / "shake", dst)


def _whole_draw(draw):
    """Every registered lifetime of one draw, from r1's real records (as tests/test_pgx_reading.py builds it)."""
    def make(dst):
        for (arm, b, d, p) in red.schedule((draw,)):
            src = "E-hyb" if arm.startswith("E-") else arm
            lt_dir = dst / arm / str(b) / f"d{d}"
            lt_dir.mkdir(parents=True, exist_ok=True)
            lt = _r1(f"shake/{src}/8/d1/lifetime.json")
            lt.update(arm=arm, budget=b, draw=d)
            if arm == "E-hyb-u":
                lt["profile"] = {"kind": "uniform-assumed", "sha256": None}
            if arm in K:
                lt["k_registered"] = lt["k_measured"] = K[arm][b]
            if arm == "E-hyb" and b != "all":
                lt["vram_tier_bytes"] = min(lt["vram_tier_bytes"], b * 10**9)
            (lt_dir / "lifetime.json").write_text(json.dumps(lt))
            rec = _r1(f"shake/{src}/8/d1/P2.json")
            rec["prompts_sha"] = _reg()["sets"][red.PASS_SET[p]]
            (lt_dir / f"{p}.json").write_text(json.dumps(rec))
    return make


# ---------------------------------------------------------------- the three classes
def test_b_on_the_shakedowns_real_lifetimes(tmp_path):
    """r1's four real 8 GB lifetimes, as box 2-1 (draws 1 and 2) of the reading would hold them: every other row is
    missing, so the record reads PART_FAULTS; it verifies, so the box is class (b), not a strike."""
    d = _box(tmp_path, "pgx-4090-1-2-1", _r1_read, 27)
    res = be.classify(str(d), "pgx-4090-1-2-1", 27)
    assert res["class"] == be.PART_FAULTS, res
    assert res["draws"] == [1, 2] and res["spec"] == "2-1"
    assert res["no_serve"] == [] and not res["stop_for_person"]    # r1's lifetimes served; the missing ones were not run
    assert be.exit_code(res) == 10
    rec = json.loads((d / "part_verdict.json").read_text())
    real = {f"{a}|8|1|{p}" for a in sch.ARMS for p in ("P1", "P2")}         # r1 ran P1 and P2 only: those rows are clean
    assert not real & set(rec["faults"])
    assert set(rec["faults"]) == {"|".join(map(str, k)) for k in red.schedule((1, 2))} - real


def test_a_on_a_whole_draw(tmp_path):
    d = _box(tmp_path, "pgx-4090-1-2-2", _whole_draw(3), 0)
    res = be.classify(str(d), "pgx-4090-1-2-2", 0)
    assert res["class"] == be.PART_OK and be.exit_code(res) == 0, res


def test_b_on_a_retry_id_names_the_box_it_replaces(tmp_path):
    d = _box(tmp_path, "pgx-4090-r2-2-2", _whole_draw(3), 0)
    res = be.classify(str(d))
    assert res["class"] == be.PART_OK and res["spec"] == "2-2" and res["draws"] == [3]


def test_not_started(tmp_path):
    d = _box(tmp_path, "pgx-4090-1-2-1", _r1_read, 27)
    (d / "STARTED").unlink()
    assert be.classify(str(d))["class"] == be.NOT_STARTED
    assert be.main([str(d)]) == 13


# ---------------------------------------------------------------- (c): the mutants
def _drop(rel):
    return lambda d: (d / rel).unlink()


def _rewrite(rel, fn):
    def go(d):
        p = d / rel
        p.write_text(fn(p.read_text()))
    return go


def _edit_record(fn):
    def go(d):
        rec = json.loads((d / "part_verdict.json").read_text())
        fn(rec)
        (d / "part_verdict.json").write_text(json.dumps(rec))
    return go


def _pop_fault(rec):
    rec["faults"].pop(sorted(rec["faults"])[0])


def _pop_void(rec):
    rec["void_lifetimes"].pop()


B_MUTANTS = {
    "no part record (the reduction crashed: 27 from the reader's catch-all)": (_drop("part_verdict.json"),
                                                                                "part_verdict.json missing"),
    "a truncated part record": (_rewrite("part_verdict.json", lambda t: t[: len(t) // 2]), "not JSON"),
    "a record that hides one faulted row": (_edit_record(_pop_fault), "does not verify"),
    "a record that hides one VOID lifetime": (_edit_record(_pop_void), "does not verify"),
    "a record for fewer draws than the box's": (_edit_record(lambda r: r.update(draws=[1])), "does not verify"),
    "a lifetime gone from read/ after the reduction": (lambda d: shutil.rmtree(d / "read" / "S4" / "8" / "d1"),
                                                       "does not verify"),
    "no registration": (_drop("registration.json"), "registration.json missing"),
    "no read/": (lambda d: shutil.rmtree(d / "read"), "read/ missing"),
    "a box schedule that is not the registered one": (_rewrite("box_schedule.txt", lambda t: "\n".join(t.splitlines()[1:])),
                                                     "box_schedule.txt differs"),
    "no box schedule": (_drop("box_schedule.txt"), "box_schedule.txt missing"),
    "no exit code for the nonce": (_drop(f"PGX_EXIT_CODE.{NONCE}"), "no readable PGX_EXIT_CODE"),
    "another nonce": (_rewrite("PGX_RUN_NONCE", lambda t: "f" * 16), "no readable PGX_EXIT_CODE"),
    "an unreadable exit code": (_rewrite(f"PGX_EXIT_CODE.{NONCE}", lambda t: "x"), "no readable PGX_EXIT_CODE"),
    "a PART_OK record under rc 27": (_edit_record(lambda r: r.update(verdict="PART_OK")), "does not verify"),
    "no RUN_ID": (_drop("RUN_ID"), "RUN_ID missing"),
    "an unregistered id": (_rewrite("RUN_ID", lambda t: "pgx-4090-1-3-1"), "K 3 is not the registered plan's K = 2"),
    "a shakedown id": (_rewrite("RUN_ID", lambda t: "pgx-shake-r1"), "not a registered reading id"),
}


@pytest.mark.parametrize("name", sorted(B_MUTANTS))
def test_c_every_unverifiable_end_is_a_strike(tmp_path, name):
    d = _box(tmp_path, "pgx-4090-1-2-1", _r1_read, 27)
    assert be.classify(str(d))["class"] == be.PART_FAULTS          # the unmutated box is (b)
    mutate, why = B_MUTANTS[name]
    mutate(d)
    res = be.classify(str(d))
    assert res["class"] == be.STRIKE, (name, res)
    assert any(why in w for w in res["why"]), (name, res["why"])
    assert be.main([str(d)]) == 12


@pytest.mark.parametrize("rc", [9, 11, 12, 25, 33, 40, 41, 78, 130, 1, 26])
def test_c_any_other_code_after_started_is_a_strike(tmp_path, rc):
    d = _box(tmp_path, "pgx-4090-1-2-1", _r1_read, 27)
    (d / f"PGX_EXIT_CODE.{NONCE}").write_text(f"{rc}\n")
    res = be.classify(str(d))
    assert res["class"] == be.STRIKE and res["rc"] == rc, res
    assert any("only 0 and 27" in w for w in res["why"]), res["why"]   # caught by the code, before any record is read


def test_c_the_codes_must_match_their_records(tmp_path):
    ok = _box(tmp_path / "a", "pgx-4090-1-2-2", _whole_draw(3), 0)
    (ok / f"PGX_EXIT_CODE.{NONCE}").write_text("27\n")                        # PART_OK record under 27
    assert be.classify(str(ok))["class"] == be.STRIKE
    ok2 = _box(tmp_path / "b", "pgx-4090-1-2-2", _whole_draw(3), 0)
    (ok2 / f"PGX_SUCCESS.{NONCE}").unlink()                                  # 0 without the nonce's marker
    assert any("success marker" in w for w in be.classify(str(ok2))["why"])
    bad = _box(tmp_path / "c", "pgx-4090-1-2-1", _r1_read, 27)
    (bad / f"PGX_EXIT_CODE.{NONCE}").write_text("0\n")
    (bad / f"PGX_SUCCESS.{NONCE}").write_text("")                            # 0 over a PART_FAULTS record
    assert be.classify(str(bad))["class"] == be.STRIKE


def test_c_the_caller_and_the_box_must_agree(tmp_path):
    d = _box(tmp_path, "pgx-4090-1-2-1", _r1_read, 27)
    assert be.classify(str(d), "pgx-4090-1-2-2")["class"] == be.STRIKE      # a directory for another box
    assert be.classify(str(d), rc=0)["class"] == be.STRIKE                   # a code the caller read differently
    assert be.classify(str(d), "pgx-4090-1-2-1", 27)["class"] == be.PART_FAULTS


def test_an_unreadable_started_never_reads_not_started(tmp_path, monkeypatch):
    """NOT_STARTED is the permissive outcome (no strike), so only FileNotFoundError may mean "absent": a STARTED that
    cannot be stat'ed raises, and the caller stops for a person."""
    d = _box(tmp_path, "pgx-4090-1-2-1", _r1_read, 27)
    real = be.os.lstat

    def lstat(path, *a, **k):
        if str(path).endswith("STARTED"):
            raise PermissionError(13, "Permission denied", str(path))
        return real(path, *a, **k)
    def exists(path):                                        # os.path.exists' own rule: ANY stat error reads as False
        try:
            lstat(path)
        except OSError:
            return False
        return True
    monkeypatch.setattr(be.os, "lstat", lstat)
    monkeypatch.setattr(be.os.path, "exists", exists)
    with pytest.raises(PermissionError):
        be.classify(str(d))


@pytest.mark.parametrize("nonce", ["../x", "0123456789ABCDEF", "0123456789abcde", "0123456789abcdef0",
                                   "0123456789abcdef/../../x", "01234567 89abcdef"])
def test_c_a_malformed_nonce_is_never_used_as_a_path(tmp_path, nonce):
    d = _box(tmp_path, "pgx-4090-1-2-1", _r1_read, 27)
    (d / "PGX_RUN_NONCE").write_text(nonce + "\n")
    target = d / f"PGX_EXIT_CODE.{nonce}"                    # a readable 27 where an unvalidated nonce would lead
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("27\n")
    except OSError:
        pass
    res = be.classify(str(d))
    assert res["class"] == be.STRIKE and res["rc"] is None, res
    assert any("PGX_RUN_NONCE malformed" in w for w in res["why"]), res["why"]


# ---------------------------------------------------------------- the second rule: an arm that served nothing
def _fail(arm, b, d, how):
    def go(box):
        lt_dir = box / "read" / arm / str(b) / f"d{d}"
        lt = json.loads((lt_dir / "lifetime.json").read_text())
        if how == "startup":
            lt.update(startup_failed="rc 45", counters_ok=False)
            for p in lt_dir.glob("P*.json"):
                p.unlink()
        else:                                        # every request of every pass faulted
            for p in lt_dir.glob("P*.json"):
                rec = json.loads(p.read_text())
                rec["faults"] = rec["n"]
                p.write_text(json.dumps(rec))
        (lt_dir / "lifetime.json").write_text(json.dumps(lt))
    return go


def _box_then(tmp_path, rid, read_from, mutate):
    def make(dst):
        read_from(dst)
        mutate(dst.parent)
    return _box(tmp_path, rid, make, 27)


@pytest.mark.parametrize("how", ["startup", "every request faulted"])
def test_an_arm_that_served_no_lifetime_stops_for_a_person(tmp_path, how):
    d = _box_then(tmp_path, "pgx-4090-1-2-2", _whole_draw(3), _fail("V8", 4, 3, how))
    res = be.classify(str(d))
    assert res["class"] == be.PART_FAULTS and res["no_serve"] == [["V8", 4]] and res["stop_for_person"], res
    assert be.exit_code(res) == 11


def test_one_serving_lifetime_of_two_is_enough(tmp_path):
    def both(dst):
        _whole_draw(1)(dst)
        _whole_draw(2)(dst)
    d = _box_then(tmp_path, "pgx-4090-1-2-1", both, _fail("V8", 4, 1, "startup"))
    res = be.classify(str(d))
    assert res["class"] == be.PART_FAULTS and res["no_serve"] == [], res
    d2 = _box_then(tmp_path / "x", "pgx-4090-1-2-1", both,
                   lambda b: (_fail("V8", 4, 1, "startup")(b), _fail("V8", 4, 2, "every request faulted")(b)))
    assert be.classify(str(d2))["no_serve"] == [["V8", 4]]


def test_a_lifetime_skipped_for_time_is_not_a_serve_failure(tmp_path):
    d = _box_then(tmp_path, "pgx-4090-1-2-2", _whole_draw(3),
                  lambda b: shutil.rmtree(b / "read" / "E-hyb" / "4" / "d3"))
    res = be.classify(str(d))
    assert res["class"] == be.PART_FAULTS and res["no_serve"] == [], res


def test_a_control_that_served_nothing_stops_too(tmp_path):
    d = _box_then(tmp_path, "pgx-4090-1-2-2", _whole_draw(3), _fail("E-allvram-g", "all", 3, "startup"))
    assert be.classify(str(d))["no_serve"] == [["E-allvram-g", "all"]]


# ---------------------------------------------------------------- one set of lifetimes
@pytest.mark.parametrize("spec", ["2-1", "2-2"])
def test_the_box_schedule_and_the_reducer_name_the_same_lifetimes(spec):
    rows = sch.box_rows(spec)
    timed = {(r["arm"], r["budget"], r["draw"], p) for r in rows if r["kind"] == "timed" for p in r["passes"]}
    draws = tuple(sorted({r["draw"] for r in rows}))
    assert timed == set(red.schedule(draws))


def test_the_cli_prints_one_json_line(tmp_path):
    d = _box(tmp_path, "pgx-4090-1-2-1", _r1_read, 27)
    r = subprocess.run([sys.executable, str(PGX / "pgx_box_end.py"), str(d), "--run-id", "pgx-4090-1-2-1", "--rc", "27"],
                       capture_output=True, text=True)
    assert r.returncode == 10, r.stdout + r.stderr
    assert r.stdout.splitlines()[0].startswith("BOX_END PART_FAULTS pgx-4090-1-2-1 rc 27")
    assert json.loads(r.stdout.splitlines()[-1])["class"] == be.PART_FAULTS


def test_the_page_carries_amendment_4():
    page = (PGX / "PREREG-pgx.md").read_text()
    a4 = page[page.index("## Amendment 4"):]
    for want in ("PART_OK", "PART_FAULTS", "not a strike", "pgx_box_end.py", "STOP and report",
                 "changes no reading rule", "pgx-shake-1", "strike two"):
        assert want in a4, want
