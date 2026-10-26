"""P238 amendment 3: the ONE recorded-good function (bench/p238/p238_hosts.py), on store trees built from REAL committed records:
the seven-burn evidence's own records where they are in this repository (two pre-amendment-83 burns, two amendment-83 burns that
fail at a capped power limit, one amendment-83 burn that passes at 575 W, one run judge's REFUSE) and a real training clock log.
Machine ids are made up; every record byte is the stored one (sha256-pinned)."""
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
FX = REPO / "tests" / "fixtures" / "card_clock"
HOSTS = REPO / "bench" / "p238" / "p238_hosts.py"
CC = REPO / "bench" / "common" / "card_clock.py"

LEGACY_430 = FX / "p238-shake-1_card_clock_burn.json"        # pre-am83 burn, 430 W card, 2,493.5 MHz
LEGACY_575 = FX / "p238-shake-r1_card_clock_burn.json"       # pre-am83 burn, 575 W card, 2,827 MHz (no Reliability count)
AM83_500 = FX / "p238-shake-r2_card_clock_burn.json"         # am83 burn, 500 W card, 2,647 MHz
AM83_475 = FX / "p238-shake-r3_card_clock_burn.json"         # am83 burn, 475 W card, 2,587 MHz
AM83_575_PASS = FX / "sc2f-prove-11_qualify_card_clock.json" # am83 burn, 575 W card, 2,715 MHz, passing
RUN_REFUSE = FX / "sc2f-prove-13_qualify_run_judge.json"     # run judge REFUSE: Reliability on 143 of 143 busy samples
TRAIN_LOG = FX / "axo1767b-5090-1_gpuclk_hp1shake_axolotl.txt"  # a training run's clock log that passes clock mode (3007 MHz)
FIXTURE_SHA256 = {
    LEGACY_430.name: "b65d02d2c9eb8b7ac263516a87c71af32787fdf2084355fee160a14fbf3630b9",
    LEGACY_575.name: "d8b02836665bfc2b8729e895bbdd895ab4831f151e365ee5824bff0ed55c9bdf",
    AM83_500.name: "63a3465c3391d2d8e490f4e4dac8c854a7b7dfb23999105c922a4085d05687f7",
    AM83_475.name: "ac87417a884368a47d6d97035c2f4561aff5457a1ad2cd680dbd228a92a2865f",
    AM83_575_PASS.name: "444cf05ecc258ca173a19fbee38a225474642a14e13f4147f63667e2963115ea",
    RUN_REFUSE.name: "f1c6dea2772d72406fc22fde757d0e8a4c143c34f8e621a950a9ddab532f96b2",
    TRAIN_LOG.name: "9a8216173f187ecc586ab351f23a8b2370deec17a54b568670cf3aa5c43db42e",
}


def _load(path, name, src=None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if src is None:
        spec.loader.exec_module(mod)
    else:
        exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def cc():
    return _load(CC, "card_clock")


@pytest.fixture(scope="module")
def H():
    return _load(HOSTS, "p238_hosts")


def _box(store, run, machine, at, power_w=None, burn=None, sc2_burn=None, run_judge=None, log=None, gpu="RTX 5090"):
    """One run directory in a store tree: receipt.json (machine, start), host_facts.json (power), and the records copied byte for
    byte from the fixtures: a P238 burn (p238/card_clock_burn.json), an SC2 burn and run judge (qualify/...), a run log (tc1/)."""
    rd = store / "receipts" / "experts4bit-qlora" / at[:10] / run
    rd.mkdir(parents=True)
    env = {"vast_gpu_name": gpu}
    if machine is not None:
        env["vast_machine_id"] = machine
    (rd / "receipt.json").write_text(json.dumps({"environment": env, "started_at": at}))
    if power_w is not None:
        (rd / "host_facts.json").write_text(json.dumps({"gpu_power_limit_w": power_w}))
    if burn is not None:
        (rd / "p238").mkdir()
        shutil.copyfile(burn, rd / "p238" / "card_clock_burn.json")
    if sc2_burn is not None or run_judge is not None:
        (rd / "qualify" / "box").mkdir(parents=True)
        if sc2_burn is not None:
            shutil.copyfile(sc2_burn, rd / "qualify" / "box" / "card_clock.json")
        if run_judge is not None:
            shutil.copyfile(run_judge, rd / "qualify" / "run_judge.json")
    if log is not None:
        (rd / "tc1").mkdir()
        shutil.copyfile(log, rd / "tc1" / "gpuclk_train.txt")
    return rd


def test_the_fixtures_are_the_stored_bytes():
    for p in (LEGACY_430, LEGACY_575, AM83_500, AM83_475, AM83_575_PASS, RUN_REFUSE, TRAIN_LOG):
        assert hashlib.sha256(p.read_bytes()).hexdigest() == FIXTURE_SHA256[p.name], p.name


def test_the_r3_host_is_not_good_and_why(tmp_path, H, cc):
    """p238-shake-r3's host: a training log passing at 475 W made it recorded-good under amendment 2; under amendment 3 that log
    does not vouch (the power clause), and its own am83 burn at 475 W fails the floor."""
    _box(tmp_path, "tc1-train", "m114", "2026-10-22T00:45:21Z", power_w=475.0, log=TRAIN_LOG)
    v = H.recorded_good(str(tmp_path / "receipts"), cc)["m114"]
    assert v["good"] is False and "475.0 W < 575 W" in v["why"] and v["records"] == [["run_log", True]]
    _box(tmp_path, "p238-shake-r3", "m114", "2026-10-26T19:50:41Z", power_w=475.0, burn=AM83_475)
    v = H.recorded_good(str(tmp_path / "receipts"), cc)["m114"]
    assert v["good"] is False and v["run"] == "p238-shake-r3" and "2587 MHz < floor 2700" in v["why"]


def test_a_passing_training_log_at_full_power_is_good(tmp_path, H, cc):
    _box(tmp_path, "tc1-train", "m1", "2026-10-22T00:45:21Z", power_w=575.0, log=TRAIN_LOG)
    assert H.recorded_good(str(tmp_path / "receipts"), cc)["m1"]["good"] is True


def test_a_passing_am83_burn_vouches_and_a_run_judge_refuse_overrides_it(tmp_path, H, cc):
    """An am83-era passing burn (575 W, 2,715 MHz) is the gate's own load: good. The same receipt with the run judge's REFUSE
    (persistent Reliability, which a power-capped burn masks) is not good, whatever the burn read."""
    _box(tmp_path, "sc2-a", "m55", "2026-10-26T09:44:50Z", power_w=575.0, sc2_burn=AM83_575_PASS)
    _box(tmp_path, "sc2-b", "m34", "2026-10-26T14:47:22Z", power_w=600.0, sc2_burn=AM83_575_PASS, run_judge=RUN_REFUSE)
    out = H.recorded_good(str(tmp_path / "receipts"), cc)
    assert out["m55"]["good"] is True and out["m55"]["records"] == [["burn", True]]
    assert out["m34"]["good"] is False and out["m34"]["records"] == [["burn", True], ["run_judge", False]]
    assert "run judge REFUSE" in out["m34"]["why"] and "persistent" in out["m34"]["why"]


def test_a_legacy_burn_is_rejudged_and_cannot_vouch(tmp_path, H, cc):
    """r1's pre-am83 burn (575 W, 2,827 MHz) is re-judged in clock mode, never through its stored pass field: the Reliability
    clause reads UNREAD, so the host is not good. The shake-1 legacy burn fails its floor as well."""
    assert json.loads(LEGACY_575.read_text())["pass"] is False          # the stored field, which is never read
    _box(tmp_path, "p238-shake-r1", "m38", "2026-10-26T03:25:21Z", power_w=575.0, burn=LEGACY_575)
    _box(tmp_path, "p238-shake-1", "m15", "2026-10-26T03:13:33Z", power_w=430.0, burn=LEGACY_430)
    out = H.recorded_good(str(tmp_path / "receipts"), cc)
    assert out["m38"]["good"] is False and "UNREAD" in out["m38"]["why"]
    assert out["m15"]["good"] is False and "2493.5 MHz < floor" in out["m15"]["why"]


def test_the_latest_receipt_decides_both_ways(tmp_path, H, cc):
    _box(tmp_path, "old-good", "mA", "2026-10-20T00:00:00Z", power_w=575.0, log=TRAIN_LOG)
    _box(tmp_path, "new-bad", "mA", "2026-10-26T00:00:00Z", power_w=500.0, burn=AM83_500)
    _box(tmp_path, "old-bad", "mB", "2026-10-20T00:00:00Z", power_w=500.0, burn=AM83_500)
    _box(tmp_path, "new-good", "mB", "2026-10-26T00:00:00Z", power_w=575.0, sc2_burn=AM83_575_PASS)
    out = H.recorded_good(str(tmp_path / "receipts"), cc)
    assert (out["mA"]["good"], out["mA"]["run"]) == (False, "new-bad")
    assert (out["mB"]["good"], out["mB"]["run"]) == (True, "new-good")


def test_unattributed_other_class_and_unparseable(tmp_path, H, cc):
    _box(tmp_path, "no-machine", None, "2026-10-26T00:00:00Z", power_w=600.0, log=TRAIN_LOG)
    _box(tmp_path, "a-4090", "m4090", "2026-10-26T00:00:00Z", power_w=450.0, log=TRAIN_LOG, gpu="RTX 4090")
    rd = _box(tmp_path, "bad-json", "mX", "2026-10-26T00:00:00Z", power_w=600.0, sc2_burn=AM83_575_PASS)
    (rd / "qualify" / "box" / "card_clock.json").write_text("{not json")
    out = H.recorded_good(str(tmp_path / "receipts"), cc)
    assert set(out) == {"mX"}                                           # no machine vouches for nothing; a 4090 is out of class
    assert out["mX"]["good"] is False and "unparseable" in out["mX"]["why"]


@pytest.mark.skipif(sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="needs POSIX permissions as a non-root user")
def test_the_read_rule_an_unreadable_store_raises(tmp_path, H, cc):
    rd = _box(tmp_path, "r", "m1", "2026-10-26T00:00:00Z", power_w=575.0, log=TRAIN_LOG)
    os.chmod(rd.parent, 0)
    try:
        with pytest.raises(PermissionError):
            H.recorded_good(str(tmp_path / "receipts"), cc)
    finally:
        os.chmod(rd.parent, 0o755)


def test_mutant_without_the_power_clause_rejoins_the_r3_host(tmp_path, cc):
    """The amendment-2 rule (no power clause) as a mutant: the 475 W training log reads good again, which is how r3 was spent."""
    src = HOSTS.read_text(encoding="utf-8")
    old = '"good": all_pass and (power_ok or burn_pass)'
    assert src.count(old) == 1
    M = _load(HOSTS, "p238_hosts_mutant", src.replace(old, '"good": all_pass'))
    _box(tmp_path, "tc1-train", "m114", "2026-10-22T00:45:21Z", power_w=475.0, log=TRAIN_LOG)
    assert M.recorded_good(str(tmp_path / "receipts"), cc)["m114"]["good"] is True
