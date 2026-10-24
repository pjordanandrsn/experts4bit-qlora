# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""bench/ho3/ho3_public.py: an HO3 box's public receipt copy. The committed copy rebuilds its verdict through HO3's reducer, the
sums cover every file of the box's receipt, no committed file names the host, and ``derive`` nulls exactly the GPU-UUID values
(two per row receipt, one per plan) and refuses a copy that still names the host or that would differ in more than those."""
import hashlib
import importlib.util
import json
import pathlib
import re
import shutil

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
RECEIPTS = REPO / "bench" / "ho3" / "receipts"
BOXES = sorted(p.parent.parent for p in RECEIPTS.glob("*/public/SHA256SUMS"))
GPU_UUID = re.compile(r"\b(?:GPU|MIG)-[0-9a-f]{8}-[0-9a-f]{4}-")
#: each started box's verdict, as RESULTS-ho3.md reports it
VERDICT = {"ho3-4090-2": "VOID", "ho3-4090-r1": "VOID", "ho3-5090-r1": "COVERED", "ho3-a4000-1": "MISS", "ho3-a4000-r2": "MISS"}
#: never committed: the launcher's files, and the box files the maintainer keeps store-only (forensics, the outer log)
NEVER = ("receipt.json", "instance.json", "known_hosts", "teardown-proof.json", "ho3/forensics.txt", "ho3/outer.log")


def _tool():
    spec = importlib.util.spec_from_file_location("ho3_public", REPO / "bench" / "ho3" / "ho3_public.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sums(path):
    return dict(reversed(ln.split("  ", 1)) for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#"))


def test_every_started_box_has_a_public_copy():
    assert [p.name for p in BOXES] == sorted(VERDICT), BOXES


@pytest.mark.parametrize("rd", BOXES, ids=lambda p: p.name)
def test_the_public_copy_rebuilds_its_verdict(rd, tmp_path):
    out = tmp_path / "verdict_ho3.json"
    _tool().rebuild(str(rd), str(out))
    assert out.read_bytes() == (rd / "ho3" / "run" / "verdict_ho3.json").read_bytes()
    assert json.loads(out.read_text())["verdict"] == VERDICT[rd.name]


@pytest.mark.parametrize("rd", BOXES, ids=lambda p: p.name)
def test_the_sums_hold_and_cover_every_file(rd):
    for name, h in _sums(rd / "public" / "SHA256SUMS").items():
        assert hashlib.sha256((rd / "public" / name).read_bytes()).hexdigest() == h, name
    every, store_only = _sums(rd / "SHA256SUMS"), _sums(rd / "STORE-ONLY.sha256")
    committed = {str(p.relative_to(rd)): p for p in (rd / "ho3").rglob("*") if p.is_file()}
    assert set(committed) | set(store_only) == set(every) and not set(committed) & set(store_only)
    for name, p in committed.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == every[name], name
    assert all(store_only[n] == every[n] for n in store_only)
    assert all(n in store_only for n in NEVER if n in every) and "ho3/forensics.txt" in every, sorted(committed)
    originals = {n for n in store_only if n.startswith(("ho3/run/receipts_", "ho3/run/plan_"))}
    assert {"ho3/" + n for n in _sums(rd / "public" / "SHA256SUMS")} == originals


@pytest.mark.parametrize("rd", BOXES, ids=lambda p: p.name)
def test_no_committed_file_names_the_host(rd):
    for f in rd.rglob("*"):
        if f.is_file():
            t = f.read_text(errors="replace")
            assert not GPU_UUID.search(t), f
            assert "vast.ai" not in t and "\r" not in t, f
    for f in (rd / "public" / "run").glob("receipts_*/*.json"):
        js = json.loads(f.read_text())
        assert js["hardware"]["gpu"]["uuid"] is None and js["plan"]["hardware"]["gpu"]["uuid"] is None, f.name
    for f in (rd / "public" / "run").glob("plan_*.json"):
        assert json.loads(f.read_text())["hardware"]["gpu"]["uuid"] is None, f.name


def test_the_rebuild_reads_the_runners_deadline_skips(tmp_path):
    # No HO3 box skipped a row, so this lays one out: the reducer reads SKIPPED lines from summary.txt beside run/, and a
    # rebuild without that file would read the skipped row as lost (VOID) rather than unread.
    rd = tmp_path / "box"
    shutil.copytree(RECEIPTS / "ho3-a4000-r2", rd)
    for f in (rd / "public" / "run" / "receipts_lfm2_2048").glob("*.json"):
        f.unlink()
    with open(rd / "ho3" / "summary.txt", "a", encoding="utf-8") as f:
        f.write("SKIPPED train lfm2_2048\n")
    out = tmp_path / "verdict_ho3.json"
    _tool().rebuild(str(rd), str(out))
    assert json.loads(out.read_text())["cohorts"]["lfm2"]["verdict"] == "UNREAD"


UUID = "GPU-" + "0" * 8 + "-1111-2222-3333-444444444444"


def _store(tmp_path, note="trained", plan_text=None):
    d = tmp_path / "store"
    (d / "ho3" / "run" / "receipts_x").mkdir(parents=True)
    (d / "instance.json").write_text(json.dumps({"instance_id": 9990001, "machine_id": 88812}))
    rec = {"hardware": {"gpu": {"name": "RTX", "uuid": UUID}},
           "plan": {"hardware": {"gpu": {"name": "RTX", "uuid": UUID}}, "total": 1}, "status": "OK", "note": note}
    plan = {"hardware": {"gpu": {"name": "RTX", "uuid": UUID}}, "total": 1}
    (d / "ho3" / "run" / "receipts_x" / "a.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    (d / "ho3" / "run" / "plan_x.json").write_text(plan_text or json.dumps(plan, indent=1, sort_keys=True))
    return d, rec, plan


def test_derive_nulls_exactly_the_uuids(tmp_path):
    d, rec, plan = _store(tmp_path)
    out = tmp_path / "pub"
    assert _tool().derive(str(d), str(out)) == ["run/plan_x.json", "run/receipts_x/a.json"]
    want = json.loads(json.dumps(rec))
    want["hardware"]["gpu"]["uuid"] = want["plan"]["hardware"]["gpu"]["uuid"] = None
    assert (out / "run" / "receipts_x" / "a.json").read_text() == json.dumps(want, indent=1, sort_keys=True)
    want = json.loads(json.dumps(plan))
    want["hardware"]["gpu"]["uuid"] = None
    assert (out / "run" / "plan_x.json").read_text() == json.dumps(want, indent=1, sort_keys=True)


def test_derive_refuses_a_copy_that_still_names_the_host(tmp_path):
    d, _, _ = _store(tmp_path, note="ran on instance 9990001")      # an identifier in a field derive keeps
    with pytest.raises(SystemExit, match="still names the host"):
        _tool().derive(str(d), str(tmp_path / "pub"))
    assert not (tmp_path / "pub").exists()                         # nothing is written when it refuses


def test_derive_refuses_an_original_it_cannot_reproduce(tmp_path):
    plan = {"hardware": {"gpu": {"name": "RTX", "uuid": UUID}}, "total": 1}
    d, _, _ = _store(tmp_path, plan_text=json.dumps(plan, indent=2))   # not Loggetta's serialization
    with pytest.raises(SystemExit, match="does not serialize back"):
        _tool().derive(str(d), str(tmp_path / "pub"))
    assert not (tmp_path / "pub").exists()
