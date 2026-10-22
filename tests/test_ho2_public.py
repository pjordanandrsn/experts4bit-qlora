# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""bench/ho2/ho2_public.py: an HO2 box's public receipt copy. The committed copy rebuilds its verdict through HO2's reducer, the
sums cover every file of the box's receipt, no committed file names the host, and ``derive`` nulls exactly the two GPU-UUID
values and refuses a copy that still names the host."""
import hashlib
import importlib.util
import json
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
RECEIPTS = REPO / "bench" / "ho2" / "receipts"
BOXES = sorted(p.parent.parent for p in RECEIPTS.glob("*/public/SHA256SUMS"))
GPU_UUID = re.compile(r"\b(?:GPU|MIG)-[0-9a-f]{8}-[0-9a-f]{4}-")


def _tool():
    spec = importlib.util.spec_from_file_location("ho2_public", REPO / "bench" / "ho2" / "ho2_public.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sums(path):
    return dict(reversed(ln.split("  ", 1)) for ln in path.read_text().splitlines())


def test_both_boxes_have_a_public_copy():
    assert [p.name for p in BOXES] == ["ho2-4090-2", "ho2-5090-1"], BOXES


@pytest.mark.parametrize("rd", BOXES, ids=lambda p: p.name)
def test_the_public_copy_rebuilds_its_verdict(rd, tmp_path):
    out = tmp_path / "verdict_ho2.json"
    _tool().rebuild(str(rd), str(out))
    assert out.read_bytes() == (rd / "ho2" / "run" / "verdict_ho2.json").read_bytes()
    assert json.loads(out.read_text())["verdict"] == "COVERED"


@pytest.mark.parametrize("rd", BOXES, ids=lambda p: p.name)
def test_the_sums_hold_and_cover_every_file(rd):
    for name, h in _sums(rd / "public" / "SHA256SUMS").items():
        assert hashlib.sha256((rd / "public" / name).read_bytes()).hexdigest() == h, name
    every, store_only = _sums(rd / "SHA256SUMS"), _sums(rd / "STORE-ONLY.sha256")
    committed = {str(p.relative_to(rd)): p for p in (rd / "ho2").rglob("*") if p.is_file()}
    assert set(committed) | set(store_only) == set(every) and not set(committed) & set(store_only)
    for name, p in committed.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == every[name], name
    assert all(store_only[n] == every[n] for n in store_only)
    originals = {n for n in store_only if n.startswith("ho2/run/receipts_")}
    assert {"ho2/" + n for n in _sums(rd / "public" / "SHA256SUMS")} == originals


@pytest.mark.parametrize("rd", BOXES, ids=lambda p: p.name)
def test_no_committed_file_names_the_host(rd):
    for f in rd.rglob("*"):
        if f.is_file():
            t = f.read_text(errors="replace")
            assert not GPU_UUID.search(t), f
            assert "vast.ai" not in t and "\r" not in t, f
    for f in (rd / "public").rglob("*.json"):
        js = json.loads(f.read_text())
        assert js["hardware"]["gpu"]["uuid"] is None and js["plan"]["hardware"]["gpu"]["uuid"] is None, f.name


def _store(tmp_path, note="trained"):
    d = tmp_path / "store"
    (d / "ho2" / "run" / "receipts_x").mkdir(parents=True)
    (d / "instance.json").write_text(json.dumps({"instance_id": 9990001, "machine_id": 88812}))
    rec = {"hardware": {"gpu": {"name": "RTX", "uuid": "GPU-" + "0" * 8 + "-1111-2222-3333-444444444444"}},
           "plan": {"hardware": {"gpu": {"name": "RTX", "uuid": "GPU-" + "0" * 8 + "-1111-2222-3333-444444444444"}}, "total": 1},
           "status": "OK", "note": note}
    (d / "ho2" / "run" / "receipts_x" / "a.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    return d, rec


def test_derive_nulls_exactly_the_two_uuids(tmp_path):
    d, rec = _store(tmp_path)
    out = tmp_path / "pub"
    assert _tool().derive(str(d), str(out)) == ["run/receipts_x/a.json"]
    want = json.loads(json.dumps(rec))
    want["hardware"]["gpu"]["uuid"] = want["plan"]["hardware"]["gpu"]["uuid"] = None
    assert (out / "run" / "receipts_x" / "a.json").read_text() == json.dumps(want, indent=1, sort_keys=True)


def test_derive_refuses_a_copy_that_still_names_the_host(tmp_path):
    d, _ = _store(tmp_path, note="ran on instance 9990001")         # an identifier in a field derive keeps
    with pytest.raises(SystemExit, match="still names the host"):
        _tool().derive(str(d), str(tmp_path / "pub"))
    assert not (tmp_path / "pub").exists()                         # nothing is written when it refuses
