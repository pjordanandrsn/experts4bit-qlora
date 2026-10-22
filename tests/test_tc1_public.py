# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""bench/tc1/tc1_public.py: a TC1 box's public receipt copy. The committed copy rebuilds its RESULTS through the registered reducer, the
derived files carry none of the fields that name the host, and ``derive`` drops exactly those fields and refuses a copy that still
names it."""
import hashlib
import importlib.util
import json
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
RECEIPTS = REPO / "bench" / "h2h-2026-10-02" / "tc1" / "receipts"
PUBLIC = sorted(p.parent.parent for p in RECEIPTS.glob("*/public/RESULTS-tc1.md"))
GPU_UUID = re.compile(r"\b(?:GPU|MIG)-[0-9a-f]{8}-[0-9a-f]{4}-")


def _tool():
    spec = importlib.util.spec_from_file_location("tc1_public", REPO / "bench" / "tc1" / "tc1_public.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_there_is_a_public_copy():
    assert any(p.name == "tc1-5090-152" for p in PUBLIC), PUBLIC


@pytest.mark.parametrize("rd", PUBLIC, ids=lambda p: p.name)
def test_the_public_copy_rebuilds_its_results(rd, tmp_path):
    out = tmp_path / "RESULTS-tc1.md"
    _tool().rebuild(str(rd), str(out))
    assert out.read_bytes() == (rd / "public" / "RESULTS-tc1.md").read_bytes()


@pytest.mark.parametrize("rd", PUBLIC, ids=lambda p: p.name)
def test_the_public_sums_hold(rd):
    for ln in (rd / "public" / "SHA256SUMS").read_text().splitlines():
        h, name = ln.split()
        assert hashlib.sha256((rd / "public" / name).read_bytes()).hexdigest() == h, name


@pytest.mark.parametrize("rd", PUBLIC, ids=lambda p: p.name)
def test_no_committed_file_names_the_host(rd):
    box = json.loads((rd / "public" / "box.json").read_text())
    assert not set(box) & {"instance_id", "hostname"}
    for f in (rd / "public").glob("*.json"):
        host = (json.loads(f.read_text()).get("env") or {}).get("host")
        if host is not None:
            assert not set(host) & {"uuid", "vast_instance_id", "container_host"}, f.name
    for f in rd.rglob("*"):
        if f.is_file():
            t = f.read_text(errors="replace")
            assert not GPU_UUID.search(t), f
            if f.name != "README.md":                                     # the README names the dropped fields
                assert "vast_instance_id" not in t and '"container_host"' not in t, f
    assert not any(ln.startswith("BOX ") for ln in (rd / "public" / "summary.txt").read_text().splitlines())


def _store(tmp_path, cell_host):
    d = tmp_path / "store" / "tc1"
    d.mkdir(parents=True)
    (d / "box.json").write_text(json.dumps({"box": "A", "instance_id": "9990001", "hostname": "h0st0", "gpu": "x"}))
    (d / "q_e4b_a.json").write_text(json.dumps({"arm": "a", "env": {"host": cell_host}, "peak_vram_gb": 1.0}))
    (d / "summary.txt").write_text("FIXTURE x\nBOX A run r1 instance 9990001 deadline 1\nq/e4b/a rc=0 CELL OK {}\nLOADGATE q/e4b/a attempt 0\n")
    return d


def test_derive_drops_exactly_the_identifying_fields(tmp_path):
    uuid = "GPU-" + "0" * 8 + "-1111-2222-3333-444444444444"
    d = _store(tmp_path, {"uuid": uuid, "vast_instance_id": "9990001", "container_host": "h0st0", "name": "RTX"})
    out = tmp_path / "pub"
    assert _tool().derive(str(d), str(out)) == ["box.json", "q_e4b_a.json", "summary.txt"]
    assert json.loads((out / "box.json").read_text()) == {"box": "A", "gpu": "x"}
    assert json.loads((out / "q_e4b_a.json").read_text()) == {"arm": "a", "env": {"host": {"name": "RTX"}}, "peak_vram_gb": 1.0}
    assert (out / "summary.txt").read_text() == "FIXTURE x\nq/e4b/a rc=0 CELL OK {}\nLOADGATE q/e4b/a attempt 0\n"


def test_derive_refuses_a_copy_that_still_names_the_host(tmp_path):
    d = _store(tmp_path, {"uuid": "u", "note": "ran on h0st0"})        # an identifier in a field derive keeps
    with pytest.raises(SystemExit, match="still names the host"):
        _tool().derive(str(d), str(tmp_path / "pub"))
    assert not (tmp_path / "pub" / "box.json").exists()                 # nothing is written when it refuses
