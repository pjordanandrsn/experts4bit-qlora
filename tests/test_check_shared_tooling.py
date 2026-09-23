"""scripts/check_shared_tooling.py: every failure it claims to detect, detected.

Each case builds two throwaway repositories from THIS repository's real shared
files, then breaks exactly one thing. The identical pair must pass; every broken
pair must fail naming what broke. A check only ever seen passing carries no
information, so the failures are the point of this file.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("check_shared_tooling", ROOT / "scripts" / "check_shared_tooling.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_shared_tooling"] = mod
    spec.loader.exec_module(mod)
    return mod


cst = _load()
PACKAGES = ("experts4bit-qlora", "grouped-nf4-gemm")


def _repo(base: pathlib.Path, name: str, package: str) -> pathlib.Path:
    root = base / name
    (root / "docs").mkdir(parents=True)
    (root / "pyproject.toml").write_text(f'[project]\nname = "{package}"\nversion = "0.0.0"\n')
    manifest = {"packages": {"runtime": {"package": PACKAGES[0]}, "kernels": {"package": PACKAGES[1]}}}
    (root / "docs" / "system-manifest.json").write_text(json.dumps(manifest))
    for rel in cst.SHARED:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / rel).read_bytes())
    return root


@pytest.fixture
def pair(tmp_path):
    return _repo(tmp_path, "a", PACKAGES[0]), _repo(tmp_path, "b", PACKAGES[1])


def _fails(root, sibling):
    _, _, fail = cst.check(root, sibling)
    return fail


def test_this_repository_carries_every_shared_file():
    assert _fails(ROOT, None) == []


def test_the_shared_list_names_itself():
    assert "scripts/check_shared_tooling.py" in cst.SHARED


def test_an_identical_pair_passes(pair):
    a, b = pair
    assert _fails(a, b) == [] and _fails(b, a) == []


def test_one_differing_byte_fails_and_names_the_file(pair):
    a, b = pair
    rel = cst.SHARED[-1]
    data = bytearray((b / rel).read_bytes())
    data[-1] ^= 0x01
    (b / rel).write_bytes(bytes(data))
    fail = _fails(a, b)
    assert len(fail) == 1 and rel in fail[0] and "differ" in fail[0]


def test_a_shared_file_missing_in_the_sibling_fails(pair):
    a, b = pair
    (b / cst.SHARED[1]).unlink()
    assert any(cst.SHARED[1] in f and "missing in the sibling" in f for f in _fails(a, b))


def test_a_shared_file_missing_here_fails(pair):
    a, _ = pair
    (a / cst.SHARED[1]).unlink()
    assert any("missing here" in f for f in _fails(a, None))


def test_comparing_a_repository_with_itself_fails(pair):
    a, _ = pair
    assert any("same package" in f for f in _fails(a, a))


def test_a_sibling_outside_the_system_fails(pair, tmp_path):
    a, _ = pair
    stranger = _repo(tmp_path, "c", "some-other-package")
    assert any("not one of this system's packages" in f for f in _fails(a, stranger))


def test_an_empty_shared_list_fails(pair, monkeypatch):
    a, b = pair
    monkeypatch.setattr(cst, "SHARED", ())
    assert any("SHARED is empty" in f for f in _fails(a, b))


def test_a_same_named_unshared_script_is_a_note_not_a_failure(pair):
    a, b = pair
    (a / "scripts" / "forked.py").write_text("x = 1\n")
    (b / "scripts" / "forked.py").write_text("x = 2\n")
    _, notes, fail = cst.check(a, b)
    assert fail == [] and any("forked.py" in n and "not yet reconciled" in n for n in notes)
