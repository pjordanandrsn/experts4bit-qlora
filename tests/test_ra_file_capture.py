"""CPU descriptor controls; fixture environment substitution is explicit."""

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "bench/ra/ra_file_capture.py"


@pytest.fixture
def capture(monkeypatch):
    spec = importlib.util.spec_from_file_location("file_capture_control", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_environment", lambda: None)
    return module


def prepared(tmp_path, data=b"complete synthetic file"):
    root = tmp_path.resolve() / "owned"
    root.mkdir()
    path = root / "asset"
    path.write_bytes(data)
    return path, data, hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("size", [1, 71, 1048575, 1048576, 1048577, 2097169])
def test_complete_capture_and_closed_descriptors(capture, tmp_path, size):
    path, data, sha = prepared(tmp_path, b"x" * size)
    output, record = capture.capture_file(str(path), size, sha)
    assert output == data
    assert record["read_bytes"] == size
    assert record["status"] == "CAPTURED_BYTES_CLOSED_PENDING_GATES"
    assert record["cleanup_proven"] and all(d["closed"] for d in record["descriptors"])
    assert all(
        record[k] is False
        for k in ["source_authenticated", "runtime_authenticated", "native_read_authenticated", "consumer_verified"]
    )
    for item in record["descriptors"]:
        with pytest.raises(OSError):
            os.fstat(item["fd"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", Path("/tmp/file")),
        ("path", "relative"),
        ("path", "/"),
        ("path", "/a//b"),
        ("path", "/a/../b"),
        ("path", "/a/./b"),
        ("path", "/a/"),
        ("path", "/" + "/".join(["a"] * 65)),
        ("path", "/a\x00b"),
        ("size", True),
        ("size", 1.0),
        ("size", 0),
        ("size", -1),
        ("size", 33554433),
        ("sha", True),
        ("sha", "A" * 64),
        ("sha", "g" * 64),
        ("sha", "0" * 63),
        ("sha", "0" * 65),
    ],
)
def test_typed_pin_and_path_refusals_before_allocation(capture, tmp_path, field, value):
    path, data, sha = prepared(tmp_path)
    arguments = {"path": str(path), "size": len(data), "sha": sha}
    arguments[field] = value
    with pytest.raises(capture.CaptureError) as error:
        capture.capture_file(arguments["path"], arguments["size"], arguments["sha"])
    assert error.value.record["descriptors"] == []
    assert error.value.record["cleanup_proven"]


@pytest.mark.parametrize(
    "mutation", ["wrong_size", "wrong_sha", "symlink", "parent_symlink", "hardlink", "directory", "fifo", "missing"]
)
def test_leaf_and_parent_refusals(capture, tmp_path, mutation):
    path, data, sha = prepared(tmp_path)
    size = len(data)
    if mutation == "wrong_size":
        size += 1
    elif mutation == "wrong_sha":
        sha = "0" * 64
    elif mutation == "symlink":
        link = path.parent / "link"
        link.symlink_to(path)
        path = link
    elif mutation == "parent_symlink":
        link = tmp_path.resolve() / "parent-link"
        link.symlink_to(path.parent, target_is_directory=True)
        path = link / path.name
    elif mutation == "hardlink":
        os.link(path, path.parent / "alias")
    elif mutation == "directory":
        path = path.parent
    elif mutation == "fifo":
        path = path.parent / "pipe"
        os.mkfifo(path)
    elif mutation == "missing":
        path = path.parent / "missing"
    with pytest.raises(capture.CaptureError) as error:
        capture.capture_file(str(path), size, sha)
    assert error.value.record["cleanup_proven"]


@pytest.mark.parametrize(
    "mutation",
    [
        "empty_read",
        "oversize_read",
        "extra_eof",
        "read_error",
        "byte_drift",
        "same_byte_metadata",
        "replace_leaf",
        "replace_parent",
    ],
)
def test_full_read_and_after_read_drift(capture, tmp_path, monkeypatch, mutation):
    path, data, sha = prepared(tmp_path)
    original = os.pread
    changed = False

    def read(fd, count, offset):
        nonlocal changed
        block = original(fd, count, offset)
        if offset == len(data) and mutation == "extra_eof":
            return b"x"
        if changed or offset:
            return block
        changed = True
        if mutation == "empty_read":
            return b""
        if mutation == "oversize_read":
            return block + b"extra"
        if mutation == "read_error":
            raise OSError("owned injected read failure")
        if mutation == "byte_drift":
            return b"z" + block[1:]
        if mutation == "same_byte_metadata":
            path.write_bytes(data)
            info = path.stat()
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1000000))
        elif mutation == "replace_leaf":
            replacement = path.parent / "replacement"
            replacement.write_bytes(data)
            replacement.replace(path)
        elif mutation == "replace_parent":
            parent = path.parent
            parent.rename(parent.with_name("original-owned"))
            parent.mkdir()
            path.write_bytes(data)
        return block

    monkeypatch.setattr(capture.os, "pread", read)
    with pytest.raises(capture.CaptureError) as error:
        capture.capture_file(str(path), len(data), sha)
    assert error.value.record["cleanup_proven"]
    assert "error" in error.value.record


def test_replaced_descriptor_is_preserved(capture, tmp_path, monkeypatch):
    path, data, sha = prepared(tmp_path)
    other = path.parent / "other"
    other.write_bytes(data)
    original_read = os.pread
    original_close = os.close
    replacement = os.open(other, os.O_RDONLY)
    replaced = []

    def read(fd, count, offset):
        block = original_read(fd, count, offset)
        if not replaced:
            os.dup2(replacement, fd, inheritable=False)
            replaced.append(fd)
        return block

    monkeypatch.setattr(capture.os, "pread", read)
    try:
        with pytest.raises(capture.CaptureError) as error:
            capture.capture_file(str(path), len(data), sha)
        record = error.value.record
        assert record["cleanup_proven"] is False
        assert os.fstat(replaced[0]).st_ino == other.stat().st_ino
        assert record["descriptors"][-1]["closed"] is False
    finally:
        for fd in replaced:
            original_close(fd)
        original_close(replacement)


def test_close_failure_is_unproven_and_other_descriptors_close(capture, tmp_path, monkeypatch):
    path, data, sha = prepared(tmp_path)
    original = capture._close_owned
    failed = []

    def close(fd, identity):
        if not failed:
            failed.append(fd)
            raise OSError("owned injected close failure")
        original(fd, identity)

    monkeypatch.setattr(capture, "_close_owned", close)
    try:
        with pytest.raises(capture.CaptureError) as error:
            capture.capture_file(str(path), len(data), sha)
        assert error.value.record["cleanup_proven"] is False
        assert all(d["closed"] for d in error.value.record["descriptors"][:-1])
        assert os.fstat(failed[0]).st_ino == path.stat().st_ino
    finally:
        os.close(failed[0])


def test_unknown_allocation_identity_is_not_closed(capture, tmp_path, monkeypatch):
    path, data, sha = prepared(tmp_path)
    original = os.fstat
    failed = []

    def fstat(fd):
        if not failed:
            failed.append(fd)
            raise OSError("owned injected initial identity failure")
        return original(fd)

    monkeypatch.setattr(capture.os, "fstat", fstat)
    try:
        with pytest.raises(capture.CaptureError) as error:
            capture.capture_file(str(path), len(data), sha)
        assert error.value.record["cleanup_proven"] is False
        assert error.value.record["descriptors"][0]["identity"] is None
        assert original(failed[0])
    finally:
        os.close(failed[0])


def test_original_interruption_and_cleanup_record(capture, tmp_path, monkeypatch):
    path, data, sha = prepared(tmp_path)
    interruption = KeyboardInterrupt("owned interruption")

    def read(*args):
        raise interruption

    monkeypatch.setattr(capture.os, "pread", read)
    with pytest.raises(KeyboardInterrupt) as error:
        capture.capture_file(str(path), len(data), sha)
    assert error.value is interruption
    assert error.value.capture_record["cleanup_proven"]


def test_original_environment_in_fresh_child(tmp_path):
    path, data, sha = prepared(tmp_path)
    code = (
        "import importlib.util,json; s=importlib.util.spec_from_file_location('capture',"
        + repr(str(SOURCE))
        + ");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        + "b,r=m.capture_file("
        + repr(str(path))
        + ","
        + str(len(data))
        + ","
        + repr(sha)
        + ");print(json.dumps(r))"
    )
    # Production boundary: fresh isolated child; parent conftest remains active.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "E4B_", "GNF4_"))}
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code], capture_output=True, text=True, timeout=20, env=env
    )
    if sys.platform == "linux" and os.uname().machine == "x86_64":
        assert result.returncode == 0, result.stderr
        assert '"cleanup_proven": true' in result.stdout
    else:
        assert result.returncode != 0
        assert "Linux x86_64 -I -S -B required" in result.stderr


def test_partial_read_prefix_retained(capture, tmp_path, monkeypatch):
    path, data, sha = prepared(tmp_path)
    original = os.pread
    calls = []

    def read(fd, count, offset):
        calls.append(offset)
        if offset:
            raise OSError("owned partial read refusal")
        return original(fd, 3, offset)

    monkeypatch.setattr(capture.os, "pread", read)
    with pytest.raises(capture.CaptureError) as error:
        capture.capture_file(str(path), len(data), sha)
    assert calls == [0, 3]
    assert error.value.record["read_bytes"] == 3
    assert error.value.record["cleanup_proven"]


def test_readback_failure_never_complete_cleanup(capture, tmp_path, monkeypatch):
    path, data, sha = prepared(tmp_path)
    monkeypatch.setattr(capture.fcntl, "fcntl", lambda *args: 0)
    with pytest.raises(capture.CaptureError) as error:
        capture.capture_file(str(path), len(data), sha)
    assert error.value.record["cleanup_proven"] is False
    assert all(
        "closed descriptor remains present; cleanup unproven" == d["cleanup_error"]
        for d in error.value.record["descriptors"]
    )


def test_original_read_error_retained_with_cleanup_failure(capture, tmp_path, monkeypatch):
    path, data, sha = prepared(tmp_path)
    original_error = OSError("owned original read error")
    original_close = capture._close_owned
    failed = []

    def read(*args):
        raise original_error

    def close(fd, identity):
        if not failed:
            failed.append(fd)
            raise OSError("owned independent cleanup error")
        original_close(fd, identity)

    monkeypatch.setattr(capture.os, "pread", read)
    monkeypatch.setattr(capture, "_close_owned", close)
    try:
        with pytest.raises(capture.CaptureError) as error:
            capture.capture_file(str(path), len(data), sha)
        assert error.value.__cause__ is original_error
        assert error.value.record["error"] == str(original_error)
        assert error.value.record["cleanup_proven"] is False
        assert "cleanup error" in error.value.record["descriptors"][-1]["cleanup_error"]
    finally:
        os.close(failed[0])


def test_changed_bytes_matching_supplied_pin_are_not_provenance(capture, tmp_path):
    path, data, sha = prepared(tmp_path, "supplied bytes Δ".encode())
    renamed = path.with_name("asset-Δ")
    path.rename(renamed)
    output, record = capture.capture_file(str(renamed), len(data), sha)
    assert output == data and record["source_authenticated"] is False
