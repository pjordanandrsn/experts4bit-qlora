"""Collection controls with explicit sealed-view doubles and real host file capture."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def fresh():
    spec = importlib.util.spec_from_file_location("asset_views_control", ROOT / "bench/ra/ra_asset_views.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ViewDouble:
    """Explicit copy facade. This fixture makes no host kernel-sealing claim."""

    def __init__(self, data, sha, owner):
        self.data, self.sha, self.owner = data, sha, owner
        self.name = owner.names[len(owner.views)]
        self.closed = False
        self._record = {
            "status": "SEALED_CAPTURED_COPY_OPEN",
            "cleanup_proven": False,
            "owned_descriptor_closed": False,
        }
        owner.views.append(self)

    @property
    def record(self):
        return copy.deepcopy(self._record)

    @property
    def path(self):
        return "/explicit-double/" + ("same" if self.owner.duplicate else self.name)

    def check(self):
        if self.owner.check_fault == self.name:
            self.closed = True
            self._record.update(status="FAILED", cleanup_proven=True, owned_descriptor_closed=True)
            raise ValueError("explicit failed view readback")
        return {"sha256": hashlib.sha256(self.data).hexdigest(), "bytes": len(self.data), "kernel_seals": 15}

    def close(self):
        self.owner.closed.append(self.name)
        if self.owner.close_fault == self.name:
            self._record.update(status="FAILED", cleanup_proven=False)
            raise ValueError("explicit unproven view close")
        self.closed = True
        self._record.update(
            status="SEALED_CAPTURED_COPY_CLOSED_PENDING_GATES", cleanup_proven=True, owned_descriptor_closed=True
        )


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    gate = fresh()
    capture, _ = gate.peers()
    monkeypatch.setattr(capture, "_environment", lambda: None)  # named host-only fixture substitution
    owner = SimpleNamespace(
        names=gate.COMMON,
        views=[],
        closed=[],
        check_fault=None,
        close_fault=None,
        construct_fault=None,
        duplicate=False,
        after_create=None,
    )

    def create(data, sha):
        name = owner.names[len(owner.views)]
        if owner.construct_fault == name:
            exc = ValueError("explicit constructor failure")
            exc.record = {"cleanup_proven": True, "owned_descriptor_closed": True, "status": "FAILED"}
            raise exc
        view = ViewDouble(data, sha, owner)
        if owner.after_create:
            owner.after_create(name)
        return view

    monkeypatch.setattr(gate, "peers", lambda: (capture, SimpleNamespace(CapturedView=create)))
    root = tmp_path.resolve() / "positive"
    root.mkdir()

    def inputs(model="Qwen/Qwen3-30B-A3B"):
        owner.names = gate.FILES[model]
        pins = {}
        for name in owner.names:
            data = ("synthetic complete " + name).encode()
            (root / name).write_bytes(data)
            pins[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return str(root), model, pins

    return gate, capture, owner, root, inputs


@pytest.mark.parametrize("model", list(fresh().FILES))
def test_complete_fixed_set_copy_paths_and_reverse_cleanup(prepared, model):
    gate, _, owner, root, inputs = prepared
    collection = gate.CapturedAssets(*inputs(model))
    assert set(collection.paths) == set(gate.FILES[model])
    assert all((root / v.name).read_bytes() == v.data for v in owner.views)
    snapshot = collection.record
    snapshot["source_authenticated"] = True
    snapshot["files"][0]["capture"]["cleanup_proven"] = False
    assert not collection.record["source_authenticated"]
    assert all(
        r["capture"]["cleanup_proven"] and r["post_capture"]["cleanup_proven"] for r in collection.record["files"]
    )
    result = collection.close()
    assert result["cleanup_proven"] and result["status"] == "SEALED_ASSET_COLLECTION_CLOSED_PENDING_GATES"
    assert owner.closed == list(reversed(owner.names))
    assert all(
        result[k] is False
        for k in ["source_authenticated", "runtime_authenticated", "native_read_authenticated", "consumer_verified"]
    )
    for operation in (collection.close, collection.check, lambda: collection.paths):
        with pytest.raises(ValueError, match="terminal"):
            operation()


@pytest.mark.parametrize(
    "fault",
    [
        "root_path",
        "root_relative",
        "root_trailing",
        "root_parent",
        "root_empty",
        "model",
        "unknown",
        "missing",
        "size_bool",
        "size_float",
        "size_zero",
        "size_large",
        "sha_upper",
        "pin_extra",
        "pin_float",
        "total_large",
    ],
)
def test_preflight_refusals_before_any_copy(prepared, fault):
    gate, _, owner, _, inputs = prepared
    root, model, pins = inputs()
    name = gate.COMMON[0]
    if fault == "root_path":
        root = Path(root)
    elif fault == "root_relative":
        root = "relative"
    elif fault == "root_trailing":
        root += "/"
    elif fault == "root_parent":
        root += "/../other"
    elif fault == "root_empty":
        root = "/"
    elif fault == "model":
        model = "other"
    elif fault == "unknown":
        pins["extra"] = pins[name]
    elif fault == "missing":
        del pins[name]
    elif fault.startswith("size_"):
        pins[name]["size"] = {"size_bool": True, "size_float": 1.0, "size_zero": 0, "size_large": gate.LIMIT + 1}[
            fault
        ]
    elif fault == "sha_upper":
        pins[name]["sha256"] = "F" * 64
    elif fault == "pin_extra":
        pins[name]["extra"] = 1
    elif fault == "pin_float":
        pins[name] = 1.0
    elif fault == "total_large":
        for row in pins.values():
            row["size"] = gate.LIMIT
    with pytest.raises(gate.AssetsError) as error:
        gate.CapturedAssets(root, model, pins)
    assert owner.views == [] and error.value.record["files"] == []
    assert error.value.record["cleanup_proven"]


@pytest.mark.parametrize("stage", ["capture", "constructor", "post_hash", "post_identity", "source_symlink"])
def test_partial_prefix_and_delayed_source_mutants(prepared, stage):
    gate, _, owner, root, inputs = prepared
    args = inputs()
    if stage == "capture":
        (root / owner.names[2]).write_bytes(b"wrong")
    elif stage == "constructor":
        owner.construct_fault = owner.names[2]
    else:

        def delayed(name):
            if name == owner.names[-1]:
                path = root / owner.names[0]
                if stage == "post_hash":
                    data = path.read_bytes()
                    path.write_bytes(b"X" + data[1:])
                elif stage == "post_identity":
                    replacement = root / "same-byte-new-inode"
                    replacement.write_bytes(path.read_bytes())
                    replacement.replace(path)
                else:
                    path.rename(root / "retained-original")
                    path.symlink_to(root / "retained-original")

        owner.after_create = delayed
    with pytest.raises(gate.AssetsError) as error:
        gate.CapturedAssets(*args)
    assert error.value.record["status"] == "FAILED" and error.value.record["cleanup_proven"]
    assert owner.closed == [v.name for v in reversed(owner.views)]
    assert all(v.closed for v in owner.views)


@pytest.mark.parametrize("stage", ["check", "duplicate", "close", "context", "context_close"])
def test_terminal_readback_and_cleanup_failures(prepared, stage):
    gate, _, owner, _, inputs = prepared
    collection = gate.CapturedAssets(*inputs())
    if stage in ("context", "context_close"):
        if stage == "context_close":
            owner.close_fault = owner.names[2]
        original = RuntimeError("original caller failure")
        with pytest.raises(RuntimeError) as error:
            with collection:
                raise original
        assert error.value is original
    else:
        if stage == "check":
            owner.check_fault = owner.names[2]
        elif stage == "duplicate":
            owner.duplicate = True
        else:
            owner.close_fault = owner.names[2]
        with pytest.raises(gate.AssetsError):
            collection.close() if stage == "close" else collection.check()
    assert collection.record["status"] == "FAILED"
    assert collection.record["cleanup_proven"] is (stage not in ("close", "context_close"))
    # One close failure must not stop cleanup of the remaining known-owned views.
    assert set(owner.closed) == set(owner.names) - ({owner.names[2]} if stage == "check" else set())
    before = list(owner.closed)
    with pytest.raises(ValueError, match="terminal"):
        collection.close()
    assert owner.closed == before


def test_matching_forged_pins_still_have_no_authority(prepared):
    gate, _, _, root, inputs = prepared
    path, model, pins = inputs()
    name = gate.COMMON[0]
    forged = b"arbitrary caller data"
    (root / name).write_bytes(forged)
    pins[name] = {"size": len(forged), "sha256": hashlib.sha256(forged).hexdigest()}
    with gate.CapturedAssets(path, model, pins) as collection:
        assert collection.record["source_authenticated"] is False
    assert collection.record["cleanup_proven"]


def test_caller_pin_changes_and_later_source_changes_do_not_change_captured_copies(prepared):
    gate, _, owner, root, inputs = prepared
    path, model, pins = inputs()
    collection = gate.CapturedAssets(path, model, pins)
    pins[gate.COMMON[0]]["sha256"] = "0" * 64
    original = owner.views[0].data
    (root / gate.COMMON[0]).write_bytes(b"later source change")
    assert collection.check() and owner.views[0].data == original
    assert collection.close()["cleanup_proven"]


def test_unknown_constructor_cleanup_is_unproven(prepared, monkeypatch):
    gate, capture, owner, _, inputs = prepared
    args = inputs()

    def unknown(data, sha):
        raise ValueError("constructor omitted cleanup record")

    monkeypatch.setattr(gate, "peers", lambda: (capture, SimpleNamespace(CapturedView=unknown)))
    with pytest.raises(gate.AssetsError) as error:
        gate.CapturedAssets(*args)
    assert owner.views == [] and error.value.record["cleanup_proven"] is False


@pytest.mark.parametrize("model", list(fresh().FILES))
def test_original_helpers_in_fresh_isolated_child(tmp_path, model):
    gate = fresh()
    root = tmp_path.resolve() / "child-inputs"
    root.mkdir()
    pins = {}
    for name in gate.FILES[model]:
        data = ("fresh child synthetic " + name).encode()
        (root / name).write_bytes(data)
        pins[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    code = (
        "import importlib.util,json; s=importlib.util.spec_from_file_location('collection',"
        + repr(str(ROOT / "bench/ra/ra_asset_views.py"))
        + ");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        + "c=m.CapturedAssets("
        + repr(str(root))
        + ","
        + repr(model)
        + ","
        + repr(pins)
        + ");"
        + "p=c.paths;r=c.close();print(json.dumps({'paths':p,'record':r}))"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "E4B_", "GNF4_"))}
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code], capture_output=True, text=True, timeout=20, env=env
    )
    if sys.platform == "linux" and os.uname().machine == "x86_64":
        assert result.returncode == 0, result.stderr
        value = json.loads(result.stdout)
        assert set(value["paths"]) == set(gate.FILES[model])
        assert value["record"]["cleanup_proven"]
        assert all(r["view"]["cleanup_proven"] for r in value["record"]["files"])
    else:
        assert result.returncode != 0 and "Linux x86_64 -I -S -B required" in result.stderr
