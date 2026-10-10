"""Complete gate composition with explicit host guard and collection doubles."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source = load("indexed_views_asset_fixture", ROOT / "tests/test_ra_tokenizer_assets.py")
sample = source.sample
stage = source.stage
cli_fixture = source.cli_fixture


def fresh():
    return load("indexed_views_control", ROOT / "bench/ra/ra_indexed_views.py")


@pytest.fixture
def prepared(sample, cli_fixture, tmp_path, monkeypatch):
    component = fresh()
    gate, views = component.peers()
    # Explicitly bind the complete synthetic common gate's own pin fixture.
    monkeypatch.setattr(gate, "peers", lambda: (source.inputs, source.cp))
    monkeypatch.setattr(source.cp, "load_inputs", lambda: source.inputs)
    _, manifest = cli_fixture
    owner = SimpleNamespace(
        collections=[],
        raw_calls=[],
        close_fault=False,
        construct_fault=False,
        bad_copy=False,
        post_create=None,
        guard_drift=False,
    )
    original_guard = {"pid": 31, "expected_parent": 22, "observed_parent": 22, "signal": 9, "platform": "linux"}

    def guard(value):
        gate.manifest_fields(value)
        return {**original_guard, "pid": 32 if owner.guard_drift else 31}

    monkeypatch.setattr(gate, "parent_guard", guard)  # explicit HOST guard double
    original_collection = views.CapturedAssets

    class CollectionDouble:
        """Real complete copied bytes in OWN files, with explicit kernel/cleanup doubles."""

        def __init__(self, root, model, pins):
            self.pins = views.fields(root, model, pins)
            self._record = {"status": "SEALED_ASSET_COLLECTION_OPEN_PENDING_GATES", "cleanup_proven": False}
            self.copyroot = tmp_path / ("explicit-copy-" + str(len(owner.collections)))
            self.copyroot.mkdir()
            self._paths = {}
            owner.collections.append(self)
            if owner.construct_fault:
                error = ValueError("explicit constructor refusal")
                error.record = {"status": "FAILED", "cleanup_proven": True}
                raise error
            for name in pins:
                data = (Path(root) / name).read_bytes()
                if owner.bad_copy and name == "tokenizer.json":
                    data += b" "
                path = self.copyroot / name
                path.write_bytes(data)
                self._paths[name] = str(path)
            if owner.post_create:
                owner.post_create()

        @property
        def record(self):
            return copy.deepcopy(self._record)

        def check(self):
            if self._record["cleanup_proven"]:
                raise ValueError("explicit closed copy")
            return copy.deepcopy(self._paths)

        @property
        def paths(self):
            return self.check()

        def close(self):
            if owner.close_fault:
                self._record.update(status="FAILED", cleanup_proven=False)
                raise ValueError("explicit unproven close")
            self._record.update(status="CLOSED", cleanup_proven=True)
            return self.record

    monkeypatch.setattr(views, "CapturedAssets", CollectionDouble)
    monkeypatch.setattr(component, "peers", lambda: (gate, views))
    rows = iter(copy.deepcopy([sample[2], sample[3]] * 4))

    def fetch(url):
        owner.raw_calls.append(url)
        return next(rows)

    return component, gate, views, manifest, fetch, owner, original_collection


def test_complete_guarded_gate_derived_copy_and_scope(prepared):
    component, gate, views, manifest, fetch, owner, _ = prepared
    collection = component.IndexedViews(manifest, fetch)
    record = collection.record
    assert len(owner.raw_calls) == 8
    assert owner.raw_calls == list(gate.peers()[1].urls(collection._model, collection._checked[1]["revision"])) * 4
    assert len(record["bindings"]) == 2
    assert record["bindings"][0] == record["bindings"][1]
    assert record["copies"]["files"] == record["bindings"][0]["assets"]["files"]
    assert record["bindings"][0]["common_inputs"]["status"] == "INPUT_BYTES_AND_PROJECTIONS_MATCH"
    assert set(collection.paths) == set(views.FILES[collection._model])
    assert record["proves_indexed_copy_equality"] is True
    assert all(
        record[k] is False
        for k in (
            "runtime_authenticated",
            "native_read_authenticated",
            "consumer_verified",
            "tokenizer_executed",
            "launch_authority",
        )
    )
    record["copies"]["files"].clear()
    assert collection.record["copies"]["files"]
    assert collection.close()["cleanup_proven"] is True
    with pytest.raises(ValueError, match="terminal"):
        collection.paths


@pytest.mark.parametrize(
    "kind",
    [
        "override",
        "bad_lock",
        "bad_copy",
        "constructor",
        "post_source",
        "post_guard",
        "after_index",
        "typed_index",
        "missing_index",
        "wrong_index",
    ],
)
def test_construction_refusals_cleanup_and_prefix(prepared, monkeypatch, kind):
    component, gate, _, manifest, fetch, owner, _ = prepared
    if kind == "override":
        manifest = {**manifest, "caller_pins": {}}
    elif kind == "bad_lock":
        manifest = {**manifest, "input_lock_sha256": "f" * 64}
    elif kind == "bad_copy":
        owner.bad_copy = True
    elif kind == "constructor":
        owner.construct_fault = True
    elif kind == "post_source":
        spec = source.inputs.read_json(manifest["input_spec"])
        target = Path(spec["trees"]["checkpoint"]) / "tokenizer.json"
        owner.post_create = lambda: target.write_bytes(target.read_bytes() + b" ")
    elif kind == "post_guard":
        owner.post_create = lambda: setattr(owner, "guard_drift", True)
    else:
        original = fetch

        def changed(url):
            row = original(url)
            if len(owner.raw_calls) == 5:
                if kind == "after_index":
                    row["disabled"] = True
                elif kind == "typed_index":
                    row["siblings"][0]["size"] = True
                elif kind == "missing_index":
                    row["siblings"].pop()
                elif kind == "wrong_index":
                    row["sha"] = "f" * 40
            return row

        fetch = changed
    with pytest.raises(ValueError) as caught:
        component.IndexedViews(manifest, fetch)
    assert caught.value.record["status"] == "FAILED"
    assert caught.value.record["cleanup_proven"] is True
    assert all(c.record["cleanup_proven"] for c in owner.collections) or kind == "constructor"
    assert len(owner.raw_calls) <= 8
    if kind in ("override", "bad_lock"):
        assert not owner.raw_calls


@pytest.mark.parametrize("kind", ["helper", "source", "common", "copy_tail", "copy_bool", "guard", "close"])
def test_post_construction_drift_terminal_no_retry(prepared, monkeypatch, kind):
    component, _, _, manifest, fetch, owner, _ = prepared
    collection = component.IndexedViews(manifest, fetch)
    if kind == "helper":
        original = collection._helper_records
        monkeypatch.setattr(collection, "_helper_records", lambda: {**original(), "extra": {"size": 0}})
    elif kind in ("source", "common"):
        spec = source.inputs.read_json(manifest["input_spec"])
        target = (
            Path(spec["trees"]["checkpoint"]) / "tokenizer.json" if kind == "source" else Path(manifest["input_spec"])
        )
        target.write_bytes(target.read_bytes() + b" ")
    elif kind in ("copy_tail", "copy_bool"):
        target = Path(owner.collections[0]._paths["tokenizer.json"])
        data = target.read_bytes()
        target.write_bytes(data + b" " if kind == "copy_tail" else data.replace(b'"id":7', b'"id":true'))
    elif kind == "guard":
        owner.guard_drift = True
    else:
        owner.close_fault = True
    with pytest.raises(ValueError) as caught:
        collection.close() if kind == "close" else collection.paths
    assert collection.record["status"] == "FAILED"
    assert caught.value.record["cleanup_proven"] is (kind != "close")
    assert len(owner.raw_calls) == 8
    with pytest.raises(ValueError, match="terminal"):
        collection.close()


def test_original_context_error_preserved_with_unproven_cleanup(prepared):
    component, _, _, manifest, fetch, owner, _ = prepared
    collection = component.IndexedViews(manifest, fetch)
    owner.close_fault = True
    with pytest.raises(RuntimeError, match="original caller error"):
        with collection:
            raise RuntimeError("original caller error")
    assert collection.record["status"] == "FAILED"
    assert collection.record["cleanup_proven"] is False
    assert collection.record["error_type"] == "RuntimeError"


def test_production_guard_refuses_before_network_or_allocation(monkeypatch):
    component = fresh()
    manifest = {
        "schema": 1,
        "input_spec": "/unused-spec",
        "stage": "/unused-stage",
        "input_lock": "/unused-lock",
        "input_lock_sha256": "a" * 64,
        "expected_parent": 0,
    }
    with pytest.raises(ValueError) as caught:
        component.IndexedViews(manifest, lambda url: pytest.fail("network before invalid manifest"))
    assert caught.value.record["cleanup_proven"] is True
    assert not caught.value.record["bindings"]


def test_final_output_input_spec_byte_recheck(prepared, monkeypatch):
    component, gate, _, manifest, fetch, owner, _ = prepared
    collection = component.IndexedViews(manifest, fetch)
    original = gate.configuration

    def drift(model, payloads):
        value = original(model, payloads)
        path = Path(manifest["input_spec"])
        path.write_bytes(path.read_bytes() + b" ")
        return value

    monkeypatch.setattr(gate, "configuration", drift)
    with pytest.raises(ValueError, match="final helper/input/guard drift") as caught:
        collection.paths
    assert caught.value.record["cleanup_proven"] is True
    assert len(owner.raw_calls) == 8


def test_fresh_isolated_unprotected_child_refuses():
    environment = load("indexed_views_env", ROOT / "bench/ra/ra_env.py")
    code = """import importlib.util,json,os,sys
s=importlib.util.spec_from_file_location('control',sys.argv[1])
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
manifest={'schema':1,'input_spec':'/unused-spec','stage':'/unused-stage','input_lock':'/unused-lock','input_lock_sha256':'a'*64,'expected_parent':os.getppid()}
try:m.IndexedViews(manifest,lambda url:(_ for _ in ()).throw(RuntimeError('unexpected network')))
except ValueError as e:print(json.dumps({'error':str(e),'record':e.record}))
else:raise RuntimeError('unprotected child accepted')
"""
    child = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code, str(ROOT / "bench/ra/ra_indexed_views.py")],
        env=environment.clean(
            os.environ,
            component="training",
            fixture={},
            venv=Path(sys.prefix),
            cache=ROOT / "bench/ra/receipts/indexed-child-cache",
            threads=1,
            allocator="",
        )[0],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert child.returncode == 0, child.stderr
    result = json.loads(child.stdout)
    assert ("inherited SIGKILL" if sys.platform == "linux" else "selected Linux") in result["error"]
    assert result["record"]["bindings"] == []
    assert result["record"]["cleanup_proven"] is True
