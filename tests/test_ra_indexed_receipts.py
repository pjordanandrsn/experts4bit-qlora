"""Explicit kernel/process/index doubles; real conftest remains active."""

import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source = load("indexed_receipt_explicit_fixture", ROOT / "tests/test_ra_indexed_views.py")
sample, stage, cli_fixture, prepared = source.sample, source.stage, source.cli_fixture, source.prepared


@pytest.fixture
def wrapper(prepared, cli_fixture, monkeypatch, tmp_path):
    indexed, gate, views, manifest, _, owner, _ = prepared
    module = load("indexed_receipts_under_test", ROOT / "bench/ra/ra_indexed_receipts.py")
    monkeypatch.setattr(module, "peers", lambda: indexed)
    inputs, cp = gate.peers()
    root, _ = cli_fixture
    parent = manifest["expected_parent"]
    guard = {
        "pid": os.getpid(),
        "expected_parent": parent,
        "observed_parent": parent,
        "signal": 9,
        "platform": "linux",
    }
    monkeypatch.setattr(gate, "parent_guard", lambda _: guard)
    original_collection = views.CapturedAssets

    class ExplicitReportedCollectionDouble(original_collection):
        @property
        def record(self):
            result = copy.deepcopy(self._record)
            clean = result["cleanup_proven"]
            if result["status"] == "CLOSED":
                result["status"] = "SEALED_ASSET_COLLECTION_CLOSED_PENDING_GATES"
            result.update(
                schema=1,
                scope="FIXED_SUPPLIED_PIN_CAPTURED_COLLECTION_ONLY",
                files=[],
                source_authenticated=False,
                runtime_authenticated=False,
                native_read_authenticated=False,
                consumer_verified=False,
            )
            depth = len(Path(self._fixture_root).parts) + 1
            for i, name in enumerate(sorted(self._fixture_pins)):
                pin = self._fixture_pins[name]
                ids = [[j, j] for j in range(depth - 1)] + [[900 + i, 900 + i]]
                capture = {
                    "schema": 1,
                    "status": "CAPTURED_BYTES_CLOSED_PENDING_GATES",
                    "scope": "SUPPLIED_PIN_CAPTURE_ONLY",
                    "cleanup_proven": True,
                    "descriptors": [
                        {"fd": j, "identity": v, "closed": True, "cleanup_proven": True} for j, v in enumerate(ids)
                    ],
                    "read_bytes": pin["size"],
                    "source_authenticated": False,
                    "runtime_authenticated": False,
                    "native_read_authenticated": False,
                    "consumer_verified": False,
                    "captured_sha256": pin["sha256"],
                    "captured_bytes": pin["size"],
                    "source_descriptor_identity": ids[-1],
                }
                view = {
                    "schema": 1,
                    "status": "SEALED_CAPTURED_COPY_CLOSED_PENDING_GATES" if clean else "SEALED_CAPTURED_COPY_OPEN",
                    "owned_descriptor_closed": clean,
                    "cleanup_proven": clean,
                    "copy_scope": "CAPTURED_COPY_ONLY",
                    "source_authenticated": False,
                    "runtime_authenticated": False,
                    "native_read_authenticated": False,
                    "consumer_verified": False,
                    "captured_sha256": pin["sha256"],
                    "captured_bytes": pin["size"],
                    "reader_cleanup_proven": True,
                    "kernel_seals": 15,
                }
                result["files"].append(
                    {
                        "name": name,
                        "capture": capture,
                        "post_capture": copy.deepcopy(capture),
                        "view": view,
                        "view_attempted": True,
                    }
                )
            return result

        def __init__(self, root, model, pins):
            self._fixture_root, self._fixture_pins = root, pins
            super().__init__(root, model, pins)

    monkeypatch.setattr(views, "CapturedAssets", ExplicitReportedCollectionDouble)
    # Reuse the complete indexed fixture's registered synthetic sibling/tree rows.
    _, _, _, _, fetch, _, _ = prepared

    class ExplicitRetainedReplay(cp.Collector):
        def __call__(self, url):
            value = fetch(url)
            data = json.dumps(value, separators=(",", ":")).encode()
            path = self.out / f"response-{len(self.records):02d}.json"
            with path.open("xb") as dst:
                dst.write(data)
            self.records.append(
                {
                    "url": url,
                    "file": path.name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                    "started_at": cp.clock(),
                    "finished_at": cp.clock(),
                }
            )
            return json.loads(data)

    monkeypatch.setattr(cp, "Collector", ExplicitRetainedReplay)
    output = root / "standalone-receipt"
    argv = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        module.__file__,
        "--manifest",
        str(root / "manifest.json"),
        "--out",
        str(output),
    ]
    process = {
        "argv": argv,
        "status": "OK",
        "returncode": 0,
        "cleanup_complete": True,
        "pid": os.getpid(),
        "attempts": 1,
        "parent_death_guard": {
            "required": True,
            "verified": True,
            "expected_parent": parent,
            "script_sha256": inputs.file_record(ROOT / "bench/ra/ra_child_guard.py")["sha256"],
            "observed": {
                "status": "ARMED",
                "pid": os.getpid(),
                "expected_parent": parent,
                "parent_before": parent,
                "parent_after": parent,
                "signal": 9,
                "platform": "linux",
                "clock": cp.clock(),
                "exec_argv_sha256": hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest(),
            },
        },
    }
    return module, indexed, gate, inputs, cp, root, manifest, output, process, argv, owner


def produce(values):
    module, _, _, _, _, root, _, output, _, _, _ = values
    return module.produce(root / "manifest.json", output)


def parent(values):
    module, _, _, inputs, _, root, manifest, output, process, argv, _ = values
    return module.retained(manifest, str(output), process, inputs.file_record(root / "manifest.json")["sha256"], argv)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def test_complete_child_and_parent_copies_without_kernel_attestation(wrapper):
    result = produce(wrapper)
    assert parent(wrapper) == result
    assert result["parent_independent_kernel_attestation"] is False
    assert all(result["indexed_collection"][name] is False for name in wrapper[0].AUTHORITY)
    assert len(list(wrapper[7].iterdir())) in (14, 16)
    assert len(wrapper[-1].raw_calls) == 8
    assert all(v.record["cleanup_proven"] for v in wrapper[-1].collections)


@pytest.mark.parametrize(
    "kind",
    [
        "result_tail",
        "typed_schema",
        "typed_pid",
        "extra_result",
        "copy_tail",
        "raw_resealed",
        "raw_url",
        "raw_order",
        "raw_clock",
        "extra_file",
        "missing_copy",
        "symlink_copy",
        "copy_swap",
        "collection_extra",
        "seal_typed",
        "cleanup_missing",
        "descriptor_unknown",
        "descriptor_typed",
        "capture_tail",
        "configuration",
        "binding_tail",
        "authority_true",
        "manifest_whitespace",
        "helper",
        "spec_whitespace",
    ],
)
def test_parent_rejects_complete_receipt_and_late_input_mutants(wrapper, kind, monkeypatch):
    module, _, _, inputs, _, root, _, output, _, _, _ = wrapper
    result = produce(wrapper)
    names = sorted(result["indexed_collection"]["copies"]["files"])
    row = result["indexed_collection"]["collection"]["files"][0]
    if kind == "result_tail":
        result["indexed_collection"]["copies"]["files"][names[-1]]["sha256"] = "f" * 64
    elif kind == "typed_schema":
        result["schema"] = True
    elif kind == "typed_pid":
        result["pid"] = float(result["pid"])
    elif kind == "extra_result":
        result["extra"] = False
    elif kind == "copy_tail":
        path = output / ("copy-" + names[-1])
        path.write_bytes(path.read_bytes() + b" ")
    elif kind == "raw_resealed":
        path = output / "response-00.json"
        path.write_bytes(b"{}")
        result["observations"][0].update(sha256=inputs.file_record(path)["sha256"], bytes=2)
    elif kind == "raw_url":
        result["observations"][0]["url"] = "http://example.invalid/"
    elif kind == "raw_order":
        result["observations"].reverse()
    elif kind == "raw_clock":
        result["observations"][0]["finished_at"] = "2000-01-01T00:00:00+00:00"
    elif kind == "extra_file":
        (output / "extra").write_bytes(b"x")
    elif kind == "missing_copy":
        (output / ("copy-" + names[0])).rename(output / "renamed-positive")
    elif kind == "symlink_copy":
        path = output / ("copy-" + names[0])
        path.rename(output / "positive-original")
        path.symlink_to(output / "positive-original")
    elif kind == "copy_swap":
        a, b = [output / ("copy-" + n) for n in names[:2]]
        a.write_bytes(b.read_bytes())
    elif kind == "collection_type":
        result["indexed_collection"] = []
    elif kind == "collection_extra":
        row["extra"] = False
    elif kind == "seal_typed":
        row["view"]["kernel_seals"] = 15.0
    elif kind == "cleanup_missing":
        row["view"]["reader_cleanup_proven"] = False
    elif kind == "descriptor_unknown":
        row["capture"]["descriptors"][0]["identity"] = None
    elif kind == "descriptor_typed":
        row["capture"]["descriptors"][0]["fd"] = False
    elif kind == "capture_tail":
        row["post_capture"]["captured_sha256"] = "f" * 64
    elif kind == "configuration":
        result["indexed_collection"]["copies"]["configuration"]["extra"] = 1
    elif kind == "binding_tail":
        result["indexed_collection"]["bindings"][1]["authority"]["extra"] = 1
    elif kind == "authority_true":
        result["indexed_collection"]["runtime_authenticated"] = True
    elif kind == "manifest_whitespace":
        path = root / "manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
        wrapper[8]["argv"] = wrapper[9]
        # Pass the ORIGINAL committed child manifest hash rather than bless new bytes.
        with pytest.raises(ValueError):
            module.retained(wrapper[6], str(output), wrapper[8], result["manifest_sha256"], wrapper[9])
        return
    elif kind == "spec_whitespace":
        path = Path(wrapper[6]["input_spec"])
        path.write_bytes(path.read_bytes() + b" ")
    else:
        original = inputs.file_record

        def drift(path):
            value = original(path)
            return {**value, "sha256": "f" * 64} if Path(path).name == "ra_indexed_receipts.py" else value

        monkeypatch.setattr(inputs, "file_record", drift)
    write(output / "result.json", result)
    with pytest.raises(ValueError):
        parent(wrapper)


@pytest.mark.parametrize(
    "kind",
    [
        "pid",
        "guard",
        "reap",
        "return",
        "bool_return",
        "attempts",
        "argv",
        "exec_hash",
        "guard_script",
        "guard_parent",
        "null_return",
        "timeout",
        "signal",
        "clock",
    ],
)
def test_parent_process_premise_refusals(wrapper, kind):
    produce(wrapper)
    process = wrapper[8]
    if kind == "pid":
        process["pid"] += 1
    elif kind == "guard":
        process["parent_death_guard"]["verified"] = False
    elif kind == "reap":
        process["cleanup_complete"] = False
    elif kind == "return":
        process["returncode"] = 1
    elif kind == "bool_return":
        process["returncode"] = False
    elif kind == "attempts":
        process["attempts"] = 2
    elif kind == "argv":
        process["argv"] = [*process["argv"], "--extra"]
    elif kind == "exec_hash":
        process["parent_death_guard"]["observed"]["exec_argv_sha256"] = "f" * 64
    elif kind == "guard_script":
        process["parent_death_guard"]["script_sha256"] = "f" * 64
    elif kind == "guard_parent":
        process["parent_death_guard"]["expected_parent"] += 1
    elif kind == "null_return":
        process["returncode"] = None
    elif kind == "timeout":
        process["kill_wait_timeout"] = True
    elif kind == "signal":
        process["parent_death_guard"]["observed"]["signal"] = True
    else:
        process["parent_death_guard"]["observed"]["clock"] = "2000-01-01T00:00:00"
    with pytest.raises(ValueError):
        parent(wrapper)


@pytest.mark.parametrize(
    "kind",
    [
        "close",
        "construction",
        "export",
        "after_export_spec",
        "post_output_spec",
        "manifest",
        "extra_output",
        "raw",
        "unprotected",
    ],
)
def test_child_failure_prefix_cleanup_and_no_retry(wrapper, kind, monkeypatch):
    module, indexed, gate, inputs, _, root, manifest, output, _, _, owner = wrapper
    if kind == "close":
        owner.close_fault = True
    elif kind == "construction":
        owner.construct_fault = True
    elif kind == "unprotected":
        real = load("original_guard_for_refusal", ROOT / "bench/ra/ra_tokenizer_assets.py")
        monkeypatch.setattr(gate, "parent_guard", real.parent_guard)
    elif kind in ("after_export_spec", "manifest", "extra_output", "raw"):
        original_close = indexed.IndexedViews.close

        def close(self):
            result = original_close(self)
            if kind == "after_export_spec":
                path = Path(manifest["input_spec"])
                path.write_bytes(path.read_bytes() + b" ")
            elif kind == "manifest":
                path = root / "manifest.json"
                path.write_bytes(path.read_bytes() + b" ")
            elif kind == "extra_output":
                (output / "extra").write_bytes(b"x")
            else:
                (output / "response-00.json").write_bytes(b"{}")
            return result

        monkeypatch.setattr(indexed.IndexedViews, "close", close)
    elif kind == "post_output_spec":
        original_record = inputs.file_record
        changed = [False]

        def record(path):
            if Path(path) == root / "manifest.json" and (output / "result.json").exists() and not changed[0]:
                changed[0] = True
                target = Path(manifest["input_spec"])
                target.write_bytes(target.read_bytes() + b" ")
            return original_record(path)

        monkeypatch.setattr(inputs, "file_record", record)
    else:
        original_check = indexed.IndexedViews.check

        def check(self):
            result = original_check(self)
            if any(output.glob("copy-*")):
                raise ValueError("explicit post-export check failure")
            return result

        monkeypatch.setattr(indexed.IndexedViews, "check", check)
    with pytest.raises(ValueError):
        produce(wrapper)
    result = inputs.read_json(output / "result.json")
    assert result["status"] == "FAILED"
    assert len(owner.raw_calls) == (0 if kind == "unprotected" else 4 if kind == "construction" else 8)
    if kind == "unprotected":
        assert result["indexed_collection"] is None
    elif kind == "close":
        assert result["indexed_collection"]["cleanup_proven"] is False
    elif kind not in ("construction", "export"):
        assert result["indexed_collection"]["cleanup_proven"] is True
    with pytest.raises(ValueError, match="fresh protected"):
        produce(wrapper)


@pytest.mark.parametrize("kind", ["extra_manifest", "bool_schema", "caller_command", "protected", "duplicate_json"])
def test_fixed_manifest_and_output_preflight(wrapper, kind):
    module, _, _, _, _, root, manifest, _, _, _, owner = wrapper
    output = wrapper[7]
    if kind == "protected":
        output = Path(manifest["stage"]) / "new-output"
    elif kind == "duplicate_json":
        (root / "manifest.json").write_text('{"schema":1,"schema":1}')
    else:
        changed = copy.deepcopy(manifest)
        if kind == "bool_schema":
            changed["schema"] = True
        elif kind == "caller_command":
            changed["command"] = ["caller"]
        else:
            changed["extra"] = False
        write(root / "manifest.json", changed)
    with pytest.raises(ValueError):
        module.produce(root / "manifest.json", output)
    assert not output.exists() and not owner.raw_calls and not owner.collections


def test_matching_forged_child_identity_boundary_is_not_kernel_authentication(wrapper):
    result = produce(wrapper)
    rows = result["indexed_collection"]["collection"]["files"]
    for row in rows:
        for slot in ("capture", "post_capture"):
            capture = row[slot]
            for descriptor in capture["descriptors"]:
                descriptor["identity"] = [v + 500000 for v in descriptor["identity"]]
            capture["source_descriptor_identity"] = capture["descriptors"][-1]["identity"]
    write(wrapper[7] / "result.json", result)
    accepted = parent(wrapper)
    assert accepted["parent_independent_kernel_attestation"] is False
    assert accepted["indexed_collection"]["native_read_authenticated"] is False


def test_output_result_unknown_replacement_is_preserved(wrapper, monkeypatch):
    module, _, _, inputs, _, root, _, output, _, _, _ = wrapper
    original = inputs.file_record
    changed = [False]

    def record(path):
        if Path(path) == root / "manifest.json" and (output / "result.json").exists() and not changed[0]:
            changed[0] = True
            (output / "result.json").rename(output / "owned-result-before-replacement.json")
            (output / "result.json").write_bytes(b"unknown replacement")
        return original(path)

    monkeypatch.setattr(inputs, "file_record", record)
    with pytest.raises(ValueError, match="owned result replaced"):
        produce(wrapper)
    assert (output / "result.json").read_bytes() == b"unknown replacement"
    assert original(output / "owned-result-before-replacement.json")["size"] > 0
    assert all(v.record["cleanup_proven"] for v in wrapper[-1].collections)
