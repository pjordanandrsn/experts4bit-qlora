"""Standalone guarded indexed-copy receipts; no tokenizer/worker/ABBA wiring.

The parent independently rechecks retained bytes and current indexed inputs.
Child kernel/descriptor snapshots report observations, not independent past
kernel attestation. Supplied process/transport/reference premises may be forged.
"""

import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path

LIMIT = 32 * 1024 * 1024
AUTHORITY = {
    "runtime_authenticated",
    "native_read_authenticated",
    "consumer_verified",
    "tokenizer_executed",
    "launch_authority",
}


def require(value, message):
    if not value:
        raise ValueError("indexed-receipts: " + message)


def peers():
    base = Path(__file__).absolute().parent
    spec = importlib.util.spec_from_file_location("ra_indexed_views", base / "ra_indexed_views.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def helper_records(inputs, indexed):
    base = Path(__file__).absolute().parent
    names = (*indexed.PEERS, "ra_indexed_receipts", "ra_process", "ra_child_guard")
    return {str(p): inputs.file_record(p) for p in [base / (n + ".py") for n in names] + [base / "source-pins.json"]}


def utc(value):
    require(type(value) is str, "typed clock read")
    time = datetime.datetime.fromisoformat(value)
    require(time.utcoffset() == datetime.timedelta(0), "UTC clock read")
    return time


def false_fields(record, names):
    require(all(record.get(n) is False for n in names), "authority remains false")


def identity(value):
    require(
        type(value) in (list, tuple) and len(value) == 2 and all(type(v) is int and v >= 0 for v in value),
        "typed child-reported descriptor identity",
    )
    return list(value)


def capture_observation(record, pin, depth):
    fields = {
        "schema",
        "status",
        "scope",
        "cleanup_proven",
        "descriptors",
        "read_bytes",
        "source_authenticated",
        "runtime_authenticated",
        "native_read_authenticated",
        "consumer_verified",
        "captured_sha256",
        "captured_bytes",
        "source_descriptor_identity",
    }
    require(
        type(record) is dict
        and set(record) == fields
        and type(record["schema"]) is int
        and record["schema"] == 1
        and record["status"] == "CAPTURED_BYTES_CLOSED_PENDING_GATES"
        and record["scope"] == "SUPPLIED_PIN_CAPTURE_ONLY"
        and record["cleanup_proven"] is True
        and type(record["read_bytes"]) is int
        and record["read_bytes"] == pin["size"]
        and type(record["captured_bytes"]) is int
        and record["captured_bytes"] == pin["size"]
        and record["captured_sha256"] == pin["sha256"],
        "complete child-reported capture observation",
    )
    false_fields(
        record, {"source_authenticated", "runtime_authenticated", "native_read_authenticated", "consumer_verified"}
    )
    rows = record["descriptors"]
    require(type(rows) is list and len(rows) == depth, "complete reported capture descriptor inventory")
    ids, fds = [], set()
    for row in rows:
        require(
            type(row) is dict
            and set(row) == {"fd", "identity", "closed", "cleanup_proven"}
            and type(row["fd"]) is int
            and row["fd"] >= 0
            and row["fd"] not in fds
            and row["closed"] is True
            and row["cleanup_proven"] is True,
            "reported descriptor close/ownership",
        )
        fds.add(row["fd"])
        ids.append(identity(row["identity"]))
    require(identity(record["source_descriptor_identity"]) == ids[-1], "reported leaf identity join")
    return ids


def collection_observation(record, pins, root):
    # These are strictly checked CHILD observations. JSON cannot attest past
    # kernel state or authenticate a fabricated cooperating-process record.
    fields = {
        "schema",
        "status",
        "scope",
        "cleanup_proven",
        "files",
        "source_authenticated",
        "runtime_authenticated",
        "native_read_authenticated",
        "consumer_verified",
    }
    require(
        type(record) is dict
        and set(record) == fields
        and type(record["schema"]) is int
        and record["schema"] == 1
        and record["status"] == "SEALED_ASSET_COLLECTION_CLOSED_PENDING_GATES"
        and record["scope"] == "FIXED_SUPPLIED_PIN_CAPTURED_COLLECTION_ONLY"
        and record["cleanup_proven"] is True,
        "complete child-reported collection close",
    )
    false_fields(
        record, {"source_authenticated", "runtime_authenticated", "native_read_authenticated", "consumer_verified"}
    )
    rows = record["files"]
    require(type(rows) is list and len(rows) == len(pins), "complete child-reported collection inventory")
    ancestry = None
    depth = len(Path(root).parts) + 1
    for row, name in zip(rows, sorted(pins)):
        require(
            type(row) is dict
            and set(row) == {"name", "capture", "post_capture", "view", "view_attempted"}
            and row["name"] == name
            and row["view_attempted"] is True,
            "fixed reported asset order",
        )
        before = capture_observation(row["capture"], pins[name], depth)
        after = capture_observation(row["post_capture"], pins[name], depth)
        require(before == after and (ancestry is None or before[:-1] == ancestry), "reported capture identity joins")
        ancestry = before[:-1]
        view = row["view"]
        fields = {
            "schema",
            "status",
            "owned_descriptor_closed",
            "cleanup_proven",
            "copy_scope",
            "source_authenticated",
            "runtime_authenticated",
            "native_read_authenticated",
            "consumer_verified",
            "captured_sha256",
            "captured_bytes",
            "reader_cleanup_proven",
            "kernel_seals",
        }
        require(
            type(view) is dict
            and set(view) == fields
            and type(view["schema"]) is int
            and view["schema"] == 1
            and view["status"] == "SEALED_CAPTURED_COPY_CLOSED_PENDING_GATES"
            and view["copy_scope"] == "CAPTURED_COPY_ONLY"
            and view["owned_descriptor_closed"] is True
            and view["cleanup_proven"] is True
            and view["reader_cleanup_proven"] is True
            and type(view["kernel_seals"]) is int
            and view["kernel_seals"] == 15
            and type(view["captured_bytes"]) is int
            and view["captured_bytes"] == pins[name]["size"]
            and view["captured_sha256"] == pins[name]["sha256"],
            "complete child-reported seal/read/close observation",
        )
        false_fields(
            view, {"source_authenticated", "runtime_authenticated", "native_read_authenticated", "consumer_verified"}
        )
    return copy.deepcopy(record)


def raw_records(inputs, cp, output, rows, lock):
    require(type(rows) is list and len(rows) == 8, "eight retained index responses")
    values, last = [], None
    for i, row in enumerate(rows):
        require(
            type(row) is dict
            and set(row) == {"url", "file", "sha256", "bytes", "started_at", "finished_at"}
            and row["url"] == (cp.urls(lock["model"], lock["revision"]) * 4)[i]
            and row["file"] == f"response-{i:02d}.json"
            and type(row["bytes"]) is int
            and 0 < row["bytes"] <= cp.LIMIT
            and cp.sha(row["sha256"], 64),
            "fixed bounded raw record",
        )
        require(
            inputs.file_record(output / row["file"]) == {"sha256": row["sha256"], "size": row["bytes"]},
            "complete retained raw byte pin",
        )
        start, end = utc(row["started_at"]), utc(row["finished_at"])
        require(start <= end and (last is None or last <= start), "sequential clock reads")
        last = end
        values.append(inputs.read_json(output / row["file"]))
    return values


def copied_bytes(inputs, cp, gate, model, output, binding):
    payloads, files = {}, {}
    for name, pin in sorted(binding["assets"]["files"].items()):
        path = output / ("copy-" + name)
        require(inputs.file_record(path) == pin and 0 < pin["size"] <= LIMIT, "complete retained copy pin")
        with path.open("rb") as src:
            data = src.read(LIMIT + 1)
        require(
            len(data) == pin["size"] and hashlib.sha256(data).hexdigest() == pin["sha256"], "copy changed during read"
        )
        addressed = binding["authority"]["files"][name]
        require(
            addressed["size"] == len(data)
            and (
                cp.git_blob(data) == addressed["git_blob_oid"]
                if addressed["lfs_sha256"] is None
                else hashlib.sha256(data).hexdigest() == addressed["lfs_sha256"]
            ),
            "complete copy Git/LFS identity",
        )
        payloads[name], files[name] = data, pin
    configuration = gate.configuration(model, payloads)
    require(
        canonical(configuration) == canonical(binding["assets"]["configuration"]),
        "complete retained copy configuration",
    )
    return {"files": files, "configuration": configuration}


def receipt_inventory(inputs, output, pins, result=False):
    names = {f"response-{i:02d}.json" for i in range(8)} | {"copy-" + n for n in pins}
    if result:
        names.add("result.json")
    require(
        {p.name for p in output.iterdir()} == names
        and all(p.is_file() and not p.is_symlink() for p in output.iterdir()),
        "exact fixed regular receipt inventory",
    )
    return inputs.inventory(output)


def indexed_record(binding, copies, collection):
    return {
        "schema": 1,
        "status": "INDEXED_COLLECTION_CLOSED_PENDING_RUNTIME_GATES",
        "scope": "INDEXED_CAPTURED_COLLECTION_ONLY",
        "bindings": [binding, copy.deepcopy(binding)],
        "copies": copies,
        "collection": collection,
        "cleanup_proven": True,
        "proves_indexed_copy_equality": True,
        **{k: False for k in AUTHORITY},
    }


def retained(manifest, output, process, manifest_sha256, expected_argv):
    """Recheck bytes after an OWN independently verified process/argv/absence premise.

    Supplied process/reference/transport premises are not authenticated here.
    The caller must verify original guard evidence and actual reap/group absence.
    Child kernel/descriptor JSON is reported evidence, not parent attestation.
    """
    indexed = peers()
    gate, _ = indexed.peers()
    inputs, cp = gate.peers()
    m = gate.manifest_fields(manifest)
    output = inputs.absolute(output)
    require(
        cp.sha(manifest_sha256, 64)
        and type(process) is dict
        and type(expected_argv) is list
        and len(expected_argv) in (9, 10)
        and all(type(v) is str for v in expected_argv)
        and Path(expected_argv[0]).is_absolute()
        and expected_argv[1:4] == ["-I", "-S", "-B"]
        and process.get("argv") == expected_argv
        and (len(expected_argv) == 9 or expected_argv[4] == "-O")
        and Path(expected_argv[-5]).is_absolute()
        and expected_argv[-4] == "--manifest"
        and expected_argv[-2:] == ["--out", str(output)],
        "verified selected argv premise",
    )
    manifest_path = inputs.absolute(expected_argv[-3])
    initial_manifest = inputs.file_record(manifest_path)
    require(
        initial_manifest["sha256"] == manifest_sha256
        and canonical(inputs.read_json(manifest_path)) == canonical(manifest),
        "current manifest bytes/premise",
    )
    guard = process.get("parent_death_guard", {})
    require(type(guard) is dict, "guard observation object")
    observed = guard.get("observed", {})
    require(type(observed) is dict, "original exec guard observation object")
    require(
        process.get("status") == "OK"
        and type(process.get("returncode")) is int
        and process["returncode"] == 0
        and process.get("cleanup_complete") is True
        and type(process.get("pid")) is int
        and process["pid"] > 0
        and type(process.get("attempts")) is int
        and process["attempts"] == 1
        and guard.get("required") is True
        and guard.get("verified") is True
        and type(guard.get("expected_parent")) is int
        and guard["expected_parent"] == manifest["expected_parent"]
        and observed.get("status") == "ARMED"
        and type(observed.get("pid")) is int
        and observed["pid"] == process["pid"]
        and all(
            type(observed.get(k)) is int and observed[k] == manifest["expected_parent"]
            for k in ("expected_parent", "parent_before", "parent_after")
        )
        and type(observed.get("signal")) is int
        and observed["signal"] == 9
        and observed.get("platform") == "linux"
        and observed.get("exec_argv_sha256")
        == hashlib.sha256(json.dumps(expected_argv, separators=(",", ":")).encode()).hexdigest()
        and guard.get("script_sha256") == inputs.file_record(Path(__file__).with_name("ra_child_guard.py"))["sha256"],
        "OWN verified process/guard/reap/group absence required",
    )
    require(
        not any(
            k in process
            for k in ("kill_wait_timeout", "cleanup_error_type", "cleanup_wait_error_type", "cleanup_probe_error_type")
        ),
        "process cleanup errors refuse",
    )
    utc(observed.get("clock"))
    before_helpers = helper_records(inputs, indexed)
    checked = cp.check(m)
    spec, lock, _ = checked
    result = inputs.read_json(output / "result.json")
    require(type(result) is dict, "complete result object")
    before = receipt_inventory(
        inputs, output, lock["trees"]["checkpoint"].keys() & gate.POLICIES[lock["model"]][2], True
    )
    values = raw_records(inputs, cp, output, result.get("observations"), lock)
    bindings = []
    expected_guard = {
        "pid": process["pid"],
        "expected_parent": manifest["expected_parent"],
        "observed_parent": manifest["expected_parent"],
        "signal": 9,
        "platform": "linux",
    }
    for start in (0, 4):
        iterator = iter(values[start : start + 4])
        audited = gate.audit(manifest, lambda url: next(iterator))
        bindings.append({**audited, "parent_guard": expected_guard})
    require(canonical(bindings[0]) == canonical(bindings[1]), "complete eight-record bracketing identity")
    pins = bindings[0]["assets"]["files"]
    require(set(pins) == gate.POLICIES[lock["model"]][2], "complete independently derived copy inventory")
    copies = copied_bytes(inputs, cp, gate, lock["model"], output, bindings[0])
    require(type(result.get("indexed_collection")) is dict, "complete indexed collection object")
    child = result["indexed_collection"].get("collection")
    child = collection_observation(child, pins, spec["trees"]["checkpoint"])
    expected = {
        "schema": 1,
        "status": "PASS",
        "pid": process["pid"],
        "manifest_sha256": manifest_sha256,
        "helper_inputs": before_helpers,
        "indexed_collection": indexed_record(bindings[0], copies, child),
        "observations": result["observations"],
        "parent_independent_kernel_attestation": False,
        "retained_copy_scope": "REGULAR_EXPORTED_COPY_BYTES_ONLY",
    }
    require(
        canonical(result) == canonical(expected)
        and helper_records(inputs, indexed) == before_helpers
        and canonical(cp.check(m)) == canonical(checked)
        and inputs.inventory(output) == before
        and inputs.file_record(manifest_path) == initial_manifest,
        "complete parent result/current helper/input/receipt differs",
    )
    return expected


def produce(manifest_path, output):
    """Fixed standalone child; the public CLI has no fetch/command overrides."""
    indexed = peers()
    gate, _ = indexed.peers()
    inputs, cp = gate.peers()
    manifest = inputs.read_json(manifest_path)
    m = gate.manifest_fields(manifest)
    spec = inputs.read_json(m["input_spec"])
    require(
        type(spec) is dict
        and set(spec) == inputs.SPEC
        and type(spec["trees"]) is dict
        and set(spec["trees"]) == inputs.TREES
        and type(spec["files"]) is dict
        and set(spec["files"]) == inputs.FILES,
        "common input roles",
    )
    output = inputs.absolute(output)
    protected = [inputs.absolute(v) for v in (*spec["trees"].values(), *spec["files"].values())]
    protected += [inputs.absolute(m[k]) for k in ("input_spec", "input_lock", "stage")]
    protected += [inputs.absolute(manifest_path), Path(__file__).absolute().parent]
    require(
        not output.exists()
        and all(not (output == p or output in p.parents or p in output.parents) for p in protected),
        "fresh protected receipt directory",
    )
    initial = inputs.file_record(manifest_path)
    output.mkdir()
    fetch, owned, written_identity = None, None, None
    result = {
        "schema": 1,
        "status": "FAILED",
        "pid": os.getpid(),
        "manifest_sha256": initial["sha256"],
        "indexed_collection": None,
        "observations": [],
        "parent_independent_kernel_attestation": False,
        "retained_copy_scope": "REGULAR_EXPORTED_COPY_BYTES_ONLY",
    }
    try:
        guard = gate.parent_guard(manifest)  # before any index request or seal allocation
        helpers = helper_records(inputs, indexed)
        checked = cp.check(m)
        fetch = cp.Collector(output, cp.urls(checked[1]["model"], checked[1]["revision"]))
        with indexed.IndexedViews(manifest, fetch) as owned:
            paths = owned.check()
            for name in sorted(paths):
                with open(paths[name], "rb") as src:
                    data = src.read(LIMIT + 1)
                pin = owned.record["copies"]["files"][name]
                require(
                    len(data) == pin["size"] and hashlib.sha256(data).hexdigest() == pin["sha256"],
                    "complete export pin",
                )
                with (output / ("copy-" + name)).open("xb") as dest:
                    require(dest.write(data) == len(data), "complete export write")
                owned.check()
        result["indexed_collection"] = owned.record
        pins = owned.record["copies"]["files"]
        execution_inputs = owned.record["bindings"][0]["execution_inputs"]
        collection_observation(owned.record["collection"], pins, checked[0]["trees"]["checkpoint"])
        raw_records(inputs, cp, output, fetch.records, checked[1])
        require(
            canonical(copied_bytes(inputs, cp, gate, checked[1]["model"], output, owned.record["bindings"][0]))
            == canonical(owned.record["copies"]),
            "complete exported copy comparison",
        )
        inventory = receipt_inventory(inputs, output, pins)
        require(
            inputs.file_record(manifest_path) == initial
            and {name: inputs.file_record(name) for name in execution_inputs} == execution_inputs
            and helper_records(inputs, indexed) == helpers
            and canonical(cp.check(m)) == canonical(checked)
            and gate.parent_guard(manifest) == guard,
            "final manifest/helper/input/guard drift",
        )
        result.update(status="PASS", helper_inputs=helpers)
        result["observations"] = fetch.records
        with (output / "result.json").open("x") as dst:
            created = os.fstat(dst.fileno())
            written_identity = (created.st_dev, created.st_ino)
            dst.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        require(
            inputs.file_record(manifest_path) == initial
            and {name: inputs.file_record(name) for name in execution_inputs} == execution_inputs
            and helper_records(inputs, indexed) == helpers
            and canonical(cp.check(m)) == canonical(checked)
            and gate.parent_guard(manifest) == guard
            and canonical(inputs.read_json(output / "result.json")) == canonical(result)
            and receipt_inventory(inputs, output, pins, True)
            == {**inventory, "result.json": inputs.file_record(output / "result.json")},
            "post-output manifest/helper/input/guard/receipt drift",
        )
        return result
    except BaseException as exc:
        result.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
        result["indexed_collection"] = (
            owned.record
            if owned is not None
            else copy.deepcopy(getattr(exc, "record", getattr(exc, "indexed_views_record", None)))
        )
        result["observations"] = fetch.records if fetch is not None else []
        # Only this owned, exclusively created result may be replaced to mark
        # post-output refusal. Never retry a child or delete its failed prefix.
        path = output / "result.json"
        if path.exists():
            require(
                written_identity is not None
                and not path.is_symlink()
                and path.is_file()
                and (path.stat().st_dev, path.stat().st_ino) == written_identity,
                "owned result replaced; failure receipt unproven",
            )
            path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        else:
            with path.open("x") as dst:
                dst.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    produce(args.manifest, args.out)


if __name__ == "__main__":
    main()
