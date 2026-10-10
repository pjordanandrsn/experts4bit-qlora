"""Guarded indexed pins to fixed captured copies; no tokenizer or consumer wiring."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

PEERS = (
    "ra_indexed_views",
    "ra_tokenizer_assets",
    "ra_checkpoint",
    "ra_inputs",
    "ra_stage",
    "ra_asset_views",
    "ra_file_capture",
    "ra_sealed_view",
)
LIMIT = 32 * 1024 * 1024


def require(value, message):
    if not value:
        raise ValueError("indexed-views: " + message)


class IndexedViewsError(ValueError):
    """A terminal refusal with complete binding prefix and cleanup outcomes."""

    def __init__(self, message, record):
        super().__init__("indexed-views: " + message)
        self.record = copy.deepcopy(record)


def peers():
    """Two fixed sibling entry modules; this does not authenticate runtime provenance."""
    base = Path(__file__).absolute().parent
    modules = []
    for name in ("ra_tokenizer_assets", "ra_asset_views"):
        spec = importlib.util.spec_from_file_location(name, base / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class IndexedViews:
    """Own an indexed captured collection through a cooperating caller's lifetime.

    Constructor pins/root/model derive exclusively from the original complete
    indexed/common gate. Two original verifies require Linux -I -S -B and an
    inherited SIGKILL guard and consume exactly eight fixed index responses.
    They bracket construction and complete independent copy/config joins before
    paths are exposed. check/close rebind original common/source/config inputs
    without additional HTTP, as well as every complete sealed copy.

    No CLI, receipt parent, constructor, role map, library activation, worker or
    ABBA wiring. A supplied fetch/reference may be forged; local helper hashes
    detect drift, not runtime/class/site/extension or publisher authentication.
    Cooperating serialized descriptor operations only, no arbitrary same-process
    tamper/syscall-race authority. Copy paths never prove original native reads.
    """

    def __init__(self, manifest, fetch):
        self._terminal = False
        self._collection = None
        self._attempted = False
        self._record = {
            "schema": 1,
            "status": "FAILED",
            "scope": "INDEXED_CAPTURED_COLLECTION_ONLY",
            "bindings": [],
            "copies": None,
            "collection": None,
            "cleanup_proven": False,
            "proves_indexed_copy_equality": False,
            "runtime_authenticated": False,
            "native_read_authenticated": False,
            "consumer_verified": False,
            "tokenizer_executed": False,
            "launch_authority": False,
        }
        try:
            self._gate, self._views = peers()
            self._gate.manifest_fields(manifest)
            self._manifest = copy.deepcopy(manifest)
            self._inputs, self._cp = self._gate.peers()
            self._guard = self._gate.parent_guard(self._manifest)
            self._helpers = self._helper_records()
            before = self._gate.verify(self._manifest, fetch)
            self._record["bindings"].append(copy.deepcopy(before))
            self._before = before
            self._checked = self._cp.check(self._gate.manifest_fields(self._manifest))
            spec, lock, common = self._checked
            require(canonical(common) == canonical(before["common_inputs"]), "derived common input identity")
            self._model = lock["model"]
            self._root = self._inputs.absolute(spec["trees"]["checkpoint"])
            self._pins = copy.deepcopy(before["assets"]["files"])
            require(set(self._pins) == set(self._views.FILES[self._model]), "derived fixed indexed inventory")
            self._attempted = True
            try:
                self._collection = self._views.CapturedAssets(str(self._root), self._model, self._pins)
            except BaseException as exc:
                self._record["collection"] = copy.deepcopy(getattr(exc, "record", None))
                raise
            self._copies = self._copy_records()
            self._record["copies"] = copy.deepcopy(self._copies)
            after = self._gate.verify(self._manifest, fetch)
            self._record["bindings"].append(copy.deepcopy(after))
            require(canonical(after) == canonical(before), "complete bracketing indexed binding drift")
            self._local_check()
            self._record.update(
                status="INDEXED_COLLECTION_OPEN_PENDING_RUNTIME_GATES", proves_indexed_copy_equality=True
            )
        except BaseException as exc:
            self._refuse(exc)

    def _helper_records(self):
        base = Path(__file__).absolute().parent
        names = [base / (name + ".py") for name in PEERS] + [base / "source-pins.json"]
        return {str(path): self._inputs.file_record(path) for path in names}

    def _copy_records(self):
        paths = self._collection.check()
        require(set(paths) == set(self._pins), "complete copy path inventory")
        payloads, records = {}, {}
        for name in sorted(self._pins):
            descriptor = os.open(paths[name], os.O_RDONLY)
            with os.fdopen(descriptor, "rb") as src:
                data = src.read(LIMIT + 1)
            pin = self._pins[name]
            require(
                0 < len(data) <= LIMIT
                and len(data) == pin["size"]
                and hashlib.sha256(data).hexdigest() == pin["sha256"],
                "complete indexed copy pin",
            )
            index = self._before["authority"]["files"][name]
            require(index["size"] == len(data), "complete indexed copy size")
            if index["lfs_sha256"] is None:
                require(self._cp.git_blob(data) == index["git_blob_oid"], "copy ordinary Git identity")
            else:
                require(hashlib.sha256(data).hexdigest() == index["lfs_sha256"], "copy materialized LFS identity")
            payloads[name] = data
            records[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        require(
            canonical(self._gate.configuration(self._model, payloads))
            == canonical(self._before["assets"]["configuration"]),
            "complete captured configuration differs",
        )
        require(self._collection.check() == paths, "copy path identity drift")
        return {"files": records, "configuration": copy.deepcopy(self._before["assets"]["configuration"])}

    def _local_check(self):
        require(self._gate.parent_guard(self._manifest) == self._guard, "parent guard drift")
        require(
            self._helper_records() == self._helpers
            and {name: self._inputs.file_record(name) for name in self._before["execution_inputs"]}
            == self._before["execution_inputs"],
            "complete helper/input byte drift",
        )
        require(
            canonical(self._cp.check(self._gate.manifest_fields(self._manifest))) == canonical(self._checked),
            "complete common input drift",
        )
        require(
            canonical(self._gate.bind(self._root, self._before["authority"], self._checked[1]["trees"]["checkpoint"]))
            == canonical(self._before["assets"]),
            "complete local indexed asset/config drift",
        )
        require(canonical(self._copy_records()) == canonical(self._copies), "complete captured copy drift")
        require(
            self._helper_records() == self._helpers
            and {name: self._inputs.file_record(name) for name in self._before["execution_inputs"]}
            == self._before["execution_inputs"]
            and canonical(self._cp.check(self._gate.manifest_fields(self._manifest))) == canonical(self._checked)
            and self._gate.parent_guard(self._manifest) == self._guard,
            "final helper/input/guard drift",
        )

    @property
    def record(self):
        result = copy.deepcopy(self._record)
        if self._collection is not None:
            result["collection"] = self._collection.record
        return result

    def _finish(self):
        if self._terminal:
            return
        self._terminal = True
        if self._collection is not None:
            record = self._collection.record
            if (
                not record.get("cleanup_proven")
                and record.get("status") == "SEALED_ASSET_COLLECTION_OPEN_PENDING_GATES"
            ):
                try:
                    self._collection.close()
                except BaseException as exc:
                    self._record.update(close_error_type=type(exc).__name__, close_error=str(exc))
            self._record["collection"] = self._collection.record
        record = self._record["collection"]
        self._record["cleanup_proven"] = (not self._attempted) or (
            type(record) is dict and record.get("cleanup_proven") is True
        )

    def _refuse(self, exc):
        self._record.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
        self._finish()
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            exc.indexed_views_record = self.record
            raise exc
        raise IndexedViewsError(str(exc), self.record) from exc

    def check(self):
        require(not self._terminal, "terminal indexed collection")
        try:
            self._local_check()
            return self._collection.paths
        except BaseException as exc:
            self._refuse(exc)

    @property
    def paths(self):
        return self.check()

    def close(self):
        require(not self._terminal, "terminal indexed collection")
        self.check()
        self._finish()
        if not self._record["cleanup_proven"] or "close_error" in self._record:
            self._refuse(ValueError("known-owned cleanup unproven"))
        self._record["status"] = "INDEXED_COLLECTION_CLOSED_PENDING_RUNTIME_GATES"
        return self.record

    def __enter__(self):
        self.check()
        return self

    def __exit__(self, kind, value, traceback):
        if value is not None:
            self._record.update(status="FAILED", error_type=type(value).__name__, error=str(value))
        try:
            self.close()
        except BaseException:
            if value is None:
                raise
        if value is not None:
            self._record.update(status="FAILED", error_type=type(value).__name__, error=str(value))
        return False
