"""Unwired fixed supplied-pin asset collection; no constructor or source authority."""

import copy
import hashlib
import importlib.util
from pathlib import Path

LIMIT = 32 * 1024 * 1024
COMMON = ("config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")
FILES = {
    "Qwen/Qwen3-30B-A3B": COMMON,
    "ibm-granite/granite-3.1-3b-a800m-instruct": tuple(
        sorted((*COMMON, "added_tokens.json", "special_tokens_map.json"))
    ),
}


class AssetsError(ValueError):
    """Refusal retaining the complete prefix and all cleanup outcomes."""

    def __init__(self, message, record):
        super().__init__("asset-views: " + message)
        self.record = copy.deepcopy(record)


def require(value, message):
    if not value:
        raise ValueError(message)


def peers():
    """Load only the two fixed sibling helpers; this is not runtime authentication."""
    base = Path(__file__).absolute().parent
    modules = []
    for name in ("ra_file_capture", "ra_sealed_view"):
        spec = importlib.util.spec_from_file_location(name, base / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def fields(root, model, pins):
    require(type(root) is str and root.startswith("/") and "\x00" not in root, "absolute typed root")
    parts = root.split("/")[1:]
    require(1 <= len(parts) <= 63 and all(p not in ("", ".", "..") for p in parts), "canonical bounded root")
    require(type(model) is str and model in FILES, "fixed model label")
    require(type(pins) is dict and set(pins) == set(FILES[model]), "exact fixed supplied asset inventory")
    total = 0
    for name in FILES[model]:
        row = pins[name]
        require(type(row) is dict and set(row) == {"size", "sha256"}, "exact supplied pin fields")
        require(type(row["size"]) is int and 0 < row["size"] <= LIMIT, "typed nonempty bounded size")
        sha = row["sha256"]
        require(
            type(sha) is str and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha),
            "typed lowercase supplied SHA256",
        )
        total += row["size"]
    require(total <= LIMIT, "bounded complete collection bytes")
    return copy.deepcopy(pins)


class CapturedAssets:
    """Five or seven supplied-pin files, captured and sealed in fixed filename order.

    All files are captured again after construction, with repeated directory and
    leaf descriptor identities, complete hashes and known-owned capture cleanup.
    check() subsequently verifies the sealed copies, not the original paths.
    Cleanup closes every known-owned view in reverse order, including after a
    partial prefix or context exception. Unknown ownership stays unproven. A
    failure is terminal and never retried. The caller must serialize descriptor
    operations. Labels and pins may be forged, helpers are not authenticated,
    and arbitrary same-process tampering/syscall races are outside this scope.
    No config interpretation, constructor role map, library read or CLI wiring.
    """

    def __init__(self, root, model, pins):
        self._terminal, self._items = False, []
        self._record = {
            "schema": 1,
            "status": "FAILED",
            "scope": "FIXED_SUPPLIED_PIN_CAPTURED_COLLECTION_ONLY",
            "cleanup_proven": False,
            "files": [],
            "source_authenticated": False,
            "runtime_authenticated": False,
            "native_read_authenticated": False,
            "consumer_verified": False,
        }
        try:
            self._pins = fields(root, model, pins)
            self._names = FILES[model]
            self._root = root
            self._capture, self._sealed = peers()
            ancestry = None
            for name in self._names:
                item = {"name": name, "capture": None, "post_capture": None, "view": None, "view_attempted": False}
                self._items.append({"record": item, "view": None})
                pin = self._pins[name]
                data, captured = self._read(name, "capture")
                directories = [v["identity"] for v in captured["descriptors"][:-1]]
                require(ancestry is None or directories == ancestry, "collection directory identity drift")
                ancestry = directories
                try:
                    item["view_attempted"] = True
                    view = self._sealed.CapturedView(data, pin["sha256"])
                except BaseException as exc:
                    item["view"] = copy.deepcopy(getattr(exc, "record", None))
                    raise
                finally:
                    del data
                self._items[-1]["view"] = view
                item["view"] = view.record
            # A delayed earlier-file change must refuse before returning paths.
            for item in self._items:
                row = item["record"]
                data, captured = self._read(row["name"], "post_capture")
                require(
                    captured["source_descriptor_identity"] == row["capture"]["source_descriptor_identity"]
                    and [v["identity"] for v in captured["descriptors"][:-1]] == ancestry,
                    "post-capture source identity drift",
                )
                require(hashlib.sha256(data).hexdigest() == self._pins[row["name"]]["sha256"], "post-capture bytes")
                del data
            self._record["status"] = "SEALED_ASSET_COLLECTION_OPEN_PENDING_GATES"
            self.check()
        except BaseException as exc:
            if self._terminal:
                raise
            self._refuse(exc)

    def _read(self, name, slot):
        row = self._items[self._names.index(name)]["record"]
        pin = self._pins[name]
        try:
            data, captured = self._capture.capture_file(self._root + "/" + name, pin["size"], pin["sha256"])
        except BaseException as exc:
            row[slot] = copy.deepcopy(getattr(exc, "record", getattr(exc, "capture_record", None)))
            if row[slot] is None:
                row[slot] = {"cleanup_proven": False, "error": "capture cleanup receipt unavailable"}
            raise
        row[slot] = copy.deepcopy(captured)
        require(captured["cleanup_proven"] is True, "capture cleanup unproven")
        return data, captured

    @property
    def record(self):
        result = copy.deepcopy(self._record)
        result["files"] = [copy.deepcopy(item["record"]) for item in self._items]
        return result

    def _finish(self):
        if self._terminal:
            return
        self._terminal = True
        clean = True
        for item in reversed(self._items):
            row, view = item["record"], item["view"]
            if view is not None:
                before = view.record
                if not (before.get("cleanup_proven") is True and before.get("owned_descriptor_closed") is True):
                    if before.get("status") == "SEALED_CAPTURED_COPY_OPEN":
                        try:
                            view.close()
                        except BaseException as exc:
                            row["close_error_type"], row["close_error"] = type(exc).__name__, str(exc)
                row["view"] = view.record
            for slot in ("capture", "post_capture", "view"):
                value = row[slot]
                if value is not None:
                    clean = clean and value.get("cleanup_proven") is True
            # A view constructor without a cleanup receipt cannot prove absence.
            if view is None and row["view_attempted"]:
                clean = clean and row["view"] is not None
        self._record["cleanup_proven"] = clean

    def _refuse(self, exc):
        self._record.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
        self._finish()
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            exc.assets_record = self.record
            raise exc
        raise AssetsError(str(exc), self.record) from exc

    def check(self):
        require(not self._terminal, "terminal asset collection")
        try:
            require(
                tuple(item["record"]["name"] for item in self._items) == self._names, "complete fixed view inventory"
            )
            paths = {}
            for item in self._items:
                name, view = item["record"]["name"], item["view"]
                pin = self._pins[name]
                require(
                    view.check() == {"sha256": pin["sha256"], "bytes": pin["size"], "kernel_seals": 15},
                    "complete sealed view pin",
                )
                paths[name] = view.path
                item["record"]["view"] = view.record
            require(len(set(paths.values())) == len(self._names), "distinct owned view paths")
            return paths
        except BaseException as exc:
            self._refuse(exc)

    @property
    def paths(self):
        return self.check()

    def close(self):
        require(not self._terminal, "terminal asset collection")
        self.check()
        self._finish()
        if not self._record["cleanup_proven"] or any("close_error" in v["record"] for v in self._items):
            self._refuse(ValueError("collection close failure or cleanup unproven"))
        if self._record["status"] != "FAILED":
            self._record["status"] = "SEALED_ASSET_COLLECTION_CLOSED_PENDING_GATES"
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
        return False
