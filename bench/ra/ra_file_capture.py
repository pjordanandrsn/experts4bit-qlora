"""Unwired no-follow file capture. Supplied pins do not authenticate provenance."""

import copy
import errno
import fcntl
import hashlib
import os
import stat
import sys

LIMIT = 32 * 1024 * 1024


class CaptureError(ValueError):
    """Refusal retaining original error and all owned descriptor cleanup results."""

    def __init__(self, message, record):
        super().__init__("file-capture: " + message)
        self.record = copy.deepcopy(record)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _environment():
    _require(
        sys.platform == "linux"
        and os.uname().machine == "x86_64"
        and sys.flags.isolated
        and sys.flags.no_site
        and sys.dont_write_bytecode,
        "Linux x86_64 -I -S -B required",
    )


def _identity(value):
    return value.st_dev, value.st_ino


def _fingerprint(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _close_owned(fd, identity):
    # Serialized cooperating process only; this cannot eliminate fd reuse races.
    _require(
        identity is not None and _identity(os.fstat(fd)) == identity,
        "descriptor identity unknown or replaced; cleanup unproven",
    )
    os.close(fd)
    try:
        fcntl.fcntl(fd, fcntl.F_GETFD)
    except OSError as exc:
        _require(exc.errno == errno.EBADF, "closed descriptor EBADF readback")
    else:
        raise ValueError("closed descriptor remains present; cleanup unproven")


def capture_file(path, expected_size, expected_sha256):
    """Return complete captured bytes and a cleanup record, with supplied pins only.

    Every component is opened relative to its retained no-follow parent directory.
    The leaf must be a regular single-link file; reads and path/descriptor checks
    are bounded. Directory identities and complete leaf metadata repeat after the
    full read. All known-owned descriptors must close with EBADF readback before
    returning bytes. Unknown/replaced descriptors are preserved and refuse.
    Callers must serialize their own process's descriptor operations. The result
    does not authenticate the expected pin, arbitrary process tampering, original
    library asset-path consumption, runtime, a tokenizer or a consumer.
    """
    record = {
        "schema": 1,
        "status": "FAILED",
        "scope": "SUPPLIED_PIN_CAPTURE_ONLY",
        "cleanup_proven": False,
        "descriptors": [],
        "read_bytes": 0,
        "source_authenticated": False,
        "runtime_authenticated": False,
        "native_read_authenticated": False,
        "consumer_verified": False,
    }
    owned, failure, data = [], None, None
    try:
        _require(type(path) is str and path.startswith("/") and "\x00" not in path, "absolute typed path")
        parts = path.split("/")[1:]
        _require(
            1 <= len(parts) <= 64 and all(p not in ("", ".", "..") for p in parts), "canonical bounded path components"
        )
        _require(type(expected_size) is int and 0 < expected_size <= LIMIT, "typed bounded nonempty size")
        _require(
            type(expected_sha256) is str
            and len(expected_sha256) == 64
            and all(c in "0123456789abcdef" for c in expected_sha256),
            "typed lowercase SHA256",
        )
        _environment()
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC

        def acquire(name, flags, parent=None):
            fd = os.open(name, flags, dir_fd=parent)
            item = {"fd": fd, "identity": None, "closed": False, "cleanup_proven": False}
            owned.append(item)  # retain allocation before possible fstat failure
            item["identity"] = _identity(os.fstat(fd))
            _require(os.get_inheritable(fd) is False, "close-on-exec acquisition")
            return fd

        directories = [(acquire("/", directory_flags), None, "/")]
        for component in parts[:-1]:
            parent = directories[-1][0]
            directories.append((acquire(component, directory_flags, parent), parent, component))
        parent = directories[-1][0]
        leaf = acquire(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, parent)
        initial = os.fstat(leaf)
        _require(stat.S_ISREG(initial.st_mode) and initial.st_nlink == 1, "regular single-link file")
        _require(initial.st_size == expected_size, "exact supplied file size")
        for fd, _, _ in directories:
            _require(stat.S_ISDIR(os.fstat(fd).st_mode), "directory descriptor")
        chunks, offset = [], 0
        while offset < expected_size:
            block = os.pread(leaf, min(1024 * 1024, expected_size - offset), offset)
            _require(
                type(block) is bytes and 0 < len(block) <= expected_size - offset,
                "complete bounded file read progress",
            )
            chunks.append(block)
            offset += len(block)
            record["read_bytes"] = offset
        _require(os.pread(leaf, 1, expected_size) == b"", "exact EOF")
        data = b"".join(chunks)
        _require(hashlib.sha256(data).hexdigest() == expected_sha256, "complete supplied file pin")
        _require(_fingerprint(os.fstat(leaf)) == _fingerprint(initial), "file descriptor metadata drift")
        _require(
            _fingerprint(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)) == _fingerprint(initial),
            "leaf path identity or metadata drift",
        )
        for (fd, parent_fd, component), item in zip(directories, owned):
            current = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            _require(
                stat.S_ISDIR(current.st_mode) and _identity(current) == item["identity"] == _identity(os.fstat(fd)),
                "directory path or descriptor identity drift",
            )
        record.update(
            captured_sha256=expected_sha256,
            captured_bytes=expected_size,
            source_descriptor_identity=list(_identity(initial)),
        )
    except BaseException as exc:
        failure = exc
        record.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        for item in reversed(owned):
            try:
                _close_owned(item["fd"], item["identity"])
            except BaseException as exc:
                item.update(cleanup_error_type=type(exc).__name__, cleanup_error=str(exc))
                failure = failure or exc
            else:
                item.update(closed=True, cleanup_proven=True)
        record["descriptors"] = copy.deepcopy(owned)
        record["cleanup_proven"] = all(item["cleanup_proven"] for item in owned)
    if failure is not None:
        if isinstance(failure, (KeyboardInterrupt, SystemExit)):
            # Keep interruption identity/traceback; attach retained cleanup when possible.
            failure.capture_record = copy.deepcopy(record)
            raise failure
        raise CaptureError(str(failure), record) from failure
    _require(record["cleanup_proven"], "complete owned cleanup")
    record["status"] = "CAPTURED_BYTES_CLOSED_PENDING_GATES"
    return data, record
