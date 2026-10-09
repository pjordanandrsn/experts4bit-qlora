"""Unwired sealed captured-copy helper; supplied bytes carry no source authority."""

import copy
import ctypes
import errno
import fcntl
import hashlib
import os
import stat
import sys

LIMIT = 32 * 1024 * 1024
# Linux v6.6 x86_64 UAPI: fcntl.h, memfd.h and syscall_64.tbl.
# Explicit constants also support selected Python builds without seal names.
F_ADD_SEALS, F_GET_SEALS, SEALS = 1033, 1034, 15


class ViewError(ValueError):
    """Refusal with a snapshot, including partial construction/cleanup failures."""

    def __init__(self, message, record):
        super().__init__("sealed-view: " + message)
        self.record = copy.deepcopy(record)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_bytes(data, expected_sha256):
    """Check one complete supplied payload, without authenticating its source."""
    require(type(data) is bytes and 0 < len(data) <= LIMIT, "bounded nonempty bytes")
    require(
        type(expected_sha256) is str
        and len(expected_sha256) == 64
        and all(c in "0123456789abcdef" for c in expected_sha256),
        "typed lowercase SHA256",
    )
    require(hashlib.sha256(data).hexdigest() == expected_sha256, "complete supplied byte pin")
    return len(data)


def require_environment():
    require(
        sys.platform == "linux"
        and os.uname().machine == "x86_64"
        and ctypes.sizeof(ctypes.c_void_p) == 8
        and sys.flags.isolated
        and sys.flags.no_site
        and sys.dont_write_bytecode,
        "Linux x86_64 -I -S -B required",
    )


def _create():
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.syscall
    call.argtypes = [ctypes.c_long, ctypes.c_char_p, ctypes.c_uint]
    call.restype = ctypes.c_long
    descriptor = call(319, b"ra-sealed-captured-copy", 3)  # CLOEXEC | ALLOW_SEALING
    if descriptor < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return descriptor


def _identity(descriptor):
    value = os.fstat(descriptor)
    require(stat.S_ISREG(value.st_mode), "regular captured descriptor")
    return value.st_dev, value.st_ino


def _close_owned(descriptor, identity):
    # Cooperating process only: this check cannot eliminate same-process races.
    require(_identity(descriptor) == identity, "descriptor ownership drift; cleanup unproven")
    os.close(descriptor)
    try:
        fcntl.fcntl(descriptor, fcntl.F_GETFD)
    except OSError as exc:
        require(exc.errno == errno.EBADF, "close readback must be EBADF")
    else:
        raise ValueError("closed descriptor still present; cleanup unproven")


class CapturedView:
    """One bounded immutable copy. No constructor, role map, guard or CLI wiring.

    The original payload's provenance and downstream reads are separate gates.
    The caller must serialize all descriptor operations in its owned process.
    """

    def __init__(self, data, expected_sha256):
        self._fd, self._identity, self._terminal = None, None, False
        self._record = {
            "schema": 1,
            "status": "FAILED",
            "owned_descriptor_closed": False,
            "cleanup_proven": False,
            "copy_scope": "CAPTURED_COPY_ONLY",
            "source_authenticated": False,
            "runtime_authenticated": False,
            "native_read_authenticated": False,
            "consumer_verified": False,
        }
        try:
            self._size = validate_bytes(data, expected_sha256)
            self._sha = expected_sha256
            self._record.update(captured_sha256=self._sha, captured_bytes=self._size)
            require_environment()
            self._fd = _create()
            require(type(self._fd) is int and self._fd >= 0, "owned descriptor allocation")
            self._identity = _identity(self._fd)  # before partial writes/sealing
            offset = 0
            while offset < self._size:
                count = os.write(self._fd, data[offset:])
                require(type(count) is int and 0 < count <= self._size - offset, "complete capture write progress")
                offset += count
            require(os.lseek(self._fd, 0, os.SEEK_SET) == 0, "capture rewind")
            fcntl.fcntl(self._fd, F_ADD_SEALS, SEALS)
            self.check()
            self._record.update(status="SEALED_CAPTURED_COPY_OPEN", kernel_seals=SEALS)
        except BaseException as exc:
            self._fail(exc)
            self._finish()
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise ViewError(str(exc), self._record) from exc

    @property
    def path(self):
        require(not self._terminal and self._fd is not None, "terminal captured view")
        return "/proc/self/fd/" + str(self._fd)

    @property
    def record(self):
        return copy.deepcopy(self._record)

    def _fail(self, exc):
        self._record.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))

    def _finish(self):
        if self._terminal:
            return
        self._terminal = True  # refusal/close is terminal; never retry
        if self._fd is None:
            self._record.update(
                cleanup_proven=self._record.get("reader_cleanup_proven") is not False, owned_descriptor_closed=True
            )
            return
        try:
            require(self._identity is not None, "allocation identity unavailable; cleanup unproven")
            _close_owned(self._fd, self._identity)
        except BaseException as exc:
            self._record.update(status="FAILED", cleanup_error_type=type(exc).__name__, cleanup_error=str(exc))
        else:
            self._fd = None
            self._record.update(
                cleanup_proven=self._record.get("reader_cleanup_proven") is not False, owned_descriptor_closed=True
            )

    def _check_descriptor(self, descriptor):
        require(_identity(descriptor) == self._identity, "captured descriptor identity drift")
        require(os.fstat(descriptor).st_size == self._size, "exact captured size")
        require(fcntl.fcntl(descriptor, F_GET_SEALS) == SEALS, "all four kernel seals")
        require(os.get_inheritable(descriptor) is False, "close-on-exec descriptor")

    def check(self):
        require(not self._terminal and self._fd is not None, "terminal captured view")
        reader, identity, failure = None, None, None
        try:
            self._check_descriptor(self._fd)
            reader = os.open(self.path, os.O_RDONLY | os.O_CLOEXEC)
            identity = _identity(reader)
            self._check_descriptor(reader)
            digest, offset = hashlib.sha256(), 0
            while offset < self._size:
                block = os.pread(reader, min(1024 * 1024, self._size - offset), offset)
                require(
                    type(block) is bytes and 0 < len(block) <= self._size - offset, "complete captured read progress"
                )
                digest.update(block)
                offset += len(block)
            require(
                os.pread(reader, 1, self._size) == b"" and digest.hexdigest() == self._sha,
                "complete captured readback pin",
            )
            self._check_descriptor(reader)
            self._check_descriptor(self._fd)
        except BaseException as exc:
            failure = exc
        finally:
            if reader is not None:
                try:
                    require(identity is not None, "reader identity unavailable; cleanup unproven")
                    _close_owned(reader, identity)
                except BaseException as exc:
                    self._record.update(reader_cleanup_proven=False, reader_cleanup_error=str(exc))
                    failure = failure or exc
                else:
                    self._record["reader_cleanup_proven"] = True
        if failure is not None:
            self._fail(failure)
            self._finish()
            if isinstance(failure, (KeyboardInterrupt, SystemExit)):
                raise failure
            raise ViewError(str(failure), self._record) from failure
        return {"sha256": self._sha, "bytes": self._size, "kernel_seals": SEALS}

    def close(self):
        require(not self._terminal and self._fd is not None, "terminal captured view")
        failure = None
        try:
            self.check()
        except BaseException as exc:
            failure = exc
        self._finish()
        if failure is not None or not self._record["cleanup_proven"]:
            raise ViewError(str(failure or self._record.get("cleanup_error")), self._record) from failure
        if self._record["status"] != "FAILED":
            self._record["status"] = "SEALED_CAPTURED_COPY_CLOSED_PENDING_GATES"
        return self.record

    def __enter__(self):
        self.check()
        return self

    def __exit__(self, kind, value, traceback):
        if value is not None:
            self._fail(value)
        try:
            self.close()
        except BaseException:
            if value is None:
                raise
            # Keep the caller's original exception and its traceback. Failed
            # cleanup remains visible through view.record; never claim PASS.
        return False
