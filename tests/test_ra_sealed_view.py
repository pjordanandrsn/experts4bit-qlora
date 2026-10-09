"""Supplied-byte/cleanup controls; simulated kernel and actual child separated."""

import errno
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fresh():
    return load("ra_sealed_view_control", ROOT / "bench/ra/ra_sealed_view.py")


DATA = b"synthetic captured copy\x00tail"
SHA = hashlib.sha256(DATA).hexdigest()


@pytest.mark.parametrize("data", [None, "text", True, bytearray(b"x"), memoryview(b"x"), b""])
def test_exact_payload_type_and_nonempty(data):
    with pytest.raises(ValueError, match="nonempty bytes"):
        fresh().validate_bytes(data, SHA)


@pytest.mark.parametrize("sha", [None, True, 0, "", "0" * 63, "0" * 65, "F" * 64, "g" * 64, "0" * 64])
def test_full_typed_pin(sha):
    with pytest.raises(ValueError):
        fresh().validate_bytes(DATA, sha)


def test_complete_tail_and_oversized_payload():
    gate = fresh()
    with pytest.raises(ValueError, match="complete supplied byte pin"):
        gate.validate_bytes(DATA[:-1] + b"X", SHA)
    with pytest.raises(ValueError, match="bounded"):
        gate.validate_bytes(b"x" * (gate.LIMIT + 1), SHA)
    assert gate.validate_bytes(DATA, SHA) == len(DATA)


class KernelDouble:
    """Explicit stateful syscall facade; never patch the real os/fcntl modules."""

    F_GETFD = 1
    O_RDONLY, O_CLOEXEC, SEEK_SET = 0, 0x80000, 0

    def __init__(self):
        self.fds, self.next, self.fail, self.closed = {}, 20, None, []
        self.calls = []

    def trip(self, name):
        self.calls.append(name)
        if self.fail == name:
            raise OSError(errno.EIO, "explicit simulated " + name)

    def create(self):
        self.trip("create")
        fd = self.next
        self.next += 1
        self.fds[fd] = {"data": bytearray(), "inode": fd, "seals": 0, "inheritable": False}
        return fd

    def fstat(self, fd):
        self.trip("stat")
        if fd not in self.fds:
            raise OSError(errno.EBADF, "closed")
        v = self.fds[fd]
        return SimpleNamespace(st_dev=2, st_ino=v["inode"], st_size=len(v["data"]), st_mode=stat.S_IFREG)

    def write(self, fd, data):
        self.trip("write")
        if self.fail == "zero_write":
            return 0
        if self.fail == "oversized_write":
            return len(data) + 1
        n = min(3, len(data))  # partial positive writes must complete
        self.fds[fd]["data"].extend(data[:n])
        return n

    def lseek(self, fd, position, whence):
        self.trip("rewind")
        return 1 if self.fail == "bad_rewind" else 0

    def fcntl(self, fd, command, argument=None):
        if fd not in self.fds:
            raise OSError(errno.EBADF, "closed")
        if command == 1033:
            self.trip("seal")
            self.fds[fd]["seals"] = argument
        elif command == 1034:
            self.trip("get_seals")
            return 7 if self.fail == "missing_seal" else self.fds[fd]["seals"]
        return 0

    def get_inheritable(self, fd):
        self.trip("inheritable")
        return self.fail == "inherited" or self.fds[fd]["inheritable"]

    def open(self, path, flags):
        self.trip("open")
        origin = int(path.rsplit("/", 1)[-1])
        fd = self.next
        self.next += 1
        self.fds[fd] = self.fds[origin]
        if self.fail == "wrong_reader":
            self.fds[fd] = {**self.fds[origin], "inode": fd}
        return fd

    def pread(self, fd, size, offset):
        self.trip("read")
        if self.fail == "short_read":
            return b""
        data = bytes(self.fds[fd]["data"])
        if self.fail == "changed_tail":
            data = data[:-1] + b"!"
        if self.fail == "extra_read" and offset == len(data):
            return b"!"
        return data[offset : offset + size]

    def close(self, fd):
        self.trip("reader_close" if fd != 20 else "close")
        self.closed.append(fd)
        if self.fail != "close_noop":
            del self.fds[fd]


@pytest.fixture
def simulated(monkeypatch):
    gate, kernel = fresh(), KernelDouble()
    monkeypatch.setattr(gate, "require_environment", lambda: None)
    monkeypatch.setattr(gate, "_create", kernel.create)
    monkeypatch.setattr(gate, "os", kernel)
    monkeypatch.setattr(gate, "fcntl", kernel)
    return gate, kernel


def test_simulated_complete_readback_close_and_snapshot(simulated):
    gate, k = simulated
    view = gate.CapturedView(DATA, SHA)
    assert len([c for c in k.calls if c == "write"]) > 1
    assert view.check()["sha256"] == SHA
    snapshot = view.record
    snapshot["source_authenticated"] = True
    assert not view.record["source_authenticated"]
    result = view.close()
    assert result["cleanup_proven"] and result["reader_cleanup_proven"]
    assert result["status"] == "SEALED_CAPTURED_COPY_CLOSED_PENDING_GATES"
    assert not k.fds
    for operation in (view.check, view.close, lambda: view.path):
        with pytest.raises(ValueError, match="terminal"):
            operation()


@pytest.mark.parametrize(
    "fault",
    [
        "create",
        "write",
        "zero_write",
        "oversized_write",
        "rewind",
        "bad_rewind",
        "seal",
        "get_seals",
        "missing_seal",
        "inheritable",
        "inherited",
        "open",
        "read",
        "short_read",
        "changed_tail",
        "extra_read",
        "wrong_reader",
    ],
)
def test_simulated_construction_refusal_cleans_known_identity(simulated, fault):
    gate, k = simulated
    k.fail = fault
    with pytest.raises(gate.ViewError) as failure:
        gate.CapturedView(DATA, SHA)
    assert failure.value.record["status"] == "FAILED"
    assert failure.value.record["cleanup_proven"]
    assert not k.fds


@pytest.mark.parametrize("fault", ["stat", "close", "reader_close", "close_noop"])
def test_simulated_unknown_or_failed_cleanup_never_complete(simulated, fault):
    gate, k = simulated
    if fault in ("stat", "reader_close", "close_noop"):
        k.fail = fault
        with pytest.raises(gate.ViewError) as failure:
            gate.CapturedView(DATA, SHA)
        record = failure.value.record
    else:
        view = gate.CapturedView(DATA, SHA)
        k.fail = fault
        with pytest.raises(gate.ViewError) as failure:
            view.close()
        record = failure.value.record
    assert record["status"] == "FAILED"
    assert not record["cleanup_proven"]
    assert k.fds  # retained explicit double descriptors, not real leaked descriptors


def test_simulated_replacement_not_closed_and_refusal_terminal(simulated):
    gate, k = simulated
    view = gate.CapturedView(DATA, SHA)
    k.fds[20] = {**k.fds[20], "inode": 99}
    with pytest.raises(gate.ViewError, match="identity drift") as failure:
        view.check()
    assert not failure.value.record["cleanup_proven"]
    assert 20 in k.fds and 20 not in k.closed
    with pytest.raises(ValueError, match="terminal"):
        view.close()


@pytest.mark.parametrize("fault", [None, "close", "reader_close"])
def test_simulated_context_preserves_original_exception(simulated, fault):
    gate, k = simulated
    view = gate.CapturedView(DATA, SHA)
    original = RuntimeError("original caller failure")
    with pytest.raises(RuntimeError) as failure:
        with view:
            k.fail = fault
            raise original
    assert failure.value is original
    assert view.record["status"] == "FAILED"
    assert view.record["cleanup_proven"] is (fault is None)


def test_real_host_environment_refuses_before_allocation(monkeypatch):
    gate = fresh()
    reached = []
    monkeypatch.setattr(gate, "_create", lambda: reached.append(True))
    if sys.platform == "linux" and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode:
        # pytest normally lacks these flags; if supplied, exercise a real flag refusal.
        monkeypatch.setattr(gate, "sys", SimpleNamespace(platform="unsupported"))
    with pytest.raises(gate.ViewError, match="Linux x86_64") as failure:
        gate.CapturedView(DATA, SHA)
    assert not reached and failure.value.record["cleanup_proven"]


KERNEL_SCRIPT = r"""
import errno, fcntl, hashlib, importlib.util, json, os, sys
spec = importlib.util.spec_from_file_location('sealed_control_child', sys.argv[1])
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
controls = []
def demand(ok, name):
    if not ok: raise ValueError('kernel control: ' + name)
    controls.append(name)
data = b'synthetic kernel copy\0' + bytes(range(256)) * 4097
sha = hashlib.sha256(data).hexdigest()
v = g.CapturedView(data, sha)
fd = int(v.path.rsplit('/', 1)[-1])
demand(v.check()['sha256'] == sha, 'full readback')
demand(fcntl.fcntl(fd, g.F_GET_SEALS) == 15, 'all four seals')
demand(not os.get_inheritable(fd), 'close on exec')
for kind in ('owned', 'duplicated', 'reopened'):
    target = fd if kind == 'owned' else os.dup(fd) if kind == 'duplicated' else os.open(v.path, os.O_RDWR | os.O_CLOEXEC)
    try:
        for operation in ('write', 'shrink', 'grow'):
            try:
                if operation == 'write': os.pwrite(target, b'X', 20)
                else: os.ftruncate(target, len(data) + (1 if operation == 'grow' else -1))
            except OSError as exc:
                demand(exc.errno == errno.EPERM, kind + ':' + operation + ':EPERM')
            else: raise ValueError('kernel control allowed mutation')
    finally:
        if target != fd: os.close(target)
demand(v.check()['sha256'] == sha, 'unchanged after mutation attempts')
r = v.close()
demand(r['cleanup_proven'] and r['owned_descriptor_closed'], 'complete owned cleanup')
try: os.fstat(fd)
except OSError as exc: demand(exc.errno == errno.EBADF, 'terminal EBADF')
else: raise ValueError('kernel control still open')
for name in ('check', 'close', 'path'):
    try: getattr(v, name)() if name != 'path' else v.path
    except ValueError: demand(True, 'terminal:' + name)
    else: raise ValueError('kernel control retry allowed')
# Both replacement and displaced original are OWN controls, closed by controller.
v = g.CapturedView(data, sha)
fd = int(v.path.rsplit('/', 1)[-1])
original = os.dup(fd)
replacement = g.CapturedView(data, sha)
replacement_fd = int(replacement.path.rsplit('/', 1)[-1])
os.dup2(replacement_fd, fd, inheritable=False)
try:
    try: v.check()
    except g.ViewError as exc:
        demand(not exc.record['cleanup_proven'], 'replacement cleanup unproven')
        demand('identity drift' in str(exc), 'same size same bytes replacement refused')
    else: raise ValueError('kernel control replacement accepted')
    demand(os.fstat(fd).st_ino == os.fstat(replacement_fd).st_ino, 'replacement preserved')
finally:
    os.close(fd)
    os.close(original)
    replacement.close()
# Partial write/seal/read and cleanup injections alter only helper-private facade.
for fault in ('write', 'seal', 'read'):
    made = []
    create = g._create
    def capture():
        fd = create(); made.append(fd); return fd
    g._create = capture
    real = g.os if fault != 'seal' else g.fcntl
    writes = []
    class Facade:
        def __getattr__(self, key):
            if key == {'write':'write', 'read':'pread', 'seal':'fcntl'}[fault]:
                def refuse(*args):
                    if fault == 'seal' and args[1] != g.F_ADD_SEALS: return real.fcntl(*args)
                    if fault == 'write' and not writes:
                        writes.append(True); return real.write(args[0], args[1][:3])
                    raise OSError(errno.EIO, 'explicit owned helper fault ' + fault)
                return refuse
            return getattr(real, key)
    if fault == 'seal': g.fcntl = Facade()
    else: g.os = Facade()
    try:
        try: g.CapturedView(data, sha)
        except g.ViewError as exc:
            demand(exc.record['cleanup_proven'], fault + ':partial cleanup')
        else: raise ValueError('kernel control fault accepted')
        try: os.fstat(made[0])
        except OSError as exc: demand(exc.errno == errno.EBADF, fault + ':EBADF')
        else: raise ValueError('kernel control partial descriptor retained')
    finally:
        g._create = create
        if fault == 'seal': g.fcntl = real
        else: g.os = real
# Context preserves the original caller exception while owned close failure
# is explicitly unproven. Controller closes only its verified own descriptor.
v = g.CapturedView(data, sha)
fd = int(v.path.rsplit('/', 1)[-1])
real = g.os
class CloseFault:
    def __getattr__(self, key):
        if key == 'close':
            def close(target):
                if target == fd: raise OSError(errno.EIO, 'owned close fault')
                return real.close(target)
            return close
        return getattr(real, key)
original_error = RuntimeError('original caller')
try:
    try:
        with v:
            g.os = CloseFault()
            raise original_error
    except RuntimeError as exc:
        demand(exc is original_error, 'original context exception preserved')
    demand(not v.record['cleanup_proven'] and v.record['status'] == 'FAILED', 'context cleanup failure retained')
    demand(os.fstat(fd).st_size == len(data), 'failed close retained owned descriptor')
finally:
    g.os = real
    os.close(fd)
# Reader close failure never becomes complete cleanup even when owner closes.
opened = []
class ReaderFault:
    def __getattr__(self, key):
        if key == 'open':
            def open(*args):
                fd = real.open(*args); opened.append(fd); return fd
            return open
        if key == 'close':
            def close(fd):
                if fd in opened: raise OSError(errno.EIO, 'owned reader close fault')
                return real.close(fd)
            return close
        return getattr(real, key)
g.os = ReaderFault()
try:
    try: g.CapturedView(data, sha)
    except g.ViewError as exc:
        demand(exc.record['owned_descriptor_closed'], 'reader failure owner closed')
        demand(not exc.record['cleanup_proven'], 'reader failure aggregate cleanup unproven')
        demand(exc.record['reader_cleanup_proven'] is False, 'reader refusal retained')
    else: raise ValueError('kernel control reader close failure accepted')
finally:
    g.os = real
    for fd in opened: os.close(fd)
print(json.dumps({'controls': controls, 'passed':len(controls), 'payload_bytes':len(data), 'sha256':sha,
                  'scope':'stdlib owned kernel controls; no tokenizer/source/runtime authority'}, sort_keys=True))
"""


def test_actual_child_platform_refusal_or_kernel_controls(tmp_path):
    env_gate = load("sealed_env_control", ROOT / "bench/ra/ra_env.py")
    env, _ = env_gate.clean(
        os.environ,
        component="training",
        fixture={},
        venv=Path(sys.prefix),
        cache=tmp_path,
        threads=1,
        allocator="expandable_segments:True",
    )
    if sys.platform == "linux" and os.uname().machine == "x86_64":
        code = KERNEL_SCRIPT
    else:
        code = """import importlib.util, sys
s = importlib.util.spec_from_file_location('sealed_child', sys.argv[1])
g = importlib.util.module_from_spec(s); s.loader.exec_module(g)
try: g.CapturedView(b'x', '2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881')
except g.ViewError as e:
    if 'Linux x86_64' not in str(e) or not e.record['cleanup_proven']: raise
else: raise ValueError('unsupported host accepted')
print('explicit unsupported platform refusal')
"""
    child = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code, str(ROOT / "bench/ra/ra_sealed_view.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert child.returncode == 0, child.stdout + child.stderr
