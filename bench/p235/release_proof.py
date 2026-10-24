"""Prove an installed distribution IS a published release, by its bytes, not by how it was installed.

    python release_proof.py DIST==VERSION [DIST==VERSION ...]

For each pin: read PyPI's JSON for exactly that version (``https://pypi.org/pypi/<dist>/<version>/json``), take the one
published wheel whose tags are the installed dist's ``WHEEL`` ``Tag:`` lines (or, when no filename carries them, the one
whose python and abi tags are the install's and whose platform is ``linux_<arch>`` or ``many``/``musllinux*_<arch>``
for the install's ``linux_<arch>``: a wheel can be NAMED manylinux while its own ``WHEEL`` says ``linux_<arch>``, as
vLLM's are; the pick only selects, and ``WHEEL`` is itself a compared RECORD entry), stream it to disk with a timeout and check
its sha256 against the digest PyPI lists, then compare the wheel's own ``RECORD`` with the install:

(a) every file the wheel records, bar ``RECORD``/``INSTALLER``/``REQUESTED``/``direct_url.json``, is installed, and the
    bytes ON DISK hash to the wheel's sha256 for it (the disk is read, not the installed ``RECORD``, so a file edited
    after install fails too);
(b) every entry the install's RECORD lists (read as written: ``importlib.metadata``'s ``files()`` drops entries whose
    file is missing on some Python versions and not others) that the wheel does not is either ``__pycache__/*.pyc`` bytecode, or a launcher
    outside site-packages whose name is one of the dist's own ``[console_scripts]``/``[gui_scripts]`` in
    ``entry_points.txt``; any other installed-only file (an added module, a ``.pth``) refuses.

(c) every top-level name the wheel installs IMPORTS from the proven install, in this interpreter as it stands
    (``importlib.util.find_spec``: a package's ``__init__.py``, a module's file, or a namespace package's one search
    location, realpath'd, is the install's file the wheel records), so a checkout earlier on ``sys.path``
    (``PYTHONPATH``, a ``.pth``, ``python -m`` from a repository root) cannot shadow the proven files while the metadata
    still finds them. Call it in the same interpreter, with the same environment, as the read it proves.

Bytecode: (b) allows ``__pycache__/*.pyc``, and CPython loads a ``.pyc`` whose header matches its source's mtime and size
instead of the source. :func:`fresh_pycache` re-executes the caller under ``PYTHONPYCACHEPREFIX`` set to a fresh empty
temporary directory (Python then reads and writes bytecode only there and compiles from the proven sources); the proof
records ``pycache_prefix`` and whether it is that fresh directory.

A wheel with ``.data/`` entries is refused: they install elsewhere and this check does not follow them. A refusal names
the first file that differs, so a wheel built locally at the release's version, a checkout, or an index serving other
bytes never reads as the release.

Standalone and stdlib-only (urllib, hashlib, zipfile, importlib.metadata), so it drops into any kit unchanged:
``prove(dist, version)`` returns ``{"dist", "version", "wheel", "wheel_pick" ("filename" or "arch"), "sha256", "files", "record": "RECORD equal",
"record_equal": True, "first_diff": None, "imports": {name: path}, "pycache_prefix", "pycache_fresh"}`` or raises :class:`ReleaseProofError` (its ``first_diff`` names the file when
there is one). The CLI prints one JSON object per pin and exits 1 with the reason on the first refusal.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import itertools
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from importlib import metadata

PYPI = "https://pypi.org/pypi/{dist}/{version}/json"
#: the .dist-info files pip writes at install time, so they never match a wheel's
PIP_WRITES = {"RECORD", "INSTALLER", "REQUESTED", "direct_url.json"}
TIMEOUT_S = 60
#: set, with PYTHONPYCACHEPREFIX, to the fresh directory fresh_pycache made, so a prefix someone else set never reads as fresh
FRESH_MARK = "RELEASE_PROOF_FRESH_PYCACHE"
CHUNK = 1 << 20


class ReleaseProofError(Exception):
    def __init__(self, message, first_diff=None):
        super().__init__(message)
        self.first_diff = first_diff


def fetch(url):
    """The bytes at ``url`` (a small JSON document); None on a 404."""
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def download(url, dest):
    """Stream ``url`` to the file ``dest`` (each read under the timeout); its sha256 hex, or None on a 404."""
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r, open(dest, "wb") as out:
            for chunk in iter(lambda: r.read(CHUNK), b""):
                h.update(chunk)
                out.write(chunk)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    return h.hexdigest()


def record_hash(path):
    """A RECORD hash for the file at ``path``: ``sha256=`` and the urlsafe base64 digest with no padding."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return "sha256=" + base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode()


def wheel_tags(filename):
    """The set of tags a wheel filename names (its compressed tag sets expanded): ``x-1-py2.py3-none-any.whl`` ->
    {py2-none-any, py3-none-any}."""
    py, abi, plat = filename[: -len(".whl")].split("-")[-3:]
    return {"-".join(t) for t in itertools.product(py.split("."), abi.split("."), plat.split("."))}


def _split(tag):
    py, abi, plat = tag.split("-")
    return py, abi, plat


def arch_candidates(urls, tags):
    """The published wheels the arch fallback may pick for installed ``tags`` whose platforms are all ``linux_<arch>``
    for one arch: the same python and abi tags, and every platform ``linux_<arch>`` or ``manylinux*_<arch>`` /
    ``musllinux*_<arch>``. Empty when the install is not ``linux_<arch>``."""
    parts = [_split(t) for t in tags]
    plats = {p for _, _, p in parts}
    if len(plats) != 1 or not next(iter(plats)).startswith("linux_"):
        return []
    arch = next(iter(plats))[len("linux_"):]
    same_arch = re.compile(rf"(linux|(many|musl)linux(1|2010|2014|_\d+_\d+))_{re.escape(arch)}")
    py_abi = {(py, abi) for py, abi, _ in parts}
    out = []
    for u in urls:
        ft = [_split(t) for t in wheel_tags(u["filename"])]
        if {(py, abi) for py, abi, _ in ft} == py_abi and all(same_arch.fullmatch(p) for _, _, p in ft):
            out.append(u)
    return out


def installed_tags(dist):
    text = dist.read_text("WHEEL") or ""
    return {line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("Tag:")}


def launcher_names(dist):
    """The launcher names pip generates for ``dist``: its ``[console_scripts]`` and ``[gui_scripts]`` entry points."""
    return {e.name for e in dist.entry_points if e.group in ("console_scripts", "gui_scripts")}


def generated_by_pip(path, launchers):
    """True for an installed-only entry pip writes or generates from compared files, which (b) allows."""
    p = str(path)
    parts = p.split("/")
    if p.startswith("../"):                 # outside site-packages: only a launcher entry_points.txt names
        name = parts[-1]
        return name in launchers or (name.endswith(".exe") and name[:-4] in launchers)
    if "__pycache__" in parts[:-1]:
        return parts[-1].endswith(".pyc")   # bytecode pip compiles from compared sources
    return len(parts) == 2 and parts[0].endswith(".dist-info") and parts[1] in PIP_WRITES


def _record_paths(text):
    return [row[0] for row in csv.reader(io.StringIO(text)) if row]


def published_record(wheel_path, filename):
    """{path: "sha256=..."} from the wheel's own RECORD, bar the files pip rewrites."""
    with zipfile.ZipFile(wheel_path) as z:
        records = [n for n in z.namelist() if n.count("/") == 1 and n.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise ReleaseProofError(f"{filename} carries {len(records)} RECORD files, not one")
        rows = list(csv.reader(io.StringIO(z.read(records[0]).decode("utf-8"))))
    out = {}
    for row in rows:
        if not row:
            continue
        path, h = row[0], (row[1] if len(row) > 1 else "")
        top = path.split("/")[0]
        if top.endswith(".data"):
            raise ReleaseProofError(f"{filename} has .data entries ({path}), which this check does not follow", path)
        if top.endswith(".dist-info") and path.count("/") == 1 and path.split("/")[1] in PIP_WRITES:
            continue
        if not h.startswith("sha256="):
            raise ReleaseProofError(f"{filename}'s RECORD gives {path} no sha256", path)
        out[path] = h
    return out


def fresh_pycache(argv=None, execve=os.execve):
    """Return the bytecode prefix when this process already runs under a fresh one this function made; otherwise
    re-execute ``[sys.executable] + argv`` (default ``sys.argv``) with ``PYTHONPYCACHEPREFIX`` and :data:`FRESH_MARK` set to
    a new empty temporary directory. Call it first thing, before the code under proof is imported."""
    if sys.pycache_prefix and os.environ.get(FRESH_MARK) == sys.pycache_prefix:
        return sys.pycache_prefix
    d = tempfile.mkdtemp(prefix="release-proof-pycache-")
    execve(sys.executable, [sys.executable, *(sys.argv if argv is None else argv)],
           {**os.environ, "PYTHONPYCACHEPREFIX": d, FRESH_MARK: d})
    return d                                # only reached when ``execve`` is a stand-in


def top_level(published):
    """{import name: (the RECORD path its spec must resolve to, or None for a namespace package; its top path)} for
    every top-level package or module the wheel installs."""
    out = {}
    for path in published:
        parts = path.split("/")
        top = parts[0]
        if len(parts) > 1:
            if top.endswith((".dist-info", ".data")) or top in out:
                continue
            out[top] = (f"{top}/__init__.py" if f"{top}/__init__.py" in published else None, top)
        elif top.endswith(".py"):
            out[top[:-3]] = (top, top)
        elif top.endswith((".so", ".pyd")):
            out[top.split(".")[0]] = (top, top)
    return out


def check_imports(dist_name, dist, published, find_spec=importlib.util.find_spec):
    """{name: origin} when every top-level name the wheel installs imports from the proven install (see (c))."""
    out = {}
    for name, (rel, top) in sorted(top_level(published).items()):
        try:
            spec = find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            raise ReleaseProofError(f"{dist_name}: {name} is not importable here", top)
        if rel is not None:
            origin = spec.origin and os.path.realpath(spec.origin)
            if origin != os.path.realpath(dist.locate_file(rel)):
                raise ReleaseProofError(f"{dist_name}: {name} imports from {spec.origin}, not the proven install", rel)
        else:
            locs = [os.path.realpath(x) for x in (spec.submodule_search_locations or [])]
            origin = os.path.realpath(dist.locate_file(top))
            if locs != [origin]:
                raise ReleaseProofError(f"{dist_name}: namespace {name} searches {locs}, not only the proven install", top)
        out[name] = origin
    return out


def prove(dist_name, version, get=fetch, fetch_to=download, distribution=metadata.distribution,
          find_spec=importlib.util.find_spec):
    """The proof that the installed ``dist_name`` is byte-for-byte the wheel PyPI publishes for ``version`` (see the
    module docstring), or raises :class:`ReleaseProofError`."""
    try:
        dist = distribution(dist_name)
    except metadata.PackageNotFoundError:
        raise ReleaseProofError(f"{dist_name} is not installed") from None
    if dist.version != version:
        raise ReleaseProofError(f"{dist_name} is installed at {dist.version}, not {version}")
    raw = get(PYPI.format(dist=dist_name, version=version))
    if raw is None:
        raise ReleaseProofError(f"{dist_name} {version} is not on PyPI")
    tags = installed_tags(dist)
    if not tags:
        raise ReleaseProofError(f"{dist_name}'s installed WHEEL names no Tag")
    urls = [u for u in json.loads(raw)["urls"] if u["packagetype"] == "bdist_wheel"]
    wheels, pick = [u for u in urls if wheel_tags(u["filename"]) == tags], "filename"
    if not wheels:
        wheels, pick = arch_candidates(urls, tags), "arch"
    if len(wheels) != 1:
        raise ReleaseProofError(f"{dist_name} {version}: {len(wheels)} published wheels carry the installed tags "
                                f"{sorted(tags)}" + (" or match their python, abi and arch" if pick == "arch" else "")
                                + ", not exactly one")
    w = wheels[0]
    with tempfile.TemporaryDirectory(prefix="release-proof-") as tmp:
        path = os.path.join(tmp, w["filename"])
        got = fetch_to(w["url"], path)
        if got != w["digests"]["sha256"]:
            raise ReleaseProofError(f"{w['filename']} as downloaded is not the sha256 PyPI lists")
        published = published_record(path, w["filename"])
    for p, h in sorted(published.items()):
        f = dist.locate_file(p)
        if not os.path.isfile(f):
            raise ReleaseProofError(f"{dist_name}: {p} is in the published wheel but not installed", p)
        if record_hash(f) != h:
            raise ReleaseProofError(f"{dist_name}: {p} differs from the published wheel's", p)
    installed = dist.read_text("RECORD")
    if installed is None:
        raise ReleaseProofError(f"{dist_name} has no installed RECORD")
    launchers = launcher_names(dist)
    extra = sorted(p for p in _record_paths(installed) if p not in published and not generated_by_pip(p, launchers))
    if extra:
        raise ReleaseProofError(f"{dist_name}: {extra[0]} is installed but not in the published wheel", extra[0])
    imports = check_imports(dist_name, dist, published, find_spec)
    return {"dist": dist_name, "version": version, "wheel": w["filename"], "wheel_pick": pick, "sha256": w["digests"]["sha256"],
            "files": len(published), "record": "RECORD equal", "record_equal": True, "first_diff": None, "imports": imports,
            "pycache_prefix": sys.pycache_prefix,
            "pycache_fresh": bool(sys.pycache_prefix) and os.environ.get(FRESH_MARK) == sys.pycache_prefix}


def main(argv):
    if not argv or any("==" not in a for a in argv):
        print("usage: python release_proof.py DIST==VERSION [DIST==VERSION ...]", file=sys.stderr)
        return 2
    for pin in argv:
        dist, version = pin.split("==", 1)
        try:
            print(json.dumps(prove(dist, version), sort_keys=True))
        except ReleaseProofError as e:
            print(f"refused: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
