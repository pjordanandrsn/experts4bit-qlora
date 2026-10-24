"""Prove an installed distribution IS a published release, by its bytes, not by how it was installed.

    python release_proof.py DIST==VERSION [DIST==VERSION ...]

For each pin: read PyPI's JSON for exactly that version (``https://pypi.org/pypi/<dist>/<version>/json``), take the one
published wheel whose tags are the installed dist's ``WHEEL`` ``Tag:`` lines, stream it to disk with a timeout and check
its sha256 against the digest PyPI lists, then compare the wheel's own ``RECORD`` with the install:

(a) every file the wheel records, bar ``RECORD``/``INSTALLER``/``REQUESTED``/``direct_url.json``, is installed, and the
    bytes ON DISK hash to the wheel's sha256 for it (the disk is read, not the installed ``RECORD``, so a file edited
    after install fails too);
(b) every entry the install's RECORD lists (read as written: ``importlib.metadata``'s ``files()`` drops entries whose
    file is missing on some Python versions and not others) that the wheel does not is either ``__pycache__/*.pyc`` bytecode, or a launcher
    outside site-packages whose name is one of the dist's own ``[console_scripts]``/``[gui_scripts]`` in
    ``entry_points.txt``; any other installed-only file (an added module, a ``.pth``) refuses.

A wheel with ``.data/`` entries is refused: they install elsewhere and this check does not follow them. A refusal names
the first file that differs, so a wheel built locally at the release's version, a checkout, or an index serving other
bytes never reads as the release.

Standalone and stdlib-only (urllib, hashlib, zipfile, importlib.metadata), so it drops into any kit unchanged:
``prove(dist, version)`` returns ``{"dist", "version", "wheel", "sha256", "files", "record": "RECORD equal",
"record_equal": True, "first_diff": None}`` or raises :class:`ReleaseProofError` (its ``first_diff`` names the file when
there is one). The CLI prints one JSON object per pin and exits 1 with the reason on the first refusal.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import itertools
import json
import os
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


def prove(dist_name, version, get=fetch, fetch_to=download, distribution=metadata.distribution):
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
    wheels = [u for u in json.loads(raw)["urls"] if u["packagetype"] == "bdist_wheel" and wheel_tags(u["filename"]) == tags]
    if len(wheels) != 1:
        raise ReleaseProofError(f"{dist_name} {version}: {len(wheels)} published wheels carry the installed tags "
                                f"{sorted(tags)}, not exactly one")
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
    return {"dist": dist_name, "version": version, "wheel": w["filename"], "sha256": w["digests"]["sha256"],
            "files": len(published), "record": "RECORD equal", "record_equal": True, "first_diff": None}


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
