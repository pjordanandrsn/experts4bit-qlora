#!/usr/bin/env python3
"""provenance.py -- lane P235's release check (bench/p235/PREREG-p235.md, "Pins and provenance"). A version string is
not a release: each distribution the lane reads must be at its registered version AND come from a release, i.e. a
PyPI index install (no direct_url.json, or one whose url is an https PyPI archive with no vcs_info or dir_info). An
editable install, a file:// or local-directory install, any git install (whatever its URL or commit: a fork, a mirror
or a local repo can carry the release's tag) and any other host are refused. The pins are the constants below, checked
unconditionally.

Usage: provenance.py check NAME=VERSION [NAME=VERSION ...] --out provenance.json
Exit 0 and write {dist: {version, provenance, direct_url}} when every distribution passes; otherwise exit 1, naming
each failure, and write nothing. stdlib only: it runs in the image python and in each vLLM venv."""
import argparse
import json
import sys
from importlib import metadata
from urllib.parse import urlparse

#: the registered pins: e4b and its kernel (image python), and the two vLLM builds (each in its own venv)
E4B = {"experts4bit-qlora": "0.58.0", "grouped-nf4-gemm": "0.51.0"}
VLLM = {"primary": "0.32.0", "secondary": "0.30.0"}
PYPI_HOSTS = ("files.pythonhosted.org", "pypi.org")


def registered(name, version):
    """True when NAME=VERSION is one of this lane's pins."""
    if name in E4B:
        return E4B[name] == version
    if name == "vllm":
        return version in VLLM.values()
    return False


def _find(name, path):
    """The distribution an import would see first (on sys.path, or on `path` when given), or None."""
    if path is None:
        try:
            return metadata.distribution(name)
        except metadata.PackageNotFoundError:
            return None
    return next(iter(metadata.distributions(name=name, path=list(path))), None)


def check_one(name, version, path=None):
    """(record, failures) for one NAME=VERSION."""
    if not registered(name, version):
        return None, [f"{name}=={version} is not a registered pin"]
    dist = _find(name, path)
    if dist is None:
        return None, [f"{name}: not installed"]
    if dist.version != version:
        return None, [f"{name}: installed version {dist.version} is not the registered {version}"]
    raw = dist.read_text("direct_url.json")
    if raw is None:
        return {"version": dist.version, "provenance": "pypi", "direct_url": None}, []
    try:
        du = json.loads(raw)
    except ValueError as e:
        return None, [f"{name}: unreadable direct_url.json ({e})"]
    url = str(du.get("url", ""))
    dir_info = du.get("dir_info")
    if isinstance(dir_info, dict) and dir_info.get("editable"):
        return None, [f"{name}: an editable install ({url}) is not a release"]
    if url.lower().startswith("file:"):
        return None, [f"{name}: a file:// install ({url}) is not a release"]
    if dir_info is not None:
        return None, [f"{name}: a local-directory install ({url}) is not a release"]
    vcs = du.get("vcs_info")
    if vcs is not None:
        commit = vcs.get("commit_id") if isinstance(vcs, dict) else None
        return None, [f"{name}: a git install ({url} @ {commit}) is not an index release"]
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in PYPI_HOSTS:
        return None, [f"{name}: installed from {url}, which is not PyPI"]
    return {"version": dist.version, "provenance": "pypi", "direct_url": du}, []


def check(pins, path=None):
    """(records, failures) for [(name, version), ...]; an empty pin list is a failure."""
    records, failures = {}, []
    if not pins:
        failures.append("no distributions named")
    for name, version in pins:
        rec, errs = check_one(name, version, path)
        failures += errs
        if rec is not None:
            records[name] = rec
    return records, failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("pins", nargs="+", metavar="NAME=VERSION")
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    pins = []
    for p in a.pins:
        name, sep, version = p.partition("=")
        if not sep or not name or not version:
            print(f"PROVENANCE FAIL: {p!r} is not NAME=VERSION")
            return 1
        pins.append((name, version))
    records, failures = check(pins)
    for f in failures:
        print(f"PROVENANCE FAIL: {f}")
    if failures:
        return 1
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(records, fh, indent=1, sort_keys=True)
        fh.write("\n")
    for name, rec in records.items():
        print(f"PROVENANCE OK: {name} {rec['version']} {rec['provenance']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
