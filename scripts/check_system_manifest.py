#!/usr/bin/env python3
"""Validate docs/system-manifest.json -- the cross-repository system manifest --
against THIS repository, and optionally against the sibling repository.

Standard library only. One network read: ``git ls-remote --tags`` of the
kernel repository, for the CI pin check below. Prints one ``OK:`` or ``FAIL:``
line per check; exit 1 when any check failed, 2 when the check itself cannot
run.

Against this repository:
  * the manifest parses and carries every top-level table the scripts read
    (system, packages, capability_ownership, compatibility, evidence_vocabulary,
    authority, invariants, router);
  * ``packages.runtime``: package name and repository match ``[project]`` in
    pyproject.toml; ``import_names`` equals docs/capabilities.json
    ``project.import_names`` and every name is a module this tree ships;
  * exactly one ``compatibility`` record's ``consumer_versions`` range contains
    pyproject's version; its ``kernel`` and ``extra`` name the ``fast`` extra's
    grouped-nf4-gemm requirement and its ``floor`` equals that requirement's
    ``>=`` floor -- the number in the manifest is validated against pyproject,
    never trusted from the manifest;
  * ``evidence_vocabulary`` keys cover docs/claims.json ``status_vocabulary``
    (extras are reported, not failed);
  * ``capability_ownership.runtime`` equals the set of capability ids in
    docs/capabilities.json;
  * every invariant has a non-empty ``id``, ``statement`` and ``checked_by``,
    and ids are unique.

The consumer's CI pin of the kernel package (``.github/workflows/ci.yml``,
``pip install "grouped-nf4-gemm @ git+...@<sha>"``) against the manifest:
  * the pinned sha is the commit of a RELEASE TAG of the kernel package --
    the tag ``compatibility[current].consumer_ci_pin`` names (exact: OK), or a
    later release (OK with a NOTE that the prose is behind); any other sha, or
    a tag below the one named, FAILS. Tags are read with
    ``git ls-remote --tags <repository>`` (annotated tags peeled); with no
    network the check prints ``NOTE: ... skipped`` and never passes silently
    as if it had run -- and under ``--require-tags`` (CI, where the network
    is there) an unreadable tag list is exit 2, so the pin is never left
    unverified on a green run. The version ``consumer_ci_pin`` names is the
    last ``vX.Y.Z`` in its prose (the last bare ``X.Y.Z`` when none carries
    a ``v``), so prose that says what it moved from still names what it
    moved to; the pin line is a non-comment ``pip install`` line;
  * every extra of pyproject that pins the kernel package (``fast``, ``test``,
    ...) has a ``>=`` floor at or above the current record's floor.

``--sibling PATH`` (the kernel repository's checkout; CI clones it at its
latest release tag for this):
  * PATH/docs/system-manifest.json is byte-identical to ours;
  * the sibling's pyproject name is ``packages.kernels.package`` and its version
    satisfies every ``floor`` this manifest names for that package -- the
    ``kernel-first`` invariant (a floor is never raised past a released kernel);
  * when the sibling has docs/capabilities.json, ``capability_ownership.kernels``
    equals its capability ids; when it has docs/claims.json, that register's
    vocabulary is covered by ``evidence_vocabulary`` too.

Version ranges understood (``consumer_versions``, ``floor``): clauses
``>=``, ``<=``, ``>``, ``<``, ``==``, ``!=`` and ``~=`` joined by commas, and
the wildcards ``a.b.x`` / ``a.b.*``; a bare ``x.y.z`` means exactly that
version. Pre-release and local version suffixes are outside this contract.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discovery_common import (  # noqa: E402
    ContractError, load_claims, load_pyproject, module_file, module_shipped, pep503_name, read_text,
)

MANIFEST = "docs/system-manifest.json"
CI_WORKFLOW = ".github/workflows/ci.yml"
CAPABILITIES = "docs/capabilities.json"
CLAIMS = "docs/claims.json"
KERNEL_PACKAGE = "grouped-nf4-gemm"
EXTRA = "fast"
REQUIRED_TOP = ("schema_version", "system", "packages", "capability_ownership", "compatibility",
                "evidence_vocabulary", "authority", "invariants", "router")

_RELEASE = re.compile(r"^\d+(\.\d+)*$")
_CLAUSE = re.compile(r"^(>=|<=|==|!=|~=|>|<)?\s*v?(\d+(?:\.\d+)*(?:\.[x*])?)$")


# ------------------------------------------------------------------ versions --

def parse_version(s: str) -> tuple[int, ...]:
    """``"0.30.0"`` (or ``"v0.30.0"``) -> ``(0, 30, 0)``; anything else raises."""
    s = s.strip()
    if s[:1] == "v":
        s = s[1:]
    if not _RELEASE.match(s):
        raise ValueError(f"not a release version: {s!r}")
    return tuple(int(p) for p in s.split("."))


def _pad(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)), b + (0,) * (n - len(b))


def _cmp(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    a, b = _pad(a, b)
    return (a > b) - (a < b)


def version_in_range(version: str, spec: str) -> bool:
    """Does ``version`` satisfy ``spec``? ``spec`` is comma-joined clauses
    (``">=0.35.0"``, ``">=0.30.0,<0.40"``, ``"0.34.x"``, ``"~=0.34.1"``,
    ``"0.34.0"``). An unsupported clause raises ValueError -- a range this
    checker cannot read must not pass as satisfied."""
    v = parse_version(version)
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        m = _CLAUSE.match(raw)
        if not m:
            raise ValueError(f"unsupported version clause {raw!r} in {spec!r}")
        op, val = m.group(1), m.group(2)
        if val.endswith((".x", ".*")):
            if op not in (None, "=="):
                raise ValueError(f"a wildcard takes no operator: {raw!r}")
            prefix = parse_version(val[:-2])
            vv, _ = _pad(v, prefix)
            if vv[:len(prefix)] != prefix:
                return False
            continue
        w = parse_version(val)
        c = _cmp(v, w)
        if op in (None, "=="):
            ok = c == 0
        elif op == "!=":
            ok = c != 0
        elif op == ">=":
            ok = c >= 0
        elif op == "<=":
            ok = c <= 0
        elif op == ">":
            ok = c > 0
        elif op == "<":
            ok = c < 0
        else:                                   # ~= : >= w and same prefix up to w's second-to-last part
            vv, ww = _pad(v, w)
            ok = c >= 0 and vv[:len(w) - 1] == ww[:len(w) - 1]
        if not ok:
            return False
    return True


def floor_of(spec: str) -> str | None:
    """The version of the single ``>=`` clause in ``spec``; ``None`` when there
    is no ``>=`` clause or more than one."""
    floors = []
    for raw in spec.split(","):
        m = _CLAUSE.match(raw.strip())
        if m and m.group(1) == ">=":
            floors.append(m.group(2))
    return floors[0] if len(floors) == 1 else None


def fast_requirement(pyproject: dict, package: str = KERNEL_PACKAGE, extra: str = EXTRA) -> tuple[str, str] | None:
    """``(requirement, specifier)`` of ``package`` in ``[project.optional-dependencies] <extra>``
    (``("grouped-nf4-gemm>=0.30.0", ">=0.30.0")``), or ``None`` when absent.
    ``pyproject`` is the ``[project]`` table (``load_pyproject``'s shape)."""
    for req in (pyproject.get("optional-dependencies") or {}).get(extra) or []:
        m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req)
        if m and pep503_name(m.group(1)) == pep503_name(package):
            spec = req[m.end():].split(";", 1)[0]
            spec = re.sub(r"^\s*\[[^\]]*\]", "", spec).strip()
            return req.strip(), spec
    return None


def current_record(manifest: dict, consumer: str, version: str) -> tuple[list[dict], str | None]:
    """The ``compatibility`` records for ``consumer`` whose range contains
    ``version`` (``(records, error)``; the error names an unreadable range)."""
    hits = []
    for rec in manifest.get("compatibility") or []:
        if pep503_name(str(rec.get("consumer", ""))) != pep503_name(consumer):
            continue
        try:
            if version_in_range(version, str(rec.get("consumer_versions", ""))):
                hits.append(rec)
        except ValueError as e:
            return [], str(e)
    return hits, None


# -------------------------------------------------------------------- report --

class Report:
    def __init__(self) -> None:
        self.failed = 0

    def ok(self, msg: str) -> None:
        print("OK:", msg)

    def fail(self, msg: str) -> None:
        self.failed += 1
        print("FAIL:", msg)

    def check(self, cond: bool, msg: str, detail: str = "") -> bool:
        (self.ok if cond else self.fail)(msg if cond or not detail else f"{msg}: {detail}")
        return cond


def _github_slug(url: str) -> str:
    m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$", str(url))
    return m.group(1).lower() if m else str(url).lower()


# ---------------------------------------------------------------------- main --

def check_local(root: Path, manifest: dict, rep: Report) -> None:
    py = load_pyproject(root)
    name, version = str(py.get("name", "")), str(py.get("version", ""))
    pk = manifest["packages"]
    runtime, kernels = pk["runtime"], pk["kernels"]

    rep.check(pep503_name(runtime["package"]) == pep503_name(name),
              f"packages.runtime.package {runtime['package']!r} is pyproject's name", f"pyproject name is {name!r}")
    src = (py.get("urls") or {}).get("Source", "")
    rep.check(_github_slug(runtime["repository"]) == _github_slug(src),
              f"packages.runtime.repository is [project.urls] Source ({src})",
              f"manifest says {runtime['repository']!r}")
    cap_path = root / CAPABILITIES
    cap = json.loads(read_text(cap_path))
    cap_imports = list(cap["project"].get("import_names") or [])
    rep.check(list(runtime.get("import_names") or []) == cap_imports,
              f"packages.runtime.import_names == {CAPABILITIES} project.import_names {cap_imports}",
              f"manifest says {runtime.get('import_names')!r}")
    for mod in runtime.get("import_names") or []:
        rep.check(module_shipped(mod, py) and module_file(root, mod, py) is not None,
                  f"import name {mod!r} is a module this tree ships")
    direction = list(manifest["system"].get("dependency_direction") or [])
    rep.check(f"{runtime['package']} -> {kernels['package']}" in direction,
              "system.dependency_direction names runtime -> kernels", f"{direction!r}")

    fr = fast_requirement(py, kernels["package"])
    if not rep.check(fr is not None, f"pyproject [{EXTRA}] extra requires {kernels['package']}"):
        return
    req, spec = fr
    py_floor = floor_of(spec)
    rep.check(py_floor is not None, f"pyproject {EXTRA} requirement {req!r} has a single >= floor")
    hits, err = current_record(manifest, runtime["package"], version)
    if err:
        rep.fail(f"compatibility: unreadable consumer_versions range: {err}")
        return
    if not rep.check(len(hits) == 1,
                     f"exactly one compatibility record contains {runtime['package']} {version}",
                     f"{len(hits)} records match: {[h.get('consumer_versions') for h in hits]}"):
        return
    rec = hits[0]
    rep.check(pep503_name(str(rec.get("kernel", ""))) == pep503_name(kernels["package"])
              and rec.get("extra") == EXTRA,
              f"record {rec.get('consumer_versions')!r} names kernel {kernels['package']!r} via extra {EXTRA!r}",
              f"kernel={rec.get('kernel')!r} extra={rec.get('extra')!r}")
    rec_floor = floor_of(str(rec.get("floor", "")))
    rep.check(rec_floor is not None and py_floor is not None and parse_version(rec_floor) == parse_version(py_floor),
              f"record {rec.get('consumer_versions')!r} floor {rec.get('floor')!r} equals pyproject's {req!r}",
              f"manifest floor {rec.get('floor')!r} vs pyproject {spec!r}")
    rep.check(bool(str(rec.get("why", "")).strip()), "the current compatibility record says why")

    try:
        _claims, vocab = load_claims(root, CLAIMS)
    except (ContractError, OSError, ValueError) as e:
        rep.fail(f"{CLAIMS}: {e}")
        return
    ev = manifest["evidence_vocabulary"]
    missing = sorted(set(vocab) - set(ev))
    rep.check(not missing, f"evidence_vocabulary covers {CLAIMS} status_vocabulary {sorted(vocab)}",
              f"missing {missing}")
    extra = sorted(set(ev) - set(vocab))
    if extra:
        rep.ok(f"evidence_vocabulary has {extra} beyond this register's vocabulary (used by the other register)")

    ids = [c["id"] for c in cap["capabilities"]]
    owned = list(manifest["capability_ownership"].get("runtime") or [])
    rep.check(len(set(owned)) == len(owned), "capability_ownership.runtime has no duplicates")
    rep.check(set(owned) == set(ids), f"capability_ownership.runtime == {CAPABILITIES} ids",
              f"manifest-only {sorted(set(owned) - set(ids))}, capabilities-only {sorted(set(ids) - set(owned))}")

    seen: set[str] = set()
    inv_ok = True
    for i, inv in enumerate(manifest["invariants"]):
        for key in ("id", "statement", "checked_by"):
            if not isinstance(inv.get(key), str) or not inv[key].strip():
                rep.fail(f"invariants[{i}]: missing or empty {key!r}")
                inv_ok = False
        if inv.get("id") in seen:
            rep.fail(f"invariants[{i}]: duplicate id {inv.get('id')!r}")
            inv_ok = False
        seen.add(str(inv.get("id")))
    if inv_ok:
        rep.ok(f"{len(manifest['invariants'])} invariants carry id, statement and checked_by; ids unique")


# ------------------------------------------------------------------ CI pin --

_PIN = re.compile(r"git\+https://github\.com/(?P<slug>[^/\s\"']+/[^/\s\"'@]+?)(?:\.git)?@(?P<sha>[0-9a-f]{7,40})\b")
_TAG_LINE = re.compile(r"^(?P<sha>[0-9a-f]{40})\s+refs/tags/v?(?P<ver>\d+(?:\.\d+)*)(?P<peel>\^\{\})?$")
_PIN_VERSION = re.compile(r"\b(v?)(\d+\.\d+\.\d+)\b")
_PIP_INSTALL = re.compile(r"\bpip3?\s+install\b")


def pin_version(prose: str) -> str | None:
    """The release ``consumer_ci_pin`` names: the LAST ``vX.Y.Z`` in the prose,
    or the last bare ``X.Y.Z`` when none is v-prefixed -- the prose may name the
    version it moved from before the one it moved to."""
    hits = _PIN_VERSION.findall(prose)
    if not hits:
        return None
    tagged = [ver for v, ver in hits if v]
    return (tagged or [ver for _, ver in hits])[-1]


def ci_kernel_pin(text: str, package: str = KERNEL_PACKAGE) -> tuple[str, str] | None:
    """``(repository slug, sha)`` of the ``pip install "<package> @ git+...@<sha>"``
    line in the workflow text, or None when there is none. Comment lines and
    lines that do not ``pip install`` (an echo, a note) are not pins."""
    for line in text.splitlines():
        if line.lstrip().startswith("#") or not _PIP_INSTALL.search(line):
            continue
        if pep503_name(package) not in pep503_name(line):
            continue
        m = _PIN.search(line)
        if m:
            return m.group("slug"), m.group("sha")
    return None


def release_tags(repo_url: str, timeout: float = 60.0) -> dict[str, str] | None:
    """``{version: commit sha}`` for every ``refs/tags/vX.Y.Z`` of the remote
    (annotated tags peeled to the commit), or None when the network or git is
    unavailable -- the caller prints a NOTE and skips, never passes."""
    try:
        p = subprocess.run(["git", "ls-remote", "--tags", repo_url], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if p.returncode != 0:
        return None
    tags: dict[str, str] = {}
    peeled: dict[str, str] = {}
    for line in p.stdout.splitlines():
        m = _TAG_LINE.match(line.strip())
        if not m:
            continue
        (peeled if m.group("peel") else tags)[m.group("ver")] = m.group("sha")
    tags.update(peeled)                      # the peeled commit wins over the tag object
    return tags


def check_ci_pin(root: Path, manifest: dict, rep: Report) -> bool:
    """The consumer CI pin is a release-tag commit at or above the tag the current
    compatibility record's ``consumer_ci_pin`` names. Returns False only when the
    release tags could not be read (the pin is then unverified, not passed)."""
    ci = root / CI_WORKFLOW
    if not ci.is_file():
        rep.ok(f"no {CI_WORKFLOW}: the CI pin check does not apply")
        return True
    py = load_pyproject(root)
    hits, err = current_record(manifest, str(py.get("name", "")), str(py.get("version", "")))
    if err or len(hits) != 1:
        return True                              # check_local has already failed this
    rec = hits[0]
    prose = str(rec.get("consumer_ci_pin", ""))
    pin = ci_kernel_pin(read_text(ci))
    if not prose.strip():
        rep.check(pin is None, "the current record names no consumer_ci_pin and CI pins no kernel commit",
                  f"CI pins {pin}")
        return True
    if not rep.check(pin is not None, f"{CI_WORKFLOW} pins {KERNEL_PACKAGE} at a git commit (consumer_ci_pin: {prose!r})"):
        return True
    slug, sha = pin
    named = pin_version(prose)
    if not rep.check(named is not None, f"consumer_ci_pin names a release version: {prose!r}"):
        return True
    kernels = manifest["packages"]["kernels"]
    rep.check(_github_slug(slug) == _github_slug(kernels["repository"]),
              f"the CI pin's repository {slug} is packages.kernels.repository", f"manifest says {kernels['repository']!r}")
    tags = release_tags(kernels["repository"])
    if tags is None:
        print(f"NOTE: could not read release tags from {kernels['repository']} (no network?): the CI pin "
              f"{sha[:12]} was NOT verified against tag v{named} -- skipped, not passed")
        return False
    if not rep.check(named in tags, f"release tag v{named} (consumer_ci_pin) exists in {kernels['repository']}",
                     f"tags seen: {sorted(tags)[-5:]}"):
        return True
    pinned = [v for v, c in tags.items() if c.startswith(sha) or sha.startswith(c)]
    if not rep.check(bool(pinned), f"CI pin {sha[:12]} is the commit of a release tag of {KERNEL_PACKAGE}",
                     f"no tag's commit matches; v{named} is {tags[named][:12]}"):
        return True
    pv = max(pinned, key=parse_version)
    if parse_version(pv) == parse_version(named):
        rep.ok(f"CI pin {sha[:12]} is the commit of v{named}, the tag consumer_ci_pin names")
    elif parse_version(pv) > parse_version(named):
        rep.ok(f"CI pin {sha[:12]} is the commit of v{pv}, a release above the v{named} that consumer_ci_pin names")
        print(f"NOTE: consumer_ci_pin says {prose!r} but CI installs v{pv}; update the prose in both repositories' "
              f"{MANIFEST} (the manifest is byte-identical across them)")
    else:
        rep.fail(f"CI pin {sha[:12]} is v{pv}, BELOW the v{named} that consumer_ci_pin names")
    return True


def kernel_extras(pyproject: dict, package: str = KERNEL_PACKAGE) -> dict[str, str]:
    """``{extra: requirement}`` for every optional-dependency extra that names the kernel package."""
    out = {}
    for extra in (pyproject.get("optional-dependencies") or {}):
        fr = fast_requirement(pyproject, package, extra)
        if fr is not None:
            out[extra] = fr[0]
    return out


def check_extra_floors(root: Path, manifest: dict, rep: Report) -> None:
    """Every extra that pins the kernel package has a ``>=`` floor at or above the current record's."""
    py = load_pyproject(root)
    hits, err = current_record(manifest, str(py.get("name", "")), str(py.get("version", "")))
    if err or len(hits) != 1:
        return
    rec_floor = floor_of(str(hits[0].get("floor", "")))
    if rec_floor is None:
        return
    for extra, req in sorted(kernel_extras(py).items()):
        fl = floor_of(fast_requirement(py, KERNEL_PACKAGE, extra)[1])
        if not rep.check(fl is not None, f"extra {extra!r} pins {KERNEL_PACKAGE} with a single >= floor ({req!r})"):
            continue
        rep.check(_cmp(parse_version(fl), parse_version(rec_floor)) >= 0,
                  f"extra {extra!r} floor >= {fl} is at or above the current record's {rec_floor}",
                  f"{req!r} is below the record's floor {hits[0].get('floor')!r}")


def check_sibling(root: Path, sibling: Path, manifest: dict, rep: Report) -> None:
    ours = (root / MANIFEST).read_bytes()
    theirs_path = sibling / MANIFEST
    if not rep.check(theirs_path.is_file(), f"sibling has {MANIFEST} ({theirs_path})"):
        return
    rep.check(theirs_path.read_bytes() == ours, f"sibling {MANIFEST} is byte-identical to ours")
    spy = load_pyproject(sibling)
    kernels = manifest["packages"]["kernels"]
    sname, sversion = str(spy.get("name", "")), str(spy.get("version", ""))
    if not rep.check(pep503_name(sname) == pep503_name(kernels["package"]),
                     f"sibling pyproject name {sname!r} is packages.kernels.package", f"expected {kernels['package']!r}"):
        return
    n = 0
    for rec in manifest.get("compatibility") or []:
        if pep503_name(str(rec.get("kernel", ""))) != pep503_name(sname):
            continue
        n += 1
        try:
            ok = version_in_range(sversion, str(rec.get("floor", "")))
        except ValueError as e:
            rep.fail(f"compatibility record for {rec.get('consumer_versions')!r}: unreadable floor: {e}")
            continue
        rep.check(ok, f"kernel-first: sibling {sname} {sversion} satisfies floor {rec.get('floor')!r} "
                      f"(consumer {rec.get('consumer_versions')!r})")
    rep.check(n > 0, f"the manifest names at least one floor for {sname}")
    scap = sibling / CAPABILITIES
    if scap.is_file():
        sids = [c["id"] for c in json.loads(read_text(scap))["capabilities"]]
        owned = list(manifest["capability_ownership"].get("kernels") or [])
        rep.check(set(owned) == set(sids), f"capability_ownership.kernels == sibling {CAPABILITIES} ids",
                  f"manifest-only {sorted(set(owned) - set(sids))}, sibling-only {sorted(set(sids) - set(owned))}")
    if (sibling / CLAIMS).is_file():
        try:
            _c, svocab = load_claims(sibling, CLAIMS)
            missing = sorted(set(svocab) - set(manifest["evidence_vocabulary"]))
            rep.check(not missing, f"evidence_vocabulary covers the sibling's status_vocabulary {sorted(svocab)}",
                      f"missing {missing}")
        except (ContractError, OSError, ValueError) as e:
            rep.fail(f"sibling {CLAIMS}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="this repository's root")
    ap.add_argument("--sibling", default=None, metavar="PATH",
                    help="the sibling repository's checkout (byte-identical manifest, kernel-first floors)")
    ap.add_argument("--require-tags", action="store_true",
                    help="exit 2 when the kernel repository's release tags cannot be read (CI: the pin must be verified, "
                         "never skipped)")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    rep = Report()
    try:
        manifest = json.loads(read_text(root / MANIFEST))
    except (OSError, ValueError) as e:
        print(f"FAIL: {MANIFEST}: {e}")
        return 1
    missing = [k for k in REQUIRED_TOP if k not in manifest]
    if not rep.check(not missing, f"{MANIFEST} parses and carries {list(REQUIRED_TOP)}", f"missing {missing}"):
        return 1
    try:
        check_local(root, manifest, rep)
        check_extra_floors(root, manifest, rep)
        pin_verified = check_ci_pin(root, manifest, rep)
        if a.sibling:
            check_sibling(root, Path(a.sibling).resolve(), manifest, rep)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"FAIL: the check could not run: {e!r}")
        return 2
    if rep.failed:
        print(f"FAIL: {rep.failed} check(s) failed")
        return 1
    if not pin_verified and a.require_tags:
        print(f"FAIL: --require-tags: the release tags of {KERNEL_PACKAGE} could not be read, so the CI pin is "
              "unverified -- the check cannot run")
        return 2
    print(f"OK: {MANIFEST} agrees with pyproject.toml, {CLAIMS} and {CAPABILITIES}"
          + (" and with the sibling" if a.sibling else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
