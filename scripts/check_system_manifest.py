#!/usr/bin/env python3
"""Validate docs/system-manifest.json -- the cross-repository system manifest --
against THIS repository, and optionally against the sibling repository.

One file, byte-identical in experts4bit-qlora (the runtime package) and
grouped-nf4-gemm (the kernel package). It reads which package this repository
is from the manifest -- the ``packages`` entry whose ``package`` is
pyproject's [project].name, or failing that the one whose ``repository`` is
its Source URL, the way scripts/check_capabilities.py tells its role -- and
runs that role's rules; nothing keys on a hard-coded package prefix.
Standard library only. The kernel role makes no network call; the runtime
role makes one, ``git ls-remote --tags`` of the kernel repository, for the CI
pin check. Prints ``OK:`` / ``FAIL:`` / ``SKIP:`` / ``NOTE:`` lines; exit 1
when any check failed, 2 when the check itself cannot run.

Both roles (the manifest is the same bytes in both, so a rule about the
manifest alone runs in both):
  * the manifest parses and carries every top-level table the scripts read
    (schema_version, system, packages, capability_ownership, compatibility,
    evidence_vocabulary, authority, invariants, router);
  * ``packages.<role>``: ``package`` is pyproject's name; ``repository`` is
    [project.urls] Source (equal after URL normalisation AND the same GitHub
    owner/repo); every ``import_names`` entry is a module pyproject ships and
    the tree carries; ``import_names`` equals docs/capabilities.json
    ``project.import_names`` (same names, same order);
  * ``system.dependency_direction`` names runtime -> kernels;
  * a [project.urls] entry naming the other package (``Consumer:
    experts4bit-qlora`` in the kernel's pyproject, ``Kernel: grouped-nf4-gemm``
    in the runtime's), when present, is that package's ``pypi`` URL;
  * ``compatibility``: at least one record names the kernel package, and each
    such record carries consumer / consumer_versions / kernel / floor / extra /
    since / why, a ``consumer_versions`` range this checker can read, a single
    ``>=X.Y.Z`` floor, ``packages.runtime.package`` as its consumer and a
    non-empty ``why``;
  * ``evidence_vocabulary`` covers this repository's docs/claims.json
    ``status_vocabulary`` (extras are reported, not failed);
  * ``capability_ownership.<role>`` equals the capability ids in
    docs/capabilities.json; neither package's list has a duplicate, and no id
    is owned by both;
  * every invariant has a non-empty string ``id``, ``statement`` and
    ``checked_by``, and ids are unique; the router carries its five entries.

Kernel role only -- the kernel-first invariant from the kernel's side:
  * every kernel record's floor is <= pyproject's version (a floor never names
    an unreleased kernel version) and, when final ``vX.Y.Z`` tags are present
    locally (``git tag -l``, never the network), <= the latest one. Without
    local tags that comparison is a SKIP, and exit 2 under ``--require-tags``;
  * the record current for the consumer (the highest consumer range) is
    reported.

Consumer rules -- on this pyproject in the runtime role, on the sibling's
pyproject in the kernel role under ``--sibling``:
  * the ``fast`` extra requires the kernel package with a single ``>=`` floor;
  * exactly one compatibility record's ``consumer_versions`` contains the
    consumer's version; it names the kernel package via extra ``fast``; its
    floor equals the ``fast`` requirement's, as a release and as the same
    specifier text -- the number in the manifest is validated against
    pyproject, never trusted from the manifest;
  * every extra that pins the kernel package (``fast``, ``test``, ...) has a
    single ``>=`` floor at or above that record's.

Runtime role only -- the consumer's CI pin of the kernel package
(``.github/workflows/ci.yml``, ``pip install "grouped-nf4-gemm @ git+...@<sha>"``):
  * the pinned sha is the commit of a RELEASE TAG of the kernel package -- the
    tag ``compatibility[current].consumer_ci_pin`` names (exact: OK), or a
    later release (OK with a NOTE that the prose is behind); any other sha, or
    a tag below the one named, FAILS. Tags are read with ``git ls-remote
    --tags <repository>`` (annotated tags peeled); with no network the check
    prints ``NOTE: ... skipped`` and never passes silently as if it had run,
    and under ``--require-tags`` (CI, where the network is there) an
    unreadable tag list is exit 2. The version ``consumer_ci_pin`` names is
    the last ``vX.Y.Z`` in its prose (the last bare ``X.Y.Z`` when none carries
    a ``v``); the pin line is a non-comment ``pip install`` line.

``--sibling PATH`` (the other package's checkout: the runtime's CI clones the
kernel at its latest release tag; the kernel's CI does not run it):
  * PATH/docs/system-manifest.json is byte-identical to this manifest;
  * the sibling's pyproject name is the other package's;
  * runtime role: the sibling kernel's version satisfies every floor the
    manifest names for it (kernel-first, from the consumer's side); kernel
    role: the consumer rules above, on the sibling's pyproject;
  * ``capability_ownership.<other role>`` equals the sibling's
    docs/capabilities.json ids, and ``evidence_vocabulary`` covers the
    sibling's docs/claims.json vocabulary, when those files exist.

Versions: manifest and pyproject versions are read strictly -- ``X.Y.Z`` with
an optional ``v``; ranges are comma-joined ``>=``, ``<=``, ``>``, ``<``,
``==``, ``!=``, ``~=`` clauses or an ``a.b.x`` / ``a.b.*`` wildcard, a bare
``x.y.z`` meaning exactly that version (a bare ``a.b`` is refused: write
``a.b.x``) -- and a range this checker cannot read never passes as satisfied (``parse_version``, ``version_in_range``,
``floor_of``). ``release_tuple``, ``floor_version`` and ``range_lower_bound``
are the lenient readers scripts/check_dependency_floor.py uses for torch /
triton floors and for the statements documents make; a pre-release suffix
there is ignored, never a finding.
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
    load_claims, load_pyproject, module_file, module_shipped, pep503_name, read_text, requirement_extras,
)

MANIFEST = "docs/system-manifest.json"
CI_WORKFLOW = ".github/workflows/ci.yml"
CAPABILITIES = "docs/capabilities.json"
CLAIMS = "docs/claims.json"
KERNEL_PACKAGE = "grouped-nf4-gemm"
EXTRA = "fast"
ROLES = ("runtime", "kernels")
REQUIRED_TOP = ("schema_version", "system", "packages", "capability_ownership", "compatibility",
                "evidence_vocabulary", "authority", "invariants", "router")
RECORD_KEYS = ("consumer", "consumer_versions", "kernel", "floor", "extra", "since", "why")
ROUTER_ENTRIES = 5
#: The [project.urls] label under which each role's pyproject links the OTHER
#: package (``Consumer: experts4bit-qlora`` / ``Kernel: grouped-nf4-gemm``).
OTHER_URL_LABEL = {"kernels": "Consumer", "runtime": "Kernel"}

_RELEASE = re.compile(r"^\d+(\.\d+)*$")
_CLAUSE = re.compile(r"^(>=|<=|==|!=|~=|>|<)?\s*v?(\d+(?:\.\d+)*(?:\.[x*])?)$")
_VERSION = re.compile(r"v?(\d+(?:\.\d+)*)")
_LENIENT_CLAUSE = re.compile(r"\s*(>=|<=|==|!=|~=|>|<)?\s*v?(\d+(?:\.\d+)*)(\.\*|\.x)?\s*$")
_FINAL_TAG = re.compile(r"^v(\d+(?:\.\d+)*)$")


def _other(role: str) -> str:
    return "kernels" if role == "runtime" else "runtime"


# ---------------------------------------------------------- strict versions --

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


def parse_range(spec: str) -> list[tuple[str | None, tuple[int, ...], bool]]:
    """``spec`` as ``(operator, version, is_wildcard)`` clauses, every clause
    read before any is evaluated, so an unreadable clause raises ValueError
    wherever it sits (an empty range used to match every version)."""
    out = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            raise ValueError(f"empty version clause in {spec!r}")
        m = _CLAUSE.match(raw)
        if not m:
            raise ValueError(f"unsupported version clause {raw!r} in {spec!r}")
        op, val = m.group(1), m.group(2)
        wild = val.endswith((".x", ".*"))
        if wild and op not in (None, "=="):
            raise ValueError(f"a wildcard takes no operator: {raw!r}")
        ver = parse_version(val[:-2] if wild else val)
        if op is None and not wild and len(ver) < 3:
            # the two copies this file replaces read a bare "0.34" differently (the
            # kernel's as 0.34.*, the runtime's as exactly 0.34.0); neither reading
            # is safe to keep, so the range must say which it means
            raise ValueError(f"a bare version is X.Y.Z; write {val}.x for the series: {raw!r}")
        out.append((op, ver, wild))
    return out


def version_in_range(version: str, spec: str) -> bool:
    """Does ``version`` satisfy ``spec``? ``spec`` is comma-joined clauses
    (``">=0.35.0"``, ``">=0.30.0,<0.40"``, ``"0.34.x"``, ``"~=0.34.1"``,
    ``"0.34.0"``). An unsupported or empty clause raises ValueError -- a range
    this checker cannot read must not pass as satisfied."""
    v = parse_version(version)
    for op, w, wild in parse_range(spec):
        if wild:
            vv, _ = _pad(v, w)
            if vv[:len(w)] != w:
                return False
            continue
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


def _same_release(a: str, b: str) -> bool:
    try:
        return parse_version(a) == parse_version(b)
    except ValueError:
        return False


# --------------------------------------------------------- lenient versions --

def release_tuple(v: str) -> tuple[int, ...]:
    """The release segment of a version as a tuple: ``0.30.0`` -> (0, 30, 0);
    ``2.8.0.dev0`` -> (2, 8, 0). Pre/post/dev suffixes are ignored on purpose:
    a floor here is a release floor. For torch / triton floors and document
    statements; the manifest's own versions go through ``parse_version``."""
    m = _VERSION.match(str(v).strip())
    if not m:
        raise ValueError(f"not a version: {v!r}")
    return tuple(int(x) for x in m.group(1).split("."))


def floor_version(floor: str) -> tuple[int, ...]:
    """``">=0.30.0"`` -> (0, 30, 0). Only the single-clause >= form is a floor."""
    m = _LENIENT_CLAUSE.match(floor)
    if not m or m.group(1) != ">=" or m.group(3):
        raise ValueError(f"floor must be a '>=X.Y.Z' specifier, got {floor!r}")
    return release_tuple(m.group(2))


def range_lower_bound(rng: str) -> tuple[int, ...]:
    """The lowest version a range admits, for ordering records when no
    consumer version is at hand."""
    lows = []
    for clause in rng.split(","):
        m = _LENIENT_CLAUSE.match(clause)
        if m and m.group(1) in (None, ">=", ">", "==", "~="):
            lows.append(release_tuple(m.group(2)))
    return max(lows) if lows else (0,)


def final_release_tag(tag: str) -> tuple[int, ...] | None:
    """``v0.30.0`` -> (0, 30, 0); a pre-release, post or dev tag (``v0.31.0rc1``,
    ``v0.31.0.dev2``) -> None. Only a final release satisfies the kernel-first
    invariant: a consumer floor must name a version that actually shipped."""
    m = _FINAL_TAG.match(tag.strip())
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


def latest_tag(root: Path) -> tuple[int, ...] | None:
    """The highest final ``v*`` tag in the LOCAL repository (``git tag -l``; no
    network), or None when git, the repository or the tags are absent."""
    try:
        out = subprocess.run(["git", "-C", str(root), "tag", "-l", "v*"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    tags = [tt for tt in (final_release_tag(t) for t in out.split()) if tt is not None]
    return max(tags) if tags else None


# ------------------------------------------------------------------ records --

def kernel_records(manifest: dict, kernel_pkg: str) -> list[dict]:
    """The ``compatibility`` records whose ``kernel`` is ``kernel_pkg``."""
    return [r for r in manifest.get("compatibility", []) if pep503_name(str(r.get("kernel", ""))) == pep503_name(kernel_pkg)]


def current_kernel_record(manifest: dict, kernel_pkg: str, consumer_version: str | None = None) -> dict | None:
    """The kernel's compatibility record that is current for the consumer: with
    a consumer version, the one whose consumer_versions contains it (None when
    zero or several do; ``version_in_range`` reads it, strictly); without one,
    the record whose range starts highest. The kernel-side reader, keyed on the
    kernel package -- ``current_record`` is the consumer-side one."""
    recs = kernel_records(manifest, kernel_pkg)
    if not recs:
        return None
    if consumer_version is not None:
        hits = [r for r in recs if version_in_range(consumer_version, str(r["consumer_versions"]))]
        return hits[0] if len(hits) == 1 else None
    return max(recs, key=lambda r: range_lower_bound(str(r["consumer_versions"])))


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


def kernel_extras(pyproject: dict, package: str = KERNEL_PACKAGE) -> dict[str, str]:
    """``{extra: requirement}`` for every optional-dependency extra that names the kernel package."""
    out = {}
    for extra in (pyproject.get("optional-dependencies") or {}):
        fr = fast_requirement(pyproject, package, extra)
        if fr is not None:
            out[extra] = fr[0]
    return out


def _pin_text(req: str) -> str:
    """The specifier text of a requirement with its name, extras, marker and
    spaces removed (``"grouped-nf4-gemm >= 0.30.0"`` -> ``">=0.30.0"``): the
    manifest floor must be this text, not merely the same release."""
    return re.sub(r"^[A-Za-z0-9._-]+(\[[^\]]*\])?", "", req).split(";")[0].replace(" ", "")


# --------------------------------------------------------------- identities --

def _norm_url(u: str) -> str:
    return str(u).strip().rstrip("/").removesuffix(".git").lower()


def _github_slug(url: str) -> str:
    m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$", str(url))
    return m.group(1).lower() if m else str(url).lower()


def same_repository(a: str, b: str) -> bool:
    """Both repository comparisons the two copies of this script made: the
    normalised URL (kernel side) and the GitHub owner/repo (runtime side)."""
    return _norm_url(a) == _norm_url(b) and _github_slug(a) == _github_slug(b)


def system_role(manifest: dict, pyproject: dict) -> str | None:
    """``"runtime"`` / ``"kernels"``: the ``packages`` entry whose ``package`` is
    pyproject's name -- as scripts/check_capabilities.py decides its role --
    or, when none is, the one whose ``repository`` is pyproject's Source URL
    (so a renamed package still gets its own role's checks, which then fail on
    the name); ``None`` when neither says."""
    packages = manifest.get("packages")
    if not isinstance(packages, dict):
        return None
    name = pep503_name(str(pyproject.get("name", "")))
    for role in ROLES:
        p = packages.get(role)
        if isinstance(p, dict) and pep503_name(str(p.get("package", ""))) == name:
            return role
    src = str((pyproject.get("urls") or {}).get("Source", ""))
    for role in ROLES:
        p = packages.get(role)
        if src and isinstance(p, dict) and (_norm_url(p.get("repository", "")) == _norm_url(src)
                                            or _github_slug(p.get("repository", "")) == _github_slug(src)):
            return role
    return None


# -------------------------------------------------------------------- report --

class Report:
    def __init__(self) -> None:
        self.failed = 0

    def ok(self, msg: str) -> None:
        print("OK:", msg)

    def fail(self, msg: str) -> None:
        self.failed += 1
        print("FAIL:", msg)

    def skip(self, msg: str) -> None:
        print("SKIP:", msg)

    def check(self, cond: bool, msg: str, detail: str = "") -> bool:
        (self.ok if cond else self.fail)(msg if cond or not detail else f"{msg}: {detail}")
        return cond


# -------------------------------------------------------------- both roles --

def check_package(root: Path, py: dict, manifest: dict, role: str, caps: dict, rep: Report) -> None:
    """``packages.<role>`` against pyproject.toml and docs/capabilities.json."""
    pk = manifest["packages"]
    me, other_role = pk[role], _other(role)
    other = pk[other_role]
    name = str(py.get("name", ""))
    rep.check(pep503_name(str(me.get("package", ""))) == pep503_name(name),
              f"packages.{role}.package {me.get('package')!r} is pyproject's name", f"pyproject name is {name!r}")
    src = str((py.get("urls") or {}).get("Source", ""))
    rep.check(same_repository(str(me.get("repository", "")), src),
              f"packages.{role}.repository is [project.urls] Source ({src})",
              f"manifest says {me.get('repository')!r}")
    names = list(me.get("import_names") or [])
    unshipped = [i for i in names if not module_shipped(i, py)]
    absent = [i for i in names if i not in unshipped and module_file(root, i, py) is None]
    rep.check(not unshipped and not absent,
              f"packages.{role}.import_names ({len(names)}) are all shipped by pyproject and present in the tree",
              f"not shipped (py-modules / packages): {unshipped}; shipped but no module in the tree: {absent}")
    # One public import surface, not two: the manifest's list is what the site
    # and packages.json publish, capabilities.json's is what this repository
    # publishes.
    cap_imports = list((caps.get("project") or {}).get("import_names") or [])
    detail = (" (same names, different order)" if set(names) == set(cap_imports) else
              f": manifest-only {sorted(set(names) - set(cap_imports))}, "
              f"capabilities-only {sorted(set(cap_imports) - set(names))}")
    rep.check(names == cap_imports,
              f"packages.{role}.import_names == {CAPABILITIES} project.import_names {cap_imports}",
              f"manifest says {names!r}{detail}")
    direction = list(manifest["system"].get("dependency_direction") or [])
    rep.check(f"{pk['runtime'].get('package')} -> {pk['kernels'].get('package')}" in direction,
              "system.dependency_direction names runtime -> kernels", f"{direction!r}")
    label = f"{OTHER_URL_LABEL[role]}: {other.get('package')}"
    url = (py.get("urls") or {}).get(label)
    if url and other.get("pypi"):
        rep.check(_norm_url(url) == _norm_url(other["pypi"]),
                  f"pyproject [project.urls] {label!r} is packages.{other_role}.pypi", f"{url!r} != {other['pypi']!r}")


def check_compatibility(manifest: dict, rep: Report) -> None:
    """The shape of every record that names the kernel package."""
    pk = manifest["packages"]
    kernel, consumer = str(pk["kernels"].get("package", "")), str(pk["runtime"].get("package", ""))
    recs = kernel_records(manifest, kernel)
    rep.check(bool(recs), f"compatibility has a record whose kernel is {kernel!r}")
    for r in recs:
        cv = r.get("consumer_versions")
        lacking = [k for k in RECORD_KEYS if k not in r]
        if lacking:
            rep.fail(f"compatibility record for consumer {cv!r} lacks {lacking}")
            continue
        problems = []
        try:
            floor_version(str(r["floor"]))
        except ValueError as e:
            problems.append(str(e))
        try:
            parse_range(str(r["consumer_versions"]))
        except ValueError as e:
            problems.append(f"consumer_versions: {e}")
        if pep503_name(str(r["consumer"])) != pep503_name(consumer):
            problems.append(f"consumer {r['consumer']!r} is not packages.runtime.package")
        if not str(r["why"]).strip():
            problems.append("its 'why' is empty")
        rep.check(not problems, f"compatibility record {cv!r}: a readable range, a single >=X.Y.Z floor "
                                f"({r['floor']}), consumer {consumer}, says why", "; ".join(problems))


def check_registers(root: Path, manifest: dict, role: str, caps: dict, rep: Report) -> None:
    """evidence_vocabulary vs docs/claims.json; capability_ownership vs docs/capabilities.json."""
    ev = manifest["evidence_vocabulary"]
    try:
        _claims, vocab = load_claims(root, CLAIMS)
    except Exception as e:  # noqa: BLE001 -- any malformed register is a finding, reported not raised
        rep.fail(f"{CLAIMS}: {e}")
        vocab = None
    if vocab is not None:
        missing = sorted(set(vocab) - set(ev))
        rep.check(not missing, f"evidence_vocabulary covers {CLAIMS} status_vocabulary {sorted(vocab)}",
                  f"missing {missing}")
        extra = sorted(set(ev) - set(vocab))
        if extra:
            rep.ok(f"evidence_vocabulary has {extra} beyond this register's vocabulary (used by the other register)")

    ids = [c["id"] for c in caps["capabilities"]]
    lists = {r: list(manifest["capability_ownership"].get(r) or []) for r in ROLES}
    for r in ROLES:                              # the manifest alone: both lists, in both roles
        dups = sorted({i for i in lists[r] if lists[r].count(i) > 1})
        rep.check(not dups, f"capability_ownership.{r} has no duplicates", f"duplicated: {dups}")
    owned = lists[role]
    rep.check(set(owned) == set(ids), f"capability_ownership.{role} == {CAPABILITIES} ids ({len(set(ids))})",
              f"manifest-only {sorted(set(owned) - set(ids))}, capabilities-only {sorted(set(ids) - set(owned))}")
    both = sorted(set(lists["runtime"]) & set(lists["kernels"]))
    rep.check(not both, "no capability id is owned by both packages", f"owned by both: {both}")


def check_invariants_router(manifest: dict, rep: Report) -> None:
    ids: list = []
    inv_ok = True
    for i, inv in enumerate(manifest["invariants"]):
        for key in ("id", "statement", "checked_by"):
            if not isinstance(inv.get(key), str) or not inv[key].strip():
                rep.fail(f"invariants[{i}]: missing or empty {key!r}")
                inv_ok = False
        if inv.get("id") in ids:                  # raw ids: two missing ids are a duplicate too
            rep.fail(f"invariants[{i}]: duplicate id {inv.get('id')!r}")
            inv_ok = False
        ids.append(inv.get("id"))
    if inv_ok:
        rep.ok(f"{len(ids)} invariants carry id, statement and checked_by; ids unique")
    rep.check(len(manifest["router"]) == ROUTER_ENTRIES, f"router carries {ROUTER_ENTRIES} entries",
              f"it has {len(manifest['router'])}")


# ------------------------------------------------------------- kernel role --

def check_kernel_first(root: Path, py: dict, manifest: dict, rep: Report) -> bool:
    """Every kernel floor is <= this kernel's pyproject version and <= its latest
    local final release tag. Returns False when no such tag could be read (the
    tag comparison is then a SKIP -- unverified, not passed). Never the network:
    this is the kernel's own history, already in its checkout."""
    kernel = str(manifest["packages"]["kernels"].get("package", ""))
    version = str(py.get("version", ""))
    try:
        here: tuple[int, ...] | None = parse_version(version)
    except ValueError as e:
        rep.fail(f"kernel-first: pyproject version {version!r} cannot be compared to the floors: {e}")
        here = None
    tag = latest_tag(root)
    recs = kernel_records(manifest, kernel)
    for r in recs:
        try:
            fv = floor_version(str(r.get("floor", "")))
        except ValueError:
            continue                              # check_compatibility has already failed this floor
        cv = r.get("consumer_versions")
        if here is not None:
            rep.check(_cmp(fv, here) <= 0, f"kernel-first vs pyproject: record {cv!r} floor {r['floor']} <= {version}",
                      "a floor never names an unreleased kernel version")
        if tag is not None:
            t = "v" + ".".join(map(str, tag))
            rep.check(_cmp(fv, tag) <= 0, f"kernel-first vs tags: record {cv!r} floor {r['floor']} <= {t}",
                      f"the latest local release tag is {t} (no such release exists yet)")
    if tag is None:
        rep.skip("no v* tags available locally; the floor-vs-released-tag comparison did not run (pyproject comparison did)")
    if recs and all("consumer_versions" in r for r in recs):
        cur = current_kernel_record(manifest, kernel)
        rep.ok(f"current record (highest consumer range without a sibling): consumer {cur['consumer_versions']} -> "
               f"{kernel}{cur.get('floor')} [{cur.get('extra')}] since {cur.get('since')}")
    return tag is not None


# ------------------------------------------------------------ consumer rules --

def check_consumer(py: dict, manifest: dict, rep: Report, who: str) -> None:
    """The consumer's side of the compatibility contract, on a runtime pyproject
    (this one in the runtime role, the sibling's in the kernel role)."""
    pk = manifest["packages"]
    kernel, consumer = str(pk["kernels"].get("package", "")), str(pk["runtime"].get("package", ""))
    version = str(py.get("version", ""))
    fr = fast_requirement(py, kernel)
    py_floor = None
    if rep.check(fr is not None, f"{who} [{EXTRA}] extra requires {kernel}"):
        req, spec = fr
        py_floor = floor_of(spec)
        rep.check(py_floor is not None, f"{who} {EXTRA} requirement {req!r} has a single >= floor")
        if requirement_extras(req):
            rep.skip(f"{who} {EXTRA} requirement carries extras {requirement_extras(req)}; not validated")
    hits, err = current_record(manifest, consumer, version)
    if err:
        rep.fail(f"compatibility: unreadable consumer_versions range: {err}")
        return
    if not rep.check(len(hits) == 1, f"exactly one compatibility record contains {consumer} {version}",
                     f"{len(hits)} records match: {[h.get('consumer_versions') for h in hits]}"):
        return
    rec = hits[0]
    cv, floor = rec.get("consumer_versions"), str(rec.get("floor", ""))
    rep.check(pep503_name(str(rec.get("kernel", ""))) == pep503_name(kernel) and rec.get("extra") == EXTRA,
              f"record {cv!r} names kernel {kernel!r} via extra {EXTRA!r}",
              f"kernel={rec.get('kernel')!r} extra={rec.get('extra')!r}")
    rec_floor = floor_of(floor)
    if fr is not None:
        same = (rec_floor is not None and py_floor is not None and _same_release(rec_floor, py_floor)
                and _pin_text(fr[0]) == floor.replace(" ", ""))
        rep.check(same, f"record {cv!r} floor {floor!r} equals {who}'s {fr[0]!r}",
                  f"{who} pins {kernel}{_pin_text(fr[0])} but the record says {floor!r}")
    if rec_floor is None:
        return
    for extra, req in sorted(kernel_extras(py, kernel).items()):
        fl = floor_of(fast_requirement(py, kernel, extra)[1])
        if not rep.check(fl is not None, f"{who} extra {extra!r} pins {kernel} with a single >= floor ({req!r})"):
            continue
        rep.check(_cmp(parse_version(fl), parse_version(rec_floor)) >= 0,
                  f"{who} extra {extra!r} floor >= {fl} is at or above the current record's {rec_floor}",
                  f"{req!r} is below the record's floor {floor!r}")


# ---------------------------------------------------- runtime role: CI pin --

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
    unavailable -- the caller prints a NOTE and skips, never passes. The only
    network read in this script, and the runtime role's alone."""
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
        return True                              # check_consumer has already failed this
    rec = hits[0]
    kernels = manifest["packages"]["kernels"]
    kernel = str(kernels.get("package", KERNEL_PACKAGE))
    prose = str(rec.get("consumer_ci_pin", ""))
    pin = ci_kernel_pin(read_text(ci), kernel)
    if not prose.strip():
        rep.check(pin is None, "the current record names no consumer_ci_pin and CI pins no kernel commit",
                  f"CI pins {pin}")
        return True
    if not rep.check(pin is not None, f"{CI_WORKFLOW} pins {kernel} at a git commit (consumer_ci_pin: {prose!r})"):
        return True
    slug, sha = pin
    named = pin_version(prose)
    if not rep.check(named is not None, f"consumer_ci_pin names a release version: {prose!r}"):
        return True
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
    if not rep.check(bool(pinned), f"CI pin {sha[:12]} is the commit of a release tag of {kernel}",
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


# ------------------------------------------------------------------ sibling --

def check_sibling(sibling: Path, manifest: dict, raw: bytes, role: str, rep: Report) -> None:
    """The other package's checkout: the same manifest bytes, the other package's
    name, and that package's side of the contract."""
    other_role = _other(role)
    other = manifest["packages"][other_role]
    theirs = sibling / MANIFEST
    if rep.check(theirs.is_file(), f"sibling has {MANIFEST} ({theirs})"):
        rep.check(theirs.read_bytes() == raw, f"sibling {MANIFEST} is byte-identical to ours")
    try:
        spy: dict | None = load_pyproject(sibling)
    except (OSError, ValueError) as e:
        rep.fail(f"sibling pyproject: {e}")
        spy = None
    if spy is not None:
        sname, sversion = str(spy.get("name", "")), str(spy.get("version", ""))
        if rep.check(pep503_name(sname) == pep503_name(str(other.get("package", ""))),
                     f"sibling pyproject name {sname!r} is packages.{other_role}.package",
                     f"expected {other.get('package')!r}"):
            if role == "kernels":
                check_consumer(spy, manifest, rep, f"sibling {sname} {sversion}")
            else:
                n = 0
                for rec in kernel_records(manifest, sname):
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
        owned = list(manifest["capability_ownership"].get(other_role) or [])
        rep.check(set(owned) == set(sids), f"capability_ownership.{other_role} == sibling {CAPABILITIES} ids",
                  f"manifest-only {sorted(set(owned) - set(sids))}, sibling-only {sorted(set(sids) - set(owned))}")
    if (sibling / CLAIMS).is_file():
        try:
            _c, svocab = load_claims(sibling, CLAIMS)
        except Exception as e:  # noqa: BLE001 -- a malformed sibling register is a finding, not a skip
            rep.fail(f"sibling {CLAIMS}: {e}")
        else:
            missing = sorted(set(svocab) - set(manifest["evidence_vocabulary"]))
            rep.check(not missing, f"evidence_vocabulary covers the sibling's status_vocabulary {sorted(svocab)}",
                      f"missing {missing}")


# ---------------------------------------------------------------------- main --

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="this repository's root")
    ap.add_argument("--manifest", default=MANIFEST, help=f"the manifest, relative to --root (default {MANIFEST})")
    ap.add_argument("--sibling", default=None, metavar="PATH",
                    help="the other package's checkout (byte-identical manifest, its side of the contract)")
    ap.add_argument("--require-tags", action="store_true",
                    help="exit 2 when the kernel's release tags cannot be read -- runtime role: the remote's, for the "
                         "CI pin; kernel role: the local checkout's, for the floor-vs-tag comparison (CI: verified, "
                         "never skipped)")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    rep = Report()
    try:
        raw = (root / a.manifest).read_bytes()
    except OSError as e:
        print(f"FAIL: {a.manifest}: {e}")
        return 1
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except ValueError as e:                      # UnicodeDecodeError is a ValueError
        print(f"FAIL: {a.manifest}: not valid JSON: {e}")
        return 1
    missing = [k for k in REQUIRED_TOP if k not in manifest]
    if not rep.check(not missing, f"{a.manifest} parses ({len(raw)} bytes) and carries {list(REQUIRED_TOP)}",
                     f"missing {missing}"):
        return 1
    try:
        py = load_pyproject(root)
        absent = [r for r in ROLES if not isinstance(manifest["packages"].get(r), dict)]
        if not rep.check(not absent, f"packages carries {list(ROLES)}", f"missing {absent}"):
            return 1
        role = system_role(manifest, py)
        if role is None:
            rep.fail(f"pyproject name {py.get('name')!r} (Source {(py.get('urls') or {}).get('Source')!r}) is neither "
                     f"packages.runtime nor packages.kernels: this repository is not a package of the system")
            print(f"FAIL: {rep.failed} check(s) failed in {a.manifest}")
            return 1
        rep.ok(f"this repository is packages.{role} ({manifest['packages'][role].get('package')})")
        caps = json.loads(read_text(root / CAPABILITIES))
        check_package(root, py, manifest, role, caps, rep)
        check_compatibility(manifest, rep)
        if role == "kernels":
            # The kernel's own pyproject version and tags are what a floor must not
            # pass; in the runtime role those are the consumer's, so the runtime side
            # of the same invariant is the --sibling version check.
            tags_read = check_kernel_first(root, py, manifest, rep)
        else:
            # The consumer's pyproject carries the `fast` extra; the kernel's does
            # not, so the kernel role applies these to the sibling under --sibling.
            check_consumer(py, manifest, rep, "pyproject")
        check_registers(root, manifest, role, caps, rep)
        check_invariants_router(manifest, rep)
        if role == "runtime":
            # Only the consumer's CI pins the kernel, and only this check reads the
            # network; the kernel role never runs it.
            tags_read = check_ci_pin(root, manifest, rep)
        if a.sibling:
            check_sibling(Path(a.sibling).resolve(), manifest, raw, role, rep)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        print(f"FAIL: the check could not run: {e!r}")
        return 2
    if rep.failed:
        print(f"FAIL: {rep.failed} check(s) failed in {a.manifest}")
        return 1
    if not tags_read and a.require_tags:
        if role == "runtime":
            print(f"FAIL: --require-tags: the release tags of {manifest['packages']['kernels'].get('package')} could "
                  "not be read, so the CI pin is unverified -- the check cannot run")
        else:
            print("FAIL: --require-tags: no final v* release tag is available in this checkout, so the floor-vs-tag "
                  "comparison is unverified -- the check cannot run (fetch the tags: actions/checkout fetch-depth: 0)")
        return 2
    print(f"OK: {a.manifest} agrees with pyproject.toml, {CLAIMS} and {CAPABILITIES}"
          + (" and with the sibling" if a.sibling else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
