#!/usr/bin/env python3
"""The claims register, checked against its ONE schema (docs/claims-schema.md).

This file and docs/claims-schema.md are byte-identical in experts4bit-qlora and
grouped-nf4-gemm (scripts/check_shared_tooling.py). Until 2026-09-23 each
repository had its own copy of both, and they had drifted into two schemas: one
accepted ``path#fragment`` strings and free-text quotes and refused bare URLs, the
other accepted directories, ``{"path", "section"}`` objects and URL strings and
refused fragments. A claim written to either repository's rules failed the other's
check (14 findings one way, 38 the other). The data was migrated to the converged
rules below. Standard library only, no network; one ``git ls-files`` per checkout.

Every rule keys on the register's STRUCTURE, which is what the 2026-09-05 audits in
both repositories found no check reading (receipt logs that were never committed,
annotated paths, superseded rows with no successor, "pending" on rows measured days
earlier, "best licensed" after the licence was withdrawn):

  * **The file.** Top-level keys are ``schema``, ``package``, ``generated``,
    ``status_vocabulary`` and ``claims``. The vocabulary names only statuses of the
    schema (``discovery_common.load_claims`` refuses any other: exit 2). Ids are unique and share one namespace (``e4b.``, ``gnf4.``). A row
    carries only fields the schema defines, and ``area``, ``tier``, ``validity``,
    ``row_status`` and ``parity_verdict`` take only their listed values. A
    ``package`` field names this repository's package.
  * **Locations.** An evidence path or a ``quoted_in`` entry is a location:
    ``path`` or ``path#anchor``. The path is a FILE in the git tree at HEAD
    (``git ls-files``, because ``*.log`` is gitignored and a receipt log is evidence
    only once it is force-added). It is relative, never through ``..``, never a
    directory, and carries no annotation or glob characters. The anchor is ``L<n>``
    or ``L<n>-L<m>`` (lines the file has), or the GitHub anchor of a Markdown
    heading in that file -- the fragment github.com itself scrolls to, so a
    location is a working link (the site links a row's ``evidence[0]`` at the
    pinned commit).
  * **Evidence.** Each ``evidence[]`` entry is a location string, or
    ``{"url": ...}`` (an issue or pull request of one of the system's
    repositories), or ``{"repository": <package>, "path": <location>}`` (a file in
    the other package's repository, resolved in a ``--sibling`` checkout of it and
    listed as SKIP without one). ``measured`` / ``confirmed`` / ``verified`` rows
    need a non-empty ``evidence`` whose FIRST entry is a location string (the
    receipt a reader lands on); ``measured-private`` rows need a non-empty
    ``evidence_private`` (paths outside the repository: listed, never resolved).
  * **Dates.** ``measured_on`` is required on measured / measured-private /
    confirmed / verified rows; wherever present it is an ISO calendar date.
  * **Successors.** ``superseded_by`` sits only on ``superseded`` rows (required)
    and on ``retired`` rows that name a restatement (optional). It names an ACTIVE
    row DIRECTLY -- no chains; when a successor is itself superseded, every row
    that pointed at it is re-pointed -- and that row lists it in ``supersedes``.
    Every ``supersedes`` entry exists, is superseded or retired, and names this row
    as its ``superseded_by``. ``retired`` rows carry a non-empty ``retired_reason``
    and no other row does.
  * **Placeholders.** ``claim`` and ``notes`` of an ACTIVE row carry no
    ``pending``, ``TBD`` or ``TODO``; an ``open`` row may.
  * **Licences.** An ACTIVE row whose ``claim`` asserts a licence -- an occurrence
    of "licensed" that no negation immediately precedes and that is not the
    citation form ``licensed by `<id>` `` -- carries ``licensed_by``, the ACTIVE
    row whose receipt holds the verdict (a verdict row names itself). A citation
    must name a row of this register that itself carries ``licensed_by``, in
    ``claim`` and in ``notes``. ``pack_fingerprint`` is ``sha256:<64 hex>``, and an
    artifact-backed licence names the same bytes on the row and on its verdict.

    python scripts/check_claims_register.py                                # CI gate
    python scripts/check_claims_register.py --sibling ../experts4bit-qlora  # and the cross-repository entries

Exit 0 when clean, 1 on findings, 2 when the check itself cannot run (an unreadable
register, or a ``--sibling`` that is not the other package of this system -- the
cross-repository entries would be skipped while the summary called them checked).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discovery_common import ACTIVE_STATUSES, ContractError, load_claims, load_pyproject, read_text  # noqa: E402

CLAIMS = "docs/claims.json"
MANIFEST = "docs/system-manifest.json"

#: A register's ``status_vocabulary`` names a subset of the schema's eight statuses; ``discovery_common.load_claims``
#: refuses any other (a ContractError, so exit 2), and refuses a row whose status the file does not define.
PUBLIC_RUN = frozenset({"measured", "confirmed", "verified"})
PRIVATE_RUN = frozenset({"measured-private"})
DATED = PUBLIC_RUN | PRIVATE_RUN
#: Every field a row may carry (docs/claims-schema.md, "Fields").
ROW_FIELDS = frozenset({
    "id", "package", "area", "claim", "value", "unit", "model", "hardware", "conditions", "measured_on",
    "status", "tier", "evidence", "evidence_private", "supersedes", "superseded_by", "retired_reason",
    "retired_on", "licensed_by", "pack_fingerprint", "quoted_in", "notes",
    "validity", "row_status", "parity_verdict", "row_reason",
})
TOP_LEVEL = frozenset({"schema", "package", "generated", "status_vocabulary", "claims"})
ENUMS: dict[str, frozenset] = {
    "area": frozenset({"train", "offload", "serve", "kernel", "parity", "quality", "provenance", "portability",
                       "roadmap"}),
    "tier": frozenset({"confirmed", "measured", "projected"}),
    "validity": frozenset({"VALID", "VOID"}),
    "row_status": frozenset({"OK", "HARNESS_ERROR", "REFUSED", "EXPERIMENTAL"}),
    "parity_verdict": frozenset({"REF", "PASS", "VOID", "no pair", None}),
}

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PLACEHOLDER = re.compile(r"\b(pending|TBD|TODO)\b", re.I)
_PACK_FP = re.compile(r"^sha256:[0-9a-f]{64}$")
_LICENSED = re.compile(r"\blicensed\b", re.I)
#: A negation that ends right before ONE occurrence of "licensed" disclaims that occurrence only (looked for in
#: the ``_NEGATION_WINDOW`` characters before it). "unlicensed" never matches ``_LICENSED``; "un-licensed" does.
_NEGATION = re.compile(r"(?:\bun-?|\bnot\s+(?:a\s+)?|\bnever\s+|\bno\s+)$", re.I)
_NEGATION_WINDOW = 12
_CITATION = re.compile(r"\blicensed by `([a-z0-9]+\.[A-Za-z0-9._+-]+)`", re.I)
_UNCLEAN = re.compile(r"[\s()*?\[\]~]")
_LINES = re.compile(r"^L(\d+)(?:-L(\d+))?$")
_ATX = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_ISSUE = re.compile(r"^(https://github\.com/[^/\s]+/[^/\s]+)/(?:issues|pull)/\d+$")


# ------------------------------------------------------------------ anchors --

def github_slug(heading: str) -> str:
    """The fragment github.com gives a Markdown heading (before de-duplication):
    link text kept and its target dropped, tags dropped, lower-cased, every
    character that is not a letter, digit, ``_``, ``-`` or space removed, spaces to
    ``-``. ``## 0.24.0 — 2026-08-31`` -> ``0240--2026-08-31``."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = re.sub(r"<[^>]*>", "", text).lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def github_anchors(text: str) -> set[str]:
    """Every heading fragment of a Markdown document as github.com assigns them:
    ATX headings outside fenced code, the second of two equal slugs ``-1``, then
    ``-2``, and so on."""
    seen: dict[str, int] = {}
    out: set[str] = set()
    fence = None
    for line in text.splitlines():
        m = _FENCE.match(line)
        if m:
            if fence is None:
                fence = m.group(1)[0]
            elif m.group(1)[0] == fence:
                fence = None
            continue
        if fence is not None:
            continue
        h = _ATX.match(line)
        if not h:
            continue
        base = github_slug(h.group(2))
        n = seen.get(base, 0)
        out.add(base if n == 0 else f"{base}-{n}")
        seen[base] = n + 1
    return out


# ------------------------------------------------------------------- files --

def tracked_files(root: Path) -> frozenset[str] | None:
    """Every path in the git tree at ``root`` (``git ls-files``), or None when ``root``
    is not a checkout or git is unavailable -- the working tree then stands in and
    the output says so."""
    try:
        p = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if p.returncode != 0:
        return None
    return frozenset(x for x in p.stdout.split("\0") if x)


def location_problem(root: Path, loc, tracked: frozenset[str] | None) -> str | None:
    """Why ``loc`` is not ``path`` / ``path#anchor`` of a file of the repository at
    ``root``, or None when it is."""
    if not isinstance(loc, str) or not loc.strip():
        return f"{loc!r} is not a non-empty string"
    path, _, anchor = loc.partition("#")
    if path.startswith(("/", "~")) or ".." in path.split("/"):
        return f"{loc!r} is not a repository-relative path (absolute, or through '..')"
    if not path or _UNCLEAN.search(path):
        return (f"{loc!r} is annotated or a glob -- a location is a bare path, optionally #anchor; what a path was "
                "run with goes in notes (docs/claims-schema.md)")
    if tracked is not None:
        if path not in tracked:
            if any(t.startswith(path.rstrip("/") + "/") for t in tracked):
                return f"{path!r} is a directory -- a location is a file"
            return f"{path!r} is not in the git tree at HEAD (an untracked or ignored file is not evidence until it is added)"
    elif (root / path).is_dir():
        return f"{path!r} is a directory -- a location is a file"
    elif not (root / path).is_file():
        return f"{path!r} does not exist in the working tree (not a git checkout, so HEAD could not be read)"
    if not anchor:
        return None if not loc.endswith("#") else f"{loc!r} has an empty anchor"
    text = read_text(root / path)
    m = _LINES.match(anchor)
    if m:
        total = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        if not 1 <= lo <= hi <= total:
            return f"{loc!r}: lines {lo}-{hi} are outside the file ({total} lines)"
        return None
    if not path.endswith(".md"):
        return f"{loc!r}: a non-Markdown file takes a line anchor (#L<n> or #L<n>-L<m>), not {anchor!r}"
    if anchor not in github_anchors(text):
        return f"{loc!r}: {path} has no heading whose GitHub anchor is {anchor!r}"
    return None


# ------------------------------------------------------------------ system --

class System:
    """What docs/system-manifest.json says about the packages of this system: this
    repository's pyproject name, and every package's repository URL."""

    def __init__(self, package: str | None, packages: dict[str, str] | None = None):
        self.package = package
        self.packages = dict(packages or {})

    @property
    def repositories(self) -> set[str]:
        return {u.rstrip("/") for u in self.packages.values() if u}


def load_system(root: Path) -> System:
    py = load_pyproject(root)
    name = str(py.get("name")) if py.get("name") else None
    pk: dict[str, str] = {}
    man = root / MANIFEST
    if man.is_file():
        doc = json.loads(read_text(man))
        for p in (doc.get("packages") or {}).values():
            if isinstance(p, dict) and p.get("package"):
                pk[str(p["package"])] = str(p.get("repository") or "")
    return System(name, pk)


# ---------------------------------------------------------------- licences --

def licence_citations(text: str) -> list[str]:
    """The ids ``text`` cites in the form ``licensed by `<id>` ``, in order."""
    return _CITATION.findall(text)


def asserts_licence(sentence: str) -> re.Match | None:
    """The first occurrence of "licensed" in ``sentence`` that is neither a citation
    (``licensed by `<id>` ``, resolved separately) nor immediately preceded by a
    negation; None when every occurrence is one or the other."""
    for m in _LICENSED.finditer(sentence):
        if _CITATION.match(sentence, m.start()):
            continue
        if not _NEGATION.search(sentence[max(0, m.start() - _NEGATION_WINDOW):m.start()]):
            return m
    return None


def iso_date(value) -> bool:
    if not isinstance(value, str) or not _ISO.match(value):
        return False
    try:
        _dt.date.fromisoformat(value)
    except ValueError:
        return False
    return True


# ------------------------------------------------------------------- check --

class Sibling:
    """A ``--sibling`` checkout of the other package: its root, name and git tree."""

    def __init__(self, root: Path, package: str, tracked: frozenset[str] | None):
        self.root, self.package, self.tracked = root, package, tracked


class Result:
    """What one run found: findings, cross-repository skips, and the counts."""

    def __init__(self, findings: list[str], skips: list[str], n_claims: int, n_active: int, tracked: bool):
        self.findings, self.skips = findings, skips
        self.n_claims, self.n_active, self.tracked = n_claims, n_active, tracked


def open_sibling(path: Path, system: System) -> Sibling:
    """A ``--sibling`` checkout, refused (ContractError -> exit 2) unless it is the
    OTHER package of this system: pointing it anywhere else would leave every
    cross-repository entry unchecked under a green summary."""
    try:
        name = load_pyproject(path).get("name")
    except (OSError, ValueError) as e:
        raise ContractError(f"cannot read the sibling's pyproject.toml at {path}: {e}") from e
    if not name:
        raise ContractError(f"the sibling {path} has no pyproject project.name; cross-repository evidence would be "
                            "skipped, not checked")
    if name == system.package:
        raise ContractError(f"the sibling {path} is {name!r}, this repository's own package")
    if system.packages and name not in system.packages:
        raise ContractError(f"the sibling {path} is {name!r}, which is not a package of this system "
                            f"({sorted(system.packages)})")
    return Sibling(path, str(name), tracked_files(path))


def _evidence(root: Path, cid: str, i: int, e, tracked, system: System, sibling: Sibling | None,
              skips: list[str]) -> list[str]:
    where = f"{cid}: evidence[{i}]"
    if isinstance(e, str):
        if e.startswith(("http://", "https://")):
            return [f"{where}: {e!r} is a URL -- write {{\"url\": ...}} for an issue or pull request"]
        p = location_problem(root, e, tracked)
        return [f"{where}: {p}"] if p else []
    if not isinstance(e, dict):
        return [f"{where}: unsupported type {type(e).__name__}"]
    keys = set(e)
    if keys == {"url"}:
        u = str(e["url"])
        m = _ISSUE.match(u)
        if not m:
            return [f"{where}: url {u!r} is not a github.com issue or pull request"]
        if m.group(1) not in system.repositories:
            return [f"{where}: url {u!r} is not in one of this system's repositories ({sorted(system.repositories)})"]
        return []
    if keys == {"repository", "path"}:
        pkg, loc = str(e["repository"]), e["path"]
        if pkg == system.package:
            return [f"{where}: repository {pkg!r} is this repository -- cite the path directly"]
        if pkg not in system.packages:
            return [f"{where}: repository {pkg!r} is not a package of this system ({sorted(system.packages)})"]
        if isinstance(loc, str) and (loc.startswith(("/", "~")) or ".." in loc.partition("#")[0].split("/")):
            return [f"{where}: path {loc!r} is not a repository-relative path"]
        if sibling is None or sibling.package != pkg:
            skips.append(f"{cid}: {pkg}:{loc} -- not resolved (no --sibling checkout of {pkg} given)")
            return []
        p = location_problem(sibling.root, loc, sibling.tracked)
        return [f"{where}: in the sibling {pkg}: {p}"] if p else []
    return [f"{where}: an evidence object is {{\"url\"}} or {{\"repository\", \"path\"}}, got {sorted(keys)}"]


def check_claims(root: Path, doc: dict, claims: list[dict], *, sibling: Sibling | None = None,
                 tracked: frozenset[str] | None = None, system: System | None = None,
                 skips: list[str] | None = None) -> list[str]:
    """Every finding over one register (``doc`` is the whole file, ``claims`` its rows)."""
    system = system or load_system(root)
    skips = skips if skips is not None else []
    out: list[str] = []
    by = {c["id"]: c for c in claims}

    for k in sorted(set(doc) - TOP_LEVEL):
        out.append(f"top-level key {k!r} is not in the schema ({sorted(TOP_LEVEL)})")
    heads = {str(c["id"]).split(".", 1)[0] for c in claims}
    if len(heads) > 1:
        out.append(f"ids span more than one namespace ({sorted(heads)}); a register is one package's")

    def active(x: str) -> bool:
        return x in by and by[x].get("status") in ACTIVE_STATUSES

    for c in claims:
        cid, st = c["id"], c.get("status")
        for k in sorted(set(c) - ROW_FIELDS):
            out.append(f"{cid}: field {k!r} is not in the schema (docs/claims-schema.md, Fields)")
        for k, allowed in ENUMS.items():
            if k in c and c[k] not in allowed:
                out.append(f"{cid}: {k} {c[k]!r} is not one of {sorted(allowed, key=str)}")
        if "package" in c and system.package and c["package"] != system.package:
            out.append(f"{cid}: package {c['package']!r} is not this repository's ({system.package!r})")

        ev = c.get("evidence")
        if ev is None:
            ev = []
        elif not isinstance(ev, list):
            out.append(f"{cid}: evidence must be a list")
            ev = []
        for i, e in enumerate(ev):
            out += _evidence(root, cid, i, e, tracked, system, sibling, skips)
        evp = c.get("evidence_private")
        if evp is not None and (not isinstance(evp, list) or any(not isinstance(x, str) or not x.strip() for x in evp)):
            out.append(f"{cid}: evidence_private must be a list of non-empty strings")
        if st in PUBLIC_RUN:
            if not ev:
                out.append(f"{cid}: status {st!r} needs a public receipt: evidence is empty")
            elif not isinstance(ev[0], str) or ev[0].startswith(("http://", "https://")):
                out.append(f"{cid}: status {st!r}: evidence[0] must be a location (the receipt a reader lands on), "
                           f"got {ev[0]!r}")
        if st in PRIVATE_RUN and not evp:
            out.append(f"{cid}: status 'measured-private' needs evidence_private (where the receipt lives)")

        if st in DATED and "measured_on" not in c:
            out.append(f"{cid}: status {st!r} needs measured_on (the ISO date of the run)")
        elif "measured_on" in c and not iso_date(c["measured_on"]):
            out.append(f"{cid}: measured_on {c['measured_on']!r} is not an ISO calendar date (YYYY-MM-DD)"
                       + (" -- omit the field instead of writing null" if c["measured_on"] is None else ""))

        sb = c.get("superseded_by")
        if st == "superseded" and not sb:
            out.append(f"{cid}: status 'superseded' needs superseded_by (the claim to quote instead)")
        elif sb:
            if st not in ("superseded", "retired"):
                out.append(f"{cid}: superseded_by on a {st!r} row -- only a superseded row, or a retired row that "
                           "names its restatement, carries one")
            elif sb not in by:
                out.append(f"{cid}: superseded_by {sb!r} is not in the register")
            elif not active(sb):
                out.append(f"{cid}: superseded_by {sb!r} has status {by[sb].get('status')!r}; it must name the ACTIVE "
                           "row directly (re-point it at that row's own successor)")
            elif cid not in (by[sb].get("supersedes") or []):
                out.append(f"{cid}: superseded_by {sb!r} does not list {cid!r} in its supersedes")
        for old in c.get("supersedes") or []:
            if old not in by:
                out.append(f"{cid}: supersedes {old!r}, which is not in the register")
            elif by[old].get("status") not in ("superseded", "retired"):
                out.append(f"{cid}: supersedes {old!r}, whose status is {by[old].get('status')!r}, not superseded "
                           "or retired")
            elif by[old].get("superseded_by") != cid:
                out.append(f"{cid}: supersedes {old!r}, whose superseded_by is {by[old].get('superseded_by')!r}, "
                           f"not {cid!r}")
        reason = c.get("retired_reason")
        if st == "retired" and not (isinstance(reason, str) and reason.strip()):
            out.append(f"{cid}: status 'retired' needs retired_reason (one sentence, with the measurement that "
                       "retired it)")
        elif st != "retired" and "retired_reason" in c:
            out.append(f"{cid}: retired_reason on a {st!r} row -- one status per row: retire it, or move the text "
                       "to notes")

        qi = c.get("quoted_in")
        if qi is not None and not isinstance(qi, list):
            out.append(f"{cid}: quoted_in must be a list")
        for i, q in enumerate(qi if isinstance(qi, list) else []):
            p = location_problem(root, q, tracked)
            if p:
                out.append(f"{cid}: quoted_in[{i}]: {p}")

        if st in ACTIVE_STATUSES:
            for fld in ("claim", "notes"):
                m = _PLACEHOLDER.search(str(c.get(fld, "")))
                if m:
                    out.append(f"{cid}: the {fld} of an active ({st}) row carries the placeholder word {m.group(0)!r} "
                               "-- state what is measured, or make the row open")
            sentence = str(c.get("claim", ""))
            m = asserts_licence(sentence)
            if m and not c.get("licensed_by"):
                out.append(f"{cid}: the sentence asserts a licence ({sentence[max(0, m.start() - 20):m.end()]!r}) and "
                           "the row has no licensed_by (the verdict claim id) -- add it, or reword")
        for fld in ("claim", "notes"):
            for cited in licence_citations(str(c.get(fld, ""))):
                if cited not in by:
                    out.append(f"{cid}: {fld} cites `{cited}` as a licence, which is not in the register")
                elif not by[cited].get("licensed_by"):
                    out.append(f"{cid}: {fld} cites `{cited}` as a licence, but that row carries no licensed_by "
                               "(neither a licensed row nor a verdict row)")
        lb = c.get("licensed_by")
        if lb is not None:
            if lb not in by:
                out.append(f"{cid}: licensed_by {lb!r} is not in the register")
            elif not active(lb):
                out.append(f"{cid}: licensed_by {lb!r} has status {by[lb].get('status')!r}; a licence comes from an "
                           "active verdict row")
        pf = c.get("pack_fingerprint")
        if pf is not None and (not isinstance(pf, str) or not _PACK_FP.fullmatch(pf)):
            out.append(f"{cid}: pack_fingerprint {pf!r} is not sha256:<64 lowercase hex>")
        if st in ACTIVE_STATUSES and lb and active(lb):
            row_fp = pf if isinstance(pf, str) else None
            ver_fp = by[lb].get("pack_fingerprint")
            if row_fp or ver_fp:
                if not row_fp:
                    out.append(f"{cid}: licensed_by {lb} carries pack_fingerprint but this row does not -- an "
                               "artifact-backed licence names the same bytes on both rows")
                elif not ver_fp:
                    out.append(f"{cid}: pack_fingerprint is set but licensed_by {lb} has none -- the verdict row is "
                               "the licence basis and must name the bytes")
                elif row_fp != ver_fp:
                    out.append(f"{cid}: pack_fingerprint {row_fp} != licensed_by {lb}'s {ver_fp}")
    return out


def check(root: Path, *, claims_path: str = CLAIMS, sibling: Path | None = None) -> Result:
    """Load the register at ``root`` and check it. Raises ContractError when the check
    cannot run (an unreadable register, or an unusable ``--sibling``)."""
    claims, _vocab = load_claims(root, claims_path)
    doc = json.loads(read_text(root / claims_path))
    system = load_system(root)
    sib = open_sibling(sibling, system) if sibling is not None else None
    tracked = tracked_files(root)
    skips: list[str] = []
    findings = check_claims(root, doc, claims, sibling=sib, tracked=tracked, system=system, skips=skips)
    n_active = sum(1 for c in claims if c.get("status") in ACTIVE_STATUSES)
    return Result(findings, skips, len(claims), n_active, tracked is not None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--claims", default=CLAIMS)
    ap.add_argument("--sibling", default=None, metavar="PATH",
                    help="a checkout of the other package's repository: cross-repository evidence is resolved there")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    try:
        r = check(root, claims_path=a.claims, sibling=Path(a.sibling).resolve() if a.sibling else None)
    except (ContractError, OSError, ValueError) as e:
        print(f"FAIL: {e}")
        return 2
    if not r.tracked:
        print(f"NOTE: {root} is not a git checkout (or git is unavailable): locations are checked against the working "
              "tree, not the git tree")
    for s in r.skips:
        print(f"SKIP: {s}")
    for f in r.findings:
        print(f"FAIL: {f}")
    if r.findings:
        print(f"FAIL: {len(r.findings)} finding(s) in {a.claims}; the schema is docs/claims-schema.md")
        return 1
    tree = "the git tree" if r.tracked else "the working tree (not a checkout)"
    print(f"OK: {a.claims}: {r.n_claims} claims ({r.n_active} active) -- every location is a file in {tree} with a "
          "resolving anchor, evidence forms and first entries are the schema's, dated rows carry an ISO measured_on, "
          "successors are direct and named back, no active row is pending, licence labels name their verdict row"
          + (f"; {len(r.skips)} cross-repository entr{'y' if len(r.skips) == 1 else 'ies'} not resolved without "
             "--sibling" if r.skips else "")
          + (f"; cross-repository entries resolved in {a.sibling}" if a.sibling else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
