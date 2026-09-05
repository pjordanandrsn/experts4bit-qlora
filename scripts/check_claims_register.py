#!/usr/bin/env python3
"""The claims register's own hygiene, checked mechanically (docs/claims-schema.md,
"What the register check enforces"). Standard library only; no network (one
``git ls-files`` of this checkout for the evidence rule).

Every check keys on the register's STRUCTURE, which is exactly what the
2026-09-05 audit found no check reading: three ``measured`` rows named receipt
logs that were never committed, annotated and scratch paths sat in
``evidence[]``, ``superseded`` rows had no successor, ``notes`` said "pending"
for rows measured days earlier, and a sentence said "best licensed" after the
register had withdrawn the licence. For every claim:

  * ``evidence[]`` entries resolve: a string is a repository-relative path of
    a FILE in the git tree -- ``git ls-files``, not the working tree, because
    ``*.log`` is gitignored and a receipt log is only evidence once it is
    force-added -- with an optional ``#fragment`` (not checked); never
    absolute, never through ``..``, never a bare directory (outside a git
    checkout the working tree is used and the output says so); a dict is
    ``{"repository": "owner/name", "path": ...}`` (a file in another
    repository, checked when ``--sibling PATH`` names a checkout whose
    ``pyproject`` slug matches) or ``{"url": "https://github.com/..."}`` (an
    issue or pull request on github.com). Free text, globs and annotations are
    findings -- the site links ``evidence[0]`` blindly.
  * ``measured`` / ``measured-private`` / ``verified`` / ``confirmed`` rows carry
    ``measured_on`` as an ISO date (``YYYY-MM-DD``).
  * ``superseded`` rows carry ``superseded_by``; every ``superseded_by``
    (on a superseded or a retired row), followed through other superseded
    rows and never a cycle, reaches an ACTIVE claim; an ACTIVE row never
    carries ``superseded_by``; ``retired`` rows carry a non-empty
    ``retired_reason`` and no other row carries one; every ``supersedes`` id
    exists.
  * ``quoted_in`` entries are ``<path>[#fragment][ free text]`` whose path exists.
  * no ``notes`` or ``claim`` of an ACTIVE row says "pending" or "TBD" (an
    ``open`` row may).
  * an ACTIVE row whose ``claim`` sentence asserts a licence -- an occurrence
    of the word "licensed" that no negation immediately precedes ("not
    licensed", "not a licensed", "never licensed", "no licensed";
    "unlicensed" is its own word) and that is not the citation form below --
    carries ``licensed_by``, the id of the ACTIVE claim whose receipt holds
    the K8 verdict (a row whose own receipt holds it names itself). The rule
    is per occurrence: one "unlicensed" elsewhere in the sentence does not
    excuse a bare "licensed stack" beside it. Every ``licensed_by`` resolves
    to an ACTIVE row.
  * the citation form ``licensed by `<id>` `` refers to ANOTHER row's licence
    and asserts none of its own (the p37 head-to-head names the stack
    licensed by the bo6c verdict it could not reproduce): the row needs no
    ``licensed_by`` for it -- ``licensed_by`` says the row's OWN configuration
    is licensed and never goes on a row whose configuration is not -- and
    ``<id>`` must be a claim in the register that itself carries
    ``licensed_by`` (a licensed row, or a verdict row, which names itself). A
    citation of a missing id, or of a row with no ``licensed_by``, is a
    finding, in ``claim`` and in ``notes``.
  * ids are unique and every status is in the file's vocabulary
    (``discovery_common.load_claims``).

    python scripts/check_claims_register.py                      # CI gate
    python scripts/check_claims_register.py --sibling ../grouped-nf4-gemm

Exit 0 when clean, 1 on findings, 2 when the check itself cannot run -- a
``--sibling`` whose owner/name slug cannot be resolved is exit 2, not a pass
with the cross-repository entries silently skipped.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discovery_common import ACTIVE_STATUSES, ContractError, load_claims, load_pyproject, self_slug  # noqa: E402

CLAIMS = "docs/claims.json"
#: Statuses whose evidence is a run: the date of that run is required.
DATED = frozenset({"measured", "measured-private", "verified", "confirmed"})
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PENDING = re.compile(r"\b(pending|TBD)\b", re.I)
_LICENSED = re.compile(r"\blicensed\b", re.I)
#: A negation that ends right before ONE occurrence of "licensed" disclaims that
#: occurrence only (looked for in the ``_NEGATION_WINDOW`` characters before it).
#: "unlicensed" never matches ``_LICENSED`` (no word boundary); "un-licensed" does.
_NEGATION = re.compile(r"(?:\bun-?|\bnot\s+(?:a\s+)?|\bnever\s+|\bno\s+)$", re.I)
_NEGATION_WINDOW = 12
#: The citation form: "licensed by `<id>`" refers to that row's licence and asserts none of its own.
_CITATION = re.compile(r"\blicensed by `(e4b\.[A-Za-z0-9._+-]+)`", re.I)
_ISSUE_URL = re.compile(r"^https://github\.com/[^/\s]+/[^/\s]+/(issues|pull)/\d+$")
_SLUG = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _split_path(entry: str) -> str:
    """``docs/METHODOLOGY.md#13 (corrected in #353)`` -> ``docs/METHODOLOGY.md``."""
    return entry.split(" ", 1)[0].split("#", 1)[0]


def licence_citations(text: str) -> list[str]:
    """The ids ``text`` cites in the form ``licensed by `<id>` ``, in order."""
    return _CITATION.findall(text)


def asserts_licence(sentence: str) -> re.Match | None:
    """The first occurrence of "licensed" in ``sentence`` that is neither a
    citation (``licensed by `<id>` ``, resolved separately) nor immediately
    preceded by a negation; None when every occurrence is one or the other."""
    for m in _LICENSED.finditer(sentence):
        if _CITATION.match(sentence, m.start()):
            continue
        if not _NEGATION.search(sentence[max(0, m.start() - _NEGATION_WINDOW):m.start()]):
            return m
    return None


def tracked_files(root: Path) -> frozenset[str] | None:
    """Every path in the git tree at ``root`` (one ``git ls-files``, relative to
    ``root``), or None when ``root`` is not a checkout or git is unavailable --
    the caller then falls back to the working tree and says so."""
    try:
        p = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if p.returncode != 0:
        return None
    return frozenset(x for x in p.stdout.split("\0") if x)


def _not_a_repository_file(root: Path, path: str, tracked: frozenset[str] | None) -> str | None:
    """Why ``path`` is not a file of this repository, or None when it is: absolute
    or escaping through ``..``, a directory, or not in the git tree (``tracked``;
    the working tree when None)."""
    if path.startswith("/") or ".." in path.split("/"):
        return f"{path!r} is not a repository-relative path (absolute, or through '..')"
    if tracked is not None:
        if path in tracked:
            return None
        if any(t.startswith(path.rstrip("/") + "/") for t in tracked):
            return f"{path!r} is a directory -- evidence is a file"
        return f"{path!r} is not in the git tree at HEAD (an untracked or ignored file is not evidence until it is added)"
    if (root / path).is_dir():
        return f"{path!r} is a directory -- evidence is a file"
    if not (root / path).is_file():
        return f"{path!r} does not exist in the working tree (not a git checkout, so HEAD could not be read)"
    return None


def check_evidence(root: Path, c: dict, sibling: Path | None, sibling_slug: str | None,
                   tracked: frozenset[str] | None = None) -> list[str]:
    out = []
    cid = c["id"]
    ev = c.get("evidence")
    if ev is None:
        return out
    if not isinstance(ev, list):
        return [f"{cid}: evidence must be a list"]
    for i, e in enumerate(ev):
        where = f"{cid}: evidence[{i}]"
        if isinstance(e, str):
            if not e.strip():
                out.append(f"{where}: empty entry")
                continue
            if " " in e or "*" in e or "(" in e:
                out.append(f"{where}: {e!r} is annotated or a glob -- evidence is a bare repository path "
                           f"(optionally #fragment) or a structured cross-repository / issue entry")
                continue
            path = e.split("#", 1)[0]
            if e.startswith(("http://", "https://")):
                out.append(f"{where}: {e!r} is a URL -- use {{\"url\": ...}} for an issue or pull request")
            else:
                problem = _not_a_repository_file(root, path, tracked)
                if problem:
                    out.append(f"{where}: {problem}")
        elif isinstance(e, dict):
            if "url" in e and set(e) == {"url"}:
                if not _ISSUE_URL.match(str(e["url"])):
                    out.append(f"{where}: url {e['url']!r} is not a github.com issue or pull request URL")
            elif "repository" in e and "path" in e and set(e) == {"repository", "path"}:
                repo, path = str(e["repository"]), str(e["path"])
                if not _SLUG.match(repo):
                    out.append(f"{where}: repository {repo!r} is not owner/name")
                if not path or path.startswith("/") or " " in path or ".." in path.split("/"):
                    out.append(f"{where}: path {path!r} is not a bare repository-relative path")
                if sibling is not None and sibling_slug and repo.lower() == sibling_slug.lower():
                    if not (sibling / path.split("#", 1)[0]).is_file():
                        out.append(f"{where}: {repo}:{path} is not a file in the sibling checkout {sibling}")
            else:
                out.append(f"{where}: a structured entry is {{\"repository\", \"path\"}} or {{\"url\"}}, got {sorted(e)}")
        else:
            out.append(f"{where}: unsupported type {type(e).__name__}")
    return out


def _follow_successors(cid: str, by: dict[str, dict]) -> str | None:
    """Walk ``superseded_by`` from ``cid`` until an ACTIVE row; a chain of superseded
    rows is fine (behaves -> chunk-free -> no-reference), a missing id, a cycle
    or a chain that ends on an inactive row is the finding returned."""
    seen, cur = [cid], by[cid].get("superseded_by")
    while cur:
        if cur not in by:
            return f"superseded_by {cur!r} is not in the register (chain {' -> '.join(seen)})"
        if cur in seen:
            return f"superseded_by chain is a cycle ({' -> '.join(seen + [cur])})"
        seen.append(cur)
        st = by[cur].get("status")
        if st in ACTIVE_STATUSES:
            return None
        if st != "superseded":
            return f"superseded_by chain ends on {cur!r} with status {st!r}; a successor must be active ({' -> '.join(seen)})"
        cur = by[cur].get("superseded_by")
    return f"superseded_by chain ends on {seen[-1]!r}, which names no successor ({' -> '.join(seen)})"


def check_claims(root: Path, claims: list[dict], sibling: Path | None = None, sibling_slug: str | None = None,
                 tracked: frozenset[str] | None = None) -> list[str]:
    """Every finding over ``claims``; ``tracked`` is ``tracked_files(root)`` (None:
    the working tree stands in for the git tree)."""
    findings: list[str] = []
    by = {c["id"]: c for c in claims}

    def active(cid: str) -> bool:
        return cid in by and by[cid].get("status") in ACTIVE_STATUSES

    for c in claims:
        cid, st = c["id"], c.get("status")
        findings += check_evidence(root, c, sibling, sibling_slug, tracked)
        if st in DATED:
            m = c.get("measured_on")
            if not isinstance(m, str) or not _ISO.match(m):
                findings.append(f"{cid}: status {st!r} needs measured_on as YYYY-MM-DD, got {m!r}")
        sb = c.get("superseded_by")
        if st == "superseded" and not sb:
            findings.append(f"{cid}: superseded without superseded_by")
        elif sb:
            if st in ACTIVE_STATUSES:
                findings.append(f"{cid}: an active row carries superseded_by {sb!r} -- a row is superseded or active, "
                                "not both")
            problem = _follow_successors(cid, by)
            if problem:
                findings.append(f"{cid}: {problem}")
        reason = str(c.get("retired_reason") or "").strip()
        if st == "retired" and not reason:
            findings.append(f"{cid}: retired without retired_reason")
        elif st != "retired" and "retired_reason" in c:
            findings.append(f"{cid}: retired_reason on a {st!r} row -- one status per row: retire it, or move the "
                            "text to notes")
        for s in c.get("supersedes") or []:
            if s not in by:
                findings.append(f"{cid}: supersedes {s!r}, which is not in the register")
        for i, q in enumerate(c.get("quoted_in") or []):
            if not isinstance(q, str) or not q.strip():
                findings.append(f"{cid}: quoted_in[{i}] is empty")
                continue
            path = _split_path(q)
            if not (root / path).exists():
                findings.append(f"{cid}: quoted_in[{i}] {q!r}: {path!r} does not exist")
        if st in ACTIVE_STATUSES:
            for field in ("claim", "notes"):
                m = _PENDING.search(str(c.get(field, "")))
                if m:
                    findings.append(f"{cid}: active row's {field} says {m.group(0)!r} -- state what is measured, or open a row")
            sentence = str(c.get("claim", ""))
            m = asserts_licence(sentence)
            if m and not c.get("licensed_by"):
                findings.append(f"{cid}: the sentence asserts a licence ({sentence[max(0, m.start() - 20):m.end()]!r}) "
                                "and the row has no licensed_by (the K8 verdict claim id) -- add it, or reword")
        for field in ("claim", "notes"):
            for cited in licence_citations(str(c.get(field, ""))):
                if cited not in by:
                    findings.append(f"{cid}: {field} cites `{cited}` as a licence, which is not in the register")
                elif not by[cited].get("licensed_by"):
                    findings.append(f"{cid}: {field} cites `{cited}` as a licence, but that row carries no licensed_by "
                                    "(neither a licensed row nor a verdict row)")
        lb = c.get("licensed_by")
        if lb is not None:
            if lb not in by:
                findings.append(f"{cid}: licensed_by {lb!r} is not in the register")
            elif not active(lb):
                findings.append(f"{cid}: licensed_by {lb!r} has status {by[lb].get('status')!r}; a licence comes from an active verdict row")
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--claims", default=CLAIMS)
    ap.add_argument("--sibling", default=None, metavar="PATH",
                    help="a checkout of the sibling repository: cross-repository evidence naming it is verified there")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    try:
        claims, _vocab = load_claims(root, a.claims)
    except (ContractError, OSError, ValueError) as e:
        print(f"FAIL: {e}")
        return 2
    sibling = Path(a.sibling).resolve() if a.sibling else None
    sibling_slug = None
    if sibling is not None:
        try:
            sibling_slug = self_slug(sibling, load_pyproject(sibling))
        except (OSError, ValueError) as e:
            print(f"FAIL: cannot read the sibling's pyproject: {e}")
            return 2
        if not sibling_slug:
            print(f"FAIL: cannot resolve the sibling's owner/name from {sibling} (pyproject [project.urls] or its git "
                  "remote): cross-repository evidence would be skipped, not checked")
            return 2
    tracked = tracked_files(root)
    if tracked is None:
        print(f"NOTE: {root} is not a git checkout (or git is unavailable): evidence paths are checked against the "
              "working tree, not the git tree")
    findings = check_claims(root, claims, sibling, sibling_slug, tracked)
    for f in findings:
        print(f"FAIL: {f}")
    if findings:
        print(f"FAIL: {len(findings)} finding(s) in {a.claims}")
        return 1
    n_active = sum(1 for c in claims if c.get("status") in ACTIVE_STATUSES)
    n_lic = sum(1 for c in claims if c.get("licensed_by"))
    tree = "the git tree" if tracked is not None else "the working tree (not a checkout)"
    print(f"OK: {a.claims}: {len(claims)} claims ({n_active} active) -- every evidence entry is a file in {tree} or a "
          f"structured entry, dated rows carry an ISO measured_on, superseded/retired rows resolve, quoted_in "
          f"resolves, no active row is pending, {n_lic} licence label(s) name their verdict row"
          + (f"; cross-repository evidence checked against {sibling}" if sibling else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
