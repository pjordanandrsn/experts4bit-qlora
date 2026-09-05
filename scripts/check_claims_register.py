#!/usr/bin/env python3
"""The claims register's own hygiene, checked mechanically (docs/claims-schema.md,
"What the register check enforces"). Standard library only; no network.

Every check keys on the register's STRUCTURE, which is exactly what the
2026-09-05 audit found no check reading: three ``measured`` rows named receipt
logs that were never committed, annotated and scratch paths sat in
``evidence[]``, ``superseded`` rows had no successor, ``notes`` said "pending"
for rows measured days earlier, and a sentence said "best licensed" after the
register had withdrawn the licence. For every claim:

  * ``evidence[]`` entries resolve: a string is a repository path that exists
    at HEAD (an optional ``#fragment`` is allowed and not checked); a dict is
    ``{"repository": "owner/name", "path": ...}`` (a file in another
    repository, checked when ``--sibling PATH`` names a checkout whose
    ``pyproject`` slug matches) or ``{"url": "https://github.com/..."}`` (an
    issue or pull request on github.com). Free text, globs and annotations are
    findings -- the site links ``evidence[0]`` blindly.
  * ``measured`` / ``measured-private`` / ``verified`` / ``confirmed`` rows carry
    ``measured_on`` as an ISO date (``YYYY-MM-DD``).
  * ``superseded`` rows carry ``superseded_by``, and following it (through
    other superseded rows, never a cycle) reaches an ACTIVE claim; ``retired``
    rows carry a non-empty ``retired_reason``; every ``supersedes`` and
    ``superseded_by`` id exists.
  * ``quoted_in`` entries are ``<path>[#fragment][ free text]`` whose path exists.
  * no ``notes`` or ``claim`` of an ACTIVE row says "pending" or "TBD" (an
    ``open`` row may).
  * an ACTIVE row whose ``claim`` sentence asserts a licence -- the word
    "licensed", not negated by "not licensed" / "never licensed" /
    "unlicensed" / "not a licensed" -- carries ``licensed_by``, the id of the
    ACTIVE claim whose receipt holds the K8 verdict (a row whose own receipt
    holds it names itself); every ``licensed_by`` resolves to an ACTIVE row.
  * ids are unique and every status is in the file's vocabulary
    (``discovery_common.load_claims``).

    python scripts/check_claims_register.py                      # CI gate
    python scripts/check_claims_register.py --sibling ../grouped-nf4-gemm

Exit 0 when clean, 1 on findings, 2 when the check itself cannot run.
"""
from __future__ import annotations

import argparse
import os
import re
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
#: A sentence carrying one of these disclaims the licence it names.
_NEGATED = re.compile(r"\b(not licensed|never licensed|unlicensed|not a licensed)\b", re.I)
_ISSUE_URL = re.compile(r"^https://github\.com/[^/\s]+/[^/\s]+/(issues|pull)/\d+$")
_SLUG = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _split_path(entry: str) -> str:
    """``docs/METHODOLOGY.md#13 (corrected in #353)`` -> ``docs/METHODOLOGY.md``."""
    return entry.split(" ", 1)[0].split("#", 1)[0]


def check_evidence(root: Path, c: dict, sibling: Path | None, sibling_slug: str | None) -> list[str]:
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
            elif not (root / path).exists():
                out.append(f"{where}: {path!r} does not exist at HEAD")
        elif isinstance(e, dict):
            if "url" in e and set(e) == {"url"}:
                if not _ISSUE_URL.match(str(e["url"])):
                    out.append(f"{where}: url {e['url']!r} is not a github.com issue or pull request URL")
            elif "repository" in e and "path" in e and set(e) == {"repository", "path"}:
                repo, path = str(e["repository"]), str(e["path"])
                if not _SLUG.match(repo):
                    out.append(f"{where}: repository {repo!r} is not owner/name")
                if not path or path.startswith("/") or " " in path:
                    out.append(f"{where}: path {path!r} is not a bare repository-relative path")
                if sibling is not None and sibling_slug and repo.lower() == sibling_slug.lower():
                    if not (sibling / path.split("#", 1)[0]).exists():
                        out.append(f"{where}: {repo}:{path} does not exist in the sibling checkout {sibling}")
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


def check_claims(root: Path, claims: list[dict], sibling: Path | None = None, sibling_slug: str | None = None) -> list[str]:
    findings: list[str] = []
    by = {c["id"]: c for c in claims}

    def active(cid: str) -> bool:
        return cid in by and by[cid].get("status") in ACTIVE_STATUSES

    for c in claims:
        cid, st = c["id"], c.get("status")
        findings += check_evidence(root, c, sibling, sibling_slug)
        if st in DATED:
            m = c.get("measured_on")
            if not isinstance(m, str) or not _ISO.match(m):
                findings.append(f"{cid}: status {st!r} needs measured_on as YYYY-MM-DD, got {m!r}")
        if st == "superseded":
            sb = c.get("superseded_by")
            if not sb:
                findings.append(f"{cid}: superseded without superseded_by")
            else:
                problem = _follow_successors(cid, by)
                if problem:
                    findings.append(f"{cid}: {problem}")
        elif c.get("superseded_by") and c.get("superseded_by") not in by:
            findings.append(f"{cid}: superseded_by {c['superseded_by']!r} is not in the register")
        if st == "retired" and not str(c.get("retired_reason", "")).strip():
            findings.append(f"{cid}: retired without retired_reason")
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
            if _LICENSED.search(sentence) and not _NEGATED.search(sentence) and not c.get("licensed_by"):
                findings.append(f"{cid}: the sentence asserts a licence and the row has no licensed_by (the K8 verdict "
                                "claim id) -- add it, or reword the sentence")
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
    findings = check_claims(root, claims, sibling, sibling_slug)
    for f in findings:
        print(f"FAIL: {f}")
    if findings:
        print(f"FAIL: {len(findings)} finding(s) in {a.claims}")
        return 1
    n_active = sum(1 for c in claims if c.get("status") in ACTIVE_STATUSES)
    n_lic = sum(1 for c in claims if c.get("licensed_by"))
    print(f"OK: {a.claims}: {len(claims)} claims ({n_active} active) -- every evidence entry resolves, dated rows carry "
          f"an ISO measured_on, superseded/retired rows resolve, quoted_in resolves, no active row is pending, "
          f"{n_lic} licence label(s) name their verdict row"
          + (f"; cross-repository evidence checked against {sibling}" if sibling else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
