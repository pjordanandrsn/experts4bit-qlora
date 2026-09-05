#!/usr/bin/env python3
"""README = current ``main``. Two mechanical facts about README.md, checked
offline (standard library only, no network):

1. **Every headline number is the register's current value.** In every
   markdown table whose header has a ``status`` column, each row names its
   claim ids in backticks (``e4b.…``; a ``*`` makes it a glob over
   docs/claims.json). Every number in the row's result column must be a
   number of one of those claims (its ``value``, ``unit`` or ``claim``
   sentence -- never its ``notes``, which carry the unlicensed arms), read at
   the README's own precision (``155`` for 154.9, ``0.046`` for 0.04645);
   every named claim's ``value`` must appear in the row; a ``superseded`` or
   ``retired`` id fails outright (the message names what to quote instead);
   and the status cell must name the weakest status among the row's claims,
   so a ``measured-private`` receipt is never presented as ``measured``.
   Outside the tables, every backticked id must exist, and an inactive one
   may only be cited on a line that says so.

2. **The release block is generated, not typed.** The text between
   ``<!-- release-block:start -->`` and ``<!-- release-block:end -->`` must be
   exactly what this script renders from the latest ``## <version> — <date>``
   heading of CHANGELOG.md (``released_version`` in check_readme_links.py,
   cross-checked against pyproject.toml). Everything else in the README links
   ``main``; the block is the one place a release is named.

    python scripts/check_readme_claims.py                        # CI gate
    python scripts/check_readme_claims.py --write-release-block  # release recipe: regenerate the block

Exit 0 when clean, 1 on findings, 2 when the check itself cannot run.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_readme_links import ReleaseVersionError, released_version  # noqa: E402
from discovery_common import KNOWN_STATUSES, load_pyproject, read_text, self_slug, write_text  # noqa: E402

README = "README.md"
CLAIMS = "docs/claims.json"
BLOCK_START = "<!-- release-block:start -->"
BLOCK_END = "<!-- release-block:end -->"

#: Never a current number, whatever the row says.
INACTIVE = frozenset({"superseded", "retired"})
#: Weakest first: the status cell must name the weakest one present in the row.
_WEAKNESS = ("open", "projected", "measured-private", "measured", "confirmed", "verified")

_ID = re.compile(r"`(e4b\.[A-Za-z0-9._+*-]+)`")
_STATUS_WORD = re.compile(r"measured-private|measured|verified|confirmed|projected|superseded|retired|open")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


class ContractError(Exception):
    """The check cannot run (malformed input); exit 2, never a green pass."""


# ------------------------------------------------------------------ numbers --

def _normalise(text: str) -> str:
    """Typography the README uses, folded to ASCII: minus signs, en/em dashes,
    ``≈``/``~`` approximations, ``×`` to a space (``6.40×`` is a number, not a
    name); links reduced to their text so a URL's digits are never read as a
    result."""
    text = _LINK.sub(r"\1", text)
    for a, b in (("−", "-"), ("–", "-"), ("—", "-"), ("×", " "), ("≈", ""), ("~", ""), (" ", " ")):
        text = text.replace(a, b)
    return _DATE.sub(" ", text)


def _to_decimal(tok: str) -> Decimal | None:
    try:
        return Decimal(tok.replace(",", ""))
    except InvalidOperation:
        return None


def result_numbers(cell: str) -> list[Decimal]:
    """The numbers a result cell states, at the cell's own precision.

    Skipped, because they are names and not results: digits glued to a letter
    on either side (``30B``, ``fp8``, ``64k``, ``A800M``), a hyphenated name
    (``Gemma-4``, ``round-1``), a key (``B=16``, ``K=8``), an issue (``#359``)
    and ``1/2``-style fractions. A range ``1.52-1.81`` yields both ends; a
    leading ``-``/``+`` after whitespace is a sign.
    """
    text = _normalise(cell)
    out: list[Decimal] = []
    for m in _NUMBER.finditer(text):
        i, j = m.start(), m.end()
        before = text[i - 1] if i else " "
        before2 = text[i - 2] if i >= 2 else " "
        after = text[j] if j < len(text) else " "
        if after.isalnum() or after == "_" or (after == "." and j + 1 < len(text) and text[j + 1].isdigit()):
            continue
        if before.isalnum() or before in "._=#":
            continue
        if before == "/" and before2.isdigit():
            continue
        sign = ""
        if before in "+-":
            if before2.isalpha() or before2 == "_":
                continue                                     # Gemma-4, round-1: a name
            if not before2.isdigit():
                sign = before                                # -0.053, +37.1: a sign
            # else 1.52-1.81: a range, this is its unsigned upper end
        d = _to_decimal(sign + m.group(0))
        if d is not None:
            out.append(d)
    return out


def _signed_numbers(text: str) -> list[Decimal]:
    """Every number in ``text`` with its sign (a leading + or - not preceded by a digit)."""
    out: list[Decimal] = []
    for m in _NUMBER.finditer(text):
        i = m.start()
        before = text[i - 1] if i else " "
        before2 = text[i - 2] if i >= 2 else " "
        sign = before if before in "+-" and not before2.isdigit() else ""
        d = _to_decimal(sign + m.group(0))
        if d is not None:
            out.append(d)
    return out


def claim_numbers(claim: dict) -> list[Decimal]:
    """Every number the claim states in its ``value``, ``unit`` and ``claim``
    fields. ``notes`` are excluded on purpose: they carry the measured-but-
    unlicensed arms, which must never become a headline by matching here."""
    parts = []
    v = claim.get("value")
    if isinstance(v, bool):
        v = None
    if isinstance(v, (int, float)):
        parts.append(repr(v))
    elif isinstance(v, str):
        parts.append(v)
    for k in ("unit", "claim"):
        if isinstance(claim.get(k), str):
            parts.append(claim[k])
    text = _normalise(" ".join(parts))
    return _signed_numbers(text)


def value_numbers(claim: dict) -> list[Decimal]:
    """The numbers of the claim's ``value`` alone (a range string gives both ends)."""
    v = claim.get("value")
    if v is None or isinstance(v, bool):
        return []
    if isinstance(v, (int, float)):
        return [Decimal(repr(v))]
    text = _normalise(str(v))
    # the same signed extraction as the row/claim text: a register value such as
    # "-0.0528 wikitext / -0.0662 c4val1" keeps its minus signs
    return _signed_numbers(text)


def number_matches(n: Decimal, pool: list[Decimal]) -> bool:
    """``n`` equals a pool number at ``n``'s own precision: 155 for 154.9,
    0.046 for 0.04645, 1.0290 for 1.029. Rounding only -- 100 never matches 98.3."""
    exp = n.as_tuple().exponent
    q = Decimal(1).scaleb(exp) if isinstance(exp, int) else None
    for p in pool:
        if p == n:
            return True
        if q is None:
            continue
        try:
            if p.quantize(q, rounding=ROUND_HALF_UP) == n or p.quantize(q, rounding=ROUND_HALF_EVEN) == n:
                return True
        except InvalidOperation:
            continue
    return False


# ------------------------------------------------------------------- tables --

def _split_row(line: str) -> list[str]:
    cells, cur, esc = [], [], False
    for ch in line.strip():
        if esc:
            cur.append(ch)
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == "|":
            cells.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    cells.append("".join(cur))
    cells = [c.strip() for c in cells]
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def parse_tables(text: str) -> list[dict]:
    """``[{"line": int, "header": [...], "rows": [(line, [...]), ...]}]`` for
    every pipe table; ``line`` numbers are 1-based."""
    tables, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
            header = _split_row(lines[i])
            rows, j = [], i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append((j + 1, _split_row(lines[j])))
                j += 1
            tables.append({"line": i + 1, "header": header, "rows": rows})
            i = j
        else:
            i += 1
    return tables


def claim_tables(tables: list[dict]) -> list[tuple[dict, int]]:
    """The tables this check owns: those with a ``status`` header cell, with its index."""
    out = []
    for t in tables:
        low = [h.strip().lower() for h in t["header"]]
        if "status" in low:
            out.append((t, low.index("status")))
    return out


def resolve_ids(ids: list[str], claims: dict[str, dict]) -> tuple[dict[str, dict], list[str]]:
    """Literal ids and ``*`` globs -> ``{id: claim}``; the second item lists
    the ids and globs that matched nothing."""
    found, missing = {}, []
    for ref in ids:
        if "*" in ref:
            hits = [k for k in claims if fnmatch.fnmatchcase(k, ref)]
            if not hits:
                missing.append(ref)
            for k in hits:
                found[k] = claims[k]
        elif ref in claims:
            found[ref] = claims[ref]
        else:
            missing.append(ref)
    return found, missing


def _weakest(statuses: set[str]) -> str | None:
    for s in _WEAKNESS:
        if s in statuses:
            return s
    return None


def check_row(line: int, cells: list[str], status_col: int, claims: dict[str, dict]) -> list[str]:
    """Findings for one table row (empty when clean)."""
    findings = []
    if len(cells) <= status_col:
        return [f"README.md:{line}: row has {len(cells)} cell(s), no status column"]
    status_cell = cells[status_col]
    result_cols = [k for k in range(len(cells)) if k != status_col]
    if not result_cols:
        return [f"README.md:{line}: row has no result column"]
    ids = [i for k in result_cols for i in _ID.findall(cells[k])]
    if not ids:
        return [f"README.md:{line}: row names no claim id (backticked e4b.… id or glob) -- every headline number "
                "must name the claim it quotes"]
    found, missing = resolve_ids(ids, claims)
    for ref in missing:
        findings.append(f"README.md:{line}: `{ref}` is not in {CLAIMS}")
    for cid, c in found.items():
        if c.get("status") in INACTIVE:
            instead = c.get("superseded_by")
            hint = f"; quote `{instead}` instead" if instead else "; it is not a current number"
            findings.append(f"README.md:{line}: `{cid}` is {c.get('status')}{hint}")
    if not found:
        return findings
    statuses = {str(c.get("status")) for c in found.values()}
    named = set(_STATUS_WORD.findall(status_cell))
    unknown = named - KNOWN_STATUSES
    if unknown:
        findings.append(f"README.md:{line}: status cell names {sorted(unknown)}, not in the register's vocabulary")
    weakest = _weakest(statuses - INACTIVE)
    if weakest and weakest not in named:
        findings.append(f"README.md:{line}: status cell says {status_cell!r} but the row's claims include "
                        f"{weakest!r} ({', '.join(sorted(k for k, c in found.items() if c.get('status') == weakest))})"
                        " -- the weakest status in the row is the one the cell must name")
    # The result column is the last non-status column with a number in it (the
    # description column carries the ids; a two-column table has just the one).
    result_col = max(result_cols)
    pool = [n for c in found.values() for n in claim_numbers(c)]
    for n in result_numbers(cells[result_col]):
        if not number_matches(n, pool):
            findings.append(f"README.md:{line}: {n} is not a current value of any claim this row names "
                            f"({', '.join(sorted(found))}) -- drift, or a hand-typed number")
    row_numbers = result_numbers(cells[result_col])
    for cid, c in sorted(found.items()):
        for v in value_numbers(c):
            if not any(number_matches(n, [v]) for n in row_numbers):
                findings.append(f"README.md:{line}: `{cid}` has value {c.get('value')!r} and the row does not quote it")
    return findings


def check_tables(text: str, claims: dict[str, dict]) -> list[str]:
    findings, tables = [], claim_tables(parse_tables(text))
    if not tables:
        return ["README.md: no table with a `status` column -- the results table is missing"]
    for t, status_col in tables:
        for line, cells in t["rows"]:
            findings += check_row(line, cells, status_col, claims)
    return findings


def check_ids_outside_tables(text: str, claims: dict[str, dict]) -> list[str]:
    """Every backticked id anywhere in the README exists; an inactive one is
    cited only on a line that says it is (``superseded``/``retired``)."""
    findings = []
    for ln, line in enumerate(text.splitlines(), 1):
        for ref in _ID.findall(line):
            found, missing = resolve_ids([ref], claims)
            if missing:
                findings.append(f"README.md:{ln}: `{ref}` is not in {CLAIMS}")
            for cid, c in found.items():
                st = c.get("status")
                if st in INACTIVE and st not in line.lower():
                    findings.append(f"README.md:{ln}: `{cid}` is {st} and the line does not say so")
    return findings


# ------------------------------------------------------------ release block --

def render_release_block(version: str, slug: str, package: str) -> str:
    """The block, byte for byte. ``version`` is bare (``0.35.0``); the tag is ``v`` + it."""
    gh = f"https://github.com/{slug}"
    tag = f"v{version}"
    return "\n".join([
        BLOCK_START,
        "<!-- generated by `python scripts/check_readme_claims.py --write-release-block` from CHANGELOG.md's "
        "latest release heading; do not edit by hand -->",
        f"**Latest released package:** [`{package}` {version}](https://pypi.org/project/{package}/{version}/) · "
        f"**Current development status:** [`docs/STATUS.md`]({gh}/blob/main/docs/STATUS.md) on `main` "
        f"(this README describes `main`) · "
        f"**Released documentation for {version}:** [`docs/`]({gh}/tree/{tag}/docs) · "
        f"[`README.md`]({gh}/blob/{tag}/README.md) · [`CHANGELOG.md`]({gh}/blob/{tag}/CHANGELOG.md)",
        BLOCK_END,
    ])


def find_block(text: str) -> tuple[int, int] | None:
    i = text.find(BLOCK_START)
    if i < 0:
        return None
    j = text.find(BLOCK_END, i)
    if j < 0:
        return None
    return i, j + len(BLOCK_END)


def replace_block(text: str, block: str) -> str:
    span = find_block(text)
    if span is None:
        raise ContractError(f"{README} has no {BLOCK_START} … {BLOCK_END} markers")
    return text[:span[0]] + block + text[span[1]:]


def check_release_block(text: str, block: str) -> list[str]:
    span = find_block(text)
    if span is None:
        return [f"README.md: no {BLOCK_START} … {BLOCK_END} markers; the release block must exist and be generated"]
    if text[span[0]:span[1]] != block:
        return ["README.md: the release block differs from the generated one -- run "
                "`python scripts/check_readme_claims.py --write-release-block` (never edit it by hand)"]
    return []


# --------------------------------------------------------------------- main --

def load_claim_map(root: Path) -> dict[str, dict]:
    doc = json.loads(read_text(root / CLAIMS))
    claims = doc.get("claims") if isinstance(doc, dict) else doc
    out = {}
    for c in claims or []:
        if not isinstance(c, dict) or "id" not in c:
            raise ContractError(f"{CLAIMS}: a claim without an id")
        out[c["id"]] = c
    if not out:
        raise ContractError(f"{CLAIMS}: no claims")
    return out


def expected_block(root: Path) -> str:
    pyproject = load_pyproject(root)
    slug = self_slug(root, pyproject)
    if not slug:
        raise ContractError("cannot derive owner/repo from pyproject [project.urls] or the git remote")
    try:
        version = released_version(str(root))
    except (ReleaseVersionError, FileNotFoundError) as e:
        raise ContractError(str(e)) from e
    return render_release_block(version, slug, str(pyproject["name"]))


def check(root: Path, readme: str = README) -> list[str]:
    text = read_text(root / readme)
    claims = load_claim_map(root)
    findings = check_release_block(text, expected_block(root))
    findings += check_tables(text, claims)
    findings += check_ids_outside_tables(text, claims)
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--readme", default=README)
    ap.add_argument("--write-release-block", action="store_true",
                    help="regenerate the block between the markers from CHANGELOG.md, then check")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    try:
        if a.write_release_block:
            text = read_text(root / a.readme)
            new = replace_block(text, expected_block(root))
            if new != text:
                write_text(root / a.readme, new)
                print(f"wrote the release block into {a.readme}")
            else:
                print(f"{a.readme}: release block already current")
        findings = check(root, a.readme)
    except ContractError as e:
        print(f"FAIL: {e}")
        return 2
    for f in findings:
        print(f"FAIL: {f}")
    if findings:
        print(f"FAIL: {len(findings)} finding(s); README numbers come from {CLAIMS} and the release block "
              "from CHANGELOG.md")
        return 1
    n_tables = len(claim_tables(parse_tables(read_text(root / a.readme))))
    print(f"OK: {a.readme} = current main -- release block generated, {n_tables} results table(s) hold the "
          f"register's current values, no superseded/retired id, no private receipt presented as public")
    return 0


if __name__ == "__main__":
    sys.exit(main())
