#!/usr/bin/env python3
"""The prose quotes the register: the documents that state numbers and cite
claim ids agree with docs/claims.json. Checked offline, standard library only,
no network.

One file, byte-identical in experts4bit-qlora (the runtime package) and
grouped-nf4-gemm (the kernel package). It reads which package this repository
is from docs/system-manifest.json -- ``check_system_manifest.system_role``: the
``packages`` entry whose ``package`` is pyproject's [project].name, or failing
that the one whose ``repository`` is its Source URL -- and applies that role's
``PROFILES`` entry. The claim-id namespace (``e4b.`` / ``gnf4.``) is read from
the register itself (``id_pattern``), never hard-coded, so a sibling's id
quoted in prose is left alone. A repository the manifest does not name cannot
be checked (exit 2).

Both roles:

1. **Every headline number is the register's current value.** In each results
   document (``table_docs``), every markdown table whose header has a status
   column is a results table, and there is at least one. Each row names its
   claim ids in backticks (a ``*`` makes an id a glob over docs/claims.json);
   every number in the row's result cell must be a number of one of those
   claims (its ``value``, ``unit`` or ``claim`` sentence -- never its
   ``notes``, which carry the unlicensed arms), read at the document's own
   precision (``155`` for 154.9, ``1.70`` for 1.7, ``0.046`` for 0.04645);
   every named claim's ``value`` must appear in the result cell; a
   ``superseded`` or ``retired`` id fails outright (the message names what to
   quote instead); the status cell must name the weakest status among the
   row's claims, so a ``measured-private`` receipt is never presented as
   ``measured``; and a row needs a status cell and a result cell that is not
   the status cell.

2. **Every cited id exists, and an inactive one says so.** In README.md, the
   results documents, the position prose that quotes ids by name
   (``POSITION_DOCS`` -- docs/STATUS.md, docs/SOLUTIONS.md,
   docs/SERVING-THROUGHPUT.md, docs/SERVING-PARITY.md, docs/METHODOLOGY.md,
   docs/ARCHITECTURE_SUPPORT.md, docs/CHOOSING.md, whichever exist) and every
   docs/solutions/*.md page: every cited id -- backticked, ``<code>``-wrapped,
   or the text of a link -- is in the register, and a superseded or retired id
   appears only on a line whose PROSE names that status (``superseded`` /
   ``retired``) or says ``historical``. The prose is the line after links are
   reduced to their text, HTML comments are removed and every claim id (in any
   form, or bare) is stripped (``prose_of``), so a link URL, a comment or the
   id's own name (``e4b.retired.x``) never vouches for it.

Runtime role only (each is a place the two copies differed; ``PROFILES`` says,
for each, what the other role's rule would break here):

  * README.md carries the generated release block: the text between
    ``<!-- release-block:start -->`` and ``<!-- release-block:end -->`` is
    exactly what this script renders from the latest ``## <version> — <date>``
    heading of CHANGELOG.md (``released_version`` in check_readme_links.py,
    cross-checked against pyproject.toml). ``--write-release-block`` rewrites
    it (the release recipe); in the kernel role that flag is exit 2;
  * results tables are the ones headed ``status``, and a row's result cell is
    its last non-status cell;
  * a position document that is anchored (a sibling ``.ots`` file or the
    ``<!-- ots-attestation-footer -->`` marker) is skipped: a finding there
    could only be fixed by editing an anchored file;
  * ``+-`` in a number is left to the tokeniser (it reads ``+-4.2`` as -4.2),
    as the runtime copy did.

Kernel role only:

  * docs/STATUS.md is a results document as well as README.md, and both must
    exist;
  * results tables are headed ``status`` or ``tier``, and a row's result cell
    is the column headed ``result`` / ``measured`` / ``value``, else the last
    column that is neither the status nor the claim-id column;
  * no document is skipped for being anchored;
  * a status word inside backticks (``is `retired```) does not count as the
    line saying so: the line must say it in both readings, ``prose_of`` and
    ``code_free_prose`` (every code span removed first);
  * ``+-`` is a tolerance, folded like ``±`` (``+-4.2%`` is the unsigned 4.2).

    python scripts/check_readme_claims.py                        # CI gate (both repositories)
    python scripts/check_readme_claims.py --write-release-block  # runtime release recipe: regenerate the block

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
from check_system_manifest import MANIFEST, system_role  # noqa: E402
from discovery_common import (  # noqa: E402
    KNOWN_STATUSES, ContractError, load_claims, load_pyproject, read_text, self_slug, write_text,
)

README = "README.md"
STATUS = "docs/STATUS.md"
CLAIMS = "docs/claims.json"
BLOCK_START = "<!-- release-block:start -->"
BLOCK_END = "<!-- release-block:end -->"

#: Never a current number, whatever the row says.
INACTIVE = frozenset({"superseded", "retired"})
#: Besides the status itself, the word that lets a line cite an inactive id: a
#: dated record that names the old row as history.
HISTORICAL = "historical"
#: Weakest first: the status cell must name the weakest one present in the row.
_WEAKNESS = ("open", "projected", "measured-private", "measured", "confirmed", "verified")
#: Header cells that mark the status column, and the result column (``result_rule="headed"``).
STATUS_HEADERS = ("status", "tier")
RESULT_HEADERS = ("result", "measured", "value")

#: Position prose that quotes claim ids by name: the id rules, not the table rules.
POSITION_DOCS = ("docs/STATUS.md", "docs/SOLUTIONS.md", "docs/SERVING-THROUGHPUT.md", "docs/SERVING-PARITY.md",
                 "docs/METHODOLOGY.md", "docs/ARCHITECTURE_SUPPORT.md", "docs/CHOOSING.md")
SOLUTIONS_GLOB = "docs/solutions/*.md"
ANCHOR_MARKER = "<!-- ots-attestation-footer -->"

#: What each role checks beyond the rules both share. Every key is a place the
#: two copies this file replaces differed. For every key but one, the other
#: role's value either fails this repository's current ``main`` or passes an
#: input this repository's copy failed, so the difference is kept, by role,
#: instead of picking one; the exception, the runtime's ``skip_anchored``, is
#: kept by choice (below).
PROFILES: dict[str, dict] = {
    "runtime": {
        "role": "runtime",
        # README only: the runtime's docs/STATUS.md has no table with a status
        # column (its tables are the training-support matrix and a noise-floor
        # reading), so the kernel's "STATUS is a results document" fails there.
        "table_docs": (README,),
        # `tier` stays the kernel's header: recognising it here would accept a
        # README whose results table is headed `tier`, which the runtime copy
        # fails as having no results table (and would move the status column of
        # a table carrying both words). The kernel README is headed `tier`.
        "status_headers": ("status",),
        # The runtime copy reads each ROW's last non-status cell as the result;
        # the kernel's header rule would skip a trailing cell the header does not
        # name, passing a number there that the runtime copy fails. The kernel
        # README ends in its claim-ID column, which this rule would read.
        "result_rule": "last",
        # The kernel README has no release block (its tag pins are
        # check_dependency_floor.py's), so this is the runtime's alone.
        "release_block": True,
        # The runtime copy exempts anchored position documents, because a finding
        # there could only be fixed by editing an anchored file. Kept by choice:
        # no runtime position document is anchored today, so the runtime's main
        # would pass without it. The kernel copy never had the exemption, and
        # adding it there would stop checking a document that becomes anchored.
        "skip_anchored": True,
        # Two lines of the runtime's docs/STATUS.md carry the status word only in
        # backticks (`e4b.retired.13.47x-training-speedup` is `retired`;
        # `e4b.train.energy-honest` is `superseded`): the kernel's reading fails
        # them. Unquoting the two words there lets this key go.
        "code_spans_are_prose": True,
        # The runtime copy reads `+-4.2` as -4.2. Folding it would let a bare 4.2
        # match a `+-4.2` claim, which that copy fails. Not folding it in the
        # kernel would fail the kernel README, whose claims write `+-`.
        "fold_tolerance": False,
    },
    "kernels": {
        "role": "kernels",
        "table_docs": (README, STATUS),
        "status_headers": STATUS_HEADERS,
        "result_rule": "headed",
        "release_block": False,
        "skip_anchored": False,
        "code_spans_are_prose": False,
        "fold_tolerance": True,
    },
}

_STATUS_WORD = re.compile(r"measured-private|measured|verified|confirmed|projected|superseded|retired|open")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
#: A closed comment, one opened on the line, or the tail of one opened on an earlier line.
_HTML_COMMENT = re.compile(r"<!--.*?-->|<!--.*$|^(?:(?!<!--).)*?-->")
_CODE_SPAN = re.compile(r"`[^`]*`")
_ID_TAIL = r"\.[A-Za-z0-9._+*-]+"
#: The id forms when no register is at hand: any lower-case namespace. The checks
#: themselves always pass the register's own pattern (``id_pattern``).
_ANY_ID = re.compile(rf"`((?:[a-z][a-z0-9]*){_ID_TAIL})`")


# --------------------------------------------------------------------- role --

def repo_role(root) -> str | None:
    """``"runtime"`` / ``"kernels"``: this repository's ``packages`` entry in
    docs/system-manifest.json, read by ``check_system_manifest.system_role``;
    ``None`` when the manifest or pyproject.toml is missing or names neither."""
    try:
        manifest = json.loads(read_text(Path(root) / MANIFEST))
        return system_role(manifest, load_pyproject(root))
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return None


def profile_for(root) -> dict:
    """``PROFILES`` for this repository's role; a ``ContractError`` (exit 2)
    when the role cannot be read -- a check that guessed would run the wrong rules."""
    role = repo_role(root)
    if role is None:
        raise ContractError(f"cannot tell which package of the system {Path(root)} is: {MANIFEST} names no "
                            "packages entry for pyproject's [project].name or Source URL")
    return PROFILES[role]


# ------------------------------------------------------------------ numbers --

#: Typography folded to ASCII, in this order (``+-`` is folded first, when it is).
#: ``\u202f`` is the narrow no-break space, written as an escape so it cannot turn
#: into a plain space unseen (the kernel copy's list carried a space-to-space no-op).
_TYPOGRAPHY = (("±", " "), ("−", "-"), ("–", "-"), ("—", "-"), ("×", " "), ("≈", ""), ("~", ""), ("→", " "),
               ("\u202f", " "))


def _normalise(text: str, fold_tolerance: bool = True) -> str:
    """Typography the documents use, folded to ASCII: minus signs, en/em
    dashes, ``≈``/``~`` approximations, ``±`` (and, with ``fold_tolerance``,
    ``+-``) tolerances (``±4.2%`` is the unsigned 4.2 on both sides), ``×`` and
    ``→`` to a space (``6.40×`` is a number, not a name), the narrow no-break
    space to a space; links reduced to their text so a URL's digits are never
    read as a result; dates blanked."""
    text = _LINK.sub(r"\1", text)
    if fold_tolerance:
        text = text.replace("+-", " ")
    for a, b in _TYPOGRAPHY:
        text = text.replace(a, b)
    return _DATE.sub(" ", text)


def _to_decimal(tok: str) -> Decimal | None:
    try:
        return Decimal(tok.replace(",", ""))
    except InvalidOperation:
        return None


def result_numbers(cell: str, fold_tolerance: bool = True) -> list[Decimal]:
    """The numbers a result cell states, at the cell's own precision.

    Skipped, because they are names and not results: digits glued to a letter
    on either side (``30B``, ``fp8``, ``64k``, ``A800M``, ``sm_86``), a
    hyphenated name (``Gemma-4``, ``round-1``), a key (``B=16``, ``E=256``), an
    issue (``#359``) and the second half of a ``144/144``-style fraction. A
    range ``1.52-1.81`` yields both ends; a leading ``-``/``+`` after whitespace
    is a sign.
    """
    text = _normalise(cell, fold_tolerance)
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


def claim_numbers(claim: dict, fold_tolerance: bool = True) -> list[Decimal]:
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
    return _signed_numbers(_normalise(" ".join(parts), fold_tolerance))


def value_numbers(claim: dict, fold_tolerance: bool = True) -> list[Decimal]:
    """The numbers of the claim's ``value`` alone (a range string gives both ends)."""
    v = claim.get("value")
    if v is None or isinstance(v, bool):
        return []
    if isinstance(v, (int, float)):
        return [Decimal(repr(v))]
    # the same signed extraction as the row/claim text: a register value such as
    # "-0.0528 wikitext / -0.0662 c4val1" keeps its minus signs
    return _signed_numbers(_normalise(str(v), fold_tolerance))


def number_matches(n: Decimal, pool: list[Decimal]) -> bool:
    """``n`` equals a pool number at ``n``'s own precision: 155 for 154.9,
    1.70 for 1.7, 0.046 for 0.04645. Rounding only -- 100 never matches 98.3."""
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


# ---------------------------------------------------------------------- ids --

def id_pattern(claims: dict[str, dict]) -> re.Pattern:
    """Backticked ids of THIS register: the prefixes before the first ``.`` of
    every claim id (``e4b`` / ``gnf4``), so a sibling id quoted in prose is left alone."""
    prefixes = sorted({k.split(".", 1)[0] for k in claims if "." in k})
    if not prefixes:
        raise ContractError(f"{CLAIMS}: no dotted claim ids to derive a prefix from")
    alt = "|".join(re.escape(p) for p in prefixes)
    return re.compile(rf"`((?:{alt}){_ID_TAIL})`")


_FORMS: dict[str, tuple[re.Pattern, re.Pattern, re.Pattern, re.Pattern]] = {}


def _id_forms(pat: re.Pattern | None) -> tuple[re.Pattern, re.Pattern, re.Pattern, re.Pattern]:
    """``(backticked, <code>-wrapped, link text, bare)`` patterns for the ids
    ``pat`` (an ``id_pattern``) matches; ``None`` means any namespace."""
    pat = pat or _ANY_ID
    forms = _FORMS.get(pat.pattern)
    if forms is None:
        m = re.fullmatch(r"`\((.+)\)`", pat.pattern)
        if not m:
            raise ContractError(f"not a backticked claim-id pattern: {pat.pattern!r}")
        inner = m.group(1)
        forms = (pat, re.compile(rf"<code>\s*({inner})\s*</code>"), re.compile(rf"\[`?({inner})`?\]\("),
                 re.compile(rf"\b{inner}"))
        _FORMS[pat.pattern] = forms
    return forms


def _namespaces(pat: re.Pattern) -> str:
    """``e4b.…`` / ``gnf4.…`` for messages."""
    m = re.match(r"`\(\(\?:([^)]*)\)", pat.pattern)
    names = m.group(1).replace("\\", "").split("|") if m else []
    return "/".join(f"{n}.…" for n in names) or "claim"


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


def cited_ids(line: str, pat: re.Pattern | None = None) -> list[str]:
    """The claim ids (or globs) a line cites -- backticked, ``<code>``-wrapped,
    or the text of a link -- in order of appearance, each once."""
    bt, code, link, _bare = _id_forms(pat)
    hits = sorted((m.start(), m.group(1)) for rx in (bt, code, link) for m in rx.finditer(line))
    out: list[str] = []
    for _, ref in hits:
        if ref not in out:
            out.append(ref)
    return out


def prose_of(line: str, pat: re.Pattern | None = None) -> str:
    """The line's own words, lower-cased, for the superseded / retired /
    historical test: links reduced to their text (a URL or fragment is not
    prose), HTML comments removed, and every claim id -- ``<code>``-wrapped,
    backticked, or a bare ``e4b.…`` token -- stripped, so an id such as
    ``e4b.retired.13.47x-training-speedup`` cannot vouch for itself. Other code
    spans stay: ``is `retired``` says so here (``code_free_prose`` is the
    reading where it does not)."""
    bt, code, _link, bare = _id_forms(pat)
    text = _LINK.sub(r"\1", line)
    text = _HTML_COMMENT.sub(" ", text)
    text = code.sub(" ", text)
    text = bt.sub(" ", text)
    return bare.sub(" ", text).lower()


def code_free_prose(line: str, pat: re.Pattern | None = None) -> str:
    """The stricter reading: every code span removed from the line FIRST (a
    status word in backticks is not prose), then links reduced to their text
    with a space either side (so removing a URL never joins two words), HTML
    comments, ``<code>`` ids and bare ids removed. Every step only removes text,
    so a word found here is a word of the line with its code spans removed."""
    _bt, code, _link, bare = _id_forms(pat)
    text = _CODE_SPAN.sub(" ", line)
    text = _LINK.sub(r" \1 ", text)
    text = _HTML_COMMENT.sub(" ", text)
    text = code.sub(" ", text)
    return bare.sub(" ", text).lower()


def says_so(line: str, status: str, pat: re.Pattern | None = None, *, code_spans_are_prose: bool = False) -> bool:
    """Does the line's prose name ``status`` (or say ``historical``)? In every
    reading: ``prose_of``, and ``code_free_prose`` unless code spans count."""
    readings = [prose_of(line, pat)]
    if not code_spans_are_prose:
        readings.append(code_free_prose(line, pat))
    return all(status in p or HISTORICAL in p for p in readings)


def check_doc_ids(rel: str, text: str, claims: dict[str, dict], pat: re.Pattern | None = None, *,
                  code_spans_are_prose: bool = False) -> list[str]:
    """Every cited id in the document (``cited_ids``) exists; an inactive one
    is cited only on a line that says so (``says_so``: ``superseded`` /
    ``retired`` matching its status, or ``historical``)."""
    pat = pat or id_pattern(claims)
    findings = []
    for ln, line in enumerate(text.splitlines(), 1):
        refs = cited_ids(line, pat)
        if not refs:
            continue
        for ref in refs:
            found, missing = resolve_ids([ref], claims)
            if missing:
                findings.append(f"{rel}:{ln}: `{ref}` is not in {CLAIMS}")
            for cid, c in found.items():
                st = c.get("status")
                if st in INACTIVE and not says_so(line, st, pat, code_spans_are_prose=code_spans_are_prose):
                    findings.append(f"{rel}:{ln}: `{cid}` is {st} and the line does not say so")
    return findings


#: The kernel copy's name for the same rule.
check_ids = check_doc_ids


def check_ids_outside_tables(text: str, claims: dict[str, dict], pat: re.Pattern | None = None, *,
                             readme: str = README, code_spans_are_prose: bool = False) -> list[str]:
    """``check_doc_ids`` on the README: every cited id exists; an inactive one
    is cited only on a line whose prose says it is."""
    return check_doc_ids(readme, text, claims, pat, code_spans_are_prose=code_spans_are_prose)


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


def table_columns(header: list[str], status_headers: tuple[str, ...] = STATUS_HEADERS,
                  result_rule: str = "headed") -> tuple[int, int | None, int | None] | None:
    """``(status_col, id_col, result_col)`` for a results table, or None when no
    header cell is one of ``status_headers``. The id column is the one whose
    header names ``claim``. ``result_rule="headed"`` (the kernel's): the result
    column is the one headed ``result`` / ``measured`` / ``value``, else the
    last column that is neither status nor id. ``result_rule="last"`` (the
    runtime's): ``None`` -- each row's last non-status cell (``check_row``)."""
    low = [h.strip().lower() for h in header]
    status = next((k for k, h in enumerate(low) if h in status_headers), None)
    if status is None:
        return None
    id_col = next((k for k, h in enumerate(low) if "claim" in h), None)
    if result_rule == "last":
        return status, id_col, None
    result = next((k for k, h in enumerate(low) if h in RESULT_HEADERS), None)
    if result is None:
        rest = [k for k in range(len(low)) if k != status and k != id_col]
        result = max(rest) if rest else status
    return status, id_col, result


def claim_tables(tables: list[dict], status_headers: tuple[str, ...] = STATUS_HEADERS,
                 result_rule: str = "headed") -> list[tuple[dict, tuple[int, int | None, int | None]]]:
    """The tables this check owns, with their column roles (``table_columns``)."""
    out = []
    for t in tables:
        cols = table_columns(t["header"], status_headers, result_rule)
        if cols is not None:
            out.append((t, cols))
    return out


def _weakest(statuses: set[str]) -> str | None:
    for s in _WEAKNESS:
        if s in statuses:
            return s
    return None


def check_row(doc: str, line: int, cells: list[str], cols: tuple[int, int | None, int | None],
              claims: dict[str, dict], pat: re.Pattern, *, fold_tolerance: bool = True) -> list[str]:
    """Findings for one table row (empty when clean)."""
    status_col, _id_col, result_col = cols
    if len(cells) <= status_col:
        return [f"{doc}:{line}: row has {len(cells)} cell(s), no status column"]
    if result_col is None:
        # the runtime's reading: the row's last non-status cell (the description
        # column carries the ids; a two-column table has just the one)
        others = [k for k in range(len(cells)) if k != status_col]
        result_col = max(others) if others else None
    elif len(cells) <= result_col:
        return [f"{doc}:{line}: row has {len(cells)} cell(s); the header promises more"]
    findings = []
    if result_col == status_col:
        result_col = None
    if result_col is None:
        # a finding, not the end of the row: its ids and status are still checked
        findings.append(f"{doc}:{line}: row has no result column")
    status_cell = cells[status_col]
    ids = [i for k, c in enumerate(cells) if k != status_col for i in pat.findall(c)]
    if not ids:
        return findings + [f"{doc}:{line}: row names no claim id (backticked {_namespaces(pat)} id or glob) -- every "
                           "headline number must name the claim it quotes"]
    found, missing = resolve_ids(ids, claims)
    for ref in missing:
        findings.append(f"{doc}:{line}: `{ref}` is not in {CLAIMS}")
    for cid, c in found.items():
        if c.get("status") in INACTIVE:
            instead = c.get("superseded_by")
            hint = f"; quote `{instead}` instead" if instead else "; it is not a current number"
            findings.append(f"{doc}:{line}: `{cid}` is {c.get('status')}{hint}")
    if not found:
        return findings
    statuses = {str(c.get("status")) for c in found.values()}
    named = set(_STATUS_WORD.findall(status_cell))
    unknown = named - KNOWN_STATUSES
    if unknown:
        findings.append(f"{doc}:{line}: status cell names {sorted(unknown)}, not in the register's vocabulary")
    weakest = _weakest(statuses - INACTIVE)
    if weakest and weakest not in named:
        findings.append(f"{doc}:{line}: status cell says {status_cell!r} but the row's claims include "
                        f"{weakest!r} ({', '.join(sorted(k for k, c in found.items() if c.get('status') == weakest))})"
                        " -- the weakest status in the row is the one the cell must name")
    if result_col is None:
        return findings
    pool = [n for c in found.values() for n in claim_numbers(c, fold_tolerance)]
    row_numbers = result_numbers(cells[result_col], fold_tolerance)
    for n in row_numbers:
        if not number_matches(n, pool):
            findings.append(f"{doc}:{line}: {n} is not a current value of any claim this row names "
                            f"({', '.join(sorted(found))}) -- drift, or a hand-typed number")
    for cid, c in sorted(found.items()):
        for v in value_numbers(c, fold_tolerance):
            if not any(number_matches(n, [v]) for n in row_numbers):
                findings.append(f"{doc}:{line}: `{cid}` has value {c.get('value')!r} and the row does not quote it")
    return findings


def check_tables(doc: str, text: str, claims: dict[str, dict], pat: re.Pattern, *,
                 status_headers: tuple[str, ...] = STATUS_HEADERS, result_rule: str = "headed",
                 fold_tolerance: bool = True) -> list[str]:
    findings, tables = [], claim_tables(parse_tables(text), status_headers, result_rule)
    if not tables:
        heads = "/".join(f"`{h}`" for h in status_headers)
        return [f"{doc}: no table with a {heads} column -- the results table is missing"]
    for t, cols in tables:
        for line, cells in t["rows"]:
            findings += check_row(doc, line, cells, cols, claims, pat, fold_tolerance=fold_tolerance)
    return findings


# ------------------------------------------------------ position documents --

def position_documents(root: Path, skip_anchored: bool = True) -> list[str]:
    """The id documents beyond the results documents, relative to ``root``:
    ``POSITION_DOCS`` that exist plus every solution page. With
    ``skip_anchored`` (the runtime's exemption; the kernel role passes False),
    an anchored document (a sibling ``.ots`` or the footer marker) is left out,
    because a finding there could only be fixed by editing an anchored file."""
    rels = [r for r in POSITION_DOCS if (root / r).is_file()]
    rels += [p.relative_to(root).as_posix() for p in sorted(root.glob(SOLUTIONS_GLOB))]
    out = []
    for rel in rels:
        p = root / rel
        if skip_anchored and (p.with_name(p.name + ".ots").is_file() or ANCHOR_MARKER in read_text(p)):
            continue
        out.append(rel)
    return out


def check_position_docs(root: Path, claims: dict[str, dict], pat: re.Pattern | None = None,
                        profile: dict | None = None) -> tuple[list[str], int]:
    """``(findings, n_docs)`` over every position document, with the id rules of
    ``profile`` (default: this repository's, ``profile_for``)."""
    prof = profile or profile_for(root)
    pat = pat or id_pattern(claims)
    findings, n = [], 0
    for rel in position_documents(root, prof["skip_anchored"]):
        n += 1
        findings += check_doc_ids(rel, read_text(root / rel), claims, pat,
                                  code_spans_are_prose=prof["code_spans_are_prose"])
    return findings, n


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


def replace_block(text: str, block: str, readme: str = README) -> str:
    span = find_block(text)
    if span is None:
        raise ContractError(f"{readme} has no {BLOCK_START} … {BLOCK_END} markers")
    return text[:span[0]] + block + text[span[1]:]


def check_release_block(text: str, block: str, readme: str = README) -> list[str]:
    span = find_block(text)
    if span is None:
        return [f"{readme}: no {BLOCK_START} … {BLOCK_END} markers; the release block must exist and be generated"]
    if text[span[0]:span[1]] != block:
        return [f"{readme}: the release block differs from the generated one -- run "
                "`python scripts/check_readme_claims.py --write-release-block` (never edit it by hand)"]
    return []


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


# --------------------------------------------------------------------- main --

def load_claim_map(root: Path) -> dict[str, dict]:
    """``{id: claim}`` through ``discovery_common.load_claims`` (unique ids, every
    status in the file's vocabulary); any way the file cannot be read is a
    ``ContractError`` -- exit 2, never a green pass and never a traceback."""
    try:
        claims, _vocab = load_claims(root, CLAIMS)
    except (OSError, ValueError) as e:
        raise ContractError(f"{CLAIMS}: {e}") from e
    if not claims:
        raise ContractError(f"{CLAIMS}: no claims")
    return {c["id"]: c for c in claims}


def _table_docs(profile: dict, readme: str) -> list[str]:
    return [readme if d == README else d for d in profile["table_docs"]]


def documents(root: Path, readme: str = README, profile: dict | None = None) -> list[str]:
    """Every document checked, in order: the results documents (which must
    exist), then the position documents not already among them."""
    prof = profile or profile_for(root)
    out = _table_docs(prof, readme)
    out += [r for r in position_documents(root, prof["skip_anchored"]) if r not in out]
    return out


def check(root: Path, readme: str = README, profile: dict | None = None) -> tuple[list[str], int, int]:
    """``(findings, n_docs, n_tables)`` for this repository, with its role's
    ``PROFILES`` entry unless ``profile`` is given."""
    prof = profile or profile_for(root)
    claims = load_claim_map(root)
    pat = id_pattern(claims)
    block = expected_block(root) if prof["release_block"] else None
    table_docs = _table_docs(prof, readme)
    findings: list[str] = []
    n_tables = 0
    docs = documents(root, readme, prof)
    for rel in docs:
        p = root / rel
        if not p.is_file():
            findings.append(f"{rel}: missing")
            continue
        text = read_text(p)
        if block is not None and rel == readme:
            findings += check_release_block(text, block, readme)
        if rel in table_docs:
            findings += check_tables(rel, text, claims, pat, status_headers=tuple(prof["status_headers"]),
                                     result_rule=prof["result_rule"], fold_tolerance=prof["fold_tolerance"])
            n_tables += len(claim_tables(parse_tables(text), tuple(prof["status_headers"]), prof["result_rule"]))
        findings += check_doc_ids(rel, text, claims, pat, code_spans_are_prose=prof["code_spans_are_prose"])
    return findings, len(docs), n_tables


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="this repository's root")
    ap.add_argument("--readme", default=README, help=f"the README, relative to --root (default {README})")
    ap.add_argument("--write-release-block", action="store_true",
                    help="runtime role: regenerate the block between the markers from CHANGELOG.md, then check")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    try:
        prof = profile_for(root)
        if a.write_release_block:
            if not prof["release_block"]:
                raise ContractError(f"--write-release-block: this repository is packages.{prof['role']}, which has "
                                    "no generated release block")
            text = read_text(root / a.readme)
            new = replace_block(text, expected_block(root), a.readme)
            if new != text:
                write_text(root / a.readme, new)
                print(f"wrote the release block into {a.readme}")
            else:
                print(f"{a.readme}: release block already current")
        findings, n_docs, n_tables = check(root, a.readme, prof)
    except (ContractError, OSError, ValueError) as e:
        print(f"FAIL: {e}")
        return 2
    for f in findings:
        print(f"FAIL: {f}")
    if findings:
        print(f"FAIL: {len(findings)} finding(s); numbers and ids come from {CLAIMS}"
              + (" and the release block from CHANGELOG.md" if prof["release_block"] else ""))
        return 1
    print(f"OK: {n_docs} document(s) quote {CLAIMS} (packages.{prof['role']}) -- "
          + ("release block generated from CHANGELOG.md; " if prof["release_block"] else "")
          + f"{n_tables} results table(s) in {', '.join(_table_docs(prof, a.readme))} hold the register's current "
          "values, no superseded/retired id is quoted as current, no private receipt presented as public; every "
          "cited id exists, an inactive one only where the line says so")
    return 0


if __name__ == "__main__":
    sys.exit(main())
