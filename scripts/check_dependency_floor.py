#!/usr/bin/env python3
"""Dependency-version statements in the CURRENT documents, checked against
their canonical sources. Standard library only; no network.

One file, byte-identical in experts4bit-qlora (the runtime package) and
grouped-nf4-gemm (the kernel package). It reads which package this repository
is from docs/system-manifest.json -- ``check_system_manifest.system_role``: the
``packages`` entry whose ``package`` is pyproject's [project].name, or failing
that the one whose ``repository`` is its Source URL -- and applies that role's
``PROFILES`` entry. A repository the manifest does not name cannot be checked
(exit 2).

Both roles:

1. **The kernel floor.** Every statement of the grouped-nf4-gemm floor equals
   the current floor (``PROFILES`` ``floor_source``: the runtime's ``fast``
   extra, the kernel's current compatibility record). A statement is
   ``grouped-nf4-gemm>=X`` / ``>= X`` / ``≥ X``, ``requires grouped-nf4-gemm
   X``, a bare ``>= X`` on a line that names the ``fast`` extra (or, in the
   runtime role, grouped-nf4-gemm or gnf4: ``PROFILES`` ``bare_floor_mentions``;
   ``statements``), or a bare ``>=X.Y.Z`` on a line about a floor that pins
   neither package (``floor_statements``). A ``>=`` that
   belongs to another dependency of the package (``torch>=2.2``,
   ``bitsandbytes >= 0.50.0``) is not one. Documents: README.md, AGENTS.md,
   llms.txt, docs/SOLUTIONS.md, docs/INDEX.md, docs/capabilities.json,
   .github/workflows/ci.yml (the ``--requires`` assertion), docs/solutions/*.md
   and every document docs/INDEX.md lists under its "Current" heading, whichever
   exist; the blockquoted examples of docs/RELEASE_NOTES_GUIDE.md are not
   statements. A statement on a historical line (``PROFILES``
   ``floor_exemption``) is excused and printed, so a reviewer sees what was
   excused.

2. **This package's own version** (pyproject.toml's): a self-repository link
   pinned to a release tag (``github.com/<owner>/<repo>/blob/vX.Y.Z/``),
   ``version X.Y.Z`` and ``latest ... X.Y.Z`` (not beside another dependency's
   name), ``<this package> X.Y.Z``, and ``<this package>==X`` / ``~=X``.

3. **The other hand-copied floors**, validated rather than removed: torch and
   triton floors against pyproject dependencies (at the document's own
   precision: "2.8" matches ``>=2.8.0.dev0``); "Python 3.11 ... CI" against
   the ``python-version`` of .github/workflows/ci.yml; "pyproject says >=3.9"
   against requires-python; the README ``## License`` section names
   [project].license.

   Documents for 2 and 3: README.md, AGENTS.md, llms.txt, docs/SOLUTIONS.md,
   docs/INDEX.md and docs/capabilities.json (each must exist), the
   docs/STATUS.md header (up to its first ``---`` rule; it must exist) and
   docs/solutions/*.md. A line carrying ``was``, ``were``, ``raised from``,
   ``raised to``, ``until`` or ``no longer`` (``HISTORICAL``) is a record of a
   past state and is skipped.

Runtime role only (each is a place the two copies differed; ``PROFILES`` says,
for each, what the other role's rule would break here):

  * the floor is the single ``>=`` of the grouped-nf4-gemm requirement in
    ``[project.optional-dependencies] fast`` (exit 2 when there is none), and
    docs/system-manifest.json's compatibility record whose ``consumer_versions``
    contains this version must carry it (older records are history by
    construction);
  * rule 1 excuses a line carrying any of ``HISTORICAL_MARKERS`` as a whole
    word, or a ``vX.Y.Z`` tag pin (``historical_marker``);
  * rule 1 skips an anchored document (a sibling ``.ots`` file or the
    ``<!-- ots-attestation-footer -->`` marker);
  * a bare ``>= X`` counts on a line that names grouped-nf4-gemm or gnf4 as
    well as on one that names the ``fast`` extra.

Kernel role only:

  * the floor is the manifest's current kernel record (the highest consumer
    range, ``check_system_manifest.current_kernel_record``): this package has
    no ``fast`` extra on itself; a floor statement with no such record FAILS;
  * ``experts4bit-qlora>=Y`` / ``==Y`` states the lower bound of that record's
    consumer range;
  * rule 1 excuses only the ``HISTORICAL`` words of rules 2 and 3;
  * no document is skipped for being anchored.

Never scanned: CHANGELOG.md, receipts, and pyproject.toml's own comment ladder
(the source is not a document).

    python scripts/check_dependency_floor.py      # CI gate (both repositories)

Prints one line per floor statement (OK / historical / FAIL) and one FAIL per
stale version statement as file:line -> statement -> canonical value. Exit 0
when clean, 1 on any FAIL, 2 when the check itself cannot run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_system_manifest import (  # noqa: E402
    EXTRA, KERNEL_PACKAGE, MANIFEST, current_kernel_record, current_record, fast_requirement, floor_of,
    floor_version, parse_version, range_lower_bound, release_tuple, system_role,
)
from discovery_common import (  # noqa: E402
    ContractError, load_pyproject, pep503_name, read_text, requirement_name, self_slug,
)

#: The consumer package, for the kernel role's ``experts4bit-qlora>=Y`` pins.
CONSUMER_PACKAGE = "experts4bit-qlora"

#: Rules 2 and 3: each must exist (a missing one is a finding).
CURRENT_DOCS = ["README.md", "AGENTS.md", "llms.txt", "docs/SOLUTIONS.md", "docs/INDEX.md", "docs/capabilities.json"]
#: Rules 2 and 3 read only its header, up to the first ``---`` rule.
STATUS_HEADER = "docs/STATUS.md"
SOLUTIONS_GLOB = "docs/solutions/*.md"
#: The runtime copy's name for the same glob.
SOLUTION_GLOB = SOLUTIONS_GLOB
CI_WORKFLOW = ".github/workflows/ci.yml"
#: Rule 1: always scanned when present, in addition to the "Current" list in
#: docs/INDEX.md. docs/STATUS.md is read whole here (the kernel copy read its
#: header for this rule, and both INDEX.md files list it as current).
FIXED_DOCUMENTS = tuple(CURRENT_DOCS) + (CI_WORKFLOW, STATUS_HEADER)
INDEX = "docs/INDEX.md"
RELEASE_NOTES_GUIDE = "docs/RELEASE_NOTES_GUIDE.md"
ANCHOR_MARKER = "<!-- ots-attestation-footer -->"

#: Rules 2 and 3 (and rule 1 in the kernel role): a line carrying one of these
#: words records a past state.
HISTORICAL = re.compile(r"\b(was|were|raised (?:from|to)|until|no longer)\b", re.I)
#: Rule 1 in the runtime role: a line carrying one of these (whole words,
#: case-insensitive) is describing history, not stating the current floor.
#: Extend deliberately: every marker is a way for a stale line to be excused,
#: and excused lines are printed.
HISTORICAL_MARKERS = ("raised from", "raised to", "landed in", "first shipped", "first released",
                      "below the current", "no longer", "used to", "at the time", "until", "older",
                      "was", "were", "history", "historical")

#: What each role checks beyond the rules both share. Every key is a place the
#: two copies this file replaces differed. For each, the comment says what the
#: other role's value would do here: fail this repository's current ``main``,
#: pass an input this repository's copy failed, or read a statement this
#: repository's documents do not make. So the difference is kept, by role,
#: instead of picking one; the runtime's ``skip_anchored`` is kept by choice.
PROFILES: dict[str, dict] = {
    "runtime": {
        "role": "runtime",
        # Where rule 1's floor comes from. The runtime's pyproject sets it (the
        # `fast` extra), and the manifest record containing this version is
        # then held to it. The kernel's source here (the highest record, no
        # pyproject floor, no record check) would pass a document that agrees
        # with a stale manifest while pyproject moved on.
        "floor_source": "pyproject",
        # Which lines rule 1 excuses. Two lines of the runtime's
        # docs/capabilities.json state an older floor as "below the current
        # [fast] floor" (grouped-nf4-gemm>=0.7.0 and >=0.12.0, the capabilities'
        # own history): the kernel's narrower words fail them.
        "floor_exemption": "markers",
        # The runtime copy exempts anchored documents from rule 1, because a
        # finding there could only be fixed by editing an anchored file. Kept by
        # choice, as in check_readme_claims.py: the runtime's main passes rule 1
        # without it. (Its AGENTS.md counts as anchored because it quotes the
        # marker in prose; rules 2 and 3 still read it.)
        "skip_anchored": True,
        # `experts4bit-qlora==X` in the runtime's own documents pins the runtime
        # itself (rule 2's own-version pin); the kernel's reading, the lower
        # bound of a consumer range, is not a statement the runtime makes.
        "consumer_pins": False,
        # What a line must name for a bare `>= X` on it to be a kernel floor:
        # grouped-nf4-gemm, gnf4 or the `fast` extra. The runtime names the
        # kernel only where it talks about the kernel; the kernel's narrower
        # set would pass `grouped-nf4-gemm pins >= 0.28.0`, which the runtime
        # copy fails.
        "bare_floor_mentions": "kernel-or-fast",
    },
    "kernels": {
        "role": "kernels",
        # The kernel package has no `fast` extra on itself (the runtime's source
        # is exit 2 here): the floor it is held to is the consumer's, recorded in
        # the manifest.
        "floor_source": "manifest",
        # The runtime's broader markers (`history`, `older`, `landed in`, a
        # `vX.Y.Z` tag pin) would excuse lines the kernel copy fails.
        "floor_exemption": "words",
        # The kernel copy never had the exemption; adding it would stop checking
        # a document that becomes anchored (one INDEX "Current" document,
        # docs/K3-PROVENANCE-CHAIN.md, already is, and is read).
        "skip_anchored": False,
        # `experts4bit-qlora>=Y` / `==Y` in the kernel's documents is the consumer
        # range its current record covers; the runtime value would stop checking it.
        "consumer_pins": True,
        # The kernel's own name is on nearly every line of its documents, so a
        # bare `>=` beside it is as likely a floor of something it does not
        # depend on (bitsandbytes, transformers: not excluded by name there) as
        # the consumer's; here only the `fast` extra marks the consumer floor.
        "bare_floor_mentions": "fast",
    },
}

OTHER_PACKAGE = re.compile(r"(bitsandbytes|torch|triton|python|transformers|numpy|pip|cff|unsloth|vllm)", re.I)
VERSION_WORD = re.compile(r"\bversion\s+v?(\d+\.\d+\.\d+)\b", re.I)
LATEST = re.compile(r"\blatest\b[^.\n]{0,80}?\bv?(\d+\.\d+\.\d+)\b|\bv?(\d+\.\d+\.\d+)\b[^.\n]{0,80}?\blatest\b", re.I)
#: The kernel copy's own-name patterns (``_self_patterns`` builds them for any package).
SELF_NAME_VERSION = re.compile(r"grouped-nf4-gemm\s+v?(\d+\.\d+\.\d+)\b")
SELF_PIN = re.compile(r"grouped-nf4-gemm\s*(==|>=|~=)\s*v?(\d+(?:\.\d+)+)")
#: ``grouped-nf4-gemm >= X`` in any role (rule 1), and ``experts4bit-qlora >= / == Y``.
KERNEL_PIN = SELF_PIN
CONSUMER_PIN = re.compile(r"experts4bit-qlora\s*(>=|==)\s*v?(\d+(?:\.\d+)+)")
BARE_FLOOR = re.compile(r"(?<![\w.-])>=\s*v?(\d+\.\d+\.\d+)\b")
TORCH_FLOOR = re.compile(r"\btorch\s*(?:>=|≥)\s*v?(\d+(?:\.\d+)*)", re.I)
TRITON_FLOOR = re.compile(r"\btriton\s*(?:>=|≥)\s*v?(\d+(?:\.\d+)*)", re.I)
PYTHON_CI = re.compile(r"\bPython\s+(\d+\.\d+)\b(?![.\d])")
PYTHON_PYPROJECT = re.compile(r"pyproject says\s*>=\s*(\d+\.\d+)")
FAST_CONTEXT = re.compile(r"\[fast\]|`fast`|fast extra|consumer floor|floors? on|floor", re.I)

#: ``name >= 1.2.3`` / ``name>=1.2.3`` / ``>= 1.2.3`` -- the name is optional and may be prose.
_STATEMENT = re.compile(r"(?:(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)(?P<gap>\s*))?(?:>=|≥)\s*v?(?P<ver>\d+\.\d+\.\d+)")
_REQUIRES = re.compile(r"\brequires\s+grouped-nf4-gemm\s+v?(?P<ver>\d+\.\d+\.\d+)", re.I)
_TAG_PIN = re.compile(r"(?<![A-Za-z0-9])v\d+\.\d+\.\d+(?![A-Za-z0-9])")
_MENTIONS = re.compile(r"grouped-nf4-gemm|(?<![a-z0-9])gnf4(?![a-z0-9])|\[fast\]|`fast`|\bfast\s+extra", re.I)
_FAST_MENTIONS = re.compile(r"\[fast\]|`fast`|\bfast\s+extra", re.I)
#: ``PROFILES`` ``bare_floor_mentions``: what a line must name for a bare ``>= X`` on it to count.
_MENTION_SETS = {"kernel-or-fast": _MENTIONS, "fast": _FAST_MENTIONS}
_KERNEL_NAMES = frozenset({pep503_name(KERNEL_PACKAGE), "gnf4"})
_NAME = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


# --------------------------------------------------------------------- role --

def repo_role(root, manifest_rel: str = MANIFEST) -> str | None:
    """``"runtime"`` / ``"kernels"``: this repository's ``packages`` entry in
    docs/system-manifest.json, read by ``check_system_manifest.system_role``;
    ``None`` when the manifest or pyproject.toml is missing or names neither."""
    try:
        manifest = json.loads(read_text(Path(root) / manifest_rel))
        return system_role(manifest, load_pyproject(root))
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return None


def profile_for(root, manifest_rel: str = MANIFEST) -> dict:
    """``PROFILES`` for this repository's role; a ``ContractError`` (exit 2)
    when the role cannot be read -- a check that guessed would run the wrong rules."""
    role = repo_role(root, manifest_rel)
    if role is None:
        raise ContractError(f"cannot tell which package of the system {Path(root)} is: {manifest_rel} names no "
                            "packages entry for pyproject's [project].name or Source URL")
    return PROFILES[role]


# -------------------------------------------------------- rule 1: statements --

def historical_marker(line: str) -> str | None:
    """The first HISTORICAL_MARKER the line carries as a whole word, else None."""
    low = line.lower()
    for m in HISTORICAL_MARKERS:
        if re.search(r"(?<![a-z])" + re.escape(m) + r"(?![a-z])", low):
            return m
    if _TAG_PIN.search(line):
        return "vX.Y.Z tag pin"
    return None


def historical_word(line: str) -> str | None:
    """The first ``HISTORICAL`` word the line carries, else None."""
    m = HISTORICAL.search(line)
    return m.group(1).lower() if m else None


def _statement_spans(line: str, other_packages: frozenset[str],
                     mention: re.Pattern = _MENTIONS) -> list[tuple[int, str]]:
    """``(offset, version)`` of every statement ``statements`` reads; a bare
    ``>= X`` counts on a line ``mention`` matches."""
    found = [(m.start("ver"), m.group("ver")) for m in _REQUIRES.finditer(line)]
    mentions = mention.search(line) is not None
    for m in _STATEMENT.finditer(line):
        name = m.group("name")
        if name is not None:
            n = pep503_name(name)
            if n in _KERNEL_NAMES:
                found.append((m.start("ver"), m.group("ver")))
                continue
            if n in other_packages or not m.group("gap"):
                continue                        # torch>=2.2.0, bitsandbytes >= 0.50.0, or glued to a foreign name
        if mentions:
            found.append((m.start("ver"), m.group("ver")))
    return found


def statements(line: str, other_packages: frozenset[str]) -> list[str]:
    """Every grouped-nf4-gemm version the line states: ``grouped-nf4-gemm>=X``,
    ``requires grouped-nf4-gemm X``, or a bare ``>= X`` on a line that names
    grouped-nf4-gemm, gnf4 or the ``fast`` extra, unless the ``>=`` belongs to
    one of ``other_packages`` (or is glued to a foreign name)."""
    return [v for _, v in _statement_spans(line, other_packages)]


def _pin_spans(line: str) -> list[tuple[int, str]]:
    """``grouped-nf4-gemm >= X`` (any X.Y form), and a bare ``>= X.Y.Z`` on a
    line about a floor that pins neither package and is not beside another
    dependency's name -- the kernel copy's reading of a floor statement."""
    out = [(m.start(2), m.group(2)) for m in KERNEL_PIN.finditer(line) if m.group(1) == ">="]
    if FAST_CONTEXT.search(line) and not KERNEL_PIN.search(line) and not CONSUMER_PIN.search(line):
        for m in BARE_FLOOR.finditer(line):
            if not OTHER_PACKAGE.search(line[max(0, m.start() - 24):m.start()]):
                out.append((m.start(1), m.group(1)))
    return out


def floor_statements(line: str, other_packages: frozenset[str], mention: re.Pattern = _MENTIONS) -> list[str]:
    """Every grouped-nf4-gemm floor the line states, in either reading
    (``statements`` -- a bare ``>= X`` counting on a line ``mention`` matches --
    and the kernel copy's ``_pin_spans``), each once."""
    seen: dict[tuple[int, str], None] = {}
    for span in sorted(_statement_spans(line, other_packages, mention) + _pin_spans(line)):
        seen.setdefault(span)
    return [v for _, v in seen]


def is_anchored(path: Path) -> bool:
    if path.with_name(path.name + ".ots").is_file():
        return True
    try:
        return ANCHOR_MARKER in read_text(path)
    except (OSError, UnicodeDecodeError):
        return False


def index_current_documents(root: Path) -> list[Path]:
    """The documents docs/INDEX.md links under its ``## Current`` heading."""
    idx = root / INDEX
    if not idx.is_file():
        return []
    out: list[Path] = []
    in_current = False
    for line in read_text(idx).splitlines():
        if line.startswith("## "):
            in_current = line.lower().startswith("## current")
            continue
        if not in_current:
            continue
        for target in re.findall(r"\]\(([^)\s#]+)", line):
            if target.startswith(("http://", "https://")):
                continue
            p = (idx.parent / target).resolve()
            if p.suffix == ".md" and p.is_file() and p not in out:
                out.append(p)
    return out


def scan_documents(root: Path) -> list[Path]:
    """Rule 1's documents, in order: ``FIXED_DOCUMENTS`` that exist, the
    solution pages, then docs/INDEX.md's "Current" list."""
    root = Path(root).resolve()
    docs: list[Path] = []
    for rel in FIXED_DOCUMENTS:
        p = root / rel
        if p.is_file():
            docs.append(p)
    docs += sorted(root.glob(SOLUTIONS_GLOB))
    for p in index_current_documents(root):
        if p not in docs:
            docs.append(p)
    return docs


def other_dependencies(py: dict) -> frozenset[str]:
    """Every package this pyproject requires, by PEP 503 name, but the kernel's:
    a ``>=`` beside one of these is that package's floor, not the kernel's."""
    return frozenset(
        pep503_name(_NAME.match(r).group(1))
        for reqs in [py.get("dependencies") or []] + list((py.get("optional-dependencies") or {}).values())
        for r in reqs if _NAME.match(r)
    ) - _KERNEL_NAMES


# ------------------------------------------------------- rules 2-3: helpers --

def _status_header(text: str) -> list[tuple[int, str]]:
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip() == "---":
            break
        out.append((i, line))
    return out


def _lines(rel: str, text: str) -> list[tuple[int, str]]:
    if rel == STATUS_HEADER:
        return _status_header(text)
    return list(enumerate(text.splitlines(), 1))


def _dep_floor(py: dict, name: str) -> tuple[int, ...] | None:
    for req in py.get("dependencies") or []:
        if requirement_name(req) == name:
            m = re.search(r">=\s*v?(\d+(?:\.\d+)*)", req)
            return release_tuple(m.group(1)) if m else None
    return None


def _ci_python_versions(root: Path) -> set[str]:
    ci = root / CI_WORKFLOW
    if not ci.is_file():
        return set()
    return set(re.findall(r'python-version:\s*"?(\d+\.\d+)"?', read_text(ci)))


def _truncate_eq(doc: tuple[int, ...], canon: tuple[int, ...]) -> bool:
    """A document floor is compared at its own precision: "2.8" matches a
    pyproject floor of 2.8.0.dev0; "2.9" does not."""
    return canon[:len(doc)] == doc


def _self_patterns(name: str) -> tuple[re.Pattern, re.Pattern]:
    """``(name X.Y.Z, name ==|>=|~= X)`` for this package's own name (the kernel
    copy's ``SELF_NAME_VERSION`` / ``SELF_PIN`` when it is grouped-nf4-gemm)."""
    n = re.escape(name)
    return (re.compile(n + r"\s+v?(\d+\.\d+\.\d+)\b"),
            re.compile(n + r"\s*(==|>=|~=)\s*v?(\d+(?:\.\d+)+)"))


# --------------------------------------------------------------------- main --

def kernel_floor(py: dict, manifest: dict, prof: dict) -> tuple[tuple[int, ...] | None, str, dict | None]:
    """``(floor, label, record)``: the floor rule 1 holds statements to, how the
    messages name its source, and the compatibility record it came from (the
    kernel role's; the runtime's record is checked separately). A
    ``ContractError`` when the runtime's ``fast`` extra has no single floor."""
    if prof["floor_source"] == "pyproject":
        fr = fast_requirement(py)
        if fr is None:
            raise ContractError(f"pyproject.toml has no {KERNEL_PACKAGE} requirement in the {EXTRA} extra")
        req, spec = fr
        floor = floor_of(spec)
        if floor is None:
            raise ContractError(f"the {EXTRA} requirement {req!r} has no single >= floor")
        return parse_version(floor), f"pyproject's {EXTRA} floor", None
    rec = current_kernel_record(manifest, str(py.get("name", "")))
    if rec is None:
        return None, "the manifest's current record floor", None
    try:
        return floor_version(str(rec["floor"])), "the manifest's current record floor", rec
    except ValueError as e:
        raise ContractError(f"{MANIFEST}: current compatibility record: {e}") from e


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="this repository's root")
    ap.add_argument("--manifest", default=MANIFEST, help=f"the manifest, relative to --root (default {MANIFEST})")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    try:
        prof = profile_for(root, a.manifest)
        py = load_pyproject(root)
        manifest = json.loads(read_text(root / a.manifest))
        floor_t, floor_label, rec = kernel_floor(py, manifest, prof)
    except (ContractError, OSError, ValueError) as e:
        print(f"FAIL: {e}")
        return 2
    name = pep503_name(str(py.get("name", "")))
    version = str(py.get("version", "")).strip()
    slug = self_slug(root, py)
    if slug is None:
        print("FAIL: self-repo slug unknown (no github.com URL in [project.urls] and no git origin)")
        return 2
    floor = ".".join(map(str, floor_t)) if floor_t is not None else None
    if prof["floor_source"] == "pyproject":
        print(f"pyproject {EXTRA} floor: {fast_requirement(py)[0]}")
    elif rec is not None:
        print(f"manifest current record: {rec.get('consumer_versions')} -> {KERNEL_PACKAGE}{rec.get('floor')}")
    else:
        print(f"manifest current record: none names {KERNEL_PACKAGE}")

    # -- rule 1: the kernel floor ----------------------------------------------
    other_packages = other_dependencies(py)
    exempt = historical_marker if prof["floor_exemption"] == "markers" else historical_word
    mention = _MENTION_SETS[prof["bare_floor_mentions"]]
    failures = 0
    n_ok = n_hist = 0
    for doc in scan_documents(root):
        rel = doc.relative_to(root).as_posix()
        if prof["skip_anchored"] and is_anchored(doc):
            print(f"skip (anchored): {rel}")
            continue
        for i, line in enumerate(read_text(doc).splitlines(), 1):
            if rel == RELEASE_NOTES_GUIDE and line.lstrip().startswith(">"):
                continue                        # an example release note, not a statement
            vers = floor_statements(line, other_packages, mention)
            if not vers:
                continue
            marker = exempt(line)
            excerpt = line.strip()[:110]
            for ver in vers:
                if floor_t is not None and release_tuple(ver) == floor_t:
                    n_ok += 1
                    print(f"OK: {rel}:{i}: {KERNEL_PACKAGE} >= {ver}")
                elif marker:
                    n_hist += 1
                    print(f"historical ({marker!r}): {rel}:{i}: {KERNEL_PACKAGE} >= {ver}: {excerpt}")
                elif floor_t is None:
                    failures += 1
                    print(f"FAIL: {rel}:{i}: states {KERNEL_PACKAGE} >= {ver} but the manifest has no current record: "
                          f"{excerpt}")
                else:
                    failures += 1
                    print(f"FAIL: {rel}:{i}: states {KERNEL_PACKAGE} >= {ver}, {floor_label} is {floor}: {excerpt}")
    # Runtime: docs/system-manifest.json -- only the record for this version is current.
    if prof["floor_source"] == "pyproject":
        if "compatibility" not in manifest:
            print(f"note: {MANIFEST} has no compatibility table (scripts/check_system_manifest.py reports that)")
        else:
            hits, err = current_record(manifest, str(py.get("name", "")), version)
            if err or len(hits) != 1:
                failures += 1
                why = err or f"{len(hits)} compatibility records contain version {version}"
                print(f"FAIL: {MANIFEST}: {why}")
            else:
                rec_floor = floor_of(str(hits[0].get("floor", "")))
                if rec_floor is not None and parse_version(rec_floor) == floor_t:
                    n_ok += 1
                    print(f"OK: {MANIFEST}: record {hits[0].get('consumer_versions')!r} floor {hits[0].get('floor')}")
                else:
                    failures += 1
                    print(f"FAIL: {MANIFEST}: record {hits[0].get('consumer_versions')!r} floor "
                          f"{hits[0].get('floor')!r} != {floor}")

    # -- rules 2 and 3: own version, the other floors ---------------------------
    self_name_version, self_pin = _self_patterns(name)
    self_link = re.compile(rf"github\.com/{re.escape(slug)}/(?:blob|tree)/v(\d+(?:\.\d+)+)/")
    consumer_low = (range_lower_bound(str(rec["consumer_versions"]))
                    if prof["consumer_pins"] and rec is not None else None)
    torch_floor = _dep_floor(py, "torch")
    triton_floor = _dep_floor(py, "triton")
    ci_py = _ci_python_versions(root)
    req_py = re.search(r">=\s*(\d+\.\d+)", str(py.get("requires-python", "")))
    lic = py.get("license")
    lic = lic.get("text") if isinstance(lic, dict) else lic

    docs = (list(CURRENT_DOCS) + [STATUS_HEADER]
            + [p.relative_to(root).as_posix() for p in sorted(root.glob(SOLUTIONS_GLOB))])
    fails: list[str] = []
    n_statements = 0

    def check(rel: str, ln: int, what: str, stated: str, canonical: str, equal: bool) -> None:
        nonlocal n_statements
        n_statements += 1
        if not equal:
            fails.append(f"{rel}:{ln} -> {what} states {stated!r} but the canonical value is {canonical!r}")

    for rel in docs:
        p = root / rel
        if not p.is_file():
            fails.append(f"{rel}: missing")
            continue
        for ln, line in _lines(rel, read_text(p)):
            if HISTORICAL.search(line):
                continue
            for m in self_link.finditer(line):
                check(rel, ln, "self-repository link pinned to tag", "v" + m.group(1), "v" + version,
                      m.group(1) == version)
            for m in VERSION_WORD.finditer(line):
                if OTHER_PACKAGE.search(line[max(0, m.start() - 40):m.start()]):
                    continue
                check(rel, ln, "own version", m.group(1), version, m.group(1) == version)
            for m in LATEST.finditer(line):
                v = m.group(1) or m.group(2)
                if OTHER_PACKAGE.search(line[max(0, m.start() - 40):m.end()]):
                    continue
                check(rel, ln, "'latest' version", v, version, v == version)
            for m in self_name_version.finditer(line):
                check(rel, ln, "own version beside the package name", m.group(1), version, m.group(1) == version)
            for m in self_pin.finditer(line):
                op, v = m.group(1), m.group(2)
                if op != ">=":                  # >= of the kernel is rule 1's; of the runtime, no rule's
                    check(rel, ln, f"own version pin {op}", v, version, release_tuple(v) == release_tuple(version))
            if prof["consumer_pins"]:
                for m in CONSUMER_PIN.finditer(line):
                    if consumer_low is None:
                        continue
                    check(rel, ln, "consumer version of the current record", m.group(1) + m.group(2),
                          str(rec["consumer_versions"]), release_tuple(m.group(2)) == consumer_low)
            if torch_floor is not None:
                for m in TORCH_FLOOR.finditer(line):
                    check(rel, ln, "torch floor", m.group(1), ".".join(map(str, torch_floor)),
                          _truncate_eq(release_tuple(m.group(1)), torch_floor))
            if triton_floor is not None:
                for m in TRITON_FLOOR.finditer(line):
                    check(rel, ln, "triton floor", m.group(1), ".".join(map(str, triton_floor)),
                          _truncate_eq(release_tuple(m.group(1)), triton_floor))
            if ci_py and re.search(r"\bCI\b", line):
                for m in PYTHON_CI.finditer(line):
                    check(rel, ln, "Python version CI tests", m.group(1), "/".join(sorted(ci_py)), m.group(1) in ci_py)
            if req_py:
                for m in PYTHON_PYPROJECT.finditer(line):
                    check(rel, ln, "requires-python floor", m.group(1), req_py.group(1), m.group(1) == req_py.group(1))
    if lic and (root / "README.md").is_file():
        readme = read_text(root / "README.md")
        sec = re.search(r"^## License.*?(?=^## |\Z)", readme, re.M | re.S)
        n_statements += 1
        if sec and str(lic) not in sec.group(0):
            fails.append(f"README.md -> the License section does not name pyproject's license {lic!r}")

    for f in fails:
        print("FAIL:", f)
    if failures or fails:
        print(f"FAIL: {failures} floor statement(s) drift from {floor_label} ({floor}); {len(fails)} stale version "
              f"statement(s); pyproject is {version}"
              + (f", current consumer record {rec['consumer_versions']} -> {name}{rec['floor']}" if rec else ""))
        return 1
    print(f"OK: {n_ok} statement(s) equal {floor_label} {floor}; {n_hist} historical line(s) excused and listed above; "
          f"{n_statements} version statements across {len(docs)} current documents agree with pyproject {version}"
          + (f" and the manifest's current record ({rec['consumer_versions']} -> {name}{rec['floor']})" if rec else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
