"""Shared, standard-library-only helpers for the discoverability scripts
(check_capabilities, check_discovery_contract, check_docs_examples,
build_llms_bundle, check_wheel_metadata).

The scripts import this with ``sys.path.insert(0, os.path.dirname(__file__))``
-- the precedent is scripts/mode_matrix_common.py. Nothing here touches torch
or the network. Python >= 3.11 (tomllib); on an older interpreter
``load_pyproject`` exits 2 with a clear message instead of guessing.

Byte-identical between experts4bit-qlora and grouped-nf4-gemm on purpose:
every repository-specific fact is read from pyproject.toml or docs/.
"""
from __future__ import annotations

import fnmatch
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------- claim status --

#: Statuses that may back a capability (docs/claims-schema.md): a run happened
#: and a receipt exists, public or private.
ACTIVE_STATUSES = frozenset({"verified", "confirmed", "measured", "measured-private"})
#: Statuses that never back a capability. ``open`` has no evidence either way.
INACTIVE_STATUSES = frozenset({"retired", "superseded", "open"})
#: Neither active nor inactive: allowed in listings, labelled, never evidence.
PROJECTED_STATUSES = frozenset({"projected"})
KNOWN_STATUSES = ACTIVE_STATUSES | INACTIVE_STATUSES | PROJECTED_STATUSES


class ContractError(Exception):
    """A contract file is malformed, or the environment cannot check it.

    Raised, not printed, so a caller decides between exit 1 (a finding) and
    exit 2 (the check could not run) and unit tests can assert on it."""


# ------------------------------------------------------------------ text I/O --

def read_text(path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path, s: str) -> None:
    Path(path).write_text(s, encoding="utf-8")


# ----------------------------------------------------------------- pyproject --

def load_pyproject(root) -> dict:
    """The ``[project]`` table of ``<root>/pyproject.toml`` plus a ``"tool"``
    key carrying the ``[tool]`` table. Exits 2 below Python 3.11: the regex
    fallback the scripts used to carry read two fields and silently agreed with
    whatever the caller expected, which is not a check."""
    try:
        import tomllib
    except ImportError:
        print("FAIL: the discoverability scripts need Python >= 3.11 (tomllib); "
              f"this interpreter is {sys.version.split()[0]}", file=sys.stderr)
        sys.exit(2)
    data = tomllib.loads(read_text(Path(root) / "pyproject.toml"))
    out = dict(data.get("project", {}))
    out["tool"] = data.get("tool", {})
    return out


def pep503_name(name: str) -> str:
    """PEP 503 normalisation: ``Grouped_NF4.gemm`` == ``grouped-nf4-gemm``."""
    return re.sub(r"[-_.]+", "-", name).lower()


_GITHUB_SLUG = re.compile(r"github\.com[:/]([^/]+/[^/.]+)")


def self_slug(root, pyproject: dict) -> str | None:
    """``owner/repo`` of this repository: from ``[project.urls]`` Source /
    Repository / Homepage when one points at github.com, else from
    ``git remote get-url origin``. ``None`` when neither says -- callers must
    FAIL on None rather than skip the self-repo link checks silently."""
    urls = pyproject.get("urls") or {}
    for label in ("Source", "Repository", "Homepage"):
        m = _GITHUB_SLUG.search(str(urls.get(label, "")))
        if m:
            return m.group(1)
    try:
        url = subprocess.run(["git", "-C", str(root), "remote", "get-url", "origin"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    m = _GITHUB_SLUG.search(url)
    return m.group(1) if m else None


def module_shipped(module: str, pyproject: dict) -> bool:
    """Does the wheel ship ``module``? With ``[tool.setuptools] py-modules``
    the module itself (or its top package, from ``packages``) must be listed;
    otherwise the first dotted component must match a ``packages.find``
    ``include`` glob, an explicit ``packages`` entry, or the project name with
    ``-`` replaced by ``_``."""
    st = (pyproject.get("tool") or {}).get("setuptools") or {}
    top = module.split(".", 1)[0]
    packages = st.get("packages")
    explicit = packages if isinstance(packages, list) else []
    py_modules = st.get("py-modules")
    if py_modules is not None:
        return module in py_modules or top in explicit
    if isinstance(packages, dict) and isinstance(packages.get("find"), dict):
        include = packages["find"].get("include") or ["*"]
        return any(fnmatch.fnmatchcase(top, pat) for pat in include)
    if explicit:
        return top in explicit
    return top == str(pyproject.get("name", "")).replace("-", "_")


def module_file(root, module: str, pyproject: dict) -> Path | None:
    """The source file of ``module`` under this tree, honouring
    ``[tool.setuptools] package-dir`` (``{"" = "kernel"}`` puts flat modules in
    kernel/; ``{"pkg" = "dir"}`` maps one package). ``None`` when absent."""
    root = Path(root)
    st = (pyproject.get("tool") or {}).get("setuptools") or {}
    pkg_dir = st.get("package-dir") or {}
    parts = module.split(".")
    bases: list[tuple[Path, list[str]]] = []
    if parts[0] in pkg_dir:
        bases.append((root / pkg_dir[parts[0]], parts[1:]))
    bases.append((root / pkg_dir.get("", ""), parts))
    for base, rest in bases:
        if rest:
            cands = (base.joinpath(*rest[:-1], rest[-1] + ".py"), base.joinpath(*rest, "__init__.py"))
        else:
            cands = (base / "__init__.py",)
        for c in cands:
            if c.is_file():
                return c
    return None


# -------------------------------------------------------------------- claims --

def load_claims(root, claims_path) -> tuple[list[dict], dict]:
    """``(claims, status_vocabulary)`` from docs/claims.json. Loud on any
    status the file itself does not define, on a vocabulary entry this module
    cannot classify, and on missing or duplicate ids -- a claim whose status
    is a typo must not pass as 'not retired'."""
    doc = json.loads(read_text(Path(root) / claims_path))
    if isinstance(doc, dict):
        claims, vocab = doc.get("claims"), doc.get("status_vocabulary")
    else:
        claims, vocab = doc, None
    if not isinstance(claims, list):
        raise ContractError(f"{claims_path}: no 'claims' list")
    if not isinstance(vocab, dict) or not vocab:
        raise ContractError(f"{claims_path}: no 'status_vocabulary' -- every status a claim uses must be defined there")
    unknown = sorted(set(vocab) - KNOWN_STATUSES)
    if unknown:
        raise ContractError(f"{claims_path}: status_vocabulary defines {unknown}, which discovery_common cannot "
                            f"classify (active {sorted(ACTIVE_STATUSES)}, inactive {sorted(INACTIVE_STATUSES)}, "
                            f"projected {sorted(PROJECTED_STATUSES)})")
    seen: set[str] = set()
    for c in claims:
        cid = c.get("id")
        if not isinstance(cid, str) or not cid:
            raise ContractError(f"{claims_path}: a claim has no id: {json.dumps(c)[:120]}")
        if cid in seen:
            raise ContractError(f"{claims_path}: duplicate claim id {cid!r}")
        seen.add(cid)
        st = c.get("status")
        if st not in vocab:
            raise ContractError(f"{claims_path}: claim {cid!r} has status {st!r}, not in the file's "
                                f"status_vocabulary {sorted(vocab)}")
    return claims, vocab


# ------------------------------------------------------------ install routes --

#: Options whose next token is a value, never a package.
_VALUE_OPTIONS = frozenset({"-i", "--index-url", "--extra-index-url", "-f", "--find-links",
                            "-r", "--requirement", "-c", "--constraint", "--target", "-t",
                            "-e", "--editable"})
_SOURCE_OPTIONS = frozenset({"-e", "--editable"})
_SEGMENT_END = frozenset({"&&", "||", ";", "|"})
_PIP = re.compile(r"pip[\d.]*")
_NAME_END = re.compile(r"[\[<>=!~;@ ]")


def _is_source(target: str) -> bool:
    return (target in (".", "..")
            or target.startswith(("git+", "http://", "https://", "file:", "./", "../", "/", ".["))
            or target.endswith((".whl", ".tar.gz", ".zip")))


def requirement_name(req: str) -> str:
    """PEP 503 name of a requirement string (``"Foo_Bar[x]>=1; m"`` -> ``foo-bar``)."""
    m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req)
    return pep503_name(m.group(1)) if m else ""


def requirement_extras(req: str) -> list[str]:
    """Normalised extras of a requirement or install target (``pkg[Fast, x]`` -> ``["fast", "x"]``)."""
    m = re.search(r"\[([^\]]*)\]", req)
    return [pep503_name(x.strip()) for x in m.group(1).split(",") if x.strip()] if m else []


def install_targets(cmd: str) -> list[tuple[str, str]]:
    """Every positional target of every ``pip install`` in ``cmd`` (multi-line
    allowed; ``#`` comments dropped) as ``(name, kind)``. ``kind`` is
    ``"package"`` for a distribution name -- extras, specifiers and quotes
    stripped -- or ``"source"`` for git+/URL/path/-e targets. Options are
    skipped, together with the value of the value-taking ones, so
    ``pip install -i URL name`` yields ``name``, never ``URL``."""
    return [(_NAME_END.split(t, 1)[0] if kind == "package" else t, kind) for t, kind in install_targets_raw(cmd)]


def install_targets_raw(cmd: str) -> list[tuple[str, str]]:
    """As ``install_targets`` but with the package token intact (extras and
    specifier kept), for callers that need the extras."""
    out: list[tuple[str, str]] = []
    for line in cmd.splitlines():
        try:
            toks = shlex.split(line, comments=True)
        except ValueError:
            toks = line.split()
        i = 0
        while i < len(toks):
            if toks[i] != "install" or i == 0 or not _PIP.fullmatch(Path(toks[i - 1]).name):
                i += 1
                continue
            i += 1
            while i < len(toks) and toks[i] not in _SEGMENT_END:
                t = toks[i]
                if t in _SOURCE_OPTIONS:
                    if i + 1 < len(toks):
                        out.append((toks[i + 1], "source"))
                    i += 2
                elif t in _VALUE_OPTIONS:
                    i += 2
                elif t.startswith("-"):
                    i += 1                      # a flag, or --option=value
                elif t == "@":
                    i += 2                      # ``name @ url``: the name was taken
                else:
                    out.append((t, "source" if _is_source(t) else "package"))
                    i += 1
    return out


# ------------------------------------------------------------------ markdown --

#: A fenced block; the back-reference closes it, so a 4-backtick fence that
#: contains a 3-backtick block is one block, not three.
FENCE = re.compile(r"^(`{3,})(\w+)?[^\n]*\n(.*?)^\1[ \t]*$", re.M | re.S)


def md_fences(text: str) -> list[tuple[str, str, int]]:
    """``(lang, body, start_line)`` per fenced block; ``lang`` lower-cased
    (``""`` when absent) and ``start_line`` the 1-based line of the opening
    fence."""
    return [((m.group(2) or "").lower(), m.group(3), text.count("\n", 0, m.start()) + 1)
            for m in FENCE.finditer(text)]


def strip_fences(text: str) -> str:
    """``text`` with every fenced block blanked; the line count is unchanged so
    positions found afterwards still map to the original file."""
    return FENCE.sub(lambda m: re.sub(r"[^\n]+", "", m.group(0)), text)


_TARGET = r"\(\s*<?([^\s()<>]+)>?(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)"
_NESTED_IMAGE_LINK = re.compile(r"\[!\[[^\[\]]*\]" + _TARGET + r"\]" + _TARGET)
_LINK = re.compile(r"!?\[[^\[\]]*\]" + _TARGET)


def md_links(text: str) -> list[str]:
    """Link targets in prose only (fenced blocks stripped), in document order:
    plain ``[t](url)``, titled ``[t](url "title")``, images, and a badge-style
    nested image link ``[![alt](img)](target)`` -- both the inner image and the
    OUTER target, which a single regex used to miss."""
    prose = strip_fences(text)
    found: list[tuple[int, str]] = []

    def take_nested(m: re.Match) -> str:
        found.append((m.start(), m.group(1)))
        found.append((m.start() + 1, m.group(2)))
        return " " * len(m.group(0))

    rest = _NESTED_IMAGE_LINK.sub(take_nested, prose)
    for m in _LINK.finditer(rest):
        found.append((m.start(), m.group(1)))
    return [t for _, t in sorted(found, key=lambda x: x[0])]


def word_in(needle: str, haystack_lower: str) -> bool:
    """Whole-token containment: ``needle`` (lower-cased) occurs in
    ``haystack_lower`` with no ``[a-z0-9]`` on either side, so ``gpu`` does
    not count ``gpus`` and ``nf4`` does not count ``nf4gemm``."""
    return re.search(r"(?<![a-z0-9])" + re.escape(needle.lower()) + r"(?![a-z0-9])", haystack_lower) is not None


# --------------------------------------------------------------- llms bundle --

def bundle_sources(root, cfg: dict) -> list[dict]:
    """docs/llms-bundle.json ``sources`` with every ``glob`` item expanded to
    one ``{"path": ...}`` entry per matching file (sorted), so a single emit
    path serves both. Other keys of a glob item are carried onto each file."""
    root = Path(root)
    out: list[dict] = []
    for item in cfg["sources"]:
        if item.get("glob"):
            rest = {k: v for k, v in item.items() if k not in ("glob", "path")}
            for f in sorted(root.glob(item["glob"])):
                out.append({"path": f.relative_to(root).as_posix(), **rest})
        else:
            out.append(dict(item))
    return out
