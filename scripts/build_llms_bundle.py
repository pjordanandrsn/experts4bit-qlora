#!/usr/bin/env python3
"""Build llms-full.txt -- a deterministic, curated context bundle for language
models and coding agents. A project-generated convenience, NOT part of the
llms.txt proposal; llms.txt (hand-written, at the repository root) is the
orientation file, this is the concatenation it points to.

Inputs are the curated list in docs/llms-bundle.json (README opening up to a
marker, SOLUTIONS index and pages, capabilities.json, STATUS, the named
architecture/API documents, and a projection of docs/claims.json). Excluded
by construction: the full CHANGELOG, raw receipts, binaries, .ots files,
telemetry, historical duplicates. No timestamps. Each source gets a boundary
heading with its canonical main-branch URL; a ``glob`` source becomes one
section per matching file.

The claims projection lists every claim with an ACTIVE status under its own
``[status]`` (measured-private ones flagged "receipt private"), then ``open``
claims under "Open items" and ``projected`` ones under "Projections";
retired and superseded claims are omitted on purpose. Each line quotes the
claim's opening, up to ``CLAIM_CHARS``, ending at a sentence where one fits:
the bundle routes a reader to a claim, and docs/claims.json carries the rest.
Full texts grew the bundle by about 600 bytes per read and pushed it past its
size cap twice in a week. A README
``until_marker`` that is not found is a failure, not the whole file.

The bundle is read as PLAIN TEXT (by language-model tools and chat clients), where an autolinker takes a URL up to the
next whitespace. A markdown link inside parentheses, ``([notes](https://.../README.md)).``, then becomes a request
for ``.../README.md)).`` and a 404: GitHub's traffic showed exactly those paths (2026-10-20). So every URL is
written in the plain-text-safe form: ``[text](url)`` becomes ``text <url>`` (a badge, ``[![alt](img)](url)``,
becomes ``alt <url>``), a bare URL that ends in ``)`` or ``.`` becomes ``<url>`` with that punctuation outside it (a
bare URL ending otherwise, e.g. inside a JSON string, is left byte-identical), and each section's source URL is in
angle brackets. CODE is never rewritten: fenced blocks and inline code spans are copied byte for byte, so an example
a reader copies still runs.
The build refuses a bundle in which any URL is still directly followed by ``)`` or ``.``.

    python scripts/build_llms_bundle.py            # write llms-full.txt
    python scripts/build_llms_bundle.py --check    # exit 1 if stale
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discovery_common import ACTIVE_STATUSES, ContractError, bundle_sources, load_claims, read_text, write_text  # noqa: E402


#: a badge: a linked image, [![alt](image-url)](link-url), rendered as its alt text and the link
_MD_BADGE = re.compile(r"\[!\[([^\]\n]*)\]\((https?://[^\s)]+)\)\]\((https?://[^\s)]+)\)")
#: a markdown link or image whose target is an absolute URL: [text](url) or ![alt](url), optionally with a title
_MD_LINK = re.compile(r"!?\[([^\]\n]*)\]\((https?://[^\s)]+)(?:\s+\"[^\"]*\")?\)")
#: a bare URL not already inside angle brackets
_BARE_URL = re.compile(r"(?<![<\w])https?://[^\s<>\"']+")
#: an inline code span, and a fence line (``` or ~~~): code is copied byte for byte, never rewritten
_CODE_SPAN = re.compile(r"`[^`\n]*`")
_FENCE = re.compile(r"^\s*(```|~~~)")


def _trim_url(url: str) -> tuple[str, str]:
    """(url, trailing) with the trailing punctuation an autolinker should never take: sentence marks, and a closing
    parenthesis the URL itself does not open (GitHub's autolink rule)."""
    tail = ""
    while url:
        if url[-1] in ".,;:!?":
            tail, url = url[-1] + tail, url[:-1]
        elif url[-1] == ")" and url.count(")") > url.count("("):
            tail, url = ")" + tail, url[:-1]
        else:
            break
    return url, tail


def _in_code(spans: list[tuple[int, int]], i: int) -> bool:
    return any(a < i < b for a, b in spans)


def _prose_links(text: str) -> str:
    """One run of prose: links and bare URLs made plain-text safe, inline code spans left byte for byte. A link whose
    TEXT is code (``[`x`](url)``) is still a link: its ``[`` sits outside the span."""
    spans = [m.span() for m in _CODE_SPAN.finditer(text)]

    def link(m: re.Match, out: str) -> str:
        return m.group(0) if _in_code(spans, m.start()) else out
    text = _MD_BADGE.sub(lambda m: link(m, f"{m.group(1)} <{m.group(3)}>"), text)
    spans = [m.span() for m in _CODE_SPAN.finditer(text)]
    text = _MD_LINK.sub(lambda m: link(m, f"{m.group(1)} <{m.group(2)}>" if m.group(1).strip()
                                        else f"<{m.group(2)}>"), text)
    spans = [m.span() for m in _CODE_SPAN.finditer(text)]

    def bare(m: re.Match) -> str:
        url, tail = _trim_url(m.group(0))
        if not tail or not url or _in_code(spans, m.start()):    # nothing to swallow, or code: leave the bytes
            return m.group(0)
        return f"<{url}>{tail}"
    return _BARE_URL.sub(bare, text)


def plain_text_links(text: str) -> str:
    """Every URL in its plain-text-safe form (see the module docstring). Fenced code blocks are copied unchanged."""
    out, prose, fenced = [], [], False
    for line in text.splitlines(keepends=True):
        if _FENCE.match(line):
            if not fenced and prose:
                out.append(_prose_links("".join(prose)))
                prose = []
            out.append(line)
            fenced = not fenced
        elif fenced:
            out.append(line)
        else:
            prose.append(line)
    if prose:
        out.append(_prose_links("".join(prose)))
    return "".join(out)


def unsafe_urls(text: str) -> list[str]:
    """URLs a plain-text autolinker would extend into punctuation that is not the URL's: not in angle brackets, and
    ending in a sentence mark or an unbalanced ')' (the URL read as such a linker reads it: up to whitespace). A ')'
    the URL itself opens, as in a Wikipedia title, is the URL's own and is not flagged. Code (fences and inline
    spans) is not prose and is not flagged: it is copied verbatim for a reader to run."""
    bad, fenced = [], False
    for line in text.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        spans = [m.span() for m in _CODE_SPAN.finditer(line)]
        bad += [m.group(0) for m in _BARE_URL.finditer(line) if _trim_url(m.group(0))[1] and not _in_code(spans, m.start())]
    return bad


def _readme_opening(text: str, marker: str, path: str) -> str:
    i = text.find(marker)
    if i < 0:
        raise ContractError(f"{path}: until_marker {marker!r} not found -- the opening would silently become the whole file")
    return text[:i].rstrip() + "\n"


#: a projected claim quotes at most this many characters of its text (docs/claims.json is authoritative)
CLAIM_CHARS = 320


def _claim_opening(text: str, limit: int = CLAIM_CHARS) -> str:
    """The claim's opening on one line: whole if it fits, else cut at the last sentence end (``.`` or ``;`` before a
    space; a decimal point never matches) past a third of ``limit``, else at the last word, and marked `` …``."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    head = text[:limit + 1]
    ends = [m.end() for m in re.finditer(r"[.;](?=\s)", head) if m.end() >= limit // 3]
    end = ends[-1] if ends else (head.rfind(" ") if head.rfind(" ") > 0 else limit)
    return text[:end].rstrip(" ,;:") + " …"


def _claims_projection(root: Path, claims_path: str) -> str:
    claims, _vocab = load_claims(root, claims_path)

    def line(c: dict) -> str:
        st = c["status"]
        flag = " (receipt private)" if st == "measured-private" else ""
        return f"- {c['id']} [{st}]{flag}: {_claim_opening(c.get('claim', ''))}"

    active = [c for c in claims if c["status"] in ACTIVE_STATUSES]
    open_items = [c for c in claims if c["status"] == "open"]
    projected = [c for c in claims if c["status"] == "projected"]
    lines = [f"Active claims (projection of docs/claims.json, each quoted up to {CLAIM_CHARS} characters; the file is "
             "authoritative and carries the full text and evidence paths):", ""]
    lines += [line(c) for c in active]
    if open_items:
        lines += ["", "Open items (no evidence either way):", ""]
        lines += [line(c) for c in open_items]
    if projected:
        lines += ["", "Projections (arithmetic, not runs):", ""]
        lines += [line(c) for c in projected]
    lines.append("")
    lines.append(f"({len(active)} active claims listed, {len(open_items)} open, {len(projected)} projected; "
                 "retired and superseded claims are omitted on purpose.)")
    return "\n".join(lines) + "\n"


def build(root: Path, cfg: dict) -> str:
    base = cfg["canonical_url_base"].rstrip("/") + "/"
    out = [f"# {cfg['title']}", "", cfg["preamble"].strip(), "",
           "Generated by scripts/build_llms_bundle.py from the sources named in docs/llms-bundle.json.",
           "Each section below starts with `## Source:` and the canonical URL of the file it was taken from.", ""]
    for item in bundle_sources(root, cfg):
        if item.get("kind", "file") == "claims":
            body = _claims_projection(root, item["path"])
        else:
            body = read_text(root / item["path"])
            if item.get("until_marker"):
                body = _readme_opening(body, item["until_marker"], item["path"])
        out.append(f"## Source: {item['path']} <{base}{item['path']}>")
        out.append("")
        out.append(plain_text_links(body.strip()))
        out.append("")
    text = "\n".join(out).rstrip() + "\n"
    bad = unsafe_urls(text)
    if bad:
        raise ContractError(f"{len(bad)} URL(s) in the bundle are directly followed by ')' or '.', which a plain-text "
                            f"autolinker takes into the link (a 404): {bad[:3]}")
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--config", default="docs/llms-bundle.json")
    ap.add_argument("--out", default="llms-full.txt")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    cfg = json.loads(read_text(root / a.config))
    try:
        text = build(root, cfg)
    except ContractError as e:
        print(f"FAIL: {e}")
        return 1
    outp = root / a.out
    limit = cfg.get("max_bytes", 400_000)
    n_bytes = len(text.encode("utf-8"))
    if n_bytes > limit:
        print(f"FAIL: bundle is {n_bytes} bytes > max_bytes {limit}; trim docs/llms-bundle.json")
        return 1
    if a.check:
        if not outp.exists() or read_text(outp) != text:
            print(f"FAIL: {a.out} is stale -- run: python scripts/build_llms_bundle.py")
            return 1
        print(f"OK: {a.out} is current ({n_bytes} bytes, {len(cfg['sources'])} sources)")
        return 0
    write_text(outp, text)
    print(f"wrote {a.out} ({n_bytes} bytes, {len(cfg['sources'])} sources)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
