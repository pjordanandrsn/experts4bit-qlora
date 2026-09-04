#!/usr/bin/env python3
"""Offline discoverability contract: docs/discovery-queries.json maps realistic
user queries to the solution page that should answer them, and this script
checks that each page can. Standard library only; deterministic.

For every record it verifies that:
  * the expected document exists;
  * every required concept group is present in that page as a whole token
    (a group is a list of alternative phrasings; one must appear,
    case-insensitively, with no letter or digit touching it -- ``gpu`` does
    not count ``gpus``);
  * the page names the canonical package and carries a canonical
    ``pip install`` route: install commands are read only from fenced blocks
    and inline code spans, every positional target of every one of them is
    checked against the alias set, and the expected package must be one of
    them -- directly, or through an extra of the canonical package whose
    ``[project.optional-dependencies]`` entry pins it (``experts4bit-qlora
    [fast]`` installs grouped-nf4-gemm; a trailing comment naming it does
    not);
  * the page has a limitations section;
  * cross-project routing: when the expected project is the related one, the
    page links to that project's repository or package;
  * an optional ``forbidden_phrases`` list does not appear (wrong-project
    interpretation).

``--bm25`` prints a small BM25 ranking of the curated docs for each query as
a regression aid. It is a local proxy only -- never evidence of ranking in
any external search or assistant. ``--bm25-min-top1 N`` turns it into a
floor: fail when fewer than N queries rank their own page first.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discovery_common import (  # noqa: E402
    install_targets_raw, load_pyproject, md_fences, pep503_name, read_text, requirement_extras, requirement_name,
    strip_fences, word_in,
)

TOKEN = re.compile(r"[a-z0-9][a-z0-9_.+-]*")
INLINE_CODE = re.compile(r"`([^`\n]+)`")
LIMITATIONS_HEADING = re.compile(r"^#+\s*.*limitation", re.I | re.M)


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def _bm25_report(root: Path, corpus_globs: list[str], queries: list[dict], k1=1.5, b=0.75) -> int:
    """Print the ranking; return how many queries rank their page first."""
    docs = {}
    for g in corpus_globs:
        for p in sorted(root.glob(g)):
            docs[p.relative_to(root).as_posix()] = _tokens(read_text(p))
    if not docs:
        print("bm25: no corpus")
        return 0
    N = len(docs)
    avgdl = sum(len(t) for t in docs.values()) / N
    df = Counter()
    for toks in docs.values():
        df.update(set(toks))
    hits = 0
    for q in queries:
        qt = _tokens(q["query"])
        scores = {}
        for name, toks in docs.items():
            tf = Counter(toks)
            dl = len(toks)
            s = 0.0
            for t in qt:
                if t not in tf:
                    continue
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / avgdl))
            scores[name] = s
        ranked = sorted(scores, key=lambda n: -scores[n])
        want = q["expected_document"]
        pos = ranked.index(want) + 1 if want in ranked else None
        hits += 1 if pos == 1 else 0
        print(f"  bm25 top1={'yes' if pos == 1 else 'no ':3} rank={pos!s:>4}  {q['query'][:60]!r} -> {ranked[0]}")
    print(f"bm25: {hits}/{len(queries)} queries rank their expected page first (local proxy, not a ranking claim)")
    return hits


def _install_commands(text: str) -> str:
    """Every line that can carry an install command: fenced-block bodies and
    the inline code spans of the prose. Prose is never an install route."""
    parts = [body for _lang, body, _line in md_fences(text)]
    parts += INLINE_CODE.findall(strip_fences(text))
    return "\n".join(parts)


def _page_facts(root: Path, rel: str, cache: dict, canonical: str, extra_provides: dict[str, set[str]]) -> dict | None:
    """Per-document facts, computed once however many queries route to it.
    ``routes`` is every package an install command on the page actually
    installs: each positional target, plus what the canonical package's
    extras pin when the target is ``canonical[extra]``."""
    if rel not in cache:
        p = root / rel
        if not p.is_file():
            cache[rel] = None
        else:
            text = read_text(p)
            targets = install_targets_raw(_install_commands(text))
            packages = [requirement_name(t) for t, kind in targets if kind == "package"]
            routes = set(packages)
            for t, kind in targets:
                if kind == "package" and requirement_name(t) == canonical:
                    for ex in requirement_extras(t):
                        routes |= extra_provides.get(ex, set())
            cache[rel] = {
                "text": text,
                "low": text.lower(),
                "n_targets": len(targets),
                "packages": packages,
                "routes": routes,
                "has_limitations": LIMITATIONS_HEADING.search(text) is not None,
            }
    return cache[rel]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--queries", default="docs/discovery-queries.json")
    ap.add_argument("--capabilities", default="docs/capabilities.json")
    ap.add_argument("--bm25", action="store_true", help="print the local BM25 ranking (informational)")
    ap.add_argument("--bm25-min-top1", type=int, default=0, metavar="N",
                    help="fail when fewer than N queries rank their expected page first (default 0: never)")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    q = json.loads(read_text(root / a.queries))
    cap = json.loads(read_text(root / a.capabilities))
    proj = cap["project"]
    canonical = proj["canonical_package"]
    canonical_norm = pep503_name(canonical)
    aliases = {pep503_name(x) for x in proj.get("aliases", [])}
    related = {r["canonical_package"]: r for r in proj.get("related_projects", [])}
    py = load_pyproject(root)
    extra_provides = {pep503_name(ex): {requirement_name(req) for req in reqs}
                      for ex, reqs in (py.get("optional-dependencies") or {}).items()}
    records = q["queries"]
    errors = []
    if len(records) < q.get("minimum_queries", 30):
        errors.append(f"only {len(records)} queries; the corpus requires at least {q.get('minimum_queries', 30)}")
    seen = set()
    facts_cache: dict = {}
    for r in records:
        key = r["query"].strip().lower()
        if key in seen:
            errors.append(f"duplicate query: {r['query']!r}")
        seen.add(key)
        rel = r["expected_document"]
        facts = _page_facts(root, rel, facts_cache, canonical_norm, extra_provides)
        if facts is None:
            errors.append(f"{r['query']!r}: expected document missing: {rel}")
            continue
        text, low = facts["text"], facts["low"]
        for group in r["required_concepts"]:
            alts = [group] if isinstance(group, str) else group
            if not any(word_in(alt, low) for alt in alts):
                errors.append(f"{r['query']!r}: {rel} lacks concept {alts} (whole-token match)")
        for phrase in r.get("forbidden_phrases", []):
            if phrase.lower() in low:
                errors.append(f"{r['query']!r}: {rel} contains forbidden phrase {phrase!r}")
        exp_pkg = r["expected_package"]
        exp_norm = pep503_name(exp_pkg)
        if exp_norm in aliases:
            errors.append(f"{r['query']!r}: expected_package {exp_pkg!r} is an alias, not a canonical package")
        if not word_in(exp_pkg, low):
            errors.append(f"{r['query']!r}: {rel} never names the expected package {exp_pkg!r}")
        if facts["n_targets"] == 0:
            errors.append(f"{r['query']!r}: {rel} has no pip install route (in a fenced block or inline code)")
        for pkg in facts["packages"]:
            if pkg in aliases:
                errors.append(f"{r['query']!r}: {rel} installs alias {pkg!r}")
        if exp_norm not in facts["routes"]:
            errors.append(f"{r['query']!r}: {rel} has no install route for {exp_pkg!r} (neither a pip install target "
                          f"nor pinned by an extra of {canonical} that the page installs)")
        if not facts["has_limitations"]:
            errors.append(f"{r['query']!r}: {rel} has no Limitations section")
        if exp_pkg != canonical:
            rp = related.get(exp_pkg)
            if rp is None:
                errors.append(f"{r['query']!r}: expected project {exp_pkg!r} is neither this project nor a related one")
            elif rp["repository"] not in text and f"pypi.org/project/{exp_pkg}" not in text:
                errors.append(f"{r['query']!r}: {rel} routes to {exp_pkg!r} without linking its repository or PyPI page")
    if errors:
        for e in errors:
            print("FAIL:", e)
        return 1
    print(f"OK: {len(records)} queries route to existing pages with their concepts, canonical install routes and limitations")
    if a.bm25 or a.bm25_min_top1 > 0:
        hits = _bm25_report(root, q.get("bm25_corpus", ["docs/solutions/*.md", "docs/SOLUTIONS.md", "README.md"]), records)
        if hits < a.bm25_min_top1:
            print(f"FAIL: bm25 top-1 hits {hits} < --bm25-min-top1 {a.bm25_min_top1}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
