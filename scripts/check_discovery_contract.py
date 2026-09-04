#!/usr/bin/env python3
"""Offline discoverability contract: docs/discovery-queries.json maps realistic
user queries to the solution page that should answer them, and this script
checks that each page can. Standard library only; deterministic.

For every record it verifies that:
  * the expected document exists;
  * every required concept group is present in that page (a group is a list
    of alternative phrasings; one must appear, case-insensitively);
  * the page names the canonical package and carries a canonical
    ``pip install`` route (no alias package in any install command);
  * the page has a limitations section;
  * cross-project routing: when the expected project is the related one, the
    page links to that project's repository or package;
  * an optional ``forbidden_phrases`` list does not appear (wrong-project
    interpretation).

``--bm25`` prints a small BM25 ranking of the curated docs for each query as
a regression aid. It is a local proxy only -- never evidence of ranking in
any external search or assistant.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9][a-z0-9_.+-]*")


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def _bm25_report(root: Path, corpus_globs: list[str], queries: list[dict], k1=1.5, b=0.75) -> int:
    docs = {}
    for g in corpus_globs:
        for p in sorted(root.glob(g)):
            docs[str(p.relative_to(root))] = _tokens(p.read_text(errors="replace"))
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
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--queries", default="docs/discovery-queries.json")
    ap.add_argument("--capabilities", default="docs/capabilities.json")
    ap.add_argument("--bm25", action="store_true")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    q = json.loads((root / a.queries).read_text())
    cap = json.loads((root / a.capabilities).read_text())
    proj = cap["project"]
    canonical = proj["canonical_package"]
    aliases = set(proj.get("aliases", []))
    related = {r["canonical_package"]: r for r in proj.get("related_projects", [])}
    records = q["queries"]
    errors = []
    if len(records) < q.get("minimum_queries", 30):
        errors.append(f"only {len(records)} queries; the corpus requires at least {q.get('minimum_queries', 30)}")
    seen = set()
    for r in records:
        key = r["query"].strip().lower()
        if key in seen:
            errors.append(f"duplicate query: {r['query']!r}")
        seen.add(key)
        doc = root / r["expected_document"]
        if not doc.exists():
            errors.append(f"{r['query']!r}: expected document missing: {r['expected_document']}")
            continue
        text = doc.read_text(errors="replace")
        low = text.lower()
        for group in r["required_concepts"]:
            alts = [group] if isinstance(group, str) else group
            if not any(alt.lower() in low for alt in alts):
                errors.append(f"{r['query']!r}: {r['expected_document']} lacks concept {alts}")
        for phrase in r.get("forbidden_phrases", []):
            if phrase.lower() in low:
                errors.append(f"{r['query']!r}: {r['expected_document']} contains forbidden phrase {phrase!r}")
        exp_pkg = r["expected_package"]
        if exp_pkg in aliases:
            errors.append(f"{r['query']!r}: expected_package {exp_pkg!r} is an alias, not a canonical package")
        if exp_pkg not in text:
            errors.append(f"{r['query']!r}: {r['expected_document']} never names the expected package {exp_pkg!r}")
        installs = re.findall(r"pip install\s+(?:-[\w-]+\s+)*\"?([A-Za-z0-9_.-]+)", text)
        if not installs:
            errors.append(f"{r['query']!r}: {r['expected_document']} has no pip install route")
        for pkg in installs:
            if pkg in aliases:
                errors.append(f"{r['query']!r}: {r['expected_document']} installs alias {pkg!r}")
        if exp_pkg not in installs and not any(exp_pkg in cmd for cmd in re.findall(r"pip install[^\n]*", text)):
            errors.append(f"{r['query']!r}: {r['expected_document']} has no install route for {exp_pkg!r}")
        if not re.search(r"^#+\s*.*limitation", text, re.I | re.M):
            errors.append(f"{r['query']!r}: {r['expected_document']} has no Limitations section")
        if exp_pkg != canonical:
            rp = related.get(exp_pkg)
            if rp is None:
                errors.append(f"{r['query']!r}: expected project {exp_pkg!r} is neither this project nor a related one")
            elif rp["repository"] not in text and f"pypi.org/project/{exp_pkg}" not in text:
                errors.append(f"{r['query']!r}: {r['expected_document']} routes to {exp_pkg!r} without linking its repository or PyPI page")
    if errors:
        for e in errors:
            print("FAIL:", e)
        return 1
    print(f"OK: {len(records)} queries route to existing pages with their concepts, canonical install routes and limitations")
    if a.bm25:
        _bm25_report(root, q.get("bm25_corpus", ["docs/solutions/*.md", "docs/SOLUTIONS.md", "README.md"]), records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
