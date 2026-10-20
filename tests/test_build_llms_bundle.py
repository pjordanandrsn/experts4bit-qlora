"""scripts/build_llms_bundle.py writes every URL in the plain-text-safe form and refuses a bundle with an unsafe one.

GitHub's traffic (2026-10-20) showed visitors landing on 404 paths ending in ")." and "))." -- markdown links inside
parentheses, read as plain text by a tool whose autolinker took the URL up to the next whitespace."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("build_llms_bundle", ROOT / "scripts" / "build_llms_bundle.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

U = "https://github.com/pjordanandrsn/grouped-nf4-gemm/blob/main/bench/phase1/README.md"


@pytest.mark.parametrize("src,want", [
    (f"about 8.1x ([phase 1 notes]({U})).", f"about 8.1x (phase 1 notes <{U}>)."),
    (f"Full list: [CHANGELOG]({U}).", f"Full list: CHANGELOG <{U}>."),
    (f"[![PyPI](https://img.shields.io/x)]({U})", f"PyPI <{U}>"),                 # a badge: alt text and the link
    (f"see {U}).", f"see <{U}>)."),                                               # bare: punctuation stays outside
    (f"see ({U})", f"see (<{U}>)"),
    (f'"repository": "{U}",', f'"repository": "{U}",'),                          # nothing to swallow: byte-identical
    ("https://en.wikipedia.org/wiki/Foo_(bar) is fine", "https://en.wikipedia.org/wiki/Foo_(bar) is fine"),
    (f"already <{U}>.", f"already <{U}>."),
])
def test_every_url_is_written_plain_text_safe(src, want):
    out = B.plain_text_links(src)
    assert out == want and B.unsafe_urls(out) == []


def test_unsafe_urls_finds_what_an_autolinker_would_swallow():
    assert B.unsafe_urls(f"x {U}). y {U}. z <{U}>.") == [f"{U}).", f"{U}."]


def test_the_build_refuses_a_bundle_with_an_unsafe_url(monkeypatch, tmp_path):
    """The rendering cannot make every URL safe (a URL ending in '.' inside a quoted string, say); the build then fails
    rather than ship one, and the check says how many and which."""
    monkeypatch.setattr(B, "plain_text_links", lambda text: text)                # a renderer that misses one
    monkeypatch.setattr(B, "bundle_sources", lambda root, cfg: [{"path": "x.md"}])
    monkeypatch.setattr(B, "read_text", lambda p: f"see ({U}).")
    with pytest.raises(B.ContractError, match=r"1 URL\(s\) in the bundle are directly followed"):
        B.build(tmp_path, {"canonical_url_base": "https://example.invalid/", "title": "t", "preamble": "p"})


def test_the_committed_bundle_is_plain_text_safe():
    text = (ROOT / "llms-full.txt").read_text(encoding="utf-8")
    assert B.unsafe_urls(text) == []                       # (a relative link, with no scheme, is never autolinked)
    assert ".md (https://" not in text                      # every section heading's source URL is in angle brackets


# ---- code is copied byte for byte (review of #574): an example a reader copies must still run
def test_a_fenced_code_block_is_never_rewritten():
    src = ("prose ([notes](https://x.org/a.md)).\n"
           "```python\n"
           "url = 'https://example.com/a.json'\n"
           "see(https://example.com/b).\n"
           "```\n"
           "after (https://x.org/c).\n")
    out = B.plain_text_links(src)
    lines = out.splitlines()
    assert lines[2] == "url = 'https://example.com/a.json'" and lines[3] == "see(https://example.com/b)."
    assert lines[0] == "prose (notes <https://x.org/a.md>)." and lines[5] == "after (<https://x.org/c>)."
    assert B.unsafe_urls(out) == []                         # the fenced line is code, not prose: never flagged


def test_an_inline_code_span_is_never_rewritten_but_a_link_with_code_text_is():
    src = "run `curl https://x.org/y).` then [`changelog`](https://x.org/CHANGELOG.md)."
    out = B.plain_text_links(src)
    assert "`curl https://x.org/y).`" in out and "`changelog` <https://x.org/CHANGELOG.md>." in out
    assert B.unsafe_urls(out) == []


def test_a_single_quoted_url_is_left_whole():
    """' is not URL punctuation to trim: a single-quoted URL in prose keeps its quotes and is not rewritten."""
    src = "set it to 'https://example.com/a.json' first"
    assert B.plain_text_links(src) == src and B.unsafe_urls(src) == []


def test_the_check_still_refuses_unsafe_prose():
    assert B.unsafe_urls("see (https://x/y).") == ["https://x/y)."]
    assert B.unsafe_urls("```\nsee (https://x/y).\n```\n") == []
