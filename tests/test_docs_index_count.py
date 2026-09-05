"""docs/INDEX.md's opening count is the number of distinct documents it links.

The count used to be typed by hand ("Forty-two") and drifted to 43-plus rows;
the README's copy of it drifted with it. The README no longer states a number
and this test holds INDEX.md's to what the index itself links, so adding a
document without updating the sentence fails CI instead of aging quietly.
Links are read with ``discovery_common.md_links`` (fenced blocks stripped,
titled links handled), fragments dropped and ``./`` normalised, so the same
document linked twice counts once.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "INDEX.md"
sys.path.insert(0, str(ROOT / "scripts"))

from discovery_common import md_links  # noqa: E402


def linked_documents(text: str) -> set[str]:
    out = set()
    for target in md_links(text):
        target = target.split("#", 1)[0]
        while target.startswith("./"):
            target = target[2:]
        if target.endswith(".md") and not target.startswith(("http://", "https://")):
            out.add(target)
    return out


def test_linked_documents_counts_each_local_document_once():
    text = ("[a](A.md) [a again](./A.md#top) [b](B.md \"title\") [ext](https://x/y.md) [not md](x.json)\n"
            "```\n[fenced](C.md)\n```\n")
    assert linked_documents(text) == {"A.md", "B.md"}


def test_index_count_equals_the_documents_it_links():
    text = INDEX.read_text(encoding="utf-8")
    m = re.search(r"^(\d+) documents accumulated", text, re.M)
    assert m, "docs/INDEX.md must open with '<N> documents accumulated ...'"
    docs = linked_documents(text)
    missing = sorted(d for d in docs if not (INDEX.parent / d).is_file())
    assert not missing, f"INDEX.md links documents that do not exist: {missing}"
    assert int(m.group(1)) == len(docs), f"INDEX.md says {m.group(1)} documents but links {len(docs)} distinct ones"
