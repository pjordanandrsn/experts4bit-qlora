"""docs/INDEX.md's opening count is the number of distinct documents it links.

The count used to be typed by hand ("Forty-two") and drifted to 43-plus rows;
the README's copy of it drifted with it. The README no longer states a number
and this test holds INDEX.md's to what the index itself links, so adding a
document without updating the sentence fails CI instead of aging quietly.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "INDEX.md"


def linked_documents(text: str) -> set[str]:
    return {t for t in re.findall(r"\]\(([^)\s#]+)\)", text) if t.endswith(".md") and not t.startswith(("http://", "https://"))}


def test_index_count_equals_the_documents_it_links():
    text = INDEX.read_text(encoding="utf-8")
    m = re.search(r"^(\d+) documents accumulated", text, re.M)
    assert m, "docs/INDEX.md must open with '<N> documents accumulated ...'"
    docs = linked_documents(text)
    missing = sorted(d for d in docs if not (INDEX.parent / d).is_file())
    assert not missing, f"INDEX.md links documents that do not exist: {missing}"
    assert int(m.group(1)) == len(docs), f"INDEX.md says {m.group(1)} documents but links {len(docs)} distinct ones"
