"""CHANGELOG.md's Unreleased section as changelog.d/ fragments.

`scripts/changelog_fragments.py` replaces the hand-edited `## Unreleased`
section, the one hunk every pull request rewrote (2026-10-05: about ten pull
requests an hour, each merge leaving the rest DIRTY, and #1136's commit
a5f88f2a cutting the 8,187-line file to 31 lines). These tests pin what decides
a pass: what a fragment is, that Unreleased stays the pointer, that released
history is append-only against the merge base, that an unreleased fragment
leaves only by being released, and the release cut's order. The last tests run
the real check on this repository.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import changelog_fragments as cf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "changelog_fragments.py"
RELEASED = "## 0.2.0 — 2026-10-02 — second\n\n### Two\n\n- b\n\n## 0.1.0 — 2026-10-01 — first\n\n### One\n\n- a\n"
CHANGELOG = f"# Changelog\n\n## Unreleased\n\n{cf.POINTER}\n\n{RELEASED}"


def _git(root: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True, env=env).stdout


def _write(root: Path, rel: str, text: str) -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text, encoding="utf-8")


def _commit(root: Path, msg: str) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", msg)


def _run(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), *args], capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "r"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _write(root, "CHANGELOG.md", CHANGELOG)
    _write(root, "changelog.d/README.md", "# changelog.d\n\n## not a fragment\n")
    _commit(root, "base")
    return root


# --------------------------------------------------------------- fragments --

@pytest.mark.parametrize("name,text,needle", [
    ("1150.md", "### Title\n\n- body\n", None),
    ("a-b_c.d.md", "### Title only\n", None),
    ("x.md", "### Title\n\n```bash\n# a comment, not a heading\n## nor this\n```\n#### an inner heading\n", None),
    ("has space.md", "### T\n", "the name must match"),
    ("-lead.md", "### T\n", "the name must match"),
    ("x.md", "", "empty"),
    ("x.md", "\n### T\n", "first line"),
    ("x.md", "- body without a title\n", "first line"),
    ("x.md", "### One\n\n### Two\n", "2 '### ' headings"),
    ("x.md", "### T\n\n## A release-level heading\n", "'## ' heading"),
    ("x.md", "### T\n\n# A title\n", "'# ' heading"),
])
def test_fragment_rules(name, text, needle):
    probs = cf.fragment_problems(name, text)
    if needle is None:
        assert probs == []
    else:
        assert any(needle in p for p in probs), probs


def test_only_markdown_fragments_and_the_readme_live_in_the_directory(repo):
    _write(repo, "changelog.d/notes.txt", "x\n")
    (repo / "changelog.d" / "sub").mkdir()
    frags, probs = cf.read_fragments(repo)
    assert frags == []
    assert any("notes.txt: not a .md fragment" in p for p in probs)
    assert any("sub: not a file" in p for p in probs)


# -------------------------------------------------------- CHANGELOG's shape --

def test_unreleased_is_the_pointer_and_rewrapping_it_is_not_an_edit():
    assert cf.shape_problems(CHANGELOG) == []
    wrapped = CHANGELOG.replace(cf.POINTER, cf.POINTER.replace("; the release", ";\nthe release"))
    assert cf.shape_problems(wrapped) == []


def test_an_entry_under_unreleased_is_refused_with_the_way_out():
    text = CHANGELOG.replace(cf.POINTER, cf.POINTER + "\n\n### My lane read\n\n- x")
    (p,) = cf.shape_problems(text)
    assert "1 entry" in p and "### My lane read" in p and "changelog.d/<pr-or-slug>.md" in p


def test_other_text_in_unreleased_is_refused():
    (p,) = cf.shape_problems(CHANGELOG.replace(cf.POINTER, "Hand-written notes."))
    assert "not the pointer paragraph" in p


@pytest.mark.parametrize("text,needle", [
    ("# Changelog\n\n## Unreleased\n\n### A lane entry\n", "no '## X.Y.Z"),        # a5f88f2a's shape
    (f"# Changelog\n\n{RELEASED}", "exactly one '## Unreleased'"),
    (f"# Changelog\n\n{RELEASED}\n## Unreleased\n", "below it"),
])
def test_a_changelog_without_its_structure_is_a_finding(text, needle):
    (p,) = cf.shape_problems(text)
    assert needle in p


# ------------------------------------------------------ released history --

def test_lost_lines_allows_additions_anywhere():
    old = ["## 0.1.0 — 2026-10-01", "", "- a", "", "- b"]
    new = ["## 0.2.0 — 2026-10-02", "", "- new", ""] + old[:3] + ["- inserted note"] + old[3:] + ["tail"]
    assert cf.lost_lines(old, new) == []


def test_lost_lines_names_an_edit_and_a_deletion():
    old = ["## 0.1.0 — 2026-10-01", "", "- a", "- b", "- c"]
    assert cf.lost_lines(old, ["## 0.1.0 — 2026-10-01", "", "- a", "- B", "- c"]) == [(3, "- b")]
    assert cf.lost_lines(old, old[:2] + old[3:]) == [(2, "- a")]


def test_lost_lines_exempts_conflict_markers():
    old = ["## 0.1.0 — 2026-10-01", "<<<<<<< HEAD", "- a", "=======", "- a", ">>>>>>> abc (x)"]
    assert cf.lost_lines(old, ["## 0.1.0 — 2026-10-01", "- a", "- a"]) == []


def test_lost_lines_verdict_is_exact_where_difflib_alignment_is_not():
    # old is a subsequence of new; difflib's longest-block alignment may pair
    # the lines differently, but the verdict is the two-pointer test.
    old = ["a", "b", "c", "a", "b"]
    new = ["a", "b", "x", "c", "a", "y", "a", "b", "c", "b"]
    assert cf.lost_lines(old, new) == []


def test_a_fragment_passes_against_the_merge_base(repo):
    _git(repo, "checkout", "-q", "-b", "lane")
    _write(repo, "changelog.d/1150-lane.md", "### Lane read\n\n- x\n")
    r = _run(repo, "--check", "--base", "main")
    assert r.returncode == 0, r.stdout
    assert "1 fragment(s)" in r.stdout and "merge base" in r.stdout


def test_an_edited_released_line_fails_unless_the_label_allows_it(repo):
    _git(repo, "checkout", "-q", "-b", "lane")
    _write(repo, "CHANGELOG.md", CHANGELOG.replace("- a\n", "- a, corrected\n"))
    r = _run(repo, "--check", "--base", "main")
    assert r.returncode == 1
    assert "released history is append-only" in r.stdout and "'- a'" in r.stdout
    r = _run(repo, "--check", "--base", "main", "--allow-history-edit")
    assert r.returncode == 0, r.stdout
    assert "NOTE (--allow-history-edit)" in r.stdout


def test_the_a5f88f2a_replacement_fails_against_the_merge_base(repo):
    _git(repo, "checkout", "-q", "-b", "lane")
    _write(repo, "CHANGELOG.md", "# Changelog\n\n## Unreleased\n\n### TC1 amendment 39 registered\n\n- x\n")
    r = _run(repo, "--check", "--base", "main")
    assert r.returncode == 1
    assert "no '## X.Y.Z" in r.stdout
    assert f"{len(cf.released_lines(CHANGELOG))} line(s) of the released sections" in r.stdout


def test_a_dropped_fragment_fails_and_a_renamed_one_passes(repo):
    _write(repo, "changelog.d/1150-lane.md", "### Lane read\n\n- x\n")
    _commit(repo, "lane merged")
    _git(repo, "checkout", "-q", "-b", "next")
    (repo / "changelog.d" / "1150-lane.md").rename(repo / "changelog.d" / "lane-read.md")
    assert _run(repo, "--check", "--base", "main").returncode == 0
    (repo / "changelog.d" / "lane-read.md").unlink()
    r = _run(repo, "--check", "--base", "main")
    assert r.returncode == 1
    assert "1150-lane.md" in r.stdout and "leaves only by being released" in r.stdout


def test_empty_base_skips_history_and_a_bad_base_cannot_run(repo):
    assert _run(repo, "--check", "--base", "").returncode == 0
    r = _run(repo, "--check", "--base", "no-such-ref")
    assert r.returncode == 2 and "no merge base" in r.stdout


# ----------------------------------------------------------------- release --

def test_release_writes_newest_first_deletes_and_keeps_history(repo):
    _write(repo, "changelog.d/1150-old.md", "### Old\n\n- o\n")
    _commit(repo, "1150")
    _write(repo, "changelog.d/1151-a.md", "### Same commit A\n")
    _write(repo, "changelog.d/1152-b.md", "### Same commit B\n")
    _commit(repo, "1151+1152")
    _git(repo, "checkout", "-q", "-b", "release")
    _write(repo, "changelog.d/zz-uncommitted.md", "### Not committed yet\n")
    pre = _git(repo, "rev-parse", "HEAD").strip()
    _write(repo, "intro.md", "**0.3.0.** Why to upgrade.\n")
    r = _run(repo, "--release", "## 0.3.0 — 2026-10-06 — third", "--intro-file", str(repo / "intro.md"))
    assert r.returncode == 0, r.stdout
    text = (repo / "CHANGELOG.md").read_text(encoding="utf-8")
    assert text == (f"# Changelog\n\n## Unreleased\n\n{cf.POINTER}\n\n## 0.3.0 — 2026-10-06 — third\n\n"
                    "**0.3.0.** Why to upgrade.\n\n### Not committed yet\n\n### Same commit B\n\n### Same commit A\n\n"
                    f"### Old\n\n- o\n\n{RELEASED}")
    assert sorted(p.name for p in (repo / "changelog.d").iterdir()) == ["README.md"]
    (repo / "intro.md").unlink()
    _commit(repo, "release 0.3.0")
    r = _run(repo, "--check", "--base", pre)
    assert r.returncode == 0, r.stdout


def test_render_shows_the_order_the_release_will_use(repo):
    _write(repo, "changelog.d/1.md", "### First\n")
    _commit(repo, "1")
    _write(repo, "changelog.d/2.md", "### Second\n")
    r = _run(repo, "--render")
    assert r.returncode == 0
    assert r.stdout == "## Unreleased\n\n### Second\n\n### First\n"


@pytest.mark.parametrize("heading,needle,code", [
    ("## 0.2.0 — 2026-10-06 — again", "already has a 0.2.0 section", 1),
    ("0.3.0 — 2026-10-06", "one '## X.Y.Z", 2),
    ("## 0.3.0 — 2026-10-06 — x\n\nmore", "one '## X.Y.Z", 2),
])
def test_release_refuses_a_bad_or_repeated_heading(repo, heading, needle, code):
    _write(repo, "changelog.d/1.md", "### First\n")
    r = _run(repo, "--release", heading)
    assert r.returncode == code and needle in r.stdout
    assert (repo / "changelog.d" / "1.md").is_file()


def test_release_refuses_with_no_fragments_or_an_invalid_one(repo):
    r = _run(repo, "--release", "## 0.3.0 — 2026-10-06 — x")
    assert r.returncode == 1 and "nothing to release" in r.stdout
    _write(repo, "changelog.d/1.md", "no title\n")
    r = _run(repo, "--release", "## 0.3.0 — 2026-10-06 — x")
    assert r.returncode == 1 and "first line" in r.stdout
    assert (repo / "CHANGELOG.md").read_text(encoding="utf-8") == CHANGELOG


# ------------------------------------------------------ this repository --

def test_this_repository_passes_the_check():
    r = _run(ROOT, "--check")
    assert r.returncode == 0, r.stdout


def test_this_repositorys_release_heading_is_the_one_check_readme_links_reads():
    import check_readme_links as crl
    m = crl._RELEASE_HEADING.search((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    first = next(ln for ln in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").split("\n")
                 if cf.RELEASE_HEADING.match(ln))
    assert first.startswith(f"## {m.group('ver')} ")
