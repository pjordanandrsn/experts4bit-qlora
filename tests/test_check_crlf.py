"""The CI line-ending guard refuses CRLF without rewriting files or old history."""
import importlib.util
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_crlf.py"
spec = importlib.util.spec_from_file_location("check_crlf", SCRIPT)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE).decode().strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Fixture")
    git(tmp_path, "config", "user.email", "fixture@example.invalid")
    (tmp_path / "old.md").write_bytes(b"existing\r\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    return tmp_path, git(tmp_path, "rev-parse", "HEAD")


@pytest.mark.parametrize("contents, expected", [(b"one\ntwo\n", 0), (b"one\r\ntwo\r\n", 1), (b"one\rtwo\r", 0)])
def test_added_file_lf_passes_and_cr_endings_fail(repo, contents, expected, capsys):
    root, base = repo
    path = root / "new file.md"
    path.write_bytes(contents)
    git(root, "add", str(path))
    git(root, "commit", "-qm", "addition")
    assert guard.main(["--root", str(root), "--base", base]) == expected
    assert path.read_bytes() == contents
    output = capsys.readouterr()
    assert ("new file.md: CRLF line endings" in output.err) == bool(expected)
    assert "old.md" not in output.err


def test_changed_file_crlf_refuses(repo):
    root, base = repo
    path = root / "old.md"
    path.write_bytes(b"changed\r\n")
    assert guard.main(["--root", str(root), "--base", base]) == 1


def test_csv_binary_deleted_and_symlink_are_exempt(repo):
    root, base = repo
    (root / "data.csv").write_bytes(b"a,b\r\n1,2\r\n")
    (root / "binary.bin").write_bytes(b"\0binary\r\n")
    (root / "old.md").unlink()
    (root / "link.md").symlink_to(root / "data.csv")
    git(root, "add", "-A")
    assert guard.main(["--root", str(root), "--base", base]) == 0


def test_untracked_addition_refuses_and_ignored_file_is_excluded(repo):
    root, base = repo
    (root / ".gitignore").write_bytes(b"ignored.md\n")
    (root / "ignored.md").write_bytes(b"ignored\r\n")
    (root / "new.md").write_bytes(b"new\r\n")
    checked, failures = guard.check(root, base)
    assert checked == 2
    assert failures == ["new.md: CRLF line endings on line(s) 1"]


def test_missing_base_fails_instead_of_skipping(repo):
    root, _ = repo
    assert guard.main(["--root", str(root), "--base", "missing-ref"]) == 2


def test_without_base_refuses_instead_of_scanning_history(repo):
    root, _ = repo
    assert guard.main(["--root", str(root)]) == 2
