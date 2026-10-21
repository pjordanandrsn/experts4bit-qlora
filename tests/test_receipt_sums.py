"""Every file a committed receipt's SHA256SUMS lists is either tracked in git or named, at the same sha256, by that
receipt's STORE-ONLY.sha256.

``.gitignore`` has ``*.log``, so ``git add`` of a receipt directory silently drops its logs. That happened on the P99
read and again on P115 Phase C (#1342). Force-add them: ``git add -f bench/pNN/receipts/<run>``. A file that is
deliberately not committed (it carries a host identifier, carriage returns, or is kept outside git) gets a line in the
receipt's STORE-ONLY.sha256 instead: its sha256 then its path, under a ``#`` header that says where the bytes are.

Tracked, not present: a log that sits on a contributor's disk but is ignored by git would pass a file-exists check
locally and fail on a fresh checkout. The STORE-ONLY line must carry the SAME sha256 as the SHA256SUMS line it accounts
for. Tracked files are checked for presence only: a Windows checkout with ``core.autocrlf`` rewrites text files, so
their bytes are not hashed here.
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = re.compile(r"^([0-9a-f]{64})\s+\*?(?:\./)?(.+?)\s*$")


def _entries(path: Path) -> dict:
    """{relative path: sha256} from a SHA256SUMS or STORE-ONLY.sha256 file; ``#`` lines and blanks are skipped."""
    out = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = ENTRY.match(ln)
        if m and not ln.lstrip().startswith("#"):
            out[m.group(2)] = m.group(1)
    return out


def tracked_files(root: Path) -> set:
    """Repository-relative paths git tracks under ``root`` (the index, so a file staged but not yet committed counts)."""
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True).stdout
    return {p for p in out.decode("utf-8").split("\0") if p}


def unaccounted(root: Path, tracked: set) -> dict:
    """{receipt directory: [problems]} for every SHA256SUMS line in bench/**/receipts/** that is neither a tracked file
    nor a same-sha line of that receipt's STORE-ONLY.sha256."""
    out = {}
    for sums in sorted((root / "bench").rglob("SHA256SUMS")):
        rel_sums = sums.relative_to(root)
        if "receipts" not in rel_sums.parts or rel_sums.as_posix() not in tracked:
            continue
        d = sums.parent
        store_only = _entries(d / "STORE-ONLY.sha256") if (d / "STORE-ONLY.sha256").is_file() else {}
        problems = []
        for rel, sha in sorted(_entries(sums).items()):
            if (d / rel).relative_to(root).as_posix() in tracked:
                continue
            if rel not in store_only:
                problems.append(f"{rel}: not tracked and not in STORE-ONLY.sha256")
            elif store_only[rel] != sha:
                problems.append(f"{rel}: STORE-ONLY.sha256 has {store_only[rel][:12]}, SHA256SUMS has {sha[:12]}")
        if problems:
            out[d.relative_to(root).as_posix()] = problems
    return out


def test_every_file_a_receipt_lists_is_tracked_or_store_only():
    bad = unaccounted(ROOT, tracked_files(ROOT))
    assert not bad, (
        "SHA256SUMS lists files that are neither tracked nor in STORE-ONLY.sha256 at the same sha256. A receipt log needs "
        "`git add -f` (.gitignore has *.log); a file deliberately kept out needs a STORE-ONLY.sha256 line: "
        + "; ".join(f"{d}: {', '.join(p[:3])}{' ...' if len(p) > 3 else ''}" for d, p in bad.items()))


SAME = re.compile(r"^#\s*same bytes:\s*(\S+)\s+at\s+([0-9a-f]{7,40})\s*$")


def same_bytes_claims(path: Path) -> list:
    """[(entry path, entry sha256, tracked path, commit)] for each STORE-ONLY entry whose preceding line is a
    ``# same bytes: <tracked path> at <commit>`` annotation."""
    out, pending = [], None
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = SAME.match(ln)
        if m:
            pending = m.groups()
            continue
        e = ENTRY.match(ln)
        if e and not ln.lstrip().startswith("#"):
            if pending:
                out.append((e.group(2), e.group(1), *pending))
            pending = None
        elif ln.strip() and not ln.lstrip().startswith("#"):
            pending = None
    return out


def wrong_same_bytes(root: Path) -> list:
    """Every ``same bytes`` annotation whose tracked path at that commit does not hash to the entry's sha256."""
    import hashlib
    bad = []
    for so in sorted((root / "bench").rglob("STORE-ONLY.sha256")):
        for entry, sha, tracked, commit in same_bytes_claims(so):
            got = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{tracked}"], capture_output=True)
            if got.returncode != 0 or hashlib.sha256(got.stdout).hexdigest() != sha:
                bad.append(f"{so.parent.relative_to(root).as_posix()}/{entry}: {tracked} at {commit}")
    return bad


def test_every_same_bytes_line_is_true():
    """A STORE-ONLY entry annotated ``same bytes: <path> at <commit>`` has exactly those bytes in git at that commit."""
    claims = [c for so in (ROOT / "bench").rglob("STORE-ONLY.sha256") for c in same_bytes_claims(so)]
    assert len(claims) >= 66                                        # 66 when these lines were written
    assert wrong_same_bytes(ROOT) == []


def test_a_same_bytes_line_that_names_other_bytes_fails(tmp_path):
    import hashlib
    root = tmp_path / "repo"
    (root / "kit").mkdir(parents=True)
    (root / "kit" / "box.py").write_text("print(1)\n", encoding="utf-8")
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "kit/box.py"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "kit"], check=True)
    head = subprocess.run(git[:3] + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    d = root / "bench" / "pX" / "receipts" / "pX-1"
    d.mkdir(parents=True)
    right = hashlib.sha256(b"print(1)\n").hexdigest()
    (d / "STORE-ONLY.sha256").write_text(f"# same bytes: kit/box.py at {head[:12]}\n{right}  box.py\n", encoding="utf-8")
    assert wrong_same_bytes(root) == []
    (d / "STORE-ONLY.sha256").write_text(f"# same bytes: kit/box.py at {head[:12]}\n{'c' * 64}  box.py\n", encoding="utf-8")
    assert wrong_same_bytes(root) == [f"bench/pX/receipts/pX-1/box.py: kit/box.py at {head[:12]}"]


def test_the_check_reads_the_receipts_it_guards():
    sums = [p for p in (ROOT / "bench").rglob("SHA256SUMS") if "receipts" in p.relative_to(ROOT).parts]
    assert len(sums) >= 170                                         # 170 when this check was tightened
    assert len(list((ROOT / "bench").rglob("STORE-ONLY.sha256"))) >= 30


# --- the check against small synthetic receipts: each of the three ways a receipt can be incomplete fails it ---

SHA_A, SHA_B = "a" * 64, "b" * 64


def _receipt(tmp_path: Path, *, store_only: str | None) -> Path:
    root = tmp_path / "repo"
    d = root / "bench" / "pX" / "receipts" / "pX-1"
    d.mkdir(parents=True)
    (d / "SHA256SUMS").write_text(f"{SHA_A}  ./summary.txt\n{SHA_B}  logs/run.log\n", encoding="utf-8")
    (d / "summary.txt").write_text("ok\n", encoding="utf-8")
    if store_only is not None:
        (d / "STORE-ONLY.sha256").write_text(store_only, encoding="utf-8")
    return root


def _tracked(root: Path, *extra: str) -> set:
    base = {"bench/pX/receipts/pX-1/SHA256SUMS", "bench/pX/receipts/pX-1/summary.txt"}
    if (root / "bench/pX/receipts/pX-1/STORE-ONLY.sha256").is_file():
        base.add("bench/pX/receipts/pX-1/STORE-ONLY.sha256")
    return base | set(extra)


def test_a_tracked_log_or_a_same_sha_store_only_line_accounts_for_it(tmp_path):
    root = _receipt(tmp_path, store_only=None)
    assert unaccounted(root, _tracked(root, "bench/pX/receipts/pX-1/logs/run.log")) == {}
    root2 = _receipt(tmp_path / "b", store_only=f"# header\n\n# a group header\n{SHA_B}  logs/run.log\n")
    assert unaccounted(root2, _tracked(root2)) == {}


def test_an_ignored_log_on_disk_but_untracked_fails(tmp_path):
    """The .gitignore trap: the file exists locally, so a presence check would pass; git does not track it."""
    root = _receipt(tmp_path, store_only=None)
    (root / "bench/pX/receipts/pX-1/logs").mkdir()
    (root / "bench/pX/receipts/pX-1/logs/run.log").write_text("x\n", encoding="utf-8")
    bad = unaccounted(root, _tracked(root))
    assert list(bad) == ["bench/pX/receipts/pX-1"] and "not tracked and not in STORE-ONLY" in bad["bench/pX/receipts/pX-1"][0]


def test_a_missing_store_only_line_fails(tmp_path):
    root = _receipt(tmp_path, store_only=f"# only another file is listed\n{SHA_A}  other.txt\n")
    bad = unaccounted(root, _tracked(root))
    assert bad == {"bench/pX/receipts/pX-1": ["logs/run.log: not tracked and not in STORE-ONLY.sha256"]}


def test_a_store_only_line_with_another_sha_fails(tmp_path):
    root = _receipt(tmp_path, store_only=f"{SHA_A}  logs/run.log\n")
    bad = unaccounted(root, _tracked(root))
    assert bad and "STORE-ONLY.sha256 has aaaaaaaaaaaa, SHA256SUMS has bbbbbbbbbbbb" in bad["bench/pX/receipts/pX-1"][0]


def test_an_untracked_receipt_is_not_read(tmp_path):
    """A SHA256SUMS that git does not track (a local scratch copy) is not a committed receipt."""
    root = _receipt(tmp_path, store_only=None)
    assert unaccounted(root, set()) == {}
