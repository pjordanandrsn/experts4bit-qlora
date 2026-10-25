# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The byte guard (scripts/check_public_ids.py) refuses identifiers where `git grep -I` cannot see them.

Each case plants one identifier in a throwaway repository and runs the guard there. The values are built at run time,
so this file names none. The mutants the redaction review asked for: a UUID inside a NUL-byte file, an escaped-JSON instance_id,
"Vast box <n>", an id wrapped onto the next line, and a UUID cut to its prefix. Each must be refused."""
import gzip
import hashlib
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
UUID = "-".join(("3" * 8, "4" * 4, "5" * 4, "6" * 4, "7" * 12))
ZERO = "-".join(("0" * 8, "0" * 4, "0" * 4, "0" * 4, "0" * 12))
ID = str(7 * 10 ** 7 + 4321)
HOST = "ssh7." + "vast.ai"   # split, so this file plants nothing the redaction test would find


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("needs git")
    r = tmp_path / "r"
    (r / ".github").mkdir(parents=True)
    (r / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "check_public_ids.py", r / "scripts" / "check_public_ids.py")
    (r / ".github" / "private-markers.txt").write_text("# markers\n", encoding="utf-8")
    (r / "README.md").write_text("a public tree\n", encoding="utf-8")
    _git(r, "init", "-q", "-b", "main")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "base")
    _git(r, "branch", "base")
    return r


def _run(repo, *args):
    return subprocess.run([sys.executable, "scripts/check_public_ids.py", *args], cwd=repo, capture_output=True,
                          text=True)


def _add(repo, name, data):
    p = repo / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "change")


def test_a_clean_tree_and_a_clean_change_pass(repo):
    _add(repo, "notes.md", "a box with 2 GPUs, instance count 3, GPU-redacted-1a2b3c4d, GPU-" + ZERO + ", GPU-accelerated\n")
    assert _run(repo, "--tree").returncode == 0
    assert _run(repo, "--added", "--base", "base").returncode == 0


def test_a_uuid_inside_a_nul_byte_file_is_refused(repo):
    _add(repo, "bench/trace.nsys-rep", b"\0\1\2header\0GPU-" + UUID.encode() + b"\0tail")
    r = _run(repo, "--tree")
    assert r.returncode == 1 and "bench/trace.nsys-rep: a GPU UUID" in r.stdout


def test_a_uuid_inside_a_gzip_member_is_refused(repo):
    _add(repo, "bench/steps.jsonl.gz", gzip.compress(('{"uuid": "%s"}\n' % UUID).encode(), mtime=0))
    assert _run(repo, "--tree").returncode == 1


def test_the_canary_uuid_stays_allowed(repo):
    _add(repo, "bench/canary.txt", "GPU-" + ZERO + "\n")
    assert _run(repo, "--tree").returncode == 0


def test_a_pinned_exempt_fixture_is_skipped_and_a_changed_one_refused(repo):
    body = ("GPU-" + UUID + "\n").encode()
    pin = hashlib.sha256(body).hexdigest()
    (repo / ".github" / "private-markers.txt").write_text(f"#exempt-fixture: bench/held.txt 1 {pin}\n", encoding="utf-8")
    _add(repo, "bench/held.txt", body)
    assert _run(repo, "--tree").returncode == 0
    _add(repo, "bench/held.txt", body + b"one more line\n")
    r = _run(repo, "--tree")
    assert r.returncode == 1 and "no longer holds its pin" in r.stdout


@pytest.mark.parametrize("why,line", [
    ("escaped-JSON instance_id", '{"proof": "{\\"instance_id\\": \\"%s\\"}"}' % ID),
    ("JSON instance_id", '"vast_instance_id": %s,' % ID),
    ("list_after", '"list_after": ["%s"]' % ID),
    ("Vast box", "ran on Vast box %s (EPYC)" % ID),
    ("Vast instance", "Vast instance **%s** was destroyed" % ID),
    ("glued", "see instance%s's receipt" % ID),
    ("container label", "VAST_CONTAINERLABEL=C.%s" % ID),
    ("ssh endpoint", "ssh root@%s:%s" % (HOST, ID[:5])),
    ("ssh endpoint, port first", "ssh -p %s root@%s" % (ID[:5], HOST)),
    ("GPU UUID prefix", "the card (GPU-%s) was pinned" % UUID[:8]),
    ("MIG prefix", "MIG-%s" % UUID[:13]),
    ("public IP after its key", '{"public_ipaddr": "%s"}' % ".".join(("81", "2", "3", "4"))),
    ("public IP after @", "ssh root@%s" % ".".join(("81", "2", "3", "4"))),
    ("public IP", "PUBLIC_IPADDR=" + ".".join(("203", "0", "113", "9"))),
    ("home directory", "read from /Users/" + "someone" + "/runs/x"),
])
def test_an_added_identifier_is_refused(repo, why, line):
    _add(repo, "bench/RESULTS.md", line + "\n")
    r = _run(repo, "--added", "--base", "base")
    assert r.returncode == 1, (why, r.stdout)


def test_a_private_or_documentation_address_passes(repo):
    _add(repo, "bench/RESULTS.md", 'ssh root@10.0.0.5 and {"public_ipaddr": "%s"}\n' % ".".join(("203", "0", "113", "9")))
    assert _run(repo, "--added", "--base", "base").returncode == 0


def test_an_id_wrapped_onto_the_next_line_is_refused(repo):
    _add(repo, "bench/RESULTS.md", "the box was destroyed with Vast instance\n%s and absent.\n" % ID)
    r = _run(repo, "--added", "--base", "base")
    assert r.returncode == 1 and "wrapped onto the next line" in r.stdout


def test_an_old_line_is_not_judged_by_the_added_rules(repo):
    _add(repo, "bench/RESULTS.md", "Vast instance %s\n" % ID)
    _git(repo, "branch", "-f", "base")
    _add(repo, "bench/other.md", "a new, clean line\n")
    assert _run(repo, "--added", "--base", "base").returncode == 0
