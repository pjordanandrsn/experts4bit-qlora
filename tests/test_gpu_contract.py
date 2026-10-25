"""bench/gpu-contract: the GPU contract a kernel PR's review runs on a rented card (README there).

The driver refuses every malformed input before any connection: an unpinned or short sha, a card class it cannot read, a
run id outside gc-<slug>-<n>, a dirty launch tree. The box's card check reads compute capability and SM count, never a name,
and its junit summary lists every skipped test. The box runner itself needs a GPU and /root, so it is read here, not run.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GC = REPO / "bench" / "gpu-contract"
SHA_A, SHA_B = "a" * 40, "b" * 40

sys.path.insert(0, str(GC))
import gc_box  # noqa: E402


def test_the_box_helper_self_test_passes():
    assert gc_box.self_test() == 0


@pytest.mark.parametrize("probe,declared,ok", [
    ("ok torch=2.8.0+cu128 cc=8.9 sms=128", "8.9/128", True),       # the RTX 4090 class
    ("ok torch=2.8.0+cu128 cc=8.9 sms=142", "8.9/128", False),      # sm_89, another count
    ("ok torch=2.8.0+cu128 cc=8.6 sms=128", "8.9/128", False),      # 128 SMs, another capability
    ("ok torch=2.8.0+cu128 cc=12.0 sms=170", "12.0/>=160", True),   # a floor
    ("ok torch=2.8.0+cu128 cc=12.0 sms=84", "12.0/>=160", False),
    ("no-cuda torch 2.8.0", "8.9/128", False)])
def test_the_card_class_is_capability_and_sm_count(probe, declared, ok):
    assert gc_box.card_ok(probe, declared)[0] is ok


@pytest.fixture
def tool(tmp_path: Path) -> Path:
    """The tool alone in a clean, committed tree: the driver refuses a dirty one."""
    root = tmp_path / "repo"
    (root / "bench" / "gpu-contract").mkdir(parents=True)
    for f in ("gc_drive.sh", "gc_run.sh", "gc_box.py"):
        shutil.copy(GC / f, root / "bench" / "gpu-contract" / f)
    (root / "bench" / "common").mkdir()
    shutil.copy(GC.parent / "common" / "drive_tail.sh", root / "bench" / "common" / "drive_tail.sh")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "tool"]):
        subprocess.run(["git", "-C", str(root), *cmd], check=True, env=env, capture_output=True)
    return root


def _drive(root: Path, tmp_path: Path, **over) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E4B_RENT_", "GC_"))}
    env.update({"E4B_RENT_SSH_HOST": "ssh8.vast.ai", "E4B_RENT_SSH_PORT": "27278", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/x",
                "E4B_RENT_RUN_DIR": str(tmp_path / "run"), "E4B_RENT_RUN_ID": "gc-gnf4-553-4090-1",
                "E4B_RENT_DEADLINE_EPOCH": "9999999999", "E4B_RENT_INSTANCE_ID": "7000123",
                "GC_GNF4_SHA": SHA_A, "GC_E4B_SHA": SHA_B, "GC_CARD": "8.9/128", "GC_DRIVE_DRYRUN": "1"})
    env.update(over)
    env = {k: v for k, v in env.items() if v is not None}
    return subprocess.run(["bash", str(root / "bench" / "gpu-contract" / "gc_drive.sh")], capture_output=True, text=True,
                          env=env, timeout=60, cwd=tmp_path)


def test_a_well_formed_launch_plans_the_run_and_names_the_tool_commit(tool, tmp_path):
    r = _drive(tool, tmp_path)
    head = subprocess.run(["git", "-C", str(tool), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert r.returncode == 0 and f"DRYRUN tool {head}" in r.stdout, r.stdout + r.stderr
    assert f"GC_GNF4_SHA={SHA_A}" in r.stdout and f"GC_E4B_SHA={SHA_B}" in r.stdout and "GC_CARD=8.9/128" in r.stdout


@pytest.mark.parametrize("over,needle", [
    ({"GC_GNF4_SHA": None}, "GC_GNF4_SHA"),
    ({"GC_GNF4_SHA": "9adeffc"}, "not a full 40-character"),             # a short sha is an unpinned ref
    ({"GC_E4B_SHA": "main"}, "not a full 40-character"),
    ({"GC_E4B_SHA": "A" * 40}, "not a full 40-character"),               # lowercase hex only
    ({"GC_CARD": "4090"}, "GC_CARD"),
    ({"GC_CARD": None}, "GC_CARD"),
    ({"E4B_RENT_RUN_ID": "p133-4090-2"}, "not gc-<slug>-<n>"),
    ({"E4B_RENT_RUN_ID": "gc-Gnf4-1"}, "not gc-<slug>-<n>"),
    ({"E4B_RENT_RUN_ID": "gc-dry-a2000-1"}, "dry run's id"),            # a dry id on the paid path
    ({"E4B_RENT_SSH_OPTS": None}, "E4B_RENT_SSH_OPTS")])
def test_a_malformed_input_is_refused_before_any_connection(tool, tmp_path, over, needle):
    r = _drive(tool, tmp_path, **over)
    assert r.returncode == 78 and needle in r.stdout, r.stdout + r.stderr
    assert "DRYRUN" not in r.stdout


def test_a_dirty_launch_tree_is_refused(tool, tmp_path):
    (tool / "bench" / "gpu-contract" / "gc_box.py").write_text("# edited after the commit\n")
    r = _drive(tool, tmp_path)
    assert r.returncode == 78 and "dirty" in r.stdout, r.stdout + r.stderr


def test_the_box_runs_both_suites_with_every_skip_listed():
    text = (GC / "gc_run.sh").read_text()
    assert 'GNF4_TESTS="test_nf4_gemv_bw.py test_dispatch_counts.py"' in text
    assert 'E4B_TESTS="tests/test_decode_graph_buckets.py tests/test_kv_step_select.py tests/test_gemv_bw_served_gpu.py"' in text
    runs = re.findall(r"python -m pytest \$(GNF4|E4B)_TESTS -q -rs -p no:cacheprovider --junitxml=\$W/logs/(\w+)\.xml", text)
    assert sorted(runs) == [("E4B", "e4b"), ("GNF4", "gnf4")], runs
    assert text.count("grep -E '^SKIPPED'") == 2, "each suite's -rs skip lines go into the summary"


def test_the_box_refuses_an_unpinned_or_dirty_checkout():
    text = (GC / "gc_run.sh").read_text()
    assert '[ ${#val} -eq 40 ] || { say "refusing: $v is not 40 characters (an unpinned ref)"; finish 78; }' in text
    assert '[ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ] && [ -z "$(git -C "$dir" status --porcelain)" ]; }' in text
    assert "assert inst == src" in text, "the installed nf4_grouped.py must be the checkout's"


def test_a_run_dir_that_already_holds_receipts_is_refused_not_deleted(tool, tmp_path):
    old = tmp_path / "run" / "gc"
    old.mkdir(parents=True)
    (old / "summary.txt").write_text("an earlier run's receipt\n")
    r = _drive(tool, tmp_path)
    assert r.returncode == 78 and "already holds receipts" in r.stdout, r.stdout + r.stderr
    assert (old / "summary.txt").read_text() == "an earlier run's receipt\n"


def test_a_class_check_that_fails_to_run_is_refused(tool, tmp_path):
    """Only 0 (inside) or 15 (outside) from the class check lets the plan through: a crash is a refusal."""
    (tool / "bench" / "gpu-contract" / "gc_box.py").write_text("import sys; sys.exit(1)\n")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "-C", str(tool), "commit", "-qam", "a broken class check"], check=True, env={**os.environ, **env})
    r = _drive(tool, tmp_path)
    assert r.returncode == 78 and "class check rc 1" in r.stdout, r.stdout + r.stderr


def _junit(tmp_path: Path) -> Path:
    p = tmp_path / "r.xml"
    p.write_text('<testsuites><testsuite><testcase classname="kernel.t" name="ok"/>'
                 '<testcase classname="kernel.t" name="skipped"><skipped message="no card"/></testcase></testsuite></testsuites>')
    return p


@pytest.mark.parametrize("ids,rc", [("kernel.t::ok", 0), ("kernel.t::skipped", 30), ("kernel.t::absent", 30),
                                    ("kernel.t::ok kernel.t::skipped", 30), ("not-an-id", 78)])
def test_a_required_test_must_pass_not_merely_not_fail(tmp_path, ids, rc):
    r = subprocess.run([sys.executable, str(GC / "gc_box.py"), "--require", ids, "--junits", str(_junit(tmp_path))],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == rc, r.stdout + r.stderr


def test_gc_require_travels_to_the_box_and_a_malformed_id_is_refused(tool, tmp_path):
    (tmp_path / "m::ta").write_text("")                           # an unquoted t[ab] would glob to this file
    r = _drive(tool, tmp_path, GC_REQUIRE="kernel.test_dispatch_counts::test_x m::t[ab]")
    assert r.returncode == 0 and "GC_REQUIRE=kernel.test_dispatch_counts::test_x\\ m::t\\[ab\\]" in r.stdout, r.stdout + r.stderr
    r = _drive(tool, tmp_path, GC_REQUIRE="kernel.t::ok;rm")
    assert r.returncode == 78 and "GC_REQUIRE id" in r.stdout, r.stdout + r.stderr


def test_each_suite_is_gated_on_the_time_left_and_the_required_tests_are_read():
    text = (GC / "gc_run.sh").read_text()
    assert 'if can_run "${GC_NEED_GNF4_S:-900}" "gnf4 kernel tests"; then' in text
    assert 'if can_run "${GC_NEED_E4B_S:-600}" "e4b served tests"; then' in text
    assert text.count('perl -e "alarm $(alarm); exec @ARGV"') == 2, "each suite's alarm stops at the receipt margin"
    assert 'python $W/gc_box.py --require "$GC_REQUIRE" --junits logs/gnf4.xml logs/e4b.xml' in text


def test_the_driver_refuses_without_the_shared_tail(tool, tmp_path):
    (tool / "bench" / "common" / "drive_tail.sh").unlink()
    r = _drive(tool, tmp_path)
    assert r.returncode == 78 and "drive_tail.sh is missing" in r.stdout, r.stdout + r.stderr


# -- the whole driver on a stub box (GC_DRY=1): staging, the nonce handshake, the poll, the fetch and the shared tail ---------------
FAKE_RUN = """#!/bin/bash
# a stand-in for gc_run.sh: binds the nonce and leaves the records the case asks for
cd "$(dirname "$0")" || exit 1
printf '%s\\n' "$GC_RUN_NONCE" > GC_RUN_NONCE
echo "GC CONTRACT FAKE" > summary.txt
case "$FAKE_CASE" in
  ok) echo 0 > "GC_EXIT_CODE.$GC_RUN_NONCE"; : > "GC_SUCCESS.$GC_RUN_NONCE";;
  nomarker) echo 0 > "GC_EXIT_CODE.$GC_RUN_NONCE";;
  link) echo 0 > "GC_EXIT_CODE.$GC_RUN_NONCE"; : > "GC_SUCCESS.$GC_RUN_NONCE"; ln -s /etc/hostname leak;;
  *) echo "$FAKE_CASE" > "GC_EXIT_CODE.$GC_RUN_NONCE";;
esac
: > "TP_DONE.$GC_RUN_NONCE"
sleep 3   # a real runner outlives the driver's start handshake, which reads an exited child as a failed start
"""
# the box: every command runs locally, with the box's /root/gc moved into the test's own directory
STUB_BOX = """#!/bin/bash
exec bash -c "${1//\\/root\\/gc/$BOX_ROOT}"
"""


@pytest.fixture
def dry_tool(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "bench" / "gpu-contract").mkdir(parents=True)
    (root / "bench" / "common").mkdir()
    for f in ("gc_drive.sh", "gc_box.py"):
        shutil.copy(GC / f, root / "bench" / "gpu-contract" / f)
    (root / "bench" / "gpu-contract" / "gc_run.sh").write_text(FAKE_RUN)
    shutil.copy(GC.parent / "common" / "drive_tail.sh", root / "bench" / "common" / "drive_tail.sh")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "tool"]):
        subprocess.run(["git", "-C", str(root), *cmd], check=True, env=env, capture_output=True)
    return root


@pytest.mark.parametrize("case,want", [
    ("ok", 0),             # rc 0 with this nonce's marker
    ("nomarker", 24),      # rc 0 without it: a success the box never recorded
    ("13", 13), ("14", 14), ("17", 17), ("18", 18),   # the launcher's host codes pass through
    ("30", 30),            # gc_run.sh's own code: the contract failed
    ("5", 26),             # a code gc_run.sh never exits with
    ("link", 22),          # rule 8: a link the box left behind refuses the fetch
])
def test_the_dry_run_end_to_end_hands_the_launcher_the_box_code(dry_tool, tmp_path, case, want):
    box = tmp_path / "box"
    stub = tmp_path / "box_exec.sh"
    stub.write_text(STUB_BOX)
    stub.chmod(0o755)
    r = _drive(dry_tool, tmp_path, GC_DRY="1", GC_DRIVE_DRYRUN=None, GC_BOX_EXEC=str(stub), GC_POLL_S="1",
               E4B_RENT_RUN_ID="gc-dry-tail-1", E4B_RENT_SSH_HOST=None, E4B_RENT_SSH_PORT=None, E4B_RENT_SSH_OPTS=None,
               BOX_ROOT=str(box), FAKE_CASE=case)
    assert r.returncode == want, r.stdout + r.stderr
    if case == "link":
        assert "symlink in the fetched artifacts" in r.stdout, r.stdout
    else:
        assert (tmp_path / "run" / "gc" / "summary.txt").is_file()          # the receipts were fetched before the verdict
