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
