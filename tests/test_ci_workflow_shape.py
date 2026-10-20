"""The CI workflow's shape, pinned: no required job can SKIP on a pull_request (GitHub counts a skipped required check
as passing), the concurrency block is the reviewed one, and the guards' triggers still cover every path to main."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml  # a hard import: a skipped module would pass every assertion below by never running them

WF = Path(__file__).resolve().parents[1] / ".github" / "workflows"
REQUIRED_CI_JOBS = ("lint-and-test", "wheel-smoke", "discoverability")
GATE_IF = ("${{ github.event_name == 'pull_request' && github.event.action != 'ready_for_review' && "
           "!contains(github.event.pull_request.labels.*.name, 'ready-to-merge') }}")
GROUP = ("${{ github.event_name == 'push' && 'ci-main' || github.event_name == 'pull_request' && "
         "format('ci-pr-{0}', github.event.pull_request.number) || format('ci-run-{0}', github.run_id) }}")


def _load(name: str) -> dict:
    d = yaml.safe_load((WF / name).read_text(encoding="utf-8"))
    d["on"] = d.pop(True, d.get("on"))   # PyYAML reads the bare key `on` as the boolean True
    return d


@pytest.mark.parametrize("job", REQUIRED_CI_JOBS)
def test_a_required_ci_job_cannot_skip(job):
    """A skipped required check counts as passing, so no job-level if:, and the first step is the failing gate. The
    PR-cancel arm of the concurrency block is safe only while this holds."""
    j = _load("ci.yml")["jobs"][job]
    assert "if" not in j, f"{job} has a job-level if: again; a skip would pass the required check"
    assert "needs" not in j, f"{job} needs another job: if that job fails or skips, {job} SKIPS, which passes"
    first = j["steps"][0]
    assert first.get("if") == GATE_IF and "exit 1" in first.get("run", "")


def test_the_concurrency_block_is_the_reviewed_one():
    c = _load("ci.yml")["concurrency"]
    assert c == {"group": GROUP, "cancel-in-progress": "${{ github.event_name == 'pull_request' }}"}


def test_ci_still_runs_on_main_pushes_labels_and_dispatch():
    on = _load("ci.yml")["on"]
    assert on["push"] == {"branches": ["main"]}
    assert set(on["pull_request"]["types"]) == {"labeled", "ready_for_review"}
    assert "workflow_dispatch" in on


def test_the_guards_still_scan_every_path_to_main():
    """Every pull_request head (default types) and every push to main; private-marker-guard also every branch push."""
    conflict = _load("conflict-marker-guard.yml")["on"]
    assert conflict["push"] == {"branches": ["main"]} and conflict["pull_request"] in (None, {})
    assert _load("private-marker-guard.yml")["on"] == ["push", "pull_request"]
