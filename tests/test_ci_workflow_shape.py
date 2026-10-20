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


def _release_job():
    return _load("release.yml")["jobs"]["pypi-publish"]


def _step(job, prefix):
    return next(st for st in job["steps"] if st.get("name", "").startswith(prefix))


def test_the_release_refuses_to_publish_without_green_ci_at_the_tag():
    """CI collapses pending main runs, so a tagged commit may have no run of its own: the publish job checks first."""
    job = _release_job()
    assert job["permissions"] == {"id-token": "write", "actions": "read"}
    assert "if" not in job and "continue-on-error" not in job
    names = [st.get("name", st.get("uses", "")) for st in job["steps"]]
    gate = next(i for i, n in enumerate(names) if n.startswith("CI is green at the tag's commit"))
    publish = next(i for i, n in enumerate(names) if n.startswith("Publish to PyPI"))
    assert gate < publish
    g, pub = job["steps"][gate], job["steps"][publish]
    assert "if" not in g and "continue-on-error" not in g, "a gate that can be skipped or ignored gates nothing"
    assert "if" not in pub and "continue-on-error" not in pub
    run = g["run"]
    assert "actions/workflows/ci.yml/runs?head_sha=${GITHUB_SHA}&status=success" in run
    # a pull_request run's head_sha is the PR head, but it tested the merge ref: only push / dispatch runs count
    assert 'select(.event == "push" or .event == "workflow_dispatch")' in run and "total_count" not in run


FAKE_GH = """#!/usr/bin/env python3
# A stand-in for `gh api URL --jq FILTER`: it applies only the release gate's own filter, so a changed filter
# (or any other call) fails loudly instead of being guessed at.
import json, os, sys
if os.environ.get("FAKE_GH_FAIL"):
    sys.exit(1)
args = sys.argv[1:]
want = '[.workflow_runs[] | select(.event == "push" or .event == "workflow_dispatch")] | length'
if args[:1] != ["api"] or "--jq" not in args or args[args.index("--jq") + 1] != want:
    sys.exit(97)
runs = json.loads(os.environ["FAKE_GH_RUNS"])
print(sum(1 for r in runs if r["event"] in ("push", "workflow_dispatch")))
"""


@pytest.mark.parametrize("runs,fail,ok", [
    ([], False, False),                                                   # no successful run at the commit
    ([{"event": "pull_request"}, {"event": "pull_request"}], False, False),  # it tested the merge ref, not this commit
    ([{"event": "push"}], False, True),
    ([{"event": "workflow_dispatch"}], False, True),
    ([], True, False),                                                    # the API call itself fails: refuse, never assume
])
def test_the_gate_script_itself_refuses_unless_a_push_or_dispatch_run_is_green(tmp_path, runs, fail, ok):
    """Run the gate step's own run: block, as written in release.yml, against a fake gh."""
    import json
    import os
    import shutil
    import subprocess
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    fake = tmp_path / "gh"
    fake.write_text(FAKE_GH, encoding="utf-8")
    fake.chmod(0o755)
    script = tmp_path / "gate.sh"
    script.write_text(_step(_release_job(), "CI is green at the tag's commit")["run"], encoding="utf-8")
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}", "GITHUB_REPOSITORY": "o/r",
           "GITHUB_SHA": "a" * 40, "GITHUB_REF_NAME": "v0.0.0", "FAKE_GH_RUNS": json.dumps(runs)}
    if fail:
        env["FAKE_GH_FAIL"] = "1"
    out = subprocess.run([bash, str(script)], env=env, capture_output=True, text=True, check=False)
    assert (out.returncode == 0) == ok, (out.returncode, out.stdout, out.stderr)
    if not ok:
        assert out.returncode != 97, "the gate's gh call no longer matches the reviewed filter"
