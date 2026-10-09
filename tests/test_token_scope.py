"""Fake whoami metadata and actual driver staging blocks; no credentials/network."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "bench/common/token_scope.py"
spec = importlib.util.spec_from_file_location("token_scope", PATH)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
Scope = helper.Scope
TOKEN = "FAKE_TOKEN_ONLY_FOR_TESTS"
READ_PERMISSION = "repo.content.read"


def who(role="read", *, global_permissions=None, scoped_permissions=None):
    access = {"role": role, "displayName": "never log this name"}
    if role == "fineGrained":
        access["fineGrained"] = {"global": global_permissions or [], "scoped": [
            {"entity": {"type": "model", "name": "fake/model"}, "permissions": scoped_permissions or []}]}
    return {"auth": {"type": "access_token", "accessToken": access}}


@pytest.mark.parametrize("response", [who(), who("fineGrained", scoped_permissions=[READ_PERMISSION]),
                                      who("fineGrained", global_permissions=["read-gated"], scoped_permissions=["repo.read"]),
                                      who("fineGrained")])
def test_classic_and_fine_grained_read_only_metadata(response):
    assert helper.token_scope(response) is Scope.READ


@pytest.mark.parametrize("permission", ["repo.write", "repo.content.write", "inference.serverless.write", "jobs.read",
                                        "job.write", "endpoint.read", "endpoints.write", "webhook.read", "webhooks.write",
                                        "future.manage", "repo.read.write", "compute.run_job.read", "endpointAccess.read"])
@pytest.mark.parametrize("location", ["global", "scoped"])
def test_any_non_read_job_endpoint_or_webhook_permission_is_unsafe(permission, location):
    response = who("fineGrained", global_permissions=[permission] if location == "global" else [],
                   scoped_permissions=[permission] if location == "scoped" else [READ_PERMISSION])
    assert helper.token_scope(response) is Scope.UNSAFE


@pytest.mark.parametrize("role", ["write", "admin", "contributor"])
def test_classic_unsafe_role(role):
    assert helper.token_scope(who(role)) is Scope.UNSAFE


@pytest.mark.parametrize("response", [None, [], {}, {"auth": []}, {"auth": {"type": "app_token"}}, who("futureRole"),
                                      {"auth": {"type": "access_token", "accessToken": {"role": "fineGrained"}}},
                                      who("fineGrained", scoped_permissions=[{}]),
                                      who("fineGrained", global_permissions=[""])])
def test_unrecognized_metadata_does_not_authorize_staging(response):
    assert helper.token_scope(response) is Scope.UNVERIFIED


def test_explicit_file_token_and_suppressed_library_diagnostics(tmp_path, capsys):
    token_file = tmp_path / "fake-token"
    token_file.write_text(TOKEN + "\n")
    calls = []

    def api(*, token):
        import logging
        calls.append(token)
        print(TOKEN, "repo.write", "private-name")
        logging.critical("scope diagnostic: %s", TOKEN)
        return who()

    assert helper.check_file(token_file, whoami=api) is Scope.READ
    assert calls == [TOKEN]
    assert not capsys.readouterr().out


def test_api_failure_never_logs_exception_token_or_scope(tmp_path, capsys):
    token_file = tmp_path / "fake-token"
    token_file.write_text(TOKEN)

    def failing(*, token):
        raise RuntimeError(token + " repo.write private-name")

    assert helper.check_file(token_file, whoami=failing) is Scope.UNVERIFIED
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_api_deadline_is_not_swallowed_by_generic_http_retry(tmp_path):
    token_file = tmp_path / "fake-token"
    token_file.write_text(TOKEN)

    def retries(*, token):
        try:
            time.sleep(10)
        except Exception:
            return who()  # an ordinary TimeoutError would be swallowed here

    start = time.monotonic()
    assert helper.check_file(token_file, whoami=retries, timeout_s=1) is Scope.UNVERIFIED
    assert time.monotonic() - start < 3


@pytest.mark.parametrize("role,code,message", [("read", 0, "token read-only: yes"),
                                              ("write", 78, "token read-only: no; refusing token staging"),
                                              ("futureRole", 3, "token unverified, staging none")])
def test_actual_cli_outputs_only_decision(tmp_path, role, code, message):
    token_file = tmp_path / "fake-token"
    token_file.write_text(TOKEN)
    package = tmp_path / "fake-api"
    package.mkdir()
    (package / "huggingface_hub.py").write_text(
        "class HfApi:\n    def __init__(self, *, endpoint):\n        assert endpoint == 'https://huggingface.co'\n    def whoami(self, *, token):\n"
        f"        assert token == {TOKEN!r}\n        print(token, 'private-scope')\n"
        f"        return {who(role)!r}\n")
    result = subprocess.run(["python3", str(PATH), "--token-file", str(token_file)], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(package)}, timeout=5)
    assert result.returncode == code and result.stdout.strip() == message and result.stderr == ""
    assert TOKEN not in result.stdout and "private-scope" not in result.stdout


@pytest.mark.parametrize("folder,script", [("tc1", "tc1"), ("p127", "p127"), ("fam", "fam")])
@pytest.mark.parametrize("role,exit_code,staged", [("read", 0, True), ("fineGrained", 0, True), ("write", 78, False), ("futureRole", 0, False),
                                               ("NETWORK_FAIL", 0, False)])
def test_actual_driver_stages_only_verified_read_token(tmp_path, folder, script, role, exit_code, staged):
    text = (ROOT / f"bench/{folder}/{script}_drive.sh").read_text()
    start = text.index('if [ -s "$HF_TOKEN_FILE" ]; then')
    end = text.index('\nLANE_STARTED_AT=', start)
    block = text[start:end]
    token_file = tmp_path / "fake-token"
    token_file.write_text(TOKEN)
    package = tmp_path / "fake-api"
    package.mkdir()
    (package / "huggingface_hub.py").write_text(
        "class HfApi:\n    def __init__(self, *, endpoint):\n        assert endpoint == 'https://huggingface.co'\n    def whoami(self, *, token):\n"
        + ("        raise RuntimeError(token + ' private-scope')\n" if role == "NETWORK_FAIL" else
           f"        return {who(role, scoped_permissions=[READ_PERMISSION])!r}\n"))
    calls = tmp_path / "calls"
    setup = f'''REPO="{ROOT}"; HF_TOKEN_FILE="{token_file}"; PASS="KNOWN=1"
say(){{ printf '%s\\n' "$*"; }}
fake_scp(){{ printf 'scp\\n' >> '{calls}'; }}
fake_ssh(){{ printf 'ssh\\n' >> '{calls}'; }}
SCP=fake_scp; SSH=fake_ssh; HOST=fake
'''
    result = subprocess.run(["bash", "-uc", setup + block + '\nprintf "CONTINUED:%s\\n" "$PASS"'],
                            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(package)}, timeout=5)
    assert result.returncode == exit_code, (result.stdout, result.stderr)
    assert calls.exists() is staged
    if staged:
        assert calls.read_text().splitlines() == ["scp", "ssh"]
        assert "HF_HUB_DISABLE_IMPLICIT_TOKEN" not in result.stdout
    elif role != "write":
        assert "HF_HUB_DISABLE_IMPLICIT_TOKEN=1" in result.stdout and "CONTINUED:" in result.stdout
    else:
        assert "CONTINUED:" not in result.stdout
    assert TOKEN not in result.stdout + result.stderr


def test_locality_never_stages_a_token():
    text = (ROOT / "bench/locality-1469/locality_drive.sh").read_text()
    assert "HF_TOKEN_FILE" not in text and "token_scope.py" not in text
    assert "unknown_probes" in text
