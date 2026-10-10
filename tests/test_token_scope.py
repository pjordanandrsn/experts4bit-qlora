"""Fake whoami metadata and actual driver staging blocks; no credentials/network.

The helper is standard library only (one GET of whoami-v2). Subprocess tests put a sitecustomize.py on PYTHONPATH that
replaces urllib.request.build_opener with a fake asserting the exact URL, the UNREDIRECTED Bearer header and the
deadline, so nothing reaches the network. Two in-process tests use real urllib against local servers."""
from __future__ import annotations

import http.server
import importlib.util
import json
import os
import threading
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


def fake_net(tmp_path, *, response=None, mode="ok", status=200, body=None, block_hub=False):
    """A PYTHONPATH dir whose sitecustomize.py fakes urllib.request.build_opener (and, with block_hub, makes huggingface_hub
    unimportable). mode "ok" answers with `response` as JSON or a raw `body`; "fail" raises URLError carrying the token."""
    d = tmp_path / "fake-net"
    d.mkdir(exist_ok=True)
    payload = body if body is not None else json.dumps(response)
    src = [
        "import urllib.error, urllib.request",
        f"TOKEN, MODE, STATUS, BODY = {TOKEN!r}, {mode!r}, {status!r}, {payload!r}",
        "class _R:",
        "    status = STATUS",
        "    def read(self, n=-1):",
        "        return BODY.encode()",
        "    def __enter__(self):",
        "        return self",
        "    def __exit__(self, *a):",
        "        return False",
        "class _Opener:",
        "    def open(self, req, timeout=None):",
        "        assert req.full_url == 'https://huggingface.co/api/whoami-v2', req.full_url",
        "        assert req.unredirected_hdrs.get('Authorization') == 'Bearer ' + TOKEN",
        "        assert 'Authorization' not in req.headers, 'the token header must be unredirected'",
        "        assert timeout is not None and timeout <= 20",
        "        if MODE == 'fail':",
        "            raise urllib.error.URLError(TOKEN + ' private-scope')",
        "        return _R()",
        "urllib.request.build_opener = lambda *handlers: _Opener()",
    ]
    (d / "sitecustomize.py").write_text("\n".join(src) + "\n")
    if block_hub:
        (d / "huggingface_hub").mkdir(exist_ok=True)
        (d / "huggingface_hub" / "__init__.py").write_text("raise ImportError('huggingface_hub is not installed here')\n")
    return d

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
    package = fake_net(tmp_path, response=who(role))
    result = subprocess.run(["python3", str(PATH), "--token-file", str(token_file)], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(package)}, timeout=5)
    assert result.returncode == code and result.stdout.strip() == message and result.stderr == ""
    assert TOKEN not in result.stdout and "private-scope" not in result.stdout


def test_no_huggingface_hub_is_needed(tmp_path):
    """The controllers' python has no huggingface_hub; the check must still verify (it is standard library only)."""
    token_file = tmp_path / "fake-token"
    token_file.write_text(TOKEN)
    package = fake_net(tmp_path, response=who("read"), block_hub=True)
    probe = subprocess.run(["python3", "-c", "import huggingface_hub"], capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": str(package)}, timeout=5)
    assert probe.returncode != 0, "the blocker must make huggingface_hub unimportable"
    result = subprocess.run(["python3", str(PATH), "--token-file", str(token_file)], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(package)}, timeout=5)
    assert result.returncode == 0 and result.stdout.strip() == "token read-only: yes" and result.stderr == ""


@pytest.mark.parametrize("kw", [{"mode": "fail"}, {"status": 401, "body": TOKEN + " private-scope"},
                                {"body": "not json " + TOKEN + " private-scope"}])
def test_network_failure_or_a_bad_answer_is_unverified_and_never_echoed(tmp_path, kw):
    token_file = tmp_path / "fake-token"
    token_file.write_text(TOKEN)
    package = fake_net(tmp_path, **kw)
    result = subprocess.run(["python3", str(PATH), "--token-file", str(token_file)], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(package)}, timeout=5)
    assert result.returncode == 3 and result.stdout.strip() == "token unverified, staging none" and result.stderr == ""
    assert TOKEN not in result.stdout + result.stderr and "private-scope" not in result.stdout + result.stderr


@pytest.mark.parametrize("folder,script", [("tc1", "tc1"), ("p127", "p127"), ("fam", "fam"), ("sd1", "sd1")])
@pytest.mark.parametrize("role,exit_code,staged", [("read", 0, True), ("fineGrained", 0, True), ("write", 78, False), ("futureRole", 0, False),
                                               ("NETWORK_FAIL", 0, False)])
def test_actual_driver_stages_only_verified_read_token(tmp_path, folder, script, role, exit_code, staged):
    text = (ROOT / f"bench/{folder}/{script}_drive.sh").read_text()
    start = text.index('if [ -s "$HF_TOKEN_FILE" ]; then')
    end = text.index('\nLANE_STARTED_AT=', start)
    block = text[start:end]
    token_file = tmp_path / "fake-token"
    token_file.write_text(TOKEN)
    package = (fake_net(tmp_path, mode="fail") if role == "NETWORK_FAIL"
               else fake_net(tmp_path, response=who(role, scoped_permissions=[READ_PERMISSION])))
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


def _server(handler_cls):
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_a_redirect_is_refused_and_never_carries_the_token(tmp_path, monkeypatch):
    """Real urllib, two local servers: A answers 302 to B. The check must come out UNVERIFIED with nothing sent to B."""
    seen = []

    class B(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(dict(self.headers))
            body = json.dumps(who("read")).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    b = _server(B)

    class A(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{b.server_port}/api/whoami-v2")
            self.end_headers()

        def log_message(self, *a):
            pass

    a = _server(A)
    try:
        monkeypatch.setattr(helper, "WHOAMI_URL", f"http://127.0.0.1:{a.server_port}/api/whoami-v2")
        token_file = tmp_path / "fake-token"
        token_file.write_text(TOKEN)
        assert helper.check_file(token_file, timeout_s=5) is Scope.UNVERIFIED
        assert all("Authorization" not in h and TOKEN not in json.dumps(h) for h in seen), seen
        assert seen == [], "the redirect must be refused, not followed"
    finally:
        a.shutdown()
        b.shutdown()


def test_the_token_header_is_unredirected(monkeypatch):
    """urllib copies Request(headers=...) onto a redirected request; an unredirected header it never copies."""
    captured = []

    class Opener:
        def open(self, req, timeout=None):
            captured.append(req)
            raise OSError("stop here")

    monkeypatch.setattr(helper.urllib.request, "build_opener", lambda *handlers: Opener())
    with pytest.raises(OSError):
        helper.whoami_v2(token=TOKEN, timeout_s=5)
    (req,) = captured
    assert req.unredirected_hdrs.get("Authorization") == "Bearer " + TOKEN
    assert "Authorization" not in req.headers
