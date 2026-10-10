"""Controller-only token scope check; never echo credentials or permission data.

Standard library only (e4b#1313). The controllers' own python has no huggingface_hub: on the Mac mini, Homebrew
python 3.14.8 raised ModuleNotFoundError. The guarded block turned that into UNVERIFIED, so every lane launched from
there staged no token. The check is one GET of the Hub's whoami-v2 endpoint with the token as a Bearer header -- what
HfApi.whoami does -- under the same 20 s deadline, the same metadata rules and the same refusal / unverified semantics."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
from pathlib import Path
import signal
import urllib.request
from enum import Enum

WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
MAX_BODY = 1 << 20


class Scope(Enum):
    READ = 0
    UNSAFE = 78
    UNVERIFIED = 3


def token_scope(who: object) -> Scope:
    """Interpret whoami access-token metadata; other auth shapes stay unverified."""
    if not isinstance(who, dict):
        return Scope.UNVERIFIED
    auth = who.get("auth")
    if not isinstance(auth, dict) or auth.get("type") != "access_token":
        return Scope.UNVERIFIED
    access = auth.get("accessToken")
    if not isinstance(access, dict):
        return Scope.UNVERIFIED
    role = access.get("role")
    if role == "read":
        return Scope.READ
    if role in ("write", "admin", "contributor"):
        return Scope.UNSAFE
    if role != "fineGrained":
        return Scope.UNVERIFIED
    fine = access.get("fineGrained")
    if not isinstance(fine, dict) or set(fine) != {"global", "scoped"}:
        return Scope.UNVERIFIED
    global_permissions, scoped = fine["global"], fine["scoped"]
    if not isinstance(global_permissions, list) or not isinstance(scoped, list):
        return Scope.UNVERIFIED
    permissions = list(global_permissions)
    for item in scoped:
        if not isinstance(item, dict) or not isinstance(item.get("entity"), dict) or not isinstance(item.get("permissions"), list):
            return Scope.UNVERIFIED
        permissions.extend(item["permissions"])
    if any(not isinstance(p, str) or not p for p in permissions):
        return Scope.UNVERIFIED
    for permission in permissions:
        if any(capability in permission.lower() for capability in ("job", "endpoint", "webhook")):
            return Scope.UNSAFE
        # Fine-grained read permissions use .read; read-gated is download-only.
        if not (permission == "read-gated" or permission == "read" or permission.endswith(".read")):
            return Scope.UNSAFE
    return Scope.READ


class _VerificationDeadline(BaseException):
    """Avoid being swallowed by HTTP retry handlers catching Exception."""


def whoami_v2(*, token: str, timeout_s: float = 20) -> object:
    """GET the Hub's whoami-v2 with the token as a Bearer header; the parsed JSON, or an exception. Never a body in an
    error: a non-200 status or a body that is not JSON raises without its content."""
    req = urllib.request.Request(WHOAMI_URL, headers={"Authorization": f"Bearer {token}", "Accept": "application/json",
                                                      "User-Agent": "e4b-token-scope/1"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # looked up at call time: tests patch it
        if getattr(resp, "status", 200) != 200:
            raise RuntimeError("whoami-v2: non-200 status")
        body = resp.read(MAX_BODY + 1)
    if len(body) > MAX_BODY:
        raise RuntimeError("whoami-v2: oversized response")
    try:
        return json.loads(body)
    except ValueError:
        raise RuntimeError("whoami-v2: not JSON") from None


def check_file(path: Path, *, whoami=None, timeout_s=20) -> Scope:
    """Bound API verification and suppress library diagnostics, including errors."""
    old_level = logging.root.manager.disable
    old_handler = signal.getsignal(signal.SIGALRM)
    old_alarm = signal.alarm(0)

    def expired(_signum, _frame):
        raise _VerificationDeadline

    try:
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(timeout_s)
        logging.disable(logging.CRITICAL)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            # Never fall back to the controller's cached token or environment token.
            token = path.read_text().strip()
            if not token:
                return Scope.UNVERIFIED
            if whoami is None:
                return token_scope(whoami_v2(token=token, timeout_s=timeout_s))
            return token_scope(whoami(token=token))
    except (Exception, _VerificationDeadline):
        return Scope.UNVERIFIED  # no exception text: it could include credentials
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_alarm:
            signal.alarm(old_alarm)
        logging.disable(old_level)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args(argv)
    scope = check_file(args.token_file)
    if scope is Scope.READ:
        print("token read-only: yes")
    elif scope is Scope.UNSAFE:
        print("token read-only: no; refusing token staging")
    else:
        print("token unverified, staging none")
    return scope.value


if __name__ == "__main__":
    raise SystemExit(main())
