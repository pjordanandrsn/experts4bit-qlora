# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Vast.ai adapter for the governed launcher (e4b#455) — armed only with ``E4B_RENT_LIVE=1`` and a key file.

What this module promises (every line is a rule that cost money before it was a rule):

* The API key is read from a mode-600 file by SHAPE (``[0-9a-f]{40,}``), never from argv or the
  environment, never echoed, never written into a receipt or a log. No key → a named refusal, never a fallback.
* ``list_ids`` NEVER returns an empty set on an error. Vast's v0 ``/instances/`` is deprecated and answers
  HTTP 200 ``{"success": false, "error": "deprecated_endpoint"}``; a parser that reads that as "zero instances"
  is the false-zero failure of 2026-08-25 / 2026-08-29 (two live boxes billed for days). Only v1's
  ``{"instances": [...]}`` counts; anything else raises :class:`BackendUnavailable`.
* ``destroy`` is ``DELETE /api/v0/instances/{id}/`` with a ``{}`` body (v1 answers 404); ``actual_status ==
  "exited"`` still bills — only destroy frees. Teardown proof is the id ABSENT from a fresh authenticated
  v1 listing, which the launcher and the guard already check.
* Offer ids rotate (the same machine came back as two ids minutes apart): search by spec at launch, record
  the offer id, the ``machine_id`` the console shows and the contract ``id`` the API returns — they never match.
* The offer payload has no ``verified`` key — it is ``verification``; the server-side filter
  ``{"verified": {"eq": true}}`` works. Units: ``cpu_ram`` MB, ``disk_space`` GB, ``dph_total`` $/h.
* Pre-flight before any command: ssh authenticates within a bounded window, the image is not stuck
  ``loading``, disk/RAM as ordered, ≥ 40 MB/s to the box (four of four boxes failed one of these on
  2026-08-27/28). A failed pre-flight raises :class:`PreflightFailed`; the launcher destroys and writes an
  ``invalid`` receipt with the reason and the cost to that point.
* Nothing here creates an instance unless :meth:`VastProvider.launch` is called by the launcher after the
  policy gate; tests use :class:`FakeTransport` with recorded response shapes and never touch the network.
"""
from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Protocol

API_BASE = "https://console.vast.ai/api"
KEY_RE = re.compile(r"^[0-9a-f]{40,}$")
DEFAULT_KEY_PATH = Path.home() / ".vast" / "secrets.env"
KIND = "vast:verified-secure"


class BackendUnavailable(RuntimeError):
    """The provider did not answer with the shape the call requires. Never interpreted as 'nothing there'."""


class VastRefused(RuntimeError):
    """Refusal before any instance exists: no key, bad mode, auth failure, no offer in the allowed class."""


class PreflightFailed(RuntimeError):
    """The instance exists but is unusable; the caller destroys it and writes an ``invalid`` receipt."""


# ---------------------------------------------------------------------------------------------- key
def load_api_key(path: Path | str = DEFAULT_KEY_PATH) -> str:
    """The key by shape from a mode-600 file. The returned value must never be printed or stored."""
    p = Path(path)
    if not p.is_file():
        raise VastRefused(f"no Vast key file at {p} (expected VAST_API_KEY=<hex> in a mode-600 file); refusing")
    mode = stat.S_IMODE(p.stat().st_mode)
    if mode & 0o077:
        raise VastRefused(f"refusing to read {p}: mode {mode:o} is not 600")
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith("VAST_API_KEY="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if KEY_RE.match(value):
                return value
            raise VastRefused(f"VAST_API_KEY in {p} does not have the shape of a Vast key ({len(value)} chars); refusing")
    raise VastRefused(f"no VAST_API_KEY line in {p}; refusing")


# ---------------------------------------------------------------------------------------------- transport
class Transport(Protocol):
    def request(self, method: str, path: str, *, query: dict[str, str] | None = None,
                body: Any = None, timeout: float = 30.0) -> tuple[int, Any]: ...


class UrllibTransport:
    """Real HTTP. Returns (status, parsed JSON | text). Network failure → BackendUnavailable. The key is a
    header, never part of a URL, an error message or a log line."""

    def __init__(self, key: str, base: str = API_BASE):
        self._key = key
        self.base = base.rstrip("/")

    def request(self, method: str, path: str, *, query: dict[str, str] | None = None,
                body: Any = None, timeout: float = 30.0) -> tuple[int, Any]:
        url = self.base + path + (("?" + urllib.parse.urlencode(query)) if query else "")
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Authorization": f"Bearer {self._key}", "Accept": "application/json",
                                              **({"Content-Type": "application/json"} if data is not None else {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, _parse(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, _parse(e.read())
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise BackendUnavailable(f"network error talking to Vast ({method} {path}): {getattr(e, 'reason', e)}") from e


def _parse(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        return text


class FakeTransport:
    """Recorded response shapes for tests: ``routes[(method, path)] = [(status, body), ...]`` consumed in
    order (the last one repeats). Records every call so a test can assert what was sent."""

    def __init__(self, routes: dict[tuple[str, str], list[tuple[int, Any]]]):
        self.routes = {k: list(v) for k, v in routes.items()}
        self.calls: list[tuple[str, str, dict | None, Any]] = []

    def request(self, method: str, path: str, *, query: dict[str, str] | None = None,
                body: Any = None, timeout: float = 30.0) -> tuple[int, Any]:
        self.calls.append((method, path, query, body))
        key = (method, path)
        if key not in self.routes:
            raise BackendUnavailable(f"fake transport has no route for {method} {path}")
        seq = self.routes[key]
        status, resp = seq[0] if len(seq) == 1 else seq.pop(0)
        return status, json.loads(json.dumps(resp))


# ---------------------------------------------------------------------------------------------- provider
def offer_filter(gpu: str, *, min_disk_gb: int, min_ram_gb: int) -> dict[str, Any]:
    """The server-side search filter for the policy's class; recorded verbatim in the receipt."""
    return {
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "external": {"eq": False},
        "gpu_name": {"eq": gpu},
        "num_gpus": {"eq": 1},
        "disk_space": {"gte": min_disk_gb},
        "cpu_ram": {"gte": min_ram_gb * 1024},  # MB
        "type": "on-demand",
        "order": [["dph_total", "asc"]],
    }


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class VastProvider:
    """The interface the launcher and the guard use (as ``FakeProvider``): ``launch`` / ``list_ids`` /
    ``destroy``; plus ``launch_facts``, ``preflight`` and ``actual_cost`` which the launcher uses when present."""

    kind = KIND

    def __init__(self, transport: Transport, *, run_label: str, min_disk_gb: int = 320, min_ram_gb: int = 98,
                 ssh_pubkey: str | None = None,
                 ssh_runner: Callable[[str, int, str, float], tuple[int, str]] | None = None,
                 bandwidth_probe: Callable[[str, int], float] | None = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep):
        self.t = transport
        self.run_label = run_label
        self.min_disk_gb = min_disk_gb
        self.min_ram_gb = min_ram_gb
        self.ssh_pubkey = ssh_pubkey
        self._ssh = ssh_runner or _ssh_run
        self._bandwidth = bandwidth_probe or _bandwidth_over_ssh
        self._clock = clock
        self._sleep = sleep
        self._facts: dict[str, str] = {}
        self._dph: float | None = None

    # ---- auth
    def auth_probe(self) -> dict[str, Any]:
        status, body = self.t.request("GET", "/v0/users/current/")
        if status != 200 or not isinstance(body, dict) or "id" not in body:
            raise VastRefused(f"Vast auth probe failed: HTTP {status}, body {_shape(body)} — the key was not accepted")
        return {"user_id": str(body.get("id")), "email": str(body.get("email", "UNKNOWN")),
                "balance": str(body.get("balance", "UNKNOWN")), "credit": str(body.get("credit", "UNKNOWN"))}

    # ---- offers
    def search_offers(self, gpu: str) -> list[dict[str, Any]]:
        q = offer_filter(gpu, min_disk_gb=self.min_disk_gb, min_ram_gb=self.min_ram_gb)
        status, body = self.t.request("GET", "/v0/bundles/", query={"q": json.dumps(q, sort_keys=True)})
        if status != 200 or not isinstance(body, dict) or not isinstance(body.get("offers"), list):
            raise BackendUnavailable(f"offer search did not answer with offers: HTTP {status}, body {_shape(body)}")
        # The payload's key is `verification`, not `verified` (2026-07-30 trap) — check it client-side too.
        offers = [o for o in body["offers"] if o.get("verification") == "verified" and o.get("rentable", True)]
        offers.sort(key=lambda o: float(o.get("dph_total", 1e9)))
        return offers

    # ---- lifecycle
    def launch(self, *, gpu: str, wallclock_h: float, image: str) -> str:
        offers = self.search_offers(gpu)
        if not offers:
            raise VastRefused(f"no verified rentable {gpu} offer with ≥{self.min_disk_gb} GB disk and ≥{self.min_ram_gb} GB RAM right now; refusing")
        offer = offers[0]
        body = {"client_id": "me", "image": image, "disk": self.min_disk_gb, "label": self.run_label,
                "runtype": "ssh", "onstart": None}
        status, resp = self.t.request("PUT", f"/v0/asks/{offer['id']}/", body=body)
        if status != 200 or not isinstance(resp, dict) or not resp.get("success") or "new_contract" not in resp:
            raise BackendUnavailable(f"create did not succeed: HTTP {status}, body {_shape(resp)}")
        iid = str(resp["new_contract"])
        self._dph = float(offer.get("dph_total")) if offer.get("dph_total") is not None else None
        q = offer_filter(gpu, min_disk_gb=self.min_disk_gb, min_ram_gb=self.min_ram_gb)
        self._facts = {
            "vast_offer_id": str(offer.get("id")),
            "vast_machine_id": str(offer.get("machine_id", "UNKNOWN")),
            "vast_contract_id": iid,
            "vast_dph_total": str(offer.get("dph_total", "UNKNOWN")),
            "vast_gpu_name": str(offer.get("gpu_name", "UNKNOWN")),
            "vast_disk_space_gb": str(offer.get("disk_space", "UNKNOWN")),
            "vast_cpu_ram_mb": str(offer.get("cpu_ram", "UNKNOWN")),
            "vast_verification": str(offer.get("verification", "UNKNOWN")),
            "vast_search_filter": json.dumps(q, sort_keys=True),
            "vast_image": image,
            "vast_created_at": _utc(),
        }
        return iid

    def launch_facts(self) -> dict[str, str]:
        return dict(self._facts)

    def list_ids(self) -> set[str]:
        """Live instance ids from v1. An error body, a deprecated-endpoint body or a missing `instances` key
        raises — it is never an empty set."""
        status, body = self.t.request("GET", "/v1/instances/")
        if status != 200 or not isinstance(body, dict):
            raise BackendUnavailable(f"instance list unavailable: HTTP {status}, body {_shape(body)}")
        if body.get("success") is False or "error" in body:
            raise BackendUnavailable(f"instance list answered an error, not a list: {_shape(body)} (false-zero refused)")
        inst = body.get("instances")
        if not isinstance(inst, list):
            raise BackendUnavailable(f"instance list has no `instances` array: {_shape(body)} (false-zero refused)")
        return {str(i.get("id")) for i in inst if isinstance(i, dict) and i.get("id") is not None}

    def instance(self, instance_id: str) -> dict[str, Any]:
        status, body = self.t.request("GET", f"/v0/instances/{instance_id}/")
        if status != 200 or not isinstance(body, dict):
            raise BackendUnavailable(f"instance {instance_id} unreadable: HTTP {status}, body {_shape(body)}")
        rec = body.get("instances", body)
        if isinstance(rec, list):
            rec = rec[0] if rec else {}
        if not isinstance(rec, dict) or str(rec.get("id")) != str(instance_id):
            raise BackendUnavailable(f"instance {instance_id} record has the wrong shape: {_shape(body)}")
        return rec

    def destroy(self, instance_id: str) -> dict[str, Any]:
        status, body = self.t.request("DELETE", f"/v0/instances/{instance_id}/", body={})
        ok = status == 200 and isinstance(body, dict) and body.get("success", True) is not False
        if status == 404:
            # v1 404s on destroy and a wrong id 404s too: not proof of anything — the caller's fresh v1 listing is.
            return {"method": "vast-destroy", "instance_id": instance_id, "http": 404, "note": "404 on v0 destroy — absence proven only by the listing", "at": _utc()}
        if not ok:
            raise BackendUnavailable(f"destroy {instance_id} did not succeed: HTTP {status}, body {_shape(body)}")
        return {"method": "vast-destroy", "instance_id": instance_id, "http": status, "at": _utc()}

    def attach_ssh_key(self, instance_id: str, pubkey: str) -> None:
        status, body = self.t.request("POST", f"/v0/instances/{instance_id}/ssh/", body={"ssh_key": pubkey})
        if status != 200 or (isinstance(body, dict) and body.get("success") is False):
            raise BackendUnavailable(f"attaching the ssh key to {instance_id} failed: HTTP {status}, body {_shape(body)}")

    # ---- pre-flight
    def preflight(self, instance_id: str, *, timeout_s: float = 600.0, ssh_timeout_s: float = 30.0,
                  min_mb_per_s: float = 40.0, poll_s: float = 10.0) -> dict[str, str]:
        """Usable, or PreflightFailed with the reason. Bounded: the instance must be `running` (not stuck
        `loading`) within timeout_s; disk/RAM as ordered; ssh authenticates within ssh_timeout_s; ≥ min MB/s."""
        t0 = self._clock()
        rec: dict[str, Any] = {}
        while True:
            rec = self.instance(instance_id)
            st = str(rec.get("actual_status", "UNKNOWN"))
            if st == "running":
                break
            if self._clock() - t0 > timeout_s:
                raise PreflightFailed(f"instance {instance_id} is `{st}` after {int(timeout_s)} s (stuck loading)")
            self._sleep(poll_s)
        disk = float(rec.get("disk_space") or 0)
        ram_mb = float(rec.get("cpu_ram") or 0)
        if disk < self.min_disk_gb:
            raise PreflightFailed(f"disk {disk:.0f} GB < ordered {self.min_disk_gb} GB")
        if ram_mb < self.min_ram_gb * 1024:
            raise PreflightFailed(f"host RAM {ram_mb / 1024:.0f} GB < ordered {self.min_ram_gb} GB")
        host, port = rec.get("ssh_host"), rec.get("ssh_port")
        if not host or not port:
            raise PreflightFailed(f"instance {instance_id} has no ssh endpoint yet (ssh_host={host!r}, ssh_port={port!r})")
        if self.ssh_pubkey:
            self.attach_ssh_key(instance_id, self.ssh_pubkey)
        rc, out = self._ssh(str(host), int(port), "true", ssh_timeout_s)
        if rc != 0:
            raise PreflightFailed(f"ssh to {host}:{port} did not authenticate within {int(ssh_timeout_s)} s (rc {rc}: {out.strip()[:120]})")
        mbps = float(self._bandwidth(str(host), int(port)))
        if mbps < min_mb_per_s:
            raise PreflightFailed(f"download bandwidth {mbps:.1f} MB/s < {min_mb_per_s:.0f} MB/s on {host}:{port}")
        return {"vast_preflight": "ok", "vast_ssh": f"{host}:{port}", "vast_actual_status": st,
                "vast_disk_space_gb": f"{disk:.0f}", "vast_cpu_ram_mb": f"{ram_mb:.0f}", "vast_bandwidth_mb_s": f"{mbps:.1f}",
                "vast_preflight_seconds": f"{self._clock() - t0:.0f}"}

    # ---- cost
    def actual_cost(self, instance_id: str, runtime_s: float) -> tuple[float, str]:
        """dph_total × measured runtime, from the offer the provider accepted. The provider's own billing
        endpoint is UNKNOWN to this adapter; the method is recorded so nobody reads this as an invoice."""
        if self._dph is None:
            return 0.0, "UNKNOWN: no dph_total recorded for this instance"
        return round(self._dph * runtime_s / 3600.0, 4), "dph_total × measured runtime (provider billing endpoint UNKNOWN; not an invoice)"


def _shape(body: Any) -> str:
    """A description of a response body for error text — keys and types only, never values that could be a secret."""
    if isinstance(body, dict):
        return "{" + ", ".join(f"{k}: {type(v).__name__}" + (f"={v!r}" if k in ("success", "error", "msg") else "") for k, v in list(body.items())[:8]) + "}"
    if isinstance(body, list):
        return f"list[{len(body)}]"
    return f"{type(body).__name__}({str(body)[:60]!r})"


def _ssh_run(host: str, port: int, command: str, timeout_s: float) -> tuple[int, str]:
    try:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                              "-o", f"ConnectTimeout={int(timeout_s)}", "-p", str(port), f"root@{host}", command],
                             capture_output=True, text=True, timeout=timeout_s + 5)
        return out.returncode, (out.stdout + out.stderr)
    except subprocess.TimeoutExpired:
        return 124, f"ssh timed out after {timeout_s} s"


def _bandwidth_over_ssh(host: str, port: int) -> float:
    """MB/s of a 100 MB download measured ON the box (what the run will see), via curl over ssh."""
    rc, out = _ssh_run(host, port, "curl -s -o /dev/null -w '%{speed_download}' --max-time 60 https://speed.cloudflare.com/__down?bytes=100000000", 90)
    if rc != 0:
        return 0.0
    try:
        return float(out.strip().split()[-1]) / 1e6
    except (ValueError, IndexError):
        return 0.0


def _under_test() -> bool:
    """True inside pytest / unittest. Tests NEVER go live: on 2026-09-06 12:38Z a test that meant to exercise the
    "no key file" refusal read the real key (the key path was bound as a default argument, so a monkeypatched
    module constant did nothing), rented an RTX 5090 for ten minutes and paid for it. Belt and braces below."""
    argv0 = (sys.argv[0] if sys.argv else "").rsplit("/", 1)[-1]
    return (bool(os.environ.get("PYTEST_CURRENT_TEST")) or "pytest" in sys.modules or "_pytest" in sys.modules
            or argv0 in ("pytest", "py.test") or any(a in ("pytest", "unittest") for a in sys.argv[:2]))


def provider_from_env(*, run_label: str, key_path: Path | str | None = None, **kw: Any) -> VastProvider:
    """The launcher's factory: armed only by E4B_RENT_LIVE=1 and a readable key file; the auth probe runs first.
    Refuses under a test runner whatever the environment says — a live adapter in a test is a rented box."""
    if _under_test():
        raise VastRefused("live provider refused: this process is a test runner (tests never rent; use FakeTransport)")
    if os.environ.get("E4B_RENT_LIVE") != "1":
        raise VastRefused("live provider vast:verified-secure is not armed in this process: set E4B_RENT_LIVE=1 (a fake run is --dry-run)")
    key = load_api_key(DEFAULT_KEY_PATH if key_path is None else key_path)  # read at call time, never bound at import
    prov = VastProvider(UrllibTransport(key), run_label=run_label, **kw)
    prov.auth_probe()
    return prov
