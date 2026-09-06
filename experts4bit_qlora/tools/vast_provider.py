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
import shlex
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


class OrphanSwept(BackendUnavailable):
    """A create answered a shape this code does not understand, and the sweep found instance(s) carrying this
    run's label, destroyed them and saw them absent. Money was spent: the caller's receipt is ALARM, complete."""

    def __init__(self, msg: str, swept: list[str]):
        super().__init__(msg)
        self.swept = swept


class PossibleOrphan(BackendUnavailable):
    """A create answered a shape this code does not understand and the sweep could NOT prove nothing is running
    under this run's label (the listing failed, a destroy failed, or an id stayed present). Never a clean refusal:
    the caller's receipt is ALARM, complete=False, and a human checks the console for the label."""


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
        # Keep injected probes as the small float-returning test seam.  The live
        # probe uses the evidence-returning path so a conservative 0.0 refusal
        # says which endpoint/attempt failed and how; R1 attempt 3 collapsed four
        # exhausted attempts into an otherwise uninterpretable 0.0 MB/s.
        self._bandwidth = bandwidth_probe
        self._clock = clock
        self._sleep = sleep
        self._facts: dict[str, str] = {}
        self._dph: float | None = None

    # ---- auth
    def auth_probe(self) -> dict[str, Any]:
        status, body = self.t.request("GET", "/v0/users/current/")
        if status != 200 or not isinstance(body, dict) or "id" not in body:
            raise VastRefused(f"Vast auth probe failed: HTTP {status}, body {_shape(body)} — the key was not accepted")
        # No `email`: it is personal data one `environment[...] = str(v)` away from a receipt in the corpus (CEO read, #460).
        return {"user_id": str(body.get("id")),
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
    def launch(self, *, gpu: str, wallclock_h: float, image: str, max_dph: float | None = None) -> str:
        offers = self.search_offers(gpu)
        if not offers:
            raise VastRefused(f"no verified rentable {gpu} offer with ≥{self.min_disk_gb} GB disk and ≥{self.min_ram_gb} GB RAM right now; refusing")
        offer = offers[0]
        # e4b#464 (the money path): the approval line is `--usd-per-hour × cap`; an offer priced above the declared rate would make
        # the receipt's estimate a lie. The cheapest verified offer must fit under the declared ceiling or nothing is created.
        dph = offer.get("dph_total")
        if max_dph is not None and (dph is None or float(dph) > float(max_dph)):
            raise VastRefused(f"cheapest verified {gpu} offer is ${dph}/h, above the declared --usd-per-hour ${max_dph}/h the approval was given for; raise the ceiling (a new approval line) or wait — nothing created")
        body = {"client_id": "me", "image": image, "disk": self.min_disk_gb, "label": self.run_label,
                "runtype": "ssh", "onstart": None}
        status, resp = self.t.request("PUT", f"/v0/asks/{offer['id']}/", body=body)
        if status == 200 and isinstance(resp, dict) and resp.get("success") is False:
            # Understood: the provider refused and nothing was created (insufficient_credit, offer gone, …).
            raise BackendUnavailable(f"create did not succeed: HTTP {status}, body {_shape(resp)}")
        if status != 200 or not isinstance(resp, dict) or not resp.get("success") or "new_contract" not in resp:
            # NOT understood (a 5xx after the contract committed, a 200 without `new_contract`, a non-JSON body):
            # the contract may exist server-side with nobody holding its id. Sweep by this run's label — labels
            # are per run — and never let this become a clean refusal receipt (CEO read, #460 MEDIUM-2).
            self._sweep_orphans(f"create answered a shape this code does not understand: HTTP {status}, body {_shape(resp)}")
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

    def _sweep_orphans(self, why: str) -> None:
        """After a create this code could not parse: list every instance carrying this run's label, destroy
        each, and prove absence with a fresh listing. Raises OrphanSwept (something was running and is now
        gone), PossibleOrphan (nothing can be proven), or BackendUnavailable (the listing is clean: nothing
        under the label ever existed). Never returns normally — the create is failed either way."""
        console = f"POSSIBLE ORPHAN — check the console for label {self.run_label!r}"
        try:
            mine = [str(r.get("id")) for r in self.list_instances() if str(r.get("label", "")) == self.run_label and r.get("id") is not None]
        except BackendUnavailable as e:
            raise PossibleOrphan(f"{why}; the sweep could not list instances ({e}); {console}") from e
        if not mine:
            raise BackendUnavailable(f"{why}; a fresh v1 listing shows nothing under label {self.run_label!r}, so nothing was created")
        failures: list[str] = []
        for iid in mine:
            try:
                self.destroy(iid)
            except BackendUnavailable as e:
                failures.append(f"{iid}: {e}")
        try:
            after = self.list_ids()
        except BackendUnavailable as e:
            raise PossibleOrphan(f"{why}; swept {mine} by label but the listing after destroy failed ({e}); {console}") from e
        still = sorted(i for i in mine if i in after)
        if failures or still:
            raise PossibleOrphan(f"{why}; swept {mine} by label: destroy failed for {failures or 'none'}, still present {still or 'none'}; {console}")
        raise OrphanSwept(f"{why}; the sweep found {mine} under label {self.run_label!r}, destroyed them, absent in the listing after", mine)

    MAX_PAGES = 50

    def list_instances(self) -> list[dict[str, Any]]:
        """Every live instance record from v1, following `next_token` to the end. An error body, a deprecated-endpoint
        body, a missing `instances` key, a page count beyond MAX_PAGES, or per-page / total counts that disagree with
        the arrays raise — the result is never a partial or empty list on an error. (The real v1 body carries
        `instances, instances_found, label_counts, next_token, success, total_instances` — CEO read 2026-09-06; a
        running instance on a second page read as "gone" is the one false-negative this module promises never to make.)"""
        out: list[dict[str, Any]] = []
        token: Any = None
        for page in range(1, self.MAX_PAGES + 1):
            query = {"next_token": str(token)} if token is not None else None
            status, body = self.t.request("GET", "/v1/instances/", query=query)
            if status != 200 or not isinstance(body, dict):
                raise BackendUnavailable(f"instance list unavailable (page {page}): HTTP {status}, body {_shape(body)}")
            if body.get("success") is False or "error" in body:
                raise BackendUnavailable(f"instance list answered an error, not a list (page {page}): {_shape(body)} (false-zero refused)")
            inst = body.get("instances")
            if not isinstance(inst, list):
                raise BackendUnavailable(f"instance list has no `instances` array (page {page}): {_shape(body)} (false-zero refused)")
            found = body.get("instances_found")
            if isinstance(found, int) and not isinstance(found, bool) and found != len(inst):
                raise BackendUnavailable(f"instance list page {page}: instances_found={found} but the array holds {len(inst)} — refusing to trust it")
            out.extend(r for r in inst if isinstance(r, dict))
            token = body.get("next_token")
            if token in (None, "", 0, False):
                total = body.get("total_instances")
                if isinstance(total, int) and not isinstance(total, bool) and total != len(out):
                    raise BackendUnavailable(f"instance list: total_instances={total} but {len(out)} record(s) were read across {page} page(s) — refusing to trust it")
                return out
        raise BackendUnavailable(f"instance list did not end within {self.MAX_PAGES} pages (next_token still non-null) — refusing to trust it")

    def list_ids(self) -> set[str]:
        """Live instance ids from v1 across every page. Raises rather than answering empty on any error."""
        return {str(r.get("id")) for r in self.list_instances() if r.get("id") is not None}

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
        if status == 200 and isinstance(body, dict) and body.get("success") is False and str(body.get("error", "")) == "no_such_instance":
            # The provider saying "already gone" (the incident's own 12:48:54Z answer) — like the 404, the listing decides.
            return {"method": "vast-destroy", "instance_id": instance_id, "http": 200, "note": "no_such_instance on v0 destroy — absence proven only by the listing", "at": _utc()}
        if not ok:
            raise BackendUnavailable(f"destroy {instance_id} did not succeed: HTTP {status}, body {_shape(body)}")
        return {"method": "vast-destroy", "instance_id": instance_id, "http": status, "at": _utc()}

    def attach_ssh_key(self, instance_id: str, pubkey: str) -> str:
        """Attach the controller's public key to the instance. Returns "yes" on a 200 with success, "already" when the
        API reports the key is already associated (e4b#468: an account key is attached to every new instance by Vast
        itself, so the per-instance attach is an idempotent no-op that the API reports as success=False with that
        message — R1 attempt 1 went NOT_RUN on it); anything else raises (#465 MEDIUM-1: attached is never assumed)."""
        status, body = self.t.request("POST", f"/v0/instances/{instance_id}/ssh/", body={"ssh_key": pubkey})
        if status == 200 and isinstance(body, dict):
            if body.get("success") is not False:
                return "yes"
            if "already associated" in str(body.get("msg", "")).lower():
                return "already"
        raise BackendUnavailable(f"attaching the ssh key to {instance_id} failed: HTTP {status}, body {_shape(body)}")

    # ---- pre-flight
    def preflight(self, instance_id: str, *, timeout_s: float = 600.0, ssh_timeout_s: float = 30.0,
                  min_mb_per_s: float = 40.0, poll_s: float = 10.0, ssh_ready_s: float = 180.0) -> dict[str, str]:
        """Usable, or PreflightFailed with the reason. Bounded: the instance must be `running` (not stuck
        `loading`) within timeout_s; disk/RAM as ordered; ssh authenticates within ssh_ready_s (each attempt
        bounded by ssh_timeout_s); ≥ min MB/s."""
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
        attached: dict[str, str] = {}
        if self.ssh_pubkey:
            # raises on anything but a 200 → the pre-flight fails; "already" = the account key Vast attached itself (#468)
            attached["vast_ssh_key_attached"] = self.attach_ssh_key(instance_id, self.ssh_pubkey)   # #465 MEDIUM-1: after the 200
        rc, out, tries = self._ssh_until_ready(str(host), int(port), ssh_timeout_s=ssh_timeout_s, ready_s=ssh_ready_s,
                                                poll_s=min(poll_s, 5.0))
        if rc != 0:
            raise PreflightFailed(f"ssh to {host}:{port} did not authenticate within {int(ssh_ready_s)} s "
                                  f"({tries} attempt{'s' if tries != 1 else ''}; last rc {rc}: {out.strip()[:120]})")
        bandwidth_probe: dict[str, Any] = {}
        if self._bandwidth is None:
            mbps, bandwidth_probe = _bandwidth_over_ssh_with_evidence(
                str(host), int(port), stop_at_mb_s=min_mb_per_s,
            )
        else:
            mbps = float(self._bandwidth(str(host), int(port)))
        if mbps < min_mb_per_s:
            evidence = (f"; probe={json.dumps(bandwidth_probe, separators=(',', ':'), sort_keys=True)}"
                        if bandwidth_probe else "")
            raise PreflightFailed(
                f"download bandwidth {mbps:.1f} MB/s < {min_mb_per_s:.0f} MB/s on {host}:{port}{evidence}"
            )
        return {"vast_preflight": "ok", "vast_ssh": f"{host}:{port}", "vast_actual_status": st,
                "vast_disk_space_gb": f"{disk:.0f}", "vast_cpu_ram_mb": f"{ram_mb:.0f}", "vast_bandwidth_mb_s": f"{mbps:.1f}",
                "vast_preflight_seconds": f"{self._clock() - t0:.0f}", "vast_ssh_attempts": str(tries),
                **({"vast_bandwidth_probe": json.dumps(bandwidth_probe, separators=(',', ':'), sort_keys=True)}
                   if bandwidth_probe else {}), **attached}

    def _ssh_until_ready(self, host: str, port: int, *, ssh_timeout_s: float, ready_s: float,
                         poll_s: float) -> tuple[int, str, int]:
        """A rented box reports `running` before its container's sshd accepts connections, so a single probe is a
        race the launcher loses: a refused connect answers at once (rc 255) and `ConnectTimeout` never applies.
        R1 attempt 2 (private receipt p41-r1-granite-2, 2026-09-06T19:00:58Z: `connect to host … port …: Connection refused`, 35 s):
        probe until sshd answers or the ready window is spent, and record how many attempts it took. Each attempt
        keeps its own bound; the window is what changed, not the per-attempt timeout. An authentication REFUSAL
        (rc 255 with a permission/auth message) is not a not-yet-up box — it fails immediately, as before."""
        t0 = self._clock()
        tries = 0
        while True:
            tries += 1
            rc, out = self._ssh(host, port, "true", ssh_timeout_s)
            if rc == 0:
                return rc, out, tries
            if _is_auth_refusal(out):
                return rc, out, tries
            if self._clock() - t0 >= ready_s:
                return rc, out, tries
            self._sleep(poll_s)

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


# What "the box is not up yet" looks like versus "this box will never let us in". The first is worth waiting for; the
# second is a refusal the pre-flight must report at once, so a misconfigured key never burns the whole ready window.
AUTH_REFUSAL = ("permission denied", "publickey", "authentication failed", "too many authentication failures",
                "host key verification failed", "no matching host key")


def _is_auth_refusal(out: str) -> bool:
    low = out.lower()
    return any(m in low for m in AUTH_REFUSAL)


def _ssh_run(host: str, port: int, command: str, timeout_s: float) -> tuple[int, str]:
    try:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                              "-o", f"ConnectTimeout={int(timeout_s)}", "-p", str(port), f"root@{host}", command],
                             capture_output=True, text=True, timeout=timeout_s + 5)
        return out.returncode, (out.stdout + out.stderr)
    except subprocess.TimeoutExpired:
        return 124, f"ssh timed out after {timeout_s} s"


# Endpoint 1 is the largest size Cloudflare actually serves: 20/50/75 MB answer 200, 100 MB answers 403 with a
# one-byte body (measured from the controller and independently by the CEO, 2026-09-06). A bigger sample makes
# the handshake a smaller share of the window; the transfer-only timing below removes what is left of it.
BANDWIDTH_URLS = ("https://speed.cloudflare.com/__down?bytes=75000000", "http://speedtest.tele2.net/100MB.zip")
BANDWIDTH_MIN_BYTES = 5_000_000
BANDWIDTH_WINDOW_S = 45


def _bandwidth_reading(out: str) -> tuple[float, float, float, str]:
    """Return (MB/s, bytes, seconds, HTTP status), rejecting error bodies and undersized transfers.

    Two shapes are accepted. `<bytes> <seconds> <status>` — the seconds are already transfer-only, which is what
    the python3 and wget legs print, because they start their clock at the first byte. `<bytes> <total>
    <starttransfer> <status>` — curl's, where `time_total` includes DNS, TCP and TLS setup; the link is
    `size / (time_total - time_starttransfer)`. Timing the handshake understates a box: measured from the
    controller on 2026-09-06, 20 MB read 7.01 MB/s naive against 8.29 MB/s transfer-only (18 % low) and 75 MB
    read 40.6 against 42.7 (5 %). Against a 40 MB/s floor an 18 % understatement destroys a healthy box — the
    same class of error as timing a 403 body, one level up (CEO read of #474, 2026-09-06T20:59Z). A
    non-positive or absent transfer window falls back to the total rather than dividing by zero, and says so."""
    parts = out.strip().split()
    if len(parts) < 3:
        raise ValueError(f"no <bytes> <seconds> <status> line: {out.strip()[:120]!r}")
    if len(parts) >= 4:
        size, total, start, code = float(parts[0]), float(parts[1]), float(parts[2]), parts[3]
        secs = total - start
        if secs <= 0:                     # a cached or instant body: the total is the only honest window left
            secs = total
    else:
        size, secs, code = float(parts[0]), float(parts[1]), parts[2]
    if code != "200":
        raise ValueError(f"HTTP {code} after {size:.0f} B — an error body is not a measurement")
    if size < BANDWIDTH_MIN_BYTES:
        raise ValueError(f"only {size:.0f} B transferred (< {BANDWIDTH_MIN_BYTES} B floor)")
    return size / max(secs, 1e-9) / 1e6, size, secs, code


def _bandwidth_over_ssh(host: str, port: int) -> float:
    """Compatibility wrapper for callers that need only the measured MB/s."""
    mbps, evidence = _bandwidth_over_ssh_with_evidence(host, port)
    if mbps <= 0:
        raise PreflightFailed(
            f"download bandwidth could not be measured on {host}:{port}; "
            f"probe={json.dumps(evidence, separators=(',', ':'), sort_keys=True)}"
        )
    return mbps


def _bandwidth_over_ssh_with_evidence(host: str, port: int, *,
                                      stop_at_mb_s: float | None = None) -> tuple[float, dict[str, Any]]:
    """Measure a 100 MB download on the box and retain bounded, non-secret evidence for every attempt.

    Each endpoint is tried twice before the next one.  A positive result below ``stop_at_mb_s`` does not suppress
    the fallback endpoint; the first result that clears the registered floor may return immediately.  Without a
    floor all attempts run and the best reading wins.  R1 attempt 3 showed why a bare 0.0 is insufficient: four
    failures and about four minutes of wall time could not distinguish curl, HTTP, parsing or box-egress failure.
    """
    best = 0.0
    attempts: list[dict[str, str | int]] = []
    cap_cmd = ('for t in curl wget python3; do p=$(command -v "$t" 2>/dev/null) || continue; '
               'printf \'%s=%s\\n\' "$t" "$p"; done')
    cap_rc, cap_out = _ssh_run(host, port, cap_cmd, 15)
    capabilities = {}
    for line in cap_out.splitlines():
        tool, sep, path = line.partition("=")
        if sep and tool in ("curl", "wget", "python3") and path.startswith("/"):
            capabilities[tool] = path[:160]
    evidence: dict[str, Any] = {
        "capability": {"rc": cap_rc, "tools": capabilities,
                       "sample": " ".join(cap_out.strip().split())[:160] or "<empty>"},
        "attempts": attempts,
    }
    if cap_rc != 0 or not capabilities:
        attempts.append({"endpoint": 0, "attempt": 0, "rc": cap_rc, "sample": "no measurable downloader",
                         "result": "capability-failed"})
        return best, evidence

    if "curl" in capabilities:
        tool = "curl"

        def command(url: str) -> str:
            return ("curl --location --fail --silent --show-error -o /dev/null "
                    f"-w '%{{size_download}} %{{time_total}} %{{time_starttransfer}} %{{http_code}}' "
                    f"--max-time {BANDWIDTH_WINDOW_S} {shlex.quote(url)}")
    elif "wget" in capabilities and "python3" in capabilities:
        tool = "wget"
        wget_code = ("import subprocess,sys,time;"
                     f"p=subprocess.Popen(['wget','-q','-O','-','--timeout={BANDWIDTH_WINDOW_S}',sys.argv[1]],"
                     "stdout=subprocess.PIPE);"
                     "n=0;t=None\n"
                     "while True:\n"
                     " b=p.stdout.read(1048576)\n"
                     " if not b:break\n"
                     " if t is None:t=time.monotonic()\n"     # the clock starts at the first byte, not at connect
                     " n+=len(b)\n"
                     f" if time.monotonic()-t>{BANDWIDTH_WINDOW_S}:break\n"
                     "rc=p.wait();print(n,(time.monotonic()-t) if t else 0.0,200);raise SystemExit(rc)")

        def command(url: str) -> str:
            return f"python3 -c {shlex.quote(wget_code)} {shlex.quote(url)}"
    elif "python3" in capabilities:
        tool = "python3"
        python_code = ("import sys,time,urllib.request as u;"
                       "q=u.Request(sys.argv[1],headers={'User-Agent':'curl/8'});"
                       f"r=u.urlopen(q,timeout={BANDWIDTH_WINDOW_S});"
                       "n=0;t=None\nwhile True:\n b=r.read(1048576)\n if not b:break\n"
                       " if t is None:t=time.monotonic()\n"   # the connect and TLS handshake are not the link
                       " n+=len(b)\n"
                       f" if time.monotonic()-t>{BANDWIDTH_WINDOW_S}:break\n"
                       "print(n,(time.monotonic()-t) if t else 0.0,r.status);r.close()")

        def command(url: str) -> str:
            return f"python3 -c {shlex.quote(python_code)} {shlex.quote(url)}"
    else:
        attempts.append({"endpoint": 0, "attempt": 0, "rc": 127, "sample": "wget has no python3 timer",
                         "result": "capability-failed"})
        return best, evidence

    for endpoint_index, url in enumerate(BANDWIDTH_URLS, start=1):
        for attempt in range(1, 3):
            rc, out = _ssh_run(host, port, command(url), 90)
            sample = " ".join(out.strip().split())[:160] or "<empty>"
            item: dict[str, str | int] = {
                "endpoint": endpoint_index, "attempt": attempt, "tool": tool, "rc": rc, "sample": sample,
            }
            if rc != 0:
                item["result"] = "command-failed"
                attempts.append(item)
                continue
            try:
                measured, size, secs, code = _bandwidth_reading(out)
            except (ValueError, IndexError) as e:
                item["result"] = "invalid-reading"
                item["error"] = str(e)[:160]
                attempts.append(item)
                continue
            best = max(best, measured)
            item["mb_s"] = f"{measured:.1f}"
            item["bytes"] = f"{size:.0f}"
            item["seconds"] = f"{secs:.3f}"
            item["http_status"] = code
            item["result"] = "ok"
            attempts.append(item)
            if stop_at_mb_s is not None and measured >= stop_at_mb_s:
                return best, evidence
    return best, evidence


def _under_test() -> bool:
    """True inside pytest / unittest. Tests NEVER go live: on 2026-09-06 12:38Z a test that meant to exercise the
    "no key file" refusal read the real key (the key path was bound as a default argument, so a monkeypatched
    module constant did nothing), rented an RTX 5090 for ten minutes and paid for it. Belt and braces below."""
    argv0 = (sys.argv[0] if sys.argv else "").rsplit("/", 1)[-1]
    return (bool(os.environ.get("PYTEST_CURRENT_TEST")) or "pytest" in sys.modules or "_pytest" in sys.modules
            or argv0 in ("pytest", "py.test") or any(a in ("pytest", "unittest") for a in sys.argv[:2]))


def _no_live_env() -> bool:
    """E4B_NO_LIVE crosses a process boundary where _under_test() cannot: the test fixture sets it, a subprocess a
    test spawns inherits it, and provider_from_env refuses on it whatever E4B_RENT_LIVE says (CEO read, #460)."""
    return os.environ.get("E4B_NO_LIVE", "") not in ("", "0")


def provider_from_env(*, run_label: str, key_path: Path | str | None = None, **kw: Any) -> VastProvider:
    """The launcher's factory: armed only by E4B_RENT_LIVE=1 and a readable key file; the auth probe runs first.
    Refuses under a test runner whatever the environment says — a live adapter in a test is a rented box."""
    if _under_test():
        raise VastRefused("live provider refused: this process is a test runner (tests never rent; use FakeTransport)")
    if _no_live_env():
        raise VastRefused("live provider refused: E4B_NO_LIVE is set in this environment (a test fixture sets it and every child inherits it; tests never rent)")
    if os.environ.get("E4B_RENT_LIVE") != "1":
        raise VastRefused("live provider vast:verified-secure is not armed in this process: set E4B_RENT_LIVE=1 (a fake run is --dry-run)")
    key = load_api_key(DEFAULT_KEY_PATH if key_path is None else key_path)  # read at call time, never bound at import
    prov = VastProvider(UrllibTransport(key), run_label=run_label, **kw)
    prov.auth_probe()
    return prov
