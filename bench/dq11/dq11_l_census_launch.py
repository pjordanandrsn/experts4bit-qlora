"""Read-only stable-host packet and guarded DQ11 L-census launch admission.

No command in this module creates compute except ``launch``, which requires a
merged clean source, a time-limited maintainer envelope, a bound manifest, a
fresh CPU gate, five minutes of provider observations and a final pinned quote
before it calls the already-reviewed pod-launch controller.
"""

from __future__ import annotations

import argparse
import hashlib
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from pathlib import Path

from dq11_bootstrap import preflight_urls
from dq11_census_transport import direct_available
from dq11_policy_preflight import preflight_policy
from dq11_science_stage import stage

SCIENCE = "13b55990433f357d63232f75926ce19f5bda5b42"
SCIENCE_MANIFEST_SHA = "4691e7329f12ad2f4b7518b75c4d3e351207fc4e5220aee514fdd42c429fef01"
INSTRUMENT_MERGE = "6e197a3f42599c8d10667a4c133627496e40f28f"
OFFER_SET_MERGE = "a41496d1c448735eb3c77a34bba8e29e686eb4bb"
MIB = 1 << 20
GIB = 1 << 30
RAW_TRACE_MAX = 512 * MIB
PACKET_MAX = 2 * GIB
MIN_STABLE_SECONDS = 300
MAX_SAMPLE_GAP_SECONDS = 30
FRESH_SECONDS = 60
POLL_SECONDS = 20
DISCOVERY_CUTOFF_SECONDS = 3 * 60 * 60
RUN_ID_PLACEHOLDER = "__DQ11_CENSUS_RUN_ID__"
MAX_ATTEMPTS = 10
ENVELOPE_HOURS = 24
WALL_HOURS = Decimal("0.75")
HOURLY_CAP = Decimal("0.85")
DOWNLOAD_GB = Decimal(27)
DOWNLOAD_RATE_CAP = Decimal("0.011")
UPLOAD_GB = Decimal(3)
UPLOAD_RATE_CAP = Decimal("0.05")
ALL_IN_CAP = Decimal("1.10")
DISK_GB = 320
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"
# 147454: stuck loading at the pre-flight on two envelopes in a row (census-7 and census-8, 2026-10-19). A journal's own
# stall exclusion covers only its envelope, so the registered list carries it into the next one.
EXCLUDED_MACHINES = {"45511", "147454"}
NEXT_RUN_ID = "next"      # --run-id next: the first declared id the envelope journal has not used
INSTRUMENT_FILES = (
    "dq11_l_census_run.py", "dq11_l_census_capture.py", "dq11_l_census_reduce.py"
)
OPERATIONAL_FILES = (
    "dq11_l_census_launch.py", "dq11_l_census_box.sh", "dq11_l_census_drive.sh",
    "dq11_wheel_retry.py", "dq11_census_transport.py",
)
TRANSPORT_FILE = "bench/tc1/tc1_drive.sh"
TEMPLATE_FILE = "bench/dq11/dq11_l_census_manifest_template.json"
#: The seat that runs the launch: POD_LAUNCH_AGENT, a well-formed CTO identity, which the controller records in the
#: receipt. The envelope binds ONE executor by the SHA-256 of that identity (``executor_sha256``: the ACK is a public
#: comment, so it carries no seat name), and the launch refuses any other seat.
EXECUTOR_RE = re.compile(r"CTO(/[A-Za-z0-9._-]+)+")


def executor_sha256(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()
#: The launcher refuses any top-level manifest key it does not read (adertha-agents#243), so the census's own record
#: (its status, closure and authorization basis) travels in ``lane_record``, which the launcher copies into the receipt.
LANE_RECORD_FIELDS = frozenset({"status", "l_census", "authorization_basis"})
#: The top-level fields a bound manifest adds to the registered template, all of them launcher terms.
BINDING_FIELDS = frozenset({"heads", "command", "vast_pin_machine", "vast_pin_offers", "vast_required_driver"})
ENVELOPE_CRITERIA = {
    "gpu": "RTX 5090", "num_gpus": 1, "verification": "verified",
    "direct_ssh_required": True,
    "vast_direct_ssh": True,
    "rentable": True, "external": False, "minimum_reliability": "0.98",
    "minimum_driver_major": 580, "exact_driver_components": 3,
    "minimum_gpu_ram_mb": 30000, "maximum_gpu_ram_mb": 34000,
    "minimum_host_ram_mb": 98 * 1024, "ordered_disk_gb": DISK_GB,
    "maximum_gpu_storage_usd_per_hour": str(HOURLY_CAP),
    "maximum_download_usd_per_gb": str(DOWNLOAD_RATE_CAP),
    "maximum_upload_usd_per_gb": str(UPLOAD_RATE_CAP),
    "maximum_all_in_usd": str(ALL_IN_CAP), "maximum_draws": 1,
    "maximum_paid_wall_seconds": 2700,
    "minimum_continuous_seconds": MIN_STABLE_SECONDS,
    "maximum_poll_gap_seconds": MAX_SAMPLE_GAP_SECONDS,
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def number(value: object, name: str) -> Decimal:
    if type(value) not in (str, int, float):
        raise ValueError(f"{name} is not a numeric rate")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} is not a numeric rate") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} is negative or nonfinite")
    return result


def reserve(hourly: Decimal, download: Decimal, upload: Decimal) -> Decimal:
    return hourly * WALL_HOURS + download * DOWNLOAD_GB + upload * UPLOAD_GB


def check_static_budget() -> None:
    if reserve(HOURLY_CAP, DOWNLOAD_RATE_CAP, UPLOAD_RATE_CAP) != Decimal("1.0845"):
        raise ValueError("registered all-in budget arithmetic changed")
    if PACKET_MAX > UPLOAD_GB * Decimal(1_000_000_000):
        raise ValueError("upload reservation is smaller than evidence packet cap")


def offer_row(offer: dict, provider, *, excluded=EXCLUDED_MACHINES, all_in: Decimal = ALL_IN_CAP) -> dict:
    """Quote one raw controller-search offer; absent price evidence is a refusal. ``excluded`` and ``all_in`` are the
    envelope journal's bounds for this attempt (``Permit``): machines earlier attempts stalled on, and the all-in left."""
    reasons = []
    if not direct_available(offer):
        reasons.append("direct SSH unavailable")
    machine = str(offer.get("machine_id", ""))
    oid = str(offer.get("id", ""))
    if not machine.isdigit() or not oid.isdigit() or machine in excluded:
        reasons.append("bad or excluded machine/offer identity")
    if offer.get("verification") != "verified" or offer.get("rentable") is not True or offer.get("external") is True:
        reasons.append("outside verified rentable secure provider class")
    if offer.get("gpu_name") != "RTX 5090" or offer.get("num_gpus") != 1:
        reasons.append("not one RTX 5090")
    try:
        reliability = number(offer.get("reliability2"), "reliability")
        if reliability < Decimal("0.98") or reliability > 1:
            reasons.append("reliability below 0.98 or above 1")
    except ValueError as exc:
        reasons.append(str(exc))
        reliability = None
    try:
        gpu_ram = number(offer.get("gpu_ram"), "gpu RAM")
        cpu_ram = number(offer.get("cpu_ram"), "host RAM")
        disk = number(offer.get("disk_space"), "disk")
        number(offer.get("storage_cost"), "storage tariff")
        if not Decimal(30000) <= gpu_ram <= Decimal(34000):
            reasons.append("not a normal 32 GB GPU")
        if cpu_ram < Decimal(98 * 1024) or disk < DISK_GB:
            reasons.append("host RAM or ordered disk below floor")
    except ValueError as exc:
        reasons.append(str(exc))
    driver = str(offer.get("driver_version", ""))
    if not re.fullmatch(r"[0-9]{3}\.[0-9]{1,3}(?:\.[0-9]{1,3})?", driver) or int(driver.split(".")[0]) < 580:
        reasons.append("driver absent or below 580")
    try:
        hourly = number(provider.hourly(offer), "GPU+storage hourly")
        download = number(offer.get("inet_down_cost"), "download tariff")
        # The generic rental driver estimates download only. This launch has an
        # additional explicit upload allowance; an absent tariff is not free.
        upload = number(offer.get("inet_up_cost"), "upload tariff")
    except ValueError as exc:
        reasons.append(str(exc))
        hourly = download = upload = None
    if hourly is not None:
        if hourly > HOURLY_CAP:
            reasons.append("GPU+storage hourly exceeds $0.85")
        if download > DOWNLOAD_RATE_CAP or upload > UPLOAD_RATE_CAP:
            reasons.append("transport tariff exceeds registered cap")
        if reserve(hourly, download, upload) > all_in:
            reasons.append("all-in 45-minute reservation exceeds $1.10" if all_in == ALL_IN_CAP else
                           f"all-in 45-minute reservation exceeds the envelope's remaining ${all_in} of $1.10")
    return {
        "id": oid, "machine_id": machine, "driver_version": driver,
        "direct_port_count": offer.get("direct_port_count"),
        "public_ipaddr": offer.get("public_ipaddr"),
        "reliability": str(reliability) if reliability is not None else None,
        "gpu_name": offer.get("gpu_name"), "gpu_ram": offer.get("gpu_ram"),
        "num_gpus": offer.get("num_gpus"), "cpu_ram": offer.get("cpu_ram"),
        "disk_space": offer.get("disk_space"), "verification": offer.get("verification"),
        "rentable": offer.get("rentable"), "external": offer.get("external"),
        "hourly": str(hourly) if hourly is not None else None,
        "download_per_gb": str(download) if download is not None else None,
        "upload_per_gb": str(upload) if upload is not None else None,
        "all_in_reserved": str(reserve(hourly, download, upload)) if hourly is not None else None,
        "eligible": not reasons, "refusal_reasons": reasons,
        "controller_filter": provider._filter("RTX 5090"),
    }


def stable_ids(samples: list[dict], machine: str, required: list[str] | None = None) -> list[str]:
    """One continuously listed machine; approved qualifying IDs may rotate."""
    if not machine.isdigit() or not samples:
        raise ValueError("machine and successful observations required")
    first, last = samples[0]["monotonic"], samples[-1]["monotonic"]
    if last - first < MIN_STABLE_SECONDS:
        raise ValueError("less than five continuous minutes of observations")
    if any(b["monotonic"] <= a["monotonic"] or b["monotonic"] - a["monotonic"] > MAX_SAMPLE_GAP_SECONDS
           for a, b in pairwise(samples)):
        raise ValueError("observation gap exceeds 30 seconds")
    seen = set()
    drivers = {}
    for sample in samples:
        if sample.get("status") != "PASS" or sample.get("machine_id") != machine:
            raise ValueError("failed or foreign controller search in stability interval")
        rows = sample.get("offers")
        if not isinstance(rows, list) or not rows:
            raise ValueError("machine absent from controller search")
        eligible = {row.get("id") for row in rows}
        if (any(not row.get("eligible") or row.get("machine_id") != machine or not str(row.get("id", "")).isdigit()
                for row in rows)):
            raise ValueError("machine returned an ineligible offer")
        if len(eligible) != len(rows):
            raise ValueError("duplicate offer identity in a search")
        if required is not None and (not required or eligible - set(required)):
            raise ValueError("offer outside ACK set during stability interval")
        for row in rows:
            previous = drivers.setdefault(row["id"], row.get("driver_version"))
            if row.get("driver_version") != previous:
                raise ValueError("driver changed during stability interval")
        seen.update(eligible)
    if required is not None and (len(set(required)) != len(required) or not seen <= set(required)):
        raise ValueError("observed union exceeds ACK offer set")
    return sorted(seen, key=int)


class HostContinuity:
    """Apply the registered continuous-presence rule independently to every host."""

    def __init__(self, excluded=EXCLUDED_MACHINES) -> None:
        self.excluded = frozenset(excluded)
        self.active: dict[str, dict] = {}
        self.maximum_seconds: dict[str, float] = defaultdict(float)
        self.resets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def _end(self, machine: str, reason: str) -> None:
        previous = self.active.pop(machine, None)
        if previous:
            samples = previous["samples"]
            self.maximum_seconds[machine] = max(
                self.maximum_seconds[machine], samples[-1]["monotonic"] - samples[0]["monotonic"])
            self.resets[machine][reason] += 1

    def step(self, poll: dict) -> tuple[str, list[dict], list[str], str] | None:
        if poll.get("status") != "PASS":
            for machine in list(self.active):
                self._end(machine, "search_failure")
            return None
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in poll["offers"]:
            grouped[str(row.get("machine_id", ""))].append(row)
        for machine in list(self.active):
            if machine not in grouped:
                self._end(machine, "absent")
        qualified = []
        for machine, rows in grouped.items():
            ids = [str(row.get("id", "")) for row in rows]
            if (not machine.isdigit() or machine in self.excluded or len(set(ids)) != len(ids)
                    or any(not row.get("eligible") or row.get("machine_id") != machine or not oid.isdigit()
                           for row, oid in zip(rows, ids))):
                self._end(machine, "ineligible_or_outside")
                continue
            drivers = {row.get("driver_version") for row in rows}
            if len(drivers) != 1:
                self._end(machine, "mixed_driver")
                continue
            driver = next(iter(drivers))
            if (not isinstance(driver, str)
                    or not re.fullmatch(r"[0-9]{3}\.[0-9]{1,3}\.[0-9]{1,3}", driver)
                    or int(driver.split(".")[0]) < 580):
                self._end(machine, "non_exact_driver")
                continue
            previous = self.active.get(machine)
            if previous and (poll["monotonic"] <= previous["samples"][-1]["monotonic"]
                             or poll["monotonic"] - previous["samples"][-1]["monotonic"] > MAX_SAMPLE_GAP_SECONDS
                             or driver != previous["driver"]):
                self._end(machine, "gap_or_driver_change")
                previous = None
            if previous is None:
                previous = {"driver": driver, "samples": []}
                self.active[machine] = previous
            previous["samples"].append({
                "utc": poll["utc"], "monotonic": poll["monotonic"], "status": "PASS",
                "machine_id": machine, "filter": poll["filter"], "offers": rows,
            })
            samples = previous["samples"]
            span = samples[-1]["monotonic"] - samples[0]["monotonic"]
            self.maximum_seconds[machine] = max(self.maximum_seconds[machine], span)
            if span >= MIN_STABLE_SECONDS:
                qualified.append((machine, samples, stable_ids(samples, machine), driver))
        if not qualified:
            return None
        qualified.sort(key=lambda item: int(item[0]))
        return qualified[0]


def observe_all_hosts(provider, packet_dir: Path, *, clock=time.monotonic,
                      sleep=time.sleep, cutoff: float = DISCOVERY_CUTOFF_SECONDS,
                      excluded=EXCLUDED_MACHINES, all_in: Decimal = ALL_IN_CAP) -> dict:
    """Write every GET-only poll before admitting the first qualifying host."""
    polls_path = packet_dir / "all-host-polls.jsonl"
    history_path = packet_dir / "stable-host.json"
    continuity = HostContinuity(excluded)
    start = clock()
    due = start
    count = 0
    selected = None
    with polls_path.open("x") as stream:
        while True:
            delay = due - clock()
            if delay > 0:
                sleep(delay)
            if clock() - start >= cutoff:
                break
            try:
                offers = provider.search_offers("RTX 5090")
                poll = {"utc": now(), "monotonic": clock(), "status": "PASS",
                        "filter": provider._filter("RTX 5090"),
                        "offers": [offer_row(offer, provider, excluded=excluded, all_in=all_in) for offer in offers]}
            except (RuntimeError, ValueError, OSError, TypeError) as exc:
                poll = {"utc": now(), "monotonic": clock(), "status": "ERROR",
                        "filter": provider._filter("RTX 5090"), "offers": [],
                        "error_type": type(exc).__name__, "error": str(exc)}
            stream.write(json.dumps(poll, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            count += 1
            if poll["monotonic"] - start >= cutoff:
                continuity.step({**poll, "status": "ERROR", "offers": []})
                break
            selected = continuity.step(poll)
            if selected:
                break
            due += POLL_SECONDS
            due = max(due, poll["monotonic"])
    report = {"schema": "dq11-l-census-all-host-discovery/1", "utc": now(),
              "status": "PASS" if selected else "CUTOFF_NO_STABLE_HOST",
              "mode": "READ_ONLY_CONTROLLER_SEARCH_NO_CREATE",
              "poll_interval_seconds": POLL_SECONDS, "minimum_stable_seconds": MIN_STABLE_SECONDS,
              "max_gap_seconds": MAX_SAMPLE_GAP_SECONDS, "cutoff_seconds": cutoff,
              "poll_count": count, "polls_sha256": sha(polls_path),
              "per_machine_max_continuous_seconds": dict(continuity.maximum_seconds),
              "per_machine_reset_counts": {k: dict(v) for k, v in continuity.resets.items()},
              "machine_id": selected[0] if selected else None,
              "offer_ids": selected[2] if selected else None,
              "driver": selected[3] if selected else None}
    if selected:
        machine, samples, offers, _ = selected
        history = {"schema": "dq11-l-census-stable-host/1", "utc": now(),
                   "machine_id": machine, "offer_ids": offers, "samples": samples,
                   "first_utc": samples[0]["utc"], "last_utc": samples[-1]["utc"],
                   "minimum_stable_seconds": MIN_STABLE_SECONDS,
                   "max_gap_seconds": MAX_SAMPLE_GAP_SECONDS,
                   "mode": "READ_ONLY_CONTROLLER_SEARCH_NO_CREATE",
                   "all_host_polls_sha256": report["polls_sha256"]}
        with history_path.open("x") as stream:
            json.dump(history, stream, indent=2)
            stream.write("\n")
        report["stable_history_sha256"] = sha(history_path)
    with (packet_dir / "all-host-summary.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    if not selected:
        raise ValueError("three-hour all-host search ended without a qualifying window")
    return report


def check_qualifying_window(packet_dir: Path, summary: dict) -> dict:
    """Re-read the sealed poll and selected window before binding the manifest."""
    history_path = packet_dir / "stable-host.json"
    if (summary.get("status") != "PASS"
            or summary.get("polls_sha256") != sha(packet_dir / "all-host-polls.jsonl")
            or summary.get("stable_history_sha256") != sha(history_path)):
        raise ValueError("qualifying-window SHA evidence is absent or changed")
    history = json.loads(history_path.read_text())
    machine = str(summary["machine_id"])
    if (history.get("machine_id") != machine or history.get("offer_ids") != summary.get("offer_ids")
            or history.get("all_host_polls_sha256") != summary["polls_sha256"]
            or stable_ids(history["samples"], machine) != summary["offer_ids"]
            or any(row["driver_version"] != summary["driver"]
                   for sample in history["samples"] for row in sample["offers"])):
        raise ValueError("selected host window does not match the complete poll record")
    return history


def packet_files(root: Path) -> tuple[int, dict[str, dict]]:
    """Bound private transfer by uncompressed regular-file bytes; refuse links."""
    if root.is_symlink():
        raise ValueError("evidence packet root is a symlink")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("evidence packet is not a directory")
    total = 0
    rows = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(("hf-cache/", "venv", "adapters/", ".cache/", "gnf4-src/",
                           "unsloth_compiled_cache/")):
            continue  # the transport excludes these entire trees
        if rel == "data/alpaca_data_cleaned.json":
            continue
        if path.is_symlink():
            raise ValueError("evidence packet contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("evidence packet contains a non-file")
        size = path.stat().st_size
        if path.name in ("proof-trace.json", "capture-trace.json") and size > RAW_TRACE_MAX:
            raise ValueError("raw trace exceeds 512 MiB")
        total += size
        if total > PACKET_MAX:
            raise ValueError("uncompressed evidence packet exceeds 2 GiB")
        rows[rel] = {"bytes": size, "sha256": sha(path)}
    if not rows:
        raise ValueError("empty evidence packet")
    return total, rows


class ReadOnlyTransport:
    def __init__(self, delegate):
        self.delegate = delegate

    def request(self, method, path, **kwargs):
        if method != "GET" or path != "/v0/bundles/":
            raise ValueError("read-only stability controller refuses non-search request")
        return self.delegate.request(method, path, **kwargs)


def live_provider(machine: str | None, controller: Path, e4b: Path,
                  required_driver: str | None = None, *, excluded=EXCLUDED_MACHINES):
    merged_clean(e4b)
    if not (e4b / "bench/dq11/DQ11-L-CENSUS-LAUNCH.md").is_file():
        raise ValueError("merged L-census launch amendment required before live search")
    merged_clean(controller)
    if subprocess.call(["git", "-C", str(controller), "merge-base", "--is-ancestor", OFFER_SET_MERGE, "HEAD"]):
        raise ValueError("merged offer-set controller is not an ancestor")
    sys.path.insert(0, str(controller))
    from adertha.compute import vast_provider as vp

    provider = vp.provider_from_env(
        run_label="dq11-l-census-read-only", min_disk_gb=DISK_GB,
        min_gpu_ram_gb=30, excluded_machine_ids=set(excluded),
        download_gb=float(DOWNLOAD_GB), max_download_usd_per_gb=float(DOWNLOAD_RATE_CAP),
        required_driver=required_driver, pinned_machine_id=machine,
    )
    provider.t = ReadOnlyTransport(provider.t)
    return provider


def observation(provider, machine: str, *, clock=time.monotonic, excluded=EXCLUDED_MACHINES,
                all_in: Decimal = ALL_IN_CAP) -> dict:
    # One call to the same provider adapter and filter the paid driver uses.
    offers = provider.search_offers("RTX 5090")
    return {
        "utc": now(), "monotonic": clock(), "status": "PASS",
        "machine_id": machine, "filter": provider._filter("RTX 5090"),
        "offers": [offer_row(offer, provider, excluded=excluded, all_in=all_in)
                   for offer in offers if str(offer.get("machine_id")) == machine],
    }


def observe(provider, machine: str, required: list[str] | None = None, *, clock=time.monotonic, sleep=time.sleep) -> dict:
    samples = []
    while True:
        samples.append(observation(provider, machine, clock=clock))
        elapsed = samples[-1]["monotonic"] - samples[0]["monotonic"]
        if elapsed >= MIN_STABLE_SECONDS:
            break
        sleep(min(20, MIN_STABLE_SECONDS - elapsed))
    offers = stable_ids(samples, machine, required)
    return {"schema": "dq11-l-census-stable-host/1", "utc": now(), "machine_id": machine,
            "offer_ids": offers if required is None else required, "samples": samples,
            "first_utc": samples[0]["utc"], "last_utc": samples[-1]["utc"],
            "minimum_stable_seconds": MIN_STABLE_SECONDS, "max_gap_seconds": MAX_SAMPLE_GAP_SECONDS,
            "mode": "READ_ONLY_CONTROLLER_SEARCH_NO_CREATE"}


def merged_clean(repo: Path) -> str:
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise ValueError(f"dirty source checkout: {repo}")
    if subprocess.call(["git", "-C", str(repo), "merge-base", "--is-ancestor", head, "origin/main"]):
        raise ValueError(f"unmerged source checkout: {repo}")
    return head


def validate_envelope(envelope: dict, envelope_path: Path, template_path: Path,
                      e4b: Path, controller: Path, *, at: datetime | None = None) -> None:
    """Refuse stale, broader, or differently sourced maintainer approval."""
    if template_path.resolve() != (e4b / TEMPLATE_FILE).resolve():
        raise ValueError("envelope template is not the merged census template")
    if envelope.get("schema") != "dq11-l-census-envelope-ack/1":
        raise ValueError("missing L-census envelope schema")
    url = envelope.get("approval_url")
    if (not isinstance(url, str) or not re.fullmatch(
            r"https://github\.com/pjordanandrsn/experts4bit-qlora/issues/1083#issuecomment-[0-9]+", url)):
        raise ValueError("envelope ACK needs the maintainer's citable GitHub comment")
    try:
        issued = datetime.fromisoformat(envelope["issued_at"].replace("Z", "+00:00"))
        expires = datetime.fromisoformat(envelope["expires_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError) as exc:
        raise ValueError("envelope lacks valid issue and expiry times") from exc
    current = at or datetime.now(timezone.utc)
    if (issued.tzinfo is None or expires.tzinfo is None or current.tzinfo is None
            or not issued <= current < expires
            or (expires - issued).total_seconds() != ENVELOPE_HOURS * 3600):
        raise ValueError("envelope is not yet valid, expired or not exactly 24 hours")
    if not (isinstance(envelope.get("executor_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", envelope["executor_sha256"])):
        raise ValueError("envelope must bind its one executor (executor_sha256)")
    if (json.dumps(envelope.get("criteria"), sort_keys=True) != json.dumps(ENVELOPE_CRITERIA, sort_keys=True)
            or type(envelope.get("maximum_draws")) is not int or envelope["maximum_draws"] != 1
            or not valid_run_ids(envelope)):
        raise ValueError("host or budget is outside the exact envelope criteria")
    if (envelope.get("e4b_head") != merged_clean(e4b)
            or envelope.get("controller_head") != merged_clean(controller)
            or envelope.get("manifest_template_sha256") != sha(template_path)):
        raise ValueError("envelope lapsed after a source or manifest template change")
    if envelope_path.is_symlink() or not envelope_path.is_file():
        raise ValueError("envelope record must be a regular file")


def verify_envelope_comment(envelope: dict) -> None:
    """Resolve the ACK URL to the maintainer's current exact JSON on issue #1083."""
    comment_id = envelope["approval_url"].rsplit("#issuecomment-", 1)[1]
    result = subprocess.run(
        ["gh", "api", f"repos/pjordanandrsn/experts4bit-qlora/issues/comments/{comment_id}"],
        capture_output=True, text=True, check=False, timeout=15)
    if result.returncode:
        raise ValueError("maintainer envelope ACK comment could not be read")
    try:
        comment = json.loads(result.stdout)
        attested = json.loads(comment["body"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("maintainer envelope ACK is not exact JSON") from exc
    expected = {key: value for key, value in envelope.items() if key != "approval_url"}
    user = comment.get("user") if isinstance(comment, dict) else None
    if (not isinstance(comment, dict)
            or comment.get("html_url") != envelope["approval_url"]
            or not isinstance(user, dict) or user.get("login") != "pjordanandrsn"
            or attested != expected):
        raise ValueError("envelope record differs from the maintainer's GitHub ACK")
    try:
        created = datetime.fromisoformat(comment["created_at"].replace("Z", "+00:00"))
        updated = datetime.fromisoformat(comment["updated_at"].replace("Z", "+00:00"))
        issued = datetime.fromisoformat(envelope["issued_at"].replace("Z", "+00:00"))
        expires = datetime.fromisoformat(envelope["expires_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError) as exc:
        raise ValueError("envelope ACK lacks authoritative GitHub times") from exc
    if (created.tzinfo is None or updated.tzinfo is None or updated != created
            or issued.tzinfo is None or expires.tzinfo is None
            or abs((issued - created).total_seconds()) > 120
            or (expires - issued).total_seconds() != ENVELOPE_HOURS * 3600):
        raise ValueError("envelope ACK was edited, backdated or extended")


def valid_run_id(run_id: object) -> bool:
    return isinstance(run_id, str) and re.fullmatch(r"dq11-l-census-[0-9]+", run_id) is not None


def valid_run_ids(envelope: dict) -> bool:
    ids = envelope.get("run_ids")
    return (isinstance(ids, list) and 1 <= len(ids) <= MAX_ATTEMPTS
            and all(valid_run_id(run_id) for run_id in ids) and len(set(ids)) == len(ids)
            and type(envelope.get("maximum_attempts")) is int
            and envelope["maximum_attempts"] == len(ids))


def assert_attempt_template(attempt: dict, template: dict, run_id: str) -> None:
    """Compare an actual attempt to the registered template with just run_id substituted."""
    expected = deepcopy(template)
    expected["run_id"] = run_id
    if (template.get("run_id") != RUN_ID_PLACEHOLDER or not valid_run_id(run_id)
            or attempt != expected):
        raise ValueError("unregistered attempt or changed field beyond run_id")


def attempt_template(template: dict, run_id: str) -> dict:
    result = deepcopy(template)
    result["run_id"] = run_id
    assert_attempt_template(result, template, run_id)
    return result


def receipt_paths(receipts: Path, run_id: str) -> list[Path]:
    return list((receipts / "receipts/experts4bit-qlora").glob(f"*/{run_id}/receipt.json"))


def zero_instance_refusal(receipt: dict, approval_url: str, run_id: str) -> bool:
    """Only completed canonical provider refusals prove the created-instance draw is unspent."""
    proof = receipt.get("teardown_proof", {})
    return (receipt.get("experiment_id") == run_id
            and receipt.get("work_id") == "experts4bit-qlora#1083"
            and receipt.get("status") == "REFUSED" and receipt.get("complete") is True
            and receipt.get("instance_id") == "none"
            and receipt.get("cost_usd", {}).get("actual") == 0
            and proof.get("complete") is True and proof.get("method") == "not-launched"
            and proof.get("instance_id") == "none"
            and any(a.get("approval_url") == approval_url for a in receipt.get("approvals", [])))


def preflight_stall(receipt: dict, approval_url: str, run_id: str) -> tuple[str, Decimal] | None:
    """A created instance that ended NOT_RUN at the controller's pre-flight with its command never run and its box
    destroyed (census-7: stuck loading). Returns (machine, actual cost), or None. Every field read is the controller's
    own structured record (status, environment.vast_preflight, the teardown proof and its reason); nothing is
    inferred from a process or a missing file."""
    proof = receipt.get("teardown_proof") or {}
    env = receipt.get("environment") or {}
    instance, machine = str(receipt.get("instance_id", "")), str(env.get("vast_machine_id", ""))
    try:
        cost = number((receipt.get("cost_usd") or {}).get("actual"), "stalled attempt cost")
    except ValueError:
        return None
    if (receipt.get("experiment_id") == run_id
            and receipt.get("work_id") == "experts4bit-qlora#1083"
            and receipt.get("status") == "NOT_RUN" and receipt.get("result") == "invalid"
            and receipt.get("complete") is True
            and env.get("vast_preflight") == "failed"
            and "command not run" in str(receipt.get("notes", ""))
            and instance.isdigit() and machine.isdigit()
            and proof.get("complete") is True and proof.get("method") == "vast-destroy"
            and proof.get("reason") == "preflight-failed" and str(proof.get("instance_id")) == instance
            and any(a.get("approval_url") == approval_url for a in receipt.get("approvals", []))):
        return machine, cost
    return None


class Permit:
    """One admitted attempt and the bounds the envelope journal leaves it: its run id, the spend of earlier attempts
    that stalled before their command ran (the all-in is cumulative), and every machine they stalled on (excluded
    from discovery). Calling it marks the durable controller boundary."""

    def __init__(self, run_id: str, spent: Decimal, excluded, mark) -> None:
        self.run_id, self.spent_usd, self.excluded_machines, self._mark = run_id, spent, frozenset(excluded), mark

    @property
    def budget_usd(self) -> Decimal:
        return ALL_IN_CAP - self.spent_usd

    def __call__(self) -> None:
        self._mark()


def write_attempt_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(state, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def committed_bytes(receipts: Path, path: Path) -> bytes:
    """Accept proof only from tracked, unchanged HEAD bytes, including the index."""
    relative = path.relative_to(receipts).as_posix()
    try:
        subprocess.run(["git", "-C", str(receipts), "ls-files", "--error-unmatch", "--", relative],
                       check=True, capture_output=True)
        status = subprocess.check_output(
            ["git", "-C", str(receipts), "status", "--porcelain", "--", relative], text=True)
        content = subprocess.check_output(["git", "-C", str(receipts), "show", "HEAD:" + relative])
        if status or path.is_symlink() or content != path.read_bytes():
            raise ValueError("receipt/ledger proof must be tracked and clean at HEAD")
        return content
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("receipt/ledger proof must be tracked and clean at HEAD") from exc


@contextmanager
def attempt_permit(envelope: dict, run_id: str, receipts: Path, state_dir: Path):
    """Serialize attempts, persist pre-controller phases, and fail closed on uncertainty. ``run_id`` NEXT takes the
    first declared id the journal has not used. A prior attempt passes only as a $0 provider refusal or as a
    pre-command host stall (``preflight_stall``) whose cost the ledger journals; anything else -- an instance that ran
    its command, or any doubt -- stops for the maintainer. The yielded ``Permit`` carries the cumulative all-in left
    and the stalled machines."""
    if not valid_run_ids(envelope) or (run_id != NEXT_RUN_ID and run_id not in envelope["run_ids"]):
        raise ValueError("invalid envelope run-id range or attempt cap")
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (state_dir / "attempts.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another census envelope attempt is active") from exc
        url = envelope["approval_url"]
        identity = hashlib.sha256(json.dumps(envelope, sort_keys=True).encode()).hexdigest()
        path = state_dir / (url.rsplit("-", 1)[-1] + ".json")
        state = json.loads(path.read_text()) if path.exists() else {
            "envelope_sha256": identity, "attempts": []}
        if state.get("envelope_sha256") != identity:
            raise ValueError("envelope changed after first attempt")
        attempts = state["attempts"]
        used = {row["run_id"]: row for row in attempts}
        if not set(used).issubset(envelope["run_ids"]) or len(used) != len(attempts):
            raise ValueError("invalid attempt journal")
        if run_id == NEXT_RUN_ID:
            run_id = next((x for x in envelope["run_ids"] if x not in used), None)
            if run_id is None:
                raise ValueError("run id burned or attempt cap reached: every declared id is used")
        spent, stalled = Decimal(0), set()
        ledger = receipts / "receipts/experts4bit-qlora/ledger.jsonl"
        # Match the controller's ledger_check.load_ledger_entries exactly:
        # comments and blank lines are legal; malformed entries still raise.
        ledger_rows = []
        for raw_line in committed_bytes(receipts, ledger).decode().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            ledger_rows.append(json.loads(line))
        for candidate in envelope["run_ids"]:
            paths = receipt_paths(receipts, candidate)
            dirs = list((receipts / "receipts/experts4bit-qlora").glob(f"*/{candidate}"))
            rows = [row for row in ledger_rows if row.get("run_id") == candidate]
            prior = used.get(candidate)
            if prior is None:
                if paths or dirs or rows:
                    raise ValueError("run id already used outside this envelope journal")
                continue
            if prior.get("phase") == "not-invoked":
                if paths or dirs or rows:
                    raise ValueError("not-invoked attempt has unexpected receipt, run dir or ledger row")
                continue
            if prior.get("phase") not in ("admitted", "controller-invoked"):
                raise ValueError("invalid attempt journal phase")
            if len(paths) != 1 or len(dirs) != 1 or len(rows) != 1:
                raise ValueError("created instance or uncertain prior attempt: stop for maintainer")
            receipt = json.loads(committed_bytes(receipts, paths[0]))
            if rows[0].get("cost_usd") == 0 and zero_instance_refusal(receipt, url, candidate):
                continue
            stall = preflight_stall(receipt, url, candidate)
            try:
                journaled = number(rows[0].get("cost_usd"), "journaled attempt cost")
            except ValueError:
                journaled = None
            if stall is None or journaled != stall[1]:
                raise ValueError("created instance or uncertain prior attempt: stop for maintainer")
            stalled.add(stall[0])
            spent += stall[1]
        if run_id in used or len(used) >= envelope["maximum_attempts"]:
            raise ValueError("run id burned or attempt cap reached")
        if spent >= ALL_IN_CAP:
            raise ValueError("the envelope's cumulative all-in is spent: stop for maintainer")
        row = {"run_id": run_id, "utc": now(), "phase": "admitted"}
        attempts.append(row)
        write_attempt_state(path, state)

        def controller_invoked():
            if row["phase"] != "admitted":
                raise ValueError("controller already invoked")
            row["phase"] = "controller-invoked"
            write_attempt_state(path, state)

        try:
            yield Permit(run_id, spent, stalled, controller_invoked)
        except Exception:
            # A caught failure before the durable controller boundary proves no
            # controller call occurred. Process death/interrupt leaves admitted.
            if row["phase"] == "admitted":
                row["phase"] = "not-invoked"
                write_attempt_state(path, state)
            raise


def launch(template_path: Path, envelope_path: Path, e4b: Path, controller: Path,
           assets: Path, packet_dir: Path, *, run_id: str) -> int:
    """Run one fresh attempt under the range envelope; one created instance ends it."""
    envelope = json.loads(envelope_path.read_text())
    validate_envelope(envelope, envelope_path, template_path, e4b, controller)
    verify_envelope_comment(envelope)
    receipts = Path(os.environ.get("RECEIPTS_REPO", str(Path.home() / "adertha-receipts")))
    with attempt_permit(envelope, run_id, receipts,
                        Path.home() / ".local/state/dq11-census-envelopes") as permit:
        return launch_attempt(template_path, envelope_path, e4b, controller, assets, packet_dir,
                              run_id=permit.run_id, before_controller=permit)


def bind_manifest(template_path: Path, envelope_path: Path, summary: dict,
                  e4b: Path, controller: Path, assets: Path, output: Path, *, run_id: str,
                  excluded=EXCLUDED_MACHINES) -> dict:
    """Pin by inclusion to exactly one qualified machine and its observed union."""
    if output.exists() or output.is_relative_to(e4b) or output.is_relative_to(controller):
        raise ValueError("bound manifest must be write-once and outside source")
    envelope = json.loads(envelope_path.read_text())
    validate_envelope(envelope, envelope_path, template_path, e4b, controller)
    if summary.get("status") != "PASS" or not summary.get("offer_ids"):
        raise ValueError("no qualifying all-host window")
    machine = str(summary["machine_id"])
    offers = [str(x) for x in summary["offer_ids"]]
    driver = summary["driver"]
    if (not machine.isdigit() or machine in EXCLUDED_MACHINES or machine in excluded
            or len(set(offers)) != len(offers) or not all(x.isdigit() for x in offers)
            or not isinstance(driver, str)
            or not re.fullmatch(r"[0-9]{3}\.[0-9]{1,3}\.[0-9]{1,3}", driver)
            or int(driver.split(".")[0]) < 580):
        raise ValueError("qualifying host cannot be bound to an exact inclusion pin")
    if run_id not in envelope["run_ids"]:
        raise ValueError("run id is outside the envelope")
    template = json.loads(template_path.read_text())
    manifest = attempt_template(template, run_id)
    lane = manifest["lane_record"]       # the launcher refuses unknown top-level keys (adertha-agents#243): ours live here
    lane["status"] = "ENVELOPE_BOUND_PENDING_FRESH_GATES"
    manifest["heads"] = {"e4b": merged_clean(e4b), "adertha": merged_clean(controller)}
    manifest["command"] = (f"DQ11_ASSET_DIR={assets.resolve()} DQ11_REQUIRED_DRIVER={driver} "
                           "bash bench/dq11/dq11_l_census_drive.sh")
    manifest["vast_pin_machine"] = machine
    manifest["vast_pin_offers"] = offers
    manifest["vast_required_driver"] = driver
    directory = e4b / "bench/dq11"
    detail = lane["l_census"]
    detail["instrument_sha256"] = {name: sha(directory / name) for name in INSTRUMENT_FILES}
    detail["operational_sha256"] = {name: sha(directory / name) for name in OPERATIONAL_FILES}
    detail["transport_sha256"] = sha(e4b / TRANSPORT_FILE)
    detail["manifest_template_sha256"] = sha(template_path)
    lane["authorization_basis"] = {
        "envelope_ack_url": envelope["approval_url"],
        "envelope_sha256": sha(envelope_path), "issued_at": envelope["issued_at"],
        "executor_sha256": envelope["executor_sha256"],
        "expires_at": envelope["expires_at"], "maximum_draws": 1,
        "run_ids": envelope["run_ids"], "maximum_attempts": envelope["maximum_attempts"],
        "qualifying_window_sha256": summary["stable_history_sha256"],
        "all_host_polls_sha256": summary["polls_sha256"],
    }
    validate_manifest(manifest, e4b, controller)
    manifest.pop("_asset_path")  # validation scratch, never a manifest field
    with output.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest


def validate_manifest(manifest: dict, e4b: Path, controller: Path) -> None:
    check_static_budget()
    try:
        rate = number(manifest.get("usd_per_hour"), "manifest hourly cap")
        hours = number(manifest.get("wallclock_h"), "manifest wall cap")
        estimate = number(manifest.get("est_usd"), "manifest runtime/storage estimate")
        download_gb = number(manifest.get("download_gb"), "manifest download GB")
        down_rate = number(manifest.get("max_download_usd_per_gb"), "manifest download tariff")
    except ValueError as exc:
        raise ValueError("manifest lacks numeric L-census limits") from exc
    if (manifest.get("heads") != {"e4b": merged_clean(e4b), "adertha": merged_clean(controller)}
            or manifest.get("work_id") != "experts4bit-qlora#1083"
            or manifest.get("role") != "CTO"
            or manifest.get("agent_env") != "POD_LAUNCH_AGENT"
            or not valid_run_id(manifest.get("run_id"))
            or manifest.get("preregistration") != "bench/dq11/DQ11-L-CENSUS-PREREG.md"
            or manifest.get("vast_direct_ssh") is not True
            or manifest.get("provider") != "vast:verified-secure"
            or manifest.get("gpu") != "RTX 5090"
            or manifest.get("image") != IMAGE
            or manifest.get("vast_pin_machine") in (None, "")
            or not isinstance(manifest.get("vast_pin_offers"), list)
            or not manifest["vast_pin_offers"]
            or "vast_pin_offer" in manifest
            or manifest.get("storage") != DISK_GB
            or rate != HOURLY_CAP
            or hours != WALL_HOURS
            or estimate != HOURLY_CAP * WALL_HOURS
            or download_gb != DOWNLOAD_GB
            or down_rate != DOWNLOAD_RATE_CAP
            or manifest.get("command_cwd") != "e4b"):
        raise ValueError("manifest is not the registered pinned L-census launch")
    command = manifest.get("command")
    match = re.fullmatch(r"DQ11_ASSET_DIR=(/[A-Za-z0-9._/-]+) DQ11_REQUIRED_DRIVER=([0-9]{3}\.[0-9]{1,3}\.[0-9]{1,3}) bash bench/dq11/dq11_l_census_drive\.sh", command or "")
    if not match:
        raise ValueError("manifest command must bind the canonical asset path and L-census driver")
    manifest["_asset_path"] = match.group(1)
    lane = manifest.get("lane_record")
    if not isinstance(lane, dict) or not isinstance(lane.get("l_census"), dict):
        raise TypeError("manifest lacks L-census closure")
    if set(lane) != LANE_RECORD_FIELDS:
        raise ValueError("bound manifest lane_record has unregistered fields")
    detail = lane["l_census"]
    try:
        upload_gb = number(detail.get("upload_gb"), "manifest upload GB")
        upload_rate = number(detail.get("max_upload_usd_per_gb"), "manifest upload tariff")
        all_in = number(detail.get("all_in_cap_usd"), "manifest all-in cap")
    except ValueError as exc:
        raise ValueError("manifest lacks numeric L-census transport limits") from exc
    if (detail.get("science_head") != SCIENCE
            or detail.get("science_manifest_sha256") != SCIENCE_MANIFEST_SHA
            or detail.get("raw_trace_max_bytes") != RAW_TRACE_MAX
            or detail.get("packet_max_bytes") != PACKET_MAX
            or upload_gb != UPLOAD_GB
            or upload_rate != UPLOAD_RATE_CAP
            or all_in != ALL_IN_CAP):
        raise ValueError("manifest lacks exact L-census science/transfer caps")
    template_path = e4b / TEMPLATE_FILE
    if detail.get("manifest_template_sha256") != sha(template_path):
        raise ValueError("bound manifest template differs from merged source")
    template = attempt_template(json.loads(template_path.read_text()), manifest["run_id"])
    actual_template = {key: deepcopy(value) for key, value in manifest.items()
                       if key not in BINDING_FIELDS | {"_asset_path"}}
    actual_template["lane_record"] = deepcopy(template["lane_record"])     # its bound fields are checked below
    registered = json.loads(template_path.read_text())
    assert_attempt_template(actual_template, registered, manifest["run_id"])
    if set(detail) != set(registered["lane_record"]["l_census"]) | {
            "instrument_sha256", "operational_sha256", "transport_sha256", "manifest_template_sha256"}:
        raise ValueError("bound manifest has unregistered closure fields")
    if lane.get("status") != "ENVELOPE_BOUND_PENDING_FRESH_GATES":
        raise ValueError("manifest was not bound under an envelope")
    if any(manifest.get(key) != value for key, value in template.items() if key != "lane_record"):
        raise ValueError("bound manifest changed a registered template field")
    if any(detail.get(key) != value for key, value in template["lane_record"]["l_census"].items()):
        raise ValueError("bound manifest changed a registered L-census template field")
    machine = str(manifest["vast_pin_machine"])
    offers = [str(x) for x in manifest["vast_pin_offers"]]
    if (not machine.isdigit() or machine in EXCLUDED_MACHINES or len(set(offers)) != len(offers)
            or not all(x.isdigit() for x in offers)):
        raise ValueError("invalid exact machine/offer-set pin")
    driver = manifest.get("vast_required_driver")
    if (not isinstance(driver, str)
            or not re.fullmatch(r"[0-9]{3}\.[0-9]{1,3}\.[0-9]{1,3}", driver)
            or int(driver.split(".")[0]) < 580
            or match.group(2) != driver):
        raise ValueError("exact approved driver 580+ required")
    instrument_dir = e4b / "bench/dq11"
    if not (e4b / "bench/dq11/DQ11-L-CENSUS-LAUNCH.md").is_file():
        raise ValueError("launch amendment is absent from source")
    if subprocess.call(["git", "-C", str(e4b), "merge-base", "--is-ancestor", INSTRUMENT_MERGE, "HEAD"]):
        raise ValueError("merged instrument is not an ancestor")
    if subprocess.call(["git", "-C", str(controller), "merge-base", "--is-ancestor", OFFER_SET_MERGE, "HEAD"]):
        raise ValueError("merged offer-set controller is not an ancestor")
    if sha(instrument_dir / "science.sha256") != SCIENCE_MANIFEST_SHA:
        raise ValueError("13b science closure changed")
    for name in INSTRUMENT_FILES:
        if not (instrument_dir / name).is_file():
            raise ValueError("missing instrument file " + name)
    if detail.get("instrument_sha256") != {name: sha(instrument_dir / name) for name in INSTRUMENT_FILES}:
        raise ValueError("manifest instrument bytes differ from merged source")
    if detail.get("operational_sha256") != {name: sha(instrument_dir / name) for name in OPERATIONAL_FILES}:
        raise ValueError("manifest operational bytes differ from merged source")
    if detail.get("transport_sha256") != sha(e4b / TRANSPORT_FILE):
        raise ValueError("manifest transport controller bytes differ from merged source")


def cpu_gate(manifest_path: Path, e4b: Path, controller: Path, assets: Path, output: Path) -> dict:
    if output.exists() or output.is_relative_to(e4b):
        raise ValueError("write-once gate output must be outside source checkout")
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest, e4b, controller)
    if assets.resolve() != Path(manifest["_asset_path"]).resolve():
        raise ValueError("CPU gate assets differ from bound launch command")
    directory = e4b / "bench/dq11"
    stage(directory, assets)
    report = preflight_urls(directory)
    policy = preflight_policy(directory, SCIENCE)
    report.update(
        schema="dq11-l-census-cpu-gate/1", utc=now(), manifest_sha256=sha(manifest_path),
        e4b_head=manifest["heads"]["e4b"], controller_head=manifest["heads"]["adertha"],
        science_source=SCIENCE, science_manifest_sha256=SCIENCE_MANIFEST_SHA,
        instrument={name: sha(directory / name) for name in INSTRUMENT_FILES},
        operational={name: sha(directory / name) for name in OPERATIONAL_FILES},
        transport_sha256=sha(e4b / TRANSPORT_FILE),
        cpu_policy=policy,
    )
    report["passed"] = report["passed"] is True and policy["passed"] is True
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    if not report["passed"]:
        raise ValueError("fresh URL/CPU-policy gate refused before quote or rent")
    return report


def quote(manifest_path: Path, history_path: Path, gate_path: Path, provider, output: Path, *,
          excluded=EXCLUDED_MACHINES, all_in: Decimal = ALL_IN_CAP) -> dict:
    if output.exists():
        raise ValueError("quote output must be write-once")
    manifest, history, gate = [json.loads(p.read_text()) for p in (manifest_path, history_path, gate_path)]
    machine = str(manifest["vast_pin_machine"])
    approved = [str(x) for x in manifest["vast_pin_offers"]]
    stable_ids(history["samples"], machine, approved)
    if any(row["driver_version"] != manifest.get("vast_required_driver")
           for sample in history["samples"] for row in sample["offers"]):
        raise ValueError("approved history includes a different exact driver")
    if (history.get("machine_id") != machine or history.get("offer_ids") != approved
            or gate.get("passed") is not True or gate.get("manifest_sha256") != sha(manifest_path)):
        raise ValueError("history/gate do not bind exact approved manifest")
    sample = observation(provider, machine, excluded=excluded, all_in=all_in)
    rows = {r["id"]: r for r in sample["offers"]}
    if len(rows) != len(sample["offers"]):
        raise ValueError("duplicate offer identity in final search")
    live_ids = set(rows)
    if not live_ids or live_ids - set(approved) or any(not rows[oid]["eligible"] for oid in live_ids):
        raise ValueError("final search has absent, outside or ineligible offer")
    if any(rows[oid]["driver_version"] != manifest.get("vast_required_driver") for oid in live_ids):
        raise ValueError("approved offer changed its exact driver")
    result = {
        "schema": "dq11-l-census-quote/1", "mode": "READ_ONLY_CONTROLLER_SEARCH_NO_CREATE",
        "utc": sample["utc"], "monotonic": sample["monotonic"],
        "manifest_sha256": sha(manifest_path), "history_sha256": sha(history_path),
        "gate_sha256": sha(gate_path), "machine_id": machine, "offer_ids": approved,
        "offers": [rows[oid] for oid in approved if oid in live_ids], "all_in_cap_usd": str(ALL_IN_CAP),
        "all_in_remaining_usd": str(all_in), "excluded_machines": sorted(excluded),
        "worst_registered_reservation_usd": str(reserve(HOURLY_CAP, DOWNLOAD_RATE_CAP, UPLOAD_RATE_CAP)),
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def controller_call(command: list[str], env: dict, packet_dir: Path, name: str) -> int:
    """Keep exact output even when the controller refuses before making a receipt."""
    stdout_path = packet_dir / f"{name}.stdout.txt"
    stderr_path = packet_dir / f"{name}.stderr.txt"
    with stdout_path.open("x") as stdout, stderr_path.open("x") as stderr:
        result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, check=False)
    with (packet_dir / f"{name}.json").open("x") as stream:
        json.dump({"utc": now(), "command": command, "returncode": result.returncode,
                   "stdout_sha256": sha(stdout_path), "stderr_sha256": sha(stderr_path)}, stream, indent=2)
        stream.write("\n")
    return result.returncode


def launch_attempt(template_path: Path, envelope_path: Path, e4b: Path, controller: Path,
                   assets: Path, packet_dir: Path, *, run_id: str, before_controller=None) -> int:
    """Under one envelope, bind the first qualified host and launch once."""
    agent = os.environ.get("POD_LAUNCH_AGENT") or ""
    if not EXECUTOR_RE.fullmatch(agent):
        raise ValueError("POD_LAUNCH_AGENT must identify the executing CTO/... seat before discovery")
    envelope = json.loads(envelope_path.read_text())
    validate_envelope(envelope, envelope_path, template_path, e4b, controller)
    if executor_sha256(agent) != envelope.get("executor_sha256"):
        raise ValueError("POD_LAUNCH_AGENT is not the executor this envelope binds; nothing searched or created")
    verify_envelope_comment(envelope)
    if packet_dir.exists() or packet_dir.is_relative_to(e4b) or packet_dir.is_relative_to(controller):
        raise ValueError("write-once launch packet must be outside source checkouts")
    # the envelope journal's bounds for this attempt (a Permit from attempt_permit; the registered constants otherwise)
    excluded = frozenset(EXCLUDED_MACHINES) | getattr(before_controller, "excluded_machines", frozenset())
    all_in = getattr(before_controller, "budget_usd", ALL_IN_CAP)
    spent = getattr(before_controller, "spent_usd", Decimal(0))
    provider = live_provider(None, controller, e4b, excluded=excluded)
    packet_dir.mkdir(mode=0o700, parents=False)
    summary = observe_all_hosts(provider, packet_dir, excluded=excluded, all_in=all_in)
    history = check_qualifying_window(packet_dir, summary)
    manifest_path = packet_dir / "bound-manifest.json"
    history_path = packet_dir / "stable-host.json"
    gate_path = packet_dir / "fresh-cpu-gate.json"
    quote_path = packet_dir / "final-quote.json"
    manifest = bind_manifest(template_path, envelope_path, summary, e4b, controller,
                             assets, manifest_path, run_id=run_id, excluded=excluded)
    env = dict(os.environ, E4B_REPO=str(e4b), ADERTHA_REPO=str(controller))
    command = ["bash", str(controller / "tools/pod-launch.sh"), str(manifest_path), envelope["approval_url"]]
    validation = [*command[:2], "--validate", *command[2:]]
    if controller_call(validation, env, packet_dir, "controller-validate"):
        raise ValueError("controller --validate refused; see packet output; nothing created")
    cpu_gate(manifest_path, e4b, controller, assets, gate_path)
    gate_end = time.monotonic()
    quote(manifest_path, history_path, gate_path, provider, quote_path, excluded=excluded, all_in=all_in)
    invocation = time.monotonic()
    validate_envelope(envelope, envelope_path, template_path, e4b, controller)
    if (invocation - history["samples"][-1]["monotonic"] > FRESH_SECONDS
            or invocation - gate_end > FRESH_SECONDS):
        raise ValueError("gate/quote/launch exceeded 60-second stable-host window; nothing created")
    with (packet_dir / "launch-admission.json").open("x") as stream:
        json.dump({"schema": "dq11-l-census-launch-admission/1", "utc": now(),
                   "manifest_sha256": sha(manifest_path), "history_sha256": sha(history_path),
                   "gate_sha256": sha(gate_path), "quote_sha256": sha(quote_path),
                   "envelope_ack_url": envelope["approval_url"],
                   "envelope_sha256": sha(envelope_path),
                   "all_host_polls_sha256": summary["polls_sha256"],
                   "qualifying_window_sha256": summary["stable_history_sha256"],
                   "selected_machine": manifest["vast_pin_machine"],
                   "selected_offer_ids": manifest["vast_pin_offers"],
                   "envelope_spent_usd": str(spent), "envelope_all_in_remaining_usd": str(all_in),
                   "excluded_machines": sorted(excluded),
                   "from_last_search_seconds": invocation - history["samples"][-1]["monotonic"],
                   "from_gate_seconds": invocation - gate_end,
                   "guarded_controller_invoked": True}, stream, indent=2)
        stream.write("\n")
    if sha(envelope_path) != manifest["lane_record"]["authorization_basis"]["envelope_sha256"]:
        raise ValueError("envelope record changed after manifest binding; nothing created")
    check_qualifying_window(packet_dir, summary)
    validate_envelope(envelope, envelope_path, template_path, e4b, controller)
    verify_envelope_comment(envelope)
    actual_call = time.monotonic()
    if (actual_call - history["samples"][-1]["monotonic"] > FRESH_SECONDS
            or actual_call - gate_end > FRESH_SECONDS):
        raise ValueError("gate/quote/launch exceeded 60-second stable-host window; nothing created")
    if before_controller is not None:
        before_controller()  # fsync the irreversible boundary before any paid controller call
    return controller_call(command, env, packet_dir, "controller")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    host = sub.add_parser("observe", help="five-minute read-only controller offer search")
    host.add_argument("--machine", required=True)
    host.add_argument("--e4b", type=Path, required=True)
    host.add_argument("--controller", type=Path, required=True)
    host.add_argument("--offer", action="append", default=[])
    host.add_argument("--out", type=Path, required=True)
    size = sub.add_parser("size", help="hash and bound a private evidence packet before transfer")
    size.add_argument("--packet", type=Path, required=True)
    size.add_argument("--out", type=Path, required=True)
    gate = sub.add_parser("gate", help="fresh sealed CPU gate without a quote or rental")
    gate.add_argument("--manifest", type=Path, required=True)
    gate.add_argument("--e4b", type=Path, required=True)
    gate.add_argument("--controller", type=Path, required=True)
    gate.add_argument("--assets", type=Path, required=True)
    gate.add_argument("--out", type=Path, required=True)
    prices = sub.add_parser("quote", help="one fresh read-only quote after gate")
    prices.add_argument("--manifest", type=Path, required=True)
    prices.add_argument("--history", type=Path, required=True)
    prices.add_argument("--gate", type=Path, required=True)
    prices.add_argument("--e4b", type=Path, required=True)
    prices.add_argument("--controller", type=Path, required=True)
    prices.add_argument("--out", type=Path, required=True)
    paid = sub.add_parser("launch", help="all-host five-minute search and same-minute guarded launch under envelope ACK")
    paid.add_argument("--run-id", required=True,
                      help="a declared id, or 'next' for the first one the envelope journal has not used")
    paid.add_argument("--template", type=Path, required=True)
    paid.add_argument("--envelope", type=Path, required=True)
    paid.add_argument("--e4b", type=Path, required=True)
    paid.add_argument("--controller", type=Path, required=True)
    paid.add_argument("--assets", type=Path, required=True)
    paid.add_argument("--packet-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.mode == "observe":
            if args.out.exists():
                raise ValueError("observation output must be write-once")
            provider = live_provider(args.machine, args.controller.resolve(), args.e4b.resolve())
            result = observe(provider, args.machine, args.offer or None)
            with args.out.open("x") as stream:
                json.dump(result, stream, indent=2)
                stream.write("\n")
        elif args.mode == "size":
            if args.out.exists():
                raise ValueError("size output must be write-once")
            count, files = packet_files(args.packet)
            with args.out.open("x") as stream:
                json.dump({"schema": "dq11-l-census-packet-size/1", "utc": now(), "bytes": count,
                           "cap_bytes": PACKET_MAX, "files": files}, stream, indent=2)
                stream.write("\n")
        elif args.mode == "gate":
            cpu_gate(args.manifest, args.e4b.resolve(), args.controller.resolve(), args.assets, args.out)
        elif args.mode == "quote":
            manifest = json.loads(args.manifest.read_text())
            provider = live_provider(str(manifest["vast_pin_machine"]), args.controller.resolve(),
                                     args.e4b.resolve())
            quote(args.manifest, args.history, args.gate, provider, args.out)
        else:
            raise SystemExit(launch(args.template.resolve(), args.envelope.resolve(), args.e4b.resolve(),
                                    args.controller.resolve(), args.assets.resolve(), args.packet_dir.resolve(),
                                    run_id=args.run_id))
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        parser.exit(78, str(exc) + "\n")


if __name__ == "__main__":
    main()
