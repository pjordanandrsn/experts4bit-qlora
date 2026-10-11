"""Read-only stable-host packet and guarded DQ11 L-census launch admission.

No command in this module creates compute except ``launch``, which requires a
merged clean source, a bound manifest, an approval permalink, a fresh CPU gate,
five minutes of provider observations and a final pinned quote before it calls
the already-reviewed pod-launch controller.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from dq11_bootstrap import preflight_urls
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
WALL_HOURS = Decimal("0.5")
HOURLY_CAP = Decimal("0.85")
DOWNLOAD_GB = Decimal("27")
DOWNLOAD_RATE_CAP = Decimal("0.011")
UPLOAD_GB = Decimal("3")
UPLOAD_RATE_CAP = Decimal("0.05")
ALL_IN_CAP = Decimal("0.90")
DISK_GB = 320
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"
EXCLUDED_MACHINES = {"45511"}
INSTRUMENT_FILES = (
    "dq11_l_census_run.py", "dq11_l_census_capture.py", "dq11_l_census_reduce.py"
)
OPERATIONAL_FILES = (
    "dq11_l_census_launch.py", "dq11_l_census_box.sh", "dq11_l_census_drive.sh"
)
TRANSPORT_FILE = "bench/tc1/tc1_drive.sh"


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
    if reserve(HOURLY_CAP, DOWNLOAD_RATE_CAP, UPLOAD_RATE_CAP) != Decimal("0.872"):
        raise ValueError("registered all-in budget arithmetic changed")
    if PACKET_MAX > UPLOAD_GB * Decimal(1_000_000_000):
        raise ValueError("upload reservation is smaller than evidence packet cap")


def offer_row(offer: dict, provider) -> dict:
    """Quote one raw controller-search offer; absent price evidence is a refusal."""
    reasons = []
    machine = str(offer.get("machine_id", ""))
    oid = str(offer.get("id", ""))
    if not machine.isdigit() or not oid.isdigit() or machine in EXCLUDED_MACHINES:
        reasons.append("bad or excluded machine/offer identity")
    if offer.get("verification") != "verified" or offer.get("rentable") is not True or offer.get("external") is True:
        reasons.append("outside verified rentable secure provider class")
    if offer.get("gpu_name") != "RTX 5090" or offer.get("num_gpus") != 1:
        reasons.append("not one RTX 5090")
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
        if reserve(hourly, download, upload) > ALL_IN_CAP:
            reasons.append("all-in 30-minute reservation exceeds $0.90")
    return {
        "id": oid, "machine_id": machine, "driver_version": driver,
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
           for a, b in zip(samples, samples[1:])):
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


def live_provider(machine: str, controller: Path, e4b: Path,
                  required_driver: str | None = None):
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
        min_gpu_ram_gb=30, excluded_machine_ids=EXCLUDED_MACHINES,
        download_gb=float(DOWNLOAD_GB), max_download_usd_per_gb=float(DOWNLOAD_RATE_CAP),
        required_driver=required_driver, pinned_machine_id=machine,
    )
    provider.t = ReadOnlyTransport(provider.t)
    return provider


def observation(provider, machine: str, *, clock=time.monotonic) -> dict:
    # One call to the same provider adapter and filter the paid driver uses.
    offers = provider.search_offers("RTX 5090")
    return {
        "utc": now(), "monotonic": clock(), "status": "PASS",
        "machine_id": machine, "filter": provider._filter("RTX 5090"),
        "offers": [offer_row(offer, provider) for offer in offers if str(offer.get("machine_id")) == machine],
    }


def observe(provider, machine: str, required: list[str] | None = None, *, clock=time.monotonic, sleep=time.sleep) -> dict:
    samples = []
    start = clock()
    while True:
        samples.append(observation(provider, machine, clock=clock))
        if clock() - start >= MIN_STABLE_SECONDS:
            break
        sleep(min(20, max(0, MIN_STABLE_SECONDS - (clock() - start))))
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
            or not str(manifest.get("run_id", "")).startswith("dq11-l-census-")
            or manifest.get("preregistration") != "bench/dq11/DQ11-L-CENSUS-PREREG.md"
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
    detail = manifest.get("l_census")
    if not isinstance(detail, dict):
        raise ValueError("manifest lacks L-census closure")
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


def quote(manifest_path: Path, history_path: Path, gate_path: Path, provider, output: Path) -> dict:
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
    sample = observation(provider, machine)
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
        "worst_registered_reservation_usd": str(reserve(HOURLY_CAP, DOWNLOAD_RATE_CAP, UPLOAD_RATE_CAP)),
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def launch(manifest_path: Path, e4b: Path, controller: Path, assets: Path,
           packet_dir: Path, approval: str) -> int:
    """Fresh five-minute search, CPU gate, quote, then the guarded rental."""
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest, e4b, controller)
    if assets.resolve() != Path(manifest["_asset_path"]).resolve():
        raise ValueError("launch assets differ from bound command")
    if not approval or "UNKNOWN" in approval or any(ch.isspace() for ch in approval):
        raise ValueError("fresh exact-machine/offer-set maintainer approval permalink required")
    if packet_dir.exists() or packet_dir.is_relative_to(e4b) or packet_dir.is_relative_to(controller):
        raise ValueError("write-once launch packet must be outside source checkouts")
    machine = str(manifest["vast_pin_machine"])
    approved = [str(x) for x in manifest["vast_pin_offers"]]
    provider = live_provider(machine, controller, e4b, manifest["vast_required_driver"])
    packet_dir.mkdir(mode=0o700, parents=False)
    history_path = packet_dir / "stable-host.json"
    gate_path = packet_dir / "fresh-cpu-gate.json"
    quote_path = packet_dir / "final-quote.json"
    history = observe(provider, machine, approved)
    with history_path.open("x") as stream:
        json.dump(history, stream, indent=2)
        stream.write("\n")
    cpu_gate(manifest_path, e4b, controller, assets, gate_path)
    gate_end = time.monotonic()
    quote(manifest_path, history_path, gate_path, provider, quote_path)
    invocation = time.monotonic()
    if (invocation - history["samples"][-1]["monotonic"] > FRESH_SECONDS
            or invocation - gate_end > FRESH_SECONDS):
        raise ValueError("gate/quote/launch exceeded 60-second stable-host window; nothing created")
    with (packet_dir / "launch-admission.json").open("x") as stream:
        json.dump({"schema": "dq11-l-census-launch-admission/1", "utc": now(),
                   "manifest_sha256": sha(manifest_path), "history_sha256": sha(history_path),
                   "gate_sha256": sha(gate_path), "quote_sha256": sha(quote_path),
                   "approved_machine": machine, "approved_offer_ids": approved,
                   "approval_permalink": approval, "from_last_search_seconds": invocation - history["samples"][-1]["monotonic"],
                   "from_gate_seconds": invocation - gate_end,
                   "guarded_controller_invoked": True}, stream, indent=2)
        stream.write("\n")
    env = dict(os.environ, E4B_REPO=str(e4b), ADERTHA_REPO=str(controller))
    command = ["bash", str(controller / "tools/pod-launch.sh"), str(manifest_path), approval]
    return subprocess.run(command, env=env, check=False).returncode


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
    paid = sub.add_parser("launch", help="five-minute search and same-minute guarded launch; requires fresh ACK")
    paid.add_argument("--manifest", type=Path, required=True)
    paid.add_argument("--e4b", type=Path, required=True)
    paid.add_argument("--controller", type=Path, required=True)
    paid.add_argument("--assets", type=Path, required=True)
    paid.add_argument("--packet-dir", type=Path, required=True)
    paid.add_argument("--approval", required=True)
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
            raise SystemExit(launch(args.manifest.resolve(), args.e4b.resolve(), args.controller.resolve(),
                                    args.assets.resolve(), args.packet_dir.resolve(), args.approval))
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        parser.exit(78, str(exc) + "\n")


if __name__ == "__main__":
    main()
