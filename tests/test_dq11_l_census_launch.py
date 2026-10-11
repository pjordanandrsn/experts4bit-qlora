"""CPU refusals for the paid census's read-only host and transport gates."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

LANE = Path(__file__).parents[1] / "bench/dq11"
sys.path.insert(0, str(LANE))
launch = importlib.import_module("dq11_l_census_launch")


class FakeProvider:
    def __init__(self, offers):
        self.offers = offers
        self.calls = 0

    def _filter(self, gpu):
        assert gpu == "RTX 5090"
        return {"gpu_name": {"eq": gpu}, "disk_space": {"gte": 320}}

    def search_offers(self, gpu):
        self.calls += 1
        return self.offers

    def hourly(self, offer):
        return offer["hourly"]


def offer(oid="100", machine="200", *, down=.011, up=.05, hourly=.85):
    return {
        "id": oid, "machine_id": machine, "verification": "verified", "rentable": True,
        "external": False, "gpu_name": "RTX 5090", "num_gpus": 1,
        "gpu_ram": 32607, "cpu_ram": 131072, "disk_space": 1200,
        "driver_version": "580.159.03", "storage_cost": .2,
        "inet_down_cost": down, "inet_up_cost": up, "hourly": hourly,
    }


def samples(*offer_ids, gap=20):
    rows = [launch.offer_row(offer(oid), FakeProvider([])) for oid in offer_ids]
    return [{"status": "PASS", "machine_id": "200", "monotonic": i * gap,
             "offers": rows} for i in range(16)]


def test_registered_budget_and_tariff_refusal():
    launch.check_static_budget()
    p = FakeProvider([])
    row = launch.offer_row(offer(), p)
    assert row["eligible"] is True and row["all_in_reserved"] == "0.872"
    for changed in (
        {"inet_up_cost": None}, {"inet_down_cost": .012}, {"inet_up_cost": .051},
        {"hourly": .86}, {"storage_cost": None}, {"driver_version": "570.1"},
        {"gpu_ram": None}, {"rentable": False},
    ):
        raw = dict(offer(), **changed)
        assert launch.offer_row(raw, p)["eligible"] is False


def test_stability_is_continuous_machine_with_union_of_approved_rotating_ids():
    history = samples("100", "101")
    for i, sample in enumerate(history):
        sample["offers"] = sample["offers"][i % 2:i % 2 + 1]
    assert launch.stable_ids(history, "200", ["100", "101"]) == ["100", "101"]
    only_a = samples("100")
    assert launch.stable_ids(only_a, "200", ["100", "101"]) == ["100"]
    history[7]["offers"] = [launch.offer_row(offer("102"), FakeProvider([]))]
    with pytest.raises(ValueError, match="outside ACK set"):
        launch.stable_ids(history, "200", ["100", "101"])
    history[7]["offers"] = []
    with pytest.raises(ValueError, match="machine absent"):
        launch.stable_ids(history, "200", ["100", "101"])
    history[7]["offers"] = [dict(launch.offer_row(offer("101"), FakeProvider([])), eligible=False)]
    with pytest.raises(ValueError, match="ineligible offer"):
        launch.stable_ids(history, "200", ["100", "101"])


def test_stability_refuses_gap_short_window_or_driver_change():
    history = samples("100")
    history[-1]["monotonic"] = 299
    with pytest.raises(ValueError, match="less than five"):
        launch.stable_ids(history, "200")
    history = samples("100")
    history[5]["monotonic"] = history[4]["monotonic"] + 31
    with pytest.raises(ValueError, match="gap"):
        launch.stable_ids(history, "200")
    history = samples("100")
    history[5]["offers"] = [dict(history[5]["offers"][0], driver_version="595.84")]
    with pytest.raises(ValueError, match="driver changed"):
        launch.stable_ids(history, "200")


def test_observe_uses_provider_search_and_no_create():
    p = FakeProvider([offer()])
    t = [0.0]

    def clock():
        return t[0]

    def sleep(seconds):
        t[0] += seconds

    result = launch.observe(p, "200", ["100"], clock=clock, sleep=sleep)
    assert result["offer_ids"] == ["100"]
    assert result["samples"][0]["monotonic"] == 0
    assert result["samples"][-1]["monotonic"] == 300
    assert p.calls == 16


def test_packet_cap_checks_uncompressed_bytes_and_links(tmp_path):
    root = tmp_path / "packet"
    root.mkdir()
    (root / "proof.json").write_text("proof")
    size, rows = launch.packet_files(root)
    assert size == 5 and rows["proof.json"]["sha256"] == launch.sha(root / "proof.json")
    (root / "capture-trace.json").open("wb").truncate(launch.RAW_TRACE_MAX + 1)
    with pytest.raises(ValueError, match="raw trace"):
        launch.packet_files(root)
    (root / "capture-trace.json").unlink()
    (root / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        launch.packet_files(root)


def test_read_only_transport_refuses_create():
    class Delegate:
        def request(self, method, path, **kwargs):
            return method, path

    transport = launch.ReadOnlyTransport(Delegate())
    assert transport.request("GET", "/v0/bundles/") == ("GET", "/v0/bundles/")
    with pytest.raises(ValueError, match="read-only"):
        transport.request("PUT", "/v0/asks/123/")


def test_final_quote_accepts_one_rotating_id_and_refuses_outside_id(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"vast_pin_machine": "200", "vast_pin_offers": ["100", "101"],
                                         "vast_required_driver": "580.159.03"}))
    history_path = tmp_path / "history.json"
    history = samples("100", "101")
    for i, sample in enumerate(history):
        sample["offers"] = sample["offers"][i % 2:i % 2 + 1]
    history_path.write_text(json.dumps({"machine_id": "200", "offer_ids": ["100", "101"],
                                        "samples": history}))
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps({"passed": True, "manifest_sha256": launch.sha(manifest_path)}))
    result = launch.quote(manifest_path, history_path, gate_path, FakeProvider([offer("100")]), tmp_path / "quote.json")
    assert result["offer_ids"] == ["100", "101"] and [r["id"] for r in result["offers"]] == ["100"]
    history_path.write_text(json.dumps({"machine_id": "200", "offer_ids": ["100", "101"],
                                        "samples": samples("100")}))
    one_seen = launch.quote(manifest_path, history_path, gate_path, FakeProvider([offer("100")]), tmp_path / "one-seen.json")
    assert one_seen["offer_ids"] == ["100", "101"] and [r["id"] for r in one_seen["offers"]] == ["100"]
    history_path.write_text(json.dumps({"machine_id": "200", "offer_ids": ["100", "101"],
                                        "samples": history}))
    with pytest.raises(ValueError, match="outside or ineligible"):
        launch.quote(manifest_path, history_path, gate_path, FakeProvider([offer("102")]), tmp_path / "outside.json")
    assert not (tmp_path / "outside.json").exists()
    for sample in history[1::2]:
        sample["offers"][0]["driver_version"] = "595.91.07"
    history_path.write_text(json.dumps({"machine_id": "200", "offer_ids": ["100", "101"],
                                        "samples": history}))
    with pytest.raises(ValueError, match="different exact driver"):
        launch.quote(manifest_path, history_path, gate_path, FakeProvider([offer("100")]), tmp_path / "mixed-driver.json")


def test_paid_controller_refuses_expired_search_before_driver(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"vast_pin_machine": "200", "vast_pin_offers": ["100"],
                                    "vast_required_driver": "580.159.03"}))
    monkeypatch.setattr(launch, "validate_manifest", lambda m, *args: m.__setitem__("_asset_path", str(tmp_path)))
    monkeypatch.setattr(launch, "live_provider", lambda *args: FakeProvider([offer()]))
    history = {"samples": [{"monotonic": 0.0}]}
    monkeypatch.setattr(launch, "observe", lambda *args: history)

    def gate(*args):
        args[-1].write_text(json.dumps({"passed": True}))

    def quote(*args):
        args[-1].write_text("{}")

    monkeypatch.setattr(launch, "cpu_gate", gate)
    monkeypatch.setattr(launch, "quote", quote)
    monkeypatch.setattr(launch, "time", SimpleNamespace(monotonic=lambda: 61.0))

    def would_spend(*args, **kwargs):
        raise AssertionError("paid controller was called after freshness expired")

    monkeypatch.setattr(launch.subprocess, "run", would_spend)
    e4b = tmp_path / "e4b"
    controller = tmp_path / "controller"
    e4b.mkdir()
    controller.mkdir()
    with pytest.raises(ValueError, match="exceeded 60-second"):
        launch.launch(manifest, e4b, controller, tmp_path, tmp_path / "packet", "https://approval.example/1")


def test_box_refuses_wrong_science_before_gpu_or_network(tmp_path):
    script = LANE / "dq11_l_census_box.sh"
    nonce = "cpu-refusal"
    result = subprocess.run(
        ["bash", str(script)], cwd=tmp_path,
        env=dict(os.environ, DQ11_W=str(tmp_path), TC1_RUN_NONCE=nonce,
                 TC1_DEADLINE_EPOCH="9999999999", E4B_SHA="wrong"),
        capture_output=True, text=True, timeout=5, check=False,
    )
    assert result.returncode == 78
    assert "science source is not 13b" in result.stdout
    assert (tmp_path / f"TC1_EXIT_CODE.{nonce}").read_text().strip() == "78"
    assert not (tmp_path / "logs/hardware-precheck.log").exists()
