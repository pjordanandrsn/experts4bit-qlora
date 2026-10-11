"""CPU refusals for the paid census's read-only host and transport gates."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
        "driver_version": "580.159.03", "storage_cost": .2, "reliability2": .99,
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
        {"gpu_ram": None}, {"rentable": False}, {"reliability2": .979},
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


def test_observe_measures_window_from_first_completed_search():
    t = [0.0]

    class FirstSearchLatency(FakeProvider):
        def search_offers(self, gpu):
            if self.calls == 0:
                t[0] += 0.5
            return super().search_offers(gpu)

    def clock():
        return t[0]

    def sleep(seconds):
        t[0] += seconds

    p = FirstSearchLatency([offer()])
    result = launch.observe(p, "200", ["100"], clock=clock, sleep=sleep)
    assert result["samples"][0]["monotonic"] == 0.5
    assert result["samples"][-1]["monotonic"] == 300.5
    assert result["samples"][-1]["monotonic"] - result["samples"][0]["monotonic"] == 300
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


def test_all_host_continuity_selects_first_qualified_and_refuses_outside_envelope():
    continuity = launch.HostContinuity()
    selected = None
    for t in range(0, 301, 20):
        good = launch.offer_row(offer(str(100 + (t // 20) % 2), "200"), FakeProvider([]))
        low_reliability = launch.offer_row(dict(offer("300", "100"), reliability2=.97), FakeProvider([]))
        selected = continuity.step({"utc": "test", "monotonic": float(t), "status": "PASS",
                                    "filter": {}, "offers": [good, low_reliability]})
        if t < 300:
            assert selected is None
    assert selected and selected[0] == "200" and selected[2] == ["100", "101"]
    assert selected[3] == "580.159.03"
    assert "100" not in continuity.active

    continuity = launch.HostContinuity()
    initially_eligible = launch.offer_row(offer("300", "100"), FakeProvider([]))
    continuity.step({"utc": "test", "monotonic": 0.0, "status": "PASS",
                     "filter": {}, "offers": [initially_eligible]})
    below_envelope = launch.offer_row(dict(offer("300", "100"), reliability2=.97), FakeProvider([]))
    continuity.step({"utc": "test", "monotonic": 20.0, "status": "PASS",
                     "filter": {}, "offers": [below_envelope]})
    assert continuity.resets["100"]["ineligible_or_outside"] == 1

    two_part = launch.offer_row(dict(offer(), driver_version="595.84"), FakeProvider([]))
    continuity.step({"utc": "test", "monotonic": 40.0, "status": "PASS",
                     "filter": {}, "offers": [two_part]})
    assert "200" not in continuity.active

    continuity = launch.HostContinuity()
    good = launch.offer_row(offer("100"), FakeProvider([]))
    assert continuity.step({"utc": "test", "monotonic": 0.0, "status": "PASS",
                            "filter": {}, "offers": [good]}) is None
    assert continuity.step({"utc": "test", "monotonic": 20.0, "status": "PASS",
                            "filter": {}, "offers": []}) is None
    assert continuity.resets["200"]["absent"] == 1


def test_all_host_collector_seals_first_five_minute_window_without_create(tmp_path):
    class RotatingProvider(FakeProvider):
        def search_offers(self, gpu):
            self.calls += 1
            return [offer(str(100 + self.calls % 2))]

    tick = [0.0]

    def clock():
        return tick[0]

    def sleep(seconds):
        tick[0] += seconds

    provider = RotatingProvider([])
    summary = launch.observe_all_hosts(provider, tmp_path, clock=clock, sleep=sleep)
    assert summary["status"] == "PASS" and provider.calls == 16
    assert summary["machine_id"] == "200" and summary["offer_ids"] == ["100", "101"]
    history = launch.check_qualifying_window(tmp_path, summary)
    assert history["samples"][-1]["monotonic"] - history["samples"][0]["monotonic"] == 300
    assert summary["polls_sha256"] == launch.sha(tmp_path / "all-host-polls.jsonl")


def test_envelope_exact_criteria_source_template_and_expiry(tmp_path, monkeypatch):
    e4b = tmp_path / "e4b"
    controller = tmp_path / "controller"
    template = e4b / launch.TEMPLATE_FILE
    template.parent.mkdir(parents=True)
    template.write_text("{}")
    controller.mkdir()
    record = tmp_path / "envelope.json"
    record.write_text("{}")
    monkeypatch.setattr(launch, "merged_clean", lambda path: "e" * 40 if path == e4b else "c" * 40)
    issued = datetime(2026, 10, 11, 8, tzinfo=timezone.utc)
    envelope = {"schema": "dq11-l-census-envelope-ack/1",
                "approval_url": "https://github.com/pjordanandrsn/experts4bit-qlora/issues/1083#issuecomment-123",
                "issued_at": issued.isoformat(), "expires_at": (issued + timedelta(hours=24)).isoformat(),
                "e4b_head": "e" * 40, "controller_head": "c" * 40,
                "manifest_template_sha256": launch.sha(template),
                "criteria": launch.ENVELOPE_CRITERIA, "maximum_draws": 1}
    launch.validate_envelope(envelope, record, template, e4b, controller, at=issued + timedelta(hours=1))
    with pytest.raises(ValueError, match="expired"):
        launch.validate_envelope(envelope, record, template, e4b, controller, at=issued + timedelta(hours=24))
    broader = dict(envelope, criteria=dict(launch.ENVELOPE_CRITERIA, minimum_reliability="0.97"))
    with pytest.raises(ValueError, match="outside the exact envelope"):
        launch.validate_envelope(broader, record, template, e4b, controller, at=issued + timedelta(hours=1))
    template.write_text("{\"changed\":true}")
    with pytest.raises(ValueError, match="source or manifest template change"):
        launch.validate_envelope(envelope, record, template, e4b, controller, at=issued + timedelta(hours=1))
    template.write_text("{}")
    monkeypatch.setattr(launch, "merged_clean", lambda path: "new" if path == e4b else "c" * 40)
    with pytest.raises(ValueError, match="source or manifest template change"):
        launch.validate_envelope(envelope, record, template, e4b, controller, at=issued + timedelta(hours=1))


def test_envelope_comment_must_be_maintainer_exact_json(monkeypatch):
    url = "https://github.com/pjordanandrsn/experts4bit-qlora/issues/1083#issuecomment-123"
    envelope = {"approval_url": url, "schema": "dq11-l-census-envelope-ack/1",
                "maximum_draws": 1}
    comment = {"html_url": url, "user": {"login": "pjordanandrsn"},
               "body": json.dumps({key: value for key, value in envelope.items() if key != "approval_url"})}

    def answer(*args, **kwargs):
        assert args[0][-1].endswith("/issues/comments/123")
        return SimpleNamespace(returncode=0, stdout=json.dumps(comment))

    monkeypatch.setattr(launch.subprocess, "run", answer)
    launch.verify_envelope_comment(envelope)
    comment["user"]["login"] = "someone-else"
    with pytest.raises(ValueError, match="differs from the maintainer"):
        launch.verify_envelope_comment(envelope)
    comment["user"]["login"] = "pjordanandrsn"
    comment["body"] = "{}"
    with pytest.raises(ValueError, match="differs from the maintainer"):
        launch.verify_envelope_comment(envelope)


def test_qualifying_window_makes_one_inclusion_pinned_launch(tmp_path, monkeypatch):
    e4b = LANE.parents[1]
    controller = tmp_path / "controller"
    controller.mkdir()
    template = e4b / launch.TEMPLATE_FILE
    envelope = tmp_path / "envelope.json"
    approval_url = "https://github.com/pjordanandrsn/experts4bit-qlora/issues/1083#issuecomment-123"
    envelope.write_text(json.dumps({"approval_url": approval_url, "issued_at": "2026-10-11T08:00:00Z",
                                    "expires_at": "2026-10-12T08:00:00Z"}))
    assets = tmp_path / "assets"
    assets.mkdir()
    packet = tmp_path / "packet"
    monkeypatch.setattr(launch, "validate_envelope", lambda *args: None)
    monkeypatch.setattr(launch, "verify_envelope_comment", lambda *args: None)
    monkeypatch.setattr(launch, "merged_clean", lambda path: "e" * 40 if path == e4b else "c" * 40)
    monkeypatch.setattr(launch.subprocess, "call", lambda *args: 0)
    monkeypatch.setattr(launch, "live_provider", lambda *args: FakeProvider([offer()]))

    def discover(_, packet_dir):
        (packet_dir / "all-host-polls.jsonl").write_text("one synthetic poll\n")
        latest = launch.time.monotonic()
        history = samples("100", "101")
        for index, sample in enumerate(history):
            sample["monotonic"] = latest - 300 + 20 * index
        history_record = {"machine_id": "200", "offer_ids": ["100", "101"],
                          "all_host_polls_sha256": launch.sha(packet_dir / "all-host-polls.jsonl"),
                          "samples": history}
        history_path = packet_dir / "stable-host.json"
        history_path.write_text(json.dumps(history_record))
        return {"status": "PASS", "machine_id": "200", "offer_ids": ["100", "101"],
                "driver": "580.159.03", "polls_sha256": history_record["all_host_polls_sha256"],
                "stable_history_sha256": launch.sha(history_path)}

    monkeypatch.setattr(launch, "observe_all_hosts", discover)
    monkeypatch.setattr(launch, "cpu_gate", lambda *args: args[-1].write_text("{}"))
    monkeypatch.setattr(launch, "quote", lambda *args: args[-1].write_text("{}"))
    calls = []

    def controller_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launch.subprocess, "run", controller_run)
    assert launch.launch(template, envelope, e4b, controller, assets, packet) == 0
    assert len(calls) == 1 and calls[0][-1] == approval_url
    bound = json.loads((packet / "bound-manifest.json").read_text())
    assert bound["vast_pin_machine"] == "200" and bound["vast_pin_offers"] == ["100", "101"]
    assert bound["vast_required_driver"] == "580.159.03"
    assert "_asset_path" not in bound
    admission = json.loads((packet / "launch-admission.json").read_text())
    assert admission["selected_offer_ids"] == ["100", "101"]
    assert admission["from_last_search_seconds"] < 60


def test_paid_controller_refuses_expired_search_before_driver(tmp_path, monkeypatch):
    template = tmp_path / "template.json"
    template.write_text("{}")
    envelope = tmp_path / "envelope.json"
    envelope.write_text(json.dumps({"approval_url": "https://github.com/pjordanandrsn/experts4bit-qlora/issues/1083#issuecomment-1"}))
    monkeypatch.setattr(launch, "validate_envelope", lambda *args: None)
    monkeypatch.setattr(launch, "verify_envelope_comment", lambda *args: None)
    monkeypatch.setattr(launch, "live_provider", lambda *args: FakeProvider([offer()]))
    history = {"samples": [{"monotonic": 0.0}]}

    def discover(_, packet_dir):
        (packet_dir / "stable-host.json").write_text(json.dumps(history))
        return {"polls_sha256": "a" * 64, "stable_history_sha256": "b" * 64}

    monkeypatch.setattr(launch, "observe_all_hosts", discover)
    monkeypatch.setattr(launch, "check_qualifying_window", lambda *args: history)

    def bind(*args):
        args[-1].write_text(json.dumps({"vast_pin_machine": "200", "vast_pin_offers": ["100"]}))
        return {"vast_pin_machine": "200", "vast_pin_offers": ["100"]}

    monkeypatch.setattr(launch, "bind_manifest", bind)

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
        launch.launch(template, envelope, e4b, controller, tmp_path, tmp_path / "packet")


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
