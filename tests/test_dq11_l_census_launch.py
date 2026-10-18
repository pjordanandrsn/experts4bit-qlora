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
RUN_IDS = tuple(f"dq11-l-census-{i}" for i in range(5, 15))


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
        "public_ipaddr": "8.8.8.8", "direct_port_count": 1,
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
    assert launch.WALL_HOURS == launch.Decimal("0.75")
    assert launch.ALL_IN_CAP == launch.Decimal("1.10")
    assert launch.HOURLY_CAP == launch.Decimal("0.85")
    assert launch.ENVELOPE_CRITERIA["maximum_paid_wall_seconds"] == 2700
    assert launch.ENVELOPE_CRITERIA["maximum_all_in_usd"] == "1.10"
    assert launch.ENVELOPE_CRITERIA["maximum_draws"] == 1
    assert launch.reserve(launch.HOURLY_CAP, launch.DOWNLOAD_RATE_CAP,
                          launch.UPLOAD_RATE_CAP) < launch.ALL_IN_CAP
    p = FakeProvider([])
    row = launch.offer_row(offer(), p)
    assert row["eligible"] is True and row["all_in_reserved"] == "1.0845"
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
                "criteria": launch.ENVELOPE_CRITERIA, "maximum_draws": 1,
                "run_ids": list(RUN_IDS), "maximum_attempts": 10,
                "executor_sha256": launch.executor_sha256("CTO/a/seat")}
    launch.validate_envelope(envelope, record, template, e4b, controller, at=issued + timedelta(hours=1))
    for bad in ({"executor_sha256": None}, {"executor_sha256": "CTO/a/seat"}, {"executor_sha256": "0" * 63}):
        with pytest.raises(ValueError, match="bind its one executor"):
            launch.validate_envelope(dict(envelope, **bad), record, template, e4b, controller,
                                     at=issued + timedelta(hours=1))
    unbound = {k: v for k, v in envelope.items() if k != "executor_sha256"}
    with pytest.raises(ValueError, match="bind its one executor"):
        launch.validate_envelope(unbound, record, template, e4b, controller, at=issued + timedelta(hours=1))
    with pytest.raises(ValueError, match="expired"):
        launch.validate_envelope(envelope, record, template, e4b, controller, at=issued + timedelta(hours=24))
    shorter = dict(envelope, expires_at=(issued + timedelta(hours=23)).isoformat())
    with pytest.raises(ValueError, match="exactly 24 hours"):
        launch.validate_envelope(shorter, record, template, e4b, controller, at=issued + timedelta(hours=1))
    broader = dict(envelope, criteria=dict(launch.ENVELOPE_CRITERIA, minimum_reliability="0.97"))
    with pytest.raises(ValueError, match="outside the exact envelope"):
        launch.validate_envelope(broader, record, template, e4b, controller, at=issued + timedelta(hours=1))
    for change in ({"run_ids": list(RUN_IDS[:-1])},
                   {"run_ids": [*RUN_IDS, "dq11-l-census-15"]},
                   {"maximum_attempts": 11}, {"maximum_attempts": True}):
        with pytest.raises(ValueError, match="outside the exact envelope"):
            launch.validate_envelope(dict(envelope, **change), record, template, e4b, controller,
                                     at=issued + timedelta(hours=1))
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
                "maximum_draws": 1, "issued_at": "2026-10-11T08:00:00Z",
                "expires_at": "2026-10-12T08:00:00Z"}
    comment = {"html_url": url, "user": {"login": "pjordanandrsn"},
               "created_at": "2026-10-11T08:00:00Z", "updated_at": "2026-10-11T08:00:00Z",
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
    comment["body"] = json.dumps({key: value for key, value in envelope.items() if key != "approval_url"})
    comment["updated_at"] = "2026-10-11T08:00:01Z"
    with pytest.raises(ValueError, match="edited, backdated or extended"):
        launch.verify_envelope_comment(envelope)
    comment["updated_at"] = comment["created_at"]
    envelope["issued_at"] = "2026-10-11T08:02:01Z"
    comment["body"] = json.dumps({key: value for key, value in envelope.items() if key != "approval_url"})
    with pytest.raises(ValueError, match="edited, backdated or extended"):
        launch.verify_envelope_comment(envelope)
    envelope["issued_at"] = comment["created_at"]
    envelope["expires_at"] = "2026-10-12T08:00:01Z"
    comment["body"] = json.dumps({key: value for key, value in envelope.items() if key != "approval_url"})
    with pytest.raises(ValueError, match="edited, backdated or extended"):
        launch.verify_envelope_comment(envelope)


@pytest.mark.parametrize("validation_rc,controller_rc", [(0, 0), (1, 0), (0, 97)])
def test_qualifying_window_makes_one_inclusion_pinned_launch(tmp_path, monkeypatch, validation_rc, controller_rc):
    monkeypatch.setenv("POD_LAUNCH_AGENT", "CTO/example/executor")
    e4b = LANE.parents[1]
    controller = tmp_path / "controller"
    controller.mkdir()
    template = e4b / launch.TEMPLATE_FILE
    envelope = tmp_path / "envelope.json"
    approval_url = "https://github.com/pjordanandrsn/experts4bit-qlora/issues/1083#issuecomment-123"
    envelope.write_text(json.dumps({"approval_url": approval_url, "issued_at": "2026-10-11T08:00:00Z",
                                    "expires_at": "2026-10-12T08:00:00Z",
                                    "run_ids": list(RUN_IDS), "maximum_attempts": 10,
                                    "executor_sha256": launch.executor_sha256("CTO/example/executor")}))
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
    (controller / "tools").mkdir()
    (controller / "tools/pod-launch.sh").write_text(
        f'if [ "$1" = "--validate" ]; then\n'
        f'  echo "validation output"; echo "validation diagnostic" >&2; exit {validation_rc}\n'
        f'fi\necho "controller output"\n'
        f'echo "REFUSED[97] exact refusal" >&2\nexit {controller_rc}\n'
    )
    if validation_rc:
        with pytest.raises(ValueError, match="controller --validate refused"):
            launch.launch_attempt(template, envelope, e4b, controller, assets, packet, run_id=RUN_IDS[0])
        assert not (packet / "final-quote.json").exists()
        assert not (packet / "launch-admission.json").exists()
        assert not (packet / "controller.json").exists()
        assert (packet / "controller-validate.stderr.txt").read_text() == "validation diagnostic\n"
        assert json.loads((packet / "controller-validate.json").read_text())["returncode"] == 1
        return
    boundaries = []

    def boundary():
        assert (packet / "controller-validate.json").exists()
        assert (packet / "launch-admission.json").exists()
        assert not (packet / "controller.json").exists()
        boundaries.append("durable boundary before paid call")

    assert launch.launch_attempt(template, envelope, e4b, controller, assets, packet,
                                 run_id=RUN_IDS[0], before_controller=boundary) == controller_rc
    assert len(boundaries) == 1
    validation = json.loads((packet / "controller-validate.json").read_text())
    outcome = json.loads((packet / "controller.json").read_text())
    assert "--validate" in validation["command"] and "--validate" not in outcome["command"]
    assert validation["command"][-2:] == outcome["command"][-2:] == [str(packet / "bound-manifest.json"), approval_url]
    assert validation["returncode"] == 0
    assert validation["stdout_sha256"] == launch.sha(packet / "controller-validate.stdout.txt")
    assert (packet / "controller-validate.stdout.txt").read_text() == "validation output\n"
    assert (packet / "controller.stdout.txt").read_text() == "controller output\n"
    assert outcome["returncode"] == controller_rc
    assert outcome["stderr_sha256"] == launch.sha(packet / "controller.stderr.txt")
    assert (packet / "controller.stderr.txt").read_text() == "REFUSED[97] exact refusal\n"
    bound = json.loads((packet / "bound-manifest.json").read_text())
    assert bound["vast_pin_machine"] == "200" and bound["vast_pin_offers"] == ["100", "101"]
    assert bound["vast_required_driver"] == "580.159.03"
    assert "_asset_path" not in bound
    stray = json.loads(json.dumps(bound))
    stray["unexpected_field"] = True
    with pytest.raises(ValueError, match="changed field beyond run_id"):
        launch.validate_manifest(stray, e4b, controller)
    stray = json.loads(json.dumps(bound))
    stray["lane_record"]["l_census"]["unexpected_field"] = True
    with pytest.raises(ValueError, match="unregistered closure"):
        launch.validate_manifest(stray, e4b, controller)
    # adertha-agents#243: the launcher refuses unknown top-level keys, so the census's record is in lane_record
    registered = json.loads(template.read_text())
    assert set(bound) == set(registered) | launch.BINDING_FIELDS
    assert not {"status", "l_census", "authorization_basis"} & set(bound)
    assert set(bound["lane_record"]) == launch.LANE_RECORD_FIELDS
    assert bound["lane_record"]["status"] == "ENVELOPE_BOUND_PENDING_FRESH_GATES"
    assert bound["lane_record"]["authorization_basis"]["envelope_ack_url"] == approval_url
    assert bound["lane_record"]["authorization_basis"]["executor_sha256"] == launch.executor_sha256("CTO/example/executor")
    stray = json.loads(json.dumps(bound))
    stray["lane_record"]["note"] = "x"
    with pytest.raises(ValueError, match="lane_record has unregistered fields"):
        launch.validate_manifest(stray, e4b, controller)
    stray = json.loads(json.dumps(bound))
    stray["l_census"] = stray["lane_record"].pop("l_census")              # the pre-#243 layout refuses
    with pytest.raises(TypeError, match="lacks L-census closure"):
        launch.validate_manifest(stray, e4b, controller)
    admission = json.loads((packet / "launch-admission.json").read_text())
    assert admission["selected_offer_ids"] == ["100", "101"]
    assert admission["from_last_search_seconds"] < 60


def test_paid_controller_refuses_expired_search_before_driver(tmp_path, monkeypatch):
    monkeypatch.setenv("POD_LAUNCH_AGENT", "CTO/example/executor")
    template = tmp_path / "template.json"
    template.write_text("{}")
    envelope = tmp_path / "envelope.json"
    envelope.write_text(json.dumps({"approval_url": "https://github.com/pjordanandrsn/experts4bit-qlora/issues/1083#issuecomment-1",
                                    "executor_sha256": launch.executor_sha256("CTO/example/executor")}))
    monkeypatch.setattr(launch, "validate_envelope", lambda *args: None)
    monkeypatch.setattr(launch, "verify_envelope_comment", lambda *args: None)
    monkeypatch.setattr(launch, "live_provider", lambda *args: FakeProvider([offer()]))
    history = {"samples": [{"monotonic": 0.0}]}

    def discover(_, packet_dir):
        (packet_dir / "stable-host.json").write_text(json.dumps(history))
        return {"polls_sha256": "a" * 64, "stable_history_sha256": "b" * 64}

    monkeypatch.setattr(launch, "observe_all_hosts", discover)
    monkeypatch.setattr(launch, "check_qualifying_window", lambda *args: history)

    def bind(*args, **kwargs):
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

    def validate_only(command, **kwargs):
        if "--validate" not in command:
            would_spend()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launch.subprocess, "run", validate_only)
    e4b = tmp_path / "e4b"
    controller = tmp_path / "controller"
    e4b.mkdir()
    controller.mkdir()
    with pytest.raises(ValueError, match="exceeded 60-second"):
        launch.launch_attempt(template, envelope, e4b, controller, tmp_path, tmp_path / "packet",
                              run_id=RUN_IDS[0])


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


@pytest.mark.parametrize("identity", [None, "", "someone-else", "CTO", "CTO/", "CTO/a b", "cto/example/executor",
                                      "CTO/seat;rm"])
def test_launch_refuses_missing_or_wrong_executor_before_discovery(tmp_path, monkeypatch, identity):
    monkeypatch.delenv("POD_LAUNCH_AGENT", raising=False)
    if identity is not None:
        monkeypatch.setenv("POD_LAUNCH_AGENT", identity)
    with pytest.raises(ValueError, match="POD_LAUNCH_AGENT must identify"):
        launch.launch_attempt(*(tmp_path / name for name in ("template", "envelope", "e4b", "controller", "assets", "packet")),
                              run_id=RUN_IDS[0])
    assert not (tmp_path / "packet").exists()



@pytest.mark.parametrize("identity,refused", [("CTO/a/seat", False), ("CTO/another/seat", True)])
def test_launch_runs_only_as_the_executor_the_envelope_binds(tmp_path, monkeypatch, identity, refused):
    """The envelope binds ONE executor (the SHA-256 of its CTO identity); POD_LAUNCH_AGENT must be exactly that seat,
    checked before any search. A matching seat goes on (to the comment check, stubbed here to stop)."""
    monkeypatch.setenv("POD_LAUNCH_AGENT", identity)
    envelope = tmp_path / "envelope.json"
    envelope.write_text(json.dumps({"executor_sha256": launch.executor_sha256("CTO/a/seat")}))
    monkeypatch.setattr(launch, "validate_envelope", lambda *args: None)

    def comment(_):
        raise ValueError("comment check reached")
    monkeypatch.setattr(launch, "verify_envelope_comment", comment)
    monkeypatch.setattr(launch, "live_provider", lambda *a: pytest.fail("no search before the executor check"))
    with pytest.raises(ValueError) as err:
        launch.launch_attempt(tmp_path / "template", envelope, *(tmp_path / n for n in ("e4b", "controller", "assets", "packet")),
                              run_id=RUN_IDS[0])
    assert ("not the executor this envelope binds" in str(err.value)) is refused
    assert ("comment check reached" in str(err.value)) is (not refused)
    assert not (tmp_path / "packet").exists()

def test_registered_range_template_identity_and_sha256():
    template = LANE / "dq11_l_census_manifest_template.json"
    assert json.loads(template.read_text())["run_id"] == launch.RUN_ID_PLACEHOLDER
    assert launch.sha(template) == "9dd6fdee2b5797fc596fe6693c8abb107b8b18f35ec057e21b7e0d9b7fa211c9"



def test_the_template_keeps_only_launcher_terms_at_top_level():
    """adertha-agents#243: the launcher refuses a top-level key it does not read. The census's own record (its status,
    closure and, once bound, the authorization basis) sits in lane_record, which the launcher copies into the receipt."""
    template = json.loads((LANE / "dq11_l_census_manifest_template.json").read_text())
    assert not {"status", "l_census", "authorization_basis"} & set(template)
    assert set(template["lane_record"]) == {"status", "l_census"}
    assert template["lane_record"]["status"] == "ENVELOPE_TEMPLATE_NO_PAID_GO"


def test_budget_template_matches_registered_wall_and_full_reservation():
    template = json.loads((LANE / "dq11_l_census_manifest_template.json").read_text())
    assert template["vast_direct_ssh"] is True
    assert launch.ENVELOPE_CRITERIA["vast_direct_ssh"] is True
    assert template["wallclock_h"] == .75
    assert template["est_usd"] == .6375
    assert template["lane_record"]["l_census"]["all_in_cap_usd"] == 1.1
    assert template["lane_record"]["l_census"]["upload_gb"] == 3
    assert template["lane_record"]["l_census"]["max_upload_usd_per_gb"] == .05


def test_static_budget_refuses_changed_upload_allowance(monkeypatch):
    monkeypatch.setattr(launch, "UPLOAD_GB", launch.Decimal(0))
    with pytest.raises(ValueError, match="budget arithmetic changed"):
        launch.check_static_budget()



FIXTURE = Path(__file__).parent / "fixtures/dq11-census-refused"


def range_envelope(ids=RUN_IDS):
    return {"approval_url": "https://github.com/pjordanandrsn/experts4bit-qlora/issues/1083#issuecomment-123",
            "run_ids": list(ids), "maximum_attempts": len(ids)}


def git_fixture(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)


@pytest.fixture
def receipt_repo(tmp_path):
    git_fixture(tmp_path, "init")
    git_fixture(tmp_path, "config", "user.name", "Receipt fixture")
    git_fixture(tmp_path, "config", "user.email", "fixture@example.invalid")
    ledger = tmp_path / "receipts/experts4bit-qlora/ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("")
    git_fixture(tmp_path, "add", ".")
    git_fixture(tmp_path, "commit", "-m", "Initial ledger")
    return tmp_path


def refusal_receipt(root, run_id, *, commit=True, **changes):
    receipt = json.loads((FIXTURE / "receipt.json").read_text())
    receipt["experiment_id"] = run_id
    receipt["approvals"][0]["approval_url"] = range_envelope()["approval_url"]
    receipt.update(changes)
    path = root / "receipts/experts4bit-qlora/2026-10-17" / run_id / "receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt))
    ledger = root / "receipts/experts4bit-qlora/ledger.jsonl"
    row = json.loads((FIXTURE / "ledger.jsonl").read_text())
    row["run_id"] = run_id
    with ledger.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    if commit:
        git_fixture(root, "add", "receipts")
        git_fixture(root, "commit", "-m", "Refusal receipt")
    return path


def test_template_substitution_changes_only_run_id():
    template = json.loads((LANE / "dq11_l_census_manifest_template.json").read_text())
    for run_id in (*RUN_IDS, "dq11-l-census-88"):
        attempt = launch.attempt_template(template, run_id)
        launch.assert_attempt_template(attempt, template, run_id)
        attempt["stray"] = True
        with pytest.raises(ValueError, match="changed field"):
            launch.assert_attempt_template(attempt, template, run_id)
    for invalid in ("../run", "dq11-l-census-five", None):
        with pytest.raises(ValueError, match="unregistered"):
            launch.attempt_template(template, invalid)


def test_zero_instance_refusal_burns_id_but_allows_fresh_attempt(receipt_repo):
    root = receipt_repo
    state = root / "state"
    with launch.attempt_permit(range_envelope(), RUN_IDS[0], root, state) as invoked:
        invoked()
    refusal_receipt(root, RUN_IDS[0])
    with pytest.raises(ValueError, match="burned"):
        with launch.attempt_permit(range_envelope(), RUN_IDS[0], root, state):
            pytest.fail("reused run id")
    with launch.attempt_permit(range_envelope(), RUN_IDS[1], root, state):
        pass
    rows = json.loads((state / "123.json").read_text())["attempts"]
    assert [row["run_id"] for row in rows] == list(RUN_IDS[:2])


@pytest.mark.parametrize("changes", [
    {"instance_id": "123", "status": "ERROR"},
    {"complete": False}, {"cost_usd": {"actual": .001}},
    {"experiment_id": "other-run"}, {"work_id": "other-work"},
    {"approvals": [{"approval_url": "another approval"}]},
    {"teardown_proof": {"complete": True, "method": "destroy", "instance_id": "none"}},
])
def test_created_or_uncertain_receipt_blocks_any_next_draw(receipt_repo, changes):
    root = receipt_repo
    with launch.attempt_permit(range_envelope(), RUN_IDS[0], root, root / "state") as invoked:
        invoked()
    refusal_receipt(root, RUN_IDS[0], **changes)
    with pytest.raises(ValueError, match="created instance or uncertain"):
        with launch.attempt_permit(range_envelope(), RUN_IDS[1], root, root / "state"):
            pytest.fail("next draw allowed")


@pytest.mark.parametrize("invoked", [False, True])
def test_process_death_without_receipt_fails_closed(receipt_repo, invoked):
    root = receipt_repo
    with launch.attempt_permit(range_envelope(), RUN_IDS[0], root, root / "state") as boundary:
        if invoked:
            boundary()
    with pytest.raises(ValueError, match="uncertain"):
        with launch.attempt_permit(range_envelope(), RUN_IDS[1], root, root / "state"):
            pytest.fail("uncertain attempt retried")


@pytest.mark.parametrize("reason", ["no host by cutoff", "validate refusal", "60-second window", "envelope changed"])
def test_caught_pre_controller_exit_marks_not_invoked_and_allows_next_id(receipt_repo, reason):
    root = receipt_repo
    with pytest.raises(ValueError, match=reason):
        with launch.attempt_permit(range_envelope(), RUN_IDS[0], root, root / "state"):
            raise ValueError(reason)
    row = json.loads((root / "state/123.json").read_text())["attempts"][0]
    assert row["phase"] == "not-invoked"
    with launch.attempt_permit(range_envelope(), RUN_IDS[1], root, root / "state"):
        pass


def test_controller_exception_stays_uncertain(receipt_repo):
    root = receipt_repo
    with pytest.raises(ValueError, match="controller died"):
        with launch.attempt_permit(range_envelope(), RUN_IDS[0], root, root / "state") as invoked:
            invoked()
            raise ValueError("controller died")
    with pytest.raises(ValueError, match="uncertain"):
        with launch.attempt_permit(range_envelope(), RUN_IDS[1], root, root / "state"):
            pass


def test_concurrent_attempts_are_exclusive(receipt_repo):
    root = receipt_repo
    with launch.attempt_permit(range_envelope(), RUN_IDS[0], root, root / "state"):
        with pytest.raises(ValueError, match="active"):
            with launch.attempt_permit(range_envelope(), RUN_IDS[1], root, root / "state"):
                pytest.fail("concurrent attempt")


def test_ten_refusals_exhaust_range(receipt_repo):
    root = receipt_repo
    for run_id in RUN_IDS:
        with launch.attempt_permit(range_envelope(), run_id, root, root / "state") as invoked:
            invoked()
        refusal_receipt(root, run_id)
    with pytest.raises(ValueError, match="burned or attempt cap"):
        with launch.attempt_permit(range_envelope(), RUN_IDS[-1], root, root / "state"):
            pytest.fail("eleventh attempt")


@pytest.mark.parametrize("kind", ["receipt", "ledger", "directory"])
def test_fresh_envelope_refuses_any_preexisting_run_id(receipt_repo, kind):
    root = receipt_repo
    if kind == "receipt":
        refusal_receipt(root, RUN_IDS[0])
    elif kind == "ledger":
        ledger = root / "receipts/experts4bit-qlora/ledger.jsonl"
        ledger.write_text(json.dumps({"run_id": RUN_IDS[0], "cost_usd": 0}) + "\n")
        git_fixture(root, "add", "receipts")
        git_fixture(root, "commit", "-m", "Orphan ledger")
    else:
        (root / "receipts/experts4bit-qlora/2026-10-17" / RUN_IDS[0]).mkdir(parents=True)
    with pytest.raises(ValueError, match="already used"):
        with launch.attempt_permit(range_envelope(), RUN_IDS[1], root, root / "state"):
            pass


@pytest.mark.parametrize("dirty", ["untracked-receipt", "modified-receipt", "staged-receipt", "ledger"])
def test_uncommitted_proofs_refuse(receipt_repo, dirty):
    root = receipt_repo
    with launch.attempt_permit(range_envelope(), RUN_IDS[0], root, root / "state") as invoked:
        invoked()
    path = refusal_receipt(root, RUN_IDS[0], commit=dirty != "untracked-receipt")
    if dirty in ("modified-receipt", "staged-receipt"):
        path.write_text(path.read_text() + "\n")
        if dirty == "staged-receipt":
            git_fixture(root, "add", "receipts")
    if dirty == "ledger":
        with (root / "receipts/experts4bit-qlora/ledger.jsonl").open("a") as stream:
            stream.write("\n")
    with pytest.raises(ValueError, match="tracked and clean"):
        with launch.attempt_permit(range_envelope(), RUN_IDS[1], root, root / "state"):
            pass


def test_changed_envelope_cannot_reset_attempt_journal(receipt_repo):
    root = receipt_repo
    with launch.attempt_permit(range_envelope(), RUN_IDS[0], root, root / "state"):
        pass
    with pytest.raises(ValueError, match="envelope changed"):
        with launch.attempt_permit(dict(range_envelope(), issued_at="different"), RUN_IDS[1], root, root / "state"):
            pass


def test_later_envelope_can_use_fresh_ids_without_code_change(receipt_repo):
    env = range_envelope(["dq11-l-census-98", "dq11-l-census-99"])
    with launch.attempt_permit(env, env["run_ids"][0], receipt_repo, receipt_repo / "state"):
        pass


@pytest.mark.parametrize("ids,cap", [([], 0), (["wrong"], 1), ([RUN_IDS[0]] * 2, 2),
                                     (list(RUN_IDS) + ["dq11-l-census-15"], 11),
                                     (list(RUN_IDS), 9), (list(RUN_IDS), True)])
def test_invalid_envelope_lists_refuse(ids, cap):
    assert not launch.valid_run_ids({"run_ids": ids, "maximum_attempts": cap})


def test_each_permitted_attempt_reenters_full_gated_launcher(receipt_repo, monkeypatch):
    root = receipt_repo
    envelope = root / "envelope.json"
    envelope.write_text(json.dumps(range_envelope()))
    monkeypatch.setattr(launch, "validate_envelope", lambda *args: None)
    monkeypatch.setattr(launch, "verify_envelope_comment", lambda *args: None)
    monkeypatch.setattr(launch.Path, "home", lambda: root)
    monkeypatch.setenv("RECEIPTS_REPO", str(root))
    calls = []

    def attempt(*args, run_id, before_controller):
        calls.append((run_id, args[-1]))
        before_controller()
        refusal_receipt(root, run_id)
        return 2

    monkeypatch.setattr(launch, "launch_attempt", attempt)
    for i in range(2):
        packet = root / f"packet{i}"
        assert launch.launch(root / "template", envelope, root / "e4b", root / "controller",
                             root / "assets", packet, run_id=RUN_IDS[i]) == 2
    assert calls == [(RUN_IDS[i], root / f"packet{i}") for i in range(2)]


def test_not_invoked_with_unexpected_receipt_stops(receipt_repo):
    root = receipt_repo
    with pytest.raises(ValueError):
        with launch.attempt_permit(range_envelope(), RUN_IDS[0], root, root / "state"):
            raise ValueError("pre-controller gate")
    refusal_receipt(root, RUN_IDS[0])
    with pytest.raises(ValueError, match="not-invoked attempt has unexpected"):
        with launch.attempt_permit(range_envelope(), RUN_IDS[1], root, root / "state"):
            pass


@pytest.mark.parametrize("malformed", [False, True])
def test_canonical_ledger_header_reaches_used_id_guard_and_bad_entry_refuses(receipt_repo, malformed):
    root = receipt_repo
    ledger = root / "receipts/experts4bit-qlora/ledger.jsonl"
    content = (FIXTURE / "canonical-ledger.jsonl").read_text()
    assert len(content.splitlines()) == 2
    assert content.startswith("# Append-only ledger:")
    assert '"run_id": "dq11-l-census-4"' in content
    if malformed:
        lines = content.splitlines()
        lines[1] = "{malformed non-comment row"
        content = "\n".join(lines) + "\n"
    ledger.write_text(content)
    git_fixture(root, "add", "receipts")
    git_fixture(root, "commit", "-m", "Canonical ledger with header")
    env = range_envelope(["dq11-l-census-4", "dq11-l-census-5"])
    expected = json.JSONDecodeError if malformed else ValueError
    message = "Expecting property name" if malformed else "already used outside this envelope"
    with pytest.raises(expected, match=message):
        with launch.attempt_permit(env, "dq11-l-census-5", root, root / "state"):
            pytest.fail("used id or malformed row accepted")
    assert not (root / "state/123.json").exists()  # no attempt burned


def test_canonical_ledger_whitespace_and_comments_allow_unused_id(receipt_repo):
    root = receipt_repo
    ledger = root / "receipts/experts4bit-qlora/ledger.jsonl"
    content = (FIXTURE / "canonical-ledger.jsonl").read_text()
    ledger.write_text("  \n  # leading whitespace comment\n" + content + "\n")
    git_fixture(root, "add", "receipts")
    git_fixture(root, "commit", "-m", "Legal ledger whitespace")
    with launch.attempt_permit(range_envelope(), RUN_IDS[0], root, root / "state"):
        pass
