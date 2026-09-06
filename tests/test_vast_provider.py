"""e4b#455 — VastProvider behind E4B_RENT_LIVE=1: key by shape from a 600 file, false-zero refused, destroy via
v0 with a body, pre-flight that destroys on failure, launch facts. Recorded response shapes; no network, no money."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from experts4bit_qlora.tools.vast_provider import (
    BackendUnavailable, FakeTransport, PreflightFailed, VastProvider, VastRefused, load_api_key, offer_filter, provider_from_env,
)

KEY = "ab" * 20 + "0123456789abcdef"  # 56 hex chars, built at runtime so nothing key-shaped is at rest
OFFER = {"id": 42274235, "machine_id": 12345, "gpu_name": "RTX 5090", "dph_total": 0.61, "disk_space": 512.0,
         "cpu_ram": 128 * 1024, "verification": "verified", "rentable": True, "num_gpus": 1}
UNVERIFIED = dict(OFFER, id=42274236, verification="unverified", dph_total=0.30)
INSTANCE = {"id": 7000123, "actual_status": "running", "ssh_host": "ssh5.vast.ai", "ssh_port": 12345, "disk_space": 512.0,
            "cpu_ram": 128 * 1024, "gpu_name": "RTX 5090"}


def key_file(tmp_path: Path, mode: int = 0o600, value: str = KEY) -> Path:
    p = tmp_path / "secrets.env"
    p.write_text(f"# vast\nVAST_API_KEY={value}\n")
    p.chmod(mode)
    return p


def routes(extra=None):
    base = {
        ("GET", "/v0/users/current/"): [(200, {"id": 633178, "email": "x@example.com", "balance": 0, "credit": 400})],
        ("GET", "/v0/bundles/"): [(200, {"offers": [UNVERIFIED, OFFER]})],
        ("PUT", "/v0/asks/42274235/"): [(200, {"success": True, "new_contract": 7000123})],
        ("GET", "/v1/instances/"): [(200, {"instances": [INSTANCE]})],
        ("GET", "/v0/instances/7000123/"): [(200, {"instances": INSTANCE})],
        ("DELETE", "/v0/instances/7000123/"): [(200, {"success": True})],
        ("POST", "/v0/instances/7000123/ssh/"): [(200, {"success": True})],
    }
    base.update(extra or {})
    return base


def provider(tr: FakeTransport, **kw) -> VastProvider:
    kw.setdefault("ssh_runner", lambda h, p, c, t: (0, ""))
    kw.setdefault("bandwidth_probe", lambda h, p: 95.0)
    kw.setdefault("sleep", lambda s: None)
    return VastProvider(tr, run_label="test-run", **kw)


# ---- key
def test_key_is_read_by_shape_from_a_600_file(tmp_path: Path):
    assert load_api_key(key_file(tmp_path)) == KEY


def test_key_file_loose_mode_or_wrong_shape_or_missing_refuses(tmp_path: Path):
    with pytest.raises(VastRefused, match="not 600"):
        load_api_key(key_file(tmp_path, mode=0o644))
    with pytest.raises(VastRefused, match="shape"):
        load_api_key(key_file(tmp_path, value="not-a-key"))
    with pytest.raises(VastRefused, match="no Vast key file"):
        load_api_key(tmp_path / "absent.env")


def test_error_text_never_contains_the_key(tmp_path: Path):
    p = key_file(tmp_path, value=KEY)
    tr = FakeTransport(routes({("GET", "/v0/users/current/"): [(401, {"success": False, "error": "unauthorized"})]}))
    with pytest.raises(VastRefused) as ei:
        provider(tr).auth_probe()
    assert KEY not in str(ei.value) and "401" in str(ei.value)
    assert KEY not in json.dumps(tr.calls)  # the fake transport never sees the key: it is a header in the real one only
    _ = p


# ---- arming
def test_a_test_runner_can_never_go_live(tmp_path: Path, monkeypatch):
    """Whatever the environment says, provider_from_env refuses inside pytest (the 2026-09-06 incident)."""
    monkeypatch.setenv("E4B_RENT_LIVE", "1")
    with pytest.raises(VastRefused, match="test runner"):
        provider_from_env(run_label="x", key_path=key_file(tmp_path))


def test_arming_rules_outside_a_test_runner(tmp_path: Path, monkeypatch):
    from experts4bit_qlora.tools import vast_provider
    monkeypatch.setattr(vast_provider, "_under_test", lambda: False)
    monkeypatch.delenv("E4B_RENT_LIVE", raising=False)
    with pytest.raises(VastRefused, match="E4B_RENT_LIVE=1"):
        provider_from_env(run_label="x", key_path=key_file(tmp_path))
    monkeypatch.setenv("E4B_RENT_LIVE", "1")
    with pytest.raises(VastRefused, match="no Vast key file"):
        provider_from_env(run_label="x", key_path=tmp_path / "absent.env")
    monkeypatch.setattr(vast_provider, "DEFAULT_KEY_PATH", tmp_path / "also-absent.env")
    with pytest.raises(VastRefused, match="also-absent"):
        provider_from_env(run_label="x")  # the default is read at call time, not bound at import


# ---- false zero
@pytest.mark.parametrize("status,body", [
    (200, {"success": False, "error": "deprecated_endpoint"}),  # the 2026-08-25 / 08-29 body
    (200, {"msg": "ok"}),                                        # no `instances` key
    (200, []),                                                   # wrong shape
    (401, {"error": "unauthorized"}),
    (500, "Internal Server Error"),
])
def test_list_ids_refuses_every_false_zero_shape(status, body):
    tr = FakeTransport(routes({("GET", "/v1/instances/"): [(status, body)]}))
    with pytest.raises(BackendUnavailable):
        provider(tr).list_ids()


def test_list_ids_reads_only_v1_instances_array():
    tr = FakeTransport(routes({("GET", "/v1/instances/"): [(200, {"instances": [{"id": 1}, {"id": "2"}, {"nope": 3}]})]}))
    assert provider(tr).list_ids() == {"1", "2"}
    assert tr.calls[-1][:2] == ("GET", "/v1/instances/")
    empty = FakeTransport(routes({("GET", "/v1/instances/"): [(200, {"instances": []})]}))
    assert provider(empty).list_ids() == set(), "a genuine empty v1 array is the only empty answer"


# ---- offers + launch
def test_offer_filter_is_the_policy_class_in_provider_units():
    q = offer_filter("RTX 5090", min_disk_gb=320, min_ram_gb=98)
    assert q["verified"] == {"eq": True} and q["gpu_name"] == {"eq": "RTX 5090"}
    assert q["disk_space"] == {"gte": 320} and q["cpu_ram"] == {"gte": 98 * 1024}


def test_launch_picks_the_cheapest_verified_offer_and_records_facts():
    tr = FakeTransport(routes())
    p = provider(tr)
    iid = p.launch(gpu="RTX 5090", wallclock_h=2.0, image="pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel")
    assert iid == "7000123"
    put = [c for c in tr.calls if c[0] == "PUT"][0]
    assert put[1] == "/v0/asks/42274235/" and put[3]["image"].startswith("pytorch/") and put[3]["disk"] == 320
    facts = p.launch_facts()
    assert facts["vast_offer_id"] == "42274235" and facts["vast_machine_id"] == "12345" and facts["vast_contract_id"] == "7000123"
    assert json.loads(facts["vast_search_filter"])["verified"] == {"eq": True}
    assert facts["vast_verification"] == "verified"
    cost, method = p.actual_cost(iid, 3600)
    assert cost == 0.61 and "UNKNOWN" in method


def test_launch_refuses_when_no_verified_offer_and_when_create_fails():
    tr = FakeTransport(routes({("GET", "/v0/bundles/"): [(200, {"offers": [UNVERIFIED]})]}))
    with pytest.raises(VastRefused, match="no verified rentable"):
        provider(tr).launch(gpu="RTX 5090", wallclock_h=1, image="img")
    tr = FakeTransport(routes({("PUT", "/v0/asks/42274235/"): [(200, {"success": False, "error": "insufficient_credit"})]}))
    with pytest.raises(BackendUnavailable, match="create did not succeed"):
        provider(tr).launch(gpu="RTX 5090", wallclock_h=1, image="img")


# ---- destroy
def test_destroy_uses_v0_with_a_body_and_404_is_not_proof():
    tr = FakeTransport(routes())
    ev = provider(tr).destroy("7000123")
    call = tr.calls[-1]
    assert call[:2] == ("DELETE", "/v0/instances/7000123/") and call[3] == {}
    assert ev["method"] == "vast-destroy" and ev["http"] == 200
    tr404 = FakeTransport(routes({("DELETE", "/v0/instances/7000123/"): [(404, {"detail": "Not found"})]}))
    ev = provider(tr404).destroy("7000123")
    assert ev["http"] == 404 and "absence proven only by the listing" in ev["note"]
    trerr = FakeTransport(routes({("DELETE", "/v0/instances/7000123/"): [(200, {"success": False, "error": "x"})]}))
    with pytest.raises(BackendUnavailable):
        provider(trerr).destroy("7000123")


# ---- pre-flight
def test_preflight_passes_and_records_the_box():
    tr = FakeTransport(routes())
    facts = provider(tr, ssh_pubkey="ssh-ed25519 AAAA test").preflight("7000123", timeout_s=60)
    assert facts["vast_preflight"] == "ok" and facts["vast_ssh"] == "ssh5.vast.ai:12345" and facts["vast_bandwidth_mb_s"] == "95.0"
    assert any(c[:2] == ("POST", "/v0/instances/7000123/ssh/") for c in tr.calls)


def test_preflight_fails_on_stuck_loading_bad_disk_bad_ssh_and_slow_link():
    loading = dict(INSTANCE, actual_status="loading")
    clock = iter([0, 5, 700, 700, 700])
    tr = FakeTransport(routes({("GET", "/v0/instances/7000123/"): [(200, {"instances": loading})]}))
    with pytest.raises(PreflightFailed, match="stuck loading"):
        provider(tr, clock=lambda: next(clock)).preflight("7000123", timeout_s=600)
    small = dict(INSTANCE, disk_space=100.0)
    tr = FakeTransport(routes({("GET", "/v0/instances/7000123/"): [(200, {"instances": small})]}))
    with pytest.raises(PreflightFailed, match="disk 100 GB"):
        provider(tr).preflight("7000123")
    tr = FakeTransport(routes())
    with pytest.raises(PreflightFailed, match="did not authenticate"):
        provider(tr, ssh_runner=lambda h, p, c, t: (255, "Permission denied")).preflight("7000123")
    with pytest.raises(PreflightFailed, match="bandwidth 12.0 MB/s"):
        provider(tr, bandwidth_probe=lambda h, p: 12.0).preflight("7000123")


def test_nothing_in_this_module_creates_an_instance_on_import(monkeypatch):
    monkeypatch.delenv("E4B_RENT_LIVE", raising=False)
    tr = FakeTransport(routes())
    provider(tr)
    assert tr.calls == []
    assert os.environ.get("E4B_RENT_LIVE") is None
