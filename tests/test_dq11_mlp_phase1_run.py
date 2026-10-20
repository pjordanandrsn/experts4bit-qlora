"""The DQ11 MLP Phase 1 runner's CPU-checkable parts (bench/dq11/dq11_mlp_phase1_run.py): the lock check against a
wheel's RECORD, the claim refusal, and the reduction over the part records."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata as md
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "dq11"))
import dq11_mlp_phase1_run as run  # noqa: E402

DIST = "pytest"                                   # any installed distribution with a RECORD stands in for the lock's


def _wheel(tmp_path, record_text, name="fake-1.0-py3-none-any.whl"):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{DIST}-{md.version(DIST)}.dist-info/RECORD", record_text)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_wheel_identity_matches_installed_files_and_catches_a_changed_one(tmp_path):
    record = md.distribution(DIST).read_text("RECORD")
    path, sha = _wheel(tmp_path, record)
    row = {"filename": path.name, "version": md.version(DIST), "sha256": sha}
    got = run.wheel_identity(DIST, row, tmp_path)
    assert got["files_match_wheel"] and got["wheel_sha256"] == got["lock_sha256"] and got["installed"] == got["locked"]
    line = next(r for r in record.splitlines() if r.split(",")[0].endswith(".py") and r.split(",")[1])
    tampered = record.replace(line, line.rsplit(",", 2)[0] + ",sha256=AAAA," + line.rsplit(",", 1)[1])
    path2, sha2 = _wheel(tmp_path, tampered, "fake2-1.0-py3-none-any.whl")
    assert run.wheel_identity(DIST, {"filename": path2.name, "version": md.version(DIST), "sha256": sha2},
                              tmp_path)["files_match_wheel"] is False
    with pytest.raises(RuntimeError, match="not in the install cache"):
        run.wheel_identity(DIST, {"filename": "absent.whl", "version": "1", "sha256": "x"}, tmp_path)


def test_every_part_refuses_without_the_bus_claim(monkeypatch, tmp_path):
    monkeypatch.delenv("DQ11_GPU_CLAIM_CONFIRMED", raising=False)
    with pytest.raises(RuntimeError, match="bus claim is not confirmed"):
        run.require_runtime(tmp_path, tmp_path)


def _records():
    prefetch = {"devices": {"cuda:0": {"prefetches": 10}}}
    after = {"devices": {"cuda:0": {"prefetches": 74}}}
    lora_today = {f"layers.{i}.mlp.{p}": 2 for i in range(32) for p in ("gate_proj", "up_proj", "down_proj")}
    lora_today.update({f"layers.{i}.self_attn.{p}": 2 for i in range(32) for p in ("q_proj", "k_proj", "v_proj", "o_proj")})
    attn = {k: v for k, v in lora_today.items() if "self_attn" in k}
    step = {"training": {"losses": [1.25]}, "adapters_sha256": {"a": "1"}, "prefetch_before": prefetch,
            "prefetch_after": after, "engine_imported": False}
    k8 = {"quality_initial": {"alpaca-heldout": {"ppl": 5.0}, "wikitext-test": {"ppl": 6.0}},
          "quality_final": {"alpaca-heldout": {"ppl": 4.5}, "wikitext-test": {"ppl": 5.9}},
          "engine_stats": {"fallback": 0}}
    return {
        "stage-check": {"model_files": {"hashed": True}, "inputs": {"status": "PASS"}},
        "reads": {"read": {"outcome": "PHASE1_PASS"}},
        "reported": {"read": {"outcome": "PHASE1_PASS, RANK1_UNRESOLVED: 15:dx"}},
        "step-today": dict(step, lora_forwards=lora_today),
        "step-off": dict(step, lora_forwards=lora_today, knob_requested=False, engine_imported=True),
        "step-fused": dict(step, lora_forwards=attn, knob_requested=True, enabled_adapters=96, refused={},
                           engine_imported=True, engine_stats={"calls": 64, "fallback": 0}),
        "k8-unfused": copy.deepcopy(k8),
        "k8-fused": dict(copy.deepcopy(k8), engine_stats={"fallback": 0, "calls": 9}),
    }


def test_reduce_passes_a_clean_set_and_names_each_failure():
    good = _records()
    out = run.reduce(good)
    assert out["outcome"] == "PHASE1_PASS" and out["reported_outcome"].startswith("PHASE1_PASS, RANK1")
    cases = {
        "KNOB_OFF: the off step is not today's": lambda r: r["step-off"].__setitem__("adapters_sha256", {"a": "2"}),
        "ENGAGE_COUNT": lambda r: r["step-fused"].__setitem__("enabled_adapters", 93),
        "VOID_FALLBACK": lambda r: r["step-fused"]["engine_stats"].__setitem__("fallback", 1),
        "ENGAGE_PEFT_MLP_FORWARDS": lambda r: r["step-fused"]["lora_forwards"].__setitem__("layers.0.mlp.up_proj", 1),
        "ENGAGE_QKVO_CHANGED": lambda r: r["step-fused"]["lora_forwards"].__setitem__("layers.0.self_attn.q_proj", 3),
        "STREAMING_CHANGED": lambda r: r["step-fused"].__setitem__("prefetch_after", {"devices": {"cuda:0": {"prefetches": 75}}}),
        "RECOMPUTE_CHANGED": lambda r: r["step-fused"]["engine_stats"].__setitem__("calls", 32),
        "K8: quality_final alpaca-heldout": lambda r: r["k8-fused"]["quality_final"]["alpaca-heldout"].__setitem__("ppl", 4.56),
        "PHASE1_FAIL: 0:dB_up": lambda r: r["reads"]["read"].__setitem__("outcome", "PHASE1_FAIL: 0:dB_up"),
        "STAGE": lambda r: r["stage-check"]["model_files"].__setitem__("hashed", False),
    }
    for name, breaks in cases.items():
        r = copy.deepcopy(good)
        breaks(r)
        got = run.reduce(r)["outcome"]
        assert got.startswith("PHASE1_FAIL") and name in got, (name, got)
    r = copy.deepcopy(good)
    r["k8-fused"]["quality_final"]["wikitext-test"]["ppl"] = 5.9 + 0.05 - 1e-9          # inside the band
    assert run.reduce(r)["outcome"] == "PHASE1_PASS"


def test_the_model_check_names_a_missing_or_resized_file(tmp_path, monkeypatch):
    root = tmp_path / "hf-cache/model"
    root.mkdir(parents=True)
    manifest = {"model": "m", "revision": "r", "files": [{"name": "a.json", "size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()},
                                                         {"name": "b.bin", "size": 2, "sha256": "x"}]}
    here = tmp_path / "here"
    here.mkdir()
    (here / "model_files.json").write_text(__import__("json").dumps(manifest))
    monkeypatch.setattr(run, "HERE", here)
    (root / "a.json").write_bytes(b"abc")
    with pytest.raises(RuntimeError, match=r"\['b.bin'\]"):
        run.verify_model(tmp_path, hashes=False)
    (root / "b.bin").write_bytes(b"zz")
    assert run.verify_model(tmp_path, hashes=False)["files"] == 2
    with pytest.raises(RuntimeError, match=r"\['b.bin'\]"):
        run.verify_model(tmp_path, hashes=True)                             # right size, wrong bytes


def test_a_quoted_record_reads_as_its_unquoted_install(tmp_path):
    """torch's wheel writes RECORD paths quoted ("functorch/__init__.py",sha256=...); pip installs them unquoted."""
    record = md.distribution(DIST).read_text("RECORD")
    quoted = "\n".join(('"' + r.split(",", 1)[0] + '",' + r.split(",", 1)[1]) if r.count(",") >= 2 else r
                       for r in record.splitlines())
    assert run._records(quoted) == run._records(record) and run._records(record)
    path, sha = _wheel(tmp_path, quoted, "quoted-1.0-py3-none-any.whl")
    got = run.wheel_identity(DIST, {"filename": path.name, "version": md.version(DIST), "sha256": sha}, tmp_path)
    assert got["files_match_wheel"] is True


# --- Amendment 4 (c) and reduce_a4 --------------------------------------------------------------------------------------

def _records_a4(tmp_path, *, read_null=True, c_parts=None, off_equal=True):
    base = _records()
    stage, reads = base["stage-check"], {"read": {"outcome": "PHASE1_PASS, ACCURACY_OVER_REORDER: 0:out_gate"}}
    rec = {"stage-check": stage, "reads": reads, "reported": base["reported"], "step-fused": base["step-fused"],
           "k8-unfused": base["k8-unfused"], "k8-fused": base["k8-fused"]}
    for i, p in enumerate(run.TODAY_READ):
        r = copy.deepcopy(base["step-today"])
        if not read_null and i:
            r["adapters_sha256"] = {"a": str(i)}
        rec[p] = r
    rec["step-off"] = copy.deepcopy(base["step-today"]) if off_equal else dict(copy.deepcopy(base["step-today"]),
                                                                               adapters_sha256={"a": "x"})
    rec.update(c_parts or {})
    return rec


def _b_file(tmp_path, name, scale):
    import torch
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(0)
    t = {f"s{i:03d}.B": torch.randn(8, 4, generator=g) for i in range(224)}
    noise = torch.Generator().manual_seed(sum(map(ord, name)))
    t = {k: v + scale * torch.randn(v.shape, generator=noise) for k, v in t.items()}
    save_file(t, str(tmp_path / f"{name}-B.safetensors"))
    return {"name": f"{name}-B.safetensors", "sha256": hashlib.sha256((tmp_path / f"{name}-B.safetensors").read_bytes()).hexdigest()}


def _c(tmp_path, scales, *, same=False):
    out = {}
    for p, sc in zip(run.C_PARTS, scales):
        out[p] = {"adapters_sha256": {"a": "c" if same else p}, "training": {"losses": [1.0]}, "b_file": _b_file(tmp_path, p, sc)}
    return out


def test_need_c_and_each_knob_off_branch(tmp_path):
    held = _records_a4(tmp_path)
    assert run.need_c(held) is False and run.knob_off(held, tmp_path)["ok"] is True
    assert run.knob_off(_records_a4(tmp_path, off_equal=False), tmp_path)["ok"] is False
    failed = _records_a4(tmp_path, read_null=False)
    assert run.need_c(failed) is True and run.knob_off(failed, tmp_path)["ok"] is False      # C parts missing
    c_held = _records_a4(tmp_path, read_null=False, c_parts=_c(tmp_path, (0.01, 0.01, 0.01, 0.01), same=True))
    assert run.knob_off(c_held, tmp_path)["how"].startswith("bitwise under C") and run.knob_off(c_held, tmp_path)["ok"]
    spread = run.knob_off(_records_a4(tmp_path, read_null=False, c_parts=_c(tmp_path, (0.01, 0.01, 0.01, 0.01))), tmp_path)
    assert spread["how"].startswith("KNOB_OFF_BY_SPREAD") and spread["ok"] is True
    far = run.knob_off(_records_a4(tmp_path, read_null=False, c_parts=_c(tmp_path, (0.01, 0.01, 0.01, 0.2))), tmp_path)
    assert far["ok"] is False                                                # off 20x the null's spread


def test_reduce_a4_passes_names_and_never_fails_an_indeterminate(tmp_path):
    good = _records_a4(tmp_path)
    assert run.reduce_a4(good, tmp_path)["outcome"] == "PHASE1_PASS, ACCURACY_OVER_REORDER: 0:out_gate"
    ind = copy.deepcopy(good)
    ind["reads"]["read"]["outcome"] = "INDETERMINATE: calibration failed 0:dx:R4"
    assert run.reduce_a4(ind, tmp_path)["outcome"].startswith("INDETERMINATE")
    bad = copy.deepcopy(ind)
    bad["step-off"]["adapters_sha256"] = {"a": "x"}                          # a real failure outranks an indeterminate
    assert run.reduce_a4(bad, tmp_path)["outcome"].startswith("PHASE1_FAIL: KNOB_OFF")
    spread = _records_a4(tmp_path, read_null=False, c_parts=_c(tmp_path, (0.01, 0.01, 0.01, 0.01)))
    assert run.reduce_a4(spread, tmp_path)["outcome"].endswith(", KNOB_OFF_BY_SPREAD")
