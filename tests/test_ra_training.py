"""CPU process/failure tests and native fixture mutations; no GPU evidence."""
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench/ra" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


load("ra_env")
load("ra_stage")
process = load("ra_process")
training = load("ra_training")


def spec(tmp_path):
    out = {"battery": "proof", "venv": str(tmp_path / "venv"), "cache": str(tmp_path / "cache"),
           "threads": 8, "allocator": "expandable_segments:True", "expect_trainable": 100,
           "deadline_epoch_s": time.time() + 1000, "timeout_s": 30}
    for name in ("data", "tokens", "prereg"):
        p = tmp_path / name
        p.write_text("synthetic input\n")
        out[name] = {"path": str(p), "sha256": process.file_digest(p)}
    pins = json.loads((ROOT / "bench/ra/source-pins.json").read_bytes())
    body = {"train": [list(range(8)) for _ in range(16)], "eval": [list(range(8)) for _ in range(8)]}
    tk = {**body, "fam": "granite", "tokenizer": training.MODEL["proof"][1],
          "revision": pins["models"][training.MODEL["proof"][1]], "dataset_sha256": out["data"]["sha256"],
          "template": "alpaca", "seq": 2048,
          "sha256": hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()}
    Path(out["tokens"]["path"]).write_text(json.dumps(tk, indent=2))
    out["tokens"]["sha256"] = process.file_digest(out["tokens"]["path"])
    return out


def native(s, pins, profile=False):
    return {"status": "ok", "framework": "e4b", "arm": "fused", "fam": "granite",
            "model": training.MODEL["proof"][1], "revision": pins["models"][training.MODEL["proof"][1]],
            "tag": "training_profile" if profile else "training", "steps": 20, "seq": 2048,
            "micro_batch": 2, "accum": 4, "autocast": False, "r": 16, "alpha": 16, "lr": 2e-4,
            "seed": 3407, "offload": False, "attn_4bit": True, "adapter_dtype": "fp32",
            "lora_init": "matched:3407", "dgrad": True, "template": "alpaca",
            "expect_trainable": s["expect_trainable"], "profile_steps": 10 if profile else 0,
            "profile_warm": 10, "prereg": s["prereg"]["path"],
            "tokens": {"sha256": training.token_digest(s, pins), "pack": False, "eval_rows_used": 8},
            "optimizer": "adamw_8bit(lr=0.0002, weight_decay=0.001) schedule=linear warmup_steps=5",
            "eval_curve": [{"step": 0}, {"step": 20}],
            "lr_per_step": [round(2e-4 * min(i / 5, (20 - i) / 15), 8) for i in range(20)],
            "profile": {"profiled_steps": 10} if profile else None}


@pytest.mark.parametrize("profile", [False, True])
def test_frozen_cli_has_field_autocast_and_independent_profile(tmp_path, profile):
    s = spec(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    pins = json.loads((ROOT / "bench/ra/source-pins.json").read_bytes())
    (stage / "source-pins.json").write_text(json.dumps(pins))
    argv, fam, model = training.command(s, stage, tmp_path / "out", profile=profile)
    opts = dict(zip(argv[3::2], argv[4::2]))
    assert fam == "granite" and opts["--revision"] == pins["models"][model]
    assert opts["--autocast"] == "0" and opts["--steps"] == "20" and opts["--seq"] == "2048"
    assert opts["--profile-steps"] == ("10" if profile else "0") and opts["--profile-warm"] == "10"
    assert opts["--tokens-sha"] == training.token_digest(s, pins) != s["tokens"]["sha256"]
    training.check_native(native(s, pins, profile), s, pins, profile=profile)


@pytest.mark.parametrize("key,value", [("autocast", True), ("micro_batch", 1), ("revision", "wrong"),
                                      ("optimizer", "adamw_torch"), ("profile", {}), ("dgrad", False),
                                      ("eval_curve", [{"step": 0}, {"step": 10}])])
def test_native_mutations_cannot_pass_fixture_binding(tmp_path, key, value):
    s = spec(tmp_path)
    pins = json.loads((ROOT / "bench/ra/source-pins.json").read_bytes())
    row = native(s, pins)
    row[key] = value
    with pytest.raises(ValueError):
        training.check_native(row, s, pins, profile=False)


def test_changed_input_or_unknown_spec_refused_before_launch(tmp_path):
    s = spec(tmp_path)
    Path(s["tokens"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="bytes"):
        training.command(s, tmp_path / "stage", tmp_path / "out", profile=False)
    s["extra_flags"] = ["--steps", "1"]
    with pytest.raises(ValueError, match="fields"):
        training.command(s, tmp_path / "stage", tmp_path / "out", profile=False)


def test_resealed_file_cannot_hide_changed_native_token_payload(tmp_path):
    s = spec(tmp_path)
    pins = json.loads((ROOT / "bench/ra/source-pins.json").read_bytes())
    p = Path(s["tokens"]["path"])
    tk = json.loads(p.read_bytes())
    tk["train"][0][0] += 1
    p.write_text(json.dumps(tk))
    s["tokens"]["sha256"] = process.file_digest(p)
    with pytest.raises(ValueError, match="payload digest"):
        training.token_digest(s, pins)


def test_deadline_keeps_retrieval_margin_and_refuses_nonfinite():
    with pytest.raises(ValueError, match="ten minutes"):
        process.window(time.time() + 620, 30)
    with pytest.raises(ValueError, match="invalid"):
        process.window(float("inf"), 30)


def test_symlink_traversal_cannot_relabel_input_bytes(tmp_path):
    p = tmp_path / "bytes"
    p.write_text("synthetic")
    link = tmp_path / "link"
    link.symlink_to(p)
    with pytest.raises(ValueError, match="symlink"):
        process.file_digest(link)


@pytest.mark.parametrize("code,status", [(0, "OK"), (7, "PROCESS_FAILED")])
def test_real_cpu_child_receipt_preserves_nonzero_exit_no_retry(tmp_path, code, status):
    script = tmp_path / "child.py"
    script.write_text(f"print('CPU process control'); raise SystemExit({code})\n")
    kwargs = dict(env=os.environ.copy(), cwd=tmp_path, log=tmp_path / "process.log",
                  receipt=tmp_path / "process.json", deadline=time.time() + 1000, timeout=10)
    if code:
        with pytest.raises(RuntimeError, match="retained"):
            process.run([sys.executable, str(script)], **kwargs)
    else:
        process.run([sys.executable, str(script)], **kwargs)
    rec = json.loads(kwargs["receipt"].read_bytes())
    assert rec["status"] == status and rec["returncode"] == code and rec["attempts"] == 1
    assert rec["log_sha256"] == process.file_digest(kwargs["log"])
    before = kwargs["receipt"].read_bytes()
    with pytest.raises(FileExistsError):
        process.run([sys.executable, str(script)], **kwargs)
    assert kwargs["receipt"].read_bytes() == before


def test_real_timeout_kills_only_owned_group_and_preserves_log(tmp_path):
    import subprocess
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"], start_new_session=True)
    try:
        with pytest.raises(RuntimeError, match="retained"):
            process.run([sys.executable, "-c", "import time; print('started',flush=True); time.sleep(10)"],
                        env=os.environ.copy(), cwd=tmp_path, log=tmp_path / "process.log",
                        receipt=tmp_path / "process.json", deadline=time.time() + 1000, timeout=0.2)
        rec = json.loads((tmp_path / "process.json").read_bytes())
        assert rec["status"] == "TIMEOUT" and rec["returncode"] < 0 and foreign.poll() is None
        assert "started" in (tmp_path / "process.log").read_text()
    finally:
        foreign.kill()
        foreign.wait()


def test_native_function_failure_is_never_promoted(tmp_path):
    s = spec(tmp_path)
    pins = json.loads((ROOT / "bench/ra/source-pins.json").read_bytes())
    row = native(s, pins)
    row["status"] = "c1_failed"
    before = copy.deepcopy(row)
    training.check_native(row, s, pins, profile=False)
    assert row == before and row["status"] == "c1_failed"


@pytest.mark.parametrize("mutate", [False, True])
def test_composition_retains_native_and_rechecks_inputs_after_process(tmp_path, monkeypatch, mutate):
    s = spec(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    pins = json.loads((ROOT / "bench/ra/source-pins.json").read_bytes())
    (stage / "source-pins.json").write_text(json.dumps(pins))
    verified = []
    monkeypatch.setattr(training.ra_stage, "verify", lambda p: verified.append(p))
    monkeypatch.setenv("E4B_FUSED_RMSNORM", "1")
    row = native(s, pins)

    def cpu_instrument(argv, **kwargs):
        assert "E4B_FUSED_RMSNORM" not in kwargs["env"]
        assert "PYTHONPATH" not in kwargs["env"]
        (kwargs["cwd"] / "native/granite_e4b_training.json").write_text(json.dumps(row))
        if mutate:
            Path(s["tokens"]["path"]).write_text("changed during CPU composition")
        return {"status": "OK", "attempts": 1}

    monkeypatch.setattr(training.ra_process, "run", cpu_instrument)
    out = tmp_path / "out"
    if mutate:
        with pytest.raises(ValueError, match="bytes"):
            training.execute(s, stage, out)
        assert json.loads((out / "component.json").read_bytes())["status"] == "NATIVE_BINDING_FAILED"
    else:
        result = training.execute(s, stage, out)
        assert result["status"] == "NATIVE_RECORDED_PENDING_ENGAGEMENT"
        assert result["proves_gpu_engagement"] is False
    assert verified == [stage, stage]
    assert json.loads((out / "native/granite_e4b_training.json").read_bytes()) == row
