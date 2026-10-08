"""Falsify DQ8's instrument on CPU; fixture bytes are not measurements."""
import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench/dq8"), str(ROOT / "bench/dq7")]
try:
    import dq8_reduce as reducer
    import dq8_refusal as control
    import dq7_subject as subject
finally:
    del sys.path[:2]


def setup(name, placement):
    return {"base": "nf4", "placement": placement, "r": 16, "alpha": 32, "adapter_dtype": "fp32",
            "targets": ["attn_in", "attn_out", "mlp_in", "mlp_out"], "attn_impl": "sdpa",
            "loss_chunk": 0 if name == "llama31_8b" else 512}


def fixture():
    proof = {"status": "PASS", **dict.fromkeys(("loss_bitwise", "all_gradients_bitwise", "optimizer_updates_bitwise",
                                               "reload_logits_bitwise", "frozen_unchanged"), True)}
    refusal = {"schema": "dq8-refusal/1", "status": "REFUSED", "subject": "qwen3_32b", "placement": "device",
               "seq": 4096, "config_sha256": subject.CONFIG_HASHES["qwen3_32b"], "allocated_before": 0,
               "allocated_after": 0, "plan": {"status": "refused", "hardware": {"gpu": {"name": reducer.DEVICE}},
                   "model": {"model_type": "qwen3", "n_layers": 64}, "workload": {"seq_len": 4096, "steps": 2},
                   "constraints": {"fixed": setup("qwen3_32b", "device")}}}
    receipts = []
    for name, placement in sorted(reducer.ARMS):
        chosen = setup(name, placement)
        receipts.append({"backend": "dense", "status": "OK", "setup": chosen, "workload": {"steps": 2},
            "hardware": {"gpu": {"name": reducer.DEVICE, "memory_total": 24*2**30}},
            "dq7": {"subject": name, "placement": placement, "seq": 4096, "config_sha256": subject.CONFIG_HASHES[name],
                    "synthetic": True, "pretrained": False, "allocator": "default", "real_tokens_per_row": 4096,
                    "padded_tokens": 0, "row_tokens_sha256": "fixture rows", "loggetta_sha": "fixture loggetta",
                    "e4b_sha": "fixture e4b"},
            "engaged": {"setup": copy.deepcopy(chosen), "chunked_loss_engaged": bool(chosen["loss_chunk"]),
                        "adapter_targets": ["fixture projection"]*(7*reducer.LAYERS[name]),
                        "dense_offload": {"layers": reducer.LAYERS[name], "late_bound_4bit": 7*reducer.LAYERS[name],
                                          "all_pinned": True}},
            "comparison": {"device_allocator": {"estimated": 1000, "measured": 1000},
                           "device_driver": {"estimated": 1250}},
            "measured": {"device_reserved_peak_bytes": 1200},
            "plan": {"selected": {"lines": [{"name": "CUDA context fixture", "where": "device", "bytes": 50}]}}})
    return proof, refusal, receipts


def test_complete_boundary_fixture_and_one_byte_failures():
    assert reducer.reduce(*fixture())["verdict"] == "BOUNDARY_PASS"
    proof, refusal, rows = fixture()
    rows[0]["comparison"]["device_allocator"]["estimated"] -= 1
    assert reducer.reduce(proof, refusal, rows)["verdict"] == "ESTIMATE_UNDER"
    proof, refusal, rows = fixture()
    rows[0]["comparison"]["device_driver"]["estimated"] -= 1
    result = reducer.reduce(proof, refusal, rows)
    assert result["verdict"] == "RESERVE_UNDER" and not result["pass_licensed"]


def test_failed_proof_and_changed_boundary_never_pass():
    proof, refusal, rows = fixture()
    proof["all_gradients_bitwise"] = False
    assert reducer.reduce(proof, refusal, rows)["verdict"] == "FUNCTION_FAIL"
    proof, refusal, rows = fixture()
    refusal["status"] = "BOUNDARY_CHANGED"
    assert reducer.reduce(proof, refusal, rows)["verdict"] == "BOUNDARY_CHANGED"
    proof, refusal, rows = fixture()
    refusal["allocated_after"] = 1
    assert reducer.reduce(proof, refusal, rows)["verdict"] == "VOID"


@pytest.mark.parametrize("field,value", [("padded_tokens", 1), ("config_sha256", "bad"), ("synthetic", False),
                                       ("allocator", "expandable"), ("seq", 2048), ("e4b_sha", "different")])
def test_changed_reading_is_void(field, value):
    proof, refusal, rows = fixture()
    rows[0]["dq7"][field] = value
    assert reducer.reduce(proof, refusal, rows)["verdict"] == "VOID"


def test_incomplete_duplicate_wrong_card_missing_peaks_and_unengaged_are_void():
    proof, refusal, rows = fixture()
    assert reducer.reduce(proof, refusal, rows[:-1])["verdict"] == "VOID"
    assert reducer.reduce(proof, refusal, rows + rows[:1])["verdict"] == "VOID"
    for mutate in (lambda r: r[0]["hardware"]["gpu"].update(name="A2000"),
                   lambda r: r[0]["measured"].clear(),
                   lambda r: next(x for x in r if x["dq7"]["placement"] == "stream")["engaged"]["dense_offload"].clear(),
                   lambda r: r[0]["setup"].update(r=8)):
        changed = copy.deepcopy(rows)
        mutate(changed)
        assert reducer.reduce(proof, refusal, changed)["verdict"] == "VOID"


def test_actual_24gb_planner_refuses_the_resident_32b_control(tmp_path, monkeypatch):
    pytest.importorskip("loggetta")
    import torch
    from loggetta import hardware
    from loggetta.backends import dense

    fact = lambda value: hardware.Fact(value, "inferred", "CPU fixture")  # noqa: E731
    gpu = hardware.GPU(0, "nvidia", reducer.DEVICE, None, fact((8, 9)), fact(24*2**30), fact(int(23.5*2**30)),
                       fact("fixture"), fact(4), fact(16), fact(4), fact(16))
    host = hardware.Host(fact("fixture"), fact(16), fact(128*2**30), fact(100*2**30), fact(128*2**30))
    monkeypatch.setattr(hardware, "probe", lambda: hardware.HardwareProfile((gpu,), host, "Linux x86_64"))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    raw = (ROOT/"bench/dq7/configs/qwen3_32b.json").read_bytes()
    (tmp_path/"config.json").write_bytes(raw)
    result = control.refusal(str(tmp_path))
    assert result["status"] == "REFUSED" and result["plan"]["status"] == "refused"
    assert result["plan"]["model"]["n_layers"] == 64
    for name in ("qwen3_14b", "llama31_8b", "qwen3_32b"):
        config = subject.config_for(name)
        describe = dense.describe
        with monkeypatch.context() as m:
            m.setattr(dense, "describe", lambda *args: describe(config))
            from dq7_arm import make_plan
            for placement in (("stream",) if name == "qwen3_32b" else ("device", "stream")):
                assert make_plan(tmp_path, placement, 4096).status == "feasible"


def test_wrong_card_refusal_nonce_and_shell_syntax(tmp_path):
    for file in ("dq8_run.sh", "dq8_drive.sh"):
        assert subprocess.run(["bash", "-n", str(ROOT/"bench/dq8"/file)]).returncode == 0
    (tmp_path/"bin").mkdir()
    p = tmp_path/"bin/nvidia-smi"
    p.write_text('#!/bin/sh\necho "NVIDIA RTX A2000 12GB, 12282"\n')
    p.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}/bin:/usr/bin:/bin", "DQ8_W": str(tmp_path),
           "TC1_RUN_NONCE": "fixture", "E4B_SHA": "fixture"}
    result = subprocess.run(["bash", str(ROOT/"bench/dq8/dq8_run.sh")], env=env, capture_output=True, text=True)
    assert result.returncode == 19 and (tmp_path/"TP_DONE.fixture").exists()
    assert not (tmp_path/"logs/pip.log").exists()
