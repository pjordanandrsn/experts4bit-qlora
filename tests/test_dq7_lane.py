"""DQ7 instrument falsification on CPU. These are synthetic receipts, never measured results."""
import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
LANE = ROOT / "bench/dq7"
sys.path.insert(0, str(LANE))
try:
    import dq7_reduce as reducer
    import dq7_subject as subject
finally:
    sys.path.pop(0)


def fixtures():
    proof = {"schema": "dq7-proof/1", "status": "PASS", **{key: True for key in (
        "loss_bitwise", "all_gradients_bitwise", "optimizer_updates_bitwise", "reload_logits_bitwise", "frozen_unchanged")}}
    receipts = []
    for name, seqs in reducer.SEQS.items():
        for placement in ("device", "stream"):
            for seq in seqs:
                peak = reducer.ANCHOR_BYTES.get((placement, seq), 2**30) if name == "qwen3_32b" else 2**30
                chunk = 0 if name == "llama31_8b" else 512
                setup = {"base": "nf4", "placement": placement, "r": 16, "alpha": 32, "adapter_dtype": "fp32",
                         "attn_impl": "sdpa", "loss_chunk": chunk, "targets": ["attn_in", "attn_out", "mlp_in", "mlp_out"]}
                receipts.append({"backend": "dense", "status": "OK", "setup": setup,
                    "hardware": {"gpu": {"name": reducer.DEVICE, "memory_total": 32 * 2**30}}, "workload": {"steps": 2},
                    "model": {"n_layers": reducer.LAYERS[name]},
                    "dq7": {"subject": name, "placement": placement, "seq": seq, "config_sha256": subject.CONFIG_HASHES[name],
                            "synthetic": True, "pretrained": False, "real_tokens_per_row": seq, "padded_tokens": 0,
                            "allocator": "default", "row_tokens_sha256": "fixture rows only",
                            "loggetta_sha": "fixture software only", "e4b_sha": "fixture software only"},
                    "engaged": {"setup": copy.deepcopy(setup), "chunked_loss_engaged": bool(chunk),
                                "adapter_targets": ["fixture projection"] * (7 * reducer.LAYERS[name]),
                                "dense_offload": {"layers": reducer.LAYERS[name], "all_pinned": True,
                                                  "late_bound_4bit": 7 * reducer.LAYERS[name]}},
                    "comparison": {"device_allocator": {"estimated": peak + peak // 50, "measured": peak}}})
    return proof, receipts


def test_complete_anchored_fixture_passes_and_reports_bytes():
    result = reducer.reduce(*fixtures())
    assert result["verdict"] == "NEVER_UNDER" and result["pass_licensed"]
    assert len(result["rows"]) == 12
    assert all(row["residual_bytes"] <= 0 for row in result["rows"])
    assert not result["quality_read"] and not result["calibration_replacement_licensed"]


def test_a_one_byte_underestimate_fails_the_decisive_gate():
    proof, receipts = fixtures()
    comparison = receipts[0]["comparison"]["device_allocator"]
    comparison["estimated"] = comparison["measured"] - 1
    result = reducer.reduce(proof, receipts)
    assert result["verdict"] == "ESTIMATE_UNDER" and not result["pass_licensed"]


def test_anchor_miss_keeps_oos_verdict_but_prevents_pass_quotation():
    proof, receipts = fixtures()
    receipt = next(row for row in receipts if row["dq7"]["subject"] == "qwen3_32b")
    receipt["comparison"]["device_allocator"]["measured"] += reducer.ANCHOR_DRAW_SPREAD_BYTES["device"] + 1
    result = reducer.reduce(proof, receipts)
    assert result["verdict"] == "ANCHOR_MISS" and result["out_of_sample_verdict"] == "NEVER_UNDER"
    assert not result["pass_licensed"]


@pytest.mark.parametrize("field,value", [("padded_tokens", 1), ("config_sha256", "wrong fixture hash"),
                                       ("synthetic", False), ("row_tokens_sha256", "different fixture rows"),
                                       ("allocator", "expandable"), ("e4b_sha", "different fixture software")])
def test_changed_subject_or_pair_is_void(field, value):
    proof, receipts = fixtures()
    receipts[1]["dq7"][field] = value
    assert reducer.reduce(proof, receipts)["verdict"] == "VOID"


@pytest.mark.parametrize("field,value", [("attn_impl", "eager"), ("loss_chunk", 0), ("r", 8),
                                       ("adapter_dtype", "bf16")])
def test_changed_executed_setup_is_void(field, value):
    proof, receipts = fixtures()
    receipts[0]["setup"][field] = value
    assert reducer.reduce(proof, receipts)["verdict"] == "VOID"


def test_missing_duplicate_wrong_card_and_unengaged_stream_are_void():
    proof, receipts = fixtures()
    assert reducer.reduce(proof, receipts[:-1])["verdict"] == "VOID"
    assert reducer.reduce(proof, receipts + receipts[:1])["verdict"] == "VOID"
    receipts[0]["hardware"]["gpu"]["name"] = "fixture A2000"
    assert reducer.reduce(proof, receipts)["verdict"] == "VOID"
    proof, receipts = fixtures()
    receipts[2]["engaged"]["dense_offload"]["late_bound_4bit"] = 0
    assert reducer.reduce(proof, receipts)["verdict"] == "VOID"


def test_failed_proof_is_not_a_reading():
    proof, receipts = fixtures()
    proof["all_gradients_bitwise"] = False
    assert reducer.reduce(proof, receipts)["verdict"] == "VOID"


def test_configuration_bytes_are_pinned_and_wrong_bytes_refused(tmp_path):
    for name in subject.SUBJECTS:
        config = subject.config_for(name)
        assert config.num_hidden_layers == reducer.LAYERS[name]
    (tmp_path / "qwen3_14b.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        subject.config_for("qwen3_14b", tmp_path)


def test_builder_is_reproducible_on_a_reduced_cpu_fixture(tmp_path, monkeypatch):
    from transformers import Qwen3Config
    from safetensors.torch import load_file

    monkeypatch.setattr(subject, "config_for", lambda *a: Qwen3Config(hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=32))
    first, second = tmp_path / "a", tmp_path / "b"
    a, b = subject.build("tiny", first), subject.build("tiny", second)
    assert a == b and a["checkpoint_bytes"] > 0 and a["synthetic"] and not a["pretrained"]
    for file in a["files"]:
        assert load_file(str(first / file)).keys() == load_file(str(second / file)).keys()
        assert (first / file).read_bytes() == (second / file).read_bytes()
    with pytest.raises(FileExistsError):
        subject.build("tiny", first)


def test_runner_refuses_a2000_before_install_and_binds_nonce(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    smi = bin_dir / "nvidia-smi"
    smi.write_text('#!/bin/sh\necho "NVIDIA RTX A2000 12GB, 12282"\n')
    smi.chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()
    env = {**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin", "DQ7_W": str(work), "TC1_RUN_NONCE": "fixture-nonce",
           "E4B_SHA": "fixture not installed"}
    result = subprocess.run(["bash", str(LANE / "dq7_run.sh")], env=env, capture_output=True, text=True)
    assert result.returncode == 19 and (work / "TP_DONE.fixture-nonce").exists()
    assert not (work / "REFUSAL").exists() and not (work / "logs/pip.log").exists()


def test_runner_shell_syntax_and_reducer_self_test():
    for file in ("dq7_run.sh", "dq7_drive.sh"):
        assert subprocess.run(["bash", "-n", str(LANE / file)]).returncode == 0
    result = subprocess.run([sys.executable, str(LANE / "dq7_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert result.returncode == 0 and "self-test OK" in result.stdout
