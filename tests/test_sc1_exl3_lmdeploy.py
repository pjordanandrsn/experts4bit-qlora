# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""CPU-only tests for the SC1 ExLlamaV3 and LMDeploy drivers (bench/sc1/{exl3,lmdeploy}/).

The engines are NOT installed here and must not be; every test drives the drivers' pure functions or monkeypatches the
engine factory with a fake. What is exercised: the prompt-file contract refusals (batch, length, distinct rows,
digest), the exactly-N-tokens assertion, the slope arithmetic (p37's), the receipt shape, the K8 window digest refusal
and NLL arithmetic, and each driver's `--selftest`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DRIVERS = {
    "exl3_arm": REPO / "bench/sc1/exl3/sc1_exl3_arm.py",
    "exl3_nll": REPO / "bench/sc1/exl3/sc1_exl3_nll.py",
    "lmd_arm": REPO / "bench/sc1/lmdeploy/sc1_lmdeploy_arm.py",
    "lmd_nll": REPO / "bench/sc1/lmdeploy/sc1_lmdeploy_nll.py",
}


def _load(name):
    spec = importlib.util.spec_from_file_location(f"sc1_{name}", DRIVERS[name])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(params=["exl3_arm", "lmd_arm"])
def arm(request):
    return _load(request.param)


@pytest.fixture(params=["exl3_nll", "lmd_nll"])
def nll(request):
    return _load(request.param)


def _rows(batch, n=512, seed=3):
    return [[(seed * (i + 1) * 7919 + j * 31) % 151936 for j in range(n)] for i in range(batch)]


def _prompt_file(tmp_path, rows, **over):
    pf = {"batch": len(rows), "prompts": rows, "prompts_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
          "rows_sha256": hashlib.sha256(json.dumps([hashlib.sha256(json.dumps(r).encode()).hexdigest() for r in rows]).encode()).hexdigest()}
    pf.update(over)
    p = tmp_path / f"prompts_b{len(rows)}.json"
    p.write_text(json.dumps(pf))
    return p


# ---------------------------------------------------------------------------------------------- prompt contract

def test_prompt_file_accepted_and_digest_recorded(arm, tmp_path):
    rows = _rows(16)
    pf = arm.load_prompts(str(_prompt_file(tmp_path, rows)), 16)
    assert pf["prompts"] == rows and len(pf["prompts_sha256"]) == 64 and pf["prompt_tokens"] == [512] * 16


def test_prompt_digest_mismatch_refused(arm, tmp_path):
    rows = _rows(2)
    p = _prompt_file(tmp_path, rows)
    d = json.loads(p.read_text())
    d["prompts"][1][5] += 1          # one token tampered, digest stale
    p.write_text(json.dumps(d))
    with pytest.raises(arm.Refusal, match="digest mismatch"):
        arm.load_prompts(str(p), 2)


@pytest.mark.parametrize("breach", ["batch", "length", "distinct"])
def test_prompt_contract_breaches_refused(arm, tmp_path, breach):
    rows = _rows(2)
    if breach == "length":
        rows[0] = rows[0][:511]
    if breach == "distinct":
        rows[1] = list(rows[0])
    p = _prompt_file(tmp_path, rows)
    batch = 3 if breach == "batch" else 2
    with pytest.raises(arm.Refusal):
        arm.load_prompts(str(p), batch)


def test_ttft_file_accepts_4096_row(arm, tmp_path):
    rows = _rows(1, n=4096)
    pf = arm.load_prompts(str(_prompt_file(tmp_path, rows)), 1, prompt_len=None)
    assert pf["prompt_tokens"] == [4096]


# ---------------------------------------------------------------------------------------------- N-token assertion

def test_n_token_assertion(arm):
    assert arm.check_tokens({"0": [1] * 32, "1": [2] * 32}, 32, 2) == 64
    with pytest.raises(arm.Refusal, match="stopped early"):
        arm.check_tokens({"0": [1] * 32, "1": [2] * 31}, 32, 2)
    with pytest.raises(arm.Refusal):
        arm.check_tokens({"0": [1] * 32}, 32, 2)


def test_short_row_from_engine_refused(arm):
    rows = _rows(2)
    with pytest.raises(arm.Refusal):
        arm.run_slope(arm.FakeEngine(short_by=1), rows, 2)


# ---------------------------------------------------------------------------------------------- slope arithmetic

def test_slope_arithmetic_matches_p37(arm):
    s = arm.slope([1.0, 1.1, 1.2], [2.0, 2.1, 2.2], 16)
    extra = (128 - 32) * 16
    assert s["decode_tok_s"] == round(extra / 1.0, 1)
    assert s["decode_ms_per_step"] == round(1.0 / 96 * 1e3, 4)        # p37 rounds to 4 decimals
    assert s["decode_tok_s_median"] == round(extra / (2.1 - 1.1), 1)
    assert s["end_to_end_tok_s_long"] == round(128 * 16 / 2.0, 1)
    assert s["wall_short_s"] == 1.0 and s["wall_long_s"] == 2.0
    with pytest.raises(arm.Refusal):
        arm.slope([2.0, 2.0, 2.0], [1.0, 1.0, 1.0], 1)


def test_run_slope_warm_plus_three_reps_each_length(arm):
    eng = arm.FakeEngine()
    r = arm.run_slope(eng, _rows(2), 2)
    assert eng.calls == [32, 32, 32, 32, 128, 128, 128, 128]
    assert len(r["walls_short_s"]) == 3 and len(r["walls_long_s"]) == 3
    assert set(r["tokens"]) == {"0", "1"} and all(len(v) == 128 for v in r["tokens"].values())


# ---------------------------------------------------------------------------------------------- receipt shape

def test_receipt_shape_with_stubbed_engine(arm, tmp_path, monkeypatch):
    rows = _rows(2)
    p = _prompt_file(tmp_path, rows)
    out = tmp_path / "r.json"
    monkeypatch.setattr(arm, "make_engine", lambda *a, **k: arm.FakeEngine())
    monkeypatch.setattr(arm, "resolve_model_dir", lambda m, r: "/fake/model")
    monkeypatch.setenv("SC1_ARM", "unit")
    monkeypatch.setenv("SC1_BATCH", "2")
    monkeypatch.setenv("SC1_PROMPTS", str(p))
    monkeypatch.setenv("SC1_OUT", str(out))
    monkeypatch.setenv("SC1_INSTANCE_ID", "123")
    assert arm.main([]) == 0
    rec = json.loads(out.read_text())
    assert all(k in rec for k in arm.RECEIPT_KEYS_SLOPE), [k for k in arm.RECEIPT_KEYS_SLOPE if k not in rec]
    assert rec["batch"] == 2 and rec["mode"] == "slope" and rec["vast_instance_id"] == "123"
    assert rec["prompts_sha256"] == json.loads(p.read_text())["prompts_sha256"]
    assert rec["generation"] and rec["resolved"] == {"fake": True}


def test_ttft_mode_receipt(arm, tmp_path, monkeypatch):
    rows = _rows(1, n=4096)
    p = _prompt_file(tmp_path, rows)
    out = tmp_path / "t.json"
    monkeypatch.setattr(arm, "make_engine", lambda *a, **k: arm.FakeEngine())
    monkeypatch.setattr(arm, "resolve_model_dir", lambda m, r: "/fake/model")
    monkeypatch.setenv("SC1_ARM", "ttft")
    monkeypatch.setenv("SC1_BATCH", "1")
    monkeypatch.setenv("SC1_PROMPTS", str(p))
    monkeypatch.setenv("SC1_OUT", str(out))
    assert arm.main(["--ttft"]) == 0
    rec = json.loads(out.read_text())
    assert rec["mode"] == "ttft" and rec["prompt_tokens"] == [4096] and rec["ttft_prompt_tokens"] == 4096 and rec["max_new_tokens"] == 8
    assert len(rec["ttft_walls_s"]) == 3 and rec["tokens_generated"] == [8, 8, 8]


# ---------------------------------------------------------------------------------------------- K8 window + NLL

def _window(tmp_path, nll_mod, n=2561, tamper=None):
    ids = [(i * 131 + 17) % 151936 for i in range(n)]
    sha = hashlib.sha256(struct.pack(f"<{2561}q", *ids[:2561])).hexdigest() if n >= 2561 else "0" * 64
    if tamper == "sha":
        sha = "f" * 64
    p = tmp_path / "k8_window_wikitext.json"
    p.write_text(json.dumps({"ids": ids, "text_sha": sha}))
    return p, ids, sha


def test_text_sha_is_int64_le_digest_of_first_2561_ids(nll, tmp_path):
    p, ids, sha = _window(tmp_path, nll)
    w = nll.load_window(str(p))
    assert w["text_sha"] == sha == nll.text_sha(ids)
    assert len(w["ids"]) == 2561


def test_window_refusals(nll, tmp_path):
    p, _, _ = _window(tmp_path, nll, tamper="sha")
    with pytest.raises(nll.Refusal, match="text_sha"):
        nll.load_window(str(p))
    p2, _, _ = _window(tmp_path, nll, n=2000)
    with pytest.raises(nll.Refusal, match="needed"):
        nll.load_window(str(p2))


def test_nll_receipt_shape_and_ppl(nll):
    r = nll.receipt("prefill", 1.8621, "ab" * 32, 12.3, {"version": "x", "logits_dtype": "torch.float16"})
    assert all(k in r for k in nll.RECEIPT_KEYS), [k for k in nll.RECEIPT_KEYS if k not in r]
    assert abs(r["ppl"] - math.exp(1.8621)) < 1e-9 and r["steps"] == 2048 and r["prompt_len"] == 512
    assert "prefill" in r["mode_available"]


def test_exl3_nll_declares_decode_and_lmdeploy_does_not():
    e, lm = _load("exl3_nll"), _load("lmd_nll")
    assert e.MODE_AVAILABLE == ["prefill", "decode"]
    assert lm.MODE_AVAILABLE[0] == "prefill" and not any(m == "decode" for m in lm.MODE_AVAILABLE)
    assert "no token-granular" in lm.DECODE_NOTE


def test_lmdeploy_ce_window_and_logits_nll():
    torch = pytest.importorskip("torch")
    lm = _load("lmd_nll")
    assert abs(lm.ce_window(4096.0, 1024.0) - 1.5) < 1e-12
    rows = torch.zeros(5, 777)
    assert abs(lm.nll_from_logits_rows(rows, [0, 1, 2, 3, 4]) - math.log(777)) < 1e-5
    peaked = torch.full((2, 10), -50.0)
    peaked[0, 3] = 0.0
    peaked[1, 7] = 0.0
    assert lm.nll_from_logits_rows(peaked, [3, 7]) < 1e-6


def test_exl3_nll_mean_from_rows():
    e = _load("exl3_nll")
    assert abs(e.nll_from_rows(lambda t, tgt: -math.log(50.0), [1] * 2048) - math.log(50.0)) < 1e-12


# ---------------------------------------------------------------------------------------------- selftests

@pytest.mark.parametrize("name", sorted(DRIVERS))
def test_driver_selftest_runs(name):
    r = subprocess.run([sys.executable, str(DRIVERS[name]), "--selftest"], capture_output=True, text=True, timeout=300,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SELFTEST OK" in r.stdout
