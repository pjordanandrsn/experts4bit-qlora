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
    cap = rec["capacity"]
    if arm.ENGINE == "exllamav3":
        assert cap["cache_tokens"] == 2 * 2048 and cap["cache_tokens_source"] == "default:B*2048" and cap["max_batch_size"] == 2
    else:
        assert cap["session_len"] == 2048 and cap["session_len_source"] == "default:2048" and cap["max_batch_size"] == 2


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
    cap = rec["capacity"]
    if arm.ENGINE == "exllamav3":
        assert cap["cache_tokens"] == 4352 and cap["cache_tokens_source"].startswith("default:ttft")
    else:
        assert cap["session_len"] == 4104 and cap["session_len_source"].startswith("default:ttft")


# ---------------------------------------------------------------------------------------------- registered capacity rule

def test_exl3_capacity_rule_b_times_2048_and_ttft_4352():
    e = _load("exl3_arm")
    assert e.cache_tokens_for(16, 512, 128, {}) == (32768, "default:B*2048")
    assert e.cache_tokens_for(1, 512, 128, {}) == (2048, "default:B*2048")
    assert e.cache_tokens_for(1, 512, 8, {}) == (2048, "default:B*2048")
    assert e.cache_tokens_for(1, 4096, 8, {}) == (4352, "default:ttft:roundup256(prompt+max_new)")
    assert all(t % 256 == 0 for t in (32768, 2048, 4352))
    assert e.cache_tokens_for(16, 512, 128, {"SC1_EXL3_CACHE_TOKENS": "16384"}) == (16384, "env:SC1_EXL3_CACHE_TOKENS")
    with pytest.raises(e.Refusal):                      # not a page multiple
        e.cache_tokens_for(16, 512, 128, {"SC1_EXL3_CACHE_TOKENS": "1000"})
    with pytest.raises(e.Refusal):                      # cannot hold 16 x roundup256(640) = 12288
        e.cache_tokens_for(16, 512, 128, {"SC1_EXL3_CACHE_TOKENS": "8192"})
    blk = e.capacity_block(16, 512, 128, 32768, "default:B*2048")
    assert blk["sequences_x_2048"] == 16 and blk["max_batch_size"] == 16 and blk["page_size"] == 256


def test_lmdeploy_capacity_rule_session_len_2048_and_ttft_4104():
    lm = _load("lmd_arm")
    assert lm.session_len_for(512, 128, {}) == (2048, "default:2048")
    assert lm.session_len_for(512, 8, {}) == (2048, "default:2048")
    assert lm.session_len_for(4096, 8, {}) == (4104, "default:ttft:prompt+max_new")
    assert lm.session_len_for(512, 128, {"SC1_LMDEPLOY_SESSION_LEN": "3000"}) == (3000, "env:SC1_LMDEPLOY_SESSION_LEN")
    with pytest.raises(lm.Refusal):                     # 600 < 512 + 128 + 1
        lm.session_len_for(512, 128, {"SC1_LMDEPLOY_SESSION_LEN": "600"})
    blk = lm.capacity_block(16, 512, 128, 2048, "default:2048", {"cache_max_entry_count_resolved": 0.8})
    assert blk["kv_tokens_held_target"] == 16 * 2048 and blk["cache_max_entry_count_resolved"] == 0.8


def test_scorers_set_their_own_capacity_above_the_arm_defaults():
    e, lm = _load("exl3_nll"), _load("lmd_nll")
    assert e.NLL_CACHE_TOKENS_MIN == 2816 and e.nll_cache_tokens({}) == (2816, "default:roundup256(P+S+1)")
    assert e.nll_cache_tokens({"SC1_EXL3_CACHE_TOKENS": "3072"}) == (3072, "env:SC1_EXL3_CACHE_TOKENS")
    with pytest.raises(e.Refusal):                      # 2560 < 2816: the window would not fit
        e.nll_cache_tokens({"SC1_EXL3_CACHE_TOKENS": "2560"})
    with pytest.raises(e.Refusal):                      # 2048 = the arm's B=1 default is BELOW the scorer floor
        e.nll_cache_tokens({"SC1_EXL3_CACHE_TOKENS": "2048"})
    assert lm.NLL_SESSION_LEN_MIN == 2562 and lm.nll_session_len({}) == (2816, "default:2816 (>= P+S+2)")
    assert lm.nll_session_len({"SC1_LMDEPLOY_SESSION_LEN": "2562"})[0] == 2562
    with pytest.raises(lm.Refusal):                     # 2561 input + 1 generated does not fit
        lm.nll_session_len({"SC1_LMDEPLOY_SESSION_LEN": "2561"})
    with pytest.raises(lm.Refusal):                     # the arm's 2048 default is below the scorer floor
        lm.nll_session_len({"SC1_LMDEPLOY_SESSION_LEN": "2048"})


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
    r = nll.receipt("prefill", 1.8621, "ab" * 32, 12.3, {"version": "x", "logits_dtype": "torch.float16",
                                                       "capacity": {"independent_of_arm_defaults": True}})
    assert all(k in r for k in nll.RECEIPT_KEYS), [k for k in nll.RECEIPT_KEYS if k not in r]
    assert r["capacity"]["independent_of_arm_defaults"] is True
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


# --- A3 (sc1b-prove-5): the ExLlamaV3 tripwire reads the version module and the real v1.5.3 binding names -----------------

EXL3_SYMBOLS = {   # m.def / py::class_ names at exllamav3 v1.5.3 (exllamav3/exllamav3_ext/bindings.cpp:160-272; libtorch/*_bc.h)
    "exl3_gemm", "exl3_mgemm", "exl3_moe", "exl3_gemv", "exl3_gemv_int8_max_k",
    "BC_LinearEXL3", "BC_BlockSparseMLP", "BC_Attention",
}


def test_exl3_tripwire_reads_the_version_module(tmp_path):
    """sc1b-prove-5: v1.5.3 keeps `__version__` in exllamav3/version.py and its __init__ does not import it, so
    `exllamav3.version.__version__` raised AttributeError and the install read rc 13. Run the tripwire's own version lines
    against a stand-in package laid out like v1.5.3."""
    src = (REPO / "bench" / "sc1" / "exl3" / "install.sh").read_text()
    lines = [ln for ln in src.splitlines() if ln.startswith(("import exllamav3", "ver = "))]
    assert any(ln.startswith("import exllamav3.version") for ln in lines), lines
    pkg = tmp_path / "exllamav3"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "version.py").write_text('__version__ = "1.5.3"\n')
    code = "\n".join(lines + ['assert ver == "1.5.3", ver', 'print("ok")'])
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path,
                       env={**os.environ, "PYTHONPATH": str(tmp_path)})
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
    # positive control: the registered line, on the same package, is the AttributeError the box hit
    old = 'import exllamav3\nver = getattr(exllamav3, "__version__", None) or exllamav3.version.__version__\n'
    r = subprocess.run([sys.executable, "-c", old], capture_output=True, text=True, cwd=tmp_path,
                       env={**os.environ, "PYTHONPATH": str(tmp_path)})
    assert r.returncode != 0 and "AttributeError" in r.stderr


def test_exl3_tripwire_requires_only_symbols_that_exist_in_v1_5_3():
    import re
    src = (REPO / "bench" / "sc1" / "exl3" / "install.sh").read_text()
    m = re.search(r"caps = \{k: hasattr\(E, k\) for k in \((.*?)\)\}", src, re.S)
    assert m, "the tripwire's symbol tuple moved"
    got = set(re.findall(r'"([A-Za-z0-9_]+)"', m.group(1)))
    assert got == EXL3_SYMBOLS, got ^ EXL3_SYMBOLS
