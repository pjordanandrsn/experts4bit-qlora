# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lever G1's GPU smoke kit (``bench/kv-pool-g/g1-smoke/``), on the CPU: the reducer's self-test; the smoke's own
workload and schedule driven through the real scheduler, KV pool and decode buckets with the ownership-checking stand-in
graph from ``tests/test_kv_pool_shared.py``, where claim-all and shared must compare IDENTICAL and the tight run must
wait for blocks and still finish; and the box and controller scripts' plumbing (the pin, the run order, the refusals).
"""
import importlib.util
import pathlib
import subprocess

import pytest
import torch

from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
from experts4bit_qlora.engines.paged_runner import PagedModelRunner
from experts4bit_qlora.engines.scheduler import ContinuousScheduler
from test_kv_pool_shared import V, OwnershipCheckingDecode

KIT = pathlib.Path(__file__).resolve().parents[1] / "bench" / "kv-pool-g" / "g1-smoke"
_spec = importlib.util.spec_from_file_location("g1_smoke", KIT / "g1_smoke.py")
g1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g1)
BUCKETS = (1, 2, 4, 8, 16)


def test_the_self_test_passes():
    assert g1.self_test() == 0


def _engine(monkeypatch, shared, lookahead, tight=False):
    """build_engine's composition on the CPU: 16 slots at 2,048 tokens, the scheduler admitting by reserved blocks
    exactly when the pool is shared."""
    monkeypatch.setenv("E4B_KV_STEP_SELECT", "1")
    kv = Fp8PagedKV(2, 2, 32, batch=16, max_tokens_per_seq=2048, device="cpu", scratch_slots=16)
    r = PagedModelRunner(torch.nn.Module(), kv, device="cpu", bulk_kv=True)
    r.incremental_claims = shared
    r.enable_decode_graphs(BUCKETS, capture=False, verbose=False)
    hist = torch.zeros(kv.B + kv.n_scratch, dtype=torch.long)
    seen = []
    for b in BUCKETS:
        r._graphs[b] = OwnershipCheckingDecode(r, b, hist, seen)

    def prefill(chunks):
        first = {}
        for rid, start, take in chunks:
            slot, n = r.slot_of[rid], len(r.tokens[rid])
            if start + take >= n:
                kv.seq_lens[:, slot] = n
                hist[slot] = sum((i + 1) * t for i, t in enumerate(r.tokens[rid])) % 2_147_483_647
                first[rid] = int(hist[slot]) % V
                r.tokens[rid].append(first[rid])
                r.pos_of[rid] = n + 1
        return first

    r.run_prefill = prefill
    s = ContinuousScheduler(runner=r, max_seqs=16, kv_slots=16, chunk_tokens=512, lookahead=lookahead,
                            kv_blocks=(kv.B * kv.blocks_per_seq) if shared else None, block_tokens=kv.bt)
    if tight:
        s.kv_blocks = g1.TIGHT_BLOCKS
    return s, kv


def _record(s, kv, shared, lookahead, tight=False):
    reqs = g1.workload(lambda t: [b % V for b in t.encode()], eos_ids=[V], stop_extra=ord(".") % V)   # V: an EOS the stand-in never emits
    recs, steps, peak = g1.drive(s, reqs, kv)
    return {"pool": "shared" if shared else "claim-all", "lookahead": lookahead, "tight": tight, "graphs": True,
            "replays": steps, "rows_end": g1.rows_held(kv), "incremental_claims": shared,
            "workload_sha256": g1.digest(reqs), "requests": recs, "steps": steps, "rows_peak": peak,
            "scheduler": s.stats(), "kv_blocks": s.kv_blocks}


@pytest.fixture
def runs(monkeypatch):
    out = {}
    for name, shared, la, tight in (("ca_la0_a", False, False, False), ("sh_la0", True, False, False),
                                    ("ca_la0_b", False, False, False), ("ca_la1", False, True, False),
                                    ("sh_la1", True, True, False), ("sh_tight", True, False, True)):
        s, kv = _engine(monkeypatch, shared, la, tight)
        out[name] = _record(s, kv, shared, la, tight)
    return out


def test_the_smoke_workload_reads_pass_on_the_cpu_engine(runs):
    v = g1.verdict(runs)
    assert v["verdict"] == "PASS", v
    assert all(line.startswith("IDENTICAL") for line in v["g1"].values()) and v["floor"].startswith("IDENTICAL")
    for k in ("sh_la0", "sh_la1"):                               # shared holds fewer rows at its peak
        assert runs[k]["rows_peak"] < runs["ca_la0_a"]["rows_peak"]


def test_the_workload_exercises_what_the_design_names(runs):
    r = runs["ca_la0_a"]["requests"]
    assert r[g1.ABORT_RID]["finish"] == "abort"                  # the abort lands mid-decode
    assert any(x["finish"] == "stop" for x in r)                 # a stop id fires before max_new
    assert sum(1 for x in r if x["finish"] == "length") >= 10
    assert runs["sh_tight"]["scheduler"]["kv_blocks_waits"] > 0  # the tight reservation makes admission wait
    assert runs["ca_la0_a"]["scheduler"].get("kv_blocks_waits", 0) == 0


def test_a_claim_that_misses_a_block_fails_the_smoke(monkeypatch):
    """The CPU stand-in's ownership check is the GPU's corrupted-value failure made loud: with the per-step claim
    removed, the shared run cannot even finish."""
    monkeypatch.setattr(PagedModelRunner, "_claim_step", lambda self, write_pos: None)
    monkeypatch.setattr(PagedModelRunner, "_claim_through", lambda self, slot, write_pos: None)
    s, kv = _engine(monkeypatch, True, False)
    with pytest.raises(AssertionError, match="owns"):
        _record(s, kv, True, False)


def test_compare_names_the_first_divergence():
    a = g1._fake("claim-all", False, outs=([1, 2, 3, 4], [5]))
    b = g1._fake("shared", False, outs=([1, 2, 7, 4], [5]))
    rc, line = g1.compare(a, b)
    assert rc == 1 and "request 0: token 2" in line


def _sh(*args, env=None):
    return subprocess.run(["bash", *args], capture_output=True, text=True, env=env)


def test_the_pin_covers_every_staged_file_and_matches():
    pin = (KIT / "staged.sha256").read_text().split("\n")
    names = sorted(line.split()[1] for line in pin if line.strip() and not line.startswith("#"))
    assert names == sorted(["g1_smoke_run.sh", "g1_smoke.py", "k8_bake.py", "calib.json"])
    out = subprocess.run(["bash", str(KIT / "make_pin.sh"), "--check"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_the_box_runs_the_six_runs_in_order_with_their_pools():
    src = (KIT / "g1_smoke_run.sh").read_text()
    order = [line.split()[1] for line in src.splitlines() if line.startswith("one ")]
    assert order == list(g1.RUNS)
    for line in src.splitlines():
        if line.startswith("one "):
            name = line.split()[1]
            assert ("E4B_PAGED_KV_POOL=shared" in line) == name.startswith("sh_")
            assert ("E4B_PAGED_DECODE_LOOKAHEAD=1" in line) == name.endswith("la1")
            assert ("--tight" in line) == (name == "sh_tight")


def test_the_driver_refuses_without_the_launcher(tmp_path):
    env = {"PATH": "/usr/bin:/bin"}
    out = _sh(str(KIT / "g1_smoke_drive.sh"), env=env)
    assert out.returncode == 78 and "E4B_RENT_SSH_HOST" in out.stdout


def test_the_box_script_parses():
    for f in ("g1_smoke_run.sh", "g1_smoke_drive.sh", "make_pin.sh"):
        assert _sh("-n", str(KIT / f)).returncode == 0, f
