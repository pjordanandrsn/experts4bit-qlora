# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lever G3's GPU smoke kit (``bench/kv-pool-g/g3-smoke/``), on the CPU: the reducer's self-test; the smoke's own
workload and schedule at 64 slots on the default bucket list, driven through the real scheduler, KV pool and decode
buckets with the ownership-checking stand-in graph from ``tests/test_kv_pool_shared.py``: claim-all and shared compare
IDENTICAL, a step runs in pieces of the largest bucket, and the short pool (physically smaller than its slots can
reach) waits for blocks and still finishes; and the box and controller scripts' plumbing.
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

KIT = pathlib.Path(__file__).resolve().parents[1] / "bench" / "kv-pool-g" / "g3-smoke"
_spec = importlib.util.spec_from_file_location("g3_smoke", KIT / "g3_smoke.py")
g3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g3)
BUCKETS = (1, 2, 4, 8, 16)
MAX_TOKENS = 2048


def test_the_self_test_passes():
    assert g3.self_test() == 0


def _short_rows(kv_full_rows, scratch, bps, slots):
    """The short run's target, as g3_smoke._short_resolver computes it: least + (claim-all - least) // 8."""
    least = scratch + max(bps, slots * (-(-min(1024, MAX_TOKENS) // 16)))
    return least + (kv_full_rows - least) // 8


def _engine(monkeypatch, shared, lookahead, short=False):
    """build_engine's composition on the CPU at 64 slots on the default bucket list: the scheduler admits by the
    pool's rows exactly when the pool is shared; the short run's pool is physically smaller than claim-all."""
    monkeypatch.setenv("E4B_KV_STEP_SELECT", "1")
    rows = None
    if short:
        rows = _short_rows(64 * (MAX_TOKENS // 16) + 16, 16, MAX_TOKENS // 16, 64)
    kv = Fp8PagedKV(2, 2, 32, batch=64, max_tokens_per_seq=MAX_TOKENS, device="cpu", scratch_slots=16, pool_rows=rows)
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
    s = ContinuousScheduler(runner=r, max_seqs=64, kv_slots=64, chunk_tokens=512, lookahead=lookahead,
                            kv_blocks=(kv._n_rows - kv.n_scratch) if shared else None, block_tokens=kv.bt)
    return s, kv


def _record(s, kv, shared, lookahead, short=False):
    reqs = g3.workload(lambda t: [b % V for b in t.encode()], eos_ids=[V], stop_extra=ord(".") % V)   # V: never emitted
    recs, steps, peak, wide = g3.drive(s, reqs, kv)
    return {"pool": "shared" if shared else "claim-all", "lookahead": lookahead, "short": short, "graphs": True,
            "replays": steps, "rows_end": g3.rows_held(kv), "incremental_claims": shared, "max_seqs": 64,
            "buckets": list(BUCKETS), "widest_decode_step": wide, "pool_rows": kv._n_rows,
            "claim_all_rows": kv.claim_all_rows, "workload_sha256": g3.digest(reqs), "requests": recs, "steps": steps,
            "rows_peak": peak, "scheduler": s.stats(), "kv_blocks": s.kv_blocks}


@pytest.fixture(scope="module")
def runs():
    mp = pytest.MonkeyPatch()
    out = {}
    try:
        for name, shared, la, short in (("ca_a", False, False, False), ("sh", True, False, False),
                                        ("ca_b", False, False, False), ("ca_la1", False, True, False),
                                        ("sh_la1", True, True, False), ("sh_short", True, False, True)):
            s, kv = _engine(mp, shared, la, short)
            out[name] = _record(s, kv, shared, la, short)
    finally:
        mp.undo()
    return out


def test_the_smoke_workload_reads_pass_on_the_cpu_engine(runs):
    v = g3.verdict(runs)
    assert v["verdict"] == "PASS", v
    assert all(line.startswith("IDENTICAL") for line in v["g3"].values()) and v["floor"].startswith("IDENTICAL")


def test_the_workload_exercises_what_the_design_names(runs):
    r = runs["ca_a"]
    assert r["requests"][g3.ABORT_RID]["finish"] == "abort"
    assert any(x["finish"] == "stop" for x in r["requests"])
    assert r["widest_decode_step"] == 64 > max(BUCKETS)           # a 64-row step ran as chained 16-row replays
    t = runs["sh_short"]
    assert t["pool_rows"] < t["claim_all_rows"] and t["scheduler"]["kv_blocks_waits"] > 0
    assert t["rows_peak"] <= t["pool_rows"] * 2 - 2 * 16           # never more than the pool's rows, per layer x 2
    assert runs["sh"]["scheduler"]["kv_blocks_waits"] == 0         # the claim-all-sized shared pool never binds here


def test_a_claim_that_misses_a_block_fails_the_smoke(monkeypatch):
    monkeypatch.setattr(PagedModelRunner, "_claim_step", lambda self, write_pos: None)
    monkeypatch.setattr(PagedModelRunner, "_claim_through", lambda self, slot, write_pos: None)
    s, kv = _engine(monkeypatch, True, False)
    with pytest.raises(AssertionError, match="owns"):
        _record(s, kv, True, False)


def test_compare_names_the_first_divergence():
    a = g3._fake("claim-all", False, outs=([1, 2, 3, 4], [5]))
    b = g3._fake("shared", False, outs=([1, 2, 7, 4], [5]))
    rc, line = g3.compare(a, b)
    assert rc == 1 and "request 0: token 2" in line


def _sh(*args, env=None):
    return subprocess.run(["bash", *args], capture_output=True, text=True, env=env)


def test_the_pin_covers_every_staged_file_and_matches():
    pin = (KIT / "staged.sha256").read_text().split("\n")
    names = sorted(line.split()[1] for line in pin if line.strip() and not line.startswith("#"))
    assert names == sorted(["g3_smoke_run.sh", "g3_smoke.py", "k8_bake.py", "calib.json"])
    out = subprocess.run(["bash", str(KIT / "make_pin.sh"), "--check"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_the_box_runs_the_six_runs_in_order_with_their_pools():
    src = (KIT / "g3_smoke_run.sh").read_text()
    order = [line.split()[1] for line in src.splitlines() if line.startswith("one ")]
    assert order == list(g3.RUNS)
    for line in src.splitlines():
        if line.startswith("one "):
            name = line.split()[1]
            assert ("E4B_PAGED_KV_POOL=shared" in line) == name.startswith("sh")
            assert ("E4B_PAGED_DECODE_LOOKAHEAD=1" in line) == name.endswith("la1")
            assert ("--short" in line) == (name == "sh_short")
    assert "E4B_PAGED_MAX_SEQS=64 " in src and "E4B_PAGED_BUCKETS=1,2,4,8,16 " in src   # the multi-piece path


def test_the_driver_refuses_without_the_launcher(tmp_path):
    env = {"PATH": "/usr/bin:/bin"}
    out = _sh(str(KIT / "g3_smoke_drive.sh"), env=env)
    assert out.returncode == 78 and "E4B_RENT_SSH_HOST" in out.stdout


def test_the_box_script_parses():
    for f in ("g3_smoke_run.sh", "g3_smoke_drive.sh", "make_pin.sh"):
        assert _sh("-n", str(KIT / f)).returncode == 0, f


def test_the_short_target_is_the_kits():
    """The test's short pool is the box's: g3_smoke._short_resolver's least + (claim-all - least) // 8."""
    import inspect
    src = inspect.getsource(g3._short_resolver)
    assert 'c["least_rows"] + (c["claim_all_rows"] - c["least_rows"]) // 8' in src
