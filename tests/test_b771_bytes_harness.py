"""Lane B771's byte-level harness (bench/b771/b771_bytes.py) counts what it claims to count.

The box runs it against gnf4's real fused append with the sm_89+ hardware cast; nothing here can. What CI can pin
is the harness's own layout math: with a pure-torch stand-in for ``fp8_kv_append_bt1`` that writes
``quantize_kv_fp8``'s bytes at the kernel's addresses, it must report 0 differing bytes, and exactly the bytes a
mutation flips. Otherwise a 0 from the box would say nothing.
"""
import importlib.util
import pathlib
import types

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
fp8_kv = pytest.importorskip("fp8_kv")
if not hasattr(torch, "float8_e4m3fn"):
    pytest.skip("torch without float8_e4m3fn", allow_module_level=True)


def _harness():
    spec = importlib.util.spec_from_file_location("b771_bytes", REPO / "bench" / "b771" / "b771_bytes.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fake_kv(flip=None):
    """fp8_kv_append_bt1's contract in torch: one token per slot, row = tbl[slot, lens[slot] // bt],
    payload at fill*H*D + h*D, scales (fp32) at pay_bytes + (fill*H*G + h*G)*4. ``flip`` = (call, byte) to corrupt."""
    calls = {"n": 0}

    def append_bt1(x, pool, tbl, slot_idx, lens, row_bytes, pay_bytes, bt, groups):
        S, H, D = x.shape
        q, s = fp8_kv.quantize_kv_fp8(x, group=None if groups == 1 else D // groups)
        qb = q.view(torch.uint8).reshape(S, H * D)
        sb = s.float().reshape(S, -1).view(torch.uint8)
        for i in range(S):
            slot = int(slot_idx[i])
            pos = int(lens[slot])
            row = int(tbl[slot, pos // bt])
            fill = pos % bt
            base = row * row_bytes
            pool[base + fill * H * D: base + (fill + 1) * H * D] = qb[i]
            sc = base + pay_bytes + fill * H * groups * 4
            pool[sc: sc + H * groups * 4] = sb[i]
        if flip is not None and calls["n"] == flip[0]:
            pool[flip[1]] ^= 1
        calls["n"] += 1
    return types.SimpleNamespace(fp8_kv_append_bt1=append_bt1, quantize_kv_fp8=fp8_kv.quantize_kv_fp8)


@pytest.mark.parametrize("groups", [1, 4])
def test_the_harness_reads_zero_when_the_append_writes_the_reference(groups):
    r = _harness().run(groups, 1.0, calls=3, S=8, H=2, D=32, dev="cpu", kv=_fake_kv())
    assert r["values"] == 8 * 2 * 32 * 3
    assert r["payload_bytes_differing"] == 0 and r["scales_differing"] == 0, r


def test_the_harness_counts_a_flipped_payload_byte():
    r = _harness().run(1, 1.0, calls=3, S=8, H=2, D=32, dev="cpu", kv=_fake_kv(flip=(1, 5)))
    assert r["payload_bytes_differing"] == 1 and r["scales_differing"] == 0, r
