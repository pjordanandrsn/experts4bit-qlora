"""HP1's evaluator library (bench/hp1/hp1_lib.py, HP1-PREREG.md): the canonical adapter is layout-free, the fingerprint is exact,
the evaluator's dequantizer reads a stored slot back bit for bit, and its eager expert forward is HF's math when no adapter is on."""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "hp1"))
import hp1_lib  # noqa: E402

E, R, H, INTER = 4, 3, 16, 8          # experts, rank, hidden, intermediate


def _per_expert(seed):
    g = torch.Generator().manual_seed(seed)
    gu = [(torch.randn(R, H, generator=g), torch.randn(2 * INTER, R, generator=g)) for _ in range(E)]
    dn = [(torch.randn(R, INTER, generator=g), torch.randn(H, R, generator=g)) for _ in range(E)]
    q = (torch.randn(R, H, generator=g), torch.randn(H, R, generator=g))
    return gu, dn, q


def _e4b_model_and_slots(gu, dn, q, scaling):
    """e4b's layout: ExpertsLoRA <prefix>_lora_A [E, r, in] / _lora_B [E, out, r]; LoRALinear lora_A [r, in] / lora_B [out, r]."""
    root = nn.Module()
    ex, qp = nn.Module(), nn.Module()
    ex.scaling, qp.scaling = scaling, scaling
    ex.gate_up_lora_A = nn.Parameter(torch.stack([a for a, _ in gu]))
    ex.gate_up_lora_B = nn.Parameter(torch.stack([b for _, b in gu]))
    ex.down_lora_A = nn.Parameter(torch.stack([a for a, _ in dn]))
    ex.down_lora_B = nn.Parameter(torch.stack([b for _, b in dn]))
    qp.lora_A, qp.lora_B = nn.Parameter(q[0].clone()), nn.Parameter(q[1].clone())
    root.add_module("ex", ex)
    root.add_module("qp", qp)
    slots = [{"key": (0, k, e), "A": getattr(ex, f"{k}_lora_A"), "B": getattr(ex, f"{k}_lora_B"), "sel": ("index", e), "module": f"ex.{k}_lora_A"}
             for k in ("gate_up", "down") for e in range(E)]
    slots.append({"key": (0, "q", -1), "A": qp.lora_A, "B": qp.lora_B, "sel": ("all",), "module": "qp"})
    return root, slots


def _peft_model_and_slots(gu, dn, q, scaling):
    """PEFT 0.21.2's layout: ParamWrapper A [r*E, in] (expert e = rows e*r:(e+1)*r), B [out, r*E] viewed [out, r, E]."""
    root = nn.Module()
    wg, wd, qp = nn.Module(), nn.Module(), nn.Module()
    for m in (wg, wd, qp):
        m.scaling = {"default": scaling}
    root.add_module("wg", wg)
    root.add_module("wd", wd)
    root.add_module("qp", qp)
    slots = []
    for name, m, pairs, kind in (("wg", wg, gu, "gate_up"), ("wd", wd, dn, "down")):
        A = torch.cat([a for a, _ in pairs], 0)
        B = torch.stack([b for _, b in pairs], dim=2).reshape(pairs[0][1].shape[0], R * E)
        m.A, m.B = nn.Parameter(A), nn.Parameter(B)
        slots += [{"key": (0, kind, e), "A": m.A, "B": m.B, "sel": ("rows", e * R, (e + 1) * R), "module": name} for e in range(E)]
    qp.A, qp.B = nn.Parameter(q[0].clone()), nn.Parameter(q[1].clone())
    slots.append({"key": (0, "q", -1), "A": qp.A, "B": qp.B, "sel": ("all",), "module": "qp"})
    return root, slots


def test_canonical_export_is_layout_free(tmp_path):
    gu, dn, q = _per_expert(0)
    m1, s1 = _e4b_model_and_slots(gu, dn, q, 1.0)
    m2, s2 = _peft_model_and_slots(gu, dn, q, 1.0)
    r1 = hp1_lib.export_canonical(m1, s1, str(tmp_path / "e4b.safetensors"))
    r2 = hp1_lib.export_canonical(m2, s2, str(tmp_path / "peft.safetensors"))
    t1, sc1, _ = hp1_lib.load_canonical(r1["path"], "cpu")
    t2, sc2, _ = hp1_lib.load_canonical(r2["path"], "cpu")
    assert sorted(t1) == sorted(t2) == ["0.down.A", "0.down.B", "0.gate_up.A", "0.gate_up.B", "0.q.A", "0.q.B"]
    for k in t1:
        assert torch.equal(t1[k], t2[k]), k
    assert sc1 == sc2 == {"0.down": 1.0, "0.gate_up": 1.0, "0.q": 1.0}
    assert t1["0.gate_up.A"].shape == (E, R, H) and t1["0.gate_up.B"].shape == (E, 2 * INTER, R)
    for e in range(E):                       # expert e's slice is exactly that expert's (A, B)
        assert torch.equal(t1["0.down.A"][e], dn[e][0]) and torch.equal(t1["0.down.B"][e], dn[e][1])


def test_canonical_export_refuses_duplicates_and_mixed_scaling(tmp_path):
    gu, dn, q = _per_expert(1)
    m, s = _e4b_model_and_slots(gu, dn, q, 1.0)
    with pytest.raises(ValueError, match="duplicate"):
        hp1_lib.export_canonical(m, s + [s[0]], str(tmp_path / "dup.safetensors"))
    m.qp.scaling = 2.0
    s_bad = [dict(x, key=(0, "gate_up", x["key"][2]) if x["key"][1] == "q" else x["key"]) for x in s]
    s_bad[-1]["key"] = (0, "down", E)        # the q pair renamed into the down stack, with its different scaling
    with pytest.raises(ValueError, match="mixed scalings|do not chain"):
        hp1_lib.export_canonical(m, s_bad, str(tmp_path / "mixed.safetensors"))


def test_fingerprint_is_exact_and_order_sensitive():
    w = torch.randn(32, 16).to(torch.bfloat16)
    assert hp1_lib.slot_fingerprint(w) == hp1_lib.slot_fingerprint(w.clone())
    w2 = w.clone()
    w2.view(torch.int16).view(-1)[5] ^= 1          # one mantissa bit of one value
    assert hp1_lib.slot_fingerprint(w2) != hp1_lib.slot_fingerprint(w)
    assert hp1_lib.slot_fingerprint(w.flip(0)) != hp1_lib.slot_fingerprint(w)
    d1 = hp1_lib.fold_fingerprints({(0, "q", -1): (w.numel(), *hp1_lib.slot_fingerprint(w))})
    d2 = hp1_lib.fold_fingerprints({(0, "q", -1): (w.numel(), *hp1_lib.slot_fingerprint(w2))})
    assert d1[0] != d2[0] and d1[1] == 1


def test_dequant_raw_reads_a_stored_nf4_slot_back_bitwise():
    bf = pytest.importorskip("bitsandbytes.functional")
    w = torch.randn(32, 64, dtype=torch.bfloat16)
    try:
        packed, qs = bf.quantize_4bit(w, blocksize=64, quant_type="nf4")
    except Exception as e:                   # a bitsandbytes build without a CPU 4-bit path
        pytest.skip(f"no CPU quantize_4bit here: {e}")
    ref = bf.dequantize_4bit(packed, quant_state=qs).to(torch.bfloat16)
    raw = {"fmt": "nf4", "packed": packed.reshape(-1), "absmax": qs.absmax.float(), "shape": [32, 64], "blocksize": 64,
           "code": qs.code.float(), "transposed": False, "dtype": str(qs.dtype).replace("torch.", "")}
    assert torch.equal(hp1_lib.dequant_raw(raw, "cpu"), ref)
    raw_t = dict(raw, transposed=True)
    assert torch.equal(hp1_lib.dequant_raw(raw_t, "cpu"), ref.t())


def test_eager_experts_without_adapter_is_hf_math_and_the_delta_moves_it():
    g = torch.Generator().manual_seed(3)
    gate_up = (torch.randn(E, 2 * INTER, H, generator=g) * 0.1).to(torch.bfloat16)
    down = (torch.randn(E, H, INTER, generator=g) * 0.1).to(torch.bfloat16)
    x = torch.randn(6, H, generator=g).to(torch.bfloat16)
    idx = torch.tensor([[0, 1], [2, 3], [1, 2], [0, 3], [3, 1], [2, 0]])
    wts = torch.softmax(torch.randn(6, 2, generator=g), -1)
    act = nn.SiLU()
    ref = torch.zeros(6, H, dtype=torch.float32)
    for t in range(6):
        for k in range(2):
            e = int(idx[t, k])
            gt, up = (x[t:t + 1] @ gate_up[e].t()).chunk(2, dim=-1)
            y = (act(gt) * up) @ down[e].t()
            ref[t] += y[0].float() * wts[t, k]
    deq = lambda kind, e: (gate_up if kind == "gate_up" else down)[e]       # noqa: E731
    m = hp1_lib.EagerExperts(deq, E, act)
    out = m(x, idx, wts)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2)
    zA = {"gate_up": (torch.zeros(E, R, H), torch.zeros(E, 2 * INTER, R), 1.0), "down": (torch.zeros(E, R, INTER), torch.zeros(E, H, R), 1.0)}
    m.adapter = zA
    assert torch.equal(m(x, idx, wts), out)                                 # a zero delta changes nothing
    nzA = dict(zA, down=(torch.randn(E, R, INTER, generator=g), torch.randn(E, H, R, generator=g), 1.0))
    m.adapter = nzA
    assert not torch.equal(m(x, idx, wts), out)



def _whole_slot_fingerprint(w):
    """The fingerprint as it was computed before chunking: the whole slot as one int64 tensor (HPN read 6's OOM)."""
    w = w.detach().contiguous()
    if w.element_size() != 2:
        w = w.to(torch.bfloat16)
    v = w.view(torch.int16).reshape(-1).to(torch.int64)
    pos = (torch.arange(v.numel(), device=v.device, dtype=torch.int64) % hp1_lib._P) + 1
    return int(v.sum().item()), int((v * pos).sum().item())


def _equal_at(shape):
    g = torch.Generator().manual_seed(shape[0])
    w = (torch.randn(shape, generator=g) * 0.05).to(torch.bfloat16)
    assert hp1_lib.slot_fingerprint(w) == _whole_slot_fingerprint(w)


@pytest.mark.parametrize("shape", [(1536, 2048), (2048, 768), (5 * (1 << 23),)])
def test_the_chunked_fingerprint_is_bit_identical_to_the_whole_slot_form(shape):
    """An expert's gate_up (1,536 x 2,048) and down (2,048 x 768) at Qwen3-30B-A3B's real sizes, and a slot of 2.5 default chunks
    (it crosses two chunk boundaries, under 1 GiB with the whole-slot reference), as bf16 on the CPU: the sums are equal."""
    _equal_at(shape)


@pytest.mark.skipif(os.environ.get("HP1_REALSIZE") != "1", reason="opt-in (HP1_REALSIZE=1): the whole-slot reference peaks at ~8 GiB")
def test_the_chunked_fingerprint_is_bit_identical_at_the_lm_heads_real_size():
    """Qwen3-30B-A3B's lm_head / embed, 151,936 x 2,048: the slot that ran the A2000 out of memory at L8. Opt-in, because its
    whole-slot reference rebuilds the three int64 copies on the host; its recorded run is on the PR."""
    _equal_at((151936, 2048))


@pytest.mark.parametrize("chunk,n", [(7, 1), (7, 7), (7, 50), (1000, 4096 + 13), (1 << 10, 1 << 16)])
def test_the_chunk_boundaries_do_not_move_the_fingerprint(monkeypatch, chunk, n):
    g = torch.Generator().manual_seed(n)
    w = torch.randn(n, generator=g).to(torch.bfloat16)
    want = _whole_slot_fingerprint(w)
    monkeypatch.setattr(hp1_lib, "FINGERPRINT_CHUNK", chunk)
    assert hp1_lib.slot_fingerprint(w) == want


def test_the_whole_slot_form_cannot_wrap_for_any_slot_this_lane_reads():
    """The weighted sum's terms are at most 2**15 x 65521; it would wrap int64 only past 2**63 / (2**15 x 65521), about 4.3e9
    elements per slot. Qwen3's largest slot is 311M, so there is no wraparound to reproduce (the chunked accumulators would wrap
    identically anyway: they are int64 adds, regrouped)."""
    limit = (2 ** 63 - 1) // (2 ** 15 * hp1_lib._P)
    assert limit > 4_000_000_000 > 151936 * 2048
    assert hp1_lib.FINGERPRINT_CHUNK * 8 * 3 <= 384 * 2 ** 20                    # the stated bound: three int64 chunks
