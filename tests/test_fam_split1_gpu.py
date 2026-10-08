# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane FAM's ``split1`` floor arm (bench/fam/PREREG-fam.md; e4b#1362) is a neutral perturbation the real kernel honours.

``split1`` forces every decode attention to ONE KV split (``n_split=1``, passed through ``Fp8PagedKV.attention``'s
``**kw`` to grouped-nf4-gemm's ``fp8_paged_decode_attention``). The kernel fixes its reduction order per (config,
split count), so one split is the same sums in another order. For it to be a floor draw it must (a) stay inside the
kernel's own serving tolerance of the automatic split count and (b) change some bits at the instrument's lengths --
otherwise the arm repeats R and is not a draw. Both are checked here, at the head dims FAM's families serve: 64
(Granite, gpt-oss; f32 compute) and 128.
"""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device (the real paged decode attention kernel)", allow_module_level=True)
pytest.importorskip("fp8_paged_attn", reason="needs grouped-nf4-gemm's paged decode attention")

from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV  # noqa: E402

LENS = [640, 513, 700]                     # prompt 512 + up to 128 decoded positions, as the instrument runs


@pytest.mark.parametrize("hq,hkv,d", [(24, 8, 64), (32, 4, 128)])
def test_one_split_is_a_reordering_inside_tolerance(hq, hkv, d):
    g = torch.Generator().manual_seed(1362)
    kv = Fp8PagedKV(1, hkv, d, batch=len(LENS), max_tokens_per_seq=768, k_groups=max(4, d // 32), device="cuda")
    for seq, t in enumerate(LENS):
        kv.append(0, seq, (torch.randn(t, hkv, d, generator=g) * 1.5).cuda(), torch.randn(t, hkv, d, generator=g).cuda())
    q = (torch.randn(len(LENS), hq, d, generator=g) * 0.5).to(torch.bfloat16).cuda()
    auto = kv.attention(0, q)
    one = kv.attention(0, q, n_split=1)
    again = kv.attention(0, q, n_split=1)
    assert torch.equal(one, again), "n_split=1 is not deterministic"
    diff = (one.float() - auto.float()).abs().max().item()
    assert diff <= 2e-2, f"n_split=1 moved the output by {diff}: not a reordering"
    assert not torch.equal(one, auto), "n_split=1 repeats the automatic split bit for bit: not a floor draw here"
