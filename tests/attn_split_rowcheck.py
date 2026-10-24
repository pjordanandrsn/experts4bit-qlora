# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""attn_split_rowcheck.py COMPUTE S [S ...] (run by tests/test_attn_split_fixed.py in a subprocess, since TRITON_INTERPRET
is read when the kernels are defined): does one row's paged-attention output depend on the batch it decodes in, at
a FIXED n_split? grouped-nf4-gemm's real fp8_paged_decode_attention under TRITON_INTERPRET=1 on CPU, over pools written
by e4b's own Fp8PagedKV. Slot r holds its own KV (seeded by r, LENS[r] tokens); row 0 decodes in a batch of 2 and of
16, and its output bytes are compared with every per-call choice pinned. The only stubs are CUDA's: the availability
probe, the per-stream arrival-counter cache, and (fp8) the device-capability check."""
import os
import sys
import time

os.environ["TRITON_INTERPRET"] = "1"
import torch  # noqa: E402
import fp8_paged_attn as fpa  # noqa: E402
from experts4bit_qlora.engines import fp8_paged_kv as kvmod  # noqa: E402

fpa.paged_attn_available = lambda: True
fpa._fuse_counters = lambda n, device: torch.zeros(n, dtype=torch.int32, device=device)
H_KV, G, D = 2, 4, 64
LENS = [300, 77, 512, 129, 33, 450, 260, 64, 201, 389, 17, 498, 140, 222, 311, 95]

kv = kvmod.Fp8PagedKV(1, H_KV, D, batch=16, max_tokens_per_seq=640, device="cpu")
for r, n in enumerate(LENS):
    g = torch.Generator().manual_seed(1000 + r)
    kv.append(0, r, (torch.randn(n, H_KV, D, generator=g) * 0.5).to(torch.bfloat16),
              (torch.randn(n, H_KV, D, generator=g) * 0.5).to(torch.bfloat16))
Q = torch.stack([torch.randn(H_KV * G, D, generator=torch.Generator().manual_seed(5000 + r)) for r in range(16)]).to(
    torch.bfloat16)


def row0(n_rows, compute, S):
    kf, vf, tbl, lens = kv.kernel_args(0, list(range(n_rows)))
    assert int(lens[0]) == LENS[0]
    o = fpa.fp8_paged_decode_attention(Q[:n_rows].contiguous(), kf, vf, tbl, lens, n_kv_heads=H_KV, head_dim=D,
                                       block_tokens=kv.bt, k_groups=kv.kgs[0], k_row_bytes=kv.k_row,
                                       v_row_bytes=kv.v_row, n_split=S, compute=compute, fuse_combine=True,
                                       num_warps=4, ktile=64 if compute == "fp8" else 32, num_stages=3,
                                       pack_heads=False)
    return o[0].contiguous()


compute = sys.argv[1]
if compute == "fp8":
    fpa.fp8_compute_unsupported = lambda *a, **k: None
# the CONTROL: each batch at the library's own plan for its row count (an RTX 5090's 170 SMs), the default the knob
# replaces; row 0 must DIFFER between B=2 and B=16, or this check could not see a batch dependence at all
plan = {2: 12, 16: 2}       # P223's recorded default plan at buckets 2 and 16 on the card (Mixtral, an RTX 5090)
a, b = row0(2, compute, plan[2]), row0(16, compute, plan[16])
print(f"CONTROL {compute} plan B=2 {plan[2]} B=16 {plan[16]}: "
      f"{'EQUAL' if torch.equal(a.view(torch.int16), b.view(torch.int16)) else 'DIFFERS'} max|d| "
      f"{float((a.float() - b.float()).abs().max()):.3g}", flush=True)
for S in [int(s) for s in sys.argv[2:]]:
    t0 = time.time()
    try:
        a, b = row0(2, compute, S), row0(16, compute, S)
        same = torch.equal(a.view(torch.int16), b.view(torch.int16))
        print(f"ROWCHECK {compute} S={S}: {'EQUAL' if same else 'DIFFERS'} max|d| "
              f"{float((a.float() - b.float()).abs().max()):.3g} ({time.time() - t0:.0f} s)", flush=True)
    except Exception as e:  # noqa: BLE001 -- report the case and go on
        print(f"ROWCHECK {compute} S={S}: could not run: {type(e).__name__}: {str(e)[:240]}", flush=True)
