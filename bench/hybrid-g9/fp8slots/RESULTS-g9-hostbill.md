# RESULTS — G9 host-bill: NO-GO for the python fix; the bill is the KV append storm

Scored under [PREREG-g9-hostbill.md](PREREG-g9-hostbill.md); receipts
in [receipts-g8g9/](receipts-g8g9/) (`g9_buckets.json`,
`g9_cprofile.txt`). Same box/session as the G8 arms.

## The buckets (B=16, gen 128, the serving operating point)

step 207.5 ms = attention-host **114.5** + other-submission **77.7** +
dram-experts 14.6 + scheduler 0.3 + drain 0.4; device-side: attention
kernels **103.2**, GPU expert kernels 29.9.

## Go/no-go: NO-GO, as registered

Host-side python outside CUDA kernels ≈ 207 − 133 ≈ **36%** of the
step — below the 50% bar. The registered phase-2 python fix is NOT
opened. The hypothesis "python is dead weight" is REFUTED as stated:
the attention bucket is ~90% real device kernel time.

## But the attribution names the actual disease

The cProfile pass is unambiguous: `Fp8PagedKV.append` — called
**per-sequence, per-layer, per-side** — is 15.85 s of the 31.9 s decode
window (~112 ms/step ≈ the entire attention bucket). 196,608
single-token `quantize_kv_fp8` calls (one per seq × layer × side),
372k `.to`, 490k `copy_`. The "103 ms of device attention kernels" is
mostly this storm of micro-kernels; the SDPA itself is a rounding
error. The per-call docstrings in the append path show the authors
already fought this battle once (host-mirror rows to avoid stream
syncs); the batch dimension is the one axis nobody batched.

## Disposition

Phase 2 as registered: CLOSED (no-go). Successor line, registered
next per the standing approval: **batched KV append**
(`PREREG-g9-kvappend.md`) — one quantize per layer per side over the
whole batch instead of B, bit-identical by construction (FP8 group
scales are per-token), with the bucket table above as its baseline.
