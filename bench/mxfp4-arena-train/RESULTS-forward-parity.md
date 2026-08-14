# MXFP4 forward parity — measured on a rented card, 2026-08-14

This is **not** the PREREG's experiment. That one measures whether a 284B model
trains from its own MXFP4 bytes on one 24 GB card (arms STOCK/ARENA, predictions
P1–P4). This is the correctness gate that has to pass *before* that run is worth
buying: does the MXFP4 forward compute the right thing at all?

No checkpoint, no arena bake at scale, ~1 MB of fixtures. The arena is baked from
bytes `tests/test_mxfp4_arena_train.py` writes, so the source of truth is known
exactly.

## What was graded against what

The fused Triton kernel is compared to the **pure-torch oracle** —
`formats.mxfp4.dequantize_mxfp4` then `@` — and never to another accelerated
lane. Comparing two fast paths measures whether they round alike, not whether
either is right. The oracle's weights are decoded from the *source stacks*, so
its provenance is independent of the arena, the tier, and the staged buffers.

Both of the kernel's branches are exercised: `gemm_mxfp4_grouped` dispatches
all-size-1 groups to a GEMV reduction and everything else to the grouped GEMM.
They are separate kernels, and a decode-shaped bug is invisible to a
prefill-shaped test.

## Environment

| | |
|---|---|
| card | NVIDIA L40S (secure on-demand) |
| torch | 2.8.0+cu128 (unchanged by the install — checked, not assumed) |
| triton | 3.4.0 |
| experts4bit-qlora | 0.18.0 (published wheel) + this branch's 4-file overlay |
| grouped-nf4-gemm | 0.11.0 (published wheel, unmodified) |

Overlay sha256s are in `overlay_provenance.json`; they match the bytes the CPU
gate ran, so both legs are about identical code.

## Measured — relative max error

| quantity | tokens | kernel branch | rel max err |
|---|---|---|---|
| gate_up kernel vs oracle | 1 | gemv | **0.000e+00** |
| down kernel vs oracle | 1 | gemv | **0.000e+00** |
| gate_up kernel vs oracle | 6 | grouped gemm | **0.000e+00** |
| down kernel vs oracle | 6 | grouped gemm | **0.000e+00** |
| forward: fused vs reference lane | 1 | gemv | 5.882e-03 |
| forward: fused vs reference lane | 6 | grouped gemm | 4.367e-03 |
| forward: reference lane vs source oracle | 1 / 6 | — | **0.000e+00** |
| CONTROL — fused vs UNCLAMPED oracle | 1 | gemv | 1.000e+00 |
| CONTROL — fused vs UNCLAMPED oracle | 6 | grouped gemm | 9.991e-01 |

**The projections are exact on this fixture.** That is a measurement at these
shapes, not a general guarantee: fp4 code points are dyadic and the fixture's
exponent range is bounded, so every product is representable and the fp32
accumulation has nothing to round. The durable claim is the asserted bound,
`< 2e-2` — the same bar `grouped-nf4-gemm`'s own `test_mxfp4_grouped.py` holds
this kernel to, arrived at independently here before that file was read.

**The whole-forward residual (4–6e-3) is two known ordering differences, both
algebraically identity.** The fused lane applies the router weight *after* the
down projection (the vendored `ExpertsNbit` contract); `_DeepseekV4ForwardMixin`
applies it before, in fp32. And `_epilogue` rounds V4's fp32 gate back to bf16 a
step earlier than V4's own forward does. The down projection is linear and these
modules carry no down bias, so scaling commutes with it exactly; what is left is
bf16 rounding.

**The CONTROL is why the tolerances mean anything.** An oracle with V4's clamps
removed — plain SwiGLU, the failure `lora._epilogue` warns is silent, where the
model trains and the loss falls while it optimises a function the frozen base
does not compute — sits at ~1.0, roughly 200x outside the bound. A fixture whose
values never reached `limit` would have let the parity numbers above pass with
the epilogue wired wrong.

## Positive control on the CPU gate

Before any of this was rented, the CPU gate was checked against a deliberate
mutation: expert `e` reading expert `e+1`'s rows — real bytes, right shapes,
wrong weights. It failed at **rel max err 1.21**, 60x the tolerance, and the
runner reported `GATE FAILED` with a nonzero exit. A green suite that cannot go
red is not a gate.

**A second mutation found a test that was already green and already blind.** The
training-composition tests were added afterwards, passed first try, and were then
mutated by making `_lora` return zeros — the adapter present but never reaching
the MXFP4 path. The gradient test caught it; **the forward-parity test did not.**
`_rel_err` normalises by the max, this fixture's e8m0 exponents span
2**-8..2**7, and the base output's heavy tail set the denominator, so a real
adapter delta vanished inside the 2e-2 bar. Its "the adapter is contributing"
guard was worse: it compared against a base-only oracle written in V4's ordering,
so it was passing on bf16 ordering noise rather than on the adapter.

Rewritten as a scale-free ratio — the module must land far closer to the oracle
that HAS the delta than to an identically-ordered one that does not. Measured
margin on the clean code: `err_without = 6.5e-3` against a `1e-4` bar (65x), with
`err_with` exactly `0.000e+00`. Both tests now reject the mutation.

The lesson generalises past this file: **a max-normalised relative error is the
wrong instrument for a small perturbation on a wide-dynamic-range tensor.** It is
right for the kernel comparison above, where a wrong answer is gross, and wrong
for grading a delta.

## Cost and teardown

Five pods, all secure on-demand, **~$0.11 total** against the $35/job cap.
Every one verified gone (`GET /v1/pods/<id>` → 404) and the account list
confirmed empty at `0 pods`, with spend/hr back to the $0.005 idle-volume floor.

Three attempts did not measure anything, and each was infrastructure or harness,
never the code under test:

1. **3090, ENV_BROKEN** — the payload assumed `python3` owned torch. It does not
   on a RunPod pytorch image, and a non-interactive `ssh host cmd` never sources
   the profile that puts the venv on PATH. Reported "torch missing in the image",
   which was false. Fixed by *probing* for the interpreter and recording what was
   probed.
2. **A5000, WEDGED** — `uptime 0` at 120 s, the known consumer-pool wedge. Caught
   in 2 min by the GraphQL uptime check rather than by waiting out a REST
   timeout, torn down, ladder advanced.
3. **3090, FAILED** — all five GPU tests died in the test helper's
   `mod.to("cuda")`. `build_meta_experts` declares on `meta`; staging replaces
   the four expert tensors and leaves the rest (the NF4 codebook, which this lane
   never reads) meta, and `Module.to` refuses to copy out of a meta tensor. The
   forward was never reached. Fixed by staging onto the device — which is also
   what the production tier does, so the test now runs the same mechanism.

The A40 in run 4 also wedged; the ladder advanced to the L40S, which came up in
30 s and ran the suite green. 4090 is excluded from the ladder entirely: it
reaches `RUNNING` with `publicIp ""` and never gets one, recorded three times.

None of these is a retry of the measurement. The measurement ran once and is
reported as it came out.
