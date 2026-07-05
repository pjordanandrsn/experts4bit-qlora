# Offload memory facts — peak decomposition, code-verified (debt T4)

```
status:   code-read + arithmetic against already-measured peaks. No new GPU runs.
queue:    POST_AUDIT_WORK_QUEUE.md T4 (audit §4 hypothesis check)
sources:  experts4bit_qlora/_vendor/experts.py (storage + dequant paths),
          experts4bit_qlora/offload.py (staging), bundle olmoe-qlora-grid-20260705-1351
          (grid peak-GB cells, single A5000 host)
```

## The mechanism, from source (never from behavior)

`ExpertsNbit._dequantize_expert` (`_vendor/experts.py:428-457`) is the entire
precision-dependent compute path:

- **16-bit passthrough**: `packed[expert_idx].reshape(shape).to(dtype)`. With bf16 storage and
  bf16 compute, `.to(dtype)` is an identity — a **view, zero allocation**. (fp16 storage under
  bf16 compute does cast — one transient [out, in] tensor per projection, ~8–13 MB, invisible
  at 2-decimal GB; consistent with fp16 and bf16 measuring identical peaks.)
- **8-bit / 4-bit**: `F.dequantize_blockwise` / `F.dequantize_4bit` allocate a fresh
  **compute-dtype** [out, in] weight per projection per expert — the dequant workspace. Its
  size is set by `compute_dtype` (bf16), NOT by storage width — the audit's "bf16-sized
  workspace" is the correct shape of the claim.

Storage slabs, from module construction (`experts.py:246-272`): 4-bit = packed uint8 (2
values/byte) + fp32 absmax (1 per 64 block); 8-bit = uint8 (1/byte) + absmax; 16-bit = the
weights themselves, **no absmax**.

## Exact slab arithmetic (OLMoE-1B-7B: 16 layers × 64 experts, H=2048, I=1024)

Per-layer expert params: 64 × (2048·2048 + 2048·1024) = 402,653,184.

| scheme | packed B/layer | absmax B/layer | slab GB/layer | ×16 GB |
|---|---|---|---|---|
| nf4/fp4 | 201.3 MB | 25.2 MB | 0.2265 | 3.624 |
| int8/fp8 | 402.7 MB | 25.2 MB | 0.4278 | 6.845 |
| bf16/fp16 | 805.3 MB | — | 0.8053 | 12.885 |

## Resident peaks: slab-additive to three digits

`fixed := measured resident training peak − total slab`:

| scheme | measured GB | slab GB | fixed GB |
|---|---|---|---|
| 4-bit | 5.28 | 3.624 | **1.656** |
| 8-bit | 8.50 | 6.845 | **1.655** |
| 16-bit | 14.54 | 12.885 | **1.655** |

Three independent widths agree on `fixed = 1.655 ± 0.001 GB` (non-expert weights, LoRA,
optimizer, activations under checkpointing). Resident peak = `fixed + slab(p)`, with no
visible precision-dependent workspace (the per-expert dequant temp is transient ~13 MB,
below resolution).

## Offload peaks: `fixed + slab(p)/16 + W·[p quantized]`

Predicted = `fixed + one layer's slab` (single-slot staging, offload.py):

| scheme | predicted GB | measured GB | residual |
|---|---|---|---|
| nf4 | 1.882 | 2.52 | **+0.64** |
| fp4 | 1.882 | 2.52 | **+0.64** |
| int8 | 2.083 | 2.72 | **+0.64** |
| fp8 | 2.083 | 2.72 | **+0.64** |
| bf16 | 2.460 | 2.41 | **−0.05** |
| fp16 | 2.460 | 2.41 | **−0.05** |

The residual is a **constant ≈0.64 GB for the four quantized schemes and ≈0 for passthrough**.
Precision-independence across 4-bit and 8-bit storage is exactly what a compute-dtype (bf16)
workspace predicts — storage width sets the slab, the workspace does not scale with it.

## Verdict on the audit §4 hypothesis

**Confirmed in form, measured in size, not yet derived in size.**

- Confirmed from code: quantized offload paths materialize bf16 dequant weights; the 16-bit
  path allocates nothing (bf16) — the workspace term exists iff quantized. Slab is the only
  other precision term. The predicted ordering (16-bit offload BELOW 4-bit, despite an 806 MB
  vs 227 MB staged slab) is reproduced by the decomposition.
- Measured: W ≈ 0.64 GB, identical across nf4/fp4/int8/fp8.
- **Open (flagged, not papered over):** a naive one-live-temp reading of the dequant loop
  predicts ~13 MB transient, not 0.64 GB (~79% of one fully-dequantized layer, 0.805 GB). The
  code-read cannot pin why that many compute-dtype bytes coexist at the peak instant under
  offload but not resident — candidate explanations are caching-allocator/stream interactions
  with the per-layer staging pattern (fresh slab allocations each layer rotating block pools)
  rather than temps outliving their matmuls. A one-step `torch.cuda.memory` snapshot timeline
  under offload would pin it; that is a GPU probe, queued behind the certificate work.

## T4b (Addendum 3 §2): shape-derived models vs measured peaks — P-C1 FAILS

Addendum 3 refutes the *persistent*-workspace reading (correct — the dequant is a transient;
this doc never claimed persistence but used the "workspace" name loosely; retired here) and
commits P-C1: a shape-derived model reproduces the six offload train peaks with per-mode
residual < 0.10 GB. Executed:

| model (from tensor shapes only) | nf4/fp4 pred | int8/fp8 pred | bf16/fp16 pred | residuals vs 2.52/2.72/2.41 |
|---|---|---|---|---|
| A/B — fixed 1.655 + staged slab + per-expert transient (0.013) | 1.89 | 2.10 | 2.46 | **+0.63 / +0.62 / −0.05** |
| C — fixed + staged packed+absmax + full-layer bf16 dequant (0.805) | 2.69 | 2.89 | 2.46 | **−0.17 / −0.17 / −0.05** |

(The packed-bytes-adder model and A/B coincide when stated from shapes; Addendum 3's
delta-over-bf16 arithmetic implicitly assumed model C.) **No shape model reaches <0.10 GB
per-mode residual → P-C1 FAILS**; per Addendum 3 §6, no mechanism sentence ships. What the
data does pin: the quantized-over-shape-model excess is constant across 4-bit and 8-bit
(~0.63 under A/B, ~0.17 under C) — a precision-independent term of unresolved identity. The
n=64 eval-job peaks cannot disambiguate (the eval script never reset after load, so
load-phase quantize transients pollute them). Resolver: the one-step
`torch.cuda.memory` timeline probe (gated), or a rev3-style leg-scoped eval measurement.

## Z3 (lanes addendum 1): compute path per ladder mode — SHARED

Source read for `SPECULATIVE_LANES_ADDENDUM_1.md` Z3/Z4. The loader passes
`compute_dtype=dtype` (bf16) to every `ExpertsNbit` regardless of storage scheme
(`loader.py:159`), and every mode's multi-row projection flows through the same call chain —
`ExpertsLoRA._base_project → ExpertsNbit._project → _FrozenLinearRecomputeBackward.apply →
F.linear(x, W_bf16)` — with a bf16 weight of identical shape/stride/dtype. **The matmul path
is one path.** The only per-mode difference is how that bf16 weight is materialized: identity
view (bf16 storage), fp16→bf16 cast kernel (fp16 storage), or bnb dequant kernel (4-/8-bit).
The GEMV branch is decode-only (single-row, `no_grad`) and never runs in the ladder evals.

Consequence for Z4: the fp16 ladder point is NOT an "activation-channel intercept" —
activations are bf16 in all modes by construction. fp16-vs-bf16 weight values differ only
where bf16 magnitudes fall into fp16's subnormal range (|w| ≲ 2⁻¹⁷ loses relative precision;
|w| < 2⁻²⁴ flushes), so fp16 is the **minimal weight-perturbation point on the same curve**
(its exact W_RMS is measured by `scripts/wrms_per_format.py`). Per P-A6, since the paths are
matched by construction, the observed ρ(int8, fp16) cannot be a kernel-path artifact and
stands as example-level smooth-sensitivity, pending the n=1024 replication.

### W_RMS measured (O-4 + Addendum-1 S-A x-axis)

`scripts/wrms_per_format.py` (relative RMS of packed-dequantized experts vs the bf16
reference, per projection, whole model):

| format | W_RMS |
|---|---|
| fp16 | 0.00000 |
| int8 | 0.00970 |
| fp8 | 0.02478 |
| nf4 | 0.09241 |
| fp4 | 0.12307 |

**O-4 resolved clean:** the weight-perturbation ordering is *exactly* the test-pinned
reconstruction chain `fp4 > nf4 > fp8 > int8 > fp16` (worst→best) — no metric conflict, no
finding. **Addendum-1's committed "fp8 lands nearer the 4-bit formats (~2e-2, ≲ nf4)" is
REFUTED:** fp8 (0.0248) is 3.7× closer to int8 (0.0097) than to nf4 (0.0924). This converges
with the n=1024 routing telemetry (fp8 = 6.0 flips vs bf16, with the int8/fp16 family at
2.2–3.8, NOT the 4-bit family at 16.7–20.4). The n=64 "fp8 behaves like 4-bit" covariance
reading was itself the shared-outlier artifact the Z1 Spearman check flagged — fp8 is 8-bit
on weights, on routing, and on the flat-top ladder.

## The floor headline (audit §4), now code-backed

**bf16-offload trains at 2.41 GB — below the 4-bit floor (2.52 GB), with no quantization at
all — vs 14.54 GB resident**, because passthrough offload carries no absmax, no dequant
workspace, and no dequant kernel; its only cost above `fixed` is one layer's staged slab.
This is a memory claim about the measured host and model; per POST_AUDIT_WORK_QUEUE.md §T4 it
is held as the campaign headline pending T1 only insofar as any *eval-quality* rider is
attached — the memory numbers themselves are 3-seed (repeat grid) and single-run (grid)
measurements independent of the training-eval anomaly.

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T22:22:03Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `42a0dfeb0ec349e75c3739b8cf6d0a729816e76b1e03046416896db5e6c8ef35` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- **Prior disclosed pre-footer hashes (chain, newest first):**
  - `2026-07-05T20:03:55Z` `e113368d9c3c991b2efb8ce823fb9316b8b4254ffa7311837d5560af89311bb3`
  - `2026-07-05T18:18:30Z` `0f9e8a7dc51d38082ae0eca173e2b3d4987ccbc4ce7fc5982f66436c79db3672`
  - `2026-07-05T17:13:39Z` `770f2824bdd35f586fc3352e265b69872b910a8ad7076281cd08b56aea49ced6`
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[o+%.!$?@.?&~o#?=]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|   +B+.          |
|  .++o .         |
|  ..  =          |
|   o B   . .     |
|    = B S =      |
|   o * @ o o     |
|    = X E        |
|     * +.O ..    |
|     .+oo +o.    |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info OFFLOAD_MEMORY_FACTS.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify OFFLOAD_MEMORY_FACTS.md.ots OFFLOAD_MEMORY_FACTS.md` succeeds against the on-disk bytes.
- Anchor file: `OFFLOAD_MEMORY_FACTS.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
