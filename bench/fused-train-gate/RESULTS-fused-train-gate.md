# The fused training path at 48 layers: bit-exact, 0.95% eval parity, 0.768x peak VRAM — and one gate that failed on my own fixture
### 2026-07-29/30 · RTX 4090 (25.26 GB visible, sm_89) · torch 2.8.0+cu128 · grouped-nf4-gemm **0.2.4** · e4b working tree that became **0.6.5**

Until this run, `grouped-nf4-gemm`'s expert GEMM was forward-only: anything
needing `dL/dx` through the expert projection fell back to
dequantize-then-matmul, so `enable_fast` was an inference-only flag. This is the
verification of the differentiable path — `nf4_qlora.FusedGroupedNf4` +
`enable_fast_train` — on a real 30B-class MoE at full depth.

**Evidence tier: `measured`, not `confirmed`.** The protocol
([`docs/PREREG-flagship-matrix.md`](../../docs/PREREG-flagship-matrix.md)) was
committed before any data existed — commit `b08747f`, amended pre-data in
`b33c553` — but it is **not** OpenTimestamps-stamped. Pre-data existence rests
on public git history, not a blockchain anchor. That is a weaker guarantee than
this project's `confirmed` tier and is labelled as such. The bands below are the
ones in that file, evaluated as written; none was moved.

## Fixture

Qwen3-30B-A3B (48 layers, 128 experts, top-8), NF4 through the streaming loader,
`offload=True` + gradient checkpointing (`use_reentrant=False`), seq 512, r=8,
α=16, AdamW lr 1e-4, batch 1. Both arms differ only in whether
`enable_fast_train` is on.

`__version__` reads `0.6.4` in the receipts: the code under test is the working
tree that shipped as 0.6.5, and the version bump landed after these runs. The
fused-path source is identical.

## Run B — the registered comparison ([`n11_fast.json`](n11_fast.json))

20 steps on the matrix's own 1,200-example `clinical` set (varied inputs), scored
on the **held-out** split the optimizer never sees. One model load with an
adapter snapshot/restore between arms, so both start from a bit-identical init
and see the same data in the same order.

| | reference | fused | delta |
|---|---|---|---|
| held-out eval, after | **0.337073** | **0.340271** | 0.003198 abs — **0.95 %** |
| held-out eval, before (B=0) | 3.737679 | 3.741370 | 0.003691 abs — **0.099 %** |
| train median abs Δ | — | — | **0.013555** ≤ 0.05 ✓ |
| train final abs Δ | — | — | **0.006472** ≤ 0.05 ✓ |
| peak VRAM | 9.825 GB | **7.544 GB** | **0.768×** |
| s/step | 14.333 | **8.696** | **1.65×** |
| kernel calls | **0** | **6912** | control silent, treatment live |
| trainable tensors | 576 | 576 | — |

**B2 passes on both registered criteria.** The 0.099 % row is the floor: LoRA `B`
is zero-initialised, so before any optimizer step the adapter delta is
identically zero and that difference is *pure kernel and summation order*
(group-sorted vs ascending expert id). The trained-model difference, 0.95 %, is
within an order of magnitude of that floor — the paths are doing the same
arithmetic, not merely similar arithmetic.

The memorisation guard passes: tail loss scale 0.350698, three orders of
magnitude above the 1e-3 floor. That guard exists because of Run A.

## Run A — the 48-layer gate, and why it reads FAIL ([`n11_gate48.json`](n11_gate48.json))

| gate | result | |
|---|---|---|
| G1 — fused trains all 48 layers | `true` | ✓ |
| G2 — frozen experts bit-exact, **both** arms | `true` / `true` | ✓ |
| G2 — byte-flip positive control fires | `true` | ✓ |
| G3 — kernel called (fused) / silent (reference) | 1152 / 0 | ✓ |
| G4 — peak ratio, no OOM, no gradient scaling | 6.87 / 9.09 GB = **0.756×** | ✓ |
| G5 — loss parity, median abs Δ ≤ 0.05 | **0.07124** | ✗ |
| G5 — loss parity, final abs Δ ≤ 0.05 | 0.03768 | ✓ |
| **GATE** | **FAIL** | |

It is published as FAIL because that is what the reducer printed against the
registered band. **The cause was my fixture, not the code.** Run A trained one
repeated sequence: the companion 24-step run
([`n11_converge.json`](n11_converge.json)) shows the reference arm driving that
single sequence to loss **3.6e-05** — memorisation. On a degenerate fixture the
6-step median is dominated by the opening steps, where both arms are far from
converged and a summation-order perturbation is largest in absolute terms; the
*final* delta was already inside the band. Run B is the same comparison on the
data the protocol registered, and it passes.

The convergence run's own verdict string reads `CONVERGING`, and that verdict
should be discounted: with tail loss at 3.6e-05 the absolute deltas collapse
along with the loss, so the trend ratio (0.0003) is measuring the loss scale.
Its *relative* tail delta was 40 %. Both files are published because the
sequence of a wrong fixture, a red gate and a corrected re-run is the actual
record.

## Bit-exactness — the hard gate, and how it was made non-vacuous

192 expert tensors per arm, **16,307,453,952 bytes (16.31 GB) hashed**,
**0 empty tensors skipped**, **0 tensors changed**, in both arms of both runs.
QLoRA trains adapters only; a single changed byte in the frozen 4-bit stack would
void every number here.

The first version of this check was worthless and passed anyway. Under expert
offload `mod.gate_up_proj` is a **0-element placeholder** on every layer, so it
compared `sha256(b"")` against itself and reported "192/192 exact" while hashing
nothing. It now reads `state_dict()` (offload maps each expert to its CPU home),
asserts **bytes hashed > 0** and **empties skipped == 0**, and carries a positive
control that deliberately flips one byte and requires the comparison to *fire*.
`control_detects_flipped_byte: true` is the receipt that the gate can fail.

## Three bugs this campaign found, all of one kind

Each was a case where the instrument could not see the defect, or the fix was
not in the code under test:

1. **`enable_fast` was unreachable in train mode.** `ExpertsLoRA._delegate_to_base()`
   requires `not self.training`; loaders return train mode and `timed_decode`
   never called `.eval()`. Measured 0 kernel calls / 8.3 tok/s against 288 calls
   / 33.6 tok/s — a 2.95× floor lost to a mode check. A patch count is not a call
   count.
2. **Autograd pinned offload-staged experts.** Stashing packed weights on the
   backward context held every staged layer by refcount, defeating the
   single-resident-layer policy; all 48 accumulated and the run OOMed at
   22.41 GB. Fixed by stashing a **closure** (`weights_fn`) so backward re-reads
   whatever is staged now. Two earlier fixes for this were **inert** — one keyed
   off a marker attribute that does not exist, one added a parameter no call site
   passed — and both passed the test suite, because no test enabled offload. The
   bug lived exactly in the untested intersection.
3. **The LoRA `alpha/r` scaling was missing** from the fused delta, making every
   fused update **exactly half** the reference's. Invisible to memory, to
   bit-exactness, to kernel-call counts, and to step-0 forward parity — `B` is
   zero-initialised, so a wrong multiplier on zero is still zero. Only the
   multi-step trajectory exposed it. There is now a regression test asserting
   `scaling * F.linear(F.linear(x, A), B)` across three (r, α) shapes and that
   `scaling=1` and `scaling=2` differ.

## What this does not license

- **One dataset.** B2 is settled on `clinical`. The registered matrix called for
  all five datasets in both arms; see
  [`../flagship-matrix/RESULTS-flagship-matrix.md`](../flagship-matrix/RESULTS-flagship-matrix.md)
  for what actually ran and what is missing.
- **One card, one architecture** (sm_89). Gradient correctness tests pass 5/5 on
  sm_86 and sm_89 with relative error 0.0, but the 48-layer end-to-end
  verification is single-architecture.
- **The speedups are indicative, not barred.** 1.65× (Run B) and 1.46× (Run A,
  9.396 → 6.414 s/step) are single same-process A/B pairs with no repetition, so
  there is no variance estimate and no paired band. Two independent runs agreeing
  in direction and rough magnitude is supporting evidence, not a confirmatory
  result. Treat the memory ratio the same way — though 0.756× and 0.768× from
  separate runs, both showing the fused path using *less*, is a robust direction.
- **Not a convergence claim.** 20 and 6 steps at batch 1 rank arms; they do not
  characterise training quality.
- **Decode is a separate question, still open.** The kernel's 9.7–20.9× over a
  naive per-expert dequant loop does not appear at bs=1 end-to-end `generate()`,
  where an interleaved A/B measured ~1.0× against a 20 % noise floor. That is
  Amdahl outside the expert GEMM, not a refutation of the kernel number, and it
  is unresolved until the expert GEMM is timed directly.

**Teardown:** every rented instance was 404-verified after evidence was pulled;
the model-cache volume was left intact.
