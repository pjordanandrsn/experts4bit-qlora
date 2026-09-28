# B511 — does a bucketed decode-graph replay decode exactly as the padded eager step, on an RTX 5090? (registered 2026-09-28, before any run)

Issue: experts4bit-qlora#511. Code under test: PR #757 (`PagedModelRunner.enable_decode_graphs`,
`Fp8PagedKV(..., scratch_slots=)`). Rule under which the rental runs: the owner's standing no-ask tier for a
single run under $15 (2026-09-26), with the unchanged mechanics: this pre-registration merged before the launch,
receipts plus a ledger row, and a proven teardown. The authorization permalink the launcher cites is posted on #511
before the launch. The rental is not part of this change.

## Question

#757 captures one CUDA graph per batch bucket and replays it with the active set padded to the bucket by scratch KV
slots. Its end-to-end test (`tests/test_decode_graph_buckets.py::test_a_bucket_replay_decodes_exactly_as_the_padded_eager_step`)
cannot run on the home RTX A2000: the fp8 paged KV needs native e4m3, which Triton compiles only on sm_89+. So the
central claim, that a replay equals the same padded step run eagerly bit for bit, has never run on hardware. B511 runs
it on an RTX 5090 (sm_120), with one mutation that must break it.

The same box also runs, at sm_120, the GPU tests of the other changes merged 2026-09-28. So far each has run only on
the sm_86 A2000: #750 (`test_hybrid_cold_dest.py`), #753 (`test_capture_quantized_moe.py`, `test_capture.py`,
`test_fast_lora.py`), #754 (`test_int4_attn_pack.py`) and #748 (`test_router_epilogue.py`).

## Instrument

`bench/b511/b511_run.sh` on the box, staged by `b511_drive.sh` and pinned by `staged.sha256` (CI:
`tests/test_b511_staged_pin.py`):

- grouped-nf4-gemm installed at **`fb15cf5f5b9f2fb107fa1a39218fd9d910b34d94`** (the v0.33.5 release, no deps).
- experts4bit-qlora cloned at the launcher-proven `heads.e4b` (**#757's head at launch**, rebased on the `main` that
  carries this pre-registration) and installed editable with `[test]`, under a constraints file that pins the image's
  own torch and triton. Tripwires refuse if the import does not resolve to the clone, or if torch changed during install.
- **Arm A.** The seven files above in one pytest run (`armA.xml`, `armA.txt`).
- **Arm M.** `mut_b511.py` removes the bound-bucket branch of `Fp8PagedKV.kernel_args`, so attention falls back to the
  per-tuple selector cache, which a captured graph bakes at capture time (on the scratch slots). Then the replay test
  alone runs (`armM.xml`). The clone is restored afterwards.

The runner exits 0 when both arms ran; it does not decide the verdict. The verdict below is read from the two junit
files.

## Predictions (written before the data)

- **P1.** In arm A, `test_a_bucket_replay_decodes_exactly_as_the_padded_eager_step` PASSES: all three buckets
  capture (`graph_status` = graph for 1, 2 and 4), the replayed token streams equal the padded-eager streams exactly,
  every bucket replays, and bucket 4 carries padding.
- **P2.** In arm M, the same test FAILS.
- **P3.** Every other arm-A test passes or skips for a reason that is not the card. A skip for "needs CUDA", "needs
  sm_89+" or "needs CUDA + gnf4_native" is a failure of the lane, since this box has all three.

## Decision rule

- **P1 and P2 hold:** #757 is verified. Its body and CHANGELOG entry cite the receipt, and it is merged under the usual
  self-review rule.
- **P1 fails:** #757 does not merge. The failing output is the finding; fix, then re-run B511 as `b511-5090-<n+1>`
  under this same pre-registration.
- **P1 holds and P2 fails (the mutation is not caught):** the test is inert for the bug it names. #757 does not merge
  until the test is strengthened and B511 re-run.
- **P3 fails:** file an issue for each failing test and decide it on its own; this does not block #757.

The unpadded eager decode is printed by the test, not asserted. A bf16 GEMM can round differently at a different row
count, so a difference there is not a finding about the graphs.

## Box and cost

One rental, **`b511-5090-<n>`**: one RTX 5090, Vast verified/secure, image
`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` (it has gcc, which the native CPU kernels and Triton need). No model
download. **Guard 1 h at ≤ $0.75/h (≤ $0.75).** The guard does not exceed one hour, so no proving rental precedes it.
The whole lane is expected to take about 20 minutes on the box (install, arm A, arm M). **Lane ceiling $1.50** (two
attempts); hard stop $2.

## Rehearsal

Before the rental, the runner is exercised on the home NAS's RTX A2000 (sm_86) with the card-class refusal lifted in a
local copy, to prove the install, the tripwires, both arms and the junit output end to end. That is not a reading: on
sm_86 the replay test skips by name.

## What this lane cannot say

Nothing about speed. It measures correctness on one host class, on a tiny random Qwen3 dense model and two MoE
fixtures, not on a served checkpoint. The #511 experiment (a forced 16 → 1 active-set trace on NF4 Qwen3) is a separate
registration.

## Receipts

`receipts/experts4bit-qlora/<date>/b511-5090-<n>/` in the adertha receipt store: the launcher's `receipt.json` and
`teardown-proof.json`, plus the fetched `b511/` directory (`forensics.txt`, `versions.txt`, `constraints.txt`,
`armA.{xml,txt}`, `armM.{xml,txt}`, `summary.txt`, `logs/`).
