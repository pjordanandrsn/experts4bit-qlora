# DQ11 Amendment 4: bitwise observer proof under a deterministic configuration

Prospective registration for `dq11-5090-4`, under [DQ11-PREREG.md](DQ11-PREREG.md)
and Amendments [1](DQ11-AMENDMENT-1.md), [2](DQ11-AMENDMENT-2.md) and
[3](DQ11-AMENDMENT-3.md). Earlier registrations and failed receipts remain
unchanged. This amendment does **not** authorize a rental. It requires reviewed
code and this registration on main, a fifth fresh zero-rental A2000 rehearsal
that passes the complete amended instrument, its measured time-left record,
and a separate maintainer draw-4 ACK before any quote or rental.

## Evidence and the limited inference

The third paid draw, source `858e7f6956f56d6b489deb93a0526be61cf61c0a`, passed the
full 101-package bootstrap and actual Mistral preparation, then refused in the
L proof while hashing a scalar loss. No binding proof or science reading
completed. Actual cost was $0.183; with the first two draws, aggregate actual
spend is **$0.235**. All three draws remain `VOID`. The immutable private packet is
`adertha://experts4bit-qlora%231083/reviews/dq11-5090-3-refusal-evidence.tar.gz`,
SHA-256 `01ebc64ae03c2598780f44cdc4647203898c621ec2f0391a7b9fa857faa94bcd`.

Four subsequent local rehearsals also refused. The fourth used merge
`57a605c29113d5ffcc3b4ef9303bf19f26260212`, passed the full rehearsal stack,
admission and preparation, and streamed 32 real CUDA NF4 handles totaling
2,359,296 bytes. Its original same-arm observer loss/gradient bitwise check
refused before a proof JSON or any scientific reading. Raw evidence is private:
`adertha://experts4bit-qlora%231083/reviews/dq11-a2000-57a605c2-fourth-refusal.tar.gz`,
SHA-256 `8936ee97f5948ae08b132b7b3a1eb40935a7e0e9bfe7e175a66c171266e4db54`.

A source-unchanged diagnostic on the tiny random Mistral, A2000 and
Python 3.11.13 / Torch 2.11.0+cu128 established variation even without the
observer. A second diagnostic performed one checksum smoke backward and four
passes in each of A without a checksum tap, A with the tap, B with synchronous
staging (`train_prefetch=False`), and C with the three policies below. Every
pass used the same canonical rehearsal initializer, original first 2048-token
training block, frozen NF4 bytes and zero optimizer updates. Before pass 4, all
handles were evicted and the prefetch scheduler's `last` and `phase` reset.

| Diagnostic | P2 vs P3 gradients | P1 vs reset P4 gradients |
| --- | --- | --- |
| A, shipped policies, no checksum tap | Different | Different |
| A, shipped policies, checksum tap | Different | Different |
| B, synchronous staging, checksum tap | Different | Different |
| C, deterministic policies, checksum tap | All 448 hashes identical | All 448 hashes identical |

C's loss and every gradient hash were identical across all four passes. A/B
variation reached maximum absolute gradient difference
0.0004725120961666107 and maximum relative difference
0.00855491585619106, using each tensor's larger maximum absolute value as the
relative denominator. All 224 bound projection checksum pairs agreed across
passes and A/B/C. Each pair is an int64 whole-tensor sum and stride-7 sum of a
storage-sharing int32 packed-weight view, queued on the compute stream into
preallocated outputs and read after the passes. These are **noncryptographic
reductions**: equality does not prove byte identity. Synchronous staging did not
remove the variation; equal routes still produced different gradients; a reset
did not restore pass 1. The controls support removable GPU execution
nondeterminism and do not support the proposed stale-byte or route-rounding
signatures. They do not establish a universal absence of offloader bugs or name
one kernel: C changes three policies together. Independent attribution is
informative and does not gate this draw.

A separate owned, source-unchanged one-factor attribution then isolated Flash
Attention backward on this local stack: math SDPA alone produced identical
hashes in two fresh four-pass processes; cuBLAS workspace alone and deterministic
algorithms with `warn_only=True` alone remained variable. The determinism-only
processes warned that Flash Attention defaults to a nondeterministic algorithm.
The hash-verified attribution core is
`adertha://experts4bit-qlora%231083/reviews/dq11-attribution-57a605c2-core.tar.gz`,
SHA-256 `9438f1bc357d02cc745ff8a21c195a9355d7ab636c90ca286bd76983c26da04f`.
Raw math-only FP32 gradients are
`adertha://experts4bit-qlora%231083/reviews/dq11-attribution-57a605c2-grad-C3-mathsdpa.tar.gz`,
SHA-256 `4aff2418200979d5e4f53b3261d3c8dee7263928e515edf776874a4e1e0702da`.
This attribution strengthens the reason for math SDPA in the proof; the
already tested full C bundle is retained. An empty determinism-warning list is
expected under math SDPA. A nonempty list is a reported finding for maintainer
review before any paid ACK, even if the bitwise check passes. Other warnings
remain in the complete warning record. These are local-stack observations,
not a blanket attribution for other models, devices or framework versions.

The complete diagnostic is measured-private evidence, not a full-model,
cu130/SM120, quality, timing or performance result. Its core packet contains the
source and input seals, worker scripts, all process metadata, layer routes,
checksums, raw logs, launch/exit/idle-resource records and CPU re-derivation:

- `adertha://experts4bit-qlora%231083/reviews/dq11-discriminating-57a605c2-core.tar.gz`, SHA-256 `6826c22decc3d6218c57393e4fac0031d7b39fc4645aee66484bcec97a1f9ebc`.
- Smoke FP32 gradients: `adertha://experts4bit-qlora%231083/reviews/dq11-discriminating-57a605c2-tap-smoke.tar.gz`, SHA-256 `21c17fcde36afee36570719af25d40682c32b399c2897501f057cda8bdfd2476`.
- A without tap: `adertha://experts4bit-qlora%231083/reviews/dq11-discriminating-57a605c2-A-no-tap.tar.gz`, SHA-256 `f61dd3057d9121a58269de0c06cb2abd78f952bfff16e68390691bf171bf08df`.
- A with tap: `adertha://experts4bit-qlora%231083/reviews/dq11-discriminating-57a605c2-A-tap.tar.gz`, SHA-256 `d1289d5229d6a1387062ad6a5d41a58e3d50dcc4e9642b332c374ec2619b34cc`.
- B synchronous: `adertha://experts4bit-qlora%231083/reviews/dq11-discriminating-57a605c2-B-sync-tap.tar.gz`, SHA-256 `ea5753c713ba131417c06b514534e85b7f4c1961608a6759029ddcdfec10c975`.
- C deterministic: `adertha://experts4bit-qlora%231083/reviews/dq11-discriminating-57a605c2-C-deterministic-tap.tar.gz`, SHA-256 `45b067ef27107df691405a94d9e11affbe960d446ff5520c7ba3ec041f68e91f`.

## The observer comparison remains bitwise

Each fresh proof process sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing
Torch/Unsloth or initializing CUDA. Inherited workspace policy and late CUDA
initialization refuse. Only the observed/clean same-arm pair enables
`torch.use_deterministic_algorithms(True, warn_only=True)` and the math-only
SDPA context. The pair retains the original tokens, canonical FP32 adapters,
frozen base, non-reentrant checkpointing, no updates and exact loss/all-gradient
hash comparison. Any bitwise difference or nonfinite value still refuses.
No numerical tolerance is introduced.

The scoped deterministic/SDPA flags are restored on success or failure. Initial
quality retains its existing scorer outside the scoped observer pair; its proof
process necessarily has the cuBLAS workspace setting established at process
startup. All six science read processes retain shipped execution policies,
including the default workspace policy. There is no attempt to undo cached
cuBLAS handles by unsetting a variable in an already initialized process.

`warn_only=True` is retained explicitly. Every warning inside the pair is saved,
with operator names where the warning identifies them and the complete message
otherwise. A nonempty warned-op list is reported even if the bitwise comparison
passes. The durable `observer-policy-<arm>.json` sidecar survives a comparison
refusal so warnings remain available when no proof JSON is written. The reducer
requires the actual C settings, warning record and policy restoration; warnings
alone do not authorize a tolerance or silently switch to strict mode.

## Report the shipped-setting spread without gating its magnitude

Before the three observer proofs, three additional fresh untimed processes run
L, U and U0 under their shipped settings. Each prepares the same model, installs the
same initializer, runs the existing initial scorer, and performs four
same-token forward/backward passes on training block 0 with gradients cleared
between passes and no optimizer updates. No observer or checksum tap is added.
Only CPU gradient copies are retained between passes. The four raw FP32 gradient
files per arm are preserved with filename, byte count and SHA-256 in each pass
record, allowing independent re-derivation. Their tensor payload is about 2 GB
in total on the full registered model, within the unchanged 320 GB storage and
100 GB transport reservations; these are additional untimed receipt writes,
not scientific timing or a measured duration.

Each `spread-<arm>.json` records four loss and all 448 gradient hashes, all six
pairwise comparisons, per-tensor maximum absolute difference and relative
difference using the larger maximum absolute gradient, execution flags, source,
input/runtime identity, unchanged initializer, frozen census and bindings. Zero
numerical difference with distinct raw hashes (including signed zero) stays
visible. Reported values must be finite and receipts complete and consistent;
**no spread magnitude is a pass/fail gate**. Proof receipts bind these sidecars,
and the reducer includes the spread and observer warning records in its output.
Read execution policies must match the shipped spread process for that arm.
None of this work is inserted into a timed read or changes its 40-update loop.

## Require live streaming during each scientific L read

This is a separately registered operational guard. Shipped backend policies
remain unchanged. Immediately before the original shared L training loop and
immediately after its 40 successful updates, before final evaluation, read the
real CUDA offloader counters. Require all 32 handles on one live train-prefetch
schedule, positive unchanged streamed bytes, exactly 40 updates and strictly
positive deltas in `uses`, `fwd_prefetch_issued` and `bwd_prefetch_issued`.
Historical positive counts without in-loop increases do not satisfy the guard.

The shared sealed `dq11_stream_witness.py` validates both modes. Scientific
receipts use `dq11-train-prefetch/1` with `science_eligible=true`; the existing
rehearsal schema remains explicitly ineligible for science. Every L reading
must bind its witness to the same source and nonce; the final reducer requires
it for both repetitions. A durable sidecar records the snapshots/deltas on pass
or refusal, including loss of the live schedule after training. No copy, kernel,
optimizer, threshold, training-loop arithmetic or scorer is changed. Science
never uses the rehearsal's tiny-model builder or `min_bytes=0` override. The
launch controller refuses inherited rehearsal flags or a proof workspace
variable before the URL gate, quotes or rental.

## Time-left verification and unchanged ceiling

The three added spread processes each have a 900-second requested cap. As for
all original phases, the effective alarm is the lesser of that cap and remaining
time until the unchanged two-hour guard minus the unchanged 300-second
receipt/teardown reserve. Exhausted reserve refuses before the phase; incomplete
work remains `VOID`. CPU fixtures must include these phases when checking
ordering, caps and near-deadline refusal.

The fifth rehearsal's actual phase durations, deadline and remaining reserve
are **pending measurement**, not estimated here. Its durable phase-time record
and a sibling measured supplement must be reviewed before the paid draw-4 ACK.
A tiny A2000 completion does not certify full Mistral/cu130/SM120 completion in
two hours. There is no prior complete full-model DQ11 timing to substitute.

The original $2.80 ceiling minus $0.235 actual spend leaves **$2.565 maximum**:
at most **$1.465 runtime/storage**, or **$0.7325/hour for two hours**, plus the
unchanged 100 GB transport reservation at most **$1.10**. Normal 32 GB RTX 5090,
320 GB ordered storage, secure-provider policy, quote driver at least 580,
exclusion of machine 45511, role/global admission and actual quote gates remain
required. The merged rental-driver ceiling repair is required. The actual
merged-source all-101-URL `--gate-only` check must pass before read-only quotes,
and the launch runs it again against its final bound source/manifest.

Model revision, data, canonical science initializer, full scientific package
versions/URLs/hashes, arms, optimizer, update count, six fresh reads and reversed
order, K8 quality budget, headroom cutoff, frozen/binding checks and the
controlled-comparison scope remain unchanged. The rehearsal-specific tiny model,
cu128 lock, threshold override and receipt schema never enter a scientific draw.
All changed payloads and the new proof-policy helper/this sibling registration
are resealed in the science closure. Previously reviewed payload repairs remain
explicit: rank-0 hashing retains exact bits, the defining-module/tiny-model
adapters stay rehearsal-only, and original ordered token IDs/full vocabulary
remain available for the local instrument. Their changed source payloads do
not license a changed scientific model, initializer, tokens or thresholds. No hotpatch, fallback, widened budget,
retuned gate or automatic replacement run is authorized.
