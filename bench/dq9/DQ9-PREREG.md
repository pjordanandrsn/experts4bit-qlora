# DQ9: paired load-cache and training-fragmentation diagnosis

This is a prospective diagnostic on architectures already observed in DQ7,
under issue #1382. DQ7 remains **VOID**, its underestimates remain underestimates,
and D7 reserve import and DQ8's original launch gate remain closed. DQ9 cannot
license capacity generalisation, a reserve fraction, or a release. A new
out-of-sample subject needs its own prospective registration before any reading.

## Fixed execution and source contract

Use the merged Loggetta full-logit correction at
`efe1c93d55a09a66d760f9f5f7a0232e15e7e5a0`, the merged e4b commit containing
this registration, and GNF4 `6ee2e10408161a9d3c874975c9191a7f2957e6f4`.
`runtime.json` pins Torch 2.8.0+cu128, bitsandbytes 0.50.2, Transformers
5.18.0 and PEFT 0.21.2. The runner verifies installed VCS objects, exact
versions, the bnb mirror contract and `instrument.sha256` before reading.
There is no new kernel, dependency floor, coefficient, tolerance or threshold.

DQ7's public configuration bytes, seed-731 synthetic checkpoint builder and
actual planner setup are reused unchanged. Inputs are architecture-shaped
synthetic weights, never pretrained weights; no quality or timing claim is
made. Record checkpoint manifest and real-token row hashes in every arm.
The two training steps use NF4, rank 16, alpha 32, fp32 LoRA, all four linear
roles, SDPA, AdamW at 2e-4 and the constant schedule. Llama uses full logits;
Qwen uses the existing chunk-512 loss. Actual admission, including the merged
mandatory streamed headroom, must succeed. A refusal stops the lane; no forced
plan or smaller margin is permitted. Qwen-32B is registered at 2048 only.

## Ordered fresh processes

For each row below, run baseline first, then empty. This gives sixteen fresh
reading processes; `dq9_reduce.ARMS` is the machine-readable order.

| Subject | Placement | Sequence |
|---|---|---:|
| Llama-3.1-8B | resident | 2048 |
| Llama-3.1-8B | streamed | 2048 |
| Llama-3.1-8B | resident | 4096 |
| Llama-3.1-8B | streamed | 4096 |
| Qwen3-32B | resident | 2048 |
| Qwen3-32B | streamed | 2048 |
| Qwen3-14B | streamed | 512 |
| Qwen3-14B | streamed | 4096 |

Baseline observes the actual executor without changing cache policy. Empty
adds exactly one `torch.cuda.empty_cache()` after dense setup and before
optimizer creation. There is no cache clear inside training. Neither arm adds
CUDA synchronization, peak resets, tensor copies, or tensor reads. This
differs from DQ7's clip observer, which synchronized after clipping, and DQ4,
which cleared cache after setup and before each rung. The executor's existing
synchronization and peak resets remain in place.

At loader start/completion, setup completion, cache boundary, optimizer
creation, each root forward/loss, backward, clip and optimizer update, record
host allocator A, R, R-A and cumulative allocator peaks. Keep load and
training intervals distinct: the executor resets peaks before training.
Record driver process peak separately through its existing sampler. Counter
snapshots do not imply GPU completion or exact transient attribution.

Setup census contains unique storage metadata for parameters, buffers, bnb
quantization state and offloaded homes; optimizer census contains state
storage metadata after both updates. They retain no tensors or storages.
They are not an inventory of every temporary, CUDA context or allocator block.

## Correctness before readings

Run four fresh tiny proof processes: resident/streamed crossed with
baseline/empty. Use deterministic algorithms, math SDPA, TF32 off, seed 731
and CUBLAS_WORKSPACE_CONFIG=:4096:8. The fixture exercises small resident
projections and streamed MLP projections above the 1 MiB threshold, with
non-reentrant checkpointing, chunked loss, pinned homes and late-bound backward.
Both steps must have identical loss bytes, all 28 gradient tensors before
clipping, and every updated adapter tensor across all four processes.
Sampled frozen integrity must hold; every arm exports and reloads its adapter
with exact tensor and logits equality. Correctness copies are permitted only
in these proof processes. This is a bounded structural proof, not bitwise
validation of the full-size diagnostic subjects or model quality.

A failed proof, mismatched engagement, missing/duplicate arm, changed source,
runtime, config, checkpoint, real-token rows, phase order or cache-call count
produces **VOID** and prevents subsequent readings where detectable. Partial
receipts and refusal/error logs are preserved. No replacement draw follows a
failure without a reviewed amendment.

## Decision and interpretation

Complete valid data produce **COMPLETE_DIAGNOSTIC**, never a capacity pass.
Report all allocator and full-device residuals, including a one-byte miss,
paired allocated/reserved/driver peak differences, load peaks, R-A after
setup, and the allocated/reserved changes at the intervention boundary.
Name a cache release only when the empty arm records a positive decrease in
R with A unchanged at that boundary. If A changes, retain the change and do
not classify it as cache-only release. Zero release is a valid diagnostic.

Cross-process peak differences remain intervention observations; allocator
address layout, pending work and tensor liveness can also differ. Do not
attribute an old anchor residual to cache from these observations, infer a
fragmentation fraction, choose a reserve coefficient, or tune away the
roughly 0.39 GB surplus of the structural full-logit correction. The new
12-byte-per-token-vocabulary term stays structural; DQ7 stays failed.

If cache hygiene improves capacity, shipping it requires a separate executor
PR, prospective verification of that changed mechanism and independent
maintainer review. The mandatory streamed safety margin stays in place.

## Lifecycle and budget

One normal 32 GB RTX 5090, the existing secure TC1 guard and two-hour deadline,
320 GB ordered storage, no HF credential, and the same VRAM/egress refusal
probes. Runtime/storage reservation is at most $1.70; the default 100 GB
transport allowance adds $1.10, for a $2.80 reservation. The controller's
quote and standing compute policy must admit the actual reservation.
No draw before the complete registration is independently reviewed, CI is
green, and it is on main. Record nonce, manifest, exact source objects,
actual invoice and verified teardown/instance absence.

Synthetic checkpoints stay under excluded `hf-cache/`; every adapter export
is explicitly under excluded plural `adapters/`. Keep logs, source/config,
shard/row hashes and raw scientific receipts. Receipt publication uses a
separate PR after the draw; no credential, model weights or private bus URI
belongs in the public tracked files.
