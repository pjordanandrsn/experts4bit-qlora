# DQ7 — Loggetta's dense executor: proof, anchor and out-of-sample capacity

Work: the dense continuation of [#1083](https://github.com/pjordanandrsn/experts4bit-qlora/issues/1083), implemented in
[Loggetta #20](https://github.com/pjordanandrsn/loggetta/pull/20). This registration licenses no new dense kernel,
quality claim or planner coefficient change. Registration review precedes every billable action. No box has been acquired.

## Question and subjects

Does the actual Loggetta executor run exactly its selected setup, and does its allocator estimate ever fall below
the measured training peak on architectures that did not fit its activation coefficient?

All weights are **synthetic**, deterministically generated bf16 tensors, then loaded and quantized through Loggetta.
This is architecture-only capacity. Loss values are used for finiteness and proof equality; no loss or quality number
is read, reported as a result, or quoted. The checkpoint builder holds one tensor at a time and writes safetensors.
Each tensor seed is the low eight bytes, little-endian, of SHA-256(subject + ":" + tensor name), modulo 2^63−1.
Matrices are bf16 normal draws scaled by config.initializer_range; norm vectors are ones and biases are zeros.
The subject manifest records names, shapes, seeds and every shard's SHA-256.

The following downloaded config.json files are preserved **byte for byte** under configs/, copied unchanged into
the generated checkpoint, and checked against these hashes before either planning or reading:

| subject | public source | SHA-256 |
|---|---|---|
| qwen3_14b | [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json) | e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a |
| llama31_8b | [NousResearch/Meta-Llama-3.1-8B config mirror](https://huggingface.co/NousResearch/Meta-Llama-3.1-8B/blob/main/config.json) | 54acfad3cffe057640904ca8a1e83525e6551c70c7a04c641f5a9eda0bbf64bd |
| qwen3_32b | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json) | 97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb |

Qwen3-14B (40 layers, intermediate 17408) and Llama-3.1-8B (32 layers, hidden 4096) are **out-of-sample**.
Qwen3-32B is an **in-sample anchor**, because its DQ4 reading fitted the coefficient and overestimate brackets.

## Setup, proof and reading

One normal 32 GB RTX 5090, PCIe gen ≥5 / x16. Capacity is the question; link and step time are recorded descriptively.
Software pins are runtime.json: torch 2.8.0+cu128, bitsandbytes 0.50.2, transformers 5.18.0, PEFT 0.21.2, the
grouped-nf4-gemm v0.43.0 commit, and the reviewed Loggetta executor commit. The launch checkout pins e4b itself.
The runner checks pip's direct_url commit IDs and exact package versions and requires the mirrored bnb sources to
match, so the late-bound backward can engage. A registration pin change is reviewed before launch.

The typed opt-in Constraints.allow_development_executor=True is explicit. Every capacity arm plans through Loggetta
and executes that same plan:
NF4 double-quant blocksize64, bf16 compute; PEFT fp32 r16/alpha32/dropout0 on all seven classified projections;
SDPA; non-reentrant checkpointing; micro-batch1, accumulation1; two AdamW steps, lr2e-4 constant, clip1.0;
default allocator (both allocator configuration environment variables unset). Qwen loss_chunk=512. Llama
loss_chunk=0, because e4b does not cover Llama's post-head path; its full logits are explicitly priced.
Every plan field is echoed and checked by the reducer. No setup switch or silent fallback is allowed.

**Proof first.** The tiny Qwen3 subject has 2 layers, hidden1024/intermediate4096, 8 query/4 KV heads of dimension128,
vocab512. Attention's packed codes are below the 1 MiB threshold and stay resident; the three MLP projections per
layer exceed it and stream. Both paths must work in the same subject. This is a CUDA correctness proof, never speed.
Seed731, TF32 off, deterministic algorithms, math SDPA (flash and memory-efficient SDPA disabled), 64 real token ids,
no dropout. Independently prepared resident and streamed models must match **bitwise** in loss, every LoRA gradient
and one AdamW update. Sampled first/last frozen bytes must stay unchanged. Fourteen NF4 PEFT wrappers must engage;
streaming must pin 2 layers and late-bind 6 projections. Export/reload must preserve adapter tensors and logits bitwise.
Failure stops before every capacity arm.

**Reading.** Stock SDPA selection in fresh processes, separate from the proof's deterministic algorithm settings:

| subject | sequences | placements | role |
|---|---|---|---|
| qwen3_14b | 512, 2048, 4096 | resident, streamed | decisive out-of-sample |
| llama31_8b | 512, 2048, 4096 | resident, streamed | decisive out-of-sample |
| qwen3_32b | 2048, 4096 | resident, streamed | DQ4 anchor |

Order: subjects as above, sequences ascending, resident then streamed. Every row contains exactly seq_len **real
token ids**, no padding. Deterministic WordLevel text maps w3..w511 to the same ids; two full rows are validated before
loading. Their SHA-256 must equal the execution data receipt and agree across placement pairs. There is no attention
isolation mask, variable row length, masked-token shrinkage, repeated short dataset or pretrained text evaluation.

Each arm uses execute(), including user-data preparation, the shared measured training loop and adapter export.
Allocator training peak, reserved peak, sampled driver peak and separate load peak stay distinct. Record the estimate's
itemized lines beside each peak, with residual measured−estimated in bytes and GB (1 GB=10^9 B). No observations feed
planning during this reading; the inferred 20% reserve remains unchanged. The allocator comparison excludes reserve
and CUDA context, exactly as the standard ExecutionReceipt does.

The shared loop clips gradients, while DQ4 did not. A transparent wrapper records cumulative allocator peak immediately
before and after each clip and the added cumulative peak; it introduces no tensor allocation. Predict zero incremental
training peak from clip (AdamW's larger workspace should dominate), with [0,32 MiB] as a descriptive expectation.
Any excess remains a separate clip line; it is never folded into the activation coefficient.

## Registered decision

The reducer requires all 16 completed arms, matching software, the exact device/memory range, config hashes,
setup and row fingerprints, and engaged chunked loss where specified. Streamed arms must pin all homes and late-bind
7×layers projections. Any missing, failed, OOM, changed or duplicate arm is **VOID/INCOMPLETE**, with the cause retained.
A failed deterministic proof is **FUNCTION_FAIL** and prevents the reading. Wrong-card refusal is rc19, not machine
evidence. No arm is removed to improve the result.

**Out-of-sample verdict:** **NEVER_UNDER** only if estimated allocator bytes ≥ measured peak in every decisive arm;
one byte below in any arm is **ESTIMATE_UNDER**. Overestimate bounds are predictions, not this pass condition.

**Anchor:** compare with exact DQ4 c_def ladder peaks from receipts/dq4-5090-2:

| placement | L2048 bytes | L4096 bytes | empirical ladder/fresh spread at binding rung |
|---|---:|---:|---:|
| resident | 23577589760 | 26273331200 | 65536 B (L7168) |
| streamed | 8459810816 | 11155552256 | 3711488 B (L14336) |

An absolute anchor residual above its arm's empirical spread is **ANCHOR_MISS**. Out-of-sample verdicts are still
reported, but no pass is quoted until the anchor miss is attributed through itemized bytes, clip census and residuals.
**ATTRIBUTED** means, at every anchor arm, abs(new peak − DQ4 peak − named accounted delta) ≤ that arm's empirical
spread. The only new peak term the current instrument measures is clip_added_cumulative_peak_bytes when that clip's
post-call peak equals the final training peak (maximum across the two steps). A clip transient surpassed by AdamW
accounts for zero final-peak delta. Two consistent before/after clip readings are required. The load peak has an
explicitly zero delta in this comparison because training resets peaks
after loading in both harnesses. Both lines and the unexplained remainder are reported. If clipping does not account
for the miss, it is **UNATTRIBUTED** and no pass is licensed. Itemized estimate lines are retained for diagnosis;
an estimated line or a newly named post-read term cannot absorb the residual. A further attribution instrument needs
its own review/registration before a reread; activation/reserve coefficients stay unchanged.
The empirical spread is deliberately not replaced with the earlier 4%/10% estimate brackets.

**Predictions, graded separately:**

- Out-of-sample allocator estimates are never under, with overestimates ≤4% resident and ≤10% streamed. These brackets
  were tuned on DQ4 and are now predictions to test, not a licensed general bound.
- The 32B anchor reproduces DQ4 within its observed spread; the changed loop is the stated risk.
- Every deterministic proof comparison is bitwise; all frozen samples remain unchanged.
- Default streaming frees substantial allocator memory on each architecture. Step ratios are descriptive, not graded.

No reserve/intercept replacement is licensed by this registration. D7 requires a subsequent registered rule using
dense's own receipts, held-out checks and the unchanged DQ4 bracket floor; a miss is preserved rather than tuned away.

## Budget, launch and prelaunch checks

One verified-secure rented RTX5090 at declared $0.85/h, guard2h, estimate≤$1.70 before recorded storage charges.
Request 200 GB storage, normal ≥98 GB host RAM, gen5/x16; use the existing guarded pod-launch and receipt store.
The provider's actual quoted total must fit the confirmed single-run no-ask ceiling; no bypass of policy or ledger.
The experiment will not start until registration review and prelaunch correctness are complete. Teardown proof,
receipt/ledger commit and immutable evidence publication are required, even on a failed lane.

Prelaunch at the reviewed pins: CPU reducer mutation tests, shell/refusal tests and config checks pass; rehearse the
actual tiny CUDA proof on the available A2000 pool under a bus resource claim. The rented runner must refuse A2000 at
rc19 before install. A2000 contributes correctness only and supplies no speed, capacity prediction or host filter.
The controller delegates nonce/heartbeat/fetch to tc1_drive, excludes synthetic checkpoints under hf-cache, and fetches
proof, subject manifests, per-arm receipts, logs and reducer output. It stages no HF credentials for these subjects.

Exit codes: 9 install/config/instrument/disk error; 11 failed proof, arm or insufficient guard time; 12 reducer error;
14 registered egress refusal; 18 registered 28 GiB host VRAM-floor refusal; 19 wrong card. No partial result becomes a pass.
No redraw or coefficient adjustment is licensed; a failed run returns to review with its receipts and cause.
