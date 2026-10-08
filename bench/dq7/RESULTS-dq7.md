# DQ7 — results of the actual dense executor

The [unchanged registration](DQ7-PREREG.md) at `4ca5a349` grades `dq7-5090-1` (2026-10-08) **VOID**. Fourteen of sixteen arms completed. The Qwen3-32 B resident 4096 anchor was refused before weight loading: the plan's device estimate **30.69 GiB + policy headroom 1.57 GiB > 31.34 GiB free**. This is a registration defect: that selected rung does not fit the planner's admission policy on the registered card. The runner stopped, so the Qwen3-32 B streamed 4096 arm was never drawn. No arm is discarded or threshold changed.

The completed out-of-sample Llama-3.1-8 B arms also falsify the never-under prediction: the allocator estimate is below the measured peak at 2048 and 4096 tokens in both placements. The overall verdict remains VOID under the complete-arm rule; it is neither NEVER_UNDER nor a successful calibration. The [D7 report](receipts/dq7-5090-1/d7-unlicensed.json) is **DQ7_UNLICENSED**, with no fit and no reserve import. DQ8 was not drawn. The development opt-in and inferred 20% reserve remain.

All checkpoints contain deterministic synthetic bf16 weights, generated tensor at a time from the preserved public configs. These are architecture capacity readings, with no model quality or pretrained-weight conclusion. Setup: NF4/doublequant64, bf16 compute, all-projection fp32 LoRA r16/alpha32, SDPA, non-reentrant checkpointing, micro-batch 1/accumulation 1, two AdamW steps, constant lr 2e-4, clip 1.0, default allocator. Qwen uses loss_chunk512; Llama uses full logits (loss_chunk0).

## Proof and completed readings

The [tiny CUDA proof](receipts/dq7-5090-1/proof.json) passed bitwise resident/streamed loss, all 28 LoRA gradient tensors, optimizer updates and adapter reload logits, with sampled frozen bytes unchanged. Both sub-threshold resident and six late-bound streamed projections engaged; all homes were pinned. This is correctness only.

One RTX 5090, 32607 MiB, driver 595.91.07, PCIe gen 5/x16; software pins and commits are in [runtime.json](receipts/dq7-5090-1/runtime.json) and [run provenance](receipts/dq7-5090-1/run-provenance.json). Each reading used a fresh CUDA process with two full real-token rows and matching placement-pair SHA-256.

The comparison excludes allocator reserve and CUDA context, exactly as the registered ExecutionReceipt comparison does. Bytes are authoritative; GB below is decimal (10^9 bytes), rounded to six places. R = resident; S = streamed.

| subject | placement | tokens | allocator estimate GB | measured allocated GB | measured − estimate GB | reserved GB |
|---|---|---:|---:|---:|---:|---:|
| llama31_8b | R | 2048 | 9.749680 | 9.947172 | +0.197492 | 11.463033 |
| llama31_8b | R | 4096 | 13.008536 | 13.670707 | +0.662171 | 16.307454 |
| llama31_8b | R | 512 | 7.305539 | 7.154522 | -0.151017 | 7.910457 |
| llama31_8b | S | 2048 | 6.478123 | 6.675091 | +0.196968 | 9.160360 |
| llama31_8b | S | 4096 | 9.736978 | 10.398625 | +0.661647 | 13.885243 |
| llama31_8b | S | 512 | 4.033981 | 3.882440 | -0.151541 | 5.056233 |
| qwen3_14b | R | 2048 | 12.920708 | 12.660288 | -0.260420 | 13.530825 |
| qwen3_14b | R | 4096 | 14.749291 | 14.535191 | -0.214100 | 15.883829 |
| qwen3_14b | R | 512 | 12.164152 | 11.939569 | -0.224583 | 12.599689 |
| qwen3_14b | S | 2048 | 6.644980 | 6.334917 | -0.310063 | 8.516534 |
| qwen3_14b | S | 4096 | 8.473564 | 8.213359 | -0.260205 | 10.991174 |
| qwen3_14b | S | 512 | 5.888425 | 5.615508 | -0.272916 | 7.432307 |
| qwen3_32b | R | 2048 | 24.317259 | 23.583964 | -0.733295 | 25.222447 |
| qwen3_32b | S | 2048 | 9.202036 | 8.439692 | -0.762344 | 11.815354 |

Complete bytes, every itemized line and named anchor-accounting terms are in the [unchanged reducer output](receipts/dq7-5090-1/dq7-rederived-4ca5a349.json).

## Full device plan against measured driver use

The full plan includes its inferred 20% allocator reserve and 536870912 B CUDA context. It still underestimates **every completed streamed arm**, by as much as **2395904403 B (2.395904403 GB)**, and the resident Llama 4096 arm by 820943251 bytes. Neither keeping the current reserve nor a successful allocator-only comparison establishes a safe device-total estimate. Loggetta must retain its execution opt-in, warn on dense plans and require at least 2.4 GB streamed admission headroom until a separately registered repair is validated. The existing policy headroom is separate from the full device estimate in this table.

| subject | placement | tokens | full plan bytes | measured driver bytes | driver − plan bytes |
|---|---|---:|---:|---:|---:|
| llama31_8b | R | 2048 | 12236487069 | 12123635712 | -112851357 |
| llama31_8b | R | 4096 | 16147113581 | 16968056832 | +820943251 |
| llama31_8b | R | 512 | 9303517184 | 8571060224 | -732456960 |
| llama31_8b | S | 2048 | 8310618525 | 9892265984 | +1581647459 |
| llama31_8b | S | 4096 | 12221245037 | 14617149440 | +2395904403 |
| llama31_8b | S | 512 | 5377648640 | 5788139520 | +410490880 |
| qwen3_14b | R | 2048 | 16041720324 | 14191427584 | -1850292740 |
| qwen3_14b | R | 4096 | 18236020295 | 16544432128 | -1691588167 |
| qwen3_14b | R | 512 | 15133853313 | 13260292096 | -1873561217 |
| qwen3_14b | S | 2048 | 8510847492 | 9248440320 | +737592828 |
| qwen3_14b | S | 4096 | 10705147463 | 11723079680 | +1017932217 |
| qwen3_14b | S | 512 | 7602980481 | 8164212736 | +561232255 |
| qwen3_32b | R | 2048 | 29717581660 | 25883049984 | -3834531676 |
| qwen3_32b | S | 2048 | 11579314012 | 12547260416 | +967946404 |

## Llama itemized allocator estimates beside measured peaks

These tables retain the lines that miss; the reserve is not folded into their comparison. The R/S underestimate differs by exactly 524288 B at both failing sequences, pointing toward their shared full-logit/activation path rather than streamed weights. This is a diagnostic lead, not measured attribution. The current logits/loss and activation terms must be inspected before any measurement amendment; coefficients remain unchanged.

| itemized device bytes | R2048 | S2048 | R4096 | S4096 |
|---|---:|---:|---:|---:|
| frozen decoder linears | 3600416768 | 0 | 3600416768 | 0 |
| embeddings + lm head (bf16) | 2101346304 | 2101346304 | 2101346304 | 2101346304 |
| norms, biases and other parameters (bf16) | 532480 | 532480 | 532480 | 532480 |
| LoRA adapters | 167772160 | 167772160 | 167772160 | 167772160 |
| adapter gradients | 167772160 | 167772160 | 167772160 | 167772160 |
| optimizer state (adamw) | 335544320 | 335544320 | 335544320 | 335544320 |
| activations | 3258855427 | 3258855427 | 6517710854 | 6517710854 |
| dequantization transient | 117440512 | 117440512 | 117440512 | 117440512 |
| allocator reserve (cached, unallocated blocks) | 1949936026 | 1295624602 | 2601707111 | 1947395687 |
| CUDA context + library workspaces | 536870912 | 536870912 | 536870912 | 536870912 |
| frozen decoder linears, two layers staged | 0 | 218103808 | 0 | 218103808 |
| frozen decoder linears, quantization statistics | 0 | 110755840 | 0 | 110755840 |
| **allocator subtotal (graded)** | 9749680131 | 6478123011 | 13008535558 | 9736978438 |
| **measured allocator peak** | 9947172352 | 6675090944 | 13670706688 | 10398625280 |
| **measured − subtotal** | 197492221 | 196967933 | 662171130 | 661646842 |

The full-logit loss term is `T × 128256 × 10` bytes; checkpointed inputs are `T × (32 × 4096 × 2) × 1.178`. The estimate chooses the larger of one-layer recomputation and loss workspace and retains the DQ4 coefficient fitted on Qwen3-32 B. Llama has an untied head and intermediate width 14336 (ratio 3.5, against Qwen3-32 B's5). The observed error is not used to refit either term. A source and tensor-lifetime diagnosis follows separately.

## Anchor and operational failure

At 2048 tokens, new minus DQ4 allocator peak is **+6373888 B R** and **−20119040 B S**, outside the registered empirical spreads (65536 B R / 3711488 B S). Clip adds zero cumulative peak at both calls; the registered load-scope delta is zero. The remainder is **unattributed**. No pass may be quoted from these anchor arms, and neither new labels nor estimate lines absorb their residual. The 4096 anchors are missing.

The runner ended rc11/HARNESS_ERROR. The [refusal log](receipts/dq7-5090-1/read-qwen3_32b-device-4096.log) and [summary](receipts/dq7-5090-1/summary.txt) preserve its cause. The partial fetch also returned rc24 while source files changed; the final fetch completed at 19:36:58Z. Synthetic adapter weights escaped TC1's plural `adapters` exclusion because the executor's default export directory is singular `adapter`; their original trees and SHA-256 manifest were preserved recoverably outside the receipt Git repository. That transport error changes neither training nor the scientific verdict.

The guard was armed before acquisition. Teardown at 19:36:59Z received provider HTTP 200 and independently verified instance 54895866 absent; [proof](receipts/dq7-5090-1/teardown-proof.json). The invoice total was **$0.513** ($0.392 GPU + $0.075 storage + $0.007 download + $0.039 upload). The launched reservation was $2.80, comprising the registered runtime/storage ceiling $1.70 and the launcher's default 100 GB × $0.011/GB transport allowance $1.10; that allowance was omitted from the initial registration's budget prose and is disclosed here. No billable follow-up is authorized by this result.

## Reproduce the reading

Verify `SHA256SUMS` inside the public receipt directory, then run the reducer from the clean registration commit 4ca5a349:

```bash
python bench/dq7/dq7_reduce.py /absolute/path/to/receipts/dq7-5090-1
```

It must return VOID with fourteen rows. D7's report-only reader from Loggetta's merged reader registration returns DQ7_UNLICENSED. The raw scientific JSON files are unchanged copies of the final fetch; their hashes accompany them. Original protocol, reducer, thresholds and planner coefficients are unchanged.
