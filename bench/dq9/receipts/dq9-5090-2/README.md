# DQ9 replacement: complete diagnostic, no training-peak cache benefit

`dq9-5090-2` on 2026-10-08 returns **COMPLETE_DIAGNOSTIC**, sixteen readings and eight pairs, under amendment 1 at `a04b7dc4e73bc780f0fae2e0a1bb114cc4578ec2`. The [standalone reducer output](dq9-read.json) re-derived byte-identically with the pinned reducer. The four fresh tiny resident/streamed × baseline/one-clear proofs passed identical losses, all 28 LoRA gradient tensors, updates and reload logits, with sampled frozen bytes unchanged. This is known-subject diagnosis, not a capacity or calibration pass. The first failed draw remains preserved separately.

One setup clear released **33,554,432 B (32 MiB)** on Qwen3-32B resident 2048, with allocated bytes unchanged. The other seven pairs released zero. Every pair had **zero allocated, reserved and sampled-driver training-peak deltas**. No cache-clear executor change follows. The corrected 12-byte full-logit workspace leaves every known-subject allocator estimate above its measured peak; all **ten streamed full-device plans remain below sampled driver peaks**, by at most **1,135,096,620 B**. DQ7 stays VOID, its D7 reserve import stays gated, and DQ8 has not run. The development opt-in and mandatory streamed margin remain.

R = resident; S = streamed. Baseline and one-clear training peaks are identical. Setup release is the one-clear arm's reserved reduction. Values are exact bytes, with no headroom folded into estimates.

| subject | placement | tokens | allocated peak B | reserved peak B | sampled driver peak B | full estimate B | driver minus estimate B | setup clear releases B |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama31_8b | R | 2048 | 9947172352 | 11463032832 | 12123635712 | 12866890960 | -743255248 | 0 |
| llama31_8b | S | 2048 | 6675090944 | 9160359936 | 9892265984 | 8941022416 | 951243568 | 0 |
| llama31_8b | R | 4096 | 13670706688 | 16307453952 | 16968056832 | 17407921364 | -439864532 | 0 |
| llama31_8b | S | 4096 | 10398625280 | 13885243392 | 14617149440 | 13482052820 | 1135096620 | 0 |
| qwen3_32b | R | 2048 | 23583963648 | 25222447104 | 25883049984 | 29717581660 | -3834531676 | 33554432 |
| qwen3_32b | S | 2048 | 8439691776 | 11815354368 | 12547260416 | 11579314012 | 967946404 | 0 |
| qwen3_14b | S | 512 | 5615508480 | 7430209536 | 8162115584 | 7602980481 | 559135103 | 0 |
| qwen3_14b | S | 4096 | 8213359104 | 10991173632 | 11723079680 | 10705147463 | 1017932217 | 0 |

For Llama streamed 4096, loader R/A = 2,214,592,512 / 2,213,095,424 B and setup R/A = 2,382,364,672 / 2,380,867,584 B. Setup cached bytes were 1,497,088, with no observed release. Training peak R/A = 13,885,243,392 / 10,398,625,280 B. The large reserved slack developed during training. Peak reserved minus peak allocated is a peak-comparison metric; those peaks need not occur together. No new synchronization, peak reset or tensor-data reads were added by the phase collector. No reserve coefficient is fitted or imported here.

One RTX 5090, 32607 MiB, driver 595.91.07, PCIe gen5/x16. Loggetta efe1c93; torch2.8.0+cu128, bitsandbytes0.50.2, transformers5.18.0 and PEFT0.21.2; exact pins in runtime.json. Synthetic architecture-only checkpoints, no pretrained weights or quality claim. NF4/doublequant64/bf16 compute, fp32 all-projection LoRA r16/alpha32, SDPA/non-reentrant checkpointing, micro-batch1/grad-accum1, two AdamW steps at constant2e-4, clip1, default allocator. Qwen loss chunk512; Llama full logits. All eight shapes passed actual planner admission before full checkpoint generation.

Guard armed 21:36:28Z, runner finished and final fetch completed 22:01:43Z. [Teardown](teardown-proof.json) at 22:01:44Z verified provider HTTP200 and instance54915655 absent. Actual invoice **$0.343**: GPU$0.288 + storage$0.047 + download$0.008 + upload$0.000; reservation$2.80. No model or adapter weight files were fetched. [Run provenance](run-provenance.json) records the invoice, source receipt hash and pinned local re-derivation. No further draw is authorized by this record.

## Reproduce without a GPU

The modest [raw scientific archive](raw-science.tar.gz) contains unchanged JSON receipts, four proofs, exact token rows, config/admission/checkpoint manifests and executed registered instrument sources. It contains no credentials, model weights, adapter tensors or private bus URI. Plain SHA256SUMS verifies the public files; SCIENCE-SHA256SUMS inside the archive verifies every raw member. The registered source checksums are also present.

From this receipt directory:

```sh
shasum -a 256 -c SHA256SUMS
mkdir /tmp/dq9-read
tar -xzf raw-science.tar.gz -C /tmp/dq9-read
cd /tmp/dq9-read/raw
shasum -a 256 -c SCIENCE-SHA256SUMS
shasum -a 256 -c instrument.sha256
python dq9_reduce.py receipts --runtime runtime.json --e4b-sha a04b7dc4e73bc780f0fae2e0a1bb114cc4578ec2 > /tmp/dq9-rederived.json
```

The output must be byte-identical to the standalone dq9-read.json (SHA256 9a77ab73cadc31b35a89c061cb6d6cfabb5c3da09ace44e359bb0c79ab9a83c6), with COMPLETE_DIAGNOSTIC and both license flags false. The archive's reducer/source files match the registered a04b7dc4 instrument; the same result can be obtained with that checkout's reducer and `PYTHONPATH=bench/dq7`.
