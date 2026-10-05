# DQ3 — results

Pre-registration: [DQ3-PREREG.md](DQ3-PREREG.md) (Amendments 0–3). Work item: #1083. Prototype:
`enable_dense_offload(..., train_prefetch=True)` (#1167) with the late-bound 4-bit backward (#1183).

## Verdict — `dq3-5090-5`, 2026-10-05: **PROTO_PASS** (step PASS, coverage PASS, capacity PASS)

**Bottom line.** With the frozen NF4 weights of a 64-layer Qwen3-32B-shaped QLoRA model streamed one layer ahead over
PCIe gen 5 x16:
- training stays **bitwise identical** to the resident model;
- it runs at **1.0023×** resident step time;
- it frees **15.08 GB** of GPU memory, **99.75 %** of the slot prediction.

| gate (registered) | reading | result |
|---|---|---|
| parity: loss and every LoRA gradient sha256 equal across arms (deterministic pass), with an R-vs-R control | all six arms equal; no `function_fail` | PASS |
| step time T(S)/T(R) ≤ 1.10 | **1.0023** (2.929 s against 2.922 s) | PASS |
| coverage: per steady step exactly 62 fwd + 62 bwd prefetches, blocking ≤ 2, residency ≤ 2, 0 unscheduled | 62 + 62, **0 blocking**, high-water **2**, 0 unscheduled, every timed step | PASS |
| capacity: saving ≥ 0.9 × 62 × 243,793,920 B (13.60 GB) | **15,077,867,520 B** (24.77 → 10.73 GiB peak allocated) against a prediction of 15,115,223,040 B | PASS |
| VOID / NOISY | no VOID cause; R self-pair 0.9976, in its band | — |

**Descriptive, not gated:**
- S0 (today's synchronous offload, on the same late-bound backward) costs **T(S0)/T(R) = 1.18** (3.454 s) and frees
  slightly more (peak 10.27 GiB). The overlap is what turns 18 % into 0.2 %.
- In the forward, 9–13 of the 62 prefetches per step were still in flight when their layer started, so compute waited
  on them. They are counted as `waited`, not `blocking`: none fell back to a synchronous copy. All 62 backward
  prefetches had completed before use.

| arm | mean step (6 timed) | peak allocated | peak reserved | streamed |
|---|---|---|---|---|
| 1 R | 2.9259 s | 24.77 GiB | 28.01 GiB | — |
| 2 S | 2.9297 s | 10.73 GiB | 22.77 GiB | 64 layers / 448 tensors, 15.60 GB pinned |
| 3 S0 | 3.4537 s | 10.27 GiB | 20.45 GiB | same |
| 4 S0 | 3.4537 s | 10.27 GiB | 20.45 GiB | same |
| 5 S | 2.9286 s | 10.73 GiB | 22.77 GiB | same |
| 6 R | 2.9189 s | 24.77 GiB | 28.01 GiB | — |

**Allocated against reserved.** The registered capacity gate is on `max_memory_allocated`. Peak **reserved** drops less
(28.01 → 22.77 GiB), because the caching allocator keeps streamed layers' freed blocks cached. PyTorch releases cached
blocks and retries before it raises an OOM, so allocated is the capacity a larger model can use. `nvidia-smi` shows the
reserved figure, though. A follow-up should measure the largest model that actually fits, rather than infer it.

| | |
|---|---|
| box | RTX 5090 (driver 595.84, 575 W), **PCIe gen 5 x16** (Vast `pcie_bw` 53.9 GB/s), AMD Ryzen 9 9950X, machine 151044, Thailand |
| subject | Qwen3-32B architecture, 64 layers, random NF4 (bnb, blocksize 64, double-quant), PEFT LoRA r16/α32 on all seven projections (448 `peft.tuners.lora.bnb.Linear4bit`, fp32 adapters), non-reentrant checkpointing, 2048 tokens, AdamW |
| software | torch 2.8.0+cu128, bitsandbytes 0.50.2, transformers 5.18.0, peft 0.21.2 (tripwire: e4b `0bf98cf3`) |
| launch | e4b `0bf98cf3` (contains #1167, #1171, #1173, #1183, #1185), adertha `56a229d1` |
| pre-checks on the box | link gen 5 x16; VRAM probe 29.5 GiB; egress probe 3.16 MB/s |
| cost | $0.282; teardown proven (`vast-destroy`, instance absent) |
| evidence | [`receipts/dq3-5090-5/`](receipts/dq3-5090-5/) (`SHA256SUMS`); launcher receipt and ledger row in the store, commit `9ecbf055` |

## The root cause found on the way: bitsandbytes keeps the frozen weight on `ctx`

Without #1183, offloaded training through bitsandbytes `Linear4bit` **saves no VRAM at all**.

**The mechanism.** bitsandbytes 0.50.2's `MatMul4Bit.forward` stores the packed weight as a ctx attribute
(`ctx.tensors = (None, B)`, where `B` is a view of `Linear4bit.weight`) whenever the input needs grad. It does not go
through `save_for_backward`, so non-reentrant checkpointing's saved-tensor hooks never see it. Every layer's weight
storage stays referenced by the autograd graph from its forward to its backward, and offload's
`p.data = placeholder` frees nothing.

**What it did here.** In `dq3-5090-3`, every probe passed and R completed (peak 24.77 GiB). S then ran out of memory
with 30.38 GiB in use: streaming had cost memory.

**The fix (#1183).** It mirrors the expert side's `_FrozenLinearRecomputeBackward`:
- every offloaded `Linear4bit`'s grad-mode matmul goes through `_LateBoundMatMul4Bit`;
- ctx keeps the module, not a tensor;
- the forward is bnb's own no-grad `gemm_4bit` call;
- the backward is `MatMul4Bit.backward`'s expressions on the weight bound at backward time.

It is bitwise identical to stock bnb by construction (and in this read). It is pinned to bnb 0.50.2's source by sha256,
and on any mismatch it warns and keeps stock bnb.

**bnb 0.50.2 behaviour, worked around locally; not reported upstream** (owner decision, 2026-10-05).

## The rehearsal-gate lesson

The A2000 rehearsal at real width (4 layers) **had already shown the wrong-way peak**: S at 6.47 GiB against R at 5.82.
It was read as green, because it checked that every arm ran, parity and counters, and never compared the one quantity
the lane exists to change.

`bench/dq3/dq3_rehearsal_check.py` (Amendment 3) now gates the launch on that quantity: `peak(S), peak(S0) ≤ peak(R) −
(L−2) layers + ½ layer`. Run over the pre-fix rehearsal receipts, it fails every streamed arm. It passed at the launch
commit, at R 5.820 / S 5.366 / S0 4.912 GiB.

## Launch attempts ($0.625 in all)

| run | machine / host | outcome | cause | fix | cost |
|---|---|---|---|---|---|
| `dq3-5090-1` | machine 139937, host 564677 (Ryzen 9 9950X, driver 595.84, BG) | HARNESS_ERROR, rc 11 | the host refused the subject's first 2.90 GiB allocation with 30.85 GiB free | #1171: VRAM probe, rc 18, before any install | $0.058 |
| `dq3-5090-2` | machine 147454 (Ryzen 7 7800X3D, Shanghai) | HARNESS_ERROR, rc 9 | GitHub egress ~33 KB/s; the install could not finish inside pip's alarm (the stuck pip was killed at ~9.5 min) | #1173: egress probe, rc 14 | $0.203 |
| `dq3-5090-3` | machine 37087, host 223199 (Ryzen 9 9950X, driver 595.84, SE) | HARNESS_ERROR, rc 11 | arm S out of memory: **bnb pins every weight on ctx** (above) | #1183: late-bound backward; Amendment 3 (#1185) | $0.068 |
| `dq3-5090-4` | machine 147454 again | NOT_RUN | stuck `loading` 600 s; not machine evidence (adertha#164) | relaunch | $0.014 |
| `dq3-5090-5` | machine 151044, host 681936 (Ryzen 9 9950X, driver 595.84, TH) | **OK — PROTO_PASS** | — | — | $0.282 |

**Host notes for adertha#164's pair work:**
- Machine 151044 loaded and ran cleanly here. Its stuck load on `dq2-5090-1` stays a single.
- Runs 1, 3 and 5 all drew the same class: Ryzen 9 9950X, driver 595.84. Run 3's host allocated the subject normally,
  and run 5 completed. So run 1's refusal of a large allocation was specific to host 564677, not to the class.
- Machine 147454 has now been drawn twice. It failed both times, once on egress and once stuck loading, and no receipt
  class can exclude it.

## Independent review (before the PR)

**Who reviewed, and what they found.** An agent separate from the run re-checked everything from the raw files and found
**no defect that changes the verdict**. It verified that:
- the six arm receipts are in the registered order and match the subject, engagement and launch commit;
- the reducer re-run is byte-identical, and its self-test passes;
- T(S)/T(R) = 1.00230 and the saving of 15,077,867,520 B recompute independently from the raw arms;
- all 896 gradient hashes and the loss bits are equal across all six arms on both steps;
- the counters hold on 12 of 12 steady steps;
- the VOID/NOISY order is as registered;
- the S arms really streamed: 64 layers, 448 tensors, 243,793,920 B per layer, 15.60 GB pinned;
- teardown is proven.

**Its minor findings, recorded here and not hidden:**

1. **Some registered VOID conditions are enforced on the box, not in the reducer.** The link check is the runner's
   rc 13, and identical software across arms comes from one tripwire'd install. The arm receipts don't record package
   versions, so "software differs across arms" can't be re-checked from them alone.
2. **The late-bound route's engagement is not recorded in the arm receipt.** The evidence is indirect: S and S0 peak
   14 GiB below R, and there is no bnb-mismatch warning in the logs. Follow-up: have `dense_offload_report` return the
   routed-projection count and `dq3_arm.py` record it.
3. **The parity pass ran `use_deterministic_algorithms(True, warn_only=True)`, where the prereg says `(True)`.** The
   logs show no determinism warnings, and the R1 = R2 control holds. At step 1 every `lora_A` gradient is zero (PEFT
   initialises B at zero), so step-1 parity exercises `lora_B` only. Step 2 has 896 distinct hashes.
4. **One stamped prediction missed (ungraded).** The design note predicted "waited ≤ 6 per step". The forward waited
   9–13 per step, and the backward 0. The reducer also leaves out pinned-host reserved bytes; for the record they are
   17,716,740,096 B against 15,602,810,880 requested, a ratio of 1.1355 (power-of-two rounding).
5. **Reserved against allocated** (above). Part of the reserved gap is cache carried over from the parity pass:
   `dq3_arm.py` doesn't call `empty_cache()` before `reset_peak_memory_stats()`. That doesn't touch the registered gate,
   but it matters for any footprint claim.
6. **Bookkeeping.**
   - Amendment 3 says "run 4 re-measures"; run 4 never started (stuck loading), so run 5 is the first full read.
   - `dq3_run.sh`'s header comment still cites Amendments 0–1.
   - The launch went through `pod-launch.sh` → `rent.py` → `tc1_drive.sh`.
   - The launch-commit rehearsal gate's receipts are committed at
     [`receipts/rehearsal-gate-0bf98cf3/`](receipts/rehearsal-gate-0bf98cf3/) (REHEARSAL CHECK OK).

## Scope: what this does and does not show

- **One card, one host, one link:** an RTX 5090 on PCIe gen 5 x16 (53.9 GB/s measured by Vast).
  - DQ2 put the break-even near 1024 tokens on this link. At 2048 tokens the copy hides behind compute with margin.
  - On **gen 4 x16 (~27 GB/s)**, DQ2's ratio predicts the copy no longer fully hides at 2048. That needs its own read;
    nothing here generalises past gen 5.
  - A different card, such as a 4090 or an RTX 6000-class card, changes both compute and VRAM and needs its own read.
- **One layer stack** at Qwen3-32B width (hidden 5120, intermediate 25600, 64 layers), random weights. Parity, time
  and bytes do not depend on weight values (Amendment 1). A different width changes the per-layer copy-to-compute
  ratio.
- **One micro-batch shape** (1 × 2048), non-reentrant checkpointing, PEFT + bnb QLoRA.
- **S0/R is descriptive.**
- **No end-to-end "a bigger model fits" demonstration.** The 15.08 GB is allocator headroom on this subject. The
  reserved-memory note above applies.

## Reproduce

The verdict re-derives byte for byte from the committed arm receipts:

```
cd bench/dq3/receipts/dq3-5090-5
python3 ../../dq3_reduce.py arm1_R.json arm2_S.json arm3_S0.json arm4_S0.json arm5_S.json arm6_R.json | cmp - dq3_read.json
python3 ../../dq3_reduce.py --self-test
```

`SHA256SUMS` covers every committed file. The box-side run is `bench/dq3/dq3_run.sh`, launched through
`bench/tc1/tc1_drive.sh` with `TC1_RUNNER=dq3_run.sh` and the stage list in the launch manifest.

## What next (not registered here)

1. **A capacity read:** the largest model, or the longest sequence at a fixed model, that trains on one 5090 with S
   but not R. This turns the 15 GB into a user-visible number, reserved bytes included.
2. **A gen 4 x16 read** of the same lane, to bound the break-even DQ2 predicted.
3. **Whether `train_prefetch` should become the training default** for dense offload. That is a design decision for
   the maintainer after (1).
