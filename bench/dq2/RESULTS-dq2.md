# DQ2 — results

Pre-registration: [DQ2-PREREG.md](DQ2-PREREG.md). Work item: #1083.

## Launch attempts before the first read (host failures, no lane data)

The launcher's opt-in PCIe band (adertha-agents#162) did select gen 5 x16 offers. Five attempts died at its pre-flight or
were refused before a rental. Each has a committed receipt and ledger row in the store.

| run | machine | outcome | cost |
|---|---|---|---|
| `dq2-5090-1` | 151044 (Ryzen 9 9950X, gen 5 x16, Vast pcie_bw 53.8) | NOT_RUN: instance stuck `loading` for 600 s | $0.014 |
| `dq2-5090-2` | — | REFUSED before rental: a stuck-loading receipt cannot exclude a machine (filed as adertha-agents#164) | $0 |
| `dq2-5090-3` | 151044 again | NOT_RUN: HF-CDN probe 13.0 MB/s < 20 | $0.076 |
| `dq2-5090-4` | — | REFUSED before rental: one bandwidth refusal does not name a machine (pair rule) | $0 |
| `dq2-5090-5` | 147454 | NOT_RUN: HF-CDN probe 0.0 MB/s (timeout); generic endpoints 22.7 | $0.038 |

None of them is a lane read. DQ2's box downloads no checkpoint, so the HF-CDN floor tests a path the lane never uses.
Making that gate optional for such lanes has been proposed to the launcher's owner; it is not changed here.

## Run 1 — `dq2-5090-6`, 2026-10-05: READ, stream **C_UNCOVERED** (the registered rule withholds the stream verdict)

| | |
|---|---|
| box | RTX 5090 (driver 595.91.07, 500 W), **PCIe gen max 5, width 16**, AMD Ryzen Threadripper PRO 7965WX, machine 151831 |
| software | torch 2.8.0+cu128, CUDA 12.8, bitsandbytes 0.50.2, transformers 5.18.0, peft 0.21.2 (tripwire) |
| launch | e4b `de7faf46` (#1135's merge), adertha `af56983f` |
| cost | $0.034 actual; teardown proven (`vast-destroy` HTTP 200, instance absent) |
| evidence | [`receipts/dq2-5090-6/`](receipts/dq2-5090-6/) (`SHA256SUMS`); launcher receipt and ledger row in the store, commit `bc718597` |

**What held:**
- The link was gen 5 x16.
- Engagement: `peft.tuners.lora.bnb.Linear4bit` ×7, fp32 adapters.
- Integrity: finite, 7/7 LoRA-B gradients nonzero, frozen-storage sha256 unchanged.
- Every warm-up reached steady.
- Self-pairs: 1 of 10 out of band (budget 1).
- The lane reads READ.

**What failed: one coverage flag.** One of the 25 streaming draws, the copy-under-forward reading at M = 512, was not
covered end to end. Every draw at the graded row (2048) was covered. The rule gives C_UNCOVERED if any loaded reading,
in any row, was uncovered, so the stream verdict is withheld. Its readings are not graded and are not quoted here.

### Diagnosis: the instrument at a host-bound row

At M = 512 the layer's checkpointed forward takes ~4.8 ms and is host-launch-bound (PEFT + bnb + checkpoint dispatch).
The probe's copy-under-load window enqueued all its forwards before any copy, on the assumption that enqueueing is much
faster than the GPU work. That does not hold at this row: the host finished enqueueing as the GPU finished computing,
so the copies started near the end of the forward window.

The draw's own numbers show it. The copy read 55.9 GB/s "under load" against 54.2 alone, so its tail was unloaded.

### Registered consequence, and what changes

C_UNCOVERED earns one re-run on another gen 5 host; a second stops the lane. Re-running the same probe would re-read the
same artifact at 512 on any host. So **Amendment 1**
([DQ2-PREREG.md](DQ2-PREREG.md#amendment-1-registered-2026-10-05-after-run-1-read-c_uncovered-before-run-2)) changes
only the copy-under-load window: the first forward is enqueued and marked, the side stream waits on the mark, the copies
are enqueued, then the remaining forwards. The rule (`dq2_reduce.py`) is byte-identical (pinned by sha256 in
`tests/test_dq2_lane.py`). The predictions and consequences are unchanged.

## Run 2 — `dq2-5090-7`, 2026-10-05, under Amendment 1: READ, **C_ALIVE**

| | |
|---|---|
| box | RTX 5090 (driver 595.91.07, 470 W), **PCIe gen max 5, width 16**, Intel Core Ultra 9 285K, machine 151350. A different host from run 1. |
| software | torch 2.8.0+cu128, bitsandbytes 0.50.2, transformers 5.18.0, peft 0.21.2 (tripwire) |
| launch | e4b `93e6725e` (#1144's merge), adertha `8fbaf56d`; Vast search banded to gen ≥ 5 / x16 / ≥ 40 GB/s, with the pre-flight's HF-CDN gate off (`preflight_bandwidth: none`; this lane pulls no checkpoint) |
| cost | $0.032 actual; teardown proven (`vast-destroy` HTTP 200, instance absent) |
| evidence | [`receipts/dq2-5090-7/`](receipts/dq2-5090-7/) (`SHA256SUMS`); launcher receipt and ledger row in the store, commit `b194aa62` |
| instrument | Self-pairs **0 / 10** out of band (worst 1.002); **25 / 25** streaming draws covered, including M = 512 (Amendment 1); every warm-up steady (179.5–180.0 TF/s, end within 0.5%). Engagement: `lora.bnb.Linear4bit` ×7, fp32. Integrity: finite, 7/7 LoRA-B gradients nonzero, frozen sha256 unchanged. |

An independent re-computation that did not import the reducer reproduces every reading and the verdict.

| M | T_fwd ms | T_bwd ms | X ms | H2D alone / loaded GB/s | forward slowdown under DMA | R_fwd | R_bwd | **Rmin** |
|---|---|---|---|---|---|---|---|---|
| 512 | 4.96 | 8.85 | 5.41 | 48.4 / 46.5 | 1.017 | 0.92 | 1.64 | 0.92 |
| 1024 | 7.79 | 16.02 | 5.33 | 48.6 / 47.2 | 1.017 | 1.46 | 3.00 | 1.46 |
| **2048** | **15.02** | **31.96** | **5.25** | **48.6 / 47.9** | **1.016** | **2.86** | 6.09 | **2.86** |
| 4096 | 30.17 | 66.48 | 5.18 | 48.7 / 48.6 | 1.018 | 5.82 | 12.83 | 5.82 |
| 8192 | 63.36 | 142.63 | 5.19 | 48.6 / 48.4 | 1.040 | 12.20 | 27.46 | 12.20 |

Break-even (reported, not graded): Rmin ≥ 1.0 and ≥ 1.25 are both first met at **1024**.

### Against the stamped predictions: 8 of 9 clauses held

| clause | registered | measured | |
|---|---|---|---|
| T_fwd(2048) | [11, 15] ms | 15.02 | **missed by 0.019 ms (0.13%)** |
| T_bwd(2048) | [22, 34] ms | 31.96 | held |
| H2D under load | [40, 56] GB/s | 47.9 | held |
| copy_ratio | ≥ 0.9 | 0.986 | held |
| forward slowdown under DMA | ≤ 1.05 | 1.016 | held |
| X | [4.5, 6.3] ms | 5.25 | held |
| **Rmin(2048) = R_fwd → verdict** | **[1.8, 3.2] → C_ALIVE** | **2.86 → C_ALIVE** | **held** |
| R_fwd(1024) | [0.9, 1.6] | 1.46 | held |
| break-even row (≥ 1.25) | 1024 or 2048 | 1024 | held |

The T_fwd miss comes from the timing method: each rep synchronises, so it carries a little host overhead. The
probe's back-to-back forward under no load reads 14.42 ms at 2048. A pipelined implementation would see that figure,
giving R_fwd ≈ 2.79 at 2048, 1.40 at 1024 and 0.82 at 512. No verdict changes.

### What the read supports, and how far

1. **Robust.** The draws at 2048 are tight (forward 14.63–15.03 ms, loaded bandwidth 47.7–48.0 GB/s), and the
   worst-case pairing gives R 2.78. Flipping to below 1.25 would take a 2.3× error: a link under ~21 GB/s (PCIe 4.0
   class), or a forward half as long. Run 1, on another gen 5 host, reads 14.82 / 31.45 ms at 2048 (within 1.6%) and an
   ungraded R_fwd of 3.31. The R gap between runs tracks the link (56 vs 48.6 GB/s).
2. **One transfer per phase is the right model.**
   - In the forward, layer i+1 is copied while layer i's checkpointed forward runs (no saved activations).
   - In the backward, one resident copy of layer i serves both its recompute and bnb's dgrad dequant.
   - That is 2L transfers a step. Even two transfers in the backward would leave R_bwd/2 = 3.04 > R_fwd.
   - Two resident slots (current + prefetch) cost ~0.5 GB for Qwen3-32B (~0.9 GB at 70B).
3. **The planner rule, measured.** Use the layer's measured rate, not the linears':
   - R = 1 at M* ≈ 3.4e4 / B tokens and R = 1.25 at ≈ 4.25e4 / B, with B the pinned H2D under load in GB/s, for this
     layer class on an RTX 5090.
   - DQ1's linears-only rule (0.26·F/B with F = 207 TF/s) is conservative by ~1.6×, because attention, norms, RoPE and
     LoRA add ~55% to the linears' time.
   - At the low end of Vast's gen 5 band (~41.6 GB/s), R_fwd(1024) ≈ 1.27, so 1024 is marginal there and 2048 is not.
4. **Scope.** This is one layer, on gen 5 x16 hosts with ≥ ~45 GB/s, at ≥ ~1024 tokens a micro-batch. Not measured, in
   order of risk:
   - whole-model overlap: the prefetch scheduler's hooks and recompute trigger, host-launch bubbles at M ≤ 1024, and
     the ~5 ms pipeline fill in each phase;
   - pinned host memory: ~35–37 GB for a 70B model's NF4 weights; the probe re-reads one 251 MB buffer;
   - PCIe contention from other host-to-device traffic: activation or optimizer offload, data loading;
   - other models: for a 70B layer, R_fwd(2048) ≈ 2.5 (derived);
   - faster kernels (fused LoRA, faster dequant) cut T_fwd and so cut R proportionally.

   The capacity benefit itself only matters where the frozen weights do not fit. Qwen3-32B NF4 (~16 GB plus embeddings)
   fits resident on a 5090; the payoff begins around 50B+ on this card.

### Registered consequence

**C_ALIVE.** The capacity axis is earned at the prototype level: a streamed frozen-weight prototype under autograd is
licensed **on a branch, with no new repository**. Its own registration must carry:

- **Outcome gates:**
  - gradient parity, streamed against resident, on a model that fits;
  - streamed step time within a registered margin of resident at ≥ 2048 tokens a micro-batch;
  - a fence proving the copy stream never overwrites a slot still being read (the write-after-read hazard
    grouped-nf4-gemm#467 fenced for the MoE tier).
- **Host requirements:** measured pinned H2D under load, pinnable host RAM ≥ the model's NF4 bytes, and gen 5 x16.
- **Then a matched-work lane:** a model larger than VRAM against FSDP-QLoRA with CPU offload.

The repository gate in [SUMMARY-dq1.md](../dq1/SUMMARY-dq1.md) is unchanged: no package split until the prototype and
its matched-work result exist.
