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
