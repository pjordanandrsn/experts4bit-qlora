# DQ5 — results

Pre-registration: [DQ5-PREREG.md](DQ5-PREREG.md) (DQ3's lane and rule, on PCIe gen 4 x16). Work item: #1083.

**Scope:** one RTX 5090 on a **PCIe gen 4 x16** link (machine 145701, AMD EPYC 7B13), e4b `3db3fd46`, adertha `86f114e0`,
and DQ3's registered subject and recipe.

## Verdict — `dq5-5090-1`, 2026-10-06: **PROTO_PASS** (step PASS, coverage PASS, capacity PASS)

On gen 4 x16, streaming the frozen NF4 weights one layer ahead keeps QLoRA training **bitwise identical** to resident at
**1.0050×** its step time, and frees 15.08 GB.

| gate (DQ3's, registered) | gen 4 x16 (DQ5) | gen 5 x16 (DQ3, #1188) |
|---|---|---|
| parity: loss and every LoRA gradient sha256 equal across arms (deterministic pass, R-vs-R control) | equal across all six arms | equal |
| step time T(S)/T(R) ≤ 1.10 | **1.0050** (2.979 vs 2.964 s) | 1.0023 |
| coverage: per steady step 62 fwd + 62 bwd prefetches, blocking ≤ 2, residency ≤ 2, 0 unscheduled | 62 + 62, **0 blocking**, high-water 2, 0 unscheduled, every timed step | same |
| capacity: saving ≥ 0.9 × 62 × 243,793,920 B | **15,076,622,336 B** = 99.74 % of the slot prediction | 99.75 % |
| NOISY: R self-pair in [0.97, 1.03] | 0.9962 | 0.9976 |
| S0/R (descriptive) | **1.51** (4.483 s) | 1.18 |
| forward copies still in flight when their layer started (`waited`), per steady step | **20–59** of 62 (backward 0–1) | 9–13 |

No VOID cause, no FUNCTION_FAIL, not NOISY.

**Against the registered predictions:**
- T(S)/T(R) [1.00, 1.05] read 1.0050, inside the band.
- Coverage and capacity passed, as predicted.
- S0/R, predicted ~1.25–1.4, read 1.51, above the range. This is descriptive: each synchronous copy costs more at gen 4's
  bandwidth.

**What gen 4 changes:**
- More forward copies are still in flight when their layer starts: 20–59 against 9–13 on gen 5. A copy of one layer
  takes about 11 ms at 21 GB/s, against about 15 ms of forward compute.
- The overlap still absorbs them. The streamed arm pays **0.5 %** of step time where the synchronous arm pays **51 %**.

## The host, and how much of 1.0050 is noise

**The host.** Machine 145701 (AMD EPYC 7B13, 256 vCPU) is the host on which TC1 amendment 45 recorded:
- a container cgroup quota of **31 CPUs**;
- host load1 medians of **4–75** over its arms ([TC1 README, amendment 45](../h2h-2026-10-02/tc1/README.md)).

**What DQ5's receipts record, and what they don't.** They do not carry host load during the arms. The launcher's
pre-flight CPU probe read `nproc` 256 and ran a 2M int multiply-add loop in 0.0872 s.

**Noise against the 0.5 % difference.** Within each arm, the six timed steps spread **3.8–6.8 %** (min to max over the
mean):

| arm | R | S | S0 | S0 | S | R |
|---|---|---|---|---|---|---|
| spread | 3.9 % | 6.1 % | 4.7 % | 3.8 % | 6.8 % | 5.5 % |

That is wider than the 0.5 % between S and R. So the reading is: **S and R are indistinguishable within this host's
step-to-step noise, and both are far inside the 1.10 gate.** It is not a claim that streaming costs exactly 0.5 %. The
registered stability check, R's self-pair, is 0.9962, inside [0.97, 1.03].

**The thinner margin is the honest headline beside 1.0050.** At gen 4, **20–59 of 62** forward copies per step were still
in flight when their layer started, against 9–13 at gen 5. That still means **0 blocking**: the copy hides, but closer to
the edge.

## Independent review (before the PR)

**Who reviewed, and what they found.** An agent separate from the run re-checked everything from the raw receipts and
found **no defect that changes the verdict**. It confirmed:
- the reducer re-run is byte-identical, and the self-test gives 25 OK;
- every staged box file matches `3db3fd46` by sha256, and the tripwire checks the full sha;
- the six arms match the registered device, subject, engagement and order;
- S and S0 stream 64 layers / 448 tensors, with `late_bound_4bit` 448;
- independently recomputed: T(S)/T(R) 1.00503 (the median of steps gives 1.0024), S0/R 1.5125, R self-pair 0.99625, the
  saving, parity (all 896 grad hashes and the loss bits, both steps), and the counters on all 12 steady steps;
- the link evidence: the max gate, `['16, 4']` ×3 under load, idle 1/16, and Vast's 4.0 / 16 / 21.8;
- the H2D probe cannot refuse;
- teardown is proven.

**The link was really gen 4.** S0's extra cost, about 1.52 s for 128 layer copies a step, implies about 20.5 GB/s during
training, consistent with the probe. So the copies really crossed a gen 4 link, and S hides about 99 % of that cost.

**Its findings, recorded here:**
1. **Missed descriptive predictions:**
   - S0/R read 1.51, against the predicted 1.25–1.4.
   - Back-to-back H2D read 22.5 GB/s, against the ~27.9 from DQ1's gen 4 host. The rate is per host, as the existing
     memory finding says.
   - Forward `waited` reached up to 59 of 62 in a step.
2. **Budget bookkeeping:** the launcher reserved and approved $1.95, including a 100 GB download allowance; the prereg
   stated ≤ $0.85. Actual cost was $0.401.
3. **The launcher receipt's artifacts entry is stale:** it records 12,266 B with sha `bafb94…` for a file now 12,416 B with
   sha `b43d66…`. Its `vast_actual_status` reads "running", and `reviewed_by` is null. The lane receipts committed here
   are the evidence.
4. **`h2d.json` has no `lspci_lnk` key,** because lspci was absent on the box, and the absence is not recorded.
5. **The A2000 rehearsal ran at `62d4b81a`, not the launch commit.** The arm and engine code is identical between the
   two. The only change after it is the width gate, which was validated on its own (`dq5-a2000-linkgate`).
6. **Outside the graded steady steps,** the parity pass and first warm step together held one forward blocking fetch
   (187 vs 186 issued, cumulatively).

## The link: what was measured, and how the gate read it

| instrument | reading |
|---|---|
| `nvidia-smi pcie.link.gen.max,width.max` (box gate, part 1) | **4, 16** |
| `pcie.link.width.current` and `gen.current` under a pinned copy (box gate, part 2, `dq5_link_gate.py`) | **16, 4** (×3) |
| the same, at idle (H2D probe) | **1, 16**: the generation idles at 1, which is why the generation is gated on its maximum and the width under load |
| Vast's offer (`pci_gen`, `gpu_lanes`, `pcie_bw`) | 4.0, 16, 21.8 GB/s |
| pinned H2D, **descriptive only, no bandwidth claim** (`h2d.json`): single 64 MiB / 40 back-to-back 64 MiB / single 243.8 MB layer | **21.71 / 22.46 / 20.94 GB/s**, single over back-to-back 0.967 |

**The width gate was added during this lane, before any 5090 data.** The A2000 rehearsal read `width.max` = 16 while its
link actually ran x8 under load, so a max-only gate would have admitted an x8 5090. DQ3's gen 5 gate is max-only and
should get the same under-load width check. DQ3's host measured 53.9 GB/s, consistent only with x16, so its read
stands.

| | |
|---|---|
| box | RTX 5090 (driver 595.84), **PCIe gen 4 x16**, AMD EPYC 7B13, machine 145701, Italy |
| software | torch 2.8.0+cu128, bitsandbytes 0.50.2, transformers 5.18.0, peft 0.21.2 (tripwire: e4b `3db3fd46`) |
| offer search | `vast_pcie {min_gen 4, min_lanes 16, max_gen 4}`, the new ceiling from adertha-agents#177 |
| cost | $0.401; teardown proven |
| evidence | [`receipts/dq5-5090-1/`](receipts/dq5-5090-1/) (`SHA256SUMS`); launcher receipt and ledger row in the store, commit `ae253242`, which the SC1g session committed as-is alongside its own receipt and pushed |

**Build time:** about 240 s per arm, against about 84 s on the gen 5 hosts. Arm build is CPU-bound NF4 quantization, and
this host's EPYC 7B13 runs at a lower clock. Build is not inside any gate.

## Not shown

- Gen 3, or x8 links.
- Any card other than the 5090.
- A capacity (sequence-length) boundary at gen 4. DQ4's bytes do not depend on the link.
- Which H2D probe "is the link".

## Reproduce

```
cd bench/dq5/receipts/dq5-5090-1
python3 ../../../dq3/dq3_reduce.py arm1_R.json arm2_S.json arm3_S0.json arm4_S0.json arm5_S.json arm6_R.json | cmp - dq3_read.json
python3 ../../../dq3/dq3_reduce.py --self-test
```

The output is byte-identical. `SHA256SUMS` covers every committed file.
