# DQ5 — pre-registration: DQ3's streamed-QLoRA lane on a PCIe **gen 4** x16 link

Work item: #1083. Owner, 2026-10-05: "register the gen 4 read and launch it".

**Why.** DQ3 (#1188) read PROTO_PASS on a gen 5 x16 RTX 5090: bitwise parity, T(S)/T(R) = 1.0023, 0 blocking
prefetches, 15.08 GB freed. DQ4 (#1206) turned that into 2.00× the trainable sequence length. DQ3's results limited the
claim to gen 5, and noted that DQ2's ratio predicts a tighter, still hiding, overlap at about half the bandwidth. DQ5
measures whether the prototype keeps its speed on the most common slower link a 5090 is installed on.

## What runs

**DQ3's lane, unchanged, on a different link.** The same:
- subject: Qwen3-32B architecture, 64 layers, random NF4, PEFT LoRA r16 on all seven projections, non-reentrant
  checkpointing, 2048 tokens, Hugging Face's stock loss;
- arm script (`bench/dq3/dq3_arm.py`);
- arms: R resident, S `train_prefetch` plus #1183's late-bound backward, S0 synchronous offload;
- palindrome: R S S0 S0 S R;
- deterministic parity pass, then the timing pass (2 warm + 6 timed);
- software pins: bitsandbytes 0.50.2, transformers 5.18.0, peft 0.21.2, torch 2.8.0+cu128, grouped-nf4-gemm `a5edec87`;
- host gates: DQ3's VRAM probe (rc 18) and egress probe (rc 14).

#1203's default-on `auto` chunked loss applies only through `enable_fast_train` and the CLI trainer. `dq3_arm.py` calls
neither, so the loss path is DQ3's.

**What differs:**
- **The link gate**, in two parts:
  - `nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max` must read **4, 16** on an RTX 5090. For the
    generation, the maximum is the instrument, because the *current* generation idles at 1.
  - `bench/dq5/dq5_link_gate.py` must then read `pcie.link.width.current` = **16** *while a pinned copy is in flight*.
    `width.max` is only what the GPU and slot can negotiate: the A2000 rehearsal read max 16 while its link ran x8
    under load, so a 5090 negotiated at x8 would otherwise pass the gate and quietly halve the lane's bandwidth. Anything else exits **rc 19**, "out of
  band". That code is deliberately not one adertha admits as machine evidence (13/14/17/18): a gen 5 host is a good
  host and must not be excluded from other lanes' searches.
- **A descriptive pinned-H2D probe** (`bench/dq5/dq5_h2d_probe.py`, receipt `h2d.json`). It records:
  - a single synchronized 64 MiB copy;
  - 40 back-to-back 64 MiB copies;
  - a single copy of one layer's 243,793,920 B;
  - nvidia-smi's link generation and width, max, idle and under load;
  - lspci's LnkSta when lspci is present.

  **It never gates.** Two gen 4 5090 hosts have read single/back-to-back efficiencies of 0.64 and 0.87–0.96, so which
  probe "is the link" is itself a reading, and a slow-but-gen-4 host is in band.
- **The offer search:** `vast_pcie = {"min_gen": 4, "min_lanes": 16, "max_gen": 4}`, with no bandwidth floor
  (adertha-agents#177 adds the ceiling) and `preflight_bandwidth: none`.

## The rule

**DQ3's, as registered (`bench/dq3/DQ3-PREREG.md`, Amendments 0–3) and as `bench/dq3/dq3_reduce.py` implements it.**

| verdict | condition |
|---|---|
| VOID | DQ3's VOID conditions |
| FUNCTION_FAIL | S or S0 differ from R bitwise in the deterministic pass (R1 = R2 control) |
| NOISY | R's self-pair is outside [0.97, 1.03] |
| **PROTO_PASS** | all three gates pass: **step** T(S)/T(R) ≤ 1.10; **coverage** 62 + 62 prefetches per steady step, blocking ≤ 2, residency ≤ 2, 0 unscheduled; **capacity** saving ≥ 0.9 × 62 × 243,793,920 B |

DQ3's reducer does not check the link; the runner's rc-19 gate does, before any install. DQ5's verdict is DQ3's reducer
output on the six arm receipts of a box that passed that gate.

## Predictions (DQ1–DQ4's rented-5090 readings only)

- **Per layer:**
  - forward ≈ 15 ms at 2048 tokens (DQ2);
  - one layer's 243.8 MB copy ≈ 8.7 ms at gen 4 x16's ~27.9 GB/s back-to-back (DQ1 run 2's gen 4 host), and up to
    ~16.6 ms at the ~14.7 GB/s single-copy rate one gen 4 host showed.
- **Step time:**
  - in forward, the copy should hide, or be exposed by at most ~1.6 ms a layer, which is at most ~0.1 s over 64 layers;
  - in backward, a layer's recompute plus backward is ≈ 40+ ms, so the copy hides there.
  - Predicted **T(S)/T(R) ∈ [1.00, 1.05]** (DQ3 at gen 5: 1.0023), so **PROTO_PASS**.
- **Coverage:** expected PASS. More forward copies are expected to be in flight at use (`waited`; DQ3: 9–13 of 62), but
  `blocking` should stay 0.
- **Capacity:** the same bytes as DQ3, so 15.08 GB of 15.12. PASS.
- **Descriptive:** S0/R ≈ 1.25–1.4 (DQ3 at gen 5: 1.18), since each synchronous copy is longer.

**Consequences:**
- **PROTO_PASS:** the prototype's speed claim extends to gen 4 x16.
- **A step FAIL** (T(S)/T(R) > 1.10): the overlap does not hide the copy at gen 4. Write up the per-phase `waited`
  counts and the H2D record; a fix is a new registration.
- **A coverage or capacity FAIL with the step inside 1.10:** the same write-up; the speed claim does not extend (all three gates are
  the rule).
- **FUNCTION_FAIL:** a defect independent of the link (DQ3 read bitwise parity at gen 5). Stop: no re-draw until it is found and fixed
  under a new registration.
- **NOISY or VOID:** no reading. One re-draw on another gen 4 x16 machine under this registration (avoiding the first, by an
  admitted exclusion only if its refusal names the host); a second NOISY/VOID closes DQ5 UNTESTED.
  (Added at review, 2026-10-05, before any box.)

## Budget, guard, rehearsal

- **Run:** one RTX 5090 on `vast:verified-secure`, gen 4 x16 by the band above. The guard is 1.0 h (DQ3's), at
  $0.85/h, estimated **≤ $0.85**. This is the standing no-ask tier for a single run under $15 (#564).
- **Pre-launch gate** (A2000, correctness and memory only, never timing). The first rehearsal (`dq5-a2000-rehearse-1`)
  found the width.max weakness above. The gate was added before any 5090 data existed.
  - DQ3's rehearsal shapes A and B through `dq5_run.sh`'s arm path, at the launch commit, must pass
    `bench/dq3/dq3_rehearsal_check.py`: bitwise parity, and the memory direction S, S0 ≤ R − (L−2) layers.
  - On the A2000 itself, which is gen 3 x8, the runner must refuse at **rc 19**.

## Not claimed

- Gen 3 or x8 links.
- Another card.
- A capacity (sequence-length) boundary at gen 4. DQ4's bytes do not depend on the link.
- Which H2D probe is "right".
