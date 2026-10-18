# R2-Mixtral — one full-depth QLoRA step on Mixtral-8x22B-Instruct-v0.1 on one card (DRAFT, Phase 0: registered before any run)

Issue: experts4bit-qlora#1630. The family ladder's first rung past Qwen3.

**Phase 0 is preparation only.** It rents nothing, approves no budget, changes no default and makes no public claim. A box runs
only after an amendment registers it, and only once every gate below has passed. This registration reuses Q235's harness **by
citation, not co-ownership**: it cites Q235's documents and tools at named commits. When one of them changes after it is cited,
this registration amends to the new commit; it never inherits a change silently.

## The milestone

**Two reproducible, real, full-depth QLoRA optimizer steps on one GPU:** all 56 decoder layers, every routed expert, the
checkpoint `mistralai/Mixtral-8x22B-Instruct-v0.1` at revision `cc88a6cc19fbd17d9f1c0ee0b0d70a748dce698d`.

The step, its records and its decision rules are Q235's, cited at `bench/q235/PREREG-q235.md` as of main `13b55990`:
- "The step" (load, data, optimizer, adapter dtype, reproducibility);
- "Decision rules for the eventual box", with rules 3–4 as replaced by Q235 Amendment 2 (two steps; the per-factor LoRA rules);
- the records Q235's milestone lists.

Two things differ from Q235: the subject and checkpoint, and the fidelity statistic and bands (below). Q235's full-state bands
(its Amendment 1) are **not** used. The maintainer's depth ruling of 2026-10-11 voided them for deep boxes.

## Scale

| | Mixtral-8x7B (calibration) | Mixtral-8x22B (subject) |
|---|---|---|
| decoder layers | 32 | 56 |
| hidden size / expert intermediate | 4,096 / 14,336 | 6,144 / 16,384 |
| experts / top-k | 8 / 2 | 8 / 2 |
| vocabulary | 32,000 | 32,768 |
| revision | `eba92302a2861cdc0098cc54bc9f17cb2c47eb61` | `cc88a6cc19fbd17d9f1c0ee0b0d70a748dce698d` |

Nothing measured on 8x7B is inherited by 8x22B except the band's calibration, and the calibration is applied to 8x22B only
through the band test, whose two directions both fail closed. Raw loss is never compared across the two models.

## Provenance

- **Subject:** 59 safetensors shards (281,260,367,720 B) and side files, held in the local cache. The cache's manifest (69 files)
  matches the hub's byte total and revision. Phase 1 re-verifies every file by content against the hub listing at this revision.
- **Calibration model:** 36 files (190,483,875,117 B), fetched 2026-10-11 and verified against the hub's listing (LFS sha256 and
  size; small files by git blob id), 0 mismatches.

## Capacity (planning rows, loggetta-e4b-gnf4, e4b `4bf1ccae`)

Shape A = 2,048 × 1 × 1 and B = 2,048 × 2 × 4, as Q235. Device GiB are fp32/bf16 adapters, experts in pinned host RAM.

| class | A | B |
|---|---|---|
| RTX 5090 | 18.8 / 14.6 | 26.5 / 20.3 |
| H100, RTX PRO 6000 | same (host-resident) | same |
| H200 | also GPU-resident (91.1–103.0) | |

- **Host RAM:** 97.5 GiB pinned (homes rounded to powers of two, plus a 3 GiB baseline). **The registration requires ≥ 128 GB of
  host RAM.**
- **Disk:** the shards are 261.9 GiB. **The registration requires ≥ 350 GB free.**
- **Link floor:** 141.8 GiB host-to-device per micro-batch, about 7.3 s at the 5090's measured 20.75 GB/s.
- **Class to register first:** RTX 5090. It is the only R2 candidate that fits one, and the only class with a launcher policy
  rate.
- **Not modelled:** the offload staging transient, the same gap as Q235's. The $0 shakedown on the A2000 (below) peaked at
  5.5 GB allocated for 4 layers.

## Fidelity: the per-layer update statistic

**Statistic.** For each decoder layer `i`, over real tokens only:

`u_i = ‖(h_i − h_{i−1})_box − (h_i − h_{i−1})_ref‖ / ‖(h_i − h_{i−1})_ref‖`

where `h_i` is layer `i`'s output and the reference is `bench/q235/instrument/layer_stream_ref.py`'s layer-streamed bf16
forward on the same tokens. The reference reads the checkpoint through transformers' own conversion mapping since e4b#1614.
`u_0` is undefined (no stored input) and reported as none, as `q235_compare` does.

**Depth class.** Interior = layers ceil(4n/48) … floor(44n/48) (0-based), Q235 Amendment 3's 30B interior (4 … 44 of 48) scaled
by depth: **3 … 29 on 8x7B** and **5 … 51 on 8x22B**. The layers outside it are reported without a bar. If Amendment 3 registers
a different rule, this registration amends to match, by commit, before any box.

**The band (its form, registered before any number exists).** STRICT, at every interior layer:

`u_i ≤ 2 × U`

where `U` is the **interior maximum of `u_i` on the unplanted 8x7B box**. It is an **8x7B calibration applied to 8x22B**, and
every receipt says so.

**The calibration** (a $0 A2000 run; its numbers land as Amendment 1, before any 8x22B box):
1. **Tokens:** `q235_tokens.py` on the calibration model, 8 rows, truncation 2,048 (built 2026-10-11: 8 × 365, 1,192 real
   tokens).
2. **Reference:** the full-depth layer-streamed bf16 reference (32 layers).
3. **Runs:** `q235_box.py --repeat` (forward-only) unplanted, then each of Q235's plants (`routing`, `epilogue`) at
   L16 = floor(32/2).
4. **Reads:** the `u_i` profile of all three runs, computed twice (`q235_compare`'s update statistic once it lands, and an
   independent script), with the state statistic reported beside it.
5. **The calibration's own separation check:** both 8x7B plants must exceed `2 × U` at L16. If either does not, the band's form is
   void on this family, and a new amendment is needed before any box.

**On the 8x22B box itself (both directions fail closed):**
- **Plants:** the box's own plants at **L28 = floor(56/2)** must each exceed the band at L28. If either does not, the bands are
  VOID.
- **Unplanted:** the unplanted box must pass the band at every interior layer. If it does not, it is a FAIL, with its full `u_i`
  profile reported. That covers 8x22B NF4 noise exceeding 8x7B's.
- **Null forward:** the plants, their placement, the clean null forward they are compared against, and its floor are as Q235
  Amendment 3 registers them, cited by commit before any box. Each plant's L28 output must differ from that null before any band
  is read.
- **Engagement:** a plant that does not change its own layer, or changes an earlier one, makes the box VOID (instrument), as
  e4b#1623's verdict tool records it, before any band is read.

**Reported, not gated:**
- the state statistic (median taken as the upper middle, as `q235_compare` computes it);
- router top-k set agreement and overlap. The recorded top-k is the pre-plant choice, so it cannot see a routing plant at its own
  layer.

## Gates before any box

1. **Q235 has reached L2** and reproduced it.
2. **The maintainer's escalation gate:** SD2, DQ11, P1/P2 prefill and the routing/cache work are measured and adjudicated.
3. **Q235 Amendment 3 is merged** and cited here by commit.
4. **This registration's Amendment 1 is merged:** the 8x7B calibration numbers, `U`, and the separation check's outcome.
5. **The instrument PRs this registration depends on are merged and cited by commit:** the update statistic in `q235_compare`
   (e4b#1624) and the verdict tool with engagement asserts (e4b#1623).
6. **Q235's Phase 0 → 1 gates, applied to this subject:** a tiny-scale rehearsal of the box script, the time-left fit test scaled
   from measured parts, the pre-rental fetch gate over every locked URL (59 shards plus side files and wheels), self-installed
   tools, and no placeholder values.
7. **Spend:** a pinned host named in the ACK (`vast_pin_machine` / `vast_pin_offer`), the launcher's per-run cap, and the owner's
   approval above $15.

## The $0 shakedown already run (2026-10-10, before this registration)

`rehearse.sh`, unchanged at e4b `18c96356`, ran on a 4-layer slice of the subject (`mk_slice.py`; 8 × 365 tokens) on the RTX A2000.
The evidence bundle (records, log and scripts) has sha256 9838a732; Phase 1 commits it beside this registration's receipts.
- **The full path ran clean:**
  - Q235's strict rules all held, and 127 frozen tensors were unchanged after both steps;
  - the fresh-process repeat was exact;
  - the device peak was 5.5 GB allocated (8.13 GB reserved), and host RSS 30.4 GB.
- **The verdict was BANDS_VOID under the old state statistic.** Both plants engaged (layers 2–3 changed), but at L2 the layer's
  own update is 0.43 % of the state.
- **The update statistic separated them:** the box's u at L2 was 0.270, the plants' 0.698 / 0.669.

The Qwen3-30B depth read (48 layers, plants at L24) points the same way: the plants stayed inside the full-state bands (max 0.241 and
0.371), while `u_24` was 0.346 for the box against 1.063 and 8.507 for the plants, and the box's interior maximum was 0.388. The
blind cross-check record (sha256 0903b16d) matched tc-campaign's read to three decimals. Both findings are why this registration
uses `u_i`.

## What this registration cannot say

- **Speed:** one step's elapsed time is a record, not a benchmark.
- **Training dynamics**, or quality after training.
- **Other shapes, models, adapter scopes, or serving.** The paged server's speed knobs for Mixtral are FAM's (PASS at ×0.90, off
  until a speed read).
- **Hardware not run.**

## Amendment 1 (2026-10-11, before any 8x22B box): the calibration, the band's form void on Mixtral, and a depth-local band

**Why.** The calibration registered above ran, and its separation check failed for the routing plant. The registered band's form is
therefore void on this family, as registered. A replacement detector was registered before an out-of-sample read
on new tokens, and that read validated it. Numbers: `bench/fam235/r2-mixtral-a1-numbers.json`.

**The two Mixtral-8x7B reads** (RTX A2000, $0, plants at L16 = ⌊32/2⌋, forward-only `q235_box.py --repeat`):

| read | rows (tokens) | e4b | evidence sha256 | box u_16 | routing u_16 | epilogue u_16 | box interior max |
|---|---|---|---|---|---|---|---|
| calibration | 0–7 (8 × 365, 1,192 real) | `a6c60914` | `f8a6aa9b` | 0.237 | 0.670 | 2.471 | 0.353 (L26) |
| validation | 8–15 (8 × 508, 1,645 real) | `2061cd36` | `d8ab6df7` | 0.220 | 0.678 | 2.435 | 0.311 |

The box script, the reference and the token builder are byte-identical between the two e4b commits. Two implementations of u_i (the
registration's independent script and `q235_compare`'s update statistic, e4b#1624) agree within 5e-6 on every layer of both reads.

**1. The calibration's outcome.** U = 0.3526 (L26), so 2U = 0.7052. The separation check: epilogue 2.471 > 0.7052, but **routing
0.670 < 0.7052: FAILED.** The form `u_i ≤ 2U` is void on this family. The routing plant still reads 2.83× the box's own u_16. One band across depth
is loose at mid-depth: U is set at L26, where the box's u is 1.49× its u_16. (Qwen3's routing plant reads about 1.0 at both depths and
both sizes, which is why the same form held there.)

**2. The replacement: one depth-local band, used both ways.** For 8x22B layer j:

`band_j = 2 × u_cal(m(j))`,  `m(j) = round(32j / 56) = round(4j / 7)`

- **u_cal** is the calibration's unplanted 8x7B box u at that layer (`depth_local.band` in the numbers file).
- **The map is tie-free.** 4j/7 has a fractional part k/7, never ½, so no rounding mode needs defining.
- **The map covers the calibrated interior exactly.** It sends the 8x22B interior 5…51 onto 3…29, hitting every calibrated layer,
  and L28 onto L16. Floor would send L5 to L2, outside the calibrated interior.
- **It is still a cross-size calibration:** an 8x7B profile applied to 8x22B, a different model. Every receipt says so. What makes it
  fail closed is the 8x22B box's own plants, as Q235 Amendment 3's plants do for 30B → 235B.
- **The plants** (routing, epilogue, at L28) must each read u_28 > band_28 = 2 × 0.2372 = **0.4743**. Otherwise the box is VOID.
- **The unplanted box** must read u_j ≤ band_j at every interior layer 5…51. Otherwise it is a FAIL, with its full u profile.
- **One threshold.** At L28 the plants must exceed the same number the box must stay under, so the plants certify the detector that
  gates the box.
- **Not chosen:**
  - A band built from the judged box itself cannot fail that box.
  - A split form (a global fidelity band, plus sensitivity paired to the box's own u) would let a plant pass under the band that gates
    the box. A real defect of the routing plant's size would then pass fidelity.
  - An MoE-block-only statistic is the principled longer-term fix. It is an instrument change, so it is left for later.

**3. The validation read.** Its reading lines were registered at 02:50:07Z UTC, before it ran. The run used the same
model, rows 8–15 (disjoint from the calibration's), the same three runs, and the identity depth map.
- **Engagement:** both plants left L0–15 bit-identical to the box and changed L16.
- **Line F:** the box's u_i ≤ band_i at every interior layer 3…29. The smallest band_i / u_i was 2.006 (L3), with a range of
  2.01–2.37.
- **Line S:** routing 0.678 and epilogue 2.435 both exceed band_16 = 0.474.
- **Verdict: VALIDATED,** applied mechanically. The maintainer re-derived lines F and S from the raw records and confirmed.
- **Reported:** against the old global band, the routing plant (0.678) would VOID again. The paired ratios were 3.08× / 11.04×
  (calibration 2.83× / 10.4×).

**4. Residual risk.** Both cases below fail closed, with no wrong PASS:
- **8x22B NF4 noise beyond 8x7B's at depth:** the unplanted box FAILs. That reads as noise, not as a defect.
- **The 8x22B routing plant diluted below band_28:** the box is VOID and gives no reading.
- **Untested margins:** the validation measured row-to-row variation on the same model, not the cross-size margin. The box's own plants
  and its full profile are the 8x22B measurement.

**5. Gates.**
- **Gate 3:** Q235 Amendment 3 is merged at `d2480ac6`. Its plant placement (⌊n/2⌋), its in-process null with that null's
  1e-3 tolerance, its engagement tolerance and its structural unplant check apply to this box. Its interior rule is this
  registration's. Its 0.775 band is Qwen3's and is not used here.
- **Gate 4:** met by this amendment.
- **Gate 5:** e4b#1624 (`71da462a`) and e4b#1623 (`8ff9b1b4`) are merged. #1623's verdict tool applies Q235's single band, so a
  mechanical verdict for band_j is a Phase 1 deliverable before any box, checked against the numbers file.

**Unchanged:** the statistic, the interior rule, the state statistic and router overlap (reported), and everything else above.

## Amendment 2 (2026-10-16, before any 8x22B box): the host-RAM floor, measured (128 → 134 GB)

**Why.** Q235 Amendment 4 moved the box's four fp32 check copies to the host, and Amendment 5 measured Q235's host floor with a
depth-scaling read. This box runs `q235_box.py` itself, so both apply here. The check copies are anonymous host memory. The
registered ≥ 128 GB (the estimate's pinned homes, a 3 GiB baseline and headroom) does not carry them, and it would admit hosts too
small to finish the step. Numbers: `bench/fam235/r2-memslope.json`.

**The scaling read** (2026-10-16 18:35–18:55 UTC, RTX A2000, $0, e4b `f3f7487b`; evidence sha256 `5bc99c61`). It uses Q235
Amendment 5's method: `q235_box.py --repeat` in a fresh process per depth, on 2-, 4-, 6- and 8-layer `mk_slice.py` cuts of the
subject checkpoint, with `host_memory` read at the load end and fitted against depth by least squares. Values are in GB:

| depth | `RssShmem` | `RssAnon` | `VmHWM` (reported) |
|---|---|---|---|
| 2 | 4.324 | 0.679 | 17.11 |
| 4 | 7.947 | 0.714 | 29.28 |
| 6 | 11.571 | 0.853 | 33.62 |
| 8 | 15.097 | 0.782 | 34.13 |

- **The pinned homes are exact from 2 to 6 layers:** 1.81194 and 1.81187 GB per layer, against the estimate's 1.81194 (94.5 GiB at
  56 layers). The 2–6 fit's largest residual is 44 kB.
- **The 6 → 8 step reads 97.6 MB short.** The host had about 18.1 GB of swap in use. Shared pages swapped out leave RSS, so under
  pressure RSS can undercount, but it cannot overcount. The 2–6 points are therefore the trustworthy ones, and the floor comes
  from their fit, not from all four points. This read recorded no swap fields. Any further read records the process's `VmSwap` and
  the system's `SwapTotal − SwapFree`.
- **The fixed terms are small:** 0.70 GB of shared memory and 0.57 GB of anonymous memory, plus 0.044 GB of anonymous memory per
  layer (2–6 fit). Q235's were 2.23 and 3.78 GB.

**The floor.** It uses Q235 Amendments 4–5's derivation, assuming no heap reuse:
- **Load end at 56 layers:** `RssAnon + RssShmem` from the 2–6 fit, 1.274 + 1.8554 × 56 = 105.18 GB.
- **The four fp32 host check copies:** 4 × 2 × 950,534,144 B, the estimate's bf16 adapters (expert and attention), = 7.604 GB.
- **The held step-0 layer outputs:** 56 × 2,920 × 6,144 × 2 B = 2.009 GB.
- **Step end:** 114.79 GB. Adding Q235's 18.5 GB allowance gives 133.29 GB, so **the registration requires ≥ 134 GB of host RAM**
  (it was ≥ 128 GB).
- **Reported, not gated:** the box's `host_memory` at a step end is expected at `RssAnon + RssShmem` ≤ 114.8 GB. The four-point fit
  would give 112.91 GB (131.41 with the allowance).
- **The device:** Amendment 4's move leaves the copies off the device. The estimate at 8 × 365 positions, bf16 adapters and 56 layers
  is 13.98 GiB, inside the RTX 5090 class.

**Citations by commit.** Q235 Amendment 4 is at `9f6e0088` (the host check copies and the `host_memory` record), and Amendment 5
at `f3f7487b` (the method). The box-registering amendment re-cites Q235 at the commit it builds on, and re-runs `r2_verdict.py`'s
self-test there.

**The harness, decided** (for the box-registering amendment):
- **Pieces:** R2's own locks; thin wrappers that run Q235's Python tools unchanged, with constants set per call (as
  `r2_verdict.py` does); and an R2 copy of the bash runner only.
- **A drift test:** it diffs that runner against `q235_run.sh` at the cited commit. It allows only a declared list of subject lines
  (the class, the floors, the lock, manifest and closure names, the box's shape argument) and fails on any other difference.
- **Q235 stays byte-identical.**
- **Later:** a subject-manifest harness is the refactor for a third subject.

**Unchanged:** everything else above.

## Amendment 3 (2026-10-17, before any 8x22B box): the box, registered

**Why.** Phase 1's $0 work is done:
- the band's mechanical verdict (e4b#1660);
- the harness (#1666, #1670, #1674);
- the host-RAM floor (Amendment 2);
- the A2000 rehearsal of R2's own runner, and its fit (#1673).

This amendment registers the box's class, floors and caps. The numbers are the maintainer's to approve.

**Citations by commit.**
- R2 cites Q235 at `f17f0744`, Amendment 7 (`r2_drift.CITED`). Every Q235 tool R2 runs or seals is read at that commit, and launch
  staging refuses once Q235's tree moves past it (#1674).
- Amendments 6 and 7 run Q235's box in shape B, with the timed MoE pass. R2's box keeps **shape A** (8 rows in one micro-batch, as
  registered) with no timed pass (`--time-moe` is off, and off leaves no trace in the record), as a declared line of R2's runner.
  Amendment 3's plants, null and tolerances, which this registration adopted in Amendment 1, are unchanged for shape A.
  `r2_verdict.py`'s self-test passes at `f17f0744` (Q235's rule 8 is gated on shape-B records).

**The rehearsal.** RTX A2000, 2026-10-16 23:51–23:57 UTC, $0, e4b `a3e73429`. It was `r2_run.sh` in `a2000` mode on a 4-layer slice;
its rehearsal directory is sha256 `5470cfae`.
- Every phase exited 0. Amendment 2's nine STRICT rules held, and the frozen base was unchanged.
- The repeat equals step 0 bit for bit. Both plants changed layer 2 only.
- The device peak was 5.10 GB. Host `RssAnon + RssShmem` at the step end was 9.30 GB, against 9.38 GB for Amendment 2's terms at 4 layers.

**The floors.** The maintainer's decision (A), 2026-10-17T00:00Z.
- H2D 5.8, host copy 5.14 and disk 0.743 GB/s. These are Q235's sealed cold floors, measured on the same RTX A2000 host.
- The leg read the slice it had just fetched warm, at 2.71 GB/s. That is above the host's own O_DIRECT 1.88, so its own rates would
  set floors that refuse it.
- The pull floor is **0.063 GB/s**, the rate at which the checkpoint fits R2's fetch cap / 1.5.

**The caps** (`bench/fam235/r2-fit/`: `r2_fit.py` at #1673's merge commit, re-run at this commit with exit 0):
- **How they are derived.** The rehearsal's box and repeat are slowed to the floors part by part, then × 1.18 (the slow end of the RTX
  5090 per-step spread), × 56/4 and × 1.5, each rounded up to the minute. The fixed phases keep Q235's caps (240 / 120 / 600 / 300 s).

| phase | slowed + spread | scaled | budget | cap |
|---|---|---|---|---|
| box | 170.6 s | 2,389 s | 3,584 s | **3,600 s** |
| repeat | 95.0 s | 1,330 s | 1,995 s | **2,040 s** |
| fetch (500 Mbps) | | 4,715 s | 7,073 s | **7,080 s** |
| install | | 212 s | 318 s | **360 s** |

- **The repeat's cap.** The maintainer's packet said 1,980 s, from a flat +32 s per process. The per-part rule gives 2,040 s because the
  repeat's own load read warm, at 29 GB/s.
- **Wallclock and cost.** The worst-case wallclock is 4.57 h, under the 6 h policy maximum. That is $3.88 at $0.85/h and $9.13 at
  $2.00/h, inside the no-ask tier.

**The registered box** (`r2_run.sh`'s science line, `r2_verdict.REGISTERED`):
- the card: an **NVIDIA GeForce RTX 5090** with at least 32,000 MiB, and exactly one GPU;
- the host: at least **134 GB** of RAM (Amendment 2) and **350 GB** of free disk;
- the estimates at the box's shape (2,920 positions): 13.98 GiB on the device and 141.75 GiB of link per micro-batch. The device
  deviation is reported, not gated.

**The market.** At 2026-10-17T00:10:36Z, 7 verified, rentable RTX 5090 offers had at least 134 GB of RAM and 350 GB of disk. 6 of them
(6 machines) also listed at least 504 Mbps down (the pull floor) and PCIe at 5.8 GB/s or more. They ran $0.670–4.002/h, median
$0.870. The floors leave the lane runnable.

**The gates, as of this amendment.**
- **Met:** gates 3–5, and gate 6's fit and rehearsal.
- **Still before any box:**
  - **Gate 1:** Q235 reaches L2. Its milestone verdict is pending.
  - **Gate 2:** the maintainer's escalation gate.
  - **Gate 6:** the pre-rental fetch gate (`r2_fetch_gate.py`), run within a day of the launch.
  - **Gate 7:** the pinned-host ACK. The CPU rehearsal mode is exercised by the tests up to its refusals only.

**Unchanged:** everything else above.

## Amendment 4 (2026-10-18, before any 8x22B box): the re-cite to Q235 at `9480ed03`

**Why.** #1709 changed `bench/tc1/tc1_arm.py`, which R2's closure seals. R2 seals Q235's files at `r2_drift.CITED`, so it takes a
change only by re-citing. The re-cite takes every Q235 change since `f17f0744` with it:
- Amendments 8 and 9;
- Amendment 7's box refit and read (#1676, #1679, #1680, #1701);
- Amendment 10 (#1709).

`CITED` is now #1709's merge commit, `9480ed03`.

**What reaches R2's box** (shape A, no timed pass):
- **Amendment 8's dX record: taken, and STRICT.** Every box step records each decoder layer's output-gradient norm.
  `step0_dx_finite_nonzero` and `step1_dx_finite_nonzero` fail the record closed (Q235's rules 3–5), whatever the flags. This is
  the one rule this amendment adds to R2.
- **Amendment 8's held-out read: not taken.** R2's tokens phase now also writes Q235's held-out rows (the line is copied as Q235
  wrote it). R2's box gets no `--heldout` (its declared `extra=()` line), so no held-out read runs and rule 10 is not read.
- **Amendment 9's staging record: reported only.** `E4B_OFFLOAD_STATS=1` is set during the step, and no rule reads it.
- **Amendment 7's box refit (13,847 s): not taken.** It times Q235's shape-B box with its timed MoE pass, which R2 does not run.
  R2 keeps Amendment 3's caps.
- **#1709's hooks: off.** `--order-seed` and the HP1 export are off by default. A default arm's receipt keeps its key set and never
  imports `hp1_lib`; `tc1_arm.py`'s self-test asserts both.

**R2's runner copy** (`r2_run.sh`, checked line for line by `r2_drift.py` at `9480ed03`):
- the tokens line is copied as Q235 wrote it;
- Q235's caps comment is now four lines (Amendment 7's refit). R2's caps comment is four lines too, each named by a declared rule.
  Those rules replace Amendment 3's two, which matched Q235's old two lines.

**The moved rows** (`r2.sha256`, resealed with `r2_stage.py --write` at `9480ed03`; each hash read from the file):

| file | sealed at Amendment 3 | sealed now |
|---|---|---|
| `r2_run.sh` (R2's) | `43316f3efbf036ee3f85fc4c20e6d05bf17a62fb786bea3c0870e54f3224aa65` | `c4072bf156f9622237db2cc43f2517ede88f5d77dae146d57e7b87c7a887c0dc` |
| `q235_hostrate.py` | `34ec780ade877e2f7ef16fa2962274b4a30a3b94473e99c6ff0b8fd2e6af67ba` | `cacd222beea37be1476fcca32ef0e2dfed3f29dae51dec6d9bbce7cdb3e6cb27` |
| `q235_box.py` | `50b7b4f9da9f06791c9644618598e2e974f8c5fb9f6f5dcd1a93bc1582fccecb` | `0aa826f6b7be2eab71388f75bbecb913061bf6e1ca5b906d8ce06866fb3053b6` |
| `q235_tokens.py` | `598439b8da9f05d9cec0893ffe9d4565d17f31127f7ec192c24b9d002ea01a3d` | `9590d01a9347eae27f70f03711af773803fdb5118ddb5f9f5515a378605c666b` |
| `tc1_arm.py` | `fbe790d7d6e868c31598c76c7e77afb8c5632f16e61cab85c457854af30bdadc` | `9f43b45b267d3b4fbc0855fd26e8b1c15de7e92832344fe42c74ecba5e2a4939` |

The other eight rows are unchanged.

**Before any box.** Amendment 3's A2000 rehearsal ran R2's box at `f17f0744`. The box now carries the dX probe and the staging
instrumentation, so gate 6's rehearsal is re-run at this cite before any 8x22B box, and its result is posted on #1630. A phase
that no longer fits its cap's 1.5× margin returns as an amendment.

**Unchanged:** everything else above. No rental, budget or default change.
