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
