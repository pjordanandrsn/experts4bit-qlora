# P230: does grouped-nf4-gemm's programmatic dependent launch, as it ships, decode the DEFAULT NF4 server's tokens exactly, and faster, at one request? On one RTX 5090 (registered before any run)

Lane number claimed by `p230/registration`. Issue: #1313. Follows grouped-nf4-gemm K28 (LEVER, grouped-nf4-gemm#451),
P112 (VOID) and P113 (CAP_DEFAULT on SC1's int4 configuration, `bench/p113/RESULTS-p113.md`).

## Why this lane

**What P113 read, and what it could not.** P113 read `GNF4_PDL` on SC1's int4 serving configuration, where 913
switched kernels run, and grouped-nf4-gemm turned it on by default, capped to launches of at most 8 activation rows
(`GNF4_PDL_MAX_ROWS=8`). P113 said nothing about the DEFAULT NF4 server: then it reached only two switched kernels.

**What changed.** K33, grouped-nf4-gemm's single-row NF4 decode GEMV (`_gemv_nf4_bw`), now takes the switch, and it is
the default route at Qwen3-30B-A3B's two expert shapes on parts with 160 or more SMs (`GNF4_GEMV_BW=auto`). So the
default server at one request launches four switched kernels a layer: K33's gate_up and down, `swiglu_rows` and
`combine_rows`, about 192 a decode step. The switch has shipped on this path without ever being read there.

**Where the cap lets it engage.** K33 keys the cap on its (token, slot) rows T, and `swiglu_rows` / `combine_rows` on
their rows R. One request at top-8 gives T = R = 8, inside the cap. Two or more requests give 16 or more, above it.
So PDL engages on the default server at one request only; at 16 requests (W16) it launches nothing with PDL, which
makes W16 a control.

## Subject

The default server: `PagedServeConfig.from_env()` + `build_engine`, graphs on, every serving lever unset (the
runner's `unset` list), resolved as e4b at the launch commit resolves it. Its configuration (max_seqs, buckets,
placement) is recorded; buckets must include 1 and 16, all captured as graphs.
- **Model:** `Qwen/Qwen3-30B-A3B` @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, NF4 arena baked on the box by P39's
  `k8_bake.py`.
- **Stack:** e4b at the launch commit; grouped-nf4-gemm **v0.50.0** (`41ef086e`), the release that carries K39's prmt
  decode (qwen3_moe's served decode) and the switch as shipped (`PDL_DEFAULT` True, `PDL_MAX_ROWS_DEFAULT` 8);
  transformers 5.17.0; torch 2.8.0+cu128 (the image's).

## Arms, workloads and timing

Four arms in a palindrome, **OFF1 DEF1 DEF2 OFF2**, each a fresh process (`bench/p230/p230_box.py`):

| setting | environment |
|---|---|
| OFF | `GNF4_PDL=0` |
| DEF | `GNF4_PDL` and `GNF4_PDL_MAX_ROWS` unset: the shipped default (on, capped at 8 rows) |

**Workloads** are P109's: W1 is row 0 alone, and W16 is 16 distinct 512-token wikitext prompts at once. Each runs 32
and 160 new tokens, one warm pass and 3 timed passes.

**Decode-only timing** (P112 Amendment 2, P113). Every pass records each row's `Request.ttft`; the rows arrive
together. The decode-only time is the wall minus the pass's largest ttft, and p37's slope (P109's `slope`) is taken
over those times. The whole-pass slope is reported beside it.

**Recorded per arm:** every pass's wall, ttfts and token digest, and the last pass's tokens per row; the resolved server
configuration and `graph_status`; `pdl_active` and `pdl_cap`; P112 Amendment 1's launch accounting of the build over
int4_b32's switched kernels and K33 (each kernel's launches, those with PDL, compile launches, compiled variants and
variants with PDL); the same launches **per decode bucket**, each tagged with the bucket being captured (the box wraps
`PagedModelRunner._capture_bucket` for the build only), so the receipt reads where PDL engaged; memory.

## The rule (`bench/p230/p230_reduce.py`, self-tested on 38 cases)

The verdict is the first of these that applies.

1. **VOID.** Any of:
   - an arm is missing or not ok;
   - another e4b or grouped-nf4-gemm commit, or another model revision;
   - different prompts or lengths, or different resolved server configurations, across arms;
   - a void decode-only slope;
   - wrong engagement:
     - every arm reports `graph` for every one of its buckets, and its buckets include 1 and 16;
     - `pdl_active` is off in OFF and on in DEF, and `pdl_cap` is 8 in every arm;
     - every build launches the same number of switched kernels, more than zero, with no more compile launches than
       variants; K33, `swiglu_rows` and `combine_rows` each launch in every build;
     - OFF carries no PDL anywhere. In DEF the cap splits the build (at least one handled launch carries PDL and at
       least one does not), and at least one handled K33 launch carries PDL.
     - **Per bucket, read from the capture:** buckets 1 and 16 both record switched launches in every arm (the control
       is a positive read, never an absence); in DEF, bucket 1 has PDL launches and bucket 16 has none; in OFF, no
       bucket has any. So the W1 graph carries PDL and the W16 control does not, as read, not argued from the code.
2. **FUNCTION_FAIL.** Any arm emits a token different from OFF1's on any row of either workload at either length, or
   an arm's timed reps do not all digest the same. It is read BEFORE the timing rungs: token identity does not depend
   on timing, so noise never hides a broken bitwise contract.
3. **NOISY.** A self-pair (OFF2/OFF1 or DEF2/DEF1, decode tok/s, either workload) falls outside **[0.995, 1.005]**, or
   so does either DEF/OFF pair on W16 (the control: no PDL launch there).
4. **KEEP.** g1 = min(DEF1/OFF1, DEF2/OFF2) on W1 is at least **1.005**.
5. **SLOWER.** h1 = max(DEF1/OFF1, DEF2/OFF2) on W1 is at most **0.995**.
6. **NULL** otherwise.

**Reported:** every pair ratio, g1 and h1, ms per step, the whole-pass slopes, and the launch accounting.

**Why these bars (from variance only).** The bars are set by the instrument's own run-to-run spread, never by an
effect. P113's self-pairs read 0.9988–1.0018 on this timing, P111's 0.997–1.002. A band of **[0.995, 1.005]** admits
that spread with room; a self-pair outside it means the box cannot resolve the bar, and the reading is NOISY. KEEP
asks the SMALLER of the two independent W1 pairs to clear the widest tolerated noise (1.005); SLOWER asks the LARGER to
fall below it (0.995). Anything between is NULL: the switch's W1 effect is inside what this instrument can tell from
zero. W16 must read inside the same band, because the cap launches nothing with PDL there.

## Predictions

Written before any P230 data. They are expectations, not the sizing basis.

| # | prediction |
|---|---|
| Q1 | every arm's engagement holds, including DEF's split and at least one K33 launch with PDL |
| Q2 | every arm's tokens equal OFF1's on every row (the switch is bitwise by its contract) |
| Q3 | the self-pairs and the W16 control read inside [0.995, 1.005] |
| Q4 | g1 lies between 1.000 and 1.015 |
| Q5 | the verdict is NULL (about 50 %), KEEP (about 45 %) or SLOWER (about 5 %) |

## Consequence, registered now

- **KEEP.** The shipped default stands on the NF4 server too. A register row
  `e4b.serve.p230.gnf4-pdl.qwen3-nf4.5090.<date>` carries g1, and the families inventory (`bench/fam/INVENTORY-fam.md`)
  notes the read.
- **NULL.** The default stands (it is bitwise and costs nothing measurable), and the record says the switch buys
  nothing this instrument can resolve at one request on the NF4 server.
- **SLOWER.** No default moves here. An e4b-side opt-out (`GNF4_PDL=0` for the NF4 server) becomes its own registered
  change, citing this read.
- **FUNCTION_FAIL.** The switch's bitwise contract broke on the served path: reported to grouped-nf4-gemm with the
  rows, and e4b sets `GNF4_PDL=0` for the NF4 server in its own PR until it is explained.
- **VOID or NOISY.** No consequence; the one rerun (`p230-5090-2`), or an amendment.

**Scope.** Qwen3-30B-A3B's NF4 store, one RTX 5090, one request and sixteen. Nothing for other families, cards or
batch sizes between 2 and 15.

## The premise (on the card, before anything is fetched; rc 25)

- e4b's `tests/test_decode_graph_buckets.py`: **7 passed**, none skipped.
- From a grouped-nf4-gemm clone at `41ef086e`: `kernel/test_pdl.py`, **26 passed**, none skipped (every switched
  int4_b32 wrapper bitwise on and off, eager and graph-captured; the engagement probe; the row cap), and K33's two PDL
  tests in `kernel/test_nf4_gemv_bw.py` (`test_the_pdl_preamble_precedes_every_global_load`,
  `test_pdl_on_is_bitwise_pdl_off`), **2 passed**.

## Run ids, outcomes and the stop

**Run ids.** `p230-5090-1` is the reading and `p230-5090-2` its one rerun. Every box needs its own id and receipt (the
ledger holds one row per run, so an id is never reused), and a box that does not start uses an id while counting toward
no started slot. The retry ids **`p230-5090-r1` to `p230-5090-r8`** follow any box that did not start, at the same
terms. The ids are labels, not budget: the caps below alone govern, and started boxes are counted by outcome, never by
id.

**The start.** The runner writes `STARTED` once every pre-install check has passed (the refusals and the pre-install
fit), immediately before the first install command. A box is box-started from that marker.

**The classification is mechanical** (`p230_reduce.py --classify --work W`, run on every fetched box; it fails closed,
exit 2, unless W holds exactly one exit-code file and `RUN_ID`, and a refused classify halts the lane until the box's
files are read by hand):

| runner exit | before `STARTED` | after it |
|---|---|---|
| 0 | (cannot occur) | an ANSWER: KEEP, NULL, SLOWER or FUNCTION_FAIL |
| 13 free disk, 15 card / compute capability / VRAM, 16 host RAM, 18 CUDA unusable | **HOST_REFUSED**: no start; counts toward the refusal cap | (cannot occur) |
| 41 the guard does not fit the deadline (a slow boot) | **FIT_REFUSED**: no start, a retry id; two CONSECUTIVE are one strike (bench/common's Guard fit) | (cannot occur) |
| 9 staging, install or pins, 10 torch does not import, 28 the image's torch, 78 run id / env / frozen knobs, 130 a signal, or any other code | **PRE_INSTALL_HALT**: no start, not capped; a kit, image or launch fault, so the lane halts until it is fixed | a started box WITHOUT an answer: a strike |
| 27 (VOID or NOISY) and any other code | (cannot occur) | a started box WITHOUT an answer: a strike |

**NOT_RUN** (the launcher's pre-flight tore down a created instance before the runner ran: no `RUN_ID`, exit file or
marker) is decided from the launcher's receipt, never classified by `--classify`; it counts toward no cap and uses one
id. A pre-create refusal (no instance) uses no id.

**The stop (nonadaptive):**
- The first ANSWER ends the lane.
- **At most 2 box-started runs** (the reading and one rerun).
- **Two strikes end the lane** with no reading. A strike is a started box without an answer, or two consecutive
  FIT_REFUSED boxes.
- **HOST_REFUSED cap:** 4 refusals; then the lane posts the market's state and pauses, and resumes on a later market
  read with no amendment. A second trip (8 in all) ends the lane, UNRUNNABLE.
- No top-up, no change of arms, bars or lengths after data.

## Budget

- **One RTX 5090** (Vast verified/secure), at most USD 1.05/h including a 180 GB disk. No power floor: the read is a
  ratio inside one box, and the power limit is recorded.
- **The needs** (the runner's named constants): the fetch 1,210 s and the bake 550 s (the slowest measured, 806 s and
  362 s, x 1.5), each arm 600 s (P113's), install and premise 450 s, and a 600 s reserve. **The fit is 5,210 s**, and
  with the cross-lane 25 % headroom the pre-install check needs **6,512 s** left.
- **Guard 2.1 h (7,560 s).** The boot slack, the guard less the fit's need, is **1,048 s**: at least the launcher's ssh
  wait (`ssh_ready_s` 420 s) plus a 300 s staging margin, as the test checks, so a slow-ssh host cannot halt the lane at
  the fit.
- **Worst per box:** 2.1 h x USD 1.05 + 66 GB x USD 0.011 = **USD 2.93**, inside the USD 15 per-box rule.
- **Worst per box that does not start:** its time to the refusal at the box rate. A FIT_REFUSED box has used more than
  the 1,048 s slack, about USD 0.31; a NOT_RUN box up to about 600 s, USD 0.18; a HOST_REFUSED box less. Each is priced
  here at USD 0.31.
- **Lane worst:** the ids bound the boxes at 10 (`-1`, `-2`, `r1` to `r8`), at most 2 of them started: 2 x USD 2.93 +
  8 x USD 0.31 = **USD 8.34**.
- **Expected:** about 45 minutes and about USD 1.5 for the reading.

## Launch

- **One exact launch commit:** the merge of this registration, through pod-launch with `--validate` passing at that
  commit, the host pinned, and rent-watch armed as the mini-side alarm. Every pin (staged files, the gnf4 commit, the
  model revision) is checked against that commit: `tests/test_p230.py` checks the staged pins against their sources at
  the commit, and the runner's tripwire checks the installed commits on the box.
- **STOP-4:** the driver refuses a dirty tree, an `E4B_SHA` override, or a staged file that differs from
  `staged-p230.sha256`.

## What was seen before this page (stated, not hidden)

- **P113's records** (`bench/p113/RESULTS-p113.md`): its self-pairs, the basis of the band above, and its readings on
  SC1's int4 configuration.
- **The code census** (grouped-nf4-gemm v0.50.0): the switched launch sites and their row keys, as described under "Why".
- **Locally:** the box's self-test (10 cases) and the reducer's (38), and `tests/test_p230.py`, pass. Triton does not
  install on the development hosts, so the premise's test counts are read from the sources at the pinned commit.

## What this lane cannot say

- Nothing about SC1's int4 configuration (P113's), other families, other cards, or batch sizes between 2 and 15.
- Nothing about e4b's own decode kernels, which the switch does not reach.
- Nothing about uncapped PDL on this server: the arms are the shipped default and off.

## Receipts

Fetched to the run directory's `p230/` and committed to `bench/p230/receipts/<run>/`:
- `arm_{OFF1,DEF1,DEF2,OFF2}.json`, `verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `prompts.json`,
  `work/bake.json`;
- the logs, with `SHA256SUMS` over every file, and `STORE-ONLY.sha256` for the files kept only in the private receipt
  store.

`RESULTS-p230.md` is written from those files.
