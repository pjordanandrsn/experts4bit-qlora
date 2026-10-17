# P133 — does `GNF4_GEMV_BW=1` decode the default `serve_paged` server faster at one request on an RTX 4090 (sm_89, 128 SMs), where auto keeps it off? Track B1's read R-c

Envelope: `ENVELOPES-R-abc.md` v2, R-c (the maintainer's ACK of 2026-10-17T16:36:32Z, with its pin rule and run-id
terms). Design: P116's (`bench/p116/PREREG-p116.md`), changed only where this page says so. Harness: `p133_run.sh` (box),
`p133_drive.sh` (controller), `p133_box.py`, `p133_reduce.py`, `p133_tiny.py` (the dry fixture), all at
`staged.sha256`.

## Why now

P116 read K33's decode GEMV on the served W1 step of Qwen3-30B-A3B on an RTX 5090: **g1 = 1.2417**, g16 = 1.0003,
quality within P110's bar (`RESULTS-p116.md`; register row `e4b.serve.p116.gemv-bw.qwen3.5090.2026-10-07`).
grouped-nf4-gemm then made `GNF4_GEMV_BW=auto` the default at Qwen3's two shapes, **on ≥ 160-SM parts only**. An RTX 4090
has 128 SMs, so auto leaves it off there; dot-pad carries the same gate, so the 4090 serves Qwen3's single-row GEMV on
the scalar route. This read tests the gate's untested side.

## The subject and the pins

The shipped default server: graphs on, `max_seqs` auto, buckets auto, all-vram, the fusion knobs at their default,
`GNF4_PDL` at its default (inert below sm_90). Qwen3-30B-A3B @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, its NF4 arena
baked on the box by P39's `k8_bake.py`. Exact pins, each refused if different:

| | pin |
|---|---|
| e4b | `c8d7e629e4c8431fc86970cf7195d8e67f72e22a` (main at #1691), installed by the box; `E4B_PIN` in the runner and driver |
| grouped-nf4-gemm | **0.46.0**, `6b7c2e7474d20c8843814dd46dad6cfc3d94b440` (its `_BW_PLANS` are K33's plans) |
| torch | 2.8.0+cu128, the image's (P116's) |
| transformers / bitsandbytes | 5.17.0 / 0.50.2 |

**Card:** RTX 4090 class, compute capability 8.9 and **128 SMs**, checked before any install (rc 15). A pass licenses
that configuration only.

## The arms

Four fresh processes, ABBA: **B0a B1a B1b B0b**, differing only in `GNF4_GEMV_BW`:
- **B0:** unset, the shipped default (auto). At 128 SMs that is the scalar GEMV. Envelope v2 wrote B0 as `=0`; at 128
  SMs auto engages neither K33 nor dot-pad (the tripwire asserts it), so both are the scalar route, and unset is what
  ships.
- **B1:** `GNF4_GEMV_BW=1`, no plan or decode override: the shipped plan table, `prmt32`.

**Workloads:** P109's, at its registered bytes. W1 (row 0 alone) is the subject. The control is **Wn at the box's own
width**, n = min(`max_seqs`, 16). `max_seqs` auto never resolves below 16 (`choose_max_seqs`'s floor), so the control
is W16 unless the server cannot build, which the fit check catches. 32 and 160 new tokens, 1 warm then 3 timed passes.

**Engagement** (`nf4_grouped.dispatch_counts()`): B1 dispatches only `bw_prmt32`; B0 none to `bw_*` and some to the
scalar GEMV, none to dot-pad.

## The fit check (before any arm)

One B0 process builds the default server and decodes W1 once at 32 tokens (`p133_box.py --mode fit`). If it cannot, the
read is **VOID with no rerun** (exit 31), and no arm runs.

## The quality read

P116's, unchanged: P110's teacher-forced instrument on the default server built eager, one window per pass (T == 1), on
wikitext and c4val1, **24 windows each**, 512 prompt tokens and 128 positions. OFF (B0) scores R, `rep`, `chunk` and
`mutant_scale`; ON (B1) scores against R. Bars: bias_ON ≤ B_floor + 0.01 and spread_ON ≤ 2 × max(S_floor, 0.005) on
both texts; |K8 Δppl| ≤ 0.05 on wikitext. Integrity as P116's, with device grouping on in every pass.

## The rule (`p133_reduce.py`, self-tested on 38 cases)

First rung that applies:
1. **VOID:** the fit check failed (no rerun); a record missing or not ok; another e4b or grouped-nf4-gemm commit or
   model revision; a dry record in a reading or the reverse; a bucket not captured; the arms differ in prompts, lengths,
   fusion census, width or buckets; a void slope; the engagement rule fails on an arm or a quality phase; the quality
   integrity fails; the scale mutant passes the bar.
2. **NOISY:** a self-pair (B0b/B0a, B1b/B1a) outside **[0.985, 1.015] at W1** or **[0.96, 1.04] at the control**.
3. **FUNCTION_FAIL:** B0b ≠ B0a or B1b ≠ B1a on any row, workload or length, or one arm's timed reps differ.
4. **QUALITY_FAIL:** either text fails P110's bias or spread bar, or wikitext's K8.
5. **SLOWER:** g1 = min(B1a/B0a, B1b/B0b) at W1 **< 1.03**, or gW (the same at the control) **< 0.99**.
6. **LICENSED:** otherwise.

## Predictions (written before any data)

| # | prediction |
|---|---|
| Q1 | the fit holds at `max_seqs` 16: P116's 5090 arm held 23,722 MiB (nvidia-smi) against the 4090's 24,564 MiB; serve_paged's own estimate is 22.10 GiB at 16 slots |
| Q2 | engagement exact, every bucket captured, B1 ≠ B0 on some W1 rows, self-pairs bitwise |
| Q3 | **g1 ∈ [1.05, 1.30]**; the plans were picked on sm_120, so the low end allows a partial win |
| Q4 | **gW ∈ [0.99, 1.01]** at W16 |
| Q5 | quality within the bar, as on the 5090 |
| Q6 | LICENSED, about 60 % |

## Consequence, registered now

- **LICENSED:** a grouped-nf4-gemm PR may extend auto's engagement to sm_89 parts with 128 SMs at Qwen3's two shapes,
  with its own tests and release. Nothing else is licensed: no other SM count, card or family.
- **SLOWER:** the gate stays. g1 < 1.03 reads "needs a plan sweep on sm_89", not a gate change.
- **QUALITY_FAIL / FUNCTION_FAIL:** the gate stays; the defect is examined first.
- **NOISY or VOID:** one rerun (`p133-4090-2`), except a fit VOID, which ends the lane.

## The premise, the proof and the dry run

**Premise**, before anything is fetched (rc 25). Each suite must give rc 0, exactly the registered pass count, and
exactly the registered skipped ids, read from its junit report (`p133_box.PREMISE`), so a new skip is a failure:
- grouped-nf4-gemm's `kernel/test_nf4_gemv_bw.py` at the pin: **33 passed**. Skipped, for "griddepcontrol needs sm_90+
  (the switch is inert below)": `test_nf4_gemv_bw::test_the_pdl_preamble_precedes_every_global_load` and
  `test_nf4_gemv_bw::test_pdl_on_is_bitwise_pdl_off`.
- `tests/test_decode_graph_buckets.py`, `tests/test_kv_step_select.py`, `tests/test_gemv_bw_served_gpu.py`: **18 passed**,
  none skipped. (On sm_86 only, their five fp8-KV cases skip for "the fp8 paged KV needs native e4m3 (sm_89+)": 13 passed.)

**Proof** (`p133-4090-prove-1`, `P133_PROVE=1`): the whole box on Granite-3.1-3b-a800m on a 4090, at 8 / 24 tokens, 1
rep, 12 windows × 32 positions. A VOID fails the proof (rc 27). It is not a reading.

**Dry run** (`P133_DRY=1`, run ids `p133-dry-<n>`, $0): the driver and the whole box on `p133_tiny.py`'s fixture (a
2-layer random Qwen3-MoE, Qwen3's tokenizer) on any CUDA card, 4 windows per text, the reducer in its dry mode. The box
is reached through `P133_BOX_EXEC` and tar instead of ssh, scp and rsync. Evidence: below, under "What was seen".

**First executed on the paid proof.** The dry run's card is the RTX A2000 (sm_86), where the default server builds
eager. So these run for the first time in this lane on the proof:
- **the quality phase, OFF and ON.** P116's instrument (`p110_box.paged_pass`, at P116's bytes) decodes through the
  bucketed path, and binding a bucket needs the fused fp8 KV append (sm_89+). On sm_86 it stops at the first window.
  It ran on the 5090 in P116 at these bytes, and the premise's fp8 cases check the append on the card first;
- **the reducer past its first VOID rung on real records.** In the dry run it read the real fit and speed records and
  VOIDed on the missing quality records; its later rungs are covered by its self-test;
- **decode-graph capture and replay** of the arms (sm_86 builds no graphs);
- **device grouping** at T > 1 (sm_86 groups on the host);
- **the fp8 paged KV and its fused append**, and the five premise cases that cover them;
- **the card and configuration refusal** (the RTX 4090 class and `cc=8.9 sms=128`, rc 15), which DRY skips;
- the ssh, scp and rsync transport (unchanged from P116's driver).

## Budget, host and STOP

- **Host, by the ACK's rule:** Vast verified-secure, reliability2 ≥ 0.98, direct SSH, ≤ $0.011/GB download, one RTX 4090
  at ≤ $0.49/h, RAM ≥ 60 GiB, disk ≥ 150 GB. The pin is the cheapest passing offer, held 40 s (three GETs 20 s apart),
  posted on this PR and the bus, then launched within the minute with `--validate`. The re-quoted worst must not exceed
  this page's.
- **Run ids:** `p133-4090-prove-1`, `p133-4090-1`, and one rerun `p133-4090-2`. The driver refuses any other. One
  instance per attempt, at most three create attempts per run id.
- **Proof:** guard 0.75 h, worst **$1.04**. **Reading:** guard 1.25 h, `--download-gb 61.1` ($0.67), expected about
  1.0 h = **$1.16**, worst **$1.28**. Proof + reading + one rerun = $3.61. **Lane ceiling $3.75, hard stop $4.50.**
- **STOP-1:** refusals before any install: CUDA unusable 18, dud box 10, torch not the pin 28, card or configuration 15,
  disk 13, host RAM 16; premise 25.
- **STOP-2:** every time-left check fits inside its guard, enforced by `tests/test_p133_staged_pin.py`.
- **STOP-3:** a VOID or NOISY reading is not retried inside the same launch. A fit VOID is not retried at all.
- **STOP-4:** the driver refuses an undeclared run id, an e4b other than the pin, or a staged file that differs from
  `staged.sha256`.
- **STOP-5:** quality does not start when a speed arm failed; the reducer VOIDs.
- The ACK lapses on a halt, or at 2026-10-20T16:40Z.

## What was seen before this page

- **No P133 data exists.** No served run has set `GNF4_GEMV_BW=1` on a 128-SM part.
- **On the RTX A2000** (sm_86), a probe before the harness: the default server builds and decodes the tiny fixture
  eager, with host grouping and `max_seqs` auto at 64.
- **The dry run, `p133-dry-2`** (2026-10-17, 17:57:00Z–18:01:10Z, the RTX A2000 in a container reached through
  `P133_BOX_EXEC`; driver rc 0, 36 entries fetched). It ran at this page's `staged.sha256` and this page's driver: the
  rerun after the driver's required-variable loop was made literal (the cross-lane known-hosts contract, CI on #1695).
  `p133-dry-1` (17:10:12Z–17:15:23Z, the earlier driver) read the same on every line below.
  - the box installed and the tripwire confirmed e4b 0.54.0 @ `c8d7e629`, grouped-nf4-gemm 0.46.0 @ `6b7c2e74`, torch
    2.8.0+cu128, triton 3.4.0, transformers 5.17.0 and bitsandbytes 0.50.2; all four self-tests passed;
  - premise: the kernel contract **33 passed, 2 skipped** (the two PDL ids), and the served route **13 passed, 5
    skipped** (the five fp8-KV ids, sm_86's list), each matched against `PREMISE`;
  - fixture, bake and prompts; the fit check passed (`max_seqs` auto 64, buckets 1–64, eager, control W16);
  - the four arms ran (rc 0). Engagement was exact: B0 3,140 scalar dispatches and none to `bw_*`, B1 3,140 `bw_prmt32`
    and nothing else;
  - quality OFF stopped at its first window (the fused-append refusal above), so ON did not run;
  - the reducer (dry mode) returned **VOID**, "record(s) missing or not ok: quality_off, quality_on".
  The same speed records, reduced with stand-in quality records, pass every speed check and read NOISY (W1 self-pairs
  0.86 and 0.84 for a 2-layer model decoding eager on a shared card). That shows the records parse; it is not a reading.

## What this lane cannot say

- Nothing about other SM counts, sm_89 parts other than this configuration, other families, or T > 1.
- Nothing about plans tuned for sm_89: B1 runs the shipped table.

## Receipts

`bench/p133/receipts/<run>/`: `fit.json`, `arm_{B0a,B1a,B1b,B0b}.json`, `quality_{off,on}.json`, `verdict.json`,
`summary.txt`, `forensics.txt`, `versions.txt`, `prompts.json`, `logs/` (with the premise junit reports),
`work/bake.json` and `SHA256SUMS`. The checkpoint, the arena, the reference log-probs and the gnf4 clone stay on the box.

## Amendment 1 (2026-10-17, after `p133-4090-prove-1` VOIDed on a defect in the reducer)

**What happened.** `p133-4090-prove-1` ($0.17, Vast contract 56454024 on machine 51612) ran every registered box step:
- refusals passed (`cc=8.9 sms=128`, torch 2.8.0+cu128); the tripwire and self-tests passed;
- the premise read 33 + 2 and 18 + 0, by id;
- the fit check passed (`max_seqs` auto 64, buckets 1–64, graphs on);
- the four arms ran with every bucket captured and exact engagement;
- both quality phases ran (rc 0).

The reducer VOIDed it (lane exit 27). The quality pass integrity read the server's buckets from the record (1–64 here)
instead of the buckets P110's instrument enables, `p110_box.BUCKETS = (1, 2, 4, 8, 16)`.

**The deviation, plainly:** the reducer derived the buckets from the record; this amendment restores P116's rule.

**The correction.**
- `INSTRUMENT_BUCKETS` is read from `p110_box.py`'s own source (`BUCKETS`), staged beside the reducer, never hard-coded.
- No rule, bar, prediction, size or budget changes.
- The self-test grows to 41 cases. They include the proof's shape (server buckets 1–64, passes at 1–16), a pass whose
  bucket captured, and a pass that misses bucket 8.
- `tests/test_p133_staged_pin.py` reruns the pre-amendment reducer (#1695's merge) on that shape: it VOIDs on the pass
  `graph_status` alone, the amended reducer passes it, and the bucket-8 mutant still VOIDs.

**The proof (the maintainer's ruling, 2026-10-17T22:06Z).** The re-reduction stands as the proof, under this page's own
criterion: a VOID fails the proof, and NOISY at the proof's lengths is not a failure. No `p133-4090-prove-2` is declared.
The verdict of record comes from the MERGED reducer:
1. after this amendment merges, `prove-1`'s committed records are re-reduced at the merge commit (`--proof`);
2. the verdict must be byte-identical to the re-reduction run before the merge (sha256
   `d07c19db63c122794900c00efb0eba2eccfdc200443ba675edea93b45e61ac6f`, NOISY);
3. it is posted with the reducer's merge sha on #1695 and the bus;
4. then the reading `p133-4090-1` may launch under the pin rule.

The B1-against-B0 comparison and the quality deltas stay unread until the reading.

**Disclosed:** that pre-merge re-reduction was run before the maintainer's terms were read. Its reasons, the four self-pair
ratios at the proof's lengths, were seen. They are noise diagnostics, per the ruling.

**Spend:** $0.17 of the $3.75 ceiling.
