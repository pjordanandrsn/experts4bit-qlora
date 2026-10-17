# P134 — does `GNF4_GEMV_BW=1` decode the default `serve_paged` server faster at one request on Granite-3.1-3b-a800m and OLMoE-1B-7B, on an RTX 5090, where auto engages only at Qwen3's shapes? Track B1's read R-a

Envelope: `ENVELOPES-R-abc.md` v2, R-a, under the maintainer's ACK of 2026-10-17T16:36:32Z with its pin rule and run-id
terms. Design: P133's (`bench/p133/PREREG-p133.md`), changed only where this page says so. Harness: `p134_run.sh` (box),
`p134_drive.sh` (controller), `p134_box.py` and `p134_reduce.py`, plus P133's dry fixture `p133_tiny.py`, all at
`staged.sha256`.

## Why now

K33 selected plans for Granite's and OLMoE's expert shapes. Against the scalar GEMV, its kernel took Granite from 1.644 to
0.385 ms per decoded token (32 layers) and OLMoE from 1.618 to 0.391 ms (16 layers). grouped-nf4-gemm 0.46.0 ships those
plans in `_BW_PLANS`. But `GNF4_GEMV_BW=auto` engages only at the exact shapes P116 read, Qwen3-30B-A3B's two. So both
families decode on the scalar GEMV today, even on a 5090.

## The subject and the pins

The shipped default server, as P133: graphs on, `max_seqs` auto, buckets auto, all-vram, fusion knobs at their default,
`GNF4_PDL` at its default. Pins (each refused if different) are P133's:
- e4b `c8d7e629e4c8431fc86970cf7195d8e67f72e22a`;
- grouped-nf4-gemm **0.46.0** at `6b7c2e7474d20c8843814dd46dad6cfc3d94b440`;
- torch 2.8.0+cu128, transformers 5.17.0, bitsandbytes 0.50.2.

**Two families, one after the other on one box**, each with its own fetch, bake, prompts, fit check, four arms, two
quality phases and verdict:

| family | checkpoint | gate_up / down (N, K) | layers |
|---|---|---|---|
| `granite` | `ibm-granite/granite-3.1-3b-a800m-instruct` @ `a02780686e08a03fe0d2679a293b5c74a90efa89` (P94's pin) | (1024, 1536) / (1536, 512) | 32 |
| `olmoe` | `allenai/OLMoE-1B-7B-0924-Instruct` @ `7f1c97f440f06ce36705e4f2b843edb5925f4498` (P99's served pin) | (2048, 2048) / (2048, 1024) | 16 |

**Card:** RTX 5090 class, compute capability 12.0 and **≥ 160 SMs**, checked before any install (rc 15).

## The arms

Four fresh processes per family, ABBA (B0a B1a B1b B0b), differing only in `GNF4_GEMV_BW`:
- **B0:** unset, the shipped default. At both families' shapes that is the scalar GEMV on any part. Neither family is in
  `_BW_SHAPES` or in dot-pad's table, which the tripwire asserts. Envelope v2 wrote B0 as `=0`, which is the same route
  here; unset is what ships.
- **B1:** `GNF4_GEMV_BW=1`, no plan or decode override: the shipped plans. Granite runs 16/512/8/1 and 16/256/8/1;
  OLMoE runs 16/1024/4/1 and 16/256/4/1 (BLOCK_N, KC, warps, split-K).

Workloads, the control (W16 at the box's own width), engagement (B1 only `bw_prmt32`; B0 only the scalar GEMV) and the
fit check before the arms are P133's, per family. A family whose fit fails is VOID with no rerun; the other family still
runs, and the run exits 31.

## The quality read

P133's (P116's instrument at one window per pass, T == 1), per family: wikitext and c4val1, **24 windows each**, 512
prompt tokens and 128 positions. The bars are P110's: bias_ON ≤ B_floor + 0.01 and spread_ON ≤ 2 × max(S_floor, 0.005)
on both texts, and |K8 Δppl| ≤ 0.05 on wikitext.

## The rule (`p134_reduce.py --family F`, self-tested on 31 cases)

P133's rungs, applied **per family**: VOID, NOISY (W1 self-pairs outside **[0.985, 1.015]**, control outside
**[0.96, 1.04]**), FUNCTION_FAIL, QUALITY_FAIL, SLOWER (g1 **< 1.03** or gW **< 0.99**), else **LICENSED**. A family's
records run under another family's name, or a dry record in a reading, read VOID.

## Predictions (written before any data)

| # | prediction |
|---|---|
| Q1 | both families fit at `max_seqs` 64 (3B and 7B NF4 on 32 GB) |
| Q2 | engagement exact; every bucket captured; B1 ≠ B0 on some W1 rows; self-pairs bitwise |
| Q3 | **Granite g1 ∈ [1.17, 1.37]**: S/(S − t × 1.259 ms) for t in [0.55, 1.0] and P123's 4.67 ms W1 step |
| Q4 | **OLMoE g1 ∈ [1.10, 1.70], provisional:** no W1 step on today's stack is on record. The proof computes the band from its own B0 W1 step by Q3's formula (Δ = 1.227 ms), and that band is posted on this PR and the bus **before the reading**; it replaces this one |
| Q5 | gW ∈ [0.99, 1.01] for both |
| Q6 | quality within the bar for both |

## Consequence, registered now

- **LICENSED (per family):** a grouped-nf4-gemm PR may add that family's two (N, K) shapes to `_BW_SHAPES` (auto, ≥ 160
  SMs), with its own tests and release. It licenses **shapes, not families**: Granite's two shapes also cover the
  committed Granite-4.0-h fixture (not served today); OLMoE's are OLMoE's alone among the fixtures. Under auto's
  ≥ 160-SM gate a licensed shape also engages on unmeasured parts above 160 SMs, such as the RTX PRO 6000 (188 SMs):
  P116's precedent for Qwen3's shapes.
- **SLOWER:** that family's shapes stay out of auto.
- **QUALITY_FAIL / FUNCTION_FAIL:** they stay out; the defect is examined first.
- **NOISY or VOID:** one rerun (`p134-5090-2`) for the whole box, except a fit VOID, which ends that family.

## The premise, the proof and the dry run

**Premise**, before anything is fetched (rc 25), read from junit against `p134_box.PREMISE`:
- on the 5090: the kernel contract **35 passed, none skipped** (PDL runs on sm_120);
- the served route **18 passed, none skipped**.

**Proof** (`p134-5090-prove-1`, `P134_PROVE=1`): the whole box on **both families**, at 8 / 24 tokens, 1 rep and 12
windows × 32 positions. A VOID for either family fails the proof (rc 27). The reducer reports each family's
`proof_band`, from its own B0 W1 step, before the quality phase can void it. It is not a reading.

**Dry run** (`P134_DRY=1`, run ids `p134-dry-<n>`, $0): the driver and the whole box on two tiny families (`tinya`,
`tinyb`), each built by `p133_tiny.py`, on any CUDA card. The family loop, staging, fetch and per-family reductions all
run. Evidence is below.

**First executed on the paid proof:**
- the quality phase (on sm_86 it stops at the fused fp8 KV append, as P133's did);
- the reducer past VOID on real quality records;
- Granite's and OLMoE's checkpoints themselves on this harness;
- the proof band on real steps;
- decode-graph capture;
- device grouping;
- the fp8 KV append;
- the card and configuration refusal;
- the ssh, scp and rsync transport.

## Budget, host and STOP

- **Host, by the ACK's rule:** Vast verified-secure, reliability2 ≥ 0.98, direct SSH, ≤ $0.011/GB download, one RTX 5090
  at ≤ $0.85/h, RAM ≥ 60 GiB, disk ≥ 100 GB. The pin is the cheapest passing offer, held 40 s (three GETs 20 s apart),
  posted on this PR and the bus, then launched within the minute with `--validate`. The re-quoted worst must not exceed
  this page's.
- **Run ids:** `p134-5090-prove-1`, `p134-5090-1`, and one rerun `p134-5090-2`. The driver refuses any other. One
  instance per attempt, at most three create attempts per run id.
- **Proof:** guard 0.75 h, worst **$0.86**.
- **Reading:** guard 2.0 h, `--download-gb 20.4` ($0.22). Expected about 1.4 h = **$1.42**, worst **$1.93**.
- Proof + reading + one rerun = $4.72. **Lane ceiling $5.00, hard stop $6.00.**
- **STOP-1:** refusals before any install: CUDA unusable 18, dud box 10, torch not the pin 28, card or configuration 15,
  disk 13, host RAM 16; premise 25.
- **STOP-2:** every time-left check fits inside its guard (`tests/test_p134_staged_pin.py`).
- **STOP-3:** a VOID or NOISY reading is not retried inside the same launch.
- **STOP-4:** the driver refuses an undeclared run id, an e4b other than the pin, or a staged file that differs from
  `staged.sha256`.
- **STOP-5:** a family's quality phase does not start when one of its speed arms failed.
- The ACK lapses on a halt, or at 2026-10-20T16:40Z.

## What was seen before this page

- **No P134 data exists.** No served run has set `GNF4_GEMV_BW=1` on Granite or OLMoE.
- **The dry run, `p134-dry-2`** (2026-10-17, 18:01:10Z–18:07:57Z, the RTX A2000 in a container reached through
  `P134_BOX_EXEC`; driver rc 0, 43 entries fetched, including both `work/<family>/bake.json`). It ran at this page's
  `staged.sha256` and driver:
  - the tripwire confirmed P133's pins; all four self-tests passed;
  - premise: `kernel_dry` 33 passed + 2 skipped and `served_dry` 13 + 5, each matched by id;
  - for **each** of `tinya` and `tinyb`: the fixture, the bake, the prompts, and a fit check that passed (`max_seqs` 64,
    control W16). The four arms ran with exact engagement (B0 3,140 scalar dispatches; B1 3,140 `bw_prmt32`);
  - quality OFF stopped at window 0 (sm_86), and each family's reducer returned VOID ("quality_off, quality_on"
    missing).
  `p134-dry-1` (17:49Z) ran the same box to the same records, but its driver file was edited while it ran, so its fetch
  is not evidence. The run was repeated.

## What this lane cannot say

- Nothing about T > 1, other families or shapes, parts with fewer than 160 SMs, or plans other than the shipped ones.

## Receipts

`bench/p134/receipts/<run>/` holds, per family F:
- `fit_F.json`, `arm_{B0a,B1a,B1b,B0b}_F.json`, `quality_{off,on}_F.json`, `verdict_F.json`, `prompts_F.json`, `work/F/bake.json`;
- shared: `summary.txt`, `forensics.txt`, `versions.txt`, `logs/` (with the premise junit reports) and `SHA256SUMS`.

The checkpoints, arenas, reference log-probs and the gnf4 clone stay on the box.

## Amendment 1 (2026-10-17, before any box)

`p134_reduce.py` inherited P133's quality pass-integrity defect: each pass's `graph_status` was checked over the
record's server buckets, not over P110's instrument buckets (P133 Amendment 1). It is corrected the same way:
- `INSTRUMENT_BUCKETS = (1, 2, 4, 8, 16)`, P116's rule;
- the self-test grows to 33 cases;
- `tests/test_p134_staged_pin.py` pins the constant to `p110_box.BUCKETS`.

No rule, bar, prediction, size or budget changes.
