# P125 — calibrated int4 attention (and the int4 lm_head) on the shipped default's single-stream decode: license, speed, memory (registered 2026-10-09, before any run)

Issue: experts4bit-qlora#1313 (single-stream decode; the owner's no-ask tier for a single run under $15). The lane
number was claimed by `prereg/p125` (pushed 2026-10-09T01:13:32Z). It follows P123 (#1394, #1405, #1411; read #1418).
The maintainer set its shape on the bus between 00:54Z and 01:15Z:
- the arms and the gates per path;
- K8 reported, never gated;
- the calibration on the box;
- the composition test first;
- the memory and slot accounting;
- the tightened bias bound, the window count from Phase D's spread, and the UNDERPOWERED rung;
- a sure-fail mutant as the VOID rung.

## The question

P123 priced the shipped default's one-request step on Qwen3-30B-A3B NF4 at 4.75 ms. Its largest class is the dense bf16
GEMVs at 38 % (1.81 ms): the attention projections stream about 1.81 GB a token and the lm_head about 0.62 GB. The
levers already exist:
- **`E4B_SERVE_ATTN_INT4_CALIB=1`:** GPTQ-style Hessians from C4 at the build, packing the 192 attention projections on
  the int4-b32 grid (0.51 GB);
- **`E4B_SERVE_LMHEAD_INT4_CALIB=1`:** the head too.

Neither has been read on today's default (the B=1 fused stack, the bandwidth GEMV), and **no read ever exercised the
one-row path**. At one row, `Int4Linear` quantises its activations (`quant_x_rows`, then `gemv_int4_b32`). At 2–16 rows
it runs K16 on bf16 activations, and above 16 rows a cached bf16 copy through cuBLAS. K8's licenses (bo6c, P55x) were
teacher-forced multi-row passes, and RTN int4 attention failed K8 on c4val1 by +0.132 ppl.

**Does calibrated int4 attention, alone and with the head, license on the shipped default at the paths one request and
sixteen requests actually serve, and what does it buy in speed, memory and served slots?**

## The premise (the composition test, A2000, 2026-10-09T01:04Z; correctness only)

A tiny random Qwen3-MoE at Qwen3-30B-A3B's per-layer dimensions (2 layers, 16 experts), e4b `478998e5`,
grouped-nf4-gemm 0.43.0, the default server with the fusion knobs unset (eager on the A2000). The calibration ran on the
box from C4. Records: the session's scratchpad `int4comp/comp_{U,F,F2,H}.json`; bus 2026-10-09T01:05Z.
1. **`fuse_qkv` composes with `Int4Linear`.** The census reads 2/2, every fold engages, and the sources are
   `default-allowlisted`. The fused module's packed bytes and scales equal `cat(q, k, v)`.
2. **The fused module is bitwise equal to `cat(q(x), k(x), v(x))`** at 1, 4 and 16 rows: the gemv and K16 paths. At 17
   and 64 rows (the bf16 copy through cuBLAS) it differs by arithmetic order, max |Δ| 0.031.
3. **Tokens:** fused against unfused int4 attention decoded B=1 identically. **B=4 differed, as expected, not gated:**
   the prefill's bf16-copy path changes arithmetic order on random weights with near-tied logits.
4. **Routes:** all three paths engage (`gemv:1`, `k16:4`, `bf16:64`), and the count-only route counter reads them.
5. **The calibration is deterministic.** Two fresh builds gave identical packed digests on every `Int4Linear` and
   identical tokens.
6. **The bf16 copy is resident** after the first prefill, for the attention projections and for the head. The head's
   copy is ~593 MiB, plus a ~2 GiB transient at the first prefill.

**Premise on the card (rc 25):** 39 passed, none skipped:
- P115 Phase D's four tests (19);
- `tests/test_int4_attn.py` and `tests/test_int4_attn_calib.py` (20; 20 passed on the A2000).

## The subject and the arms

Every arm runs **the shipped default server**. It is built by `PagedServeConfig.from_env()` + `build_engine` with every
fusion and decode-GEMV knob unset:
- the B=1 fused stack resolves `auto` on `qwen3_moe` (census 48 / 193 / [48, 48] / 48);
- the bandwidth GEMV is at its default (`bw_prmt32`, no dot-pad);
- 16 slots named; graphs on for speed, eager for quality.

The arms differ only in their int4 lever. The calibration runs on the box exactly as a user's build runs it (32 × 512
tokens of C4 validation shard 0; its knobs unset).

| arm | lever | role |
|---|---|---|
| A | none | the default; writes R |
| B | `E4B_SERVE_ATTN_INT4_CALIB=1` | the subject: 192 projections, 96 `Int4Linear` modules (fused q/k/v and o) |
| C | B + `E4B_SERVE_LMHEAD_INT4_CALIB=1` | the head too (97 modules); licensed only if B is |
| M | `E4B_SERVE_ATTN_INT4=1` | RTN attention: the reported sensitivity check |
| K | B, then every `Int4Linear`'s scales **rolled** one 32-block along K | the sure-fail mutant: the VOID rung |

## The instrument

**Speed (reported, not ruled).** Six processes in palindromic order, A1 B1 C1 C2 B2 A2:
- the default graph server at 16 slots, through P109's W16 and W1 (32 / 160 new tokens, 1 warm and 3 timed passes,
  decode-only slopes);
- `g1_X = min(X1/A1, X2/A2)` at W1, and `g16` the same at W16;
- each record carries the fusion and int4 census, every `Int4Linear`'s digest, grouped-nf4-gemm's dispatch tally and
  memory.

**Memory and slots.** Recorded per speed arm:
- the free device memory before load;
- the peaks after the build, over **the first prefill** (when the bf16 copies are built) and over the runs;
- the bf16 copies resident afterwards;
- **each arm's auto slot count** (`E4B_PAGED_MAX_SEQS=auto`) at 2048 and 4096 tokens a slot, computed by e4b's own
  `choose_max_seqs` on that measured free memory. Its estimate prices `attn_int4` (the grid, the resident bf16 copy and
  the workspaces) but has no head flag, so C is also corrected by hand (the head's int4 grid on top).

The arithmetic, before any data, at 29.5–31.3 GiB free:

| | 2048 tokens a slot | 4096 tokens a slot |
|---|---|---|
| A needs (64 slots / 32 slots) | 25.86 GiB | 26.11 GiB |
| B | +0.58 GiB | +0.58 GiB |
| C | ~+0.16 GiB more | ~+0.16 GiB more |

**Every arm resolves 64 and 32 slots: no slot cost.** C's first-prefill transient (~2 GiB, unpriced by any estimate)
fits the ~3 GiB headroom left after the 1.5 GiB margin.

**Quality (the gate).** Five processes, A first: A writes R's fp32 log-probs per gate, which stay on the box. The server
is built eager, through P115's `p115_quality.measure_phase` (Phase D's instrument at its registered bytes):
- **`t1`:** wikitext at **one window per pass** (`T == 1`: the activation-quantised `gemv_int4_b32` path), **108
  windows** (K: the first 12);
- **`k16`:** wikitext in **16-row pieces** (K16 on bf16 activations), **112 windows** (K: 16);
- **`c4`:** c4val1 in 16-row pieces, 16 windows. It feeds only the reported K8-style ppl.
- Each window is 512 prompt tokens + 128 scored positions.

**A count-only route counter** records every `Int4Linear` call by route and row count over each gate's passes.

## The gates (`p125_reduce.py`, self-tested on 25 cases)

Per gate (`t1`, `k16`) and arm against A:
- **d** per window = the arm's mean continuation NLL − A's. From it: the mean d, its SD, and SE = SD / √n.
- **bound = ln(1 + 0.05 / ppl_A)**, with ppl_A = exp(A's mean NLL over that gate's own windows). This is K8's +0.05 ppl
  budget in nats: about 0.0057 on Phase D's wikitext windows (ppl 8.76), about 3.5× tighter than SANE's 0.02.

The first rung that applies:
1. **FAIL:** mean argmax agreement < 0.95, or mean d − 2 SE > bound (decisive at any power).
2. **UNDERPOWERED:** SE > bound / 3. Not licensed, not failed, and reported with the window count it would have needed.
3. **FAIL:** mean d > bound. One-sided, as K8's calibrated rule: an improvement is neither a failure nor claimed.
4. **PASS.**

**The window counts (the prior, written before any data).** Phase D's `p115d-5090-1`, 12 wikitext windows at
`T == 1`, read:
- a per-window d of mean +0.00541 and **SD 0.01957**;
- ppl_R 8.762, so bound = ln(1 + 0.05 / 8.762) = 0.00569.

SE ≤ bound / 3 = 0.0019 needs (0.01957 / 0.0019)² = **107 windows**. So `t1` reads 108 and `k16` reads 112 (seven
16-row passes). Phase D's spread is the prior, not a promise: an int4 weight change may spread wider, which is the
UNDERPOWERED rung's job.

**Licences:**
- **B is LICENSED** iff `t1` and `k16` both PASS **and** its calibration is deterministic across its three builds
  (speed B1, B2 and quality B: identical digests on every `Int4Linear`).
- **C is LICENSED** iff its own gates PASS, its builds agree, **and B is LICENSED**.
- Otherwise each reads **NOT_LICENSED** (a FAIL) or **UNDERPOWERED**.

**The verdict**, the first rung that applies:
1. **VOID**, if any of these holds:
   - a record is missing or not ok;
   - another e4b or grouped-nf4-gemm commit, or another model revision;
   - not the shipped default: a fusion or GEMV knob set, the census or its sources off the family's default, or the
     build's GEMV dispatch off `bw_prmt32`-without-dot-pad;
   - an int4 census off its arm: projections, modules, or the head's type;
   - a gate's window count, group or text off;
   - a pass's route counts off shape:
     - `t1` wants `gemv` at 1 row only, exactly windows × 127 × modules, with no K16 or wide;
     - `k16` and `c4` want K16 at 16 rows only, exactly passes × 127 × modules, with no gemv or wide;
     - every gate needs a prefill on the bf16 copy;
   - **the sure-fail mutant K not FAILING both gates.**
2. **READ** otherwise. Self-pairs outside [0.97, 1.03] mark the speed NOISY; the licences stand.

**Reported, never ruled:**
- **RTN (M)** against its written prediction: **FAIL at `t1`**. This is the instrument's sensitivity check.
- **The K8-style ppl** (P115 Phase B's in-box method, exp(mean NLL)) on wikitext from `t1` and c4val1 from `c4`. K8
  itself is not run. **Neither method runs the one-row path for c4val1**, and step_decomp's K8 never ran it at all.
- **The speed ratios, the memory peaks and the auto slots per arm (the slot trade), the calibration seconds** (an int4
  arm's build minus A's) and the C4 fetch time.

## Predictions (written before any data)

| | predicted | basis |
|---|---|---|
| g1_B | **1.10–1.30** | 1.81 GB → 0.51 GB of attention, at P123's 1.34 TB/s dense rate ≈ 1.0 ms saved, less the quantise launches (96 a step) |
| g1_C | 1.15–1.40 | the head adds 0.62 → 0.17 GB, ~0.3 ms more |
| g16_B | 1.00–1.08 | K16 at 16 rows on the int4 bytes; the 18.2 ms step is mostly experts (P123: 69 %) |
| g16_C | 1.00–1.10 | the head on K16 at 16 rows |
| B `t1` | PASS, about 50 % | calibrated attention passed K8 only beside int4 experts; the one-row path's activation quantisation is unread |
| B `k16` | PASS, about 65 % | K16 keeps bf16 activations |
| B licensed | about 40 % | |
| C | **NOT_LICENSED**, about 75 % | #373 read the calibrated head at +0.0085 nats in-stack on Qwen3, over the 0.0057 bound |
| M (RTN) `t1` | **FAIL** | its K8 failure; the sensitivity check |
| K | FAIL on both gates | every weight on its neighbour's scale |
| UNDERPOWERED at `t1` | about 25 % | int4 weights may spread wider than Phase D's arithmetic change |
| memory over A, after the first prefill | B +0.4–0.8 GiB; C +0.5–1.0 GiB | the int4 grid and workspaces on top of a resident copy the size of the bf16 weights; C's head grid |
| auto slots | **unchanged**: 64 at 2048, 32 at 4096, every arm | the arithmetic above |
| calibration seconds, Qwen3 | 60–180 s a build (unmeasured) | **measured first on the proof and stated in the read** |
| the verdict | READ about 80 % | |

## The proving rental and the run

**Proof** (`p125-prove-<n>`): the whole box on Granite-3.1-3b-a800m at 8 / 24 tokens and 1 rep, every gate capped at 16
windows (K at 12 at `t1`).
- Granite's default leaves the fusions off, so the int4 arms carry four `Int4Linear` modules a layer (128; 129 with the
  head) and the scalar GEMV.
- It VOIDs unless every int4 arm engages on shape and K FAILs both gates. A VOID fails the proof (rc 27). Its
  licences are not a reading (16 windows are UNDERPOWERED by design).
- **The proof measures the calibration build time and the C4 fetch first.** If they project the reading past its
  guard, an amendment adjusts before the reading.

**Order on the box:**
1. Refusals: CUDA 18, card class 15, disk 13, host RAM 16.
2. Install, then the tripwire:
   - the pins;
   - the default resolving `auto` on `qwen3_moe` and `0` on `granitemoe`;
   - `Int4Linear`'s row thresholds (1, 16), `fuse`, the calibration functions, `gemv_int4_b32` and `quant_x_rows`;
   - the calibration knobs unset;
   - the bandwidth GEMV.
3. The self-tests, then the premise (39).
4. Fetch: the checkpoint, then C4 validation shards 0 and 1, timed. Then the bake and the prompts.
5. The six speed arms, then the five quality arms (A first).
6. The reducer.

**Time (reading):**

| step | time |
|---|---|
| fetch + C4 | ~6 min |
| bake | ~6 min |
| premise | ~2 min |
| speed arms | ~24 min (six builds with calibration) |
| quality A | ~25 min (12 s a `t1` window + R saves) |
| quality B, C, M | ~22 min each |
| quality K | ~6 min |

That is **about 2.25 h** against **guard 2.5 h**. Phase D ran `t1`-style windows at 9.8 s (ON) and 12.3 s (OFF).

## Budget and STOP

- **Proof:** one RTX 5090, **guard 0.75 h**, `--download-gb 7`, about $0.56 at the launcher's policy rate.
- **Reading:** one RTX 5090, **guard 2.5 h**, `--download-gb 61`, about $1.9 at the policy rate. The maintainer approved
  one box at 2.5 h (2026-10-09T01:15Z).
- **Ceiling:** $4.50 for the lane (the proof, the reading and one rerun), hard stop $5.50. Every run sits inside the
  owner's standing no-ask tier for a single run under $15; anything over $15 needs the maintainer lane's approval.
- **STOP:**
  - the refusals above;
  - every time-left check inside its guard (`NEED_FETCH`, `NEED_BAKE`, `NEED_ARM`, `NEED_QUALITY`);
  - no in-launch retry;
  - the driver refuses a dirty tree or a staged file off `bench/p125/staged-p125.sha256`.

**Receipts.** The run directory's `p125/` is committed to `bench/p125/receipts/<run>/`:
- `arm_{A1,B1,C1,C2,B2,A2}.json`, `quality_{A,B,C,M,K}.json`;
- `verdict_p125.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `prompts.json`, `bake.json`;
- `logs/` (added with `git add -f`) and `SHA256SUMS`.

The reference log-probs stay on the box.

## The consequence

- **B LICENSED, g1_B ≥ 1.05, and the speed READ** (every self-pair inside [0.97, 1.03]): a separate PR may propose
  `E4B_SERVE_ATTN_INT4_CALIB` as the default on `qwen3_moe`, family-scoped like the fused stack.
  - **A NOISY speed leaves the licence standing but allows no flip PR** until a speed-only rerun reads it.
  - **The flip PR must name the trades, not only the B=1 speed:**
    - **startup:** the calibration seconds per build, **and a C4 fetch through `datasets` at every startup.** A server
      without network access cannot calibrate, so the flip PR must also carry an offline answer: a pinned attention pack
      installed by fingerprint, or the lever staying opt-in where offline. The maintainer weighs startup against speed;
    - **memory:** the measured rest and first-prefill peaks;
    - **slots:** a flip that costs auto slots at 2048 or 4096 tokens a slot names that trade against the B=1 gain. The
      arithmetic says none;
    - **16 requests:** g16 as measured.
- **C LICENSED** (only with B): the same, for the head.
- **NOT_LICENSED or UNDERPOWERED:** no flip. The next single-request lever is the launch-bound glue (P123: 635 launches
  in 0.88 ms).

## What this does not claim

- No default change here. A flip is its own PR and its own review.
- K8 is not run; the K8-style ppl is reported, not gated.
- No claim about prompts longer than 512 tokens, other families or other cards.
- A2000 numbers are correctness only.
