# P119 — a descriptive census of the 64-slot server's steps: per-kernel device time of SC2e's decode steps at 16, 32 and 64 rows and of the 512-token prefill, on one RTX 5090 (registered 2026-10-08, before any run)

Issue: experts4bit-qlora#846 (the serving campaign; the owner's no-ask tier for a single run under $15). Lane number
claimed by `prereg/p119` (pushed 2026-10-08T06:28:57Z). Follows SC2e (#1333), P117 (#1343) and the buckets default
(#1346). Shared with the TTFT lane (`CTO/codex-desktop/ttft`): one profiling run serves both, so neither registers
duplicate profiling. Derived from P117 (`bench/p117/PREREG-p117.md`) by named substitutions; descriptive only.

**Amendment 1 (2026-10-08, before any reading).** `p119-prove-1` (HARNESS_ERROR, $0.069) failed in the box: it called
`len()` on `kv_layers()`, which returns the pool's layer count as an int. The box now records that int; the CPU test's
stand-in pool returns an int as the real one does, so the same defect fails on CPU. `staged.sha256` is re-pinned. No
bracket, rule, prediction, guard or budget changes.

## Why this lane

- **SC2e:** 64 slots with buckets up to 64 serve 12 req/s on Qwen3-30B-A3B int4, one RTX 5090. The 64-row decode step
  runs in 18.4 ms, 16.0 ms of it on the device; a 512-token prefill forward takes about 40 ms.
- **The capacity model** on SC2e's own costs (within 0.075 of every measured cell): at 16–20 req/s every miss is TTFT,
  with TPOT p50 about 34 ms. Prefill × 0.5 alone lifts 16 req/s to 0.90 attainment; prefill × 0.5 with decode × 0.7
  puts the ceiling at 20.
- **Nobody has attributed either step by kernel on this stack.** SC1b's census (2026-10-03) read B = 16 on SC1's stack,
  before the folds, K23's lean glue, bulk KV bookkeeping and wide buckets. P107's prefill profile ran before bulk KV
  bookkeeping became the default (#1200); its ~50,000 device-to-device copies per 4,096-token prefill may have been the
  per-layer flush that bulk bookkeeping replaces.

**The question:** on SC2e's served stack, where does each step's device time go, by kernel and by class, at 16, 32 and
64 decode rows and in the 512-token prefill? The answer names the next kernel lanes; it decides nothing by itself.

## Instrument

**The model is SC2e's served stack**, built exactly as P117 built it:
- `PagedServeConfig.from_env()` + `build_engine` with `bench/sc1/sc1_run.sh`'s `SPEEDENV`, `ROUTEENV` and
  `E4B_PAGED_FUSE_QKV=1`, byte for byte;
- built eager with one slot of 768 tokens and no prefill graph: the box's brackets build their own pools;
- the NF4 arena baked on the box by P39's `k8_bake.py`.

**The eager twins.** Each decode bucket's graph is captured from its padded eager step, and the first-chunk prefill
graph from the eager forward: the same code path at the same shapes. That shared route, and the shapes the box
records, are what let the eager steps stand for the served ones; P109's bit-identical outputs (P117's G64 at 64 rows)
establish numerical equivalence, not identical kernels. `torch.profiler` (CUDA activities) attributes the eager steps'
device time per kernel, as P102 and P107 did. The captured kernels stay unprofiled here, and host gaps and graph
launch are SC2e's to read.

**Brackets** (`bench/p119/p119_box.py`; each on a fresh `Fp8PagedKV` and `PagedModelRunner`, device grouping on and
bulk KV bookkeeping as the server runs them; windows from wikitext-2-raw test as P117 takes them, 512 prompt tokens):

| bracket | what runs | profiled |
|---|---|---|
| `d16` | 16 windows, buckets 1–16 | 8 decode steps after 3 warm: one 16-row padded eager step each |
| `d32` | 32 windows, buckets 1–32 | 8 steps: one 32-row step each |
| `d64` | 64 windows, buckets 1–64 | 8 steps: one 64-row step each |
| `d64x4` | 64 windows, buckets 1–16 | 8 steps: four 16-row pieces each (SC2e's s64c) |
| `p512_off` | 512-token first-chunk prefill, `E4B_PAGED_LAST_LOGITS` off | 3 prefills after 1 warm, each its own window, flush included |
| `p512_on` | the same with it on (#1337) | 3 prefills; refused and recorded if the forward has no explicit keyword |
| `head` | the LM head alone on bf16 rows of 1, 16, 64 and 512 | 5 calls each |

**Recorded per bracket:** every device kernel's calls and device ms per step (or per prefill); the classes
(`CLASSES` in the box, frozen now: memcpy D2D, other copies, attention, int4 experts, MoE routing, int4 dense, dense
GEMM, norms and glue, sampling, elementwise, other); D2D copies per step and per layer; the runner's bucket statistics
and graph status; the profiled wall per step, reported and never read as a speed; the LM head's shape, dtype and
module.

## The rule (`bench/p119/p119_reduce.py`, self-tested on 22 cases)

**VOID**, any of:
- another e4b or grouped-nf4-gemm commit, or another model revision;
- on the reading, a stack other than SC2e's: int4 experts on fewer than every MoE layer, no int4 attention, or a fold
  or fused q/k/v missing;
- a decode bracket off its registered split: profiled bucket statistics other than 8 steps × its pieces, rows and
  padding, a bucket other than `eager: capture=False`, or KV bookkeeping other than bulk;
- the prefill OFF bracket missing, refused or run with `last_logits` on;
- any bracket whose profiler saw no device kernel.

A refused prefill ON bracket is reported, not VOID.

**READ** otherwise. The reducer tabulates device ms per step by class; the marginal device ms per decode row by class
from 16 to 32 to 64; the chained step over the one-piece step; D2D copies per layer; the prefill by class; the LM
head at each row count; and prefill ON minus OFF. Floats are summed with `math.fsum`, so the verdict file is
byte-identical on any Python.

## Predictions (written before any data)

Each prediction says what a hit or a miss changes. None gates the verdict.

| # | prediction | if it holds | if it misses |
|---|---|---|---|
| Q1 | at 64 rows the int4 expert class is the largest, ≥ 40 % of device ms | the next decode lane is K19 above 256 routed rows | the largest class is named as the next decode lane instead |
| Q2 | `d64` device ms within ±15 % of SC2e's 16.0 ms device time | the eager twins stand for the served replay | the census reads shares, not absolute times; RESULTS says so |
| Q3 | `d64x4` / `d64` device ms ≥ 1.8 | the chained step repeats the weight reads, as SC2e's 2× wall suggested | the chained step's cost is host-side; the census says what |
| Q4 | the marginal device cost per row from 16 to 64 is ≤ 0.20 ms | SC2e's 0.18–0.21 ms per row is device time | the gap to SC2e's per-row cost is host time |
| Q5 | `p512_off` device ms in [30, 42] | the 40 ms forward is device-bound | the prefill forward is host-bound; the TTFT lane's levers move first |
| Q6 | D2D copies per layer in `p512_off` ≤ 10 | P107's ~131 per layer were the per-layer flush that bulk bookkeeping removed | they remain; the census names the kernels that issue them |
| Q7 | the LM head at 512 rows ≥ 1.5 ms and at 1 row ≤ 0.3 ms; `p512_on` saves ≥ 1 ms per prefill | the TTFT lane's last-logits read is worth registering for speed | its speed case rests on host time or is small |
| Q8 | the box runs ≤ 15 minutes, the int4 repack included | — | the guard was generous |

## Consequence, registered now

**READ** has no default, licence or claim consequence. RESULTS-p119.md tabulates the census and names, per the
predictions above, the next kernel lane for the decode step and for prefill; each is its own registration. The TTFT
lane may cite the prefill brackets as a prior, never as its reading.

**VOID or NO_READING:** no consequence. A rerun is a new attempt, or an amendment if the cause is the harness.

## The proving rental

No local card runs the fp8 paged KV or the bucketed path; `tests/test_p119_box.py` covers the box's bookkeeping on CPU
with the runner and the pool stood in (splits, warm and profiled steps, modes passed through, a refusal recorded, the
head bracket, the kernel table).

**The proof** runs the whole box end to end on `ibm-granite/granite-3.1-3b-a800m-instruct` @
`a02780686e08a03fe0d2679a293b5c74a90efa89` (its NF4 store, no int4 levers, as P117's proof), with 40 rows: the
refusals, install and tripwire, the reducer self-test, the premise (`tests/test_decode_graph_buckets.py`, 7 passed,
none skipped), fetch, bake, every bracket and the reducer. It is **PROVED** iff the lane exits 0 with a verdict
other than VOID or NO_READING. The proof's tables are not a reading.

## Budget and STOP rules

- **Proof:** one RTX 5090 (Vast verified/secure), guard 0.75 h at ≤ $0.75/h, estimate $0.5625.
- **Reading:** one RTX 5090, guard 1.5 h at ≤ $0.75/h, estimate $1.125. Expected about 45 minutes, most of it the
  61 GB fetch (P117's took 28 minutes); the box itself about 10.
- **Lane ceiling:** $3.00, inside the owner's $15 no-ask tier.
- **STOP-1:** the refusals (exit codes) run before any install: dud box (10), card class (15), disk < 150 GB (13), host
  RAM < 60 GiB (16), premise (25).
- **STOP-2:** every time-left check is a per-mode variable that fits its own guard (enforced by
  `tests/test_p119_staged_pin.py`).
- **STOP-3:** a VOID is not retried inside the same launch.
- **STOP-4:** the driver refuses a dirty tree or a staged file that differs from `bench/p119/staged.sha256`.

## What this lane cannot say

- Nothing about speed: the profiled walls are not speeds, and host gaps are SC2e's to read. Eager device times are not
  captured-step times or request TTFT. The eager attribution rests on the shared implementation route and the
  recorded shapes; it says nothing about timing or launch behaviour under capture.
- Nothing about the graphs' private memory pools: no graph is captured here, so they are reported as unavailable. The
  box's `max_mem_gb` is the process peak, a different quantity.
- Nothing about a prefill's cost to the decodes riding its step, or about served TTFT under load: the TTFT lane's
  registration carries both (bus, 2026-10-08T08:06Z).
- Nothing about quality: the TTFT lane reads last-logits quality on its own box.
- Nothing about other models, cards, prompt lengths or chunk sizes.
- Nothing about which change would help most: it names candidates, and each fix is its own registered lane.

## Receipts

Fetched to the run directory's `p119/` and committed to `bench/p119/receipts/<run>/`:
- `box.json`, `verdict.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `bake.json`;
- the logs, and the teardown proof;
- `SHA256SUMS`.

`RESULTS-p119.md` is written from those files.
