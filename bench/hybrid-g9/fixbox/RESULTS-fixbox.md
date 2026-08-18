# Fix-round box run: rows term, pool floor, and the placement mechanism that actually moved the needle

Follow-up to `../box/RESULTS-box-run.md` (the G8-balance / G9 miss, one shared
cause: the DRAM tier achieving ~2% of its calibrated bandwidth). This run
validates the two kernel-level fixes built after that diagnosis, and measures
what each is worth **in the executor** — where the attribution came out
different from the microbenches, in a way worth recording precisely.

## Instrument

Rented serving-class host: AMD EPYC 9755 (Zen 5, AVX-512), RTX 5090,
single NUMA node as seen by the container, **48 online CPUs under a 46.08-core
cgroup quota** (the box is bigger; the quota is what the pool could use).
Calibration (`calib.json`): grouped scatter **268.2 GB/s** (48t), 5090 triad
**1572.2 GB/s**, PCIe link 28.5/28.2 GB/s. Model: Qwen3-30B-A3B, NF4 arena
(60.9 GB, 48 layers x 128 experts, 2,654,208 bytes/expert), VRAM budget
10 GiB -> placement 4045 vram / 2099 dram / 0 nvme in every arm.

## Kernel-level fixes, validated on-box

**1. Per-row dequant hoist (gnf4 `1403466`) — the rows term is dead.**
`rows_curve_zen5.txt`, per-expert fit over T=1..32 rows:

```
t_us = 234.0 + (-1.5) * rows        # slope ~0; was strongly positive pre-fix
T=1: 1956 us   T=8: 1446 us   T=32: 1631 us   (flat, not linear)
```

On AVX-512 hosts, rows are now free: batch concentration no longer raises the
CPU tier's per-expert cost. (Exactness of the hoist: bit-identical by
construction — same mul+add order, weights computed once instead of per row —
and 28/28 on the AVX-512 dev container. pytest was missing on this box, so the
suite was not re-run here; the codepath is ISA-identical.)

**2. Pool wake/join floor (`pool_floor_zen5.txt`) — U-curve confirmed on this
host class.** Grouped call, serving shape (~30 MB touched), cold = first call
after 2 ms idle:

```
workers=32: cold 441-443 us (42.6-42.8 GB/s) | hot 374-381 us (~50 GB/s)
workers=16: cold 719 us | workers=8: cold 1297-1323 us | workers=64+: >1 ms
```

32 workers is the floor on this quota; `pool_start(0)` (= all visible cores)
is the wrong serving default.

## In-executor: three arms, one G9

`g8_balance.py` arms, identical model/arena/budgets/seed, one process each,
5-step average at decode shape (B tokens, T=1, `use_cache=False`):

| arm | pool | solver cost model | B  | dram ms | gpu ms | ratio | uniq dram/step |
|-----|------|-------------------|----|---------|--------|-------|----------------|
| A   | 48w (threads=0) | bandwidth-only | 8 | 92.34 | 19.86 | 0.215 | 179 |
| B   | 32w  | bandwidth-only    | 8  | 104.87  | 18.96  | 0.181 | 179 |
| C   | 32w  | fixed=55us, per_row=2us | 8 | **49.39** | 16.57 | **0.335** | 178 |
| C   | 32w  | fixed=55us, per_row=2us | 16 | **48.73** | 16.43 | **0.337** | 190 |

G9 (full stack: scheduler + paged FP8 KV + hybrid tier, 32w pool + constants,
B=8, 512-token prompts, 64 new tokens): **aggregate 40.5 tok/s**, per-stream
5.1, TTFT p50 17.7 s, `offload_steps=0`. Prior serving-class run (Threadripper
host, pre-fix): 23.9 tok/s — cross-host, so not a controlled before/after.

## Attribution — the surprise worth keeping

**The pool resize alone bought nothing in-executor** (A 92.3 -> B 104.9 ms is
inside this shared box's run-to-run band). **The whole in-executor win came
from the cost model reshaping placement**, through a mechanism I did not
design it for:

- unique DRAM experts touched per step are EQUAL across arms (179 vs 178) —
  same bytes, so the win is not "less DRAM work";
- the greedy solver is a completion-time balancer: it interleaves assignments
  by running bus totals. The constants raise per-expert CPU cost from ~4 us
  (bandwidth term alone) to ~34 us (fixed contributes ~22, per-row ~8), so the
  interleave ratio goes from ~6:1 to ~50:1 gpu:cpu — VRAM capacity exhausts
  ~5 layers earlier in the walk, and the DRAM tier **concentrates**;
- placement repro (deterministic; run locally from `calib.json` + geometry):
  bandwidth-only puts 18-19 DRAM experts in *every* layer 0-36 plus 11 full
  late layers; with-term drains layers 0-31 to 2-3 experts and makes the last
  16 layers all-DRAM;
- the DRAM tier pays a per-call floor (~441 us cold at 32w, x2 calls/layer).
  Scattered: ~96 thin calls/step -> floor model predicts ~105 ms (measured
  104.87). Concentrated: same bytes in ~half the effective calls -> ~52 ms
  (measured 49.39). The model closes both arms to ~6%.

So on AVX-512 hosts the per-row term (which motivated the constants) is moot —
rows are free, `offload_rows` never fired — and the **fixed** term is the
operative half: it encodes "every touched CPU expert costs a call share", and
the balancer responds by giving the CPU tier fewer, fatter calls. Call COUNT,
not worker count, is the floor lever the executor actually has.

## Gate posture

- **G8 balance clause: still a MISS** against the 0.80 bar — best 0.335. But
  the batch trend is fixed: TR run degraded 0.124 (B=8) -> 0.074 (B=16);
  this run holds 0.335 -> 0.337. The clause's failure mode changed from
  "CPU tier collapses with batch" to "CPU tier pays a per-call floor".
- **G9 throughput: still a MISS** (40.5 vs 140 tok/s). Step decomposition at
  B=8: 197 ms/step in the engine vs 113 ms bare forward at the same shape
  (arm C `step_s`) -> ~84 ms/step in attention+engine outside the expert path;
  expert path itself is 49 (dram) + 17 (gpu) ms. The DRAM tier is no longer
  the dominant term.

Ranked next levers: (1) per-job worker-subset wake + call coalescing (attack
the 441 us x calls floor directly); (2) decompose the ~84 ms attention/engine
overhead; (3) batch prefill across sequences — mixed-mode prefill WAS active
in this run (`PagedModelRunner` defaults `gpu_only_prefill=True`), but
`run_prefill` forwards one sequence per chunk (`ids[None]`), so 8 prompts
prefill serially and TTFT p50 = ~4.5 sequential chunk forwards (~3.9 s per
512-token chunk); (4) overlap the two expert buses.

## Loose ends recorded

- pytest absent on the box -> exactness suite not re-run there (covered on the
  AVX-512 dev container; same codepath).
- A-vs-B spread (92 vs 105 ms, same config class) unattributed: shared-tenant
  noise on a quota'd box. Neither arm's pool config explains it; treat +-13%
  as this instrument's band.
- The balance harness only recorded tier COUNTS, which are capacity-clamped
  and identical across arms — the membership shift that explains the result
  had to be reproduced locally from the solver's determinism. Harness patched
  to embed the per-layer DRAM histogram in its JSON.

## Receipts

`calib.json` (host block: quota, flags, THP), `rows_curve.json` +
`rows_curve_zen5.txt`, `pool_floor_zen5.txt`, `g8_A_pool128.json` (name kept
from launch script; actual pool = 48w by quota), `g8_B_pool32.json`,
`g8_C_term.json`, `g9_fixed.json`. Box destroyed after pull; SSH refused and
API shows no instance.
