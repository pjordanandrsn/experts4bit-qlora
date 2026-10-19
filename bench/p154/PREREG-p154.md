# P154: the W16 NF4 expert byte floor of Qwen3-30B-A3B's served step, on the replay's own route (REPORTING)

**Status:** registered before any data. This is a REPORTING instrument for stream (B) of e4b#1313. It prices the
headroom of the expert GEMM P152 named, and names no lever: a lever is its own later registration. It changes no
default, no gate and no speed claim. The maintainer reserved the lane number on 2026-10-19.

## Why

[P152](../p152/RESULTS-p152.md) read Qwen3-30B-A3B's served step on one RTX 5090. At 16 requests the grouped NF4 small-M
expert GEMM (`_gemm_nf4_grouped_smallm`) takes **71.2 %** of the unprofiled 16.497 ms step: gate_up 8.3951 ms and down
3.3530 ms. Its expert byte floor read **UNREAD**. The registration assumed int4 experts through K19, but the served
store is NF4. Its eager routing pass, with `DEVICE_GROUPING` off, host-grouped the 16-row expert calls instead of taking
the replay's device-grouped K25 route. The route check recorded both faults, as designed.

Context arithmetic, outside P152's rule, put gate_up at about 38 % of the copy rate and down at about 48 %. That would
be up to about 7 ms of headroom at W16, against about 0.3 ms for the W1 GEMV candidate. P154 reads the floor properly,
before any kernel work is built against an assumed one.

## The instrument

P152's runner (`bench/p152/p152_run.sh`) cut to its routing pass, by counted substitutions:
- **Family and server:** Qwen3-30B-A3B @ `ad44e777`, the shipped default at 16 slots, every knob unset except one:
  `E4B_INT8_HEAD_BW=0`. e4b#1838 made the int8 output head the qwen3_moe default after P152 ran. A head that rounds the
  logits changes tokens, and so the routing; P152's GEMM times were recorded on the bf16 head, so the routing pass
  runs on P152's server. It records the head, and an int8 head module is a fault.
- **Pins:** e4b at this registration's merge commit; grouped-nf4-gemm **v0.47.1** at
  `38b308887688ea5bda7dd449bd3cc3dc1426b995`, P152's pin, kept on purpose although e4b CI has since moved to v0.48.0: it
  is the K25 build whose GEMM times P152 recorded, so the route and the store are read on the kernel that was timed;
  transformers 5.17.0, torch 2.8.0+cu128.
- **Kept from P152:** the refusals (CUDA, card class, disk, host RAM), the **575 W power floor** before any install (rc
  17; unreadable, 15), the install and tripwire, the arena bake, P109's prompts and the GPU state at the routing pass's
  start and end.
- **Dropped:** the speed arms, Nsight and the captures. P152's GEMM times are compared **as recorded**, from its
  committed verdict (`bench/p152/receipts/p152-5090-2/verdict_p152.json`, staged and sha-pinned); nothing is re-timed.

**Added:**
1. **The premise on this card**, before the fetch: K25's row-exact GPU tests (`tests/test_k25_row_exact_gpu.py`), all 4
   passing, none skipped; else a refusal (rc 25). The tripwire also asserts K25 in the kernel package, K25's mode `auto`
   by default, and the library's grouping rule at 16 NF4 rows: host-grouped with `DEVICE_GROUPING` off, device-grouped
   with it on.
2. **The routing pass on the replay's route** (`p154_route.py`): P152's bracketed W16 window (the same prompts, ramp, 32
   skipped and 64 recorded full-decode steps), run eager so a forward hook on every router fires. After the engine
   builds, the pass sets `hot_residency.DEVICE_GROUPING` on itself, as the server does under graphs. It records:
   - **the route:** `_collapsed_grouping(16, store)`, which must be (singleton, device) = (False, True), and K25's mode;
   - **K25's calls inside the window**, counted by a wrapper on the kernel module: exactly 2 a MoE layer a step (gate_up
     and down), 6,144 in all;
   - **the library's own route record across the same window** (`hot_residency.ROUTE_SEEN`): exactly one
     `nf4_k25|le256` a MoE layer a step, 3,072 in all, and no other route;
   - **the store**, on every MoE layer's residency state: the NF4 stacks K25 reads (`h_gu_p`, `h_gu_a`, `h_dn_p`,
     `h_dn_a`), their dtypes, shapes and device, and **bytes per expert read off the loaded tensors**, not assumed;
   - **provenance:** the arena they were loaded from must have been written after this run started. That makes it the
     box's own bake, not a shipped file;
   - **a device copy rate**, reported only: bytes read plus written over 20 copies of 512 MiB, as P141 counts it.

   Any other route, store kind, layer count or K25 count is a fault, and the read is **VOID**, never a silently
   different floor.

## The rule (`p154_reduce.py`, fixed before data)

- **VOID:** the routing record missing; any fault in it; other than 64 steps; bytes per expert unread; P152's verdict
  not at its pinned bytes, or its two GEMM lines (`136x48`, `136x64`, 48 launches a step) missing.
- **READ:** for gate_up and for down:
  - the **floor** is the mean unique experts a step (summed over the 48 layers) x that projection's bytes per expert,
    over P141's copy rate (**1,498.9 GB/s**, the rate P152's floor used);
  - the **recorded** time is P152's removable time for that GEMM line;
  - the **fraction of copy** is floor over recorded;
  - the **headroom** is recorded minus floor.
- **Reported beside, never deciding:** the same floor at this box's own copy rate, and at the step totals' minimum and
  maximum.

## Predictions (graded in the read, never ruling)

- The route is device-grouped K25 on an NF4 store on all 48 layers, with 6,144 K25 calls and 3,072 `nf4_k25|le256`
  route records in the window.
- Bytes per expert: gate_up **1,769,472** and down **884,736**: 0.5625 B a weight, packed nibbles plus an fp32 absmax
  per 64, the arena's 2,654,208-byte row.
- Mean unique experts per layer-step on the replay's route: 50-65 (P152's eager, host-grouped window read 57.02).
- gate_up's floor 2.7-3.7 ms, at a fraction of copy of 0.32-0.45. down's floor 1.35-1.85 ms, at 0.40-0.55. Headroom
  over both at least 5 ms.
- The box's own copy rate is 1,350-1,650 GB/s.

## Budget and sequencing

- **One RTX 5090**, declared at $1.05/h, **guard 0.8 h**, 61 GB at <= $0.011/GB: worst 0.8 x $1.05 + 61 x $0.011 =
  **$1.51**, under the standing rule. Disk 200 GB (the box's floor is 150 GB free).
- **The guard is set by the fit.** Every time-left check must fit it given the measured setup, P149 Amendment 3's
  lesson. The allowances are P152's read, rounded up: boot 300 s, install 450 s, fetch 900 s, bake and prompts 180 s.
  With the runner's 600 s reserve after each step:
  - the fetch check: 300 + 450 + 1200 + 600 = 2550 s;
  - the bake check: 300 + 450 + 900 + 300 + 600 = 2550 s;
  - the routing check: 300 + 450 + 900 + 180 + 300 + 600 = **2730 s**, which fits 0.8 h (2880 s) and not 0.75 h
    (2700 s). `tests/test_p154.py` asserts both.
- **Hosts:** only those with a recorded power cap of at least 575 W (`vast_min_power_limit_w: 575`,
  `vast_power_limit_unknown: exclude`), screened for a cap read within the last three days. The box's own floor check is
  the guard.
- **Run ids** `p154-5090-1` to `-3`. Two consecutive HARNESS_ERROR, VOID or NOT_RUN runs stop the lane. A pre-flight
  host stall or a power-floor refusal (rc 17) is not a strike.
- **Sequencing:** the rental waits for this registration's merge and the maintainer's final pass, then the shared
  pre-flight and `--validate` on the exact manifest.

## Receipts

Fetched to the run directory's `p154/` and published under `bench/p154/receipts/<run>/` by the 2026-10-19 citation
rules:
- `route_b16.json`, `verdict_p154.json`, `gpu_phases.csv`, `summary.txt`, `forensics.txt`, `versions.txt`, the logs
  without carriage returns, and `work/bake.json`;
- `SHA256SUMS`, and `STORE-ONLY.sha256` for the rest.

`RESULTS-p154.md` is written from those files.

## What this lane cannot say

- Nothing about speed or a lever: it prices headroom. A kernel change and its gain are their own registrations.
- Not that the floor and the recorded GEMM times come from one box. The routing and the store are read here; the GEMM
  times are P152's, on another RTX 5090 at the same pins.
- Not that the routing pass saw the replay's experts step for step. It runs the replay's route eagerly, which is not the
  captured graph, and the window's tokens are not compared.
- Nothing about other cards, families or loads: one RTX 5090, Qwen3-30B-A3B, 16 rows in steady decode.
