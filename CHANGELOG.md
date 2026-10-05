# Changelog

## Unreleased

### TC1 amendment 33 registered: load-gated draws, and the same-stack pair re-asked under them (bench and tests only)

- **Why.** On multi-tenant rental hosts the host-bound training step slows when the host is busy, and the arms' own samplers show the
  load average following the unstable draws (`tc1-5090-66`, `-68`, `-70`, `-72`). The same-stack pair (P50, P51) has gone unread on three
  boxes.
- **What.** `tc1_run.sh`'s `arm` now runs `arm_once` through a gate. With `TC1_LOAD_GATE` set, an OK arm whose median host load1 over its
  own run exceeds the gate is set aside to `loadvoid/` and run again (at most `TC1_LOAD_RETRIES`). The last attempt stands. Every
  attempt writes a `LOADGATE` line, which the reducer prints. Unset, nothing changes.
  - `tc1_drive.sh` forwards both variables.
  - A functional test runs the script's own `arm()` with `arm_once` stubbed: a busy draw is voided and re-run, the cap holds, an unset
    gate is one attempt.
- **The box.** `qwen3samestack` over 60 steps with `TC1_LOAD_GATE=6.0`, off the busy and unexplained machines. P50 and P51 bands are
  unchanged and the reading is final.

### Read: TC1 amendment 28 — the double-quantized absmax costs Mixtral 2.3 % of its step for 2.04 GB (P57 HELD); Qwen3's speed pair unstable (P56, P58 UNTESTED); amendment 31 re-asks it over 60 steps

- `tc1-5090-71` ($1.27): Mixtral-8x7B resident, `_dq1`/`_dq0` 1.023 [1.004, 1.043], peak 31.07 → 29.03 GB, held-out −0.0028.
- `tc1-5090-70` ($1.13): Qwen3-30B-A3B. Peak 27.15 → 25.82 GB and held-out +0.0026, but both speed pairs were unstable (5.6 % and
  23.8 %), with step times following the host's load average.
- By the registered rule `E4B_ABSMAX_DQ` stays opt-in. Amendment 31 re-asks P56 (and P58's Qwen3 half) with `TC1_STEPS=60` off the
  machines that went unstable. Row `e4b.train.absmax-dq.mixtral.5090.2026-10-05`.

### TC1 amendment 32 registered: one variable, Triton 3.4 against 3.7.1, on one RTX 5090 (P59–P61) (bench and tests only)

- **Why.** Amendment 24's 0.882 changed torch, transformers and triton together. An unregistered RTX A2000 decomposition put the whole
  gain in triton 3.7.1's code for grouped-nf4-gemm's training kernels: torch 2.8 with triton 3.7.1 read 0.931×, and torch 2.12 and
  transformers 5.5 added nothing. The A2000 runs that slice device-bound, so the 5090 reads it here.
- **Token `qwen3tritonab`.** venv-e4b with its triton 3.4 against triton 3.7.1 put first on the arm's `PYTHONPATH`, matched and shipped
  arms, two draws a side, 60-step runs, prebound launches off on both sides.
- **Predictions.** P59 matched in [0.82, 0.95]; P60 shipped in [0.85, 0.98]; P61 held-out within 0.005. The reducer scores them (one new
  self-test case).

### Default: e4b's prebound Triton launches (`E4B_TRITON_PREBIND`) are on; `=0` turns them off

- **Why.** TC1 amendments 26 and 30 read the fused RMSNorm and rotary kernels' prebound launches, with grouped-nf4-gemm's
  (`GNF4_TRITON_PREBIND`), against the flags off on one RTX 5090 each, triton 3.4.
  - The training step reads 0.973× on the matched arm (`tc1-5090-69`).
  - It reads 0.980× on the shipped arm over 60 steps (`tc1-5090-73`).
  - Held-out moves by 0.0012 or less.

  The registered rule (P53, P54, P55 held) makes both defaults on. grouped-nf4-gemm's flip is its own PR.
- **What.** `prebind_requested()` reads the flag as on unless it is `0`. Values are unchanged (the same compiled kernel). Triton releases
  other than 3.4 and 3.6 keep Triton's own launch.

### CI on grouped-nf4-gemm 0.40.0 (no default changes there)

- CI now tests against grouped-nf4-gemm **0.40.0** (`cfc79904`). That release changes no default. It carries:
  - the pinned-slot fence (`ColdTier.fence`, #60): a fill into a pinned slot waits for any queued device copy that
    still reads it, which the arena training path (`nvme_train`) gets through `segment_into`;
  - opt-in `GNF4_TRITON_PREBIND=1`: the training GEMMs launch without Triton's per-call binding, bit-identical. TC1
    amendment 26 is registered to read it.
- The `[fast]` floor stays `>=0.30.0`.

### Docs: `SERVING-THROUGHPUT.md`'s gpt-oss line notes what shipped after it (docs only)

- The 0.32.0-era line said gpt-oss was "NF4 only", with int4 experts and the router fold both refused. Both were built
  out since: the native MXFP4 store (decode only; prefill on the kept NF4 stacks) and the `topk_softmax` router fold.
  A dated sub-note says so, and the historical line is left as measured.

### Serve estimate: the prefill graph's pool is named, and `ServeSetup` carries `prefill_graph`

- Since SC2b, `E4B_PAGED_PREFILL_GRAPH=auto` is the server's default. When the first-chunk prefill graph engages,
  it keeps its forward's working set in a private pool for its life: +3.3 GiB on Qwen3-30B-A3B int4.
  `estimate_serve_footprint` does not price that pool. It now lists the pool as not modelled wherever the graph can
  engage (decode graphs with device grouping, no linear-attention layers).
- `ServeSetup.prefill_graph` (`auto`, `1` or `0`) is passed through `to_env()`, so a caller that wants memory bounded
  by the estimate can plan `0`.

### Fix: `serve_paged` decodes eagerly below sm_89 instead of dying in Triton's compiler

- **The bug.** Bucketed decode graphs (`E4B_PAGED_GRAPHS=auto`, the all-VRAM default since P109) need
  grouped-nf4-gemm's fused FP8 KV append. That kernel casts to e4m3 with Triton's `tl.float8e4nv`, which Triton
  compiles only on sm_89+. On an RTX A2000 (sm_86, triton 3.4.0) the first graphed decode step died: "type fp8e4nv
  not supported in this architecture". Every Ampere card (sm_80/86) took the same path by default.
- **The fix.**
  - `fp8_paged_kv.fused_append_unsupported(capability)` states the floor.
  - The fused append degrades to the eager append below it; an explicit `E4B_FUSED_KV_APPEND=1` is refused in
    words.
  - `E4B_PAGED_GRAPHS=auto` decodes eagerly there; `=1` is refused in words.
  - An unknown capability (no CUDA) changes nothing.
- **Unchanged on sm_89+**, where every registered serving number was read (RTX 5090, H100).

## 0.47.0 — 2026-10-05 — serve_paged's first-chunk prefill graph is on by default (`auto`; lane SC2b: serial TTFT 1.30-1.65x faster with byte-identical text, +3.3 GiB, capacity unchanged); LFM2, Granite-4.0-H, ERNIE-4.5 and Nemotron-H supported for fast training (MG1); CI on grouped-nf4-gemm 0.39.0

**0.47.0.** One default changes, by lane SC2b's licence: `serve_paged`'s first-chunk prefill graph is `auto`.

- **Serving: `E4B_PAGED_PREFILL_GRAPH=auto` by default.** Every 512-token first chunk replays one CUDA graph of the
  prefill forward, wherever the graph's startup check passes (bitwise against eager).
  - **What SC2b read** (#846; one RTX 5090, Qwen3-30B-A3B int4, 16 sequences): the graph engaged on all 508 requests,
    and serial text was byte-identical with it off and on in both draws. Serial TTFT was **1.645× and 1.299×** faster,
    and nothing regressed at any arrival rate. The capacity ceiling stays at **1 req/s**: under load, the per-prefill
    stall is mostly work outside the graphed forward.
  - **Memory.** The graph's private pool holds its forward's working set for its life (**+3.3 GiB** in SC2b), so
    `auto` stands down when the device's free memory after capture is below that pool. Where it stands down, or the
    check refuses, the server runs with eager prefill and `/health`'s `prefill_graph` says why.
  - `E4B_PAGED_PREFILL_GRAPH=0` restores eager prefill; `1` stops the server on a refusal. `prefill_graph.requested` is
    now the setting string.
- **Training support (MG1).** LFM2-8B-A1B, granite-4.0-h-tiny, ERNIE-4.5-21B-A3B and Nemotron-3.5-Lightning-30B-A3B
  enter `fast_train = supported`; each passed tp1's loss-parity verdict on one RTX 5090. Qwen3.6-35B-A3B also passes,
  but stays `experimental` until its dgrad counter (P2) is read. Gemma-4 and OLMoE re-read clean after #1048.
- **CI on grouped-nf4-gemm 0.39.0.** Its `GNF4_TRAIN_GEMM=auto` takes the dense route off sm_90 for training calls with
  at most 16 present experts (TC1 amendment 22: Mixtral-8x7B's step 0.651× on an RTX 5090; Qwen3-like layers stay
  fused). The `[fast]` floor stays `>=0.30.0`; `pip install -U grouped-nf4-gemm` picks it up.
- **New: before-load serve planning.** `estimate_serve_footprint`, `ServeSetup` and `paged_kv_pool_bytes` price what
  `serve_paged` will hold under the all-VRAM placement, from the config and a `meta` tree, before anything loads. The
  KV pool's bytes match the pool's own allocation exactly; the working set is a stated heuristic.
- **Opt-in: `E4B_TRITON_PREBIND=1`.** The fused RMSNorm and rotary kernels launch without Triton's per-call argument
  binding. TC1 amendment 26 is registered to read it.
- **Also in this release:**
  - TC1 amendments 25 and 27. On one software stack the matched set holds on two boxes (P52), and e4b's same-stack
    speed pairs were unstable both times (P50 and P51 UNTESTED, final). Amendment 28, the double-quantized absmax as a
    default, is registered;
  - MG1 amendment 2: the ladder OOMs at Qwen3.6's licensed configuration, so P2 stays unread and `qwen3_5_moe` stays
    `experimental`;
  - the provenance of TC2 amendment 8's Mixtral row: grouped-nf4-gemm `@bb56b42`, not the v0.38.0 tag.

### Read: TC1 amendment 26 — prebound Triton launches take 2.7 % off the matched arm's step (P54 HELD); the shipped pair was unstable (P53, P55 UNTESTED), so the flags stay opt-in; amendment 30 re-asks it over 60 steps

- `tc1-5090-69` ($0.43, Core Ultra 9 285K). Matched arm `_pb1`/`_pb0` 0.973 [0.958, 0.988], held-out within 0.0012, with 73,591
  prebound e4b launches and 24,546 prebound grouped-nf4-gemm launches on the `_pb1` side.
- The shipped arm's flags-off draws were 8.3 % apart. Its `_pb1` draws (1.972 / 1.970) read slower than both `_pb0` draws, which would
  have falsified P53 had the pair been stable.
- By the registered rule `E4B_TRITON_PREBIND` and `GNF4_TRITON_PREBIND` stay opt-in. Amendment 30 re-asks the shipped pair with
  `TC1_STEPS=60` on another machine; its reading is final.

### Read: TC1 amendment 27, the re-draw — the matched set holds again (P52), e4b's same-stack pair unstable (P50, P51 UNTESTED, final); amendment 29 registered: the pair over 60 steps

- `tc1-5090-68` ($1.86, EPYC 7663): Unsloth 8.989 / 9.246 s/step and e4b's field-image arm 4.388 / 4.314 were stable. e4b's same-stack
  arm, 4.060 / 3.755, was 7.8 % apart, so no ratio is read and P50 and P51 stay UNTESTED under amendment 25.
- Across two boxes, three different pairs lost stability over TC1's 10-step median window. Amendment 29 re-asks P50 and P51, with their
  bands unchanged, on one box with `TC1_STEPS=60` (50-step medians), the reference arm not run. Its reading is final.

### MG1 amendment 2 read: the ladder OOMs at Qwen3.6's licensed configuration, so P2 stays unread and `qwen3_5_moe` stays experimental; amendment 3 registered, reading P2 on tp1's own fused arm (bench, tests and docs)

- **The read** ([`bench/moegen/mg1/mg1-a2-5090-2/RESULTS-mg1-a2.md`](bench/moegen/mg1/mg1-a2-5090-2/RESULTS-mg1-a2.md)).
  - The first draw was refused at pre-flight: the host read 2.4 MB/s to the HF CDN and 2.0 MB/s to the generic endpoints.
  - On the second, on the same host that read MG1's PASS, the ladder's `fused` rung OOMed in AdamW's first step at r 8 with
    fp32 adapters. The backward had run, but the counter died with the process.
  - By the amendment's rule that is a row, and the status is unchanged.
  - The ladder holds a GPU copy of the initial adapters that tp1's driver does not (1.73 GiB of the 4.45 GiB above the arm's
    peak; the rest is not attributed).
- **Amendment 3** (`bench/moegen/MG1-PREREG.md`). `MG1_P2_ARM=1` runs only tp1's fused arm, from the staged driver file
  unchanged, under the new `bench/moegen/p2_hook.py`. The hook runs the driver as `__main__` and writes grouped-nf4-gemm's
  `DGRAD_STATS`, e4b's `FAST_TRAIN_STATS` and the driver's sha256 at interpreter exit. P8 needs loop 0 with kernel > 0 and the
  driver's sha equal to tp1's file. `tc1_drive.sh` forwards the knob.
  - `tests/test_mg1_lane.py` checks that the runner starts the staged driver file under the hook. It also checks that the hook
    records the counters and keeps the driver's exit code, including a stub's.
- **Spend.** $0.357 for amendment 2; $2.24 for the MG1 lane in all.

### Before-load serve planning: `estimate_serve_footprint`, `ServeSetup`, `paged_kv_pool_bytes`

- **New: before-load serve planning.** `estimate_serve_footprint(describe_moe(id), ServeSetup(...))` itemizes the
  device memory `serve_paged.build_engine` holds under the all-VRAM placement. The expert stacks and bf16 dense
  weights are sized from this package's own modules. The FP8 paged KV pool is `paged_kv_pool_bytes`, which uses
  `Fp8PagedKV`'s own arithmetic; tests compare it with a constructed pool. The prefill/decode working set is a
  stated heuristic. CUDA graph pools, the CUDA context and fragmentation are listed as not modelled. The solver's
  tiered placements are refused in words until they are measured.
  - `MoETopology` carries the paged pool's KV geometry: `kv_heads`, `kv_head_dims` and `kv_layers`. They are read by
    `serve_paged._kv_geometry` and `paged_runner.kv_layers`, as `build_engine` calls them.
  - `ServeSetup.to_env()` is the `E4B_PAGED_*` environment that builds the priced setup; `PagedServeConfig.from_env`
    reads it back (tested).
  - **Checked against lane P109's receipts** (`bench/p109/receipts/p109-5090-2`: Qwen3-30B-A3B, one RTX 5090,
    16 sequences × 4096 tokens, NF4 experts, no int4). The estimate is under the measured allocator peak by
    165 MB with eager decode and 158–210 MB with decode graphs: 0.7–0.9%. Allocator reserve slack on those arms
    was 0.1–0.9%, far below a trainer's.

### TC1 amendment 28 registered: the double-quantized expert absmax as a default, A/B on Qwen3-30B-A3B and Mixtral-8x7B (P56–P58) (bench and tests only)

- **Why.** The memory census found the expert absmax is the only static class where e4b and Unsloth differ. e4b keeps it in fp32,
  1.81 GB on Qwen3-30B-A3B; `E4B_ABSMAX_DQ=1` stores it in 0.46 GB as Unsloth does. The switch's own cost has never been read as an A/B.
- **Tokens `qwen3dqab` and `mixtraldqab`.** The matched arm, resident, `E4B_ABSMAX_DQ=0` against `=1`, two draws a side in ABBA order.
- **Predictions.** P56 (Qwen3) and P57 (Mixtral): the step within [0.97, 1.03] and the peak lower by [1.25, 1.45] / [1.9, 2.3] GB.
  P58: held-out within 0.005. All three HELD makes the switch the default for the resident fused path. The reducer scores them (two
  new self-test cases).

### Default: serve_paged's first-chunk prefill graph is `auto` (lane SC2b licensed it); `auto` stands down on memory

- **What changes.** `E4B_PAGED_PREFILL_GRAPH` now defaults to `auto`. The graph engages wherever its startup check
  passes (bitwise against eager, #1070). Where the check refuses, the server runs with eager prefill, and `/health`'s
  `prefill_graph` block reads `refused` with `why`. An explicit `1` still stops the server on a refusal; `0` keeps
  eager prefill.
- **The licence.** SC2b (#846; #1072 registered, #1086 read; one RTX 5090, Qwen3-30B-A3B int4, 16 sequences) read
  DEFAULT_LICENSED:
  - every gate passed, and the graph engaged on all 508 requests, none eager;
  - serial text was byte-identical with the graph off and on, in both draws;
  - nothing regressed at any rate;
  - serial TTFT OFF/ON was **1.645 and 1.299**. That is above the licence's 1.10, but P1's predicted 1.5 was missed
    in one draw, so P1 is REFUTED.
- **What it does not do.** The capacity ceiling stays at 1 req/s (P3 REFUTED). Under load, most of the per-prefill
  stall on running decodes is work outside the graphed forward: K/V staging and flush, and scheduler bookkeeping.
- **Memory, and the stand-down.** The graph keeps its 512-token forward's working set in a private pool for its life:
  SC2b read **+3.3 GiB** at ready on Qwen3-30B-A3B int4. An eager prefill only borrows that, and a later chunk still
  runs eagerly and needs it again. So `auto` also stands down when the device's free memory after capture is below
  the pool. That is stricter than the licence, never looser. `/health` reports `pool_mib` and `free_after_mib`.
- **`/health` change.** `prefill_graph.requested` is now the setting (`auto`, `1` or `0`), not a bool.
- **Tests.**
  - The headroom refusal releases the graph and engages when there is room (GPU).
  - `engage_prefill_graph` covers all three settings (CPU).
  - An `auto` refusal is recorded and reported.
- **Verified on the A2000** (`bench/prefill-graph-auto-2026-10-04/`).
  - With nothing set, `auto` engaged, and prefill was bitwise against eager on the int4 Qwen3-MoE and on the Granite
    NF4 store.
  - At `max_seqs` 1, `auto` served eagerly and reported `refused`, while `1` stopped the server.
- **The pool measure.** It is the allocator's segments for the graph's own pool id. The first cut used the growth of
  `memory_reserved`, which read 0 once segments were recycled, so the headroom rule would have failed open. The A2000
  run caught it.

### Read: SC2b (#846) -- the prefill graph is value-identical, cuts serial TTFT 1.30-1.65×, and is licensed as a default (`auto`); capacity stays at 1 req/s

- **What ran.** `sc2b-5090-1` ($1.326, a 400 W 5090) drove e4b `serve_paged` with `E4B_PAGED_PREFILL_GRAPH` OFF against
  ON, paired, on today's stack (grouped-nf4-gemm v0.38.0; k19 / flash read from the servers' own `/health`).
- **Gates.** Every gate passed. Engagement: 508 replays for 508 requests, 0 eager chunks. Identity: every request's
  serial text byte-equal OFF vs ON in both draws.
- **Verdicts.**
  - P1 REFUTED: TTFT OFF / ON 1.645 and 1.299 against a predicted ≥ 1.5.
  - P2 HELD: TPOT unchanged.
  - P3 REFUTED: ON's ceiling is still 1 req/s; attainment at 2 req/s was 0.90 / 1.00.
  - P4 HELD: no regression.
  - **Licence: DEFAULT_LICENSED**, for a default of `auto` (engage where the startup check passes).
- **Why the ceiling held** (post hoc, `sc2_trace.py --plan`). The graph cut the per-prefill stall under load only about
  17% (0.32 s to 0.27 s). That stall is about 1.6× the graphed serial TTFT, so most of it is per-prefill work outside
  the forward, which names the next lever.
- **Memory.** The graph costs +3.3 GiB at ready on Qwen3-30B-A3B (20,308 → 23,686 MiB).
- **Total and read page.** SC2b cost $1.566. Read page: `bench/h2h-2026-10-02/sc2b/README.md`. `sc2_trace.py` gains
  `--plan`.

### Read: TC1 amendment 25, first box — on one stack the matched set holds (P52 HELD); the speed pairs were unstable (P50, P51 UNTESTED); amendment 27 registers one re-draw

- `tc1-5090-67` ($0.61, Core Ultra 9 285K): e4b's matched set in Unsloth's venv steps in 2.220 / 2.188 s, stable. Unsloth's draws were
  18.2 % apart and e4b's field-image draws 8.4 % apart, so neither ratio is read.
- On one stack Unsloth and e4b's reference are EQUIVALENT to e4b's fused arm, and parity PASSES (P52).
- Amendment 27: one re-draw of the same token on another machine, with amendment 25's predictions unchanged; its reading is final.
  Bundle and `RESULTS-tc1-samestack-box1.md`; no register row until the re-draw is read.

### TC1 amendment 26 registered: prebound Triton launches, A/B on one RTX 5090 (P53–P55) (bench and tests only)

- **Why.** e4b's training step is host-bound, and much of each hot launch's host time is Triton's per-call binding. #1078
  (`E4B_TRITON_PREBIND=1`) and grouped-nf4-gemm#468 (`GNF4_TRITON_PREBIND=1`) are opt-in and bit-identical, and they cut those calls'
  host time on an RTX A2000 host.
- **Token `qwen3prebindab`.** The shipped and matched arms, both flags 0 against both 1, two draws a side in ABBA order, in venv-e4b
  (triton 3.4). `tc1_arm.py` records `prebind_ab` (each side's requested flag and its prebound and Triton launch counts) on every e4b
  receipt.
- **Predictions.** P53 shipped in [0.90, 0.98]; P54 matched in [0.92, 0.99]; P55 held-out within 0.005 on each arm. All three HELD
  turns both flags on by default for the Triton versions they cover. The reducer scores them (two new self-test cases).

### Tests: the family-blind training parity test counts grouped-nf4-gemm's dense dgrad route

- `tests/test_fused_train_parity.py::test_fused_train_is_family_blind` checked that the dgrad kernel or the grouped_mm route served
  both frozen GEMMs. Since grouped-nf4-gemm#463, `GNF4_TRAIN_GEMM=auto` takes the dense route off sm_90 for calls with at most 16
  present groups, so the Mixtral case (8 experts) failed on CUDA cards other than the H100. CI has no GPU and skips the test. The
  count now includes `DGRAD_STATS["dense"]`, read with `.get` for older grouped-nf4-gemm. Test only.

### Opt-in: the fused RMSNorm and rotary kernels launch without Triton's per-call argument binding (`E4B_TRITON_PREBIND=1`)

- **What.** `engines/triton_prebind.py`'s `prebind(kernel)` lets the first launch of each specialization go through Triton and keeps
  the compiled kernel that launch returns. Its key is what Triton specializes on: each tensor's dtype and 16-byte alignment, each
  integer's value, every other argument's type and value, the launch options, the device and the debug knobs. Later launches with
  that key call the kernel's own launcher. `rmsnorm_train` (`_rms_fwd`, `_rms_bwd`) and `rope_train` (`_rope_fwd`) launch through
  it. The flag is read when those modules are imported; off, `prebind` returns the kernel itself and nothing changes.
- **Values.** Same compiled binary, arguments and stream, so outputs are bit-identical. Triton's own launch serves a Triton release
  other than 3.4 and 3.6, a registered launch or pre-run hook, a callable grid, a changed global the kernel reads, and an argument of
  another type. One knob is read less often: triton 3.4 re-reads `TRITON_DEBUG` at every launch, the prebound path once per kernel
  (3.6 itself reads it once).
- **Measured** on the RTX A2000 box's host: a Xeon W-1250 at load average 18–44 on its 12 threads, the bench at nice 10. Host µs per
  call, each timed after a synchronize; median of 2,000 interleaved off/on pairs; two runs per torch. Qwen3-30B-A3B shapes: sequence
  2048 × micro-batch 2, hidden 2048, 32 query and 4 key-value heads of 128.

  | call | torch 2.8.0 / triton 3.4.0: off → on | torch 2.11.0 / triton 3.6.0: off → on |
  |---|---|---|
  | `RMSNormFrozen.apply`, hidden / q_norm / k_norm | 152–160 → 117–121 / 137–170 → 105–127 / 132–144 → 101–111 | 143–146 → 129–131 / 154–197 → 135–168 / 133–140 → 120–126 |
  | `RMSNormFrozen.backward`, hidden / q_norm / k_norm | 110–116 → 74–77 / 124–126 → 82–83 / 82–95 → 55–64 | 92–96 → 78–82 / 118–156 → 98–125 / 81 → 70 |
  | `rope_qk` forward (two launches) | 232–255 → 149–168 | 204–211 → 171–174 |
  | `_RopeQK.backward` (two launches) | 185–197 → 102–109 | 135–138 → 108–109 |
  | one launch alone (the kernels and shapes above) | 53–90 → 28–46 | 40–69 → 30–48 |

  Most of what is left in an RMSNorm forward is torch's: `autograd.Function.apply` (24–31 µs for a no-op), the two allocations (10–12)
  and the views (9–13).
- **Not measured.** No training step, and no H100 or RTX 5090 host. The default stays off until a registered A/B reads it.
- **Tested.** `tests/test_triton_prebind.py`. Against Triton's own launch, outputs and dx match under `torch.equal`: RMSNorm in bf16
  and fp16, all four variants, odd widths and row counts, misaligned pointers; rotary at odd lengths. Every prebound launch is the
  very compiled kernel Triton's lookup returns. A launch hook, a callable grid, a changed global, a tensor passed by keyword and an
  unsupported Triton release take Triton's path. On an RTX A2000 under torch 2.8.0 and 2.11.0, flag off and on: 41 passed with
  `test_rmsnorm_train.py` and `test_rope_train.py`. The fused-training suites agree flag off and on (156 passed;
  `test_fast_v4.py::test_gptoss_inside_lora_is_still_skipped` fails either way on that box).

### CI on grouped-nf4-gemm 0.39.0; where the dense-route rows came from (docs and register notes only)

- **CI** now tests against grouped-nf4-gemm **0.39.0** (`a5edec87`). In that release `GNF4_TRAIN_GEMM=auto` takes the
  dense route off sm_90 for training calls with at most 16 present experts, so Mixtral-like layers train densely and
  Qwen3-like layers stay fused. The rule was read in TC1 amendment 22. The `[fast]` floor stays `>=0.30.0`.
- **Provenance.** TC2 amendment 8's Mixtral row (`…mixtral.5090.2026-10-04.dense-default`) ran grouped-nf4-gemm
  `@bb56b42`: #463's merge on main, after the v0.38.0 tag, with its version string still reading 0.38.0. The row's notes
  and `docs/STATUS.md` now say so, and that the rule ships in 0.39.0.
- **`docs/MOE_RUNTIME_PORTABILITY.md`** states `auto`'s current rule (grouped_mm on sm_90; dense for at most 16 present
  experts elsewhere; fused otherwise).

### MG1 read: LFM2, Granite-4.0-H, ERNIE-4.5 and Nemotron-H enter `fast_train = supported`; Qwen3.6 PASSES and waits on P2; Gemma-4 and OLMoE re-read clean after #1048 (one RTX 5090; `bench/moegen/mg1/mg1-5090-2/RESULTS-mg1.md`)

- **Why.** #1048 made the training stack's glue structural, so other MoE families could inherit the Qwen work. Lane MG1
  (registered before any box, amended once before any box) puts the new families through tp1's licence instrument unchanged:
  tp1's arm driver, the clinical fixture, N 60, tp1_reduce v2.1's verdict and the train anchor.
- **Every family PASSES.** OLMoE-Instruct and Gemma-4-26B-A4B-it are the regression anchors. Gemma-4's step-wise median
  moved further inside the band after #1048 changed its fused RMSNorm to its own fp32 multiply.
  - LFM2-8B-A1B, granite-4.0-h-tiny, ERNIE-4.5-21B-A3B and Nemotron-3.5-Lightning-30B-A3B enter
    `qlora-fused-moe-experts.model_families`, with `quantize` / `reference_train` / `fast_train` supported (claims
    `e4b.train.parity.mg1.<family>.*.2026-10-04`).
  - Qwen3.6-35B-A3B trained resident and PASSES. It is `experimental`, not supported, because MG1's rule also needs P2, the
    dgrad kernel's engagement, whose counter was not read on that box. MG1 amendment 2 reads it.
- **Predictions.** P1, P3, P4 and P6 HELD. P2 HELD on the six laddered families and is untested on Qwen3.6, whose ladder OOMed
  at r 16. P5 was **falsified** for Nemotron-H's norm: the probe correctly read its fp32 multiply, so the prediction was wrong,
  not the code. ERNIE's interleaved rotary was refused on semantics, as predicted.
- **Draws.** Proving run $0.044; one box refused by the train anchor ($0.048, `h2d.self_pair` 1.0321 > 1.03); the reading
  $1.79. $1.88 in all.

### SC2b registered (#846): does a CUDA-graphed prefill chunk lift e4b `serve_paged`'s request-level capacity?

- **Why.** SC2 read e4b's request-level capacity as prefill-bound: each 512-token prefill forward stalls every running
  decode. That forward is launch-bound (about 15k kernels per chunk; P102's TTFT moved 4× with the host CPU alone).
  `E4B_PAGED_PREFILL_GRAPH=1` (#1070) captures the first chunk once, verifies it bitwise at startup and replays it.
- **What runs.**
  - A new box F, e4b only: the code at `373c89ac` with grouped-nf4-gemm v0.38.0.
  - Prefill routes are main's defaults (k19 + flash), with neither SC1 pin exported. The server's own `/health` routes
    are asserted, and a test executes the box's export block.
  - Knob OFF against ON, paired: same seeds within a draw, counterbalanced order, a determinism repeat.
  - Engagement is checked right after warm-up.
- **The rule.** `bench/sc2/sc2b_reduce.py`, 10 self-test cases.
  - Gates: ROUTES, ENGAGED, PROMPTS, DETERMINISM and IDENTITY (every request's serial text byte-equal OFF vs ON).
  - P1: serial TTFT OFF/ON ≥ 1.5. P2: TPOT unchanged. P3: ON's ceiling ≥ 2 req/s. P4: no regression.
  - The licence, decoupled from P1: the gates, P4 and TTFT ≥ 1.10× in both draws. It licenses a default of `auto`.
- **Instrument additions.** `sc2_identity.py` (the serial identity gate); the driver records `usage.prompt_tokens`; a
  server error saves the full `/health` JSON.
- **Budget.** Proof 1.0 h and reading 2.5 h, each stated with download charges this time.

### MG1 amendment 2, before its box: P2 for Qwen3.6, read from the fused rung's dgrad counter at the licensed configuration (bench and prereg only)

- **Why.** The MG1 reading PASSED Qwen3.6-35B-A3B, but P2 went unread: tp1's arm driver predates `DGRAD_STATS`, and Qwen3.6's
  ladder OOMed at its default r 16. The decision rule needs P2, so `qwen3_5_moe` waits as `experimental`.
- **What.** `mg1_run.sh` gains two knobs, both forwarded by `tc1_drive.sh` and pinned by `tests/test_mg1_lane.py`.
  `MG1_LADDER_ONLY=1` skips the anchor and the arms, and `MG1_LADDER_ARGS` is appended to the ladder. The registered shape is
  the `fused` rung at r 8 / alpha 16 / fp32 adapters / bf16 attention, with no profiler: the licensed arms' configuration.
- **P7.** `DGRAD_STATS["loop"] == 0`. A zero loop flips the row to `supported`; anything else is a row, and the status is
  unchanged. Guard 0.75 h, about $0.64.

## 0.46.0 — 2026-10-04 — CI on grouped-nf4-gemm 0.38.0, whose pinned-tier sizing models PyTorch's power-of-two allocator; before-load planning (`describe_moe`, `prepare_qlora_training`); `serve_paged` reports its prefill routes; two opt-ins, a first-chunk prefill graph verified at startup and the double-quantized expert absmax

**0.46.0.** No default in this package changes. CI now tests against grouped-nf4-gemm 0.38.0's commit.

- **grouped-nf4-gemm 0.38.0.** `capacity_for_bytes(..., pinned=True)` is the helper e4b's NVMe and host-RAM tier
  messages tell you to size `hot_rows` with. It now models PyTorch's power-of-two pinned allocator: exact and never
  over budget, where the flat 1.9 wasted up to half a budget. It was measured on cgroup v1 and v2 (lane K29,
  CONFIRMED; grouped-nf4-gemm#71 closed). Its new `GNF4_TRAIN_GEMM=dense` route is opt-in. The `[fast]` floor stays
  `>=0.30.0`; `pip install -U grouped-nf4-gemm` picks 0.38.0 up.
- **New: before-load planning.** `describe_moe`, `QLoRASetup`, `estimate_qlora_footprint` and
  `prepare_qlora_training` describe a checkpoint's MoE structure and memory from its config and a `meta` module tree,
  with no weights read, and hold the recipe's choices in one place.
- **`serve_paged`.**
  - `/health` reports the prefill routes the server resolves (`prefill_routes`).
  - An opt-in first-chunk prefill graph (`E4B_PAGED_PREFILL_GRAPH=1`) engages only if it verifies bitwise against
    eager at startup, and refuses otherwise. Its speed is for lane SC2b to read.
- **Opt-in: the frozen expert absmax stored double-quantized** (`E4B_ABSMAX_DQ=1`). TC1 amendment 23 measured it on
  Qwen3-30B-A3B: 1.812 GB → 0.460 GB, which brings e4b's peak to 0.43 GB above Unsloth's, all transient.

**Readings in this release:**
- **TC1c amendment 8.** On an H100 NVL at default settings on 0.45.0, e4b is faster per step than Unsloth: Unsloth/e4b
  **1.061** [1.047, 1.075]. This is the new H100 position of record. It supersedes amendment 1's 0.817, which was read
  on the fused kernels.
- **TC2 amendment 8.** On Mixtral at default settings, Unsloth is faster: **0.836**. On Qwen3.6 at micro-batch 1, e4b is
  2.05× faster (a labelled row).
- **SC2** (#846), request-level serving on one RTX 5090. vLLM and SGLang hold the SLO to 8 req/s; e4b's `serve_paged`
  holds it only to 1 req/s. Every miss is a TTFT miss, set by the prefill forward stalling running decodes.
  - *Correction:* e4b ran SC1's prefill route pins, not the registered k19 + flash. Lane SC2b re-reads it on the
    defaults, with and without the prefill graph.
- **P55** (#344): the host-memory reading of the Gemma-4 load fault is REFUTED.
- **#392:** the energy claim was remeasured on a released bitsandbytes. Decode is now at break-even (0.91–1.06×).
- **moe-generalize.** MG1's regression anchors pass after #1048; there are more RTX A2000 ladders, and a portability
  map.

### serve_paged: opt-in first-chunk prefill graph (`E4B_PAGED_PREFILL_GRAPH=1`)

- **What.** With the knob on, every first chunk of exactly `E4B_PAGED_CHUNK_TOKENS` tokens replays one CUDA graph of
  the prefill forward (`PagedModelRunner.enable_prefill_graph`). Later chunks and other first-chunk lengths run
  eagerly and are counted by reason. A first chunk reads no history, so one graph serves every slot; after a replay
  the graph's K/V outputs are staged for the request's slot, copied when the prompt continues.
- **Engages only if verified at startup.** It needs device grouping, no linear-attention state, and a capture that
  succeeds. On two seeded prompts, each replay must equal an eager forward bit for bit in the logits and every layer's
  staged K/V, and the two prompts must give different logits. Otherwise the server refuses at startup (`/health`
  `prefill_graph.status: "refused"`, with the reason) rather than falling back silently.
- **`/health`.** The `prefill_graph` block reports `off` / `on` / `loading` / `refused`; when on, it adds `T`,
  `replays`, `eager_chunks` and `eager_reasons`.
- **Tests.**
  - CPU: routing, counters, and the copy for a continuing prompt, with a mutation arm that drops the copy.
  - GPU: bitwise against eager on the FP8 pool and first tokens, the same mutation arm, and every refusal, each made
    to fire.
- **What the A2000 caught.** The first version kept the graph's input ids but not its positions tensor. Its startup
  check ran while the positions were still alive, so it passed, and every served replay then read freed memory: the
  bitwise test failed on the A2000. The graph now keeps everything it reads that was allocated outside the capture.
  The startup check now runs only after the capture's scope has returned and the allocator has been churned with a
  sentinel. A mutation test drops the positions after capture, and the check must refuse it.
- **Verified on the A2000** (`bench/prefill-graph-knob-2026-10-04/`). The GPU tests pass (7), and so do their
  neighbours (82). Through `build_engine`, on tiny random models with SC2's int4 stack (Qwen3-MoE) and with the
  Granite NF4 store, the knob engages, and 5 prompts each prefill bitwise against eager in the first token and the
  pool. At `max_seqs` 1 it refuses at startup.
- **Basis.** The A2000 census (`bench/prefill-graph-census-2026-10-04/`). Off by default; its speed is for lane
  SC2b to read.

### Prefill-graph feasibility census on the A2000 (bench only)

- **Result.** `serve_paged`'s 512-token prefill forward makes **zero host syncs**, on chunk 1 and on a chunk with 512
  tokens of history, on both the `k19` and `mtile` routes. It captures as a CUDA graph with no code change, and
  replays are bitwise-equal to eager on the logits and every layer's staged K/V. Two instruments agree after
  calibration: `set_sync_debug_mode` sites, and the profiler's stream-sync count against a no-op baseline.
- **The blocker** for an `E4B_PAGED_PREFILL_GRAPH` knob is the Python-side K/V staging (a list plus a whole-prompt
  `torch.cat` every chunk), not syncs.
- **Setup.** A tiny random Qwen3-MoE with Qwen3-30B-A3B's attention geometry, served by the unmodified `build_engine`
  under SC2's int4 stack with the routes unset. `bench/prefill-graph-census-2026-10-04/`. No claim row; nothing about
  speed.

### TC1 amendment 24 registered: the RTX 5090's fp32 `bmm` host cost and the torch 2.12 environment, on one RTX 5090 (P44–P49) (bench and tests only)

- **Why.** On both 5090 profiles the largest single e4b host row is `aten::bmm` in grouped-nf4-gemm's padded LoRA delta on the matched
  arm's fp32 adapters: 197–324 µs of host time per call, against 15 µs for bf16 on the same box and 30 µs for fp32 on an H100. The
  launch APIs read 2.5–6 µs a call, so this is not backpressure. TC1's native box also read e4b's matched arm 0.864× faster on torch
  2.12.1+cu130 than on torch 2.8.0+cu128; that reading was never attributed. Every 5090 position runs e4b on torch 2.8 and Unsloth on 2.12.1.
- **Token `qwen3bmmab`.** First a replay with no model, `bench/tc1/bmm_bench.py`. It times the delta's `bmm` at the 96 recorded
  real-router shapes in fresh processes, in fp32, bf16, under cublasLt and under TF32, on new, repeated, fixed and bucketed shapes, in both
  environments. Then the matched and shipped arms on venv-e4b against venv-unsloth, two draws a side.
- **Predictions.** P44: the fp32 call's host time reproduces outside training. P45: it is paid per new shape. P46: torch 2.12 removes it.
  P47: the matched arm gains 0.70–0.95×. P48: the shipped arm gains at least 0.05 less. P49: held-out moves at most 0.01. The reducer
  scores all six (four new self-test cases, including the fallen-back venv and an off-card replay).

### Read: TC1 amendment 23 — the memory census: with the absmax double-quantized, e4b's peak is 0.43 GB above Unsloth's, all transient (P41, P42 HELD; P43 FALSIFIED)

- `tc1-5090-65` ($0.38): Qwen3-30B-A3B's matched set at micro-batch 1, one draw per arm, the allocator census on, no speed read.
- Every static class is byte-for-byte the same in e4b and Unsloth except the expert absmax. e4b's fp32 absmax is exactly the
  analytic 1.812 GB, and `E4B_ABSMAX_DQ=1` stores it in 0.460 GB (P41). At least 99.96 % of each peak is attributed (P42).
- Peaks: e4b at defaults 26.02 GB, e4b with the absmax double-quantized 24.68 GB, Unsloth 24.24 GB. The 0.43 GB that remains is
  transient, mostly grouped-nf4-gemm's padded LoRA delta in the adapters' fp32. That is below P43's [0.5, 2.5] GB band.
- Row `e4b.train.memory-census.qwen3.5090.2026-10-04`. Results file `bench/h2h-2026-10-02/tc1/RESULTS-tc1-memcensus.md`.

### serve_paged: `/health` reports the prefill routes the server resolves

- **What.** `GET /health` gains a `prefill_routes` block, computed at each request by the same functions the forward
  calls: `int4_prefill` (the resolved `E4B_INT4_PREFILL`), `int4_prefill_above_256_rows` (the route prefill calls
  above 256 rows take when `device_grouping` is on), `prefill_attn` (the resolved `E4B_PAGED_PREFILL_ATTN`), the raw
  environment values, and `device_grouping`. A route the environment makes invalid reads `invalid: …`.
- **Why.** SC2's e4b arm inherited `E4B_INT4_PREFILL=loop` and `E4B_PAGED_PREFILL_ATTN=math` from the box script while
  its registration said `k19` and `flash`. A box's environment is not evidence of the route the server took; the
  server's own report is. Documented in `docs/SERVING.md`.

### #392: the energy claim remeasured with a recorded, released bitsandbytes (bench and docs only)

- **What.** The same NAS RTX A2000 and the unchanged `bench/_upstream/bench_energy.py`, run on **bitsandbytes 0.50.2**
  with every version recorded, three passes. A per-process GPU monitor shows no other workload.
- **Result.** `matmul_4bit` vs native bf16 total J/op:
  - decode **0.91–1.06×** (break-even; the 0.50.0.dev0 fork read 1.18×);
  - prefill 1.29–1.49×;
  - train 1.64–2.15× (fork 2.25×).

  The dequantize-then-linear arm reads decode 2.37–2.43×.
- **Records.** New register row `e4b.train.energy-honest.a2000-bnb0502.2026-10-04`. The fork-build row stands, and its
  notes point here. METHODOLOGY §10 carries a dated note. The receipts are in `bench/energy-remeasure-2026-10-04/`.

### moe-generalize: five more RTX A2000 ladders, the dequantize-then-GEMM geometry and engagement on the hybrids (bench only; `bench/moegen/RESULTS-moegen-ladders.md`)

- **Receipts:** the ERNIE-4.5, Nemotron-H and Qwen3.6 layer slices, OLMoE in the shipped bf16-adapter configuration, and LFM2's
  route rungs. These are informational within-box readings, not positions.
- **Engagement:** every MoE layer patched, with no dgrad loop.
  - Nemotron-H's non-gated relu² experts ran on real weights.
  - The RMSNorm probe read each family's own formula. Qwen3.6's centered norms fused for the first time; Nemotron-H's run its
    fp32 multiply.
  - ERNIE's interleaved rotary was refused on semantics, so it keeps its own.
  - With mamba-ssm, causal-conv1d and flash-linear-attention installed, no recurrent block fell back.
- **Geometry:** the dequantize-then-GEMM prototype's device ratio follows rows per expert at seq 512. It was cheaper on every
  family with 32 or more (LFM2 0.71, ERNIE 0.76, OLMoE 0.79, Granite-H 0.81, Qwen3 0.87) and not at 16–24 (Qwen3.6 0.97,
  Nemotron-H 1.07). That is the key for a measured dispatch rule, pending full-step readings per card.

### Before-load planning: `describe_moe`, `QLoRASetup`, `estimate_qlora_footprint`, `prepare_qlora_training`

- **Why.**
  - Asking "will this fit, and how" meant loading the model, or re-deriving its structure from config attributes
    whose names differ per family. Two call sites carried different lists of the routed top-k's spellings.
  - The training recipe's choices were split across env vars read at import (`train.py`) and keyword arguments
    spread over five calls (the documented fast path).
  - The only automatic memory decision was halving `TOKEN_BUDGET` after an OOM.
- **What.**
  - `arch.topology.describe_moe(model)` returns a `MoETopology` built from the config plus the module tree on
    `meta`, with no weights read. It records which layers carry an expert stack at the loader's path, each stack's
    shape, the non-expert parameter count (a tied head counted once), the attention projections, the loader's
    refusal, and the provenance of each fact.
  - `recipe.QLoRASetup` holds every mechanism choice in one place: storage, adapters, expert residency, expert
    kernel, dgrad, NF4 attention and kept MoE layers.
  - `recipe.estimate_qlora_footprint(topology, setup, tokens_per_microbatch=...)` returns an itemized estimate. Each
    item is `derived` (sized from this package's own `Experts4bit`/`ExpertsLoRA` built on `meta`, the classes the
    load builds) or `heuristic` (a stated formula: activations, kept MoE layers, the offload staging transient), and
    the estimate lists what it does not model. Pinned host homes are rounded to a power of two (grouped-nf4-gemm#71).
  - `recipe.setup_refusals` refuses by structure, never by family name. Two examples: biased expert stacks can't
    take `ExpertsLoRA`, and the grouped kernel reads NF4 only.
  - `recipe.prepare_qlora_training(model_id, setup)` builds exactly the priced setup: the documented fast path as
    one call. Every requested accelerated path must engage or it raises.
- **Refactors, behaviour unchanged.**
  - The loader's three config-level refusals moved into `loader.check_admission` (the loader calls it first, and
    quant_guard's source scan still sees every literal raise). `loader.admission_refusal` returns the message instead
    of raising.
  - `serve_paged._routed_topk` and `int4_experts._top_k` now read one alias list
    (`arch.topology.ROUTED_TOP_K_KEYS`, the union of the two), so each now also accepts the spelling only the other
    knew.
- **Tests.**
  - `tests/test_topology_recipe.py` checks topology against the real model built from the same config, for
    qwen3_moe, olmoe and granitemoe (exact parameter counts).
  - The derived footprint items are checked against the built modules.
  - Structural refusals are tested on a gpt-oss config and on Llama.
  - A CUDA test runs `prepare_qlora_training` on a tiny checkpoint with both expert kernels and asserts the trainable
    count is the count the estimate priced.
  - On an RTX A2000: 121 passed together with `test_int4_experts_topk_alias.py` and `test_loader_architectures.py`.
    The loader, serving, int4, moe-keep and quant_guard suites pass too.
- `train.py` is untouched; adopting `prepare_qlora_training` there is a separate change.

### Read: TC2 amendment 8 — at default settings Unsloth is faster on Mixtral, 0.836; on Qwen3.6 at micro-batch 1 e4b is 2.05× faster (P24–P28 HELD)

- **Box M** (`tc1-5090-63`, $0.60): Mixtral-8x7B with e4b at default settings, where grouped-nf4-gemm 0.38.0's `auto` takes the dense
  route on every expert call. e4b steps in 3.57 s against Unsloth's 2.98 s: Unsloth/e4b **0.836** [0.828, 0.844], the pair
  COMPARABLE, e4b's peak 31.08 GB against 29.13. The host is a Core Ultra 9 285K, the Unsloth-favouring end of the range seen. By the
  registered rule this becomes the family's position and supersedes amendment 6's 0.697 on the fused kernels.
- **Box Q** (`tc1-5090-64`, $1.31): Qwen3.6-35B-A3B resident at micro-batch 1 × accum 8, e4b with the absmax double-quantized and the
  non-routed projections in NF4. Unsloth/e4b **2.049** [2.028, 2.070] (18.95 against 9.25 s/step), the pair COMPARABLE, step-0 gap 0.019
  nats. Qwen3.6's first quoted position, labelled; e4b's defaults do not fit this family on 32 GB.
- Rows `e4b.train.h2h.unsloth.mixtral.5090.2026-10-04.dense-default` and `…qwen3_5.5090.2026-10-04.mb1-dq-frozen4`. Lane page
  `bench/h2h-2026-10-02/tc2/README.md`.

### Correction: SC2's e4b rows ran SC1's prefill route pins, not the PREREG's k19 + flash (#846)

- **What happened.** `bench/sc1/sc1_run.sh` exports `E4B_INT4_PREFILL=loop` and `E4B_PAGED_PREFILL_ATTN=math` to every
  box. Box E's `serve_paged` inherited them, so under device grouping SC2's e4b prefill ran the int4 M-tile and `math`
  attention, not the stated defaults (k19, flash).
- **What stands.** The comparators, the measurements of e4b as run, and the prefill-bound mechanism.
- **What changes.** Q4 and Q5 now read REFUTED for e4b as run, and UNREAD for the registered configuration.
- **Where to read it.** The read page's Correction section (`bench/h2h-2026-10-02/sc2/README.md`). Lane SC2b runs e4b
  at the defaults with the pins unset, and records the resolved routes.

### TC1 amendment 23 registered: a memory census of e4b against Unsloth on one RTX 5090 (P41–P43)

- **The box** (`qwen3memcensus`). Qwen3-30B-A3B at TC1's pin and tokens, micro-batch 1 × accum 8, one draw per arm: e4b at its defaults
  (fp32 expert absmax), e4b with `E4B_ABSMAX_DQ=1`, and Unsloth on grouped_mm. Every arm runs the new memory census. No speed is read.
- **Predictions.** P41: the census finds e4b's fp32 expert absmax within 2 % of the analytic 1.81 GB, and the double-quantized one
  within 2 % of that over 3.94. P42: it attributes at least 90 % of each arm's peak to named groups. P43: with the absmax
  double-quantized, e4b's peak exceeds Unsloth's by 0.5–2.5 GB.
- **Harness.** `tc1_arm.py --mem-census 1` (any framework, off by default) records PyTorch's allocator history from the arm's start, in
  a ring of 1,000,000 events with Python stacks. Each time the run's peak grows by 32 MiB, the box snapshots the ring and reduces it on
  the spot. The reduction finds the peak and groups the allocations live at it by static class, else by the first e4b,
  grouped-nf4-gemm, Unsloth, bitsandbytes or optimizer frame. A static census by class runs after setup and after training. The
  receipt gains `mem_census`; a failure inside the census is recorded there, and the arm finishes as usual. Smoke-tested on torch 2.8 on
  an RTX A2000. The Unsloth venv's torch 2.12 is handled by reading the recorder's signature, but has not run.
- **Reducer.** The family is registered with no position quoted. Its scorer prints P41–P43 beside side-by-side census tables. Four new
  self-test cases, 81 in all. `bench/tc1/TC1-PREREG.md` amendment 23.

### Read: TC1c amendment 8 — on an H100 at default settings e4b is faster per step than Unsloth: 1.061 (P24, P25, P26 HELD)

- **The box.** `tc1c-h100-15` ($2.44) ran e4b 0.45.0's code with grouped-nf4-gemm 0.37.0 and nothing set. `auto` took the grouped_mm
  route on every fused arm.
- **Position.** Unsloth/e4b **1.061** [1.047, 1.075] (2.544 against 2.397 s/step), at 2.93 GB more peak VRAM on e4b and ×1.23 Unsloth's
  energy. The matched set is EQUIVALENT.
- **Register.** The new H100 position of record is `e4b.train.h2h.unsloth.qwen3.h100.release-0.45.0`. It supersedes amendment 1's 0.817
  (the fused kernels) as the default-settings row. Lane page `bench/h2h-2026-10-02/tc1c/README.md`.

### Read: SC2 (#846) -- request-level serving on one RTX 5090. vLLM and SGLang hold the SLO to 8 req/s; e4b's `serve_paged` holds it only at 1 req/s

- **What ran.** `sc2-5090-1` ($2.349, a 400 W-capped 5090) drove e4b `serve_paged` (SC1's int4 levers, plus the NF4
  default as a labelled row), vLLM 0.30.0, SGLang 0.5.20 and llama.cpp `552f18f` through one driver. All 5,060 requests
  were VALID.
- **The rule's verdicts.**
  - Capacity ceilings (attainment ≥ 0.95 in both draws): vLLM 8, SGLang 8, llama.cpp 2, e4b_int4 1, e4b_nf4 1 req/s.
  - **Q4 and Q5 REFUTED.** Q6 REFUTED by UNSTABLE rows; no row is INVALID.
  - **Q1–Q3 UNREAD**, because e4b_int4's serial row is UNSTABLE (p50 TTFT 0.293 vs 0.246 s). Seen, not read: TTFT
    5.18× vLLM's, TPOT 1.29×.
- **Why** (post hoc, from e4b's own trace; `bench/sc2/sc2_trace.py`).
  - Every SLO miss is a TTFT miss.
  - A request's decode time fits 6.15 ms/token + 0.356 s per other request's prefill landing during it (R² 0.985).
  - A 512-token prefill step stalls every running decode for about a third of a second, so e4b's capacity is set by
    prefill, not decode.
- **The lever it names:** cheaper prefill, or prefill that doesn't stall decode.
- **Total and read page.** SC2 cost $4.225 across 3 receipts. Read page: `bench/h2h-2026-10-02/sc2/README.md`.

### TC2 amendment 8 registered: Mixtral at default settings with the dense route, and Qwen3.6's micro-batch-1 pair (P24–P28). Two RTX 5090s

- **Box M** (`tc2mixtralres`): Mixtral alone, every e4b arm resident at e4b's default settings, so grouped-nf4-gemm's `auto` takes its
  dense route (grouped-nf4-gemm#463). It predicts e4b's peak at most 31.6 GB and Unsloth/e4b in [0.75, 1.20].
- **Box Q** (`tc2qwen35mb1`): Qwen3.6 alone, resident, the primary pair at micro-batch 1 × accum 8, two draws a side: e4b with
  `E4B_ABSMAX_DQ=1 TRAIN_FROZEN_4BIT=1` against Unsloth with the family's expert target parameters. It predicts each e4b peak at most
  31.8 GB, Unsloth/e4b in [1.50, 2.60] and a step-0 gap of at most 0.02 nats.
- **Harness.** `tc2_big_family` reads two knobs only box Q's token sets: `TC2_PRIMARY_RECIPE` (the four primary arms' recipe; an `mb1`
  primary pair runs no `_mb1` secondary) and `TC2_E4B_ARM_ENV` (extra environment on e4b arms only). Every other token runs as before.
- **Reducer.** A primary pair off the field recipe's micro-batch names its recipe on its position line, and the lane's P4, whose e4b leg
  reads the field recipe's step, is UNTESTED on such a box. Two new self-test cases, 77 in all. `bench/tc1/TC2-PREREG.md` amendment 8.

### Read: TC1 amendment 22 — grouped-nf4-gemm's dense route is 0.651× the fused kernels' step on Mixtral and 2.947× on Qwen3-30B-A3B (P38, P40 HELD; P39 FALSIFIED)

- **Mixtral-8x7B** (`tc1-5090-62`, $0.51): dense/fused **0.651** [0.648, 0.654], 5.52 → 3.59 s/step, held-out Δ −0.0021, peak unchanged.
- **Qwen3-30B-A3B** (`tc1-5090-59`, $0.45): **2.947** [2.915, 2.979], far slower. The per-expert loop adds about 295,000 launches to a
  launch-bound step. Held-out Δ −0.0001.
- **Decision.** Off sm_90, grouped-nf4-gemm's `auto` takes the dense route for calls with at most 16 present groups (grouped-nf4-gemm#463).
  Rows `e4b.train.dense-route.{mixtral,qwen3}.5090.2026-10-04`; lane page `bench/h2h-2026-10-02/tc1/README.md`.

### MG1 amendment 1, before any box: the regression anchors re-read what was licensed (bench and prereg only)

- **OLMoE re-reads `OLMoE-1B-7B-0924-Instruct` @ `7f1c97f4`, the checkpoint tp1 licensed.** The registration had named the base
  model, which no licence reads.
- **Gemma-4-26B-A4B-it @ `4d7ae498` joins as a second regression anchor.** #1048 changed its fused RMSNorm numerics, and its tp1
  median (0.04742 against a 0.05 band) has the least margin of any licensed family. P6: PASS with the median at most 0.0574.
- The guard is 6 h, about $5.10, still under the $15 no-ask tier.

### The Qwen training stack, by structure: what another MoE family inherits (moe-generalize; `docs/MOE_RUNTIME_PORTABILITY.md`)

- **Why.** The expert-side work behind Qwen3-30B-A3B's fused training step already applied to any family `ExpertsLoRA` wraps:
  host-sync removal, pinned staging, host reuse, the lean delta, the tile rule, the dgrad kernel and the grouped_mm route. The
  glue around it did not. It was matched on Qwen's names or Qwen's formulas, so other families silently got less, and one got a
  wrong rotation path.
- **Fused frozen RMSNorm** now probes each norm's own formula and the kernel computes that formula. The candidates are
  Llama-rounded `w·round(n)`, Gemma-4's fp32 `w·n`, Qwen3.5/3.6's centered fp32 `(1+w)·n`, and `(1+w)·round(n)`. The closest
  candidate inside the old tolerance wins, and an exact one wins outright.
  - Qwen3.5/3.6's norms were all left on the composite before.
  - **Gemma-4's and Nemotron-H's numerics change slightly.** Their norms multiply in fp32. `main`'s probe accepted them as the
    Llama formula, inside its tolerance, and fused them with the Llama rounding. They now run their own fp32 multiply, which is
    closer to the reference composite.
  - Llama-formula families take the same kernel path as before: Qwen3, OLMoE, Mixtral, Granite-3.1, Granite-4.0-H, LFM2 and
    ERNIE (each checked on transformers' module or in a ladder's census).
  - Fallback calls are counted (`RMSNORM_TRAIN_STATS["fallback_calls"]`).
- **Fused RoPE** needs a patch-time semantics probe that reproduces the composite exactly. ERNIE-4.5's `apply_rotary_pos_emb`
  has the Hugging Face signature but rotates interleaved pairs; only its fp32 tables kept it off the kernel. Refusals are listed
  (`ROPE_TRAIN_STATS_REFUSED`).
- **MoE-activation retention** (`E4B_MOE_KEEP_LAYERS`, opt-in) matches every checkpointed layer that holds an expert stack,
  rather than `self_attn` + `mlp` by name.
  - Each other weighted child stays checkpointed on its own: attention, Gated DeltaNet, short conv, Mamba, dense or shared MLP.
  - Granite, LFM2, Jamba and Nemotron-H went from 0 kept layers to all; Qwen3.5/3.6 from 10 of 40 to 40.
  - A Qwen3-MoE layer is unchanged (test-pinned).
  - A layer whose experts are offloaded is never kept: offload's backward needs the recompute.
- **`enable_fast_train` has a storage gate and an engagement census** (`FAST_TRAIN_STATS`).
  - A positively known non-NF4, non-4-bit, non-64-blocksize or K%64 base is skipped to the reference path and counted in
    `skipped`, rather than decoded through the NF4 table. A partially engaged arm says so.
  - `recurrent_fallbacks` names the hybrid families' recurrent blocks that run transformers' PyTorch path because mamba-ssm,
    causal-conv1d or flash-linear-attention is absent, with a `RuntimeWarning`. On an RTX A2000, installing the kernels cut
    Granite-4.0-H's fused step by 36 % of device time.
- **Attention projections:** `out_proj` is admitted beside bias-free `q_proj` and `k_proj`. LFM2's 24 attention projections now
  get attention LoRA and `TRAIN_ATTN_4BIT`; short-conv, Mamba and GDN mixers stay excluded.
- **`python -m experts4bit_qlora.train`** picks trainables by owning module (`ExpertsLoRA`, `LoRALinear`) rather than by
  parameter name. Nemotron-H's attention adapters (under `mixer.`) were frozen before.
  - The router is the one `[num_experts, hidden]` weight beside each expert stack (`lora.router_param_ids`). Only Qwen-style
    `mlp.gate` routers were found before, so `TRAIN_ROUTER=1` trained no router on Granite, Gemma-4, Nemotron-H or LFM2.
  - Attention subtrees are not searched. A block with more than one candidate is listed in `lora.ROUTER_AMBIGUOUS`, and the
    trainer refuses `TRAIN_ROUTER=1` there rather than guess. `TRAIN_ROUTER=1` with no router found also refuses.
- **Tests and harness.**
  - The fused training path is under the composition contract on the candidate families' scaled geometry, including
    Nemotron-H's non-gated relu² stack (its first GPU run), with dgrad-kernel engagement asserted.
  - `bench/moegen/` holds the per-family ladder harness (`ladder.py`: rungs interleaved A..Z Z..A, device-busy time, the
    engagement census), the layer-slice tool, the MG1 preregistration, box runner and reducer, and the RTX A2000 receipts.
  - [`bench/moegen/RESULTS-moegen-ladders.md`](bench/moegen/RESULTS-moegen-ladders.md) reads them. These are informational, not
    positions. Every family's fused rung engages on every MoE layer with no dgrad loop, and the Qwen-general switches cut host
    syncs by about nine in ten everywhere. tp1's arm driver passes LFM2-8B-A1B (d_final 0.0088) and Granite-4.0-H-tiny (0.0062)
    on the A2000, outside tp1's anchor band, so these license nothing.
  - `bench/tc1/tc1_drive.sh`'s liveness check follows `TC1_RUNNER`.

### Read: TC2 amendment 7 — on the current code e4b is faster than HF and axolotl on Granite and than Unsloth on OLMoE; the absmax double-quantized brings e4b's Mixtral peak level with Unsloth's; Qwen3.6 trains resident at micro-batch 1 (P15, P17 HELD; P16 half HELD; P18–P21, P23 FALSIFIED; P22 UNTESTED)

- **Box S** (`tc1-5090-54`, $2.59): HF/e4b **1.299** and axolotl/e4b **1.150** on Granite, Unsloth/e4b **1.821** on OLMoE. All three
  are e4b faster; on 2026-10-02 they read 0.971, 0.903 and 1.201. Parity PASS, and the pairs read as before.
- **Box D** (`tc1-5090-55`, $1.14, `E4B_ABSMAX_DQ=1`): Mixtral's e4b peak is 29.03 GB, against 31.07 with the fp32 absmax and 29.13
  for Unsloth. Unsloth/e4b reads 0.542 on this desktop-CPU host, outside P18's band, so the switch stays opt-in. Qwen3.6 still OOMs.
- **Box F** (`tc1-5090-56`, $3.00, plus `--frozen-4bit`): Qwen3.6 trains resident at micro-batch 1 (31.36 GB, 9.24 s/step), the first
  time on a 32 GB card. Unsloth's micro-batch-1 arm took 18.49 s on the same box (one draw each, not quoted). Mixtral reads 0.644.
- Rows: three superseding positions, plus labelled `.absmax-dq` and `.fit-mb1-dq-frozen4` rows. Lane page
  `bench/h2h-2026-10-02/tc2/README.md`.

### TC1 amendment 22 registered: grouped-nf4-gemm's dense route against its fused kernels, A/B on one RTX 5090 (P38–P40) (bench and tests only)

- **Why.** TC2 amendment 7's box D read e4b's reference loop (each expert dequantized, then a dense GEMM) on Mixtral-8x7B at
  4.69 s/step against the fused kernels' 5.52 s. An RTX A2000 replay of the expert GEMMs timed the dense path at 0.13–0.58 of
  the fused kernels' time at large groups. grouped-nf4-gemm#459 adds the opt-in `GNF4_TRAIN_GEMM=dense`.
- **The box.** Two tokens on one RTX 5090, each the matched arm with `GNF4_TRAIN_GEMM=fused` against `=dense`, two draws a side
  in ABBA order:
  - `qwen3denseab`: Qwen3-30B-A3B on TC1's recipe and tokens;
  - `mixtraldenseab`: Mixtral-8x7B-Instruct at TC2's pin, field recipe and tokens, resident, with `E4B_ABSMAX_DQ=1` on both sides.
- **Engagement.** Each arm's `route_ab` record must show the route its tag names: dense forward and dgrad calls counted on the
  dense side, no dense forward on the fused side, and on Mixtral `absmax_dq` on both. A side that misses reads VOID.
- **Predictions.** P38: Mixtral dense/fused s/step in [0.55, 0.90]. P39: Qwen3-30B-A3B in [0.85, 1.15]. P40: on each family the
  two routes' mean held-out at N within 0.01.
- **Decision.** P38 and P40 HELD: grouped-nf4-gemm's `auto` takes the dense route off sm_90 for calls with at most 16 present
  groups. P39 HELD at or below 0.95: dense for every group count. Otherwise fused stays for that family.
- **Harness.** `tc1_run.sh` gains both families (no Unsloth venv on either token). `tc1_reduce.py` gains the registration,
  the engagement predicate and the P38–P40 scorer; its self-test goes from 70 to 75 cases. `bench/tc1/TC1-PREREG.md`
  amendment 22.

### Lane K29's driver retries its fetch (bench and tests only)

- **Why.** `k29-5090-1` ($0.06) drew a cgroup v2 box and ran to TP_DONE. Then the driver's single `rsync` died on a
  closed connection, the driver exited 22, and the launcher tore the box down with the results still on it.
- **What.** The fetch retries `rsync` up to 4 times with backoff, then falls back to `tar` over a fresh ssh, all
  inside the run's deadline. `tests/test_k29_lane.py` pins it. The registered rerun (a harness fault, inside the
  $3.00 ceiling) is `k29-5090-2`.

### Lane K29's runner (grouped-nf4-gemm#71): what a pinned host byte costs a cgroup v2 container (bench and tests only)

- **Why.** grouped-nf4-gemm#457 models pinned-tier sizing as PyTorch's power-of-two rounding (measured on cgroup v1),
  replacing the flat 1.9. It hands back more rows, so its release waits for a cgroup **v2** reading. Rented boxes are
  v2.
- **What.** `bench/k29/`:
  - the box script reads forensics, checks for cgroup v2 (STOP-1) and runs the probe at ten sizes ×2, pinned and
    pageable;
  - the driver is P55's shape;
  - the probe is byte-identical to grouped-nf4-gemm's `kernel/receipts-71/pinned_charge_probe.py`;
  - the reducer implements the registered rule (VOID / CONFIRMED / PREMIUM / MIXED) with a 10-case self-test.
- **Rehearsed at $0** on the QNAP A2000 (cgroup v1). STOP-1 fired, the reducer returned VOID, and v1's 18 pinned rows
  read r in [1.0043, 1.005]. The receipts are in `bench/k29/receipts/rehearsal-a2000-v1/`. `tests/test_k29_lane.py`
  covers the pins, the box script's contract, the self-test and the rule constants.

### #674's last question decided: the licensed Qwen3 K8 row names the software it was read on; no re-read (register note only)

- **Question.** Lane P85 left an owner decision open: should the licensed row
  (`e4b.serve.buildout.bo6c.qwen3.all-calibexp-streamed-64k.k8.2026-09-05`, read on grouped-nf4-gemm `0b25d13`) name its
  software, or be re-read on the current release? The decision is made under the owner's delegation.
- **Decision.** The row stands as read. Its notes now tie it to P85: from grouped-nf4-gemm 0.33.6 the same recipe reads
  wikitext K8 1.85065 (ppl 6.36396), 0.00049 nats lower, and P85 isolated #413's fused-append fix as the whole move.
  That is ~5% of the 0.0095-nat floor, so the licence is unchanged and no re-read is registered.
- #674's titled gap, an artifact for the calibrated attention, was already closed by #754 (2026-09-28): a hash-pinned
  attention pack with dump and licensed load. The issue closes with this note.

### Read: TC1c amendment 7 — `auto` takes the grouped_mm route on an H100 with nothing set (P22 HELD); the box is not quoted (P21 UNTESTED, P23 FALSIFIED). Amendment 8 registered

- **The box.** `tc1c-h100-11` ($2.15 invoiced) ran e4b 0.45.0 with grouped-nf4-gemm 0.37.0 and nothing set. Every e4b fused arm took
  the route (16,896 forward and 7,680 dgrad calls).
- **Not quoted.** e4b's draws were 5.1 % apart, over the 5 % rule; the pooled ratio of about 1.03 is reported, not quoted. Unsloth's
  first draw read COMPARABLE (0.0061 against a 0.0054 band). The receipt reads ALARM: its teardown took the emergency retry path,
  and the instance was proven absent.
- **Amendment 8** re-asks the same box once, the last re-ask under these rules. Lane page `bench/h2h-2026-10-02/tc1c/README.md`.

### `llms-full.txt`'s size cap raised from 500,000 to 600,000 bytes

- The bundle reached 499,973 bytes, and every read PR adds register rows to it. The cap was last raised (from 400,000) on 2026-10-01,
  for the same reason. `docs/llms-bundle.json` records both.

### P55 read (#344): the host-memory reading of the Gemma-4 load fault is REFUTED; the fault does not reproduce on the current loader

- **Run.** `p55-5090-2`, the one rerun Amendment 2 allowed, cost $0.164; the lane's total is $0.179 against its $3.00
  ceiling.
  - The host was in the failing class: an RTX 5090 with MemTotal 60.5 GiB and a cgroup limit of 58.1 GiB.
  - `google/gemma-4-26B-A4B-it` @ `4d7ae49` loaded cleanly three times: A unarmed; B with staged synchronisation and
    `CUDA_LAUNCH_BLOCKING=1`, every stage clean; and C with the cgroup's headroom down to 5.72 GiB against the
    46.48 GiB shard.
- **The rule.** P1 is REFUTED. P2 to P4 are undecided because nothing failed. The memory-headroom reading is dead
  alongside the driver and GPU leads, and #344 stays open as host-specific and unreproduced. No second box (STOP-3).
- **What this is not.** It does not show the September failures were not memory-related on those two machines (they
  cannot be rented). The failing runs used e4b as of 2026-09-03, so the current loader is what was tested.
- **Docs.** STATUS's #344 entry, the README's model-coverage line and the bitsandbytes solutions page now say
  "unreproduced", not "fails". `bench/p55/RESULTS-p55.md` is generated by the reducer from
  `bench/p55/receipts/p55-5090-2/`. Run 1's receipts (`p55-5090-1`, rc 11, $0.015) are beside it.

### TC2 amendment 7 registered: the small families re-read, and the big families with the absmax double-quantized (P15–P23). Three RTX 5090s

- **Box S** re-reads Granite and OLMoE on the current code. Their 2026-10-02 positions predate e4b's ~1.4× faster step.
- **Box D** runs Mixtral and Qwen3.6 resident with `E4B_ABSMAX_DQ=1`. It predicts Mixtral's peak at most 29.6 GB (was 31.07)
  with Unsloth/e4b in [0.62, 0.78], and Qwen3.6 fitting at micro-batch 2.
- **Box F** adds `--frozen-4bit` on top of D, so Qwen3.6 runs on Unsloth's 4-bit bytes. It predicts a step-0 gap of at most 0.01 nats and
  Unsloth/e4b in [1.30, 2.50]: the family's first matched position.
- **Decision.** `E4B_ABSMAX_DQ` becomes the training default only if box D holds the Mixtral prediction within 0.005 nats.
  `bench/tc1/TC2-PREREG.md` amendment 7.

### `lora.quantize_frozen_linears_4bit`: a measurement hook that puts e4b's other frozen projections in a 4-bit comparator's regime (not a training option)

- **What.** It stores every frozen, bias-free bf16 `nn.Linear` in NF4, the way `TRAIN_ATTN_4BIT` stores the attention. It keeps the
  routers, the `shared_expert_gate`s, `lm_head` and the attention projections (those stay under their own switch). On
  Qwen3.6-35B-A3B that is the 150 Gated DeltaNet projections and the shared experts' 120 gate/up/down; on Qwen3-30B-A3B and
  Mixtral it converts nothing.
- **Why.** A bitsandbytes `load_in_4bit` comparator quantizes those modules. On Qwen3.6 that starts it 0.05 nats higher on TC1's
  held-out rows than e4b's bf16 default (1.194–1.196 against 1.143), which VOIDs a same-box pair by the lane's step-0 rule. The TC1
  harness's `--frozen-4bit` (default from `TRAIN_FROZEN_4BIT`, so `TC1_E4B_ENV` reaches e4b arms only) gives a matched pair the
  same bytes.
- **Not offered.** `experts4bit_qlora.train` does not read it, and the bf16 default stays (`docs/ARCHITECTURE_SUPPORT.md`).
- **Tested.** `tests/test_frozen_linear_4bit.py`: the selection on CPU, and the conversion on CUDA (an RTX A2000: 15 passed with
  the attention-projection tests). The harness test pins the flag's order: after the attention census, before the LoRA.

### Opt-in: the frozen expert absmax stored double-quantized (`E4B_ABSMAX_DQ=1`, `compress_expert_absmax_`)

- **What.** `compress_expert_absmax_(model)` stores each ExpertsLoRA-wrapped NF4 stack's absmax the way bitsandbytes'
  `quantize_4bit(compress_statistics=True)` does: the mean as an offset, 8-bit codes, one fp32 per 256 values. That is
  3.94x fewer bytes than one fp32 per 64 weights. `E4B_ABSMAX_DQ=1` turns it on in `python -m experts4bit_qlora.train`
  (after load, before training) and in the TC1 harness (`--absmax-dq`, recorded on the receipt).
- **Why.** On a 32 GB RTX 5090, e4b's resident Mixtral-8x7B peak was 31.07 GB against Unsloth's 29.12 GB. Qwen3.6-35B-A3B
  did not fit, where Unsloth trains it at 30.47 GB. The fp32 absmax is about 2.8 GB on Mixtral, 1.8 GB on Qwen3-30B-A3B and
  2.1 GB on Qwen3.6. Unsloth trains on bitsandbytes' double-quantized statistics.
- **What reads it.** grouped-nf4-gemm's kernels are unchanged. `enable_fast_train` and the ExpertsLoRA reference loop expand one
  layer's absmax to fp32 just in time (`expert_absmax_fp32`). Expert offload, `enable_batched_train`, the hot, cold,
  pipelined, hybrid and NVMe engines, and the gpt-oss and DeepSeek-V4 per-expert forwards refuse a compressed model by name
  (`AbsmaxCompressedError`). The trainer refuses the switch with `OFFLOAD_EXPERTS=1` or `TRAIN_ARENA`.
- **Values.** On, the absmax values are bitsandbytes' double-quantized ones: lossy against fp32, and bit-identical to
  bitsandbytes' own nested path for the same stack. Off, nothing changes.
- **Tests.** `tests/test_absmax_dq.py` checks the selection, refusals and layout on CPU. It checks that the stored bytes are
  bitsandbytes' own, that the reference loop and the fused training path (both dgrad settings, on an RTX A2000) are
  bit-identical to the same code fed the dequantized absmax, and that the stored absmax bytes drop by the exact layout ratio.
  The TC1 tests cover the flag, its refusals and the frozen-base probe (SAME-BYTES against bitsandbytes' nested path).
  No speed or memory claim is registered yet.

### P55 Amendment 2 (#344): the box script installs the loader's stack and git, rehearsed end to end at $0; one rerun (bench and tests only)

- **Why.** `p55-5090-1` drew the low-RAM class (62 GB allotment, effective 59.9 GiB) and then died at the fetch, before
  any arm, on `No module named 'huggingface_hub'`. e4b's base dependencies are torch and bitsandbytes only, and the
  registered runner installed the bare package, so it could never have loaded the model. The run cost $0.015.
- **What.**
  - The runner installs as P113's does (torch held; transformers 5.17.0, bitsandbytes 0.50.2, safetensors,
    huggingface_hub, accelerate; bounded, exit 9).
  - A tripwire checks the installed commit and the loader's imports.
  - The runner installs git when the image lacks it. The lane's image does, and P113 had relied on Vast's runtime layer.
- **Rehearsed.** The whole box script ran on the QNAP A2000 in the lane's image against a 6.6 GB granite MoE. Rehearsal 1
  found the missing git. Rehearsal 2 ran with rc=0: all three arms loaded, and the reducer read the result correctly.
- **Rerun.** STOP-3 is amended for this case only: exactly one rerun, `p55-5090-2`, because `p55-5090-1` observed nothing.
  Its outcome is final.
- **Tests.** `tests/test_p55_staged_pin.py`: the install, the tripwire, and git ensured before pip.

### P55 Amendment 1 (#344): the launcher can now draw the low-RAM host class, and the lane reads the class correctly (bench and tests only)

- **Why.** P55 tests whether #344's Gemma-4 load failure (`CUDA error: invalid argument` on 2 of 6 rented 5090s) is a
  host-memory class. Preparing its launch found five defects:
  - the launcher could only rent hosts with ≥ 98 GB;
  - STOP-1 read MemTotal, which inside a Vast container is the whole host's RAM, not the allotment;
  - P4 read the host's MemAvailable for the same reason;
  - the shard's "49.9 GiB" was 49.9 GB, which is 46.48 GiB;
  - the first arm downloaded the 51.6 GB checkpoint inside its own load, unbounded and on the Xet backend.
- **What.**
  - The launch uses adertha-agents#145's host-RAM band, `[48, 72]` GB.
  - The class is the memory a process there can have, min(MemTotal, cgroup limit). The new `bench/p55/p55_ram.py`
    computes it, and it is staged and pinned.
  - P4's headroom is min(MemAvailable, limit − usage) against 46.48 GiB, and the forensics record `ulimit -a`.
  - The checkpoint is 51.6 GB, not ~12 GiB. It is fetched before the arms, bounded and with Xet disabled (exit 11 on failure).
    The launcher's estimate is $1.95, and the lane ceiling is $3.00.
- **Tests.** `tests/test_p55_staged_pin.py`, 17 tests: pins, the memory arithmetic on cgroup v1, v2 and none, and the
  reducer on synthetic receipts.

### TC1c amendment 7 registered: the H100 position at default settings on 0.45.0 / grouped-nf4-gemm 0.37.0 (P21–P23). One H100 NVL

- **Why.** Amendment 6 read the grouped_mm route at 1.030 with it forced, as a labelled row. grouped-nf4-gemm 0.37.0 makes the
  route its sm_90 default, and 0.45.0 is the first release on it. This box reads the position with nothing set.
- **Predictions.** P21 Unsloth/e4b in [0.95, 1.12]. P22 `auto` engages the route with `GNF4_TRAIN_GEMM` unset (16,896 forward and
  7,680 dgrad calls). P23 the matched set EQUIVALENT.
- **Decision.** All three held: this becomes the H100 default-settings position, superseding amendment 1's 0.817.
  `bench/tc1/TC1C-PREREG.md` amendment 7.

### Read: TC2 amendment 6 — e4b now trains Mixtral resident on a 32 GB 5090, and Unsloth is faster there; Qwen3.6 still does not fit (P11, P12, P14 HELD; P13 FALSIFIED)

- **The box.** `tc1-5090-53` ($3.03 invoiced) ran both big families with every e4b arm resident, against Unsloth resident.
- **Mixtral-8x7B.** e4b's fused path fits at 31.07 GB, where on 2026-10-02 it ran only under offload. Unsloth/e4b is **0.697**
  [0.679, 0.714]: Unsloth is faster per step, at 1.94 GB less peak and ×0.65 the energy. The pair is inside draw noise, and e4b's
  resident parity PASSES.
- **Qwen3.6-35B-A3B.** e4b still OOMs at both micro-batches. It now fails at step 2 instead of step 1. Unsloth trains it at 30.47 GB.
- **Why, as read from the receipts.** Mixtral's step is GEMM-bound on 2 of 8 large experts. e4b's extra VRAM matches its fp32
  expert absmax, about 2.8 GB against about 0.7 GB double-quantized. On Qwen3.6, add about 1.6 GB of non-routed weights kept in bf16.
- **Next levers,** each to be registered: double-quantized absmax, 4-bit non-routed projections, and a dequantize-then-GEMM route
  for large experts on sm_120. Rows `e4b.train.h2h.unsloth.mixtral.5090.2026-10-04` and `...qwen3_5.5090.2026-10-04`; lane page
  `bench/h2h-2026-10-02/tc2/README.md`.

### SC2 amendment A1 (#846): the first proof's harness defects fixed; the proof reruns

- `sc2-prove-1` ($0.92) read HARNESS_ERROR, NOT PROVED. vLLM and SGLang passed both smokes with every request VALID.
  e4b's server died on `No module named 'uvicorn'` (SC1 never needed `serve_paged`'s web stack), and llama.cpp failed
  every other request with `ServerDisconnectedError` (its server drops a connection after a streamed response, and the
  driver reused it).
- Fixes: box E installs e4b's `serve` extra pinned (`fastapi==0.141.1`, `uvicorn==0.54.0`); the driver opens a fresh
  connection per request on every engine (`force_close`); and the e4b start waits for `/health` status `ready`, not
  HTTP 200 (`serve_paged` answers 200 while loading). Each fix has a test, and the connection test fails without its
  fix. The rule, the plan and the predictions are unchanged; amendment A1 is in `bench/sc2/SC2-PREREG.md`.

## 0.45.0 — 2026-10-04 — CI on grouped-nf4-gemm 0.37.0, whose two new defaults were registered and read here: programmatic dependent launch capped to launches of at most 8 rows (lane P113: SC1's int4 serving decode 1.0404× at one request and 1.0000× at 16 on an RTX 5090, identical tokens) and the grouped_mm training route on sm_90 (TC1c amendment 6: Unsloth/e4b 1.030 on an H100 NVL, a labelled row)

**0.45.0.** No default in this package changes. Two change in grouped-nf4-gemm 0.37.0, each by a rule registered here, and CI now tests against 0.37.0's commit.
- **Serving: `GNF4_PDL` is on, capped at 8 activation rows.**
  - Lane P113 read it on SC1's int4 serving configuration (Qwen3-30B-A3B, one RTX 5090) under decode-only timing. Tokens were identical on every row, and decode ran **1.0404×** as fast with one request and **1.0000×** with 16 (`e4b.serve.p113.gnf4-pdl-capped.qwen3-int4.5090.2026-10-04`). Uncapped, it cost 16 requests 2.1 %.
  - The default NF4 server reaches only two of the switched kernels, so the switch was not read there. P112, the first served read, closed VOID twice.
- **Training: `GNF4_TRAIN_GEMM=auto`** takes grouped-nf4-gemm's grouped_mm route on an sm_90 card and its fused kernels elsewhere. Values change on sm_90 only; `GNF4_TRAIN_GEMM=fused` restores them.
  - TC1c amendment 6 read Unsloth/e4b at **1.030** [1.016, 1.045] on the full Qwen3-30B-A3B training step on an H100 NVL with the route on, and **1.325** with MoE activations also kept. Amendment 4's three conditions for the default all held, the held-out loss by a thin margin.
  - Both readings are LABELLED rows (`e4b.train.h2h.unsloth.qwen3.h100.2026-10-04.route-v2`, `….moe-keep-route-v2`). The H100 position of record stays amendment 1's 0.817 (Unsloth faster) until a default-settings box re-reads it on this release.
- The `[fast]` floor stays `>=0.30.0`. A fresh install gets 0.37.0; an existing environment gets both defaults with `pip install -U grouped-nf4-gemm`.

**Also in this release:**
- **TC1c amendments 2–5** on the H100 NVL:
  - amendment 2: e4b keeping its MoE activations at Unsloth/e4b 1.100 (a labelled row);
  - amendment 3: dequantize + `torch._grouped_mm` running the recorded GEMM calls in 0.50–0.60 of the fused kernels' time (a kernel replay);
  - amendment 4: the route as first shipped making e4b slower, because of its dequant kernel;
  - amendment 5: the fused kernels' own configs changing nothing worth taking.
- **SC2 registered** (#846): request-level serving, `serve_paged` against vLLM, SGLang and llama.cpp under Poisson arrivals.
- **TC2 amendment 6 registered** (#1032, merged during the release PR): Mixtral and Qwen3.6 trained with every e4b arm resident on one RTX 5090, against Unsloth resident (P11–P14).
- **Lane K28's runner** (bench and tests), and a correction pricing five TC boxes at Vast's invoiced cost.
- **Registrations with no entry of their own:** TC1c amendments 3 (#1018), 4 (#1020), 5 (#1021) and 6 (#1028). Also #1011 (tests only): the GPU-class refusal snippet runs in a temp dir.

### TC2 amendment 6 registered: Mixtral and Qwen3.6 with e4b RESIDENT on the 32 GB card (P11–P14). One RTX 5090

- **Why.** On 2026-10-02 e4b trained neither big family resident on a 5090. Qwen3.6 OOMed at both micro-batches, and Mixtral ran only
  under expert offload, where Unsloth (resident) steps about 5.3× faster. e4b's training memory has changed since (the lean LoRA delta
  on by default; the combine saving bf16, not fp32).
- **The box.** Token `tc2resident`: Mixtral, then Qwen3.6, with every e4b arm resident against Unsloth resident, two draws a side, the
  e4b reference, and the micro-batch-1 pair on an OOM. Rows earlier boxes hold are skipped.
- **Predictions.** P11 Mixtral fits resident at micro-batch 2 (at most 31.5 GB); P12 Unsloth/e4b on Mixtral in [0.45, 0.95], Unsloth
  faster on a GEMM-bound step; P13 Qwen3.6 OOMs at micro-batch 2 and fits at micro-batch 1; P14 the pairs EQUIVALENT, parity PASS
  where the reference fits.
- **Reducer.** Mixtral's footprint line and P5 (the offload pair's prediction) are drawn only when e4b's anchor ran under offload (a
  new self-test case, 70 in all). `bench/tc1/TC2-PREREG.md` amendment 6.

### Read: TC1c amendment 6 — with the dequant at bandwidth the grouped_mm route makes e4b faster than Unsloth on an H100 (P18, P19, P20 HELD)

- **What.** Amendment 4's two boxes again, with grouped-nf4-gemm at #452's merge (`81706a9`), whose route dequant is bit-equal and
  about 8.5× faster on the card. Box R `tc1c-h100-9` ($2.65 invoiced) put every e4b arm on the route; box K `tc1c-h100-10`
  ($2.76) also kept all 48 layers' MoE activations.
- **Positions.** Unsloth/e4b **1.030** [1.016, 1.045] on the route (was 0.664 with the old dequant; 0.817 with the fused kernels) and
  **1.325** [1.296, 1.356] with keep + route (was 0.934; keep alone 1.100). e4b is faster per step on the H100 in both. The matched set
  is EQUIVALENT on both boxes.
- **Why.** The dequant now takes 256 ms of device time per step (0.222 ms per call), against 2,178 ms before. The step is host-bound
  now: device-busy 0.454.
- **Decision.** Amendment 4's default rule holds, on a thin held-out margin (Δ 0.0041 on two-draw means; two of four draw pairings
  exceed 0.005). the route qualifies as grouped-nf4-gemm's sm_90 default (grouped-nf4-gemm#454). The readings are labelled rows `...h100.2026-10-04.route-v2`
  and `.moe-keep-route-v2`. Lane page `bench/h2h-2026-10-02/tc1c/README.md`.

### P113 read (RTX 5090): CAP_DEFAULT — programmatic dependent launch, capped to launches of at most 8 rows, decodes SC1's int4 serving step 4.0 % faster with one request, at no cost with sixteen, tokens identical (#1015)

Register: `e4b.serve.p113.gnf4-pdl-capped.qwen3-int4.5090.2026-10-04`; `bench/p113/RESULTS-p113.md`.
- **The reading** (`p113-5090-1`, $0.4710). The run used six palindromic arms on SC1's int4 configuration with
  decode-only timing. Capped (`GNF4_PDL=1 GNF4_PDL_MAX_ROWS=8`) the ratios were **1.0404** at B=1 and
  **1.0000** at B=16; uncapped, 1.0401 and 0.9787. The B=1 step went from
  4.38 to 4.21 ms. Tokens were identical in every arm, and the self-pairs read within
  0.18 %.
- **The predictions.** All eight held; they were written after P112 run 1's arms.
- **What follows (registered).** grouped-nf4-gemm's next release turns `GNF4_PDL` on by default, capped at 8, and this
  repository's CI moves to it.

### P113 registered (#1015): PDL everywhere or capped to small launches, on SC1's int4 serving step, under decode-only timing. One RTX 5090

- **Why.** P112 closed VOID with no reading. Its first run's arms showed grouped-nf4-gemm's `GNF4_PDL=1` helping B=1 and
  costing B=16. grouped-nf4-gemm#453 adds `GNF4_PDL_MAX_ROWS` to keep PDL for launches of at most `n` rows.
- **The lane.**
  - **Subject:** P112's.
  - **Arms:** six in a palindrome, OFF1 ALL1 CAP1 CAP2 ALL2 OFF2 (off; `GNF4_PDL=1`; `GNF4_PDL=1 GNF4_PDL_MAX_ROWS=8`).
  - **Timing:** decode-only, each pass's wall minus its largest ttft (SC1's A10).
  - **Accounting:** P112 Amendment 1's.
- **The rule** (`bench/p113/p113_reduce.py`, 22 cases): VOID / NOISY / FUNCTION_FAIL / **ALL_DEFAULT** (B=1 ≥ 1.02 and
  B=16 ≥ 1.00) / **CAP_DEFAULT** (B=1 ≥ 1.02 and B=16 ≥ 0.99) / NONE. A default verdict moves grouped-nf4-gemm's default
  in its next release.
- **Budget.** Runs are priced with the checkpoint download, about $1.95 each; the lane ceiling is $4.00. There is no
  proving rental (P112's reason).

### Read: TC1c amendment 4 — grouped-nf4-gemm's grouped_mm route as shipped makes e4b slower on an H100 (P12, P13 FALSIFIED; P14 HELD)

- **What.** Two H100 NVL boxes ran with every e4b arm on `GNF4_TRAIN_GEMM=grouped_mm` (grouped-nf4-gemm#450), at $2.49 invoiced each.
- **Box R,** route alone: Unsloth/e4b **0.664** [0.650, 0.678], e4b 3.932 s/step. That is slower than its default 3.146 s (0.817).
- **Box K,** route + MoE activations kept: **0.934** [0.916, 0.953], e4b 2.771 s. That is slower than keep alone, 2.343 s (1.100).
- **Numerics.** The matched set stays equivalent on both (P14 HELD).
- **Diagnosis.** The route's dequant kernel took 1.89 ms per call: 2,178 ms of device time per step against the fused kernels' 1,424 ms.
  That is about 4× slower than the bitsandbytes dequant amendment 3's replay timed, and the A2000 measures the same gap (2.9–4.1×).
- **Decision.** The route stays opt-in, and the readings are labelled rows (`...h100.2026-10-04.route`, `.moe-keep-route`) beside the
  default-settings and keep rows. A faster dequant kernel is the next change. Lane page `bench/h2h-2026-10-02/tc1c/README.md`.

### Read: TC1c amendment 5 — the fused kernels' own configs on the H100 change nothing worth taking (P15, P16 FALSIFIED; P17 HELD, near-exact)

- **What.** `tc1c-h100-8` (H100 NVL, $0.33 invoiced) swept grouped-nf4-gemm's fused forward (25 configs) and dgrad (9) on the 128
  recorded real-router calls.
- **Forward.** The default stays best: 0.985 re-run, every other config 0.999–1.82×. bf16 MMA is 2.3–3.5× slower. BLOCK_K 128 fits
  but does not help.
- **dgrad.** 64/128/64/w4 reads 0.853, a near-exact config (5e-5 Frobenius); the best bit-identical config reads 0.988.
- **Decision.** The gap to `torch._grouped_mm` is structural, so the grouped_mm route (TC1c amendment 4) is the H100 path, and the
  dgrad config is recorded, not taken. Lane page `bench/h2h-2026-10-02/tc1c/README.md`.

### P112 closed VOID (#1015): no reading on `GNF4_PDL` for the int4 serving step. Seen but not read: B=1 3.7 % faster and B=16 1.5 % slower, with identical tokens

- **Run 1** (`p112-5090-1`, $0.3210) VOIDed on the lane's own launch accounting (Amendment 1).
- **Run 2** (`p112-5090-2`, $1.0210 including the checkpoint download) confirmed the accounting fix: compile launches
  equal compiled variants, and every PDL launch is accounted for. It then VOIDed on a decode slope: W16 prefill jitter
  on a shared host swamped the slope's 1.2 s of decode.
- **Why the lane closes.** A third run would breach the registered $2.00 ceiling, which was priced without download
  charges, and a reading needs decode-only timing.
- **In both runs,** tokens were identical in every arm. The default NF4 server's census showed only `swiglu_rows` and
  `combine_rows` among the switched kernels.
- **Run 1's arms, not a reading:** W1 ratios 1.0393 / 1.0366 and W16 0.9851 / 0.9854.
- **What follows.** `GNF4_PDL` stays opt-in. A new lane reads PDL off, PDL on everything, and PDL at small row counts
  only, under decode-only timing. `bench/p112/RESULTS-p112.md`.

### P112 Amendment 1: the first reading VOIDed on its own launch accounting; the accounting is fixed and the lane reruns

- **What happened.** `p112-5090-1` ($0.3210) read VOID on engagement: 20 of 6,927 switched launches in each on arm
  showed no PDL.
- **Why.** Triton 3.4 hands the launch hook no function handle on the launch that compiles a variant, and the box
  counted those launches as PDL-less. The shortfall was one per compiled variant in every kernel.
- **The fix.** The box now counts compile launches and compiled variants separately (`p112_reduce.py`, 21-case
  self-test), and the driver no longer fetches the box's grouped-nf4-gemm clone.
- **Seen in the arms (not a reading).** Tokens were identical everywhere. g1 = 1.0366 (B=1 step 4.44 → 4.28 ms) and
  g16 = 0.9851 (B=16 step 9.05 → 9.19 ms), so the rule's later steps would read SLOWER. The default NF4 server's census
  shows only `swiglu_rows` and `combine_rows` among the switched kernels.
- **The rerun** `p112-5090-2` predicts SLOWER, written after seeing these numbers. Receipts are in
  `bench/p112/receipts/p112-5090-1/`.

### P112 registered (#1015): does `GNF4_PDL=1` decode SC1's int4 serving configuration's tokens exactly, and faster? One RTX 5090

- **Why.** grouped-nf4-gemm's lane K28 read programmatic dependent launch LEVER on the served B=1 layer's gnf4 kernels
  (0.323 µs saved per kernel, bit-identical; grouped-nf4-gemm#451). P112 is its registered served read.
- **The subject.** The default graph server with SC1's int4_sched levers. These are RTN int4 experts and attention, the
  fused T1 glue, the router epilogue and the fused qkv, verbatim from `bench/sc1/sc1_run.sh`, and they are where the
  913 switched kernels of SC1b's census run. e4b's default NF4 server reaches only `swiglu_rows` and `combine_rows`,
  by its defaults; the box measures that in a census after the arms.
- **The arms.** Four ABBA arms (P0 off, P1 on) through P109's W16 and W1. A launch hook proves that every switched
  kernel in P1's build carried PDL.
- **The rule** (`bench/p112/p112_reduce.py`, 17-case self-test): VOID / NOISY / FUNCTION_FAIL / SLOWER /
  **DEFAULT_ON**. On DEFAULT_ON, grouped-nf4-gemm turns `GNF4_PDL` on by default in its next release.
- **No proving rental.** The proof model (Granite) cannot run the Qwen3-only fused qkv; this was seen on the NAS A2000.
  The reading runs under a 1.0 h guard. `tests/test_p112_staged_pin.py` pins the lane.

### Correction: five TC boxes of 2026-10-04 at Vast's invoiced cost (read pages and the host-reuse register note)

- **What was wrong.** The reads of TC1 amendments 20 and 21 and TC1c amendments 1–3 quoted each box's receipt cost. That figure was the
  GPU rate × measured runtime. Vast's invoice also bills storage, download and time from create.
- **Invoiced totals,** against the receipt figures quoted:

  | box | run | invoiced | receipt figure | breakdown |
  |---|---|---|---|---|
  | `tc1-5090-51` | amendment 20 | **$3.14** | $0.36 | download $2.57: 65.7 GB at $0.039/GB on machine 150527 |
  | `tc1-5090-52` | amendment 21 | **$3.11** | $0.35 | download $2.57, same host |
  | `tc1c-h100-3` | TC1c amendment 1 | **$4.42** | $4.33 | |
  | `tc1c-h100-4` | TC1c amendment 2 | **$2.97** | $2.80 | |
  | `tc1c-h100-5` | TC1c amendment 3 | **$0.31** | $0.21 | |

  In total, $13.96 invoiced against $8.05 recorded.
- **What changed since.** The launcher now records invoiced cost (adertha-agents #142). It also stops buying hosts that bill download
  above $0.011/GB (#141), which excludes 150527.
- **What did not change.** No measured number moved. The 0.44.0 entries that quote the old figures stand as written, with this
  correction beside them.

### SC2 registered (#846): request-level serving, e4b's `serve_paged` against vLLM, SGLang and llama.cpp under Poisson arrivals

- **What it asks.** SC1 timed decode loops at fixed batches; SC2 drives each engine's own OpenAI `/v1/completions` with ONE client
  and ONE request plan: identical token-id prompts (64 rows of 512 wikitext-2 tokens), `max_tokens` drawn from [64, 256], greedy,
  `ignore_eos`, streaming, concurrency capped at 16, prefix caching off. Scheduling stays native. It reads TTFT, TPOT and
  SLO attainment (the share of requests VALID with TTFT ≤ 1.0 s and TPOT ≤ 100 ms; goodput = attainment × rate) serially
  and at Poisson rates of 1, 2, 4 and 8 req/s, two draws each, and each engine's capacity ceiling. Predictions Q1–Q6 are
  in `bench/sc2/SC2-PREREG.md`; no position sentence comes from SC2 alone.
- **The instrument.** `bench/sc2/sc2_driver.py` (tested against `serve_paged`'s real app and over real sockets),
  `sc2_prompts.py`, `sc2_reduce.py` (self-tested on 10 cases) and box E, `sc2_box_e.sh`, which `sc1_run.sh` sources with
  `SC1_BOX=E` so the lane reuses SC1's installs, checkpoints, bake and sampler. The SC2 files are staged and pinned like SC1's.
- **Budget.** Proof guard 1.25 h ≤ $0.94; reading guard 3.0 h ≤ $2.25; expected about $2.5.
### Read: TC1c amendment 3 — on an H100, dequantize + `torch._grouped_mm` runs e4b's recorded GEMM calls in 0.50–0.60 of the fused kernels' time

- **What.** `tc1c-h100-5` (H100 NVL, $0.21) replayed the 128 unique fused forward and dgrad calls of e4b's training step, recorded
  through the real router at Qwen3-30B-A3B's shapes.
- **Reading.** Forward (dequant + grouped_mm) / fused is **0.596**, and dgrad **0.494**. The grouped GEMM alone is 0.20 / 0.16, so
  the dequantize is two thirds of the route.
- **Numerics.** Every call is within 0.0024 relative Frobenius error of the fused output. P9, P10 and P11 HELD.
- **Decision.** By the registered rule, grouped-nf4-gemm takes the route as an sm_90 opt-in, and its full-step value is a separate
  box. It is a kernel replay, not a position, so no register row changes. Lane page `bench/h2h-2026-10-02/tc1c/README.md`.

### Lane K28's runner (#1015): the GNF4_PDL decode-chain bench's box side (bench and tests only)

- **What.** `bench/k28/` drives grouped-nf4-gemm's K28 (`kernel/PREREG-k28-pdl-decode-chain.md`). It is K27's runner with
  the bench and the premise replaced.
- **The premise.** grouped-nf4-gemm's `kernel/test_pdl.py`, compiled on the card. It must report 23 passed and none
  skipped, because its sm_90+ tests are the on-card bitwise and engagement checks.
- **The tripwire.** It refuses unless:
  - the installed kernel package carries the `GNF4_PDL` switch and its inline-PTX preamble;
  - Triton has `launch_pdl`;
  - the card is sm_90+, and the switch reads as active there.
- `tests/test_k28_staged_pin.py` pins the runner's bytes and shape, the exit codes and the driver's dry run.

### Read: TC1c amendment 2 — on an H100 NVL, e4b keeping its MoE activations is faster per step than Unsloth (1.100, a labelled row)

- **What.** `tc1c-h100-4` ran on the same machine as amendment 1's box, for $2.80. Every e4b arm used
  `E4B_MOE_KEEP_LAYERS=all NF4_QLORA_COMPACT_DELTA=1 GNF4_HOST_REUSE=1` (gradients identical); HF and axolotl were skipped.
- **Reading.** Unsloth/e4b is **1.100** [1.088, 1.111]: e4b faster per step (2.343 against 2.577 s), at 9.81 GB more peak VRAM and
  about equal energy. P7 HELD and P8 HELD (matched set inside the draw noise). e4b's step fell 3.146 → 2.343 s against amendment 1.
- **Labelled.** It is a labelled row, `e4b.train.h2h.unsloth.qwen3.h100.2026-10-04.moe-keep`, quoted beside the default-settings
  0.817 (Unsloth faster) and never in place of it. Lane page `bench/h2h-2026-10-02/tc1c/README.md`.

## 0.44.0 — 2026-10-04 — one KV-table selection per decode step by default (`E4B_KV_STEP_SELECT`; lane P111: identical tokens, 3.6 % faster with 16 concurrent requests); CI on grouped-nf4-gemm 0.36.0, whose per-pass host reuse steps e4b's fused training at 0.933 / 0.951 on an RTX 5090; the matched-work training positions after #945: Unsloth/e4b 1.997, axolotl/e4b 2.775 on a 5090

**0.44.0.** Two defaults move, one here and one in the kernel package. Both are value-identical.
- **Serving: `E4B_KV_STEP_SELECT` is on.** A decode-graph bucket selects every layer's KV-table rows once per step instead of once per layer.
  - P111 measured the default `serve_paged` server (Qwen3-30B-A3B NF4, one RTX 5090): identical tokens on every row, 3.6 % faster with 16 concurrent requests (772 → 800 tok/s) and 1.2 % with one.
  - `E4B_KV_STEP_SELECT=0` keeps the per-layer form.
- **Training: grouped-nf4-gemm 0.36.0's `GNF4_HOST_REUSE` is on.** TC1 amendment 20 measured e4b's fused training step at 0.933 (shipped arm) / 0.951 (matched arm) of the flag off, with held-out loss unchanged.
  - CI now tests against 0.36.0's commit.
  - The `[fast]` floor stays `>=0.30.0`. A fresh install gets 0.36.0; an existing environment gets the reuse with `pip install -U grouped-nf4-gemm`.

**Also in this release:**
- **TC1 amendment 19** read the matched-work training positions on an RTX 5090 with e4b at every default from amendments 10–15: Unsloth/e4b **1.997** and axolotl/e4b **2.775**, against 1.437 and 1.416 before #945.
- **TC1c amendment 1** read the H100 NVL position at Unsloth/e4b 0.817. Unsloth is still faster per step on that card.
- **`E4B_MOE_KEEP_LAYERS` (opt-in)** keeps MoE activations instead of recomputing them under gradient checkpointing.
  TC1 amendment 21 read it at 0.835 / 0.926 of recomputing on a 5090.
- **The fused training forward** gathers its routing weights with a scatter backward (values identical).

### Read: TC1 amendment 21 — keeping MoE activations steps e4b at 0.835 / 0.926 on a 5090 (P35, P36, P37 HELD)

- **What.** `tc1-5090-52` (AMD EPYC 7C13, $0.35) ran e4b against itself with whole-layer gradient checkpointing against
  `E4B_MOE_KEEP_LAYERS=n` with `NF4_QLORA_COMPACT_DELTA=1`.
- **Shipped arm**, 32 of 48 layers kept: **0.835** [0.823, 0.849], peak 24.58 → 29.09 GB.
- **Matched arm**, 16 kept: **0.926** [0.920, 0.932], peak 27.14 → 29.40 GB.
- **Memory and quality.** About 141 MB per kept layer. Held-out moved within +0.0037 / +0.0014. All eight arms VALID.
- **Decision.** By the registered rule it is the recommended setting where the headroom exists (README "Which door?",
  `docs/CHOOSING.md`), and it stays opt-in. Register `e4b.train.moe-keep.qwen3.5090.2026-10-04`; read page
  `bench/h2h-2026-10-02/tc1/RESULTS-tc1-keepab.md`.

### Reads: TC1 amendment 20 (host reuse, 0.933 / 0.951 on a 5090) and TC1c amendment 1 (H100 NVL, Unsloth/e4b 0.817)

- **TC1 amendment 20** (`tc1-5090-51`, AMD EPYC 7C13, $0.36). grouped-nf4-gemm's per-pass host reuse (`GNF4_HOST_REUSE`, #444,
  values identical) steps e4b's field recipe at **0.933** [0.912, 0.954] (shipped) and **0.951** [0.922, 0.980] (matched) of the flag
  off. All eight arms are VALID and every pair stable; held-out is unchanged. P33 and P34 HELD. By the registered rule the flag becomes
  grouped-nf4-gemm's default (#446). Register `e4b.train.host-reuse.qwen3.5090.2026-10-04`; read page
  `bench/h2h-2026-10-02/tc1/RESULTS-tc1-reuseab.md`. The instance billed $0.84/h against a declared $0.69/h GPU ceiling: the 320 GB
  disk is billed on top, and the launcher's ceiling does not count it (adertha-agents#140).
- **TC1c amendment 1** (`tc1c-h100-3`, AMD EPYC 9534, $4.33). With e4b after TC1 amendments 10–15, the matched position on an H100
  NVL is Unsloth/e4b **0.817** [0.799, 0.836]: Unsloth is still faster per step there, by 1.22× (1.61× before #945). The matched set
  is EQUIVALENT. P5 and P6 HELD. e4b's step fell ×0.768 since the first H100 box, and it is now mostly device-bound on that card
  (device-busy 0.665), at ~2.4× Unsloth's device time per step. Register `e4b.train.h2h.unsloth.qwen3.h100.2026-10-04`; the
  2026-10-02 row stays, labelled as the code before #945. Lane page `bench/h2h-2026-10-02/tc1c/README.md`.

### `E4B_MOE_KEEP_LAYERS=all|n`: keep MoE activations instead of recomputing them under gradient checkpointing (opt-in); the training combine saves bf16 (values identical)

- **Why.** Hugging Face checkpoints each decoder layer whole, so the fused training step's backward re-runs every MoE forward. On a
  4-layer Qwen3-30B-A3B slice on an RTX A2000 (TC1's token rows, mb2 x accum 4), the recomputed MoE forwards were 232 of 917 ms of
  device time.
- **What.**
  - **`E4B_MOE_KEEP_LAYERS=all` or a count n.** In the last n decoder layers that Hugging Face is checkpointing, whole-layer
    checkpointing goes off and `self_attn` alone is checkpointed, with the layer's own checkpoint function. The MoE block's
    activations then live from its forward to its backward. `enable_fast_train` applies it; `disable_fast_train` unwinds it
    (`engines/moe_keep.py`). Pair it with grouped-nf4-gemm's `NF4_QLORA_COMPACT_DELTA=1` (#445): the padded LoRA delta then saves
    its input rather than its padded block.
  - **`_ScatterCombine` (always on).** The training forward's combine becomes one autograd node that saves the bf16 `down`, not its
    fp32 copy. Its backward replays autograd's own sequence, so both gradients are the same bytes.
- **Measured on an RTX A2000.** The slice above, with `NF4_QLORA_COMPACT_DELTA=1` and `GNF4_HOST_REUSE=1` on the kept arms, two runs
  each:

  | `E4B_MOE_KEEP_LAYERS` | step s (median of 6) | peak allocated GB |
  |---|---|---|
  | unset | 1.2545 / 1.2689 | 4.335 |
  | 2 of 4 | 1.1349 / 1.1383 | 4.56 |
  | all 4 | 0.9697 / 0.978 | 4.786 |

  The cost is about 113 MB of peak per kept layer at this slice's largest micro-batch, which is about 5.4 GB at 48 layers. Choose n to
  fit the card.
- **Values.** Every trainable gradient is `torch.equal` with the policy on and off, under deterministic mode. `_ScatterCombine` matches
  the composite's forward and both gradients bitwise (`tests/test_moe_keep.py`). The policy is also checked on a tiny Qwen3-MoE:
  gradients equal, the last n layers changed, an exact unwind.
- **Next.** TC1 measures it on an RTX 5090, with peak memory, before it is recommended.

### The fused training forward gathers the routing weights with a scatter backward instead of a sorted one (values identical)

- **What.** `fused_experts_train_forward` gathered the routing weights as `top_k_weights[token_rows, top_pos]`. The backward of
  that two-index advanced gather is `index_put_(accumulate=True)`, which linearizes the two indices and radix-sorts them, about
  ten launches per MoE layer backward. `order` is a permutation of the `[tokens, k]` slots, so `_PermGather` now does the same
  gather as one `index_select`, and its backward is a zero fill plus one `index_copy_`. `top_pos` is no longer computed.
- **Values.** Forward and every gradient are `torch.equal` to the old form, in `tests/test_perm_gather.py` (CPU and CUDA,
  bf16 and fp32) and in one Qwen3-30B-A3B-shaped `ExpertsLoRA` layer on an RTX A2000 under deterministic mode: the output, the
  input and routing-weight gradients, and all four adapter gradients.
- **Measured on an RTX A2000** (grouped-nf4-gemm's `bench/host-reuse/moe_host.py`, router weights carrying grad,
  `GNF4_HOST_REUSE=1` in both arms), three interleaved old/new pairs of 300 repetitions, on a host shared with other jobs:
  - backward host median 3.015 / 2.866 / 2.801 → 2.775 / 2.558 / 2.578 ms;
  - forward host median 3.319 / 3.066 / 3.052 → 3.261 / 2.819 / 2.799 ms;
  - backward device span 18.88 / 19.17 / 19.38 → 18.99 / 19.25 / 19.44 ms. Each new arm ran second, and the span drifted
    upward across the run, so this is not read as a device cost.

### TC1 amendment 19 read: the matched-work positions after amendments 10–15 — Unsloth/e4b 1.997, axolotl/e4b 2.775

- **What was asked.** TC1's matched set (`tc1-5090-49`, EPYC 7C13, $1.15) and the axolotl rows (`tc1-5090-50`, Ryzen 9 9950X3D, $0.33),
  re-run with e4b carrying every default from amendments 10–15.
- **What it read.** All three predictions are HELD:

  | prediction | reading | band | was (pre-#945) |
  |---|---|---|---|
  | P30, Unsloth / e4b | **1.997** [1.980, 2.014] | 1.6–2.8 | 1.437 |
  | P31, matched set | inside the draw noise of e4b fused | — | — |
  | P32, axolotl / e4b | **2.775** [2.735, 2.814] | 1.6–2.8 | 1.416 |

  On the matched-set box's host axolotl/e4b reads 1.979. Unsloth is still 2.95 GB lower at peak (registers `e4b.train.h2h.unsloth.qwen3.5090.2026-10-03`, `e4b.train.h2h.axolotl.qwen3.5090.2026-10-03`).
- **What follows.** STATUS quotes the new positions, and the 2026-10-02 rows stand for the code before #945. Read:
  `bench/h2h-2026-10-02/tc1/RESULTS-tc1-matched19.md`.

### P111 read (RTX 5090): DEFAULT_ON -- one KV-table selection per decode step decodes identical tokens 3.6 % faster with 16 concurrent requests and 1.2 % with one; `E4B_KV_STEP_SELECT` is on by default (`0` keeps the per-layer form) (#1001)

- **Files.**
  - `bench/p111/RESULTS-p111.md`;
  - `bench/p111/receipts/p111-5090-1/`: one RTX 5090 on an AMD Ryzen 9 9950X3D, $0.1586. The lane cost $0.1959.
- **The read.** The default graph server ran Qwen3-30B-A3B NF4 in four ABBA arms that differ only in the switch.
  - Tokens are identical on every row.
  - With 16 concurrent requests, the pair ratios are 1.0352 and 1.0377 (772 → 800 tok/s); the step falls 20.72 → 19.99
    ms, SC1b's ~0.8 ms census.
  - With one request, 1.0096 and 1.0138.
  - The self-pairs read 0.997–1.002.
- **The registered consequence:**
  - `Fp8PagedKV` now reads an unset `E4B_KV_STEP_SELECT` as `1`;
  - the parser and `tests/test_kv_step_select.py` keep the bytes P111 staged;
  - `tests/test_kv_step_select_default.py` pins the default;
  - `docs/SERVING.md` and `docs/STATUS.md` say so;
  - register row `e4b.serve.p111.kv-step-select.qwen3.5090.2026-10-04`.
- **Predictions missed:** g1 (1.0096, under its 1.02–1.10 band).

### P111 registered: does one KV-table selection per decode step (`E4B_KV_STEP_SELECT=1`) decode the default `serve_paged` server's tokens exactly, and faster? (bench and tests only)

- **Why.** SC1b put about 0.8 ms of the B=16 decode step in per-layer KV-table glue. #999 ships one selection per step
  behind an opt-in switch, but neither its speed nor its identity has been read on the served model.
- **What.** One RTX 5090 runs the default graph server (Qwen3-30B-A3B NF4, P109's subject) in four ABBA arms that differ
  only in `E4B_KV_STEP_SELECT`. Each arm runs 16 concurrent requests and 1 request, at 32 and 160 new tokens.
- **The rule:** VOID, then NOISY (self-pairs outside [0.96, 1.04]), then FUNCTION_FAIL (any token differs), then SLOWER
  (either workload's min pair ratio below 1.00), then DEFAULT_ON.
- **Consequence of DEFAULT_ON:** the switch defaults on, with a register row.
- **Proof.** The whole box runs on Granite, with the premise of 13 GPU tests on the card.
- **Files:**
  - `bench/p111/`: prereg, runner, box (importing P109's at its registered bytes), reducer, driver and pin;
  - `tests/test_p111_staged_pin.py`.

### `E4B_KV_STEP_SELECT=1`: a decode-graph bucket's KV-table selection once per step instead of once per layer (opt-in; default unchanged)

- **Why.** SC1b's read (#993) put about 0.8 ms of e4b's B=16 decode step on an RTX 5090 in `fp8_paged_kv.py`: each layer
  re-selects the active set's block-table and seq-lens rows (two `index_select`) and publishes its length with one
  `seq_lens.index_add_`. That is 97 + 48 launches a step on Qwen3-30B-A3B. The active set is fixed for the whole step.
- **What.** With the switch on and a decode-graph bucket bound:
  - `Fp8PagedKV.graph_bucket_load` selects every layer's rows in one launch each, outside the graph. It stores the lengths
    attention reads, pre-step + 1.
  - `kernel_args` hands each layer its slice, with no launch.
  - `append_graph_bt1` skips its per-layer `index_add_`.
  - The new `graph_bucket_publish`, called by the runner after every bucket step, advances every layer's lengths in one
    launch.
  - The kernels see the same values: each append writes at its pre-step length and attention reads pre-step + 1. Tokens
    must therefore be identical.
- **The storage change.** The per-layer block tables are now views of one `[L, slots, blocks]` tensor (`_bt_all`), so one
  select covers every layer. Every existing write is in place through `block_table[layer]` and lands in it.
- **Tests** (`tests/test_kv_step_select.py`).
  - On CPU:
    - the stacked table's views;
    - the selection against what each layer would select after its own append;
    - `kernel_args` handing out the slices;
    - the publish touching exactly the bucket's slots;
    - the switch's parsing and refusal;
    - nothing changing with the switch off.
  - On sm_89+: a tiny Qwen3 decodes identical tokens with the switch on and off, through captured graphs and through the
    padded eager step.
- **Next.** A lane reads the exactness and speed on an RTX 5090 through the default server before the default moves.

### TC1 amendments 20 and 21 and TC1c amendments 1 and 2 registered (#835, #945) (bench only)

- #1006 (amendment 20: grouped-nf4-gemm's per-pass host reuse, A/B on one 5090), #1008 (amendment 21: keeping MoE activations
  instead of recomputing them), #1004 (TC1c amendment 1: the H100 position again) and #1010 (TC1c amendment 2: the H100 position
  with e4b keeping its MoE activations) are registration PRs that merged without a changelog line. They are listed here so that
  the release names every merged PR.

## 0.43.0 — 2026-10-03 — `serve_paged` captures bucketed decode graphs by default (`E4B_PAGED_GRAPHS=auto`): ×5.60 the old eager default with 16 concurrent requests and ×9.02 with one on an RTX 5090, at no measured quality cost (lanes P109, P110); e4b's training lead over axolotl's scattermoe replicates on a second host

**0.43.0.** One default changes. `serve_paged` now decodes with bucketed CUDA graphs on a CUDA device at the `all-vram`
placement (`E4B_PAGED_GRAPHS=auto`). `E4B_PAGED_GRAPHS=0` restores eager decode.

**The license.** On the default server (Qwen3-30B-A3B NF4, one RTX 5090):
- **Speed (P109).** Graphs ran ×5.60 the eager default with 16 concurrent requests (731–748 against 125–131 tok/s) and
  ×9.02 with one (99.4 against 10.4–11.0).
- **Function (P109).** The replay is bit-identical to its own padded eager step.
- **Quality (P110).** The arithmetic graphs bring, device grouping plus bucket padding, reads +0.0004 nats against the
  eager default, teacher-forced. That is inside the eager default's own neutral perturbations (AT_PARITY).

**What it means for a user.**
- Greedy outputs change from the old eager default's at the bf16 level, at no measured quality cost.
- Startup adds about 3 s of capture.
- A `PagedServeConfig` built directly in code keeps `graphs=False`.
- **Scope.** Read on Qwen3-30B-A3B NF4. Hybrid models' graphs are P101's; other families ride the same capture code
  without a reading of their own.

**Also in this release:**
- TC1 amendments 17–18 read e4b's steady-state training lead over axolotl's scattermoe on a second host: 1.146, against
  1.238 on the first.
- SC1b's read names where e4b's serving losses come from.
- TC1 amendment 19 is registered.

grouped-nf4-gemm is unchanged; CI still tests against 0.35.0.

### `serve_paged` captures bucketed decode graphs by default (`E4B_PAGED_GRAPHS=auto`; `0` keeps eager decode) (#770, lanes P109 and P110)

- **What changes.** `PagedServeConfig.from_env()` resolves `E4B_PAGED_GRAPHS` through `_graphs_env`:
  - `auto`, the default (also when the variable is unset or empty): graphs on a CUDA device at the `all-vram`
    placement, and eager decode elsewhere;
  - `1`: forces graphs, refusing where batched graphs are refused;
  - `0`: keeps eager decode.
  - Anything else is refused.
  - A `PagedServeConfig` built directly keeps `graphs=False`.
- **Why.** On the default server (Qwen3-30B-A3B NF4, one RTX 5090):
  - P109 read graphs ×5.60 the eager default with 16 concurrent requests and ×9.02 with one, at +0.055 GiB, with the
    replay bit-identical to its padded eager step;
  - P110 read the arithmetic graphs bring, device grouping and bucket padding, at +0.0004 nats against the eager
    default, teacher-forced and inside the eager default's own floor (AT_PARITY).
- **What a user sees.**
  - The default server decodes 5–9× faster on that host.
  - Greedy outputs differ from the old eager default's at the bf16 level, at no measured quality cost.
  - Startup adds about 3 s of capture.
  - `E4B_PAGED_GRAPHS=0` restores the old behaviour.
- **Scope.** Read on Qwen3-30B-A3B NF4. Hybrid models' graphs are P101's. Other families ride the same capture code
  without a reading of their own.
- **Tests.**
  - `tests/test_serve_paged.py` pins `auto`, `0`, `1`, the CPU and `solver` fallbacks, and the refusal.
  - `tests/test_p109_box.py`'s stub engine declares a CPU device, so an unset switch still reads as the eager arm P109
    registered.

### SC1b read (#846): e4b's B=1 loss to llama.cpp is kernel overlap it lacks, not slower kernels; its B=16 loss to vLLM is per-layer KV-table glue (bench docs, receipts and a read-time tool)

- `bench/h2h-2026-10-02/sc1b/`: the read (`README.md`) and every run's receipts: two proofs, one NOT_RUN, two box D runs.
  The nsys exports stay in the private receipt store. SC1b total: $3.04 over 5 receipts.
- **`sc1d-5090-2` under A1** (registered): Q2 REFUTED (1,550 vs 1,110 in-graph kernels, 1.396); G4 B=16 named norm_elem;
  every other gap unread. That outcome led to A3.
- **`sc1d-5090-3` under A3** (confirmatory, capture paths byte-identical to the proved commit):
  - **Q6 HOLDS**: G1 names the in-graph overlap, +1.380 of e4b's +1.199 ms/step loss to llama.cpp at B=1.
  - **Q8 HOLDS**: llama.cpp overlaps 95.5 % of consecutive in-graph kernels on one stream; e4b none. The two engines'
    summed kernel time agrees within 1.2 %.
  - Q2 REFUTED again.
  - G3 B=16 names norm_elem. G4 B=16 does too nominally, but its band floor sits 17 us under the threshold, so Q7 is
    UNREAD.
  - Q1 and Q3-Q5 UNREAD.
- **What e4b's B=16 `norm_elem` is.** About 0.8 ms/step is `fp8_paged_kv.py` re-selecting the active set's block-table
  and seq-lens rows per layer (97 `index_select`), plus a per-layer `seq_lens.index_add_` (48). Levers, untested here:
  programmatic dependent launch in e4b's decode graph; one active-set selection per step.
- `bench/sc1b/sc1b_kernels.py` (read time, descriptive only) lists one class's kernels by per-step time. It has a
  self-test in `tests/test_sc1b.py`.

### P110 read (RTX 5090): AT_PARITY -- the arithmetic decode graphs bring to `serve_paged` costs no measurable quality: +0.0004 nats against the eager default, inside its own neutral perturbations (#770)

- **Files.**
  - `bench/p110/RESULTS-p110.md`;
  - `bench/p110/receipts/p110-5090-1/`: one RTX 5090 on an AMD EPYC 7C13, $0.1904. The lane cost $0.2455.
- **The read.** The default server ran Qwen3-30B-A3B NF4, built eager, scored teacher-forced over 48 wikitext windows.
  - Device grouping plus bucket padding, which is bitwise the graph path per P109, reads +0.00036 nats (spread 0.0114)
    against the eager default.
  - The eager default's own half-batch and prefill-split draws read +0.0018 and +0.0005 (spreads 0.011–0.013).
  - Grouping alone reads +0.0015.
  - A halved decode scale reads +1.05.
  - Engagement is exact.
- **The registered consequence.** It licenses `E4B_PAGED_GRAPHS=auto` as `serve_paged`'s default; the code change is the
  follow-up PR. Register row `e4b.serve.p110.graph-arithmetic-quality.qwen3.5090.2026-10-03`. SERVING and STATUS are
  updated.
- **Predictions missed:** the floor's and P's spreads, both wider than predicted (0.0127 / 0.0114).

### P110 registered (#770): does the arithmetic decode graphs bring to `serve_paged` cost quality? Device grouping and bucket padding against the eager default, teacher-forced, judged against the eager default's own neutral perturbations (bench and tests only)

- **Why.** P109 read graphs ×5.60 / ×9.02 and the replay exact, but DIVERGENT on tokens. The divergence comes from the
  device grouping and bucket padding that graphs bring, and token agreement cannot say which arithmetic is better.
- **What.** One RTX 5090 runs the default server (Qwen3-30B-A3B NF4, built eager by `build_engine`). Teacher-forced
  paged passes on its model cover 48 wikitext windows (512 + 128) in groups of 12:
  - **R:** the eager default, host grouping;
  - **the floor:** a half-batch, a 256-token prefill split, and reversed slots;
  - **D:** device grouping;
  - **P:** device grouping plus bucket padding, which is the graph server's arithmetic (G ≡ P, per P109);
  - **a scale mutant.**
- **The rule (P108's):** AT_PARITY iff P's bias ≤ B_floor + 0.01 nats and spread ≤ 2 × max(S_floor, 0.005); VOID on
  commits, windows, engagement or a passing mutant; COST otherwise.
- **Consequence of AT_PARITY:** `E4B_PAGED_GRAPHS` defaults to `auto` (on for CUDA at all-vram), with a register row,
  and #770 closes.
- **Proof.** It runs the whole box on Granite, because no local card runs the bucketed path.
- **Files:**
  - `bench/p110/` (prereg, runner, box, reducer, driver, pin);
  - `tests/test_p110_staged_pin.py`;
  - `tests/test_p110_box.py`: every CPU-capable arm on a tiny Qwen3-MoE.

### P109 read (RTX 5090): DIVERGENT -- graphs are 5.60x the eager default at 16 requests and 9.02x at one, and the replay is bit-identical to its padded eager step, but the graph server's tokens leave the eager default's within 16 tokens on 7 of 16 rows; `serve_paged` keeps eager decode by default (#770)

- **Files.**
  - `bench/p109/RESULTS-p109.md`;
  - `bench/p109/receipts/p109-5090-2/`: one RTX 5090 on an AMD EPYC 7C13, $0.2926. The lane cost $0.4599 over five runs.
- **The read.** The default server ran Qwen3-30B-A3B NF4 at `max_seqs` 16, built by `build_engine`.
  - `E4B_PAGED_GRAPHS=1` ran 731–748 tok/s against the eager default's 125–131 at 16 concurrent requests, and 99.4 against
    10.4–11.0 for one request.
  - It cost +0.055 GiB of peak memory and +3.3 s of load.
  - G1 ≡ G2 ≡ P1, the padded eager step, bitwise on every row.
- **Why DIVERGENT.**
  - The registered sanity bar needed 12 of 16 rows agreeing with the eager default for 16 tokens; the reading had 9.
  - The decomposition shows that graph replay adds no divergence. The divergence comes from the device grouping and
    bucket padding that graphs bring.
  - Grouping alone, with no graphs, reads 10 of 16.
  - Whether either arithmetic is better is a teacher-forced quality question, and it is still open.
- **The registered consequence.**
  - Graphs stay opt-in.
  - `docs/SERVING.md` and `docs/STATUS.md` state the measured ratio, and that the registered serving-speed numbers are the
    graph path.
  - A measured register row: `e4b.serve.p109.decode-graphs-vs-eager-default.qwen3.5090.2026-10-03`.
  - #770 stays open.
- **Predictions missed:** S1, which was above its band; the median E-vs-G first divergence; and the sanity bar.

### TC1 amendments 17–18 read: e4b's steady-state lead over axolotl's scattermoe replicates on a second host (P29 HELD, 1.146)

- **What was asked.** amendment 16's box (P27: axolotl / e4b 1.238) again, byte for byte, on a different machine.
- **What it read.**
  - **P29 is HELD:** on machine 150527 (`tc1-5090-48`, EPYC 7C13, $1.23), axolotl / e4b over steps 101..200 is **1.146** [1.129, 1.164],
    the whole interval above 1.0. That machine was reached by leaving 142284 out by evidence: adertha-agents#139's
    `avoid_vast_machine_receipts` cited -46's receipt.
  - **e4b as shipped is faster at steady state on both hosts**, by 15–24 %.
  - **P28 is UNTESTED:** `tc1-5090-47` re-bought machine 142284, the cheapest 5090. That makes it a same-host repeat, and it reproduces
    -46 within 1.2 % (1.253).
- **What follows.** The register row `e4b.train.h2h.axolotl.qwen3.5090.2026-10-03.native-steady-state` records the second host. Read: `bench/h2h-2026-10-02/tc1/RESULTS-tc1-steady18.md`.

### P109 Amendment 2 (#770): the graph replay's oracle is the padded eager step, before any reading data (bench and tests only)

- **What the proof showed.** `p109-prove-2` ($0.057) proved the whole box on Granite. Its verdict, not a reading, was
  FUNCTION_FAIL: G1 = G2, but both differed from D1, the unpadded eager step, on 7 of 16 rows of the 16-request workload.
- **Why that was the registration's error.** The 16-request workload's staggered prefill decodes 1–15 active rows, and a
  graph step pads them to the next bucket. The repo's own test asserts the replay against the PADDED eager step and
  only reports the unpadded one. P82 and B771b's identity traces were all bucket-sized, so they never padded.
- **The stopped reading.** `p109-5090-1` was stopped 27 s into staging, before any data, at $0.015.
- **The change.** A sixth arm, P1 (`E4B_PAGED_GRAPHS=1` with `enable_decode_graphs(capture=False)`), is now FUNCTION's
  oracle. G-vs-D is reported. The reducer self-tests on 18 cases. Everything else is unchanged.
- **Said before the data:** the sanity bar is unchanged, and the Granite proof read 8 of 16 on it.

### P109 Amendment 1 (#770): the proving run gets its own time-left checks, before any reading data (bench only)

- **What happened.** `p109-prove-1` ($0.0207) passed everything up to the fetch on an RTX 5090: the install, the tripwire,
  both self-tests, and the premise (7 passed, none skipped). It then stopped with rc 40.
  - The runner's check before the fetch needed 2,400 s plus a 600 s margin. That is 50 minutes, more than the proof's
    whole 45-minute guard.
- **The fix.** The proof's checks are 300 / 300 / 240 s (fetch, bake, arm). The reading's are unchanged.
- **New test.** `tests/test_p109_staged_pin.py` asserts that every check fits its own guard.

### P109 registered (#770): should `serve_paged` capture decode graphs by default? Its eager default against bucketed graphs, through the server's own construction (bench and tests only)

- **Why.** Every e4b serving number in the register ran with decode graphs: SC1 and P96–P101. `serve_paged` ships with
  them off, so its default user gets an eager configuration no registered number describes.
- **What.** One RTX 5090 runs the default server: Qwen3-30B-A3B NF4, `max_seqs` 16, `all-vram`, every lever at its
  default, built by `PagedServeConfig.from_env()` + `build_engine`. It runs in five arms, each in its own process:
  - **E1, E2:** today's eager default;
  - **G1, G2:** `E4B_PAGED_GRAPHS=1`;
  - **D1:** eager with G's device grouping.

  Each arm decodes 16 concurrent requests and 1 request, at 32 and 160 new tokens.
- **The rule, in order:**
  - VOID on commits, prompts or engagement;
  - NOISY if a self-pair falls outside [0.93, 1.07];
  - FUNCTION_FAIL unless G ≡ D on every row;
  - KEEP if graphs are below ×1.25 at 16 requests or ×0.97 at 1;
  - DIVERGENT if fewer than 12 of 16 rows agree with eager for 16 tokens;
  - DEFAULT_GRAPHS otherwise.
- **Registered consequence of DEFAULT_GRAPHS:**
  - `E4B_PAGED_GRAPHS` defaults to `auto` (on for CUDA `all-vram`);
  - a register row;
  - #770 closes.
- **The proving rental** runs the whole box on Granite-3.1-3b-a800m, because no local card runs the fp8 paged KV.
- **Files:**
  - `bench/p109/` (prereg, runner, box, reducer, driver, pin);
  - `tests/test_p109_staged_pin.py`;
  - `tests/test_p109_box.py`: the box end to end on CPU over a scripted runner and a real scheduler, through the
    reducer.

### TC1 amendment 19 registered (#835): the matched-work positions again, with e4b after amendments 10–15 (P30, P31, P32) (bench only)

- #996 pre-registers re-reads of TC1's headline matched-work positions on the current e4b, before any box runs:
  Unsloth / e4b 1.437 and axolotl / e4b 1.416, both measured before #945. It merged without a changelog line, so it is
  listed here.

## 0.42.0 — 2026-10-03 — paged prefill attention on the flash kernel by default (lane P107: a 4096-token prefill's device time 906 → 378 ms on an RTX 5090); the fused training RMSNorm on by default; with grouped-nf4-gemm 0.35.0, e4b's fused training step runs ahead of axolotl's scattermoe at steady state on a 5090 (1.238); Gemma-4's paged path at parity (P108)

**0.42.0.** This package changes two defaults, and grouped-nf4-gemm 0.35.0, released the same day, carries three more.

**This package's defaults:**
- **Prefill attention takes the flash kernel** (`E4B_PAGED_PREFILL_ATTN=flash`, lane P107). `=math` restores the old path.
  - A 4096-token prefill's device time fell from 906 to 378 ms on an RTX 5090.
  - Served-prefill NLL stayed within the K8 rule (−0.014 / −0.008 ppl). This was read on Qwen3-30B-A3B only.
- **Fused training runs a fused RMSNorm** (TC1 amendment 15). `E4B_FUSED_RMSNORM=0` restores the old path.
  - Step time was 0.924 of the old path on the shipped arm and 0.959 on the matched arm.
  - It is near-exact, not exact: about 1e-5 of elements are one bf16 ulp off. Held-out loss agreed within 0.01.

**grouped-nf4-gemm 0.35.0's defaults** are value-identical:
- the pinned index ring outside capture;
- the lean padded LoRA delta;
- the prefill M-tile height from the group sizes.

**Installing the kernel package.**
- The `[fast]` floor stays `grouped-nf4-gemm>=0.30.0`, because nothing here needs the new kernel to be correct.
- A fresh install gets 0.35.0. An existing environment gets the speed with `pip install -U grouped-nf4-gemm`.
- CI now tests against 0.35.0's commit (`51a4916`).

**What the combination measured.** TC1 amendment 16 ran this combination against axolotl 0.20.0's scattermoe native-best on one RTX 5090 (EPYC 7663), over steps 101–200. It ran e4b at every default from amendments 10–15, with those kernels.
- axolotl / e4b = **1.238**: e4b was faster.
- At step 200, e4b's held-out loss was 0.024–0.027 above axolotl's matched curve.
- Register: `e4b.train.h2h.axolotl.qwen3.5090.2026-10-03.native-steady-state`.
- This is one host. Amendments 17–18 register the same ordering on a second host; they are not read at this release.

**Also in this release:**
- P108 reads Gemma-4's paged path **AT_PARITY** with transformers' own perturbations, which closes #359.
- SC1b, the per-kernel census of each serving engine's decode step, is registered.

### SC1b amendment A3 (#846): written after box D's first run -- vLLM's periodic steps kept, CUPTI's graph-id messages labelled, node-trace overhead as a band, in-graph kernel overlap as a term; a confirmatory re-run (bench and tests only)

- **`sc1d-5090-2`** (machine 45511, $0.7788, all 24 passes exit 0) reads as registered:
  - Q2 REFUTED: e4b runs 1,550 in-graph kernels per B=1 step against llama.cpp's 1,110, a ratio of 1.396 (>= 1.5
    predicted).
  - Q1 and Q3-Q5 UNREAD.
  - The only gap read is G4 at B=16 (e4b - SGLang +1.874 ms/step), named norm_elem (+1.007 ms).
- **Three instrument causes, each fixed in `sc1b_census.py`:**
  - The out-of-graph drop rule threw away vLLM's every-16th step (a KV block-table write), which voided vLLM.
  - The diagnostics gate read CUPTI's graph-id mapping messages as errors on llama.cpp B=16.
  - A flat 2 % node-trace gate blocked every B=1 arm. It is replaced by a bound on what node tracing can add, and a
    gap reads only if its reading holds across that bound.
- **G1's remainder was kernel overlap.** llama.cpp overlaps 1,060 of 1,109 consecutive in-graph kernel pairs on one stream
  at B=1 (1.35 ms per step); e4b overlaps none. The overlap is now an explicit term of the identity.
- `sc1b_read.py`: Q1-Q5 keep A1's term set and are decided across the bands. Q6-Q8 are added as replications of what
  `sc1d-5090-2` showed: G1 names the overlap, the B=16 vLLM and SGLang gaps name norm_elem, and llama.cpp overlaps
  >= 90 % of pairs where e4b overlaps <= 1 %.
- Capture paths are byte-identical to `77469c1`, which `sc1d-prove-2` proved, so box D re-runs as `sc1d-5090-3` without
  a new proof.


### P108 read (RTX 5090): AT_PARITY -- Gemma-4's paged path is indistinguishable from transformers' own arithmetically neutral perturbations over 32 windows with the sliding window binding (#359)

- **Files:** `bench/p108/RESULTS-p108.md`, `bench/p108/receipts/p108-5090-3/` (one RTX 5090, AMD Ryzen 9 9950X3D;
  $0.5181; lane $2.3632 over four runs).
- **The result:**
  - the paged path's mean NLL, 5.360, lies between the one-shot, chunked and batched transformers draws
    (5.298–5.415);
  - against transformers' cached forward its bias is −0.196 nats and spread 0.326, inside the floor's 0.258 / 0.349;
  - a halved decode scale reads +2.26, and is caught;
  - a dropped decode window reads −0.95: visible, and LOWER NLL.
- **Unexplained:** transformers' own cached order sits apart from every other arm.
- **Registered consequence:**
  - `docs/SERVING-PARITY.md`'s Gemma-4 row is now "at parity with transformers' own perturbations", with a new section
    on the floor;
  - `docs/STATUS.md`;
  - register row `e4b.parity.gemma4.p108.paged-vs-floor.5090.2026-10-03`;
  - #359 closes.
- **Predictions missed:** the floors and the paged bias (larger, and negative), and the window mutant's sign.

### P108 Amendment 3 (#359): a GPU memory leak in the box fixed, and the last attempt's budget, before any data is read (bench only)

- **Why.** `p108-5090-2` ran out of CUDA memory in group 3 of 4 (30.67 GiB allocated; $1.0068; no data). P97's
  `_paged_pass` leaves each pass's KV pool in a reference cycle (`kv.attention` is a closure over `kv`), so the pools
  waited for Python's cycle collector.
- **The A2000 confirmation:** allocated memory crept upward without a collection and stayed flat with one.
- **The fix:** `p108_box.py` collects and empties the CUDA cache after every paged pass and each group, and logs the
  allocated memory.
- **The budget:** box alarm 180 min; guard 3.25 h (≤ $2.44); lane ceiling $4.30. The hard stop stays $4.50.
- **This is the last attempt.**

### SC1b amendments A1 and A2 (#846): the evaluator for predictions Q1-Q5, merged before any box D data; the proof's e4b capture gets Granite rows (bench and tests only)

- `bench/sc1b/sc1b_read.py` decides Q1-Q5 (HOLDS / REFUTED / UNREAD) from box D's own arm and gap records and writes
  the read's tables; `SC1b-PREREG.md` gains A1, which fixes three readings of the registered text before data exists.
  - "The largest term of dP" is the largest contribution in dP's direction. The remainder O is not a term.
  - Q2's kernel count is the node-mode median per step.
  - Q4 has no noise clause.
- `tests/test_sc1b.py`: the evaluator's self-test, and an end-to-end case that feeds it the reducer's own `gap()`
  records, so a key rename on either side fails. A blocking label leaves the gap and its predictions UNREAD.
- A1 runs at read time; box D's code is unchanged by it.
- **A2.** The proof `sc1d-prove-1` ($0.41) failed on one item.
  - Its e4b Granite B=16 capture read the lane's `prompts_b16.json` before that file existed, and the file holds Qwen3 ids
    beyond Granite's vocabulary.
  - `sc1b_e4b_census.py --write-prompts` now writes seeded Granite rows that SC1's own `load_prompts` accepts, and the
    proof passes them to `e4b_census`.
  - The other four proof items had reduced cleanly. The proof re-runs as `sc1d-prove-2`.

### TC1 amendment 16 read: at steady state e4b as shipped now steps faster than axolotl's scattermoe on a 5090 (P27 HELD, 1.238)

- **What was asked.** On one RTX 5090 (`tc1-5090-46`, EPYC 7663, $0.99), amendment 8's 200-step comparison again. This time e4b ran
  with every default from amendments 10–15 (#945, #440, #442, #965, #975), against axolotl 0.20.0's scattermoe native-best. P27
  predicted axolotl / e4b over steps 101..200 in [1.05, 1.50].
- **What it read.** P27 is HELD:

  | reading | result |
  |---|---|
  | axolotl / e4b, steps 101..200 | **1.238**, cross-draw 1.231 – 1.246; both pairs stable (0.6 %) |
  | ordering | **e4b shipped faster** |
  | summed step time, 200 steps | 650–654 s against 1,373–1,389 s |
  | axolotl's energy per step | ×1.79 |
  | held-out at 200 | axolotl on the matched curve; e4b as shipped 0.024–0.027 above it |

  Register `e4b.train.h2h.axolotl.qwen3.5090.2026-10-03.native-steady-state`.
- **What follows.** STATUS says e4b as shipped is faster at steady state on that host. The 2026-10-02 rows stand for the code before
  #945. Read: `bench/h2h-2026-10-02/tc1/RESULTS-tc1-steady16.md`.

### P108 Amendment 2 (#359): the box gets 220 minutes and the reading a 4 h guard, before any data is read (bench only)

- **Why.** `p108-5090-1` ended on its 75-minute alarm (rc 142, $0.8047) after two of four groups. The box needs about
  140 minutes, so Amendment 1's 150 would leave ~10 minutes of margin on a host-bound decode. No data exists.
- **What changes:** the box alarm, 150 → 220 min; the guard, 3 h → 4 h (≤ $3.00); the lane ceiling, $3.50 → $4.25
  (the hard stop stays $4.50).
- **What stays:** the registered design (32 windows × 256 positions).

### SC1b registered (#846): a per-kernel census of where each engine's decode step goes -- SC1's largest loss (llama.cpp at B=1) and largest win (B=16), named by term (bench and tests only)

- `bench/sc1b/{SC1b-PREREG.md,sc1b_census.py,kernel_classes.json,sc1b_e4b_census.py,sc1b_vllm_census.py,sc1b_serve_census.py,sc1b_toy.py,sc1b_box_d.sh,UPSTREAM-NOTES.md,DESIGN-REVIEW.md}`,
  `tests/test_sc1b.py`; box D's dispatch points in `bench/sc1/sc1_run.sh` and `sc1_drive.sh`.
- **The box:** one RTX 5090 runs e4b, llama.cpp, vLLM and SGLang at B=1 and 16. Each (engine, B) gets three passes: SC1's
  own unprofiled arm, then an Nsight Systems 2025.6.1 graph-mode capture and a node-mode capture of decode steps 34-97.
- **The reading:** each step's period splits into kernel classes, in-graph idle (graph span minus the union of its
  records) and out-of-graph idle, with an overlap remainder. A gap is named by the first term with its sign that carries
  >= 50 % of it and exceeds twice the two arms' per-step IQRs. If the remainder carries half, the gap is UNEXPLAINED.
  Gates on profiler inflation, node-trace inflation, class-map coverage, MoE segment balance, nsys diagnostics and clocks.
- **Predictions:** e4b's B=1 loss to llama.cpp is mostly in-graph launch/dependency gaps, not its int4 expert GEMV; its
  B=16 win is llama.cpp's out-of-graph host time.
- **SC1's own scripts:** unchanged when `SC1_LAUNCH_PREFIX` is unset. The paged prefill attention is pinned to `math`, the
  route every SC1 box ran, now that the default is flash.
- Two hostile design reviews; round 2's disposition is in `DESIGN-REVIEW.md`. One more defect was found while applying it
  and is fixed with a Linux test: ending a capture with `pkill -f` on the app's pattern also signalled nsys, whose
  command line carries the app's.

### P108 Amendment 1 (#359): the box gets 150 minutes and the reading a 3 h guard, before any data is read (bench only)

- **Why.** `p108-5090-1` showed that transformers' batch-1 eager decode of Gemma-4-26B-A4B runs at about 0.25 s a step.
  The reference, its repeat and the chunked floor need about 95–100 minutes in all, and the 75-minute alarm ends the
  box before it writes a record. No per-window value was read.
- **What changes:** the box alarm, 75 → 150 min; the guard, 2 h → 3 h (≤ $2.25).
- **What stays:** the windows, arms, floor, rule, premise and predictions. The "box ≤ 60 min" prediction is already
  refuted.

### TC1 amendment 15 read (#945): e4b's fused training RMSNorm makes the step 4–8 % faster on a 5090, held-out unchanged

- **What was asked.** On one RTX 5090 (`tc1-5090-45`, $0.39), e4b against itself: the Hugging Face RMSNorm composite against #961's
  fused kernel. Shipped and matched arms, two draws each, ABBA order. (`tc1-5090-44` ended HARNESS_ERROR at $0.0004: the guard's auth
  probe got an HTTP 429.)
- **What it read.** P24, P25 and P26 are HELD:

  | reading | result | range |
  |---|---|---|
  | shipped | fused / composite **0.924** | 0.893 – 0.955 |
  | matched | **0.959** | 0.935 – 0.984 |
  | held-out at N | within 0.0021 | band 0.01 |

  Not bit-identical: the first training loss differs by 0.0090 before any update (register `e4b.train.fused-rmsnorm.qwen3.5090.2026-10-03`).
- **What follows.** The fusion is on by default (#975). Read: `bench/h2h-2026-10-02/tc1/RESULTS-tc1-rmsab.md`.

### The fused training RMSNorm is on by default (`E4B_FUSED_RMSNORM=0` turns it off)

- **Why.** TC1 amendment 15 registered a 5090 A/B of #961 with a decision rule: flip the default on if held-out loss agreed within
  0.01 (P26) and both arms stepped at or below 0.99 of the composite. The box (`tc1-5090-45`, EPYC 7663, $0.39) read all three
  predictions HELD:

  | reading | result | range |
  |---|---|---|
  | P24, shipped | fused / composite **0.924** | 0.893 – 0.955 |
  | P25, matched | **0.959** | 0.935 – 0.984 |
  | P26, held-out difference | −0.0021 shipped, −0.0015 matched | within 0.01 |

- **What.** `enable_fast_train` applies `enable_fused_rmsnorm_train` unless `E4B_FUSED_RMSNORM=0`. On the default path a model with no
  frozen, probe-matching RMSNorm is skipped quietly. An explicit `E4B_FUSED_RMSNORM=1` keeps the refusal of a vacuous enable.
- **Numerics.** Near-exact, not exact: about 1e-5 of elements are one bf16 ulp off the composite. Through Qwen3-MoE's 48 layers and
  router that showed up on the box before any update: held-out loss at step 0 differed by 0.0064 (1.9441 against 1.9505), and the
  first training loss by 0.0090 (2.0614 against 2.0705). Held-out at N stayed inside P26's band.

### E4B_PAGED_PREFILL_ATTN defaults to flash (#960, lane P107's DEFAULT=flash): a 4096-token prefill's attention on the flash kernel in bf16 instead of SDPA's fp32 math backend -- device time 906 -> 378 ms on an RTX 5090

- **What changes.** A paged prefill chunk on a layer without sinks or a sliding window now passes SDPA the lower-right
  causal mask as `causal_lower_right(T, t_total)` with `enable_gqa`, which the flash kernel takes. It used to pass an
  explicit boolean mask, which with GQA lands on the fp32 math backend.
  - `E4B_PAGED_PREFILL_ATTN=math` restores the old path.
  - Layers with sinks or a sliding window are unchanged.
  - Where no fused kernel applies (CPU, older GPUs, head dims flash does not take), SDPA serves the same mask itself.
- **The read** (`bench/p107/RESULTS-p107.md`, #971; RTX 5090, Qwen3-30B-A3B, the calibrated K8 rule on a
  served-prefill NLL, 12 fresh windows): flash - math = -0.014 / -0.008 ppl (PASS). TTFT-4096 fell from 1.984 s to
  1.742 s (1.14x) on a CPU-bound host, with device time down from 906 ms to 378 ms. First tokens were identical in
  every timed draw.
- **Scope:** read on Qwen3-30B-A3B. For other families with plain full-attention layers, the route's CPU equivalence
  tests cover the arithmetic; no reading does.
- `docs/SERVING.md` gains a "Prefill attention" note. `tests/test_paged_prefill_attn_route.py` and
  `tests/test_p107_staged_pin.py` pin the new default; P107's own runner refuses at this commit by design (its tripwire
  asserted the old default).

### P107 read (RTX 5090, #960): DEFAULT=flash -- a 4096-token prefill's device time 906 -> 378 ms; TTFT-4096 1.984 -> 1.742 s (1.14x) on a CPU-bound host; served-prefill NLL within -0.014 / -0.008 ppl of math on 12 fresh windows (bench docs and receipts only)

- **Verdict** (`p107_reduce.py`, the first full draw `p107-5090-3`, AMD EPYC 7663 host, $0.1634): `DEFAULT=flash`, no
  VOID reason.
  - TTFT-512 / TTFT-4096: math 0.249 / 1.984 s; flash 0.235 / 1.742 s. Every draw had the same first token.
  - Quality: mean dppl -0.0140 (c4val1, W=8, one window at -0.079) and -0.0085 (wikitext, W=4). PASS.
  - Census: math 0 flash kernels and 97 SGEMMs; flash 48 flash kernels.
- **The device saving is larger than the wall saving.** The fp32 SGEMMs, masking and softmax (about 477 ms) give way
  to 40 ms of flash attention. On this host the GPU then idles for most of a prefill: the wall moved 242 ms, against a
  528 ms device saving. P102's Ryzen host ran the same math route at 1.373 s.
- **Next lead:** host-side. About 50,000 device-to-device copies per 4096-token prefill remain under both routes.
- `bench/p107/RESULTS-p107.md`; receipts under `bench/p107/receipts/p107-5090-3/`. Lane total $0.1634; two $0
  launcher refusals came first (my exclusion list).

### P108 registered (#359): is e4b's paged path on Gemma-4 worse than transformers' own forward, judged against transformers' own chaos? (bench and tests)

- **Files:** `bench/p108/{PREREG-p108.md,p108_run.sh,p108_drive.sh,p108_box.py,p108_reduce.py,staged.sha256,a2000/}`,
  `tests/test_p108_box.py`, `tests/test_p108_staged_pin.py`.
- **The box:** `google/gemma-4-26B-A4B-it` at `4d7ae49` on one RTX 5090. 32 wikitext windows (1,280-token prompts, so
  the 1,024 sliding window binds; 256-token continuations) run through:
  - the reference (transformers' cached forward, batch 1) and a repeat of it;
  - three floor draws, the same model under arithmetically neutral perturbations: one-shot, chunked prefill, batched;
  - the paged path (fp8 KV, the real kernel);
  - a decode-scale mutant the bar must catch, and a window mutant (reported).
- **The rule:** AT_PARITY iff the paged bias is ≤ the floor's largest |bias| + 0.05 nats and its spread is ≤ 2× the
  floor's.
- **New premise test** `tests/test_gemma4_paged_window_gpu.py`:
  - a tiny dense Gemma-4 whose window binds, through the paged path, within 2× a non-binding control on every one of 8
    seeds;
  - variants: SDPA standing in, and the real fp8 kernel (sm_89+);
  - on the A2000 it reads 0.55–1.33×, and a mutant that ignores the window fails at 13.5–27.2×.
- **Predictions:** AT_PARITY. Paged bias 0 to +0.05, spread ≤ 1.5× the floor's. The scale mutant above +0.3.

### TC1 amendment 14 read (#945): the grouped GEMM's M-tile height from the group sizes makes e4b's training step 3–8 % faster on a 5090

- **What was asked.** On one RTX 5090 (`tc1-5090-43`, $0.41), e4b against itself: grouped-nf4-gemm's M-tile height keyed on the
  largest group (`GNF4_PREFILL_TILE_RULE=max`) against the cost rule over the actual sizes (#441). Shipped and matched arms, two
  draws each, ABBA order.
- **What it read.** P22 and P23 are HELD:

  | arm | cost / max step time | cross-draw range |
  |---|---|---|
  | shipped | **0.924** | 0.918 – 0.930 |
  | matched | **0.968** | 0.964 – 0.971 |

  Held-out loss is unchanged and energy per step lower (register `e4b.train.prefill-tile-rule.qwen3.5090.2026-10-03`).
- **What follows.** `cost` becomes grouped-nf4-gemm's default (#442). Read: `bench/h2h-2026-10-02/tc1/RESULTS-tc1-tileab.md`.

### TC1 amendments 16–18 registered (#835): the steady-state comparison with every new default (P27, read in this release), then P27's ordering on a second host (P28; P29 with the machine excluded by evidence) (bench only)

- #972 (amendment 16), #982 (amendment 17) and #983 (amendment 18) are registration PRs that shipped without a changelog line. They are listed here so that the release names every merged PR.

## 0.41.1 — 2026-10-03 — Gemma-4 serves through `serve_paged` again (a 0.41.0 regression), the paged attention's unbound fallback keeps sliding windows, and fused training makes fewer host syncs and launches

**0.41.1.** Two fixes, three training-path reductions that leave the arithmetic unchanged, and one opt-in serving knob.
No default changes the arithmetic of a path that was already correct.

- **Fix: `serve_paged` sizes a Gemma-4 KV pool again** (#964).
  - 0.41.0's composite-config check (#897) read `num_key_value_heads` first. On Gemma-4's per-layer text config that
    raises transformers' `AmbiguousGlobalPerLayerAttributeError`, a `RuntimeError`, so `build_engine` failed for the
    family.
  - Now `text_config` is read first.
  - CPU-tested. Gemma-4 under `serve_paged` has not been read on a GPU since 0.41.0.
- **Fix: the paged attention's unbound fallback keeps sliding windows and aligns a cached chunk to its last key**
  (#966).
  - transformers builds no mask for an attention implementation with no mask function. A registered model run with no
    paged context therefore dropped Gemma-4's and gpt-oss's windows, and took torch's top-left `is_causal` in a cached
    chunked prefill.
  - The fallback now builds the mask itself. The bound serving path builds its own masks and is untouched.
- **Fused training (the TC campaign, #945).** Each change is exact.
  - **Grouping reads the per-expert counts back once, not five times** (#946).
  - **The rotary embedding trains through one Triton launch each way** (on by default; `E4B_FUSED_ROPE=0` turns it
    off). It is bit-identical to the composite: `tests/test_rope_train.py` asserts `torch.equal`.
  - **LoRA skips the scaling multiply at exactly 1** (#959).
  - **What PyPI users get.** TC1's reading of 1 host sync per fused MoE layer pass instead of 13 (13–15 % faster on a
    5090) also needs grouped-nf4-gemm's pinned index ring (grouped-nf4-gemm#438, its default since #439). That is on
    grouped-nf4-gemm's main and not yet in a release, so with grouped-nf4-gemm 0.34.1 from PyPI this release delivers
    #946's share only.
- **`E4B_PAGED_PREFILL_ATTN`** (#960) is a route knob for paged prefill attention: `math`, the default and unchanged,
  or `flash`. Lane P107 (#960) is registered to read it.
- **Docs.**
  - `docs/SERVING.md` gives the Gated DeltaNet kernels' measured quality against the torch path (P106: KL 5.7e-3 /
    4.9e-3 nats, d_nll within ±0.001) and their prefill TTFT (1.10–1.12×).
  - `docs/ARCHITECTURE_SUPPORT.md` names the modules that stay bf16 on Qwen3.6 / Qwen3-Next (#899).

### TC1 amendment 11 read (#945): the steady-state comparison after the sync fix is UNTESTED -- an unstable host

- **What was asked.** On one RTX 5090 (`tc1-5090-40`, Xeon E5-2696 v4, $1.50), amendment 8's 200-step comparison of e4b as shipped
  against axolotl's scattermoe native-best, re-run after #945. P18 predicted the late-window ratio in [0.97, 1.15].
- **What it read.** UNTESTED: e4b's late-window draws are 12.0 % apart and axolotl's 9.1 %, both outside the registered 5 %, with
  the drift alternating between frameworks (the host). Reported, not quoted: e4b finishes 200 steps first (1,042–1,182 s against
  1,781–1,910 s), and axolotl reaches the matched held-out curve.
- **What follows.** No register row; `.native-steady-state` stands for the code before #945. The question is re-asked on a stable
  host after amendments 13–15. Read: `bench/h2h-2026-10-02/tc1/RESULTS-tc1-steady945.md`.

### P107 registered (#960): paged prefill attention's route A/B -- math (SDPA's fp32 math backend, today) against flash (#963's lower-right causal bias) on one engine, gated by the calibrated K8 rule on the served-prefill NLL (bench and tests)

- `bench/p107/{PREREG-p107.md,p107_run.sh,p107_drive.sh,p107_box.py,p107_reduce.py,staged.sha256}`,
  `tests/test_p107_served_prefill_scorer.py`, `tests/test_p107_staged_pin.py`.
- **Why:** under k19, P102's 4096-token profile spends about 445 of 924 device-ms in fp32 SIMT SGEMMs, masking and
  softmax -- prefill attention on SDPA's math backend, because a boolean mask with GQA rules out every fused kernel.
- **The box:** one engine, `E4B_PAGED_PREFILL_ATTN` switched between requests. TTFT at 512 and 4,096 tokens over three
  rotated rounds; a 512-token kernel census per route; and the SERVED-PREFILL NLL on 12 fresh windows (c4val1 W=8,
  wikitext W=4), each window's 2,560 tokens run through the paged prefill path in 512-token chunks and its 2,048
  predictions after the prompt scored. The decode-shaped K8 and P102's eager-attention NLL never score a logit this code produces.
- **The rule:** `DEFAULT=flash` iff flash PASSES the K8 rule (|mean dppl| <= 0.05 per text) and its TTFT-4096 is at
  most 0.9x math's.
- **The scorer** equals one non-paged forward within 1e-4 on a tiny Qwen3 for both routes (CPU test), and two of its
  mutations fail that test.
- **Predictions:** PASS with |mean dppl| <= 0.01; TTFT-4096 1.25-1.55 s -> 0.85-1.10 s; `DEFAULT=flash`.

### E4B_PAGED_PREFILL_ATTN (#960): a route knob for paged prefill attention -- math (the default, unchanged) or flash (the same lower-right causal mask as a bias the flash kernel takes)

- **Why.** For a layer without sinks or a sliding window, the paged prefill branch called SDPA with an explicit
  boolean mask and `enable_gqa`. With fewer KV heads than query heads, flash attention refuses a non-null mask and
  memory-efficient attention refuses mismatched head counts, so the call lands on SDPA's math backend: fp32, scores
  materialised.
  - P102's profile put this at ~45 % of a 4096-token Qwen3 prefill's device time under k19.
  - On an A2000 at Qwen3's shapes the math backend took 19.1 ms per 4096-context call, against 1.31 ms for flash.
- **What.** `paged_attention._prefill_attn_mode_env` reads `E4B_PAGED_PREFILL_ATTN` at every call; an unknown value
  is refused.
  - `math` (the default) leaves behaviour unchanged.
  - `flash` passes `causal_lower_right(T, t_total)` with `enable_gqa`, with no K/V expansion.
  - Layers with sinks or a sliding window keep the explicit-mask path under either value.
  - The default moves only by a registered reading (lane P107).
- **Tests.**
  - `tests/test_paged_prefill_attn_route.py` (CPU, 7): the knob; both routes equal the whole-sequence reference across
    chunk boundaries; `flash` hands SDPA a `CausalBias` with GQA on a plain layer, and never on a windowed or sink layer.
  - `tests/test_paged_prefill_attn_route_gpu.py`: Qwen3 shapes, three 512-token chunks; `flash` reaches the flash
    kernel and stays within 1 % of `math` (0.21 % on an A2000).
  - On the A2000: CPU arm 25 passed with CUDA hidden, GPU arm 26 passed. Two mutations each failed: the flash branch
    removed, and a top-left causal bias.


### Fix: `serve_paged` sizes a Gemma-4 KV pool again (a 0.41.0 regression from #897)

- **The bug.** `serve_paged._kv_geometry` read `num_key_value_heads` before looking for a composite config's
  `text_config` (#897, for Qwen3.5 / Qwen3.6 MoE). On Gemma-4's per-layer text config, which is the config of the model
  the streaming loader builds, that read raises transformers' `AmbiguousGlobalPerLayerAttributeError`. That error is a
  `RuntimeError`, so `getattr`'s default did not catch it, and `build_engine` died before reaching the per-layer branch.
  Found while drafting a Gemma-4 parity lane for #359 (unregistered; the number P107 belongs to #960's lane).
- **The fix.** `text_config` is read first. The per-layer branch is unchanged.
- **Tests** (`tests/test_linear_state.py`), using transformers' own Gemma-4 configs:
  - a tiny per-layer text config, which with the fix reverted fails with exactly that error;
  - a composite config wrapping it.

  The existing heterogeneous-config test covered the harness's copy (`bench/hybrid-g9/step_decomp.py`) only.
- **Scope.** CPU-tested. Gemma-4 under `serve_paged` has not been read on a GPU since 0.41.0.

### Fix: the paged attention's unbound fallback keeps sliding windows and aligns a cached chunk to its last key

- **The bug.** A model with `e4b_paged` registered and no paged context bound (the "transformers reference" in
  P97-style instruments, or any direct forward after `register`) falls back to SDPA. transformers builds no attention
  mask for an implementation it has no mask function for, and `e4b_paged` registers none, so the fallback ran with no
  mask at all:
  - it dropped sliding windows (Gemma-4, gpt-oss): a tiny Gemma-4's last logits moved by 0.85 at window 16;
  - in a cached chunked prefill it took torch's top-left `is_causal` alignment.
- **The fix.** When transformers passes no mask, the fallback builds the causal mask aligned to the last key, with the
  layer's window. It builds none where SDPA's own handling is already right: a square prefill with no window, or one
  query whose keys all sit inside the window.
  - No mask function is registered, because one would run HF's mask preprocessing inside the bound, graph-captured
    forwards.
  - The bound paged path builds its own masks and is untouched.
  - The fallback assumes unpadded batches.
- **Tests** (`tests/test_paged_attention_unbound_mask.py`, a tiny Gemma-4 whose window binds):
  - a one-shot forward and a cached chunked prefill with decode both match transformers' own attention;
  - with the fix reverted, both fail on ~60 % of logits, by up to 1.26.
- **Not affected.** Serving, which binds the context. P97's readings (Qwen3.6 and OLMoE: no windows, unpadded windows,
  square prefill). P106 (paged against paged). Found while drafting a Gemma-4 parity lane for #359
  (unregistered; the number P107 belongs to #960's lane).

### The rotary embedding trains through one Triton launch each way, bit-identical to the composite (on by default; `E4B_FUSED_ROPE=0` turns it off)

- **What.** `experts4bit_qlora.engines.rope_train`: `rope_qk` computes `q * cos + rotate_half(q) * sin`, for q and k, in one launch
  forward and one per tensor backward. Every op in Hugging Face's composite is elementwise, and a product of two bf16 values is
  exact in fp32. The kernel therefore rounds each product to bf16 and then the sum, and this reproduces the composite byte for byte,
  forward and gradients. `tests/test_rope_train.py` asserts `torch.equal` at three shapes, broadcast cos/sin included.
- **A Triton trap, worked around.** Triton 3.4 folds a `.to(bfloat16).to(float32)` round trip that feeds an add: 14 % of a
  product-plus-add's elements came out unrounded. The kernel rounds with integer arithmetic on the fp32 bits (`_rne_bf16`), which
  the compiler cannot elide.
- **Scope.** `enable_fast_train` points the `apply_rotary_pos_emb` of the model's own attention modules at the kernel. Only a
  function with the `(q, k, cos, sin, unsqueeze_dim=1)` contract is replaced, any call outside bf16 CUDA `[B, H, L, D]` shapes goes
  to the original, and `disable_fast_train` unwinds it.
- **Measured on an RTX A2000** (2-layer Qwen3-MoE at Qwen3-30B-A3B's layer shape, e4b's fused training step, ABBA): launches per
  step 1,344 → 1,275, which would be about 6,600 fewer a step at the 48-layer accum-4 field recipe. No step-time effect is claimed.

### `E4B_FUSED_RMSNORM=1`: frozen RMSNorms train through one Triton launch each way (opt-in; default unchanged)

- **Why.** A Qwen3-MoE layer runs four Hugging Face RMSNorms: input and post-attention, and the per-head q and k norms. Each is a
  composite of about 8 kernels forward and 10 backward, and gradient checkpointing runs the forward twice. TC1's RTX 5090 profile
  (`tc1-5090-41`) puts that at roughly 20,000 of the shipped step's 134,000 device events and 17 % of its CPU op time.
- **What.** `experts4bit_qlora.engines.rmsnorm_train`:
  - `rmsnorm_frozen` is a one-launch Triton forward and a one-launch backward. The backward returns dx only, because QLoRA freezes
    the norm weights. It mirrors the composite's casts: fp32 statistics, the normalised value rounded to the input dtype, then the
    weight multiply rounded once, and `g * w` rounded the same way in the backward.
  - `enable_fused_rmsnorm_train` patches only norms that are frozen, structurally matched and probe-verified, the decode fusion's
    rules (a centered `x * (1 + w)` variant is never patched). Other inputs fall through to the composite.
  - `enable_fast_train` applies it when `E4B_FUSED_RMSNORM=1`.
  - The result is not bit-identical: row reductions run in another order. On an RTX A2000 about 1 element in 100,000 differs, by one
    bf16 ulp, in the forward and in dx (`tests/test_rmsnorm_train.py`).
- **Measured on an RTX A2000** (2-layer Qwen3-MoE at Qwen3-30B-A3B's layer shape, r16 alpha 16, e4b's fused training step, ABBA):
  launches per step 1,344 → 1,109, device time 216.2 → 206.9 ms, wall 227.6 → 216.4 ms. The 5090 effect and a held-out quality
  reading come from TC1 before any default changes.

### LoRA: the delta's scaling multiply is skipped at exactly 1 (exact; fewer launches at alpha == r)

- **What.** `LoRALinear.forward` (the attention adapters), `ExpertsLoRA._lora` (the reference expert path) and the batched engine's
  padded delta used to compute `scaling * delta` even at scaling 1.0. They now go through `lora._scaled`, which returns the delta
  unchanged when `scaling` is exactly 1 and applies any other value (a tensor `scaling` is always applied). `1.0 * x == x` for
  every float, so outputs and gradients are bit-identical (`tests/test_lora_unit_scaling.py`: 5 (r, alpha) pairs × fp32 and bf16
  adapters, against the explicit multiply). This is the same rule as grouped-nf4-gemm#440's `_scaled` for the fused expert delta.
- **Measured on an RTX A2000** (2-layer Qwen3-MoE at Qwen3-30B-A3B's layer shape, r16 alpha 16, fp32 attention adapters, ABBA):
  launches per step 1,344 → 1,320. That is 4 projections × the forward, recompute and backward. At the 48-layer, accum-4 field
  recipe that would be 2,304 fewer launches a step. No step-time effect is claimed.

### TC1 amendment 13 read (#945): grouped-nf4-gemm's trimmed LoRA delta makes e4b's training step 6–9 % faster on a 5090

- **What was asked.** On one RTX 5090 (`tc1-5090-42`, $0.33), e4b against itself on the post-#945 sync path: grouped-nf4-gemm's
  previous padded LoRA delta (`NF4_QLORA_LEAN_DELTA=0`) against its trimmed body (#440). Shipped and matched arms, two draws each,
  ABBA order.
- **What it read.** P20 and P21 are HELD:

  | arm | trimmed / previous step time | cross-draw range |
  |---|---|---|
  | shipped | **0.939** | 0.935 – 0.944 |
  | matched | **0.911** | 0.898 – 0.925 |

  Held-out loss is unchanged, matched peak VRAM is 0.6 GB lower, and energy per step is lower (register `e4b.train.lora-delta-lean.qwen3.5090.2026-10-03`).
- **What follows.** The trimmed body stays grouped-nf4-gemm's default. Read: `bench/h2h-2026-10-02/tc1/RESULTS-tc1-leanab.md`.

### SC1 read, box A's third draw (#846): P13 holds on all three boxes (vLLM / e4b scheduler 1.167-1.203 at B=1, 1.167-1.177 at B=16); P7, P8 and P11 REFUTED, P14 HOLDS; the licence is QUALITY_FAIL a third time (bench docs and receipts only)

- `sc1a-5090-4` (adertha-receipts `c4915d3`, e4b `c19dd52` with A13, Ryzen 9 7950X, $1.9335): every arm VALID, none
  skipped. It ran at the launcher's 6.0 h cap; `sc1a-5090-3` was refused at $0 for A13's 7.5 h.
- **P13 HOLD:** B=1 A 1.187 / B 1.203 / C 1.167 (gap 3.1 %); B=16 A 1.167 / B 1.172 / C 1.177 (gap 0.8 %).
- **P7 REFUTED:** e4b's SAMEPROMPT ×1.18 < 1.3; vLLM's ×1.62 holds.
- **P8 REFUTED:** fp8-KV ×1.079 at B=1 (> 5 %); ×1.110 at B=16; Δ_bf16 +0.003 / +0.004 holds.
- **P11 REFUTED:** vLLM / e4b J/token 0.751 at B=1.
- **P14 HOLD:** 1.016 / 0.996.
- e4b's TTFT on this host is 0.82 / 7.01 s against 2.0 / 16.4 s on the EPYC boxes: the prefill loop is host-bound.
- Lane total $15.9047 over 40 receipts. `RESULTS-sc1-cross-box-a3.md` is added beside the first read.


### TC1 amendment 12 read (#945): after the sync fix, e4b's matched step is 0.740 device-busy (was 0.595) and the next target is the LoRA path's launch volume

- **What was asked.** TC1's profile instrument on one RTX 5090 (`tc1-5090-41`, $0.27). Three arms:
  - e4b shipped on the post-#945 path;
  - e4b matched on the post-#945 path;
  - e4b matched on the legacy path, as the before-picture on the same host.
- **What it read.** P19 is HELD: the matched arm's device-busy fraction went from 0.595 to **0.740**, and the shipped arm reads 0.830. Device
  work is unchanged (2,629 against 2,632 ms a step), and the legacy arm's 4,436 stream syncs a step are gone.
- **Where the time goes.** On the shipped arm the grouped GEMM takes 811 + 294 ms of 2,049 ms device time per step. It runs at under a tenth
  of the card's rooflines at this fixture's ~24 to 32 rows per expert. The next kernel is a bf16 scalar multiply of the padded expert LoRA
  delta, 94 ms with scaling 1.0. The matched arm's fp32 LoRA `bmm` costs 607 ms of CPU time a step.
- **What follows.** By amendment 12's rule the step is launch-bound, so the first change targets the LoRA path:
  - grouped-nf4-gemm#440, an exact trim of the padded delta;
  - amendment 13 (#954), which registers its 5090 A/B.

  Read: `bench/h2h-2026-10-02/tc1/RESULTS-tc1-prof945.md`, receipts `receipts/tc1-5090-41/`.

### Docs: which Qwen3.6 / Qwen3-Next modules stay bf16, and what that costs (#899)

- `docs/ARCHITECTURE_SUPPORT.md` gains a `qwen3_5_moe` / `qwen3_next` note.
  - Only the routed experts, and on request (`TRAIN_ATTN_4BIT=1`) the full-attention projections, are 4-bit.
  - The Gated DeltaNet projections, the shared experts and their gates, and `lm_head` stay bf16: about 1.14 B
    parameters and ~1.6 GB of VRAM over NF4 on Qwen3.6-35B-A3B (TC2's census).
  - In serving, the attention projections stay bf16 as well.
- The default is kept and stated as deliberate but unpriced. An opt-in would need a KL reading against it first.

### P106 read (RTX 5090): NEUTRAL -- against transformers' torch path, the Gated DeltaNet kernels cost Qwen3.6 nothing measurable in nats (KL 5.7e-3 prefill / 4.9e-3 decode, d_nll within +-0.001); prefill TTFT 1.10-1.12x (#944)

- `bench/p106/RESULTS-p106.md`, `bench/p106/receipts/p106-5090-1/` (one RTX 5090, Intel Core Ultra 9 285K; $0.2559,
  lane $0.3472).
- **Quality**, both paths in one process, teacher-forced on 8 wikitext windows, on fp32 log-probs:
  - KL(torch || kernels) 5.69e-3 nats over 16,384 prompt positions and 4.86e-3 over 512 decode steps, about the fp8
    KV's own effect (P97: 4.43e-3);
  - argmax agreement 0.969 / 0.973;
  - d_nll +3.3e-5 / -8.4e-4 nats;
  - the null pair bit-identical; the l2norm-off mutant's logits went NaN (agreement 0.0).
- **TTFT** at one request: 1.124x (512 tokens), 1.106x (2,048), 1.101x (4,096), about 32 us saved per prompt token.
  The predicted 1.3-4x at 4,096 missed: the Gated DeltaNet layers are about 10 % of prefill on this stack.
- **Registered consequence (NEUTRAL):** `docs/SERVING.md` replaces "quality ... not measured" with the measured
  numbers and adds the TTFT ratios; the recommendation stands. `docs/STATUS.md`; register row
  `e4b.serve.p106.qwen36-gdn-kernel-quality.5090.2026-10-03`; `llms-full.txt` regenerated.

### P106 registered (#944): the Gated DeltaNet kernels' quality against transformers' torch path, and their prefill TTFT, on Qwen3.6 -- both paths switched in one process (bench and tests)

- `bench/p106/{PREREG-p106.md,p106_run.sh,p106_drive.sh,p106_box.py,p106_reduce.py,gdn_toggle.py,toggle_probe.py,staged.sha256}`,
  `tests/test_p106_box.py`, `tests/test_p106_staged_pin.py`.
- **`gdn_toggle.GdnToggle`** rewrites the three cells of transformers' fallback closure for each Gated DeltaNet function
  to the torch function's values (what the closure holds without the packages) and restores them. One engine then
  serves both paths. On the A2000 the switched-off path equals a kernel-free process bit for bit (0 differing logits,
  tiny dense and MoE hybrids, under fla and under fla + causal-conv1d).
- **The box:** 8 wikitext windows x (2,048 prefill + 64 teacher-forced decode steps), the two paths in lockstep on fp32
  log-probs, with a null pair (the torch path against itself) and an l2norm-off mutant. TTFT at 512 / 2,048 / 4,096
  tokens is alternated over 5 rounds.
- **The rule:** NEUTRAL iff, on prefill and on decode, mean KL(torch || kernels) <= 0.05, agreement >= 0.85 and d_nll
  <= +0.01. TTFT is reported only.
- **Predictions:** NEUTRAL; KL 1e-3 to 2e-2; TTFT 1.3-4x at 4,096 tokens.

### Fused expert grouping reads the per-expert counts back once, not five times (#945)

- **Why.** `enable_fast` / `enable_fast_train` sized the grouped launch from host-side counts built with `bincount`,
  `nonzero` and two `.tolist()` calls: five synchronizing calls per grouped forward (measured with
  `torch.cuda.set_sync_debug_mode`). Under gradient checkpointing, a fused MoE training layer pass paid 10 of its 13
  host syncs on this grouping and grouped-nf4-gemm's index transfers.
- **What.** `engines/fast.py::_group_by_expert` counts with `scatter_add_` and reads the counts back with one
  `.tolist()`. The lists are identical: the non-empty experts in ascending id with their counts, as Python ints. All
  three grouped call sites use it. `E4B_GROUPING=legacy` restores the old form for A/B measurement.
- **Measured on an RTX A2000** (Qwen3-30B-A3B layer shape, 512 tokens, bf16 adapters): forward syncs per layer
  10 -> 6, and forward launches 89 -> 79. With grouped-nf4-gemm's opt-in `GNF4_PINNED_RING=1` (grouped-nf4-gemm#438),
  syncs are 1 forward and 0 backward. The 5090 training-step effect is measured separately.
- **Tests** (`tests/test_fast_grouping.py`):
  - the same lists as the legacy form on CPU and CUDA, including empty experts;
  - one synchronizing call against the legacy form's four or more (the control);
  - the fused training forward and every gradient bit-identical under either grouping.

### TC1 amendment 10 read (#945): one host sync per fused MoE layer pass instead of 13 makes e4b's training step 13-15 % faster on a 5090

- **What was asked.** On one RTX 5090 (`tc1-5090-38`), e4b against itself: legacy grouping with grouped-nf4-gemm's pageable
  index copies (13 host syncs per MoE layer pass), against #946's single-read grouping with grouped-nf4-gemm's pinned ring
  (1). Shipped and matched arms, two draws each, ABBA order.
- **What it read.** P16 and P17 are HELD:

  | arm | new / legacy step time | cross-draw range |
  |---|---|---|
  | shipped | **0.866** | 0.843 – 0.889 |
  | matched | **0.847** | 0.840 – 0.854 |

  Held-out loss and peak VRAM are unchanged, and energy per step is lower. Every new-path arm's ring staged 53,680
  transfers; no legacy arm's did (register `e4b.train.host-syncs.qwen3.5090.2026-10-03`).
- **What follows.** By the registered decision rule, the ring becomes grouped-nf4-gemm's default outside capture. The
  cross-framework positions measured before this change stand as measured.

## 0.41.0 — 2026-10-03 — hybrid linear-attention models (Qwen3.5 / Qwen3.6 MoE, Qwen3-Next) serve paged, under decode graphs, with the Gated DeltaNet kernels recommended; the int4 store's prefill takes K19 by default (`E4B_INT4_PREFILL=auto`, lane P102: TTFT-4096 7.21 s -> 1.37 s); bucketed decode graphs on an NF4 MoE no longer replay against a freed index (#913)

**0.41.0.** One default changes, one feature lands, and one fix ships.

- **`E4B_INT4_PREFILL` is `auto`** (#916, lane P102). The knob is new in this release (#916). Its routes for the
  int4 store's host-grouped prefill calls are `loop`, `batched`, `k19`, `mtile` and `auto`, and `auto` is the default.
  - With the variable unset, the uniform-int4 store's host-grouped prefill calls take K19 when the kernel package
    carries K19 and a CUDA device is up, and the per-expert loop otherwise. CPU-only behaviour is unchanged.
  - **Measured:** on an RTX 5090 with Qwen3-30B-A3B at `max_seqs` 1, TTFT-4096 went from 7.21 s to 1.37 s and TTFT-512
    from 0.85 s to 0.11 s. The prefill-shaped NLL stayed within +0.0115 / +0.0070 ppl of the loop, under the
    calibrated K8 rule with a 0.05 budget.
  - **Also moved:** with device grouping on (`max_seqs > 1`), prefill rows above 256 move from the int8 M-tile to K19.
    T == 1 decode is untouched.
  - **Restoring the old route:** `E4B_INT4_PREFILL=loop`. K19 needs grouped-nf4-gemm 0.34.0 or newer; with an older
    kernel package `auto` resolves to the loop.
- **Hybrid linear-attention models serve paged** (#889, #897, #905, #907, #908).
  - `PagedModelRunner` keeps a per-slot Gated DeltaNet state pool.
  - `serve_paged` builds a hybrid checkpoint with an fp8 KV pool sized to the attention layers only.
  - Decode graphs (`E4B_PAGED_GRAPHS=1`) capture the per-slot state.
  - Mamba-style layers, which transformers also labels `linear_attention`, are refused.
  - **Read on an RTX 5090 with Qwen3.6-35B-A3B:**
    - P97: the pooled state stays transformers' own, and the whole model tracks transformers' forward at 4.43e-3
      nats.
    - P101: every decode bucket replays exactly as the padded eager step; W16 449 tok/s, W1 79.7.
    - P105: with `flash-linear-attention==0.5.2` and `causal-conv1d==1.7.0` installed, graph decode runs 1.135× (W16)
      and 1.118× (W1) the torch path's on the same host. **Recommended.** See `docs/SERVING.md` for the two caveats:
      token streams differ from the torch path's, and chunked prefill depends on chunk boundaries.
- **Fixed: bucketed decode graphs on an NF4 MoE** (#913, #918).
  - **The bug:** a later bucket's capture warm-up freed the row-to-token index that an earlier bucket's graph still
    read. On Qwen3.6 that was a device-side assert (lane P98).
  - **Who was exposed:** any NF4 MoE served with `E4B_PAGED_GRAPHS=1` and `E4B_PAGED_MAX_SEQS>1` on 0.40.0. The
    earlier serving lanes' bitwise graph gates held (#913's exposure audit).
  - **The fix:** the index cache is keyed per (row count, top-k, device). A GPU test reproduces the fault with the fix
    reverted.
- **Tests:** `tests/test_linear_state_chunk_matched_gpu.py` and `tests/test_linear_state_dense_parity_gpu.py`.
  `tests/test_linear_state_gpu.py` is unchanged. Its single-seed tiny-MoE statistic is a seed lottery (lane P104), and
  this is documented, not edited, because six lanes pin its bytes.
- **Dependencies:** CI still installs grouped-nf4-gemm at the v0.34.1 commit; the `[fast]` floor stays
  `grouped-nf4-gemm>=0.30.0`.
- **The rest is evidence and bench:**
  - lanes P97–P105 (hybrid serving, its graphs, the Gated DeltaNet kernels);
  - P100 and P102 (the int4 prefill route);
  - lane SC1's and TC1–TC3's amendments and reads.

### SC1 amendment A13 (#846): box A's third draw -- a 7.5 h guard so Phase G2 runs, and the int4 prefill route pinned to the `loop` every box ran (#937 made `auto` the default afterwards) (bench and tests only)

- The read (#934) left P7, P8, P11, P14 and P13's third box unread. `sc1a-5090-2` ran out of its 5.5 h guard before
  Phase G2, and its scheduler anchor read UNSTABLE on the wall-slope estimator A10 replaced.
- `sc1a-5090-3` runs at this merge, after `sc1a-prove-12`, with a 7.5 h guard (<= $5.63). G2 needs about 70 min by the
  first draws' phase lengths.
- The box scrubs `E4B_INT4_PREFILL` and exports `loop` before the tripwire and every arm, and the tripwire asserts it.
  `ROUTEENV` is left as it was, because P100's and P102's pins compare it byte for byte. Without the pin, the redraw's TTFT, scheduler prefill and K8 licence prompt would run K19 while every other
  SC1 row ran the loop.
- Recorded, not changed: #918's row-index fix is in this draw and was not in the others. SC1's scheduler rows are speed
  only, and the fix changes the gathered rows, not the work.
- `tests/test_sc1_a13.py` (6; three fail on the registered script). `staged.sha256` is re-pinned.
### TC1 P15: axolotl's steady-state lead over e4b as shipped replicates on a second host

- **What was asked.** TC1 amendment 9 asked P14 again on a different machine (`tc1-5090-36`: Vast machine 45501, AMD Ryzen 9 3900X),
  with the harness and package byte-identical to `tc1-5090-35`'s.
- **What it read.** P15 is HELD: over steps 101..200, axolotl scattermoe / e4b shipped is **0.901 [0.892, 0.910]**, the whole interval
  below 1.0, and the point 0.001 above the band's lower bound. axolotl is faster at steady state on two hosts with different CPU
  classes (0.911 on the first). The register row `e4b.train.h2h.axolotl.qwen3.5090.2026-10-02.native-steady-state` now covers both.
- **Beside it, as on the first host.** e4b finishes the 200-step run first (743-752 s against 1,219-1,231 s of summed step time).
  axolotl spends ×1.39 the energy per step. axolotl's held-out sits within 0.003 of the matched curve, and e4b as shipped 0.026 above it.

### E4B_INT4_PREFILL defaults to auto (#916, lane P102's DEFAULT=k19): k19 where K19 can run, else loop -- TTFT-4096 7.21 s -> 1.37 s on an RTX 5090 at max_seqs 1

- **The default** (`hot_residency._int4_prefill_mode_env`) is `auto`, also when unset. It resolves to `k19` when the
  kernel package carries K19 and a CUDA device is up (`_k19_prefill_available`, read once per process), and to `loop`
  otherwise, so CPU-only behaviour is unchanged. Every route stays selectable: `loop`, `batched`, `k19`, `mtile`.
- **Why:** lane P102 (#931) read `DEFAULT=k19`. On an RTX 5090 with Qwen3-30B-A3B at `max_seqs` 1, TTFT-4096 went from
  7.207 s to 1.373 s and TTFT-512 from 0.851 s to 0.113 s. The prefill-shaped NLL stayed within +0.0115 (c4val1) and
  +0.0070 (wikitext) ppl of the loop over 12 fresh windows: the calibrated K8 rule, budget 0.05.
- **What it also moves:** with `DEVICE_GROUPING` on (`max_seqs > 1`), prefill rows above 256 move from the int8 M-tile
  to K19. T == 1 decode is untouched.
- **Tests.**
  - Five existing tests about other knobs now pin the prefill route they assumed: `loop` for the A16 collapse and the
    collapsed-grouping decisions; `mtile` for the two device-grouping tests and the K19 decode opt-in's prefill rows.
  - `test_int4_prefill_route.py` pins `auto`'s two resolutions.
  - P102's staged-pin test now pins the knob's new default string. P102's own tripwire, which requires `loop`, refuses
    at later commits by design: the lane is read and closed.
- **Docs.** The `int4_experts.py` Scope note, the knob's docstring, and a `docs/SERVING.md` paragraph citing P100 and
  P102.

### P105 read (RTX 5090): SUPPORTED, recommend -- with flash-linear-attention and causal-conv1d, Qwen3.6's hybrid paged path holds its dense parity premise and replays every decode graph exactly; graph decode 1.135x (W16) / 1.118x (W1) the torch path's on the same host (#928)

- `bench/p105/RESULTS-p105.md`, `bench/p105/receipts/p105-5090-1/`.
- All three phases SUPPORTED by P98's rule. Graph decode tok/s t / f / fc: W16 472.7 / 525.0 / 536.5, W1 90.1 /
  98.9 / 100.7. That saves 1.6 ms per step at one row and 3.3 ms at sixteen; fla carries 82-83 % of the gain.
- Plain eager 1.11-1.12x. The padded-eager oracle ran 0.87x with the kernels (not explained).
- The kernels change greedy trajectories: 41-57 % positional agreement with the torch path. Quality in nats is not
  measured.
- Against the predictions:
  - held: SUPPORTED + recommend, W1 band, f's share, memory;
  - missed: W16 band (1.135x below 1.15) and fc plain eager W16 (1.113x below 1.15).
- The registered consequence:
  - `docs/SERVING.md` lists the two kernels as supported and recommended (install line, gain, two caveats);
  - `docs/STATUS.md`;
  - register row `e4b.serve.p105.qwen36-gdn-kernels.5090.2026-10-03`.
- $0.4280 (proof + reading). The kernel question across P103-P105: $0.6137.

### P105 registered (#928): the Gated DeltaNet kernels under the hybrid paged path with a premise that is a distribution -- a dense hybrid within its control on every one of 8 seeds (bench and tests)

- `bench/p105/{PREREG-p105.md,p105_run.sh,p105_drive.sh,p105_reduce.py,staged.sha256}`, `tests/test_p105_staged_pin.py`.
- **New test** `tests/test_linear_state_dense_parity_gpu.py`: a dense hybrid through `PagedModelRunner` with the fp8
  KV pool, against transformers' chunk-matched cache, within 2x its dense all-attention control on every one of 8
  seeds. Greedy tokens are reported, not gated (the control itself flips argmax near-ties). Variants: SDPA standing in
  (any CUDA) and the real fp8 kernel (sm_89+).
  - On the A2000 the ratio is 0.66-1.60 on all seeds under the torch path, fla and fla + causal-conv1d.
  - A mutant that swaps sequences' states fails at 9.2-14.6x on every seed.
- **The lane:** P104's, with the premise replaced:
  - phase t: 11 passed;
  - kernel phases: the graph files, the chunk-matched all-linear test and the dense parity file, 9 passed;
  - the single-seed tiny-MoE checks (a seed lottery, P104) are reported only.
- **Predictions:** SUPPORTED with `recommend`; fc's graph arm 1.15-1.60x phase t's on W16 and 1.02-1.15x on W1.

### P104 stopped at its proving rental (#928), no verdict; P103's cause corrected: the hybrid parity premise's single-seed MoE statistic is a lottery in every kernel arm, the torch path included -- not the kernels, not e4b

- `bench/p104/RESULTS-p104.md`, `bench/p104/receipts/p104-prove-1/`, `bench/p104/a2000/` (probes 7 and 8).
- **The proving rental** (`p104-prove-1`, one RTX 5090, $0.0527):
  - premise t passed 9;
  - premises f and fc each read 1 failed, 7 passed: the chunk-matched fp8 hybrid test at 7.54e-2 / 8.13e-2 against a
    4.05e-2 bound, tokens equal;
  - the four chunk-matched all-linear tests and the graph tests pass in every phase.
- **The cause, at $0 on the A2000** (e4b's `PagedModelRunner` with SDPA standing in for the fp8 kernel, 10 seeds):
  - the test's 4-expert MoE hybrid is bimodal in every arm, with 5/10 torch-path seeds above the test's own bound;
  - a dense hybrid is stable (2.1-3.9e-2) and identical across the torch path, fla and fla + causal-conv1d;
  - seed 0 (the test's) lands low on the torch path and high under fla.
- **Corrected:** `bench/p103/RESULTS-p103.md` carries a dated correction withdrawing its attribution to fla's
  chunk-split variance. That variance is real, but not the cause. `docs/SERVING.md` states the corrected cause.
- `tests/test_linear_state_gpu.py` is unchanged; six lanes pin its bytes.

### SC1 read (#846): no position quoted -- box A's licence reads QUALITY_FAIL, which blocks every box. Measured, vLLM leads e4b's scheduler by 1.17-1.20x at B=1, SGLang by 1.27x, llama.cpp Q4_K_M by 1.48x; P13 holds on boxes B and C (bench docs and receipts only)

- **Receipts:** `sc1a-5090-2` (e4b `32d424e`), `sc1b-5090-3` and `sc1c-5090-5` (`9dd712b`), read with main's reducer
  (A10 census, A11 cross-box). Files under `bench/h2h-2026-10-02/sc1/`.
- **Licence QUALITY_FAIL** on both box-A draws: wikitext -0.014 / -0.011 ppl, c4val1 +0.056 / +0.040 ppl. Under the
  registered rule no position is quoted on any box. P1-P5 read UNREAD and the register does not move.
- **Measured** (comparator / e4b `int4_sched`):
  - vLLM: 1.203 (B) / 1.167 (C) at B=1, 1.172 / 1.177 at B=16.
  - SGLang 0.5.20: 1.268 / 1.191.
  - llama.cpp Q4_K_M: 1.484 / 0.624.
  - ExLlamaV3: 0.899 at B=1, B=16 UNSTABLE.
  - LMDeploy UNSUPPORTED on sm_120.
  - Served quality is CLOSE to e4b for vLLM, SGLang and llama.cpp, and COMPARABLE (closer to bf16) for ExLlamaV3.
- **Predictions:**
  - P5b HOLD (UNSUPPORTED), P12 HOLD, P13 HOLD on B and C (box A's scheduler anchor UNSTABLE on the pre-A10 estimator).
  - P6, P9 and P10 REFUTED: e4b's TTFT-4096 is 16.3 s on the EPYC boxes and 27.6 s on box C's Xeon, against vLLM's
    0.17-0.21 s. That is P100's host-bound per-expert prefill loop; P102 read `DEFAULT=k19` for it.
  - P7, P8, P11 and P14 UNREAD; box A's host-limited deadline skipped their arms.
- SGLang's prefill-shaped rows are HARNESS_ERROR, fixed by A12 (#933) without a re-run. Lane total $13.8566 over 37
  receipts.
### SC1 amendment A12 (#846): the SGLang prefill scorer reads entry 0 as v0.5.20 emits it -- a (None, id, None) triple, not a bare None (bench and tests only)

- `sc1c-5090-5` (adertha-receipts `94a8f5e`, $1.2626) read every box-C speed arm VALID. Its rc 1 is the two
  `nll_sglang_prefill_*` scorings, which died on `assert lps[0] is None`. SGLang prepends None to the logprob values and
  zips them with the ids (`logprob_result_processor.py:38`, `:50`; `tokenizer_manager.py:2888-2892`), so entry 0 is
  `(None, ids[0], None)`.
- The scorer now requires entry 0's logprob to be None and its token to be `prompt[0]`. The CPU fake emits the real triple.
  Four tests are added: the triple test fails on the registered scorer, and two tests refuse a misaligned entry 0.
  `staged.sha256` is re-pinned.
- No re-run: the prefill-shaped SGLang rows bear on no registered prediction, and SGLang's served-shape rows are VALID and
  CLOSE on both texts.

### P104 registered (#928): P103 re-asked with a chunk-matched premise -- the Gated DeltaNet kernels (flash-linear-attention, causal-conv1d) under the hybrid paged path, on one 5090 and one host (bench and tests)

- `bench/p104/{PREREG-p104.md,p104_run.sh,p104_drive.sh,p104_reduce.py,staged.sha256}`, `tests/test_p104_staged_pin.py`.
- **New test** `tests/test_linear_state_chunk_matched_gpu.py`. P103 showed that fla's chunk kernel is split-variant, so
  comparing a chunked paged path with a single-call forward charges the kernel to e4b. This test compares like with
  like:
  - an all-linear pool against transformers' cache prefilled in the same 32-token chunks, **bit for bit** (4 seeds,
    any CUDA card);
  - the fp8 hybrid against chunk-matched references, within 2× its all-attention control (sm_89+).
  On the A2000 it passes on the torch path, fla, and fla + causal-conv1d. A mutant that stops marking state between
  prefill chunks fails in every arm.
- **The lane:** P103's, with only the premise changed:
  - phase t: the three pinned files plus the chunk-matched file, 9 passed;
  - phases f and fc: the two graph files plus the chunk-matched file, 8 passed;
  - P103's single-call check is reported, not gating.
- **Predictions:** SUPPORTED with `recommend`; fc's graph arm 1.15-1.60x phase t's on W16 and 1.02-1.15x on W1.
- `bench/p103/a2000/probe6.py` and `probe6_run.sh`: the indicative per-call timing probe cited on #928.

### P102 read (RTX 5090, #916): DEFAULT=k19 -- TTFT-4096 7.21 s -> 1.37 s (5.25x) at max_seqs 1, with the prefill-shaped NLL within +0.011 / +0.007 ppl of the loop on 12 fresh windows (bench docs and receipts only)

- **Verdict** (`p102_reduce.py`, the registered redraw `p102-5090-6`, Ryzen 9 7900 host, $0.2085): `DEFAULT=k19`.
  - TTFT-512 / TTFT-4096: loop 0.851 / 7.207 s; batched 0.672 / 5.820 s; k19 0.113 / 1.373 s; mtile 0.147 / 1.652 s.
  - Quality: batched bit-identical (every window's NLL, every first token); k19 mean dppl +0.0115 (c4val1, W=8) /
    +0.0070 (wikitext, W=4); mtile +0.0080 / +0.0190. All PASS.
- **The VOID first draw** `p102-5090-5` (my stale engagement threshold; Xeon E5-2698 v4): loop 30.1 s, k19 3.4 s at
  4096 tokens. Its NLL deltas equal the reading's to every digit. The loop's TTFT-4096 runs 4.8-30.1 s across the four
  hosts measured; k19's lead over loop holds on both of P102's draws (5.25x and 8.7x).
- **Next lead:** under k19, a 4096-token prefill spends ~45 % of its device time in fp32 SIMT SGEMMs with masking and
  softmax kernels around them -- prefill attention without tensor cores.
- `bench/p102/RESULTS-p102.md`; receipts under `bench/p102/receipts/`. Lane total $0.7328 over six attempts.

### P103 stopped at its proving rental (#928), no verdict: flash-linear-attention and causal-conv1d install and engage on a 5090, and decode graphs replay exactly on them, but the hybrid premise fails; the cause is fla's chunk kernel, not e4b

- `bench/p103/RESULTS-p103.md`, `bench/p103/receipts/p103-prove-2/`, `bench/p103/a2000/`.
- **The proving rental** (`p103-prove-2`, one RTX 5090): premise t passed 4. Premise f and premise fc each read
  1 failed, 3 passed: `test_linear_state_gpu.py`'s hybrid-vs-transformers check, hybrid 7.6e-2 / 8.2e-2 against a
  4.05e-2 bound (torch path 2.6e-2), tokens equal. The graph tests passed in every phase.
- **Isolated at $0 on the A2000, over 20 seeds:**
  - fla's chunk rule is not invariant to where a prompt is split: 3.4e-3 to 5.1e-3 relative, against 6e-7 to 1.7e-3
    on the torch path;
  - transformers' own chunked prefill under fla drifts exactly as e4b's pooled chunked prefill does: identical median,
    mean and maximum;
  - on the torch path both are exact;
  - the decode-step recurrent rule is exactly batch-invariant in both implementations;
  - the layout of the state handed over does not matter.
  Not an e4b defect.
- **The reading is not run:** it could only re-measure phase t (P101). `docs/SERVING.md` records the finding, and the
  torch path remains the read configuration.
- **Unexplained:** with causal-conv1d engaged, one seed in 20 read 1.66 relative error in transformers' own chunked
  prefill (e4b's pooled path: 8.4e-3 at most). Not filed upstream.
- $0.1330: two proving rentals, one stuck `loading`.

### P102 amendment A2 (#916): the loop's engagement check is structural -- the registered 9,600-decode floor voided the first full draw (bench and tests only)

- `p102-5090-5` (adertha-receipts `796486c`, $0.3466) ran every arm. The reducer voided it because `loop` had 8,390
  reference decodes per 512-token request, under a 9,600 floor taken from a uniform-routing guess. P100 had already
  measured 8,390. The loop ran exactly as registered.
- The check is now: 48 loop calls, two decodes per distinct expert, and at least 3,072 decodes (>= 32 distinct experts
  per layer). The fixtures take the measured census; three self-test cases are added (14). Every quality and speed
  criterion is unchanged. The redraw is `p102-5090-6`.


### Lane SC1 amendment A11 (#846): the reducer reads as the registration reads -- llama.cpp's server log, box A's licence and quality rows carried across boxes, P13 on measured ratios (bench + tests)

- **llama.cpp.** `sc1_reduce.py` reads llama.cpp's engagement lines from the server log each receipt names. Box B's engaged rows
  had all VOIDed.
- **Licence carried.** The cross-box read carries box A's QUALITY_FAIL onto boxes B and C, under the registered no-side-door
  rule.
- **Quality carried.** It carries e4b's quality rows (box A) and the bf16 oracle (box B) to the boxes that lack them, on
  identical windows and labelled.
- **P13.** P13 reads the measured vLLM/e4b anchor ratios (both arms VALID and stable) instead of quoted positions.
- **Tests.** Four self-test cases, each failing when its change is reverted. Pin regenerated; `SC1-PREREG.md` A11.
### P103 registered (#928): the Gated DeltaNet kernels under the hybrid paged path -- flash-linear-attention and causal-conv1d against transformers' torch path, on one 5090 and one host (bench and tests only)

- `bench/p103/PREREG-p103.md`, `bench/p103/{p103_run.sh,p103_drive.sh,p103_reduce.py,staged.sha256}`,
  `tests/test_p103_staged_pin.py`.
- P98's measurement at its registered bytes, in three phases on one box:
  - **t**, transformers' torch path (P101's reading again);
  - **f**, `flash-linear-attention==0.5.2`;
  - **fc**, plus `causal-conv1d==1.7.0`.
  Each phase records an engagement probe (what transformers resolves at import, torch held) and runs the hybrid GPU
  premise with the kernels engaged.
- The rule:
  - per phase, P98's rule unchanged;
  - the lane VOID unless phase t holds; otherwise fc's result, else f's, else UNAVAILABLE;
  - `recommend` iff fc is SUPPORTED at 1.05x or better on a workload.
- Predictions: SUPPORTED with `recommend`; fc's graph arm 1.10-1.35x phase t's on W1 and 1.03-1.20x on W16.
- A proving rental (premise t plus both kernel installs and premises on sm_120, no model), then the reading (1.5 h
  guard, <= $1.125).

### TC1 P14: at steady state axolotl's scattermoe steps 9 % faster than e4b as shipped; e4b still finishes a 200-step run first

- **What was asked.** TC1 amendment 8 re-asked P13's axolotl half over 200 steps of the field recipe, read on steps 101..200, on
  one RTX 5090 (`tc1-5090-35`): e4b as shipped and axolotl 0.20.0's scattermoe native-best, two draws each.
- **What it read.** P14 is HELD: axolotl / e4b shipped is **0.911 [0.902, 0.921]**, inside [0.90, 1.10], both pairs stable. The
  whole interval lies below 1.0, so **axolotl is faster at steady state**. Under the amendment's decision rule that is the finding
  (register `e4b.train.h2h.axolotl.qwen3.5090.2026-10-02.native-steady-state`, a labelled row).
- **Beside it.**
  - Over the whole run e4b finishes first. axolotl's warm-up (359 / 324 s at step 1, then spikes on the same 18 steps in both
    draws) puts its summed step time at 1,934-2,023 s against e4b's 1,353-1,372 s.
  - axolotl spends ×1.32 the energy per step.
  - At step 200 axolotl's native configuration matches the box's matched e4b held-out (0.7696 against 0.7691), while e4b as
    shipped sits 0.024-0.027 above, its TC1b plateau reproduced.
- One host (Xeon E5-2698 v4). See `bench/h2h-2026-10-02/tc1/RESULTS-tc1-nativebest.md`.

### P101 read (RTX 5090): SUPPORTED -- on the fixed code, Qwen3.6's bucketed decode graphs all capture and replay through the serving stack bit for bit; 449.0 / 79.7 tok/s (W16 / W1), 2.02x / 3.28x plain eager (#919, #913)

- `bench/p101/receipts/p101-5090-5/`, `bench/p101/RESULTS-p101.md`.
  - P98's kit at its registered bytes and P98's rule.
  - e4b at `4287d07`, which includes #918's fix of #913.
- Every bucket (1, 2, 4, 8, 16) captured and replayed with no eager step (209 replays; bucket 1, the faulting bucket,
  99). Arm g's tokens equal the padded-eager oracle's on all 17 requests.
- The eager arms wrote the same tokens as P98's, on another host and an older commit, but ran about 2.2x P98's speed.
  This box was a Ryzen 9 7950X against P98's EPYC 7663, and eager hybrid decode is launch-bound. The graph-over-eager
  ratio is therefore not claimed for another host.
- Against the predictions:
  - held: SUPPORTED, the W1 band (79.7 in 30-150), W16 above 350, and peak memory 23.76 GB;
  - missed: "graphs at least 3x on both workloads", on W16 (2.02x), and "eager arms within 10 % of P98", on the host.
- The registered consequence:
  - `docs/SERVING.md` cites this lane and lifts its warning against `E4B_PAGED_GRAPHS=1` on hybrid models;
  - `docs/STATUS.md` adds it;
  - register row `e4b.serve.p101.qwen36-hybrid-decode-graphs.5090.2026-10-03`.
- $0.3192 across seven rentals. Five were refused before any lane code ran: four on the pre-flight's 40 MB/s download
  floor (two slow hosts that were the cheapest offers) and one on a Vast API TLS timeout. Then the proof, and the
  reading at $0.1937.

### P99 read (RTX 5090): LOCALISED -- P98's fault repeats at the first bucket-1 replay, also with K25 off; not with one bucket, and not on OLMoE (#913)

- `bench/p99/receipts/p99-5090-1/`, `bench/p99/RESULTS-p99.md`. e4b at `daef38c`, before #918's fix.
  - **d0g** (P98's arm g) and **d2g** (K25 off) faulted on the same kernel assert as P98
    (`indexSelectSmallIndex: srcIndex < srcSelectDimSize`). Both died at decode call 110, W16's first one-row step,
    so the first replay of bucket 1's graph.
  - **d3g** (bucket 16 only) ran: 209 replays, tokens equal its padded-eager arm.
  - **d1g** (OLMoE-1B-7B through the same hybrid expert tier, all-VRAM, device grouping on) ran: bucket 1 replayed 99
    times, tokens equal its padded-eager arm.
- Answers: **qwen36_specific yes** (the reducer's key `hybrid_state_necessary`, renamed by amendment 1),
  **k25_necessary no**, **multiple_buckets_necessary yes**. No silent mismatch.
- Against the predictions:
  - The pre-registration's k25 and OLMoE predictions missed.
  - The predictions posted on #913 from the code reading, before any data, got k25 and multiple-buckets right and
    OLMoE wrong.
  - The fault sits where #918's cause puts it: the T == 1 gather at bucket 1's first replay, after other buckets'
    warm-ups. P99 does not explain why OLMoE's run never reached the stale read's failing state.
- Observation, not this lane's question: Qwen3.6 with bucket 16 only, under graphs, decoded W16 at 421.0 tok/s and W1
  at 46.4, 3.6x and 4.3x its padded-eager arm.
- $0.196 (one RTX 5090, 21 min; destroyed, absent). P101 (#919) asks P98's question again on code that includes #918.

### P102 amendment A1 (#916): the runner installs pytest, which the premise runs under (bench and tests only)

- `p102-5090-1` (adertha-receipts `139544a`, $0.0303) stopped at the premise, rc 25, before any fetch: "No module named
  pytest". The runner was derived from P100's install line, which has none.
- The install line adds `pytest`, and the tripwire imports it. `tests/test_p102_staged_pin.py` adds a check, false on
  the registered runner, that whatever the premise runs is installed and imported first. Pin regenerated;
  `PREREG-p102.md` A1. Nothing measured, and nothing in the rule, changes.

### P102 registered (#916): the int4 store's prefill route A/B -- loop against batched, k19 and mtile on one RTX 5090 and one engine, gated by the calibrated K8 rule on a prefill-shaped NLL (bench and tests only)

- **Why.** P100 (#920) put 73 % of a 512-token prefill chunk in the per-expert host loop: about 0.43 s fixed per chunk
  plus 0.35 ms per token on a fast host, and 2.07 s per chunk on SC1 box B's. #921 added `E4B_INT4_PREFILL` with the
  default unchanged. This lane decides the default.
- **What.** `bench/p102/`:
  - The premise first: `tests/test_int4_prefill_route_gpu.py` on the card.
  - Arm `ttft`: one `serve_paged` engine at `max_seqs` 1 and chunk 512, the route switched between requests, three
    interleaved rounds at 512 and 4096 tokens. It adds P100's dispatch census and a kernel census per route, and a
    device-time profile of one 4096-token request under k19 and under mtile.
  - Arm `nll`: `step_decomp --ppl-oracle eager` (every MoE call T >= 256) on one model, over 12 fresh windows
    (c4val1 9-16, wikitext 9-12) x 4 routes.
  - `p102_reduce.py` (11 cases):
    - batched must be bit-identical (NLL and first tokens);
    - k19 and mtile must hold |mean dppl| <= 0.05 on both texts;
    - the fastest passing route (numerics break 10 % ties) becomes the default iff it at least halves loop's TTFT-4096.
  - Predicted: `DEFAULT=k19`.
- **Tests.** `tests/test_p102_staged_pin.py` (10).

### E4B_INT4_PREFILL (#916): a route knob for the int4 store's host-grouped prefill calls -- loop (the default, unchanged), batched (bit-identical), k19, mtile; the "paid once per request" comment and the int4 Scope note corrected

- **Why.** With `DEVICE_GROUPING` off (the library default, so every `max_seqs == 1` server), a T > 1 call on the
  uniform-int4 store takes host grouping and the int4 store's else-branch. That branch loops in Python over the routed
  experts: one `dequant_int4_ref` (a pure-torch reference decoder), a cast, a matmul and a copy per expert per
  projection. Lane P100 read the cost on an RTX 5090: about 0.43 s fixed per 512-token chunk (73 % of the chunk in the profile) on an AMD Ryzen 9 9950X3D host, and 2.07 s per chunk on SC1 box B's host, where TTFT-4096 read 16.6 s (#920).
- **What.** `hot_residency._int4_prefill_mode_env` reads `E4B_INT4_PREFILL` at every call (an unknown value is
  refused):
  - `loop` is the default and leaves behaviour unchanged.
  - `batched` decodes each projection's routed experts together, in slices of 16 (`_dequant_int4_bf16`, the
    reference's own elementwise operations), and keeps the same matmuls: bit-identical output, no per-expert decode.
  - `k19` and `mtile` give those calls device grouping (`_collapsed_grouping`). `k19` serves them with K19 at every
    row count: bf16 activations and in-register int4 decode, i.e. the loop's operands, within one bf16 ulp. `mtile`
    serves them with the grouped int4-b32 M-tile GEMM: int8 activations.
  - With `DEVICE_GROUPING` on (`max_seqs > 1`), prefill rows above 256 have always taken the M-tile; only `k19` moves
    them.
  - The default changes only by a registered reading (lane P102).
- **Docs.** The branch comment said the loop was "paid once per request". It runs on every call, i.e. per prefill chunk
  per layer. `int4_experts.py`'s Scope note said a T > 1 call with `DEVICE_GROUPING` off takes "the NF4 grouped path";
  with the int4 store installed it never does. Both are corrected, and `test_int4_docstring_matches_dispatch.py` now
  refuses either sentence.
- **Tests.**
  - `tests/test_int4_prefill_route.py` (CPU, 14): the knob and its refusal; `_collapsed_grouping` per route, store
    kind and `T == 1`; the batched decode bitwise against the reference; batched bitwise against the loop through
    `_fused_over_stack`, across slices and on DECODE_A16's singleton rows, with no reference decode; `k19` reaching
    K19 (refused without it) and `mtile` the M-tile at prefill rows; device-grouped rows staying on the M-tile unless
    `k19`.
  - `tests/test_int4_prefill_route_gpu.py` (2): the real kernels at Qwen3-30B-A3B's expert shapes and 4,096 / 16,384
    rows. Batched is bit-identical to loop; k19 is within 0.8 % of loop and mtile within 3 %, both with no reference
    decode. On an A2000 the deviations were 0.20 / 0.43 % (k19) and 1.30 / 1.32 % (mtile).
  - The A2000 run also exercised three mutations, each failing its test: the batched product rounded before the
    scale, the route switch removed, and K19 never selected.

### P100 read (RTX 5090, #916): REFUTED by its rule -- the per-expert prefill loop runs exactly as traced and is 73 % of a 512-token chunk, but costs 0.42 s per chunk on this host, not 2.07 s; SC1 box B's TTFT ran 3.5-3.8x faster on another 5090 host (bench docs and receipts only)

- **Verdict** (`p100_reduce.py`): REFUTED.
  - SCALING INDETERMINATE: rho = T(chunk 512) / T(chunk 2048) = 4.79 / 2.25 = 2.13, under the registered 3.0.
  - MECHANISM NOT_CONFIRMED on its size threshold: 8,390 `dequant_int4_ref` calls per 512-token chunk, under 9,600.
    Routing hits 87 distinct experts per layer, not ~128.
- **Measured** (`p100-5090-2`, AMD Ryzen 9 9950X3D host, $0.163):
  - TTFT 0.550 s at 512 tokens; 4.790 / 3.065 / 2.255 s at 4096 tokens, chunk 512 / 1024 / 2048 (SC1 box B: 2.078 /
    16.597 s for the same configuration and token ids).
  - The census confirms the trace exactly: 48 host-grouped loop calls per chunk per layer (384 for an 8-chunk
    request), two reference decodes per distinct expert, none device-grouped, none at T == 1.
  - 117,247 device kernels and 14,969 copies per 512-token request.
  - In the cProfile, the int4 branch is 0.454 s of the 0.619 s prefill, and its `ncalls` equals the census.
  - Per chunk: about 0.43 s fixed plus about 0.35 ms per token.
- **Next:** the registered consequence's two sentences disagree on this data (the time IS in the loop; the rule failed
  on the hypothesis's quantities). P102 (P101 went to the hybrid-graphs lane) is written against where the time was
  measured to be, with a fresh rule.
  `bench/p100/RESULTS-p100.md`; receipts under `bench/p100/receipts/p100-5090-2/`.


### P101 registered (#564, #913): P98's question asked again on the fixed code -- Qwen3.6's bucketed decode graphs through the serving stack, against the padded eager step, and the first hybrid decode speed (bench docs only)

- `bench/p101/PREREG-p101.md`. P98's kit runs at its registered bytes (`bench/p98/staged.sha256` unchanged), under
  P98's rule. Only the code under test differs: e4b at the launch commit, which includes #918's fix of #913.
- Predictions: SUPPORTED; graphs at least 3x plain eager on both workloads (P98's eager arms were launch-bound at
  about 85 ms per step); the eager arms reproduce P98's readings within 10 %.
- A proving rental, then the reading (1.5 h guard, <= $1.125).

### P100 registered (#916): where the paged prefill's time goes on the int4 expert store -- TTFT against chunk size, and the per-expert loop counted, with no code change (bench and tests only)

- **Why.** SC1 box B read TTFT 2.08 s at 512 tokens and 16.6 s at 4096 for Qwen3-30B-A3B on an RTX 5090 (int4
  experts, chunk 512, `max_seqs=1`): about 2.07 s per chunk. A code trace (#916) names a per-expert host loop in the
  int4 store's host-grouped branch (`hot_residency.py:721-752`, one `dequant_int4_ref` per routed expert per GEMM),
  reached because `max_seqs == 1` leaves `DEVICE_GROUPING` off. Not yet confirmed on a GPU.
- **What.** `bench/p100/`: five arms on one 5090, each a fresh engine. TTFT-512 at chunk 512; TTFT-4096 at chunk
  2048, 512 and 1024 (SC1's `sc1_e4b_sched.py --ttft`, its registered bytes, SC1's stack and token ids); and
  `step_decomp.py --cprofile-out` at `--batch 1` with int4 experts. `p100_box.py` adds an untimed census request per
  arm: every MoE call's rows, grouping and `dequant_int4_ref` calls, plus a device-kernel count in the 512-token arm.
  `p100_reduce.py` (13 cases) reads SCALING (rho = T(chunk 512) / T(chunk 2048) >= 3) and MECHANISM (48 loop calls
  per chunk, >= 9,600 dequant calls, none at T == 1).
- **A2000 pre-check ($0).** The census recorded the real loop branch exactly at Qwen3's expert shapes. The branch took
  ~104-110 ms per layer whether a call carried 4,096 or 16,384 rows, with ~3.6k device kernels per call.
- **Tests.** `tests/test_p100_staged_pin.py` (11): the pin, the borrowed files at their lanes' bytes, SC1's stack and
  token ids, the arms and their order, the tripwire strings against the library, the timed requests unwrapped, the
  exit codes, the dry run.

### Bucketed decode graphs on an NF4 MoE no longer replay against a freed row-to-token index (#913; the cause of P98's replay fault)

- **The bug.** `_HotResidency._forward_collapsed` (the all-VRAM path) and `_forward_diet` cached `rt`, the token each
  expert row reads, in a single-entry cache keyed by row count.
  - `PagedModelRunner.enable_decode_graphs` warms each bucket eagerly and then captures it, in ascending order. Each
    bucket's warm-up therefore freed the previous bucket's `rt`, while that bucket's graph still read the address: a
    CUDA graph keeps no reference to the tensors it reads.
  - Once the caching allocator handed the block out again (on the warm-ups' side stream), an earlier bucket's replay
    gathered token rows through whatever was there.
  - At T == 1, and at every T with K25 off, the gather is `x.index_select(0, rt)`. P98 hit it as an out-of-range
    assert at bucket 1. **With in-range garbage it is silent:** wrong token rows, no error.
- **The fix.** `_HotResidency._row_index(T, k, dev)` builds `(row_token, row_slot)` once per (T, k, device) and keeps
  them for the module's life, a few KB per distinct row count.
- **Test** (`tests/test_rt_cache_graph_gpu.py`, any CUDA card). K25 off; buckets 2 and 4 captured as
  `enable_decode_graphs` captures them; the warm-ups' side stream handed zero-filled blocks; bucket 2 replays new
  inputs and must equal eager bit for bit. On the NAS A2000:
  - **with the fix reverted it fails**: max abs 1.77e-2, silent;
  - with the fix it passes, and so do 27 neighbouring expert-engine GPU tests.

  A first version churned the default stream and passed either way; it was corrected before use.
- **Scope.** Any bucketed-graph serving of an NF4 MoE through the all-resident collapse ran this cache before now,
  not only hybrid models. Six earlier lanes ran bucketed decode graphs on an MoE:
  - **P80, P81, B771b and P82** (Qwen3-30B-A3B; `step_decomp` graph buckets 1–16) ran the cache at commits where every
    bucket read `rt`. The lean glue arrived later, in v0.38.0. Each lane's registered graph-vs-padded-eager token gate
    held bitwise in all 16 rows, so the fault did not show in their outputs, and their claims stand.
  - **SC1** recorded speed only, from `serve_paged`.
  - **P98** hit the fault.

  `docs/SERVING.md`'s hybrid warning stays until a lane re-reads P98's question on the card.

### P99 amendment 1 (#913): the OLMoE arm's answer reads "Qwen3.6-specific"; two mechanisms ruled out at $0 on the A2000 (bench docs only)

- `bench/p99/PREREG-p99.md`, before the launch. OLMoE differs from Qwen3.6 in more than the hybrid state (64 vs 256
  experts, no shared expert, a different attention geometry, a full-size pool). So a d1g that does not fault reads as
  "something Qwen3.6-specific is necessary", not "the hybrid state is".
- The fused tile-table builder (`build_group_tiles_fused`, which feeds the small-index `index_select` that matches
  P98's assert) matches the chained builder in every output. That held on 64, 128 and 256 experts with a poisoned
  allocator: 0 of 300 calls differ.
- K25 with device grouping and the lean glue, captured at 2 to 16 rows on Qwen3.6's exact expert shapes (and on 40
  experts) and replayed on new routings, equals eager: 0 of 120 replays differ. The k25_necessary prediction is
  revised, before any data, to no.

### Lane SC1 amendment A10 (#846): box B's full run exposed four instrument defects and an unsupported comparator; box C's stall gets bounded stops (bench + tests)

- **llama.cpp.** `llamacpp_box.sh` starts llama-server with `-lv 4`. At the pin, the offload and `flash_attn` lines the
  engagement checks read are library INFO messages, logged only at verbosity 4, so every start was refused.
- **LMDeploy.** LMDeploy 0.18.0 TurboMind W4A16 is UNSUPPORTED on sm_120: its SM80 GEMM fallback compiles to empty kernels
  there. Box B records P5b's stated alternative instead of installing it.
- **ExLlamaV3.** Both drivers import `exllamav3.version`; the quality scorer died on that read after scoring.
- **Reducer.** `sc1_reduce.py` checks the scheduler census in its own units: the pre-fusion lever count is 192, and the
  `Int4Linear` count after q/k/v fusion is 192 - 2 x `fuse_qkv_n`. The self-test fixture now carries the real shape.
- **Bounded stops.** The SGLang and llama.cpp stops never `wait` without a bound; a process stuck in the driver ignores
  SIGKILL. SGLang transitions write summary lines before they start.
- **Box B's proof** starts llama-server on the lane's Q4_K_M through the registered checks.
- **Incremental pulls.** `sc1_drive.sh` pulls the box's results every 20 min during the run, at low priority and
  bounded. A failed final fetch keeps the last pull, labelled `PARTIAL_PULL_AT`. Box C's draw was lost with its unreachable box.
- **Scheduler decode reading.** `sc1_e4b_sched.py` records per-request `Request.ttft` and reads decode from the slope of
  decode-only time, each rep's wall minus its own prefill. The registered wall slope stays recorded beside it. On box B, e4b's
  2.1 s prefill jitter had made the wall slope UNSTABLE.
- **Tests.** `tests/test_sc1_a10.py` (13), four reducer self-test cases, four scheduler self-test cases, and the amended proof-needs test. Pin regenerated;
  `SC1-PREREG.md` A10.

### P99 registered (#913, #564): localise P98's replay fault -- P98's graph run repeated beside arms without the hybrid state, without K25, and with one bucket (bench and tests only)

- **Why.** P98 (#912, VOID): Qwen3.6's bucketed decode graphs captured, then a replay hit a device-side assert. The
  same padded steps run eagerly were fine. Three candidate causes (#913): the hybrid's per-slot state; K25 under
  bucketed graphs; a buffer moved after capture.
- **What.** `bench/p99/`: `p99_box.py` (P98's `p98_box.py` at its registered bytes, plus a `P99_STEP` line flushed
  before every decode call, so a faulting replay names its step and bucket), `p99_reduce.py` (11-case self-test), the
  runner, the driver and the staged pin.
  - On one RTX 5090, seven fresh engines: d0g, P98's arm g repeated, must fault; d1 OLMoE, with no linear state; d2
    Qwen3.6 with K25 off; d3 Qwen3.6 with bucket 16 only. Each of d1-d3 runs graphs and a padded-eager oracle.
  - LOCALISED with three answers (is each cause necessary?) plus any silent mismatch against an oracle. VOID if the
    fault does not reproduce or an arm errors otherwise.
  - Guard 1 h, so no proving rental. The premise runs first on the box.
- **Tests.** `tests/test_p99_staged_pin.py` pins the diagnosis. P98's files are staged at P98's pinned bytes.

### P98 read (RTX 5090): VOID -- Qwen3.6's bucketed decode graphs all captured, then a replay hit a device-side assert (`index_select` out of range); the padded-eager and plain-eager arms ran (bench and docs)

- `bench/p98/receipts/p98-5090-2/`, `bench/p98/RESULTS-p98.md`.
  - The premise held on the card: 4 passed, including the dense hybrid's bucket graphs replaying exactly.
  - The bake held.
  - Arm g (graphs): `build_engine` reported every bucket captured, then a W16 replay raised `device-side assert
    triggered` (`indexSelectSmallIndex: srcIndex < srcSelectDimSize`). The process aborted and wrote no record, so the
    reducer reads VOID.
  - Arm e ran the same padded steps eagerly, 209 of them across all five buckets, without fault. The fault appears only
    under replay.
- Observations from the eager arms, not this lane's question. W16 128.3 tok/s (padded eager) and 107.3 (plain); W1
  11.98 / 11.46; peak 23.7 / 22.7 GB. A step costs about 85 ms at 1 row and at 16: eager hybrid decode is bound by
  launch overhead, which graphs remove.
- Not shown to be hybrid-specific: bucketed graphs with K25 (the NF4 default above T == 1 since 0.40.0) on a real NF4
  MoE have not run together before. The next lane separates the candidates. `docs/SERVING.md` now warns against
  `E4B_PAGED_GRAPHS=1` on hybrid models until then.
- $0.153 (proving $0.018; one $0 refusal; the reading $0.135). No code, gate or default moves.

### P98 registered (#564): hybrid decode under CUDA graphs through the serving stack -- Qwen3.6-35B-A3B's bucketed graph replays against its padded eager step, and the first hybrid decode speed (bench and tests only)

- **Why.** #907 and #908 capture a hybrid model's per-slot linear state. On a GPU they have been read only on small
  models (the A2000 real-capture tests in #908). P97 read Qwen3.6 eagerly, at 830 ms per 4-row step.
- **What.** `bench/p98/`:
  - `p98_bake.py`: the NF4 arena, with the checkpoint pinned;
  - `p98_box.py`: one arm per process through `serve_paged.build_engine`;
  - `p98_reduce.py` (19-case self-test);
  - the runner, the driver and the staged pin.
  - On one RTX 5090, three fresh engines: **g** bucketed graphs (1–16), **e** the same padded steps eagerly (the
    bitwise oracle), **p** plain eager. Two workloads: 16 staggered requests that walk every bucket, and one request.
  - SUPPORTED if every bucket captures and replays with no eager step and g's tokens equal e's. NOT_SUPPORTED
    otherwise. VOID on a broken oracle, layout or shape. Decode speed, graphs against plain eager, is reported.
  - The premise, on the card before any fetch: the three hybrid GPU test files pass, all four tests, none skipped.
- **Rehearsed on the A2000** (stand-in attention, solver placement):
  - the bake on Qwen3.6 now reads an offloaded layer's NF4 tensors from the offload handle's host copies (it read
    0-element placeholders first);
  - `build_engine` serves the `qwen3_5_moe` checkpoint from the arena;
  - arm p ran end to end.
- **Tests.** `tests/test_p98_box.py` covers the workload, timer and summary helpers. `tests/test_p98_staged_pin.py`
  pins the registration.

### TC1 native-best against native-best: e4b as shipped steps 1.79x faster than Unsloth's native-best; the axolotl half is untested

- **What was asked.** Amendments 5-7 of `bench/tc1/TC1-PREREG.md` set each framework's native-best configuration against the
  others on one RTX 5090, two interleaved draws each. P13 predicted that e4b as shipped is the fastest of the three.
- **What `tc1-5090-34` read.** Every arm is VALID. Unsloth native-best / e4b shipped is **1.794 [1.790, 1.797]** over two stable
  pairs, so P13's Unsloth half is HELD (register `e4b.train.h2h.unsloth.qwen3.5090.2026-10-02.native-vs-native`, a labelled row).
  axolotl's scattermoe draws were 10.8 % apart, so its half is UNTESTED and P13 is UNTESTED. No "fastest" statement is made.
- **Why axolotl's draws disagree.** The arm spends 216, 61-65 and 76-81 s on steps 1, 3 and 6 in both draws, with no Dynamo
  recompile, and runs at 0.94-0.99 of e4b shipped on steps 14-16. Amendment 8 re-asks it over steps 101..200 (P14).
- **The first box.** `tc1-5090-33`'s receipts are bundled with a re-reduction: its anchor read VOID only because the reducer
  had not registered the token, and its e4b-shipped pair was 6.1 % apart. See
  `bench/h2h-2026-10-02/tc1/RESULTS-tc1-nativebest.md`.

### The linear-state pool builds each slot tuple's index once, so a fixed-slot graph capture copies nothing to the device

- **Why.** `bench/p39/step_decomp.py`'s B=1 and batched lanes capture the model forward with fixed slots, and never
  bind a bucket selector. The linear wrapper's slot-list path then ran `torch.tensor(slots, device=...)`, a
  host-to-device copy that a CUDA-graph capture refuses. A hybrid model could not be captured there.
- **What.** `LinearStatePool._index` caches the index tensor per (slots, device), bounded at 512 entries. The eager
  warm-up step builds it, and the capture reuses it.
- **Tests** (`tests/test_linear_state_graph_gpu.py`, any CUDA card; no fp8, so the NAS A2000 runs it). An all-linear
  dense Qwen3.5 is captured for real:
  - a fixed-slot graph replays bit for bit as the eager steps;
  - a bucket-selector graph (#907), captured on scratch slots, replays the slots its selector names, reordered
    between replays, bit for bit. The capture itself writes nothing.

### Decode graphs capture a hybrid model's per-slot linear state (opt-in `E4B_PAGED_GRAPHS=1`; not yet read on a GPU)

- **Why.** `enable_decode_graphs` refused models with Gated DeltaNet layers. The per-slot state was gathered and
  scattered through `ctx.slots`, a Python list that a captured graph would bake at capture time (the scratch slots).
  Hybrid decode was therefore eager only: 830 ms per 4-row step on Qwen3.6 in P97.
- **What.**
  - On a bucketed decode step, the linear-attention wrapper gathers and scatters through the bound bucket's device
    selector (`linear_state._bucket_selector`: the KV's `_g_sel`, rewritten with the step's slot ids before every
    replay), and takes every row as carrying state.
  - `enable_decode_graphs` warms the pool before capture (one eager one-token prefill on a scratch slot, its staged K/V
    and state discarded), then freezes it. `ensure_slots` refuses to move tensors a captured graph holds.
  - The runner's block-claim and host-length loops walk the pool layers attention appends to (`pool_layers`).
  - `view` / `store` take `sel` only on a graph step, so a caller wrapping them with their original signatures keeps
    working.
- **Tests.**
  - CPU (`tests/test_linear_state.py`): the selector path is bit-identical to the slot-list path, every row and every
    pooled state; the runner's warm-up allocates every layer and leaves the scratch slot clean; a frozen pool refuses to
    grow; a selector step refuses an unallocated layer.
  - GPU (`tests/test_hybrid_decode_graphs_gpu.py`, sm_89+): on a dense Qwen3.5 hybrid with a compact pool, under the
    continuous scheduler, replay decodes exactly as the padded eager step. Every bucket captured and replayed, with
    padding rows and a recycled slot. It skips in CI; the lane that reads hybrid decode speed runs it as its premise.

### P97 read (RTX 5090): SUPPORTED -- the paged runner keeps each sequence's Gated DeltaNet state at transformers' own and tracks transformers' forward on Qwen3.6-35B-A3B at 4.43e-3 nats (bench, docs and register)

- `bench/p97/receipts/p97-5090-1/`, `bench/p97/RESULTS-p97.md`. One RTX 5090 through gnf4's fp8 kernel: 4 sequences
  interleaved, 512-token prompts and 255 batched decode steps.
  - **G1:** the pooled state at the layers before the first attention layer sits at 7.67e-3 relative error from
    transformers' cache (bar 5e-2).
  - **G2:** the whole model tracks transformers' forward at mean KL 4.43e-3 nats, argmax agreement 0.9717 (bars 0.05 /
    0.85). The OLMoE control reads 7.15e-3 / 0.9736 through the same harness.
  - **The mutant** (decode write-back rotated by one slot) reads 4.05 nats / 0.186 / state 0.879.
  - **Engagement exact:** 2,550 / 4,080 kernel calls, 8,130 linear-state stores.
  - **Cost:** $0.2489, plus the $0.0221 proving rental.
- One prediction FALSIFIED: the subject reads 0.62x the control's KL, not 1-3x. A cross-model ratio swings with the
  shape (1.78x in the rehearsal), which is why the rule does not gate on it.
- `e4b.serve.p97.qwen36-hybrid-paged-state.5090.2026-10-02` registered. `docs/SERVING.md`'s hybrid paragraph and
  `docs/STATUS.md` follow.
- This is a correctness reading: decode is eager (graphs are refused for hybrids), and the Gated DeltaNet layers ran
  transformers' torch path. No code, gate or default moves.

### The per-slot linear-state pool grows when a second runner on the same model binds more slots

- **Why.** `linear_state.install()` keeps one pool per model and returned an existing one unchanged. A second
  `PagedModelRunner` on the same model with a larger batch would have bound slots past the pool's end
  (`IndexError`, or `index_copy_` out of range).
- **What.** `LinearStatePool.ensure_slots(n)` grows the pool, keeping every existing slot's state, and never shrinks.
  `install()` calls it when it returns an existing pool.
- **Tests** (`tests/test_linear_state.py`): a 2-slot pool grows to 4 for a second install. Slot 1's state is kept bit
  for bit, the new slot 3 starts from zero (its logits equal a fresh run's), and a smaller later install leaves it at 4.

### P97 registered (#564): hybrid paged serving on the card -- the paged runner's per-slot linear state against transformers' own cache, and its whole-model error, on Qwen3.6-35B-A3B (bench and tests only)

- **Why.** #889 and #897 (per-slot Gated DeltaNet state, compact fp8 pool) are tested on CPU only, with a stand-in for
  gnf4's fp8 decode kernel (sm_89+). This lane is their first GPU reading.
- **What.** `bench/p97/`: the pre-registration, the box measurement `p97_box.py`, the reducer `p97_reduce.py` (26-case
  self-test), the runner and driver, and the staged pin.
  - On one RTX 5090, each model is measured on the same in-process weights: transformers' forward with a
    `DynamicCache` against `PagedModelRunner` (fp8 pool, 128-token prefill chunks, 4 rows decoded together),
    teacher-forced over 4 wikitext windows (512 + 256 tokens). Per step: KL(reference || paged) on fp32 log-probs,
    Δnll and argmax agreement. After the last step, each window's pooled linear state is held against transformers'
    own cache, layer by layer.
  - Subject Qwen3.6-35B-A3B (30 linear + 10 attention layers on a 10-layer pool); control OLMoE-1B-7B (16 attention
    layers).
  - **G1:** the linear layers before the first attention layer see the same tokens on both paths, so their state must
    match transformers' within 5e-2 relative error. **G2:** mean KL <= 0.05 nats and agreement >= 0.85, a bound the
    control must also pass. Both are set from an A2000 rehearsal of the real model: the working path read 6.0e-3 /
    3.0e-3 nats / 0.94, a slot-rotation mutant 0.73 / 2.95 nats / 0.34.
  - The subject also runs a mutant pass, with each decode write-back rotated by one slot, which must fail both gates.
  - The premise, on the card before any fetch: `tests/test_linear_state_gpu.py` (new) must pass, not skip. It takes
    tiny hybrid and all-attention models through the real kernel.
- **Tests.** `tests/test_p97_box.py` runs the whole measurement on CPU in CI, with the stand-in kernel. The counts must
  match; the pre-attention state must match transformers' cache (1e-7 on CPU); the mutant must fail the state check
  (1.4). A pool that drops its write-backs reads 1.5 there. `tests/test_p97_staged_pin.py` pins the registration.

### Lane TC2 read, Mixtral re-measured (#835): with the counters trace-safe, Unsloth steps in 3.74 s at 29.1 GB against e4b under offload at 19.8 s and 7.16 GB -- a footprint row, no position (bench, docs and register)

- `bench/h2h-2026-10-02/tc2/receipts/tc1-5090-32/` (TC2 amendment 5): Unsloth's Mixtral arm compiles once (34 s) and steps in 3.735 / 3.748 s (stable),
  29 Dynamo frames and no graph break; e4b under expert offload 19.751 / 21.181 s (7 % apart, unstable) at 7.16 GB. e4b's peak x4.07 lower, Unsloth about
  5.3x faster per step; P5 UNTESTED (e4b's pair unstable), P6 HELD, P7 FALSIFIED (Unsloth's held-out 0.005-0.006 nats lower, COMPARABLE).
  `e4b.train.footprint.unsloth.mixtral.5090.2026-10-02` supersedes the two retired 2026-10-02 Mixtral rows.
- `docs/STATUS.md`, the solution page and the capability list follow. No code, gate, default or licence moves.

### `serve_paged` builds a hybrid checkpoint: the fp8 KV pool holds the attention layers only, and the KV geometry reads a composite config's `text_config`

- **Why.** The fp8 KV pool pre-allocates rows for every layer it is given. A hybrid model's linear-attention layers
  keep no K/V (#889), so a pool with a layer per index would spend most of its memory on layers that never write: 30
  of 40 on Qwen3.6-35B-A3B. Qwen3.5 / Qwen3.6's composite vision-language config also keeps `num_key_value_heads` in
  `text_config`.
- **What.**
  - `paged_runner.kv_layers()` sizes the pool to the attention layers when `config.layer_types` names linear layers.
  - `kv_layer_map()` gives paged attention a model-layer to pool-layer map through `PagedAttentionContext.layer_map`.
    It is identity when the pool has a layer per index, compact when it holds exactly the attention layers, and
    refused otherwise.
  - The runner flushes K/V into the mapped layers. `serve_paged._kv_geometry` falls back to `text_config`.
- **Tests** (`tests/test_linear_state.py`): a compact pool matches a one-layer-per-index pool bit for bit on a hybrid
  model; the map and pool-size rules; the composite-config geometry.
- **Not yet:** a GPU run. Decode graphs stay refused for hybrid models.

### Lane TC1 / TC2 reads, axolotl re-run and Qwen3.6 offload (#835): axolotl/e4b 1.416 on Qwen3-30B-A3B; e4b fits Qwen3.6's matched set on 32 GB under offload; Granite's axolotl regime corrected to 4-bit (bench, reducer, docs and register)

- `bench/h2h-2026-10-02/tc1/receipts/tc1-5090-30/`: with the harness no longer breaking axolotl's fp32 routers (TC1 amendment 4), axolotl trains the
  matched set: **axolotl/e4b 1.416 [1.398, 1.435]** over two stable draws each, e4b faster per step, axolotl 0.93 GB lower at peak and x2.31 the
  energy, held-out COMPARABLE. TC1 P6 FALSIFIED (predicted [1.5, 6]). axolotl's scattermoe native-best trains too, at **0.929 x e4b's matched step**
  (one draw, its own init: a labelled row, the first arm to step faster than e4b's matched path on the 5090). `e4b.train.h2h.axolotl.qwen3.5090.2026-10-02`
  (superseding the retired `.axolotl-unsupported`) and `.scattermoe-native`.
- `bench/tc1/tc1_reduce.py`: the regime label reads axolotl's own record of its quantized expert stacks; the census's stack count misses
  parametrized storage, so axolotl's NF4 experts were labelled bf16. **Corrected:** `e4b.train.h2h.axolotl.granite.5090.2026-10-02` -- axolotl's Granite
  experts were 4-bit (64 stacks), so its 0.903 is a 4-bit-against-4-bit result in axolotl's favour, not a footprint position; the box is re-reduced
  beside its receipts.
- `bench/h2h-2026-10-02/tc2/receipts/tc1-5090-29/` (TC2 amendment 4): **under expert offload e4b trains Qwen3.6-35B-A3B's matched set on 32 GB**
  at 19.35 GB (TC2 P8 HELD; parity PASS), where it OOMed resident. No ratio: e4b's offload draws are 11 % apart, and Unsloth's arms are VOID by the
  step-0 rule because the frameworks quantise different module sets on this family (e4b keeps about 1.14 B linear-attention and shared-expert
  parameters in bf16, Unsloth 4-bits them). `.e4b-offload` registered; the earlier Qwen3.6 row is qualified as a resident-only loss.
- `docs/STATUS.md`, the solution page and the capability list follow. No code outside the bench harness, no gate, default or licence moves.

### Corrections (#835): two published head-to-head results were this harness's artifacts -- axolotl "does not train" Qwen3-MoE, and Unsloth's Mixtral numbers (bench, tests, docs and register)

- **axolotl** (TC1 amendment 4). axolotl 0.20.0's loader keeps every `*.gate` router in fp32 on purpose and its own trainer runs the forward
  under bf16 autocast; this harness drives the forward with no autocast, so on Qwen3-30B-A3B the fp32 router met bf16 activations and every
  axolotl arm died in its first forward. The receipts show the router in float32 on the axolotl arms and in bfloat16 on the HF and e4b arms.
  `e4b.train.h2h.unsloth.qwen3.5090.2026-10-02.axolotl-unsupported` is **retired**; the TC3 24 GB row and the Qwen3.6 row are corrected (the
  Qwen3.6 refusal was the harness's target naming against axolotl's vision-language model class). `bench/tc1/tc1_arm.py` now casts axolotl's
  frozen fp32 routers to bf16 after load -- what autocast computes per call -- and records it; axolotl's scattermoe arm may reach the Hub for
  its kernels and records the kernel commits it fetched.
- **Unsloth on Mixtral** (TC2 amendment 5). The harness's engagement counters were Python dict increments; inside Unsloth's compiled Mixtral
  MoE block each became a Dynamo guard that failed on every call (3,547 graphs, 743 graph breaks, the 1024 recompile limit hit twice). The
  76-89-minute "compile phase", the 6.2 / 7.0 s/step and the 12 % draw spread were the harness's. `e4b.train.h2h.unsloth.mixtral.5090.2026-10-02`
  and `.compile-warmup` are **retired**; P7 is withdrawn to UNTESTED. Every other Unsloth arm of the campaign is clean on the same counters
  (Unsloth leaves Qwen3's MoE block uncompiled), so the Qwen3, Qwen3.6, OLMoE and gpt-oss readings stand. The counters are now trace-safe
  (a registered custom op under tracing, a plain increment otherwise; tested under `torch.compile` with checkpointing), and every Dynamo
  snapshot records frames, graphs, graph breaks and recompile-limit hits (`recompiles_total` read a key torch 2.12 never fills).
- Both re-runs are registered in the amendments and run from this merge. `docs/STATUS.md`, the solution page, the capability list and the
  TC1, TC1c, TC2 and TC3 READMEs carry the corrections in place, dated.

### Lane SC1 amendment A9 (#846): SGLang runs `--dtype float16`, the GPTQ checkpoint's scales dtype (its Marlin MoE asserted on bf16 activations) (bench + tests)

- `bench/sc1/sglang/server.sh`, `one_batch.sh`: `--dtype float16` instead of `bfloat16`. Box C's re-proof `sc1c-prove-14` got past
  A8's memory pin and then hit SGLang's Marlin MoE assertion (`hidden_states.dtype == w1_scale.dtype`; the scales are float16).
  float16 is also what vLLM's `auto` resolves on the same checkpoint, so it is the matched activation dtype. The engagement check
  refuses any other resolved dtype. One CPU test that fails on the registered wrappers; pin regenerated; `SC1-PREREG.md` A9.

### Lane TC2 read, box B re-run (#835): Qwen3.6-35B-A3B at matched work does not fit e4b on 32 GB while Unsloth trains it; Mixtral-8x7B: e4b under offload at x4.5 lower peak, Unsloth faster per step once compiled but unstable across draws, no position (bench, docs and register only)

- `bench/h2h-2026-10-02/tc2/receipts/tc1-5090-27/`: box B's Qwen3.6 half re-run (TC2 amendment 3). The matched set (926,187,520 fp32 adapters over 20,520 slots) OOMs e4b's fused path at
  step 1 at both draws (32.52 GB on 31.36 GiB usable), at micro-batch 1 and in the reference loop; **Unsloth 2026.9.14 with the family's own expert names trains the same set resident at
  30.47 GB and 10.59 s/step** (its registered target list still adapts the attention only: VOID). e4b as shipped (bf16 adapters) fits at 32.48 GB and 5.09 s/step -- a labelled row. HF OOMs,
  axolotl's PEFT target names do not resolve. An e4b loss, said as such; TC2 P4 FALSIFIED. `e4b.train.h2h.unsloth.qwen3_5.5090.2026-10-02` with its `.shipped-labelled` row.
- `bench/h2h-2026-10-02/tc2/receipts/tc1-5090-26/`: the Mixtral pair re-asked (TC2 amendment 2, on a host with 1.06 TB of RAM under amendment 3). e4b under expert offload:
  15.891 / 15.719 s/step at 7.15 GB (STABLE). Unsloth resident: 6.213 / 7.011 s/step at 31.95 / 32.24 GB once compiled -- 12.1 % apart, UNSTABLE, so no position is quoted
  and P5 is UNTESTED by its rule. Unsloth 2026.9.14 spends its first six steps compiling Triton kernels on this family (76-89 minutes; 5,413 / 4,673 s of 20-step
  wall against e4b's 337 / 329 s), at a 95 GB host-RAM high-water -- what stopped box B's 98 GB host. `e4b.train.h2h.unsloth.mixtral.5090.2026-10-02` with its
  `.compile-warmup` row.
- Box B's first instance (`tc1-5090-22`) was stopped by its host with its receipts unfetched (amendment 3); three re-launches were refused at $0 by launcher rules (an 8 h guard over
  the 6 h cap; the pool's cheapest verified RTX 5090 above the lane's $0.54/h) and the boxes ran at the $0.69/h ceiling box B itself had used -- the lane's budget line says $0.54/h, the receipts say $0.69/h.
- `docs/STATUS.md`, `docs/solutions/qlora-fused-moe-experts.md`, `docs/capabilities.json`: the paragraph and the claim list follow. No code, gate, default or licence moves.

### Lane SC1 amendment A8 (#846): SGLang's chunking-off servers pin `--mem-fraction-static 0.75` (its own rule left 0.226, below the 0.514 the GPTQ weights need) (bench + tests)

- `bench/sc1/sglang/server.sh`: matched, kvfp8, ttft_matched and quality pass `--mem-fraction-static 0.75`. With chunked
  prefill off, SGLang 0.5.20 sized its activation reserve from `max_prefill_tokens` (16384 × 1.5 MB on a 32 GB card), and
  box C's proof `sc1c-prove-13` refused at start-up. The engagement check now refuses a drifted fraction or a KV pool below
  the registered capacity. Four CPU tests (three fail on the registered wrapper); pin regenerated; `SC1-PREREG.md` A8.

### `PagedModelRunner` serves hybrid linear-attention models (Qwen3.5 / Qwen3.6 MoE, Qwen3-Next): per-slot Gated DeltaNet state (engine side, CPU-tested)

- **Why.** The paged runner drives the model with `use_cache=False`. A hybrid model's linear-attention layers keep a
  causal-conv window and a recurrent state per sequence, and without a cache each call would start them from zero.
  The runner's "every layer flushes K/V" step was also the only thing refusing these models, by raising.
- **What.**
  - `engines/linear_state.py` keeps those states in a per-slot pool. For each forward, a linear layer receives
    transformers' own `LinearAttentionLayer` built from the bound rows' state, and the updated state is written back.
    The state arithmetic stays transformers'.
  - `PagedModelRunner` flushes K/V for attention layers only (`attn_layers`, from `config.layer_types`, including a VL
    `text_config`). It marks the pool after each prompt chunk and resets a slot's state wherever it resets the slot's
    KV.
  - **Refused:** Mamba-style layers (transformers labels them `linear_attention` too), unknown state-carrying layer
    types, and decode graphs for a model with linear layers (the gather / scatter is not captured yet).
- **Tests** (`tests/test_linear_state.py`, CPU, tiny random models, no download):
  - the pool against transformers' DynamicCache to 1e-5, with greedy tokens equal, across chunked prefill, batched
    decode in changing row orders, and non-contiguous slots;
  - the runner end to end on a hybrid model (3 linear + 1 attention layer) within its all-attention control's fp8 error
    (1.4e-2 against 3.4e-2), with tokens equal;
  - the refusals, and slot recycling.
- **Not yet:** a GPU run, and `serve_paged.build_engine` wiring for a hybrid checkpoint. `docs/SERVING.md` says so.

### Lane SC1 amendment A7 (#846): box C's image lacked Python.h (Triton could not build its driver); box C's SGLang JIT proof now loads the lane's own GPTQ checkpoint (bench + tests)

- `bench/sc1/sc1_run.sh`: `python3-dev` joins box C's apt line (both Granite smokes died on gcc "Python.h: No such file or
  directory"). The SGLang JIT proof's checkpoint moves from Qwen1.5-MoE GPTQ (4320 expert bias tensors SGLang's loader rejects)
  to the lane's `Qwen3-30B-A3B-GPTQ-Int4` (none). A shape test that fails on the registered line; pin regenerated; `SC1-PREREG.md` A7.

### Lane SC1 amendment A6 (#846): the box's e4b tripwire asserted each route knob's DEFAULT (0.40.0 moved one); it now asserts each knob is READ -- the arms pin all four (bench + tests)

- `bench/sc1/sc1_run.sh`: box B's proof refused because `E4B_NF4_GROUPED_SMALLM`'s default moved `0` -> `auto` (#878). Every e4b
  arm pins the four route knobs via `ROUTEENV`, so the tripwire now requires each knob to be read from the environment and logs
  its observed default (`ROUTE_DEFAULT`). A shape test that fails on the registered loop; pin regenerated; `SC1-PREREG.md` A6.

### Lanes TC2 and TC3 read (#835): the other families at matched work on the current cuts, and the memory frontier on 24 GB and 12 GB cards (bench, docs and register only)

- `bench/h2h-2026-10-02/tc3/`: Qwen3-30B-A3B at TC1's matched set on a rented 24 GB RTX 4090 and on the owned 12 GB RTX A2000, every framework with
  its own memory lever. **e4b trains the set on 24 GB under expert offload at 11.88 GB peak and 10.52 s/step** with the trajectory EQUIVALENT-TO-RESIDENT
  against TC1's 5090 reading (median per-step |Δ| 0.0025; TC3 P3 and P4 HELD); resident e4b OOMs at 24.45 GB. **Unsloth 2026.9.14 trains the same set
  resident on the 24 GB card** (24.22 GB peak, 10.03 s/step) -- the lane predicted an OOM; P1's clause (ii) is FALSIFIED and that row is the finding.
  HF OOMs resident and its accelerate offload arm refuses on a meta tensor; axolotl's plain, `layer_offloading` and ZeRO-3 levers are three refusals.
  On the 12 GB card e4b trains the set under offload at micro-batch 1 (10.46 GB peak, 69.9 s/step on an A2000, held-out 0.8483 beside the 5090's
  0.8516 / 0.8487); Unsloth's loader dispatches modules to the CPU there and refuses, axolotl's cu130 wheels need a newer driver than the host has,
  and HF OOMs at load once given the budget to reach the card (amendment 5: 11.04 GiB in use at a 20 MiB allocation after 761 s of loading) -- TC3 P2 HELD.
  A fit table, one draw per row -- no position moves.
- `docs/claims.json`: `e4b.train.frontier.qwen3.4090-24gb.2026-10-02` with its `.unsloth-resident`, `.e4b-resident-oom`, `.hf-axolotl` and
  `.offload-keeps-trajectory` rows; `e4b.train.frontier.qwen3.a2000-12gb.2026-10-02` (the owned card, a hand run, labelled so).
- `bench/h2h-2026-10-02/tc2/`: the other families at matched work on the current cuts, two rented RTX 5090 boxes. Granite-3.1-3B-A800M: HF + PEFT
  (bf16 experts) **HF/e4b 0.971 [0.965, 0.978]** and axolotl 0.903 -- both faster per step than e4b's fused 4-bit experts, which train at half HF's
  VRAM with COMPARABLE / EQUIVALENT loss (a footprint position; TC2 P1 FALSIFIED); Unsloth adapts no expert parameter on either target list.
  OLMoE-1B-7B: **Unsloth/e4b 1.201 [1.199, 1.204]** with Unsloth's grouped path engaged (e4b faster per step; Unsloth 2.21 GB lower peak and
  x0.71 the energy, the same loss) and HF/e4b 1.255 [1.253, 1.258] at twice e4b's VRAM. gpt-oss-20b: no common adapter set (e4b attention-only;
  Unsloth's packed-MXFP4 arm VOID by the step-0 rule, its bnb load a silent bf16 fallback; HF and axolotl refuse the MXFP4 checkpoint).
  e4b's parity control PASSES on every family with a reference. Box B (Qwen3.6-35B-A3B, Mixtral-8x7B) was lost with its instance -- stopped by its host at 85 % memory during the Mixtral Unsloth load, its receipts unfetched -- and nothing from it is registered; TC2 amendments 2 and 3 (#883, #884) register the re-runs (the Qwen3.6 half on its own box; the Mixtral pair with the Unsloth alarm at 7,200 s on a host with at least 192 GB of RAM), read in a follow-up. P4 and P5 UNTESTED here.
- `docs/claims.json`: `e4b.train.h2h.hf.granite.5090.2026-10-02`, `e4b.train.h2h.axolotl.granite.5090.2026-10-02`,
  `e4b.train.h2h.unsloth.granite.5090.2026-10-02.coverage`, `e4b.train.h2h.unsloth.olmoe.5090.2026-10-02`, `e4b.train.h2h.hf.olmoe.5090.2026-10-02`,
  `e4b.train.h2h.unsloth.gptoss.5090.2026-10-02.no-common-set`.
- `bench/tc1/TC3-PREREG.md` amendment 5: the 12 GB HF arm re-run once with the budget the e4b arms had (its 1,800 s alarm measured the host's disk).
- `docs/STATUS.md`, `docs/solutions/qlora-fused-moe-experts.md`, `docs/capabilities.json`: the paragraphs and the claim lists follow. No code, gate,
  default or licence moves. The TC1 read entry below merged after the v0.40.0 tag and is filed here, where it belongs.

### Lane SC1 amendment A5 (#846): box A's first draw exposed three instrument defects -- vLLM's B=1 token budget, the energy reader, the SAMEPROMPT check (bench + tests)

- `bench/sc1/vllm/`: every B=1 vLLM arm died at engine init (illegal memory access) with `max_num_batched_tokens` 8192 >
  `max_num_seqs * max_model_len` 2048, which vLLM itself warns about; the budget is capped at that product and checked before
  `LLM(**kw)` (B=16 unchanged at 8192). `bench/sc1/sc1_reduce.py`: energy now reads the sampler's headerless CSV through the
  recorded `sampler_fields`; SAMEPROMPT compares the arm's effective rows with `prompts_b16_same.json`. Three CPU tests that fail
  on the registered code; pin regenerated; `SC1-PREREG.md` gains A5 with the first draw's registered outcomes.
### Lane SC1 amendment A4 (#846): the scheduler arms applied the int4 levers twice (harness hook + serve_paged) -- they now run with the hook off (bench + tests)

- `bench/sc1/sc1_run.sh`: the P42 hook on `PYTHONPATH` and `serve_paged.build_engine` both apply the int4 levers right after the
  tier, so the second enable refused on Qwen3 (box-A reading `sc1a-5090-1`). Both `sc1_e4b_sched.py` invocations clear
  `PYTHONPATH`; the harness `step_decomp.py` arms keep the hook. `bench/sc1/sc1_e4b_sched.py` refuses if the hook is loaded.
  A shape test and three self-test checks; the pin is regenerated; `SC1-PREREG.md` gains A4.

### Lane TC1 read (#835): the field-recipe position against Unsloth at matched work is 1.437, not 4.490 (bench, docs and register only)

- `bench/h2h-2026-10-02/tc1/`: the curated receipts of the three TC1 / TC1b boxes (2026-10-01/02, rented RTX 5090s) and the read.
  With the work matched (the same 642,514,944 parameters, one per-slot LoRA init and fp32 adapters in both frameworks) and Unsloth
  2026.9.14 on torch 2.12.1+cu130 with its `grouped_mm` backend engaged on every step, Qwen3-30B-A3B at the notebooks' recipe
  reads **Unsloth/e4b 1.437 [1.434, 1.440]** (5.688 vs 8.171 s/step, two draws each), held-out EQUIVALENT at N = 20 and at every
  eval over 200 steps, e4b's parity control PASSING; Unsloth's peak VRAM is 3.57 GB lower and its energy ×0.72 at that configuration.
- `docs/claims.json`: `e4b.train.h2h.unsloth.qwen3.5090.2026-10-02` and thirteen rows beside it (quality, parity, the mb1 secondary,
  the torch-2.8 loop row 6.565 ×, native-best 2.433, shipped-vs-matched 0.671, e4b on cu130 0.864, HF OOM, host variance, the 200-step
  curve, the shipped plateau +0.0258, the tp2/P38 anchor pair 2.428 / 5.147, the scaling points); the 2026-09-19 position (4.490) and
  its `.quality-n20`, `.e4b-internal-parity` and `.secondary-mb1` rows are **superseded** — that Unsloth arm ran on torch 2.8, where
  `torch._grouped_mm` is sm_90-only and Unsloth's loader silently takes its per-expert loop.
- `bench/h2h-2026-10-02/tc1c/`: the same matched set on one rented H100 NVL (lane TC1c): **Unsloth/e4b 0.621 [0.615, 0.628]** — Unsloth faster
  by 1.61 × at lower VRAM and energy with the same loss; `e4b.train.h2h.unsloth.qwen3.h100.2026-10-02` with its `.quality-n20`,
  `.e4b-internal-parity` and `.dispatch-profile` rows (e4b issues the same ~139 k device events per step on both cards, Unsloth 612 k on
  the 5090 and 82 k on the H100). The 5090 position is card-specific; no "e4b faster" position is quoted for the H100 class.
- `docs/STATUS.md`, `docs/solutions/qlora-fused-moe-experts.md`, `docs/capabilities.json`: the position paragraph and the claim list
  follow. No code, gate, default or licence moves. axolotl 0.20.0 at its pins does not train this family (its loader hands over an
  fp32 router against bf16 activations; five attempts on three cards, `.axolotl-unsupported`); no axolotl position is quoted.

## 0.40.0 — 2026-10-02 — the NF4 store's batched decode rows take K25 by default (`E4B_NF4_GROUPED_SMALLM=auto`, lane P96: B=16 about 40 % faster on Granite and OLMoE); `serve_paged`'s batched decode graphs fixed and confirmed on a GPU

**0.40.0.** One default changes:
- **`E4B_NF4_GROUPED_SMALLM` is `auto`.** With the variable unset, the NF4 store's batched decode rows (T > 1) take K25, the grouped small-M tensor-core GEMM with the select-tree decode through TF32, instead of the served NF4 M-tile GEMM.
  - **The licence.** Lane P93 measured the speed: B=16 ×0.594 (Granite) and ×0.598 (OLMoE) on an RTX 5090. Lane P96 licensed the quality under lane P95's windowed K8 gate.
  - **Unchanged:** T == 1 stays on the decode GEMV. `=0` restores the previous route.
  - **Kernel versions:** K25 needs grouped-nf4-gemm 0.34.0 or newer; 0.34.1 adds the select tree, which is bit-identical. With an older kernel package the previous route runs, silently.
- **Fixed: `serve_paged` with batched decode graphs** (`E4B_PAGED_GRAPHS=1`, `E4B_PAGED_MAX_SEQS>1`). In 0.39.0 only bucket 1 captured; every batch above 1 ran eagerly, and `graph_status` showed it (#874).
  - Confirmed on an RTX 5090: buckets 1–16 capture (lane SC1's proof `sc1a-prove-8`, Granite NF4 at B=1 and B=16).
  - B=1 and eager servers were unaffected.
- **Dependencies:** CI still installs grouped-nf4-gemm at the v0.34.1 commit; the `[fast]` floor stays `grouped-nf4-gemm>=0.30.0`.
- **The rest is evidence and bench:** lane P96 (registration, read, receipts, register row), lane TC3's amendments, lane SC1's amendment A3 and its erratum, and serve_paged's GPU-status docs.


### Lane SC1 A3 erratum (#846): the e4b package is NOT identical between box A's and boxes B/C's commits (#878 landed between) -- the arms' arithmetic is, because SC1 pins the route knob (text only)

- `bench/sc1/SC1-PREREG.md`: A3 claimed `experts4bit_qlora/` was byte-identical between `0a2a0c8` and A3's merge. #878 (the NF4
  grouped small-M default `0` -> `auto`) merged first. Every SC1 e4b arm sets `E4B_NF4_GROUPED_SMALLM=0` explicitly, so the
  routes match. The bf16 oracle, the one call without it, reads the transformers forward at T == 1. Recorded as an erratum.

### Lane SC1 amendment A3 (#846): box B's proof found three bugs in SC1's own drivers (llama.cpp version check, two in the ExLlamaV3 tripwire) -- fixed, with tests that fail on the registered drivers (bench + tests)

- `bench/sc1/llamacpp/llamacpp_box.sh`: llama.cpp built, then the check grepped an 8-character commit where `llama-server
  --version` prints git's 7-character abbreviation; the printed abbreviation must now be a prefix of the pin.
- `bench/sc1/exl3/install.sh`: the tripwire read `exllamav3.version.__version__`, which v1.5.3's package does not expose without
  importing `exllamav3.version`, and required the symbol `exl3_gemv_int8`, which v1.5.3 binds as `exl3_gemv_int8_max_k`.
- Pin regenerated; `SC1-PREREG.md` gains A3; `UPSTREAM-NOTES.md` carries the corrected symbol list. Box A's reading runs at its
  proof's commit; B and C at A3's merge.

### `E4B_NF4_GROUPED_SMALLM` defaults to `auto`: the NF4 store's batched decode rows take K25 at the served precision (lane P96)

- **What changes.** With the variable unset, NF4 rows above T == 1 now run through K25 instead of the served NF4 M-tile
  GEMM, when the installed grouped-nf4-gemm carries it. K25 is the grouped small-M tensor-core GEMM with the select-tree
  decode through TF32 MMA at lane K27's plan.
  - T == 1 stays on the decode GEMV.
  - `0` restores the previous route.
  - A kernel package without K25 keeps the previous route silently.
- **The licence.**
  - Speed, lane P93: B=16 ×0.594 (Granite) and ×0.598 (OLMoE) on an RTX 5090.
  - Quality, lane P96: K8 of K25 against the M-tile it replaces, under lane P95's windowed gate, with mean deltas
    inside 0.05 on every text in both families.
  - P94's single-window QUALITY_FAIL stands under its own rule. P95 measured that one window cannot resolve 0.05 on
    these families.
- **Tests.**
  - `tests/test_nf4_grouped_smallm_route.py`: unset and `auto` take K25 above T == 1 and leave T == 1 alone; `0` is
    the previous route; unset without K25 is the previous route, silently.
  - `tests/test_s2_verify_mechanics.py` selects the M-tile explicitly, since its subject is the captured M-tile path.
- **Re-running an older lane.** A lane that read the M-tile at T > 1 without naming the variable (P91–P93's OFF arms
  set it explicitly) now needs `E4B_NF4_GROUPED_SMALLM=0` to read what it read.

### Lane P96 read (#564, one RTX 5090): LICENSED -- under P95's windowed K8 gate, K25 against the served NF4 M-tile reads mean t − m −0.001 / −0.015 (Granite) and −0.016 / −0.004 (OLMoE), so `E4B_NF4_GROUPED_SMALLM` defaults to `auto` (`e4b.serve.p96.nf4-families.k25-windowed-k8.5090.2026-10-02`)

- **What it read.** K8 of K25 (t) and the served M-tile (m) at T == 1 on fresh windows (c4val1 9–16, wikitext 9–12),
  on both families. All 48 arms ran. Engagement was 64 / 32 calls per step for each arm's kernel.
- **The gate.** |mean(t − m)| ≤ 0.05 on every text in both families: Granite −0.0009 / −0.0146, OLMoE −0.0158 /
  −0.0040 (c4val1 / wikitext).
- **Single windows still swing by up to 0.15.** Six of 24 exceed 0.05, and the largest (−0.150) is on a ppl-29.5
  window. A single-window gate would have failed this change, as it failed P94's.
- **Consequence.** The default moves to `auto` (rows above T == 1), in its own PR.
- `bench/p96/RESULTS-p96.md`, `bench/p96/receipts/p96-5090-1/` ($0.5799; lane $0.6425).

### Docs: `serve_paged`'s batched decode graphs confirmed on a GPU after the #874 fix

- `docs/SERVING.md`: lane SC1's proof `sc1a-prove-8` (RTX 5090, e4b `0a2a0c8`) captured all five decode-graph buckets at B=16, with
  device grouping on at B=16 and the defaults at B=1. The GPU-status paragraph now cites the receipt and states the evidence's
  scope (Granite NF4, unfused set). No code change.

### Fix: `serve_paged` with batched decode graphs failed to capture every bucket above 1 (the batched lane's device grouping was never switched on)

- `experts4bit_qlora/serve_paged.py`: the first GPU run of the server (lane SC1 proof `sc1a-prove-7`, RTX 5090, Granite-3.1-3B
  NF4) captured bucket 1 and failed buckets 2 / 4 / 8 / 16 with "operation failed due to a previous error during capture". A
  `[b, 1]` step with `b > 1` routes `T = b` MoE rows, and the library's default at `T > 1` is eager grouping, whose host-size sync
  invalidates a capture. `bench/p39/step_decomp.py`'s batched lane sets `hot_residency.DEVICE_GROUPING` and clears
  `FORCE_SINGLETON_GROUPS` before it captures; `build_engine` did not. `_batched_graph_grouping` now does the same when graphs are
  on and `max_seqs > 1`, refuses batched graphs off the all-resident placement (as the harness asserts), leaves B=1 and eager
  servers at the defaults, and reports both flags in the census (`grouping`). Four CPU tests; the GPU check is the next SC1 proof.
- `docs/SERVING.md`: the GPU-status paragraph records the first GPU run and this fix.

### Lane P96 registered (#564): K25's default asked again with P95's windowed K8 gate -- K25 against the served NF4 M-tile at T == 1 on fresh windows (bench and tests only)

- **Why.**
  - P93 measured the route at B=16 ×0.594 / ×0.598.
  - P94 failed it on one window per text.
  - P95 measured that one window cannot resolve 0.05 on these families, and named the windowed gate.
- **What it reads.** K8 of K25 (t) and the served M-tile (m) at T == 1, on both families, on fresh windows: c4val1
  9–16 and wikitext 9–12, disjoint from P94's and P95's. It also reads engagement censuses for both arms.
- **The rule** (`bench/p96/p96_reduce.py`, 12-case self-test).
  - **LICENSED** if |mean(t − m)| ≤ 0.05 over the windows on every text in both families. `E4B_NF4_GROUPED_SMALLM`
    then defaults to `auto` (rows above T == 1).
  - **QUALITY_FAIL** otherwise.
  - **VOID** below P95's window minimums (5 and 2), or on failed engagement.
- **Rentals.** A proving rental (0.5 h), then the reading (1.5 h guard, ≤ $1.125). Lane ceiling $2.50.
- `tests/test_p96_staged_pin.py` pins:
  - the windows: fresh, at least P95's minimums, and read by the reducer;
  - the reducer's minimums and σ against P95's receipts;
  - the arms, the order and the proving switch.

### Docs: `serve_paged`'s GPU construction has not yet run on a GPU (stated plainly)

- `docs/SERVING.md`: 0.39.0 shipped `experts4bit_qlora.serve_paged` CPU-tested only. Its `build_engine` (the GPU construction)
  has never executed on hardware, and the docs said so only obliquely ("written from the harness rather than measured here").
  A GPU-status paragraph now says so plainly and names the first GPU exercise: lane SC1's proving rental (#846). No code change.

## 0.39.0 — 2026-10-02 — an opt-in OpenAI-compatible server over the continuous-batching engine (`serve_paged`) with a per-request stop set in the scheduler; foreign-format loader correctness (compressed-tensors NVFP4 decoded global_scale² too large, fixed; zero points and GPTQ `gptq_v2` refused); lane P95 reads K8's window spread

**0.39.0.** No default changes.
- **New, opt-in: `python -m experts4bit_qlora.serve_paged`.** An OpenAI-compatible HTTP server over `ContinuousScheduler` + `PagedModelRunner` + `Fp8PagedKV`, for request-level serving benchmarks. Greedy only, as stated in its entry.
  - The scheduler gains an optional per-request stop set, `min_tokens`, `finish_reason` and `abort()`.
  - With no stop set, every existing caller behaves as before.
- **Fixed: compressed-tensors NVFP4 loads (#788).** The per-tensor `weight_global_scale` is a divisor, and every earlier release multiplied by it.
  - Those tensors loaded clean but `global_scale²` too large, about 5e7 on a real Qwen3-30B-A3B tensor.
  - ModelOpt FP4 was right, and no registered claim uses NVFP4.
  - If you loaded a compressed-tensors NVFP4 checkpoint with an earlier release, reload it.
- **Refused instead of mis-decoded (#789):**
  - compressed-tensors asymmetric weights (`weight_zero_point`);
  - GPTQ checkpoints whose config names `checkpoint_format: gptq_v2`.
- **Tests:** `compressed-tensors==0.18.0` joins `[test]` as the foreign-format oracle, and CI fails if it is missing.
- **Evidence:** lane P95 measured K8's per-window spread on the NF4 families. A single-window 0.05 gate cannot resolve them (UNDER_RESOLVED); a windowed gate needs ≥ 5 c4val1 windows and ≥ 2 wikitext windows.
- **Dependencies:** CI still installs grouped-nf4-gemm at the v0.34.1 commit; the `[fast]` floor stays `grouped-nf4-gemm>=0.30.0`.
- **The rest is bench, docs and tooling:**
  - lanes SC1 (serving head-to-head) and TC1/TC3 (training head-to-head): registrations and amendments;
  - the `E4B_ROUTER_EPI_CAST=0` policy text (#782);
  - a conflict-marker guard in CI.

### Lane P95 read (#564, one RTX 5090): UNDER_RESOLVED -- K8's per-window spread across equal-error arithmetics is 0.054 / 0.055 ppl on c4val1 and 0.026 / 0.030 on wikitext, so a single-window 0.05 gate cannot resolve the NF4 families (`e4b.serve.p95.nf4-families.k8-window-spread.5090.2026-10-02`)

- **What it read.** P94's three arithmetics (GEMV, served M-tile, K25 TF32) on 9 c4val1 windows and 5 wikitext
  windows per family. Window 0 is P94's, reproduced bit for bit on a third host.
- **The spread.** σ = max(SD(m − g), SD(t − m)): Granite 0.054 / 0.026, OLMoE 0.055 / 0.030 (c4val1 / wikitext). A
  windowed gate needs W ≥ 5 windows on c4val1 and W ≥ 2 on wikitext.
- **Production's own pair already crosses 0.05.** The GEMV and the M-tile differ by more than 0.05 on 1 of 8
  (Granite) and 2 of 8 (OLMoE) fresh c4val1 windows.
- **K25, descriptively.** Over the fresh windows it shows no consistent shift against the M-tile: c4val1 −0.019 /
  +0.028, SE 0.019. P94's window was an outlier for it, 2.3 / 2.7 SD above the fresh mean.
- **Consequence.** P94's verdict stands, and this lane licenses nothing. A later lane that asks about K25's default
  again uses the windowed gate on fresh windows (k ≥ 9).
- `bench/p95/RESULTS-p95.md`, `bench/p95/receipts/p95-5090-1/` ($0.7049; lane $0.7965 with four proving attempts).


### Lane SC1 amendment A2 (#846): the box script's quiesce could never succeed, and a proof could pass with its smokes skipped (bench + tests)

- `bench/sc1/sc1_run.sh`: `quiesce` counted busy processes with `pgrep -fc … || echo 0`, which reads `0\n0` on procps when
  nothing matches (`pgrep -c` prints 0 and exits 1), so every quiesce waited its full 900 s — 30–45 min of each box's guard. Now
  `pgrep -f … | wc -l`. The PROVE block wrote `PROVED` with both Granite smokes skipped by the deadline guard (`sc1a-prove-3`);
  it is now NOT PROVED (rc 23) when a smoke is skipped or fails, when a comparator its box installs did not install, or when box
  C's SGLang JIT is unset or not admitted. Box B's proof guard 0.75 → 1.0 h. Pin regenerated; three CPU tests in
  `tests/test_sc1_run_shape.py` pin both fixes and fail on the registered script.
### `E4B_ROUTER_EPI_CAST=0` stays, with a stated end condition: until #674's K8 question is measured (#782)

- 0.37.5 said `=0` (fp32 router top-k weights) would be kept "for one release", and it survived 0.37.6–0.38.1.
- It is now the only way to reproduce the licensed build's function. #674 asks whether the 0.37.5 cast default
  explains the licensed K8 gap (6.33015 against 6.36709), and measuring that needs `=0`.
- It is therefore kept until that question is measured, then removed with notice in this file.
- `router_epilogue._cast_default`'s docstring says so. `docs/STATUS.md` no longer calls the 0.37.5 default
  "(Unreleased)". No behavior changes.

### Foreign-format loader refuses what it cannot decode: compressed-tensors zero points and GPTQ `gptq_v2` (#789)

- **compressed-tensors asymmetric weights.** An `X.weight_zero_point` beside a pack-quantized or NVFP4 tuple is now
  refused by name. Before, the planner consumed the symmetric tuple and left the zero point to surface as an unmapped
  key. The decoders are symmetric only, so loading without the zero point would shift every weight.
- **GPTQ `checkpoint_format: gptq_v2`.** It stores zero points without v1's −1, and `formats/gptq.py` applies v1's
  `zeros + 1`, so a v2 checkpoint would load clean one zero-point step off. `plan_moe_checkpoint` now refuses GPTQ
  tensors when the model config names `gptq_v2`, whether transformers kept `quantization_config` as a dict or as an
  object. A config without `quantization_config` cannot say, and is read as v1, as before.
- Decoding either format, rather than refusing it, is left for a change with its own oracle.
- `tests/test_moe_plan.py`: both refusals; the symmetric tuples and v1 GPTQ plan as before.


### compressed-tensors NVFP4 decoded global_scale² too large: the global scale is a divisor (#788)

- **The bug.** `dequantize_nvfp4` multiplied by the per-tensor scale for both formats that reach it. ModelOpt's `weight_scale_2` (`amax / (6 · 448)`) is a multiplier; compressed-tensors' `weight_global_scale` (`448 · 6 / amax`) is a divisor. Every compressed-tensors NVFP4 tensor was decoded `global_scale²` too large and still loaded clean. On `RedHatAI/Qwen3-30B-A3B-NVFP4`, layer 0 expert 0 `down_proj` ships `7264.0`, so that tensor came out about 5.3e7× too large. No registered claim uses NVFP4; ModelOpt FP4 loads were right.
- **The fix.** `dequantize_nvfp4(..., convention=)` is now required: `"compressed-tensors"` divides, `"modelopt"` multiplies. It refuses a non-finite per-tensor scale, and a zero divisor. The executor takes the convention from the companion key the planner matched, and refuses any other key.
- **Why the tests passed.** The NVFP4, ModelOpt, compressed-tensors int and AWQ tests compared the decoder with itself, and the compressed-tensors ones `importorskip`ped a package `[test]` did not install, so CI skipped them. Every scale they used was positive.
  - They now check against independent references: compressed-tensors' own compress/decompress for NVFP4, `q · scale` over compressed-tensors' own `pack_to_int32` (2/4/8 bit, plus `unpack_from_int32`), an autoawq-order packer for AWQ, and hand-built bytes for ModelOpt and for an NVFP4 case that needs no compressed-tensors.
  - Scales are half negative, with 0, -0 and fp32 subnormals. Negative scales are legal, and AutoRound's symmetric export ships them.
  - Each test fails on a mutant: the old multiply, or `abs()` on the scale.
- **CI.** `compressed-tensors==0.18.0` is in `[test]`, and an unguarded import tripwire keeps those tests from skipping.
- **Not in this change** (#789): `gptq_v2` checkpoints and compressed-tensors `weight_zero_point` both change the math, and the loader reads neither.

### Lane SC1 amendment A1 (#846): proving-rental guards 0.75 / 0.75 / 1.0 h (bench text only)

- `bench/sc1/SC1-PREREG.md`: the per-proof budget as first registered (≤ 10 min, ≤ $0.15) did not count instance acquisition
  (6.4 min measured) or the box script's `can_run` admission tail, so `sc1a-prove-2` hit the deadline after the installs with
  nothing measured. Guards move to A 0.75 h / B 0.75 h / C 1.0 h; an `## Amendments` section records it with the receipts; the
  fourth proof's model revision is spelled out in full. No script, pin, gate or claim changes.

### CI: `conflict-marker-guard` refuses merge-conflict markers in tracked text (tooling only; mirror of grouped-nf4-gemm#437)

- New workflow on push and pull_request: a positive control plants a two-sided conflict and asserts both marker lines are
  flagged, then `git grep` refuses any tracked line beginning `<<<<<<< ` or `>>>>>>> ` (the lone `=======` is a legitimate
  setext underline and is not matched; a conflict always carries the other two). `guard-allow` on the line exempts a
  deliberate quotation. Motivated by this repository's two CHANGELOG races in one hour (#848's rebase staged an unresolved
  file; hotfix #852). No package code changes.
### Lane P95 registered (#564): K8's spread across arithmetics of equal per-GEMM error -- P94's three arithmetics at T == 1 on disjoint windows of each text (bench and tests only)

- **Why.** P94 read K8 on one window per text. On c4val1 its pairs differed by 0.013 to 0.168 ppl, and production's own
  GEMV and M-tile read 0.078 apart on Granite. One draw per arithmetic cannot say whether a single-window 0.05 gate is
  inside the instrument's resolution.
- **What it reads.**
  - P94's arms on both families: g, the scalar GEMV; m, the served M-tile; t, K25 TF32.
  - Disjoint windows of each text: window k starts at token k × 4096. Window 0 is P94's, the reproduction control.
    Fresh windows: c4val1 1–8, wikitext 1–4.
  - Window-major order, so a deadline trims both families evenly.
- **The rule** (`bench/p95/p95_reduce.py`, 12-case self-test).
  - σ = max(SD(m − g), SD(t − m)) over the fresh windows.
  - **RESOLVED** if σ ≤ 0.025 on every text in both families: the single-window gate stands.
  - **UNDER_RESOLVED** otherwise, with W = ⌈(σ/0.025)²⌉ windows for a windowed-mean gate that a later lane registers.
  - The lane licenses nothing, and P94's verdict stands.
- **Rentals.** A proving rental (0.5 h), then the reading (2.5 h guard, ≤ $1.875). Lane ceiling $3.00.
- `tests/test_p95_staged_pin.py` pins:
  - the windows, and that the reducer reads the same ones;
  - P94's kernel pin, harness bytes and arms;
  - the window-0 control values against P94's receipts;
  - the order and the proving switch.


### Lane SC1 registered (#846): Qwen3-30B-A3B serving head-to-head on RTX 5090s against vLLM 0.30.0, SGLang 0.5.20, llama.cpp b11327, ExLlamaV3 1.5.3 and LMDeploy 0.18.0 (bench and tests only)

- `bench/sc1/`: the pre-registration (three boxes with minute budgets; matched work on identical token ids; e4b's licensed
  pack built and K8-gated on the box at both arithmetics it serves; quality on every arm in the served and the prefill
  shape, in nats, vs the bf16 checkpoint and vs the ratioed arm; the e4b ratio taken on a scheduler-in-the-loop slope with
  the graph window as the kernel ceiling; TTFT, resources, energy; controls), `UPSTREAM-NOTES.md` (read from source at
  the pinned tags), `DESIGN-REVIEW.md` (three adversarial rounds with dispositions), the comparator drivers and their
  teacher-forced NLL scorers, the orchestration, the reducer with its self-test, the staged pin. Nothing outside `bench/`
  and `tests/` changes; no default, gate, licence or claim moves.

### serve_paged correction: `fuse_qkv` applies the env-gated folds itself -- the server no longer refuses the registered fused stack

- #853 described `--fuse-qkv` and the three fold flags as the harness's exclusive branches and refused
  `E4B_PAGED_FUSE_QKV=1` together with `E4B_FUSE_T1_GLUE` / `E4B_FUSE_T1_GLUE_R2` / `E4B_FUSE_ROUTER_EPI`. That was
  wrong: `qkv_fuse.fuse_qkv` imports and calls the three folds after fusing (one serve assembly point), so the
  registered B=1 fused stack (P54 / P58 / P88: `--fuse-qkv` WITH the fold flags) is exactly that combination. The
  refusal is removed; the fused branch now reports `fuse_t1_glue_n` / `fuse_t1_glue_r2_n` / `fuse_router_epilogue_n`
  from what the folds returned inside `fuse_qkv` (captured by wrapping them on their modules for the call, restored
  after), never a literal 0, and a `fuse_qkv` that returns without calling them refuses. `/health`, the module
  docstring and `docs/SERVING.md` say so.
- Tests: the real `fuse_qkv` on a CPU stand-in attention module with fake folds and the flags set reports the counts
  and no longer raises; the fold functions are restored after the call, including when `fuse_qkv` raises; a
  `fuse_qkv` that skips the folds refuses; the unfused branch reports the folds it called directly.
### `experts4bit_qlora.serve_paged`: an OpenAI-compatible server over the continuous-batching engine (opt-in, v1)

- **Why.** Request-level serving benchmarks (TTFT, ITL and throughput under Poisson arrivals, as `vllm bench serve`
  and `sglang.bench_serving` drive them) need an HTTP endpoint over the engine the serving campaign measures --
  `ContinuousScheduler` + `PagedModelRunner` + `Fp8PagedKV` -- not over HF `generate`. `serve.py` is the shared-GPU
  availability deployment and stays as it is.
- **What.** `python -m experts4bit_qlora.serve_paged` (127.0.0.1:8778): `/health`, `/stats`, `/v1/models`,
  `/v1/completions` (streaming SSE and non-streaming), `/v1/chat/completions` when the tokenizer has a template. One
  engine thread owns the GPU and steps the scheduler; requests arrive through a thread-safe queue with
  `Request.arrival` stamped at HTTP arrival, so the scheduler's TTFT includes queue wait. `build_engine` reproduces
  the harness's construction in its order (arena load, placement, all-VRAM override, hybrid tier, the lane hook's
  int4 levers from the same env names, amortisation off, paged attention, `fuse_qkv` OR the env-gated folds, KV with
  scratch slots, decode graphs, scheduler); a set lever that patches nothing refuses at startup and `/health` reports
  the census. Greedy only, stated: nonzero `temperature`, `logprobs`, `echo`, `n > 1`, stop strings and penalties are
  400s; `ignore_eos`, `min_tokens`, `stop_token_ids` and token-id prompts are honoured. Streaming emits one chunk per
  token through a windowed incremental detokenizer (a code point split across byte-fallback tokens is held, never
  emitted in pieces), `finish_reason` on the last token's chunk, a usage chunk under `stream_options.include_usage`,
  `[DONE]`. `E4B_PAGED_TRACE` appends a per-request JSONL (arrival / admitted / first token / finished clocks).
- **Scheduler.** Uses the stop set, `min_tokens`, `finish_reason` and `abort()` the scheduler gained in #848 (its own entry below); `ignore_eos` maps to no stop set, the engine's original contract.
- **Tests.** `tests/test_serve_paged.py` (CPU, a scripted runner and a byte-fallback stub tokenizer, the real engine
  thread): the vLLM v0.30.0 client payload verbatim, the SSE sequence, EOS vs `ignore_eos`, `min_tokens`, the
  per-sequence window refusal, TTFT in the trace, concurrent requests sharing a decode step, capacity waits without
  eviction, abort mid-generation; `tests/test_scheduler.py` gains the stop-set and abort cases.
- Docs: `docs/SERVING.md` "Continuous-batching server (opt-in, v1)".

### Scheduler: an optional per-request stop set, `min_tokens`, `finish_reason` and `abort()` (additive)

- **Why.** The continuous-batching engine stopped a sequence only at `max_new_tokens`: `PagedModelRunner` stores
  `eos_id` and never consults it, and `ContinuousScheduler._emit` knew no stop set. That is how every registered
  serving measurement ran (fixed output length, `ignore_eos` semantics) and it stays the default. A serving layer
  needs EOS to end a request in the step that produced it; otherwise the slot keeps decoding wasted tokens to
  `max_tokens` and the throughput a benchmark reads is partly waste.
- **What.** `add_request(..., stop_ids=, min_tokens=)`: a token in the set ends the sequence once `min_tokens` are
  out; the stop token is kept in `out` (computed, counted); a stop at the length boundary reports `stop`, as vLLM
  does. `Request.finish_reason` is `length` / `stop` / `abort`. `abort(rid)` drops a queued request before it takes a
  slot or frees an active one's slot now; aborted requests go to `aborted`, never `done`, so a disconnected client
  cannot move the gate's percentiles; `stats()` gains `aborted`. With `stop_ids=None` (the default, and every
  caller in `bench/`) behaviour is unchanged.
- **Tests.** `tests/test_scheduler.py`: the default contract without a stop set; a stop frees the slot in that step;
  `min_tokens` defers the stop; a boundary stop reports `stop`; abort of queued and active requests.

## 0.38.1 — 2026-10-02 — the K25 route for the NF4 store's batched decode rows runs at the served precision (select tree through TF32 MMA; opt-in, lanes P93 and P94), an instrument for the served M-tile kernel at T == 1, and CI on grouped-nf4-gemm 0.34.1

**0.38.1.** No default changes.
- **`E4B_NF4_GROUPED_SMALLM` (K25, opt-in) runs at the served kernel's precision:** the select-tree codebook decode through TF32 MMA at lane K27's plan.
  - It takes the tree when the installed grouped-nf4-gemm carries it (0.34.1 does). With 0.34.0 it uses the paired lookup, which gives bit-identical outputs.
  - Lane P93 measured the route on an RTX 5090 at B=16 ×0.594 (Granite) and ×0.598 (OLMoE).
  - Lane P94 read its K8 against the served M-tile kernel it would replace: QUALITY_FAIL in both families (c4val1 +0.102 / +0.168 ppl). So it stays opt-in.
- **New: `E4B_NF4_T1_DEVICE_GROUPING=1`.** An off-by-default instrument that serves T == 1 NF4 rows through the served M-tile kernel (lane P94's m arm).
- **Dependencies:** CI installs grouped-nf4-gemm at the v0.34.1 commit. The `[fast]` floor stays `grouped-nf4-gemm>=0.30.0`.
- **The rest is bench and evidence:**
  - lanes K26 and K27's runners;
  - lanes P93 and P94: registrations, reads, receipts and register rows;
  - the training campaign's lane TC1: its registration and amendments, bench only.

### Lane P94 read (#564, one RTX 5090): QUALITY_FAIL in both families. K25 against the served NF4 M-tile kernel it would replace moves c4val1 K8 +0.102 (Granite) and +0.168 (OLMoE); `E4B_NF4_GROUPED_SMALLM` stays `0` (`e4b.serve.p94.nf4-families.k25-vs-mtile-k8.5090.2026-10-01`)

- **K8 at T == 1, three arithmetics:** g = the scalar GEMV, m = the served M-tile through #838's instrument, t = K25
  TF32 tree. Gated on t − m:
  - Granite: −0.020 (wikitext) / **+0.102** (c4val1);
  - OLMoE: −0.005 / **+0.168**.
- **The baseline:**
  - On Granite c4val1 the M-tile itself reads −0.078 from the GEMV (`BASELINE_SHIFT`).
  - On OLMoE it reads within 0.017 of the GEMV, so P93's OLMoE swing is not the GEMV-to-tile difference that
    production carries.
- **Reproduction:** g and t reproduce P93's OFF and ON exactly, on a different host.
- **Spread:** on wikitext every pair is within 0.026; on c4val1 the pairs range from 0.013 to 0.168.
- **Next:** a lane that calibrates c4val1's K8 spread across equal-error arithmetics, registered before any noise-aware
  gate.
- `bench/p94/RESULTS-p94.md`, `bench/p94/receipts/p94-5090-2/` (`p94-5090-2` $0.1373, plus `p94-5090-1` NOT_RUN at the
  bandwidth pre-flight, $0.0164).

### `docs/system-manifest.json` follows grouped-nf4-gemm v0.34.1 (byte-identical)

- The `consumer_ci_pin` prose names v0.34.0, the release whose commit this CI installs (846b512); it still named v0.33.0. The kernel repository changed it in its 0.34.1 release, and this CI compares the manifest byte-for-byte against the kernel's latest tag. No floor, range or ownership changes.

### Lane P94 registered (#564): K8 against the arithmetic a T > 1 route replaces -- K25 against the served NF4 M-tile kernel at T == 1 (bench and tests only)

- **Why.** P92 and P93 compared K25 at T == 1 against the scalar fp32 decode GEMV. The `auto` default changes only rows
  above T == 1, where the route it replaces is the served M-tile kernel (TF32). That was a design error in both lanes;
  their verdicts stand under their own rules. P93's OLMoE c4val1 swing (+0.155, after P92's −0.107) is why the right
  baseline matters.
- **What it reads.** K8 at T == 1 under three arithmetics per family (Granite `r12epi`, OLMoE `nf4`):
  - g, the scalar GEMV;
  - m, the served M-tile through `E4B_NF4_T1_DEVICE_GROUPING=1` (#838);
  - t, K25.
  The premise covers row-exactness for t and m, plus K25's contract.
- **The rule.** LICENSED if the uncalibrated K8 gate passes with base m and candidate t in both families. With P93's
  speed, the T > 1 default then moves to `auto`. Otherwise QUALITY_FAIL. m − g is reported, not gated.
- `tests/test_p94_staged_pin.py` pins:
  - the staged bytes and the families' envs;
  - the three arms and the route's plan;
  - the premise before the fetch, and the order;
  - the exit codes;
  - the reducer's 10-case self-test;
  - the driver's dry run.

### `E4B_NF4_T1_DEVICE_GROUPING=1`: an instrument that reads the served NF4 M-tile kernel at T == 1 (off by default; lane P94)

- **Why.** K8 is decode-shaped, so it reads T == 1. The NF4 store's T == 1 rows run the scalar fp32 decode GEMV, while its
  batched rows run the served M-tile kernel (TF32 on the fp32 dequant). Lanes P92 and P93 compared K25 at T == 1
  against the GEMV, not against the M-tile arithmetic a T > 1 route actually replaces.
- **What.** With the knob set and K25 off (`E4B_NF4_GROUPED_SMALLM=0`), `_collapsed_grouping` sends the NF4 store's
  T == 1 rows to the device tile table and the served M-tile kernel. Off (the default, also when unset), nothing
  changes. Other stores are untouched, and anything but `0`/`1` is refused. Like `E4B_INT4_DECODE_A16`, it is a
  quality instrument, not a serving route.
- **Tests.**
  - The stubbed route test: the decision, the served kernel at T == 1, the refusal.
  - `tests/test_nf4_t1_device_grouping_gpu.py`: a token's rows are bit-equal alone and inside B=16 on the served
    kernel, and the T == 1 output matches the fp32 oracle.
  - NAS RTX A2000: GPU 6/6 (with K25's file), CPU routes 59 passed; a mutation that ignores the knob fails both.

### P93 read (#564): Granite LICENSED (B=16 ×0.594, B=1 ×0.854, K8 inside the gate), OLMoE QUALITY_FAIL (c4val1 K8 +0.155 ppl, the other sign from P92's); `E4B_NF4_GROUPED_SMALLM` stays `0` (bench, docs and register)

- **`p93-5090-1`** (RTX 5090, $0.33), P92's design on the served-precision route (#834):
  - Granite `r12epi`: B=16 10.535 → 6.258 ms (×0.594), B=1 ×0.854; K8 −0.026 / +0.024 ppl.
  - OLMoE `nf4`: B=16 14.352 → 8.585 ms (×0.598), B=1 ×0.859; K8 +0.011 / **+0.155** ppl.
  - In-model the TF32 tree GEMM runs at 0.41× / 0.48× of the served NF4 kernel.
- **OLMoE's failure is not the weight rounding.** P92 (bf16) moved its c4val1 K8 −0.107. P93 changed exactly the
  rounding, and it moved +0.155. Granite moved about as much at TF32 as at bf16. The next lane measures the
  instrument's spread under the production path's own arithmetic on that text, before any instrument changes.
- **Register:** `e4b.serve.p93.nf4-families.k25-tree-tf32-b16.5090.2026-10-01`.

### Lane P93 registered (#564): P92's design on the route at the served precision -- does K25-tree in TF32 make the NF4 families' B=16 decode faster without moving their K8? (bench and tests only)

- **What it is.** P92's lane, unchanged except the treatment and the pin.
  - **Treatment:** the K25 route now runs the select tree through TF32 MMA at K27's plan (#834).
  - **Pin:** grouped-nf4-gemm `908a2ca`, which carries the tree.
  - **Scope:** Granite `r12epi` and OLMoE `nf4`, OFF vs ON, on one RTX 5090. Per family: B=16 and B=1 speed, two-text
    K8, and P92's reducer and rule.
- **The registered consequence:** both families LICENSED moves `E4B_NF4_GROUPED_SMALLM` to `auto`.
- **Rehearsed on the A2000** through install, the new tripwire, the reducer's self-test, the premise (4/4, 30/30) and
  fetch/bake. Every arm stopped at the fp8 KV append, as P92's did.
- `tests/test_p93_staged_pin.py` is P92's pin test, plus the runner's tripwire plan equal to the live `_K25_PLAN`.

### The K25 route runs at the served precision: the select tree through TF32 MMA at K27's plan (opt-in; P93 reads it)

- **Why.** P92 read K25 in bf16 QUALITY_FAIL on OLMoE: bf16 weight rounding. grouped-nf4-gemm's K27 (RTX 5090) read
  K25 with the select tree at the served kernel's weight precision at 0.448 / 0.502 of the served NF4 GEMM's time
  (Granite / OLMoE B=16 shapes), with the served kernel's error (ratio 1.000). TF32_PATH by its rule.
- **What.** `_K25_PLAN = {block_n 32, kc 64, warps 4, stages 3, lut "tree", dot_bf16 False}`: fp32 weights through
  TF32 MMA, K27's best TF32 plan. A kernel package without the tree gets the paired lookup at the same precision.
  `E4B_NF4_GROUPED_SMALLM` still defaults to `0`. P92's reading was of the bf16 arithmetic; lane P93 reads this one.
- **Tests.** The route test's stub records `dot_bf16`, and the test pins the served-precision plan. NAS RTX A2000 at
  grouped-nf4-gemm main (`908a2ca`): the plan resolves as above; GPU 11/11 (K25 row-exactness, the oracle, the lean
  bits and graph capture, with K19's and K23's files); CPU routes and pins pass, except a driver dry run that needs
  `/usr/bin/python3`, which the container lacks.

### Lane K27's runner (#564): the K25-tree precision bench's box side (bench and tests only)

- **What.** `bench/k27/` drives grouped-nf4-gemm's K27 (`kernel/PREREG-k27-nf4-tree-precision.md`). It is K26's runner
  with the bench renamed, plus a tripwire that refuses a kernel package without K25's select-tree decode.
- **Rehearsed** on the NAS A2000 (marked REHEARSAL): rc 0 through the self-test (9 cases), the premise (30/30) and the
  bench.
- `tests/test_k27_staged_pin.py` pins the runner's bytes and shape, the exit codes and the driver's dry run.

### The K25 route takes the select-tree codebook decode when the kernel package carries it (bit-identical; lane K26)

- **Why.** grouped-nf4-gemm's lane K26 (RTX 5090) read the per-nibble codebook lookup as about 80 % of K25's time. An exact
  select-tree decode over the 16 fp32 codebook values gives bit-identical outputs at 0.373 / 0.383 of the time (Granite /
  OLMoE B=16 shapes). That is about 0.39× the served NF4 GEMM, where the paired lookup ran at about 1.0×.
- **What.** `_K25_PLAN`'s decode is `"tree"`. `_k25_plan` resolves it against the installed `nf4_smallm`: `"tree"` when
  its `_LUT_MODES` names it, else `"pair"`. The two give the same bits, so lane P92's reading applies to both:
  Granite's K8 inside the gate, OLMoE QUALITY_FAIL. `E4B_NF4_GROUPED_SMALLM` still defaults to `0`.
- **P92's pins.** Its runner and its pin test name the paired decode it registered, so that closed lane's runner refuses
  today's e4b by design. Its `staged.sha256` re-pins the edited GPU test, as P89's did in #822.
- **Tests.**
  - The route test runs against a kernel stub with and without the tree.
  - The GPU file compares against the resolved plan.
  - NAS RTX A2000:
    - with grouped-nf4-gemm carrying the tree: the plan resolves to `tree`, GPU 11/11 (with K19's and K23's files);
    - with v0.34.0: it resolves to `pair`, and the GPU file passes 4/4;
    - the CPU routes and pins pass, except two driver dry runs that need `/usr/bin/python3`, which the container lacks
      and CI has.

## 0.38.0 — 2026-10-01 — batched decode on grouped small-M tensor-core GEMMs (grouped-nf4-gemm 0.34.0): K19 is the default for int4 decode rows above T == 1 (lane P88, B=16 ×0.905), its lean glue is the default (P89, ×0.950), and K21 is the default for the MXFP4 store's batched rows (P90, gpt-oss-20b B=16 ×0.581, KL lower); K25 serves the NF4 store's batched rows opt-in (`E4B_NF4_GROUPED_SMALLM`; P92); lanes P82–P92 read

**0.38.0.** Three defaults change, each licensed by a pre-registered lane on an RTX 5090 and each reversible by its environment variable:
- `E4B_INT4_GROUPED_SMALLM` is `auto` (K19), lane P88;
- `E4B_INT4_LEAN_GLUE` is `auto`, lane P89;
- `E4B_MXFP4_GROUPED_SMALLM` is `auto` (K21, rows above T == 1), lane P90.

Each takes effect when the installed grouped-nf4-gemm carries its kernel (0.34.0 does); with an older kernel package the previous routes run, silently. The `[fast]` floor stays `grouped-nf4-gemm>=0.30.0`, and a fresh install resolves to 0.34.0. One route is new and opt-in: `E4B_NF4_GROUPED_SMALLM` (K25) for the NF4 store's batched rows. Lane P92 measured it on Granite (B=16 ×0.937) and read OLMoE QUALITY_FAIL, so its default stays `0`. CI installs grouped-nf4-gemm from the v0.34.0 release commit. Both repositories' `docs/system-manifest.json` still say the CI pin is v0.33.0. That `consumer_ci_pin` text moves with the next grouped-nf4-gemm release, kernel first, because the manifest is byte-identical across the two.

### Lane K26's runner (#564): the NF4 decode ablation's box side (bench and tests only)

- **What.** `bench/k26/` drives grouped-nf4-gemm's K26 (`kernel/PREREG-k26-nf4-decode-ablation.md`): a kernel microbench
  on one RTX 5090 with no model.
  - **Install:** grouped-nf4-gemm at the manifest's `GNF4_SHA`, plus pytest.
  - **Premise:** K25's contract compiled on the card (rc 23).
  - **Run:** the bench from the gnf4 clone at that pin.
- **Rehearsed** on the NAS A2000 (marked REHEARSAL, correctness only): install, tripwire, the rule's self-test
  (16 cases), the premise (28/28) and the bench all ran to rc 0.
- `tests/test_k26_staged_pin.py` pins:
  - the runner's bytes;
  - the refusals before the install, and the premise before the bench;
  - that no model is fetched;
  - the exit codes;
  - the driver's dry run.

### P92 read (#564): Granite LICENSED (B=16 ×0.937, K8 inside the gate), OLMoE QUALITY_FAIL (c4val1 K8 −0.107 ppl, step ×1.024); `E4B_NF4_GROUPED_SMALLM` stays `0` (bench, docs and register)

- **`p92-5090-2`** (RTX 5090, $0.33; `p92-5090-1` was NOT_RUN, ssh refused, $0.03), OFF = `0`, ON = `1`:
  - Granite `r12epi`: B=16 10.529 → 9.865 ms (×0.937), B=1 ×0.972; K8 wikitext −0.023, c4val1 +0.016 ppl.
  - OLMoE `nf4`: B=16 14.355 → 14.694 ms (×1.024), B=1 ×1.066; K8 wikitext +0.017, c4val1 **−0.107** ppl. The
    uncalibrated K8 rule is two-sided (|Δ| ≤ 0.05).
  - The premise held on the card (K25 row-exact; K25's contract compiled on sm_120, 28/28), and engagement held in
    every census.
- **Where the time goes.** K25's own GEMM is within 4 % of the served NF4 GEMM (Granite 6.464 vs 6.602 ms, OLMoE 10.852
  vs 10.388). Granite's gain is the glue the lean route folds away. The NF4 codebook decode, shared by both kernels, is
  the likely bottleneck (inferred, not profiled).
- **Consequence (registered):** the default moves only when both families read LICENSED, so it stays `0`. `auto` is a
  measured opt-in for Granite `r12epi`.
- **Register:** `e4b.serve.p92.nf4-families.k25-b16.5090.2026-10-01`.

### Lane P92 registered (#564): does K25 make the NF4 families' B=16 decode faster on an RTX 5090 without moving their K8? (bench and tests only)

- **What it is.** The end-to-end read of K25 (grouped-nf4-gemm #429, pinned at its merge `8cc3510`), which P91's
  decision named, through `E4B_NF4_GROUPED_SMALLM`. Granite-3.1-3B-A800M (`r12epi`) and OLMoE-1B-7B (`nf4`) each run
  OFF (`0`) and ON (`1`) on one RTX 5090:
  - the premise first, on the card: the K25 row-exact test (rc 25) and K25's contract compiled (rc 23);
  - B=16 OFF/ON/OFF/ON (first draw censused) and B=1 OFF/ON (censused), P91's harness;
  - K8 OFF/ON on wikitext and c4val1 (P44-a's arm).
- **The rule, per family.**
  - VOID on missing arms, B=16 draws more than 3 % apart, failed engagement, or K8 ON bit-equal to OFF;
  - QUALITY_FAIL if the uncalibrated K8 gate fails;
  - LICENSED if B=16 ON/OFF ≤ 0.95;
  - NOT_FASTER otherwise.

  Both families LICENSED moves the default to `auto`.
- **Rehearsed on the A2000** through install, the premise (4/4 and 28/28), and both families' fetch and bake. Every arm
  stopped at the fp8 KV append, which sm_86 cannot compile.
- `tests/test_p92_staged_pin.py` pins:
  - the staged bytes, P91's harness bytes;
  - the families' envs;
  - the gnf4 pin as a real SHA;
  - the arms and the tripwire's `_K25_PLAN`;
  - the premise before the fetch and the arm order;
  - the exit codes;
  - the reducer's 14-case self-test;
  - the driver's dry run.

### `E4B_NF4_GROUPED_SMALLM` (K25, opt-in): the NF4 store's batched decode rows through the grouped small-M tensor-core GEMM

- **Why.** Lane P91 read the NF4 families' B=16 decode steps on an RTX 5090. The served NF4 M-tile GEMM
  (`gemm_4bit_grouped_captured`) is 61.7 % of kernel time on Granite (`r12epi`) and 71.9 % on OLMoE (`nf4`). Its
  registered decision named an NF4 grouped small-M kernel. grouped-nf4-gemm #429 is that kernel, K25: K19's kernel
  with the NF4 codebook dequant.
- **What.** Under `auto` (rows above T == 1) or `1`, `_fused_over_stack` serves the NF4 store's device-grouped decode
  rows (up to 256) with `nf4_smallm.gemm_nf4_grouped_smallm`:
  - over the 16-row device tile table, with gate_up's gather folded into the kernel, so the `[R, H]` `index_select`
    is gone;
  - under K23's lean glue (the default), gate_up reads the step's token rows (`gather_div`) and down stores into the
    caller's row order (`scatter`), as on K19's rows;
  - at K25's kernel default (`_K25_PLAN`: BLOCK_N 32, KC 256, 4 warps, 2 stages, the paired codebook decode). No lane
    has swept it; every K25 plan compared on the A2000 was bit-identical.

  gpt-oss's NF4 fallback rows keep their biases by the sorted ids, and a calibration sink keeps today's route.
- **Not the served arithmetic.** K25's weight operand is the bf16 dequant (`dequant_ref(...).to(bf16)`). The served
  GEMM multiplies TF32 on the fp32 dequant, which rounds the weight less. So the default is **`0`, today's routes**,
  until a lane reads speed and quality per family.
- **The quality instrument reads the kernel.** Under `1`, `_collapsed_grouping` also sends T == 1 on the NF4 store to
  the device tile table and K25, so the decode-shaped K8 instrument reads the kernel it gates. It never moves an int4
  or MXFP4 store's T == 1.
- **Refusals.** `1` on a kernel package without K25 is a RuntimeError, and `auto` there keeps today's route silently.
  Anything else is refused.
- **Tests:**
  - `tests/test_nf4_grouped_smallm_route.py` (stubs with an NF4 dequant oracle): the default, the opt-in and `auto`
    routes, the plan, the gpt-oss epilogue, the T == 1 decisions, the refusals, prefill and the calibration sink
    untouched, and lean token rows with the expanded call's bits.
  - `tests/test_k25_row_exact_gpu.py` (real kernel, Granite's expert shapes: 40 experts, top-8, H 1536, I 512):
    - a token's rows are bit-equal alone (T = 1) and inside B=16;
    - the T = 1 output matches the fp32 dequant oracle;
    - lean token rows are bit-equal to the expanded rows;
    - the T = 1 route captures and replays bit-equal.
  - NAS RTX A2000 with grouped-nf4-gemm at #429's head, correctness only:
    - GPU 14/14 with K19's, K21's and K23's files;
    - CPU routes 64 passed;
    - two mutations each fail both the GPU file and the stubbed route tests: T == 1 never moving under `1`, and the
      lean down call without its scatter.

### P91 read: READ. The NF4 grouped expert GEMM is 62 % (Granite) and 72 % (OLMoE) of B=16 decode kernel time; the next family lane is an NF4 grouped small-M kernel (bench, docs and register)

- **`p91-5090-1`** (RTX 5090, $0.08), on the licensed configs:
  - Granite `r12epi`: B=16 9.19 ms/step, with `_gemm_nf4_grouped` at 5.48 ms (61.7 %); B=1 3.34 ms, with the NF4 GEMV at
    50.7 %;
  - OLMoE `nf4`: B=16 12.47 ms, with the NF4 GEMM at 8.61 ms (71.9 %); B=1 3.96 ms, at 45.7 %.
  - Every census reconciles within 10 %.
- **Register:** `e4b.serve.p91.nf4-families.kernel-census.5090.2026-10-01` (descriptive).

### Lane P91 registered (#564): where do the NF4 families' decode steps go? (bench and tests only)

- **What it is.** A descriptive kernel census of Granite-3.1-3B-A800M (`r12epi`) and OLMoE-1B-7B (`nf4`), each on its
  licensed NF4 configuration, at B=16 and B=1 on one RTX 5090. P88's harness, P42's replay census.
- **The rule.** READ if every census reconciles with its step within 10 %. If the NF4 grouped GEMM is at least 40 % of
  B=16 kernel time in either family, the next lane is an NF4 grouped small-M kernel; otherwise it is the largest
  non-NF4 kernel family.
- `tests/test_p91_staged_pin.py` pins:
  - the staged bytes;
  - that the families' envs equal `serve_stack.arm_env` at P44's revisions;
  - the arms;
  - the exit codes;
  - the reducer's 8-case self-test;
  - the driver's dry run.

### K21 is the default for the MXFP4 store's batched decode rows (`E4B_MXFP4_GROUPED_SMALLM` now defaults to `auto`), as lane P90 licensed

- **P90** (`e4b.serve.p90.gptoss.mxfp4.k21-b16.5090.2026-10-01`): gpt-oss-20b's B=16 step went from 22.511 to
  13.086 ms on an RTX 5090 (×0.581). The store's KL from the reference fell from 0.00192 to 0.00147 nats on an
  H100 NVL. B=1 read 1.075× slower.
- **The modes:**
  - `auto` (the default, also when unset) sends the store's device-grouped decode rows above T == 1 to K21 when the
    kernel package carries it with its masked K tail, and keeps today's route (the NF4 fallback) when it does not;
    T == 1 stays on the split-K GEMV;
  - `1` requires K21 and its masked tail, and adds T == 1;
  - `0` is today's route;
  - anything else is refused.
- **Tests.** `tests/test_mxfp4_grouped_smallm_route.py`:
  - the default takes K21 at the registered plan for batched rows, with gpt-oss's epilogue, and keeps T == 1 on the
    GEMV;
  - on a kernel side without K21, or without its masked tail, the default is today's route, silently;
  - `0` is today's route.
- **P90's runner** asserts the pre-default boolean switch in its tripwire. It ran at its registered commit and is
  read, so the pinned bytes stay as they ran.

### P90 read: LICENSED. K21 takes gpt-oss-20b's B=16 decode step to 0.581× on an RTX 5090, and the MXFP4 store's KL falls (bench, docs and register)

- **Speed** (`p90-5090-2`, RTX 5090): B=16 went from 22.511 to 13.086 ms/step (710 → 1,223 tok/s). B=1 read 1.075×
  slower, so a default covers T > 1 only.
- **Quality** (`p90-h100-1`, H100 NVL, P44's instrument): KL from the reference went from 0.001922 to 0.001466 nats,
  top-1 from 0.9814 to 0.9840. OFF reproduces P44's licensed 0.0019.
- **Register:** `e4b.serve.p90.gptoss.mxfp4.k21-b16.5090.2026-10-01`, with a STATUS mention beside the store's
  licence.
- `p90-5090-1` read VOID on the KL reference not fitting a 5090 (amendment 1, #820). Its speed is recorded
  descriptively.

### The lean glue is the default on K19's rows (`E4B_INT4_LEAN_GLUE` now defaults to `auto`), as lane P89 licensed

- **P89** (`e4b.serve.p89.qwen3.int4.k23-lean-glue-b16.5090.2026-10-01`, RTX 5090): Qwen3-30B-A3B's B=16 int4 step
  went from 10.321 to 9.805 ms (×0.950). Tokens were identical in all 16 rows, and there were 336 fewer launches per
  step.
- **The modes:**
  - `auto` (the default, also when unset) folds K19's grouping glue when the kernel package carries grouped-nf4-gemm
    K23's options, and keeps the separate launches when it predates them;
  - `1` requires K23 (absent is a refusal);
  - `0` is the old path;
  - anything else is refused.
- **Scope.** It applies to K19's rows only, so gpt-oss's epilogue and the MXFP4 store are untouched. The collapse now
  hands over its token rows whenever the mode is not `0`; a route that is not lean expands them inside, with the same
  launches as before.
- **Tests:**
  - the route tests add the default (unset / auto / AUTO / empty) taking the lean path with the same bits as `0`, and
    `auto` on a pre-K23 kernel side keeping the old launches without refusing;
  - `tests/test_k23_lean_glue_gpu.py` sets `0` for its OFF side and adds a default-is-lean check;
  - `bench/p89/staged.sha256` follows that file. P89 is read, and its receipts keep the bytes that ran.

### P89 read: LICENSED. K23's lean glue takes Qwen3-30B-A3B's B=16 int4 step to 0.950×, with identical tokens (bench, docs and register)

- `p89-5090-4` (RTX 5090): B=16 went from 10.321 to 9.805 ms/step. Tokens were identical in all 16 rows of both draw
  pairs, and there were 336 fewer launches per step (the 7 per layer K23 removes).
- Register: `e4b.serve.p89.qwen3.int4.k23-lean-glue-b16.5090.2026-10-01`.
- `p89-5090-3` read VOID on an engagement clause that named the expansion's kernel by inference (amendment 1, #818).
  It is recorded descriptively.
- To fit the new row under the bundle's cap, P82's and P83's claim texts are stated more tightly. No number moved.

### Lane P90, amendment 1: the KL arms move to an H100 NVL; one reading is a speed run and a quality run (bench and tests only)

- `p90-5090-1` read VOID: gpt-oss-20b's bf16 dequant reference (about 40 GB) does not fit a 32 GB RTX 5090. The KL
  arms died offloading it. P44-b scored that reference on an H100 NVL.
- `P90_ARMS=speed` runs the registered speed arms on the 5090. `P90_ARMS=quality` runs the registered KL arms on an
  H100 NVL. Both hold the premise and K0.
- `p90_reduce.py --speed-dir A --quality-dir B` gives the verdict under the registered rule.
- `p90-5090-1`'s speed numbers are recorded descriptively only: B=16 ×0.584, B=1 ×1.074.

### Lane P89, amendment 1: the engagement clause counts launches, not an inferred kernel name (bench and tests only)

- `p89-5090-3` read VOID on "at least 48 fewer `indexSelect` launches per step". That clause named the token-row
  expansion's kernel by inference. On torch 2.8 the expansion is `vectorized_gather_kernel`, and the route removed
  all 7 launches per layer it targets (336 per step).
- The clause is removed, the total-launch floor is now 288 per step (6 per layer), and kernel names are reported, not
  gated.
- `p89-5090-3` is recorded descriptively only (B=16 ×0.951, tokens identical); a fresh reading gives the verdict.

### Lane P90 registered (#564): does K21 make gpt-oss-20b's B=16 decode faster without moving the MXFP4 store's quality? (bench and tests only)

- **The question.** gpt-oss-20b's licensed MXFP4 store falls back to NF4 above 16 rows, and that NF4 GEMM is 79 % of
  the B=16 step. With `E4B_MXFP4_GROUPED_SMALLM=1` (#816), K21 serves those rows from the store's own bytes. P90 asks
  whether that is faster on an RTX 5090 without moving the store's quality.
- **Speed.** bo7's `store_r12`, OFF vs ON, at B=16 and B=1, two draws each, the first censused (K22's command line).
  Engagement is read from the censuses: K21 replaces the NF4 grouped GEMM at B=16 and the GEMV at B=1, 48 calls per
  step.
- **Quality.** P44's KL-from-reference instrument, unchanged, on arm `store_r12`, OFF then ON. It is decode-shaped,
  and under ON T == 1 reads K21. Two checks run first, before any fetch: the premise
  (`tests/test_k21_row_exact_gpu.py`: rows bit-equal alone and inside B=16) and the instrument's K0 controls.
- **The rule:**
  - QUALITY_FAIL if KL rises more than 0.0005 nats or top-1 drops more than 0.002;
  - LICENSED at B=16 ON/OFF <= 0.90;
  - VOID if OFF's KL does not reproduce P44's 0.0019 within 0.001, among other checks.
- `tests/test_p90_staged_pin.py` pins:
  - the staged bytes (P88's harness, P44's KL modules, the premise);
  - that the gnf4 pin is a real sha;
  - the refusal, premise, K0 and proof order;
  - the arms and their settings;
  - the exit codes;
  - the reducer's 18-case self-test;
  - the driver's dry run.

### `E4B_MXFP4_GROUPED_SMALLM=1` (K21, opt-in): the native MXFP4 store's decode rows go through K21 instead of falling back to NF4

- **Why.** gpt-oss-20b's licensed MXFP4 store has no batched kernel here. Above 16 rows (B=16 is 64) its experts fall
  back to the kept NF4 stacks, which are a requant that P44 read at 0.0222 nats against the store's 0.0019. That NF4
  grouped GEMM is 79 % of the B=16 step (grouped-nf4-gemm K22/K24 censuses, 17.9 of 22.5 ms).
  - On gpt-oss's recorded B=16 routing, K21 (#422, with #425's masked K tail) read 7.62 ms/step against the served
    route's 15.18 (K24, #428).
  - That read was VOID by its instrument and is descriptive only, so this route is opt-in, the treatment an
    end-to-end lane measures.
- **What.** Under `1`, `_fused_over_stack` keeps the MXFP4 store for every device-grouped decode row (up to 256) and
  serves both projections with `mxfp4_grouped.gemm_mxfp4_grouped_smallm`:
  - over the 16-row device tile table, with gate_up's gather folded in;
  - at K24's best plan (`_K21_PLAN`: BLOCK_N 32, KC 128, 4 warps, 3 stages; every K24 plan was bit-identical).

  gpt-oss's biases index by the sorted ids, and the unsort applies as before.
- **The quality instrument reads the kernel.** Under `1`, `_collapsed_grouping` also sends T == 1 on the MXFP4 store to
  the device tile table and K21, so P44's decode-shaped KL instrument reads the kernel it gates.
- **Default unchanged.** `0` (the default) keeps today's routes, and anything else is refused. `1` on a kernel package
  without K21, or without its masked tail, is a RuntimeError.
- **Tests:**
  - `tests/test_mxfp4_grouped_smallm_route.py` (stubs with an MXFP4 dequant oracle): the routes, the plan, the
    gpt-oss epilogue, the refusals, prefill untouched, the grouping decisions.
  - `tests/test_k21_row_exact_gpu.py` (real kernel, gpt-oss shapes, K = 2880 so the masked tail runs): a token's rows
    are bit-equal alone (T = 1) and inside B=16; the T = 1 output matches the fp32 dequant oracle through gpt-oss's
    epilogue; the T = 1 route captures and replays bit-equal.
  - NAS RTX A2000, correctness only:
    - GPU 9/9 with K19's and K23's files;
    - CPU route and MXFP4 neighbours 81 passed;
    - a mutation that drops gate_up's gather fails the B=16 row-exactness test and the stubbed route tests.

### Lane P89 registered (#564): does K23's lean glue make Qwen3-30B-A3B's B=16 int4 decode faster with the same bits? (bench and tests only)

- **The question.** P88 LICENSED K19 for T > 1 int4 decode rows. K23 (`E4B_INT4_LEAN_GLUE=1`, #814 over
  grouped-nf4-gemm #427) folds the launches around K19 into the kernels that bracket it, bit-identical by
  construction. P89 asks whether that is faster on an RTX 5090.
- **The lane.** `bench/p89/` is P88's speed instrument, with OFF / ON differing only in `E4B_INT4_LEAN_GLUE`. K19 is at
  its licensed `auto` in both arms. B=16, two draws each, the first censused.
  - There is no K8: it scores through the T == 1 loop, where the lean route never engages, so it would be inert as a
    gate.
  - The quality gates are the on-card premise, run before any fetch: `tests/test_k19_row_exact_gpu.py` +
    `tests/test_k23_lean_glue_gpu.py`, 6 passed and none skipped. Then token equality in all 16 rows of both draw
    pairs.
  - Engagement is read from the census: at least 48 fewer `index_select` launches per step (the expansion) and at least
    192 fewer launches per step overall.
  - The rule: LICENSED at ON/OFF <= 0.97, otherwise NOT_FASTER, IDENTITY_FAIL or VOID.
- `tests/test_p89_staged_pin.py` pins:
  - the staged bytes (P88's harness, the two premise tests);
  - that the gnf4 pin is a real sha;
  - the refusal, premise and proof order;
  - the arms and their settings;
  - the reducer's 14-case self-test;
  - the driver's dry run.

### `E4B_INT4_LEAN_GLUE=1` (lane K23, opt-in): K19's grouping glue folds into the builder and K19's store

- **Why.** P88 censused Qwen3-30B-A3B's B=16 step on an RTX 5090 (8 graph replays per arm).
  - Its K19 arm, against its GEMV arm, adds 0.685 ms/step of launches: the tile builder 0.430, three fills 0.096, an
    index kernel 0.094 (by its count, the unsort) and a scatter/gather kernel 0.065.
  - Both arms also pay an `index_select` of 0.46 ms/step. By its call count and per-call time it is inferred to be the
    collapse's `[T * top_k, H]` expansion of the token rows.
- **What.** On K19's rows, `_fused_over_stack` uses grouped-nf4-gemm K23 (#427):
  - it builds the table with `build_group_tiles_fused(..., lean=True, sorted_ids=True)`, one launch;
  - gate_up reads the collapse's token rows through `gather_div=top_k`, because `_forward_collapsed` now hands over
    `(x, row_token, top_k)` and the expansion is made inside only for routes that read it;
  - the down projection is stored with `scatter=order` straight into the caller's row order, so the unsort is skipped.
  - gpt-oss's epilogue reads the sorted down output, so it is excluded.
  - `0` (the default) keeps the separate launches. Anything else is refused, and `1` on a kernel package without
    K23's options is a RuntimeError, not a silent fallback.
- **No speed claim yet.** Lane P89 reads it end to end.
- **Tests:**
  - `tests/test_int4_grouped_smallm_route.py` (stubs):
    - the route takes K23's builder and the down scatter with the same bits as `0`;
    - token rows are read through `gather_div` under `1` and expanded inside under `0`, both bit-equal to the
      expanded call;
    - the refusals;
    - off K19's rows nothing changes.
  - `tests/test_k23_lean_glue_gpu.py` (real kernels): bit-equal to the default at B=16 from expanded rows and from token
    rows, eager, and captured in a CUDA graph and replayed on new inputs. It skips when the installed kernel package
    predates K23.
  - On the NAS RTX A2000 (correctness only), against K23's commit:
    - GPU 6/6 (with K19's row-exact file) and CPU route 24/24;
    - a mutation that drops the down scatter but still skips the unsort fails all three K23 GPU tests.

### Lane K24 runner (#564): K22 re-read on gpt-oss-20b with per-layer weight stores and K21's masked-tail plans (bench and tests only)

- **Why.** K22 read VOID: its bench's served NF4 kernel was 17 % under the in-model census. Descriptively, gpt-oss's B=16 step is 79 % that kernel. K24's prereg and bench live in grouped-nf4-gemm (`kernel/PREREG-k24-gptoss-per-layer.md`).
- **`bench/k24/` is K22's runner minus the prompts and routing-record phases:**
  1. census the served B=16 step;
  2. run `k24_bench.py` on K22's recorded routing, read from the gnf4 clone's receipts at the registered commit.
  Failures use rc 31 / 34.
- `tests/test_k24_staged_pin.py` pins the staged bytes, the order, the env, the instrument wiring, the codes and the dry run.

### Lane K22 runner (#564): gpt-oss-20b at B=16 on an RTX 5090 -- census, recorded routing, and K21 against the served expert route (bench and tests only)

- **Why.** The first lane of the throughput push to other model families. gpt-oss's licensed MXFP4 store has no batched kernel, so at B=16 (64 rows) the experts fall back to the kept NF4 stacks; bo7 timed that step at 21.65 ms. The prereg, the bench and the rule live in grouped-nf4-gemm (`kernel/PREREG-k22-gptoss-mxfp4-b16.md`). This repo carries the runner.
- **`bench/k22/`, one box, three phases:**
  1. census the served B=16 step (bo7's `store_r12`) with P42's replay census;
  2. tokenize 16 wikitext rows with gpt-oss's own tokenizer (`step_decomp._k8_window`, as P37) and record 128 teacher-forced B=16 steps of routing with `bench/families/record_eids.py`;
  3. run grouped-nf4-gemm's `k22_bench.py` on that routing. Its instrument is phase 1's own `_gemm_nf4_grouped` row.
- **Codes and proof.** Lane failures use rc 31–33, never the launcher's machine-exclusion codes. The proof (`K22_PROVE=1`) compiles K21's and K16's contracts on the card and fetches no model.
- `tests/test_k22_staged_pin.py` pins:
  - the staged bytes (P86's harness, the family recorder, P44's served-model builder);
  - the refusal and proof order;
  - the phase order and bo7's env;
  - the instrument wiring;
  - the exit codes;
  - the driver's dry run.

### K19 is the default for batched int4 decode rows (`E4B_INT4_GROUPED_SMALLM` now defaults to `auto`), as lane P88 licensed

- **What changes.** In the device-grouping configuration, the one every B=16 register row is measured in, int4 decode rows above T == 1 (≤ 256 rows) now run grouped-nf4-gemm's K19 when the installed kernel package carries it. Before, they ran the split-K GEMV.
- **Why.** P88 (`e4b.serve.p88.qwen3.int4.k19-b16.5090.2026-10-01`): B=16 step 0.905× on an RTX 5090, K8 +0.0062 nats (floor 0.0095).
- **What doesn't change:**
  - **T == 1** stays on the singleton GEMV (P88 read B=1 1.103× slower);
  - the library's default batched path with `DEVICE_GROUPING` off;
  - prefill rows;
  - a kernel package without K19 (released grouped-nf4-gemm ≤ 0.33.7): auto falls back to the GEMV silently.
- **Values:**
  - `auto` (default, also unset or empty);
  - `0`: the split-K GEMV everywhere;
  - `1`: requires K19, refuses if absent, and also routes T == 1 (the quality instrument's setting);
  - anything else is refused.
- `tests/test_int4_grouped_smallm_route.py` pins each value's route, the silent fallback, the refusal, and T == 1 under each. `int4_experts.py`'s Scope note says the same.
- Three tests whose subject is the split-K GEMV route now select it explicitly (`=0`): `test_int4_device_grouping.py::test_int4_decode_routes_to_gemv`, P63's `test_int4_device_grouping_gemv_is_row_exact` and `test_int4_decode_a16.py::test_on_does_not_cover_the_device_grouped_decode_gemv`. K19's own row invariance is `tests/test_k19_row_exact_gpu.py`. With the real kernel (grouped-nf4-gemm `7b7e6b1`) on an RTX A2000, the 9 dispatch and route files pass: 62 passed.
### Lane P88 read (#564): LICENSED. K19 takes the RTX 5090's B=16 int4 decode step to 0.905×, K8 +0.0062 nats; B=1 is 1.10× slower (bench, docs and register only)

- **Run:** `p88-5090-4` on an RTX 5090 with an AMD EPYC 9334 host, e4b `b848089` + grouped-nf4-gemm `7b7e6b1` (K19's plan 32/256). $0.5538. The lane cost $0.6582, including a proof and three pre-flight NOT_RUNs.
- **Steps (medians of two draws):**
  - B=16 11.200 → 10.140 ms (0.905, bar 0.95);
  - B=1 4.325 → 4.771 (1.103, SLOWER).
- **K8 on the licensed recipe:** OFF 1.8434202801176407, ON 1.8496547109991661: **+0.00623 nats** (floor 0.0095). The build equals OFF bit for bit, and the pack it dumped carries the licensed fingerprint `sha256:0c9955a9`.
- **Census at B=16:** GEMV 6.34 → K19 5.05 ms per step. Grouping adds 0.69 (tile build 0.43 + glue 0.26) against 0.60 of quantise and reduce removed.
- **Predictions:** B=16 0.78–0.86 refuted (0.905); B=1 slower held; |ΔK8| < 0.003 refuted (+0.0062, inside the floor).
- **Register:** `e4b.serve.p88.qwen3.int4.k19-b16.5090.2026-10-01`. STATUS's P86 sentence is condensed and now cites it. P84's and P86's register sentences are tightened, with their figures unchanged, to keep the bundle under its cap.
- **What follows:** K19 by default for int4 decode rows above T == 1, a separate PR.

### `bench/families/record_eids.py`: P60's routing recorder, generalised to every MoE family (bench and tests only)

- P60's recorder (`bench/p60/record_eids.py`, a registered lane's staged file, left byte-identical) hooked `.gate` on classes named `*SparseMoeBlock` and parsed Qwen's router output, so it missed gpt-oss, Granite and Gemma-4, whose routers are named `router`.
- The new copy pre-hooks each block's `experts` call and reads its `top_k_index` argument. Every admitted family calls `experts(hidden, top_k_index, top_k_weights)` in transformers 5.x, and e4b's served wrapper keeps that signature. It refuses a model with no experts module, a call shaped otherwise, and a `top_k` mismatch.
- `tests/test_record_eids_families.py` covers tiny Qwen3-MoE, OLMoE, Granite-MoE and gpt-oss models. On each, the recorded ids must equal the family's own router indices, call for call; the routers return `(logits, w, idx)`, `(idx, w, logits)` and `(logits, scores, idx)`. Groundwork for per-family routing replays (the K20 method, other families).

### Lane P88 registered (#564): P87's instrument on K19's new plan, with a CPU floor (bench and tests only)

- **Why.** grouped-nf4-gemm's K20 (#421) found K19's plan on the 5090: BLOCK_N 32 / KC 256 serves recorded B=16 routing at 0.736× the int8 GEMV route, with outputs bit-identical across plans. P87's read was VOID because its calibrated K8 build ran out of its alarm on a Broadwell host.
- **P88 is P87's arms, reducer and rule**, pinned to gnf4 `7b7e6b1`. It adds:
  - a tripwire on K19's default plan;
  - a host CPU-vendor floor (AMD; **rc 18**, which the launcher excludes the machine on, adertha#131) before any install;
  - a 5,400 s build alarm and a 3.0 h guard.
- **Predictions:** B=16 0.78–0.86; B=1 SLOWER (1.03–1.20, so a default would cover T > 1 only); |ΔK8| < 0.003.
- **Cost:** a proof (0.5 h), then the reading (3.0 h). Lane ceiling $3.50.
- `tests/test_p88_staged_pin.py` adds the vendor refusal, the plan tripwire and the alarm to P87's pins.

### Lane K20 runner (#564): K19's plan space on an RTX 5090, replaying P60's recorded B=16 routing (bench and tests only)

- The prereg, the bench and the rule live in grouped-nf4-gemm (`kernel/PREREG-k20-k19-plan-sweep-5090.md`, grouped-nf4-gemm#420). This repo carries the runner, `bench/k20/`, in K18's pattern:
  - install gnf4 at the registration's merge, then a tripwire that the installed module is the pinned cut, with K19 and the tile builder present;
  - K19's contracts, under the interpreter and then compiled on the card, before any timing (rc 21 / 22);
  - the rule's self-test, then the bench on P60's committed ids.
- `tests/test_k20_staged_pin.py` pins the staged bytes (the routing is K18's, byte-identical) and the order: contracts, then self-test, then bench.
### Lane P87 read (#564): VOID by the rule; the speed arms show K19 at 0.971× the GEMV step at B=16 and 1.236× at B=1 on an RTX 5090 (bench and docs only)

- **Why VOID.** The calibrated K8 build ran on an Intel Xeon E5-2698 v4. It needed 1,120 s for its first chunk (P85's AMD host: 360 s), hit its 3,600 s arm alarm after 3 of 5 chunks and dumped no pack, so K8 OFF/ON never ran. No register row.
- **The speed arms passed every check of their own:** the premise (K19 rows bit-equal alone and inside B=16; 3 passed on both 5090s), the 3 % draw spread, and engagement (96 K19 calls per step ON, the experts' 96 GEMV calls gone, none OFF).
- **Steps (median of two draws):**
  - B=16: 12.013 → 11.667 ms (0.971; the bar was 0.95, I predicted 0.75–0.85);
  - B=1: 4.277 → 5.286 (1.236).
- **The census.** Kernel for kernel, K19 is 1.08× the GEMV (6.50 vs 7.00 ms per step): about 75 % of the int4-b32 byte floor, against the GEMV's 70 % and Marlin MoE's 94 %. Grouping adds about 0.8 ms per step (tile build 0.50, glue 0.30), nearly cancelling the 0.73 ms of reduce and quantize removed. At B=1, one-row tiles make K19 1.91× the GEMV's expert time.
- **Why the prediction failed.** It rested on an A2000 timing (K19 1.89×), but the A2000's GEMV runs far below its own floor (11.4× slower than the 5090's). A timing on that card says nothing about the 5090.
- **What follows.** K19 stays opt-in; P87 is not re-run. Next: a 5090 kernel microbench on P60's recorded routing (K19's plan space, Marlin MoE, the tile build) before any end-to-end lane.
- Spend: $0.9851 of the $3.00 ceiling (proof $0.0329, a $0 refusal, the reading $0.9522).

### Lane P87 registered (#564): does K19 make e4b's int4 decode faster on an RTX 5090 without moving its quality? (bench and tests only)

- **The question.** On one RTX 5090, with `E4B_INT4_GROUPED_SMALLM` on against off:
  - **Speed:** B=16 and B=1 step times. P86's harness, RTN env and command line, two draws each, the first censused.
  - **Quality:** K8 on the licensed recipe (calibrated experts and int4 attention, P85's env and K8 arguments). One build dumps the pack, then OFF and ON load it by fingerprint.
  - grouped-nf4-gemm is pinned at K19's merge, `3351c9d`.
- **The premise, on the card.** `tests/test_k19_row_exact_gpu.py` runs before anything is fetched: a token's K19 rows are bit-equal alone and inside a B=16 step, which lets the B=1 K8 stand for the batched rows. If it fails, the lane stops (rc 25).
- **The rule** (`p87_reduce.py`, 17-case self-test):
  - **VOID** on a failed premise; on failed engagement (96 K19 calls per step ON, the experts' 96 GEMV calls gone, none OFF); or on a bit-equal K8 ON/OFF.
  - **QUALITY_FAIL** if |ΔK8| > 0.0095 nats.
  - **LICENSED** if B=16 ON/OFF ≤ 0.95.
  - **NOT_FASTER** otherwise.
  - B=1 is read as FASTER, NEUTRAL or SLOWER beside the verdict, and decides the scope of a proposed default.
- **Cost.** A proof (0.5 h: K19's contract tests and the premise compiled on sm_120, no model), then a 2.5 h reading. Lane ceiling $3.00.
- `tests/test_p87_staged_pin.py` pins:
  - the staged bytes, which are P86's harness bytes;
  - the gnf4 pin;
  - the refusals and the premise coming before any fetch;
  - the proof;
  - the arm order and settings;
  - the driver's dry run.

### `E4B_INT4_GROUPED_SMALLM=1` now covers T == 1 decode too (opt-in; no default changes)

- **The gap.** The route above engaged only where device grouping was already on: T > 1 under the batched harness's `DEVICE_GROUPING`. T == 1 (B=1 decode) kept the singleton int4 GEMV. K8 scores through the T == 1 loop, so a K8 read of the opt-in would have measured the GEMV it meant to replace, and the quality gate would have been inert. This was found while writing lane P87's reducer, before anything ran.
- **The change.** `_collapsed_grouping(T, int4_stores)` decides the all-resident collapse's grouping. With the opt-in and a uniform-int4 store (not MXFP4), T == 1 takes the device tile table (capture-legal, no host sync), and its 8 routed rows reach K19. Without the opt-in, the decisions are what they were.
- `tests/test_int4_grouped_smallm_route.py` adds:
  - the decision table for both T and both settings, plus MXFP4 and no store;
  - an end-to-end T == 1 call (one row per expert) through K19 against the oracle.
- `tests/test_k19_row_exact_gpu.py` (GPU; skips without CUDA or K19), on the real kernel at Qwen3-30B-A3B's expert shapes:
  - a token's rows come out bit-equal whether it decodes alone (T == 1) or inside a B=16 step, so a T == 1 instrument such as K8 stands for the batched rows;
  - K19 runs at T == 1 and matches an fp32 dequant oracle;
  - the default T == 1 route is untouched;
  - the T == 1 route captures in a CUDA graph, as the B=1 decode loop does, and a replay on new inputs equals eager to the bit.
  - On an RTX A2000 (sm_86), tokens 0, 5 and 15 were bit-equal, and the captured replays matched eager. Relative error against the oracle: K19 0.0048, the GEMV 0.0131.

### `E4B_INT4_GROUPED_SMALLM=1`: int4 decode rows through grouped-nf4-gemm's K19 grouped tensor-core GEMM (opt-in; no default changes)

- **What it routes.** At decode shapes (≤ 256 routed rows) on the int4 expert store, today's route is the split-K GEMV, the row P86 measured at 6.98 ms/step against Marlin MoE's 4.78 at B=16 (#564). The opt-in sends those rows through K19 (`int4_smallm.gemm_int4_b32_grouped_smallm`, grouped-nf4-gemm#419) instead, using the existing device-grouping branch:
  - the 16-row tile table is built once per layer;
  - gate_up runs as K19 with its gather folded in (`order`);
  - down runs as K19 on the already-sorted epilogue output;
  - the existing unsort and combine are unchanged.
- **Why it is opt-in.** K19 multiplies bf16 activations instead of int8-quantised ones, a different arithmetic, so it stays opt-in until a registered quality read licenses it.
- **Asked for and absent is refused.** If the kernel side lacks K19, the opt-in raises a `RuntimeError` naming the requirement; it never falls back to the GEMV silently. Prefill rows (> 256) are untouched.
- `tests/test_int4_grouped_smallm_route.py` (Linux CI, stubbed kernels) pins:
  - the route (gate_up with the gather, down without);
  - no GEMV, and 16-row tiles;
  - the per-row oracle in the caller's row order;
  - the default route unchanged;
  - the refusal, and prefill untouched.

### Lane P86 read (#564): READ -- vLLM's B=16 lead is the expert kernel; Marlin MoE runs the experts in 4.78 ms per step against e4b's int4 GEMV 6.98 (2.86 of the 3.32 ms gap) (bench, docs and register only)

- `p86-5090-3` ran on one RTX 5090 for $0.5170, teardown proven, with the same prompts as P58. The lane cost $0.8561: three proofs (a bandwidth NOT_RUN, the attempt that caught the CUDA 12.8 image, and a pass), an ssh NOT_RUN, and the reading.
- **Steps.** B=16: e4b 12.06 ms vs vLLM 0.30.0 8.75 (1.38x; P58 1.40x). B=1: 4.27 vs 4.00 (1.07x). Both censuses reconcile to their own step.
- **B=16 by role (e4b / vLLM / gap):** quantized linear 8.61 / 5.75 / +2.86, of which experts 6.98 + reduce 0.39 + quantize 0.35 against Marlin MoE 4.78 (the attention projections are at parity, 0.89 vs 0.97); routing glue 1.07 / 0.45 / +0.62; other +0.44; attention + KV write 0.96 / 1.35, **e4b faster**. Stated expectation held.
- **Against #564's byte floor** the e4b GEMV runs about 70 % and Marlin MoE about 94 % of its own.
- **Next (proposed):** a kernel lane on the B=16 expert matmul, benchmarked against Marlin MoE's 4.78 ms, starting with a $0 A2000 microbench of both kernels on a recorded routing.
- New row `e4b.serve.p86.qwen3.b16.kernel-census-vs-vllm-0.30.0.5090.2026-10-01`. STATUS adds P86 beside P58's comparison and condenses the #674 bullets now that P85 answered it. The read is in `bench/p86/RESULTS-p86.md`.

### Lane P86 amendment 1 (#564): the vLLM image needs a >= 12.9 CUDA toolkit; the runner refuses an older one before any install (bench only; nothing in the wheel changes)

- `p86-prove-2` caught a defect. vLLM 0.30.0 installed and imported, but its warmup died in FlashInfer's JIT (`requires GPUs with sm75 or higher`) on the sm_120 card. The lane's image was `2.8.0-cuda12.8-devel`, carried over from P84/P85; P58's registration names `2.8.0-cuda12.9-devel` for vLLM on sm_120. `p86-prove-1` was a launcher NOT_RUN (bandwidth). Together $0.1156; no reading attempted.
- **Changes, before any reading:** the image is P58's; the runner refuses a container toolkit below 12.9 with rc 24, before any install; up to two more proof attempts under the corrected image. The question, the arms, the rule and the ceiling are unchanged.

### Lane P86 registered (#564): where do e4b's and vLLM's decode steps go, kernel by kernel, on one box? (bench only; nothing in the wheel changes)

- `bench/p86/PREREG-p86.md`. P58 measured vLLM 0.30.0 at 1.087x (B=1) and 1.396x (B=16) e4b's int4 stack end to end. e4b's B=16 step has a census (P57); vLLM's never had one, so the 3.3 ms gap could not be assigned. This lane censuses both engines on one RTX 5090 with the same prompt token ids.
- **e4b:** the current release's int4 stack through P58's exact harness, fused q/k/v at both batches, timed by the graph-replay window and censused by P42's replay profiler. **vLLM 0.30.0** (P58's comparator, the GPTQ-Int4 checkpoint on Marlin): timed by P37's slope arm, unchanged, and censused by the new `p86_vllm_census.py`. The census runs the engine in-process under `torch.profiler` and applies the same 32 -> 128-token slope to each kernel. Only kernels that gain calls in the long run count as decode.
- **The rule:** kernels sorted into registered families, families into roles present in both engines, plus host and launch (step minus kernels). VOID on a missing arm, the wrong vLLM, or a census exceeding its own step by 10 %. NOT_READ if over 10 % of either engine's B=16 kernel time is unmapped. Otherwise the largest B=16 gap names the next lane. Stated expectation: quantized linear (the expert GEMV against Marlin MoE).
- **Refusals before any install:** the card class, the disk, and a driver below 580 (vLLM 0.30.0's wheels are CUDA 13.0; rc 18).
- **A2000 rehearsal of the method** (vLLM 0.11.0, Qwen3-0.6B): graph replays are recorded, each decode kernel once per layer per step. The graph arm read 5.79 ms/step of decode kernels against the eager control's 7.40.
- **A proving rental first** (0.5 h, no 30B model). It runs the census arm itself on vLLM 0.30.0 with Qwen3-0.6B, proving the profiler on this build, driver and card, which the A2000 cannot. Then the reading: guard 2.0 h at <= $0.75/h (P58 registered 3.0 h for its 18 arms); lane ceiling $2.50. `tests/test_p86_staged_pin.py` (17) pins the staged bytes, P58's harness and comparator bytes, the comparator, the refusals' order, the proof, the arms and the census settings.

### Lane P85 read (#674): CONFIRMED -- grouped-nf4-gemm#413 (the fused fp8 KV append's IEEE-rounded quotient) is the whole step that moved the recipe's fp32 K8 from 6.36709 to 6.36396 (bench, docs and register only)

- `p85-5090-4` ran on one RTX 5090 on P70's own Ryzen 7950X card for $0.6386, teardown proven. The lane cost $0.7362: a proof, two Intel-host refusals at preflight (rc 16), and one attempt whose deadline guard did not arm on a Vast HTTP 429.
- **The readings.** Every reading used P70's harness and e4b 0.37.4, and every pair repeated bit for bit. The control, P70's build, read O's known `1.8511420498367808`. The same stack with `E4B_FUSED_KV_APPEND=0` read `1.8506507749113845`, the gnf4 0.33.7 builds' float, bit for bit. gnf4 0.33.6 with the append on read O's float bit for bit.
- **So #413 is the whole KERNEL step P84 found.** P84 had inferred it from the code; this reading measures it. The newer 6.36396 is the reading with the reference quantizer's KV bytes. Whether the licensed row names its software is the owner's decision.
- **Not measured here:** the P81/P82 cast pair (#413 is its leading explanation) and the Intel-host attention calibration.
- New row `e4b.serve.p85.qwen3.int4-recipe.k8.fused-append-413.5090.2026-09-30`. P84's row and `RESULTS-p84.md` are noted as measured. The read is in `bench/p85/RESULTS-p85.md`.

### Lane P85 amendment 1 (#674): the first-chunk watchdog admits a Zen 2 host (1,500 s), the guard is 4.0 h, the ceiling $4.00 (bench only; nothing in the wheel changes)

- `p85-prove-1` PROVED on an AMD EPYC 9655 host ($0.0573). The first reading, `p85-5090-1`, landed on an Intel Core Ultra 9 285K, which the runner refused at preflight with rc 16 before any install ($0.0251). No data exists.
- **Why.** The launcher excludes a machine only on ssh-readiness failures or a lane exit of 13, 14 or 17. A refusal on rc 16 or rc 30 re-rolls onto the same cheapest offer. That offer was then an EPYC 7K62, the Zen 2 class that took 1,121 s to its first calibration chunk in P81, past the 900 s watchdog.
- **Changes, before any reading:** watchdog 900 → 1,500 s (P83's); guard 3.0 → 4.0 h at ≤ $0.75/h; lane ceiling $3.00 → $4.00; at most three vendor refusals in a row. The question, readings, rule and predictions are unchanged, and `p85-prove-1` stands as the lane's proof.

### Lane P85 registered (#674): is grouped-nf4-gemm#413 (the fused fp8 KV append's IEEE-rounded quotient) the whole step that moved the recipe's fp32 K8 from 6.36709 to 6.36396? (bench only; nothing in the wheel changes)

- `bench/p85/PREREG-p85.md`. P84 localized the move to grouped-nf4-gemm 0.33.0 → 0.33.7 and read #413 from the code as the one change on K8's path. This lane measures it on one AMD-host RTX 5090, through P70's harness and env throughout.
- **The control runs first.** `O_build` is P70's build (e4b 0.37.4 + gnf4 0.33.0), and it must read O's known mean NLL `1.8511420498367808` bit for bit. If it doesn't, the lane is VOID and stops.
- **F:** the same stack with `E4B_FUSED_KV_APPEND=0`, so every append goes through `quantize_kv_fp8`, whose bytes 0.33.7's fused kernel writes exactly. Predicted: N's float, `1.8506507749113845`. **S:** e4b 0.37.4 on gnf4 0.33.6, the append on. Predicted: O's float. Each is read twice with the O build's expert pack loaded by fingerprint.
- **The rule:** VOID, then PATH-REFUTED (F = O: the append is not on the path), CONFIRMED (F = N and S = O), MIXED (F = N, S ≠ O), REFUTED (otherwise). Stated expectation: CONFIRMED.
- **Tripwires and stamps.** Each stack must carry a pre-#413 `fp8_kv`, and 0.37.4's resolver must turn the append on by default and off under the knob. Every K8 process is stamped with the resolver's answer under its own env.
- **A2000 rehearsal.** R rc 0: both stacks installed, and the stamps read the append on and off. M1 (a 0.33.7 kernel posing as S) was refused by the pre-#413 tripwire, rc 9. M2 (the knob leaking into every process) was refused by the stamp check, rc 9. V (the default vendor on the Intel host) was refused with rc 16 before any install.
- Guard 3.0 h at ≤ $0.75/h after a 0.4 h proof; lane ceiling $3.00. `tests/test_p85_staged_pin.py` pins the staged bytes, the stacks, the known floats, the order and the control gate, each reading's env, the tripwire and the stamp, and the driver's dry run.

### Lane P84 read (#674): KERNEL. grouped-nf4-gemm 0.33.0 → 0.33.7 moved the recipe's fp32 K8 (−0.000491 nats); e4b 0.37.4 → 0.37.8 and the harness move it by exactly zero (bench, docs and register only)

- `p84-5090-3` ran on one RTX 5090 on the AMD Ryzen 7950X host that read P55x's and P70's 6.36709 (same GPU) for $1.0928, teardown proven. The lane cost $1.8328 of its $4.00 ceiling over six rentals.
- **The chain.** The control C (P82's build) read N's known float `1.8506507749113845` bit for bit. H1 (e4b 0.37.8, P70's harness) and H2 (e4b 0.37.4, P70's harness), both on gnf4 0.33.7, read the same float. Every build repeated itself, and every expert pack was the licensed `0c9955a9…`. So KERNEL (O → H2) is the whole −0.000491 nats, and PACKAGE and HARNESS are 0.0. The stated expectation, PACKAGE alone, was wrong.
- **Which commit is not measured.** Read from the code, the one change in the cut on K8's path is gnf4#413 (0.33.7), which IEEE-rounds the fused fp8 KV append's quotient (B771: the old kernel wrote a different byte in 3.9e-8 of values). K8's eager loop calls `graph_mode_init`, so every scored token appends through that kernel. 0.33.1–0.33.3 change no code, and 0.33.4 (`cold_deadline`), 0.33.5 (`int4_smallm` annotations) and 0.33.6 (MXFP4) are off this path.
- **Correction.** PREREG-p82 and PREREG-p84 said the eager K8 does not run the graph-mode append. It does. Noted in `RESULTS-p82.md`, `RESULTS-p83.md`, the P82 and P83 register rows and STATUS; the registrations stay as registered. #413 is also the first suspect for the P81/P82 cast pair (gnf4 0.33.5 vs 0.33.7), not measured.
- **Proposed (P85, not started):** P70's build with `E4B_FUSED_KV_APPEND=0` should read N's float bit for bit if #413 is the whole step.
- New row `e4b.serve.p84.qwen3.int4-recipe.k8.factor-chain.5090.2026-09-30`. The read is in `bench/p84/RESULTS-p84.md`; receipts in `bench/p84/receipts/p84-5090-3/`.

### Lane P84, reading 1 VOID and amendment 1 (#674): on an Intel host the recipe's attention calibration does not reproduce the AMD hosts' pack; P84 is restricted to AMD hosts; P83's cross-machine claim qualified (bench, docs and register only)

- `p84-5090-2` ran on one RTX 5090 on an Intel Core i9-14900K host for $0.5356, teardown proven. The lane has spent $0.7042 so far: two proof attempts and two reading attempts.
- **The control failed, so the reading is VOID.** P82's own build read K8 **6.36276** (mean NLL `1.8504622130569988`), not N's known 6.36396. Its repeat was bit-identical. The lane stopped before H1 and H2, as registered.
- **The attention pack is different: `45b4cc5a…`, not `d7cfa1f4…`.** 342 of 384 tensor payloads differ, in all 48 layers, with the same calibration token stream and toolchain. The expert pack is the licensed `0c9955a9…` again. The mechanism is not isolated: CPU vendor and driver both differ from every earlier host.
- **Amendment 1.** The runner refuses a CPU vendor other than `AuthenticAMD` before the install and the fetch (rc 16), because the known floats were read on AMD hosts. On the A2000 (an Intel Xeon) the default refuses with rc 16, and the knob lifted proves.
- **Qualified.** P83's "K8 is bit-reproducible across machines" becomes: on the AMD hosts tested. Changed in `RESULTS-p83.md`, the P82 and P83 register rows, and STATUS. The read so far is in `bench/p84/RESULTS-p84.md`.

### Lane P84 registered (#674): which factor moved the recipe's fp32 K8 from P70's 6.36709 to P82's 6.36396 -- the kernels, the package, or the harness? (bench only; nothing in the wheel changes)

- `bench/p84/PREREG-p84.md`. P83 showed the two K8 values are bit-reproducible across machines, so their known floats anchor a chain of one-factor steps built on one RTX 5090. From O (known), a new gnf4 alone gives H2 (KERNEL). A new e4b on top gives H1 (PACKAGE). P82's harness on top gives C (HARNESS), and C = N.
- **The control runs first.** C is P82's build, and it must read N's known mean NLL `1.8506507749113845` bit for bit. If it doesn't, the lane is VOID and stops before the other builds.
- **Every build repeats itself within the box.** C's repeat is K32. H1 and H2 use P55x's lic arm.
- **The verdict names every step whose ends differ** (some of KERNEL, PACKAGE, HARNESS), with each step's nats; the steps sum to N - O. Stated expectation: PACKAGE alone.
- **The first-calibration-chunk watchdog is now 900 s**, so a slow host fails fast rather than outrunning a three-build guard.
- **A2000 rehearsal.** PROVE rc 0: both stacks installed, with e4b 0.37.4 on gnf4 0.33.7 and P70's harness importing on both. A mutation with the router export deleted was refused.
- `tests/test_p84_staged_pin.py` pins:
  - the files, and the known floats against P83's receipts;
  - the control gate on P83's N and O receipts;
  - the order and the stop on a failed control;
  - each build's harness and env, the manifests, the watchdog, and the driver's dry run.

### Lane P83 read (#674): on one box the software moved the recipe's K8 (DIFFERENT, −0.00049 nats), and each value reproduces bit for bit across machines; this corrects P82's read (bench, docs and register only)

- `p83-5090-1` ran on one RTX 5090 (AMD EPYC 9755) for $0.7671, teardown proven. The lane cost $0.8179 with its proof.
- **The reading.** Both builds used the fp32 router. P70's build (e4b 0.37.4, grouped-nf4-gemm 0.33.0) read wikitext K8 **6.36709**, and P82's (0.37.8, 0.33.7) read **6.36396**. Each repeated itself bit for bit, and both produced the licensed expert pack.
- **Bit-reproducible across machines.** Each mean NLL is the same float its software gave elsewhere: P70's on P70's and P64's machines, P82's on P82's. So this K8 does not depend on the box, and the software moved it. Which change is not isolated: the attention calibration (O's bytes cannot be dumped on 0.37.4) or the decode path.
- **Correction.** P82's read said K8 "does not reproduce across boxes" and that the fp32 residual "cannot be attributed to software". Both are wrong for the fp32 router, and are corrected in place in `RESULTS-p82.md`, P82's register row, STATUS and on #674. The one pair that prompted them, P81's and P82's cast-router readings, stays unexplained.
- New row `e4b.serve.p83.qwen3.int4-recipe.k8.software-ab.5090.2026-09-29`. The read is in `bench/p83/RESULTS-p83.md`.

### Lane P83 registered (#674): on one box, do P70's build (e4b 0.37.4, grouped-nf4-gemm 0.33.0) and P82's build (0.37.8, 0.33.7) read the same K8 with the fp32 router? (bench only; nothing in the wheel changes)

- `bench/p83/PREREG-p83.md`. P82 found that the licensed recipe's K8 does not reproduce across boxes. This lane separates box from software by running both builds on ONE RTX 5090, each exactly as its lane ran it: pins, harness copy, hook and env.
- **Stack O.** P70's build, then its within-box repeat: P55x's "lic" arm, with the expert pack loaded by fingerprint and the attention calibrated live. **Stack N.** P82's build, then its repeat, K32, with both packs loaded by fingerprint. O is installed first, and N over it with `--force-reinstall --no-deps`.
- **The rule.** VOID if either stack does not repeat its own K8 bit for bit, or if a reading is off-window, not fp32, or from the wrong stack. SAME if O's and N's builds read bit-identical K8, meaning the spread between boxes is the box. DIFFERENT otherwise, meaning the software moved it.
- **Reported:** whether O reads the licensed 6.36709 and N reads P82's 6.36396, and the pack fingerprints.
- **A2000 rehearsal.** PROVE rc 0: both installs, both tripwires, and both router stamps, including 0.37.4's router module. A mutation with the router export deleted was refused. A transient `git` failure (exit 128) in the first install led to a single pip retry.
- `tests/test_p83_staged_pin.py` pins:
  - the staged files, with O's harness equal to `bench/p70`'s pins and N's to `bench/p82`'s;
  - O's commits, the 13-case rule, the fp32 export, and the step order;
  - each step's harness and env;
  - the driver's dry run, which catches an apostrophe in `${VAR:?...}` that had swallowed the rest of the script.

### Lane P82 read (#511, #674, #777): on the licensed int4 stack the fixed graph path decodes exactly as the eager runner and is 12.0–12.4× faster on this host (CONFIRMED); for #674 the attention pack reproduces across boxes but K8 does not (bench, docs and register only)

- `p82-5090-3` ran on one RTX 5090 (AMD EPYC 7R13) for $0.851, teardown proven. The lane cost $2.0438 across five rentals: two proof attempts, two host failures and the reading.
- **Decode.** P80's trace on P55x's recipe (calibrated int4 experts and attention, T=1 folds, fused router at fp32 weights), both packs loaded by fingerprint, all arms on the device grouping.
  - The eager runner, its bucket step and both graph arms decode identical tokens in every row (A1 ≡ P ≡ B1 ≡ B2). In P81 they did not.
  - B1/A1 = 12.006 and B2/A2 = 12.358; the self-pairs are 0.974 and 1.003. **CONFIRMED.**
  - The eager step is host-bound: 88–102 ms at every row count, against 13.6 ms for the graph step at 16 rows and 4.4 ms at one.
- New row `e4b.serve.p82.qwen3.licensed-int4-fp32router.dynb.graph-buckets.fixed-path.5090.2026-09-29` supersedes P81's. STATUS's P81 entry is rewritten around it. The read is in `bench/p82/RESULTS-p82.md`.
- **#674 (reported).**
  - The attention pack `d7cfa1f4…` is byte-identical to P81's, on a second box and release.
  - On one box, wikitext K8 reads 6.36396 with the fp32 router (the build and a pack-loading process alike) and 6.31811 with the cast. The cast moves K8 by −0.0072 nats but is not the whole 6.36709 → 6.33015 gap.
  - With identical packs and router setting, P81's box read 6.33015. So on this stack K8 does not reproduce across boxes to five decimals, which the registered table had assumed.
- Also filed: #784. A lane driver cannot tell a dead box from a silent one, and waited out the whole guard on one.

### `docs/claims.json` is back to two-space indentation, and a test keeps it there; the llms bundle cap is 500,000 bytes (register formatting, tests and bundle config; no row changes)

- #810 re-wrote the register at indent 1. The rows were unchanged, but every line moved, so any open pull request that touched the register conflicted on all of it. It is re-serialised as `json.dumps(indent=2, ensure_ascii=False)` plus a newline, the form it had before #810, with identical content. `tests/test_claims_json_format.py` fails on any other serialisation.
- `docs/llms-bundle.json`'s `max_bytes` goes from 400,000 to 500,000. The bundle had reached 399,974 bytes, and lanes were shortening register sentences to fit it (#810 did). It has grown by roughly 3–4 KB a day.

### Owner quotes and name credits removed from the documents (docs only)

- Verbatim chat quotes and name credits are removed from pre-registrations, RESULTS pages, planning documents, the CI workflow comment and one CHANGELOG line. Directives are paraphrased or reduced to their date and record; no criterion, band, measurement or date moved.
- Four OpenTimestamps-anchored documents were edited: `PROVENANCE.md`, `docs/NULL_LADDER_1024_AMENDMENT.md`, `docs/POST_AUDIT_WORK_QUEUE.md` and `docs/SPECULATIVE_LANES_PLAN.md`. Each now ends with a note that its `.ots` anchors the version before the edit, which git history keeps.

### Lane P82 registered (#511, #674, #777): P81 re-measured on the fixed graph path with the licensed build's fp32 router, and whether the router cast accounts for #674's K8 gap (bench only; nothing in the wheel changes)

- `bench/p82/PREREG-p82.md`. One RTX 5090. P81's build and five arms (the licensed int4 stack, both packs loaded by fingerprint, the device grouping), on grouped-nf4-gemm 0.33.7 and an e4b containing #777.
- **The router.** Every process runs `E4B_ROUTER_EPI_CAST=0`, the fp32 routing weights of the builds that read the licensed K8 6.36709. The tripwire refuses a box where the module reads otherwise. The runner stamps what `router_epilogue` read under each process's env into a sidecar beside its receipt.
- **Decisive, new:** the eager runner must decode exactly as its bucket step (A1 ≡ P), alongside B1 ≡ B2 ≡ P. P81 had reported A ≠ P as not decisive, and that divergence hid #413's append and #777's bucket-1 bug. An arm not on the fp32 router voids the read.
- **Confirmed only if all hold:** the four streams are identical in every row, every bucket captures, and B/A > 1.03 in both pairings. A confirmed read supersedes P81's register row.
- **Reported for #674.** After the verdict, two K8 arms read wikitext through both packs loaded by fingerprint: K32 with the fp32 router, K16 with the shipped cast. The registered table reads whether the cast is the whole 6.36709 → 6.33015 gap. The attention pack's fingerprint is compared with P81's `d7cfa1f4…`.
- `tests/test_p82_staged_pin.py` pins the staged files (P81's harness and hook, referenced unchanged), the 21-case rule and K8 table, the router export and the soundness of its stamp, the tripwire markers for #777 and #413, and the reduction's place before the K8 arms.

### Correction: the flagship matrix's energy range and frozen-byte figure each described one model (docs and register only)

- `e4b.train.flagship-matrix` covers Qwen3-30B-A3B and Gemma-4-26B-A4B, but its "0.86–0.92× energy" was Gemma-4's range alone (`bench/flagship-matrix-model2/RESULTS-flagship-matrix-model2.md`, C3: 0.860–0.922×). Qwen3's is 0.797–0.846× (`bench/flagship-matrix/RESULTS-flagship-matrix.md`, B3). The two-model range is **0.80–0.92×**, recomputed from the twenty per-cell receipts.
- Its "bit-identical over 16.31 GB hashed" is the fused-train gate's figure, from one Qwen3 run outside the matrix. The ten Gemma-4 cells' own check hashed 12.85 GB each. The sentence now names both, and the worst parity cell (0.03653) is labelled as Gemma-4 finance's.
- The speed (1.52–1.81×) and VRAM (0.75–0.81×) ranges already spanned both models, and the row's value and status are unchanged. Corrected in `docs/claims.json` (the row's notes keep the old wording, and the model-2 receipt joins its evidence), `docs/STATUS.md` (marked), `README.md` and `docs/solutions/qlora-fused-moe-experts.md`.

## 0.37.8 — 2026-09-29 — a correctness fix to 0.37.7's opt-in bucketed decode graphs: a bucket of one row appends to its own KV slot (#777), and with grouped-nf4-gemm ≥ 0.33.7 the graph path decodes bit-identically to the eager runner (lane B771b); the batched training path is bit-reproducible on CUDA (#765, #776); a calibrate-and-dump build names the dumped expert artifact on its provenance (#772, #773); corrections to the B511, P80 and P81 reads and to the serving-path description (#774)

**0.37.8.** If you call `PagedModelRunner.enable_decode_graphs` (new in 0.37.7), upgrade.
- **The bug.** Whenever exactly one request was active, 0.37.7's graph path appended that request's new K/V to a scratch slot, so the request decoded without the tokens it produced while alone.
- **Who it did not affect.** The package's own serving entry points (`serve.py`, `infer.py`) decode with transformers' `generate` and never used the graph path.
- **Verified.** On an RTX 5090 (lane B771b), with grouped-nf4-gemm ≥ 0.33.7, the graph path now decodes exactly the eager runner's function: identical tokens in all 16 rows of P80's trace, at 4.79–4.81× the eager runner on that host. With an older grouped-nf4-gemm, the fused KV append can still flip about one stored byte in 25 million.
- **Also.** `enable_batched_train` is now bit-reproducible on CUDA (two float-atomic sites fixed). And a calibrate-and-dump build records the dumped expert artifact's root fingerprint beside the live one.
- **Corrections.** P80's and P81's reads attributed their graph-vs-eager divergence to rounding and to the KV append. It was this bug, together with grouped-nf4-gemm 0.33.7's append fix.

### Lane B771b read (#771, #777): the fixed bucketed-graph path decodes exactly as the eager runner and is 4.79–4.81× faster on this host (CONFIRMED); it supersedes P80's row (bench, docs and register only)

- `b771b-5090-1` ran on one RTX 5090 (AMD EPYC 7K62 host) for $0.2475, teardown proven.
- **Tests on sm_120.** e4b's graph tests passed, 10 of them, including #777's invariant test, which checks after every bucketed step that each row's device KV length equals the host count. gnf4 0.33.7's fused-append byte gates passed, 19. The registered mutation, the old single-row routing, failed the invariant test in both modes.
- **Decode.** Over P80's trace, the eager runner with the device grouping, the bucket step and both graph arms decode identical tokens in all 16 rows. B1/A1 = 4.815 and B2/A2 = 4.791; the self-pairs are 1.0048 and 0.9999. **CONFIRMED**, and #771 is answered: two causes, both fixed.
- The ratio is host-bound. The graph step is within 4–10% of P80's, while eager is 2.15–2.34× slower on this Zen 2 host.
- New row `e4b.serve.b771b.qwen3.dynb.graph-buckets.fixed-path.5090.2026-09-29` supersedes P80's. STATUS's P80 entry is rewritten around it. The read is in `bench/b771b/RESULTS-b771b.md`.

### Lane B771b registered (#771, #777): does the fixed bucketed-graph path decode exactly as the eager runner, and what is P80's ratio on it? (bench only; nothing in the wheel changes)

- `bench/b771b/PREREG-b771b.md`. One RTX 5090, grouped-nf4-gemm 0.33.7, e4b at a `main` containing #777.
- **Arm T.** The GPU tests on sm_89+: #777's invariant test (never run before), the replay and routing tests, and gnf4's fused-append byte gates. **Arm M.** The registered mutation restores the old single-row routing; the invariant test must then fail.
- **Decode.** P80's NF4 Qwen3-30B-A3B trace and five arms, plus Ad, the eager runner with the device grouping.
- **Confirmed only if all hold:** every registered test passes and none is skipped; the mutation is caught; Ad ≡ P ≡ B1 ≡ B2 bitwise in every row; and B/A > 1.03 in both pairings. A confirmed read supersedes P80's register row.
- `tests/test_b771b_staged_pin.py` pins the staged files, the 11-case rule, the arm order, and the mutation's target in the shipped shim.

### Lane B771 read (#771): the fused fp8 append is fixed at the byte level, but it was not the whole cause (REFUTED); the rest is a bucket-1 append bug (bench only)

- `b771-5090-1` ran on one RTX 5090 for $0.1515, teardown proven.
- **Byte stage.** Under the shipped kernel, 21 of 5.4×10⁸ stored bytes differed from `quantize_kv_fp8`, on the hardware e4m3 cast. Under grouped-nf4-gemm#413's, 0.
- **Decode stage.** P80's NF4 trace, all arms with the device grouping.
  - The eager control held across the gnf4 swap.
  - With the fixed append, the bucket step still left the eager step, but only in row 0 and only from its first one-row step (token 129). **REFUTED** as registered.
  - That position is what exposed the bucket-1 append bug, fixed separately.
- The read is in `bench/b771/RESULTS-b771.md`, with receipts byte-identical to the store's. One deviation is recorded: the old byte stage ran after the bake, not before any model load.

### Bucketed decode graphs: a bucket of one row now appends to its own slot (a correctness fix to 0.37.7's opt-in `enable_decode_graphs`)

- **The bug.** `enable_decode_graphs` (#757, 0.37.7) initialises graph mode on a scratch slot (`graph_mode_init(seq=scratch[0])`) and binds each bucket's real slots on device. The paged-attention shim sent every single-row decode to `append_graph_t1`, which writes to that init-time slot.
  - So at **bucket 1, whenever exactly one row was active**, each new token's K/V landed in the scratch slot, and the row's own length stopped advancing.
  - Attention reads the row's slot through the bucket selector, so it ran without the tokens generated since the one-row phase began.
  - A captured bucket-1 graph baked the scratch slot in. Replay and the padded eager step shared the bug.
- **Found by lane B771** (#771, RTX 5090, $0.1515). With grouped-nf4-gemm#413's append fix in place, the eager step and the bucket step decoded identical tokens in all 16 rows through the 16-, 8-, 4- and 2-row phases, and left each other at the first one-row step (row 0, token 129) and nowhere else.
  - B771's registered question, whether the fused append was the whole cause, reads **REFUTED**. The append fix holds at the byte level on the hardware cast: 21 differing bytes of 5.4×10⁸ before, 0 after.
- **The fix.** A bound bucket takes the batch append (`append_graph_bt1`, which addresses the bound slot on device) whatever its size. An unbound single-sequence graph keeps `append_graph_t1`, whose `_g_seq` is the sequence.
- **Tests.**
  - `tests/test_bucket1_append_routing.py` pins the routing through the real shim on any machine.
  - `tests/test_decode_graph_buckets.py::test_every_bucket_step_advances_its_rows_own_kv_length` checks, after every bucketed step, that each stepped row's device length equals the host count (sm_89+). The existing replay test could not see this: both of its sides share the shim.
- **Corrections, each marked in place.**
  - B511's read: replay ≡ padded eager holds, but the padded step was wrong at bucket 1.
  - P80's and P81's reads, STATUS and register notes: their "reported, not decisive" divergence was this bug, plus, in P81, the append's.
  - P80's and P81's ratios were measured on the buggy path and are to be superseded by a re-measurement of the fixed one.

### The batched training path gives identical bits call to call on CUDA (#765)

- **What was wrong.** `enable_batched_train`'s forward (`engines/batched.py`) went through CUDA float atomics at two sites, so identical inputs could give different results call to call. Same class as grouped-nf4-gemm#408.
  - The combine summed each token's k expert rows with `index_add_`.
  - The token gather's backward (`index_select` over rows repeated k times) scattered the k gradient copies with `index_add_`.
- **What changed.** Both now write each (token, slot) cell once and reduce over k along a fixed axis, the pattern of `fast.py`'s `_scatter_combine`, kept torch-only because this path must work without grouped-nf4-gemm.
  - The combine scatters by assignment.
  - A small `autograd.Function` keeps the gather's forward (one `index_select`) and makes its backward deterministic.
  - Cost: one `[tokens·k, hidden]` fp32 transient in the forward.
- **Test.** `test_repeated_calls_are_bitwise_identical_forward_and_backward[bf16|fp32]` (CUDA; skips by name on CPU, where `index_add_` is sequential). Top-k 8 over 16 experts, 512 tokens; output, dL/dx and both LoRA gradients bit-identical over 8 repeats.
  - It needs k ≥ 3, since two terms from zero sum the same either way.
  - It needs both compute dtypes, because each hides one site. Under bf16 compute the combine's rows carry 8 significant bits and sum exactly in fp32 whatever the order; fp32 compute exposes the combine.
  - On the A2000: 32 passed with the fix. Restoring the atomic combine fails the fp32 case on the output; restoring `index_select` fails both on dL/dx.

### Lane B771 registered (#771): is the fused fp8 KV append's non-IEEE quotient the whole cause of the eager-vs-bucket-step divergence? (bench only; nothing in the wheel changes)

- `bench/b771/PREREG-b771.md`. One RTX 5090 and two grouped-nf4-gemm cuts in one process tree: the fused append as shipped (v0.33.5) and grouped-nf4-gemm#413's IEEE-rounded quotient.
- **Byte stage.** Under each cut, 5.4×10⁸ values through `fp8_kv_append_bt1` are compared byte for byte with `quantize_kv_fp8`, with the hardware e4m3 cast.
- **Decode stage.** P80's NF4 Qwen3-30B-A3B setup and trace. The eager step and the bucket step run under each cut, all four arms with the device grouping.
- **Confirmed only if all hold:** the fixed kernel writes the reference's bytes; the shipped one does not; the eager arms agree across the swap; the bucket step differs from the eager step under the old cut and equals it under the new.
- The byte harness's layout math is pinned in CI (`tests/test_b771_bytes_harness.py`); the files, rule and arm order in `tests/test_b771_staged_pin.py`.

### Correction: the package's serving entry points do not use `PagedModelRunner` (docs only)

- P81's read (#769) and STATUS said the HTTP shim and `infer` serve the int4 stack eagerly and so leave the graphs' throughput unused. Neither uses `PagedModelRunner`: `serve.py` and `infer.py` both decode with transformers' `generate`. In the shipped package only the bench harnesses drive the paged runner and scheduler.
- So P81's ratio says nothing about those entry points. Graphs-by-default (#770) is a question about `PagedModelRunner`'s own default. Corrected in `bench/p81/RESULTS-p81.md` and `docs/STATUS.md`, each marked.

### A calibrate-and-dump build now records the expert artifact's root fingerprint on its provenance (#772)

- **What was wrong.** `dump_calibrated_artifact` wrote the artifact but left the model's provenance at the live stores' fingerprint, which hashes the tensor payloads only. The artifact's root also covers the identity and assignment payloads, so the two never match. `dump_attn_int4_artifact` then copied the live value into its informative `expert_pack_fingerprint`: P81's attention pack names `c221ab32…`, which no artifact carries (its expert artifact is `0c9955a9…`).
- **What changed.** The dump now attaches `pack_artifact_fingerprint` (the root) beside the unchanged `pack_fingerprint`, so reducers that compare a calibrating run's `pack_fingerprint` keep their meaning. Receipts that merge provenance carry it, and the attention dump prefers it. After a licensed load, `pack_fingerprint` is already the root.
- **Tests.** `test_a_dump_records_the_artifact_root_beside_the_live_fingerprint` and `test_the_manifest_names_the_expert_artifact_not_the_live_hash`. On the A2000 container both pass, CPU-only as well, and both fail with the fix reverted. The pack P81 already wrote is not rewritten: it is a distributed artifact, and its field is informative.

### Lane P81 read (#511, #674): on the int4 serving recipe, bucketed CUDA-graph decode is 13.95–14.29× the eager `PagedModelRunner` on one host whose eager step is host-bound; the recipe's first attention pack loads by fingerprint (docs, register and bench only; nothing in the wheel changes)

- `p81-5090-2` ran on one RTX 5090 (AMD EPYC 7K62 host) for $1.1349, teardown proven. The lane cost $1.2057: the proof, one NOT_RUN on the launcher's bandwidth pre-flight, and the reading.
- Aggregate decode was A1 47.8, B1 683.6, B2 679.3 and A2 48.7 tok/s, with P (the bucket step, eager) at 57.7. B1/A1 = 14.29 and B2/A2 = 13.95; the self-pairs were 1.0175 and 0.9937. The graph streams equal P's bitwise, and every bucket captured. **CONFIRMED** (`e4b.serve.p81.qwen3.licensed-int4.dynb.graph-buckets.5090.2026-09-29`).
- The ratio is host cost. Eager decode took 125–139 ms per step at every row count, and this host's CPU is slow (the build's first calibration chunk ran 3.1× slower than P70's). Graphs took 16.9 ms at 16 rows and 5.4 ms at one. The ratio is not portable to another host.
- #674: both packs of one build (expert `0c9955a9…`, the licensed one; attention `d7cfa1f4…`, 192 projections) loaded by fingerprint in all five arms. The build's K8 read 6.33015 against the licensed build's 6.36709 on the same window, so the attention pack is the recipe's on e4b 0.37.7, not a byte-identification of P55x's. The cause is not isolated. The read is in `bench/p81/RESULTS-p81.md`.

### Correction: P80's read misattributed two things to padding (docs and register only)

- P80's trace pads no rows: its active sizes are exactly the bucket sizes, and `pad_rows` is 0 in every bucket of its receipts. Two statements depended on padding that never happened.
  - The first explained the graph-vs-eager token divergence by "other row counts" through bf16 GEMMs. A and P ran the same rows. They differ in the grouping and in the step path (KV append), and P81 found them differing even with the grouping held equal.
  - The second attributed P/A1 = 1.33 to "padding plus device grouping". It is device grouping plus the bucket step's path.
- Corrected in `bench/p80/RESULTS-p80.md`, `docs/STATUS.md` and the P80 row's notes. P80's verdict is unchanged: its gate was graph ≡ P, which held. The 0.37.7 section below keeps the text as released.

### Lane P81 registered (#511, #674): bucketed CUDA-graph decode on the licensed int4 stack, and the licensed attention pack by fingerprint (bench only; nothing in the wheel changes)

- `bench/p81/PREREG-p81.md` runs P80's trace, arms and rule on the configuration actually quoted for serving: calibrated int4 experts and attention, the T=1 folds and the fused router epilogue, on Qwen3-30B-A3B. P80 read NF4 only, and named licensed int4 as the case it could not speak to.
- One build (P70's recipe) dumps both packs: the experts (#405) and, for the first time on the licensed stack, the calibrated attention (#754). Every arm then installs both by fingerprint, and a reducer gate voids the read unless all five arms served the same two packs from their artifacts. The expert pack's equality with the licensed `0c9955a9…` and the build's K8 against 6.36709 are recorded, not decided.
- One departure from P80, with the reason: the eager control also runs the device grouping. With the int4 store and the grouping off, a T > 1 decode takes `_fused_over_stack`'s prefill branch (dequantise and matmul per routed expert), which is not how the licensed stack serves a batch.
- The lane copy of `step_decomp.py` is P80's plus the receipt provenance and `--dynb-grouping`. `tests/test_p81_staged_pin.py` pins the staged files, the copy's diff against P80's, both packs in the arms' load env, the trace and the reducer's 13-case rule.

### Kimi-K3's run-to-run drift is the MXFP4 prefill combine's atomics (bench and docs only; no library change)

- **The A/B ran.** Nine prefill-only processes ran on the A2000 through a wrapper that sets torch's deterministic-algorithms switch and hashes every MoE call's input, router ids, router weights and output. The driver is unmodified. With `torch.use_deterministic_algorithms(True)`, three processes are bit-identical at all 92 MoE calls: p(" Paris") 0.7144126892089844, 6,118 prefill expert rows. Without it, three processes all give 0.7119670510292053 and 6,130 rows, yet every pair differs at 3 to 8 of the 92 calls, each time with the engine returning different bits for identical inputs. With grouped-nf4-gemm#410's ordered combine alone, deterministic mode off, three processes reproduce the deterministic ones bit for bit. So the drift recorded in #761 is grouped-nf4-gemm's `out.index_add_` float atomics; in these nine processes nothing else in the forward moved. Row `e4b.parity.kimi-k3.prefill-drift-is-the-combine.a2000.2026-09-28`. The five-runs row's notes and `docs/STATUS.md` now say measured, not suspected. RESULTS §5; receipts `bench/kimi-k3-a2000/receipts/2026-09-28-det-ab/`.
- **Confirmed on the release.** grouped-nf4-gemm 0.33.6 ships the fix. Three Kimi-K3 prefill processes on it, with deterministic mode off and nothing shadowed, give 0.7144126892089844 and 6,118 rows, bit-identical at all 92 MoE calls to each other and to the deterministic runs. Row `e4b.parity.kimi-k3.reproducible-on-gnf4-0.33.6.a2000.2026-09-29`, RESULTS §6, receipts `bench/kimi-k3-a2000/receipts/2026-09-29-rel0336/`.
- **Same pattern elsewhere, not changed here.** `engines/batched.py`'s batched-training combine still uses `index_add_` over repeated token rows (#765). The NF4 serving paths (`fast.py`, `hot_residency.py`, `nvme_experts.py`) already land unique (token, slot) cells and sum once.

### Kimi-K3 on the A2000, same day: an armed cache gate, a Fireworks reference, and run-to-run drift (bench and docs only; no library change)

- **The cache gate replays routing.** Each MoE call's (expert ids, weights) is recorded along the cached path and replayed into the fresh cache-free prefill, so the logits comparison measures only the kernels and the cache. Under replay, the real cache reads cos 0.999966 (PASS against cos ≥ 0.9999 and argmax agreeing, fixed before the run). A cache with all 69 KDA recurrent states zeroed reads 0.877523 (FAIL), with argmax still agreeing, so an argmax-only check would have passed it. With free routing the same comparison reads 0.998699: 90 of 92 layers route differently somewhere in the 9 positions. Row `e4b.parity.kimi-k3.cache-gate.replayed-routing.a2000.2026-09-28`.
- **A reference ran.** Fireworks' serverless `kimi-k3` (native MXFP4 weights, MXFP8 activations) scored the same raw paragraph through its completions endpoint with `echo`. The A2000 run's per-token NLL tracks it at Pearson r = 0.9965 over the same 89 predictions (median |Δ| 0.038 nats; perplexity 3.196 vs 3.150). Both complete the prompt as " Paris. It is" (p(" Paris") 69.26 % there). Row `e4b.quality.kimi-k3.vs-fireworks.per-token.a2000.2026-09-28`.
- **The first run's 270.5 s prefill was Triton compilation.** 183 kernels compiled inside it, in a fresh cache. Warm, the prefill takes 179.0 s, and a repeat in the same process 176.9 s, against July's 178.3 s. Warm runs peak at 4.07 GB of VRAM, not 4.32 GB.
- **The forward is not reproducible run to run.** Five processes of one build gave p(" Paris") 68.90 %, 71.59 %, 71.29 %, 71.20 % and 71.20 %, with 6,115 to 6,131 prefill expert rows. Autotuning is ruled out after the first run: the ten autotuned fla kernels persisted their choices, and a run with `TRITON_PRINT_AUTOTUNING=1` benchmarked nothing. The suspect, not confirmed, is the MXFP4 prefill combine's `out.index_add_` (grouped-nf4-gemm `mxfp4_pipelined.py`), whose float atomics can reorder. So `e4b.offload.kimi-k3.full-depth.a2000.2026-09-28` is superseded by `e4b.offload.kimi-k3.full-depth.a2000.five-runs.2026-09-28`, which quotes the ranges.
- **Receipts.** `bench/kimi-k3-a2000/receipts/2026-09-28-{gate,reference,determinism}/`, each with the driver that produced it (SHA256SUMS).

## 0.37.7 — 2026-09-28 — bucketed CUDA-graph decode in `PagedModelRunner` (`enable_decode_graphs`, opt-in; #511): each step pads the active set to a bucket with scratch KV slots and replays that bucket's graph. It is verified bitwise against the padded eager step on an RTX 5090 (lane B511) and measured at 2.32–2.34× the eager runner on NF4 Qwen3-30B-A3B over a changing active set (lane P80). `dense_offload_report`'s `all_pinned` is `None`, not a vacuous `True`, when no layer is host-resident (#759). Kimi-K3 at full depth on one A2000 re-run on released packages

### Lane P80 read (#511): bucketed CUDA-graph decode is 2.32–2.34× the eager `PagedModelRunner` on a changing active set, NF4 Qwen3-30B-A3B (docs, register and bench only; nothing in the wheel changes)

- `p80-5090-1` ran on one RTX 5090 for $0.157, teardown proven. The lane cost $0.2513: one proof NOT_RUN on bandwidth, one refused at $0 by my error, one proved.
- Over the registered trace (16 → 8 → 4 → 2 → 1 active rows, 32 decode steps each), aggregate decode was A1 121.6, B1 284.5, B2 284.5 and A2 122.5 tok/s, with P (padded eager) at 162.1. B1/A1 = 2.340 and B2/A2 = 2.323, above the 1.03 bar; the self-pairs were 1.0076 and 1.0000. The graph streams equal the padded eager step's, bitwise. **CONFIRMED** (`e4b.serve.p80.qwen3.dynb.graph-buckets.5090.2026-09-28`).
- Per step, eager ran 54.5 → 42.3 ms from 16 rows to one, and graphs 33.8 → 9.95 ms. P splits the gain: 1.33× from padding plus device grouping, 1.75× from the graph itself. The read is in `bench/p80/RESULTS-p80.md`.

### Lane P80 registered (#511): bucketed CUDA-graph decode against the eager `PagedModelRunner` on a changing active set, NF4 Qwen3-30B-A3B (bench only; nothing in the wheel changes)

- `bench/p80/PREREG-p80.md` adopts the Q2 memo's registered experiment. The active set steps 16 → 8 → 4 → 2 → 1 (32 decode steps each). The arms run in the fixed order A1 → B1 → B2 → A2, plus P, the padded-eager oracle. The claim is confirmed only if B/A > 1.03 in both pairings. It departs from the memo in one place, with the reason given: the bitwise gate is graph ≡ padded eager, not graph ≡ unpadded eager (bf16 GEMMs vary with row count).
- `bench/p80/step_decomp.py` is a lane copy of the harness plus the `--dynb-mode` stage (closed lanes pin the live file). A test asserts the copy removes no live line except the `Fp8PagedKV` call it extends. `p80_reduce.py` computes the verdict; its nine-case self-test runs in CI and on the box. The runner and driver are derived from P70's. The proving path was rehearsed on the A2000; the arms need sm_89+.

### Bucketed CUDA-graph decode in `PagedModelRunner`, opt-in (#511; new API, default behaviour unchanged)

- **What.** `PagedModelRunner.enable_decode_graphs(buckets=(1, 2, 4, 8, 16))` captures one decode graph per batch bucket. Each decode step pads its active set to the next bucket, replays that bucket's graph and discards the padded rows. An active set larger than the largest bucket runs as several chunks. `Fp8PagedKV(..., scratch_slots=N)` adds N padding slots with one KV block each, which the scheduler must never be given (`bind` refuses them). A bucket's device tensors (slot ids, attention selector) persist and are rewritten in place each step, so a replay reads the current active set rather than the one present at capture.
- **Capture never touches a live sequence.** Each bucket is warmed up and captured on scratch slots only. A bucket whose capture raises runs the same padded step eagerly and prints one `DECODE_GRAPH bucket=N EAGER (...)` line. `graph_status` and `graph_stats` (replays, eager steps, rows, pad rows) record what ran. `capture=False` runs every bucket eagerly on the padded layout; that is the bitwise oracle for the replays.
- **Verified on silicon (lane B511, `bench/b511/RESULTS-b511.md`).** On an RTX 5090 (sm_120), `tests/test_decode_graph_buckets.py`'s replay test passed: buckets 1, 2 and 4 captured, and the replayed token streams equal the padded eager step's exactly, over an active set shrinking 4 → 3 → 2 → 1 with a recycled slot. Under the registered mutation (attention reading the per-tuple selector, which a graph bakes at capture) the same test fails. Arm A: 102 passed, 0 skipped. Cost $0.0666. The fp8 paged KV needs native e4m3 (sm_89+), so on an sm_86 card the test skips by name. (#757)

### Kimi-K3 at full depth on one 12 GB RTX A2000, on the released packages (bench and docs only; no library change)

- **The receipt.** `bench/kimi-k3-a2000/` re-runs the 2026-07-30 full-depth driver on 0.37.5 and grouped-nf4-gemm 0.33.4 from PyPI. All 93 layers run on real weights, on the NAS's RTX A2000, at 4.32 GB peak VRAM. The MXFP4 experts stream from the 1.446 TB SSD arena through 92 `Mxfp4NvmeResidencyK3` engines that share one 281 MB slot store. `enable_dense_offload` serves the 108.76 GB dense side from safetensors byte offsets, with 0 bytes pinned in host RAM. The model greedily completes "The capital city of France is" as " Paris. It is", with p(" Paris") = 68.90 %, at a median 92.4 s per decode token. Perplexity is 3.181 on one 90-token paragraph. Register row `e4b.offload.kimi-k3.full-depth.a2000.2026-09-28`.
- **Why a re-run.** The July run used a pre-0.9.0 e4b at an unrecorded commit and a loose, non-git copy of the kernel files. Its JSONs carry no provenance, and a later 1-step run overwrote its 4-token JSON. The new driver differs only in import paths, output path and a `provenance` block (package versions, driver sha256). The third-party stack is pinned to July's: torch 2.8.0, transformers 4.57.6, fla-core 0.5.2. The token ids are identical to July's, and the probabilities moved by up to 2.3 points. The prefill read 6,115 expert rows against July's 6,130, so the two builds route differently at near-ties. The July driver, logs and JSONs are kept under `receipts/2026-07-30/` as history.
- **The cache warning in the log is not a finding.** Cached decode and a fresh prefill of the same sequence select a different expert set in 45 of 92 layers (50 of 1,472 slots). So the cache check's cos 0.999408, with argmax agreeing, comes from discrete routing flips. The driver's `cos < 0.9999` gate was never calibrated.

### `dense_offload_report` no longer says `all_pinned: true` about a run that pins nothing (a report field's value; no runtime change)

- **The defect.** `all_pinned` was `all(h.pinned for h in handles if h.host_bytes)`. When every dense home is served from disk, that generator is empty and `all()` answers `True`. The code's own comment already called this out, but the code still did it. The 2026-09-28 Kimi-K3 receipt (`bench/kimi-k3-a2000/receipts/2026-09-28/k3_gen.log`) printed `"host_bytes": 0, "host_resident_layers": 0, "all_pinned": true`. That receipt stays as recorded.
- **The change.** `all_pinned` is `True` or `False` when `host_resident_layers > 0`, and `None` (JSON `null`) when it is 0 or there are no handles. `False` would claim that some host layer failed to pin, and there is none. A layer's own `pinned` flag, which the report and the handle's repr read, is `None` for a disk-only layer for the same reason. Host-path runs report exactly as before. On a disk-only run, a reader that treated the value as truthy now reads it as falsy. `False` still means only a real pinning failure.
- **Tests** (`tests/test_dense_disk.py`). A CPU test with only disk-served handles asserts `None` in the report, in its JSON form, per layer and for an empty handle list. On the unfixed code it fails. The host-path report test now also asserts `False` for pageable homes.

## 0.37.6 — 2026-09-28 — the calibrated int4 attention can be pinned by bytes: `dump_attn_int4_artifact` writes a hash-pinned attention pack and `enable_serve_attn_int4_from_artifact` installs one by fingerprint without recalibrating (#674; new functions and `E4B_SERVE_ATTN_INT4_{DUMP,ARTIFACT,FINGERPRINT}`, default behaviour unchanged); `enable_fast` refuses CUDA-graph capture by name, and capture is tested on a quantised MoE for both engines (#527)

### The calibrated int4 attention can be pinned by bytes: a hash-pinned attention pack (#674; new functions and env switches, default behaviour unchanged)

- **The gap.** `engines/int4_attn_calib.py` had no serialisation. The calibrated attention projections were re-derived from a Hessian pass on every load, on whatever box loaded them, so a `pack_fingerprint` identified only the expert half of the licensed Qwen3 stack. P55x measured the attention half as the one carrying the quality: RTN attention fails c4val1 at +0.13237.
- **Dump.** `dump_attn_int4_artifact(model, dir)` writes every live int4 projection (attention, and the output head or dense MLPs when packed) as a sibling of the expert pack. It uses the expert pack's hashing, hashed identity payload and verifier under its own layout, `int4_b32.attn.v1`, so the two packs cannot be confused. The payload path carries the module name, so the root fingerprint covers which module each set of bytes belongs to. Per projection it records N, K, whether a bias is carried (the bias bytes ride in the pack) and whether the projection was calibrated. It refuses fused q/k/v modules (dump before `qkv_fuse`), and it refuses an unknown checkpoint revision unless asked, as the expert dump does.
- **Licensed load.** `enable_serve_attn_int4_from_artifact(model, dir, expected_fingerprint=…)` builds every projection from the pack's bytes with `Int4Linear.from_packed`, so nothing is calibrated or re-quantised. It refuses, and never falls back to calibrating, on: a fingerprint, layout or revision mismatch; a corrupt payload; an edited manifest; a live model without `config._commit_hash`; a projection set other than the one the enable flags select; or a shape or bias disagreement. Every projection is validated before any is swapped, so a refused load leaves the model untouched.
- **Env.** `enable_from_env` gains `E4B_SERVE_ATTN_INT4_DUMP=<dir>` (write after calibrating) and `E4B_SERVE_ATTN_INT4_ARTIFACT=<dir>` with `E4B_SERVE_ATTN_INT4_FINGERPRINT` (load instead of calibrating). An artifact without its fingerprint, or without a flag naming its groups, refuses. Receipts built with `merge_provenance_into_receipt` carry `attn_pack_fingerprint` beside the expert `pack_fingerprint`.
- **Tests** (`tests/test_int4_attn_pack.py`). 14 tests: 13 on CPU with the reference-kernel stubs, and one on the real GPTQ packer and int4 kernels. On an RTX A2000 the loaded pack serves the calibrated function **bitwise** at 1, 4 and 24 rows. A mutation that makes the load re-quantise the live weights fails 4 of them.
- **Not yet done.** No pack has been dumped for the licensed Qwen3-30B-A3B configuration. That needs a card that holds the model, and a reading of whether the attention pack reproduces across boxes, which #674 leaves open. Until then the register's statement that only the expert half is artifact-backed stands. (#754)

### `enable_fast` refuses CUDA-graph capture by name; capture is tested on a quantised MoE for both engines (#527; a clearer error, no numeric change)

- **The refusal.** `enable_fast`'s grouped expert path sizes its launch from host-side per-expert counts (`counts[active].tolist()`), which is a host sync and a data-dependent launch, so it cannot be captured. The reference path it falls back to cannot be captured either. Under capture, both grouped inference forwards (`fused_experts_forward`, `fused_experts_lora_forward`) now raise a `RuntimeError` that names `enable_pipelined_residency` as the capturable engine. Before, the failure was `operation failed due to a previous error during capture` from inside `torch.bincount`. Outside capture, nothing changes.
- **The tests** (`tests/test_capture_quantized_moe.py`). `test_capture.py` built every GPU case from a dense tiny Llama, so no capture test had exercised a quantised MoE or either engine. A random 2-layer Qwen3-MoE is loaded through `load_moe_4bit_streaming`. Under `enable_pipelined_residency` (K = 0), `probe_capture` must capture and replay token-for-token as eager. Under `enable_fast` it must refuse by name and still decode eagerly. Two CPU tests pin the refusal's decision and its placement ahead of every fallback in both forwards.
- **RTX A2000 (sm_86)**: the new file plus `test_capture.py` and `test_fast_lora.py`, 19 passed. With the refusal calls removed, the named-refusal test fails on the old CUDA error and the placement test fails, while the pipelined capture still passes. (#753)

## 0.37.5 — 2026-09-28 — a routing fix and a default change under the opt-in fused router epilogue (`E4B_FUSE_ROUTER_EPI=1`): the gpt-oss router now adds its bias inside the GEMM as upstream does (on real bf16 gpt-oss-20b weights the old path licensed 16 of 24 layers and could pick a different expert set), and `softmax_topk` routers (Qwen3-MoE, OLMoE, Mixtral) return the model's dtype by default (lane P70 read the cast INDISTINGUISHABLE; `E4B_ROUTER_EPI_CAST=0` restores fp32 for one release). `requires-python` is raised to `>=3.10`, which is what the package already needed. Lanes P69 and P70 read; the direct cold landing is tested end to end (#178 closed); documentation corrections; trove classifiers

### `requires-python` is `>=3.10`, which is what the package already needed (packaging metadata; a floor changes)

- **The defect.** The wheel declared `>=3.9`, but `util.py` and `formats/fp8_blocks.py` use `X | None` annotations without `from __future__ import annotations`, which raise `TypeError` at import on 3.9. `util.py` loads with the package, so on 3.9 the package failed at import. The `[train]` and `[serve]` extras need `transformers>=5.0`, which itself requires Python 3.10, and current bitsandbytes does too. Found in the 2026-09-24 documentation review (#749, "left for the maintainer").
- **The change.** `requires-python = ">=3.10"`. On 3.9, pip now refuses the install up front, where before the import failed. Every module in `experts4bit_qlora/` parses under the 3.10 grammar (`ast.parse(..., feature_version=(3, 10))`, 66 of 66). CI runs 3.11 only, so 3.10 is the declared floor, not a tested one. `docs/capabilities.json` and one solution page state the new floor. (#751)

### The fused gpt-oss router adds its bias inside the GEMM, as upstream does; bf16 routers are licensed before fusing (a routing fix under `E4B_FUSE_ROUTER_EPI=1`)

- **The defect.** On the `topk_softmax` kind, the fused path and the probe's reference both computed the logits as a bf16 GEMM **without** the bias, then added the bias in fp32. `GptOssTopKRouter` (transformers 5.16.1) calls `F.linear(x, weight, bias)`, so its logits round to bf16 **with** the bias already added. At bf16 those are different functions. They can select a different expert **set** on a near-tie.
- **Measured on gpt-oss-20b's real router weights** (all 24 layers, bf16, the pinned `GptOssTopKRouter`). The licensing probe passed 16 of 24 layers at its fixed seed. So `E4B_FUSE_ROUTER_EPI=1` patched the model only partly and did not report it. The probe feeds 4 random N(0,1) rows. Across 50 probe seeds × 24 layers, 791 of 1200 probes passed. 263 failed the weight check, 100 failed on slot order, and **46 on the expert set**. Those 46 are rows where the fused function and upstream pick different experts. That is mis-routing, which the fused path would do at decode on such a row, not a rounding difference. GraniteMoe (granite-3.0-1b-a400m, no bias) passed 1200 of 1200 before and after.
- **The fix.** `_topk_softmax_logits` forms the logits as `F.linear(rows, weight, bias)` in the module's dtype, in both the fused forward and the probe's reference. The kernel is called with no bias. After the fix, both families pass 1200 of 1200 at bf16 and fp32, and all 24 gpt-oss layers are licensed. The probe's `rtol=2**-8` is unchanged: it is bf16's half-ulp, and the upstream k-softmax rounded to bf16 falls inside it by construction. The expert set and order are still required to match exactly. The probe's "raw" first slot now means exactly what the fused path returns. So a biased router that returned the *un*biased projection would be refused, not handed biased logits. No family does this.
- **Unchanged.** `E4B_ROUTER_EPI_CAST`, `CAST_WEIGHTS` and `_cast_for` are unchanged. `topk_softmax` still returns fp32 weights by default, pending its read. The `softmax_topk` and Gemma-4 branches are unchanged.
- **Tests** (`tests/test_router_epilogue.py`). The toy routers and the pinned transformers' own `GptOssTopKRouter` and `GraniteMoeTopKRouter` are built in bf16 **before** fusing. Each must be licensed and must select exactly the module's own experts on 32 draws of 64 decode rows. The existing default-cast test now fuses in bf16, where it used to fuse in fp32 to dodge the probe. One test pins the late-bias mechanism; another pins the first-slot refusal. The new tests fail on both mutations: the old module, and a late-bias fused path behind the fixed probe.
- **Receipts** (`bench/router-probe-bf16/`). `probe_real.py` runs the probe over the real router weights. `extract_router.py` pulls those tensors from the checkpoints using only the stdlib. `before.txt` and `after.txt` hold the counts above. (#748)

### The direct cold landing is tested end to end; #178 closed (tests only, no library change)

- `tests/test_hybrid_cold_dest.py`'s only end-to-end fixture (`INTER=64, H=128`) cannot scatter at `align=4096`, so every cold-CPU equivalence test ran the copy landing and the direct (preadv-scatter) landing never moved a byte in a test. #178 (`enable_hybrid_tier(cold_dest="cpu")` raised at enable on arenas that scatter) had been fixed by #177's separate setup tier, with nothing to hold the fix.
- A second fixture at `INTER=256, H=512` scatters. A CPU test calibrates it: the selector picks `direct-scatter` there and `copy` on the old geometry. A GPU test enables `cold_dest="cpu"` with `cold_direct` True and False, and asserts the landing names and **bitwise** equality with each other and with a DRAM placement. The old equivalence test now asserts that its own landing is `copy`.
- RTX A2000 (sm_86): 37 passed, 0 skipped. With the setup reads routed back through the serving tier (the #178 condition), the new test fails with #178's exact `RuntimeError`, while the old fixture's test still passes. (#750)

### Documentation corrections from the 2026-09-24 cross-document review (docs, `capabilities.json` prose, one docstring; no library behaviour change)

- #743 applied 85 verified findings across README, STATUS, CHOOSING, AGENTS, SECURITY, CONTRIBUTING, INDEX and 20 other pages: terms defined on first use, the quickstart names `[train]`, the vLLM bullet describes P58, dead links and pins fixed. No claim value, status or unit moved.
- #749 applied the rest. The "no shipped tool bakes the training arena" wording was false: grouped-nf4-gemm's `nvme_bake_nf4` writes exactly the four segments `enable_nvme_train_residency` stages. The offload, MXFP4 and training pages, README, STATUS and `capabilities.json` now describe the bake. The training page records Gemma-4's attention-4-bit support (lane P67) and the 4.490× tp4 position.

### Package metadata: trove classifiers (no library change)

- `pyproject.toml` declares classifiers for the first time: NVIDIA CUDA, Linux, Python 3 only, the AI topic, typed (`py.typed` already ships), and developer and research audiences. There is no `License ::` classifier, because the licence is a PEP 639 expression. PyPI shows them from the next upload.

### The fused router now returns the model's dtype by default for `softmax_topk` (#726; a default changes)

- **What changes.** With `E4B_FUSE_ROUTER_EPI=1`, the fused router epilogue on the `softmax_topk` kind (Qwen3-MoE, OLMoE, Mixtral) now casts its top-k weights to the router logits' dtype, which is bf16 on a bf16 model. That is what the upstream router returns above 64 rows, so a decode or verify step and a prefill now weight each expert with one function.
- **Why.** Lane P70 read the cast as INDISTINGUISHABLE at B = 1 decode on Qwen3-30B-A3B (`e4b.serve.p70.qwen3.b1.router-weight-cast.5090.2026-09-25`). Its registered decision rule makes it the default.
- **The `topk_softmax` kind is unchanged.** gpt-oss and GraniteMoe keep the kernel's fp32 weights until that kind is read.
- **`E4B_ROUTER_EPI_CAST`.** Unset means the per-kind default above. `=0` restores fp32 weights for every kind, as before, **for one release**. `=1` also casts `topk_softmax`, which is unread. Any other value is refused with a `ValueError` at import.
- **Harnesses.** No other lane reads the switch. P70's own runner refuses on this commit instead of silently measuring the new default, because its tripwire requires the cast off at import. The runner also unsets the variable, so rerunning P70 means running it at its registered commit, `c77aab6`.

### Lane P70 read (#726): the fused router's weights cast to bf16 is INDISTINGUISHABLE at B = 1 decode (docs and bench only; the default flips in a follow-up)

- `p70-5090-2` ran on one RTX 5090 with the licensed pack `0c9955a9…` (build K8 ppl 6.36709) for $0.6925. The lane cost $0.8156 over six runs. Validity was VALID.
- KL(cast off ‖ cast on) was **0.003309** nats/token on wikitext and **0.002806** on c4val1: 0.60× and 0.92× of the in-lane floor. dNLL was +0.00096 / +0.00216, and both intervals span 0. Verdict: **INDISTINGUISHABLE**.
- P1–P4 and P6 held. P5 (|dNLL| ≤ F_NLL) **broke on c4val1**, 0.00216 against 0.00180. It moves no class.
- Proof `p70-prove-4`, on the RTX 5090's kernel: cast-on weights were bit-equal to upstream's at matching row counts, with the same experts. Only the top-k slot order differs.
- Register row: `e4b.serve.p70.qwen3.b1.router-weight-cast.5090.2026-09-25`. The read is in `bench/p70/RESULTS-p70.md`.
- By the registered rule, the cast becomes the default for the `softmax_topk` kind in the next release, with `E4B_ROUTER_EPI_CAST=0` restoring fp32 for one release.

### `E4B_ROUTER_EPI_CAST`: the fused router's weights in the model's dtype, as a switch; lane P70 registered (#726) (one default-off switch in the wheel; bench tooling)

- **The discontinuity (P63's P7, confirmed from the pinned transformers 5.16.1 / 5.17.0 source).** With `E4B_FUSE_ROUTER_EPI=1`, a patched router returns the fused kernel's **fp32** routing weights at ≤ 64 rows (every decode and verify step) and the model's own router returns **bf16** above 64 rows (`router_top_value.to(router_logits.dtype)`): two functions of the same logits by row count.
- **The switch.** `router_epilogue.CAST_WEIGHTS` (env `E4B_ROUTER_EPI_CAST=1`, **default off**, read on every call) makes the fused `softmax_topk` and `topk_softmax` branches return `w.to(logits.dtype)`. On `softmax_topk` (Qwen3-MoE, OLMoE, Mixtral) that is the upstream function to the bit, at every row count; on `topk_softmax` the dtype matches but not every bit. Five tests in `tests/test_router_epilogue.py`. Nothing moves a default.
- **Correction (2026-09-25, P70 amendment 1).** "To the bit" above covers the routing as a map, expert to weight, at the same row count as upstream. It does not cover the top-k slot order, which can differ from `torch.topk`'s where two probabilities tie within fp32 rounding. The proving rental `p70-prove-2` failed its cast check (rc 27, $0.0345) on an instrument defect: it compared fused 1- and 64-row calls with rows of an 80-row upstream call, and a bf16 GEMM's logits change with the row count. A free NAS A2000 diagnosis (`bench/p70/diag-a2000/`) separated the two. The corrected proof compares at matching row counts and as maps; see the amendment in `bench/p70/P70-PREREG.md`.
- **Lane P70 registered** (`bench/p70/P70-PREREG.md`): P64's served Qwen3 int4 stack and scorer (imported unchanged, pins equal to P64's), scored at B = 1 with the cast off and on, against a floor whose two samples keep the router function fixed in the prefill. A router-call census by row class and returned dtype proves what each pass ran. Disclosed: P64's `a8_pc64` floor sample straddled this very switch (a 64-row prefill runs the fused router); P70 measures that sample beside its own floor, informationally. Decision rule: INDISTINGUISHABLE or not material → the cast becomes the default for the `softmax_topk` kind in the next release; MATERIAL → the default stays and the > 64-row path is lifted to fp32 instead.

### Lane P69 read (for grouped-nf4-gemm#400): the link-efficiency factor is per host, not a 5090-class constant (docs only; nothing in the wheel changes)

- `p69-5090-2` ran grouped-nf4-gemm's `bench/calibrate.py` (schema /2) twice on one gen 4 × 16 RTX 5090 (host EPYC 7663) for $0.0146: back-to-back 20.58 / 20.72 GB/s, single copy 17.97 / 19.90 → `link_eff` **0.873 / 0.960**. The registered [0.55, 0.75] (P66's other 5090 host: 0.638) is **REFUTED**; repeatability held at 9.7 %. The two hosts' difference is stated, not explained. Receipt and claim in grouped-nf4-gemm (`bench/cold-engine/calib-5090-p69/`, `gnf4.calib.link-efficiency.5090.2026-09-24`, PR #403); the read here: `bench/p69/RESULTS-p69.md`. No default moves.

### Lane P69 registered: does grouped-nf4-gemm's single-copy link probe read the gen-4 gather rate P66 implied? (for grouped-nf4-gemm#400; bench only, nothing in the wheel changes)

- **The question.** P66 found the pipelined gather running at the box's single-copy H2D rate (14.72 GB/s probed, 14.44 implied) while `cold_deadline` was given the back-to-back rate (23.07), so its transfer term under-predicted the gather 1.57–2.02× on a gen 4 × 16 RTX 5090. grouped-nf4-gemm#402 moves that probe into `bench/calibrate.py` (schema `gnf4-hybrid-calib/2`) and carries the ratio as `Costs.link_eff`. P69 runs that script, at the pinned commit, twice on a 5090 of the class the consumer serves, and reads the ratio. Prereg: `bench/p69/P69-PREREG.md`.
- **Predictions, written before the box.** `link_eff` ∈ [0.55, 0.75] (P1); back-to-back 20–30 GB/s (P2); single 12–18 GB/s (P3); the two runs within 10 % (P4). Disclosed: the NAS A2000 (gen 3 × 8) read single 5.67 vs back-to-back 5.46 → 1.0, the gen-3 control.
- **The instrument.** `bench/p69/p69_drive.sh`, controller side like `p56_prove.sh`: refuses outside the launcher's environment (78) and any card that is not an RTX 5090 (15), fetches `calibrate.py` at `GNF4_SHA` with its sha256 recorded, stages it, runs it twice with `--skip-cpu`, fetches both blobs and prints the read. Nothing installed on the box. `tests/test_p69_drive.py` holds the refusals, the exit codes and the probe fields.
- **Cost.** ≤ $0.75/h at a 0.25 h guard ($0.1875 line), no proving rental (the guard is under an hour). Decision rule: whatever P1 reads, both blobs become grouped-nf4-gemm's 5090 calibration receipt and one claim (`gnf4.calib.link-efficiency.5090.<date>`); nothing in either model is tuned to the outcome.

## 0.37.4 — 2026-09-24 — documentation, register data, evidence and repository tooling; under the package one default-off switch: `E4B_INT4_DECODE_A16` (lane P64's eager-only quality instrument, with its CUDA-graph-capture guard in `engines/hot_residency.py`). Six lanes read on rented RTX 5090s — P63 (row-exact expert routes), P64 (the int8 activation step is indistinguishable), P65 (Granite's selector is TWO_ARMS; Colla-Q replicates), P66 (residency's fixed launch count; cold_deadline under-predicts the gen-4 gather), P67 (Gemma-4's training-parity floor; attention-4-bit training supported under it), P68 (a verify from T = 1 calls is bit-identical; T > 1 inside the bar) — with their registrations and amendments; 13 claims added, P56's proxy floor superseded; `docs/ARCHITECTURE_SUPPORT.md` leaves the llms bundle

### Lane P67 read (#713): Gemma-4's training-parity floor is measured; the attention-4-bit paths are supported under it (docs, register, capabilities; no library behaviour change)

- **The draw.** `p67-gemma4-floor-1` ran on one RTX 5090 for $1.4426 (lane $1.4523 with the proof), every arm OK · VALID, both teardowns proven. Full read: `bench/p67/RESULTS-p67.md`; the box's receipts under `bench/p67/receipts/`.
- **The floor.** Five admissible draws of the reference against itself (one plain repeat, four fixed permutations of the per-expert loop): **F_hi(final) 0.121, F_hi(med) 0.1275**, bands 0.363 / 0.383. The plain repeat is not bit-identical (diverges at step 2). Registered as `e4b.train.p67.gemma4.attn4-reorder-floor.5090.2026-09-24`, superseding P56's proxy `e4b.parity.gemma4.train-floor`.
- **The verdicts.** `fused_attn4` D 0.013 / 0.036 and `batched_attn4` D 0.037 / 0.048: both **PASS**, carried by the tolerance, neither detectable against the floor; the two earlier sessions' constant-band FAILs PASS the floor band, so the reading is not MIXED. `docs/capabilities.json`: `gemma4_text.fast_train` and `reference_train` in the attention-4-bit configuration are **supported**; `batched_train` stays void (the arm ran at pad_waste_limit 64, not the shipped default).
- **Predictions.** Q1, Q2 held. Q3, Q4, Q6 refuted and Q5 mostly refuted, all in the direction of noise: the floor is larger than predicted, the loop reorder moves the step-0 loss more than the fused kernels do (0.052–0.217 vs 0.034), and the accelerated arms landed closer than their standing values. No default moves; no speed claim.

### `llms-full.txt`: `docs/ARCHITECTURE_SUPPORT.md` leaves the bundle (discoverability only; no library change)

- The bundle stood at 397,928 of its 400,000-byte cap after the P66 read, and every lane read adds a STATUS paragraph and claims. The architecture-support table (27.7 KB) is dropped from `docs/llms-bundle.json`; it stays linked from `llms.txt`, the README and `docs/INDEX.md`, so nothing becomes undiscoverable. The bundle is 370 KB.

### P67 Amendment 1: the proof and draw rate ceilings follow the verified-5090 market (#713; a lane change, no library change)

- **Why.** The registration priced P67 at $0.69/h; the verified-5090 floor has moved between $0.66 and $0.81/h in a day, and P66 burned three proof launches on provider refusals before amending. Dated `2026-09-24 16:21Z`, before any rental in this lane.
- **Amended.** Proof ≤ $0.75/h at its 600 s guard (≤ $0.125, at most three attempts); draw ≤ $0.75/h at its 3 h guard ($2.25 line); lane line $2.625. Rates and dollar lines only — arms, fixture, knobs, predictions and decision rule are unchanged, and the pinned tree is the registration's (`tests/test_p67_staged_pin.py`).

### Lane P66 read (#711): residency's launch cost is a fixed per-layer count; the bytes model under-predicts the gather on a gen 4 link (docs, register; no library behaviour change)

- **The reading.** `p66-5090-2` ran on one RTX 5090 for $0.3752; the lane cost $0.5285 over nine launches (three $0 price refusals, one Vast API NOT_RUN, one burned id, the proof, an rc-14 egress refusal, the reading), every teardown proven. All six instrument gates held. P1–P6 held, P7 was refuted, P8 and P9 are informational. Full read: `bench/p66/RESULTS-p66.md`; the box's own reducer output and rows under `bench/p66/receipts/`.
- **The count is fixed (P1, P2).** The pipelined engine adds **+10 launches, +2 copies, 0 syncs per MoE layer per token** with `n_hot > 0` (+7 / +2 / 0 at `n_hot = 0`), spread exactly 0 across cold fraction 0 / 0.5 / 1.0, eager and captured. Level M agrees by API name to the launch. Registered as `e4b.serve.p66.qwen3.pipelined-residency-fixed-count.5090.2026-09-24`.
- **The fixed tax (P3).** 1.184 ms/token of added device time captured (576 rows at 2.06 µs); 8.05 ms/token of eager wall for the same 576 submissions on a host-bound step. `…pipelined-fixed-tax.captured…`. At the RFC's 17.19 % cold the tax is 8.9 % of residency's captured cost (P8): launches cannot explain an RFC-size gap in this engine.
- **P7 REFUTED.** cold_deadline's bytes-over-link prediction is under the measured gather by **1.57–2.02×**; the gather runs at the box's *single-copy* link rate (14.4 GB/s implied against a 14.72 GB/s single-copy probe), not the 40-deep back-to-back 23.07 GB/s the calibration blob records. On the A2000 rehearsal the two probes agreed and the ratio was ~1.0. Registered as `…pipelined-gather-over-cold-deadline…`; the cause is not attributed.
- **MXFP4 and the hybrid tier (P4, P5, P6)** read as registered: NVMe engine 4 syncs/layer fixed (−1 / +3 / +3 against all-resident), pinned = all-resident, hybrid tier 4 / 10 / 8 syncs by layer composition. `…gptoss.mxfp4-and-hybrid-sync-census…`. The pinned-staging follow-up is not licensed (its licensing measurement was not taken).
- **Follow-up filed in grouped-nf4-gemm** on `kernel/cold_deadline.py`: a per-step fixed term, a measured UVA-read efficiency factor before any RFC comparison, and the mixed-layer dispatch term the deadline destination omits. No default moves.

### Lane P68 read (#725): a verify built from T = 1 calls is bit-identical to decode; the served T > 1 paths are inside the bar (docs, register; no library behaviour change)

- **The reading.** `p68-5090-2` ran on one RTX 5090 for $0.2386. The whole lane cost $0.2799, including the proof and one NOT_RUN on a stale host key, and every teardown is proven. G0 held on both stacks: 27 predictions held and 1 was refuted. Full read: `bench/p68/RESULTS-p68.md`.
- **Which component.**
  - The projections alone or the attention core alone leave layer-0 attention differing, and both together move the first difference out of it.
  - Forcing projections, core, router, LM head and norms to their T = 1 calls makes a verify **bit-identical to decode on both stacks** (264 / 264 positions). Registered as `e4b.serve.p68.qwen3.verify-from-t1-calls.row-exact.5090.2026-09-24`.
- **The refuted cell: int4 prefill under full forcing.** 159 of 160 positions differ as predicted, but position 0 is exact, because rotation there is the identity.
- **The size.** Every served T > 1 cell is WITHIN the shipped bar over 952 / 896 / 2,048 positions: KL ≤ 0.017, top-1 ≥ 0.946 (lower CI ≥ 0.9355). Registered as `e4b.serve.p68.qwen3.t-gt-1-vs-t1.within-bar.5090.2026-09-24`. P63's over-bar reads were small-sample top-1, and #725 closes.

### Lane P68 registered (#725): which part of the attention makes a verify or a prefill differ from T = 1 decode, and is it over the bar once enough positions are read? (bench only; nothing in the wheel changes)

- **The question.** P63 found every first difference between T = 1 decode and a verify or prefill in layer-0 attention. 16 of its 45 cells were under the shipped top-1 bar, all on top-1 over 64–160 positions and none on KL. P68 asks which component makes the difference, and whether the bar is crossed once enough positions are read. Prereg: `bench/p68/P68-PREREG.md`.
- **The instrument.** `bench/p68/p68_probe.py` reuses P63's probe and comparison module unchanged, and adds forcing arms.
  - A forcing makes one component (projections, attention core, router, LM head, norms) compute a multi-row call one row at a time, as T = 1 decode does.
  - The prediction is that forcing all of them makes a verify bit-exact on both stacks.
  - A size reading of the served configuration runs over P64's committed wikitext rows: ~950 verify and 2,048 prefill positions per cell, with bootstrap intervals.
  - `bench/p68/p68_reduce.py` applies the registration.
- **Rental lessons applied.**
  - The proof's guard is 0.23 h.
  - Egress is measured the way the fetch runs: four parallel ranges, in Python. The reading refuses below 80 MB/s (rc 13).
- **Tests.**
  - `tests/test_p68_force.py`: the forcings on CPU (row order, inert at one row, restoration, the core's prefix).
  - `tests/test_p68_reduce.py`: every gate and verdict branch.
  - `tests/test_p68_staged_pin.py`: the staged pin, the egress probe, the size rows.

### P66 Amendment 1: the proof and reading rate ceilings follow the verified-5090 market (#711; a lane change, no library change)

- **Why.** Two proof launches were refused at the provider before any instance existed, at $0: the cheapest verified RTX 5090 was $0.7237/h, then $0.6604/h, against the registered $0.65/h.
- **Amended.** The proof is ≤ $0.75/h with a 0.2 h guard (≤ $0.15), and the reading ≤ $0.75/h with its 2 h guard (≤ $1.50). The $2 lane ceiling holds (≤ $1.95). Registered in `bench/p66/P66-PREREG.md` "Amendment 1", before any data.

### Lane P65 read (#710): Granite's per-expert selector is TWO_ARMS; Colla-Q's cross-domain claim replicates on OLMoE and Mixtral (docs, register; no library behaviour change)

- **The reading.** `p65-5090-4` ran on one RTX 5090 for $0.5491. The whole lane cost $0.9369 over ten runs: six proofs, three boxes refused before any data, and the read. Teardown is proven for every one.
- **Completeness.** All three censuses are complete and self-checked. The reducer reproduces its output byte for byte from the committed receipts. Full read: `bench/p65/RESULTS-p65.md`.
- **Granite: TWO_ARMS.** Every ranking survives the wikitext → C4 change: `rel_act` 0.932, ρ·error 0.910, entropy 0.882 and routing frequency 0.891. Entropy is not redundant with `rel_act` (ρ ≤ 0.055).
  - The selector is written in the new `docs/SPECULATIVE_LANES_ADDENDUM_4.md`: rank by `rel_act` and by ρ·error, with frequency as the baseline, compared at matched bytes on K8. The OpenTimestamps-anchored plan documents are not edited.
  - Register row: `e4b.quality.p65.granite.selector-two-arms.5090.2026-09-24`.
- **OLMoE and Mixtral: NOT_WRITTEN.** Neither has a per-expert premise.
  - Colla-Q's comparative claim replicates on both: +0.421 and +0.479, recorded as `e4b.quality.p65.collaq-stability.olmoe-mixtral.5090.2026-09-24`.
  - Routing-frequency hot sets barely transfer between texts: Jaccard 0.141, and 0.125 (chance) on Mixtral.
- **One prediction refuted.** P1 on Mixtral: `rel_act`'s domain penalty there is 0.159 against the 0.15 survival line.
- **P44-a's census row now names its Mixtral layers.** They are {0, 1, 2, 10–22}, verified from its receipt.
- **STATUS.** Its P44 paragraph no longer calls Gemma-4's gap a served-model defect: #597 closed on P47–P51.

### Lane P64 read (#709): the int4 experts' int8 activation step at B = 1 decode is below Qwen3's arithmetic-order floor (docs, register; no library behaviour change)

- **The reading.** `p64-5090-1` ran on one RTX 5090 for $1.4193, after the proof `p64-prove-1` ($0.0201). Teardown is proven, and the validity checks were VALID.
- **What it ran on.** The licensed pack (`sha256:0c9955a9…`), rebuilt on the box and verified by `verify_artifact`. The build's K8 read ppl 6.36709, as P55x's did.
- **G_exp was INDISTINGUISHABLE.** KL(bf16 activations ‖ int8) was 0.003392 / 0.002527 nats/token on wikitext / c4val1, which is 0.61× / 0.77× of the in-lane floor. dNLL's intervals include 0. Recorded as `e4b.serve.p64.qwen3.b1.expert-int8-step.5090.2026-09-24`. W4A8 expert decode stays, and #709 closes for the experts.
- **G_attn was UNREAD.** The `a16_all` passes, the second floor pair and the NF4 anchor were skipped for time: a host-limited 67-minute pack build left too little deadline. The attention half is #728. Full read: `bench/p64/RESULTS-p64.md`; receipts: `bench/p64/receipts/`.
- **P59's B = 16 KL row now states its scope.** Its scorer leaves `DEVICE_GROUPING` off, so its decode experts took dequant + bf16 and never ran the int8 step. The fusion licence is unaffected.

### Lane P63 read (#708): which expert routes give a token the same bits alone and inside a verify or prefill (docs, register, tests; no library behaviour change)

- **The reading.** `p63-5090-1` ran on one RTX 5090 with Qwen3-30B-A3B for $0.2881, after the proof `p63-prove-1` ($0.0115) at the same commit. Teardown is proven. **Every registered prediction held**, and no path was outside its fp64 accuracy bound in 179 records. Full read: `bench/p63/RESULTS-p63.md`. Receipts: `bench/p63/receipts/`, and the reducer reproduces its JSON from them byte for byte.
- **Row-exact routes, registered.** Every experts module (48/48) returns each token's rows bit-equal to its T = 1 decode call on three routes:
  - the int4 store under `FORCE_SINGLETON_GROUPS` (`e4b.serve.p63.qwen3.int4-singleton.row-exact.5090.2026-09-24`);
  - the int4 store under `DEVICE_GROUPING` up to 256 routed rows (`…int4-device-gemv.row-exact…`);
  - the NF4 stack under `FORCE_SINGLETON_GROUPS` on the dot-pad GEMV (`…nf4-singleton.row-exact…`).
  The kernels' own row-invariance is registered in grouped-nf4-gemm (#399).
- **Different functions by row count, recorded in STATUS** (each path keeps its own quality licence; P63 moves no default):
  - the default T > 1 routes: the int4 store's dequant + bf16 matmul, and NF4's M-tile;
  - the router epilogue's fp32 weights at ≤ 64 rows against bf16 above (#726).
- **End to end, no position is exact.** Every first difference is at layer 0 in attention, never in the experts. KL mean is 0.008–0.033 nats/token. 16 of 45 cells are under the shipped top-1 bar of 0.93 (all on top-1, none on KL). Filed as #725.
- **New test.** `tests/test_p63_row_exact_gpu.py` asserts the three routes with `torch.equal` on the plan and dispatch a 5090 takes, with a control that must fail. It skips on CPU, and ran 9/9 on the NAS RTX A2000.
- **Corrected.**
  - `hot_residency.py`'s singleton comment. It said the singleton path is bitwise-equal to the grouped path at every T, "pinned in CI". That holds at T = 1 on NF4 only; at T > 1 the grouped path is a different function.
  - `tests/test_singleton_groups.py`'s docstring. It pins the dispatch algebra through a mocked GEMM, not the arithmetic.

### Lane P66 registered: the residency launch census (`bench/p66/`, for #711; nothing in the wheel changes)

- **The question.** What residency adds per token, in CUDA launches, copies and host syncs, against the
  all-resident step on the same box. Whether that moves with cold fraction. Whether grouped-nf4-gemm's cold cost
  model (`kernel/cold_deadline.py`, bytes over the link plus bytes over VRAM) predicts the measured transfer.
  vLLM #57794 left all three open, with an unexplained 2–4× cold-cost gap.
  - Paths: pipelined NF4 at five hot fractions including 0 and the RFC's 17.19 % cold; the hybrid tier (VRAM +
    NVMe); grouped-nf4-gemm's MXFP4 pinned and NVMe engines.
  - Each path is measured against its family's all-resident step, captured where capture works.
  - Families: Qwen3-30B-A3B (NF4) and gpt-oss-20b (native MXFP4) on an RTX 5090. The prereg is
    `bench/p66/P66-PREREG.md`.
- **The instrument.**
  - `p66_census.py` drives one MoE token (every layer's real expert bytes, routing set per window) through each
    engine. It counts CUDA API calls inside record_function regions, attributed to the engine's own phases, plus
    device rows, a sync-debug pass, timing, and a graph replay checked against eager.
  - `p66_reduce.py` is pure Python: the gates, the attribution against the reference, the cold-fraction test and
    the registered verdicts.
  - `p66_step.py` runs `bench/hybrid-g9/step_decomp.py` unchanged, swapping only the pipelined arm's hot sets, for
    the served-step context.
  - Runner and driver are the P60/B374 pattern (`p66_run.sh`, `p66_drive.sh`, `staged.sha256`).
- **Rehearsed on the NAS RTX A2000**, disclosed in the prereg and in `bench/p66/rehearsal-a2000/`, which is not a
  reading. The last round ran `p66_run.sh` itself end to end (rc 0) under documented rehearsal-only overrides.
  - The pipelined engine added a fixed +10 launches and +2 copies per MoE layer (+7 and +2 with no hot experts),
    and 0 syncs, identical at cold fraction 0, 0.5 and 1, eager and captured (replays bitwise equal to eager).
  - The v0-dispatch hybrid tier moved with layer composition. The MXFP4 NVMe engine held a fixed 4 syncs per
    layer.
  - The served step through step_decomp reproduced the MoE token's API deltas exactly.
  - cold_deadline's bytes model predicted the pipelined gather at 0.99–1.05× (MXFP4 engines 0.95–1.01×).
  - The rehearsals changed the instrument seventeen times before registration, each listed in the prereg. Three
    examples:
    - The profiler's device view drops records, eager and captured, while host counts stay exact. So eager
      counts come from host API calls, and captured counts from the graph's own node list: the CUDA runtime's
      `cudaGraphGetNodes`, cross-checked against torch's dot dump. That dump writes nothing, silently, on torch
      2.8 unless the graph is kept.
    - A +1.7 ms "GEMV slowdown" in round 1 was the shared card's state (+0.04 and +1.37 on later rounds). So the
      tax is split into added kernels and shared-kernel shift, and time bands are graded only on the registered
      box.
    - The runner's time guard skipped late arms by using their alarm caps as their expected times.
- **Two rentals.** The reading's guard is 2 h, so under the compute rule a **proving rental** comes first
  (`P66_MODE=prove`, 0.23 h guard, ≤ $0.15 per attempt, ≤ $0.45 over all).
  - It does the class and dud checks, an egress probe, the pinned install, the tripwire and the pin, and it times
    one checkpoint shard.
  - Following P65 Amendment 1, it records the reading-only floors (VRAM, disk, RAM, egress, pin) instead of
    enforcing them, because the proving box is not the reading's box.
  - Following P65 Amendment 2, egress is measured the way the fetch runs: eight parallel 50 MB ranges in Python,
    since the reading's `snapshot_download` uses eight workers. The reading refuses below 80 MB/s (rc 14).
  - G3 records the sha256 of the eager and replayed outputs, so a bit statement across boxes rests on hashes, not
    on difference counts.
  - It was rehearsed through the real install path, with those floors genuinely failing on the A2000.
- **Tests.**
  - `tests/test_p66_reduce.py`: counting rules, gates and every verdict branch, on synthetic event lists.
  - `tests/test_p66_census_helpers.py`: the profiler-tree walk, routing and schedule, on CPU.
  - `tests/test_p66_staged_pin.py`: the staged pin, the driver's mapping, that the driver never forwards the
    runner's rehearsal-only overrides, that the proving mode stops before any measurement, and that every arm's
    expected time is below its alarm.

### `E4B_INT4_DECODE_A16`: bf16 activations at T = 1 on the int4 expert store, as a quality instrument; lane P64 registered (#709)

- **The flag.** `E4B_INT4_DECODE_A16=1` (default off; `hot_residency.DECODE_A16`, the `FORCE_SINGLETON_GROUPS`
  pattern) routes the int4-b32 store's singleton-groups calls through the prefill branch. That is T = 1 decode in
  every default configuration. Each routed expert is dequantised and bf16-matmul'd over the same int4 bytes, and
  `quant_x_rows` is not called. There is no new kernel.
  - **Not covered:** the device-grouped batched-decode routes and the MXFP4 store.
  - **Eager only:** under CUDA-graph capture it refuses with a sentence, because the dequant loop reads the expert
    ids on the host.
  - **Off:** behaviour is unchanged.
- **Tests.** `tests/test_int4_decode_a16.py` pins both states on CPU with the int4 kernels stubbed: T = 1 is the
  GEMV on int8 activations when off, and the prefill branch bit for bit when on. It also pins that T > 1 is
  untouched, that the device-grouped route is not covered, and the capture refusal.
- **Lane P64** (`bench/p64/`: pre-registration, `kl_a16.py`, `p64_reduce.py`, `p64_run.sh`, `p64_drive.sh`,
  `staged.sha256`; `tests/test_p64_staged_pin.py`, `tests/test_p64_reduce.py`).
  - **What it reads:** the decode-scored KL A/B of that step on Qwen3-30B-A3B's licensed int4 stack, at B = 1, on
    wikitext and c4val1. The scorer is P59's, one row at a time. The floor is the arithmetic-order floor measured in
    the same instrument.
  - **The proving rental:** `P64_PROVE=1` is the proving mode the compute rule asks for before a guard over 1 h.
    It runs the install, tripwire, self-test, K0, `kl_a16.py --prove-flag` (the flag on the card's real kernels)
    and an egress probe, with no model and no pack.
  - **What was run:** the whole runner was rehearsed on the NAS RTX A2000 on OLMoE, rc 0, validity VALID, with
    every pass's counts exactly as registered. It is in `bench/p64/rehearsal-a2000/`, NOT a reading.
  - **What the rehearsal fixed:** the runner gates the pack on `verify_artifact`, not on the build process's exit
    code (a failed K8 cross-check had discarded a complete pack), and the proving run's egress probe is python (the
    image has no `curl`). No box was rented.
- **A correction to #709's premise, in the pre-registration.** P59's B = 16 KL never ran the int8 step on the
  experts. Its scorer leaves `DEVICE_GROUPING` off, so T = 16 decode takes the host-grouped dequant + bf16 branch.
  The timed B = 16 arms run the device-grouped GEMV on int8 activations.

### P65 Amendment 2: the egress floor measures the fetch's own four-stream path (#710; a lane change, no library change)

- **The single-stream egress probe under-read the fetch by 2.3×.** It read 36.1 MB/s on a box whose 4-worker Granite fetch then ran at ~83.5 MB/s. It refused every box the lane drew: 97.8, 36.1 and 21.8 MB/s, the last on the first reading box.
- **Amended.** `p65_run.sh` probes four parallel 50 MB ranges in Python, since the image has no `curl`. The floor is 80 MB/s aggregate (Mixtral ~19.5 min); the deadline logic still skips Mixtral rather than cut it. Registered in `bench/p65/P65-PREREG.md` "Amendment 2", before any reading data.

### The P65 addendum's OpenTimestamps proof completed (#736; a binary file, no text change)

- `docs/SPECULATIVE_LANES_ADDENDUM_4.md.ots` was calendar-pending at #730; `ots upgrade` fetched its Bitcoin block-header attestations (blocks 968405 and 968406). The document is byte-unchanged.

## 0.37.3 — 2026-09-24 — a memory-safety fix: the int4 singleton GEMV no longer writes past its preallocated split-K buffer at T > 1 (S2 verify); plus default-off lane switches and pre-registrations for P63, P65 and P67

### The int4 singleton GEMV no longer writes past its preallocated buffer at T > 1 (a fix; T = 1 unchanged)

- **The defect.** `enable_serve_experts_int4` sizes each store's split-K partials buffer `st["part"]` for one token's
  `top_k` rows. The singleton branch of `hot_residency._fused_over_stack` passed it to `gemv_int4_b32` at every T,
  and under `FORCE_SINGLETON_GROUPS` at T > 1 (the S2 verify default, `--moe-grouping singleton`) the GEMV has
  `T * top_k` rows and plans its split count from them. Measured on the NAS A2000 with the buffer a view into a
  sentinel-filled one: at T = 17 the call wrote 16,384, 12,288 and 180,224 fp32 elements past it at the OLMoE gate_up
  and Qwen3 gate_up/down shapes; T = 1 and T = 2 wrote nothing past it
  (`bench/p63/rehearsal-a2000/part_oob.json`). Found while mapping lane P63's routes (#708).
- **The fix.** `_int4_part_or_none` keeps the buffer when it has the rows the call's own plan needs and passes None
  otherwise, so the wrapper allocates its own (through the graph pool under capture, as the device-grouping GEMV
  branch already does). The arithmetic is unchanged: the split count was always the call's own plan. The fit is
  memoised per (rows, device) on the store, because this runs on every decode expert call and the eager B=1 step is
  host-bound. Each shape is planned once, then costs a dict lookup.
- **Tests.** `tests/test_int4_singleton_part_fits.py`, CPU with the kernel stubbed: one token keeps the store's
  buffer, a 17-token call gets None, a buffer the call's smaller plan fits is kept, and the shipped planner (where
  triton imports) decides the same way at 26 and 170 SMs.

### P65 Amendment 1: a proof records the reading-only host floors instead of refusing on them (#710; a lane change, no library change)

- **Three proving draws, none reached the install.** The first box missed the host scale-and-add floor, 0.152 s against 0.15 s. The second was a launcher refusal: I named the first receipt in the wrong exclusion class. The third missed the egress floor, 97.8 MB/s against 100. That spent $0.1009 of the registration's $0.15 proof budget. Those floors exist for the reading's Mixtral census. The proof's box is not the reading's box, and the proof never runs Mixtral.
- **Amended.** In a proof, `p65_run.sh`'s new `floor()` records the RAM, scale-and-add and egress floors (`floor_would_refuse_reading`) and continues. Class, dud and disk still refuse, and a reading refuses on every floor exactly as before.
- **Guard and budget.** The proof guard is 0.23 h, ≤ $0.15 each; launcher boot had taken 3–5 min of the old 10 min guard. The proof budget is ≤ $0.45 over all attempts, and the lane ceiling is $2.05. Registered in `bench/p65/P65-PREREG.md` "Amendment 1", before any reading.

### P67 registered: training parity read against a per-family floor, not against zero (#713) (one default-off switch in the wheel; bench tooling)

- **`bench/p67/P67-PREREG.md`, registered before its draw.** tp1's band compares an accelerated arm with the
  reference against a constant 0.05 **and against zero**, and on Gemma-4 every accelerated path fails it, the
  kernel-free one included. The lane replaces "against zero" with a measured floor: the SAME reference path, same
  box and session, run again (a plain repeat, and four runs whose per-expert loop sums the experts in a fixed
  permuted order, correct by construction). The band is `max(0.05, 3 × F_hi)` on both of tp1's quantities, on the
  TRAIN loss, with at least 3 admissible floor draws; fewer is NO-FLOOR, never FAIL. The batched arm is judged
  against that floor, not used as it: it shares code with the fused path and could not judge itself.
- **The existing receipts, re-read on CPU (`bench/p67/reread-existing/`): no session ever ran the reference twice,
  so no family has a floor draw.** Every Gemma-4 attention-4-bit row reads NO-FLOOR; every other row is a
  constant-band PASS, which the rule keeps as a PASS. The rental therefore covers Gemma-4 only: one RTX 5090,
  ≈ 2.2 h, ≈ $1.5, guard 3 h (approval line $2.07), preceded by a ≤ 10-min proving rental (`p56_prove.sh`, $0.115
  line). **Not launched.** `claims.json`, `capabilities.json` and STATUS do not change until it is read.
- **Rehearsed free on the NAS A2000** (`bench/p67/rehearsal-a2000/`, not a reading): on a small MoE the switch
  reached the loop on every call, a same-box repeat was not bit-identical, and every permuted run moved the
  trajectory (two of three already at step 0, which revised one prediction before registration).
- **In the wheel: `E4B_REFERENCE_EXPERT_ORDER` (`experts4bit_qlora/lora.py`), default off.** Unset, the reference
  loop is the shipped one and the helper is never called. `descending` or `perm:<seed>` visits the same experts in
  another order, which moves only the rounding of the sums. Anything else raises. `reference_order_stats()` reports
  what the loop did. Tests: `tests/test_reference_expert_order.py`.
- **Harness.** `tp4_arm.py` records `reference_order` on every e4b reference receipt and refuses the switch on any
  other arm (exit 19). `tp4_run.sh` gains an opt-in `TP4_P67=1` block (the floor arms, run last). `tp4_drive.sh` gains
  `TP4_RUNNER` / `TP4_EXTRA_STAGE`, which default to its own behaviour. `bench/p67/` adds the reducer, the box guard
  `p67_run.sh` (pins, registered knobs, host RAM ≥ 96 GiB), the controller `p67_drive.sh`, `registered.knobs` and
  `staged.sha256`. Tests: `tests/test_p67_reduce.py`, `tests/test_p67_harness.py` (which executes the new shell
  paths rather than parsing them) and `tests/test_p67_staged_pin.py`.

### P65 registered: per-expert activation entropy beside `rel_act`, and which ranking survives wikitext → c4val1 (#710; not yet run)

- **The calibration tap keeps a first moment, on request.**
  - `calibrate_expert_hessians(..., activation_means=)` fills a dict with each expert's fp64 mean gate/up input and
    down input, from the same tap, over the same rows. It uses `_ExpertHessianSink(means=)`.
  - Off by default. The Hessians are bitwise the same tensors either way, and the return value does not change
    (`tests/test_p65_entropy.py`).
  - The Hessians are uncentred (`2 X X^T`), so no statistic that needs per-channel means could be read from them
    alone. Nothing else in the wheel changes.
- **`bench/p65/`** (P65-PREREG.md):
  - `expert_entropy.py`: Colla-Q's activation-entropy ratio `rho = sigma2_within / sigma2_total` of each expert's
    output, read exactly from the tap's moments.
  - `p65_census.py`: P44-a's `census_row`, unchanged and RTN only, plus the entropy field, on wikitext-2 train and
    K8's c4val1 text. Each text runs in two disjoint halves; the full rows are combined exactly.
  - `p65_reduce.py`: within-layer rank stability (a split-half ceiling, and the domain shift at the same sample size),
    overlap with `rel_act` and with routing frequency (through `hot_sets_from_profile`), Colla-Q's cosine for
    replication, and the registered rule for S-C's selector.
  - `p65_run.sh` / `p65_drive.sh` / `staged.sha256`: the box/controller pair, with the pin, nonce, heartbeat and HF
    token staging. The host floors are refusals, not stalls:
    - RAM and disk;
    - HF egress ≥ 100 MB/s, because Mixtral is 93 GB;
    - a host-side Hessian update ≤ 0.15 s at Mixtral's 14336² shape. That update runs about 4,100 times, and the
      rehearsal measured 0.33 s for it on the NAS Xeon.
  - **`P65_PROVE=1`** is the proving rental the compute rule requires in front of a guard over 1 h (≤ 10 min,
    ≤ $0.15). It runs every refusal, the install and the tripwire, then Granite's first layer at `nseq 8` end to end.
    It writes `prove_census_granite.json` and `MODE=prove`, which the reducer never reads. The same cut-down census
    passed on the A2000 (`rehearsal-a2000/prove_config_*`).
- **Rehearsed on the NAS RTX A2000**, not a reading (`bench/p65/rehearsal-a2000/`). It found that the census walks
  layers in checkpoint-key order (0, 1, 10, 11, …). P44-a's "16 of 32" Mixtral layers were therefore most probably
  not 0–15; unverified here. P65 takes the first 16 of the same order (`--first-layers`).
- **Tests:**
  - `tests/test_p65_entropy.py`: known distributions, moments against raw rows, the sink, the census row keeping P44's
    `rel_act`, halves, texts, and layer order;
  - `tests/test_p65_reduce.py`: every decision branch on synthetic rows;
  - `tests/test_p65_staged_pin.py`.

### Lane P63 registered: does a token's output depend on how many rows share its forward? (#708; bench only)

- **What.** `bench/p63/`: the pre-registration, a GPU probe, its comparison module and reducer, and the rental runner.
  Nothing is claimed yet and no default moves.
- **The routes, mapped against the code.** The int4 store (T = 1 GEMV on int8 activations vs a dequant + bf16 matmul
  above it; `DEVICE_GROUPING`'s GEMV up to 256 rows), NF4 (dot-pad or scalar decode GEMV vs the TF32 M-tile), the
  three int4 attention buckets, `combine_rows`, and routes #708 did not list: the decode folds switch to the torch
  chains above 64 rows, the paged prefill attends bf16 K/V where decode and verify read fp8, and cuBLAS picks its dense
  kernel by M.
- **The probe.** Qwen3-30B-A3B on an RTX 5090, three stacks and fifteen sub-arms, each a route chosen by
  configuration. Each sub-arm is read three ways:
  - end to end: per token, layer and site, the T = 1 incremental-decode control against the same token in 17- and
    16-row verifies and a 160-row prefill;
  - a module replay of the recorded T = 1 inputs as one n-row call;
  - a kernel census against fp64.
- **The design moved before registration, and the prereg says where.**
  - Per-layer ULPs are reported only over significant elements, the B393 lesson.
  - The fp64 accuracy bound is the defect line and not a classifier, because at served K it cannot see a bf16-rounded
    weight. The REORDER vs PRECISION classes come from the kernels' declared operand models.
  - cuBLAS paths are rerun with torch's `allow_bf16_reduced_precision_reduction` off for that line.
- **Rehearsed on the NAS A2000** (OLMoE, not a reading; `bench/p63/rehearsal-a2000/`). It found a route #708 did not
  list: the fused router epilogue returns fp32 routing weights up to 64 rows and the upstream router bf16 above. That is
  registered as P7.
- **B393's side diagnostic** (`fma_attribution.py`, grouped-nf4-gemm #397, present from `f88df1e`) rides along,
  guarded and non-fatal.
- **The rental plan** is a 1.5 h guard, preceded by the ≤ $0.15, ≤ 10 min proving rental the standing rule requires,
  ≤ $1.15 in all.
- **Tests.** `tests/test_p63_compare.py`, `tests/test_p63_reduce.py`, `tests/test_p63_probe_plumbing.py` (a tiny random
  Qwen3-MoE on CPU) and `tests/test_p63_staged_pin.py`.

### The fused MoE combine's call-site comment says what lane B393 measured (a comment; no behaviour change)

- **`engines/hot_residency.py`'s call site said `combine_rows` takes "the same order and roundings as the chain
  below".** grouped-nf4-gemm's lane B393 measured otherwise on an RTX 5090 (grouped-nf4-gemm#397, #393), over 144
  census cases at every served family's shape:
  - **Not bitwise.** The kernel differs from the chain in 95 cases, 392 of 9.07 M elements. It sums in slot order,
    with a fused multiply-add exactly so on sm_86; the chain rounds each product and sums in torch's order.
  - **Both correct.** Both are within the error bound of a correct fp32 summation in every case, at the same max
    ratio (claim `gnf4.kernel.combine-rows-accuracy.5090.2026-09-23`).
- **The comment now says so.** The default stays `E4B_FUSE_COMBINE=1`, because an accuracy-equal kernel is not a defect.
  The end-to-end size of the difference is #708's probe, with `E4B_FUSE_COMBINE=0` as its control arm.

### Lane B393's runner (`bench/b393/`, for grouped-nf4-gemm#393)

- **What it runs.** The box side of grouped-nf4-gemm's pre-registered lane B393: are `combine_rows` and
  `reduce_partials` bitwise equal to the torch chains they replaced? This repository runs `combine_rows` on every MoE
  layer by default, and its call site says the fused path takes "the same order and roundings as the chain below".
- **How.** `b393_run.sh` installs grouped-nf4-gemm at `GNF4_SHA`, clones the same sha for the census script (it is not
  in the wheel), and proves from a work dir holding only the census that `int4_b32` resolves to the installed
  package. It then runs the 414-case census once. The verdict is the census JSON, read against grouped-nf4-gemm's
  pre-registration.
- **Driver and pin.** The driver and the staged pin follow B374's pattern. `tests/test_b393_staged_pin.py` mirrors the
  pin test.

### Correction: Gemma-4 attention-4-bit training HAS receipts; it is unlicensed for a different reason (documentation; nothing in the wheel changes)

- **Two entries below said the arms had not run and had "no receipt", and pointed at a re-run (#703).** That was wrong,
  and I wrote it without checking P56. Since #435 the Gemma-4 attention-4-bit arms have run cleanly on three boxes:
  `tp4-c-4`, `tp4-c-parity-2` and `p56-gemma4-ladder-3`. Each converts all 115 structural projections (25 sliding
  layers × 4, and 5 full-attention layers × 3, which have no `v_proj`), and every arm is VALID.
- **Why it is still not supported.** tp1's internal training-parity band, a constant 0.05 against zero, fails for
  every accelerated path on this model, the kernel-free batched one included (0.054 final against the fused path's
  0.102; `e4b.parity.gemma4.train-floor`, `bench/p56/RESULTS-p56.md`). The band cannot tell a defect from the model's
  own sensitivity. P56 recommends a per-family floor measured by the smallest-perturbation arm, now filed as #713.
- **What changed.** STATUS, `capabilities.json` (four strings), ARCHITECTURE_SUPPORT, the QLoRA solution page and the
  two HARNESS_ERROR register notes now say this, and point at #713 instead of #703.

### `check_change_impact.py` and `check_dependency_floor.py` are one file each, shared with grouped-nf4-gemm (repository tooling; nothing in the wheel changes)

- **Byte-copied from grouped-nf4-gemm** (pjordanandrsn/grouped-nf4-gemm#394). With these, `SHARED` lists 14 files:
  every CI script both repositories carry except the per-package `wheel_smoke.py`. Each reads its role (`runtime` here)
  through `check_system_manifest.system_role` and keeps this repository's settings in its `PROFILES` entry.
  - **Change impact diffs against `git merge-base BASE HEAD`**, where this copy diffed against the base itself. On a
    branch behind its base, the old reading blamed the PR for a claim change the base made, and it passed a PR whose
    missing companion the moved base happened to supply. CI's discoverability checkout therefore now uses
    `fetch-depth: 0`, and the depth-1 base fetch is gone, since it would make the clone shallow again. A shallow
    checkout exits 2 with that hint.
  - **Change impact now also checks, here:**
    - a version bump needs CHANGELOG;
    - a dependencies change warns without README / capabilities;
    - a capabilities entrypoint change needs CHANGELOG;
    - a claim's `unit` change is a measured-result trigger;
    - every class a trigger reports must be named in `docs/change-impact.json`. That contract now describes these
      triggers.
  - **Dependency floor now also checks, here,** the kernel copy's version statements: this package's own version and
    tag links, the torch floor, Python/CI, requires-python and the licence, across the current documents (24
    statements, all agreeing). The floor still comes from pyproject's `fast` extra, and this copy's historical markers
    and anchored-document exemption are kept.
  - **Tests.** `tests/test_readability_checks.py` fixture data only: the fixture manifest names both packages, the
    pyproject gains a Source URL, and the fixture writes the current documents and a minimal contract. No assertion
    changed.

### `check_readme_claims.py` is one file, shared with grouped-nf4-gemm (repository tooling; nothing in the wheel changes)

- **Byte-copied from grouped-nf4-gemm** (pjordanandrsn/grouped-nf4-gemm#392) and now in `SHARED` (12 files). The two
  copies had forked by 399 diff lines. The unified file reads its role through `check_system_manifest.system_role`
  and reads the claim-id namespace from the register. This repository is `runtime`, so it keeps this copy's
  settings:
  - README is the results document, headed `status`;
  - a row's result is its last non-status cell;
  - the generated release block (`--write-release-block` still works, and is a no-op on the current README);
  - anchored documents are exempt;
  - a status word in backticks counts.
- **Now also checked here.** A missing README is reported as a finding (exit 1) instead of a traceback. A register
  with no dotted id is exit 2. The manifest is required.
- **Tests.** `tests/test_check_readme_claims.py` needed four edits, all mechanical: the unified functions take the
  kernel copy's signatures (document name, column tuple, pattern and profile arguments). No assertion was loosened.

### Gemma-4's tp2 attention-4-bit void was the harness's count check, not the converter's (documentation; nothing in the wheel changes)

- **The docs said both e4b attention-4-bit arms "died on the converter's own count check".** Before #435 the converter
  had no count check. The check that voided both arms is the tp2 harness's: `bench/h2h-20260906/tp2/tp2_arm.py:526-527`
  asserts `n_attn4 == 4 · n_layers` on the converter's return value.
  - **Corrected in** STATUS, ARCHITECTURE_SUPPORT, `capabilities.json` and the QLoRA solution page. The two
    HARNESS_ERROR rows keep their headline and gain a precision note.
  - **Why it matters.** A re-run under that harness as written would void again, because Gemma-4's `k_eq_v` layers put
    the library's census (`len(detect_attention_projections)`) below `4 · n_layers` by design. tp4's harness, which
    checks the structural census, is the one that has run these arms since (see the correction below).

### `check_system_manifest.py` is one file, shared with grouped-nf4-gemm (repository tooling; nothing in the wheel changes)

- **Byte-copied from grouped-nf4-gemm** (pjordanandrsn/grouped-nf4-gemm#391), and now listed in `SHARED`. The two
  copies had forked by 734 diff lines: this repository's enforced 39 rules and the kernel's 31. The unified file reads
  its role from the manifest (`runtime` here) and runs that role's rules.
  - **Runtime role, as before:** the `fast` floor, every kernel-pinning extra, and the CI kernel pin as a release-tag
    commit (still the only network read).
  - **Now also checked here:** an ssh-form Source URL; the `Kernel:` URL against `packages.kernels.pypi`; the shape of
    every compatibility record, including the non-current 0.34.x one; every clause of a range, not just the first;
    ownership duplicates or overlap; unique invariant ids; the router's size. A `fast` pin with an upper bound
    (`>=0.30.0,<1`) now fails locally too.
  - **Output.** The check no longer stops at the first finding, so later findings print as well.
  - **Parity.** On `main` its verdicts match the old copy's in both CI forms, and the existing tests
    (`test_check_system_manifest_pin.py`, `test_readability_checks.py`) pass unchanged. No importer here needed an
    edit.

### Gemma-4 attention-4-bit: the docs point at the open re-run, not the closed count-check bug (documentation; nothing in the wheel changes)

- **Six places said Gemma-4 attention-4-bit training is "NOT supported pending #412"**, and STATUS listed #412 under
  "What is open": `docs/STATUS.md`, `docs/capabilities.json` (four strings), `docs/ARCHITECTURE_SUPPORT.md` and the
  QLoRA solution page. #412 was closed on 2026-09-23 as fixed by #435, so the docs pointed at a closed issue.
  - **What #412 fixed.** It was the converter's count check (100 of 120 projections), which killed tp2/P40's two e4b
    attn4 arms. #435 now detects projections by structure, with `v_proj` optional on `k_eq_v` layers.
  - **What stays open.** The configuration stays **not supported**. This entry first said nothing had re-run those arms
    and pointed at a re-run, #703. That was wrong: the arms have run since, as the correction below records.
  - **Register.** The two HARNESS_ERROR rows keep #412 as their evidence and gain a note: fixed by #435, not re-run,
    #703.
  - The dated tp2 receipt README is unchanged, because a receipt is a record.
  - The site's `qlora-fused-moe-experts` FLAG still fires. It is a true family-scoped caveat: the capability is
    supported, and one family's attention-4-bit cell is not.

### The serving-position WARN reads STATUS's position section (repository tooling; nothing in the wheel changes)

- **`scripts/check_capabilities.py` (shared; copied byte-for-byte from grouped-nf4-gemm, pjordanandrsn/grouped-nf4-gemm#389).**
  The WARN takes "the position" to be the newest `area: serve` claims `docs/STATUS.md` quotes. It read the whole file,
  so P61's diagnostic read counted as the position: that expert-GEMV cost split, measured 2026-09-23, is quoted under
  "What is open". The capability then warned on every CI run that it "headlines an older lane".
  - **The fix.** The rule now reads only the text before STATUS's first `## What changed` heading. The position here is
    P58's 2026-09-22 rows, which `serve-moe-on-consumer-gpu` cites, so the warning clears with no data edit.
  - **Tests.** Two new tests in `tests/test_check_capabilities_serving_position.py` use the real STATUS layout:
    - a newer lane quoted only after "What changed" is not the position;
    - a newer lane in the position section still warns.
  - **Mutation check.** With the old whole-file read restored, the first test fails.

## 0.37.2 — 2026-09-23 — documentation, register data and repository tooling only (under the package only `__version__` changes): the claims register has ONE schema, shared with grouped-nf4-gemm; lane B374's runner, whose read holds both predictions

### The claims register has ONE schema, shared with grouped-nf4-gemm, and this register is migrated to it (repository tooling and data; nothing in the wheel changes)

- **`scripts/check_claims_register.py` and `docs/claims-schema.md` are one file each, byte-identical in both
  repositories** (both now in `SHARED`). The two copies had drifted into two schemas that refused each other's data (14
  findings one way, 38 the other). The converged rules, in `docs/claims-schema.md`:
  - **Locations.** An evidence path or a `quoted_in` entry is `path` or `path#anchor`. The path is a file in the git
    tree at HEAD, never a directory, annotation or glob. The anchor is `L<n>` / `L<n>-L<m>`, or the anchor
    **github.com renders** for a Markdown heading, so every location is a working link. The checker's anchors match
    github.com's on all 371 headings of both repositories' READMEs, CHANGELOGs and docs.
  - **Evidence** is a location, `{"url"}` (an issue or pull request of a system repository), or
    `{"repository": <package>, "path": <location>}` (resolved in a `--sibling` checkout). A public-run row's FIRST
    entry is a location, because the consumer site links it.
  - **Successors** are direct and named back. `superseded_by` names the active row itself, with no chains, and that
    row lists it in `supersedes`. A retired row may name its restatement.
  - **The file is closed.** Unknown row fields and top-level keys are findings, and `area` / `tier` / lane fields take
    only their values.
  - The licence, fingerprint, date and placeholder rules are the union of both repositories' old rules.
- **Tests.** One test file, identical in both repositories, produces every rule's failure. A mutation sweep disabling
  each of 32 rules in turn was caught every time.
- **This register's migration.**
  - 20 free-text `quoted_in` entries became locations. Entries whose file no longer quotes the id were dropped, and
    every original entry is kept verbatim in the row's `notes`.
  - The cross-repository entry names the package (`grouped-nf4-gemm`), not the GitHub slug.
  - The successor graph was made direct and bidirectional. `e4b.parity.gemma4.behaves` now points at
    `…no-reference`, not through `…chunk-free`, and three successors gained the `supersedes` back-links they lacked.
- **A defect the migration found.** The featured claim `e4b.train.energy-honest.scoped-a2000` linked
  `docs/METHODOLOGY.md#10-energy--measured-benchupstream…`. GitHub's anchor keeps the underscore of `bench/_upstream`,
  so the receipt link on the consumer site has never scrolled to its section. It is fixed.

### Lane B374's runner (`bench/b374/`, for grouped-nf4-gemm#374)

- **What it runs.** The box side of grouped-nf4-gemm's pre-registered lane B374: the word-addressed NF4 decode routes
  (wide loads, and dot-pad, the default on >= 160-SM parts at its census shapes) put past THEIR 2^31 boundary on an
  RTX 5090.
- **Two passes.** `b374_run.sh` installs grouped-nf4-gemm at `GNF4_SHA` and runs its
  `kernel/test_offset_boundary_words_gpu.py` twice, from two work dirs (the test puts its own directory first on
  `sys.path`):
  - against the installed kernels, where all four cases must pass;
  - against a copy with all six eid promotions removed, where all four must fail. Tripwires record where each pass
    resolved `nf4_grouped`.
- **One process per case.** Each GPU case runs in its own pytest process, so a fault cannot poison the verdicts of the
  cases after it.
- **Driver and pin.** The driver and the staged pin are K18's, re-pointed. `tests/test_b374_staged_pin.py` mirrors the
  K18 pin test, and the driver's dry run stages, starts and fetches as written.
- **The read (pjordanandrsn/grouped-nf4-gemm#385, 2026-09-23).** Run `b374-5090-1` on one RTX 5090 cost $0.0354, and its teardown
  is proven. P1 held: all 4 cases pass on the shipped kernels. P2 held: all 4 read the decoy at the int32-wrapped
  address once the six promotions are stripped. The runner ran unchanged at `cf80b0f`. The results and the claim
  (`gnf4.kernel.word-boundary-wide-dotpad.5090.2026-09-23`) live in grouped-nf4-gemm, the repository that owns the
  kernels.

## 0.37.1 — 2026-09-23 — documentation and repository tooling only: the same-box vLLM comparison is P58's everywhere, the cross-host pack limitation is P55x's, CI runs the census cross-check, and the CI scripts shared with grouped-nf4-gemm start to become one file

**0.37.1.** Nothing a user imports changed: the package code is identical to 0.37.0's (under `experts4bit_qlora/`, `git diff v0.37.0` changes only the `__version__` literal). The documentation that ships with it is corrected. The README (which is also the PyPI description), the serving capability and the serving solution page led with the 2026-09-05 comparison against vLLM 0.28.0; they now quote the current same-box one (P58: vLLM 0.30.0 decodes 1.087× faster than this package's current int4 stack at B=1 and 1.396× at B=16, bounded to one box and prompt set). The capability's statement that the streamed calibration does not reproduce across hosts is replaced by what P55x measured. CI additionally runs the census cross-check against the kernel package's shape census (grouped-nf4-gemm#353) and fails if the CI scripts shared with grouped-nf4-gemm stop being byte-identical to its `main`. No action is needed if you are on 0.37.0; the `[fast]` floor stays `grouped-nf4-gemm>=0.30.0`.

### The CI scripts both repositories carry start to become one file (repository tooling; nothing in the wheel changes)

- **`scripts/check_shared_tooling.py`** (new, byte-identical in grouped-nf4-gemm): `SHARED` lists the scripts that are one
  file in both repositories. Nine of the thirteen same-named scripts had forked, so one check name enforced two rules.
  With `--sibling` it fails on any differing byte, a shared file missing there, a self-comparison or a sibling outside the
  system, and lists the scripts still forked as NOTEs. grouped-nf4-gemm is upstream for shared tooling (kernel-first, as
  for the manifest); a new `ci.yml` step clones its `main` and runs `--sibling`. `tests/test_check_shared_tooling.py`
  produces every failure the check claims and asserts it is detected.
- **`scripts/check_discovery_contract.py`** adopts grouped-nf4-gemm's copy, which adds `page_kind: orientation` records
  ranked over their own corpus. This repository has none, so every query keeps its kind and corpus: the 45 rankings are
  identical before and after (37/45 top-1 against the floor of 30), compared line by line.

### Docs: the same-box vLLM comparison is P58's everywhere, and the cross-host pack limitation is P55x's

- The README ("Do not use this when" and the results table), `serve-moe-on-consumer-gpu`'s limitation in
  `docs/capabilities.json` and `docs/solutions/serve-large-moe-on-a-consumer-gpu.md` still led with lane p37's 2026-09-05
  comparison against vLLM 0.28.0 (2.52× / 4.06× over the NF4 control). P58 re-measured current against current on
  2026-09-22 and re-pointed `docs/STATUS.md` and `docs/SERVING-THROUGHPUT.md`, but not these four. They now quote P58
  (vLLM 0.30.0 / e4b's current int4 stack 1.087 at B=1, 1.396 at B=16) and keep p37 as history. No register check could
  catch the omission: p37's rows are still `measured`, correctly, because they are history rather than retracted.
- `docs/STATUS.md` cited `e4b.serve.buildout.bo6.qwen3.calibexp-streamed-*`, a glob that breaks mid-segment; the
  consumer site's claim-reference check reads it as the id `…calibexp-streamed-`, which is not in the register, and
  refused the 0.37.0 re-pin on it. The three rows it stood for are now named.
- The same capability said the streamed 64k calibration "does not reproduce its licence across hosts". P55x found the
  pack byte-reproducible on one box and across boxes and a release boundary, and P37's divergent pack an outlier; the
  limitation now says that, and what stays open (the calibrated attention half is not in the fingerprint, #674).
### CI runs the census cross-check (grouped-nf4-gemm#353)

- `bench/support/census_cross_check.py --check` (#522) fails when a claimed family that the kernel's shape census covered
  stops being covered, but nothing ran it. It now runs in `lint-and-test` against `census/shape_census.json` read from the
  exact kernel commit pip installed (`direct_url.json`), so it is never a second pin. Calibrated: a census with OLMoE's
  down-proj shape removed exits 1 (`census coverage REGRESSED for: olmoe`); the v0.33.0 census exits 0 (4 of 11 probed
  claimed families covered). `--check` writes a missing baseline and passes, so the step asserts the baseline exists first.
### `check_capabilities.py` joins the shared set; the two discovery corpora stop answering one question two ways

- **`scripts/check_capabilities.py` is one file in both repositories** (added to `SHARED`). grouped-nf4-gemm's copy was a
  strict subset of this one. Adopting this copy there as it stood would have run cleanly, but only because the
  serving-position rule's id pattern hard-coded `e4b.` and so could never match a kernel id: deriving the prefix instead
  made it fire falsely in the kernel repository, whose serving capabilities are separate kernels, each on its own lane
  (two false warnings measured). The rule is now gated on the repository's role in `docs/system-manifest.json`, and the
  id namespace is the register's own. Behaviour here is unchanged (the same single WARN on `main` before and after); new
  tests pin the namespace derivation, the mixed-register refusal and the role gate.
- **One routing question, one answer per phrasing.** Both repositories' `docs/discovery-queries.json` carried "Where does
  an NVMe primitive belong versus model-level NVMe integration?", each routed to its own page, so the consumer site's
  merged corpus recorded it as unmapped with a WARN ("to be settled upstream"). The kernel repository keeps that
  phrasing (the primitive's side); this repository now asks "Where does model-level NVMe integration live versus the
  NVMe primitives?" (the integration's side), which still ranks its page first.

## 0.37.0 — 2026-09-23 — five more families admitted (`nemotron_h`, `granitemoehybrid`, `granitemoeshared`, `jamba`, `lfm2_moe`) and the five still staged say why, as tests; fused q/k/v licensed on the int4 serving lanes at B=1 and B=16 (P54, P59); a licensed int4 expert pack again, as bytes (P55x); the same-box vLLM comparator re-run on 0.30.0 (P58); two expert-GEMV levers built and refused (K17, K18), and the B=16 expert GEMV's remaining headroom bounded with no lever to follow (P60, P61)

**0.37.0.** The loader admits five more MoE families. `nemotron_h`, `granitemoehybrid`, `granitemoeshared` and
`lfm2_moe` each load a real published checkpoint (`reference-ok`); `jamba` is so far exercised only on a toy
checkpoint (`toy-ok`). The five families still refused (`axk1`, `dbrx`, `jetmoe`, `qwen3_vl_moe`,
`qwen3_vl_moe_text`) now each name a measured blocker, asserted by a test that fails the day the blocker lifts. On the
int4 serving stack, fusing q/k/v is licensed at B=1 (12.4 % per step, token-identical) and at B=16 (0.0044 nats/token).
A calibrated int4 expert pack for Qwen3-30B-A3B is licensed again, as retained bytes loaded back by fingerprint. A load
that faults on CUDA can now be re-run with `E4B_LOAD_SYNC_DEBUG=1`, which synchronises after each load stage so the
failure is reported at the stage that caused it, and a failed shard read now prints the host's memory facts. Everything
else is measurement: the same-box comparison with vLLM 0.30.0 was re-run (vLLM decodes 1.09× faster at B=1 and 1.40× at
B=16). Two expert-GEMV levers were built and refused (K17's fused reduce and K18's grouped GEMV), and the B=16
expert GEMV's remaining headroom was measured and has no lever to follow (P60, P61). Affected: loading of the five newly admitted families
(CPU and CUDA) and the int4 serving lanes on NVIDIA GPUs. Nothing changes for NF4 training or for any package default, and families
admitted before 0.37.0 load as they did (the new prefix-rename pass is a no-op for them, asserted by test; the only
difference they can see is the diagnosis a failed shard read now prints). Upgrade if you load one of the five families or want the
load-fault diagnosis; no action otherwise. The `[fast]` floor stays `grouped-nf4-gemm>=0.30.0`; CI now runs against
grouped-nf4-gemm 0.33.0.

### Loader and engine changes not filed under a lane below

- **`E4B_LOAD_SYNC_DEBUG=1` — staged synchronisation for CUDA load faults** (#660, #344). CUDA reports asynchronously, so
  the traceback of a fault raised during a streamed load need not name the kernel or the stage that caused it. With the
  flag set, the loader synchronises at each stage boundary and logs `[sync] <stage>: clean`, so the first stage that
  does not come back clean is the one that faulted. It also logs the host's `MemTotal` / `MemAvailable` / cgroup limit
  and the largest shard it will map. The flag is ignored with a log line on a non-CUDA device, and off by default
  because the synchronisations serialise the load. Separately, and always on, a shard read that raises now logs a
  diagnosis (shard, size, device, host memory facts) before re-raising unchanged. Nothing branches on the memory
  numbers: they describe a host, they do not refuse one. `tests/test_load_sync_debug.py`. #344 stays open; the
  Gemma-4 load fault it tracks is bounded to a stage by this, not fixed.
- **`E4B_BATCHED_PAD_WASTE_LIMIT`** (#662) overrides the batched engine's 4× pad-waste fallback guard (default
  unchanged). The guard only protects speed, since falling back and batching compute the same function. A parity arm
  wants the batched arithmetic on every call, and a run that falls back is measuring the reference against itself.
  `batched_fallback_stats` reports the limit in force, so a receipt says which one it ran.

### P56 read: Gemma-4's training-parity FAIL is not the fused path's, and the band has no floor term (#662, #671, #672, #675)

- **The kernel-free batched path fails the same 0.05 band** as the fused path on the same box, init and tokens:
  0.05425 final / 0.08487 median against the fused path's 0.10223 / 0.10639. It never touches grouped-nf4-gemm and has
  13× less composed gradient error, yet it closes only 1.9× of the gap, so no achievable change to the arithmetic
  reaches the band. The CPU half (#662) found e4b's expert composition exact and family-blind: every family, Gemma-4
  included, sits at the same bf16 floor against an fp32 arm over identical quantized weights.
- `e4b.parity.gemma4.train-internal` is superseded by `e4b.parity.gemma4.train-floor`, the first measurement of this
  family's training-parity floor (≥ 0.054 final / 0.085 median). The superseded row also mislabelled its unit: it said
  "held-out" while the number was `loss_last`, the final TRAIN loss. **#558 stays open as a gate defect**:
  `tp4_reduce.parity()` compares against 0.05 and against zero, with no floor term. Refusing the fused path for
  `gemma4_text` is not supported by this read.
- Two harness defects, both fixed. An unquoted expansion in the arm's assignment prefix made the next `VAR=value` the
  command, so all four arms of draw 1 died with rc 127 after measuring nothing ($0.29). They are now routed through
  `env`, and a regression test executes the real prefix from the real file (#672). The pre-registration's micro-batch
  was misstated as 1: the run used the field recipe's micro-batch 2, so the draw matches the standing claim row's
  fixture, and the erratum is filed rather than edited (#671).

### Lane tooling

- `p47_drive.sh` / `p54_drive.sh` gain the heartbeat, stall verdict and lane-dead exit (25) from `tp4_drive.sh` (#655,
  fixes #641). `bench/p319/` and `bench/p319b/` carry the pre-registrations that three committed receipts already
  cited (#669).

### llms-full.txt headroom: 399,971 → 379,460 bytes against the 400,000 cap (docs only, no number or status changed)

- `docs/STATUS.md` (−11.9 KB): serving-history paragraphs (p37, bo3, bo5, bo6) now give the verdict and point at their
  results files. Training paragraphs (tp1, p38, tp2) and restated numbers elsewhere cite the ACTIVE claim rows that the
  bundle already projects. Numbers with no projected claim row (Gemma-4's P47–P52 store map, P56's first readings) stay.
- `docs/claims.json` (−8.6 KB projected): 128 ACTIVE `claim` sentences shortened by wording. Hashes, revision pins,
  library versions and verbatim log quotes move to the same row's `notes`, and the moved text is appended verbatim. Every
  measured number, id, status and value is unchanged, checked mechanically against `main`.

### P61 read: the B=16 expert GEMV's row work and expert bytes overlap -- no lever lane (lanes `p61-5090-1`..`-6`, $0.28)

- On a 575 W RTX 5090 (`bench/p61/RESULTS-p61.md`), the served int4-b32 expert GEMV reproduces P60's replay: 6.548 vs 6.479
  ms/step. On per-layer weight stores the recorded B=16 routing's repeated rows cost **0.905 ms/step** in total.
- The pre-registered additive model (fixed + per-row + per-distinct-expert, fit to a rows x experts grid) predicts 2.42, so P1
  is refuted. The grid shows why: at fixed rows, time is flat in distinct experts up to ~32, then climbs with them. Row work
  and expert bytes overlap rather than add. By the registered rule, no lever lane follows; nothing changes a default.
- Host lesson: run 1's 5090 was power-capped at **450 W** and read the served arm 12 % slow while the light arm matched.
  Amendment 1 refuses cards below 575 W; adertha-agents #128 lets that refusal (exit 17) exclude its machine. Row
  `e4b.serve.p61.qwen3.b16.expert-gemv-cost-split.5090.2026-09-23`.

### K18 read (grouped-nf4-gemm): a grouped expert GEMV is exact and slower — P60's 0.92 ms is not reachable by sharing loads; its re-streaming reading is withdrawn (lane `k18-5090-1`, $0.13)

- **grouped-nf4-gemm lane K18** built the kernel P60 licensed: an int4-b32 split-K GEMV that loads each expert's slice
  once for up to four of its rows. It was run from `bench/k18/` (#687) against P60's recorded ids on an RTX 5090.
  - It is **bitwise the served GEMV** (0 of 256 replay checks differ) and **1.49× slower**: 9.689 against 6.520 ms/step.
    The dedup arm reads 5.564, reproducing P60 on a second host.
  - At R = 8/16 it is 1.00–1.65× the served call, worst where nothing can be shared.
  - Decision: not a lever. It stays dormant in grouped-nf4-gemm, and nothing here routes to it (row
    `gnf4.kernel.k18-grouped-expert-gemv.5090.2026-09-22`).
- **Correction to the P60 entry below.** The 0.92 ms/step dedup gap was read as "the cost of re-streaming each expert per
  row". The dedup arm also drops the repeated rows' arithmetic and programs, locality was already ruled out (P60's P3),
  and sharing the loads did not recover it. That reading is withdrawn in `bench/p60/RESULTS-p60.md` (dated correction),
  `docs/STATUS.md` and the P60 claims row. The number stands.

### P60 read: the expert GEMV's B=16 headroom is repeated rows — 0.92 ms/step, the ceiling a grouped kernel can recover (lane `p60-5090-1`, $0.23)

- **Recorded B=16 routing replayed through the shipped int4-b32 expert GEMV on one RTX 5090** (`bench/p60/RESULTS-p60.md`):
  the replay reproduces the served kernel row (**6.155 ms/step vs the census's 6.340**, −2.9 %), and one row per distinct expert
  instead of one per routed row runs **5.560 vs 6.479 ms/step** — **0.92 ms/step** (~8 % of the B=16
  step) is the cost of the repeated rows [first read as re-streaming each expert per row — withdrawn by the K18 read above]. Ordering rows by expert changes nothing (-0.55%): L2
  already serves the repeats. One row per expert sits 1.21× above the 1,512 GB/s byte floor (between bands).
- Decision: a **grouped expert GEMV** (grouped-nf4-gemm lane K18) is licensed to build, read against the recorded ids committed in
  `bench/p60/receipts/eids_b16.int16.bin`. Row `e4b.serve.p60.qwen3.b16.expert-gemv-repeat-cost.5090.2026-09-22`. No default changes.

### P59 amendment 1: the fused q/k/v path is the default at B=16 too — 0.0044 nats/token through a 128-row prefill, determinism control bit-identical (lane `p59b-5090-1`)

- **Through the harness's 128-row prefill, `KL(int4 unfused ‖ int4 fused q/k/v)` at B=16 = 0.0044 nats/token, top-1 0.9775**
  over 2,048 teacher-forced decode positions (one RTX 5090, $0.18; `bench/p59/RESULTS-p59.md`); the unfused stack rebuilt in a
  fresh process is **bit-identical** (KL exactly 0), so the number is the fusion's alone. The distance from the NF4 anchor moves
  by +0.0034 (band ± 0.005). All four registered conditions hold → **`--fuse-qkv` is the default on the int4 serving lanes at
  B=16 as well as B=1**; P54's fused B=16 row (0.223 ms/step, 1401 → 1429 tok/s on its box) is `licensed_by` this read.
- The prefill-last control (P4) is refuted as the amendment anticipated: the >16-row cuBLAS path on the cached bf16 weight is
  where fusing changes bits; the K16 decode path is bitwise invariant (run 1). Register rows
  `e4b.serve.p59b.qwen3.b16.{fqkv-kl,nf4-int4-kl,nf4-fqkv-kl}.5090.2026-09-22`.

### P59 read: fusing q/k/v changes nothing at 16-row decode (KL exactly 0) — the K16 kernel is bitwise invariant to it; P57's kernel attribution withdrawn (#682, lane `p59-5090-1`)

- **KL(int4 unfused ‖ int4 fused q/k/v) at B=16 = 0.000000 nats/token, top-1 1.0000, bit-identical on every one of 2,048
  teacher-forced decode positions** (one RTX 5090, $0.34; `bench/p59/RESULTS-p59.md`). The fusion was installed (48 modules,
  `Int4Linear` 192 → 96). A free A2000 probe (`bench/p59/probe/`) explains it: the K16 small-M GEMM is **bitwise invariant**
  to fusing at 2–16 rows (the same plan for every N, independent columns); only the >16-row cuBLAS path on the cached bf16
  weight differs (≤ 1 bf16 ulp). The anchors: the int4 stack sits 0.0717 nats/token (top-1 0.905) from the NF4 control.
- **P57's reading that P54's B=16 token divergence "is the K16 GEMM's" is withdrawn** (P57's RESULTS, STATUS and the P57 entry
  below corrected; also corrected: P57's first-divergence indices are not P54's). The leading hypothesis is now the harness's
  128-token prefill steps, which take the cuBLAS path.
- The determinism arm was skipped by the deadline guard, so the registered decision rule is not met and `--fuse-qkv` stays
  opt-in at B=16 on run 1 (amendment 1 flipped it — entry above). **Amendment 1** re-runs the gate with a 128-row chunked prefill (`kl_b16.py --prefill-chunk 8`),
  a K16-route census, and room for the determinism arm. Register rows `e4b.serve.p59.qwen3.b16.{fqkv-kl,nf4-int4-kl,nf4-fqkv-kl}.5090.2026-09-22`.

### P58 read: same box, current vs current — vLLM 0.30.0 decodes 1.09× (B=1) / 1.40× (B=16) faster than e4b's int4 stack; 0.29.0 vs 0.30.0 within 1 % at B=16 (#676 lane, run `p58-5090-1`)

- **The register's current-vs-current comparator against a production engine is re-pointed** (`bench/p58/RESULTS-p58.md`,
  one RTX 5090 on an EPYC 9655 host, identical prompt token ids for both engines, $0.31): e4b's current int4 stack (RTN
  int4 experts + uncalibrated int4 attention + K16 route, fused q/k/v at B=1) at **239.4 tok/s (B=1) / 1379.2 tok/s
  (B=16)** vs vLLM 0.30.0 serving Qwen's GPTQ-Int4 via Marlin at **260.3 / 1925.6** → ratios **1.087 / 1.396**, vLLM
  ahead, inside the pre-registered bands (P1 1.02–1.20, P2 1.35–1.75). vLLM 0.29.0: 1912.7 at B=16 (ratio 1.387;
  build-to-build 1.007, P3 holds); at B=1 its two engine starts read 280.4 and 259.9 (self-pair 1.079 > 1.03 →
  **DRIFT, no ratio quoted**) while 0.30.0's two starts agreed to 0.04 % — a vLLM B=1 engine-start variance on this
  host, recorded not explained. Same-box e4b int4/NF4: ×2.356 (B=1), ×2.851 (B=16). P37's 2026-09-05 rows stay as
  history; the engine advantage is understated (vLLM's number includes its serving loop). Quality quoted, never equated.
- **B=1 is host-bound across three 5090 hosts today**: the same fused-q/k/v int4 stack read 4.24 (P57, EPYC), 4.18
  (P58, EPYC 9655) and 3.69 ms/step (P54's box, 2026-09-21); STOP-1 informational, same-box ratios only.
- Register rows `e4b.serve.h2h.vllm-0.30.0.p58.qwen3.{b1,b16}.5090.2026-09-22`, `…vllm-0.29.0.p58.qwen3.b16…`, the
  build-to-build row and 20 per-arm rows (`bench/p58/p58_register_rows.py`); `docs/SERVING-THROUGHPUT.md` and
  `docs/STATUS.md` re-pointed. Receipts in `bench/p58/receipts/` (trimmed engine logs; full run private).

### P57 read: K17's fused split-K reduce is exact and SLOWER in the consumer at both batches; P54's B=16 divergence is not the glue's (#666, lane p57-5090-2)

- **`GNF4_GEMV_FUSED_REDUCE=1` costs 0.038 ms/step at B=1 and 0.217 ms/step at B=16** on the int4 serving stack
  (Qwen3-30B-A3B, one RTX 5090, A/A spreads ≤ 0.009 ms; `bench/p57/RESULTS-p57.md`). The route engaged exactly as
  asked (`_reduce_partials` gone from both fused censuses) and the tokens are identical, but `_gemv_int4_b32`'s own
  duration grows by more than the reduce it absorbs (1.415 → 1.642 ms at B=1; +7.0 % at R=128 rows, B=16): the
  separate reduce launches were overlapped in the graph, the fused epilogue sits on the critical path. K17's kernel
  stays opt-in in grouped-nf4-gemm; nothing in e4b changes. P1 and P2 refuted as registered.
- **With the round-2 glue forced OFF on both legs, fused q/k/v at B=16 still diverges from unfused on 15 of 16
  sequences** while the control's two draws are bit-identical: the glue path is exonerated (P3). *(Corrected by P59:
  the indices are not P54's, and the divergence is not the K16 GEMM's — see the P59 entry above.)*
  `--fuse-qkv` stays the default at B=1 and opt-in at B=16 until a KL/K8 read bounds the difference *(it did: P59
  amendment 1 above licenses it at B=16)*.
- **The distinct-expert count is READ (amendment 3, run `p57d-5090-1`, $0.07): 58.7 distinct experts per layer per
  decode step at B=16** (layer means 50.9–71.7, 73 steps counted on device; uniform expectation 82.4;
  7–41 of 128 experts per layer never touched). The first two attempts counted 3 warm-up steps because the
  counter's `torch.unique` synchronised under CUDA-graph capture (amendments 1–2 in `P57-PREREG.md`); v2 of
  `bench/p57/distinct_experts.py` accumulates on device and was verified under capture+replay on an A2000. Against
  #564's expert-tier byte roofline the floor at 58.7 is 4.89 ms/step vs the measured 6.34 ms `_gemv_int4_b32`
  row: the expert GEMV runs at ~77 % of roofline at its real routing. Row
  `e4b.serve.p57.qwen3.b16.distinct-experts.5090.2026-09-22`.
- Register rows `e4b.serve.p57.qwen3.{b1,b16}.{control,fr}.5090.2026-09-22` and
  `e4b.serve.p57.qwen3.b16.{nor2_control,nor2_fqkv}.5090.2026-09-22` (measured, same-box). Receipts in
  `bench/p57/receipts/`; `bench/p57/p57_register_rows.py`. Cost $0.09 (a dud-box bake failure, `p57-5090-1`) + $0.22.

### There is a licensed pack again, and its bytes exist (#658, #405)

Lane P55x built Qwen3-30B-A3B's streamed 64k calibrated int4 expert pack, dumped it as a hash-pinned
artifact, and ran the registered two-text K8 gate **on those bytes loaded back by fingerprint** — the
path a loader takes, not the live stores they were built from. It **passes both texts**: wikitext
−0.05275 ppl, c4val1 −0.06622, against an NF4 reference on that box bit-identical to bo6c's on both.

`pack_fingerprint sha256:0c9955a9f06d8326…` is in the register. The bytes are retained and were verified
after transfer by two independent implementations on two machines. A loader given that fingerprint
refuses anything else and never rebuilds from the recipe, so **the licence travels as bytes** — which is
what [#405](https://github.com/pjordanandrsn/experts4bit-qlora/issues/405) had been open on since
2026-09-06, when the machinery existed but no pack had been built, gated and kept.

**Three findings beyond the licence.**

- **The recipe reproduces across boxes, and #405's framing was drawn from an outlier.** This pack is
  byte-identical to the one lane P39 built on 2026-09-10 on a different rented 5090 under **e4b 0.35.3**,
  where this ran under **0.36.4** — the same 64 hex characters across a box and a release boundary. With
  P39's own box-1/box-2 agreement and this lane's two same-box builds, four builds across at least three
  boxes agree on every byte. P37's 11522/766 divergence is the outlier, not the rule, and the open
  question becomes what was different about that host. The remedy is unchanged: a licence still travels
  as bytes, because nothing here predicts which box is the next P37.
- **The half that cannot be pinned is the half carrying the quality.** `engines/int4_attn_calib.py` has
  no serialisation of any kind, so the 192 calibrated attention projections are re-derived on every load.
  The same pinned expert bytes with RTN attention instead **fail** c4val1 at +0.13237 — a +0.19858 swing
  and a budget failure. So `pack_fingerprint` names the experts, the unpinnable component is load-bearing,
  and that residual is now measured rather than asserted.
- **The recipe is byte-deterministic on one box.** Two builds in one session produced the identical
  fingerprint, method-map hash and row-count-vector hash. bo6c's determinism row compared mean_nll and
  counts for the 16k arm a day before fingerprints existed; this is the 64k recipe at byte level.

bo6c's verdict row stays **active and unsuperseded**, deliberately: three bo7 census rows take their
licence from it, and repointing them at this verdict would assert their pack is this pack — unverifiable,
since neither bo6c's nor bo7's bytes were kept. What the new row replaces is bo6c's *role*, not its
measurement. No gate, threshold, `min_rows`, damping or K8 budget moved.

### The five families still staged now say WHY, and each reason is a test (#648, #509)

`STAGED_NOT_WIRED` recorded which families no loader admits. For four of them it recorded no reason
at all — *"none was found; that absence is itself the thing to resolve."* Resolved: two of those four
(`jamba`, `lfm2_moe`) simply had not been tried and are now wired; the rest have a measured blocker.

`tests/test_staged_blockers.py` asserts each blocker as the CURRENT upstream or checkpoint fact, so it
**fails the day the blocker lifts**. A blocker kept as a comment rots into a stale excuse; one kept as
a test says when the family became wirable. Nothing here claims a family should stay unwired — only
what would have to change first.

- **`qwen3_vl_moe` / `qwen3_vl_moe_text` — transformers publishes no `ForCausalLM` class.** Only
  `Qwen3VLMoeForConditionalGeneration` exists and neither config is in `MODEL_FOR_CAUSAL_LM_MAPPING`,
  so admission would clear the architecture gate and then raise while BUILDING the tree, before a
  weight is read. This is also why #637/#639 could adjudicate the expert layout while the loader still
  refuses the family: the int4 planner builds its own tree and does not need a CausalLM class.
- **`jetmoe` — its tree has no `experts` submodule to replace.** The stacks sit directly on the MoE
  block as `mlp.input_linear` [E, 2I, H] / `mlp.output_linear`. That is granitemoe's *shape*, but
  granitemoe's tree declares `block_sparse_moe.experts.gate_up_proj`, so a checkpoint-side rename is
  enough there and cannot be here, where both sides say `input_linear`. The convention's `fused_prefix`
  names a path that does not exist — left as-is rather than changed to another wrong value. It is also
  a DUAL MoE (`self_attention.experts`), so a naive admission would quantize the MLP experts, leave the
  attention experts in bf16, and report success.
- **`dbrx` — flat 2-D stacks.** Each projection is one `[E * ffn_hidden, hidden]` tensor and the module
  declares it the same way; `Experts4bit.from_float` requires `[E, out, in]` and refuses anything not
  3-D. Reshaping is a real change with a real orientation decision — which is why the convention pins
  w1=gate / v1=up / w2=down in advance.
- **`axk1` — the keymap does not FIT the registry that would hold it.** #509 said admission and the
  rewriter must land together; measured now, they cannot yet. `CKPT_KEY_REWRITERS` holds callables the
  loader applies one key at a time (`rewrite(k) -> str | None`), while `rewrite_axk1_keys` takes
  `(checkpoint_keys, first_k_dense_replace)` and returns `(kept, dropped)` — because the
  `post_mlp_layernorm` rename is layer-CONDITIONAL: the released checkpoint ships that key on the dense
  layer 0 *and* on the MoE layers, and it must be dropped on one and renamed on the other. So wiring is
  "close over `first_k_dense_replace` and adapt the list-shaped keymap to the per-key contract", not an
  admission row plus a dict entry. Separately, the only released checkpoint is 1.04 TB, so no support
  row is obtainable here and an admission would be a claim the coverage gate has no evidence for.
- A final test requires every member of `STAGED_NOT_WIRED` to be named by a blocker in that file, so
  "no reason recorded" cannot reopen for a family added later.

### `jamba` and `lfm2_moe` are wired — the "no reason recorded" was that nobody had tried (#648)

`STAGED_NOT_WIRED` said of these two only that there was *"no reason recorded here because none was
found; that absence is itself the thing to resolve."* Resolved by trying them: both load unchanged.

- **Both are hybrid towers** — Jamba is Mamba/attention, LFM2-MoE uses short convolutions — and that
  was the stated caution for keeping them out (`READ_COMPATIBLE_CONVENTIONS`: *"hybrid Mamba towers
  whose NON-expert surface this loader has never placed"*). The caution was reasonable and the answer
  is that the non-expert surface is ordinary passthrough: it goes through `_assign` like any other
  dense tensor, no meta tensors remain, and the forward is finite.
- **Admitted through `SUPPORTED_ARCHITECTURES`, not by adding their conventions to
  `READ_COMPATIBLE_CONVENTIONS`.** Membership there would have carried them in on their STORAGE
  convention, and storage was never the open question — so each came in on its own row instead.
- **A dense layer must never be read as an expert, and that is now pinned.** Both families alternate
  MoE and dense layers, and the dense MLP sits in the SAME container under the SAME projection names
  (`feed_forward.{gate,up,down}_proj` / `feed_forward.w{1,2,3}`), differing only by the absent
  `experts.{e}.`. Matching one would build a one-expert stack for a layer the router never routes
  through. `expert_re` requires the index; `test_a_hybrids_dense_layer_is_never_read_as_an_expert`
  asserts it against the released key spellings.
- **Admitting `lfm2_moe` does not reopen the #648 activation hazard**: its config declares none of
  `_ACTIVATION_FIELDS`, so #650's rule would REFUSE it — it is carried by
  `_ACTIVATION_UNDECLARED_OK` with its evidence (upstream `Lfm2Moe` applies SiLU unconditionally),
  and a test pins that.
- **Evidence**, CPU/bf16, transformers 5.17.0 / torch 2.14.0 / bitsandbytes 0.50.2:
  `bench/support/rows/lfm2_moe.json` — `LiquidAI/LFM2-8B-A1B` (4.5 B params, 32 experts on 22 of 24
  layers): load 13.7 s, **22 quantized / 0 unquantized** (nf4), forward finite → **`reference-ok`**.
  `bench/support/rows/jamba.json` — `ai21labs/Jamba-tiny-dev` (222 M params, 8 experts on 8 of 16
  layers): load 0.9 s, **8 quantized / 0 unquantized** (nf4), forward finite → **`toy-ok`**, and the
  row says so: every larger published jamba MoE is 52 B+, gated, or (the 3 B "Jamba2"/"Reasoning"
  releases) ships `num_experts: 1` and is not MoE at all.
- `STAGED_NOT_WIRED` drops both; the derived counts go to 5 across 4.

### GraniteMoe's two aliases are wired — it was an omission, and the tree now says so (#648)

`granitemoehybrid` and `granitemoeshared` had a convention that claimed them, while plain `granitemoe`
was admitted and they were not. This confirms that was an omission rather than a decision, and closes it
with real-checkpoint rows for both.

- **The expert surface is the same, checked rather than assumed.** Upstream's `conversion_mapping` entry
  is the SAME three `WeightRenaming`s for all three model_types, compared entry by entry against
  transformers (`test_the_granite_aliases_share_granitemoes_converter_exactly`), so one rename tuple
  legitimately serves all three and the test fails if upstream ever splits them. They differ only
  OUTSIDE the experts: `granitemoeshared` adds a per-layer dense `shared_mlp`, `granitemoehybrid`
  additionally replaces most attention layers with Mamba — both pure passthrough here.
- **Admission alone would have loaded nothing.** `LEGACY_KEY_RENAMES` is keyed on **model_type**, so
  without an entry each, `block_sparse_moe.input_linear` never becomes `experts.gate_up_proj`, every
  layer looks dense, and the load ends in the zero-expert-stacks guard. Admission and the rename land
  together, and `test_an_admitted_prefused_family_cannot_be_missing_its_legacy_renames` now asserts
  that mechanically for any family admitted on this convention. Same shape as #509's axk1 point.
- **Evidence — both `reference-ok`**, CPU, bf16, transformers 5.17.0 / torch 2.14.0 / bitsandbytes 0.50.2:
  `bench/support/rows/granitemoehybrid.json` — `ibm-granite/granite-4.0-h-tiny` (4.0 B params, 64 experts,
  40 MoE layers, Mamba/attention hybrid): load ok 23.0 s, **40 quantized / 0 unquantized** (nf4), forward
  finite. `bench/support/rows/granitemoeshared.json` — `ibm-research/moe-7b-1b-active-shared-experts`
  (3.5 B params, 62 experts, 40 MoE layers): load ok 23.7 s, **40 quantized / 0 unquantized** (nf4),
  forward finite.
- `STAGED_NOT_WIRED` drops both; the registry's derived counts go to 7 across 6.

### `nemotron_h` is wired: the loader admits it, and its renames now reach the expert path (#648, #509)

The wire-or-remove decision #648 exists for, taken for the first family, with a real published
checkpoint behind it rather than a fixture.

- **Admitted.** `SUPPORTED_ARCHITECTURES` carries `nemotron_h -> mixer.experts`. `has_gate` comes
  from the convention, so the read stacks `up_proj` alone (`down(act(up(x)))`, no SwiGLU gate) instead
  of fusing a gate that does not exist, and #650's `_expert_activation_name` resolves the activation
  from the family's own `mlp_hidden_act` (`relu2`) rather than the old `"silu"` default.
- **Admission alone loaded ZERO experts, and that is the substance of this change.** Nemotron-H ships
  every tensor under `backbone.` while the tree declares `model.`. The loader anchors expert indexing
  on `^model\.layers\.` and builds every fused-target lookup from `model.layers.{i}.{expert_rel}.`, so
  with the family merely admitted `_index_per_expert_keys` returned `{}` against the real checkpoint:
  every MoE layer read as dense and the load died in the zero-expert-stacks guard — while its
  NON-expert keys mapped fine, because the `_assign` pass applies `conv.rename` and the expert path
  never did. Renames meant two different things depending on which branch a key took. That is exactly
  the defect #643/#644 fixed in the PLANNER; this is the same fix on the loader side.
  `loader._rename_ckpt_prefixes` applies a convention's renames to the part of a key BEFORE
  `layers.N.`, which is the planner's own narrow rule — a blanket rename of the whole key would rewrite
  the CONTAINER too, and mixtral's `.block_sparse_moe.` -> `.mlp.` would then destroy the substring
  `MIXTRAL.expert_re` matches on, un-recognising every mixtral expert key. Asserted to be a no-op for
  all twelve families admitted before it.
- **Evidence — two rows, both on real published checkpoints.**
  `bench/support/rows/nemotron_h.json` is `inference-optimization/NemotronH-0.3B-A0.3B` (32 routed
  experts, 2 of 5 layers MoE): load ok, `verify_moe_4bit(strict=True)` 2 quantized / 0 unquantized
  (nf4), forward finite — graded **`toy-ok`**, because at 323 M parameters it is below the probe's 1 B
  reference bar and the row says so rather than overstating it.
  `bench/support/rows/nemotron_h_30b.json` is `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
  (128 routed experts, 23 of 52 layers MoE, 14 shards): integrity clean (every shard the length its own
  header declares), load ok 155.2 s, **23 quantized / 0 unquantized** (nf4), forward finite —
  **`reference-ok`**, which is the grade `coverage-baseline.json` records. Both CPU, bf16, transformers
  5.17.0 / torch 2.14.0 / bitsandbytes 0.50.2. The small row is kept rather than replaced: it is the
  one whose two-MoE-layer shape the prefix-rename regression test is written against.
- `STAGED_NOT_WIRED` drops `nemotron_h` in the same change — `tests/test_staged_not_wired.py` fails in
  both directions, so wiring without delisting could not have merged. The registry's own count is now
  derived from the set and asserted (`test_the_stated_counts_match_the_set`): it read "ten model_types
  across seven conventions" when the conventions were **eight** — wrong the day it was written, and
  invisible because nothing read it. It is 9 across 7 now, checked.

### Measured (no default changes)

- **Lane P54 — fusing q/k/v on the int4 attention store** (`bench/p54/RESULTS-p54.md`, receipts in
  `bench/p54/receipts/`, one RTX 5090, $0.47). `Int4Linear.fuse` (0.36.4, #651) measured on Qwen3-30B-A3B,
  two interleaved draws per arm: **B=1 4.210 → 3.693 ms/step (0.516 ms, 12.4 %; 237.5 → 270.8 tok/s) with
  token-IDENTICAL output**, and B=16 11.420 → 11.197 ms (0.223 ms, 2.0 %) where the fused arm's tokens
  **diverge** from the control's on 14 of 16 sequences, reproducibly, against a bit-identical A/A. The K16
  small-M GEMM's accumulation order is a function of N, so the fused N=5120 launch rounds differently from the
  three it replaces *(explanation withdrawn by the P59 read above: the K16 decode path is bitwise invariant to
  fusing; the >16-row cuBLAS prefill path is where the bits change)*. **`--fuse-qkv` stays opt-in at B=16 and no B=16
  position is quoted** *(superseded by P59 amendment 1 above: licensed at B=16)* until a
  KL-from-checkpoint or K8 read bounds the divergence (bar: ≤ 0.10 nats, top-1 ≥ 0.93); B=1 is a licensed
  lever. Four register rows `e4b.serve.p54.qwen3.*.5090.2026-09-21`. The lane's distinct-expert arm (#564's
  unmeasured number) was **unbuildable as registered** — `--series-out` requires `--amort on`, which the
  captured B>1 stage refuses — and is re-registered in the lane's amendment 1.

## 0.36.4 — 2026-09-21 — the loader's three arrows measured against upstream (fused layout, conditional transpose, rename reach) and an activation that no longer defaults; the int4 attention stack can fuse q/k/v (P54 measures it); P52 and P53 read on Gemma-4

### The loader: three arrows measured against upstream's own code, and each one found a defect

None of these came from a wrong number in the field. Each came from executing upstream's code beside e4b's on the same
tensors and reading the disagreement. Nothing here changes which families load.

- **The expert activation is read from the family's own field, or refused** (#650, #648). The lookup was
  `hidden_activation`, then `hidden_act`, then a `"silu"` DEFAULT. `nemotron_h` declares neither and names
  `mlp_hidden_act: relu2`; it is non-gated, so `down(act(up(x)))` with SiLU instead of ReLU² is a different function with
  every shape agreeing, and the existing guard (unknown activation NAME) could not see a field it never read.
  `_ACTIVATION_FIELDS` is the ordered list (adds `mlp_hidden_act`, `activation_function`); a config that declares none of
  them raises `MoEConventionError` unless the model_type is in `_ACTIVATION_UNDECLARED_OK` with its evidence
  (`qwen3_omni_moe`, `lfm2_moe`, `dbrx`). Every admitted family's transformers default config resolves under the new rule
  (39 of 39, asserted by test); the load log names the field the activation came from.
- **`STAGED_NOT_WIRED`** (`arch/moe_conventions.py`): the ten model_types that have a convention here and that no loader
  path admits — `axk1`, `nemotron_h`, `granitemoehybrid`, `granitemoeshared`, `qwen3_vl_moe`, `qwen3_vl_moe_text`,
  `jamba`, `lfm2_moe`, `jetmoe`, `dbrx` — asserted equal to the loader's own refusal in both directions, so wiring one
  without delisting it fails and adding a convention nothing admits without listing it fails too. (`axk2` has no
  `SUPPORTED_ARCHITECTURES` row either but aliases onto `qwen2_moe` and IS admitted.) The wire-or-remove decision on #648
  and #509 stays open.
- **The fused gate/up layout is verified against upstream's own expert forward** (#630, closes #515).
  `arch/fused_layout_probe.py`: a gated expert's output is affine in `up` and nonlinear in `gate`, so scaling one index
  set of the fused axis and taking the second difference finds the layout on CPU, with no checkpoint and no GPU, for any
  activation; no affine set or more than one raises `FusedLayoutUndetermined` rather than guessing. All 13 gated
  conventions measured: contiguous gate-first everywhere except `gpt_oss`, which is interleaved. Three pre-existing
  defects: (a) `gptoss` DECLARED contiguous gate-first and is interleaved — `fused_order` had two values for a layout space
  of at least three; new field `ckpt_gate_up_packing`, and gptoss declares it; (b) the #518 refusal in `expert_layout_for`
  was raised inside a `try` whose `except MoEConventionError` fell back to `SUPPORTED_ARCHITECTURES`, so it was INERT for
  every natively pre-fused family — the `try` now wraps only the lookup; (c) the exposed set is SEVEN (`NATIVELY_PREFUSED`:
  granitemoe, gptoss, qwen3_vl_moe, gemma4, jetmoe, qwen3_5_moe, axk1), not the six or nine counted before.
- **The pre-fused transpose is conditional, because upstream's is** (#639, closes #637). Upstream's converter is
  `Transpose(check_dims=True)`: transpose only if the checkpoint tensor and the module parameter differ. e4b's
  `transpose_last2` was unconditional, so on a SQUARE expert stack (`2 * moe_intermediate_size == hidden_size`) e4b loaded a
  transposed tensor upstream would not — same shape, different values, nothing raised. `make_plan_reader(param_shape=)`
  replicates the condition; `execute_moe_plan` supplies the shape from the model and the int4 serve lane from its meta
  twin; a square stack with no expected shape is REFUSED. `tests/test_converter_arrow.py` (43 arms) runs upstream's real
  `ConversionOps` beside e4b's read path over the same tensors for eight conventions.
- **A convention's renames reach the expert fused target too** (#644, closes #643). A passthrough key went through
  `conv.rename`; a per-expert key had its fused target built from the RAW checkpoint prefix, so a family shipping
  `backbone.layers.N.` where the tree declares `model.layers.N.` (nemotron_h) mapped the two branches inconsistently.

### Serving: the int4 attention stack can fuse q/k/v (lane P54 measures it)

- **`fuse_qkv` takes the int4 store** (#651; #652). Every int4 serving lane applies `enable_serve_attn_int4` at load and
  `fuse_qkv` after it, and `fuse_qkv` read `.weight`, which an `Int4Linear` lacks -- so the two were exclusive and every
  quoted int4 census (P42, K16 P5, bo7) ran `--no-fuse-qkv`, paying q, k and v as THREE attention launches per layer
  where the bf16 stack pays one (at M=16 `k_proj`/`v_proj` sit within 1 us of the 4.6 us launch floor). New
  `Int4Linear.fuse(mods)` / `Int4Linear.from_packed(...)`: the parts' packed rows and scales concatenated along N,
  byte-identical, so the fused projection computes the parts' function on one GEMV (rows == 1) or one K16 small-M GEMM
  (rows 2..16, N = 5120 on Qwen3-30B). A mix of int4 and dense q/k/v is refused, never half-fused. **No default
  moves**; `bench/p54/P54-PREREG.md` registers the measurement (B=16 saving 0.25-0.50 ms/step predicted, B=1
  0.20-0.45, K16 calls 192 -> 96, plus the distinct-expert count #564 names as the unmeasured number) before the lane runs.

### Measured this release (no default changes; nothing licensed)

- **P52 — the Gemma-4 graded store map's gate ran properly, on held-out prompts, and DID NOT PASS** (#621, #622, #623;
  `bench/p51/RESULTS-p51.md`). The K8 two-text gate 0.36.3 promised cannot be built on this family: Gemma-4's own NLL
  moves 0.4 nats with batch shape against K8's 0.05 budget (`e4b.parity.gemma4.no-reference`), and the K8 runner needs
  the arena path, which refuses a per-layer map. Replacement bar, registered on `main` four minutes before the run: KL from
  the bf16 checkpoint ≤ 0.10 nats AND top-1 ≥ 0.93. `bench/kl_prompts_heldout.py` holds 100 NEW prompts (same strata,
  disjointness from the committed 200 asserted — the first build had 6 overlaps). Result: graded map **0.1319 nats /
  top-1 0.874**, failing both axes on every stratum, while the bar's own provenance point (gpt-oss NF4) moved 2 % between
  prompt sets and Gemma moved 20 % — the instability is the family. The map stays a documented option; **no Gemma-4
  default ships and no position is quoted**.
- **P53 — calibration does not rescue Gemma-4's experts, and sequential is WORSE** (#638, #640, #645, #646, #649; closes
  #636; `bench/p53/RESULTS-p53.md`, one H100, $3.14). All 30 expert layers quantised: NF4 RTN **1.0772** nats KL vs the
  bf16 checkpoint, GPTQ int4 all-at-once 1.1050, GPTQ int4 sequential **1.1564** (top-1 0.644 / 0.642 / 0.630). Order was
  the only difference between the calibrated arms. Both axes on #636 — reduce the perturbation (P49) and make the
  downstream absorb it — are now closed; the one lever that works is keeping the early expert layers in high precision,
  and that lever is memory. Register row `e4b.quality.gemma4.calibration-refuted`.
- **The Gemma-4 TRAINING-path parity failure is in the register before #558 closed** (#635, `e4b.parity.gemma4.train-internal`):
  e4b's fused expert path against e4b's own dense per-expert reference, same box and tokens, ends **0.08257 nats** apart on
  held-out loss against the 0.05 band on the 0.32.1 kernel cut (0.09037 on the previous cut) — the failure survives the
  kernel change, so it is the family, not the kernel. The `fused` arm is not quoted for Gemma-4.

### Tests and bench harness

- The paged-attention end-to-end test's tolerance follows the compute mode that ran (fp8 default on sm_89+ gets the
  kernel package's own 1.5e-1; f32 keeps 2e-2) and its draw is seeded (#626, closes #341).
- p41 driver: a run whose registered failure criteria fired can no longer be published as a pass (#627, closes #495).
- tp4 HF arm selects expert parameters by STRUCTURE and refuses an empty selection (#628, closes #542).
- tp4 harness: the stall alarm tells a model fetch from a hang (#625, closes #624); the prologue names where its time went
  and its residual cannot round negative (#629, #633, closes #548); "any byte" is not a stall threshold and a dead lane is
  not a slow one (#634); amendments 5 and 6 register a parity-only box C and an axolotl arm with a proof-of-work predicate
  (#631, #632).
- `bench/p47/staged.sha256` is checked in CI rather than on the controller after a box is rented (#642); `kl_serve
  --family` has the P53 family's REFERENCE entry, found only after a rented box had fetched 52 GB (#647).

## 0.36.3 — 2026-09-19 — a per-layer expert STORE MAP; the field-recipe training position on the new kernel cut (Qwen3-30B-A3B 4.49x Unsloth, parity-gated); Gemma-4 explained end to end

### The loader takes a per-layer store map

- `load_moe_4bit_streaming(quantize_layers=...)` accepts a **mapping** `{layer: spec}` beside the existing `None` and set forms.
  A spec is a scheme name (`"int8"`), a `(scheme, blocksize)` pair, a `{"quant_type": ..., "blocksize": ...}` dict, or `None` for
  the base dtype; layers the mapping does not name stay in the base dtype. `layer_store_spec()` is the documented selector, and a
  map is **refused** on the arena and dedicated-quant paths, which carry one store for the whole model. `blocksize` is also a
  first-class argument of the direct path now. Why: expert layers are not equally sensitive to quantisation — on Gemma-4 the span
  is **159x** — so one store for the whole model is the wrong shape for some families (#597).

### Measured this release (no default changes, nothing silently applied)

- **Training, the field recipe, on grouped-nf4-gemm 0.32.1** (`bench/tp4/RESULTS-tp4-p46cut.md`, 21 register rows
  `e4b.train.h2h.unsloth.*.2026-09-19`): Qwen3-30B-A3B on one RTX 5090, both frameworks training the same 642,514,944
  parameters — e4b `fused_attn4` **6.4707 s/step against Unsloth's 29.0547 (x4.490)**, 237 vs 48 tok/s, 1186 vs 3341 J/step,
  the same peak VRAM, held-out delta 0.0283 nats (COMPARABLE). On the previous cut this arm did not finish. e4b's fused path
  passes its own parity control against its dense reference on **every** family measured (0.00063 / 0.00203 / 0.00140 /
  0.00159 against a 0.05 band), which was the registered condition for this release to proceed. Granite 2.853 -> 2.366 and
  OLMoE 2.739 -> 1.395 on the same cut. Four comparisons are deliberately NOT quoted, each with its reason in the coverage row.
- **Gemma-4 (#597), explained end to end across four lanes** (`bench/p47`-`bench/p51`): the serving stack is innocent; e4b's
  modelling is faithful (0.0056 nats with 29 of 30 expert stacks bf16); the cost is NF4 on the EARLY expert layers and the
  sensitivity is **positional** — the same ~8 % expert-branch damage costs 159x more at layer 0 than at layer 27, while an 8.5x
  smaller damage at layer 0 buys 22 %; no store rescues layer 0; the tail cannot be crushed further (704 = 64 x 11). What works
  is a **graded map** — `{0..9: None, 10..19: "int8", 20..29: ("nf4", 64)}` — which at matched bytes is **1.44x better than a
  uniform high-precision head** (0.1695 nats at 25.70 GB vs 0.2448 at 25.21 GB). **No Gemma-4 position is quoted and no default
  ships**: the K8 two-text gate on the served stack is still owed.

### Also

- **CI pins grouped-nf4-gemm at the v0.32.1 release commit** (`9206352f`; the `[fast]` floor `>=0.30.0` is unchanged) and the
  system manifest is the v0.32.1 copy (`consumer_ci_pin` prose now names v0.32.1). What 0.32.1 changes for e4b training: the
  grouped-LoRA delta's `auto` path now pads unless the padded block would not fit (P46, `bench/p46/RESULTS-p46.md`: 4.22 vs
  24.46 s/step at Qwen3-30B-A3B's field recipe, same loss, same peak VRAM). **No training position moves from this entry** --
  the tp4 box-B head-to-head re-runs on the 0.32.1 commit and the position, if any, is quoted from that receipt.
- `bench/tp4/tp4_run.sh`: refuses before any fetch when the instance overlay has under `TP4_MIN_DISK_GB` (200) GB free (P48
  run 1 died ENOSPC on a 32 GB overlay; the launcher orders machine disk, the instance overlay is what the box gets).

## 0.36.2 — 2026-09-19 — the K16 small-M int4 attention route ships and defaults to `auto` (−1.06 ms/step at B=16 on the 5090, P5 read); the P43 read (T1: no collapse, host-bound; T2b: #558 is a per-family band); the P44 and P45 instruments

### K16 route: `Int4Linear` serves 2..16 rows with grouped-nf4-gemm's small-M int4 GEMM (#578, #587; lane K16, #561)

- `Int4Linear(smallm=True)` routes `1 < rows <= 16` to `int4_smallm.gemm_int4_b32_smallm` (grouped-nf4-gemm ≥ 0.32.0) on the
  SAME packed bytes, split-K workspace preallocated at construction (capture-legal), **no cached bf16 copy** for those rows
  (#561). One row keeps the int4 GEMV; more than 16 rows keep the cached bf16 matmul.
- **Default `auto`** (`resolve_smallm`, applied to BOTH `enable_serve_attn_int4` and `enable_serve_attn_int4_calib`): the route
  is ON when the installed kernel package carries `int4_smallm`, OFF with a one-line banner when it does not — never a silent
  fallback, never a refusal on an older cut. `E4B_ATTN_INT4_SMALLM=1` requires the kernel (refuses without it at enable time),
  `=0` keeps the cached-bf16 path. The `[fast]` floor stays `grouped-nf4-gemm>=0.30.0`; CI pins the kernel at the v0.32.0
  commit (`8b1acc9e…`).
- **Why the default moved — the K16 P5 read** (`bench/k16/RESULTS-k16-p5.md`, lane `k16-p5`, receipt `2026-09-19/k16-p5/`,
  P42's census protocol on one RTX 5090): `int4_b16` 12.25 ms/step → `int4_b16_smallm` **11.19 ms/step** (−1.06 ms, −8.6 %;
  1,306 → 1,429 tok/s); the bf16 GEMM family that carried the attention projections falls by 2.35 ms/step and the K16 kernel
  costs 1.30 in its place (net −1.05; the microbench predicted −0.96); the route appears in the smallm arm's census (192
  calls/step) and in no other arm's. **P5 HOLDS** (≥ 0.4 registered). B=1 is untouched, so every K8 row on record is
  unaffected; at 2..16 rows the route computes the same `x_bf16 @ dequant(W)` with fp32 accumulation, within bf16 rounding.
  Kernel side: grouped-nf4-gemm 0.32.0 (`gnf4.kernel.k16-smallm-int4-gemm.5090.2026-09-19`, measured; P4 untested there).
- `bench/k16/`: the K16 5090 runner (installs pytest on the box, #579), the P5 census lane (`k16p5_run.sh`, `k16p5_drive.sh`,
  `k16p5_reduce.py`, #584).

### P43 read (#582): T1 — no collapse, a host-bound step; T2b — #558 is a per-family parity band, not a kernel defect

- T1 (`p43-t1-qwen3-2`, 5090, three arms at the field fixture): `fused_attn4` **29.30 s/step**, `fused_bf16attn` 29.47,
  `fused_attn4_mb1` 50.65; ρ 0.70 / 0.67 / 0.16 — a +15 % drift over 20 steps at mb2, no RAMP, no CLIFF; tp4's collapse
  (24.7 s → > 200 s) did not reproduce. Int4 attention is not the tier (P2). **P4 refuted: the card sat at 16 % utilisation /
  113 W from step 0** — host-bound the whole run. The follow-up is P45 (below), not the RAMP/CLIFF lanes the rule named.
- T2b (`p43-t2b-g4sweep-2`, H100 NVL; run 1 walked the oracle's vision tower, #580): 30 text-decoder layers, 23,314 positions,
  rms(fused − reference) 0.0044 → 0.42 (**P7 holds**); the fused/reference-to-oracle ratio stays in [0.988, 1.007] on every
  layer (**P8 holds**: neither path is closer to bf16 anywhere). Under the registered rule **#558's remedy is a re-derived
  per-family parity band, not a kernel fix**; no Gemma-4 training position is quoted until that step closes.

### P44 instruments (#581, #583, #585): OLMoE two-text K8 + per-expert census, KL-from-bf16 for the families K8 cannot read

- `bench/p44/serve_stack.py` (ONE table of the arms bo7 timed + `build_served_model()` mirroring `step_decomp`'s all-VRAM
  assembly with an engagement census), `kl_serve.py` (K0-gated; reference scored once decode-shaped, cached, freed; **one child
  process per arm** because the lane hook arms only when a lever flag is set at interpreter start; a set lever that engaged
  nothing refuses the row; gpt-oss's dequant reference proven by its weights), `expert_residuals.py` (per-expert
  `sqrt(tr(DHDᵀ)/tr(WHWᵀ))` for RTN and GPTQ from the recipe's own Hessians), `p44_reduce.py` (k8_gate two-text rule, the KL
  reading rule, P3 tail statistic; a missing row is NOT_READ), runners/drivers, amendments 1–2 in `P44-PREREG.md` (2048-step K8
  window; decode-shaped scorer; the 1.1-nat Gemma-4 rows of run 2 disclosed as unexplained with two controls added).
- First OLMoE rows (`p44-a-olmoe-2`, quoted here as observations; the register rows follow the reducer): `int4all` c4val1
  **+0.255 ppl** (FAIL two-sided), `calibexp_all` (streamed 64k on wikitext-train) wikitext −0.055 / c4val1 **+0.443** (FAIL
  one-sided) — the Qwen3 recipe does not transfer to OLMoE; its licensed position stays NF4.

### P45 instrument (#586, #588): where a training step's host time goes

- `bench/tp4/tp4_arm.py --profile-steps K --profile-warm W` wraps K optimizer steps of the shared loop in `torch.profiler`
  (device-busy fraction, device events/step, CPU op self time by op family, top rows → `<receipt>_profile.json`; profiled steps
  are flagged, the timed number never comes from them); `tp4_run.sh` box E (`qwen3prof`: e4b + Unsloth at the field recipe with a
  1 s `nvidia-smi dmon` sampler); `bench/p45/P45-PREREG.md` + `p45_reduce.py`.

## 0.36.1 — 2026-09-18 — ernie4_5_moe loads its released checkpoint: tensors the text model does not build are skipped when the modeling class declares them, refused by name otherwise (#529)

**A released checkpoint that ships a speculative-decoding block now loads.**
ERNIE-4.5-21B-A3B-PT carries 12 multi-token-prediction tensors its text model
does not build; the loader died on the first of them with a bare
`AttributeError`. It now honours the modeling class's own
`_keys_to_ignore_on_load_unexpected` — what transformers' `from_pretrained`
does with those keys — and skips them before reading a byte, and any other
tensor with no module is refused by name with the two declared-drop routes.
Affects every family whose class declares such patterns (ERNIE-4.5 MoE and
DeepSeek-V4 today) and, only as a clearer error, any checkpoint carrying a
tensor the model does not build; nothing changes for a checkpoint whose every
tensor has a home, and no gate, threshold, floor or registered number moves.
**Verified on the released ERNIE-4.5-21B-A3B-PT** off the LAN (CPU, bf16):
9 shards integrity-clean, load 81.6 s, 12 tensors skipped, 27 keys renamed,
27 of 27 MoE layers quantised (nf4), forward finite — a `reference-ok` row in
`docs/ARCHITECTURE_SUPPORT.md`. Upgrade if you load ERNIE-4.5 MoE or any
checkpoint with an MTP / next-token block; no action otherwise. A patch
release: the `[fast]` extra stays at `grouped-nf4-gemm>=0.30.0` and the CI
kernel pin stays at v0.31.0.

### A checkpoint tensor the text model does not build: skipped when the modeling class declares it, refused by name otherwise (#529)

- ERNIE-4.5-21B-A3B-PT's released index carries a multi-token-prediction block — 12 tensors
  under `model.mtp_block.0.*`, `model.mtp_emb_norm.*`, `model.mtp_hidden_norm.*` and
  `model.mtp_linear_proj.*` — that `Ernie4_5_MoeModel` does not build. The loader walked to the
  first of them and died with `Ernie4_5_MoeModel has no attribute `mtp_block``, a bare
  `AttributeError` naming neither the key nor a remedy. `docs/ARCHITECTURE_SUPPORT.md` read
  `validated` for the family on fixture evidence; the fixture had no MTP weights, which is exactly
  the gap that row's own caveat warns about.
- The loader now honours the modeling class's own `_keys_to_ignore_on_load_unexpected` at the
  non-expert assignment pass — what transformers' `from_pretrained` does with those keys (ERNIE-4.5
  MoE declares `["mtp"]`, its modeling file saying "Not supporting multi-token prediction (MTP)
  atm"; DeepSeek-V4 declares `["(^|\.)mtp\..*"]`). A key with no module that matches is skipped
  before its shard is read and counted in the log (`skipped N checkpoint tensor(s) the text model
  does not build (<Class>._keys_to_ignore_on_load_unexpected)`); a key with no module that matches
  nothing raises `loader.UnplaceableTensorError` — an `AttributeError` subclass, so what caught the
  old exception still catches it — naming the key and the two declared-drop routes
  (`CKPT_KEY_REWRITERS`, the convention's `drop_re`). Family-agnostic: no per-family table, and
  DeepSeek-V4's bespoke `mtp.` rewriter stays as the first line.
- `tests/test_unbuilt_checkpoint_tensors.py`: the family's own tiny model plus tensors under the
  released MTP prefixes loads, counts the skipped tensors, and forwards; an *undeclared*
  unplaceable tensor is refused by name (the false-accept probe — the skip is gated on the
  declaration, not on "anything with no module"); the patterns are read from the class (ERNIE
  yes, Qwen3-MoE none); `_assign` refuses by name.
- **Verified on the released checkpoint** (2026-09-18, `bench/support/rows/ernie4_5_moe.json`,
  `bench/support/support_probe.py` on CPU, bf16, transformers 5.17.0 / torch 2.14.0 / bitsandbytes
  0.50.2): ERNIE-4.5-21B-A3B-PT off the LAN — 9 shards each matching the length its own header
  declares; load 81.6 s with `skipped 12 checkpoint tensor(s) the text model does not build
  (Ernie4_5_MoeForCausalLM._keys_to_ignore_on_load_unexpected)` and 27 transformers checkpoint-key
  renamings (`moe_statics`); `verify_moe_4bit(strict)` 27 quantised / 0 unquantised (nf4); one
  forward finite (loss 5.17 on synthetic ids — a finiteness smoke, not a quality measure); grade
  **`reference-ok`**. CUDA-graph capture not tested (CPU probe). The family loads by convention
  (`QWEN2_MOE`), not through `SUPPORTED_ARCHITECTURES`, so the row sits under "probed but not in the
  claimed list" by design; `docs/ARCHITECTURE_SUPPORT.md`'s hand-written row is now real-checkpoint
  evidence and its Notes say so.

## 0.36.0 — 2026-09-18 — the licensed int4 expert pack is bytes with its gptq/rtn decision recorded (#405, #530, #537); the fused gate/up order and the attention projections are declared by structure (#518, #519, #426)

**Two things stop being recipes and become records.** The calibrated int4
expert pack is an artifact with a root fingerprint, and the per-expert
gptq/rtn decision travels inside it, so a licensed pack loads byte-for-byte
on another box and a re-pack honours the recorded split instead of
re-deriving it from routing counts at the noise floor (#405, #530); a
streamed build's record now covers every chunk, not the last one (#537). The
fused `gate_up_proj` orientation is a declared, validated field on every
convention and an up-first family is refused at the loader instead of
computing `up * act(gate)` silently (#518, #519); attention projections are
found by module structure, so Gemma-4's `k_eq_v` layers are counted rather
than skipped (#426). Affects the int4 serving lanes
(`enable_serve_experts_int4*`, `dump_calibrated_artifact`,
`E4B_INT4_ASSIGNMENT`), attention-4-bit and attention-LoRA on every family,
and — only as a refusal that no shipped convention triggers — every fused MoE
load. Upgrade if you build or consume calibrated int4 packs, target
attention on Gemma-4, or pin this package by version: the 0.35.3 wheel on
PyPI predates `detect_attention_projections`, `pack_manifest` and the
assignment API, so that version string named two code states (#488) and this
release closes it. No gate, threshold, floor or existing claim value moved.
The `[fast]` extra stays at `grouped-nf4-gemm>=0.30.0`; CI pins the kernel
package at its v0.31.0 release commit.

### The gptq/rtn decision travels with the pack; a re-pack honours it (#530)

Mechanism only. No gate, threshold, floor, `min_rows`, damping, or existing claim value moved; no licence granted or withdrawn. The P37 VOID stands until a pack built from a recorded assignment passes the two-text gate on a second box.

- **Why**: the per-expert gptq/rtn choice is `routed_rows >= min_rows`, a threshold on a quantity at the router-flip noise floor, so it does not reproduce across boxes (P37: 10 of 12,288 experts flipped, `11522/766` vs the licensed `11512/776`; every e4b arm VOID). `#405` made the licensed **bytes** reproducible by artifact; it recorded only a *hash* of the decision, so a re-pack elsewhere could verify a mismatch but never avoid one.
- **`pack_manifest`**: the decision (`method_map` per `(layer, expert, role)` + `row_counts` + the `min_rows` that created it) is written as a **hashed payload** `payloads/assignment.json`, so the root `pack_fingerprint` covers it; the manifest's `method_map_hash` is derived from that payload and `verify_artifact` refuses a manifest copy that disagrees. `read_assignment(artifact_dir | assignment.json)`, `assignment_index` (duplicate keys with different methods refuse). `#405` artifacts without the payload still verify and load.
- **`enable_serve_experts_int4(assignment=...)`**: honours the record and does **not** consult `min_rows`. Refusals, never silent fallbacks: an expert the record names `gptq` that this box's calibration never routed to (no Hessian) refuses; an expert the record does not name refuses; an assignment without Hessians refuses. Where the record disagrees with what `min_rows` would have picked locally, that is **counted and reported** (`INT4EXP assignment honoured <hash>: N expert-roles where local routing disagrees`; provenance `assignment_honoured`), never applied.
- **`enable_serve_experts_int4_calibrated(assignment=...)`** and env **`E4B_INT4_ASSIGNMENT`** (file or artifact dir) so a lane hook can pin the licensed split without a code change; the argument wins over the env; a bad path refuses. Absent both, the recipe decides — unchanged.
- **Provenance and dump**: the live record now carries `method_map` and `row_counts` as lists (not only their hashes); `dump_calibrated_artifact` writes them as the hashed payload; a licensed `enable_serve_experts_int4_from_artifact` load carries them back, so dump → load → dump reproduces the same `pack_fingerprint`.
- **What it does not do, said plainly**: it fixes the *classification*. GPTQ output also depends on the Hessian, which routing flips also perturb, so bytes still reproduce only via the artifact. The experiment this enables is the one the issue asks for: re-pack on a second box from the recorded assignment and run the K8 two-text gate — if c4val1 still fails, the split was not the cause.

### Pack-manifest: licensed int4 experts are bytes, not a recipe (#405)

Code and register contract; no gate, threshold, floor, `min_rows`, damping, or existing claim value moved. Qwen3 licensed 238.1 / 1327.5 stay. No `pack_fingerprint` hashes invented for existing licence rows (the bo6c bytes were not retained).

- **`experts4bit_qlora.engines.pack_manifest`**: canonical JSON manifest, per-payload sha256/size, root `pack_fingerprint = sha256:<64 hex>` over ordered `(path, size, sha256)`. Serialize/load packed tensors + scales. Verify refuses corruption, missing files, wrong model revision / layout / fingerprint.
- **`enable_serve_experts_int4_calibrated`**: an `expected_fingerprint` loads the artifact and **refuses** a mismatch — no recipe fallback. Recipe builds remain observations; `dump_artifact_dir` writes them. Live calibrated packs attach observed provenance (`pack_fingerprint`, component hashes, method-map hash, row-count-vector hash, calibration-token SHA, toolchain) onto the model for decode receipts (`step_decomp.py`).
- **Review fixes (2026-09-06)**: the manifest's identity fields (`schema_version`, `layout`, `model_id`, `model_revision`, `layers[]`) are written a second time as a hashed payload `payloads/identity.json`, so the root `pack_fingerprint` covers which checkpoint and layout the bytes belong to; `verify_artifact` refuses a manifest whose top-level copy disagrees with the hashed one. The licensed loader takes N/K from the hashed payload bytes (`packed [E, N, K//2] uint8`, `scales [E, N, K//32]`) and treats `layers[]` only as a cross-check that refuses on disagreement. `dump_calibrated_artifact` refuses an unknown `model_revision` unless `allow_unknown_revision=True` (then the pack carries `model_revision_missing: true`).
- **p37 reducer**: when a lane names `expected_pack_fingerprint` / `E4B_EXPECTED_PACK_FINGERPRINT`, licensed arms VOID on exact fingerprint mismatch; legacy count-banner VOID stays for receipts without an expected hash. Anchored P37/bo6/bo7 receipts untouched.
- **Claims contract**: optional `pack_fingerprint` in `docs/claims-schema.md`; `scripts/check_claims_register.py` regex-checks the format and requires the field to match across an ACTIVE `licensed_by` pair once either side carries it. Step (4) determinism lane is not this PR.

### Serving census rows name the comparator (#418)

Register wording only; no value, status, gate, threshold or floor moved. The owner's ruling: the Qwen3 licensed 238.1 / 1327.5 rows stay.

- Every `e4b.serve.census.bo7.*` speed row's `unit` and `claim` name the comparator as **vs e4b's own NF4 control on the same box**, never a bare ×N speedup. Granite, OLMoE, gpt-oss, Gemma-4 and Mixtral notes carry **no field comparator measured**. Qwen3 names the P37 vLLM 0.28.0 GPTQ-Int4 / MarlinExperts comparator (footprint not recorded) and scopes the licensed position to the bo6c pack artifact (11512 gptq / 776 rtn); #405 is a notes reproduction item, not a licence withdrawal. The P37 root row is bounded to graph decode at B=1 and B=16 on one box and one prompt set.
- No structured `comparator` field (the register validator does not check one). `docs/STATUS.md` and `docs/SERVING-THROUGHPUT.md` hand-edited; README results table updated to the same wording.
- `llms-full.txt` regenerated.

### Attention 4-bit + LoRA: detect projections by STRUCTURE (#426, #412)

`quantize_attention_projections_4bit` and `add_attention_lora` now share one
detector (`detect_attention_projections`). Admission is by module STRUCTURE,
never family name: `q_proj`, `k_proj` and `o_proj` must be supported linears;
`v_proj` may be a supported linear or absent/`None` (Gemma-4 `attention_k_eq_v`
layers, where transformers sets `v_proj=None` and reuses `key_states` as V).
The expected count is `len(candidates)` on the snapshot, never `4 * n_layers`.
A layer with `q_proj`/`o_proj` but no `k_proj` is refused rather than guessed.
The bias refusal still fires after admission, so gpt-oss's "96 of 96 attention
projections carry a bias" REFUSED holds. `docs/capabilities.json` `gemma4_text`
attention-4-bit stays "not supported pending #412"; the tp2 VOID rows stay VOID.
No gate, threshold, floor or registered claim moved.

### The fused gate/up order is a declared, validated field; up-first is refused at the loader (#515 → #518, #519; #509 → #514)

- `MoEConvention.fused_order` (default `("gate", "up")`) records which half of a fused `gate_up_proj`
  `[E, 2*inter, hidden]` is the gate. It is an adjudicated fact, never inferred: gate and up are
  shape-identical, so a swap computes `up * act(gate)` with every structural gate still passing.
  `__post_init__` refuses any value but the two orders and refuses a non-default order on a non-gated
  convention; `gate_first` is the predicate consumers read. Nine conventions have an unmatchable
  `expert_re` (natively pre-fused or nested — dense, granitemoe, gptoss, qwen3_vl_moe, gemma4, axk1,
  jetmoe, dbrx, qwen3_5_moe) and all nine are gated, so for each an order exists to get wrong; the
  field is data rather than a comment because those families have no key names left to recover it from.
- `loader.expert_layout_for`, the single funnel from the convention system into the loader, raises
  `MoEConventionError` for a convention that declares up-first, until the `chunk(2, dim=-1)` consumers
  (the vendored expert forward, deepseek_v4's dense path, the hybrid and hot-residency engines,
  ExpertsLoRA) are parameterised on the order. No shipped convention declares up-first, so nothing that
  loaded before is refused now.
- `tests/test_moe_conventions.py` adds a numerical detector: pack with the real `fuse_experts`, split
  with the arithmetic the consumers apply, and compare against a reference built from the *separate*
  gate/up tensors (no fused layout to inherit a mistake from) to 1e-12 in float64 on every expert; a
  false-accept probe shows a pack with the halves exchanged fails the same comparison on 3 of 3
  experts. The three pre-existing orientation tests are regexes over upstream source text and could
  not see this pipeline disagreeing with itself.
- Still open on #515: the field states the order and the detector checks this package's packing
  against its own split; neither can tell a reader that a natively pre-fused checkpoint's own order
  differs from the adjudication.
- `axk1` is annotated **staged, not wired** (#509 → #514; comments and one doc line): `loader.py`
  admits it by neither route (`SUPPORTED_ARCHITECTURES`, or a read-compatible convention —
  `{qwen2_moe, mixtral, phimoe}`), and `rewrite_axk1_keys` is absent from `CKPT_KEY_REWRITERS`, so
  admitting it alone would load with the wrong key mapping. `axk2` maps onto `QWEN2_MOE`, which is
  read-compatible, and is admitted; `mixtral` loads by the same second route, which is why it carries
  real-weight PASS receipts without a `SUPPORTED_ARCHITECTURES` entry.

### Streamed calibration merges each chunk's provenance (#537)

- `enable_serve_experts_int4_calibrated` enables the pack one layer chunk at a time, and each chunk's
  enable called `_attach_live_pack_provenance`, which **replaced** the live record — so after a
  48-layer streamed build the record, and the hashed assignment payload `dump_calibrated_artifact`
  wrote from it, described the last chunk only. P39 box 1 dumped an assignment naming 8 of 48 layers;
  box 2's honoured build refused `layer 0 expert 0 gu: not named by the assignment` — the refusal #530
  promised, and correct. The single-chunk tests could not see it.
- A live record already on the model for the same checkpoint is now merged: component hashes by path
  (the chunk's bytes win), the decision by `(layer, expert, role)`, row counts by `(layer, expert)`,
  counts summed, honoured-disagreements summed under the same hash, and every hash and the
  `pack_fingerprint` recomputed over the union — so the record after the last chunk equals what one
  all-at-once enable writes. Test: a two-layer checkpoint enabled as `layers=[0]` then `layers=[1]`
  carries both layers' decision, summed counts, all eight payloads and the SAME `pack_fingerprint` as a
  single enable, and the artifact it dumps names both layers.

### Documentation, and what is repository-only

- `engines/int4_experts.py`'s module note describes the dispatch, not the design (#496 → #520):
  batched decode stays on the NF4 M-tile path in every default configuration (`DEVICE_GROUPING =
  [False]`, assigned nowhere in the package), and the `gemm_int4_b32_grouped_captured` branch the
  published `e4b.serve.b16.qwen3-30b.int4.5090` row was measured in is reached only when a bench flips
  that flag. `tests/test_int4_docstring_matches_dispatch.py` holds the note to the code.
- `docs/ARCHITECTURE_SUPPORT.md` is regenerated from `bench/support/` rows (#521, #522; PR title:
  "1 of 9 evidenced becomes 6 of 9"): real-checkpoint load / verify / forward rows on current versions,
  `tests/test_support_doc_matches_rows.py` fails when regenerating is not a no-op, and the
  architecture claim is joined to grouped-nf4-gemm's shape census (#353's missing artifact). The
  `SUPPORTED_ARCHITECTURES` comment in `engines/offload.py` no longer states a family count.
- `train.py` logs the structurally expected attention-projection count beside the converted one.
- Repository-only, not in the wheel: the pre-registered lanes and their box-side runners under `bench/`
  (P39, P41, P42, tp3, tp4, and the K14/K15 runners whose pre-registrations live in grouped-nf4-gemm),
  the compute-governance policy and run-receipt ledger, and the Vast rent launcher fixes. Lane results
  are registered where they moved a number; see `docs/claims.json` and the receipts they name.

## 0.35.3 — 2026-09-06 — the loader honours a pinned checkpoint revision (#404); tp2 / P40 into the register (#415)

One behaviour change (the loader threads `revision` into both hub lookups, pins remote modeling code to the same commit,
records the commit actually loaded and refuses mismatches; #404 → #410) and the claims half of the tp2 bundle (31 rows,
`training_support` per family × path from the receipts, companion docs; #415 → #419, receipts in #414). No gate, threshold,
floor or existing claim value moved; the `fast` extra's floor (grouped-nf4-gemm >= 0.30.0) and the CI pin are unchanged.

### tp2 / P40 into the register: the per-family Unsloth head-to-head (claims + docs; the claims half of the tp2 bundle, receipts merged in #414)

Documentation and register only; no code, no version bump, no gate, threshold, floor or existing claim value moved.
The receipts landed separately as `bench/h2h-20260906/tp2/` (#414); this entry registers them (#415).

- **31 rows under `e4b.train.h2h.unsloth.<family>.5090.2026-09-06`** (lane tp2 / P40, 2026-09-06, one rented RTX
  5090, Vast 50005568 on a Ryzen 7 5700X3D host; the pre-registration verbatim as the bundle's `P40-PREREG.md`;
  every number copied from `RESULTS-tp2.md` and the receipt JSONs; the `.coverage` and `.footprint` rows per the
  spec amendment on #415): one row per attempt (`….arm.<framework>.<arm>` — including Granite's Unsloth VOID,
  OLMoE's Unsloth HARNESS_ERROR, gpt-oss's three REFUSED rows and Gemma-4's two `void_attn4` rows, #412),
  position + `.quality-n60` rows on Qwen3 and Mixtral, a `.footprint` row on Mixtral, `.coverage` rows on Granite
  and OLMoE (the comparator could not train the experts there — attention-only LoRA / a crash at MoE-LoRA engage
  — its own log lines quoted; no speed ratio in those rows), `.e4b-internal-parity` PASS rows on Granite, OLMoE,
  Qwen3 and Mixtral. The positions: Qwen3-30B-A3B s/step Unsloth/e4b **1.457** (e4b faster per step; held-out
  COMPARABLE, Δ +0.0152) — the cross-lane anchor, **+3.1% from P38's 1.413, inside the pre-registered ±10%**,
  noted on the P38 row (neither supersedes the other); Mixtral-8x7B **0.361**, whose row leads with the footprint
  trade: e4b trained under its registered expert-offload design at a **3.223 GB** peak (the
  trainable-on-smaller-cards result, `….footprint`) against Unsloth resident at **29.163 GB**, its only mode, and
  what that VRAM buys it is speed per step — a footprint-vs-speed trade, not a kernel deficit (held-out
  COMPARABLE, Δ −0.0087; P4's OOM prediction falsified).
- **`training_support` updated per family × path from these receipts only** (`docs/capabilities.json`, inside the
  per-path structure — never a flat boolean): the attention-4-bit configuration (`TRAIN_ATTN_4BIT`,
  `reference_attn4`/`fused_attn4`) now has receipts — supported on `granitemoe` / `olmoe` / `qwen3_moe` /
  `mixtral` (the new claim ids cited per path); **not supported on `gemma4_text` pending #412**
  (`quantize_attention_projections_4bit converted 100 projections, expected 120`; the bf16-attention `fast_train`
  path stays exactly as tp1 left it); refused on `gpt_oss` (96 of 96 attention projections carry a bias).
  `model_families` is unchanged.
- `docs/STATUS.md` (a dated "tp2 / P40 (2026-09-06)" section and a #412 open item),
  `docs/ARCHITECTURE_SUPPORT.md` (a dated per-family head-to-head section: what trained, what refused, the
  competitor observation quoted from the receipt — statuses, never a flat flag),
  `docs/solutions/qlora-fused-moe-experts.md` (the per-family measured-result paragraph, the attention-4-bit
  scope note, evidence rows), two routing queries in `docs/discovery-queries.json`, `llms-full.txt` regenerated.
  The README results table is unchanged: no existing claim's value or status moved.
- **Serving-comparison rule applied** (the #415 spec amendment's third clause): prose touched by this change may
  not quote a serving ×N against e4b's own NF4 control without the same-box vLLM figure, its weight precision and
  its resident footprint in the same sentence (P37: vLLM 0.28.0 serving `Qwen3-30B-A3B-GPTQ-Int4`, footprint not
  recorded in the receipts). No prose touched by this change quotes such a ×N, so no sentence needed the
  annotation; the rule is recorded here for the next edit that does.

### The loader honours a pinned checkpoint revision (#404)

- `load_moe_4bit_streaming(..., revision=<commit sha | branch | tag>)` threads the revision into both hub lookups the
  loader makes (`AutoConfig.from_pretrained` and `snapshot_download`), and pins a trust-remote-code checkpoint's modeling
  module to the same commit (transformers' `code_revision`); remote code hosted in a different upstream repository cannot
  be pinned by the weights' sha and is logged as unpinned. A snapshot staged with
  `snapshot_download(model_id, revision=<sha>)` writes no `refs/main`, so before this the unpinned `main` lookup had
  nothing to resolve offline (`LocalEntryNotFoundError` at load -- five training arms in a row on 2026-09-05) and
  online the loader streamed whatever `main` pointed to that day; every lane since P38 wrote `refs/main` by hand to
  work around it (P38 amendment 2, tp2). The commit actually loaded is recorded on `config._commit_hash`
  (transformers' own receipt slot, filled from the snapshot folder when transformers left it empty) and in the log
  line `checkpoint: <id> @ <sha> (requested ...)`; a full-sha `revision` whose snapshot resolves to a different
  commit -- or a config and a snapshot from two different commits -- is refused with `ValueError` rather than
  loading other bytes. A local directory has no hub revision: noted, not verified. Default `None` still means
  `main`; for existing callers the resolved commit is now logged, and `config._commit_hash` may now be populated
  where transformers left it empty (filled from the snapshot folder's basename). Nine loader tests cover the
  threading, both refusal arms (a pinned sha that resolved elsewhere; a config and a snapshot from two different
  commits), the unpinned receipt, the basename fallback, the local-directory case, the remote-code pinning
  (same-repo and upstream-repo `auto_map`), and a real offline hub-cache regression -- `HF_HUB_OFFLINE` forced over
  a staged `snapshots/<sha>/` cache with no `refs/`, both hub lookups untouched, the unpinned load shown to die
  there and the pinned one to resolve; no kernel, gate or threshold changes.

## 0.35.2 — 2026-09-05 — documentation: head-to-head receipts (same-box vLLM; Unsloth QLoRA end-to-end)

Documentation and tooling only; no runtime change. The `fast` extra's floor (grouped-nf4-gemm >= 0.30.0), the CI
`--requires` assertion and the `>=0.35.0` compatibility record in `docs/system-manifest.json` are unchanged (the
manifest is byte-identical to 0.35.1's). This release exists so the first end-to-end head-to-head against Unsloth is
in the repository with its pre-registration, every amendment and every attempt, and so the machine-readable surfaces
the 2026-09-05 audit found unchecked -- claim sentences, evidence paths, successors, licence labels -- are held by a
check from here on.

### vLLM 0.28.0 head-to-head, same box, identical prompt token ids (lane p37, receipt `bench/h2h-20260905/p37/`)

- One rented RTX 5090 (Vast 49975016, EPYC 7Q83 host), one session, decode-vs-decode on the same 512-token prompt ids
  (dumped once from the e4b harness's own window function and fed to vLLM verbatim; the reducer refuses to divide
  receipts that disagree on the prompt sha): vLLM 0.28.0 (torch 2.13.0+cu130, triton 3.7.1) serving Qwen's
  `Qwen3-30B-A3B-GPTQ-Int4` (Marlin, default CUDA graphs, kv auto; eager and fp8-KV arms beside) against
  experts4bit-qlora 0.35.0 + grouped-nf4-gemm 0.30.0 -- the NF4 control and the licensed recipe (bo7b's
  `calibexp_all_n128`, the pack rebuilt on that box), bo7's harness pieces byte-identical, sixteen arms with
  non-adjacent self-pairs, every knob in the receipts. Pre-registered before the box was rented (`PREREG.md`); three
  amendments (a staging refusal; the comparator fetch hang left to its alarm and recovered in-lane, the guarded follow-up
  ran as `NOT NEEDED`; the registered K8 gate on that box's pack, below); no threshold, arm, prompt set or knob changed.
- **What is quoted** (`e4b.serve.h2h.vllm-0.28.0.qwen3.5090.2026-09-05`, measured): vLLM 286.0 tok/s at B=1 (3.497
  ms/step; fp8-KV 300.9; eager 20.8) and 2030.0 aggregate at B=16 (7.882 ms; fp8-KV 2206.5; eager 322.5) against the NF4
  control's 113.4 / 500.1 -- **vLLM / e4b-NF4 2.52 and 4.06**, the only licence-free ratio the lane can produce (self-pairs
  inside 1.03×). **What is not quoted:** the ratio against the licensed stack. Every licensed e4b arm on that box is VOID
  under the pre-registered pack-fingerprint rule -- the streamed calibration there packed 11522 gptq / 766 rtn expert
  matrices where the licensed pack reads 11512 / 776 (the same recipe, ten of 12,288 matrices across the `min_rows`
  threshold, not the licensed bytes; speed cannot inherit a licence). The recipe's speed on that box (236.4 / 1305.3
  tok/s, ×2.08 / ×2.61 over its NF4 control -- bo7's ×2.067 / ×2.602 reproduced within 1%) is registered per arm as
  measured and unlicensed. **Amendment 3** (pre-registered after `TP_DONE`, the registered K8 gate on that box's own
  pack, bo6c's `k8()` verbatim, `e4b.serve.h2h.vllm-0.28.0.qwen3.5090.2026-09-05.gate`): wikitext Δ −0.0230 ppl PASS, C4
  validation **Δ +0.1093 ppl FAIL** against the +0.05 budget (the licensed pack read −0.0662 on the same window; the two
  boxes' NF4 references agree to 0.0002 ppl) -- **NOT LICENSED, VOID stands**: no ratio against the licensed stack exists
  on that lane, and the streamed calibration recipe is shown not to reproduce its licence across hosts (an open item in
  `docs/STATUS.md`, filed as #405; bo6c's licence stands on its box). `e4b.serve.h2h.vllm.same-box` (2026-09-03, ×1.47 / ×1.55,
  a different box, the RTN stack, vLLM version unrecorded) is superseded for current-position use and stays as measured.
  One row per arm (`…arm.<engine>.b<B>.<arm>`, sixteen). Quality quoted, never equated.
- `docs/capabilities.json` (`serve-moe-on-consumer-gpu`: the claim and the limitation reworded), `docs/STATUS.md`
  ("Serving speed", "What changed", an open item: the licensed pack does not reproduce bit-for-bit across boxes), the
  README results table and "Do not use this when", `docs/solutions/serve-large-moe-on-a-consumer-gpu.md`, a routing
  query, `llms-full.txt` regenerated; the capability carries the host-dependence as a limitation.

### e4b vs Unsloth, QLoRA end-to-end, one identical training problem (lane p38, receipt `bench/h2h-20260905/p38/`)

- One rented RTX 5090 (Vast 49975389, train-anchor class `pcie-full/launch-fast` recorded): Qwen3-30B-A3B at one pinned
  revision, the registered `clinical` fixture tokenised ONCE into a sha-asserted file both frameworks train on, seq 512,
  r 8 / α 16 on attention q/k/v/o and every expert (321,257,472 trainable parameters asserted in every arm, router
  frozen), the same AdamW call, batch 1, loss over all tokens, the same held-out eval -- experts4bit-qlora 0.35.0 +
  grouped-nf4-gemm 0.30.0 in the image python (the fused `dgrad` path with NF4 attention, the shipped `TRAIN_ATTN_4BIT`
  mechanism; transformers 5.16.1, bitsandbytes 0.50.1) against Unsloth 2026.9.2 + unsloth_zoo 2026.9.1 in its own
  venv (its 4-bit MoE path, `native_torch` backend; transformers 5.5.0, bitsandbytes 0.50.2, peft 0.20.0 -- the
  transformers/peft difference between the two pythons is a recorded environment difference). Pre-registered before
  the box was rented (`PREREG.md` in the bundle, verbatim); every failed attempt a row; four amendments, all
  environment or instrument (the comparator venv's pins; the loader's `refs/main` and a `torchao` import; the Unsloth
  branch's snapshot-directory resolution; the U8 predicate evaluated on PEFT's wrapper, proven on the innermost module
  and re-reduced) -- no workload, fixture, threshold or knob changed.
- **The position at 60 steps** (`e4b.train.h2h.unsloth.qwen3.5090.2026-09-05`, measured): s/step ratio Unsloth/e4b
  **1.413** (2.151 vs 1.522 s -- e4b faster per step at this workload), peak VRAM 21.371 vs 23.141 GB, 157.1 vs 224.7
  J/step, time to a held-out loss of 0.32 92.5 vs 130.3 s; held-out loss comparable, 0.2923 vs 0.2975
  (`…quality-n60`, |Δ| 0.0052 ≤ the pre-registered 0.05 reading threshold). **At 200 steps the curves separate in
  Unsloth's favour: 0.2713 vs 0.2881** (`…curve-n200`) -- a measured row in Unsloth's favour, quoted beside the
  position wherever it is quoted; candidate causes (eval schedule, checkpointing mode, the two stacks' transformers /
  peft versions, the expert adapter's precision -- bf16 on this side because the loader passes the model dtype to
  `ExpertsLoRA`, fp32 on Unsloth's) are not established. e4b's fused-vs-reference pair passes its band on that box
  (`…e4b-internal-parity`: 0.00131 / 0.01138, ×2.92 per step; informational, tp1 owns the licence). One row per arm
  (`…arm.<framework>.<arm>`, eight, all VALID). The pre-registration predicted the opposite sign at this workload
  (Unsloth faster per step, lower peak) and shipped the finding either way. Nothing is superseded or licensed by this
  lane; the 2026-08-26 "1.17× ahead" memory (never a claim) is disqualified as a comparison. Found on the way:
  the loader does not honour a pinned revision (e4b#404).
- `docs/capabilities.json` (`qlora-fused-moe-experts`: the four claims and a limitation that says the 200-step curve
  favours Unsloth), `docs/STATUS.md` ("What you get today"), `docs/solutions/qlora-fused-moe-experts.md` (measured
  result, limitation, evidence), the README results table, three routing queries in `docs/discovery-queries.json`,
  `llms-full.txt` regenerated.

### The register, after the 2026-09-05 machine-readable audit

- **Superseded:** `e4b.serve.tp.qwen3.b1.5090.2026-09-04` and `.b16` -- their "best licensed" configuration class
  (round-to-nearest int4 experts + calibrated attention) failed its second text on lane bo5 -- now point at the census
  rows of the licensed stack, `e4b.serve.census.bo7.qwen3.b1.5090.2026-09-05` / `.b16` (which name them under
  `supersedes`); their values stand as measured. The other four families' 2026-09-04 rows keep their numbers with the
  "best licensed" label withdrawn in the sentence (measured, not licensed), the NF4-position rows of gpt-oss and
  Gemma-4 say what they are, and the Gemma-4 / gpt-oss build-out rows no longer call themselves licensed under a gate
  their own notes say does not exist for the family. No number, gate, threshold or verdict changed.
- **`licensed_by`** (new field, `docs/claims-schema.md`): every active claim whose sentence asserts a licence names the
  claim whose receipt holds the K8 verdict -- the bo6c Qwen3 verdict row for Qwen3's census and census-row claims,
  the bo3 Granite row (its own receipt carries the +0.019 ppl pass) for Granite's; a row whose own receipt carries the
  verdict names itself.
- **Retired by id:** the 2026-08 "vLLM 6.31× ahead" figure (`e4b.retired.vllm-6.31x-ahead`; template prompts, a
  different box) so that `e4b.serve.h2h.vllm.same-box`'s supersession resolves; that head-to-head now carries full
  `conditions` (the vLLM version is not recorded in its receipt; the e4b arm is the 2026-09-03 RTN-int4 class of
  `e4b.serve.b16.qwen3-30b.int4.5090`, not the licensed stack) and a populated `quoted_in`.
  `e4b.retired.13.47x-training-speedup` is `retired` (its "about 7.2×" restatement has no receipt and was never a
  claim); `e4b.retired.inference-md-decode-grid` names its successor; two dangling `supersedes` pointers resolve or
  are dropped with the reason in notes.
- **Evidence resolves:** the three bo3 `measured` rows that named run logs never committed now name the K8 receipts
  beside their speed receipts (`bo3/*_ppl_*.json`); annotated paths are bare paths with the annotation moved to
  notes; the scratch probe of `e4b.serve.gptoss.loader-faithful` is dropped and said so; cross-repository and
  issue evidence use the structured forms the schema now defines (`{"repository", "path"}`, `{"url"}`).
- **`measured_on`** on the ten tp1 rows that lacked it and, from their receipts' own dates (stated in notes), on the
  twelve older measured rows.
- **Stale notes:** `e4b.train.flagship-matrix`, `e4b.serve.tp.granite.*` and nine other rows no longer say "pending"
  for things measured since; each states what is measured, with ids.
- Prose: `docs/solutions/serve-large-moe-on-a-consumer-gpu.md`'s "licensed configuration per family" is rewritten
  from STATUS's current positions (Qwen3 = the bo6c streamed stack; OLMoE and Mixtral = NF4; Gemma-4 = `r1epi` on NF4
  with no instrument; Granite = `r12epi`; gpt-oss = its NF4 reference arm), its Evidence lists the census rows first;
  `docs/STATUS.md` "Serving speed" leads with the licensed position and says the 2026-09-03 RTN class failed its
  second text and is not the licensed stack; `serve-moe-on-consumer-gpu` cites the bo7 census rows and the bo6c
  licence; `docs/SERVING-THROUGHPUT.md` is dated by section; abbreviated ids in STATUS and SERVING-THROUGHPUT are
  written in full; the document count in `docs/INDEX.md` is the number it links (held by a test) and the README no
  longer repeats it.

### Checks (tooling, not the package)

- New `scripts/check_claims_register.py` (CI, discoverability job; `tests/test_check_claims_register.py`): every
  `evidence[]` entry is a path that exists at HEAD or a structured cross-repository / issue entry (cross-repository
  entries verified against `--sibling`); `measured_on` required and ISO on every measured row; a `superseded` row's
  successor chain reaches an active row; `retired` rows carry `retired_reason`; `supersedes` and `quoted_in` resolve;
  no "pending"/"TBD" on an active row; a licence label names its verdict row.
- `scripts/check_readme_claims.py` applies its id rules to the position documents (`docs/STATUS.md`,
  `docs/SOLUTIONS.md`, `docs/solutions/*.md`, `docs/SERVING-THROUGHPUT.md`, `docs/SERVING-PARITY.md`,
  `docs/METHODOLOGY.md`, `docs/ARCHITECTURE_SUPPORT.md`, `docs/CHOOSING.md`): every backticked id exists, an inactive
  one only on a line that says superseded / retired / historical.
- `scripts/check_system_manifest.py`: the CI kernel pin (`git+…@<sha>` in `ci.yml`) is the commit of a release tag at
  or above the tag `consumer_ci_pin` names (`git ls-remote --tags`; without network a visible NOTE, never a silent
  pass), and every kernel-pinning extra (`fast`, `test`) floors at or above the current record's; CI now also clones
  the kernel package at its latest release tag and runs `--sibling` (byte-identical manifest) and the register check's
  cross-repository evidence.
- `scripts/check_capabilities.py` warns (never fails) when a capability whose primary mode is `serving` cites none of
  the newest serving lane `docs/STATUS.md` quotes as the position.
- CI installs grouped-nf4-gemm from the v0.30.1 release commit (`d58b39fb…`; the comment now states the pin's real
  reason); `pyproject.toml`'s `[test]` floor is `grouped-nf4-gemm>=0.30.0`, equal to `fast`'s.
- Checks hardened after review: `check_claims_register.py` reads evidence from the git tree (`git ls-files` -- an
  untracked or gitignored receipt log, a directory, an absolute path or `..` is a finding; outside a checkout the
  working tree stands in and the output says so), applies the licence rule per occurrence (one "unlicensed" no longer
  excuses a bare "licensed stack" beside it; the citation form ``licensed by `<id>` `` refers to another row's licence
  and is resolved -- the id must carry `licensed_by` -- rather than labelled: eleven sentences reworded with no number
  changed, the p37 rows citing the bo6c verdict), follows every `superseded_by` to an active row and refuses one on an
  active row, allows `retired_reason` on retired rows only (`e4b.retired.inference-md-decode-grid` drops the field it
  carried beside `superseded`; `e4b.retired.vllm-6.31x-ahead` names the current successor), and exits 2 for a
  `--sibling` whose slug cannot be resolved instead of skipping its entries; `check_readme_claims.py` reads the
  superseded / retired / historical words from the line's prose alone (links reduced to their text, HTML comments and
  every claim id in any form stripped) and treats ids in `<code>` or link text as citations; `check_system_manifest.py`
  gains `--require-tags` (CI: an unreadable tag list is exit 2, not a NOTE on a green run), reads `consumer_ci_pin` as
  the last `vX.Y.Z` in its prose and ignores comment and non-`pip install` lines; `check_capabilities.py` warnings are
  also `::warning::` annotations; `docs/claims-schema.md` documents `validity`, `row_status`, `parity_verdict`;
  `tests/test_docs_index_count.py` counts links through `md_links`.

## 0.35.1 — 2026-09-05 — documentation (the tp1 training parity matrix) and one behaviour change (#397 → #402)

The training parity matrix on real weights (lane tp1) with its documentation, plus **one behaviour change** from the
parallel #397 fix (#402, below) — **this release is not docs-only**. The `fast` extra's floor (grouped-nf4-gemm >=
0.30.0), the CI `--requires` assertion and the `>=0.35.0` compatibility record in `docs/system-manifest.json` are
unchanged. The release exists so that the repository, the PyPI project page and the site state the training matrix
from evidence, and so that the refusals that matrix argued for ship with it.

### The stock-epilogue contract: `ExpertsLoRA` refuses what it cannot represent (#397)

- `ExpertsLoRA` re-implements the expert forward inline (the low-rank delta lands before the nonlinearity), so it owns the epilogue and can only represent the stock `down(act_fn(gate) * up)` or a module that hands its epilogue over through `_apply_gate`. gpt-oss's stack (per-expert biases, clamped sigmoid GLU, de-interleaved at load, no hook) was wrapped anyway by the arena loader under `arena_train=True` and trained against a plain SwiGLU with nothing raised. New: `experts4bit_qlora.assert_stock_epilogue(module)` decides on the module's STRUCTURE (bias buffers/parameters/tensor attributes, a non-stock `forward` without a hook, `alpha`/`limit`/`swiglu_*` scalars no hook consumes, a forward body that clamps or adds a bias itself, an interleaved-layout marker, a two-argument `act_fn`, a declared shape that is not `[2I, H]`/`[H, I]`) and raises `EpilogueContractError` (a `TypeError`) naming every offending attribute and the faithful route (grouped-nf4-gemm's `mxfp4_qlora.ExpertsMxfp4LoRA`, the `mxfp4-moe-training-and-residency` capability). Applied by `ExpertsLoRA.__init__`, the loader's `arena_train=True` branch, `enable_nvme_train_residency`'s pre-flight (before the tier opens), `enable_hybrid_train` (which used to refuse gpt-oss through the tier's own flag), and `enable_fast` / `enable_fast_train` / `enable_batched_train`, which now REFUSE a wrapper whose base violates the contract instead of skipping it (a skipped wrapper would have trained the same unfaithful reference forward). Stock SiLU, Gemma-4's gelu_tanh, non-gated stacks and DeepSeek-V4's hooked clamp are unaffected. Both new symbols are exported; `docs/capabilities.json` and the solution pages are updated by the release bundle that carries this entry.
- `enable_mxfp4_nvme_residency` refuses a module that carries per-expert bias tensors: the binding passes no biases to the engine and defaults to the V4 epilogue, so gpt-oss would have been served through the wrong GLU with its biases dropped. The MXFP4-arena fused lane (`mxfp4_experts_forward`) keeps the module's own forward for any module the contract refuses.
- `enable_batched_train`'s per-call fallbacks (pad waste past `_PAD_WASTE_LIMIT`, evicted storage, an empty batch) are counted on each patched module; `batched_fallback_stats(model)` (exported) reports `calls` / `batched` / `fallback_calls` / `by_reason` per module and in total, so an arm claiming the batched path can assert `fallback_calls == 0` (the tp1 lane read OLMoE's batched arm as void because layers fell back with nothing counting them).
- The loader logs a one-time NOTE when a family's experts are built bare (no expert adapter; `r`/`alpha` do not apply), and `python -m experts4bit_qlora.train` refuses `TRAIN_EXPERTS=1` on a model with no expert adapter instead of training attention/router alone under that flag.
- Audit of the public enable/load/train/serve entry points for the class "unsupported behaviour, plausible output, no refusal": `docs/audits/no-silent-fallback-2026-09-05.md`.
- Folded under this release from `## Unreleased` at the bundle's rebase; the capability contract and the solution page carry the refusal (`EpilogueContractError`), the MXFP4 NVMe-residency refusal of bias-carrying modules and `batched_fallback_stats` as limitations, and `training_support.gpt_oss.nvme_train` is `refused` with that code reference.

### Training on real weights, per family, under the shipped code (lane tp1, receipt `bench/train-parity-20260905/tp1/`)

- The six serving families through the shipped training path on one rented RTX 5090 (train-anchor class recorded):
  the direct `load_moe_4bit_streaming` + `verify_moe_4bit(strict=True)` path on the real checkpoints, then
  `reference` / `enable_fast_train(dgrad=True)` / `enable_batched_train` for 60 steps on the registered `clinical`
  text, verdicts by `tp1_reduce.py` in the registered B2/C2 units (`|Δ final train loss| ≤ 0.05` and median
  step-wise `|Δ| ≤ 0.05` against the family's own reference; VOID when the arm cannot be read), cost reported and
  never gated. Every log, the amended lane script beside its pre-amendment copy, the patched harness and its patch,
  the reducer, `RESULTS-tp1.md` (the reducer's output verbatim plus the reading) and a README with the three
  amendments — the first box's 10 MB/s link, the fetch-by-short-sha false start and the ≈ 4.4 h it idled, the
  harness's closure bug fixed in flight — and predictions P1–P7 scored.
- Rows (`TP_DONE` 2026-09-05T15:22Z, all six families through the registered arms): the fused path **PASSES** on every
  family that has one — OLMoE-1B-7B-Instruct (`e4b.train.parity.tp1.olmoe.fused.2026-09-05`, the first reading on a
  registered text with real weights for that family), Qwen3-30B-A3B resident on the 32 GB card (`…qwen3.fused…`),
  Gemma-4-26B-A4B-it — the `-it` checkpoint, loaded without #344 on that host — with the step-wise median inside the
  band by a small margin (`…gemma4.fused…`), and Mixtral-8x7B-Instruct under `offload=True` at half the reference
  loop's peak VRAM (`…mixtral.fused…`; the family enters `model_families` on it); the batched path **PASSES** on
  Granite-3.1-3B-A800M (`…granite.batched…`; the family's first direct real-weight load) and Mixtral, and is **VOID** on
  OLMoE, Qwen3 and Gemma-4 — `enable_batched_train` falls back to the reference forward per call above
  `_PAD_WASTE_LIMIT` with no counter, and the kernel was not reached on every layer; gpt-oss fused / batched
  **REFUSED** (0 patched: the loader builds its experts bare), attention-only QLoRA trains with the frozen stacks
  bit-exact, and grouped-nf4-gemm's experimental MXFP4 route trains its experts on its own text with the canary passing
  — experimental, never licensed. Granite's fused arm **PASSES** on its corrected-counter re-run (`…granite.fused…`,
  `TP2_DONE` 15:33Z; attempt 1 is a kept HARNESS_ERROR row, `…granite.fused.attempt1…` — a closure bug in the harness's
  kernel counter, not the shipped code; the follow-up script's own abort between them is amendment 5). Every attempt a
  row, 18 result lines, nothing pending.
- `docs/claims.json`: one claim per family × arm (`e4b.train.parity.tp1.<family>.<arm>.2026-09-05`) plus a per-family
  matrix claim; notes on `e4b.train.fast-train-dgrad` (the batched "no speed-up at real width" is a Qwen3-30B-width
  reading — at Granite's width the batched path is the faster one measured), `e4b.train.flagship-matrix` and
  `e4b.train.olmoe-converges`. Nothing superseded or retired.
- `docs/capabilities.json`: `qlora-fused-moe-experts.model_families` is evidence-gated —
  `olmoe`, `qwen3_moe` and `gemma4_text` confirmed on tp1 rows, **`mixtral` and `granitemoe` added** (Mixtral's fused
  PASS under offload; Granite's on the corrected-counter re-run), `gpt_oss` stays out with the refusal named and the
  experimental route pointed at; the batched-fallback VOID (three families), the `EpilogueContractError` refusal and
  `batched_fallback_stats` are limitations; `training_support.gpt_oss.nvme_train` is `refused` with a code reference; the
  `mxfp4-moe-training-and-residency` capability carries the tp1 canary row and stays `experimental`.
- Phase directive 2026-09-05 14:45Z, applied to the bundle's shape: every row of the receipt is exactly one of OK /
  REFUSED / HARNESS_ERROR / ALARM / OOM / NOT_RUN / EXPERIMENTAL, classified mechanically by the bundle's reducer (v2;
  the box's v1 copy kept) with the parity verdict as a separate column, every attempt a row (Granite's first fused
  attempt stays a HARNESS_ERROR row beside its re-run) and every amendment referenced from the rows it touched; the
  `e4b.train.parity.tp1.*` claims carry `row_status` / `parity_verdict`. `docs/capabilities.json` states training
  support **per path** — a `training_support` object (`headline_path`; `by_model_type` keyed by model_type: quantize / reference_train / fast_train /
  batched_train / nvme_train / native_mxfp4_train, each `supported` / `refused` / `void` / `harness_error` /
  `not_tested` / `experimental` / `n/a` with its claim ids) on `qlora-fused-moe-experts` and
  `mxfp4-moe-training-and-residency`; `model_families` is exactly the families whose `fast_train` is `supported`.
  Tooling, not the package: `docs/capabilities.schema.json` admits the object and `scripts/check_capabilities.py`
  validates it (allowed values; `supported` / `void` / `refused` cite existing claim ids, `supported` ones active;
  `refused` says why; the `model_families` rule), with `tests/test_check_capabilities_training_support.py`.
- `docs/STATUS.md` (the training position and three open items), `docs/ARCHITECTURE_SUPPORT.md` (a new dated section
  "Training on real weights (tp1, 2026-09-05)"; the existing tables are untouched), `docs/SOLUTIONS.md` and
  `docs/solutions/qlora-fused-moe-experts.md` (families from evidence; the refusal and the VOID as "what to check"),
  the README's results table and scope, two routing queries in `docs/discovery-queries.json`, `llms-full.txt`
  regenerated.

## 0.35.0 — 2026-09-04

### Calibrated int4 experts, calibrated sequentially (#384)

- Per-expert GPTQ packing for the int4-b32 expert store from the fused forward's Hessian tap: `calibrate_expert_hessians`, `enable_serve_experts_int4(..., expert_hessians=)`, and the driver that decides the result, `enable_serve_experts_int4_calibrated`, which calibrates and packs layer chunk by layer chunk so every chunk is calibrated against the already-int4 prefix (GPTQ's sequential convention) and the host never holds more than one budget of Hessians. Knobs: `E4B_INT4_GPTQ_DEVICE=cuda` (GPU solve), `E4B_INT4_GPTQ_DAMP`, `E4B_INT4_HESSIAN_BUDGET_GB`. Off by default; the serve hook enables it with `E4B_SERVE_EXP_INT4_CALIB=1`.
- Verdicts stay in the register: Granite's calibrated experts fail the registered gate on their second text (`e4b.serve.buildout.bo5.granite.b1.5090.2026-09-04`, notes); the Qwen3 and Mixtral readings under the sequential method are in the bo6 receipt and are registered with its bundle, not here. Calibrating all layers at once against the unquantised prefix is the two-step API only and is not the method that ships.

### Decode glue through the kernel side (#385)

- `silu(gate) * up` through `swiglu_rows` and the top-k combine through `combine_rows` when the kernel side has them (`E4B_FUSE_SWIGLU=0` / `E4B_FUSE_COMBINE=0` are the A/B arms); the split-K reduce through `reduce_partials`. Qwen3's licensed single-stream and batched positions after this cut: `e4b.serve.buildout.bo5.qwen3.b1.5090.2026-09-04`, `e4b.serve.buildout.bo5.qwen3.b16.5090.2026-09-04`.

### gpt-oss: the native MXFP4 expert store (#372)

- gpt-oss experts are served from their released MXFP4 blocks and scales through the grouped MXFP4 GEMM, never re-quantised onto the int4 grid; single rows take the MXFP4 decode GEMV and batched rows keep the NF4 stack (`E4B_INT4_KEEP_NF4=1`): `e4b.serve.buildout.bo5.gptoss.b1.5090.2026-09-04`, `e4b.serve.buildout.bo5.gptoss.b16.5090.2026-09-04`.

### Receipts: the second text, in the gate's own units (#386, #390)

- The bo3 and bo5 build-out bundles are in-repo. Every calibrated and int4 arm now has a second text; three read FAIL as registered, and Mixtral's licensed label is withdrawn (`e4b.serve.buildout.mixtral.*`, notes). The gate was not retuned.

### Documentation and packaging (#388, #393, #389, #394, #391)

- The agent discoverability layer: routing sections in the README, `docs/SOLUTIONS.md` and six problem-first pages, `docs/capabilities.json` under a schema, `AGENTS.md`, `llms.txt` and the generated bundle, PyPI metadata with labelled project URLs, and a CPU-only CI job for all of it.
- The bitsandbytes position is version-, workload- and shape-aware (upstream 5453368 is in 0.50.0, not in 0.49.2); `e4b.train.energy-honest` is superseded by `e4b.train.energy-honest.scoped-a2000`, scoped to its measured comparator and build. Solution pages for capacity, QLoRA on fused experts, offload and MXFP4 carry their decision structure.
- LICENSE is the verbatim MIT text; the vendored bitsandbytes-derived file names `THIRD_PARTY_NOTICES.md` in its header. Homepage, Documentation, Status and Solutions point at the cerinamroth.com routing pages. CI pins grouped-nf4-gemm at a commit that no longer tracks a stale `build/lib/` (the previous pin could ship an old `nvme_reader.py`).
- Requires grouped-nf4-gemm >= 0.30.0 for the `fast` extra.

## 0.34.0 — 2026-09-04

Optimisation pass, day one: the round-2 fold reaches the stacks it never
engaged on, the calibrated int4 set is complete (attention, output head,
dense MLP, biased projections), an op-level census instrument, and a
retraction. Every lane number below is **measured-private** until the
receipt bundle lands in-repo; the numbers are on the merged PRs. Floors
grouped-nf4-gemm >= 0.28.0 (`rope_heads`).

### Round-2 glue reaches the calibrated int4 stack (#375) and norm-less attention (#379)

- The norm + rotary fold had only licensed the FUSED-qkv attention module.
  The calibrated int4 attention lane packs q/k/v/o separately and is
  exclusive with qkv fusion, so on every family's best stack the fold
  never engaged. #375 adds the same fold for the standard
  separate-projection shape (q/k/v/o + per-head q/k norms), licensed on
  structure. Qwen3-30B-A3B, full stack, one RTX 5090: K8 1.8505 → 1.84944
  (−0.0011 nats, inside the 0.0095 floor), B=1 6.41 → 5.62 ms —
  **156.1 → 177.9 tok/s (×1.14)**, ×1.82 over the NF4 baseline.
- #379 (grouped-nf4-gemm `rope_heads`, 0.28.0): the rotary chain folded
  for attention WITHOUT a head norm (GraniteMoe, Mixtral), licensed on
  exactly `{q_proj, k_proj, v_proj, o_proj}`; gpt-oss's `sinks` refuses
  it. Granite, same box: ΔK8 −0.0026 nats (inside the 0.0033 floor),
  **×1.156 at B=1, ×1.093 at B=16**, control arm flat. Review caught a
  vacuous license (the fold required `sliding_window`, which neither
  family's attention sets) before the lane ran.

### The calibrated int4 set is complete: output head (#373), biased projections (#377), dense MLP (#378)

- `E4B_SERVE_LMHEAD_INT4_CALIB=1` (#373): the calibrated int4 output head,
  opt-in. Qwen3 +0.0085 nats in-stack (below its floor), ×1.04 at B=1;
  Gemma-4 (262k vocabulary) ×1.078 on top of its calibrated stack.
- `Int4Linear` carries a projection bias (#377): gpt-oss's q/k/v/o no
  longer refuse the calibrated lane. Speed on gpt-oss ×1.005 alone,
  ×1.015 in-stack (head_dim 64 — a small attention slice); its best
  licensed B=1 moves 133.3 → 137.4 tok/s with the head. Quality is NOT
  readable for this family on raw text (K8 falls 0.13–0.17 nats under
  every int4-attention arm — the documented OOD-regime flattery); the
  flag stays opt-in for gpt-oss until a harmony-text or KL gate exists.
- `E4B_SERVE_DENSE_INT4_CALIB=1` (#378): the dense MLP beside a routed
  block as an opt-in calibrated target (Gemma-4's shape). Measured
  ×1.010 at B=1 on Gemma-4 — not a lever there; correct by test, shipped
  for completeness, default off.

### Instrument: `--op-profile-out` (#380)

- `bench/hybrid-g9/step_decomp.py --b1d-loop eager --b1d-timed
  --op-profile-out PATH`: an op-level census behind the kernel census —
  by call site (the dispatch-mode tracer; torch 2.13 returns empty
  profiler stacks), by op launch count, by op + input shape — from two
  uncaptured steps after the timed window, reserved so the window stays
  byte-identical to a no-flag run.

### Correction: Granite's int4-expert rows fail the registered K8 gate

- The 0.33.0 notes quoted Granite-3.1-3B-A800M at 302 tok/s (×1.59) with
  int4 experts + round-1 norms + router epilogue as reaching the Qwen3-30B
  ratio. Its int4 experts cost +0.0118 nats = **+0.063 ppl** against NF4
  on the same 2048-step window, over the registered 0.05-ppl uncalibrated
  gate (`experts4bit_qlora.k8_gate`). The lane table carried the family's
  0.0033-nat noise floor and no budget verdict, so the row passed unread.
  Retracted as a parity claim in `docs/STATUS.md`; the 0.32.0 throughput
  table's Granite int4 rows (same delta) are re-labelled in
  `docs/SERVING-THROUGHPUT.md` and `docs/claims.json`. Granite's licensed
  stack keeps NF4 experts (round-1 + round-2 folds + epilogue); its
  combined number is on the validation lane.

### Round-2 glue: rotary fold for attention without a head norm

- `E4B_FUSE_T1_GLUE_R2=1` now also folds the rotary chain of the
  Llama-shaped q/k/v/o attention GraniteMoe and Mixtral use (no q/k norm),
  through the kernel side's `rope_heads` (grouped-nf4-gemm >= 0.28).
  Licensed on structure — exactly the four projections and nothing of the
  module's own — so gpt-oss's attention (`sinks`) is refused, and a kernel
  cut without `rope_heads` refuses loudly rather than silently skipping.
  Lane numbers gate the merge.

## 0.33.0 — 2026-09-04

### Throughput parity across families: the build-out, measured

Six families were run under the Qwen3-30B campaign's serving protocol on
one rented RTX 5090 class (`docs/SERVING-THROUGHPUT.md`, receipt in
`bench/hybrid-g9/throughput-20260904/`, 12 claims at tier **measured**).
Every refused arm became a change; each is listed with the lane number
that gates it.

- **Round-2 layer fold licenses on structure** (#366): exactly the four
  pre-norm children and no parameters or buffers of the layer's own.
  Gemma-4's decoder body (two more norms, a routed branch, a layer
  scalar) and GraniteMoe's (scaled residuals) were being silently
  replaced by the Qwen3-shaped body. Any pre-#366 non-Qwen fused
  number is invalid; none was published.
- **GraniteMoe-shaped layer fold** (#371, grouped-nf4-gemm #328): the
  scaled-residual body folds with the kernel's `rmsnorm_resid_rows(
  scale=)` and a one-launch `scaled_resid_add_rows`, both carrying
  upstream's two bf16 roundings. The plain fold now mirrors a
  tuple-returning MoE block (gpt-oss raised `TypeError` on the lane).
  LANE (bo3, one RTX 5090): Granite K8 round-1 1.67927 → round-1 +
  round-2 1.67927 (bit-identical) vs NF4 1.67407; B=1 216 → 222 tok/s
  (×1.027); gpt-oss bit-identical too (6.38437), 132.8 → 133.3.
- **Router epilogue kinds** (#370, grouped-nf4-gemm #327):
  `topk_softmax` (select on the logits, optional bias: gpt-oss,
  GraniteMoe), `gemma4` (normed/scaled router with a per-expert scale),
  and Mixtral's renormalising router without `norm_topk_prob`; probe-
  chosen among candidates; output order by dtype; the first slot is
  the module's own (probabilities or raw logits), recorded by the probe.
  LANE: Granite +0.0009 nats, ×1.046 at B=1; gpt-oss −0.017 nats (inside
  its floor), ×1.009; Gemma-4 ×1.011 (`gemma4` kind); Mixtral −0.003 nats, ×1.01 (its
  step is expert-bandwidth-bound; int4 experts are ×2.07 there).
  Granite's licensed stack (int4 experts + round-1 norms + epilogue)
  reaches 302 tok/s at B=1 = ×1.59 over NF4, the Qwen3-30B reference's
  own ratio, and ×1.80 at B=16 (2,582 tok/s).
- **Gemma-4 MoE convention** (#369): `gemma4` / `gemma4_text` adjudicated
  pre-fused (`experts.gate_up_proj [E, 2I, H]` gate rows first,
  `down_proj [E, H, I]`), so the int4 expert lane can plan it; the
  released multimodal index maps onto the text-only tree (the
  `model.language_model.` prefix is stripped by the loader's own rule,
  the vision tower dropped deliberately and recorded). The loader's
  dedicated path is unchanged. LANE: int4 experts plan 30 layers,
  calibrated attention 115 projections; Gemma-4 int4 experts 71.7 → 86.0 tok/s at B=1 (×1.20), 574 → 797
  at B=16 (×1.39); int4 + round-1 norms + router epilogue 121.1 tok/s
  (×1.69 — above the reference's ×1.59).
- **Matched routing** (#365/#368): `--ppl-route record|replay` pins the
  router's choices across arms (consumption counters, order-agnostic
  by dtype). On Gemma-4 it removes 0.014 of a 0.21-nat remainder —
  routing is not the mechanism.
- **Refused, with its number** (#367): 2 key groups at head_dim 64 so
  Granite and gpt-oss leave the f32 attention path. Paired A/B on one
  box, twice: the released cut on the f32 path times identically to the
  fp8 path (4.10 / 5.19 ms on Granite; gpt-oss flat), for +0.0136 nats
  on Granite (4× its floor) and +0.108 on gpt-oss (6×). The power-of-two
  rounding of the group count (head_dim 96 asked for 3) is kept; the
  floor of 4 groups is restored.
- The K8 harness gains per-layer diff (`--ppl-layer-diff`), the fp8
  kernel precision model inside a one-shot forward (`--ppl-fq`), and an
  `upstream-full` oracle (#361). `--ppl-layer-diff` must not be combined
  with route replay (not inert: +0.46 nats on the same replay).

Floors: `grouped-nf4-gemm>=0.27.0` for `[fast]` and `[test]`.


## 0.32.0 — 2026-09-04

### fp8 paged KV: key scale groups per layer, 32-wide at every head_dim

`Fp8PagedKV(k_groups=None)` (the new default) sizes each layer's key
scale groups to keep 32-wide scales — 4 at head_dim 128 (unchanged), 8
at 256, 16 at 512 — when the installed grouped-nf4-gemm unrolls that
many (a capability probe of `fp8_compute_unsupported`, never a version
string), and falls back to 4 otherwise. Gemma-4's five 512-dim layers
measured 0.046 nats of fp8 cost with 128-wide groups and 0.017 with
32-wide (P27, #359). Small heads are never coarser than before. An int
still broadcasts to every layer; `kv.kgs[layer]` is the per-layer value
and `kv.k_groups` is the uniform value or `None` under mixed geometry.

Measured end to end on a rented RTX 5090 with grouped-nf4-gemm 0.26.0
(which unrolls 8 and 16 groups): Gemma-4-26B-A4B-it's paged decode on
the P26b window moves from 3.59239 to **3.57228 nats (−0.020)** with
16 groups on its five 512-dim layers and 8 on the sliding layers; the
fake-quant instrument had predicted about −0.029. Qwen3-30B-A3B (head
dim 128, groups unchanged at 4) is bit-exact at 1.61067. The K8 harness
gains `--kv-groups` (default `auto`). The `[fast]` and `[test]` floors
move to `grouped-nf4-gemm>=0.26.0`.

Also: the #344 Gemma-4 load fault did not reproduce on a third host
running driver 580.159.03 (every copy path and the bake succeeded under
`CUDA_LAUNCH_BLOCKING=1`), so the driver-version lead is refuted and the
fault stays confined to two specific, currently unrentable machines.


## 0.31.2 — 2026-09-03

### Correction: Gemma-4 has no parity reference at 512-token resolution

No code changes. 0.31.1 said Gemma-4-26B-A4B-it's paged decode was "not
at parity: 0.247 nats, three times its floor". Three windows and a
three-forward test in plain transformers (no e4b code) say something
different: the paged path is +0.093 / +0.114 / +0.247 nats from a
one-shot forward, and transformers' *own* cached forward is −0.107 /
+0.271 / +0.081 from the same one-shot forwards. The cache is
bit-exact; what moves is the model — bf16 batch-shape variance in the
expert gathers (0.2% at layer 1) that Gemma-4's router amplifies to a
0.4-nat swing on identical tokens. Qwen3 shows the mechanism at a tenth
of the amplitude and loses 0.001. So this family has no reference at
that resolution, and no parity verdict is quoted for it. What survives
as a measured, path-specific cost is the fp8 cache and dot: 0.046 nats,
on the five 512-dim layers, 0.017 with 32-wide K groups. Register:
`e4b.parity.gemma4.chunk-free` → superseded by
`e4b.parity.gemma4.no-reference`; new `e4b.parity.gemma4.fp8-share`.
METHODOLOGY §13.2 describes the three-forward test. [#359](https://github.com/pjordanandrsn/experts4bit-qlora/issues/359)
stays open, re-scoped to the kernel's K groups and a batch-variance-proof
instrument.

The harness gained `--ppl-fq` (the fp8 kernel's precision model inside
the one-shot forward), `--ppl-chunk`, `--ppl-layer-diff` and
`--ppl-oracle upstream-full` (#361).

## 0.31.1 — 2026-09-03

### Correction: Gemma-4 is not at parity through the paged decode path

No code changes. Against a chunk-free reference (one full forward, no
chunk boundaries) on a 512-step window, Gemma-4-26B-A4B-it's paged
decode is 0.247 nats from the model's own attention — three times a
floor (0.081 nats) that is itself five to twenty-five times any other
family's. The 0.31.0 README, `docs/STATUS.md`, `docs/SERVING-PARITY.md`
and `docs/claims.json` carried this family as "behaves" on the strength
of −0.0078 nats against the *chunked* oracle over 8192 steps; that was a
comparison with an instrument, not with the model, and it is superseded
(`e4b.parity.gemma4.behaves` → `e4b.parity.gemma4.chunk-free`). Tracked
as [#359](https://github.com/pjordanandrsn/experts4bit-qlora/issues/359)
with the tests in order: the paged arm with a bf16 KV cache on the same
window first, because the fp8 K-cache scale groups are 128-wide at
`head_dim` 512.

Also in this release: Qwen3-30B-A3B's chunk-free row (0.00173 nats,
floor 0.00641) — three of four families indistinguishable from their
own attention, one not — and the #344 host tally is 3 load / 2 fail.

## 0.31.0 — 2026-09-03

### Documentation release: the README says what is measured, and every number has a register entry

No code changes. This release exists so that what PyPI renders matches
the repository: the README is distilled from 494 lines of benchmark
prose (whose serving story stopped at 0.22 tok/s, v0 figures that
`docs/INFERENCE.md` itself marks superseded) to one page of what the
package is, the doors, install, quickstart, **one table of what is
measured with each row's evidence status**, the caveats that change how
the table reads, what was retired, and where the receipts are.

- `docs/claims.json` — a machine-readable register of 31 claims, each
  with value, unit, model, hardware, conditions, date, status and
  evidence path (`docs/claims-schema.md`). Every number in the README's
  measured section maps to an entry, checked mechanically.
- **`measured-private`** is a status, not a footnote: the serving
  speeds, the parity numbers and the vLLM head-to-head come from a
  private audit tree. Real runs, real receipts, not checkable from
  this repository — and now labelled as such in the README table.
- `docs/STATUS.md` — one page: what you get today, what was retired
  (each with the measurement that retired it), what is open.
- `docs/INDEX.md` — what each of the 42 documents is for and whether it
  is current; the anchored July research record is indexed as such.
- `docs/SERVING-PARITY.md` — the per-family parity table, moved out of
  the anchored `docs/support_matrix.md`, which three same-day PRs had
  appended to after its OpenTimestamps footer. The anchored file is
  restored byte-for-byte to its 2026-07-05 anchored bytes.
- The 0.30.0 corrections (below) are carried in the register as
  `retired` entries so the retractions stay findable.

## Corrections to 0.30.0 — 2026-09-03 (same day)

The 0.30.0 entry below is left as written; this records what the same
day's measurements retired. Full detail: `docs/STATUS.md`,
`docs/METHODOLOGY.md` §13–13.1, `docs/claims.json`.

- **"The fp8 paged KV cache costs +0.047 ppl on Qwen3-30B and +0.022 on
  Granite" — RETIRED.** Those deltas are +0.0058 and +0.0028 nats, and
  the models' measured arithmetic-order floors (two correct forwards
  differing only in the order of the arithmetic) are 0.0095 and 0.0033
  nats. Both deltas sit BELOW their floor: indistinguishable from
  reordering the maths, not a cost of the cache. The rule derived from
  it — "buy headroom back from the cache first" — is retired with it.
- **"gpt-oss's +0.078 nats is 10–20× every other family, a real signal
  about the sinks and sliding-window path" — RETIRED.** Against a
  chunk-free reference (one full forward, `--ppl-oracle full`) the paged
  path sits at 0.00288 nats, below its 0.01758-nat floor. The chunked
  oracle it had been compared against is 6× further from the reference
  than the path it was judging; the 8192-step gap tracked the oracle's
  32 chunk boundaries. Established on a 512-step window; a full forward
  is quadratic in the window and cannot run at 8192.
- **"Chunked teacher forcing is not equivalent to one full forward on a
  family whose layers alternate sliding and full attention" — the
  MECHANISM is retired, the measurement stands.** Widening the window
  past the context leaves the gap (KL 0.0178); every cache class
  reproduces it. The cause is MoE router flips under rounding (4.52% of
  layer-token top-k choices on gpt-oss, 6.77% on Qwen3; flipped tokens
  carry 39× the KL), which applies to every MoE model. Every parity
  delta must be read against a per-model measured floor.
- **The pre-registered KL gate in METHODOLOGY §13 is FALSIFIED** by its
  first measurement: it rejects shipped NF4 experts (0.029 nats and
  93.6% top-1 against 0.01 / 99%). Its 0.01 was calibrated from a signed
  NLL difference and applied to a full-vocabulary KL. Left textually
  unchanged and marked falsified; not retuned.
- **The serving-parity table appended to `docs/support_matrix.md` broke
  that document's OpenTimestamps anchor** (three PRs appended after the
  attestation footer). The anchored file is restored to its anchored
  bytes; the section lives in the new, unanchored
  `docs/SERVING-PARITY.md`.
- **Gemma-4's parity row reads "behaves", not PASS**: −0.0078 nats, the
  same order as the passing families, with the absolute |Δppl| ≤ 0.05
  bar inapplicable at oracle ppl 752. Its load still fails on 2 of 4
  rented hosts (#344); a 2 GiB host-hop fix was merged and reverted the
  same day because the model's largest tensor is 1.375 GiB.

## 0.30.0 — 2026-09-03

### The paged decode path is valid beyond plain-causal attention

Three changes and a floor bump. Until this release the paged B=1 decode
path (the `paged_attention` shim over `Fp8PagedKV` and the kernel
package's split-K attention) computed one attention: full causal,
scaled by `head_dim**-0.5`, one KV geometry for every layer. It served
Qwen3, Mixtral, OLMoE and Granite correctly once #336 forwarded the
attention scale; it could not serve a sliding-window family, an
attention-sink family, or a family whose KV geometry changes per layer.

- **Oracle arm** (#338). `step_decomp.py --ppl-oracle eager` scores the
  same K8 window through transformers' own eager attention with the HF
  cache, shim not registered, in 256-token chunks with explicit
  `position_ids`. It is the reference every paged verdict below is
  measured against; a family's paged perplexity must sit within the K8
  gate of its oracle.
- **Windows and sinks** (#339). The shim reads each layer's sliding
  window (`sliding_window` kwarg, else the module attribute) and
  attention sinks (`s_aux`, else `module.sinks`) and passes them to the
  kernel for decode and verify; prefill attends through SDPA, or
  through a sink-aware manual path when sinks are present. On a kernel
  wheel older than 0.24 the options are dropped with one `PARITY
  WARNING` (the K8 gate then catches the wrong attention); the fallback
  never compares the sinks tensor against a number.
- **Per-layer KV geometry** (#340). `Fp8PagedKV` accepts per-layer KV
  head counts and head dims (Gemma-4: sliding layers at 256/8 beside
  full layers at 512/2), sizes one pool at the widest row and addresses
  each layer at its natural row; the harness reads
  `config.per_layer_config` where transformers 5.16 refuses a global
  attribute.
- **Floor**: `grouped-nf4-gemm >= 0.24.0` (windows, sinks, scale and
  stride overrides in the decode kernels).
- **Instrument fixes found by running the lane**: the K8 record now
  reports the attention compute mode that RAN, from the kernel's own
  tally, instead of the environment request whose default string is
  `f32` (#348); `--ppl-chat` builds the scored window inside the
  tokenizer's chat template for chat-only families (#343);
  `--ppl-oracle upstream` scores the same window through the model as
  transformers loads it, with no e4b loader in the process (#342).
  Known limitation recorded rather than papered over: chunked
  teacher-forced scoring is NOT equivalent to one full forward on a
  family whose layers alternate sliding and full attention (gpt-oss:
  KL 0.0165 nats, top-1 93.9%), so an oracle for such a family must be
  a single full forward.

**Parity verdicts** (paged minus oracle, 8192 sha-matched steps, RTX
5090). Quoted in nats as well as perplexity, because the `|dppl| <=
0.05` bar is only meaningful where perplexity is around 8:

| Family | oracle ppl | dppl | dnats | verdict |
|---|---|---|---|---|
| Granite-3.1-3B-A800M | 7.696 | +0.0219 | +0.0028 | PASS |
| Qwen3-30B-A3B | 8.015 | +0.0467 | +0.0058 | PASS (0.003 ppl of headroom) |
| Gemma-4-26B-A4B-it | 752.5 | −5.839 | −0.0078 | behaves; the absolute bar is inapplicable at ppl 752 |
| gpt-oss-20b | 1336.0 | +108.45 | +0.0781 | no gate exists for this family |

Two things this release makes explicit. **The fp8 paged KV cache costs
+0.047 ppl on Qwen3-30B and +0.022 on Granite against a bf16 cache.**
Every earlier K8 compared paged against paged, so error the two arms
shared cancelled and that cost was invisible; attention-side and
KV-side changes are gated against `--ppl-oracle eager` from here (see
`docs/METHODOLOGY.md` section 13). **And two families have no usable
perplexity gate**: gpt-oss-20b scores 2361 on bare wikitext through
plain transformers, Gemma-4 752, so an absolute 0.05 bar means nothing
for either. A full-vocabulary KL gate is pre-registered in METHODOLOGY
13, thresholds fixed before any KL number was computed.

For gpt-oss specifically: the loader and the served expert tier are
validated faithful (MXFP4 dequant bit-identical against an independent
decode; expert forward cosine 0.991-0.993 against the reference math;
the served layer 0 inside the running model cosine 0.998 per token),
and its +0.078 nats is 10-20x every other family, which is a real
signal about the sinks and sliding-window path rather than a regime
artefact. Serve it if you want; do not read a perplexity from this
harness as evidence about it.

## 0.29.0 — 2026-09-03

### The serving stack outside Qwen: what the first five-model sweep fixed

Four changes, all from one campaign (receipts `INT4B16/P24-GEN-*`): the
same instrument as the Qwen lanes — NF4 bake, K8 perplexity on wikitext
and an out-of-domain C4 shard, graph-timed decode at B=1 and B=16, a
fusions arm, a greedy sample — run on Mixtral-8x7B, OLMoE-1B-7B,
Granite-3.1-3B-A800M, Gemma-4-26B-A4B and gpt-oss-20b.

- **The module's attention scale reaches the decode and verify kernels**
  (#336). The paged `decode` and `verify` branches passed only q/k/v
  and slots, so the fp8 decode kernel always ran at `head_dim**-0.5`.
  GraniteMoe's `attention_multiplier` is 0.015625: NF4 perplexity 4142
  and word salad through the paged loop, **7.72** with the scale threaded
  (validated on a 5090, receipts P24-GEN-E).
  Gemma-4 (scale 1.0, folded into q_norm) is the same class. Families
  whose scale *is* `head_dim**-0.5` (Qwen3-MoE, OLMoE, Mixtral) are
  unchanged. Sliding windows (Gemma-4, gpt-oss) and attention sinks
  (gpt-oss) are still not honoured: those models remain numerically
  invalid on the paged path, and the harness says so rather than
  quoting them.
- **Hessians accumulate on the CPU** (#334, with `grouped-nf4-gemm`
  0.23.0's storage-aware accumulator; the floor moves to 0.23.0).
  Mixtral's 128 attention projections at K=4096 held 8 GB of fp32
  Hessians beside a 23 GB model and ran a 32 GB card out of memory.
  Each batch's Gram is computed on the model's device and only the
  K×K result moves. With that, the Mixtral calibration completes — and
  the one-sided gate refuses the pack (+0.09 ppl on both texts, ×1.01):
  **calibrated int4 attention is a Qwen3-30B-A3B win, not a general
  lever.** OLMoE says the same (+0.60 out of domain, no speed).
- **gpt-oss per-expert biases** (#333): the hot-residency state gathered
  their rows with a CPU index against a CUDA bias; only gpt-oss carries
  them, so the branch had never run.
- **Fusion flags on non-Qwen families** (#333): `E4B_FUSE_T1_GLUE`,
  `_R2` and `_ROUTER_EPI` were consulted only inside the Qwen3-MoE
  serve assembly; elsewhere a set flag did nothing and said nothing.
  The harness now calls the three fusions directly, and each engages
  on matching modules or refuses with a sentence (validated on Granite
  and Mixtral).
- **Gemma-4 config spellings** (#335, #336): `text_config.top_k_experts`
  for the routed top-k; per-layer attributes read uniformly on
  transformers 5.16 heterogeneous configs instead of a global read that
  raises.

What the sweep established beyond the fixes: the generic loader bakes
every one of these layouts; the int4 expert lane is quality-neutral and
**doubles decode on Mixtral** (×2.04 B=1, ×1.97 B=16 over NF4) while a
1B-active model (OLMoE) pays 1.8 % perplexity for it — the K8 gate is a
per-model verdict, not a property of the lane.

## 0.28.0 — 2026-09-03

### Calibrated int4 attention for serving, and a gate that knows what calibration does

One serving lane (#331), gated behind `E4B_SERVE_ATTN_INT4_CALIB=1`.
The uncalibrated int4 attention lane was refused on quality at +0.056
perplexity, and the fp8 lane that followed showed the obvious fix is
not a fix: 4.6× lower weight error bought almost nothing, because
weight error is not what the gate measures. This lane keeps the same
grid, bytes and kernel and changes only *which* grid point each weight
lands on: `int4_attn_calib` records `H = 2·XXᵀ` for every attention
projection through forward hooks over a short calibration text, and
`Int4Linear` takes a `packer` closed over that Hessian
(`grouped-nf4-gemm` 0.22.0's `gptq_pack_int4_b32`). A projection
without a Hessian is refused, never silently packed uncalibrated under
the calibrated banner.

What it measures (RTX 5090, receipts INT4B16/P21–P22c). Calibrated on a
C4 validation shard, the pack scores **−0.042** against bf16 attention
on the wikitext gate and **−0.115** on an out-of-domain C4 text — an
improvement on both, with the same sign. Calibrated on wikitext-2
*train*, two window choices scored −0.017 and −0.078: an improvement
that moves with the calibration windows is fitting the scored text, and
that pack is refused. Speed: **×1.06 at B=1** (5.185 → 4.888 ms/step,
204.6 tok/s on that box). At batch the int4 GEMV *loses* — each row
re-streams the projection, ×0.90 at B=16 on its row axis and ×0.52
through a per-call dequant — so `Int4Linear` serves one row on the
GEMV and every larger row count on a bf16 weight dequantised once and
cached (+≈1.8 GB for 96 projections); batched decode costs exactly what
bf16 attention costs. A batched int4 attention that wins needs a
small-M int4 GEMM with weight-tile reuse; that is a kernel, not this
release.

`experts4bit_qlora.k8_gate` is the perplexity gate as one function.
Uncalibrated formats keep the symmetric `|Δ| ≤ 0.05`. Calibrated packs
are gated one-sided, `Δ ≤ +0.05`, and an improvement is trusted only
with the same sign on two scoring texts, one outside the calibration
domain; `bench/hybrid-g9/step_decomp.py --ppl-source c4val1` scores the
K8 instrument on such a text, and `ppl_source` travels in the output
beside `text_sha`. `grouped-nf4-gemm >= 0.22.0` is now the floor.

## 0.27.0 — 2026-09-02

### Glue round 3: the router epilogue, one launch per layer

One feature (#329), gated behind `E4B_FUSE_ROUTER_EPI=1`. Rounds one
and two folded the norms, the residual add and the rotary chain; the
census's largest remaining cluster at decode was what the router does
after its GEMM — a softmax over every expert, a top-k that torch
serves with a gather plus a bitonic sort, a sum and a divide. Five
launches per layer become one. The GEMM itself does not move, which is
what separates this from the router fusion refused earlier on
occupancy: a program reads 512 bytes of logits rather than the router
weight matrix.

Patching is licensed by a semantic probe against the module's own
forward, requiring both the selected expert **set** and the weights to
match the reference epilogue — a routing change is not a rounding
change. Routers that bias logits before selection, or that softmax
only the selected logits with renormalisation off, are refused by it
despite sharing every structural attribute. A test also pins the
algebraic fact that with `norm_topk_prob` on, top-k-then-softmax *is*
the reference function (the partition function cancels), so the probe
neither can nor needs to separate those.

Measured against the round-two tip: **1.0735x at B=1** (5.657 → 5.270
ms, 176.8 → 189.8 tok/s) and **1.0464x at B=16** (13.823 → 13.210 ms,
1,157.5 → 1,211.2 tok/s aggregate), engagement confirmed by kernel
census in every treated arm and absent in every control. The paired
quality gate PASSES at **-0.01968 ppl** over 8,192 sha-matched steps.

## 0.26.0 — 2026-09-01

### Glue round 2: two more decode folds, both lanes paid

One feature (#326), gated behind `E4B_FUSE_T1_GLUE_R2=1`. Where round
one fused the RMSNorm call itself, round two folds what the censuses
showed around it:

- the decoder layer's `residual + attn_out` disappears into the
  post-attention norm, one call returning both the new residual and
  the normed activation;
- each of q/k's per-head norm folds together with the rotary chain
  into a single launch per projection.

Licensing follows the round-one lesson exactly: every structural
attribute is checked before patching, the norms must pass the semantic
probe that rejects centered variants, and the attention fold only
touches an attention this package already fused — it replaces that
forward rather than half-patching an unfused one. A cos/sin tensor
that upstream broadcasts across the batch is materialised per row, and
any other layout keeps the upstream chain rather than rotating with
the wrong positions. Off decode shapes everything falls through, and a
zero-match enable refuses instead of running as a quiet no-op.

Measured on one box against the round-one tip: **1.1557x at B=1**
(6.575 -> 5.689 ms, 152.1 -> 175.8 tok/s) and **1.0916x at B=16**
(15.305 -> 14.021 ms, 1044.7 -> 1141.1 tok/s aggregate), engagement
confirmed by kernel census in every treated arm and absent in every
control. The paired quality gate PASSES at delta -0.00105 ppl over
8192 sha-matched steps.

## 0.25.0 — 2026-09-01

### Opt-in fused RMSNorm glue for single-stream decode

One feature (#324). `fuse_t1_glue(model)` — env-gated behind
`E4B_FUSE_T1_GLUE=1` and hooked from the qkv fusion pass — swaps
structural RMSNorm sites on the decode path for the kernel package's
single-launch row-parallel norm (its #306), with decode-shape and
bf16 gates. Patching is licensed per module by a semantic probe: a
deterministic probe tensor through the module's OWN forward must
match the non-centered reference formula at rtol 2^-5, which excludes
centered variants (`x_norm * (1 + w)`) that a structural name match
cannot distinguish; a vacuous enable (zero sites patched) refuses and
reports the probe-skip count. Composed on the B=1 serve lane: 8.344 →
6.469 ms per step on the rental class, quality gate PASS (Δppl
+0.0136 @ 8192 paired sha-matched steps).

## 0.24.2 — 2026-09-01

### The fused-SwiGLU wiring is retired — it never fired and would not pay

One change (#322), a removal. Census absence across every composed
receipt showed the epilogue-fusion path added in the tail-fusion round
never executed: its activation identity gate never matched the live
activation-registry object. Two dedicated A/Bs with the gate widened
then bounded the fusion's value below A/A noise at BOTH batch sizes
(1.0001× at B=1, 1.0000× at B=16, A/A ≤ 1.0005) — under graph replay
the three-launch epilogue chain is effectively free. Dead code that
measured null twice comes out rather than being re-gated a third time;
the kernel-side helper stays in the kernel package, tested and
documented as unused by this consumer.

The same measurement pass re-baselined B=1 on the released stack:
**int4 over NF4 is now 1.197×** (9.969 → 8.327 ms) — up from 1.098× at
certification, the quantise-grid fix having compounded at B=1
unannounced. The prior tail-fusion release-note attribution is
corrected accordingly: its composed 1.104× belongs to the one-launch
tile table and the gather-folded quantise alone.


## 0.24.1 — 2026-08-31

### Batched int4 decode routes through the split-K GEMV

One change (#320), no new API: decode shapes (R ≤ 256) on the int4
expert store now take the per-row split-K GEMV instead of the M-tile
GEMM. The engine probe showed the M-tile's binding constraint at B=16
top-8 routing is padded MMA lanes (~1–2 live rows per 16-row tile), not
occupancy — a split-K M-tile barely moved, while the shipped GEMV wins
gate_up 1.92× / down 1.28× and serves rows in input order, deleting the
tile table, gather, and unsort on that path. Prefill keeps the M-tile,
where tile reuse is real.

Composed on the B=16 serving step: 21.81 → **16.61 ms** (734 → 963
tok/s aggregate), A/A ≤ 1.0016, base reproduced cross-box to 0.03 ms.


## 0.24.0 — 2026-08-31

### The int4 serve lanes, the coverage matrix, and the batched-step campaign

Everything merged since 0.23.0, receipts in the audit tree:

- **Opt-in uniform-int4 expert serving** (`enable_serve_experts_int4`):
  repacks the hot expert stacks to int4-b32 and serves decode from
  them — single-stream ×1.098 with Δppl −0.084 on the certified
  family; quality gates PASS on three families (qwen3_moe −0.084,
  olmoe −0.477, qwen15moe +0.030 over 8,192 sha-matched steps).
- **Coverage via load plans**: the enabler routes through
  `plan_moe_checkpoint` and the loader's own reader/fusion helpers, so
  it inherits every family keymap and source quant format; pre-fused
  families pack off the plan's passthrough; split-K sizing follows the
  config's routed-expert count. Named refusals for what the engine
  cannot host.
- **gpt-oss on the arena path**: per-expert biases now carried
  resident and de-interleaved to the baked layout, with the residency
  gather indexed on the biases' own device.
- **The batched (B=16) campaign**: device-grouped decode routes
  through the grouped int4 GEMM; one-launch-per-side batched KV
  append; fused tile table / gathered quantise / fused SwiGLU wiring;
  a kernel census for the batched graph replay (Stage-A budget
  contract). Composed: 39.5 → 21.8 ms per step (405 → 734 tok/s
  aggregate) against the pre-campaign graph lane.
### Corrections

- **The `>275` refutation's basis is corrected; the verdict is not.**
  0.23.0 argued `>275 tok/s REFUTED-AS-COMPOSED` from "device work alone is
  8.43 ms/step". That census is an **eager-path** Self-CUDA sum (basis disclosed
  in RESULTS-sv1), and both it and the 7.25 ms graphed-default wall sit *above*
  the certified opt-in's own **6.476 ms** wall — which, since device work cannot
  exceed wall-clock on a single stream, is the correct bound for the certified
  path. (Box provenance disclosed: 8.43 is box 48728047 / anchor 7.27 ms, 6.476 is
  48709950 / anchor 7.25 ms — same class to 0.3%; and the load-bearing inequality
  is within-run on 48709950, so it does not depend on the cross-box step.) The refutation stands: 6.476 → 3.636 ms is a **1.78×** device-work
  reduction (not 2.32×), still not available from orchestration. The 0.23.0 entry
  below is left unedited so the record of what was claimed survives; the full
  reconciliation is appended to `bench/hybrid-g9/sv1/RESULTS-sv1-census.md`.
  Do not re-derive the 250 verdict from 8.43 — against the certified wall that
  frame needs 1.62×, and it stays OPEN per `RESULTS-250-closing.md` (#283).
- **RESOLVED by measurement (2026-08-26): TR2's tokens/step is 3,086.**
  Re-ran the TR1/TR2 recipe on a rented RTX 5090 (driver 595.84, transformers
  5.5.0). The trainer's own log prints `116 tok/s` at `26.6 s/step`, i.e.
  **3,086 tokens/step** — confirming the ~3,072 that had to be inferred, and
  closing the gap recorded below. Emit `tokens_per_step` explicitly anyway; a
  rate whose denominator must be back-derived from a second printed rate is not
  a receipt. Recipe validity confirmed by held-out eval reproducing TR2's
  published figures: base 1.010 -> measured **1.0093**, grouped 1.009 ->
  measured **1.0118**.
- **The 13.47x needs restating: it is ~7.2x against a current baseline.**
  Same box, same recipe, identical 321,257,472 trainable params:

      e4b base (bnb)      26.6  s/step   116 tok/s   24.36 GB   eval 1.0093
      e4b grouped          3.7  s/step   846 tok/s   24.37 GB   eval 1.0118
      -> same-box speedup 7.19x, against the registered 13.47x

  **The grouped arm did not regress** — it reproduces TR2's 3.77 s/step at 3.7.
  What moved is the BASELINE: 50.86 -> 26.6 s/step, because transformers v5
  ships `grouped_mm_experts_forward` and fused the per-expert loop upstream.
  Roughly half the published multiple is now upstream's work, not ours. Anyone
  who reruns 13.47x on a current stack gets half of it. Restate as: *7.2x over a
  current-transformers bnb baseline, same-box, with held-out eval parity.*
- **Reproducibility gap: no shipped tool bakes the arena.** `TRAIN_ARENA` takes a
  PATH to a pre-baked arena, is undocumented in `train.py --help`, and
  `nvme_arena.bake_expert_tensors` only RELOCATES pre-quantized tensors. The HF
  checkpoint ships bf16 experts, so reproducing TR2 from published artifacts
  requires writing a quantize -> emit-nf4-snapshot step yourself (~60 lines;
  15.19 GiB snapshot in 13.3 s, 16.3 GB arena bake in 7.7 s). Ship it as
  `experts4bit_qlora.bake`. Also: passing a non-path truthy value releases
  16.31 GB of expert storage BEFORE discovering the arena is missing, then dies
  on `FileNotFoundError`. Check the path exists before `_release_expert_storage`.
- **Recorded gap (now resolved above): `tokens_per_step` is not in the TR2 receipts.**
  `tr2_report.json` records `seq: 192`, `grad_accum: 4`,
  `token_budget_effective: 1024` — but not the per-step token count that the
  published "~59 → ~800 tok/s class" figure divides by. The rate is reproducible
  only by inferring ~3,000 tokens/step from the two s/step values. The s/step
  numbers (50.86 → 3.77) and the 13.47× are unaffected and remain the primary
  quantities. Recorded rather than back-filled: the receipts are committed and a
  denominator reconstructed after the fact is not a measurement. Emit
  `tokens_per_step` explicitly in the next training run.
- **Attention int4 stays fit-only** and the lm_head stays bf16 — the
  measured refusals (chain cost at B=1, occupancy at M=16, lm_head
  quality) are recorded with receipts.


## 0.23.0 — 2026-08-26

Minor release: the batched-serving certification and the honest
closure of the single-stream arc.

### Performance (serving)

- **The B>1 CUDA-graph decode loop is certified** (RESULTS-bv3b:
  PASS 3.10×): `--b1d-loop graph --batch 16` serves **419 aggregate
  tok/s** on the reference class versus 135 for the eager scheduler
  at equal compile coverage — with device-vs-eager MoE grouping
  proven BITWISE on all 48 layers, 11/16 rows token-identical, and
  the residual divergence deep inside the registered floor. Uniform
  dynamo-limit envs (`E4B_RECOMPILE_LIMIT`/`E4B_ACCUM_RECOMPILE_LIMIT`)
  ship as the comparability mechanism — equalizing compile coverage
  also made the eager baseline ~9% faster.
- **>275 tok/s single-stream is REFUTED-AS-COMPOSED** per the SV1
  pre-commitment: device work alone is 8.43 ms/step against the
  3.64 ms the target requires. The certified single-stream ladder
  closes at ~138 default / 154.4 with the dot-pad knob; the
  throughput lane is the batched loop.

### Corrections and instruments

- RESULTS-bv3's grouping-numerics attribution is corrected in
  public: the parity probe (three review rounds of vacuity binds)
  measured the paths bitwise-identical; the real confound was
  dynamo compile coverage.
- Degeneracy handling amendment: workload rows that loop identically
  in BOTH arms are excluded with disclosure (75%-clean floor;
  treatment-induced degeneration still refuses).

## 0.22.0 — 2026-08-26

Minor release: the training arc. Headline: **QLoRA training of the
30B reference recipe is 13.47× faster** — `TRAIN_ARENA` routes expert
forward/dgrad through the grouped gnf4 kernels instead of the
per-expert bitsandbytes chain, adjudicated PASS with learning
identical to the third decimal (RESULTS-tr2, receipts committed).

### Performance (training)

- **`TRAIN_ARENA=<arena>` is the documented default path for
  arena-holding models** (RESULTS-tr2: PASS). 50.86 → 3.77 s/step on
  the reference recipe (Qwen3-30B-A3B QLoRA, RTX 5090 class): ~59 →
  **~800 tok/s class**; a training epoch of the TR1 recipe drops from
  ~17 minutes of stepping to ~75 seconds. Kernel launches per step:
  2.92M → 126k (23.1×). Final held-out evals 1.010 (bnb) vs 1.009
  (grouped). Engagement releases the loader's bnb expert storage via
  shape-preserved meta twins BEFORE the tier build (the build peak
  OOMed a 32 GB card otherwise) and refuses partial engagement.
- Census instruments behind `TR1_CENSUS=1`: CUDA-event phase
  brackets in the shipped trainer loop, loss-in-receipt, effective
  token budget recorded, profiler window (CUDA-only) in its own run.
  The TR1 census that found the launch-storm (GPU ~8–11% busy on the
  bnb path) ships with receipts; its 9.1× bound is recorded as
  falsified-conservative by the TR2 receipts.

### Performance (serving)

- **Dot-pad × F2 composition certified** (SV1, K6-B PARTIAL):
  `GNF4_GEMV_DOTPAD=1` on the 0.15.1/0.21.0 defaults measures
  **154.4 tok/s** single-stream on the reference class (6.476 vs
  7.25 ms; 127 tokens identical). Knob remains opt-in, now with a
  composed receipt.
- **B>1 CUDA-graph decode loop implemented; verdict REFUSE,
  standing** (BV3): 38.0 ms/step at B=16 — a would-be 3.44× over the
  eager scheduler (421 aggregate tok/s), reproducible to 0.01 ms —
  but one row diverged from the eager stream at step 3 and the
  registered identity gate refused. Receipts attribute the
  divergence to device-vs-eager grouping numerics; BV3b (logit-parity
  probe + a kernel-swap identity frame) is registered before any
  re-adjudication, and no BV3 wall number is citable until it lands.
  The machinery ships: `--b1d-loop graph --batch B`, slot-list graph
  KV init, batched capture-safe appends.

### Instruments and gates

- The census composer's same-workload A/A gate carries a measured
  absolute-noise floor (per-step box jitter is ~250 ms regardless of
  step duration; a purely relative gate mis-refuses fast-step runs).
  All amendments disclosed in the preregs with self-tests proving the
  refusal directions survive.

## 0.21.0 — 2026-08-25

Minor release: the serving-harness half of the K/F campaign, the
speculative-verification machinery, and two default flips. With gnf4
0.15.0, single-stream decode on the reference class moves from ~66 to
**~139–140 tok/s** at defaults. Every default cites an adjudicated
verdict.

### Performance (defaults changed)

- **`--compile-layers` compiles the dense layer bodies in `default`
  mode** (F1-B1, PARTIAL ships): the 74.3 → 94.2 tok/s rung. Paged
  attention and the MoE tier stay dynamo-disabled; the disable is
  re-asserted after any shim unwrap (the wrapper-cancellation bug is
  tested against).
- **Fused T=1 KV append is the default** (F1-B2, PASS): construction-
  time kernel resolution with graceful degrade, loud refuse on an
  explicit `E4B_FUSED_KV_APPEND=1` without triton. The 94.2 → 133.4
  rung. Rollback: `E4B_FUSED_KV_APPEND=0`.
- **Fused QKV projection is the default at load** (F2 T2,
  RESULTS-f2-tail PARTIAL ships): one matmul replaces the three
  per-layer q/k/v GEMVs; +0.120 ms/step under a 0.001 ms A/A,
  token-identical over the 127-step receipt, per-projection numerics
  inside `max|ref|·2⁻⁷` on all 48 layers. Rollback: `--no-fuse-qkv`.
  NOT claimed bitwise — the prereg records the falsification of that
  claim by its own CPU gate.

### Correctness

- **Fused KV append resolves by the KV's DEVICE, not kernel
  presence.** gnf4 0.15.0 ships `fp8_kv_append_t1`, so the old
  presence-only check enabled the fused path on CPU-device KVs and
  the first append died inside triton's driver ("0 active drivers" —
  caught by this release's own CI the hour 0.15.0 hit PyPI). A
  non-cuda KV now degrades to the eager append; an explicit
  `E4B_FUSED_KV_APPEND=1` on a non-cuda KV refuses loudly. Resolution
  factored into `_resolve_fused_append` with the full cell table
  under test.

### New capabilities

- **Speculative-verification machinery** (S2-lite/S3), shipped as
  capability with a negative headline: verify-mode paged attention
  (K+1 rows against one slot with length-staggered causality),
  `rewind`/`rewind_nosync`/`seen_device` (device-truth forwardness
  under graph mode), and device-side expert grouping
  (`DEVICE_GROUPING`, capture-legal via gnf4's grouped API).
  **The S3 verdict REFUTED grouped verification at this scale**
  (0.66× vs the decode anchor at K=16–64) — and with it the 425 tok/s
  single-stream target on the registered path. The machinery ships
  because the measurement required it and future models may invert it.
- **Batch-decode curve harness** (BV2): the eager scheduler tops at
  ~124 tok/s aggregate (B=16) and is host-bound; the registered next
  lever is a B>1 CUDA-graph decode loop (not in this release).
- Census instruments: step-budget decomposition, dispatch-mode
  elementwise attribution (`--ew-attr-out`), recompile guards.

### Measurement and research artifacts

S1 acceptance-length receipts (2.9–3.9 tokens accepted at K=16/32/64,
MARGINAL), the S2 singleton bound and its correction (the bound is a
valid conservative upper bound, not a measurement of grouped verify),
S3 grouped-verify refutation, BV2 curve receipts, and the F2 arm
receipts. All under `bench/hybrid-g9/`.

## 0.20.0 — 2026-08-23

Minor release. 0.19.1 predates the hybrid tier (Phases 1–9), the cold engine's
CPU destination, the residency and scheduling work, and the correctness round
that followed them.

### Dependency floor

**`grouped-nf4-gemm>=0.14.0` is now required** by both the `[fast]` and `[test]`
extras, raised from `>=0.12.0`. 0.14.0 carries
[grouped-nf4-gemm#205](https://github.com/pjordanandrsn/grouped-nf4-gemm/pull/205):
the MXFP4 port of the grouped kernels had dropped NF4's int64 `eid` promotion,
so `eid * stride_be` overflowed signed-int32 and an MXFP4 expert stack past
2^31 bytes raised an illegal memory access. Every arena and cold path here that
stacks MXFP4 experts crosses that boundary at DeepSeek/K3-class expert counts,
and there is no degraded mode — the launch faults.

### New capabilities

- **Hybrid three-tier engine** (`engines/hybrid.py`, `engines/hybrid_train.py`)
  — VRAM / DRAM / NVMe with a placement solver and three-bus executor (#148),
  speculative prefetch that routes layer L+1 from L (#149), and QLoRA backward
  over the whole three-tier engine (#150).
- **CPU router** (`engines/cpu_router.py`) with a native epilogue (#146, #147).
- **A second execution destination for cold experts** — NVMe → DRAM → **CPU**
  (#168), with `cold_dest="deadline"` choosing against both engines' committed
  work (#179), and setup reads given their own tier so the serving tier can
  carry a direct landing (#177).
- **Paged KV and attention** — `engines/paged_kv.py`, `engines/fp8_paged_kv.py`,
  `engines/paged_attention.py`, `engines/paged_runner.py`, `engines/fp8_kv_cache.py`:
  tiered paged KV over the generalized RowPool (#152), an FP8 KV cache with a
  quality oracle (#153), and paged FP8 attention with a continuous-batching
  engine (#155).
- **Placement and scheduling** — `engines/placement.py`, `engines/scheduler.py`,
  including the batched placement law (#155).
- **Thin-layer DRAM routing** — layers below a static population threshold serve
  their DRAM experts on the GPU (#159).
- **`protected_rows` is exposed** — R1–R10 cannot be measured without it (#172).

### Performance

Measured on the paths shipped here:

- **GPU cold stacks read the cold view (#184): −5.9% at 5% cold mass, −12.1% at
  20%.** `_TieredStack.index_select` had rebuilt every routed row with
  `segment_tensor` on each call, including for experts the cold view had already
  materialized. Gate 1 attributes ~98% of cold cost to exactly that staging.
- The hybrid tier adopts the fused expert-FFN kernel — one pool wake per DRAM
  call (#156). On the full-stack re-measure the `fused_ffn` **default flips to
  `False`** (#157): it wins only with work-stealing underneath it, so the
  default follows the measurement rather than the feature.
- Direct scatter is wired into the engine, and the engine stops building one
  cold view per module (#176).

### Correctness

- **Four unread Bugbot findings addressed (#181).** `gate1_cold_sweep`'s
  `beats_both_fixed` read an *absent* fixed arm's `exposed_ns` as `+inf`, so
  "the dynamic arm beat a measurement that does not exist" **passed** the
  clause — an inversion in a module whose stated rule is the opposite. Plus two
  fallback defects and DRAM load the deadline estimator charged to nobody.
- **The direct landing and the GPU cold path cannot share a tier (#180)**, and
  #185 then lifted the direct-landing bar for GPU and mixed destinations once
  #184 removed the `row()` call that required it — the fallback and the external
  landing are now mutually exclusive by construction rather than by timing.
- `cold_stats` forwards the reuse ratio's own denominator (#182, #183).
- Deterministic per-token combine — unique `(token, slot)` writes and one
  fixed-order sum (#151).
- `cold_dest` is a rounding path, not just a bus, with matched-reference tests
  pinning it (#173, #174).

### Measurement and research artifacts (no runtime behavior change)

`bench/hybrid-g3` … `bench/hybrid-g9` and `bench/cpu-router` hold the gate
record for the hybrid campaign. Reported as measurements, not as product claims:
G8 closed at 0.51–0.68 with the full stack, with the residual identified as
intra-call (#163); G9's program best was 45.4 tok/s (#163) and its TTFT missed
by single-digit milliseconds (#161); the call-size bandwidth ramp is the
frontier while L3-warmth and spin-window mechanisms were refuted (#164); B=8
balance closed at 0.978 on reference silicon with B=16 open at 0.618–0.698
(#166, #167); and a TTFT pool-size effect was **retracted** as a first-run
compile artifact (#158).

## 0.19.1 — 2026-08-14

### The compacted stack: measured, and closed

`engines/nvme_train.py` has always named a compacted `[R, ...]` staged stack as
what would fit a big MoE on a small card, and declined it because it splits the
kernel's expert-id space from the adapter's. Its docstring now carries the
measurement that closes the question instead of leaving it as future work.

A compacted stack only saves memory when a batch routes to far fewer than `E`
experts. Counted exactly on DeepSeek-V4-Flash's hash-routed layers — expert
selection there is a frozen `tid2eid[input_ids]` lookup, so no forward and no
149 GB download is needed — over real prose: **224 of 256** distinct experts at 128
tokens, **249** at 256, and **all 256** by 512. Compaction saves 12%, 3%, then
nothing. At decode it saves **98%**.

And where it pays, it already exists: `_HotResidency._cold_contrib` takes
`torch.unique(...)` and `index_select`s only the routed rows, and `_TieredStack`
never materializes an `[E, ...]` tensor. The change has no beneficial home.

Worth naming the quantity that misleads: this is **not** the routing skew informed
hot sets exploit. Those care which experts are hit *often*; compaction cares which
are hit *at all*, and heavy frequency-skew still leaves the tail touched.

Scope: hash-routed layers only, 3 of 43 on V4-Flash. The other 40 use the learned
top-k router and need a real forward.

The probe now ships in the **sdist** (`bench/routing/distinct_experts.py`, via a new
`MANIFEST.in`) so the table above can be re-run rather than taken on trust. Wheels
are unaffected — they carry the package only, as `tools/` and `scripts/` always have.

## 0.19.0 — 2026-08-14

### The MXFP4 arena forward: projection math, not just staging

Option B (0.18.0) made an MXFP4 arena's bytes *land* correctly; it did not make the
NF4 arithmetic interpret them, and `_e4b_mxfp4_arena` only flagged the module. This
wires the compute half.

`_dequantize_expert` is overridden per instance, so the module's **own** forward
becomes correct — whichever forward that is. Every reference lane funnels through
`_project` (`ExpertsNbit.forward`, `_DeepseekV4ForwardMixin.forward`, and
`ExpertsLoRA._base_project`), so the arch's epilogue is applied by the arch's own
code rather than re-derived. `forward` additionally routes to
`mxfp4_grouped.gemm_mxfp4_grouped` under CUDA + bf16 + **no autograd graph**: that
kernel has no `autograd.Function`, so a training step routed there would produce no
`dL/dx` and silently stop learning.

Graded against the pure-torch oracle (`dequantize_mxfp4` then matmul), never
another accelerated lane, on **two cards**. Projections 0.000e+00 on an L40S; on a
3090 the down projection's GEMV branch lands at 3.344e-06, so "the projections are
exact" was a property of that card and not of the kernel. The asserted bound is
`< 2e-2`. A control grades the same forward against an *unclamped* oracle and
requires it to be rejected at ~1.0, so the fixture demonstrably tells the epilogues
apart. Receipts in `bench/mxfp4-arena-train/`.

### Two silent-wrong-answer surfaces closed

`ExpertsLoRA._use_infer_gemv` would have routed single-row MXFP4 projections through
bitsandbytes' `gemv_4bit`. Its own probe could never have caught that:
`_gemv_4bit_matches_dequant` quantizes a synthetic weight and compares bnb against
bnb, so it never reads the module's buffers.

`enable_fast`/`enable_fast_train` would have patched the NF4 grouped kernel over
MXFP4 storage — such a base passes every check they had (`quant_type="nf4"` by
class, `_apply_gate` present) and then dies on a `view(E, n1, k1 // 64)` of a buffer
holding one scale byte per 32 elements. Both refuse explicitly now.

### The dense FP8 dequantize cost 6x what its docstring claimed

`Fp8BlockLinear` said the transient was "one weight at a time (~67 MB for V4's
largest)". That counted the bf16 result only; the fp32 route also held an expanded
fp32 scale, `weight.float()` and the fp32 product — 12-14 bytes per parameter, so
~403 MB for `wq_b [32768, 1024]`. The aligned path now decodes in `dtype` with a
broadcast scale, 2 bytes per parameter, and is **bit-exact**: e4m3 -> bf16 is
lossless and an e8m0 scale is a power of two, so the multiply cannot round. Verified
including deliberately over/underflowing exponents. Ragged shapes keep the fp32
route, which the docstring now says rather than denying.

### `util.container_free_bytes()`

`enable_nvme_residency` tells callers to size `hot_rows` from measured free RAM and
the package gave them no correct way to do it in a container. Reads cgroup **v2 and
v1** — pods are v1, so a v2-only reader measures nothing there — and counts
reclaimable page cache as free, because both versions count it as *used*: straight
after a 138 GiB arena bake the naive read said **18.3 MB**.

### `grouped-nf4-gemm` floor raised to 0.12.0 — hard, not advisory

Below it, `nvme_residency._ST_TO_TORCH` has no entry for `F8_E8M0`, the tag
DeepSeek-V4 uses for its MXFP4 expert scales, so a real V4 arena cannot be staged
for training at all. No degraded mode, nothing skips.

### The 284B claim this was built for: NOT established

`bench/mxfp4-arena-train/` registers a prereg with five OTS-stamped amendments and
reports it either way. **P1 confirmed** (the unquantized control OOMs, twice, on
memory). **P2 refuted** — no rung of the registered batch ladder fits in 24 GiB.
P3/P4 ungraded; no training step completed. The expert path itself runs end to end
against a real 284B checkpoint's own MXFP4 bytes (43/43 modules patched, gates
green); what stops it is `[E, ...]` staging at 2.12 GiB/layer, which
`engines/nvme_train.py` already documents as a deliberate trade-off it declines to
make.

## 0.18.0 — 2026-08-13

### Accepts an arena whose absmax is stored bf16

`grouped-nf4-gemm` 0.11.0 can bake absmax as bf16 — 11.1% of a Qwen3-30B row down to 5.6%,
and bitwise lossless for a bf16 checkpoint, because absmax is `|w|.amax()` over a block and
is therefore one of the source magnitudes. `check_arena_geometry` refused **any** dtype
difference between the module's home and the arena segment, which rejected such an arena
outright.

The relaxation is narrow: only casts in gnf4's exported `widening_casts()` table
(bf16/fp16 → fp32) are accepted, and this **imports that table** rather than growing a
second copy that can drift. Every other mismatch still raises — "this arena was not baked
from this model" is the far more common cause of a dtype difference.

**VRAM and the kernel contract are unchanged.** The staging destination is still the
module's fp32 home, so the kernel keeps receiving the fp32 absmax it specifies. The
geometry check returns the *module's* dtype for exactly this reason: returning the arena's
would allocate a bf16 destination, `segment_into` would take its memcpy path, and the
kernel would get bf16 absmax where its contract says fp32 — wrong scales, finite numbers,
no error.

**The `grouped-nf4-gemm` floor is raised to 0.11.0**, which is where `widening_casts` and
the CPU-scaled queue depth land.

### `hot_rows` below its floor is refused at attach, and `qd` stops being pinned

**The floor is enforced now.** A stage requests every expert one forward routed and each
protects a slot from eviction, so an undersized tier raises inside `ColdTier.ensure` — but
only when a forward is finally unlucky. Measured on Qwen3-30B-A3B at seq 384, top-8: a
forward routes a **median of 63 unique experts and a max of 97 of 128**. A tier sized to
the median survives most forwards and kills the run on one of them, minutes in, after the
checkpoint has loaded and the arena is open.

`num_experts` is the worst case that request can reach and is known at attach for free, so
`enable_nvme_train_residency` now refuses below it in the pre-flight — which opens nothing
and so has nothing to unwind. The message carries the floor, what it costs in pinned RAM at
this arena's row size, and where to size it from.

Above the floor it stays a **RAM-for-disk dial**: on Qwen3-30B, 128 rows costs ~4 GB pinned
and reads 14.4 GB/step; 3216 costs ~12 GB and reads 2.65 GB/step — 3.2× the RAM for 5.4×
fewer bytes.

**`qd` now defaults to `None`.** It was `qd: int = 4` and was forwarded on every call, so
`grouped-nf4-gemm`'s CPU-scaled queue-depth default never applied to the training path —
the exact path its measurement came from. The key is now *omitted* rather than forwarded as
`None`, because on a `grouped-nf4-gemm` older than that default (the floor is 0.10.0, which
predates it) `qd=None` would reach `ThreadPoolExecutor(max_workers=None)` and silently open
up to 32 reader threads instead of 4.

## 0.17.5 — 2026-08-13

**Docs-only. A third model corrects what two models got wrong, and the arena's cost in
TIME is measured for the first time.**

### The arena requirement is not flat

0.17.4 said the host requirement scales with expert bytes "while the arena requirement is
set by `hot_rows` and stays roughly flat". The host half holds. **The flat half is wrong.**
Gemma-4-26B-A4B — now bakeable, see below — lands between the other two and breaks it:

| | expert bytes | arena req | host req | ratio | ⇒ dense baseline |
|---|---|---|---|---|---|
| OLMoE-1B-7B | 3.62 GB | 2.28–2.42 GB | 5.91–6.17 GB | 2.56× | ~2.19 GB |
| **Gemma-4-26B-A4B** | **12.85 GB** | **5.10–5.37 GB** | **19.33–20.40 GB** | **3.80×** | **~4.94 GB** |
| Qwen3-30B-A3B | 16.31 GB | 3.89–4.03 GB | 24.70–25.77 GB | 6.40× | ~3.69 GB |

Gemma has **fewer** expert bytes than Qwen3 and a **larger** arena requirement, because its
dense side is bigger — a dense MLP in every layer plus a 262144-token vocabulary. The ratio
is expert bytes measured **against the dense baseline**, not expert bytes alone. Two points
were consistent with "flat"; three are not.

### What the arena costs in time

First timing measurement in this line, on two architectures because
[no timing claim ships on one machine](bench/host-ram-ceiling/RESULTS-timing.md):

| | RTX 3090 (Ampere) | L40S (Ada) | travels? |
|---|---|---|---|
| **load**, host/arena | 3.39× / 6.76× | 3.12× / 5.87× | **yes** |
| **step**, arena/host | 1.331 / 1.238 | **2.248 / 1.708** | **no** |

**The load saving travels; the step cost does not, and it is worse on faster hardware.**
Going 3090 → L40S the host arm's step time nearly halves while the arena arm improves only
1.2–1.4×, because part of every arena step is an NVMe read and the disk does not care which
GPU you bought. **Quote the step cost with the card attached, or not at all.**

**Size `hot_rows` to the routing floor, not to free RAM.** Sweeping 128 / 384 / 1024 on
Qwen3 produced no resolvable step-time difference, while 1024 cost resolvably more load
time and ~8× the pinned RAM. `hot_rows` also does not travel between models: it has a hard
floor at the experts one forward routes (OLMoE 89% of a layer, Qwen3 52%, Gemma-4 50%).

Both receipts carry their pre-registrations, including a mis-specified gate that was
amended before any re-run, and a `hot_rows` U-shape that one round suggested and three
rounds withdrew.

## 0.17.4 — 2026-08-13

**Docs-only. The scaling claim 0.17.3 flagged as unmeasured is now measured.**

0.17.3 said the ratio "should widen substantially on larger MoEs — but that is the
mechanism's prediction, not a measurement". On **Qwen3-30B-A3B**, which has 6144 experts
against OLMoE's 1024 and **4.50× the expert bytes**, it is **6.40×**. Receipt, raw ledger
and the pre-registration it is scored against:
[`bench/host-ram-ceiling/RESULTS-scaling.md`](bench/host-ram-ceiling/RESULTS-scaling.md).

| | expert bytes | host-RAM path | arena path | ratio |
|---|---|---|---|---|
| OLMoE-1B-7B | 3.62 GB | 5.91–6.17 GB | 2.28–2.42 GB | 2.56× |
| **Qwen3-30B-A3B** | **16.31 GB** | **24.70–25.77 GB** | **3.89–4.03 GB** | **6.40×** |

**At an 8.59 GB ceiling Qwen3-30B-A3B is OOM-killed on the host-RAM path and trains to
completion on the arena.** The host requirement grew **×4.18** against **×4.50** in expert
bytes — what "pins every expert" predicts — while the arena requirement grew only ×1.66,
and most of *that* is the larger dense side (48 layers, 151936-token vocab), not the
expert path.

**A hot row costs 1.58–1.98× the bytes it holds.** Re-running the whole ladder at
`hot_rows=512` puts the marginal cost at **4.19–5.24 MB per row** against a 2.654 MB
on-disk row. At `hot_rows=128` the expert path is therefore only ~0.5–0.7 GB of the
~3.9 GB requirement; the rest is fixed base. The cause of the per-slot overhead is not
isolated and no mechanism is offered for it.

**`hot_rows` does not travel between models.** `hot_rows=64` — correct for OLMoE — refuses
on Qwen3 with `request of 97 unique rows exceeds hot_rows=64`. That is the documented
behaviour working: the docstring already specifies a floor of `min(T*k, num_experts)`,
which is 128 here. OLMoE has exactly 64 experts per layer, so its value was silently at
the floor already and looked portable. Size from the formula, not from a previous run.

The pre-registration was committed before the checkpoint was downloaded and **all three of
its point predictions missed** — host 18–21 GB (actual 24.70–25.77), arena 2.2–3.0 GB
(actual 3.89–4.03), ratio 7–9× (actual 6.40×). Direction right, magnitudes wrong, because
both arms were sized from OLMoE's much smaller non-expert baseline. Its stop rule fired at
4.08 GB and the ratio is reported only after the investigation it demanded.

No code changed; the wheel is byte-identical to 0.17.3 apart from the version.

## 0.17.3 — 2026-08-13

**Docs-only. The case this feature exists for is now demonstrated instead of asserted.**

Every release through 0.17.2 described `enable_nvme_train_residency` as the answer to
"the experts do not fit host RAM", and every release through 0.17.2 admitted it had
never shown a model that could not be trained without it. It can now be shown, with a
number on both sides. Full receipt, raw ledger and drivers:
[`bench/host-ram-ceiling/`](bench/host-ram-ceiling/RESULTS-host-ram-ceiling.md).

**At a 5 GiB host-RAM ceiling — same model, same seed, same four steps, same box — the
host-RAM path is OOM-killed and the arena path trains to completion.** Descending the cap
until each arm stops completing brackets both requirements:

| | host-RAM offload | NVMe arena, `hot_rows=64` |
|---|---|---|
| **minimum host RAM to train** | **5.91–6.17 GB** | **2.28–2.42 GB** |
| frozen experts | all 1024 pinned (3.83 GB of homes, per 0.17.0) | ~0.2 GB pinned, 64 hot rows |
| steady RSS at `trained` | 5.88 GB | 2.34 GB |

**The saving is ~2.5×, not 2.5–3.5×.** 0.17.2's upper bound came from pairing the lowest
host sample against the lowest arena sample across *different* runs. Measured as one
quantity — the smallest ceiling in which four steps complete — it is **2.56×**, bracketed
2.44×–2.71× by the rungs either side, and steady RSS agrees independently at 2.51×. The
0.17.2 entry is left as published; this supersedes it.

**Peak RSS overstates the host arm by 2.7× — now shown causally.** That arm peaks at
16.63 GB uncapped (15.86 GB of it file-backed) and trains fine under a 6.17 GB cap, with
nothing tuned between the two runs: the kernel reclaims the mmap'd bf16 checkpoint when
RAM is scarce. The arena arm, having almost no page cache to drop, has a peak RSS that
*does* predict its threshold. Hence peak-RSS ratio **7.10×** against requirement ratio
**2.56×** — and hence the earlier 8× figure, which was this artifact.

Why it took a home box rather than a rented one: the cap has to include **swap**. A rented
container's cgroup is read-only and the kernels seen there had no `memsw` accounting, so
an over-limit process pages out and survives — a different outcome from fitting, reported
as success. `docker --memory=N --memory-swap=N` sets both limits, verified by reading them
from inside. The cap was positive-controlled in both directions before any arm ran (900 MB
under 512m → killed; the same allocation under 2g → completes).

**0.17.2 did not actually do what it says it did, and this fixes that too.** It was
titled "put the host-RAM number on the page that serves it" and put the number in
`CHANGELOG.md` — but the PyPI long description is built from **`README.md` alone**, so
no changelog text has ever appeared on the package page. The 0.17.2 page contains no
`3.83`, no `ru_maxrss`, no `steady RSS`. The release was verified by confirming it
published, not by reading the page it was for. The measured summary now sits in the
README, where the long description will carry it.

No code changed; the wheel is byte-identical to 0.17.2 apart from the version.

## 0.17.2 — 2026-08-13

**Docs-only. The published page still lacked the one number the feature is for.**

0.17.0/0.17.1 describe `enable_nvme_train_residency` as lifting the **host-RAM**
ceiling and never said by how much. The 0.17.0 entry below now carries it —
**~2.5–3.5×**, as pinned expert bytes (3.83 GB → ~0.2 GB) and steady RSS after
load (4.94–5.9 GB → 1.42–2.37 GB) — together with the reason the obvious
instruments give the wrong answer.

`ru_maxrss` is not usable here: it reports 18.6 GB for the host arm on a roomy box
and 10.8 GB for the *same arm* on a constrained one, because the 13.84 GB bf16
checkpoint is mmap'd and read in full to fuse and quantize, and those pages are
clean, file-backed and reclaimable. Peak *anonymous* is not the fix either —
~1.5 GB for **both** arms, since `smaps_rollup` counts pinned CUDA memory as
file-backed.

Established by five reproductions across two independently built stacks on one
pod whose unpinned install resolved to identical ML package versions, plus an
A/B/A memory-balloon test (18.57 → 10.80 → 18.56, bit-identical losses) that rules
out drift. No code changed; the wheel is byte-identical to 0.17.1 apart from the
version.

## 0.17.1 — 2026-08-12

**Docs-only. The published 0.17.0 page carried a timing number that was never a
measurement.**

0.17.0's release notes reported `s/step` as one measurement per arm, and from
those single samples claimed the fully-pinned arena cost **1.03×** the host-RAM
reference. Re-measured under this repo's paired protocol — 5 scored rounds, every
arm timed once per round in fixed order, warmup dropped, plus a `host_self`
control that times the *same* host-RAM model twice per round — the control came
back at **0.986 with a 0.898–1.080 spread**. That spread is the harness's
resolution limit, and `hot_rows=1024`'s 0.957 sits *inside* it.

So the honest statement is **"indistinguishable from host RAM"**, not 1.03×. The
disk-bound arms are unaffected and remain real: **1.679×** at the `hot_rows`
floor, **1.479×** at 256 — both far outside the noise, with the ladder moving as
the tier's additive law predicts.

The warmup round is the argument for the protocol: `host` and `host_self` read
3.084 vs 1.901 in round 0 — the identical model, 62% apart. No one-shot-per-arm
run can see that.

The full table, the control, and what still is **not** established (a model whose
experts genuinely exceed host RAM) are in the 0.17.0 entry below, now corrected.
No code changed; the wheel is byte-identical to 0.17.0 apart from the version.

## 0.17.0 — 2026-08-12

**Training whose frozen experts live on NVMe — plus the namespace split, and a CI
gap that had been reporting 42 tests as coverage while running none of them.**

- **`enable_nvme_train_residency(model, arena_path, hot_rows=…)`** — QLoRA on a
  MoE whose frozen experts exceed **host RAM**. The arena served inference and
  refused training in as many words (*"Load without `arena=` to train, or drop the
  adapters to serve"*); that refusal was right, since the serving engines replace
  the module's forward and would discard the adapter's delta. This moves the
  **home** instead: `_ArenaExpertOffload` is an offload handle whose homes are
  `meta` — shape and dtype, no storage — and whose rows come off the arena:
  disk row → ColdTier pinned slot → `[E, …]` device stack → kernel.

  `enable_fast_train` is untouched. `nf4_qlora`'s `weights_fn` closure already
  re-reads whatever is staged when backward runs, which is the seam that makes
  this work at all.

  **Gradient checkpointing is required, and enforced.** The evict hook fires when
  a forward returns, so the checkpoint recompute is what re-stages a layer for its
  own backward. Routed staging fills only the routed rows of a full-shaped stack,
  so a recompute that routed differently would read uninitialized memory —
  `assert_rows_staged` runs inside the weights_fn closures (i.e. **at backward**)
  and refuses instead.

  **It does not bound VRAM.** The staged stack keeps its full `[E, …]` shape so
  every consumer still indexes by global expert id; one layer is device-resident,
  as with ordinary offload. This lifts the host-RAM ceiling, not the VRAM one.

  Needs `grouped-nf4-gemm >= 0.9.0` for `nvme_residency.segment_into`.

- **Fixes a latent bug in the shared single-resident-layer policy.** The slot was
  written through `type(self)`, so a subclass bound a *second* slot and left the
  base class pointing at a handle nothing would evict — two layers resident at
  once, the exact bound that policy exists to hold. Unreachable with one handle
  class; reachable the moment there are two.

- **`enable_nvme_residency` validates before allocating now.** It built its
  `ColdTier` first, which made every refusal unreachable on a host without an
  accelerator: the tier pins its landing buffer, so a CPU-only machine raised
  "Cannot access accelerator device" and the caller got an allocator error instead
  of the message naming their mistake. On a host that does pin, a refusal leaked
  the tier. The error path also no longer closes a tier that modules are already
  serving from, and counts what *this call* attached rather than trusting a sticky
  `_e4b_hot_ref` marker that survives an earlier enable.

- **CI actually runs the arena tests now.** `.[test]` did not install
  grouped-nf4-gemm, so on the runner `tests/test_nvme_train_residency.py` went
  from 29 tests to **one skip**, and `tests/test_nvme_residency_equivalence.py`
  did the same — both reporting as coverage while executing nothing. Adding the
  dependency took the runner from **577 to 661 passing**, and immediately
  surfaced the `enable_nvme_residency` bug above, in an engine that shipped
  months earlier.

- **The README link check stopped calling throttling a dead link**, and no longer
  forwards `Authorization` across origins. It opened a fresh TLS connection per
  link and GitHub's edge dropped some of that churn; one run reported 28 of 28
  links dead on a tree where every path existed. Connection reuse fixed it
  (measured: a URL that failed four `urlopen` attempts answered 200 three times
  running under `curl`). 404 and 403 are never retried into a pass.

Also shipping here, from the preceding commits: the `arch/` `formats/` `engines/`
namespace split (old submodule paths still resolve via aliases in `__init__`), the
architecture support matrix, transformers checkpoint-key renamings, and the
gap-heuristic fix.

- **`arena_train=True` on the loader — without it the above could not be reached
  at all.** The arena branch built bare `meta` experts (its serving shape) and
  silently ignored `r`/`alpha`, so `enable_nvme_train_residency` refused every
  module with *"not ExpertsLoRA-wrapped"* and its own documented usage failed.
  29 CPU tests missed it because each constructs `ExpertsLoRA` by hand in the
  fixture — they exercised the mechanism, never the route a caller takes. It is
  gated on an explicit flag rather than on `r`, because `r` is a required
  positional and the *serving* example passes `r=8`; keying off it would have
  fixed training by breaking serving.

**Verified on a GPU** (RTX A5000, sm_86; OLMoE-1B-7B; 12 steps on Alpaca;
identical data and bit-identical starting adapters; every arm through
`enable_fast_train`, so only residency differs):

| arm | s/step | peak GB | final loss | med \|ΔL\| |
|---|---|---|---|---|
| host RAM (reference) | 1.62 | 2.26 | 0.5839 | — |
| arena, `hot_rows=64` | 2.65 | 2.03 | 0.6143 | **0.0059** |
| arena, `hot_rows=256` | 2.41 | 2.04 | 0.6426 | **0.0086** |
| arena, `hot_rows=1024` | 1.67 | 2.04 | 0.5944 | **0.0099** |

All three pass `bench/fused-train-gate`'s registered 0.05 median-|ΔL| band, 5–8×
inside it. A precondition run first established the arena's bytes are **bitwise
identical** to loader-quantized bytes, so the arms differ only in residency.

**Timing re-measured under the paired protocol** (5 scored rounds, every arm timed
once per round in fixed order so drift hits all arms equally; warmup round
dropped; optimizer not stepped so routing is identical every round). The
`s/step` column above was one measurement per arm and is superseded by this:

| arm | s/step (med) | ratio med | ratio min–max |
|---|---|---|---|
| host RAM (reference) | 1.910 | 1.000 | — |
| **`host_self` (control)** | 1.906 | **0.986** | 0.898–1.080 |
| arena, `hot_rows=64` | 3.207 | **1.679** | 1.490–1.724 |
| arena, `hot_rows=256` | 2.919 | **1.479** | 1.351–1.542 |
| arena, `hot_rows=1024` | 1.867 | **0.957** | 0.890–1.051 |

`host_self` is the *same* host-RAM model timed a second time in each round. At
0.986 the instrument is unbiased, and its 0.898–1.080 spread is the **resolution
limit** — which is the number that makes the rest readable:

- The two disk-bound arms sit far outside it, so **1.68× at the `hot_rows` floor
  and 1.48× at 256 are real**, and the ladder moves the way the tier's additive
  law predicts: the cost is disk traffic.
- **`hot_rows=1024` is indistinguishable from host RAM.** Its 0.957 sits inside
  the control's own spread, so the honest claim is *smaller than this harness can
  resolve* — not the "1.03×" a single measurement suggested.

The warmup round is why this matters: `host` and `host_self` read 3.084 vs 1.901
in round 0 — the identical model, 62% apart. A one-shot-per-arm run cannot see
that.

**How much host RAM this actually saves: ~2.5–3.5×.** The point of the feature is
the host-RAM ceiling, so here is the measurement, with the caveat that makes it
readable. Same OLMoE run, RTX A5000:

| | host-RAM offload | arena, `hot_rows=64` |
|---|---|---|
| **pinned expert bytes** | **3.83 GB** (all 1024 experts) | **~0.2 GB** (64 hot rows) |
| steady RSS after load | 4.94–5.9 GB | 1.42–2.37 GB |

**Do not measure this with `ru_maxrss`.** It reports 18.6 GB for the host arm on a
roomy box and 10.8 GB for the same arm on a constrained one — an A/B/A with a
memory balloon moved it 18.57 → 10.80 → 18.56 GB with bit-identical losses and an
unchanged unreclaimable footprint. The checkpoint is 13.84 GB of bf16
safetensors, mmap'd and read in full to fuse and quantize; those pages are clean,
file-backed and reclaimable, so the "peak" is page cache the process happened to
have mapped, not memory it needed. An earlier 8× figure came from comparing that
inflated host peak against an arena arm that never reads the expert bytes at all.

Peak *anonymous* memory is not the fix either — it is ~1.5 GB for **both** arms,
because `smaps_rollup` counts pinned CUDA memory as file-backed. Use steady-state
RSS after load, or peak of anon + `/dev/zero` mappings.

**What this still does NOT establish.** OLMoE's arena is 3.6 GB and fits
everywhere, so this shows the mechanism is correct and how the cost scales with
residency — **not** a model whose experts exceed host RAM, which is the case the
tier exists for. A demonstration of that needs a machine capped near the host
arm's true ~5–6 GB working set; attempts at 11–16 GB could not fail the host arm,
because it never needed 18 GB.

> **Superseded 2026-08-13 in 0.17.3 — demonstrated.** The prediction in the
> paragraph above was published before the measurement and held: the host arm's
> requirement is **5.91–6.17 GB**. At a 5 GiB cap it is OOM-killed while the arena
> arm trains. The ratio here, **~2.5–3.5×**, is refined to **2.56×** by measuring
> one quantity rather than pairing extremes across runs. See
> [`bench/host-ram-ceiling/`](bench/host-ram-ceiling/RESULTS-host-ram-ceiling.md). Rented-instance NVMe varies ~7× between pods, so
these ratios characterise this box and do not travel.

## 0.16.3 — 2026-08-12

**Makes 0.16.2's headline feature actually importable, and turns one opaque load
failure into an accurate refusal.**

- **`capture_decode`, `CapturedDecoder` and `probe_capture` are exported from the
  package root.** 0.16.2 shipped `capture.py` with no top-level export, no in-repo
  caller and no README mention, so the only route to it was knowing the private module
  path — the feature was published unreachable. Documented under Inference with the
  measured numbers (4.4–5.6x on 2-layer fixtures, **1.11x** on OLMoE-1B-7B, **1.04x**
  on Qwen3-30B-A3B) and both costs: `StaticCache` is allocated to `max_length` up
  front, and `step()` is greedy argmax with no logits processors, stopping criteria or
  streamer.

  Deliberately **not** used by the HTTP server: `_generate_once` needs sampling,
  repetition penalty, stop signals and streaming, and reimplementing those on a
  captured step to gain the measured ~4% at 30B is a bad trade against the risk.

- **Identity ("zero-computation") experts are refused with the counts named.**
  longcat_flash previously died with `AttributeError: ExpertsLoRA has no attribute
  '10'`, which names neither the architecture nor the limitation. LongCat-Flash
  allocates `gate_up_proj` over `n_routed_experts + zero_expert_num` (512 + 256 by
  default) but `down_proj` over the routed count only — its forward sends
  `expert_idx >= num_routed_experts` through `nn.Identity` scaled by the router weight
  and never reads those `gate_up` rows, so the surplus experts are ragged on disk by
  construction. The per-expert reader consumed `0..n_routed-1`, orphaned the rest, and
  the generic weight walk then called `get_submodule(".../experts.10")` on a path whose
  leaf was already the fused module.

  Loading only the routed experts is not a fix: the router keeps selecting over the full
  space, so it would address experts that do not exist. An identity slot belongs in the
  expert primitive, not the loader.

## 0.16.2 — 2026-08-12

**CUDA-graph decode capture, and one less allocation per decode step.**

- **`capture_decode()` / `probe_capture()`** (`experts4bit_qlora.capture`). Wraps one decode
  step in a `torch.cuda.CUDAGraph` backed by a `StaticCache`, so every step has identical
  shapes and ONE graph serves the whole generation — a growing KV cache otherwise gives a
  distinct shape, and a distinct graph, per token. The cost is explicit: the cache is
  allocated to `max_length` up front. `torch.compile` cannot be used instead; inductor dies
  on `aot_autograd() does not yet handle input mutations on views with different dtypes`,
  which is exactly the engine's one-uint8-store-viewed-as-int64-and-float32 row block.
  Capture also throws on a host sync inside the region, so a successful capture doubles as
  a check on the zero-sync decode contract.

  Measured, 16 new tokens greedy: **4.4–5.6x** on 2-layer fixtures (qwen2_moe, qwen3_moe,
  granitemoe, hunyuan_v1_moe, glm4_moe, dots1, olmoe), **1.11x** on OLMoE-1B-7B-0924-Instruct
  (3090) and **1.04x** on Qwen3-30B-A3B (A5000). The speedup is inversely proportional to
  real GPU work per step, which is what a fixed per-step launch cost predicts — so this is
  worth having for small models and for the sync contract, not as a throughput claim at
  scale. Both real-weight models replay **bit-identical** to eager decode.

  `probe_capture()` reports support rather than assuming it, and distinguishes a bf16 argmax
  tie from a real defect by measurement: it teacher-forces the same tokens down both paths
  and compares logits. The reference is eager INCREMENTAL decode against a cache — comparing
  against one full-sequence forward charges a few ulp of kernel/reduction-order difference to
  capture. `qwen3_next` is not capturable: `StaticCache` does not cover LinearAttention.

- **Persistent `row_idx` buffer in the pipelined engine** — the per-step device allocation
  and H2D copy are gone. **-16.4%** host time per decode step.

## 0.16.1 — 2026-08-12

**Correctness fix for the segmented cold source, plus per-round overhead removed.**

- **Prime each segment from its own offset.** `seg_addr` pointed every hot lane at the
  resident row START for all four segments instead of `hot_row + off[j]`. `_prime` seeds
  every slot from expert 0 with `have = -1`, so it does NOT take the hot skip — with
  expert 0 hot on the segmented (offload-homes) path it primed the absmax and down
  regions with gate_up bytes. Latent in 0.16.0: `_fetch` forces hot lanes to skip and
  they read the resident row in place, so nothing read those bytes — but that was an
  undocumented invariant holding up a wrong address table, and nothing tested it.
- **Traffic counting is now opt-in** (`E4B_PIPELINED_TRAFFIC=1`, or `count_traffic = True`
  on the engine before the first fetch). The two device reductions cost ~8.9% of the
  decode step on an A5000 to produce numbers nothing reads in production.
  `traffic()` RAISES when counting was off rather than reporting zeros, because
  `hot_d2d_bytes == 0` is a regression witness and several tests use the counters to
  prove the engine ran at all — silent zeros would let those pass while measuring nothing.

## 0.16.0 — 2026-08-12

- **Residency reads the offload homes in place** (#104, closes #86 with #87).
  Under offload the homes already hold every expert in pinned host RAM and the
  pipelined engine baked a SECOND full-size arena from them — 0.316 GiB per layer
  twice over on Qwen3-30B-A3B geometry, ~15 GiB duplicated for the one
  configuration that exists to fit a big model on a small card. The homes cannot
  be freed (prefill, grad and odd-dtype forwards still fall back to the reference
  path and need staging), so the engine stops making the copy instead.

  The homes group by tensor and the row layout groups by expert, so an expert is
  four contiguous runs and the gather issues one launch per segment. The kernel
  gained a destination offset, a length, and a separate IDENTITY vector — four
  launches read four addresses but must skip or copy together as one expert.
  Measured: RSS delta on enable 0.579 → 0.080 GiB per layer, output bit-identical
  to the copied-arena control.

  Guarded rather than assumed: offload packs homes one buffer per DTYPE with the
  offset advancing in ELEMENTS, so an odd-numel predecessor leaves the next tensor
  misaligned — undefined behaviour where the gather casts to `int64*`. Each
  segment is checked for pinned + contiguous + 8-byte-aligned base and
  8-byte-divisible length, falling back to the copied arena otherwise.
  Non-offloaded modules are unaffected.

## 0.15.0 — 2026-08-11

**Five more quantized checkpoint formats, and the DFlash drafter load path.**

- **AWQ** (`awq.py`) — the first ASYMMETRIC format, using autoawq's exact
  `[0,4,1,5,2,6,3,7]` nibble order. Packed along OUT.
- **GPTQ** (`gptq.py`) — packed along IN, sequential order, `+1` zero offset.
  Told apart from AWQ by its `g_idx` sibling; AWQ had been silently claiming all
  18624 GPTQ tensors, which is a wrong-answer bug, not a load failure. `g_idx` is
  now range- and length-validated (a negative index would wrap to the last group).
- **compressed-tensors int4** (`compressed_int.py`) — llm-compressor / vLLM.
  `num_bits`/`group_size` are DERIVED from shapes because the config often omits
  them. Vectorized unpack.
- **NVFP4** (`nvfp4.py`) — E2M1 codebook with two-level scaling; also serves
  **NVIDIA ModelOpt FP4**, verified against modelopt itself.
- **DFlash drafter** (`glimmer_draft.py`, `speculative.py`) — drafter load for both
  released spellings with coverage reconciled, plus a greedy speculative loop that
  is token-identical to plain greedy.

The dispatch matrix is pinned in BOTH directions, so a format is claimed by exactly
one decoder.

## 0.14.0 — 2026-08-10

**MoE breadth: the convention system, and every execution config measured.**

- **12 adjudicated conventions** covering 45 `model_type`s, each checked against
  transformers' own converter table so coverage DRIFT fails a test rather than
  silently going stale. Gate/up are shape-identical, so orientation can never be
  inferred — every entry is adjudicated, not guessed.
- New families: `gpt_oss` (pre-fused MXFP4 through the generic planner),
  `qwen3_vl_moe` (pre-fused + load-time transpose), `dbrx` and `jetmoe` (flat
  native stacks, bit-identical passthrough), `qwen3_5_moe` (native passthrough),
  `nemotron_h` (NON-gated: stack up/down, no gate to fuse), `minimax_m3_vl`
  (VL-prefixed mixtral), `axk1` (hybrid dense/MoE, layer-conditional keymap).
- **block-FP8 routing**, and MTP heads are never dropped silently.
- Tied heads are tied even when the checkpoint also ships the head.
- Rotary dim/theta buffers are materialized for VL vision towers.
- **Every execution config measured**, not just dtype: the decode/prefill ranking
  inverts, gains shrink as experts widen, and `dgrad=True` is the fastest training
  lane.


## 0.13.0 — 2026-08-10

**Muse Glimmer (Meta) and GLM-5 (Zhipu) checkpoint support.**

- **`glimmer.py` + `glimmer_load.py`** — serve Muse-Glimmer-30B from a released
  GGUF text tower. Glimmer is DENSE (Gemma-3 lineage), so it uses the dense lanes,
  not the expert path. Weights are decoded through grouped-nf4-gemm's k-quant lane
  (**needs gnf4 >= 0.8.0**; the import is capability-gated, so older installs get an
  actionable message rather than an AttributeError). The load streams each tensor to
  the target device as it decodes — peak host RAM is one tensor, not the ~60 GB a
  dequantized 30B would need — and ends with a coverage reconciliation: every
  text-tower parameter must be materialized or it raises.
- **`glm5.py`** — GLM-5 (`glm_moe_dsa`) checkpoint keymap and expert fusion.
  DeepSeek-V3 lineage, so it reuses the existing MLA/per-expert machinery; the new
  surface is DSA's lightning indexer.

Every mapping in both was adjudicated against the real released checkpoint AND the
instantiated transformers module tree, then reverse-armed (every model parameter must
be claimed by some checkpoint key — the direction that catches a silently dropped
weight). The traps that arithmetic alone gets wrong, now asserted:

- Glimmer's head is **untied** and must never be aliased to the embedding.
- Its four per-layer norms are centered (`x*(1+w)`) with the `+1` baked into the GGUF
  bytes, so the parameter is `gguf - 1.0` — while the FINAL norm is used as-is.
- Its `attn_q/k_norm` are uniform vectors equal to `config.qk_scale_factor` and 1.0,
  absorbed by a parameter-free norm; dropped only after asserting that identity, so a
  genuinely learned qk-norm fails loudly.
- GLM-5's checkpoint carries **one more layer than the model builds** (an MTP head);
  it is skipped explicitly, and MTP markers on a built layer raise.
- GLM-5's experts are per-expert on disk and fused in the tree, with gate/up
  concatenated as **blocks** — the interleave convention would mis-activate with every
  shape agreeing.
- Rotary `inv_freq` is computed, never shipped; it is rebuilt through the module's own
  rope initializer instead of being left on `meta`.

Validated end-to-end on an A100 80GB: the 30B loaded from Meta's released
`kquant-dynamic` GGUF (19.65 GB) in 205 s — 627 tensors assigned, 104 dropped,
0 unfilled — and generated correct text at 13.8 tok/s, 55.8 GB VRAM.

## 0.12.0 — 2026-08-06

**`serve` grows a residency dial** — the missing piece between 0.10.0's
residency-reachability fix and the deployment that needed it: `python -m
experts4bit_qlora.serve` could only stream every expert, which is why the judge
deployment decodes at 0.38 tok/s with VRAM sitting idle.

`E4B_RESIDENCY=pipelined` + `E4B_HOT_PROFILE=<jsonl>` + `E4B_HOT_PER_LAYER=<K>` attaches
`enable_pipelined_residency` after load (post-`eval()`, pre-warmup — the ordering the
wrapper's delegation preconditions require). Hot sets are **frequency-ranked from a
profile, never by index** — an index-ordered set on a 256-expert top-6 layer serves ~6%
of routed slots, so there is deliberately no by-index fallback; without a profile it
raises. `E4B_K_SLOTS` overrides the routed top-k when the config lacks
`num_experts_per_tok`. `/health` gains a `residency` block (mode, patched-module count,
profile-predicted coverage).

`E4B_EXPERT_PROFILE` now works under serve: the routing profiler was only ever attached
by `train.py`, so profiling a *serving* workload — the input the residency dial consumes —
silently wrote nothing. serve attaches it at load (no-op unless set); the JSONL lands once
at clean shutdown.

Every quiet failure mode refuses or warns instead: unknown mode, missing/most-wrong
profile (a set-count/module-count mismatch would silently shift every hot set one layer),
an engine that patches 0 modules ("residency on" in the logs, streaming in reality), and
— the subtle one — **activating a trained adapter turns residency off silently** (the
wrapper only delegates while the adapter is provably zero), so `_swap_adapter` warns when
that happens. Serving `base` (the judge/eval case) is what the dial is for.

Also: `test_reenable_with_a_different_dgrad_setting_says_so` gains a capability skip —
against a pre-0.7.0 grouped-nf4-gemm the dgrad mismatch it tests cannot be constructed
(the flag is coerced off first, with its own warning), which surfaced as a spurious
failure on any box with an old wheel installed.

## 0.11.1 — 2026-08-06

Docs-and-tests patch. The PyPI page for 0.11.0 froze training-path guidance that
same-day measurement falsified; this ships the corrected long_description plus the
receipts and one new test. No functional code changes.

- **The 24x was a toy-shape artifact, and the guidance is corrected.** 0.11.0's README
  recommended `enable_batched_train` for "VRAM to spare" on an A2000 microbench (hidden
  512) where it measures 24x. At Qwen3-30B-A3B/48-layer width it is **1.05x at the
  highest peak memory of any lane**, while `enable_fast_train(dgrad=True)` is fastest at
  **2.52x**. The README now defaults to the latter and positions `enable_batched_train`
  as the no-extras fallback it actually is. `batched.py`'s docstring — which predicted a
  bigger card "should narrow this" — carries the measured reversal.
- **The dgrad fidelity caveat is retired by measurement** (`bench/dgrad-gate/`, published
  wheels, 16 + 48 layers): dgrad adds nothing to composed gradient error; an fp32-truth
  arm shows every lane — the reference loop included — on the composed bf16 noise floor
  (vs-reference divergence is rounding *similarity*, not accuracy); a 20-step real-data
  trajectory gate passes at a third of its band, with dgrad at **2.87x** the reference's
  real-data step rate; sm_120 (RTX PRO 4500 Blackwell) runs all 95 release-tag tests
  clean with the sm_86-tuned tile default holding.
- **New test:** DeepSeek-V4's clamped SwiGLU pinned through the batched path
  (`test_batched_train.py`) — the one epilogue composition nothing covered.
- **Credit** for the `enable_batched_train` approach (@jiwoon-ahn, #38) now appears in
  the README, not only the docstring/CHANGELOG.

## 0.11.0 — 2026-08-06

**`enable_batched_train` — a kernel-free batched training path.** Training
without `grouped-nf4-gemm` fell back to `ExpertsLoRA.forward`'s per-expert Python
loop: ~10k sync-gated iterations per forward at 256 experts over 40 layers, with
the GPU idle through most of it. That extra has to build and is arch-gated, so it
is not a rare configuration.

Experts are frozen, so the decoded stack is a constant w.r.t. autograd — and it
comes out of ONE `dequantize_4bit` call, because `_quantize_stack` uses
`compress_statistics=False` and the constructor refuses straddling shapes, making
the flattened absmax an exact concatenation. Verified **bit-identical** to the
per-expert loop and pinned as a test, since a future double-quant would break it
silently. Measured 32x against the per-expert decode at E=256.

One training step, E=256, 512 tokens, top_k 8, hidden 512, RTX A2000:

| path | step | vs loop | peak |
|---|---|---|---|
| reference per-expert loop | 601.2 ms | 1.00x | 59 MB |
| `enable_fast_train` | 132.6 ms | 4.53x | 108 MB |
| `enable_batched_train` | 25.0 ms | 24.01x | 417 MB |

Faster than the kernel lane it was written to fall back *from* — and it spends
peak memory to get there, materializing a stack where the kernel lane holds one
expert. At production width that trade is ~1.6 GB per layer against a few MB, so
**the kernel lane stays the answer under offload or VRAM pressure**. The two are
mutually exclusive and each refuses to patch over the other.

The approach is [@jiwoon-ahn](https://github.com/jiwoon-ahn)'s, from #38. Two
differences from the design proposed there: the backward re-decodes rather than
letting autograd save the stack, so gradient checkpointing is an option rather
than a precondition; and the LoRA delta is a padded double-`bmm`, so expert-LoRA
trains and the package default `TRAIN_EXPERTS=1` works.

**`enable_fast_train(..., dgrad=True)`** routes the fused lane's *backward*
through `grouped-nf4-gemm >= 0.7.0`'s single-launch dgrad kernel instead of its
per-expert decode loop, which measured 78-84% of a training step. A second opt-in
rather than part of the first, because it is a second numerics change: the loop is
exact, the kernel lands near 2.9e-3. Requested against an older kernel package it
turns off with a warning rather than raising from inside a forward.

**A parity contract for training paths** (`tests/test_fused_train_parity.py`).
Gradient *values* through the `ExpertsLoRA` composition were unverified —
`enable_fast_train` was covered by a forward comparison plus
`grad is not None`. A backward wrong by a constant factor still trains and still
descends, and nothing raises. Both lanes now satisfy one contract: forward,
`dL/dx`, and `dL/d` every LoRA parameter against the reference. Tolerances are
measured, not fitted, and a control proves the contract rejects a 1% scaling
error that forward parity alone passes.

**Fixed: `fast.py`'s module header described the whole module as inference-only.**
The paragraph predated `enable_fast_train`, which lives in the same file. It was
quoted back at us in #38 as evidence the package had no training accelerator.

**Untied output heads on multimodal checkpoints were silently tied to `embed_tokens`**
(#37). `load_moe_4bit_streaming` builds the text tower by keeping only keys under the
multimodal prefix (`model.language_model.` for Gemma-4, `language_model.model.` for Kimi
K3). `lm_head.weight` sits *outside* that prefix, so the filter dropped it — and the
meta-tie fallback then assigned the embedding matrix as the output head unconditionally.
Correct for a genuinely tied checkpoint (Gemma-4 ships no head on disk); for a
`tie_word_embeddings: false` checkpoint carrying a real head it meant every logit was
computed through the wrong matrix, with nothing raised: plausibly-shaped generations,
initial train loss at `ln(vocab)`, and a LoRA that "converges" by learning to steer hidden
states into `embed_tokens` — then collapses when the adapter is served on a stack that
maps `lm_head` correctly. The symptom is quant-invariant, so it reads as a quantization
fidelity problem, and an A/B against a tied model passes because the fallback is right
there.

The loader now recovers the head from outside the prefix (logging where it found it) and
gates the tie on `tie_word_embeddings`: untied config with no head that reached the model
raises instead of tying. The multimodal test previously covered only the tied path; the
untied load and the refusal are now both tested.

## 0.10.0 — 2026-08-05

**The residency engine was unreachable for every model the streaming loader produces.**
`enable_pipelined_residency` raised `NotImplementedError` the moment every `ExpertsNbit`
under the model was an `ExpertsLoRA.base` — which is every model
`load_moe_4bit_streaming` returns, i.e. the path most callers take. The composition it
needed already existed and went unused: `ExpertsLoRA._delegate_to_base` hands the whole
forward to the base when an engine is attached and the adapter provably contributes
nothing (`B` is zero-initialised, so an untrained adapter is *identically* zero), and it
already checked for this engine's own `_e4b_pipe_ref` marker. Only the patch site was
missing.

`target_modules()` now includes `ExpertsLoRA` bases. Membership means "targetable and
index-bearing", not "reachable by every engine" — the deprecated v0 `enable_hot_residency`
is not delegated to and still skips them, consuming its `hot_sets` entry. `ExpertsLoRA(r=0)`
raises a `ValueError` naming the supported way to get a zero delta, instead of dying on
`alpha / r` with a bare `ZeroDivisionError`.

Two silent-failure modes were fixed alongside it. `enable_mxfp4_nvme_residency` refused
wrapped bases via `isinstance(m, ExpertsLoRA)` over a list that only ever held
`ExpertsNbit`, so the check could never fire. And `enable_pipelined_residency` now WARNS
when a patch installs but cannot run (train mode, or a non-zero adapter) rather than
returning a count that implies work — a residency split that never executes reproduces the
unsplit reference exactly, so a dead patch scores a perfect zero and reads as a pass.

**`dispatched_modules()`** (new, exported) closes the footgun the above created. Hook what
is CALLED, not what is patched: a wrapped base is not called until an engine is attached,
so a `register_forward_pre_hook` on one fires zero times. The usual reason to hook these
modules is to build a routing histogram for an informed hot set — and that calibration pass
runs before the engine exists, by construction. Zero counts make `topk` return `0..K-1`, so
"informed" silently becomes the by-index set it exists to beat. It fails as a plausible
null, not as an error; it did exactly that once before the helper existed.

**Hot experts are now read in place.** The hot stack and the k-slot store were separate
allocations, so a hot hit still paid a device-to-device row copy before the GEMM could read
it. One shared `[n_hot + k, row_bytes]` store lets the GEMM address a resident row directly.
`sizes` stays the host constant `[1]*k`, still one GEMM launch, and the hot/cold decision
stays device-side — the fixed-shape, zero-host-sync decode loop is untouched. OLMoE-1B-7B on
an A2000, 7 interleaved reps x 96 tokens:

| hot set | before | after | delta | p | gather MB/tok |
|---|---:|---:|---:|---:|---|
| none (pure stream) | 11.77 | 11.78 | +0.1% | 0.225 | 418.4 -> 418.4 |
| by index | 13.29 | 13.37 | +0.7% | 0.025 | 418.4 -> 356.9 |
| informed | 16.51 | **16.83** | **+1.9%** | 0.025 | 418.4 -> **263.5** |

**The byte count that motivated that change overstated it, and the correction is the more
useful result.** The re-copy was 48.5% of all gather traffic on granite, which read as a
large lever. It is not: that copy runs at HBM bandwidth (~5 us/expert), while the PCIe cold
reads are what bind — so removing 37% of gather BYTES bought 1.9% of TIME. They were the
cheap bytes. A first version also cost -0.7% (p=0.013) on pure streaming, where an empty hot
set has nothing to gain but the new per-fetch row dispatch ran anyway; guarded on `n_hot`,
after which the change is non-negative on every config measured.

**Informed hot sets are a property of the model, not just the host.**
`docs/RESIDENCY-ENGINES.md` attributed the size of the gain to the host. Holding the host
fixed at one A2000: OLMoE-1B-7B gains **+24.2% (p=0.002)** from an informed hot set over a
by-index one, and granite-3.0-1b-a400m gains **nothing** (+0.7%/+1.4%/-1.0% across three
runs, never significant) — despite coverage working exactly as designed there (49.7% of
routed slots vs 24.5%, a real 2.0x skew, and a 46% cut in cold traffic). Reads have to bind
before coverage converts, and on a 1.3B model at ~4.2 GB/s they do not.
`bench/bench_hotsets_ab.py` measures this per (model, host) instead of assuming it.

**`quantize_layers`** (loader, #63) restricts 4-bit quantization to a subset of MoE layers,
reusing the loop's existing skip semantics so no new code path appears; `None` preserves
current behaviour bit-for-bit. Motivation is measurement rather than serving: the
KL-vs-knowledge work bounded the churn-to-destruction transition to somewhere in
2.2e-02 .. 1.41e-01 KL but could not locate it, because no quantization scheme lands in that
gap.

**KL-from-reference fidelity instrument** (`bench/kl_fidelity.py`, `bench/kl_paths.py`,
`bench/kl_ikp.py`, `bench/kl_sweep.py`) with K0 control receipts gating every measurement,
a committed 200-prompt set, and a path table where every row names its reference. Its
tier-transition row — 544 GB streamed from host DRAM against an all-resident reference,
**KL exactly 0.000 over 6,813 tokens, top-1 1.000000** — is the row this release's residency
fix unblocked. That row now carries a mandatory execution witness: its expectation is 0.000,
and an engine that never ran satisfies it perfectly, so the test side must stream nonzero
cold bytes and the reference side must stream none.

## 0.9.0 — 2026-08-01

**The trainer ran at batch size 1.** Its inner loop put one variable-length row through each
forward. A fused-MoE step's cost is largely *fixed per active expert* — the reference path
dequantizes each routed expert once, the fused path launches one grouped GEMM per expert
group — so a forward carrying 100 tokens paid nearly what one carrying 2000 does. On OLMoE
(16 layers x 64 experts, top-8) a single ~100-token row was dequantizing ~128 experts.

Rows are now packed until a **token budget** is reached. Measured on an RTX A2000
(OLMoE-1B-7B, SEQ=192, alpaca, 15 steps x grad_accum 4):

| `TOKEN_BUDGET` | s/step | tok/s | peak GPU |
|---|---:|---:|---:|
| 0 (one row per forward) | 17.8 | **22** | 5.23 GB |
| 1024 | 22.2 | **144** | 5.88 GB |
| 2048 | 22.8 | **248** | 6.67 GB |

**11.3x throughput for +1.4 GB.** Steps get 28% *slower* — each carries ~15x more data — so
tok/s is the metric this moves and s/step reads like a regression. `TOKEN_BUDGET=0` restores
the one-row path the v0.2.0 convergence receipts were measured on.

**The ceiling is VRAM, and it is not knowable in advance.** 4096 OOMs on a 12 GB card — but
only sometimes, on an unlucky batch of long rows; it sustained 353 tok/s for six steps first.
A static default cannot be right for both a 12 GB card running OLMoE and a 30B model on the
offload path, so an OOM now **halves the budget and retries the step** rather than killing
the run, down to a floor of 256. Verified: a run at 4096 that previously died now backs off
to 2048 at step 7 and finishes.

Padded batching, not sequence packing: pad positions carry label `-100` and attention `0`,
so no row can see another's tokens and no padding contributes loss — correct without
touching the model. Rows are drawn length-sorted within a bucket to bound padding waste, and
the budget counts the *padded* cost (`rows * width`), which is the work the GPU actually does.

Not claimed: better optimization. The batched arms reach a lower eval loss at equal `STEPS`
(-0.351 vs -0.181) purely because they see ~15x more tokens per step. Loss-per-token parity
is unmeasured.

## 0.8.0 — 2026-08-01

**DeepSeek-V4 (Flash / Pro) loads, serves and trains.** Full V4-Flash — 43 layers
x 256 experts, 284B params — loads in ~10 s at **8.74 GiB peak VRAM** and
generates, with 147 GB of experts served from an on-disk arena. The dense side
measured 8.28 GiB against 8.40 predicted from the shard headers alone. See
`docs/DEEPSEEK-V4.md`.

V4 needed three things the package did not have. Its experts are per-expert
MXFP4 with an epilogue that is gpt-oss's *clamps* over SwiGLU's *combination*,
so neither existing class was correct. Its dense half is block-scaled FP8 rather
than bf16 — `fp8_blocks` serves it at ~1 byte/param instead of 2, which is 8.4
GiB resident against ~14, i.e. whether it fits a 12 GB card. And the published
checkpoint ships in DeepSeek's own `inference/` spelling; transformers converts
that via its central `conversion_mapping.py`, but only inside `from_pretrained`,
which the streaming loader never enters.

**Two fixes to existing code that V4 exposed.**

`mxfp4` was *value-casting* scale bytes rather than reinterpreting them. That was
right by accident for gpt-oss, which ships both blocks and scales as `U8`, and
silently catastrophic for any checkpoint labelling scales `F8_E8M0`:
`.to(torch.int32)` yields the value (`2**-5` -> 0), not the exponent byte, so
every block would be scaled by `2**-127`. torch < 2.7 fails loudly at the read;
torch >= 2.7 materializes the dtype and the error goes silent.

`ExpertsLoRA` hardcoded `act_fn(gate) * up`. Since the adapter re-implements the
expert math inline — to inject the delta before the nonlinearity — it also owns
the choice of nonlinearity, so wrapping **any** clamped-expert architecture
trained a function the frozen base does not compute, with the loss still falling.
The base now supplies its epilogue via `_apply_gate`. This is why gpt-oss and V4
were built bare; V4 is now trainable.

**Hot sets are worth choosing properly.** `expert_profile` only probed
`ExpertsLoRA`, so it found *zero* layers on gpt-oss and V4 — exactly the models
worth profiling. It now probes whichever module is dispatched, and
`hot_sets_from_profile` / `coverage_from_profile` turn a routing histogram into a
hot set. Measured on full V4-Flash: frequency-ranked hot sets are **+37.1%** over
index-ordered at identical VRAM, and index-ordered is statistically
indistinguishable from pure streaming — 4.4 GiB spent for nothing.

**Also:** `scripts/energy_probe.py` (CPU RAPL via powercap or MSR, plus GPU) for
honest J/token, which needs bare metal — containers block both interfaces.
`tools/make_v4_fixtures.py` regenerates the real-bytes test fixtures, so those
tests are coverage instead of permanent skips. README consolidated 476 -> 359
lines (30.6 -> 23.9 KB) on top of 0.7.1's rewrite, keeping every measured number and
receipt link but stating each once — the fused-train figures appeared three times. The
"which door" decision procedure is now a ten-row table on the landing page with the
reasoning in `docs/CHOOSING.md`; residency, decode and V4 long-form moved to
`docs/RESIDENCY-ENGINES.md`, `docs/INFERENCE.md` and `docs/DEEPSEEK-V4.md`. All 19 repo
links are absolute and pinned — relative links 404 on PyPI.

## 0.7.1 — 2026-07-30

**0.7.0's headline features were not importable from the top level.**
`enable_fast_train` — the differentiable fused training path that both flagship
matrices measure, twenty cells of evidence, the thing the front page leads with —
was absent from `__init__.py` entirely, as were `enable_dense_offload`,
`DenseDiskSource` / `DiskHome` / `disk_homes_for`, and `enable_nvme_residency`.
The modules shipped; the names did not. `from experts4bit_qlora import
enable_fast_train` raised `ImportError` on 0.7.0. `__all__` goes 33 → 43, and a
check now asserts every symbol the README tells you to call is actually exported.

**The README described a package two releases old.** Its "Which door? (all six,
one line each)" table listed six execution modes when there were ten, and told a
*training* reader to call nothing — which stopped being the whole answer in
0.6.5. It is now a decision procedure keyed on what ran out (VRAM, host RAM, or
disk), with every mode's entry point, when to pick it, and what it requires.

Three factual corrections in the same pass: a cost figure still quoted the *eval*
delta against a band the protocol registers on **train** loss and median
step-wise; the "measured on an RTX A2000" section header sat above numbers from
two different hosts; and a note promising `enable_hot_residency` would be removed
*in* 0.7 was falsified by 0.7.0 shipping with it still exported — withdrawn
explicitly rather than quietly edited, since someone may have planned around it.

Also in this release: the second flagship-matrix model completed all ten
registered cells under a pre-stamped protocol
(`bench/flagship-matrix-model2/`), with C1 hashing 12.85 GB per cell against the
first matrix's withdrawn gate that hashed zero — and a C4 winner that **flips
sign** when the same cell is re-run on a second host, reported beside the
registered verdict rather than in place of it.

No code behaviour changed beyond the exports.

## 0.7.0 — 2026-07-30

**If you train Gemma-4-class models with `offload=True`, 0.6.x could not do it
at all** — the first backward raised `backward re-dequantization read an
offload-evicted expert`. The evict *post*-hook fired during the
gradient-checkpoint recompute and un-staged the layer before its own backward
re-dequantized it. `offload.py` documented the opposite as an invariant
("PyTorch stops that recompute early, so the evict post-hook does **not** fire");
whether the recompute reaches the post-hook depends on where the checkpointed
region's last needed tensor is produced, which is an architecture detail. OLMoE,
Qwen3-MoE and GraniteMoe stop early, which is why the wrong premise survived.
The post-hook is now a no-op inside a backward; residency is unchanged, because
the single-resident-slot policy already evicts there.

**Two new ways to fit a model that does not fit.** `nvme_experts` serves the
cold expert tail from NVMe, and `dense_offload` + `dense_disk` serve the
*dense* side — the 114.4 GB of non-expert weights that pinned host RAM cannot
hold for a K3-class model — straight from the checkpoint's own safetensors,
byte for byte. Nothing is transformed: the alternative way to fit a 114 GB dense
side on a small card is to quantize it, which changes the model.

Also: the flagship matrix's **B1 bit-exactness gate is withdrawn** — it hashed
`getattr(module, "gate_up_proj")`, which under `offload=True` is a 0-element
placeholder, so it compared `sha256(b"")` with itself and could not fail. The
performance numbers are independent measurements and stand; the assurance that
the frozen stack was untouched during those ten cells does not, and is
separately evidenced by the fused-train gate (16.31 GB hashed, byte-flip control
fires). Its B2 table is corrected too: it reported "Δ eval" where the protocol
registers `|Δ final-**train**-loss|` *and* median step-wise `|Δ|`. Both still
pass; the worst cell is 3.4× inside the band, not the 7× the eval column implied.

The README and `METHODOLOGY.md` §11 no longer end on "a memory optimization, not
a speedup" without saying what the fused path measured: **1.75–1.81× faster per
step at 0.754–0.755× peak VRAM and 0.797–0.846× energy**, both arms offloaded.

## 0.6.7 — 2026-07-30

**`e4b serve` could not load Kimi K3 at all.** Four blockers, each hidden behind
the last: no `trust_remote_code` anywhere in the package (so `AutoConfig` raised
before the architecture gate, with an opaque message), `kimi_k3` missing from
`SUPPORTED_ARCHITECTURES`, a multimodal prefix that is per-family rather than
universal, and per-expert MXFP4. `trust_remote_code` is a new argument plus
`E4B_TRUST_REMOTE_CODE=1` and **defaults to OFF** — executing
checkpoint-supplied code is the caller's decision, never a default.

## 0.6.6 — 2026-07-29

**Per-expert MXFP4 layouts load.** `dequantize_mxfp4` ended in
`out.transpose(1, 2)`, hardcoding gpt-oss's rank-4 `[E, rows, G, B]` blocks, so a
single expert projection `[rows, G, B]` raised `IndexError: Dimension out of
range` instead of returning `[K, rows]`. That is the layout every
DeepSeek-V3-lineage checkpoint ships per expert. `transpose(-2, -1)` is
equivalent for the rank-4 case and correct for both.

## 0.6.5 — 2026-07-29

**Training could never reach the fused kernel, and `enable_fast` reported
success anyway.** Two distinct problems, both fixed here:

- `enable_fast()` patched all expert modules and returned a non-zero count while
  the kernel was invoked **zero** times in train mode. `ExpertsLoRA` hands off to
  the patched base only via `_delegate_to_base()`, which requires
  `not self.training` — and the streaming loaders return a model in `nn.Module`'s
  default train mode. Measured on an RTX 4090: 0 kernel calls / 8.34 tok/s
  against 288 calls / 33.6 tok/s. **A patch count is not a call count.**
- `enable_fast_train()` is new, and is the differentiable path: it patches the
  `ExpertsLoRA` **wrapper** — the module the model actually calls — and composes
  the frozen projection with the trainable `B(Ax)` delta at the pre-activation
  point, the only correct place, since `act(Wx + BAx) != act(Wx) + d`. Opt-in on
  purpose: it changes the expert summation order (group-sorted vs ascending
  expert id), which should be a deliberate choice in a training run.

Requires `grouped-nf4-gemm>=0.2.4`. `--help` no longer loads a model.

## 0.6.4 — 2026-07-28

**If you installed `[fast]` on 0.6.3 or earlier, the fused kernel was not
running.** `enable_fast()` patched `ExpertsNbit.forward`, but `ExpertsLoRA`
inlines the expert math and never calls `self.base(...)` — and
`load_moe_4bit_streaming` always wraps in `ExpertsLoRA`. The advertised speedup
was a silent no-op on the loader this package tells you to use. Upgrade to get
it; nothing about your code changes.

**This is a behaviour change, not only a fix.** With delegation live, the fused
path actually executes, and it is a different computation from the reference
loop — priced at **+0.023% perplexity** (see `docs/METHODOLOGY.md`). If you were
unknowingly running the reference path, your numbers will move slightly.

- **`enable_fast()` now reaches the streaming-loader path (PR #36, `c2bf990`).**
  `ExpertsLoRA` previously inlined the expert math and never called
  `self.base(...)`, so the `[fast]` fused kernel was patched onto a method that
  was never invoked — a silent no-op for every model loaded with
  `load_moe_4bit_streaming`. `ExpertsLoRA` now delegates to its base when the
  adapter provably contributes nothing (B is zero-init, so an untrained adapter
  is *identically* zero), guarded so a trained adapter is never silently dropped.
- **Docs corrections (2026-07-28).** The informed-hot-set decode gain is scoped
  to the (bandwidth-limited) hosts it was measured on — it does not replicate on
  a fat-PCIe box. The `memlock` deployment note no longer claims `cudaHostAlloc`
  is gated by `RLIMIT_MEMLOCK`; that cause is false and the observed slowdown is
  now marked unattributed. README states that both residency engines require
  standalone expert modules and refuse/skip `ExpertsLoRA`-wrapped bases.

## 0.6.3 — 2026-07-21
- **Behavior change — serve binds to `127.0.0.1` by default** (was `0.0.0.0`).
  LAN exposure is now opt-in: set `E4B_HOST=0.0.0.0` to restore the old
  default. Migration: one env var. Rationale: a localhost tool for the
  machine's owner should not be reachable from the network unless asked.
- Optional bearer auth: set `E4B_TOKEN` and the generation routes require
  `Authorization: Bearer <token>` (off by default; `/health` stays open).
- README: first-screen "It dials" bullet (informed hot sets +57-120% at
  identical VRAM); engine-tier tags on the v0 offload-path decode figures;
  the serving posture paragraph. Length pass — the storage-modes matrix,
  serving/Docker, benchmarks, and the bitsandbytes essay moved to `docs/`
  with anchor-preserving stubs.
- `[fast]` pins `grouped-nf4-gemm>=0.2.1`.

## 0.6.2 — 2026-07-21
- `enable_hot_residency` deprecated at call (superseded by
  `enable_pipelined_residency` — same capability, K is config; kept through
  0.6 so the stamped v0 receipts stay reproducible; removal in 0.7).
- README: "Which door?" decision tree covering all six execution modes with
  honest status tiers (`enable_cold_engine` labeled performance-experimental —
  the host decode is a correctness path until the AVX2 kernel lands); all
  relative links absolutized (they rendered as pypi.org 404s in the PyPI
  long_description); CPU-only bitsandbytes first-import notice documented.
- `py.typed` marker ships (the public API already carries annotations).
- Permanent built-artifact smoke in CI and the release gate: wheel installed
  into a clean venv, README-surface import battery + deprecation-warning
  check; README link check blocks publish.

## 0.6.1 — 2026-07-20
- Cold engine (`enable_cold_engine`): hot partition GPU-resident, cold tail
  computed on the host from CPU-resident NF4 (activation-sized bus traffic).
  Host decode bit-exact vs bitsandbytes' CPU `dequantize_4bit`;
  `dequant="auto"` gates bnb behind `avx512f` (on AVX2-only hosts bnb falls
  below naive torch — grouped-nf4-gemm `bench/cold-engine/` receipts).
  All-cold + `device="cpu"` is a pure-host MoE (no CUDA, no `[fast]`).
  gpt-oss epilogue supported. 0.6.0 shipped from a pre-merge tree without
  the engine; 0.6.1 is the real release.

## 0.6.0 — 2026-07-20
- Hot-expert residency (`enable_hot_residency`, #26/#27): expert-granular
  partial residency — hot experts VRAM-resident on the fused kernel, cold
  tail streamed from pinned host RAM; gpt-oss (clamped-GLU + per-expert
  biases) supported; requires `[fast]`, fails at enable time with an install
  hint.
- Routing-informed hot sets (#28): calibrate-then-pin reference driver;
  decode gain tracks routing coverage on thin-link hosts (gpt-oss +56/+120%, Gemma-4 +44%,
  OLMoE +19%); multi-socket affinity law documented (pin `taskset` before
  any cold-path number).
- Hybrid-vs-llama same-box A/B receipts + Gemma-4 gated-weights serving gate
  (`bench/RESULTS-gptoss-hybrid-ab.md`, `bench/RESULTS-informed-hotsets.md`).
- README package-family section (the `[fast]` seam with grouped-nf4-gemm).

## 0.5.0 — 2026-07-18
- `[fast]` extra: fused grouped-GEMM inference via grouped-nf4-gemm —
  `enable_fast()` routes frozen-expert inference through the single-launch
  kernel (measured 3.65× at bs=1 decode, OLMoE geometry, A2000; #25).
