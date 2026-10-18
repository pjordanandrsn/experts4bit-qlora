# Gemma-4 served default census — new registration, no run

Status: **registration draft only**. The maintainer assigns the P number at merge. This lane uses `bench/g4census/` and registers one run ID, `g4census-5090-1`. It does not reinterpret or amend P136. No box may be created before this registration merges and the maintainer acknowledges its scope. A census reading alone changes no default, portability cell, quality gate, or speed claim.

## Prior data, disclosed before this rule

P136's `p136-gemma4-5090-4` produced a **VOID** for both W1 and W16 because `NODE_TRACE_INFLATED` blocked its reducer. The public P136 closure is [#1706 comment 6246958085](https://github.com/pjordanandrsn/experts4bit-qlora/pull/1706#issuecomment-6246958085); the independent reviewer has the byte-pinned reduced records through the private artifact bus. These values are **seen**, not new data or a passing reading:

| P136 seen field | W1 | W16 |
|---|---:|---:|
| graph-delimited `P_ms` | 13.075432 | 21.089588 |
| unprofiled reference, ms/step | 12.457 | 20.5331 |
| graph replay span `S`, ms | 12.269846 | 20.235647 |
| node replay span, ms | 12.697652 | 20.814358 |
| node union `U_in`, ms | 12.285017 | 20.168633 |
| node span / graph span | 1.0349 | 1.0286 |
| node union / graph span | 1.0012 | 0.9967 |
| `omega_ms` | 0.427806 | 0.578711 |
| node kernels per graph, modal | 4173 | 4202 |
| node map residual fraction | 0.0024 | 0.0016 |

Node-mode class medians in that VOID, in ms, were `moe_expert` 3.238318/8.358530, `moe_route` 0.834612/1.223407, `attn` 0.651191/1.324462, `dense_gemm` 3.789024/4.740642, `norm_elem` 3.734116/4.496300, `sample` 0.004768/0.008000, `input_prep` 0.007680/0.023262, `memcpy` 0.002400/0.002240, and `residual` 0.029024/0.032544 (W1/W16). The P136 verdict's `why` contains the two blocking `NODE_TRACE_INFLATED` labels and its `rows` is empty. The prior graph/node ratios, omega and class medians set **none** of the new thresholds below. In particular, this registration does not widen P136's `G_NODE = 1.02` gate or retroactively read P136.

## Question, subject, and pins

Where does one steady decode step of the shipped, all-VRAM Gemma-4 default graph server spend its time, and are node-trace **class shares** repeatable when the node instrument stretches the graph? The graph trace supplies the step clock; node traces supply only a distribution over recorded kernel classes. This lane does **not** turn node-mode class duration into an absolute class time or a fraction of graph-delimited `P`.

Reuse P136's Gemma-4 subject and fixed bytes: `google/gemma-4-26B-A4B-it` at revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, 30 MoE layers, P109's fixed 512-token prompts and 32/160-token decode slope, W1 and W16, P39 NF4 bake, grouped-nf4-gemm v0.46.1 commit `70bc03cb7a61b03c4e801e7423c58c103a1b6d7d`, transformers 5.17.0, bitsandbytes 0.50.2, torch 2.8.0+cu128, and Nsight Systems CLI `2025.6.1.190-256136895201v0`. The e4b source is the exact clean merged commit supplied at the later launch; the box verifies the installed commit before model fetch. All serving and kernel knobs are unset except `E4B_PAGED_MAX_SEQS=16`; graph buckets 1 and 16 must exist, placement is all-VRAM, and `int4_store_kinds == []` and `int4_expert_layers == 0`. The frozen P136 `kernel_classes_nf4.json` is the starting class map; it is copied byte-exact into this lane's staged manifest. An incomplete map or broken 30-layer MoE segmentation VOIDs the run.

## Instrument fixed before new data

On one RTX 5090, P136's pre-fetch host, install, pin and 19/19 GPU-premise checks run first. Two **fresh unprofiled** default-server processes, S1a and S1b, supply W1 and W16 decode slopes. Their self-pair must lie in [0.97, 1.03] at each workload. After 32 skip steps, capture exactly 64 full-batch decode-only steps per workload. Capture order is W1 graph, W1 node-A, W1 node-B, W16 graph, W16 node-A, W16 node-B. Each capture uses a fresh process with the same prompts and default environment. Nsight uses `cuda-sw,nvtx`, `--capture-range=cudaProfilerApi`, and the pinned version above. Raw reports and SQLite remain on the box; reduced records, driver windows, logs and hashes are retained.

The **graph** export alone supplies `P` (delimiter-to-delimiter median), graph replay span `S`, and `P-S` as the out-of-graph interval. `S/P` is a graph-span fraction, **not** a GPU-busy fraction: this lane cannot establish in-graph idle when node tracing stretches the replay. The existing SC1b graph-mode completeness and clock/diagnostic checks apply. The graph `P` must be within 5% of the mean unprofiled reference, SC1b's pre-existing `G_INFLATE` gate; this 5% was fixed before P136 and is unchanged. Profiled wall time is never quoted as serving speed.

For each of the two **node** exports, run SC1b's frozen node classifier on all nine classes (`moe_expert`, `moe_route`, `attn`, `dense_gemm`, `norm_elem`, `sample`, `input_prep`, `memcpy`, `residual`). Let `d[c,r]` be that capture's nonnegative per-step median recorded duration for class `c` and `T[r] = sum_c d[c,r]`. Require finite values and `T[r] > 0`; define `q[c,r] = d[c,r]/T[r]`. Require `abs(q[c,A] - q[c,B]) <= 0.05` for **every** class at each workload. Five percentage points is a pre-data reporting-resolution requirement for two independent traces, not a bound inferred from the single P136 node trace. Report both share vectors, their arithmetic midpoint and maximum absolute difference. They are labelled *node-trace duration shares*, not true device-time shares. Never multiply them by graph `P`, infer class milliseconds from them, or fill P136's missing cells.

Both node exports must retain the registered 64-step window, complete nine-class table and kernel count; the existing SC1b `CLASS_MAP_INCOMPLETE` (>2% map residual), `CLASS_MAP_SEGMENT_BROKEN`, `NSYS_DIAGNOSTIC_ERRORS` and `CLOCK_MISMATCH` checks block. `NODE_TRACE_INFLATED` is recorded with each node's span and union ratios but is **informational for this share-only instrument**. No new node inflation band is fitted. Any failed window, missing export, inconsistent source/model/knob, nonfinite value, or unstable share makes the whole W1+W16 run VOID. One W alone cannot READ. All class-share predictions are checked only after that rule; a missed prediction is reported, never used to rescue or void a record.

`g4census_drive.sh` stages this lane's runner, census driver and reducer plus P136's unchanged speed box, route-exact premise and class map. `staged-g4census.sha256` pins every staged source byte by its box filename. The box captures graph/node-A/node-B in that order per workload and gives every fresh capture a distinct ID. `g4census_reduce.py` requires all three identities and all six driver records before READ; it does not accept P136's one-node receipts as substitutes. The controller refuses a dirty source checkout or an overridden source SHA, then returns only reduced artifacts and completion markers. The launcher manifest must name the same merged source commit, one registered run ID and the cost/guard limits below; this registration does not supply a live manifest or authorize a launch.

## Predictions, declared with the seen-data caveat

The following are **replication expectations informed by the VOID above**, not blind predictions or acceptance limits: W16 graph `P` exceeds W1 on this box; W1's two largest node-duration classes are `dense_gemm` and `norm_elem` in either order; W16's largest is `moe_expert`; each workload's residual node-duration share remains below 2%. Report HELD/MISSED for each, alongside the actual W1 and W16 share vectors. A miss does not change the mechanical READ/VOID rule. There is no cross-host speed comparison, default change or served-quality finding.

## Host, cost, stops, and authority

Register **one** `g4census-5090-1` attempt. A refused create still consumes the ID; any second attempt requires a new reviewed amendment. One verified, direct-SSH RTX 5090 (sm_120), >=30000 MiB VRAM, >=60 GiB host RAM, and a 230 GB ordered container disk must still have >=200 GB free immediately before fetch. Rate cap <=$0.85/h including disk, download cap 100 GB at <=$0.011/GB, and a 4 h guard yield worst `4*0.85 + 100*0.011 = $4.50`, below the assigned $15 ceiling. The quoted offer must fit all caps; otherwise stop before create. The 4 h guard includes the extra node traces. If it cannot finish them with the 600 s receipt margin, stop host-limited and amend before a different run; do not shorten the windows or silently drop a trace.

Bad staged hash, source pin, version, GPU, RAM, disk, default knob, premise, or deadline stops before fetch. Failed speed arms prevent captures. Failed capture/reduction produces VOID, never a partial reading. The controller keeps nonce/completion markers, fetches reduced receipts and teardown evidence, and never copies raw reports, checkpoint or arena into the public PR. The run requires a merged registration **and a later explicit maintainer scope ACK**; this draft grants no rental, GPU probe, or model fetch. The independent review seat is tc-campaign; the maintainer merges and controls rollout.
