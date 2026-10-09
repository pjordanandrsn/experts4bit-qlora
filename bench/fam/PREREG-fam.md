# FAM1 + FAM2a — a quality instrument whose neutral floor is drawn on each family; the B=1 fused stack at T == 1 on gpt-oss-20b (one knob per arm), Granite-3.1-3b-a800m and Qwen3.6-35B-A3B. On one RTX 5090 (registered 2026-10-08, before any run)

Issue: experts4bit-qlora#1362 (lane FAM, model families beyond Qwen3). Follows P115 (#1313): Phase C (#1342) and
Amendment 3 (#1354). The inventory that ranked this read first is `bench/fam/INVENTORY-fam.md` (#1368).

## Why this lane

**Phase C's SANE bar was set on Qwen3, and it sits on Qwen3's own floor.** SANE passes a family iff |mean d_ON| ≤ 0.02
nats and argmax agreement with the unfused reference ≥ 0.95. The neutral perturbations P115 drew (`half`, `chunk`; 48
windows a text) agree with R at 0.955–0.964 on Qwen3-30B-A3B and 0.953–0.962 on Granite (`receipts/p115-5090-8`,
`receipts/p115c-5090-1`). A change that is exactly as neutral as re-batching therefore passes 0.95 by a margin of one
to two hundredths, on those two families.

**gpt-oss-20b failed SANE at 0.924, and its own floor was never drawn.** Phase C ran R and ON only. The register's
routing-flip probe puts gpt-oss's arithmetic-order floor at 0.0176 nats against Qwen3's 0.0095 and Granite's 0.0033
(`e4b.parity.moe-routing-flip-floor`, notes). So the run cannot say whether 0.924 is the folds' arithmetic or gpt-oss's
sensitivity to any bf16 reordering.

**0.924 is an SC2g number, not a default-server number.** Phase C served gpt-oss on SC2g's e4b path
(`E4B_SERVE_EXP_INT4=1 E4B_INT4_KEEP_NF4=1 …`: the native MXFP4 store for single rows, the NF4 stacks for batched rows)
at T == 12. The default server, which a family allowlist governs, serves gpt-oss from the NF4 requantised store. **No
default-server arm below is to be compared with 0.924 directly.** The anchor cell (below) places 0.924 against a floor
drawn at its own setting.

**The allowlist now needs a T == 1 read.** P115 Phase D's proof read Granite at T == 1 with SANE bias −0.0139 against
−0.0009 at T == 12 (proof scale; `receipts/p115d-prove-2`). The maintainer then ruled (bus, 2026-10-08T19:13Z) that the
default fusion allowlist takes a family only with a T == 1 read at reading size. Granite and Qwen3.6 were read at
T == 12 only, so their entries wait on this lane.

## The instrument (FAM1)

`bench/fam/fam_box.py`, one process per (model, config). It wraps P115's instrument, `p115_quality.measure_phase` over
`p110_box.paged_pass`, staged at their registered bytes. The default server (`PagedServeConfig.from_env()` +
`build_engine`) is built eager (`E4B_PAGED_GRAPHS=0`) at all-vram with `E4B_PAGED_MAX_SEQS=16` named. Scoring is P115's:
fp32 log-probs, teacher-forced; the true token's NLL, the argmax, KL against R.

**Cells.** A cell is (text, shape, set), each with its own reference R:
- **texts:** wikitext-2-raw test and c4val1 (P115's loaders; window k starts at token k · 4096), 512 prompt tokens and
  128 teacher-forced positions per window;
- **shapes:** `1`, one window per pass: bucket 1, T == 1, the served one-request arithmetic; and `12`, Phase C's twelve
  windows per pass: T == 12, padded to bucket 16;
- **sets:** A = windows 0–11 (Phase C's wikitext windows), B = 12–23, C = 24–35. Three disjoint draws: parity is never
  read from one window set.

**Arms.**

| arm | process | what it is |
|---|---|---|
| R | OFF | the reference: the four knobs `0`; saved per cell |
| rep | OFF | R again on the first group; a floor draw only if it does not repeat R bit for bit |
| chunk | OFF | floor: prompts prefilled in 256-token chunks (prefill summation order) |
| half | OFF | floor, shape 12 only: the group decoded as two halves of 6, each padded to bucket 8 (batch grouping) |
| split1 | OFF | floor: every decode attention forced to one KV split, `n_split=1` (decode summation order) |
| mutant_scale | OFF | P108's mutant, decode softmax scale × 0.5: gross; the gate must fail it |
| mut090 | OFF | the graded mutant, softmax scale × 0.90: the gate's resolution test, on every set |
| mut095, mut098 | OFF | × 0.95 and × 0.98 on set A only: the ladder, reported, never gated |
| ON | ON_* | the config's knobs; scored against the cell's R |

`split1` and the graded mutants wrap `Fp8PagedKV.attention` below P108's counting wrapper, outside every staged file.
Each wrapped call is counted. `split1` changes only the order of the attention kernel's fp32 sums: grouped-nf4-gemm
fixes the reduction order per (config, split count), and the automatic count at these lengths is above one.

**Statistics,** per (arm, cell), over its windows. d(w) is the arm's mean continuation NLL minus R's on window w.
- bias = mean d; spread = mean |d|;
- agree = mean argmax agreement with R; kl = mean KL(R ‖ arm), reported.

**The floor** of a (text, shape) is every floor draw on every set: `chunk`, `half` (shape 12), `split1`, and `rep` if it
did not repeat R. A draw bit-identical to R on every window is not a draw; the reducer lists it.
- B_f = the largest |bias|, S_f = the largest spread, A_f = the smallest agree: the worst draw;
- K_f = the largest kl, reported.

**The gate.** An arm passes a (text, shape) iff on **every** set (its worst of three):
- |bias| ≤ B_f + **0.010** nats (P110's margin);
- spread ≤ **2** × max(S_f, **0.005**) (P110's);
- agree ≥ A_f − **0.005**: half a percent of positions, about one standard error of a 12-window mean on the P115
  floors (window agreement varies by about 0.02 there);
- **backstop:** |bias| ≤ **0.020** nats (SANE's) and agree ≥ **0.90**. A family whose own floor is wider than this
  cannot license through it.

The gated cells are both texts at both shapes on the default server. An arm passes the family iff it passes all four.

**Resolution.** `mut090` is scored as an arm in every gated cell, on the box's own card.
- If it **passes** the gate in any gated cell, the instrument cannot resolve a 10 % softmax-temperature change on that
  family. The verdict is UNRESOLVED, and **no gate is licensed for that family**, whatever ON scored.
- A PASS is therefore a null read of stated size: **no effect on that family as large as a 10 % change of the decode
  softmax temperature.** For scale, Phase C's gpt-oss folds read KL 0.030 and argmax agreement 0.924 against R. The
  seat A2000's × 0.90 on Granite read KL 0.021–0.024 and agreement 0.924–0.939 (below). So an effect the size of the one
  that held gpt-oss is inside what this gate sees.
- `mut095` and `mut098` (set A, both texts and shapes) report finer sizes, never gated.
- `mutant_scale` must fail every cell, or the read is VOID: the gate cannot fail.
- **Why × 0.90 and not × 0.98** (changed in review, before any box): on the A2000 run below, × 0.98 sat inside Granite's
  own floor on all four cells. A gate whose resolution test sits inside the reorder floor can never resolve anything;
  × 0.90 failed all four. The A2000 calibrated the choice; it does not stand in for the box, where × 0.90 must fail
  again on each family.

## The readings (FAM2a)

One box runs three families, in order of checkpoint size. A later family's fetch or bake failure still leaves the
earlier reads.

| family | pin | arena | configs (the four knobs) | census prediction (q/k/v / glue / r2 / router) |
|---|---|---|---|---|
| Granite-3.1-3b-a800m | `ibm-granite/granite-3.1-3b-a800m-instruct` @ `a02780686e08a03fe0d2679a293b5c74a90efa89` | P39's `k8_bake.py` | OFF (all `0`); ON_auto (all `auto`) | ON_auto `0 / 65 / [32, 32] / 32` |
| gpt-oss-20b | `openai/gpt-oss-20b` @ `6cee5e81ee83917806bbde320786a8fb61efebee` | P39's `k8_bake.py` | OFF; ON_glue (`E4B_FUSE_T1_GLUE=auto`, the rest `0`); ON_r2; ON_epi; ON_auto | glue `0 / 49 / [0, 0] / 0`; r2 `0 / 0 / [24, 0] / 0`; epi `0 / 0 / [0, 0] / 24`; auto `0 / 49 / [24, 0] / 24` |
| Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B` @ `995ad96eacd98c81ed38be0c5b274b04031597b0` | P98's `p98_bake.py` | OFF; ON_auto | ON_auto `0 / 0 / [0, 0] / 40` |

OFF's census is zero on every family. Fused q/k/v never engages here: its gate is the `Qwen3MoeAttention` class.

**Glue-kernel calls per decode-shaped forward** (P115's `KernelCounters`), proven on tiny models of each family on CPU
(`tests/test_fam_box.py`):

| family / config | `rmsnorm_rows` | `rmsnorm_resid_rows` | `scaled_resid_add_rows` | `rope_heads` | `router_epilogue` |
|---|---:|---:|---:|---:|---:|
| Granite ON_auto | 33 | 32 | 32 | 64 | 32 |
| gpt-oss ON_glue | 49 | | | | |
| gpt-oss ON_r2 | | 24 | | | |
| gpt-oss ON_epi | | | | | 24 |
| gpt-oss ON_auto | 25 | 24 | | | 24 |
| Qwen3.6 ON_auto | | | | | 40 |

**The anchor cell (gpt-oss only, reported, never gated).** Phase C's exact setting: SC2g's path (`SC2G_ENV` in
`fam_box.py`), wikitext, shape 12, set A. OFF scores R, the floor arms and the mutants there; ON_auto scores ON. The
reducer reports the anchor's own floor and whether Phase C's 0.924 lies within A_f − 0.005 of it. It answers the
question Phase C could not: knob or gate. It licenses nothing.

**Receipts record, for every process:** the expert store (`int4_expert_layers`, `int4_store_kinds`, the store
levers), the census, the fusion modes, grouped-nf4-gemm's NF4 dispatch tally per cell, peak memory, and every pass's
engagement.

## The rule (`bench/fam/fam_reduce.py`, self-tested on 42 cases)

The verdict for each (family, ON config) is the first of these that applies.

1. **VOID.** Any of:
   - a record missing or not ok;
   - another model revision, e4b commit or grouped-nf4-gemm commit across the family's processes;
   - a cell missing, or run at the wrong group size;
   - a prompt, prefill chunk or floor chunk off 512 / 512 / 256: every prefill forward must carry more than the folds'
     64 decode rows, or the per-step counts below do not hold;
   - the configs scored different windows (the windows' digests per cell);
   - an arm short of its windows: 12 per set; `rep` = the shape;
   - a census off the table above;
   - a pass whose decode attention calls ≠ 127 × the family's attention layers (Granite 32, gpt-oss 24, Qwen3.6 10;
     × 2 for `half`);
   - a pass whose decode-shaped forwards ≠ 127 (× 2 for `half`), plus 1 on Qwen3.6: the hybrid's one-token
     linear-state warm-up, once per padded pass, as Phase C's receipts read it;
   - a pass whose glue-kernel calls ≠ the table × its decode-shaped forwards; an OFF pass that called any;
   - a wrapped arm whose wrapper did not touch every decode attention call;
   - the default path carrying an int4 or MXFP4 expert store;
   - no peak memory;
   - a (text, shape) with no floor draw;
   - `mutant_scale` passing the gate in any cell.
2. **UNRESOLVED.** `mut090` passes the gate in any gated cell.
3. **FAIL.** ON fails the gate in any gated cell.
4. **PASS** otherwise.

## Predictions (written before any data)

| # | prediction |
|---|---|
| F1 | census and every engagement count exact on all three families; no VOID |
| F2 | `split1` is not bit-identical to R in any cell on the real kernel |
| F3 | gpt-oss's floor is wider than Qwen3's and Granite's: A_f at (wikitext, 12) in [0.88, 0.94] |
| F4 | the anchor: Phase C's 0.924 lies within gpt-oss's own SC2g floor (≥ A_f − 0.005), about 60 % |
| F5 | `mut090` fails the gate in every gated cell on every family, about 85 % (gpt-oss's wider floor is the risk); `mut098` passes everywhere it runs, about 80 % |
| F6 | Granite ON_auto PASS, about 70 %: Phase C's T == 12 read passed, and the proof's T == 1 bias of −0.0139 was at proof scale |
| F7 | Qwen3.6 ON_auto PASS, about 85 %: only the router epilogue engages, and it was token-identical in Phase C |
| F8 | gpt-oss: ON_epi PASS about 75 %; ON_glue and ON_r2 PASS about 65 % each; ON_auto PASS about 55 % |
| F9 | peak memory ≤ 26 GiB on every process (Phase C's SANE read 25.22 GiB on gpt-oss) |

## Consequence, registered now

- **PASS** on a (family, config) licenses a separate default PR, reviewed by the maintainer, that adds the family to
  that config's knobs in the allowlist. For Granite and Qwen3.6, ON_auto PASS is the T == 1 read the allowlist entry
  needs. For gpt-oss, each knob licenses alone; ON_auto licenses all three together.
- **FAIL:** the knob or knobs stay off on that family by default. The gate statistic that failed, and the cell, are
  named in RESULTS-fam.md.
- **UNRESOLVED:** × 0.90 sat inside that family's own floor, so no gate is licensed for it under this design. Any
  amendment gates on × 0.80 at the same windows. That rung is fixed here, so it is not chosen after the data.
- **VOID:** no consequence; one rerun inside the ceiling, then an amendment.
- **The anchor** changes no default. It is quoted beside the gpt-oss verdicts so 0.924 is read against its own floor.

The register gets one row per family: `e4b.serve.fam.fused-stack-t1.<family>.5090.<date>`, its value the ON_auto bias
at (wikitext, 1). For gpt-oss the per-knob verdicts and the anchor go in the claim text. STATUS's serving entry for the
fused stack is replaced, not appended to.

## The premise and the proving rental

**Premise**, on the card before anything is fetched (rc 25), after the box's, the reducer's and P115's quality box's
self-tests:
- `tests/test_fam_split1_gpu.py`: the paged decode attention at `n_split=1` is deterministic, stays within 2e-2 of
  the automatic split count, and differs from it in some bits, at head dims 64 and 128;
- `tests/test_fusion_modes.py`, which pins the knobs' `auto` semantics.

All must pass, none skipped. `tests/test_fam_box.py` (the CPU end-to-end run) and `tests/test_fam_staged_pin.py` run in
CI: they need the repository's layout.

**Proof** (`fam-prove-<n>`): Granite alone runs every process kind: OFF, ON_auto, ON_epi, and the anchor's two processes
on SC2g's path. It uses 32 teacher-forced positions on every cell. A VOID from the reducer fails the proof (rc 27); its
verdict is not a reading. The reducer's tables are scaled to the proof's positions (31 decode steps).

## Budget and STOP rules

- **Proof:** one RTX 5090, **guard 0.75 h**, `--download-gb 7`; about $0.7 at most at the policy rate ($0.85/h).
- **Reading:** one RTX 5090, **guard 4.5 h**, `--download-gb 92`; about $4.8 at most at the policy rate ($0.85/h plus
  the download), about $3.6 at the expected 3 h.
- **Expected time, about 3 h:**
  - install and premise, 5 min; fetches 15–40 min; bakes 10 min;
  - Granite about 35 min, gpt-oss about 60 min, Qwen3.6 about 55 min.
  - Basis: eager T == 1 steps of about 25 ms (P115 Phase D's proof, Granite) and about 41 ms on Qwen3.6 (P101's eager
    W1).
- **Lane ceiling:** **$10.00** for the proof, the reading and one rerun of each. Every run is inside the owner's $15
  no-ask tier and the $35 per-run cap.
- **STOP-1:** the refusals before any install: CUDA unusable (18), card class (15), disk < 200 GB (13), host RAM
  < 60 GiB (16), premise (25).
- **STOP-2:** every step checks the time left: fetch 1800 s, bake 900 s, each process 3000 s. Each check leaves 600 s
  for the fetch-back and fits inside its guard.
- **STOP-3:** a VOID is not retried inside the same launch.
- **STOP-4:** the driver refuses a dirty tree or a staged file that differs from `bench/fam/staged.sha256`.

## What was seen before this page (stated, not hidden)

- **No FAM reading exists.** Every number above is cited from P115's and P101's receipts and the register, except the
  A2000 correctness run below.
- **Locally (CPU):** `tests/test_fam_box.py` runs one family end to end on a tiny gpt-oss in bf16 (stand-in attention
  and glue kernels, R unpadded): every cell, arm and config; then the reducer's integrity checks on those records with
  the tiny model's constants, with no VOID. It also checks the census and per-step table above on tiny gpt-oss,
  Granite and Qwen3.5-MoE models.
- **On the seat A2000 (sm_86; correctness only, no timing), Granite-3.1-3b at the pin above.** This was the instrument
  end to end on real weights. Granite's NF4 arena was baked on the card at 2026-10-08T20:28:57Z. OFF ran
  20:30:54Z–21:06:51Z and ON_auto 21:08:10Z–21:14:34Z (clock-read from the logs and the record files).
  - **Setup:** grouped-nf4-gemm 0.43.0 at the pin; set A only, both texts and both shapes, 128 positions. R is unpadded
    (`device_grouping` only), because the padded bucket path needs the fused fp8 KV append (sm_89+). At shape 1 that is
    the same arithmetic. Granite's attention computes f32 on the 5090 too (head_dim 64).
  - **Integrity:** the reducer's checks pass on both records, with Granite's registered census
    (`0 / 65 / [32, 32] / 32`) and per-step counts.
  - **Floor draws:** `split1` is a floor draw on the real kernel in every cell; `rep` repeated R bit for bit.
  - **Expert route:** T == 1 ran grouped-nf4-gemm's scalar NF4 GEMV (Granite's shapes are in neither GEMV table).
  - **Peak:** 4.91 GiB.
  - **Floor** (worst draw over `chunk`, `half` and `split1`): agreement 0.956–0.962, |bias| ≤ 0.0053 nats, KL ≤ 0.0089.
    That is P115's 5090 Granite floor (0.953–0.962).
  - **Mutants:**
    - `mutant_scale` fails by 1.80–2.71 nats.
    - **× 0.98 sat inside the floor on all four cells** (agreement 0.956–0.968, KL 0.0075–0.0094).
    - × 0.95 passed wikitext and failed c4val1.
    - × 0.90 failed all four (agreement 0.924–0.939, bias 0.012–0.023, KL 0.021–0.024).
    - This is why the review moved the gating rung from × 0.98 to × 0.90 (Resolution, above).
  - **ON_auto:** inside the gate on all four cells (bias −0.0053 to +0.0041, agreement 0.964–0.971, KL 0.0071–0.0078).
    This is not a reading: A2000, one set, unpadded R. The box's read decides.

## What this lane cannot say

- Nothing about decode speed: no speed arm runs. P115 read the stack's speed on Qwen3-30B-A3B.
- Nothing about gpt-oss on SC2g's path beyond the one anchor cell, about other families, or about the int4 route.
- Nothing about bucket padding at shape 1, where there is none.

## Receipts

Fetched to the run directory's `fam/` and committed to `bench/fam/receipts/<run>/`:
- `fam_<family>_<config>.json`, `fam_gptoss_anchor_{OFF,ON_auto}.json`, `verdict.json`;
- `summary.txt`, `forensics.txt`, `versions.txt`, the bakes, `logs/` (with `git add -f`), the teardown proof and
  `SHA256SUMS`.

The references and the arenas stay on the box.

## Amendment 1 (2026-10-08, after `fam-prove-1`, before any reading)

**What the proof read.** `fam-prove-1` ran on one RTX 5090 on an AMD EPYC 7C13 host: Vast instance 54917126, launched
2026-10-08T21:48:57Z, lane done 22:17:44Z, $0.423. It VOIDed on census, as registered: the router epilogue licensed 27
of Granite's 32 routers (`failed_probe: 5` in the build log's fusion report, ON_epi and ON_auto alike).
- **Everything else held on the real card:** OFF's census zero, no int4 store, decode-attention calls, decode-shaped
  forwards, wrapper coverage, windows, prefill sizes and group sizes. The premise passed (36 tests), and so did all
  three self-tests.
- **The anchor's two processes were skipped** by the time-left rule.
- **Receipt:** the store's `e63b1f10`. The run's records go in their own receipts pull request.

**Two consequences; neither changes the rule.**
1. **The router probe.** The seat A2000 licensed 32 of 32 at the same code and weights. #1385's probe judges near ties
   on the device's own bf16 logits, over four rows. The fix belongs to the B=1 lane, and the maintainer has made it a
   release blocker.
   - The census table stands at 32, and the relaunched proof must read it on the 5090.
   - No FAM box launches before that fix and #1395 are on `main`.
2. **Time.** This is a sizing basis seen on one host, clock-read from the proof's records, not a speed claim.
   - The median eager decode step was 49.0 ms at T == 1 and 53.6 ms at T == 12.
   - Wall time per process at 32 positions: OFF 1024.1 s, ON_epi 171.1 s, ON_auto 167.1 s.
   - At 128 positions, OFF projects to about 70 minutes a family on such a host, and an ON config to about 12. The
     registered single 4.5 h box cannot hold all three families.

**Changes.**
- **One family per box.** `FAM_FAMILY` names it: `granite`, `gptoss` (which also runs the anchor) or `qw36`. Each is
  its own launch under this registration, with the same rule, gates and predictions. The driver forwards
  `FAM_FAMILY` and refuses any other value, and a reading must name one. The box reduces its own family
  (`fam_reduce.py --families`).
- **Guards,** at 1.5 × the projection, with Qwen3.6 taken at 1.5 × Granite's step:

  | run | guard | the box's checks (need / alarm cap) |
  |---|---|---|
  | Granite | 2.5 h | fetch 900 s, bake 600 s, OFF 6300 / 7200 s, each ON 1100 / 1800 s |
  | gpt-oss (+ the anchor) | 3.75 h | fetch 1200 s, bake 900 s, OFF 6300 / 7200 s, each ON 1100 / 1800 s |
  | Qwen3.6 | 4.0 h | fetch 2400 s, bake 900 s, OFF 9500 / 10800 s, each ON 1600 / 2700 s |
  | the proof | 1.25 h | fetch 300 s, bake 300 s, OFF 1800 / 2400 s, each ON 400 / 900 s |

  Each check plus 600 s of fetch-back fits inside its guard (`tests/test_fam_staged_pin.py`). The proof's longer
  guard lets the anchor's SC2g processes run.
- **Budget,** at the policy rate ($0.85/h plus $0.011/GB):
  - each run's ceiling: proof $1.14, Granite $2.2, gpt-oss $3.4, Qwen3.6 $4.2;
  - **lane ceiling $18.00** (was $10.00), which covers one rerun of the largest; $0.423 is spent;
  - every run stays under the owner's $15 no-ask tier.
- **Unchanged:** the instrument, the cells, the arms, the gate and its margins, the mutants and the × 0.90 resolution
  rung, the predictions, the census and per-step tables, and the rule's rungs.

The reducer's self-test now runs 45 cases: one family per box, the anchor only on gpt-oss's box, and an unregistered
family refused.

## Amendment 2 (2026-10-09, after `fam-qw36-1`, before Qwen3.6's rerun)

**What VOIDed.** `fam-qw36-1` ran on one RTX 5090 on an AMD EPYC 7763 host: Vast instance 54930776, launched
2026-10-08T23:33:38Z, torn down 2026-10-09T00:37:40Z, $1.639. Its OFF process scored wikitext at shape 1 (sets A,
B and C), then died building the first shape-12 runner. e4b's `LinearStatePool.ensure_slots` refused: "the
linear-state pool has 17 slots and a captured decode graph holds its tensors' addresses; growing it to 28 ...". No
record was written, so the reducer VOIDed ("record missing or not ok"). Receipt: the store's `3b11414c`.
- **Cause: a harness bug in the box, not in e4b.** A runner binds `kv.B` + scratch slots, so 1 + 16 at shape 1 and
  12 + 16 at shape 12. The padded R's `enable_decode_graphs(capture=False)` freezes the model's one pool, and a frozen
  pool never grows. e4b documents that refusal: size the first runner for the largest batch. Granite and gpt-oss have
  no linear layers, and the CPU tests run R unpadded, so nothing before the card reached it.

**Change.** `run_cells` scores the largest shape first, 12 and then 1, so the first runner sizes the pool. The rule,
cells, arms, windows, gates, mutants, predictions and census tables are unchanged, as is every Granite and gpt-oss
record already read. The box's self-test now runs 28 cases.

**Why the arithmetic is unchanged** (`tests/test_fam_shape_order.py`, on a tiny Qwen3.5-MoE hybrid on CPU):
- the refusal reproduced at the box's own slot counts (17, then 28), and the fixed order binding both;
- a shape-1 R pass bit-identical on a 28-slot and a 17-slot pool;
- a shape-1 R pass after a shape-12 pass on the same pool bit-identical to one on a fresh pool. The reset it relies on
  is `PagedModelRunner.bind` → `LinearStatePool.reset`, the call `paged_pass` makes for every window on the card too;
- the decode-graph bucket selector reading the same rows, bit for bit, from either pool size.

The padded decode itself needs the fused fp8 KV append, which neither CPU nor the seat A2000 (sm_86) has. Shape 1
decodes in bucket 1, so it has no padding rows and reads only its own, freshly reset slot.

**The rerun.** The registration's one rerun after a VOID is `fam-qw36-2`: Qwen3.6 alone, guard 4.0 h, estimate
$3.40, inside Qwen3.6's $4.2 ceiling. The lane has spent $4.111 (proofs $0.637, Granite $0.828, gpt-oss $1.007,
`fam-qw36-1` $1.639). With the rerun it stays under $8.31 plus downloads, inside the $18 lane ceiling. It launches
from this amendment's merge commit, after review.

## Amendment 3 (2026-10-09, after `fam-qw36-2`, before any gate statistic from it)

**What VOIDed.** `fam-qw36-2` (store `555ba11f`, $2.345) ran both Qwen3.6 processes to completion under Amendment 2's
order: OFF 5966 s and ON_auto 922 s, every cell at both shapes, census exact, peak 24.38 GB. The reducer VOIDed on one
engagement count, 613 times over: "decode-shaped forwards 127 != 128" (254 != 255 on `half`), with router-epilogue calls
5080 against 5120 (127 × 40 against 128 × 40).

**The registered count was wrong about e4b, not the data.** The rule expected the hybrid's one-token linear-state
warm-up once per padded pass. At `98d59921`, `PagedModelRunner._warm_linear_state` (`engines/paged_runner.py`, line 466)
returns early once the model's pool is allocated, so the warm-up runs once per process: on its first padded pass. The
records show exactly that. Each process has one +1 pass, its first (OFF: `wikitext|12|A` R, 1 of 460 passes; ON_auto:
`wikitext|12|A` ON, 1 of 78), and every other pass carries exactly the decode steps. Phase C ran one pass per process
(140 forwards = 12 + 127 + 1), so it could not tell the two apart.

**Change** (`fam_reduce.py`; the rule's gates, cells, arms, windows, mutants and predictions are unchanged). On qw36,
each process's first padded pass in visit order carries exactly one extra decode-shaped forward, with its per-step glue
calls. That pass is the first arm (R on OFF, ON otherwise) of `FIRST_CELL` = `wikitext|12|A`. Every other pass carries
none, and anything else VOIDs. Granite and gpt-oss carry none. The reducer's self-test now runs 50 cases. The new
ones put the warm-up on both processes' first passes (PASS), and on every pass, on a later cell's first pass, on no
pass, missing from the ON process, or twice in the OFF process (each VOID).

**These records are re-reduced, with no new rental** (the maintainer's ruling). That is not picking the rule from the
result. The count is an integrity check fixed by e4b's code and read off the engagement records. It was decided before
anyone computed a gate statistic from `fam-qw36-2`, and nobody computes one until this amendment merges. After the
merge, the maintainer re-derives the verdict from the store before it counts. The lane stays at $6.456.

## Amendment 4 (2026-10-09, after the readings, before any speed data): what the router epilogue buys on Qwen3.6

Designed with the maintainer on the bus (2026-10-09T04:13Z). Qwen3.6's T == 1 read passed, and this registration lets
a PASS license an allowlist pull request. On `qwen3_5_moe` the census is `0 / 0 / [0, 0] / 40`, so the flip turns on
only the router epilogue. A default that changes arithmetic should buy something, and no claim times the epilogue on
this family (P115 Phase C read its engagement and quality only). This amendment registers that timing, before the flip.

**The instrument** (`bench/fam/fam_speed.py`; `FAM_SPEED=1 FAM_FAMILY=qw36`). This is P124 Amendment 1's interleaved
method at one row.
- **One model, both settings.** The default `serve_paged` build on Qwen3.6 NF4, eager at all-vram, with
  `E4B_FUSE_ROUTER_EPI=1` and the other three knobs at 0. `EpiRoute` puts each fused router's original `forward` back
  (OFF) or keeps the fused one (ON). Two copies of the model would not fit on 32 GB.
- **Two blocks.** In block `a` OFF goes first in every pair; in block `b` ON does. Each block has one runner per
  setting, both alive, each with its own KV pool and its bucket graphs (`serve_paged.DEFAULT_BUCKETS`) captured under
  its own setting. Both are prefilled with the same 512-token wikitext prompt. They then decode greedily in strict
  alternation, one step each in lockstep: 5 warm, 256 timed and 32 more. A timed step is `run_decode`'s synchronised
  wall. Each step runs under its own setting, and the swap happens before its clock starts.
- **The hybrid's one linear-state pool.** Every runner on the model shares that pool, so two live runners both binding
  slot 0 would overwrite each other's recurrent state on alternate steps. Each runner's KV holds 2 sequences, and OFF
  binds slot 0, ON slot 1.

**The rule** (`bench/fam/fam_speed_reduce.py`). `ratio_x` is median(ON) / median(OFF) of the timed steps within block
x. The verdict is the first that applies:
1. **VOID**, if any of these holds:
   - the record is not ok;
   - another model, revision or e4b commit;
   - other knobs;
   - a census other than `0 / 0 / [0, 0] / 40`, or a setting that touched other than 40 routers;
   - a block missing, out of its order or on other slots;
   - wrong step or token counts;
   - a runner that did not capture every bucket, replayed bucket 1 other than 293 times (69 in the proof), replayed
     another bucket, or stepped eagerly;
   - a setting whose two blocks emit different tokens.
2. **NOISY**, if the two blocks' ratios differ by more than **1.5 %**.
3. **FASTER** if both ratios ≤ **0.98**; **SLOWER** if both ≥ **1.02**; otherwise **NO_GAIN**.

ON's tokens may part from OFF's: the epilogue changes arithmetic, and FAM's quality read licensed that. The reducer
reports ON-against-OFF token agreement and the first step they part, never gated. It also reports the medians, each
block's median per-pair ratio, the GPU clock log's range and peak memory.

**Prediction** (written before any data, as posted on the bus): a small gain, ratio about 0.99. One fused kernel
replaces softmax, top-k and normalisation on 40 routers, against a one-row step of roughly 10 ms. FASTER about 35 %,
NO_GAIN about 55 %, SLOWER or NOISY about 10 %.

**Consequence.**
- **FASTER, with FAM's PASS:** licenses the allowlist pull request that adds `qwen3_5_moe` to
  `FUSION_DEFAULT_FAMILIES`. The maintainer decides it on both reads.
- **NO_GAIN or SLOWER:** the knobs stay off on Qwen3.6, since a default that changes arithmetic and buys nothing does
  not ship.
- **NOISY or VOID:** one rerun inside the ceiling, then an amendment.

**Runs and budget** (the policy rate, $0.85 an hour):

| run | guard | the box's checks (need / alarm cap) |
|---|---|---|
| the reading | 1.5 h | fetch 2400 s, bake 900 s, speed 1200 / 1800 s |
| the proof | 1.0 h | fetch 2400 s, bake 900 s, speed 1200 / 1800 s |

- **The proof** is `fam-speed-prove-1`, with `FAM_PROVE=1` and 32 timed steps: the whole box end to end on the card.
  It is proved when the speed reducer does not VOID; its verdict is not a reading.
- **The reading** is `fam-speed-1`, on the maintainer's go after the proof.
- **Cost:** at most $0.85 + $1.28. The lane has spent $6.456, so it stays under $8.60 plus downloads, inside $18.

**Code and tests.**
- The speed box's self-test runs 9 cases, and the speed reducer's self-test runs 30 cases.
- `tests/test_fam_speed.py`, on CPU:
  - `EpiRoute(on=False)` reproduces the unfused model's logits bit for bit on a tiny fused Qwen3.5-MoE;
  - two alternating runners on their own slots leave each slot's recurrent state bit-identical to one runner alone,
    and emit its tokens;
  - on one shared slot the state is not identical, which is the hazard the slots avoid.
- What CPU cannot run is the captured graphs, so the proof runs them on the card.
- In the same change, `FUSION_UNLICENSED`'s reasons for `gpt_oss`, `qwen3_5_moe` and `granitemoe` cite FAM's reads.

**Unchanged:** the quality instrument, its rule, the readings and their verdicts.

## Amendment 5 (2026-10-09, before any Mixtral data): Mixtral-8x7B, its own registration

The Mixtral and Gemma-4 readings were requested for this lane; the brief makes each its own registration. The
maintainer approved this design on the bus (2026-10-09T13:43Z), raised the lane ceiling, and asked for the disk sizing
and the fp32 router check below. The instrument, the cells, the arms, the rule, the gates and margins, the mutants and
the verdict ladder are unchanged; this amendment adds Mixtral's pins and tables, a per-family proof, and two checks.

**The family.** `mistralai/Mixtral-8x7B-Instruct-v0.1` @ `eba92302a2861cdc0098cc54bc9f17cb2c47eb61`: 32 layers, 8
experts, top-2, no sliding window (its config.json at that revision). The arena is baked by P39's `k8_bake.py`, as
Granite's and gpt-oss's are. Mixtral has been served all-vram on an RTX 5090 at NF4 before
(`e4b.serve.census.bo7.mixtral.b1.5090.2026-09-05`).

**Configs and tables** (one knob per arm, as gpt-oss; measured on CPU with P115's stand-in kernels on a tiny Mixtral at
the real depth, `tests/test_fam_box.py`):

| config | census (q/k/v / glue / r2 / router) | glue-kernel calls per decode-shaped forward |
|---|---|---|
| OFF | `0 / 0 / [0, 0] / 0` | none |
| ON_glue | `0 / 65 / [0, 0] / 0` | `rmsnorm_rows` 65 |
| ON_r2 | `0 / 0 / [32, 32] / 0` | `rmsnorm_resid_rows` 32, `rope_heads` 64 |
| ON_epi | `0 / 0 / [0, 0] / 32` | `router_epilogue` 32 |
| ON_auto | `0 / 65 / [32, 32] / 32` | `rmsnorm_rows` 33, `rmsnorm_resid_rows` 32, `rope_heads` 64, `router_epilogue` 32 |

Decode attention calls are 127 × 32 per pass (× 2 for `half`), and there is no warm-up forward (no linear layers).

**Two checks the reducer adds.**
- **The router weights stay fp32.** Mixtral's router returns fp32 weights, and the fused epilogue keeps them fp32 since
  the cast guard (#1372). The box now records the build's `fusion_report`. Wherever the epilogue engages (ON_epi,
  ON_auto), its `E4B_FUSE_ROUTER_EPI` entry must read `fp32_upstream` 32, every patched router; anything else VOIDs, so
  ON_epi is read on that path.
- **The disk, refused up front.** Mixtral's checkpoint is 93.4 GB of safetensors (the `consolidated*` files are
  excluded from the fetch). With its NF4 snapshot and arena (about 25 GB each), the environment and the cells'
  references it needs about 157 GB. The box refuses with exit 13 below 220 GB free at the start (`DISK_DEF`, which
  replaces the fixed 200 GB floor for Mixtral) and again below 165 GB free just before the fetch (`FETCH_DISK_GB`),
  after the installs, so a short volume never stops it mid-fetch.

**A proof per family.** Mixtral's own proof (`FAM_PROVE=1 FAM_FAMILY=mixtral`; `fam_reduce.py --proof --families
mixtral`) runs OFF, ON_epi and ON_auto at 32 positions, with no anchor; Granite's proof is unchanged. A reading follows
only after its proof is re-derived.

**Runs and budget** (the policy rate, $0.85 an hour; sized from FAM's measured walls, Qwen3.6's needs for a 47B model):

| run | guard | the box's checks (need / alarm cap) |
|---|---|---|
| Mixtral | 4.0 h | fetch 3000 s, bake 1500 s, OFF 9500 / 10800 s, each ON 1600 / 2700 s |
| Mixtral's proof | 1.5 h | fetch 3000 s, bake 1500 s, OFF 2400 / 3000 s, each ON 600 / 900 s |

- Each check plus 600 s of fetch-back fits inside its guard (`tests/test_fam_staged_pin.py`).
- Run ceilings: the proof $1.28, the reading $3.40. The lane has spent $7.720; with these two it stays under $12.40.
- **lane ceiling $26.00** (was $18.00), which also covers Amendment 6's Gemma-4 proof and reading; every run stays
  under the $15 no-ask tier.

**Predictions** (written before any data):

| # | prediction |
|---|---|
| X1 | census, every engagement count and the fp32 router path exact; no VOID, about 85 % |
| X2 | Mixtral's floor A_f at (wikitext, 12) in [0.93, 0.97] |
| X3 | `mut090` fails every gated cell, about 80 % |
| X4 | ON_epi PASS about 70 %; ON_glue and ON_r2 about 60 % each; ON_auto about 50 % (Granite, with the same fold structure, missed by one entry) |
| X5 | peak memory ≤ 30 GiB on every process |

**Consequence.** As registered, with the maintainer's rule from Amendment 4: a PASS licenses an allowlist pull request
only together with a speed read of what the passing knobs buy on Mixtral (Amendment 4's interleaved method). A FAIL
keeps the knob off; UNRESOLVED and VOID read as registered.

**What this cannot say:** the knobs' speed (no speed arm runs); Mixtral's other checkpoints; the int4 route.

**Code.** The reducer's self-test now runs 56 cases (Mixtral's clean pass and proof; the fp32 path cast off, and the
report missing, each VOID; a proof names one registered family). `tests/test_fam_box.py` pins the table above at the
real depth, with the fp32 router path. The box's self-test is unchanged.

## Amendment 6 (2026-10-09, before any Gemma-4 data): Gemma-4-26B-A4B, its own registration

The second of the two requested readings, designed with the maintainer on the bus (2026-10-09T13:43Z) and
registered separately from Mixtral's. Everything Amendment 5 left unchanged stays unchanged; its per-family proof and
the lane ceiling ($26.00) cover this family too.

**The family.** `google/gemma-4-26B-A4B-it` @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`: 30 layers (five sliding to
one full, 25 and 5), 128 experts, top-8, a 1024-token sliding window, K = V on the full layers (head_dim 512 there,
256 on the sliding ones), a final-logit softcap of 30 (its config.json at that revision). The arena is baked by P39's
`k8_bake.py`. The paged path is at parity on this family (P108, #359).

**Configs and tables** (measured on CPU with P115's stand-in kernels on a tiny Gemma-4 at the real depth and layer
pattern, `tests/test_fam_box.py`). The r2 fold refuses Gemma-4's extra norms (`layer_scalar`, `v_norm`) and engages
nothing, so there is no ON_r2 arm:

| config | census (q/k/v / glue / r2 / router) | glue-kernel calls per decode-shaped forward |
|---|---|---|
| OFF | `0 / 0 / [0, 0] / 0` | none |
| ON_glue | `0 / 271 / [0, 0] / 0` | `rmsnorm_rows` 271 |
| ON_epi | `0 / 0 / [0, 0] / 30` | `router_epilogue` 30 |
| ON_auto | `0 / 271 / [0, 0] / 30` | `rmsnorm_rows` 271, `router_epilogue` 30 |

The glue fold takes nine norms a layer and the final one. Decode attention calls are 127 × 30 per pass (× 2 for
`half`); there is no warm-up forward. The epilogue's `gemma4` kind applies `per_expert_scale` and is not on the
`fp32_upstream` path, so Amendment 5's fp32 check does not apply here.

**Disk.** The checkpoint is 49.9 GB. With its NF4 snapshot and arena (about 13 GB each), the environment and the
references of a 262,144-token vocabulary (about 19 GB) it needs about 105 GB. The 200 GB start floor holds, and the box
refuses below 120 GB free just before the fetch (`FETCH_DISK_GB`).

**Gemma-4's own proof** (`FAM_PROVE=1 FAM_FAMILY=gemma4`): OFF, ON_epi and ON_auto at 32 positions, no anchor. It also
checks on the card what CPU cannot: the load (#344 is open, though unreproduced on the current loader) and `split1`
(`n_split=1`) on the 512-dim full layers.

**Runs and budget:**

| run | guard | the box's checks (need / alarm cap) |
|---|---|---|
| Gemma-4 | 4.0 h | fetch 2400 s, bake 900 s, OFF 9500 / 10800 s, each ON 1600 / 2700 s |
| Gemma-4's proof | 1.25 h | fetch 2400 s, bake 900 s, OFF 2400 / 3000 s, each ON 600 / 900 s |

Run ceilings: the proof $1.07, the reading $3.40. With Amendment 5's two runs the lane stays under $16.90, inside
$26.00.

**Predictions** (written before any data):

| # | prediction |
|---|---|
| G1 | census and every engagement count exact; no VOID, about 80 % (#344 and the 512-dim `split1` are the risks) |
| G2 | Gemma-4's floor A_f at (wikitext, 12) in [0.90, 0.96] |
| G3 | `mut090` fails every gated cell, about 80 % |
| G4 | ON_epi PASS about 65 %; ON_glue about 50 %; ON_auto about 45 % |
| G5 | peak memory ≤ 24 GiB on every process |

**Consequence.** As Amendment 5's: a PASS licenses an allowlist pull request only with a speed read of what the passing
knobs buy on Gemma-4; a FAIL keeps the knob off.

**What this cannot say:** at 512 + 128 tokens Gemma-4's 1024-token sliding window never binds, so the read says nothing
about the folds where it does (P108's parity read, with the window binding, stands on its own); the knobs' speed; the
int4 route.

**Code.** The reducer's self-test now runs 58 cases (Gemma-4's clean pass and its proof). `tests/test_fam_box.py` pins
the table above at the real depth and pattern, and that ON_r2 engages nothing.

## Amendment 7 (2026-10-09, after `fam-mixtral-prove-1`, before any Mixtral reading)

**What happened.** `fam-mixtral-prove-1` (store `bfc1cd3b`, $1.115) fetched Mixtral's 93 GB in 1986 s and baked it in
about 190 s, both OK. Its OFF process then ran out of GPU memory building the server: the server's own KV pool, the
default 16 slots × 4096 tokens (4.12 GiB for K and V), did not fit beside the NF4 weights (about 26.8 GiB of the card's
31.36). No record was written, and the proof was NOT PROVED (rc 27). The instrument never uses that pool: every paged
pass builds its own, at most 12 + 16 scratch slots × 656 tokens (1.15 GiB). Resizing it therefore leaves the read
unchanged.

**Changes** (Mixtral only; the rule, gates, tables, predictions and every other family are unchanged):
- Mixtral's processes build the server at `E4B_PAGED_MAX_TOKENS_PER_SEQ=768` (≥ the 512-token prompt + 128 positions +
  16), still 16 slots: its pool is 0.77 GiB. The budget is then about 26.8 + 0.77 + 1.15 = 28.7 GiB plus the context,
  about 1.7 GiB below the card. X5 (peak ≤ 30 GiB) stands as registered.
- Every arm record carries the server it was built with (`server`: `max_seqs`, `max_tokens_per_seq`), and the reducer
  VOIDs a Mixtral record built at any length but 768 (`SERVER_TOKENS`).
- The proof and the reading stay separate rentals: the second 93 GB fetch (about $0.47) buys the re-derive between
  them.
- Mixtral's proof is guarded 2.0 h, sized from that run: launch to the bake's end took 2336 s, and OFF needs 2400 + 600.
  The reading stays at 4.0 h.

| run | guard | the box's checks (need / alarm cap) |
|---|---|---|
| Mixtral's proof | 2.0 h | fetch 3000 s, bake 1500 s, OFF 2400 / 3000 s, each ON 600 / 900 s |

**Budget.** The lane has spent $8.835. `fam-mixtral-prove-2` ≤ $1.70, the reading ≤ $3.40, and Gemma-4's proof and
reading ≤ $1.07 + $3.40 bring it to at most about $18.42, inside $26.00.

**Code.** `fam_run.sh`'s `build_env_of` sets the length for Mixtral, and the box records it. The reducer's self-test
now runs 59 cases (a Mixtral record built at the default length VOIDs). `tests/test_fam_staged_pin.py` pins the env,
that 768 covers the instrument, and that the proof's guard holds the measured setup.

## Amendment 8 (2026-10-09, after `fam-mixtral-prove-2`, before any Mixtral reading): Mixtral gates on × 0.90, then × 0.80

**The trigger: an instrument property, not an outcome.** `fam-mixtral-prove-2` (store `ff615c5c`, $1.775) PROVED the box on
Mixtral, and its reducer read UNRESOLVED. At proof scale, `mut090` passed the gate on c4val1 at both shapes, and it
failed on wikitext:

| cell | `mut090` agree (A / B / C) | bar (A_f − 0.005) | `mut090` \|bias\| max | bar (B_f + 0.010) |
|---|---|---|---|---|
| c4val1, 1 | 0.956 / 0.964 / 0.971 | 0.948 | 0.0033 | 0.0167 |
| c4val1, 12 | 0.953 / 0.964 / 0.956 | 0.946 | 0.0068 | 0.0207 |

Mixtral barely moves under × 0.90: its agreement with R sits on the reorder floor's. This is a property of the
instrument on this family, the mutant's sensitivity, and not of ON against OFF. The registration fixed the next rung
before any data ("Any amendment gates on × 0.80 at the same windows"). The maintainer ruled on the bus to amend now,
Mixtral only, before the reading.

**Changes** (Mixtral only; the rule, gates, margins, cells, windows and every other family are unchanged):
- `GRADED` gains `mut080` (decode softmax scale × 0.80). Mixtral's OFF process scores `mut090` **and** `mut080` on
  every set of every cell (the box's `--gating mut090,mut080`, from `fam_run.sh`'s `gating_of`). The ladder is
  unchanged.
- **The claimed resolution is the weakest rung that fails the gate in every gated cell:** × 0.90 if the reading
  resolves it, else × 0.80, else UNRESOLVED. A stronger claim is never given away for a weaker rung.
- **A PASS is a null read of that size:** no effect on Mixtral as large as a × 0.80 (or, if × 0.90 resolves, a × 0.90)
  change of the decode softmax scale. RESULTS and the claim state which.
- The reducer reports `resolution` for every family (`mut090` wherever × 0.90 resolves). It VOIDs a Mixtral OFF
  record that did not run both rungs (`GATING`), as it VOIDs any arm short of its windows.

**Predictions** (written before any Mixtral reading):

| # | prediction |
|---|---|
| X3a | `mut090` fails every gated cell at reading scale (resolution × 0.90), about 25 % |
| X3b | `mut080` fails every gated cell (resolution at least × 0.80), about 75 % |

X3 is superseded by these; X1, X2, X4 and X5 stand.

**The reading's guard: 6.0 h, sized from measurements.** `fam-mixtral-prove-2` timed OFF at 1513 s, 245 s of it
`mut090`, and ON_epi and ON_auto at 261 and 234 s. Granite's proof-to-reading ratios (`fam-prove-2` against
`fam-granite-1`) are × 5.33 for OFF, × 5.27 for `mut090` and × 6.7 for an ON process. Scaled by those, Mixtral's
reading needs:
- setup: 2336 s (Amendment 7);
- OFF: 1513 × 5.33 + 245 × 5.27 ≈ 9360 s, the second figure for `mut080`;
- four ON processes: about 1700 s each.

That totals about 18,500 s (5.1 h). Amendment 5's 4.0 h would have skipped the last two ON configs, ON_epi and
ON_auto.

| run | guard | the box's checks (need / alarm cap) |
|---|---|---|
| Mixtral | 6.0 h | fetch 3000 s, bake 1500 s, OFF 10800 / 12600 s, each ON 2000 / 2700 s |

**Budget.** The lane has spent $10.61. The Mixtral reading is at most $6.10 (6.0 h at $0.85, plus about $1.00 for the
93 GB download). Gemma-4's proof and reading add at most about $1.61 and $3.95, downloads included. That brings the
lane to at most about $22.27, inside $26.00.

**Code.** The box's self-test now runs 38 cases: the arms with and without × 0.80, and refusals of a gating list that
does not lead with × 0.90, repeats a rung, names a ladder or unregistered rung, or does not strengthen. The reducer's
self-test now runs 68 cases: resolution at × 0.90 and at × 0.80, UNRESOLVED with both passing in any one cell, the
weakest failing rung claimed whatever × 0.80 does, a FAIL still FAILing at × 0.80, and VOID without × 0.80. The merged
readings re-derive identically apart from the new `resolution` field (`fam-granite-1`, `fam-gptoss-1`, `fam-qw36-2`
against Amendment 3's verdict, and Granite's two proofs).

## Amendment 9 (2026-10-09, after `fam-mixtral-3`, before any Mixtral or Gemma-4 reading): the runs launch past #1482

**What happened.** `fam-mixtral-3` (store `97d853b6`, $0.87) launched from Amendment 8's merge (`4b9954f5`).
- Its self-tests, premise, fetch and bake passed.
- Its OFF process then raised `TypeError: _HybridTier.forward() got an unexpected keyword argument 'residual'` on its
  first forward. Every config VOIDed, and no reading ran.
- The cause was #1477, P127's MoE residual fold, which merged after Amendment 8's review. The patched experts forward
  handed `residual=` to the hybrid tier that the default server installs, and that tier did not take it.
- #1482 (`508cdd03`) fixed it.
- Two earlier tries, `fam-mixtral-1` and `-2`, were refused at the provider before any instance existed ($0): no
  verified RTX 5090 was offered at or under the policy's $0.85/h.

**Changes.** Mixtral's and Gemma-4's remaining runs launch from this amendment's merge, past #1482, instead of from
Amendment 8's. Nothing else changes: not the rule, gates, tables or predictions, and no pinned file (`staged.sha256` is
unchanged).

**The served code under the ON configs since `fam-mixtral-prove-2` (`da17bbc5`).**
- #1472 (P127's kernel options) engages nothing at grouped-nf4-gemm 0.43.0. The pinned package has no `weights_dtype`,
  no `rope_norm_qk`, and no `gather_div` on `gemm_4bit_grouped`.
- #1477 with #1482: wherever the r2 fold folds a layer, P127 moves the MoE residual add into the experts' combine. It
  is licensed per row count, only where it is bitwise the layer's own body. On Mixtral that is ON_r2 and ON_auto; on
  Gemma-4 the r2 fold engages nothing.
- **Measured on the seat's RTX A2000.** The setup:
  - a tiny random 4-layer Mixtral;
  - the served build, eager;
  - P115's kernel counters installed before `build_engine`, as the box installs them;
  - all five configs, at `da17bbc5` and at `508cdd03`.

  The results:
  - The census and the glue-kernel calls per one-row decode forward are identical at the two commits.
  - Both follow Amendment 5's pattern at L = 4. ON_auto reads `rmsnorm_rows` 5, `rmsnorm_resid_rows` 4,
    `rope_heads` 8 and `router_epilogue` 4.
  - At `508cdd03` the residual fold licensed 4 of 4 layers under ON_r2 and ON_auto, and it calls no counted kernel.
  - So Amendment 5's tables stand.

**Before launch.** No FAM run launches until two things are done:
- the maintainer re-derives `p127-prove-2`, the 5090 check of the fixed main's decode graphs;
- this amendment merges.

**Budget.** The lane has spent $11.48. At the policy rate the rest is:
- the Mixtral reading, at most $6.10;
- Gemma-4's proof, at most $1.61;
- Gemma-4's reading, at most $3.95.

That makes at most $23.14 of $26.00.

**No second Mixtral proof** (the maintainer agreed on the bus). It would cost at most $2.70 and bring the lane to $25.84,
and four things already cover what it would check:
- The per-step counts above are identical across `da17bbc5` and `508cdd03`.
- The A2000 smokes on #1482 cover the served eager build, on a real Granite arena and a tiny Mixtral.
- `p127-prove-2` covers the 5090's decode-graph path with the residual fold.
- A harness crash in the reading surfaces at the first forward, for about $0.9, not after 6 h (`fam-mixtral-3`).
