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
| mut098 | OFF | the graded mutant, softmax scale × 0.98: the gate's resolution test |
| mut095, mut090 | OFF | × 0.95 and × 0.90 on set A only: the resolution ladder, reported |
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

**Resolution.** `mut098` is scored as an arm in every gated cell.
- If it **passes** the gate in any gated cell, the instrument lacks resolution at that size on that family. The verdict
  is UNRESOLVED, and **no gate is licensed for that family**, whatever ON scored.
- `mut095` and `mut090` (set A, both texts and shapes) report the size the instrument does resolve, for the amendment an
  UNRESOLVED read would need.
- `mutant_scale` must fail every cell, or the read is VOID: the gate cannot fail.

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

## The rule (`bench/fam/fam_reduce.py`, self-tested on 40 cases)

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
2. **UNRESOLVED.** `mut098` passes the gate in any gated cell.
3. **FAIL.** ON fails the gate in any gated cell.
4. **PASS** otherwise.

## Predictions (written before any data)

| # | prediction |
|---|---|
| F1 | census and every engagement count exact on all three families; no VOID |
| F2 | `split1` is not bit-identical to R in any cell on the real kernel |
| F3 | gpt-oss's floor is wider than Qwen3's and Granite's: A_f at (wikitext, 12) in [0.88, 0.94] |
| F4 | the anchor: Phase C's 0.924 lies within gpt-oss's own SC2g floor (≥ A_f − 0.005), about 60 % |
| F5 | `mut098` fails the gate on every family (the instrument resolves × 0.98), about 65 %; `mut090` fails everywhere, about 95 % |
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
- **UNRESOLVED:** no gate is licensed for that family. Before any default, an amendment re-reads it with more windows,
  and the window count is fixed here, from the ladder:
  - per set: 12 × ⌈(δ / 0.02)²⌉, where δ = 1 − the ladder rung closest to 1 that failed;
  - δ = 0.05 (× 0.95 failed) gives 84 windows; δ = 0.10 (only × 0.90 failed) gives 300;
  - this assumes the mutant's effect grows linearly with its deviation and the standard error falls as one over the
    square root of the windows;
  - if neither rung fails, or the count does not fit the lane ceiling, no gate is licensed for that family under this
    design.
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

- **Proof:** one RTX 5090, **guard 0.75 h**, `--download-gb 7`; about $0.6 at the launcher's policy rate.
- **Reading:** one RTX 5090, **guard 4.5 h**, `--download-gb 92`; about $3.4 at most at $0.75/h.
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

- **No FAM data exists.** Every number above is cited from P115's and P101's receipts and the register.
- **Locally (CPU):** `tests/test_fam_box.py` runs one family end to end on a tiny gpt-oss in bf16 (stand-in attention
  and glue kernels, R unpadded): every cell, arm and config; then the reducer's integrity checks on those records with
  the tiny model's constants, with no VOID. It also checks the census and per-step table above on tiny gpt-oss,
  Granite and Qwen3.5-MoE models.
- **On the seat A2000 (sm_86; correctness only, no timing):** pending. The A2000 run is in progress; this item is
  written from its records before the PR leaves draft.

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
