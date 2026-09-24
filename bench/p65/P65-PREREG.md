# P65 — which per-expert ranking survives a calibration-domain shift, and does activation entropy earn a place in S-C's selector? (registered 2026-09-23, before the run)

Work item: experts4bit-qlora#710. Owner authorization (relayed with the lane assignment, 2026-09-23): the work, and one
later rental within the standing caps. Under the compute rule relayed 2026-09-24, that rental is a proving rental
followed by the reading ("Box and cost"). Nothing here has been rented; the run this registers cites the merge commit.

**Predecessors.**
- **P44-a** (`bench/p44/RESULTS-p44.md:20-24`): the per-expert RTN/GPTQ residual census. Granite-3.1-3B's error is
  heavy-tailed (126 of 1,270 routed experts, 9.92 %, carry 50.3 %: P3 holds by 0.08 points). Mixtral-8x7B's is flat on
  the 16 layers measured (29 of 128, 22.7 %, for 50 %: P3 refuted). Per-expert precision has a premise on one family
  and not the other.
- **Calibration domain has already bitten.** OLMoE's streamed GPTQ experts calibrated on wikitext-train read
  **+0.443** on c4val1, worse than RTN's +0.255 (`RESULTS-p44.md:11-14`). On Gemma-4, GPTQ int4 reads KL 1.1050 /
  1.1564 against NF4 RTN's 1.0772 (`bench/p53/RESULTS-p53.md:9-11`).
- **S-C** (`docs/SPECULATIVE_LANES_PLAN.md:100`, per-expert bit-width from fragility) and **N3**
  (`docs/NEXT_CAMPAIGN_LANES.md:40`, the S-C mixed-precision cell, half-licensed, one CPU join short) are registered
  and unrun. Both documents are OpenTimestamps-anchored (`.ots`); neither is edited here. Anything this lane decides
  for S-C goes into a NEW sibling file (`docs/SPECULATIVE_LANES_ADDENDUM_4.md`), written only after the read.
- **Colla-Q** (Shin & Ryu, arXiv:2609.18131) allocates per-expert bits from an activation-entropy proxy weighting the
  quantisation error, and reports that its metric vectors agree across calibration datasets (C4 / Math / French /
  QA) at cosine 98.2–98.9 on Mixtral-8x7B, where a routing-statistics allocation (PMQ) agrees at 72.0–86.0 (their
  Table 4; 128 calibration sequences per dataset).

Serving mixed formats is out of scope: storage is one scheme per layer (`docs/STORAGE-MODES.md:8-9`). This lane
decides whether that work is worth starting, and with which ranking.

## Question

The census has an error metric per expert (`rel_act`) and a usage statistic (routing frequency), and nothing
label-free about the activations. On the P44-a families:

1. Rank experts **within each layer** by activation entropy, by `rel_act`, and by routing frequency. How much do the
   rankings agree with each other?
2. Which rankings **survive wikitext → c4val1**, measured against how much the SAME text's own resampling already moves
   them?
3. Does entropy earn a place in S-C's selector (the issue's step 3), and if so, as what?

## Instrument (built and rehearsed before registration)

**The entropy.** [`expert_entropy.py`](expert_entropy.py) implements Colla-Q's proxy exactly (their Eq. 5-7), on
each expert's OUTPUT `y = W_dn h`, which is where the paper evaluates it:

    rho = exp(2 H_within) / exp(2 H_total) = sigma2_within / sigma2_total
    sigma2_within = mean over tokens and channels of (y_tc - mu_c)^2      (each channel's own mean removed)
    sigma2_total  = mean over tokens and channels of (y_tc - mu_bar)^2    (one global mean removed)

`rho` lies in [0, 1]. Colla-Q treats a higher `rho` as a weaker expert that needs more bits. `1 - rho` is the share
of the output's variance that is a per-channel constant offset: `sigma2_total - sigma2_within = Var_c(mu_c)`.

- **Why this definition.** The claim under test is Colla-Q's, so the quantity is theirs. The obvious alternative, a
  per-row Shannon entropy of normalised |activation|, cannot be recovered from any accumulated moment: the tap would
  have to keep every row.
- **How it is read.** Every term is linear in the first two moments of `y`, so the census reads `rho` exactly:
  - `E[y_c^2] = (W_dn (H/2) W_dn^T)_cc`, from the down-projection Hessian `H = 2 E[h h^T]` the tap already
    accumulates;
  - `mu_c = (W_dn E[h])_c`, from the first moment the tap now also keeps
    (`calibrate_expert_hessians(..., activation_means=)`; off by default, the Hessians and the return value unchanged,
    `tests/test_p65_entropy.py`).
- **Whose activations.** The rows are the NF4-served model's, the same rows P44-a's Hessians are taken from: `h` comes
  out of NF4 gate/up weights, and `y` applies the checkpoint's bf16 `W_dn`. `rho` is therefore the entropy of the model
  as e4b serves it at NF4, not of the bf16 checkpoint. Colla-Q's text implies the latter.
  - **What that costs.** The NF4 error in `h` is a fixed property of each expert's quantised weights, so it is the same
    in both halves and both texts. Neither the split-half ceiling nor the domain comparison can see it, and it could
    reorder experts relative to a bf16-activation ranking.
  - **Scope.** P65 compares rankings of the served model with each other and across texts. It does not measure how far
    they sit from the checkpoint's. P44-a's `rel_act` has the same property: its Hessians come from the same NF4-served
    rows.
- **Cost.** One extra `[C,K] @ [K,K]` per expert, in fp32 on the device, with fp64 for the rest.
- **Determinism.** Fixed shapes and no sampling. Every census re-computes a row on the box and asserts the bits match.
- **Descriptive neighbours, read by no rule:** the same ratio on the expert's input (`rho_in`), and the normalised
  Shannon entropy of the output's per-channel energy shares (`h_energy`).

**The census.** [`p65_census.py`](p65_census.py) reuses, unchanged:
- P44-a's served model: `serve_stack.build_served_model` (the reused functions are byte-identical to P44-a's pinned
  `serve_stack.py`);
- P44-a's census row: `expert_residuals.census_row`, still P44-a's bytes, `bench/p44/staged-a.sha256`;
- the package's Hessian tap and passes.

It is called with a `min_rows` no expert reaches, so every row is **RTN**: the int4-b32 bytes do not depend on the
text, only `H` does, and no GPTQ solve runs. This is a departure from P44-a's Granite read, which used GPTQ, and it is
deliberate: a ranking difference between texts must not be a pack difference.

Each text is split into two disjoint, interleaved halves (even and odd batches), each censused on its own. A text's
`full` row is the exact row-weighted combination of its halves, since every moment and trace term is a mean over rows.
Every row carries: `rows` (token-slots routed, the routing frequency); `rtn.rel_act`, `rtn.sq_err_act` and `denom_act`
(P44-a's fields); and `entropy`.

**Selfcheck, on the box, before any row is written:**
- on one batch, the moment-derived `rho` of the two most-routed experts of the first layer equals `rho` computed from
  the raw captured rows (relative difference ≤ 1e-4);
- the captured row count equals the tap's;
- a recomputed row is bit-identical.

A family whose selfcheck fails is NOT_READ.

**The signals, per (layer, expert), per census:**
- `entropy`: the dn row's `rho`;
- `rel_act`: `sqrt(rel_gu^2 + rel_dn^2)`, the RTN relative errors of both projections, combined to first order as
  independent relative errors;
- `freq`: token-slots routed. Its top-k sets come from `hot_sets_from_profile`
  (`experts4bit_qlora/engines/expert_profile.py:225`), fed a profile written from the census, so the routing ranking
  IS the residency engine's;
- `rho_x_err`: `rho · rel_act^2`, Colla-Q's objective `L = rho ||e - e~||^2` in relative form, the arm an entropy
  selector would rank by.

An expert enters a comparison only if it saw ≥ 32 rows (the recipe's GPTQ threshold) in every census the comparison
reads.

**The statistics** ([`p65_reduce.py`](p65_reduce.py)) are computed WITHIN each layer, then averaged over layers.
Colla-Q allocates per MoE block, and so does `hot_sets_from_profile`. A pooled ranking would be dominated by layer
identity, which no text changes: on the rehearsal, the layer explains 56 % of `rel_act`'s variance and 38 % of
entropy's.
- `r_split(s)`: Spearman between half 0 and half 1 of the same text, averaged over both texts. This is the resampling
  ceiling.
- `r_cross_half(s)`: Spearman between wikitext half i and c4val1 half j, over the four (i, j). The domain shift at the
  same 16,384-token sample size as `r_split`.
- `penalty(s) = r_split − r_cross_half`: what the domain costs beyond resampling.
- Reported, not decided on:
  - `r_cross_full`;
  - pooled top-10 % Jaccard, with its chance level;
  - `hot_sets_from_profile` top-k Jaccard;
  - Colla-Q's own statistic, the per-layer cosine averaged over layers. The cosine of two positive, near-constant
    vectors is near 1 whatever their order (`test_cosine_cannot_tell_order_from_noise`), and the rehearsal's output
    `rho` has a median of 0.961.
- Bootstrap 95 % CIs resample **layers** (B = 2,000, seed 65).

`tests/test_p65_reduce.py` builds synthetic censuses with a known structure (a stable property, one the text redraws,
one resampling redraws). Every decision branch fires on the rows built to trigger it.

## Families

- **Granite-3.1-3B-A800M-instruct** (`a0278068…`; prefused stacks; 40 experts × 32 layers, top-8). The family where
  per-expert precision has a premise (P44-a P3 holds). The selector is written for it if entropy earns a place. What it
  can show: whether entropy is a stable, non-redundant axis on a heavy-tailed family. What it cannot show: Colla-Q's own
  model.
- **OLMoE-1B-7B-0924-Instruct** (`7f1c97f4…`; per-expert projections; 64 × 16, top-8). Not a P44-a census family. It
  is included because it is where the calibration domain has already cost a measured pack +0.443 on c4val1, so a
  ranking that does not survive has a documented consequence here. Its per-expert premise is P44's P3 rule read on this
  census (RTN, c4val1 full), not assumed.
- **Mixtral-8x7B-Instruct-v0.1** (`eba92302…`; per-expert projections; 8 × 32, top-2). The flat family (P44-a P3
  refuted), and the model Colla-Q tested, so it is the replication anchor for their stability claim. It cannot license
  a selector: no premise. With 8 experts a layer, within-layer ranks are coarse.
  - **Layers:** the first 16 of the plan's own enumeration order, `--first-layers 16`. That is the order P44-a's census
    walked, so these are the layers P44-a measured.
  - **Found while building this lane:** that order is the checkpoint index's key order, sorted as strings (0, 1, 10,
    11, …). It shows in the rehearsal's census order on both layouts. On the LAN store, `OLMoE-1B-7B-0924`'s and
    `Mixtral-8x22B-Instruct-v0.1`'s `model.safetensors.index.json` are key-sorted (8x22B: 0, 1, 10–19, 2, 20, …).
    P44-a's "16 of 32 layers" were therefore most probably {0, 1, 2, 10–22}, not 0–15.
  - **Unverified here:** Mixtral-8x7B's own index was not checked (not on the LAN store; gated upstream), and P44-a's
    receipt is not reachable from this checkout. The read compares P65's `layers_requested` with that receipt's
    `layers_censused`.

## Texts and steps

| text | source | per half | full |
|---|---|---|---|
| `wikitext` | wikitext-2-raw-v1 **train**, `serve_stack.calib_batches(tok, 64, "wikitext")`, the text OLMoE's failing pack calibrated on | 32 × 512 = 16,384 tokens (P44-a's census size) | 32,768 |
| `c4val1` | `allenai/c4` `en/c4-validation.00001-of-00008.json.gz`, first 2,000 documents: the exact text K8 scores as `c4val1` (`step_decomp.py:_k8_window`), windowed by `calib_batches`' own rule (tested equal) | 16,384 | 32,768 |

The windows' token ids are digested per half into every census (`texts.*.halves[].ids_sha256`). Batches are 4 × 512.
Granite and OLMoE run all layers; Mixtral runs 16.

## The rehearsal (NAS RTX A2000 12 GB, sm_86) — NOT a reading

Its output is in [`rehearsal-a2000/`](rehearsal-a2000/). It proves the tap, the entropy, the census and the reducer
end to end on real CUDA. It does **not** read the question:
- the models are different: `granite-3.0-1b-a400m-instruct` (prefused, 32 × 24) and `OLMoE-1B-7B-0924` (the base
  model, 64 × 16);
- the card is not the registered class;
- the tree is this branch's, not a pinned cut.

The image was `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` (torch 2.8.0+cu128, triton 3.4.0), with transformers
5.16.1, bitsandbytes 0.50.1 and gnf4 0.33.0 (the e4b CI pin). The reducer's thresholds were committed (`dd86a36`)
before any rehearsal row existed.

**Instrument checks.** Both selfchecks passed:
- Granite: moment `rho` equal to the raw-row `rho` to a relative 4.5e-8; tap rows = captured rows; bitwise repeats.
- OLMoE: agreement to 1.5e-8 (experts 28 and 41 of layer 0; 903 and 767 tap rows = captured rows); bitwise
  repeats. The stopped first OLMoE run printed the same 1.4525590898021376e-08, to every digit.

No channel's variance rounded below zero in either census.

**Granite-3.0-1B** (24 layers, 768 experts; census 264 s, bake ~23 s of work, peak 1.4 GB reserved):

| signal | r_split | r_cross_half [95 % CI] | penalty | Colla-Q cosine | pooled top-10 % Jaccard (chance 0.053) |
|---|---|---|---|---|---|
| entropy | 0.977 | 0.895 [0.872, 0.913] | 0.082 | 0.9998 | 0.711 |
| rel_act | 0.974 | 0.919 [0.898, 0.938] | 0.055 | 0.9999 | 0.730 |
| freq | 0.988 | 0.893 [0.867, 0.917] | 0.095 | 0.9831 | 0.540 |
| rho_x_err | 0.974 | 0.891 [0.857, 0.920] | 0.083 | 0.9994 | 0.621 |

Beside the table:
- entropy is orthogonal to `rel_act` within layers (Spearman −0.04 on both texts) and anti-correlated with frequency
  (−0.32 / −0.35);
- Colla-Q's comparative claim is UNRESOLVED here: entropy − freq is +0.002, CI [−0.033, 0.034];
- the reducer's selector reads TWO_ARMS. The premise it applies is the registered Granite's, not this model's;
- this census's own RTN P3 statistic on c4val1 is **0.1185**, above P3's 0.10 line. P44-a read 0.0992 on
  Granite-3B's GPTQ packs, a different model and method. The registered Granite premise stays P44-a's read, and the
  reducer reports the RTN figure beside it on every family (`premise.rtn_tail_c4val1`).

**OLMoE-1B-7B-0924, the base model** (the first 4 layers of plan order, {0, 1, 10, 11}; 256 experts; census 479 s,
peak 5.0 GB reserved):

| signal | r_split | r_cross_half [95 % CI] | penalty | Colla-Q cosine | pooled top-10 % Jaccard (chance 0.053) |
|---|---|---|---|---|---|
| entropy | 0.970 | 0.783 [0.715, 0.844] | **0.187** | 0.9998 | 0.576 |
| rel_act | 0.988 | 0.938 [0.912, 0.964] | 0.050 | 0.9999 | 0.576 |
| freq | 0.940 | **0.454** [0.359, 0.549] | **0.487** | 0.9141 | 0.156 |
| rho_x_err | 0.986 | 0.938 [0.909, 0.960] | 0.047 | 0.9994 | 0.576 |

Beside the table:
- routing frequency, stable within a text (0.940), is largely redrawn by the domain (0.454);
- entropy moves less (0.783), but its penalty sits past the registered 0.15, so by the rule it does NOT survive here;
- `rel_act` barely moves;
- Colla-Q's comparative claim REPLICATES on this model: entropy − freq is +0.329, CI [0.177, 0.482];
- entropy correlates with `rel_act` at 0.37 / 0.51 (not redundant);
- this census's RTN P3 statistic is 0.242, so OLMoE-base has no per-expert premise: selector NOT_WRITTEN (it would
  read REL_ACT_ONLY, entropy refused);
- the entropy cosine is 0.9998 where its rank correlation is 0.78: Colla-Q's statistic cannot see the reordering;
- four layers make a thin bootstrap, and this is not the Instruct checkpoint the lane registers.

**The OLMoE run was cut to 4 layers on purpose.** The first run, all 16 layers with an 8 GB Hessian budget, passed
its selfcheck. It then took 162 s per half-pass of a 2-layer chunk (64 s Hessians, 98 s rows), about 85 min for all
16 layers on a GPU three other agents share. It was stopped after its first half-pass
(`rehearsal-a2000/census_olmoe_run1_stopped_log.txt`) and re-run on `--first-layers 4`, a 16 GB budget and one chunk.


**What the rehearsal changed, disclosed.**
1. The census walks layers in checkpoint-key order, which led to the Mixtral finding above and to `--first-layers`
   (P65 had been drafted with layers 0–15).
2. The two projections' relative errors are **anti-correlated** within a layer (Spearman −0.52 on Granite-1B). Each
   alone is more stable than the combination (`r_cross_half` 0.967 and 0.954, against 0.919 for their
   root-sum-of-squares). `rel_act` stays the registered combination. The reducer now also reports `rel_gu`, `rel_dn`,
   `rho_in` and `h_energy` stability, **descriptive, read by no rule**, so the combination is visible.
3. On Granite-1B every signal clears the survival floor with room to spare; on OLMoE-base two do not (frequency, and
   entropy just past the penalty line). The thresholds were not moved. The rehearsal informed P1–P4, each saying where.
4. The costs it measured set two host floors that did not exist before: HF egress ≥ 100 MB/s, and a host Hessian
   scale-and-add ≤ 0.15 s at Mixtral's 14336² (0.33 s on the NAS, `accbench_*.json`). Without the second, a slow host
   would spend ~39 min on Mixtral's updates alone.
5. The reducer now enforces P0 itself. It had checked only the selfcheck, so a census the arm alarm cut would have
   been read on the layers it reached. Found while writing P0, not from a rehearsal number; `test_an_incomplete_census_is_not_read`.
6. Two harness bugs of the rehearsal's own shell were fixed. Neither is in the lane: offline `datasets` could not match
   the c4 data-files config to its cache (the box runs online), and an `echo` reported rc 0 for a failed census.

## Predictions (written before the data; informed by the rehearsal where stated)

- **P0 — instrument (gate, per family).** The selfcheck passes and the census is complete: every requested layer, both
  texts, both halves and the full rows. A family that fails is NOT_READ; nothing below is read for it.
- **P1 — `rel_act` survives on every family:** `r_cross_half` ≥ 0.8 and penalty ≤ 0.10 on Granite and OLMoE, ≥ 0.7 on
  Mixtral, where 8-expert ranks are coarse. *Basis:* RTN bytes are text-independent, so only `H` moves; the rehearsal
  read 0.919 / 0.055 (Granite-1B) and 0.938 / 0.050 (OLMoE-base).
- **P2 — entropy survives on Granite:** `r_cross_half` ≥ 0.8 and penalty ≤ 0.15. *Basis:* Granite-1B 0.895 / 0.082.
  - **OLMoE: no side predicted.** The rehearsal's base model on 4 layers reads 0.783 / 0.187, just past the penalty
    line.
  - **Mixtral:** ≥ 0.6. *Basis:* this is Colla-Q's own model and their cross-dataset agreement.
- **P3 — entropy is not `rel_act` again:** |Spearman(entropy, rel_act)| ≤ 0.8 (the redundancy line) within layers on
  both texts, every family: ≤ 0.3 on Granite (rehearsal −0.04) and in [0.3, 0.7] on OLMoE (rehearsal 0.37 / 0.51).
  Refuted means entropy is a relabelling of the error metric the census already has.
- **P4 — routing frequency and Colla-Q's comparative claim.** The claim is the paired
  `r_cross_half(entropy) − r_cross_half(freq)`, with a bootstrap CI over layers.
  - **OLMoE:** frequency does NOT survive (rehearsal 0.454 / 0.487), and the claim REPLICATES (rehearsal +0.329
    [0.177, 0.482]).
  - **Granite:** frequency survives and the claim is UNRESOLVED (rehearsal 0.893 / 0.095; +0.002 [−0.033, 0.034]).
  - **Mixtral:** the claim REPLICATES. *Basis:* the only published evidence on this model, Colla-Q's Table 4.
- **P5 — Colla-Q's cosine reproduces their magnitude and cannot discriminate:** per-layer cosine of the entropy
  vectors ≥ 0.98 on every family (theirs 98.2–98.9; rehearsal 0.9998), while `r_cross_half` sits below it. Recorded as
  the reason the rule reads Spearman.
- **P6 — continuity with P44-a (descriptive, gates nothing):** Mixtral's RTN tail fraction (P44's P3 statistic) on
  c4val1 full, on these 16 layers, is in [0.18, 0.27] (P44-a: 0.2266 on C4 shard 00000). It can differ for reasons of
  text alone.

## Decision rule (the issue's step 3)

**Registered definitions** (in code in `p65_reduce.py`, committed before the rehearsal):
- a signal **survives** iff `r_cross_half ≥ 0.5` and `penalty ≤ 0.15`;
- entropy is **redundant** iff |Spearman(entropy, rel_act)| > 0.8 within layers on either full text.

**Per family, only where the per-expert premise holds** (P44-a's P3 for Granite and Mixtral; this census's RTN P3 for
OLMoE):

| entropy | `rel_act` | S-C's selector (written in `docs/SPECULATIVE_LANES_ADDENDUM_4.md`) |
|---|---|---|
| survives, not redundant, and `rho_x_err` survives | survives | **TWO_ARMS**: rank by `rel_act`, and by `rho_x_err` (Colla-Q's weighting). The addendum compares the two at matched bytes on an outcome (two-text K8, or KL where K8 cannot read), against routing frequency as the baseline arm. |
| survives, not redundant, but `rho_x_err` does NOT survive | survives | **REL_ACT_ONLY (rho_x_err does not survive although both parts do)**: each part transfers but their product does not, so there is no TWO_ARMS selector to write. Recorded as that finding. |
| survives, not redundant | does not survive | **ENTROPY_ONLY**: the only one-text ranking that transfers is label-free. The addendum must show it predicts the serving text's error before any cell. |
| redundant | survives | **REL_ACT_ONLY (entropy redundant)**: entropy adds nothing the error metric lacks. Recorded, and the issue's premise is answered no. |
| does not survive | survives | **REL_ACT_ONLY (entropy refused)**. |
| (either) | does not survive, and entropy does not rescue it | **NO_ONE_TEXT_SELECTOR**: S-C must calibrate on the serving text or a mixture. Recorded as a finding against one-text selectors on that family. |

**Entropy earns a place** only in TWO_ARMS or ENTROPY_ONLY. **It does not** in any REL_ACT_ONLY branch.

Two further consequences:
- Mixtral has no premise, so its selector is NOT_WRITTEN whatever the rankings do. Its read answers P4 and P5.
- P4 is reported as Colla-Q's claim replicated, reversed or unresolved on each family. It does not gate the selector:
  a ranking can be the better selector without beating routing on stability.

N3's missing flip-attribution join is independent of this lane and stays where it is. Nothing here changes a default,
a kernel or a stored format.

## Box and cost

- **Box:** one RTX 5090 (Vast verified/secure).
  - `$/h` ceiling 0.65 (P44-a's);
  - ≥ 64 GB available RAM (refused rc 13 below it): Mixtral's per-layer Hessians are 7.1 GB and the chunk's weights
    5.6 GB, with the Hessian budget scaled to 40 % of RAM (16–64 GB);
  - ≥ 200 GB free disk (rc 13): Mixtral bf16 93 GB + NF4 snapshot + arena;
  - HF egress ≥ 100 MB/s on the pre-flight (rc 14). P44-a's floor was 20; at 20 MB/s Mixtral's fetch alone is 78 min.
  - host scale-and-add of a 14336² fp32 Hessian ≤ 0.15 s (rc 13, `REFUSAL` names it). Every calibration batch lands
    each expert's gram on the host and folds it into the running mean. The registered Mixtral census does ~4,100 such
    updates at 822 MB, and the cost is set by host memory bandwidth, not the GPU: 0.33 s on the NAS Xeon W-1250, 0.027 s
    on an M1 Max (`rehearsal-a2000/accbench_*.json`). The update count is 16 layers × 8 experts × 8 batches × 4 halves
    = 4,096 per projection. At the NAS's speed, with its landing copy, that alone is ~39 min for Mixtral; at the floor,
    about 18 min.
- **A proving rental comes first.** The compute rule in force (relayed 2026-09-24) puts a proving rental (≤ $0.15,
  ≤ 10 min) in front of any rental whose guard exceeds 1 h, and the reading's does. The proof is `p65_drive.sh` with
  `P65_PROVE=1`, on the same class, at the SHAs the reading will install, with a 10 min guard (≤ $0.11 at $0.65/h).
  - **What it runs.** Every refusal above, the install and the tripwire. Then Granite-3.1-3B end to end on a cut-down
    census: its first plan-order layer at `nseq 8`, covering fetch at the pin, bake, build, the on-box selfcheck, both
    texts, both halves and the full rows.
  - **What it writes.** `prove_census_granite.json` and `MODE` = `prove`. The reducer reads only `census_<family>.json`,
    so a proof can never be read as the lane's result.
  - **Rehearsed.** The same cut-down census ran on the A2000 (Granite-1B): rc 0, selfcheck 6.0e-8, 384 rows over 1 layer, census 38.5 s, bake 74 s wall. That is ~2 min of the proof's
    10, beside ~5 min of install and ~1 min of Granite-3B fetch at the egress floor. It is recorded in
    `rehearsal-a2000/prove_config_*`.
  - **Pass.** The proof passes iff rc 0. Its receipts also carry the egress, the host scale-and-add and the wall times
    of install, fetch, bake and census, which the reading's estimate below is checked against before that box is
    started.
  - **Fail.** A harness failure (rc 3, 9, 11, 12, 41) means the reading is not rented: the fault is fixed and proved
    again. A host refusal (rc 10, 13, 14, 15) is a refused box, not a failed proof, and another proving box is drawn.
    All proving attempts together stay ≤ $0.15.
- **The reading. Guard: 2 h**, started only after a passing proof at the same `E4B_SHA` / `GNF4_SHA`. **Estimate:** ~75 min typical (≈ $0.80 at $0.65/h); ≤ $1.30 at the guard. The bases are the rehearsal
  (a contended A2000 on a busy NAS, so an upper bound per call), P44-a's Mixtral run, and the host benchmark.

| step | basis | estimate on a box at the floors |
|---|---|---|
| install | P44-a's same install | ~5 min |
| Granite-3B: fetch 6.6 GB, bake, census | ≥ 100 MB/s; Granite-1B census 264 s and bake ~23 s on the A2000; the 3B has ~3.3× the Hessian elements and ~2× the forward | 10–18 min |
| OLMoE-Instruct: fetch 13.8 GB, bake, census | rehearsal bake ~5.7 min of work, reading from the NAS disk; census from the 4-layer rehearsal: per layer and half, 16 s of Hessian pass and 10 s of rows on the A2000, which is ~30 min for 16 layers there (an upper bound) | 12–39 min |
| Mixtral: fetch 93 GB | ≥ 100 MB/s (typically 200+ with the token) | 7–16 min |
| Mixtral: bake (NF4 quantise, snapshot, arena) | not rehearsed at this size; P44-a baked the same checkpoint on the same class | 10–20 min |
| Mixtral: census, 16 layers | host update ≤ ~18 min at the floor; 1,024 rows × ~0.5 s ≈ 9 min; 16–64 half-pass forwards ≈ 2–5 min; build, selfcheck, reads ≈ 5 min | 25–37 min |
| **total** | | **~70–135 min** |

- **If the guard binds.** The upper end exceeds the guard only if every host floor binds at once. Mixtral is then the
  family the runner skips (rc 40), since it starts only with its whole budget (`P65_NEED_MIXTRAL_S` = 75 min) left
  before the deadline. The single follow-up this registration allows is one Mixtral-only box: `P65_FAMILIES=mixtral`,
  1.5 h guard, ≤ $0.98, needing the owner's rental approval like any box. Its guard also exceeds 1 h, so it too is
  preceded by its own passing proof (≤ $0.15). Nothing else re-runs.
- **Ceilings:** lane $1.75 for the proof, the reading and any refusals; $2.90 with the Mixtral-only follow-up and its
  proof; hard stop $3.00. No second box on a disappointing result.
- **Downloads** (pinned revisions, authenticated with the staged token, never on a command line):
  - Granite-3.1-3B-A800M-instruct, ~6.6 GB;
  - OLMoE-1B-7B-0924-Instruct, ~13.8 GB;
  - Mixtral-8x7B-Instruct-v0.1 safetensors, ~93.4 GB;
  - wikitext-2-raw-v1, ~13 MB, and one C4 validation shard, ~40 MB;
  - e4b at the merge commit and gnf4 at the CI pin, with transformers 5.16.1, bitsandbytes 0.50.1 and datasets.
- **Order:** Granite, then OLMoE, then Mixtral. The families that can license a selector come first. A family that
  cannot finish before the deadline is skipped (rc 40, host-limited), never started and cut.

## Receipts and exit codes

**Receipts:** `census_<family>.json` (every row, the text digests, the selfcheck, per-pass timing), `logs/`,
`versions.txt`, `forensics.txt` (card, power limit, RAM, disk, egress, the Hessian budget), `summary.txt`, the box's
reducer smoke (`p65_table.md`), and the teardown proof. They go to `receipts/experts4bit-qlora/<date>/<run_id>/` and
`bench/p65/receipts/`; the receipt and its ledger row are committed together. `RESULTS-p65.md` quotes only
`p65_reduce.py` over the fetched receipts.

**Box exit codes (`p65_run.sh`):**

| rc | meaning |
|---|---|
| 0 | every registered census present, complete, self-checked |
| 3 | a census's selfcheck failed |
| 9 | staging, install or tripwire (the tripwire refuses an e4b without `activation_means=`) |
| 10 | no CUDA |
| 11 | fetch failed |
| 12 | pin mismatch or bake failed |
| 13 | host floor: disk, RAM or host Hessian scale-and-add (`REFUSAL` names which) |
| 14 | egress floor |
| 15 | wrong card class |
| 40 | a family skipped for time |
| 41 | a registered census incomplete |
| 78 | environment refusal |
| 130 | interrupted |

**Controller exit codes (`p65_drive.sh`):**

| rc | meaning |
|---|---|
| 20 | stage |
| 21 | start or nonce handshake |
| 22 | fetch |
| 23 | no TP_DONE |
| 24 | stale nonce or malformed rc |
| 25 | the lane died on the box |
| 78 | refusing |

Under `P65_PROVE=1` the same box codes apply: rc 0 means the proof passed, and `MODE` reads `prove`. The verdict is
read from the JSON, never from an exit code.

Amendments, dated, go below this line before any data is read.

## Amendment 1 (2026-09-24 ~02:10Z, after three proof draws and before any reading)

**What happened.** Three proving attempts ran under this registration, and none reached the install:

| run | outcome | cost |
|---|---|---|
| `p65-prove-1` | refused rc 13, host scale-and-add **0.152 s** against the 0.15 s floor | $0.0419 |
| `p65-prove-2` | refused by the launcher before renting: my manifest named the first receipt in the wrong exclusion class, and the run id was burned (adertha-agents#112) | $0.00 |
| `p65-prove-3` | refused rc 14, egress **97.8 MB/s** against the 100 MB/s floor | $0.0590 |

Proof spend was **$0.1009** against this registration's "all proving attempts together ≤ $0.15". No reading was rented.

**Two design flaws, both mine.**
1. **The proof enforced floors that only the reading needs.** The RAM, host scale-and-add and egress floors exist
   because Mixtral's 16-layer census has to fit the reading's 2 h guard. A proof never runs Mixtral, and **its box is
   not the reading's box**. So a proof refused on those floors proves nothing and spends the proof budget. Both real
   refusals missed by about 2 %.
2. **The 10 min guard cannot hold the proof.** Launcher boot and pre-flight took 3–5 min of the guard: the lane began
   with 438 s and 324 s left. The proof itself needs about 8 min (install ~5, the Granite-3B fetch ~1, bake and
   cut-down census ~2). So even a box passing every floor would have stopped for time.

**Amended (the code is in this change; everything else in this registration is unchanged).**
- **In proof mode (`P65_PROVE=1`), the RAM, host scale-and-add and egress floors are measured and recorded, not
  enforced.** `floor()` in `p65_run.sh` writes `floor_would_refuse_reading rc=<code> <measurement>` to `forensics.txt`
  and `summary.txt`, then continues. Card class (15), a dud box (10) and disk (13) still refuse a proof, because those
  would break the proof itself. **The reading enforces every floor exactly as registered.** The proof's recorded
  measurements are reported beside the reading's.
- **The proof's guard is 0.23 h** (≈ 14 min) at ≤ $0.65/h, so **≤ $0.15 per proof**. This matches the earlier
  proving rentals' 0.2 h guards (`p55x-prove`, `p56-prove`) and keeps each proof inside the compute rule's per-proof
  $0.15.
- **The proof budget is ≤ $0.45 over all attempts, including the $0.1009 already spent**, and each attempt is
  ≤ $0.15. The lane ceiling rises from $1.75 to **$2.05** for proofs, the reading and refusals, and stays under the
  $3.00 hard stop. With the Mixtral follow-up and its proof, the ceiling becomes $3.00, the unchanged hard stop.
- **Pass and fail are unchanged.** A proof passes iff rc 0. A harness failure means no reading. A host refusal on class,
  dud or disk draws another proving box.

**What this does not change.**
- the question, instrument, families, texts, predictions and decision rule;
- the reading's box floors, guard and estimate;
- the reducer.

The runner's reading path is byte-identical except where the three floor checks now call `floor()`, which in a reading
behaves exactly as the old inline refusal, with the same exit codes. `staged.sha256` is updated for `p65_run.sh`.
