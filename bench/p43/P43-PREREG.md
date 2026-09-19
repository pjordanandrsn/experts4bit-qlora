# P43 — TRAINING CORRECTNESS, TWO LANES: the Qwen3 field-recipe step-degradation DIAGNOSIS (T1) and the Gemma-4 fused/reference LAYER-1 ADJUDICATION with the bf16 oracle resident (T2) — pre-registered 2026-09-18, before any box is rented

Work item: adertha-agents#110 (owner directive 2026-09-18: "time to go on the throughput and training work for all models supported"). Lineage: tp4 (`bench/tp4/TP4-PREREG.md`, 2026-09-10) left two OPEN correctness items on the training side, both of which block any training claim for their family. Nothing here is an optimisation; the phase rule (2026-09-05: correctness before optimisation) puts these first.

## Claim under test

**T1.** e4b's fused training path on Qwen3-30B-A3B at the FIELD recipe (tp4's fixture: seq 2048, micro-batch 2 × accum 4, r 16 / α 16, lr 2e-4, AdamW-8bit, linear warmup 5, seed 3407, the registered Alpaca subset) completes N = 20 optimizer steps in a bounded time, with per-step time that does not grow. tp4 (2026-09-10/11) measured the contrary: Unsloth finished 20 steps in 1423 s at 24.66 GB; e4b ALARMED at 3600 s at both mb2×accum4 and mb1×accum8 (21.6–23.1 GB), having reported step 10 at 24.7 s (mb2) and 47.4 s (mb1) — so the later steps averaged far more than the early ones. GPU signature while degrading: util 100 % at ~105 W (tiny kernels). The one "sharpened" read (no step 1 in 2 h 23 min) followed a SIGKILL of a sibling process during Triton JIT and is registered as **confounded, not a measurement**.

**T2.** experts4bit-qlora#558: on Gemma-4-26B-A4B-it, e4b's fused expert path and e4b's own dense per-expert reference disagree by 0.090 nats at STEP 0 — a forward-pass difference, deterministic (within-path noise floor exactly zero), first diverging at **decoder layer 1** (layer 0's output is bit-identical across arms). A whole-stack bf16 oracle did not adjudicate (loss favoured the reference, logits the fused path, per-layer the reference 18/12/1; both 4-bit paths sit ~+1.1 nats above the oracle, the registered independence tell). The layer-1 probe with an early-exit hook (`bench/p43/g4/gemma4_layer1_probe.py`, committed here from the mini's untracked copy) isolates ONE decoder block's computation from identical input; four attempts on 32 GB hosts died on host performance before the oracle sweep. It needs the 52 GB bf16 oracle RESIDENT, i.e. an 80 GB-class card.

## T1 — design

One RTX 5090 (the class every prior training number lives on), **a CLEAN box: no process on it is ever SIGKILLed mid-JIT**; between arms the Triton cache is cleared (`rm -rf /root/.triton/cache`) so an alarmed arm cannot leave a stale compilation lock for the next one (the confound tp4 hit). `TP4_BOX=D` in `bench/tp4/tp4_run.sh` (this PR): the tp4 fetch/tokenise path unchanged (same pinned revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, same `ds_alpaca.json` sha `5324987a…`, same tokeniser step), **e4b arms only**, no Unsloth venv (the comparator already has its row: `2026-09-11/tp4-qwen-mb1-2`), N = ${TP4_STEPS:-20}, every optimizer step printed (`--log-every 1`) and **every micro-batch timed** (`--microbatch-timing 1`, T17 in `tp4_arm.py`: one cuda sync per micro-batch; recorded as `microbatch_ms` per step), `TRITON_PRINT_AUTOTUNING=1` in the arm's log, GPU util / memory / power sampled every second per arm (tp4's `vram_<arm>.txt`, unchanged).

Arms, in this order, each one process with its own alarm (`TP4_DIAG_ALARM`, default 5400 s, deadline-capped as tp4 does):

| tag | arm | recipe | what it separates |
|---|---|---|---|
| `fused_attn4` | `enable_fast_train(dgrad=True)` + attention 4-bit (the tp4 primary, as run) | field, mb2 × accum4 | the shape of the curve: CLIFF vs RAMP; prologue (LOAD OK → step 1) vs steady |
| `fused_bf16attn` | the same fused expert path, attention **bf16** (`--attn-4bit 0`) | field, mb2 × accum4 | whether the degradation lives in the attention-4-bit / int4 store path or in the expert path |
| `fused_attn4_mb1` | the tp4 secondary | mb1 × accum8 | whether the effect scales with micro-batch rows (activation memory) or with steps |

An arm that alarms is a ROW with its partial curve (`step_ms`, `microbatch_ms`, `steps_done` in the ALARM stub) — the curve up to the alarm is the deliverable, not a failure of the lane.

## T2 — design

One 80 GB-class card (policy allowlist: `H100 NVL` 94 GB is what the market offers today; `A100 80GB PCIe` / `H100 PCIE` accepted if offered), `vast:verified-secure`. `bench/p43/g4/p43_g4_drive.sh` (controller) stages `gemma4_layer1_probe.py`, `p43_g4_run.sh` and the registered tokens file; the box script refuses BEFORE any download when (a) the card holds < 80 GB (exit 15) or (b) HF CDN egress < 20 MB/s over a 50 MB range (exit 14) — each a refusal row naming its reason, never a 90-minute stall. Then: the tp4 e4b venv recipe at the pinned heads; fetch `google/gemma-4-26B-A4B-it` @ `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` (pin asserted, never coerced); the probe over **128 rows in chunks of 8** (real positions only; ids sha recorded): e4b 4-bit reference sweep → `enable_fast_train` fused sweep → free → bf16 oracle (`experts_implementation="eager"` where the class accepts it) sweep, all capturing decoder layer 0's output (= layer 1's input) and decoder layer 1's output through an early-exit hook. Output `gemma4_layer1_probe.json`: `layer0_sanity` (reference vs oracle, fused vs reference — must be ≈ 0), `layer1` rms / max / mean_abs for reference-vs-oracle, fused-vs-oracle, fused-vs-reference, the verdict and the fused/reference rms ratio, load timings, peak VRAM.

Tokens: `tokens_gemma4.json` from receipt `2026-09-10/tp4-c-4` (the tp4 Gemma-4 tokenisation of the registered Alpaca subset with this revision's tokenizer): file sha256 **`4a1e8bf309b2654ca5ae0b56315bc6c37ee104cd524fc0bbf75c845a19684333`**, inner `sha256` field **`f758d35ebff378dc06af031f03003dc73877eec1e8b2b45234ed98bb76da1821`**; the driver refuses any other bytes.

## Environment (both lanes)

e4b at this PR's merge commit (the launcher pins it as `heads.e4b`); grouped-nf4-gemm **v0.31.0** `24f8c9fb22673a77219b8645b3bb8be50aa158c4` (the released kernel; tp4 ran `d9fd170`, 0.30.2 + #357); transformers 5.17.0, bitsandbytes 0.50.2, peft 0.20.0 (tp4's pins); image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`. Versions recorded in every receipt (`versions.txt`).

## Registered predictions (falsifiable)

- **P1 (T1, shape):** the curve is a **RAMP** — `step_ms` grows monotonically (Spearman ρ of step index vs step_ms over steps 2..N > 0.8) rather than a step-function; because tp4's step-10 readings (24.7 s / 47.4 s) with a 3600 s alarm imply later steps averaged > 200 s.
- **P2 (T1, tier):** the `fused_bf16attn` arm **does not degrade** (ρ < 0.5 and last-step/step-2 ratio < 1.5) — the degradation lives in the attention-4-bit path (`Int4Linear` / `quantize_attention_projections_4bit` machinery under training), not in the fused expert kernel. Registered alternative: it degrades the same way → the expert path or the trainer loop.
- **P3 (T1, not JIT):** after step 1, `TRITON_PRINT_AUTOTUNING` emits no new autotuning lines, and within a degraded step the micro-batches are of similar length (max/min ≤ 1.5) — the growth is per step, not a recompilation on some steps.
- **P4 (T1, signature):** the sampled GPU during degraded steps shows utilisation ≥ 95 % at power ≤ 150 W on a 575 W card (the tp4 signature: many tiny launches), while the first steps show ≥ 300 W.
- **P5 (T2):** at layer 1 the **reference is closer to the oracle**: rms(fused, oracle) / rms(reference, oracle) ≥ 1.2 — the fused Gemma-4 expert path carries the discrepancy (the per-layer count favoured the reference 18/12/1). Registered alternative: ratio ≤ 0.83 → the reference (per-expert loop) is the unfaithful path on this family.
- **P6 (T2, sanity):** `layer0_sanity` rms < 1e-3 for both pairs (identical inputs to layer 1); otherwise the run is VOID (the hook captured the wrong tensor), not a verdict.

## Decision rules, registered in advance

- T1: P1's shape decides the next lane: RAMP → a per-step accumulation (memory growth, a list that grows, autograd graph retention, fragmentation) — the follow-up instruments allocator stats per step; CLIFF → something fires on a schedule (eval, checkpoint, allocator compaction) — the follow-up aligns the cliff step with the trainer's events. P2 decides the tier. No fix is written in this lane; the fix lane is registered from the curve.
- T2: ratio ≥ 1.2 → **fused-path defect** (open the e4b kernel/dispatch issue for the Gemma-4 epilogue, with the layer's numbers; no Gemma-4 training number is quoted until it closes); ratio ≤ 0.83 → **reference-path defect** (the per-expert loop, which every parity control rests on — escalate: it invalidates tp1's Gemma-4 PASS reading); 0.83 < ratio < 1.2 → **inconclusive at layer 1** (both paths equally far from the oracle: the divergence measured at layer 1 is quantisation error in both, and the 0.09 nats emerges later — the follow-up moves the hook to layer 2..k). P6 failing → VOID, re-run only after the hook is fixed.

## Budget and STOP rules

| run | class | $/h ceiling | guard | estimate |
|---|---|---|---|---|
| `p43-prove` (Motion 1 proving run) | RTX 5090 | 0.65 | 10 min | ≤ $0.15 |
| `p43-t1-qwen3` | RTX 5090 | 0.65 | 5 h | $3.25 |
| `p43-t2-g4layer1` | H100 NVL (≥ 80 GB) | 3.10 | 2 h | $6.20 |

Both under the $35/run cap; day total ≤ $10 against the $100 global budget; role CTO. STOP: a box not of the registered class (refused, exit 15); egress below the floor (refused, exit 14); fetch alarm (5400 s) → `not_run` rows, host-limited; a second host-limited draw on the same lane → stop the lane and report (no third draw without an amendment). Every rental torn down with proof; the receipt dir is the record.

## Reading rules

Ratios and curve shapes only within a box. No training THROUGHPUT position is quoted from T1 (a diagnosis arm with a sync per micro-batch is not a benchmark arm). T2 reports rms distances and their ratio; it does not report a loss. A refusal is a row; an alarm is a row with its partial curve.

## Receipts

`receipts/experts4bit-qlora/2026-09-18/<run_id>/` in adertha-receipts (the launcher's receipt + `tp4/` or `probe/` work dirs). Results memo `bench/p43/RESULTS-p43.md` after both lanes, quoting only what the rows carry.

## Amendments

(dated entries, before the data they touch)

### Amendment 1 (2026-09-19 ~02:10Z, after T2's layer-1 result and before any per-layer data): T2b, the per-layer sweep the decision rule named

T2 ran on an H100 NVL (receipt `2026-09-19/p43-t2-g4layer1`; 13 min wall, $0.61): layer-0 sanity **exactly 0** on both pairs (P6 held). At decoder layer 1, over 23,314 real positions: rms(reference, oracle) = 0.063963, rms(fused, oracle) = 0.063992, **ratio 1.0004** → the registered rule's **inconclusive band** (0.83 < ratio < 1.2): the oracle does not prefer either path. But **fused and reference DO differ from each other at layer 1: rms 0.00439, max |Δ| 1.0, mean |Δ| 0.00068** — about 7 % of either path's distance to the oracle, deterministic (the within-path noise floor is exactly zero). So the divergence e4b#558 measured is present from the first expert block, and the oracle cannot say which side of it is faithful there. P5 (ratio ≥ 1.2 at layer 1) is **refuted**; the offloaded whole-stack run's "first diverging layer = 1" is **confirmed** with a resident oracle, its per-layer 18/12/1 count is not.

T2b (`bench/p43/g4/gemma4_layer_sweep.py`, `P43_G4_PROBE=layer_sweep`): the same three arms on the same 128 rows / 16 chunks, forward hooks on **every** decoder layer (no early exit; the oracle is resident), reporting per layer rms(fused − reference), rms(reference − oracle), rms(fused − oracle), the ratio, and the **first layer at which rms(fused − reference) exceeds each of 1e-3, 1e-2 and 5e-2** (layer 1 sits at 4.4e-3, so the 1e-2 and 5e-2 crossings are the amplification onsets).

Registered predictions for T2b: **P7 (amplification)** — rms(fused − reference) grows with depth and exceeds 5e-2 by the last decoder layer (the 0.09-nat loss delta needs an output-scale difference); registered alternative: it stays within 2× its layer-1 value throughout — then the loss delta arises at the head, not in the expert stack. **P8 (faithfulness)** — the fused/reference-to-oracle ratio stays within [0.9, 1.1] on every layer (neither path is closer to bf16 anywhere: the disagreement is second-order to quantisation, and #558's remedy is a re-derived per-family parity band, not a kernel fix); registered alternative: the ratio leaves that band at some layer k and stays out — then the path that is farther from the oracle from k on is the unfaithful one and #558 becomes a defect in it.

Budget: `p43-t2b-g4sweep`, H100 NVL, $3.10/h ceiling, guard 1 h, estimate $3.10 (T2's checkpoint fetch took 7.5 min on this class; the three sweeps under a minute).

### Amendment 2 (2026-09-19, before any T2b data) — the sweep's layer discovery

Run `p43-t2b-g4sweep` (H100 NVL, instance 51521714) produced NO per-layer data: `gemma4_layer_sweep.py` found the
decoder as "the first `ModuleList` named `*.layers`", and on the bf16 oracle (the full multimodal class) that is the
vision tower's 27 encoder layers, while the e4b side had walked the 30-layer text decoder → `IndexError: index 27`
before any distance was read. HARNESS_ERROR, receipt committed. Fix: the sweep now uses the layer-1 probe's explicit
path list (`model.layers`, `model.language_model.layers`, …), pins the oracle's list to the e4b side's layer count,
records `oracle_layer_path`, and refuses if the three arms' captured shapes differ at any layer. P7/P8 and the
reading rules are unchanged; the redraw is `p43-t2b-g4sweep-2`.

## Read (2026-09-19, after both lanes; the rows are the receipts, this section only reads them)

### T1 — Qwen3 field-recipe diagnosis (`p43-t1-qwen3-2`, RTX 5090, vast 51519025; run 1 `p43-t1-qwen3` never authenticated ssh, NOT_RUN, $0.13)

Three arms, N=20 at the registered field fixture (alpaca, seq 2048, mb2 × accum 4, r 16, adamw_8bit), all `CELL OK`,
VALID; e4b `ec22050` + gnf4 v0.31.0; receipts `2026-09-19/p43-t1-qwen3-2/tp4/`:

| arm | s/step (median, steps 11+) | step_ms, steps 0..19 (s) | Spearman ρ (steps 2..N) | last / step 2 | tok/s | GPU util median / power median | kernel calls per step |
|---|---|---|---|---|---|---|---|
| `fused_attn4` | **29.30** | 32.1 29.3 28.7 28.4 27.1 28.5 28.5 27.9 27.0 31.2 27.9 27.2 29.4 28.7 29.2 28.9 31.5 31.3 31.2 33.1 | 0.699 | 1.155 | 51.8 | **16 % / 113 W** (max 35 % / 163 W) | 768 |
| `fused_bf16attn` | **29.47** | 32.3 29.0 27.6 28.0 28.3 28.6 27.8 28.1 26.8 31.9 28.4 27.4 29.6 29.4 29.0 28.2 31.2 31.2 31.2 32.4 | 0.674 | 1.173 | 51.9 | 16 % / 114 W | 768 |
| `fused_attn4_mb1` (mb1 × accum 8) | **50.65** | 50.2 46.9 51.6 51.6 51.6 50.9 53.8 49.0 51.6 51.8 48.6 50.7 50.1 50.6 49.6 48.5 52.4 54.2 53.1 53.7 | 0.162 | 1.041 | 29.8 | 14 % / 103 W | 1536 |

Held-out loss 1.918 → 0.903 (`fused_attn4`), 1.994 → 0.894 (`fused_bf16attn`), 1.918 → 0.904 (mb1): the three arms
train to the same place. `TRITON_PRINT_AUTOTUNING` emitted **zero** autotuning lines in any arm.

- **P1 (RAMP) — NOT MET, and the collapse did not reproduce.** ρ = 0.70 / 0.67 (< 0.8), last/step-2 = 1.16 / 1.17: a
  mild monotone drift (+15 % over 20 steps, 27 → 33 s) in both mb2 arms, absent in the mb1 arm (ρ 0.16), and nothing
  like tp4's reading (24.7 s at step 10 with later steps averaging > 200 s). Neither the RAMP nor the CLIFF shape the
  decision rule was written for appeared in 20 steps on this host (EPYC 7302, 16 cores, 528 GB).
- **P2 (tier) — the attention-4-bit path is NOT the tier.** `fused_bf16attn` and `fused_attn4` are indistinguishable
  (29.47 vs 29.30 s/step, same drift, same ρ); the mb1 arm with int4 attention is flat. The registered alternative
  applies: whatever drifts, drifts in the expert path or the trainer loop, and only at mb2. (P2's literal ρ < 0.5
  clause fails on `fused_bf16attn` — ρ 0.67 — so it is read with its alternative, not as a pass.)
- **P3 (not JIT) — HOLDS.** No autotuning after step 0; micro-batch spread within a step median 1.29× (max 1.62× at
  step 0, the warm-up step).
- **P4 (signature) — REFUTED in both halves.** The card never showed ≥ 300 W: **utilisation 16 % at 113 W from step 0
  to step 19** (max 35 %, 163 W), i.e. the whole run is host-bound, not a run that degrades into host-boundness.
  768 kernel calls per optimiser step at mb2 (1536 at mb1) against a 575 W card sitting at a fifth of its power.

**What T1 says.** The field-recipe collapse tp4 recorded is not a property of the code at this fixture on this host;
it did not reproduce in 20 steps. What did reproduce is the number that matters for the throughput campaign:
**e4b trains the Qwen3 field recipe at 29.3 s/step with the GPU 16 % busy** — a host-bound step, the same class of
finding as the B=16 serving step (`finding_b16_is_host_bound`). The int4 attention is free here (P2). The follow-up
the decision rule named (allocator stats for RAMP, event alignment for CLIFF) is not the follow-up the curve asks
for; the curve asks where the host time goes in a training step at mb2 (a P42-style census of the training step's
host side), and that is a new lane to register, not a fix to write here. tp4's collapse stays on record as
unreproduced, with its own host (a different machine) as the first suspect.

### T2 / T2b — Gemma-4 per-layer three-way (`p43-t2-g4layer1`, `p43-t2b-g4sweep-2`, H100 NVL)

T2b (`gemma4_layer_sweep.json`, 30 text-decoder layers on both sides — `model.layers` on the e4b model,
`model.language_model.layers` on the oracle — 23,314 real positions over 128 rows / 16 chunks; run 1 died on the
oracle's vision-tower list, amendment 2):

| layer | rms(fused − ref) | rms(ref − oracle) | rms(fused − oracle) | ratio | | layer | rms(fused − ref) | rms(ref − oracle) | rms(fused − oracle) | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0044 | 0.0640 | 0.0640 | 1.0004 | | 15 | 0.307 | 0.620 | 0.616 | 0.994 |
| 1 | 0.0138 | 0.0805 | 0.0803 | 0.9985 | | 17 | 0.400 | 0.730 | 0.725 | 0.994 |
| 3 | 0.0246 | 0.0964 | 0.0967 | 1.0031 | | 19 | 0.381 | 0.647 | 0.639 | 0.988 |
| 5 | 0.0385 | 0.136 | 0.136 | 0.995 | | 21 | 0.389 | 0.635 | 0.629 | 0.991 |
| 7 | 0.0538 | 0.188 | 0.187 | 0.995 | | 24 | 0.415 | 0.671 | 0.666 | 0.993 |
| 9 | 0.117 | 0.350 | 0.352 | 1.0069 | | 27 | 0.349 | 0.574 | 0.569 | 0.992 |
| 11 | 0.196 | 0.520 | 0.520 | 0.9998 | | 28 | 0.336 | 0.567 | 0.562 | 0.991 |
| 13 | 0.229 | 0.578 | 0.577 | 0.999 | | 29 | 0.181 | 0.309 | 0.309 | 0.998 |

First layer with rms(fused − reference) over 1e-3 / 1e-2 / 5e-2: **0 / 1 / 7**. Ratio within [0.9, 1.1] on **every**
layer (range 0.988–1.007); the fused path is the marginally closer one on 25 of 30 layers, the reference on 5.

- **P7 (amplification) — HOLDS.** rms(fused − reference) grows 0.0044 → 0.42 (layer 24) and is 0.18 at the last layer
  (> 5e-2 from layer 7 on); it does not stay within 2× its layer-1 value. The loss delta is not born at the head.
- **P8 (faithfulness) — HOLDS.** The fused/reference-to-oracle ratio never leaves [0.9, 1.1]: **neither path is closer
  to bf16 anywhere.** The two 4-bit paths disagree with each other by an amount that grows with depth to about half
  their common distance from the bf16 oracle, and the oracle sits equidistant from both at every layer. The registered
  alternative (the ratio leaves the band at some k and the farther path is the unfaithful one) did not occur.

**What T2b says about #558.** Under the registered rule, P8 holding means the disagreement is second-order to
quantisation and **#558's remedy is a re-derived per-family parity band, not a kernel fix**: on Gemma-4 two 4-bit
paths that are each equally faithful to bf16 differ from one another by 0.18–0.42 rms in hidden state at depth, and
a 0.05-nat parity band between them is not a band this family can meet. The hidden-state instrument cannot say which
path lands the *lower* loss or why (T2's oracle loss comparison already found the metrics disagree); it can say the
fused path is not the unfaithful one. Deriving the per-family band is its own registered step, with the band's
derivation fixed before the number is read; no Gemma-4 training position is quoted until that step closes.

### Cost

T1: `p43-t1-qwen3` NOT_RUN $0.13, `p43-t1-qwen3-2` ≈ $0.62 (≈ 57 min box). T2: $0.61. T2b: run 1 HARNESS_ERROR ≈ $0.35,
run 2 ≈ $0.45. Proving runs ≤ $0.15 each. Lane total under $3; every rental torn down with proof.
