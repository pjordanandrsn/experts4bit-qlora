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
