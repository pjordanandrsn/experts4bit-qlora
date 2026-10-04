# TC2 — the other families at matched work on the current cuts: Granite-3.1-3B-A800M, OLMoE-1B-7B, gpt-oss-20b (box A); Qwen3.6-35B-A3B and Mixtral-8x7B (box B) — lane TC2 of #835, 2026-10-02

Pre-registration: [`../../tc1/TC2-PREREG.md`](../../tc1/TC2-PREREG.md) (with its amendment 1). The TC1 harness (`bench/tc1/`) at e4b
`09b0f6a` (0.38.1) for box A, grouped-nf4-gemm `846b512`; the same matched-work rules as TC1 (one per-slot init, fp32 adapters, the same
tokens per family, validity predicates, verdicts, two draws for a quoted position, within-box ratios only). Receipts under `receipts/<run id>/`
(private store `receipts/experts4bit-qlora/2026-10-02/<run id>/tc1/`).

| box | token | host (Vast verified-secure) | families | cost |
|---|---|---|---|---|
| `tc1-5090-17` | `tc2small` (N = 60, 48 held-out rows every 20 steps) | instance 53800083, AMD EPYC 9755, driver 580.119.02 | granite, olmoe, gptoss | $0.90 |
| `tc1-5090-22` (box B) | `tc2big` (the field recipe, N = 20) | instance 53828163, machine 117847 — **stopped by its host at 11:41Z** with the run's receipts unfetched (below) | qwen3_5, mixtral | $2.42 (`tc1-5090-18` and `-21` before it failed pre-flight on one dead machine, $0.03) |

## What the small families say

**The per-step edge e4b holds on Qwen3-30B-A3B on a 5090 does not generalise to small experts.** On both families where another
framework trained the experts in 4-bit or in bf16, the matched-work ratio is at or below parity in speed, while e4b keeps a VRAM edge
against bf16 experts and loses it against Unsloth's 4-bit stacks on OLMoE.

### Granite-3.1-3B-A800M (`ParallelExperts`; 32 layers)

| arm | s/step | ratio vs e4b `fused_attn4_m` (1.484 s, two draws) | peak VRAM | J/step | held-out at N = 60 (e4b 0.8311) | regime |
|---|---|---|---|---|---|---|
| HF + PEFT (`target_parameters`, two draws) | 1.441 | **0.971 [0.965, 0.978]** (HF faster) | 8.50 GB (e4b 4.54) | 334.9 (e4b 343.8) | 0.8286 (Δ −0.0024, COMPARABLE) | **bf16 experts** — bitsandbytes' Linear-only quantizer leaves `ParallelExperts` unquantised |
| axolotl 0.20.0 (`quantize_moe_experts`) | 1.340 | **0.903 [0.901, 0.905]** (axolotl faster; one draw) | 4.08 GB | 351.1 | 0.8295 (Δ −0.0016, EQUIVALENT) | **4-bit experts** (corrected 2026-10-02: `quantize_moe_experts` packed all 64 stacks as parametrized NF4, which the census's `Params4bit stacks 0` does not see; `RESULTS-tc1-5090-17-rereduced-2026-10-02.md`) |
| Unsloth 2026.9.14 (both target lists) | — | VOID | — | — | — | attention-only: 5,242,880 trainable vs e4b's 99,614,720 — it adapts no expert parameter (as registered; `coverage` row of tp4 stands) |
| HF on torch 2.14 + `experts_implementation="grouped_mm"` | 1.347 | 0.908 (labelled) | — | — | 0.8278 | the same bf16 experts, a different dispatch |
| e4b as shipped (bf16 expert adapters, N(0, 1/r)) | 1.247 | 0.840 (labelled) | 3.78 GB | — | 0.8329 (Δ +0.0018) | |

e4b's parity control PASSES (Δ final 0.00153, median 0.00168; ×5.81 vs its reference at ×1.09 the peak). Read: on an 800 M-active MoE
whose experts nobody else quantises, e4b's fused 4-bit experts train at 0.90–0.97 × the speed of bf16-expert training and at 0.5 × its
VRAM (4.54 vs 8.50 GB) — a footprint position, not a speed position. TC2 P1 (HF/e4b in [1.1, 1.6]) is FALSIFIED. (Corrected 2026-10-02: that reading holds against HF; axolotl's
0.903 is 4-bit against 4-bit — its experts are NF4 — so on Granite axolotl's 4-bit path steps faster than e4b's at a lower peak.)

### OLMoE-1B-7B (16 layers, 64 experts)

| arm | s/step | ratio vs e4b `fused_attn4_m` (1.105 s, two draws) | peak VRAM | J/step | held-out at N = 60 (e4b 0.8424) | regime |
|---|---|---|---|---|---|---|
| Unsloth (grouped_mm, one draw) | 1.327 | **1.201 [1.199, 1.204]** (e4b faster) | **5.97 GB** (e4b 8.18) | **231.9** (e4b 326.2, ×0.71) | 0.8426 (Δ +0.0002, EQUIVALENT) | 4-bit expert stacks (+ double-quant) + bnb-4bit attention |
| HF + PEFT (two draws) | 1.387 | **1.255 [1.253, 1.258]** (e4b faster) | 16.32 GB | 490.0 (×1.50) | 0.8370 (Δ −0.0054, COMPARABLE) | bf16 experts |
| HF on torch 2.14 + `grouped_mm` | 1.316 | 1.191 (labelled) | 16.3 GB | — | 0.8376 | bf16 experts |
| axolotl 0.20.0 | — | UNSUPPORTED | — | — | — | its loader asserts `Original QKV code not found` on this family (both arms) |
| e4b as shipped | 0.902 | 0.816 (labelled) | 6.62 GB | — | 0.8373 (Δ −0.0051) | |

e4b's parity control PASSES (Δ final 0.00095, median 0.00159; ×6.08 vs its reference). Read: e4b is 1.20 × faster per step than Unsloth's
engaged 4-bit path and 1.26 × faster than bf16-expert HF, but **Unsloth trains this family at 2.2 GB less peak VRAM and 0.71 × the energy
per step, with the same loss**. TC2 P2 (Unsloth in [1.2, 3.0] — 1.201 sits on the edge; HF in [1.5, 2.5]) is FALSIFIED on the HF half.
The attention bytes differ between the two quantisers on this family (`q_proj` DIFFERENT, the control detecting), unlike Qwen3-30B-A3B where
they are SAME-BYTES: recorded, not resolved here.

### gpt-oss-20b (MXFP4 experts in the checkpoint)

No common adapter set, as registered: e4b has no expert path on this family (`enable_fast_train` patches 0 modules; the experts are
built bare — tp1/tp2 cited) and trains the attention only (3.820 s/step, 7,962,624 trainable, 17.82 GB, two draws STABLE). Unsloth's
bnb-4-bit load adapts the experts but left no `Params4bit` stack (VOID: 0 of 48, a silent fallback); its 16-bit load with the experts kept
packed in MXFP4 trained (`ckpt_unsloth_mxfp4`, two draws) but is VOID by the step-0 rule — its held-out loss at step 0 is 3.2109 against
the e4b anchor's 3.1100 (Δ 0.10 > 0.05): the two arms do not start from the same bytes, so no equivalence and no ratio. HF and axolotl refuse
the family at load (`The model is quantized with Mxfp4Config but you are passing a BitsAndBytesConfig`). TC2 P3 is FALSIFIED. No position.

## Box B: lost with its instance (TC2 amendments 2 and 3; re-runs registered)

`tc1-5090-22` ran from the TC3 amendment-4 merge (`39834a7`) for 4 h 40 min and stopped answering ssh at 11:42Z, inside Mixtral's Unsloth
micro-batch-1 secondary. The provider then listed the instance as exited and **stopped by the host** (memory use 85 % at the stop; its last
duration ending about 11:41Z), the ssh gateway refused every connection, the controller's deadline fetch failed and the guard destroyed the
instance with its disk (teardown complete, 12:20:45Z). The controller's heartbeat log holds the status of the cells it sampled, never a
number, so **nothing from that box is registered**; the private store's receipt for the run records the loss. What the heartbeats show, as
statuses only and with cells the sampling missed: Qwen3.6-35B-A3B — e4b's matched arms OOM at the field recipe (both draws) and at
micro-batch 1, e4b as shipped OK, Unsloth's three matched arms OK, HF OOM, axolotl's scattermoe arm refused; Mixtral-8x7B — e4b under
offload OK (both draws and at micro-batch 1), e4b as shipped OK, **both Unsloth matched draws out of their 2,400 s alarm with no step
taken** (still loading at about 30 GB on the card), HF OOM, axolotl's scattermoe arm refused. Two amendments follow from it:
amendment 2 (`tc2mixtral`, merged as `31cb04d`) re-asks Mixtral's P5 pair with the Unsloth alarm at 7,200 s, and amendment 3 (merged as
`0933d4e`) re-runs the Qwen3.6 half on its own box and gives the Mixtral box a 192 GB host-RAM floor, because Unsloth's loader
materialises the 93 GB bf16 checkpoint in host memory before it quantises. Three launches of those boxes were refused at $0 by the
launcher's own rules (`tc1-5090-23`: an 8 h guard over the 6 h teardown cap; `-24` and `-25`: the pool's cheapest verified RTX 5090 had
moved above the lane's $0.54/h, with or without the RAM floor), and the boxes ran as `tc1-5090-27` (Qwen3.6, below) and `tc1-5090-26`
(Mixtral, below) at the $0.69/h ceiling box B itself had run at — the lane's budget line says $0.54/h, and the receipts say $0.69/h.

## Box B re-run, Qwen3.6-35B-A3B (`tc1-5090-27`: instance 53859323, AMD EPYC 7663, driver 580.95.05, $0.54 actual)

40 layers, 256 routed experts, 20,520 adapter slots: the matched set is **926,187,520** trainable parameters (fp32 adapters, one per-slot init,
TC1's rules). Register `e4b.train.h2h.unsloth.qwen3_5.5090.2026-10-02` and its `.shipped-labelled` row.

| arm | status | s/step | tok/s | peak VRAM | J/step | held-out 0 → 20 | trainable | note |
|---|---|---|---|---|---|---|---|---|
| e4b `fused_attn4_m` (×2) | **OOM at step 1** | — | — | 32.52 GB | — | — | 926,187,520 | 30.86 GiB in use at a 606 MiB allocation, both draws |
| e4b `fused_attn4_m_mb1` | OOM at step 2 | — | — | 32.54 GB | — | — | 926,187,520 | the micro-batch-1 secondary |
| e4b `reference_attn4_m` | OOM at step 2 | — | — | 32.84 GB | — | — | 926,187,520 | the reference loop |
| **Unsloth `ckpt_unsloth_m_experts`** (the family's own expert names) | **OK · VALID** | 10.593 | 143.1 | **30.47 GB** | 1603.5 | 1.1940 → 0.6880 | 926,187,520 | one draw; matched init complete (20,520 slots); 4-bit expert stacks (80), `grouped_mm` 320 calls/step |
| Unsloth `ckpt_unsloth_m` / `_d2` / `_mb1` (the registered target list) | OK · **VOID** | 4.87 / 4.89 / 8.86 | — | 21.08 GB | — | 1.1940 → 0.8355 | 3,440,640 | attention-only on this family: 40 of 20,520 slots set — tp4's VOID reproduced on 2026.9.14 |
| e4b `fused_attn4_shipped` (bf16 expert adapters, N(0, 1/r)) | OK · VALID | 5.088 | 314.6 | 32.48 GB | 1200.0 | 1.1433 → 0.6729 | 926,187,520 | a fit row, never a position: different adapter precision and init from every matched arm |
| HF + PEFT `hf_peft_m` / `_mb1` | OOM at load | — | — | — | — | — | — | 31.33 GB in use at a 512 MiB allocation |
| axolotl `ckpt_axolotl_m` | UNSUPPORTED | — | — | — | — | — | — | `NoMatchingPeftModuleError`: its target-module names do not match this family's modules |
| axolotl `ckpt_axolotl_best` | UNSUPPORTED | — | — | — | — | — | — | the scattermoe KernelsPlugin fetches `kernels-community/activation` with the Hub offline |

**Read: on Qwen3.6-35B-A3B at matched work, Unsloth trains the set resident on 32 GB and e4b does not.** e4b's fused path with the matched fp32
adapters and their AdamW-8bit state over 20,520 slots needs more than the card has at both micro-batches and in its reference loop; Unsloth,
given the family's own expert names, trains the same 926 M parameters at 30.47 GB and 10.59 s/step. An e4b loss, said as such: no ratio
exists (its anchor never stepped), and the as-shipped e4b arm that does fit (bf16 expert adapters at 32.48 GB, 5.09 s/step, a lower
held-out loss) is a different adapter precision and init — the confound TC1 identified — so it is a labelled fit row beside Unsloth's
matched row, never a comparison. Unsloth's registered target list still adapts the attention only on this family (tp4's VOID, on the
current release); the lane's prediction that e4b's matched arm would land within 15 % of tp4's 6.33 s/step was wrong because that
tp4 arm ran bf16 adapters. **TC2 P4 FALSIFIED.** e4b under expert offload on this family is the obvious next row and was not registered
for this box. HF OOMs resident. axolotl's refusal here is the harness's too: it enumerated target names on an
`AutoModelForCausalLM` skeleton while axolotl's loader built the vision-language class (`is_multimodal: true`), whose language layers sit
under a different module path (TC1 amendment 4, corrected 2026-10-02; not yet fixed).

## Qwen3.6-35B-A3B with e4b under expert offload (`tc1-5090-29`, TC2 amendment 4, $0.82)

The row the box B re-run named. Every e4b arm under `--offload 1`; Unsloth resident with the family's expert target parameters on
both draws, so the pair is the reducer's ordinary matched pair. Register `e4b.train.h2h.unsloth.qwen3_5.5090.2026-10-02.e4b-offload`.

| arm | verdict | s/step | peak VRAM | held-out 0 → 20 | note |
|---|---|---|---|---|---|
| e4b `fused_attn4_m` (offload) ×2 | **VALID** | 12.372 / 13.846 | **19.35 GB** | 1.1433 → 0.6834 / 0.6831 | draws 11.2 % apart: UNSTABLE |
| e4b `reference_attn4_m` (offload) | VALID | 110.57 | 21.18 GB | 1.1381 → 0.6822 | parity PASS (final Δ 0.00133); fused ×8.94 its speed |
| Unsloth `ckpt_unsloth_m` ×2 (expert targets, resident) | **VOID** (step-0) | 10.107 / 10.618 | 30.47 GB | 1.1935 / 1.1962 → 0.6868 / 0.6904 | step-0 |Δ| 0.050 / 0.053 against e4b's anchor |

**Read.** **TC2 P8 HELD**: e4b trains Qwen3.6's 926 M-parameter matched set on the 32 GB card under expert offload, at 19.35 GB,
where it OOMed resident. **P9 UNTESTED** and **P10 parity PASS, equivalence N-A**: e4b's offload draws are unstable, and Unsloth's
arms are VOID by the step-0 rule because the two frameworks do not quantise the same module set on this family. e4b's census holds
40 expert stacks and the 40 full-attention projections in 4-bit and 311 Linear in bf16 (the 30 linear-attention layers' five
projections each, the 40 shared experts' four, `lm_head`), about 1.14 B parameters; Unsloth's holds 498 Linear4bit and 3 bf16. The
base model each trains from therefore differs by 0.05 nats at step 0 (e4b's is closer to bf16), and that difference also accounts
for about 1.6 GB of the resident footprint gap above. By step 20 the two reach 0.683 and 0.687 / 0.690. That e4b leaves this family's
linear-attention and shared-expert projections in bf16 is an e4b coverage gap worth closing, recorded here, not a comparison.

## Box B re-run, Mixtral-8x7B-Instruct-v0.1 (`tc1-5090-26`, TC2 amendment 2's token on a host with 1 TB of RAM: instance 53859501, AMD EPYC 7C13, 1.06 TB RAM, driver 595.71.05)

> **Correction (2026-10-02, TC2 amendment 5): this section's Unsloth readings are retired.** The 76–89-minute compile phase, the 6.2 / 7.0
> s/step, the 12 % draw spread and the peak were produced by this harness's engagement counters inside Unsloth's compiled Mixtral MoE block:
> Python dict increments wrapped around `torch._grouped_mm` and Unsloth's backend functions became Dynamo guards that failed on every call
> (3,547 unique graphs, 743 "HOP: Unsafe side effect" graph breaks, two hits of the 1024 recompile limit, after which frames ran eagerly).
> Unsloth compiles Mixtral's MoE block and leaves Qwen3's uncompiled, which is why no other family's Unsloth arm shows it (each is clean on
> the same counters). The candidate named below ("the harness's output-capture hooks") was the wrong mechanism. e4b's offload rows stand;
> the counters are now trace-safe and the pair is re-measured. `e4b.train.h2h.unsloth.mixtral.5090.2026-10-02` and `.compile-warmup` are retired.
> P7's FALSIFIED below, decided by Unsloth's second Mixtral draw, is withdrawn to UNTESTED; P5 stays UNTESTED; P6 (e4b's parity) stands.

32 layers, 8 experts, 640 adapter slots: the matched set is **223,346,688** trainable parameters. (The two Unsloth arm logs ship gzipped, `logs/*.log.gz`: 5.7 MB
each of kernel-compile progress bars; `gunzip -c` returns the box's bytes.) e4b runs under expert offload (the registered
lever on this family), Unsloth resident with the Unsloth alarm raised to 7,200 s.

| arm | status | s/step (median 11–20) | tok/s | peak VRAM | J/step | held-out 0 → 20 | train wall, 20 steps |
|---|---|---|---|---|---|---|---|
| e4b `fused_attn4_m` (offload) | OK · VALID | **15.891** | 105.2 | **7.151 GB** | 2,984 | 1.4297 → 0.7142 | 337 s |
| e4b `fused_attn4_m_d2` (offload) | OK · VALID | **15.719** | 107.8 | 7.143 GB | 2,968 | 1.4297 → 0.7132 | 329 s |
| Unsloth `ckpt_unsloth_m` (resident) | OK · VALID | **6.213** | 6.6 (whole run) | **31.951 GB** | 11,564 (whole run) | 1.4326 → 0.7107 | 5,413 s |
| Unsloth `ckpt_unsloth_m_d2` (resident) | OK · VALID | **7.011** | 7.6 (whole run) | 32.239 GB | 10,218 (whole run) | 1.4326 → 0.7089 | 4,673 s |
| e4b `reference_attn4_m` (offload, the parity control) | OK · VALID | 16.079 | 109.4 | 10.595 GB | 2,414 | 1.4259 → 0.7134 | — |
| HF, both axolotl arms, e4b as shipped | `not_run` stubs (amendment 2: box B holds those rows — and box B's receipts were lost; its heartbeats read HF OOM, axolotl scattermoe refused, e4b as shipped OK) | | | | | | |

**What box B's two alarms were.** Unsloth's load of the 93 GB checkpoint is not the problem: on this host it took 40 s. The first training steps
are: Unsloth 2026.9.14 compiles `MixtralSparseMoeBlock` ("without fullgraph since output capture hooks run inside it", its own log line) and
then spends the first six steps compiling Triton kernels — 864, 882, 983, 1,032, 819 and 741 s for steps 1 to 6 (89 minutes), the GPU idle and
one core pinned, 5,016 "Compiling kernels" batches over the arm — before settling to 6 to 7 s/step from step 7 on. The arm's host-RAM high-water is 95 GB, which
is what stopped box B's 98 GB host; the 192 GB floor amendment 3 asked for was right for the compile phase, not for the load. On the previous
Unsloth release tp2 ran this arm at 0.858 s/step (`e4b.train.h2h.unsloth.mixtral.5090.2026-09-06.footprint`); the compiled MoE path of
2026.9.14 on Mixtral's per-expert block is the difference, with the harness's output-capture hooks (which force the partial graph) a candidate
factor, not established here — the same hooks sit inside Qwen3-30B-A3B's stacked-expert block, where Unsloth took 8 s/step from step 1.
**Read.** e4b's draws are STABLE (1.1 %); Unsloth's steady-state medians differ by 12.1 % (6.213 vs 7.011 s/step), outside the registered 5 %,
so **no position is quoted and TC2 P5 is UNTESTED by its own rule**. Said beside it, not scored: the steady-state ratios Unsloth/e4b
(0.391 and 0.446) sit inside P5's band [0.3, 0.5], and the footprint ratio (×4.47 / ×4.51 lower peak on e4b) does not reach P5's 8× clause.
Held-out at N = 20 is within 0.005 nats across all four draws. The registered metric is the steady-state median, and that is Unsloth's
advantage here — about 2.3–2.5× faster per step once compiled. The wall-clock is the other way round on this host: 337 / 329 s for e4b's
20 steps against 5,413 / 4,673 s for Unsloth's, and — arithmetic on these receipts, not a run — the two cumulative curves cross between steps
519 and 544. A one-time compile on one host is not a property of the comparison; it is recorded because it decides which framework
finishes a short fine-tune first on this card, and because it is what lost box B.
e4b's parity control PASSES (fused vs reference under offload: final Δ 0.00195, median step 0.00166; **P6 HELD**). Under offload the fused path is
only ×1.01 faster per step than the reference loop on this host, at ×0.675 its peak: Mixtral's step under expert offload is bound by the stream of
expert bytes, not the expert GEMM. **P7 FALSIFIED**: Unsloth's first draw is EQUIVALENT to e4b's anchor (held-out |Δ| 0.0035), its second
COMPARABLE (0.0053, just outside the 0.005 band).

## Mixtral-8x7B, re-measured with the counters trace-safe (`tc1-5090-32`, TC2 amendment 5, $0.56)

The same token and arms as `tc1-5090-26`, run from the amendment-5 merge (`3e41f0f`), on an AMD EPYC 7663 host with 792 GB of RAM.
Register `e4b.train.footprint.unsloth.mixtral.5090.2026-10-02`, superseding the two retired rows.

| arm | verdict | s/step (median 11–20) | peak VRAM | held-out 0 → 20 | Dynamo |
|---|---|---|---|---|---|
| e4b `fused_attn4_m` (offload) ×2 | VALID | 19.751 / 21.181 | 7.164 / 7.145 GB | 1.4297 → 0.7147 / 0.7138 | not compiled |
| Unsloth `ckpt_unsloth_m` (resident) ×2 | VALID | **3.735 / 3.748** | 29.115 / 29.142 GB | 1.4277 → 0.7086 / 0.7095 | 29 frames, 26 graphs, 0 breaks, no limit hit |
| e4b `reference_attn4_m` (offload) | VALID | 20.618 | 10.595 GB | 1.4259 → 0.7135 | — |

**Read.** With the counters a registered custom op under tracing, Unsloth's Mixtral arm compiles once (its first step took 34 s) and
then steps in 3.5–4.1 s, counting 1,792 grouped-GEMM calls per step. The 76–89-minute "compile phase" in the retired rows above was
the old counters. Against e4b under expert offload, Unsloth is about **5.3× faster per step** and e4b's peak is **×4.07 lower** —
the footprint trade on this family, said as one. No position is quoted: e4b's two offload draws differ by 7.0 %, outside the 5 %
rule (the offload step is bound by the host link; it read 15.9 s on the earlier host), so **P5 stays UNTESTED**. The steady ratio,
about 0.18–0.19, sits far below P5's [0.3, 0.5] band, toward Unsloth, and is not scored. **P6 HELD** (parity |Δ| 0.00010; under
offload the fused path is ×1.04 the reference's speed, the step stream-bound). **P7 FALSIFIED**: Unsloth's held-out sits 0.005–0.006
nats under e4b's, COMPARABLE, just outside the equivalence band.

## Both big families with e4b RESIDENT (`tc1-5090-53`, TC2 amendment 6, $3.03 invoiced)

One RTX 5090 (Vast instance 54140024, AMD EPYC 7C13, 1.06 TB RAM, driver 595.71.05), token `tc2resident`, e4b `4ea0b48` (the amendment's merge),
grouped-nf4-gemm `34a0881`. Every e4b arm ran with `--offload 0`, after TC1 amendments 10–15 and the memory changes since 2026-10-02
(the lean LoRA delta on by default; the combine saving bf16, not fp32). Vast invoiced $3.03: GPU $0.98, storage $0.19, and $1.87 for
179 GB of checkpoint download. Receipts: [`receipts/tc1-5090-53/`](receipts/tc1-5090-53/).

### Mixtral-8x7B: e4b now fits resident, and Unsloth is 1.43× faster per step

| arm | verdict | s/step (median 11–20) | peak VRAM | J/step | held-out 0 → 20 |
|---|---|---|---|---|---|
| e4b `fused_attn4_m` (resident) ×2 | VALID | 5.885 / 5.923 | **31.07 / 31.08 GB** | 2,055 / 2,032 | 1.4313 → 0.7113 / 0.7145 |
| Unsloth `ckpt_unsloth_m` (resident) ×2 | VALID | **4.024 / 4.203** | 29.12 / 29.14 GB | 1,367 / 1,284 | 1.4291 → 0.7097 / 0.7092 |
| e4b `reference_attn4_m` (resident) | VALID | 6.719 | 30.31 GB | 1,641 | 1.4259 → 0.7135 |

- **P11 HELD.** e4b's fused path completes the matched set (223,346,688 fp32 adapters over 640 slots) resident at micro-batch 2,
  peaking at 31.07 GB, inside the registered 31.5 GB. On 2026-10-02 Mixtral had only ever run under expert offload.
- **P12 HELD.** The MATCHED POSITION is **Unsloth/e4b 0.697 [0.679, 0.714]** (4.114 against 5.904 s, two stable draws a side):
  Unsloth is faster per step, as predicted for a GEMM-bound step. Unsloth also runs at 1.94 GB less peak and ×0.65 the energy per step.
- **P14 HELD on Mixtral.** Both Unsloth draws sit inside e4b's draw noise (held-out Δ −0.0016 / −0.0021). The resident reference
  fits here, and e4b's parity PASSES (Δ final 0.00027, median step 0.00194; fused ×1.14 the reference's speed).
- Against the offload footprint row: e4b's step went from 19.8–21.2 s under offload to 5.9 s resident, so the resident pair is the
  comparison. `e4b.train.footprint.unsloth.mixtral.5090.2026-10-02` stays as the offload lever's reading.

**Why Unsloth is faster here, and where e4b's extra VRAM goes (a reading of the receipts, not yet a measured cause).** Mixtral routes
2 of 8 large experts (14,336 × 4,096), about 1,000 rows per expert per micro-batch, so the expert GEMMs dominate. That is the regime
where the H100 showed dequantize + a dense grouped GEMM beating the fused 4-bit kernels (TC1c amendment 3). e4b's fused path is only
×1.14 faster than its own reference loop on this family. The VRAM gap matches the expert absmax: e4b keeps it in fp32 (45.1 B expert
parameters / 64 × 4 bytes ≈ 2.8 GB), where Unsloth double-quantizes it (about 0.7 GB). That ≈ 2.1 GB is close to the measured 1.94 GB.

### Qwen3.6-35B-A3B: still out of memory resident, at both micro-batches

| arm | verdict | s/step | peak VRAM | note |
|---|---|---|---|---|
| e4b `fused_attn4_m` ×2 | **OOM** | — | 32.55 GB | at step 2 (2026-10-02: at step 1, 32.52 GB) |
| e4b `fused_attn4_m_mb1` | **OOM** | — | 32.68 GB | at step 2 |
| e4b `reference_attn4_m` | **OOM** | — | 32.84 GB | at step 2 |
| Unsloth `ckpt_unsloth_m` (expert targets) ×2 | VALID | 11.658 / 11.888 | **30.47 GB** | held-out 1.1962 → 0.6893 / 0.6897 |
| Unsloth `ckpt_unsloth_m_mb1` | VALID | 19.014 | 30.41 GB | the secondary pair's other side |

- **P13 FALSIFIED.** The micro-batch-2 half held (OOM again), but e4b also OOMs at micro-batch 1, which was predicted to fit. The
  memory changes moved the failure from step 1 to step 2: step 1 now completes, and the OOM comes after the first optimizer
  update has allocated the 8-bit Adam state (an inference from where it fails, not a memory trace). No position exists, and `e4b.train.h2h.unsloth.qwen3_5.5090.2026-10-02`'s loss stands, now
  re-asked on the current code.
- **P14** has no Qwen3.6 pair to read.
- Two parts of e4b's resident footprint are larger than Unsloth's on this family. The first is the fp32 expert absmax (about 2.1 GB,
  against about 0.5 GB double-quantized). The second is the 30 linear-attention layers' projections and 40 shared experts, about
  1.14 B parameters, which e4b keeps in bf16 where Unsloth stores 4-bit (about 1.6 GB; STATUS records it). Together they come to
  about 3.2 GB. e4b's recorded peak at the failure sits 2.1–2.4 GB above Unsloth's 30.47 GB, a lower bound on what it needed.

**What follows.** The next lever on both families is e4b's base footprint, not its adapters. First, double-quantized expert absmax
in grouped-nf4-gemm's kernels, a values-changing switch that needs its own A/B. Second, 4-bit storage for Qwen3.6's non-routed
projections. On Mixtral's speed, the lever is a dequantize-then-GEMM route for large experts on sm_120. Each needs its own registration.

## TC2 amendment 7 (2026-10-04): the small families re-read, and the big families with the absmax double-quantized

Pre-registration: [`../../tc1/TC2-PREREG.md`](../../tc1/TC2-PREREG.md), amendment 7. Three RTX 5090s, e4b `7145425`, grouped-nf4-gemm
v0.37.0 (`71185d6`). Receipts: [`receipts/tc1-5090-54/`](receipts/tc1-5090-54/) (box S), [`receipts/tc1-5090-55/`](receipts/tc1-5090-55/)
(box D), [`receipts/tc1-5090-56/`](receipts/tc1-5090-56/) (box F).

| box | host | e4b settings | invoiced |
|---|---|---|---|
| S | AMD EPYC 7C13 | defaults | $2.59 |
| D | Intel Core Ultra 9 285K | `E4B_ABSMAX_DQ=1` | $1.14 |
| F | AMD EPYC 7B13 | `E4B_ABSMAX_DQ=1 TRAIN_FROZEN_4BIT=1` | $3.00 |

### Box S: on the current code e4b is faster than each comparator on Granite and OLMoE

| family | pair | other/e4b | 2026-10-02 (e4b 0.38.1) | peak e4b / other | pair reads | prediction |
|---|---|---|---|---|---|---|
| Granite | HF (bf16 experts) | **1.299** [1.268, 1.332] | 0.971 | 4.31 / 8.50 GB | COMPARABLE | **P15 HELD** |
| Granite | axolotl (4-bit) | **1.150** [1.130, 1.170] | 0.903 | 4.31 / 4.08 GB | EQUIVALENT | **P15 HELD** |
| OLMoE | Unsloth (4-bit) | **1.821** [1.805, 1.838] | 1.201 | 7.67 / 5.97 GB | EQUIVALENT | **P16 HELD** (Unsloth half) |
| OLMoE | HF | not quoted: HF's draws 6.0 % apart | 1.255 | | COMPARABLE | P16's HF half UNTESTED |

e4b's parity PASSES on both families, and each quoted pair reads as on 2026-10-02 (**P17 HELD**). gpt-oss keeps no common adapter set,
and Unsloth still adapts no expert parameter of Granite, both as before. e4b's own step on this host is 2.11 s on Granite and 1.20 s on
OLMoE. On 2026-10-02 the same arms read 1.48 s and 1.11 s on other hosts; that comparison crosses hosts, so it is context, not a reading.
The three rows supersede the 2026-10-02 positions.

### Boxes D and F: the absmax double-quantized brings e4b's Mixtral peak level with Unsloth's

| | box D (285K) | box F (EPYC 7B13) | amendment 6 (fp32 absmax) |
|---|---|---|---|
| e4b peak | **29.03 GB** | 29.02 GB | 31.07 GB |
| Unsloth peak | 29.13 GB | 29.13 GB | 29.12 GB |
| Unsloth/e4b | 0.542 [0.534, 0.549] | 0.644 [0.641, 0.646] | 0.697 |
| e4b held-out, two draws | 0.7138 / 0.7155 | 0.7135 / 0.7136 | 0.7113 / 0.7145 |
| pair | COMPARABLE | COMPARABLE | EQUIVALENT |

- **P18 FALSIFIED.** Its peak clause held (at most 29.6 GB), but the position fell outside [0.62, 0.78]. On box D's desktop CPU
  Unsloth steps in 3.0 s, against 3.7 s on box F and 4.1 s on amendment 6's host; e4b's step moves much less (5.5, 5.7, 5.9 s).
  Unsloth's Mixtral step is far more host-bound than e4b's.
- **P23 FALSIFIED** on Mixtral: both pairs read COMPARABLE, not EQUIVALENT (Unsloth 0.0025–0.0054 nats lower on held-out).
- **By the registered rule `E4B_ABSMAX_DQ` stays opt-in**, because P18 failed. Its own effect is what the rule set out to protect:
  e4b's held-out moved 0.0017 nats on the two-draw mean.
- **e4b's reference loop against its fused kernels on Mixtral is host-dependent.** On box D the reference (dequantize, then a dense
  GEMM per expert) ran 4.69 s against the fused 5.52. On box F it ran 6.76 against 5.69. TC1 amendment 22 reads grouped-nf4-gemm's dense
  route against the fused kernels on the full step.

### Box F: Qwen3.6 trains resident at micro-batch 1 on Unsloth's bytes

| arm | verdict | s/step | peak |
|---|---|---|---|
| e4b `fused_attn4_m` ×2 | OOM at step 6 | — | 32.79 GB |
| e4b `fused_attn4_m_mb1` | **VALID** | **9.241** | 31.36 GB |
| Unsloth `ckpt_unsloth_m` ×2 | VALID | 10.393 / 11.008 (5.7 % apart) | 30.47 GB |
| Unsloth `ckpt_unsloth_m_mb1` | VALID | 18.494 | 30.41 GB |

- **The first resident e4b run of Qwen3.6 on a 32 GB card**, at micro-batch 1, with the absmax double-quantized and the non-routed
  projections in NF4. `--frozen-4bit` is a measurement hook that matches the comparator's bytes, not a training option.
- At micro-batch 1 e4b steps about twice as fast as Unsloth on the same box. That is one draw each, so it is reported, not a quoted position.
- **P20 FALSIFIED** (micro-batch 2 OOMs). **P21 FALSIFIED**: the step-0 gap to Unsloth is 0.011–0.014 nats, down from 0.05 with
  the projections in bf16. **P22 UNTESTED**: there is no micro-batch-2 pair.
- **Box D** (the absmax alone): Qwen3.6 OOMs at micro-batch 2 (step 2) and at micro-batch 1 (step 18). **P19 FALSIFIED.**

## Predictions scored (box A)

P1 granite FALSIFIED; P2 olmoe FALSIFIED; P3 gptoss FALSIFIED; P6 (e4b parity on every family with a reference) HELD; P7 (matched sets
EQUIVALENT where two frameworks train one adapter set) FALSIFIED on the HF rows (COMPARABLE, inside 0.006 nats, outside the 0.005 band) and
held on the axolotl (granite) and Unsloth (olmoe) rows. P4 FALSIFIED on the Qwen3.6 re-run (above); P5 UNTESTED on the Mixtral re-run (Unsloth's draws unstable, above).

## Harness notes from this box

axolotl's scattermoe row (`ckpt_axolotl_best`) refused at load on every family: the KernelsPlugin fetches `kernels-community/rotary` and the
harness runs its arms with the Hub offline — a harness limit, open; its plain 4-bit arm ran on Granite. The `hf_peft_m_t214` rows ran in the
axolotl venv's torch 2.14 and transformers 5.17.0 and dispatched as recorded in each receipt.
