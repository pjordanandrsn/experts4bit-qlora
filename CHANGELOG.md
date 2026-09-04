# Changelog

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
