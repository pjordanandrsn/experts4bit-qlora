# Measurement audit — bundle olmoe-qlora-grid-20260705-1351

An external review pass recomputed every figure number from `runs/results/*` (nothing re-run) and
returned three re-readings the original harvest missed. Numbers verified; interpretation corrected
in four places; one headline unclaimed; one confound now has a control cell. This doc records the
corrections and reprioritizes the debts. It supersedes the "best eval" framings in
`OLMOE_EXPERTSNBIT_GRID.md` and `MODE_DECOUPLED_ADAPTERS.md` where they conflict.

## 1. The ∅ ladder and the yardstick G (was latent in the bundle, computed here)

The `grid_before_eval` rows ARE the preregistered no-adapter (∅) row — a single deterministic
pinned-eval pass (A5000): fp8 1.4724 < fp16 1.4780 < int8 1.4811 < bf16 1.4818 < nf4 1.4905 <
fp4 1.5041.

- **G_int8 = 0.0094** (nf4 − int8), **G_total = 0.0087** (nf4 − bf16) → **coverage = 108%**:
  frozen int8 is 0.0007 *better* than frozen bf16 on this eval. The precision axis above int8 is
  flat/inverted here. fp8 is the best frozen base (G_fp8 = 0.0181).
- **The top of the ladder (fp8/fp16/int8/bf16) is scrambled within ~0.01.** Per-example losses
  were not kept (debt D1), and the same-mode eval reproduces only to ~0.001–0.003 across code
  paths (measured from the bundle: seed 2027 reproduces exactly, others drift 0.0012–0.0030). So
  the ladder ordering above int8 — and the coverage=108% claim — cannot currently be
  distinguished from eval-path noise. D1 is load-bearing.
- fp16 < bf16 by 0.0038 on nominally-identical weights (bf16→fp16 in-range cast is exact) → that
  gap is activation precision, ~40% the size of the weight-precision G.

## 2. Portability decomposed against G (claim_usable, 3-seed means)

Effects as fractions of G (preregistered rule):

| quantity | Δ | in G | reading |
|---|---|---|---|
| upgrade nf4→int8 serve (nf4-trained) | +0.0003 | +0.03 G | adapter carries 3% of the serve upgrade; **97% forfeited** |
| upgrade nf4-offload→int8 serve | −0.0045 | −0.48 G | upgrade *hurts* |
| downgrade int8→nf4 serve (int8-trained) | +0.0075 | 0.80 G | costs ≈ the whole frozen gap |
| downgrade int8-offload→nf4 | +0.0061 | 0.65 G | same shape |
| **int8→int8 vs nf4→int8, resident (certified-comparable)** | +0.0007 | +0.07 G | **TIE — train precision buys nothing at resident placement** |
| int8-offload→int8 vs nf4-offload→int8 | −0.0065 | −0.69 G | exists ONLY in the offload-trained rows (see §3) |

Co-adaptation branch realized: train-cheap/serve-clean does not transfer here; **serve precision
is adapter provenance, train precision is not the dial** (the nutrition-label result). The
seed-0 25-cell delta-vs-∅ table shows the adapter effect is ≈constant (−0.468 ± 0.001) on the
nf4/fp4 serve columns regardless of train mode, and train-mode-dependent (spread 0.018) only on
the int8/bf16/fp16 columns — the co-adaptation signature as column structure.

## 3. The offload-trained "best" cells are confounded — bf16 is the control

Paired per-seed training deltas (best eval, 3 seeds):

| comparison | mean Δ | in G | sign consistency |
|---|---|---|---|
| int8-off − nf4-off (precision, under offload) | −0.0031 | −0.33 G | 3/3 |
| int8-res − nf4-res (precision, resident) | +0.0000 | 0.00 G | 1/3 |
| int8-off − int8-res (placement) | −0.0052 | −0.56 G | 3/3 |

The precision effect exists **only inside the offload placement** — a precision×placement
interaction the "offload changes location, not math" invariant forbids. The single-run grid's
offload−resident best-eval by mode makes it undeniable: **bf16 −0.0108, fp8 +0.0077 (sign
reversed)**. bf16 resident and offload weights are byte-identical — no quantization exists to
explain a 0.0108 gap; only RNG / data-order / gradient-checkpoint-recompute divergence can. So
**the offload training path is running a different experiment, not a different placement.**

Consequence: "int8-offload posts the best training eval" (grid doc) and "int8-offload→int8 is the
strongest cell" (portability doc) are **one uncertified mechanism counted twice**, and both are
**downgraded from `OLMoE-supported` to `candidate, confounded`** pending debt D3. The certified,
comparable statement is the resident tie in §2 (+0.07 G): at equal placement, int8-training buys
nothing over nf4-training on this task.

## 4. Unclaimed headline: the offload floor is cheapest at 16-bit (code-verified)

Offload training peaks collapse six modes to three, by storage byte width: **{bf16, fp16} 2.41 GB
< {nf4, fp4} 2.52 < {int8, fp8} 2.72** — the *unquantized* pair is cheapest. Verified from code
(`_vendor/experts.py::_dequantize_expert`): the 16-bit path returns the staged weight directly
(`packed.reshape(shape).to(dtype)`, a no-op cast when compute dtype = storage dtype), while the
4-/8-bit paths call `dequantize_4bit`/`dequantize_blockwise`, each materializing a full bf16
expert weight *on top of* the packed slab. Resident weight-bytes ≈ 1.0 (passthrough) / 1.25
(4-bit: packed + temp) / 1.5 (8-bit) — exactly the observed ordering. The dequant workspace, not
the packed slab, sets the offload floor.

So the campaign's memory thesis upgrades from "int8 quality at a 4-bit floor" to **"bf16 training
at 2.41 GB — below the nf4-offload floor, no quantization at all — vs 14.54 GB resident (6×)."**
bf16-offload's 1.0112 is the lowest training eval in the entire bundle. Caveats: single run,
`debug_only` gate, and its advantage over bf16-resident IS the §3 confound.

## 5. Corrections to the figure / docs

- **fp8 is absent from the dashboard** despite being the best frozen base (∅ 1.4724) and showing
  the reversed placement effect. It should appear.
- **Panels carry mixed evidence gates:** per `provenance_report.json`, all 15 train/decode jobs
  are `debug_only` (missing commit); all 24 seeded query jobs are `claim_usable`. The grid,
  eval, decode, and memory panels rest on `debug_only` rows; only portability is `claim_usable`.
- **Decode session noise is large and cross-session bars are not comparable:** single-run nf4
  10.12 vs repeat-5 12.68 tok/s (same config, different session); the grid's bf16 13.26 was a
  different session than the repeat-5 panel. Only within-session, within-panel decode comparisons
  hold.
- **Expert-streaming panel is train-batch only.** `hits = 992/992` everywhere is a train-batch
  artifact (every expert routed on every forward across a batch); it says nothing about
  decode-time temporal locality. The DO-NOT-BUILD verdict is correct **for train-phase static
  pinning** and must not be read as closing the decode/routed-stream question — whose traces were
  never captured (the decode-profile jobs wrote empty profiles; bug fixed in
  `scripts/decode_repeat.py`, which loaded the model but never attached the profiler).

## 6. Debts, reprioritized

1. **D3 — one-step train certificate**, with the **bf16 resident-vs-offload pair as the test
   article** (byte-identical weights → any best-eval gap is pure placement machinery: RNG, data
   order, recompute). Convicts or acquits §3; the highest-value run in the program. Every "best"
   claim in the bundle waits on it.
2. **D1 — per-example eval losses** for the ∅ ladder: the fp8/fp16/int8/bf16 scramble spans 0.009
   with SE unknown; coverage=108% and the fp8-best-base finding both need it.
3. **D2 — ∅ placement diff** (int8-resident vs int8-offload, no adapter): one grep-or-rerun;
   localizes §3 to the forward vs backward path.
4. **Decode routed-stream Phase 1** — now unblocked by the `decode_repeat.py` profiler-attach fix;
   batch-1 decode traces + Jaccard/recall telemetry, the question the train-batch panel cannot
   answer.

None of D1–D3 are run here (no pod; explicit-instruction-gated). The decode-profiler fix ships so
the next decode-profile run is non-empty.

## 7. Debt resolution addendum (2026-07-05, post-audit session — pointers only)

All three debts were run the same day under the post-audit work queue
(`docs/POST_AUDIT_WORK_QUEUE.md`); this section records outcomes, the details live in the cited
docs:

- **D3 PAID — §3 acquitted at one-step granularity, and then some.** Five trios (bf16+int8 ×
  default/deterministic kernels × dropout OFF/ON): null AND placement **bitwise-equal on every
  object** (losses, logits, grads, weights, optimizer state). The offload training path is not a
  different one-step experiment. The anomaly is run-level: T5 forensics found the repeat grid's
  placement pairs were ALL cross-architecture (4090↔A5000, evaluator offset 0.0026–0.0054 —
  same scale as the claimed effects), which explains the "3/3 seeds" cells; the same-host
  single-run bf16 0.0108 pair remains unexplained (scoped S10; divergence-onset probe gated).
  Quarantine on offload-trained precision claims HOLDS. `docs/TRAIN_PLACEMENT_CERTIFICATE.md`.
- **D2 PAID — placement is bitwise-exact at serve.** 384/384 per-example null evals identical
  resident-vs-offload across all six modes; eval determinism repeat 64/64 bitwise. §3 localizes
  strictly outside the forward path. `runs/results/postaudit/null_ladder_per_example.md`.
- **D1 PAID — and S9 FIRED.** Per-example paired SEs put G_int8 = +0.0094 ± 0.0076 (|t|=1.24)
  and G_total = +0.0088 ± 0.0080 (|t|=1.09) at n=64: the ladder's fine ordering, coverage=108%,
  and §1's scramble reading are all within sampling noise of this eval size (the pinned-set
  ordering itself is deterministic; the noise is distributional, not measurement). The
  preregistered n=1024 re-pin (`docs/NULL_LADDER_1024_AMENDMENT.md`) is the confirmation
  instrument; no G-denominated claim ships while S9 holds.
- §4's mechanism was **confirmed in form** (workspace exists iff quantized,
  precision-independent ≈0.64 GB; 16-bit floor code-backed) — `docs/OFFLOAD_MEMORY_FACTS.md`,
  which also nails the resident decomposition (fixed = 1.655 ± 0.001 GB + slab).

---

<!-- ots-attestation-footer -->

**OpenTimestamps anchor (self-attestation footer):**

- **OTS proof timestamp for visible document:** `2026-07-05T18:11:13Z` (the moment the current `.ots` was submitted to the calendars; this is the legally operative timestamp for the visible file as published).
- **Disclosed pre-footer content hash:** `de63a344f11ac6b95402e64682d0ed452c7ded31204a5ecb20eb99c059afca9b` (the SHA-256 of the document *before* this footer was appended — disclosed inside the OTS-anchored visible document for human-readable historical reference; this hash is *not* the payload of the current `.ots` file).
- **Prior disclosed pre-footer hashes (chain, newest first):**
  - `2026-07-05T16:48:26Z` `62fc92ea4193d118538cabd40758d8678c65172000955a45a9bb71dbb067005c`
- integrity-attestor glyph (`core.fingerprint`, first 8 bytes of the disclosed pre-footer hash): `[!?0~%~oo$::%&0@#]`
- Drunken-bishop randomart (full disclosed pre-footer SHA-256, OpenSSH-style):

```
+----[SHA256]-----+
|.o.+++B .o       |
|. ==+X++. +      |
|.o..o+=.o..o     |
| o oo. . *.      |
|  +.    S .      |
|. .    = =       |
| o      = =      |
|  o    . o o     |
| E      .        |
+-----------------+
```

- **Payload hash actually covered by the current `.ots`:** see `ots info MEASUREMENT_AUDIT.md.ots`; by construction this is `SHA-256(this entire file including this footer)` and `ots verify MEASUREMENT_AUDIT.md.ots MEASUREMENT_AUDIT.md` succeeds against the on-disk bytes.
- Anchor file: `MEASUREMENT_AUDIT.md.ots`
- Calendars: a.pool.opentimestamps.org, b.pool.opentimestamps.org, a.pool.eternitywall.com, ots.btc.catallaxy.com
- **Provenance posture (load-bearing):** the **OTS proof timestamp** above is the legal anchoring time for the visible document — that is what the calendars witnessed. The **disclosed pre-footer content hash** is *not* anchored by the current `.ots` file; it is *disclosed inside* the OTS-anchored visible document as a human-readable historical record of what the file's bytes hashed to immediately before this footer was appended. A reviewer verifying the visible file runs `ots verify` against the on-disk bytes; a reviewer wanting to confirm the disclosed pre-footer hash recomputes `SHA-256` of the file with everything from `<!-- ots-attestation-footer -->` onward stripped. Both checks are independent; neither replaces the other.
