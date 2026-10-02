# P97 — Hybrid paged serving on the card: the paged runner's per-slot linear state against transformers' own cache, and its whole-model error, on Qwen3.6-35B-A3B on one RTX 5090 (registered 2026-10-02, before any run)

Issue: experts4bit-qlora#564. Code under test: #889 (per-slot Gated DeltaNet state) and #897 (compact fp8 pool,
`build_engine` wiring).

**Why.**
- **#889** taught `PagedModelRunner` to serve hybrid linear-attention models (Qwen3.5 / Qwen3.6 MoE, Qwen3-Next). Each
  Gated DeltaNet layer reads and writes a per-slot conv window and recurrent state through transformers' own
  `LinearAttentionLayer`.
- **#897** sized the fp8 KV pool to the attention layers only (Qwen3.6: 10 of 40, through
  `PagedAttentionContext.layer_map`) and wired `build_engine`.
- Both are tested on CPU only. There, a stand-in replaces the fp8 decode kernel (gnf4's Triton, which needs sm_89+).
  `docs/SERVING.md` says so: "no GPU run yet".
- **This lane** is that GPU run. It asks two things on the card, through the real kernel, at Qwen3.6's scale:
  - does each sequence's pooled linear state stay transformers' own state for that sequence?
  - is the whole model's error that of a working paged runner rather than a broken one?
- Speed is not asked. Decode graphs are refused for hybrids, so any decode time read here is eager and reported only.

Rule: the owner's standing no-ask tier for a single run under $15 (2026-09-26), with the usual mechanics:
- this page merged before the launch;
- a proving rental first, because the guard exceeds 1 h;
- receipts and ledger rows;
- proven teardown.

## The lane

- **Box and software.** One RTX 5090.
  - e4b at the launch commit.
  - grouped-nf4-gemm at `34da93d` (v0.34.1, e4b CI's pin: the `fp8_paged_attn` kernel).
  - transformers 5.17.0, the release #889 and #897 were built and tested on; bitsandbytes 0.50.2.
  - No `fla` or `causal_conv1d` is installed, so the Gated DeltaNet layers take transformers' torch path, the one the
    CPU tests ran. The record names which kernel modules were importable.
  - Every serving lever unset (e4b's defaults).
- **Models.** Each loads through `load_moe_4bit_streaming(..., quant_type="nf4")`, NF4 experts resident.
  - **Subject:** `Qwen/Qwen3.6-35B-A3B` at `995ad96` (qwen3_5_moe: 40 layers, 30 linear-attention, 10 attention, the
    first attention layer at index 3; kv heads 2, head_dim 256). Its linear-attention projections and shared experts
    stay bf16 under e4b (#899); both paths here read the same weights.
  - **Control:** `allenai/OLMoE-1B-7B-0924-Instruct` at `7f1c97f` (P96's revision; 16 attention layers). Its only
    paged error is the fp8 KV, read through the same kernel.
- **The measurement** (`p97_box.py`), per model, on the same in-process weights.
  - **Reference:** paged attention is registered but no paged context is bound, so the model runs transformers' forward
    with a `DynamicCache`, one window at a time.
  - **Paged:** `PagedModelRunner` with the fp8 pool sized by `kv_layers()`. Each window is bound to its own slot,
    prefilled in 128-token chunks, then the four decode together, one row each.
  - **Text:** wikitext-2-raw test, the K8 corpus. Window k starts at token k × 4096 (k = 0–3), each a 512-token prompt
    and a 256-token continuation, teacher-forced in both paths.
  - **Per step:** KL(reference ‖ paged) in nats over the full vocabulary, the true token's nll, and whether the argmax
    agrees. Log-probs are kept in fp32: bf16 rounding (~2e-3 at log-prob −0.5) is the size of the effect.
  - **The state, per window, after the last step:** each linear layer's pooled conv window and recurrent state against
    the reference's own `DynamicCache` state for that window (relative Frobenius error). The layers before the first
    attention layer (0, 1, 2) see the same tokens on both paths, so their state can differ only by bf16 arithmetic,
    whatever the fp8 KV does downstream. That is the model-independent quantity.
  - **The mutant pass (subject only).** A second paged pass in which each decode step writes a row's state back to the
    NEXT row's slot (rotated by one), a slot-mapping bug. Prefill (one row) is unaffected, so every window starts right
    and then reads another's state. It shows each gate can see broken state.
- **Engagement, counted.** The box counts every decode-kernel call with the pool layer it names, and every per-slot
  linear-state store with its layer.
- **The premise, on the card, before anything is fetched** (rc 25): `tests/test_linear_state_gpu.py` must PASS, not
  skip. On tiny models, through the real kernel, a hybrid with a compact pool must stay within 2× its all-attention
  control's worst relative logit error, with greedy tokens equal.
- **The order:** both checkpoints fetched (control, then subject), then the control's measurement, then the subject's.

**The reducer** (`p97_reduce.py`, 26-case self-test).
- **VOID** if any of these holds:
  - a record is missing, or ran a rehearsal knob (`--offload`, `--stand-in-attention`);
  - a record is not the registered shape (4 windows, 512 / 256 / 128), or loaded another model or commit;
  - a layer plan is not its model's. The subject must read 40 layers, 10 attention, 30 linear, on a 10-layer pool, with
    a state record for every linear layer and pre-attention layers 0, 1, 2. The control must read 16 attention layers,
    no linear state;
  - an engagement count is off:
    - decode-kernel calls ≠ 255 × attention layers (2,550 subject, 4,080 control), or not over pool layers 0..L−1;
    - linear-state stores ≠ 30 × (4 windows × 4 chunks + 255) = 8,130, or not over every linear layer;
  - the control's mean KL is zero (the paged path read the reference's own numbers);
  - the control fails G2 (the whole-model bound would sit below the fp8 KV's own error on an all-attention model);
  - the subject's mutant pass is missing, or passes G1 or G2 (that gate could not see broken state).
- **The gates, on the subject:**
  - **G1, the state:** every pre-attention linear layer's pooled conv window and recurrent state lies within **5e-2**
    relative error of transformers' own, for every window.
  - **G2, the whole model:** mean KL ≤ **0.05 nats** and argmax agreement ≥ **0.85**. The control must also pass it.
  - Both bounds were set from the rehearsal of the real model (below), between its working paged path and its
    rotation mutant: G1 about 8× above the one (6.0e-3) and 15× below the other (0.73); G2's KL ceiling 17× above the
    one (3.0e-3) and 59× below the other (2.95).
- **SUPPORTED** if G1 and G2 hold; **NOT_SUPPORTED** otherwise.
- **Reported, not gated:**
  - the subject's mean KL as a multiple of the control's;
  - each model's prefill-step KL (step 0: chunked prefill, no fp8 read), max KL, mean Δnll and max |Δnll|;
  - every linear layer's state error, including the layers after the first attention layer (their inputs carry the fp8
    KV's effect);
  - the eager decode time per step at 4 rows, the linear-state pool's size and peak GPU memory;
  - the mutant's numbers.

**The registered consequence.**
- **SUPPORTED:** `docs/SERVING.md`'s hybrid paragraph cites P97 as the GPU reading, and the next hybrid item, decode-graph
  capture of the state gather and scatter, starts from it.
- **NOT_SUPPORTED:** the hybrid path stays documented as CPU-tested only, and an issue records the reading before any
  change to it.
- **VOID:** nothing moves.

## Predictions (written before the data)

- **SUPPORTED.**
- **G1:** the pre-attention state error lands between 2e-3 and 2e-2 (rehearsal 6.0e-3, on 128 + 16 tokens). It grows
  from layer 0 to layer 2, as in the rehearsal (3.1e-3, 6.0e-3, 5.8e-3).
- **G2:** the subject's mean KL lands between 1e-3 and 1e-2, and both models agree on argmax at ≥ 0.93.
- **The subject's KL is 1–3× the control's** (rehearsal 1.78×). Its prefill-step KL is of the same order as its mean
  KL (rehearsal 5.2e-3 against 3.0e-3): Qwen3.6's chunked-prefill arithmetic, not the fp8 KV, sets its floor.
- **The mutant:** mean KL > 1 nat, agreement < 0.6, pre-attention state error > 0.3.
- **The engagement counts** match their expected values exactly. The linear-state pool holds 0.24–0.25 GiB for 4 slots
  (the rehearsal's 2 slots read 123.8 MB): 30 layers × (a 32,768-value conv window + a 32 × 128 × 128 recurrent state)
  per slot.

## Box and cost

- **`p97-prove-<n>`:** one RTX 5090, 0.5 h guard at ≤ $0.75/h (≤ $0.375). `P97_PROVE=1` runs:
  - the refusals;
  - the install with its tripwire;
  - the reducer's self-test;
  - the premise on the real kernel;
  - an HF CDN egress probe.

  It loads no model.
- **`p97-5090-<n>`:** one RTX 5090, after a passing proof, with ≥ 150 GB of disk (the runner refuses below 130 GB: a
  72 GB and a 14 GB checkpoint). **Guard 1.5 h at ≤ $0.75/h (≤ $1.125).** About 45 minutes:
  - install and premise ~6;
  - the fetch ~10–15 (86 GB);
  - the control ~5;
  - the subject's load ~10 and its three passes ~10.

  The runner stops starting steps 10 minutes before the deadline.
- **Lane ceiling $2.50; hard stop $3.50.**

## Rehearsal

Run on the NAS RTX A2000 (sm_86, 12 GB) from e4b `22629e4` (this branch's first commit), grouped-nf4-gemm at the pin,
with the local checkpoints. The knobs: `P97_MODEL_DIR`, `P97_PREMISE_ALLOW_SKIP=1`,
`P97_GPU_CLASS=A2000 P97_MIN_DISK_GB=20`, and
`P97_BOX_EXTRA="--offload --stand-in-attention --windows 2 --prompt 128 --cont 16 --chunk 64"`. The card has no native
e4m3, and 12 GB does not hold Qwen3.6 resident. Every knob marks the run REHEARSAL, and the reducer voided both
records, as it must. No time is quoted.

- **The proving run** (`P97_PROVE=1`) held, rc 0:
  - install and the tripwire: transformers 5.17.0, bitsandbytes 0.50.2, gnf4 0.34.1, no `fla` or `causal_conv1d`;
  - the reducer's self-test;
  - the premise skipped (sm_86), as the knob allows;
  - the HF CDN probe;
  - `PROVED`.
- **The full path** ran end to end, rc 0, on the real Qwen3.6-35B-A3B (experts offloaded) and OLMoE.
  - Engagement exact: the control made 240 / 240 kernel calls over 16 pool layers; the subject 150 / 150 over 10, with
    the compact map {3: 0, 7: 1, …, 39: 9} and 570 / 570 linear-state stores over 30 layers.
  - **The first rehearsal changed the rule.** As first drafted, the rule was the subject's mean KL ≤ 2 × max(the
    control's, 1e-3) and agreement ≥ the control's − 0.02. The rehearsal read the subject at 1.78× the control's KL
    (2.98e-3 against 1.68e-3), with agreement 0.9375 against a floor of 0.9488.
  - The subject's prefill step, which reads no fp8 at all, carried KL 5.2e-3, larger than its whole mean. A
    cross-model ratio therefore measures the two models' sensitivity to chunked-prefill and batched-decode
    arithmetic, and could fail a correct implementation.
  - The rule was replaced, before registration, by the state comparison (G1) and an absolute whole-model bound (G2).
    The mutant became the slot rotation: the first mutant only dropped write-backs, which leaves the first pass's
    correct final state in the pool, so a state gate could not have seen it.
- **The second rehearsal** (the kit as registered here, before the bounds were set) read:
  - G1's quantity: the pre-attention state error was 3.1e-3, 6.0e-3, 5.8e-3 at layers 0–2 (recurrent state; conv
    windows 3.3e-4–5.1e-3). The layers after the first attention layer read 8.7e-3–5.3e-2, carrying the fp8 KV's
    effect;
  - the subject: mean KL 2.98e-3, agreement 0.9375, prefill-step KL 5.2e-3. The control: mean KL 1.68e-3, agreement
    0.969;
  - the rotation mutant: mean KL 2.95 nats, agreement 0.34, pre-attention state error 0.73.
- **On CPU** (`tests/test_p97_box.py`, fp32), the same pre-attention state reads 1e-7, and the mutant 1.4. A pool that
  drops its write-backs reads 1.5.

Amendments, dated, go below this line before any data is read.
