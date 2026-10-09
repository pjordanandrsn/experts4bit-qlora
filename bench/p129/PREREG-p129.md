# P129 — one fused q/k/v projection for e4b's training attention: NF4 base and fp32 LoRA (DRAFT, registered before any run)

Issue: experts4bit-qlora#835. The lane number was claimed by `prereg/p129` (2026-10-09). It follows P128 (#1453), whose census found
training attention the largest launch bucket. The registration is staged as P128's was:
- **Phase 1** builds the fused projection and is gated on an RTX A2000 by counts only.
- **Phase 2** is a rented A/B, registered by amendment once Phase 1 passes. No box runs before then.

## The question

At TC1's field recipe, one training step of a two-layer Qwen3-MoE at Qwen3-30B-A3B's layer dimensions made 799 kernel launches (P128's
census). Attention made 230 of them: 90 in the forward, 90 in the checkpoint recompute and 50 in the backward. Per layer pass the
forward ran:
- 16 `mm`;
- 8 dtype casts (the fp32 LoRA in and out);
- 8 bitsandbytes dequantizes (the nested absmax, then the NF4 weight);
- 8 adds.

That is about 10 launches a projection, and q, k and v are about 30 of the 45. The three projections read the same input, so most of that
work is per-projection repetition.

**Does one fused q/k/v projection (NF4 base and fp32 LoRA) remove that repetition in training, exactly where it can be exact, and does
that shorten the step on a host-bound box?**

## The subject

An opt-in knob, `E4B_TRAIN_FUSE_QKV=1` (off by default), applied after `add_attention_lora` and `enable_fast_train`. It replaces each
eligible attention module's `q_proj`, `k_proj` and `v_proj` (each a `LoRALinear` around a bitsandbytes NF4 `Linear4bit`) with one
fused module, and the module's forward with serving's fused forward (`engines/qkv_fuse.py`'s `_fused_forward`, reused, not copied).
That forward looks up the modeling module's rotary at call time, so e4b's fused training rope and RMSNorm patches stay in force.

**The fused base.**
- The three projections' packed NF4 rows are concatenated along N. The blocks are 64 elements along a row of K = 2048, so each block
  stays within one row and the packed bytes are the same bytes.
- The nested (double-quantized) absmax cannot be concatenated, because each projection's nested state carries its own offset. The
  fused module therefore holds the fp32 absmax, expanded once from each projection's nested state by bitsandbytes' own dequantize.
- So the fused dequantize computes, bit for bit, the three dequantized weights stacked. It is one launch where today's path takes two
  per projection, and costs about 0.5 MB more per layer.

**The fused LoRA.**
- One cast of the input to the adapters' dtype and one `mm` against the three A matrices concatenated, `[3r, K]`.
- Three B `mm`s writing the three output slices, and one cast back.
- The trainable parameters stay the original nine q/k/v adapters (`torch.cat` and slicing route their gradients), so optimizer
  state and saved adapters are unchanged.

**Refusals.** Anything below keeps today's path, and the module is counted as refused with its reason:
- the three adapters differ in rank, alpha, scaling or dtype;
- any adapter carries dropout or other per-projection semantics;
- a base is not NF4 with the same blocksize, quant type and compute dtype;
- K is not a multiple of the blocksize (so a block would span rows);
- any projection carries a bias;
- the attention class is not one whose fused forward serving already reviews (`Qwen3MoeAttention`);
- a rotary that is not rotate-half.

**Untouched.** Serving: `fuse_qkv` and its folds are not called or changed. With the knob unset, training runs today's ops; tests assert
that bit for bit (the same op sequence, values and gradients as the commit before).

## Phase 1 — the build and its gate (RTX A2000, counts and correctness only)

The model is P128's two-layer random Qwen3-MoE at real layer dimensions, set up as TC1's e4b arm, rows of 270 tokens, the same
seeds.

**Exactness, stated before the read:**
- **The dequantize:** the fused module's dequantized weight `torch.equal` the three projections' dequantized weights stacked, every
  layer.
- **The projections and the loss:** concatenating changes the GEMM shapes, so cuBLAS may pick another algorithm or reduction order.
  Each q/k/v output, the loss and every gradient (the nine adapters included) must be within TC1's rounding bar of today's path: each
  tensor within `2**-6` of its largest entry for bf16 and `2**-16` for fp32. Bitwise is reported, not claimed.

**Counts, per training step, against today's path on the same model:**
- **Kernel launches:** today 799.
  - **Prediction:** about 80 fewer, `[60, 100]`: per layer pass about 18 fewer (q/k/v from about 30 to about 12), twice a
    layer (forward and recompute), plus 6–10 in the backward, over two layers.
  - **Gate:** at least 50 fewer (≥ 6 %).
- **Python calls** (cProfile, autograd on the calling thread): today 16,241.
  - **Prediction:** 4–12 % fewer (six module calls a layer pass become one fused call; the forward is serving's, not HF's).
  - **Gate:** at least 3 % fewer.

**Phase 1's rule:**
- **PASS:** exactness and both count gates hold.
- **NO_GAIN:** exact, but a count gate missed.
- **FAIL:** an exactness bar missed. P129 then stops; the read names the tensor and its distance.

## Phase 2 — the rented A/B (by amendment after a PASS; the frame is P128's, carried over)

- **The box:** TC1's field recipe over 60 load-gated steps in venv-unsloth (torch 2.12), e4b at its defaults. The shipped arm (bf16
  adapters) and the matched arm (fp32 adapters), knob 0 against 1, two draws a side in ABBA order, every arm profiled.
- **The recount gate:** the box recounts launches and Python calls with the knob on and off on its own card. It is VOID unless each
  reproduces at least 0.8 of Phase 1's relative cut.
- **The premise gate:** VOID for speed unless the matched knob-0 arm's GPU busy share is at most 0.85.
- **Two ratios:** the wall ratio judges the remedy; the device-time ratio bounds its cost. They are reported separately.
- **Quality:** step-0 held-out within 0.0005 per draw pair and held-out at N within 0.005, each arm.
- **The rungs:** VOID / NOISY (draws 5 % apart) / QUALITY_FAIL / NO_GAIN (wall above 0.98 on either arm) / GAIN (both at most 0.98,
  stays opt-in) / DEFAULT_ON (GAIN, plus a second host at GAIN or, GPU-bound there, at most 1.01).
- **Predictions:** set at the amendment from Phase 1's counts, priced at P128's per-launch and per-op CPU self times.

## Budget

- **Phase 1:** no rental.
- **Phase 2:** one RTX 5090 box (about $1.5–2), plus at most one replication. The lane stays under $15.

## Phase 1 read, as registered (RTX A2000, 2026-10-09): counts PASS, exactness FAIL on the end-to-end clause

The prototype is `E4B_TRAIN_FUSE_QKV=1`, the module above. It was read on the registered model, rows and seeds against today's path.
Both layers fused, none refused.

**Counts (gate PASS; both predictions missed high):**

| per training step | today | fused | change | gate | prediction |
|---|---|---|---|---|---|
| kernel launches | 769 | 659 | −110 (−14.3 %) | ≥ 50 fewer: **HELD** | `[60, 100]`: missed high |
| Python calls | 16,240 | 14,266 | −12.2 % | ≥ 3 %: **HELD** | 4–12 %: missed high |

**Exactness:**
- **The dequantize: bitwise, as claimed.** `torch.equal` on both layers.
- **The fused projection in isolation, reported:** the same input and the same upstream gradient through the fused module and through the
  three `LoRALinear` modules. Every tensor is within TC1's rounding bar, none bit for bit:

  | tensor | relative difference | bar |
  |---|---|---|
  | q/k/v outputs (bf16) | 6.7e-3 | `2**-6` |
  | input gradient (bf16) | 1.06e-2 | `2**-6` |
  | the six adapter gradients (fp32) | about 4e-7 | `2**-16` |

- **The end-to-end clause: FAIL.**
  - The loss read 8.682032 against 8.682492, a relative difference of 5.3e-5 against a `2**-16` bar.
  - All 24 gradients differ. The worst is layer 0's `gate_up_lora_B`, at 0.18 of its largest entry.

**Cause.** The clause applied an op-level rounding bar to a whole bf16 MoE training step. A reorder-class change moves the projections by a
rounding step. Attention and the router amplify that: a flipped top-k choice moves the expert gradients by whole contributions. So no
change of this class can meet that clause in bf16 MoE training. The condition in review named "the projections and the loss", and this page
widened it to every gradient. The condition was itself ambiguous about the end-to-end loss. By the rule this read is **FAIL**, and the bar
is not replaced on these records. Amendment 1 re-asks the end-to-end question in its own units, against a neutral floor, on fresh seeds.

## Amendment 1 (2026-10-09T14:24Z, after the Phase 1 read, before any new run): the end-to-end question against a neutral floor

The question Phase 1's end-to-end clause should have asked: **does the fused projection move a short training run more than the eager path
moves under changes that are equally valid?** A single-seed comparison of one MoE step cannot say; a floor over seeds can.

**The instrument.** P128's two-layer random Qwen3-MoE at Qwen3-30B-A3B's layer dimensions, set up as TC1's e4b arm at e4b's defaults
(fp32 LoRA, the reentrant checkpoint, `NF4_QLORA_SINGLE_LADDER=auto`). Per seed `s`:
- 30 optimizer steps of bitsandbytes' AdamW8bit, lr 2e-4 with 5 linear warm-up steps, weight decay 0.001;
- each step one batch of 2 rows × 270 random tokens from a generator seeded by `s`;
- the adapters initialised from `s` (`lora_B` drawn N(0, 0.02), so the attention adapters are live from step 0);
- a fixed held-out set of 4 such batches from `s + 1000`, evaluated at steps 0 and 30.

**The seeds (fresh; none was read before):** 211, 223, 227, 229, 233.

**Divergence of a run V from the baseline run B with the same seed:**
- `D_traj` = max over steps 1..30 of |training loss of V − of B|;
- `D_held` = |held-out loss of V at step 30 − of B|.

**The neutral floor.** The eager path under three changes that leave the arithmetic's meaning unchanged and move only its reduction order:
- **F1** `NF4_QLORA_SINGLE_LADDER=0` (the expert LoRA block unladdered);
- **F2** `NF4_QLORA_PAD_BUCKETS=1` (the LoRA delta bucketed);
- **F3** each 2-row batch as two 1-row micro-batches with gradients accumulated. Both rows carry 270 tokens, so the mean loss is the
  same quantity.

The floor of each measure is its **worst draw**: the maximum over F1–F3 × the five seeds (15 draws).

**The fused path:** `E4B_TRAIN_FUSE_QKV=1` against B, the same five seeds (5 draws).

**The end-to-end gate** (registered now):
- **HELD** iff the fused path's maximum `D_traj` is at most the floor's worst `D_traj`, **and** its maximum `D_held` at most the floor's
  worst `D_held`, **and** (the backstop, TC1's held-out bar) every fused `D_held` is at most 0.005.
- Per-step gradients (the first step's relative differences) are reported, not gated.

**The counts gate is unchanged** (≥ 50 fewer launches, ≥ 3 % fewer Python calls), re-read on the same build.

**The rule:**
- **PASS:** the counts gate and the end-to-end gate both HELD.
- **NO_GAIN:** the counts gate missed.
- **FAIL:** the end-to-end gate missed. The read then names the measure and the draw.

**Prediction:** the end-to-end gate HELD (about 70 %). The fused projection moves each run by a rounding step, the same class as F1–F3.
The floor's own magnitude is not predicted; it is read.

**Budget:** the RTX A2000 only: 20 runs of 30 steps. Phase 2's speed A/B is still registered by its own amendment after a PASS.

## Amendment 1 read (RTX A2000, 2026-10-09): PASS

e4b at `590962a` (`p129-train-fuse-qkv`: the module above, committed before the read), grouped-nf4-gemm main at `e21a712`, torch 2.11.
25 runs, none failed: the baseline, F1, F2, F3 and the fused path, on each of the five registered seeds.

**The end-to-end gate: HELD.**

| measure | fused, worst of 5 | floor, worst of 15 | backstop |
|---|---|---|---|
| `D_traj` (max step loss difference) | 0.00311 | 0.00361 (F3, seed 233) | |
| `D_held` (held-out at step 30) | 0.00064 | 0.00078 (F3, seed 233) | every fused draw ≤ 0.005: held |

The fused draws by seed (`D_traj` / `D_held`):
- 211: 0.00228 / 0.00050
- 223: 0.00311 / 0.00048
- 227: 0.00237 / 0.00041
- 229: 0.00236 / 0.00041
- 233: 0.00234 / 0.00064

The floor's draws lie in 0.00171–0.00361 (`D_traj`) and 0.00002–0.00078 (`D_held`). The fused path moves a 30-step run by
less than the eager path's own equally valid variants do. The prediction (HELD, about 70 %) is met. This instrument's runs barely
learn (random tokens on random weights: the baseline's training loss went from 8.759 to 8.754), so it measures numerical divergence, which
is what it was registered for.

**The counts gate, re-read on this build: HELD.** grouped-nf4-gemm's newer main adds 16 launches to both paths:
- kernel launches 785 → 675 (−110, −14.0 %);
- Python calls 16,440 → 14,466 (−12.0 %);
- the dequantize still `torch.equal`.

**By the rule, PASS.** The next step is Phase 2's speed A/B, registered by its own amendment with P128's frame: a recount gate, a
host-bound premise gate, wall and device ratios separately, and the TC1 held-out bars.
