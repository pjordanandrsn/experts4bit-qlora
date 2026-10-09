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
