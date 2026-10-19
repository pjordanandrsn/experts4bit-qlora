# DQ11 MLP-adapter BUILD — amendment 1: Phase 1's adapter state, and where the engine patches (#1083)

Prospective, before any Phase 1 data. It is filed under `DQ11-MLP-BUILD-PREREG.md` (#1802), which stays as merged
except for what is stated here. The maintainer ruled each item on 2026-10-19, during the review of the engine
(`experts4bit_qlora/engines/dense_lora_fuse.py`, #1809). This amendment authorizes no rental.

## D2's text: the MLP module's forward

D2 says the engine patches "the PEFT `LoraLayer` forward of the MLP modules". The engine, like its template
`train_qkv_fuse`, patches the **MLP module's** forward instead, because the shared gate/up rank GEMM needs both
projections in one call. The maintainer ruled this within D2, on three conditions:
- the LoRA modules, their parameters and their names stay untouched;
- the patched forward reproduces the class forward exactly, `down(act_fn(gate(x)) * up(x))`, at the pinned
  `transformers` (only `MistralMLP` is taken);
- the refusals catch every non-stock MLP: another class, a forward already patched, or a hook on a projection.

## Phase 1's adapter state (ruled)

**Why.** PEFT starts every B at 0. At that state the A gradients and the adapter part of each input gradient are exactly
0 on both paths, so the bar would test nothing there. Two AdamW updates are not enough either: at lr 2e-4, B is then
about 4e-4 per element, and the adapter share of dx is so small that a fusion bug in that term could hide under dx's
own floor. Non-zero is not enough: every checked tensor's floor must be read where the adapters visibly move it.

**The inputs.** For each checked layer (0, 15, 31), the harness captures two tensors from the fixture's first Alpaca
batch at seq 2048, run once through the frozen model:
- `x`, the MLP's input;
- `gy`, the training loss's gradient at the MLP's output.

All adapters are at PEFT's initialization during the capture (B = 0, so the model is the base model). The path is
unfused, with the knob off. Every per-module read (unfused, the 31 redraws, fused) replays the same `x` and `gy`.

**The adapters for the reads:**
- **A:** PEFT's own initialization under the fixture's seed, unchanged.
- **B, for the three MLP adapters of each checked layer:** `torch.Generator("cpu").manual_seed(3407)`, then
  `randn(shape) * σ` in FP32. They are drawn in `named_parameters` order (layers 0, 15, 31; gate, up, down within each)
  and then moved to the device.
- **σ** is the smallest of **0.01, 0.02, 0.05, 0.1** at which every checked tensor's adapter share (below) is at least
  **1 %** at all three layers. The shares are measured on the unfused path only, before any redraw or fused read, so the
  choice cannot be tuned toward the fused result.
- If no σ on that list qualifies, the read is **`VOID_LORA_SHARE`**.

**The adapter share of each checked tensor**, in FP32. Every share is recorded for every σ read, the chosen one marked:
- each projection's forward output: ‖s·(x·Aᵀ)·Bᵀ‖ / ‖output‖;
- the shared MLP input's gradient: ‖Σ over gate and up of (s·g·B)·A‖ / ‖dx‖;
- `down_proj`'s input's gradient: ‖(s·g·B_down)·A_down‖ / ‖dh‖;
- in both, `g` is each projection's **own** output gradient: `gate_proj`'s and `up_proj`'s for dx, `down_proj`'s for
  dh. It is not `gy`;
- each A and B gradient: these are wholly adapter terms, so the share is 1 by construction. Such a tensor is
  `VOID_LORA_SHARE` instead if it is all zero or non-finite.

The floor, the bar and `PHASE1_PASS` are read at this state only.

**What a PASS excludes (required).** At a 1 % share, a bug in one of the r = 16 rank components moves a tensor by about
1 % / 16 ≈ 0.06 %, which may be under a bf16 floor. So, for every checked tensor, the read records:
- its adapter-term norm (the numerator of its share);
- its floor's relative RMS times ‖tensor‖, divided by that norm. That is the smallest error, as a fraction of the adapter
  term, that the bar can see in that tensor. Call it `e_min`.

The read states `e_min` for every tensor, together with the largest one. A rank-1 error is about 1/r = 1/16 of the
adapter term. Wherever `e_min` > 1/16, a rank-1 error could hide under that tensor's floor. The outcome line then names
every such tensor: `PHASE1_PASS, RANK1_UNRESOLVED: <tensors>`. It is never a silent `PHASE1_PASS`. A wholly-adapter
tensor (each A and B gradient) has a share of 1, so its `e_min` is its floor's relative RMS.

## The two-AdamW-step state (reported, never ruled)

As realism context, the harness also reads from a second state:
- PEFT's initialization (B = 0 on all seven targets);
- then two updates of the L path, unfused, with AdamW at lr 2e-4, seed 3407, on the fixture's first two batches in
  order;
- the adapters are then snapshotted.

`x` and `gy` are captured again at that snapshot on the fixture's first batch. The unfused path, the 31 redraws and
the fused path all read the same snapshot. Their shares, floors and outcome are reported, labeled REPORTED. They never
enter `PHASE1_PASS`.

## The reference for the redraws, and R5's order

The redraws R1–R5 are computed by a reference for the unfused path. It repeats PEFT 0.19.1's `Linear4bit` LoRA forward
op for op:
- the base output, cloned;
- the input cast to FP32;
- `lora_A`, then `lora_B`, then the scale;
- the delta cast to the base dtype and added.

It changes only what each redraw names. The shared input's gradient is the sum of four terms: gate's base, gate's LoRA,
up's base and up's LoRA. The reference sums them in an explicit order.

**R5's order.** PEFT's own order was read on 2026-10-19 by a $0 A2000 instrument probe. The probe used a 256-wide random
MLP with PEFT LoRA on bnb NF4 bases, with no fixture and no fused read. Autograd accumulates the four terms left to
right as up's base, up's LoRA, gate's LoRA, then gate's base. That held on 6 of 6 seeds and matched the same probe on
CPU. In the same probe, the reference's forward and every adapter gradient were bit-identical to PEFT.
- **R5** therefore puts gate's LoRA term before up's: up's base, gate's LoRA, up's LoRA, gate's base. The registration's
  "(up first)" assumed gate was natively first; the redraw is the other order of the two LoRA terms, as registered.
- **The reference's self-check.** Before any floor is read, the reference with no redraw must equal PEFT bit for bit on
  every checked tensor, at every checked layer and in each state read. Anything else is the named failure
  `REF_MISMATCH`, and no floor or bar is read.

## Engagement and the counters

- Before any read, the harness asserts that `enable_dense_mlp_lora_fuse` returns **96** on the real fixture, with
  `DENSE_MLP_LORA_STATS["refused"]` empty. Anything else is the named failure `ENGAGE_COUNT`. An accelerate-dispatched
  MLP, for example, carries an instance forward and is refused.
- `DENSE_MLP_LORA_STATS["calls"]` counts every fused MLP forward, **checkpoint recomputes included**. Under
  non-reentrant checkpointing that is about twice a step's MLP forwards. Phase 1 reads it with that meaning. The
  registered witness stays "zero PEFT LoRA MLP forwards in a step", counted by hooks on the LoRA modules.
- If the engine at the harness's head counts fallbacks (`DENSE_MLP_LORA_STATS["fallback"]`, calls that ran the class's
  own forward), every fused read and every fused step requires it to be 0.
