# Q235 Shape B: staged bytes per step, and two layouts that cut them

An analytic note, from records and code only: no runs, no GPU, and no change to any registered path. It sizes the lever that
`bench/q235/am7/PREDICTION-C.md` (e4b#1680) left parked. That note finds the PRO 6000 box staging-bound, with overlap capped at
about 11 % of the experts time. This one asks how many bytes a step stages, and what two other layouts would stage instead.

`staged_bytes.py` reproduces every number below from the box records:

    python staged_bytes.py BOX_RECEIPTS_DIR --h2d 21.76 --h2d 40 --h2d 53

## Inputs

| record | what it gives | sha256 |
|---|---|---|
| box 1 (`q235-pro6000-2`) `box/box.json` | 254,124,490,752 B in 187 copies per Shape A step: **1.358954 GB per whole-layer staging** | `27a7863e` |
| box 1 `box/nf4.pt` (in the store's `RELOCATED.sha256`) | the routers' top-8 of 128 for all 94 layers at Shape A's 8 × 331 positions | `b074b386` |
| box 1 `tokens.pt` | the attention mask: Shape B's micro-batches rebuilt as the box builds them | `d077cc63` |
| A2000 leg `f17f0744` `box/box.json` | a Shape B step at 4 layers stages 28 times = 4 micro-batches × (3 + 4) | `dd49de5d` |
| e4b#1680 | the PRO 6000's experts compute per micro-batch per layer: 15.13 ms | — |

**Staging count, read from the step.** In each micro-batch's forward, layer 0 is still resident from the previous backward
(the backward's recompute ends at layer 0, and `stage()` is idempotent), so the forward stages n − 1 layers. The recompute stages
all n. A Shape B pass therefore stages **M × (2n − 1)** whole layers: 4 × 187 = 748 at n = 94. The A2000 leg's 28 at n = 4 is that
formula, measured.

**The micro-batches.** Consecutive row pairs, each trimmed to its longer row: 134, 266, 662 and 158 positions, as `f17f0744`
recorded. Every position a micro-batch computes, padding inside its width included, goes through the router.

**Measured routing.** Uniform top-8-of-128 routing would hit essentially every expert at 134 positions or more (1 − (120/128)^134 =
0.9998). The measured routing is far more concentrated. Mean fraction of the 128 experts hit per layer, over the 94 layers:

| micro-batch (positions) | 134 | 266 | 662 | 158 | all four together (union) |
|---|---|---|---|---|---|
| experts hit, mean over layers | 0.694 | 0.768 | 0.820 | 0.653 | 0.898 |
| range over layers | 0.578–0.859 | 0.695–0.875 | 0.711–0.898 | 0.562–0.781 | 0.836–0.945 |

These come from Shape A's routing on the same positions. Shape B routes those positions with its own reduction order, and
Amendment 6's `route_flips` record measures how many flip; the fraction is a close proxy, not Shape B's own.

## Per pass, and per box

A Shape B box (q235_box.py at 13c05fce, no `--repeat`) runs **3 Shape A passes** (`l0`, `l1`, `ga1`) and **15 Shape B passes**:
rule 8's check at steps 0 and 1 (`separate`, `token`, `token` each), the two `ga_plants`, the two registered steps, and Amendment
7's bound timing (warm-up + 4). Staging is synchronous on the compute stream, so staging seconds add to box seconds one for one.
Box staging includes the Shape A passes (3 × 187 stagings) in every row. H2D: box 1's measured 21.76 GB/s (gen 4), and 40 / 53 GB/s
for the gen-5 host, whose H2D is unmeasured until its own host-rate phase runs.

| layout | stagings per pass | GB per pass | pass s @ 21.76: staging + experts compute (a lower bound) | box staging s @ 21.76 / 40 / 53 |
|---|---:|---:|---:|---:|
| **(a) as-is** | 748 | 1,016.5 | 46.7 + 5.7 = **52.4** | **735.7** / 400.2 / 302.1 |
| **(b) layer-major** | 187 | 254.1 | 11.7 + 5.7 = **17.4** | **210.2** / 114.4 / 86.3 |
| **(c) routed experts per micro-batch** | 548.8 | 745.8 | 34.3 + 5.7 = **40.0** | **549.1** / 298.7 / 225.4 |
| **(b) + (c)** | 167.9 | 228.2 | 10.5 + 5.7 = **16.2** | **192.3** / 104.6 / 79.0 |

- **The pass seconds are lower bounds.** They add only the experts compute (#1680's 15.13 ms covers the experts' forward, recompute
  and backward), not attention, the dense layers, the router, the head or the loss. Box 1's Shape A step spent 17.6 s outside
  staging (7.203 + 22.116 − 11.68).
- **(b) cuts the staged bytes 4×: 75 % fewer. (c) cuts them 27 %; together, 78 %.** Those are exact counts.
- **The step-time effect of (b) is bounded, not known.** With experts compute alone, a Shape B pass would run 3.0× faster at 21.76 GB/s
  (2.4× at 53); that is an upper bound. With Shape A's measured 17.6 s of non-staging time in place of 5.69 s, it would be 2.2× (1.6×
  at 53). That low end holds if Shape B's non-staging time is at most Shape A's, and the A2000 legs show it is. On the same card,
  Shape B's step spent 3.520 + 5.359 − 28 × 0.222 = 2.66 s outside staging (`f17f0744`), against Shape A's 1.443 + 6.226 −
  7 × 0.2178 = 6.14 s (`9f6e0088`). Shape B computes 1,220 positions against 2,648, which outweighs paying per-micro-batch fixed
  costs four times. So the true figure lies between the two bounds, and 2.2× is the conservative one. For comparison, #1680's
  overlap cap is about 11 % of the experts time.
- **On this one box the saving is small in money.** (b) saves 526 s of staging at 21.76 GB/s: about $0.29 at an illustrative
  $2.00/h, not a quote. The lever matters for training throughput on a staging-bound card, not for Q235's single registered box.
- **These are staging seconds.** The box's other phases (the load, the frozen hashes, the fetch) don't change.

## (a) As-is

Shape B, as registered (Amendment 6): four micro-batches in sequence, each a full forward and a checkpointed backward through all
94 layers. Each layer is staged 8 times per pass (twice per micro-batch), less layer 0's four forward stagings: 748 in all. Per
position computed, that is 8.7× Shape A's bytes: 1,016.5 GB for 1,220 positions, against 254.1 GB for 2,648.

## (b) Layer-major accumulation

Run all four micro-batches through a layer while its experts are resident: for each layer, for each micro-batch, the checkpointed
layer; then the head and loss per micro-batch; then one backward of the weighted sum of the four losses. Each layer is staged once in
the forward and once in the recompute: 187 per pass, as Shape A.

- **Activation memory.** The four micro-batches' checkpointed layer inputs are alive at once: 94 × 1,220 × 4,096 × 2 B = 0.94 GB,
  against 0.51 GB for the largest micro-batch alone, so **+0.43 GB**. At ≤ 662 positions per micro-batch the run takes the stock
  loss (T × V × 4 < 1 GiB), whose saved fp32 log-softmax is alive for all four until the backward: 1,220 × 151,936 × 4 B = 0.74 GB,
  against 0.40 GB, so **+0.34 GB**. Together about **+0.8 GB**. The per-layer working set is unchanged, because the micro-batches
  still run one at a time inside a layer.
- **Checkpointing.** Unchanged granularity: one non-reentrant checkpoint per (layer, micro-batch). **Assumption A1:** autograd
  runs the backward in decreasing sequence number, so the four micro-batches of layer L are recomputed back to back, and the
  single resident slot stages each layer once. **Tested: A1 holds, and (b) needs one more change** (see "A1, tested" below).
- **Gradient summation (a registered-result question).** As-is, each LoRA tensor's gradient is four sequential `.backward()` calls
  accumulating into `p.grad`, in micro-batch order 1 → 4. Layer-major is one backward: each tensor's four contributions meet in
  autograd's input buffer in arrival order, likely 4 → 1 under A1, in the gradient's dtype (bf16 for bf16 adapters). The math is the
  same weighted sum; the bits are not. Rule 8's deterministic check compares the registered device accumulation with the host-sum
  reference. Layer-major is a different accumulation, so rule 8 at the registered bar would have to be re-read for it. It also
  redesigns rule 8's `overwrite` plant, which takes a backward per micro-batch: layer-major has a single backward.
- **Cost to build.** Two pieces. First, a layer-major driver in e4b: about 200 lines that call the decoder layers directly instead
  of the model's forward. Second, an engine change: a hold flag in `engines/offload.py`'s forward evict post-hook, so a layer stays
  resident across its micro-batches (see "A1, tested"). The post-hook is shared by inference and every training lane, so the
  flag needs offload's own tests, not just the driver's. Then gradient parity with (a) within rule 8's bar, and the staging count
  (n per direction) on the A2000's offload path, at $0. No kernel change.

## (c) Routed experts per micro-batch

Stage only the experts a micro-batch routes to, at the experts module's pre-hook, where the router's top-k is already known
(`args[1]`). That is what e4b's inference routed staging does. The forward and the recompute each stage the micro-batch's routed
experts: 0.734 of the bytes on average, measured above.

- **What it computes.** The same experts, the same arithmetic: outputs are expected to be bit-identical to (a), as routed staging
  is in inference. That needs a parity test in training.
- **Cost to build.** Small in code: today's routed path refuses grad-enabled forwards on host-RAM homes (`offload.py`'s grad
  guard), and lifting it is the change. Two risks need tests. First, an expert the backward reads but the recompute did not stage
  would compute on a 0-element placeholder, so it needs a guard that refuses an unrouted expert read. Second, the DMA count: about
  94 experts × (1–2 arena copies) per staging, against 1–2 whole-layer copies. At about 10.6 MB per expert the copies should run
  near peak, but that is **unmeasured (an assumption)**.

## Registered results

**Nothing here changes a registered result, and neither layout may run in Q235's registered box** (Shape B as Amendment 6
registers it). (b) changes the accumulation procedure that rule 8 reads. (c) changes the staging procedure. Either would need its
own amendment, with a rehearsal, parity against (a), and rule 8 re-read at its bar, before any box used it. The numbers here are
for a later training lane's planning.

## A1, tested (CPU, `a1_layer_major.py`)

A toy model with e4b's single-slot staging rule, mirrored from `engines/offload.py`:
- a forward pre-hook stages the layer, idempotent if it is resident and otherwise evicting the resident one;
- a forward post-hook evicts it, except inside a backward (`torch._C._current_graph_task_id() != -1`);
- one non-reentrant checkpoint per (layer, micro-batch).

The toy has 94 layers and 4 unequal micro-batches. It ran on torch 2.11.0 (the box's) and 2.14.1, CPU only, with identical
results:

| schedule | copies per pass (toy) | on the box | backward layer-major | gradients vs as-is |
|---|---:|---:|---|---|
| as-is (Shape B as registered) | 749 | 748 | no (one micro-batch at a time) | — |
| layer-major, e4b's hooks unchanged | 470 | 469 | **yes** (each layer's 4 recomputes back to back, 93 → 0) | max rel. diff 1.2e-7 |
| layer-major, forward evict held across a layer's micro-batches | **187** | 186 | **yes** | max rel. diff 1.2e-7 |

The toy starts with nothing staged; the box enters each pass with layer 0 resident, so each box figure is one lower.

- **A1 holds.** Autograd's backward runs layer-major on its own: the recompute stages each layer once.
- **That alone is not enough.** e4b's forward post-hook evicts after every forward call outside a backward. In a layer-major forward,
  layer L is evicted after micro-batch 1 and re-staged for micro-batches 2–4. That costs 4 copies per layer in the forward: 470,
  only −37 % against as-is.
- **The full −75 % (187) needs the layer held** across its micro-batches in the forward: suspend the evict between them, then let
  the next layer's stage evict it. That is an engine change, a hold flag in `engines/offload.py`'s evict post-hook, which inference
  and every training lane share, so it needs offload's own tests. With the layer held, the backward also starts with layer 93
  resident, which saves one more copy.
- **The gradients are the same math.** The 1.2e-7 differences are fp32 summation order: the four contributions are summed in one
  backward rather than accumulated across four. The toy is fp32. The registered adapters are bf16, where summation order moves
  bits by about 1e-3 relative, so the toy shows the same math but does not read rule 8's bar. That needs (b) on the A2000 at bf16.
- **What it decides.** (b) is buildable: autograd needs no explicit per-layer backward. The build is the layer-major driver plus the
  engine's held forward. On CUDA, the A2000's staging count would confirm A1 too. The box figure, 186, is one under the table's 187
  (0.5 %), because the held forward leaves layer 93 resident for the backward; the table's 187 already counts layer 0 as resident.
  It still waits for Q235's read, and it still needs its own amendment before any registered box uses it.

