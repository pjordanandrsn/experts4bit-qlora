# DQ1 phase summary: is there a dense low-bit training primitive in this stack?

Written 2026-10-05 after DQ1's graded read ([RESULTS-dq1.md](RESULTS-dq1.md), run 2). It answers the brief behind #1083,
using that read and the research note ([NOTE-dense-qlora-primitive.md](NOTE-dense-qlora-primitive.md)) that preceded
it.

## The answer

**Not a speed primitive. Possibly a capacity primitive, not yet earned.**

| brief's success case | status after DQ1 |
|---|---|
| A. Strong dense win | **No.** The W4A16 speed line is closed at QLoRA training rows: H(2048) = 0.097, H(4096) = 0.050. |
| B. Useful Pareto point | **Open, on the capacity axis only.** Streaming frozen NF4 weights is a link question. On PCIe 4.0 x16 it hides behind the forward only above ~1,800 tokens per micro-batch (Rmin 1.07 at 2048, 1.93 at 4096). On PCIe 5.0 it scales to roughly 2× that ratio. Not registered as settled. |
| C. Dispatch result | **Already shipped upstream.** bitsandbytes 0.50 dispatches by arch and shape, and on sm_120 it took dequant + cuBLAS at every census row. gnf4 0.39.0's `auto` does the same at G=1. Choosing among routes inside the low-bit path buys nothing here: the routes are at parity. |
| D. Architectural result | **Yes.** The stack already holds what a dense route needs: gnf4's dense route (G=1) and e4b's bnb-based pieces. A new low-level package would duplicate bitsandbytes for speed. |
| E. Negative result | **Yes, for speed.** That is the defensible headline. |

## What each piece of evidence settles

**Speed.**
- A perfect bf16-math 4-bit kernel could remove at most ~10% of base-linear time at 2048 tokens, ~5% at 4096, ~3% at
  8192.
- That share is bnb's three dequant passes per linear, within ±10%. The dequant costs a fixed ~0.25 ms on the
  gate_up/down shapes, so its share falls as 1/M.
- A whole step gains less, because attention, norms and LoRA sit outside H.

**Small rows.** Below the registered rows the headroom is real: H(1024) = 0.19, H(512) = 0.30. Training at ≤ 1k tokens
a micro-batch (very large models on small cards) is the one regime where a fused dequant-GEMM could pay. Not
registered.

**G=1.** grouped-nf4-gemm at one group is a correct dense route at parity (1.002–1.010). Its packed kernel is 3.2–4.5×
slower than bnb on dense shapes: the expert-major design does not transfer.

**LoRA.** PEFT's unfused LoRA delta costs ~9–10% of base-linear time at every row ≥ 2048. That equals or exceeds the
whole dequant headroom. It is the bigger lever on the linears, and it is the one Unsloth already pulls.

**Capacity.**
- Concurrent DMA costs the GEMMs ≤ 2.5% and keeps its full bandwidth.
- The binding phase is the forward, and its ratio follows a model-size-free rule, R_fwd ≈ 2·M·B / (b·F). Break-even is
  M* ≈ 0.26·F/B: ~1,760 tokens on this PCIe 4.0 host, ~970 on PCIe 5.0.
- F must be the layer's measured rate, not a nominal one.

## The new-repository gate (brief, 11 points)

**Not met.** No reusable dense primitive with a measured win exists, so points 1 and 9 fail outright. Nothing is moved,
split or created. The investigation's code lives where it belongs: one bench lane in experts4bit-qlora.

If the capacity axis is later earned, the candidate home is grouped-nf4-gemm's host-transfer layer, not a new
repository. `host_gather` is already tensor-level and works for any pinned row stack.

## For Loggetta (no coupling added)

Two planner facts DQ1 makes available. Both are stated as data and do not import any backend:
1. **Speed.** For dense QLoRA at ≥ 2k tokens a micro-batch, every low-bit route lands within ~1% of the others on
   sm_120. Choose by memory and capacity, never by speed.
2. **Capacity.** A streamed frozen dense layer hides behind the forward when M ≥ 0.26·F/B. B is the measured pinned H2D
   rate under load; F is the layer's measured linear rate. Loggetta already records `link_h2d_gbps` in its observation
   schema, so the rule needs no new measurement plumbing.

Loggetta does not model dense models today; its provider is `describe_moe` only.

## Recommended next, in order. None is launched; each needs its own registration.

1. **DQ2: capacity on the right link.** The same census on a PCIe 5.0 x16 host, plus the real step's per-layer forward
   time (attention, norms and LoRA included) on Qwen3-32B. Cheap: about $0.10 for the census and a few dollars for a
   short profiled step.
   - **Stop** if Rmin(2048) < 1.25 on 5.0 as well.
   - **On success,** a streamed frozen-weight prototype under autograd is licensed, on a branch. Concretely, e4b's
     `dense_offload` already stages under grad, synchronously; the prototype would make it overlap.
2. **A fused-LoRA matched-work comparison** against Unsloth on a dense model. This is only a position-taking lane, not a
   primitive: the lever exists, and Unsloth ships it.
3. **Small-row headroom (M ≤ 1k):** only if a concrete workload needs it.

## Cost and record

| | |
|---|---|
| runs | `dq1-5090-1` $0.067 (NOISY; instrument diagnosed) and `dq1-5090-2` $0.102 (READ), receipt store `aae1719c`, `a737cf91` |
| rehearsals | three $0 RTX A2000 rehearsals through the local pool |
| reviews | two independent agent reviews (pre-run instrument; post-run read), both acted on |
| PRs and issue | #1084 (lane), #1103 (Amendment 1), this PR (results); issue #1083 |
