# P125 results: calibrated int4 attention (and the int4 lm_head) on the shipped default's single-stream decode (#1313)

## The reading (`p125-5090-2`): **READ**. B and C are **NOT_LICENSED**

**Registration:**
- P125 itself: #1422, merged as `23c1e815`.
- Amendment 1 (#1428): the int4-head build fix #1426 and a 3.0 h guard.
- Amendment 2 (#1431): the gate record keeps the box's count.
- Amendment 3 (#1439): every gate's windows fit their corpus, with a preflight.

The maintainer re-derived the verdict from the receipt store (`774524d3`) with main's reducer and got a byte-identical
result.

**Code under test:**
- e4b 0.50.0 at `4934aba9`;
- grouped-nf4-gemm 0.43.0 at `6ee2e10`;
- torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, datasets 5.1.0;
- `Qwen/Qwen3-30B-A3B` at `ad44e77`, its NF4 arena baked on the box.

**Subject:** the shipped default `serve_paged` with every fusion and decode-GEMV knob unset:
- the B=1 fused stack, census 48 / 193 / [48, 48] / 48, `default-allowlisted`;
- the bandwidth GEMV (`bw_prmt32` ×288 at every build, no dot-pad);
- 16 slots.

The arms differ only in their int4 lever, calibrated on the box as a user's build runs it (32 × 512 tokens of C4
validation shard 0):

| arm | lever | int4 modules |
|---|---|---|
| A | none | 0 |
| B | `E4B_SERVE_ATTN_INT4_CALIB=1` | 96 (fused q/k/v + o, ×48) |
| C | B + `E4B_SERVE_LMHEAD_INT4_CALIB=1` | 97 |
| M | `E4B_SERVE_ATTN_INT4=1` (RTN) | 96 |
| K | B with every scale rolled one 32-block along K (the sure-fail mutant) | 96 |

### The gates

The bound is ln(1 + 0.05 / ppl_A), K8's +0.05 ppl budget in nats, computed from A's own windows. Argmax agreement must
be ≥ 0.95.
- **`t1`:** wikitext at one window per pass, the activation-quantised one-row `gemv_int4_b32` path, 108 windows (K: 12).
- **`k16`:** wikitext in 16-row pieces, K16, 112 windows (K: 16).
- **`c4`:** 16 c4val1 windows in 16-row pieces, reported only.

| arm | gate | windows | mean d (nats) | SD | SE | bound | argmax agreement | outcome |
|---|---|---|---|---|---|---|---|---|
| B | t1 | 108 | **−0.00317** | 0.0307 | 0.00296 | 0.00537 | **0.9294** | FAIL (argmax) |
| B | k16 | 112 | −0.00371 | 0.0295 | 0.00279 | 0.00541 | **0.9270** | FAIL (argmax) |
| C | t1 | 108 | +0.00090 | 0.0346 | 0.00333 | 0.00537 | **0.9180** | FAIL (argmax) |
| C | k16 | 112 | +0.00054 | 0.0309 | 0.00292 | 0.00541 | **0.9144** | FAIL (argmax) |
| M (RTN) | t1 | 108 | +0.00055 | 0.0411 | 0.00396 | 0.00537 | 0.9099 | FAIL (argmax), **as predicted** |
| M (RTN) | k16 | 112 | −0.00126 | 0.0412 | 0.00390 | 0.00541 | 0.9084 | FAIL (argmax) |
| K | t1 | 12 | +5.52 | 1.76 | 0.507 | 0.00575 | 0.147 | FAIL (decisive) |
| K | k16 | 16 | +5.88 | 1.86 | 0.466 | 0.00557 | 0.125 | FAIL (decisive) |

**The licences:**
- **B is NOT_LICENSED and C is NOT_LICENSED.** Both fail both ruled gates, on argmax.
- **The sure-fail mutant K failed both**, so the instrument is live.

**Engagement was exact in every arm.** At `t1`, `gemv:1` = windows × 127 × modules: B 1,316,736 = 108 × 127 × 96, and C
1,330,452 = 108 × 127 × 97. At `k16`, `k16:16` = 7 passes × 127 × modules: B 85,344, C 86,233. Every gate prefilled on
the bf16 copy, and nothing ran on the wide route.

### What decided it, plainly

1. **Argmax decided it, not the bias.**
   - About **7 %** (B) and **8 %** (C) of next-token argmaxes change against the default's.
   - Meanwhile the mean NLL bias sits **inside K8's budget in nats**: B's is −0.0032, an improvement that is not claimed.
   - **A K8-style ppl gate alone would have passed a lever that changes one top-1 token in fourteen.** On wikitext,
     the K8-style ppl reads B −0.029 and C +0.008, both inside +0.05.
2. **The bias half was UNDERPOWERED on its own.** This is a finding about the instrument.
   - B's per-window SE at `t1` was 0.00296, against bound / 3 = 0.00179; about **295 windows** would have been needed.
   - int4 *weights* spread the per-window difference wider (SD ≈ 0.031) than Phase D's *arithmetic-order* change
     (0.0196, the registered prior that sized 108 windows).
   - The argmax rung decides first, so no verdict depended on it. But a bias-only gate for a weight-changing lever needs
     its window count sized on a weight-change prior, not Phase D's.
3. **The speed is real, and this is what it costs.** Calibrated int4 attention decodes one request **1.154×** as fast,
   and **1.238×** with the int4 head, at the price of about one top-1 token in fourteen (twelve with the head) changing
   against the default. The registered gate does not license that trade. No looser gate is proposed here.

### RTN: the sensitivity check

The registration predicted RTN attention would **FAIL** at `t1`. It did, **but on argmax (0.910), not on bias**: its
mean d is +0.0005, inside the bound. On argmax, calibration helps: calibrated 0.929 > RTN 0.910. P55x's K8 failure of RTN
on c4val1 (+0.132 ppl) does not reproduce here as a bias: the K8-style c4val1 read is −0.016 over 16 windows, which is
noisy (SE 0.008).

### The K8-style ppl (reported, not ruled)

P115 Phase B's in-box method, exp(mean NLL) over the gates' windows. **Neither this method nor step_decomp's K8 runs the
one-row path for c4val1**, and the c4 read is 16 windows.

| | wikitext (t1, 108) ppl_A 9.2817 | c4val1 (16) ppl_A 17.0744 |
|---|---|---|
| B | 9.2523 (−0.029) | 17.1878 (**+0.113**) |
| C | 9.2900 (+0.008) | 17.3168 (**+0.242**) |
| M (RTN) | 9.2868 (+0.005) | 17.0584 (−0.016) |

### Speed (reported, not ruled)

Decode tok/s, P109's workloads, palindromic order (A1 B1 C1 C2 B2 A2):

| | A1 | B1 | C1 | C2 | B2 | A2 |
|---|---|---|---|---|---|---|
| W1 | 224.87 | 259.45 | 278.42 | 278.45 | 259.58 | 224.60 |
| W16 | 904.29 | 925.31 | 921.68 | 921.27 | 925.68 | 892.15 |

| | measured | predicted | |
|---|---|---|---|
| g1_B (the minimum over the pairs) | **1.1538** | 1.10–1.30 | held |
| g1_C | **1.2381** | 1.15–1.40 | held |
| g16_B | 1.0232 | 1.00–1.08 | held |
| g16_C | 1.0192 | 1.00–1.10 | held |

Self-pairs read 0.987–1.001, so the speed was not NOISY. Calibrated digests were equal in B1 = B2 and C1 = C2, so the
paired arms served the same weights.

### Memory, slots and the calibration

**Memory per arm:**

| | free before load | peak after build | peak, first prefill | peak, runs | auto slots (2048 / 4096 a slot) |
|---|---|---|---|---|---|
| A | 30.83 GiB | 22.56 GiB | 21.88 | 21.88 | 64 / 32 |
| B | 30.83 | 23.15 | 22.46 (+0.58) | 22.46 | 64 / 32 |
| C | 30.83 | **25.32** (the head's calibration) | 22.67 (+0.79) | 22.67 | 64 / 32 |

- **The bf16 copies are resident after the first prefill:** B 1.69 GiB over 96 modules, C 2.27 GiB with the head.
- **No auto-slot cost** at either slot size, as the arithmetic predicted.

**The calibration:**
- **Deterministic:** B1 = B2 = quality B and C1 = C2 = quality C, and C's attention bytes equal B's.
- **On this host:** 94.5 s (B) and 95.6 s (C) a build over A's.
- **Host-dependent:** 206 s and 225 s on `p125-5090-1`'s host; on Granite 49, 22–25 and 45–48 s across three hosts.
- The C4 fetch takes seconds once cached (shard 0 in 3.5 s).

### Against the predictions

- **g1_B, g1_C, g16_B, g16_C:** held.
- **Memory:** B +0.58 GiB held (0.4–0.8); C +0.79 GiB held (0.5–1.0).
- **Auto slots:** unchanged, held.
- **B `t1` PASS** (about 50 %): it FAILED, on argmax; the bias passed its bound.
- **B `k16` PASS** (about 65 %): it FAILED, on argmax.
- **B licensed** (about 40 %): not licensed.
- **C NOT_LICENSED** (about 75 %): held, on argmax rather than on the +0.0085-nat bias the registration expected from
  #373. C's bias read +0.0009.
- **RTN FAILS `t1`:** held, on argmax.
- **K FAILS both:** held.
- **UNDERPOWERED at `t1`** (about 25 %): the bias half alone would have read UNDERPOWERED; the argmax rung decided first.
- **Calibration 60–180 s a build:** 95 s on this host, 206–225 s on the first reading's host (over).
- **The verdict READ** (about 80 %): held.

### The reading (`p125-5090-2`)

**Host:** one RTX 5090 (sm_120, driver 595.91.07, **power limit 470 W**, 32607 MiB), Vast instance 54971795, 24-thread
host.

**Cost:** $0.968, 76 minutes from launch to teardown (05:35:19–06:51Z).

**Timeline:**
- the windows preflight passed (108 / 112 / 16 full) at 05:49Z;
- the six speed arms ran;
- quality A took 405.7 s at `t1` (3.8 s a window) + 53.8 s at `k16` + 7.7 s at c4;
- B, C, M and K followed.

## The history

| run | from | outcome | cost | what it found |
|---|---|---|---|---|
| `p125-prove-1` | `23c1e815` | VOID | $0.402 | the int4 head could not build the graph server (`enable_prefill_graph` read `.weight`), fixed in #1426 |
| `p125-prove-2` | `1e5bc75a` | VOID | $0.295 | the box merged `measure_phase`'s own `windows` over the gate's count (Amendment 2) |
| `p125-prove-3` | `e9e59b0f` | **PROVED** | $0.356 | the whole box on Granite |
| `p125-5090-1` | `e9e59b0f` | VOID | $1.067 | wikitext-2 test holds only 73 windows at P97's 4096 stride (Amendment 3) |
| `p125-5090-2` | `4934aba9` | **READ** | $0.968 | the reading |

**The proofs on Granite.** Granite's gates are instrument behaviour, not evidence about Qwen3. They were identical to the
digit on three boxes: calibrated attention d +0.037 nats with argmax 0.911, C +0.046, RTN +0.106, K +7.3. The
instrument ordered calibrated < RTN ≪ rolled, as on Qwen3.

**`p125-5090-1`'s speed arms** (history only, nothing decided): g1_B 1.153 and g1_C 1.224, on a 450 W board. They agree
with the reading's 1.154 and 1.238 within 1.2 %.

**Lane spend: $3.088** of the $5.50 ceiling.

## The consequence (as registered)

**No flip.** Neither lever licenses on the shipped default. The registered next single-request lever is the
launch-bound glue: P123 counted 635 small launches in 0.88 ms of the 4.75 ms step, in `moe_route` and `norm_elem`.

## Receipts

`bench/p125/receipts/<run>/` holds the runs' records and `logs/`, each with a `SHA256SUMS` that verifies:
- `arm_*.json`, `quality_*.json`;
- `verdict_p125.json`, `summary.txt`, `forensics.txt`, `versions.txt`, `prompts.json`, `bake.json`.

The private receipt store holds `receipt.json` and `teardown-proof.json` (`receipts/experts4bit-qlora/2026-10-09/p125-*`).
The reference log-probs stayed on the boxes.
