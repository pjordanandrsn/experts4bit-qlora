# P88 read — LICENSED: K19 takes the RTX 5090's B=16 int4 decode step to 0.905×, with K8 inside the floor (+0.0062 nats); B=1 is 1.10× slower

Registered in [`PREREG-p88.md`](PREREG-p88.md) (#808, `b848089`). Issue: experts4bit-qlora#564.

```
P88_VERDICT LICENSED B=16 10.140 vs 11.200 ms (x0.905); K8 +0.00623 nats; B=1 SLOWER (x1.103)
```

**The setup:**
- **Box:** one RTX 5090 (sm_120, driver 590.48, 600 W) on an AMD EPYC 9334 host, Vast machine 41531.
- **Code:** e4b at the registration's merge, `b848089`. grouped-nf4-gemm `7b7e6b1`, where K19 defaults to K20's plan, BLOCK_N 32 / KC 256.
- **Opt-in:** `E4B_INT4_GROUPED_SMALLM` on against off.

**Every check passed before the rule was applied:**
- the premise on the card (3 passed);
- engagement at both B: 96 K19 calls per step ON, the experts' 96 GEMV calls gone, none OFF;
- the draw spread;
- K8 window and steps;
- K8 ON differs from OFF.

## Runs and cost

| run | outcome | cost (ledger) |
|---|---|---|
| `p88-prove-1` | **PROVED**. RTX 5090, AMD Ryzen 9 7950X. Tripwire OK, including K19's default plan; premise 3 passed; K19 + K16 contracts compiled, 22 passed, including plan bit-identity | $0.0406 |
| `p88-5090-1` | **NOT_RUN** at the launcher's pre-flight: ssh never authenticated (machine 34181). Now excluded | $0.0308 |
| `p88-5090-2` | **NOT_RUN** at pre-flight: download bandwidth 24.8 MB/s < 40 (machine 147733) | $0.0148 |
| `p88-5090-3` | **NOT_RUN** at pre-flight: 25.1 MB/s, the same machine. Bandwidth NOT_RUNs are not exclusion evidence, so the launcher rebought it. The next launch waited for the market | $0.0182 |
| `p88-5090-4` | **the reading: LICENSED.** Destroyed 09:26:54Z, absent | $0.5538 |
| **total** | | **$0.6582** of the $3.50 ceiling |

## Speed (graph-replay window, `step_ms_clean`)

| arm | draw 1 | draw 2 | median | ON / OFF |
|---|---:|---:|---:|---:|
| B=16 OFF (int8 GEMV) | 11.2050 | 11.1942 | **11.200** | |
| B=16 ON (K19) | 10.1425 | 10.1382 | **10.140** | **0.905** (bar 0.95) |
| B=1 OFF | 4.3263 | 4.3236 | **4.325** | |
| B=1 ON | 4.7697 | 4.7715 | **4.771** | **1.103** (SLOWER) |

This host is faster than P87's Broadwell host: the OFF B=16 step reads 11.20 against 12.01 there.

## Quality (K8, the licensed recipe, wikitext window `9ef10d760ad9`, 2,048 steps)

| reading | mean NLL |
|---|---|
| build (OFF; calibrates the experts and dumps the pack) | 1.8434202801176407 |
| OFF (pack loaded by fingerprint) | 1.8434202801176407, **bit-equal to the build** |
| ON (K19) | 1.8496547109991661 |
| **ΔK8** | **+0.00623 nats** (floor 0.0095) |

- **The build dumped the licensed expert pack.** Its fingerprint is `sha256:0c9955a9…`. So this host's calibration reproduced it, and OFF and ON read the licensed experts' bytes.
- **OFF does not reproduce P85's licensed float** (1.85114). The licensed float was read under P70's environment, including `E4B_ROUTER_EPI_CAST=0` and e4b `c77aab6`; P88 runs the launch commit's defaults. That is reported, not gated; the unmeasured cast pair is open in STATUS. ON and OFF differ only in the opt-in.

## Where the time went (P42 replay census, ms per step)

**B=16:**

| kernel | OFF | ON | Δ |
|---|---:|---:|---:|
| `_gemv_int4_b32` (experts) | 6.338 | — | −6.338 |
| `_gemm_int4_b32_grouped_smallm` (K19) | — | 5.050 | +5.050 |
| `_quant_x_rows` | 0.306 | — | −0.306 |
| `_reduce_partials` | 0.289 | — | −0.289 |
| `_tile_table_r1` (tile build) | — | 0.430 | +0.430 |
| gather / scatter / elementwise glue | 0.099 | 0.355 | +0.256 |
| everything else | 3.995 | 3.945 | −0.050 |
| **kernel total** | **11.027** | **9.780** | **−1.247** |

- K19 is 1.26× the GEMV kernel for kernel here. K20 measured 1.36× for the whole route on recorded routing.
- In the model, K19's own time (5.05) sits above K20's microbench, where K19 plus the tile build was 5.20. The grouping glue costs about 0.69 ms per step (tile build 0.43 + glue 0.26), against 0.60 of quantise and reduce removed.
- **B=1:** K19 takes 1.139 ms against the experts' 0.848 share of the GEMV. With the tile build and glue, the step is 0.40 ms slower.

## The registered predictions

| prediction | read |
|---|---|
| B=16 ON/OFF 0.78–0.86 | **0.905 — refuted, though LICENSED.** The grouping glue (0.69 ms) and K19's in-model time are larger than K20's route measurement carried |
| B=1 SLOWER, 1.03–1.20 | **held: 1.103** |
| \|ΔK8\| < 0.003 nats | **refuted: +0.00623**, two thirds of the floor. It passes the registered gate. The activations K19 multiplies are bf16 where the GEMV's are int8. P64 read that change INDISTINGUISHABLE at T == 1 on the dequant path; here it is measurable, though inside the floor |
| the build equals OFF to the bit | **held** |
| OFF reproduces P87's steps | not on this host (11.20 vs 12.01). Reported, not gated |

## What follows (the registered LICENSED branch, B=1 SLOWER)

1. **A default PR:** `E4B_INT4_GROUPED_SMALLM` becomes auto, with K19 on for int4 decode rows above T == 1 when the installed kernel package has it. **T == 1 stays on the GEMV**, because B=1 is slower. `=0` keeps the GEMV everywhere; `=1` keeps the opt-in's current meaning, T == 1 included.
2. **Register:** `e4b.serve.p88.qwen3.int4.k19-b16.5090.2026-10-01` (this PR).
3. **A gnf4 release** carrying K19 and K20's plan.
4. **The next lever on this row is the grouping glue:** tile build 0.43 + gather/scatter 0.26 ms per step at B=16, about 7 % of the ON step. Folding the tile build into the router epilogue and the unsort into the combine is a kernel lane of its own.

## Receipts

[`receipts/p88-prove-1/`](receipts/p88-prove-1/) and [`receipts/p88-5090-4/`](receipts/p88-5090-4/), byte-identical to the receipt store's. For each run:
- summary, forensics, versions, the premise result;
- for the reading, also: the eight step receipts, the four censuses, the three K8 receipts, `verdict.json`, the pack manifest, and `census_diff.txt` (generated here);
- `SHA256SUMS`.

The three NOT_RUN receipts are in the store only.
