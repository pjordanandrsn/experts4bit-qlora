# `p134-5090-1`: the bandwidth GEMV on Granite and OLMoE (LICENSED, both)

One RTX 5090 (cc 12.0, 170 SMs, a **400 W** power limit) on an AMD EPYC 9554 host (256 CPUs, 1,133 GiB RAM, 100 GB
disk), machine 45708, 2026-10-18T03:30:23Z to 04:07:17Z. Cost **$0.480**.
- **Code:** e4b `c8d7e629` (the box's pin), grouped-nf4-gemm 0.46.0 at `6b7c2e74`. The harness is `bench/p134/` at the
  registration's reviewed main (#1697 with Amendment 1, #1704).
- **Stack:** torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2.
- **Models:** ibm-granite/granite-3.1-3b-a800m-instruct @ `a0278068`; allenai/OLMoE-1B-7B-0924-Instruct @ `7f1c97f4`.
- **The power limit:** this read predates the 575 W floor later lanes register. 400 W is below the class's stock limit,
  so these ratios are on a capped card. Both arms ran on it.

## Before the arms

- **The premise:** grouped-nf4-gemm's K33 kernel contract gave 35 passed (none skipped on sm_120), and the served route 18
  passed.
- **The fit:** each family's default server built at `max_seqs` 64, buckets 1-64, graphs on. The control is W16.
- **Engagement:** each B0 build dispatched only to the scalar GEMV (Granite 192, OLMoE 96 calls), and each B1 build the
  same counts to `bw_prmt32`.

## The arms

| family | process | `GNF4_GEMV_BW` | W1 tok/s | W16 tok/s |
|---|---|---|---|---|
| Granite | B0a / B1a / B1b / B0b | unset / 1 / 1 / unset | 197.73 / 276.68 / 275.13 / 196.81 | 2224.56 / 2222.68 / 2223.57 / 2219.93 |
| OLMoE | B0a / B1a / B1b / B0b | unset / 1 / 1 / unset | 246.92 / 370.64 / 371.39 / 246.46 | 1937.05 / 1928.06 / 1923.78 / 1922.84 |

## The verdicts

**LICENSED for both families** (`verdict_granite.json`, `verdict_olmoe.json`):

| | g1 (W1 pairs) | gW at W16 (pairs) | self-pairs W1 / W16 |
|---|---|---|---|
| Granite | **1.3979** (1.3993, 1.3979) | **0.9992** (0.9992, 1.0016) | 0.9953, 0.9944 / 0.9979, 1.0004 |
| OLMoE | **1.5011** (1.5011, 1.5069) | **0.9954** (0.9954, 1.0005) | 0.9981, 1.0020 / 0.9927, 0.9978 |

- **Tokens:** OLMoE's B1 decoded B0's tokens on every W1 row and on 31 of the 32 W16 rows. Granite's B1 differed from
  B0's on the W1 row at 160 tokens and on 3 of the 32 W16 rows. The kernel is not bitwise with the scalar route; it ships
  on the quality gate.
- **Quality** (P110's teacher-forced instrument, 24 windows a text, T == 1):

  | | wikitext bias (bar) | spread (bar) | wikitext K8 Δppl (gated, 0.05) | c4val1 bias (bar) | c4val1 K8 (reported) |
  |---|---|---|---|---|---|
  | Granite | +0.00260 (0.01338) | 0.01045 (0.02057) | +0.0161 | +0.00459 (0.01042) | +0.0503 |
  | OLMoE | -0.00238 (0.01087) | 0.00903 (0.02473) | -0.0234 | -0.00089 (0.01380) | -0.0124 |

  K8 is gated on wikitext only. Granite's c4val1 K8 of +0.0503 is reported and not gated.

## Files

- **Here:** the eight arm records, both families' fit, prompts and quality records, both verdicts, `summary.txt`,
  `versions.txt`, and the bake, clone, pip, premise and OLMoE prompts logs. `SHA256SUMS` covers each file here.
- **Only in the private receipt store,** byte-exact, listed by sha256 in `STORE-ONLY.sha256`:
  - `forensics.txt` and the runner's console log (the GPU's UUID);
  - the premise suites' junit reports (the container's host name);
  - the logs that carry carriage returns from progress output;
  - the instance record and teardown proof (the provider's instance).
- **No identifiers:** no file here names an instance, an offer, a host name, an endpoint or the GPU's UUID.

Each verdict re-derives byte for byte from these records with the registered reducer, run beside its two staged
siblings (both in the repository at their staged bytes), under Python 3.10 (as checked). Under 3.12 or later
only the last digits of a few floats move, never a figure or a verdict:

    mkdir -p /tmp/rd && cp bench/p134/p134_reduce.py bench/p115/p115_reduce.py bench/p110/p110_box.py /tmp/rd/
    python /tmp/rd/p134_reduce.py --dir bench/p134/receipts/p134-5090-1 --family granite --out verdict_granite.json \
        --e4b-sha c8d7e629e4c8431fc86970cf7195d8e67f72e22a      # and --family olmoe
