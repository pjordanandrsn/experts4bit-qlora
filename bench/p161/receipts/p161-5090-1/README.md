# `p161-5090-1`: the served read (FLIP)

One RTX 5090 (cc 12.0, 170 SMs, driver 595.91.07, a **575 W** power limit, enforced 575 W) on an AMD EPYC 7K62 host
(96 CPUs, 170 GB disk), 2026-10-20T17:00:07Z to 17:35:47Z. Cost **$0.726** (the provider's invoice lines: GPU $0.5410,
download $0.1640 for 62.9 GB, storage $0.0190, upload $0.0020), against the registered worst of $2.35.
- **Code:** e4b `da3eb5c5` (the registration's merge, #1904), grouped-nf4-gemm v0.50.0 at `41ef086e`.
- **Stack:** torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2; Qwen/Qwen3-30B-A3B @ `ad44e777`.

## Before the arms

- **The floors:** the power limit read 575 W before the install. The guard fit read `fit=3300s need=4125s left=5292s`.
- **The self-tests:** the reducer 96 cases, the box 29, P109's box 5.
- **The premise:** 62 passed on this card: P136's 19, K39's compiled contract (23) and e4b's served-route contract with
  c1's cases (20).
- **The early gates:** passed after the warm-up (S0w, today's plan) and after S1a (c1).

## The processes

Every process ran K25's decode at qwen3_moe's default (`prmt`, env unset, `default-allowlisted`); the arms differ only in
`E4B_K25_PLAN`.

| process | plan | K25 decode record | plan record | the spy (kernel calls by block shape) | `bw_prmt32` | W16 tok/s | W1 tok/s | load s |
|---|---|---|---|---|---|---|---|---|
| S0w (warm-up) | today | `{"prmt\|le256": 960}` | `{"today\|le256": 960}` | `{"32/64/4/3/prmt": 1920}` | 480 | 1037.77 | 270.48 | 57.94 |
| S0a | today | `{"prmt\|le256": 960}` | `{"today\|le256": 960}` | `{"32/64/4/3/prmt": 1920}` | 480 | 1038.38 | 270.97 | 41.05 |
| S1a | c1 | `{"prmt\|le256": 960}` | `{"c1\|le256": 960}` | `{"64/256/4/2/prmt": 960, "128/128/8/3/prmt": 960}` | 480 | 1528.55 | 270.06 | 40.66 |
| S1b | c1 | `{"prmt\|le256": 960}` | `{"c1\|le256": 960}` | `{"64/256/4/2/prmt": 960, "128/128/8/3/prmt": 960}` | 480 | 1523.85 | 270.32 | 40.60 |
| S0b | today | `{"prmt\|le256": 960}` | `{"today\|le256": 960}` | `{"32/64/4/3/prmt": 1920}` | 480 | 1036.83 | 270.46 | 41.01 |
| S0c | today | `{"prmt\|le256": 960}` | `{"today\|le256": 960}` | `{"32/64/4/3/prmt": 1920}` | 480 | 1039.14 | 270.24 | 40.79 |
| S1c | c1 | `{"prmt\|le256": 960}` | `{"c1\|le256": 960}` | `{"64/256/4/2/prmt": 960, "128/128/8/3/prmt": 960}` | 480 | 1522.79 | 270.62 | 41.69 |
| S1d | c1 | `{"prmt\|le256": 960}` | `{"c1\|le256": 960}` | `{"64/256/4/2/prmt": 960, "128/128/8/3/prmt": 960}` | 480 | 1525.83 | 270.22 | 40.81 |
| S0d | today | `{"prmt\|le256": 960}` | `{"today\|le256": 960}` | `{"32/64/4/3/prmt": 1920}` | 480 | 1038.49 | 270.43 | 40.31 |

Every process read every closed form exactly, under its own plan only. Each record after the runs is the same as after
the build: the timed decode steps replay the captured graphs (a replay never counts), and the prefill's rows are above
K25's bound.

## The verdict

**FLIP** (`verdict_p161.json`): g16 **1.465433** (the min of four W16 pairs) >= 1.03 and g1 **0.996642** >= 0.99, both
unrounded.

| | S1a/S0a | S1b/S0b | S1c/S0c | S1d/S0d |
|---|---|---|---|---|
| W16 | 1.4721 | 1.4697 | 1.4654 | 1.4693 |
| W1 | 0.9966 | 0.9995 | 1.0014 | 0.9992 |

- **The null:** every S0 process, the warm-up's included, decoded S0a's tokens at both workloads and both lengths, and
  every S0 arm's timed reps digest alike.
- **The contract:** every S1 process decoded S0a's tokens, and every S1 arm's timed reps digest alike.
- **Self-pairs:** W16 0.9969-1.0020, W1 0.9981-1.0010; the cross-block self-pairs 0.9962-1.0021.
- **Clocks:** no pair is clock-confounded. The median sampled SM clock over the W16 blocks read 2850 MHz in every scored
  arm (63 samples an S1 arm, 73 an S0 arm). The enforced limit read 575 W in every sample. The software power-cap event
  (0x4) appears in 0.17-0.22 of each S1 arm's W16 samples and in none of the S0 arms': c1 touches the cap, at an
  unchanged limit and an unchanged median clock.
- **The warm-up** (reported, never scored): W16 0.9994 and W1 0.9982 of S0a. Its load took 57.9 s, against 40.3-41.7 s
  for every later process.

## Files

- **Here:** the nine arm records, `verdict_p161.json`, `summary.txt`, `versions.txt`, `prompts.json`, `work/bake.json`,
  and the bake, pip and premise logs. `SHA256SUMS` covers each file in this directory.
- **Only in the private receipt store,** byte-exact, listed by sha256 in `STORE-ONLY.sha256`:
  - `forensics.txt` and the runner's console log, which carry the GPU's UUID (no reading depends on it);
  - eleven logs that carry carriage returns from the box's progress output;
  - `INSTANCE_ID` and the launcher's teardown proof, which name the provider's instance.
- **No identifiers:** no file here names an instance, an offer, a host name, an endpoint or the GPU's UUID.

The verdict re-derives byte for byte from the arm records with the registered reducer:

    python bench/p161/p161_reduce.py --dir bench/p161/receipts/p161-5090-1 --out verdict.json \
        --e4b-sha da3eb5c5e452e0031659e2e219b63bb130e79e3b

Re-derive: `python bench/p161/p161_reduce.py --dir bench/p161/receipts/p161-5090-1 --out {out} --e4b-sha da3eb5c5e452e0031659e2e219b63bb130e79e3b` writes `bench/p161/receipts/p161-5090-1/verdict_p161.json`
