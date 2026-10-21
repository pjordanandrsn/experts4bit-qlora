# `p158-5090-1`: the served read (FLIP)

One RTX 5090 (cc 12.0, 170 SMs, driver 615.71.09, a **575 W** power limit, enforced 575 W) on an AMD Ryzen 7 9800X3D
host (16 CPUs, 91 GiB RAM, 170 GB disk), machine 147099, 2026-10-20T04:03:54Z to 04:24:37Z. Cost **$0.4190** (the
provider's invoice lines: GPU $0.2460, download $0.1620 for 62.2 GB, storage $0.0100, upload $0.0010), against the
registered worst of $2.35.
- **Code:** e4b `406a391a` (the registration's merge, #1864), grouped-nf4-gemm v0.50.0 at `41ef086e`.
- **Stack:** torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2; Qwen/Qwen3-30B-A3B @ `ad44e777`.

## Before the arms

- **The floors:** the power limit read 575 W before the install. The guard fit read `fit=3300s need=4125s left=5328s`.
- **The self-tests:** the reducer 72 cases, the box 24, P109's box 5.
- **The premise:** 49 passed on this card: P136's 19, K39's compiled contract (23) and e4b's served-route contract (7).
- **The early gates:** passed after the warm-up (S0w, the tree) and after S1a (prmt).

## The processes

| process | decode | K25 calls after the build | `bw_prmt32` at the build | W16 tok/s | W1 tok/s | load s |
|---|---|---|---|---|---|---|
| S0w (warm-up) | tree | `{"tree\|le256": 960}` | 480 | 971.73 | 277.28 | 37.58 |
| S0a | tree | `{"tree\|le256": 960}` | 480 | 976.44 | 277.28 | 21.32 |
| S1a | prmt | `{"prmt\|le256": 960}` | 480 | 1066.72 | 277.47 | 20.48 |
| S1b | prmt | `{"prmt\|le256": 960}` | 480 | 1067.04 | 277.42 | 20.93 |
| S0b | tree | `{"tree\|le256": 960}` | 480 | 975.85 | 277.46 | 20.94 |
| S0c | tree | `{"tree\|le256": 960}` | 480 | 976.12 | 277.25 | 20.76 |
| S1c | prmt | `{"prmt\|le256": 960}` | 480 | 1067.28 | 277.40 | 20.82 |
| S1d | prmt | `{"prmt\|le256": 960}` | 480 | 1067.01 | 277.51 | 20.81 |
| S0d | tree | `{"tree\|le256": 960}` | 480 | 976.02 | 277.35 | 21.75 |

Every process read both closed forms exactly, under its own decode only. Each K25 record after the runs is the same 960:
the timed decode steps replay the captured graphs (a replay never counts), and the prefill's rows are above K25's bound.

## The verdict

**FLIP** (`verdict_p158.json`): g16 **1.092458** (the min of four W16 pairs) >= 1.03 and g1 **0.999856** >= 0.99, both
unrounded.

| | S1a/S0a | S1b/S0b | S1c/S0c | S1d/S0d |
|---|---|---|---|---|
| W16 | 1.0925 | 1.0934 | 1.0934 | 1.0932 |
| W1 | 1.0007 | 0.9999 | 1.0005 | 1.0006 |

- **The null:** every S0 process, the warm-up's included, decoded S0a's tokens at both workloads and both lengths, and
  every S0 arm's timed reps digest alike.
- **The contract:** every S1 process decoded S0a's tokens, and every S1 arm's timed reps digest alike.
- **Self-pairs:** W16 0.9994-1.0003, W1 0.9998-1.0006; the cross-block self-pairs 0.9997-1.0005.
- **Clocks:** no pair is clock-confounded. The median sampled SM clock over the W16 blocks read 2842 MHz in every S1 arm
  and 2835 MHz in every S0 arm (+0.25 %), from 73-75 samples an arm. The enforced limit read 575 W in every sample. The
  software power-cap event (0x4) appears in 0.40-0.44 of each S0 arm's W16 samples and in 0.07-0.18 of each S1 arm's
  (the rest read 0x400), so the tree hit the cap more often than prmt, at an unchanged limit: excursions, not a cap that
  dropped mid-run. That is part of the served difference this read measures, not a confound: a decode that draws less
  and so caps less is a property of serving with it. The median clocks differ by only 0.25 %; clocks are not the source
  of the 1.09x.
- **The warm-up** (reported, never scored): W16 0.9952 and W1 1.0000 of S0a. Its load took 37.6 s, against 20.5-21.8 s
  for every later process.

## Files

- **Here:** the nine arm records, `verdict_p158.json`, `summary.txt`, `versions.txt`, `prompts.json`, `work/bake.json`,
  and the bake, pip and premise logs. `SHA256SUMS` covers each file in this directory.
- **Only in the private receipt store,** byte-exact, listed by sha256 in `STORE-ONLY.sha256`:
  - `forensics.txt` and the runner's console log, which carry the GPU's UUID (no reading depends on it);
  - eleven logs that carry carriage returns from the box's progress output;
  - `INSTANCE_ID` and the launcher's teardown proof, which name the provider's instance.
- **No identifiers:** no file here names an instance, an offer, a host name, an endpoint or the GPU's UUID.

The verdict re-derives byte for byte from the arm records with the registered reducer:

    python bench/p158/p158_reduce.py --dir bench/p158/receipts/p158-5090-1 --out verdict.json \
        --e4b-sha 406a391a11d42ddb39154cfc36938a2da5d68216

Re-derive: `python bench/p158/p158_reduce.py --dir bench/p158/receipts/p158-5090-1 --out {out} --e4b-sha 406a391a11d42ddb39154cfc36938a2da5d68216` writes `bench/p158/receipts/p158-5090-1/verdict_p158.json`
