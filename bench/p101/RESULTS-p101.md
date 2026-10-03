# P101 — results: **SUPPORTED**. On the fixed code, Qwen3.6-35B-A3B's bucketed decode graphs all capture and replay through the serving stack, bit for bit equal to the padded eager step; W16 decodes at 449.0 tok/s and W1 at 79.7, 2.02× and 3.28× plain eager

Registration: `bench/p101/PREREG-p101.md` (#919, `4287d07`). Issues: #564 and #913. Code under test: e4b at `4287d07`,
which includes #918's fix of #913, and grouped-nf4-gemm at `34da93d`. Measurement: P98's kit at its registered bytes
(`bench/p98/staged.sha256`; the box refuses a staged file whose hash differs).

**Verdict by `p98_reduce.py`, under P98's rule: `SUPPORTED`.**
- Every bucket (1, 2, 4, 8, 16) captured and replayed with **no eager step**: 209 replays.
- Arm g's tokens equal arm e's (the padded-eager oracle) on **all 17 requests**, W16's 16 and W1's one.

## What happened (`p101-5090-5`)

- **The premise held:** the three hybrid GPU test files, 4 passed, none skipped.
- **The bake held:** 40 layers × 256 experts, a 16.88 GiB NF4 snapshot, loaded commit `995ad96`.
- **All three arms ran both workloads to their budgets**, each in its own process:
  - **g:** bucketed graphs;
  - **e:** the same buckets, every step eager on the same padded layout;
  - **p:** plain eager.

| bucket | replays | rows | padding rows | eager steps |
|---:|---:|---:|---:|---:|
| 1 | 99 | 99 | 0 | 0 |
| 2 | 6 | 12 | 0 | 0 |
| 4 | 10 | 36 | 4 | 0 |
| 8 | 20 | 132 | 28 | 0 |
| 16 | 74 | 1,048 | 136 | 0 |

- **Bucket 1, the faulting bucket, replayed 99 times.** It faulted on P98's run, and on P99's d0g and d2g at their
  first bucket-1 replay.
- **The eager arms reproduced P98's tokens exactly.** Arms e and p wrote the same tokens, on both workloads, as P98's
  arms e and p (`bench/p98/receipts/p98-5090-2/`). P98 ran on another host, at e4b `a08df83`. What moved between the two
  readings is the speed, not the arithmetic.

## Decode speed

| | arm g (graphs) | arm e (padded eager) | arm p (plain eager) | graphs ÷ plain |
|---|---:|---:|---:|---:|
| W16, tok/s (111 timed steps) | **449.0** | 277.6 | 222.7 | **2.02×** |
| W1, tok/s | **79.7** | 26.7 | 24.3 | **3.28×** |
| step at 1 row / 16 rows, ms | 12.6 / 28.1 | — | 42.0 / 50.1 | 3.3× / 1.8× |
| peak GPU memory | 23.76 GB | 23.71 GB | 22.74 GB | |
| build | 32.1 s | 23.9 s | 23.4 s | |

- **Graph and plain-eager tokens agree** on all of W1 and on 67.2 % of W16's positions. P98's padded-eager and
  plain-eager arms agreed on the same 67.2 %. Padded and unpadded steps run different row counts per GEMM, and
  free-running greedy decode diverges for good after one flipped argmax.
- **This is still the torch path.** The Gated DeltaNet layers ran transformers' torch path: no `fla`, no
  `causal_conv1d`. A graph removes that path's launch and Python cost, but not its kernels.

## Against the predictions

| prediction (written before the data) | result |
|---|---|
| SUPPORTED | **yes** |
| graphs at least 3× plain eager on both workloads | **W1 yes (3.28×); W16 no (2.02×)** |
| arm g W1 between 30 and 150 tok/s | **yes** (79.7) |
| arm g W16 above 350 tok/s | **yes** (449.0) |
| arms e and p within 10 % of P98's eager readings | **no**: about 2.2× faster (see below) |
| peak GPU memory at most 28 GB in every arm | **yes** (23.76 GB) |

**The eager arms ran about 2.2× P98's speed on a different CPU.**

| | P98 (AMD EPYC 7663, Zen 3 server) | P101 (AMD Ryzen 9 7950X, Zen 4 desktop) |
|---|---:|---:|
| arm e W16 / W1, tok/s | 128.3 / 11.98 | 277.6 / 26.7 |
| arm p W16 / W1, tok/s | 107.3 / 11.46 | 222.7 / 24.3 |
| arm p step at 1 row, ms | 90.9 | 42.0 |

- **Same arithmetic, different host.** Both boxes carried an RTX 5090 at sm_120, and the tokens are identical.
- **The arithmetic needed the CPU.** P98 found eager hybrid decode launch-bound: a step cost about the same at 1 row as
  at 16. A launch-bound step runs at the speed of the host thread issuing it.
- **The 10 % prediction ignored the host.** It assumed P98's host, and the lane does not choose its CPU.
- **The 2.02× is likely the low end.** It sets graphs against a fast host's eager path. On P98's host the eager
  denominator would be about twice as slow, while the graph replay offloads most of the host's per-step work. This
  lane did not measure graphs on that host, so that ratio is not claimed.

## The registered consequence (SUPPORTED)

- `docs/SERVING.md`'s hybrid paragraph cites this lane as the GPU reading of hybrid decode graphs, with their speed.
- Its warning against `E4B_PAGED_GRAPHS=1` on hybrid models is lifted.
- Register row: `e4b.serve.p101.qwen36-hybrid-decode-graphs.5090.2026-10-03`.
- #913 closes. Its fix (#918) has now been read on the GPU where the fault was found.

## Box and cost

Seven rentals, two that ran:

| run | status | machine | cost | why |
|---|---|---:|---:|---|
| `p101-prove-1` | NOT_RUN | 96642 | $0.0192 | pre-flight download bandwidth 10.3 MB/s < 40 |
| `p101-prove-2` | OK, PROVED | 36204 | $0.0258 | install tripwire, reducer self-test (19), premise |
| `p101-5090-1` | NOT_RUN | 136851 | $0.0223 | pre-flight bandwidth 10.0 MB/s |
| `p101-5090-2` | NOT_RUN | 136851 | $0.0238 | pre-flight bandwidth 9.9 MB/s (with -1, the pair that excluded 136851) |
| `p101-5090-3` | ALARM | 96642 | $0.0249 | pre-flight bandwidth 13.4 MB/s; teardown through the emergency retry loop, instance absent |
| `p101-5090-4` | NOT_RUN | 36204 | $0.0095 | the launcher's Vast API call timed out in the TLS handshake |
| `p101-5090-5` | **OK, SUPPORTED** | 36204 | **$0.1937** | the reading, 1,302 s |

- **The reading:** one RTX 5090 (sm_120, driver 580.95.05) on an AMD Ryzen 9 7950X, Vast instance 53959695.
  - Pre-flight 111.6 MB/s; the snapshot fetch took 16 min.
  - Destroyed at 2026-10-03T03:46:40Z, instance absent.
- **The lane: $0.3192** of a $2.50 ceiling.
- **Every refusal was destroyed and proven absent.** The ALARM and the API timeout were each also checked against a
  live v1 instance listing.
- **Two slow hosts, 136851 and 96642, refused seven launches across three lanes this morning (four and three).** They were the cheapest
  verified 5090 offers.
  - 136851 was excluded here by its same-lane pair.
  - 96642 could not be: adertha admits only NOT_RUN receipts as exclusion evidence, and its second P101 refusal is an
    ALARM.

Receipts (`receipts/p101-5090-5/`):
- the three arm records, `verdict.json`, `summary.txt`, `forensics.txt` and `versions.txt`;
- the premise, fetch, bake and arm logs;
- `work/bake.json` and the teardown proof;
- `SHA256SUMS` over every file the box wrote.

The launcher's receipts and ledger rows are in the receipt store (adertha-receipts `5208003`, `1268b67`, `df62316`,
`fe06fb8`, `64d02ff`, `6ec654e`, `493a7cd`).
