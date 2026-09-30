# P84 — results so far: reading 1 is **VOID** (the control failed on an Intel host, whose attention calibration does not reproduce the AMD hosts' pack); the lane continues under amendment 1

Registration: `bench/p84/PREREG-p84.md` (#792, `5675078`), with amendment 1 (this PR). Issue: #674. The lane's question
(which factor moved the fp32 K8 from P70's build to P82's) is **not yet answered**. The next reading will be added here.

## Runs and cost

| run | outcome | cost (ledger) |
|---|---|---:|
| `p84-prove-1` | **NOT_RUN**: the instance was stuck `loading` for 600 s | $0.0927 |
| `p84-prove-2` | **PROVED**, lane rc 0: both stacks installed on an EPYC 7C13; each harness imports under its stack; both stamps fp32 | $0.0382 |
| `p84-5090-1` | **NOT_RUN**: ssh did not authenticate (publickey). An ssh-readiness failure, so its machine (145142) is excluded from then on | $0.0377 |
| `p84-5090-2` | **reading 1: VOID**. The control failed (below), and the lane stopped after it, as registered | $0.5356 |
| **so far** | | **$0.7042** |

Every teardown is proven (`vast-destroy`, HTTP 200, instance absent).

## Reading 1 (`p84-5090-2`): the control failed on an Intel host

One RTX 5090 (driver 595.84, 575 W) on an **Intel Core i9-14900K** host
([`forensics.txt`](receipts/p84-5090-2/forensics.txt)). Stack B: e4b at `5675078` with grouped-nf4-gemm 0.33.7.
- **Timeline (UTC).** Install 17:01. The fetch ran 17:03–17:23: slow, with one resumed download. Bake 17:23. C's build
  ran 17:24–17:53 (first calibration chunk 360 s), and its repeat 17:53–17:56.
- **The control C is P82's build, bit for bit** (P82's harness, env and command). It read **mean NLL
  `1.8504622130569988`, ppl 6.36276**. Its repeat, K32 with both packs loaded by fingerprint, read the same float, so the
  box is deterministic.
- **N's known float is `1.8506507749113845` (6.36396).** They are not equal, so the rule reads **VOID**, and the lane
  stopped without running H1 or H2. The early stop saved two builds.

**Why: the attention pack is different.**

| | expert pack | attention pack |
|---|---|---|
| this host (Intel i9-14900K) | `0c9955a9…` (licensed) | **`45b4cc5a…`** |
| P81 (EPYC 7K62), P82 (EPYC 7R13), P83 (EPYC 9755) | `0c9955a9…` | `d7cfa1f4…` |

- Against P82's attention manifest, **342 of the 384 tensor payloads differ**: all 192 scale tensors and 151 of 192
  packed ones, in all 48 layers. The identity payload, the layer list, the calibration token stream (`e5d5eba5…`) and the
  recorded toolchain are the same (torch 2.8.0+cu128, triton 3.4.0, `e4b_int4_gptq_device: cuda`, damp 0.01).
- A broad, small perturbation like this is what slightly different Hessians feeding the same GPTQ solve would produce.
  **The mechanism is not isolated.**
  - In the path: the attention Hessians' Gram is formed on the GPU in fp32 and accumulated on the CPU
    (grouped-nf4-gemm's `gptq_pack.HessianAccumulator`, `hessian_device="cpu"`).
  - Against that: the expert calibration uses the same accumulator, and its pack did reproduce here. So the calibration
    forward pass is as much a candidate as the accumulation.
  - The host differs in CPU vendor (the first Intel in this series) and in driver (595.84). Neither is separated from
    the other.
- **What it means for #674 and P83.**
  - The calibrated attention reproduces **on the AMD hosts tested** (three, `d7cfa1f4…`) but **not on this Intel host**.
  - P83's "K8 is bit-reproducible across machines for a fixed software stack" holds on the AMD machines it covered
    (five readings on four hosts). It is qualified in place.
  - The attention half is pinned by its artifact (#754), not by re-running the recipe on arbitrary hardware.

## Amendment 1

The known floats O and N were read on AMD hosts only, so the chain is read on AMD hosts: the runner refuses a
non-`AuthenticAMD` CPU at preflight, with rc 16, before the install and the fetch (PREREG-p84, amendment 1). The A2000
rehearsal of the amended runner is in [`rehearsal-a2000-amendment1/`](rehearsal-a2000-amendment1/).

## Receipts

- **In this repo:**
  - [`receipts/p84-5090-2/`](receipts/p84-5090-2/): C's two K8 receipts and stamps, `stamp_B.json`, `versions_B.txt`,
    `verdict.json`, `packs.json`, `summary.txt`, `forensics.txt`, both C pack manifests, and `SHA256SUMS`. All 12 files
    are byte-identical to the store's.
  - [`receipts/p84-prove-2/`](receipts/p84-prove-2/): the proof's summary, versions, stamps and forensics.
- **In the adertha receipt store:** `receipts/experts4bit-qlora/2026-09-30/p84-{prove-1,prove-2,5090-1,5090-2}/`
  (commits `8f34950`, `86deca6`, `cb293d5`, `55302d8`).
