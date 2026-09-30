# P84 — results: **KERNEL**. grouped-nf4-gemm 0.33.0 → 0.33.7 moved the recipe's fp32 K8 by −0.000491 nats; e4b and the harness moved nothing

Registration: `bench/p84/PREREG-p84.md` (#792, `5675078`), with amendment 1 (#795, `33d0f2f`). Issue: #674.

**Verdict by `p84_reduce.py`: KERNEL.** Reading 2 (`p84-5090-3`), on one RTX 5090 on the Ryzen 7950X host that read P55x's
and P70's licensed 6.36709:
- **KERNEL** (grouped-nf4-gemm 0.33.0 → 0.33.7, on e4b 0.37.4 and P70's harness) is −0.000491 nats, the whole of N − O.
- **PACKAGE** (e4b 0.37.4 → 0.37.8) and **HARNESS** (P70's → P82's) are exactly zero.
- Every build repeated itself bit for bit.
- The stated expectation, PACKAGE alone, was wrong.

Reading 1 (`p84-5090-2`) was VOID on an Intel host and is kept below.

Which change inside the kernel cut moved it is **not measured**. Read from the code, exactly one change in the cut is on
K8's path. It is grouped-nf4-gemm#413 in 0.33.7, which IEEE-rounds the fused fp8 KV append's quotient. The registration
had assumed the eager K8 never ran that append; it does. See below: this is an inference from the diff and the call
path, and the test that would measure it is proposed, not run.

## Runs and cost

| run | outcome | cost (ledger) |
|---|---|---:|
| `p84-prove-1` | **NOT_RUN**: the instance was stuck `loading` for 600 s | $0.0927 |
| `p84-prove-2` | **PROVED**, lane rc 0: both stacks installed on an EPYC 7C13; each harness imports under its stack; both stamps fp32 | $0.0382 |
| `p84-5090-1` | **NOT_RUN**: ssh did not authenticate (publickey). An ssh-readiness failure, so its machine (145142) is excluded from then on | $0.0377 |
| `p84-5090-2` | **reading 1: VOID**. The control failed on an Intel host (below), and the lane stopped after it, as registered | $0.5356 |
| `p84-prove-3` | **NOT_RUN**: ssh refused for 180 s, on machine 146028, reading 1's Intel host. An ssh-readiness failure, so that machine is excluded too. It was the third and last registered proof attempt; the lane's proof stands on `p84-prove-2` (amendment 1 changes only the preflight vendor check, which the A2000 rehearsal exercised) | $0.0358 |
| `p84-5090-3` | **reading 2: KERNEL** | $1.0928 |
| **total** | | **$1.8328** of the $4.00 ceiling |

Every teardown is proven (`vast-destroy`, HTTP 200, instance absent). The ledger prices each run as the offer's rate ×
its runtime, not from an invoice.

## Reading 2 (`p84-5090-3`): KERNEL

- **Host.** One RTX 5090 (driver 575.57.08, 600 W) on an **AMD Ryzen 9 7950X**
  ([`forensics.txt`](receipts/p84-5090-3/forensics.txt)). Its GPU UUID, `7a6634c6…`, is the card in P55x's and P70's
  forensics, so this is the machine that read the licensed 6.36709. The amended preflight saw `AuthenticAMD`.
- **Stacks.** B: e4b `33d0f2f` (0.37.8) + grouped-nf4-gemm `9407d49` (0.33.7). A: e4b `c77aab6` (0.37.4) + the same
  0.33.7, installed over B.
- **Timeline (UTC, from the run's log).**

  | step | window |
  |---|---|
  | install B, fetch the model | 18:18–18:31 |
  | bake, then C's build | 18:31–19:03 (first calibration chunk 360 s) |
  | C's repeat | 19:03–19:05 |
  | H1's build, then its repeat | 19:05–19:43 |
  | install A, bake, then H2's build | 19:43–20:15 |
  | H2's repeat | 20:15–20:20 |

  The lane exited rc 0 at 20:20:51.

| reading | e4b | grouped-nf4-gemm | harness | K8 mean NLL | ppl | repeat |
|---|---|---|---|---|---|---|
| **O** (known: P70, P64, P83) | 0.37.4 | 0.33.0 | P70's | `1.8511420498367808` | 6.36709 | — |
| **H2** | 0.37.4 | 0.33.7 | P70's | `1.8506507749113845` | 6.36396 | bit-identical |
| **H1** | 0.37.8 | 0.33.7 | P70's | `1.8506507749113845` | 6.36396 | bit-identical |
| **C** (control; N known: P82, P83) | 0.37.8 | 0.33.7 | P82's | `1.8506507749113845` | 6.36396 | bit-identical |

- **The control held.** C reproduced N's known float bit for bit, so this box's comparisons with the known floats hold.
- **The steps.** KERNEL (O → H2) is −0.0004912749 nats. PACKAGE (H2 → H1) is 0.0 and HARNESS (H1 → C) is 0.0,
  exactly: all three builds on 0.33.7 read the same float, whatever the e4b release or harness.
- **Packs.**
  - Every expert pack is `0c9955a9…` (C, H1, H2), the licensed pack.
  - C's attention pack is `d7cfa1f4…`, the AMD hosts' pack, reproduced on a fourth AMD machine.
  - H1's and H2's attention is calibrated live, as P70's harness does, and is not dumped.
- **The predictions, scored.**
  - The stated expectation, **PACKAGE alone, was wrong**.
  - The control reproduced N's float, both new builds repeated themselves, every expert pack was `0c9955a9…`, and C's
    attention was `d7cfa1f4…`: all held.

**What the registered rule says next.** For KERNEL, it says to bisect grouped-nf4-gemm 0.33.1 → 0.33.6 on e4b 0.37.4.
Reading the cut release by release first narrows that to one release. The next section records that reading.

## Which change in the kernel cut: one is on K8's path (read from the code, not measured)

What each release changed in shipped modules, 0.33.0 → 0.33.7, and whether K8 runs it:

| release | shipped-module change | on K8's path? |
|---|---|---|
| 0.33.1–0.33.3 | none in code (`int4_b32` docstrings only) | — |
| 0.33.4 | `cold_deadline`: the cold-expert CPU-offload cost model | no: e4b reaches it only for experts not resident in VRAM (`_cold_to_cpu_deadline`), and K8 runs all-VRAM |
| 0.33.5 | `int4_smallm` postpones its annotations (#406); a `gnf4_native` docstring | no: the K16 route serves 2–16 rows, and K8's decode is 1 row, its prefill 512 |
| 0.33.6 | MXFP4 prefill combine order (#410) | no: gpt-oss's MXFP4 path, not Qwen3's int4 |
| **0.33.7** | **the fused fp8 KV append's quotient is IEEE-rounded (`div_rn`, #413)** | **yes** |

**Why the append is on the path.** Both harnesses call `kv.graph_mode_init(...)` before the teacher-forced window, in
the eager loop as in the graph loop (`bench/p39/step_decomp.py:1530`, `bench/p81/step_decomp.py:1535`).
- That sets `graph_t1`, so `paged_attention` routes every decode append to `Fp8PagedKV.append_graph_t1`.
- `append_graph_t1` takes the fused Triton `fp8_kv_append_t1` whenever it resolves on (`_resolve_fused_append`: the
  default on CUDA; no lane from P55x to P84 sets `E4B_FUSED_KV_APPEND`).
- The prompt's KV is written by the prefill through `quantize_kv_fp8`, which 0.33.7 did not change. Every scored
  token's K and V go through the changed kernel.

**The size of the change.** Lane B771 measured the old kernel (0.33.5's, identical to 0.33.0's) against `quantize_kv_fp8`'s bytes. It wrote a different
byte in 21 of 536,870,912 values (3.9e-8); 0.33.7 writes none different (`bench/b771/RESULTS-b771.md`). A K8 run
appends 2,048 steps × 48 layers × 4 KV heads × 128 × 2 sides = 100,663,296 values.
- At B771's rate that is a handful of changed fp8 bytes per run, each persisting in the cache for every later step.
- Whether a handful of flips, carried through attention and top-8 routing, adds up to −0.00049 nats over 2,048 tokens
  is **not shown**. The rate on real Qwen3 K/V is not measured either.

So #413 is the one arithmetic change on this path in the cut, and the leading explanation. It is **an inference from
the diff and the call path**. P84 measured the release range, not the commit.

**The test that would measure it (proposed as P85, not started).** P70's build (e4b 0.37.4 + grouped-nf4-gemm 0.33.0)
with `E4B_FUSED_KV_APPEND=0` appends through `quantize_kv_fp8` instead. 0.33.7's kernel writes exactly those bytes. If
#413 is the whole KERNEL step, that build reads **N's float, `1.8506507749113845`, bit for bit**. A companion reading,
0.33.6 on e4b 0.37.4 with the append on, would read **O's float**. Each prediction is one exact number.

**It also bears on the cast pair.** P81 ran grouped-nf4-gemm 0.33.5 and P82 ran 0.33.7, so #413 lies between them too.
Their cast-router readings (6.33015 and 6.31811, identical packs) are what P82 read as a box effect. #413 is now the
first suspect for that pair as well; not measured either.

## Corrections: my reading of the diff, in three places

- **PREREG-p82** (registration, left as registered) said grouped-nf4-gemm "records no arithmetic change on this path.
  0.33.7's append is the graph path's; the eager K8 appends through `quantize_kv_fp8`." **The second sentence is
  wrong.** The eager K8 loop enters graph mode for its appends, as above, so 0.33.7's append is on its path.
- **PREREG-p84's stated expectation** (left as registered) repeated it: the graph-mode fp8 append was among changes
  "none of which the eager all-VRAM K8 runs".
- **RESULTS-p82** said P81's and P82's "code on K8's path is the same", and **RESULTS-p83** listed "a difference between
  the two runs that P82's diff reading missed" as the first suspect for the cast pair. Both are annotated in place.

The P82 and P83 register rows carry the same note.

## What this means for #674

- For fixed software on the AMD hosts tested, the recipe's fp32 K8 is one float.
  - P70's software reads 6.36709. The current release reads 6.36396 on the same machine.
  - The step between them is inside grouped-nf4-gemm 0.33.0 → 0.33.7 (−0.00049 nats, a function change far inside
    the 0.0095 floor). e4b 0.37.4 → 0.37.8 and the harness move nothing.
- If #413 is the cause, the newer reading is the one whose KV bytes equal the reference quantizer's.
- Whether the licensed row should name the software it was read on, or be re-read on the current release, is the
  owner's decision. This read does not change the row.

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
  - [`receipts/p84-5090-3/`](receipts/p84-5090-3/): the six K8 receipts and their stamps, `stamp_{A,B}.json`,
    `versions_{A,B}.txt`, `verdict.json`, `packs.json`, `summary.txt`, `forensics.txt`, the four pack manifests, and
    `SHA256SUMS`. All 23 files are byte-identical to the store's.
  - [`receipts/p84-5090-2/`](receipts/p84-5090-2/): C's two K8 receipts and stamps, `stamp_B.json`, `versions_B.txt`,
    `verdict.json`, `packs.json`, `summary.txt`, `forensics.txt`, both C pack manifests, and `SHA256SUMS`. All 12 files
    are byte-identical to the store's.
  - [`receipts/p84-prove-2/`](receipts/p84-prove-2/): the proof's summary, versions, stamps and forensics.
- **In the adertha receipt store:** `receipts/experts4bit-qlora/2026-09-30/p84-{prove-1,prove-2,5090-1,5090-2,prove-3,5090-3}/`
  (commits `8f34950`, `86deca6`, `cb293d5`, `55302d8`, `8ef0f43`, `8f6f884`). `p84-5090-3` also holds the run's logs,
  both harness copies, `outer.log`, `receipt.json` and `teardown-proof.json`.
