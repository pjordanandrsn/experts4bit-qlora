# Reproduce our numbers

This page is for someone outside the project who wants to check a number we publish. Each row gives the claim, the
stack it was measured on, the command, the hardware, and what one reading cost us. Our losses and limits stand beside
the wins. Every number comes from [`claims.json`](claims.json) by id, or is marked UNVERIFIED with its record.

## Read this first

- **A reading is pinned to a commit, not a version string.** Each row names the e4b commit it was measured at and the
  first release that contains it. A receipt's `versions.txt` records the `pyproject` string, which on `main` between
  releases still names the previous release, so "0.55.0" in a receipt can mean "main after 0.55.0". Check out the
  commit for an exact reproduction. The release named beside it also carries later changes.
- **A reading at any other version is a new reading.** Report it with its own versions (e4b, grouped-nf4-gemm, torch,
  transformers, and the competitor's), its GPU, power limit and host. A different number is not by itself a refutation.
- **Every ratio says which way is better.** A *rate* ratio (tokens/s, B1/B0) above 1 is faster. A *step-time* ratio
  (ON/OFF ms per step) below 1 is faster.
- **Host matters.** The eager decode path and every wall-clock training ratio move with the CPU. Device-time ratios
  travel between hosts; absolute times do not.
- **Cost** is what one binding reading cost us on a rented box, from its receipt, not the lane total.

### Driving a lane on your own GPU box

Every lane runs through its controller (`bench/<lane>/<lane>_drive.sh`, or `bench/tc1/tc1_drive.sh` for the training
head-to-heads). The controller stages hash-pinned files to any GPU box you can ssh to, runs the box script, and fetches
the receipts. From a checkout of the row's commit:

```bash
export E4B_RENT_SSH_HOST=<box> E4B_RENT_SSH_PORT=<port> \
       E4B_RENT_SSH_OPTS='-o StrictHostKeyChecking=accept-new' \
       E4B_RENT_RUN_DIR=$PWD/out E4B_RENT_RUN_ID=<the lane's run-id form, any N> \
       E4B_RENT_DEADLINE_EPOCH=$(($(date +%s)+5*3600)) E4B_RENT_INSTANCE_ID=local
```

The controller refuses a run id outside its lane's form (for example `p158-5090-<N>`). It reads a Hugging Face token
from `HF_TOKEN_FILE` (default `~/.config/hf/token`). The box needs the model's disk and the host floor named in the
lane's PREREG.

## Where to start

### 1. Qwen3-30B-A3B QLoRA-trains on a 12 GB card

- **Claim:** `e4b.train.frontier.qwen3.a2000-12gb.2026-10-02`. **10.463 GB** peak allocated: TC1's matched set
  (642,514,944 fp32 adapters, seq 2048, r 16) at micro-batch 1 × accum 8, with `OFFLOAD_EXPERTS=1`, on an RTX A2000
  12 GB. Held-out 1.9533 → 0.8483 at N=20. One draw.
- **Limits:** micro-batch 2 (the field recipe) ran out of memory on that card, which sidecars shared (1.37 GB held at
  start); an unshared 12 GB card is marginal there. Unsloth 2026.9.14 refused at load (bitsandbytes dispatched modules to the CPU, and its loader has no
  expert-offload lever), and HF + PEFT ran out of memory at load. The older, lighter 7.16 GB reading (`e4b.offload.fits-30b-class`) recorded no model revision
  or run id and was not reproduced.
- **Stack:** e4b `516c87c0aa4253ed8ac6a683ee96fad69be58ba8` (first released in v0.40.0), grouped-nf4-gemm
  `846b512b905468c08f5748943d08769b572affa2` (v0.34.0), torch 2.8.0+cu128. Model `Qwen/Qwen3-30B-A3B` at
  `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
- **Run:** a hand run on our own card, `TC1_LOCAL_BOX=1 … TC1_FAMILIES=qwen3frontier12 bash tc1_run.sh`. The full
  command, its staged files and the snapshot layout are in [`bench/tc1/README.md`](../bench/tc1/README.md) ("TC3 hand
  run on the owned box"). Receipts: [`bench/h2h-2026-10-02/tc3/`](../bench/h2h-2026-10-02/tc3/).
- **Cost:** $0 (our card). Any 12 GB CUDA card can host it; it is a fit reading, not a speed one.

### 2. Qwen3-30B-A3B against Unsloth, packed 4,096-token rows, with e4b's memory cost beside it

- **Claim:** `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.packed-4k-device-ratio`. Unsloth spends **1.160×** e4b's GPU
  time per profiled step, both frameworks at their defaults on one stack (read 2026-10-08). The wall-clock ratio on that host was 1.472.
- **Beside it, e4b's loss:** `e4b.train.memory.packed-4k-position-defaults.5090.2026-10-07`. At its defaults e4b's
  training phase peaks **1.04 GB above** Unsloth's (25.904 vs 24.864 GB, read 2026-10-07). The opt-in `E4B_CKPT_OFFLOAD=1` brings the gap
  to 0.303 GB but costs 1.041× the step, so it stays opt-in.
- **Stack:** e4b `0b2995236f7a5d446299a252db0a7beb1d212280` (first released in v0.49.0), grouped-nf4-gemm
  `8944aa84857349cb98dce30cdf21f56c5a3640b6` (v0.43.0); Unsloth 2026.9.14, unsloth_zoo 2026.9.9, torch 2.12.1+cu130,
  transformers 5.5.0, `grouped_mm` MoE backend. Model as in row 1. The memory row ran at e4b
  `08ff61cd9acc2b8c758a02c414ab5670d550d9ad` (v0.49.0), grouped-nf4-gemm `9953bab9a94f0d5b3c932fe2e90c22f927d4b550`.
- **Run:**
  `GNF4_SHA=<gnf4> TC1_BOX=A TC1_FAMILIES=qwen3pos68 TC1_SEQ=4096 TC1_MB=1 TC1_ACCUM=4 TC1_PACK=1 TC1_FREE_OUTPUTS=1 TC1_STEPS=40 TC1_EVAL_EVERY=40 bash bench/tc1/tc1_drive.sh`
  (the memory row: `TC1_FAMILIES=qwen3pos67`, same knobs). Records:
  [`bench/h2h-2026-10-02/tc1/`](../bench/h2h-2026-10-02/tc1/README.md).
- **Hardware and cost:** one RTX 5090 and TC1's registered pre-flight floor: at least 320 GB disk, 98 GB host RAM and 40 MB/s
  ([`TC1-PREREG.md`](../bench/tc1/TC1-PREREG.md), "Budget and stop rules"). The boxes had 251-504 GB RAM. $1.96 (`tc1-5090-136`) and $1.34
  (`tc1-5090-135`).
- **The other recipe:** at TC1's field recipe (seq 2048, micro-batch 2 × accum 4) the device ratio read 1.924
  (`e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.field-device-ratio`), and a September primary pair on an older stack read
  1.413 (`e4b.train.h2h.unsloth.qwen3.5090.2026-09-05`). Name the recipe when you quote one.

### 3. Mixtral-8x7B against Unsloth: which stack decides it

- **Claim, the matched position:** `e4b.train.h2h.unsloth.mixtral.5090.2026-10-05.same-stack`. With both frameworks
  resident on one stack, on one AMD EPYC 7B13 host, Unsloth/e4b is **1.144** [1.140, 1.148], so e4b is faster per step.
  Beside it, in Unsloth's favour: Unsloth's peak is 2.07 GB lower (29.17 vs 31.24 GB; e4b's fp32 absmax), and its
  energy per step is ×0.93 of e4b's. On the same box, with e4b on the field image's stack, the ratio read **1.013**.
- **Unsloth faster:** `e4b.train.h2h.unsloth.mixtral.5090.2026-10-04.dense-default`. Both resident, e4b at its
  defaults in the field image's environment, on an Intel Core Ultra 9 285K host: Unsloth/e4b **0.836** [0.828, 0.844],
  so Unsloth was faster per step. e4b `ed08029d9fb8d210ac78e9371f64bd06098ed07d` (first released in v0.46.0). $0.60.
- **A footprint lever, not a race:** `e4b.train.h2h.unsloth.mixtral.5090.2026-09-06` reads 0.361: e4b ran under expert
  CPU offload at 3.2 GB, against Unsloth resident at 29.2 GB. Cost UNVERIFIED (it predates our receipt store).
- **Stack (the matched position):** e4b `63400e1ab0af233aeb09e9e59017492da8ca3694` (first released in v0.49.0),
  grouped-nf4-gemm `9622144c734395de4d40332f06dea92135d858de` (v0.42.0), Unsloth 2026.9.14, torch 2.12.1+cu130. The
  September row ran e4b 0.35.1 and grouped-nf4-gemm 0.30.2 from PyPI, Unsloth 2026.9.2 and torch 2.8.0. Model
  `mistralai/Mixtral-8x7B-Instruct-v0.1` at `eba92302a2861cdc0098cc54bc9f17cb2c47eb61`.
- **Run:** the matched position is family token `mixtralsamestack` through `bench/tc1/tc1_drive.sh` (records:
  [`bench/h2h-2026-10-02/tc2/`](../bench/h2h-2026-10-02/tc2/README.md)). The September row is
  [`bench/h2h-20260906/tp2/`](../bench/h2h-20260906/tp2/README.md).
- **Hardware and cost:** one RTX 5090 and a 192 GB host floor. $1.59 (`tc1-5090-84`). As the claim says, these positions are
  within-box readings of their host, so name the host when you quote one.

### 4. K25's byte-permute decode, the qwen3_moe default since 0.56.0

- **Claim:** `e4b.serve.p158.k25-prmt.qwen3.5090.2026-10-20`. Sixteen concurrent requests decode **1.0925×** as fast
  (rate ratio, the smallest of four ABBA pairs), and the tokens are identical. One request is unchanged.
- **Limits:** one card, one family, one width (W16 and W1), today's K25 plan. Nothing about quality beyond token
  identity, and nothing exact about clocks.
- **Stack:** e4b `406a391a11d42ddb39154cfc36938a2da5d68216` (first released in v0.56.0; v0.55.0 has no `E4B_K25_LUT`),
  grouped-nf4-gemm v0.50.0 (`41ef086e`), torch 2.8.0+cu128, transformers 5.17.0. Model as in row 1.
- **Run:** `E4B_RENT_RUN_ID=p158-5090-<N> bash bench/p158/p158_drive.sh`. The arms differ only in `E4B_K25_LUT`
  (`tree` vs `prmt`). Record: [`bench/p158/RESULTS-p158.md`](../bench/p158/RESULTS-p158.md).
- **Hardware and cost:** one RTX 5090 at 575 W. $0.419 (`p158-5090-1`).

### 5. CUDA-graph decode against the eager path

- **Claim, speed:** `e4b.serve.p109.decode-graphs-vs-eager-default.qwen3.5090.2026-10-03`. With `E4B_PAGED_GRAPHS=1`,
  sixteen requests decode **5.60×** the eager rate (9.02× at one request).
- **Its own verdict was DIVERGENT:** only 9 of 16 rows agreed with the eager path for their first 16 tokens (the bar
  was 12). Graphs became the default (`E4B_PAGED_GRAPHS=auto`, #770) only after a separate quality read,
  `e4b.serve.p110.graph-arithmetic-quality.qwen3.5090.2026-10-03`, read `AT_PARITY` (bias +0.00036 nats against a 0.0118 bar)
  ([`bench/p110/RESULTS-p110.md`](../bench/p110/RESULTS-p110.md)).
- **Limits:** the eager path is host-bound, so the ratio moves with the CPU. One family, the NF4 stack, `max_seqs` 16.
- **Stack:** e4b `3185b46ab041666894fd776fed904a6a96f1c396` (first released in v0.43.0), grouped-nf4-gemm
  `51a49166ae7bc1a0f84188b7b5d1f42ecbc37e00` (v0.35.0), torch 2.8.0+cu128, transformers 5.17.0.
- **Run:** `E4B_RENT_RUN_ID=p109-5090-<N> bash bench/p109/p109_drive.sh`. Record:
  [`bench/p109/RESULTS-p109.md`](../bench/p109/RESULTS-p109.md).
- **Hardware and cost:** one RTX 5090. $0.2926 (`p109-5090-2`).

### 6. vLLM is ahead

- **Claim:** `e4b.serve.h2h.vllm-0.30.0.p58.qwen3.b1.5090.2026-09-22` and the `.b16` sibling. vLLM 0.30.0's decode
  rate over e4b's int4 stack: **1.087** at one request and **1.396** at sixteen (above 1 means vLLM is faster).
- **Limits:** this compares against vLLM 0.30.0, read on 2026-09-22; a later vLLM is a new reading. vLLM ran
  `Qwen/Qwen3-30B-A3B-GPTQ-Int4` at `9b534e4318b7ebc3c961a839f13eb18b1833f441`, with `max_model_len` 2048 and
  `gpu_memory_utilization` 0.90. The engine gap is understated, not inflated: vLLM's slope includes its scheduler and
  detokenisation. Quality is quoted, not equated. TTFT, other batch sizes and footprint were not measured.
- **Stack:** e4b `4cab05a81dcf419daa8035d1f7142b266a51f37d` (first released in v0.37.0), grouped-nf4-gemm
  `65cb10410239120d25307812827a23139e84abe1` (v0.33.0), torch 2.8.0+cu129, transformers 5.16.1; vLLM 0.30.0 on torch
  2.13.0+cu130 in its own venv.
- **Run:** `E4B_RENT_RUN_ID=p58-5090-<N> bash bench/p58/p58_drive.sh`. Record:
  [`bench/p58/RESULTS-p58.md`](../bench/p58/RESULTS-p58.md).
- **Hardware and cost:** one RTX 5090. $0.31 (`p58-5090-1`).

## More readings

<!-- docs-upkeep: the linked table (rows 5, 7-8, 10-14, 16, 18-21 of the inventory and gnf4's G1-G4 / loggetta's L1)
     follows in the next commit, after the lanes' peer reviews of these six. -->

## How to report a reading

Open an issue with:
- the claim id;
- your commit (or release) for e4b and grouped-nf4-gemm, torch, transformers, the competitor's version, the GPU with
  its power limit, and the host CPU;
- the receipt directory the controller wrote.

We add it to the record beside ours, whichever way it reads.
