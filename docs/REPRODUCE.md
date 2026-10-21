# Reproducing the published numbers

This page is for someone outside the project who wants to check a published number. Each row gives the claim, the
stack it was measured on, the command, the hardware, and what one reading cost. Losses and limits stand beside the
wins. Every number comes from [`claims.json`](claims.json) by id, or is marked UNVERIFIED with its record.

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
- **Cost** is what one binding reading cost on a rented box, from its receipt, not the lane total.

### Driving a lane on your own GPU box

Every lane runs through its controller (`bench/<lane>/<lane>_drive.sh`, or `bench/tc1/tc1_drive.sh` for the training
head-to-heads). The controller stages hash-pinned files to any GPU box you can ssh to, runs the box script, and fetches
the receipts. From a checkout of the row's commit:

```bash
export E4B_RENT_SSH_HOST=<box> E4B_RENT_SSH_PORT=<port> \
       E4B_RENT_SSH_OPTS='-o StrictHostKeyChecking=accept-new' \
       E4B_RENT_RUN_DIR=$PWD/out E4B_RENT_RUN_ID=<a run id the controller accepts> \
       E4B_RENT_DEADLINE_EPOCH=$(($(date +%s)+5*3600)) E4B_RENT_INSTANCE_ID=local
```

Run ids differ by controller: some accept only their registered ids (P158's accepts `p158-5090-1` to `-3`), others
check none. Each row's command gives one that works. Every variable a row's command sets matters: a controller's
defaults are not the run's settings. The controller reads a Hugging Face token from `HF_TOKEN_FILE` (default
`~/.config/hf/token`). The box needs the model's disk and the host floor named in the lane's PREREG.

## Where to start

### 1. Qwen3-30B-A3B QLoRA-trains on a 12 GB card

- **Claim:** `e4b.train.frontier.qwen3.a2000-12gb.2026-10-02`. **10.463 GB** peak allocated: TC1's matched set
  (642,514,944 fp32 adapters, seq 2048, r 16) at micro-batch 1 × accum 8, with `OFFLOAD_EXPERTS=1`, on an RTX A2000
  12 GB. Held-out 1.9533 → 0.8483 at N=20. One draw.
- **Limits:** micro-batch 2 (the field recipe) ran out of memory on that card, which sidecars shared (1.37 GB held at
  start); an unshared 12 GB card is marginal there. Unsloth 2026.9.14 refused at load (bitsandbytes dispatched
  modules to the CPU, and its loader has no expert-offload lever), and HF + PEFT ran out of memory at load. The older,
  lighter 7.16 GB reading (`e4b.offload.fits-30b-class`) recorded no model revision or run id and was not reproduced.
- **Stack:** e4b `516c87c0aa4253ed8ac6a683ee96fad69be58ba8` (first released in v0.40.0), grouped-nf4-gemm
  `846b512b905468c08f5748943d08769b572affa2` (v0.34.0), torch 2.8.0+cu128. Model `Qwen/Qwen3-30B-A3B` at
  `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
- **Run:** a hand run on the owned card, `TC1_LOCAL_BOX=1 … TC1_FAMILIES=qwen3frontier12 bash tc1_run.sh`. The full
  command, its staged files and the snapshot layout are in [`bench/tc1/README.md`](../bench/tc1/README.md) ("TC3 hand
  run on the owned box"). Receipts: [`bench/h2h-2026-10-02/tc3/`](../bench/h2h-2026-10-02/tc3/).
- **Cost:** $0 (the owned card). Any 12 GB CUDA card can host it; it is a fit reading, not a speed one.

### 2. Qwen3-30B-A3B against Unsloth, packed 4,096-token rows, with e4b's memory cost beside it

- **Claim:** `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.packed-4k-device-ratio`. Unsloth spends **1.160×** e4b's GPU
  time per profiled step (medians of two draws), both frameworks at their defaults on one stack (read 2026-10-08). The
  wall-clock ratio on that host was 1.472.
- **Beside it, e4b's loss:** `e4b.train.memory.packed-4k-position-defaults.5090.2026-10-07`. At its defaults e4b's
  training phase peaks **1.04 GB above** Unsloth's (25.904 vs 24.864 GB, medians of two draws, read 2026-10-07). The
  opt-in `E4B_CKPT_OFFLOAD=1` brings the gap to 0.303 GB but costs 1.041× the step, so it stays opt-in.
- **Stack:** e4b `0b2995236f7a5d446299a252db0a7beb1d212280` (first released in v0.49.0), grouped-nf4-gemm
  `8944aa84857349cb98dce30cdf21f56c5a3640b6` (v0.43.0); Unsloth 2026.9.14, unsloth_zoo 2026.9.9, torch 2.12.1+cu130,
  transformers 5.5.0, `grouped_mm` MoE backend. Model as in row 1. The memory row ran at e4b
  `08ff61cd9acc2b8c758a02c414ab5670d550d9ad` (v0.49.0), grouped-nf4-gemm `9953bab9a94f0d5b3c932fe2e90c22f927d4b550`.
- **Run:**
  ```bash
  GNF4_SHA=8944aa84857349cb98dce30cdf21f56c5a3640b6 TC1_BOX=A TC1_FAMILIES=qwen3pos68 TC1_SEQ=4096 TC1_MB=1 \
    TC1_ACCUM=4 TC1_PACK=1 TC1_FREE_OUTPUTS=1 TC1_STEPS=40 TC1_EVAL_EVERY=40 bash bench/tc1/tc1_drive.sh
  ```
  The memory row is `TC1_FAMILIES=qwen3pos67` with the same knobs and `GNF4_SHA=9953bab9a94f0d5b3c932fe2e90c22f927d4b550`.
  Records: [`bench/h2h-2026-10-02/tc1/`](../bench/h2h-2026-10-02/tc1/README.md).
- **Hardware and cost:** one RTX 5090 and TC1's registered pre-flight floor: at least 320 GB disk, 98 GB host RAM and
  40 MB/s ([`TC1-PREREG.md`](../bench/tc1/TC1-PREREG.md), "Budget and stop rules"). The boxes had 251-504 GB RAM. $1.96
  (`tc1-5090-136`) and $1.34 (`tc1-5090-135`).
- **The other recipe:** at TC1's field recipe (seq 2048, micro-batch 2 × accum 4) the device ratio read 1.924 (medians
  of two draws; `e4b.train.h2h.unsloth.qwen3.5090.2026-10-08.field-device-ratio`), and a September primary pair on an
  older stack read 1.413 (a median of steps 11..60; `e4b.train.h2h.unsloth.qwen3.5090.2026-09-05`). Name the recipe when
  you quote one.

### 3. Mixtral-8x7B against Unsloth: which stack decides it

- **Claim, the matched position:** `e4b.train.h2h.unsloth.mixtral.5090.2026-10-05.same-stack`. With both frameworks
  resident on one stack, on one AMD EPYC 7B13 host, Unsloth/e4b is **1.144** [1.140, 1.148] (s/step, medians of steps 11..60 over two draws each), so e4b is faster per step.
  Beside it, in Unsloth's favour: Unsloth's peak is 2.07 GB lower (29.17 vs 31.24 GB; e4b's fp32 absmax), and its
  energy per step is ×0.93 of e4b's. On the same box, with e4b on the field image's stack (transformers 5.18.0), the
  ratio read **1.013**.
- **Unsloth faster:** `e4b.train.h2h.unsloth.mixtral.5090.2026-10-04.dense-default`. Both resident, e4b at its
  defaults in the field image's environment, on an Intel Core Ultra 9 285K host: Unsloth/e4b **0.836** [0.828, 0.844] (medians of steps 11..20 over two draws each),
  so Unsloth was faster per step. e4b `ed08029d9fb8d210ac78e9371f64bd06098ed07d` (first released in v0.46.0). $0.60.
- **A footprint lever, not a race:** `e4b.train.h2h.unsloth.mixtral.5090.2026-09-06` reads 0.361 (median of steps
  11..60): e4b ran under expert CPU offload at 3.2 GB, against Unsloth resident at 29.2 GB. Cost UNVERIFIED (it predates
  the private receipt store).
- **Stack (the matched position):** e4b `63400e1ab0af233aeb09e9e59017492da8ca3694` (first released in v0.49.0),
  grouped-nf4-gemm `9622144c734395de4d40332f06dea92135d858de` (v0.42.0), Unsloth 2026.9.14, torch 2.12.1+cu130,
  transformers 5.5.0: e4b ran inside Unsloth's environment. The September row ran e4b 0.35.1 and grouped-nf4-gemm
  0.30.2 from PyPI, Unsloth 2026.9.2 and torch 2.8.0. Model `mistralai/Mixtral-8x7B-Instruct-v0.1` at
  `eba92302a2861cdc0098cc54bc9f17cb2c47eb61`.
- **Run:** the run's own command, from its receipt:
  ```bash
  GNF4_SHA=9622144c734395de4d40332f06dea92135d858de TC1_BOX=A TC1_FAMILIES=mixtralsamestack TC1_STEPS=60 \
    TC1_LOAD_GATE=6.0 TC1_LOAD_RETRIES=2 TC1_SKIP="mixtralsamestack/e4b/reference_attn4_m" bash bench/tc1/tc1_drive.sh
  ```
  At this commit the controller's defaults differ (grouped-nf4-gemm `846b512b`, 20 steps), so set every variable shown.
  Records: [`bench/h2h-2026-10-02/tc2/`](../bench/h2h-2026-10-02/tc2/README.md). The September row is
  [`bench/h2h-20260906/tp2/`](../bench/h2h-20260906/tp2/README.md).
- **Hardware and cost:** one RTX 5090 and a 192 GB host floor. $1.59 (`tc1-5090-84`). As the claim says, these
  positions are within-box readings of their host, so name the host when you quote one.

### 4. K25's byte-permute decode, the qwen3_moe default since 0.56.0

- **Claim:** `e4b.serve.p158.k25-prmt.qwen3.5090.2026-10-20`. Sixteen concurrent requests decode **1.0925×** as fast
  (rate ratio, the smallest of four ABBA pairs), and the tokens are identical. One request is unchanged.
- **Limits:** one card, one family, two widths (W16 and W1), the K25 plan as of 2026-10-20. Nothing about quality
  beyond token identity, and nothing exact about clocks.
- **Stack:** e4b `406a391a11d42ddb39154cfc36938a2da5d68216` (first released in v0.56.0; v0.55.0 has no `E4B_K25_LUT`),
  grouped-nf4-gemm v0.50.0 (`41ef086e`), torch 2.8.0+cu128, transformers 5.17.0. Model as in row 1.
- **Run:** `E4B_RENT_RUN_ID=p158-5090-1 bash bench/p158/p158_drive.sh` (this controller accepts only `p158-5090-1` to
  `-3`). The arms differ only in `E4B_K25_LUT` (`tree` vs `prmt`). Record:
  [`bench/p158/RESULTS-p158.md`](../bench/p158/RESULTS-p158.md).
- **Hardware and cost:** one RTX 5090 at 575 W. $0.419 (`p158-5090-1`).

### 5. CUDA-graph decode against the eager path

- **Claim, speed:** `e4b.serve.p109.decode-graphs-vs-eager-default.qwen3.5090.2026-10-03`. With `E4B_PAGED_GRAPHS=1`,
  sixteen requests decode **5.60×** the eager rate (the smaller of two pairs; 9.02× at one request).
- **Its own verdict was DIVERGENT:** only 9 of 16 rows agreed with the eager path for their first 16 tokens (the bar
  was 12). Graphs became the default (`E4B_PAGED_GRAPHS=auto`, #770) only after a separate quality read,
  `e4b.serve.p110.graph-arithmetic-quality.qwen3.5090.2026-10-03`, read `AT_PARITY` (bias +0.00036 nats against a
  0.0118 bar) ([`bench/p110/RESULTS-p110.md`](../bench/p110/RESULTS-p110.md)).
- **Limits:** the eager path is host-bound, so the ratio moves with the CPU; the reading's host was an AMD EPYC 7C13.
  One family, the NF4 stack, `max_seqs` 16.
- **Stack:** e4b `3185b46ab041666894fd776fed904a6a96f1c396` (first released in v0.43.0), grouped-nf4-gemm
  `51a49166ae7bc1a0f84188b7b5d1f42ecbc37e00` (v0.35.0), torch 2.8.0+cu128, transformers 5.17.0.
- **Run:** `E4B_RENT_RUN_ID=p109-5090-2 bash bench/p109/p109_drive.sh`. Record:
  [`bench/p109/RESULTS-p109.md`](../bench/p109/RESULTS-p109.md).
- **Hardware and cost:** one RTX 5090. $0.2926 (`p109-5090-2`).

### 6. vLLM is ahead

- **Claim:** `e4b.serve.h2h.vllm-0.30.0.p58.qwen3.b1.5090.2026-09-22` and the `.b16` sibling. vLLM 0.30.0's decode
  rate over e4b's int4 stack: **1.087** at one request and **1.396** at sixteen (above 1 means vLLM is faster).
- **Limits:**
  - This compares against vLLM 0.30.0, read on 2026-09-22; a later vLLM is a new reading.
  - vLLM ran `Qwen/Qwen3-30B-A3B-GPTQ-Int4` at `9b534e4318b7ebc3c961a839f13eb18b1833f441`, with `max_model_len` 2048
    and `gpu_memory_utilization` 0.90.
  - e4b's int4 stack is P54's opt-in flags, which the lane sets itself (`bench/p58/p58_run.sh`); a plain
    `pip install` runs a different stack.
  - The host was an AMD EPYC 9655, and at one request both engines were host-bound there.
  - The engine gap is understated, not inflated: vLLM's slope includes its scheduler and detokenisation. Quality is
    quoted, not equated. TTFT, other batch sizes and footprint were not measured.
- **Stack:** e4b `4cab05a81dcf419daa8035d1f7142b266a51f37d` (first released in v0.37.0), grouped-nf4-gemm
  `65cb10410239120d25307812827a23139e84abe1` (v0.33.0), torch 2.8.0+cu129, transformers 5.16.1; vLLM 0.30.0 on torch
  2.13.0+cu130 in its own venv.
- **Run:** `GNF4_SHA=65cb10410239120d25307812827a23139e84abe1 E4B_RENT_RUN_ID=p58-5090-1 bash bench/p58/p58_drive.sh`
  (this controller refuses without `GNF4_SHA`). Record: [`bench/p58/RESULTS-p58.md`](../bench/p58/RESULTS-p58.md).
- **Hardware and cost:** one RTX 5090. $0.31 (`p58-5090-1`).

## More readings

Same rules as above: each row's commit, first release and cost come from its record. Controllers differ, so read the
lane's drive script before running it:
- **Build from the checkout's HEAD and refuse an `E4B_SHA` override** (P143, P153, P155, P161): check out the recorded
  commit and run from there.
- **Build from HEAD but accept an `E4B_SHA` override** (P135, whose run script also hard-pins its grouped-nf4-gemm):
  pass the recorded commit.
- **Hard-pin the e4b commit inside the drive** (P133, P137, P140, P144, P146): the drive can postdate the measured
  commit (P133's was added at its registration, `7896ce0e`), so run it from its registration commit or a later main;
  it installs the pinned commit itself.
- **Accept only their registered run ids** (P133, P135, P137, P140, P141, P143, P144, P146, P153, P155, P161), and **refuse to
  start without `GNF4_SHA`** (K15, K33, MG1, P58).

| Reading | Claim | Measured at e4b (first release) | GPU, host | Lane, run | Cost |
|---|---|---|---|---|---|
| Qwen3-30B-A3B trains TC1's matched set (fp32 adapters) at the field recipe under expert offload on a 24 GB card: **11.881 GB** peak, 10.521 s/step (a fit, one draw; e4b as shipped has no offload reading there) | `e4b.train.frontier.qwen3.4090-24gb.2026-10-02` | `39834a7c` (v0.40.0), gnf4 `846b512b` | RTX 4090, AMD EPYC 7642 | TC3 via `bench/tc1/tc1_drive.sh`, `tc3-4090-8` | $0.75 |
| axolotl native-best / e4b as shipped at steady state, read 2026-10-03 before the later defaults: **1.238** (s/step, median of steps 101..200 per draw, over two draws; above 1 = e4b faster); 1.146 on a second host. Held-out at step 200, reported, not judged: axolotl 0.7705 / 0.7704 against e4b 0.7948 / 0.7937 | `e4b.train.h2h.axolotl.qwen3.5090.2026-10-03.native-steady-state` | `e5859e99` (v0.42.0), gnf4 `00929a49` (pass it: the drive's default differs) | RTX 5090, AMD EPYC 7663 (1.238) / 7C13 (1.146) | TC1, `tc1-5090-46` + `-47` (1.238's two draws), `tc1-5090-48` (1.146) | $0.99 + $0.98; $1.23 |
| Layer-major offload training, stock / layer-major median s/step (the smaller of two ABBA pairs, 32 timed steps per arm, one process for the first three): **2.365** (Qwen3, 4090), **2.311** (Qwen3, 5090), **2.118** (Mixtral, 4090); the CLI's `FAST_TRAIN=1` over `LAYER_MAJOR=1`, one fresh process per arm: **1.455** | `e4b.train.p137.layer-major-offload.qwen3.4090.2026-10-18`, `e4b.train.p140.layer-major-offload.qwen3.5090.2026-10-18`, `e4b.train.p144.layer-major-offload.mixtral.4090.2026-10-18`, `e4b.train.p146.layer-major-cli.qwen3.4090.2026-10-18` | P137 and P140 `b4366b9a`, P144 `a9d140d8`, P146 `4917506a` (all v0.56.0), gnf4 `6b7c2e74` | RTX 4090 / 5090; EPYC 7742, Ryzen 7 5700X3D, EPYC 7B13 | P137, P140, P144, P146 | $0.745, $0.489, $1.167, $0.819 |
| OLMoE fused training vs the reference loop: **|Δ final loss| 0.0089** (band 0.05; one sha-pinned dataset, batch 1, seq 512, 60 steps) | `e4b.train.parity.mg1.olmoe.fused.2026-10-04` | `058dda34` (v0.46.0), gnf4 `1546db26` | RTX 5090 at 470 W, Core Ultra 9 285K | MG1, `mg1-5090-2` | $1.79 |
| int8 head bandwidth GEMV, one request: **1.052** (rate, the smallest of four ABBA pairs; outputs change). Sixteen requests read **0.9918**, below 1 in every pair: an observed cost the record leaves unexplained | `e4b.serve.p153.int8-head-bw.qwen3.5090.2026-10-19` | `585cf68b` (v0.56.0) | RTX 5090 at 575 W, Ryzen 7 9800X3D | P153, `p153-5090-1` | $0.445 |
| Attention split cap, one request: **1.0442** (decode rate, the smallest of four ABBA pairs) | `e4b.serve.p143.attn-split-cap.qwen3.5090.2026-10-19` | `e8de1799` (v0.56.0) | RTX 5090 at 450 W, EPYC 9135 | P143 (Amendment 3), `p143a3-5090-1` | $0.479 |
| Launch-bound glue, one request: **1.1321** (rate, the mean of two pairs; Phase 1 + 2 against the stack before #1448) | `e4b.serve.p127.launch-bound-glue.qwen3.5090.2026-10-09` | B `7f044dd9` (v0.52.0) vs A `a8c01d42` (v0.51.0) | RTX 5090, Ryzen 9 9950X | P127, `p127-5090-1` | $0.940 |
| Bandwidth decode GEMV, one request: **1.2417** over dot-pad (5090), **1.5145** over the scalar GEMV (4090, at Qwen3's two shapes); each the smaller of two ABBA pairs | `e4b.serve.p116.gemv-bw.qwen3.5090.2026-10-07`, `e4b.serve.p133.gemv-bw.qwen3.4090.2026-10-18` | `730c8acc` (v0.49.0); `c8d7e629` (v0.55.0) | RTX 5090 at 575 W, EPYC 7713; RTX 4090 at 450 W, EPYC 7742 | P116, `p116-5090-2`; P133, `p133-4090-2` | $1.078; $0.919 |
| Bandwidth decode GEMV on Mixtral, one request: **2.4075** over the scalar GEMV (decode rate, the smaller of two ABBA pairs) | `e4b.serve.p141.gemv-bw.mixtral.5090.2026-10-18` | `d2bf9f64` (v0.56.0) | RTX 5090 at 500 W, Xeon Gold 6240 | P141, `p141-5090-2` | $0.999 |
| Fused-stack folds on Mixtral: median step time **0.9393** ON/OFF, the larger of two interleaved blocks (below 1 = faster; 20.2-20.5 → 19.0 ms) | `e4b.serve.p135.fold-speed.mixtral.5090.2026-10-18` | `5be4c4d9` (v0.56.0) | RTX 5090, EPYC 7742 | P135, `p135-5090-2` | $1.159 |
| Server capacity with 64 slots and `E4B_PAGED_BUCKETS=auto`, on the int4 stack (int4 experts and attention, fused QKV, 2,048 tokens a slot): **12 req/s** at TTFT ≤ 1 s and TPOT ≤ 100 ms for ≥ 95 % in both Poisson draws, against 4 for the 16-slot server. No comparator ran, and it is not the default NF4 server | `e4b.serve.sc2e.64-slots-buckets-auto.qwen3.5090.2026-10-07` | `3bbcb01b` (v0.49.0) | RTX 5090 at 575 W, EPYC 7352 | SC2e via `bench/sc1/sc1_drive.sh` (`SC1_BOX=L`), `sc2e-5090-1` | $1.017 |
| **A negative, since overtaken:** K25's plan under the select-tree decode: best plan 1.0188 against a 1.15 bar, NO_GO. Under K39's `prmt` decode, P160 then read plan `c1` at 1.66× in a microbench ([`RESULTS-p160`](../bench/p160/RESULTS-p160.md)); the served read is P161 (the next row) | UNVERIFIED: no register row; [`RESULTS-p155`](../bench/p155/RESULTS-p155.md) | `4d3fd170` (v0.56.0) | RTX 5090 at 575 W, EPYC 7K62 | P155, `p155-5090-1` | $0.274 |
| K25 at plan `c1` under K39's `prmt` decode, served on the shipped default server for Qwen3-30B-A3B: sixteen requests **1.4654×** as fast as the `today` plan (the smallest of four ABBA pairs), one request unchanged (0.9966), every token identical. `qwen3_moe`'s default from v0.56.1 | `e4b.serve.p161.k25-plan-c1.qwen3.5090.2026-10-20` | `da3eb5c5` (v0.56.1) | RTX 5090 at 575 W, EPYC 7K62 | P161, `p161-5090-1` | $0.726 |
| **A quality limit:** Gemma-4 served NF4 vs bf16, KL **1.077** nats/token (a served-model defect on this family, #597; no Gemma-4 serving position) | `e4b.serve.p44.gemma4.nf4-vs-bf16.2026-09-19` | not recorded; gnf4 v0.31.0 | H100 NVL | P44-b (`bench/p44/p44b_drive.sh`); three runs agree | UNVERIFIED (lane ≈ $8.8) |

### grouped-nf4-gemm and the planner

| Reading | Claim | Measured at | GPU, host | Lane, run | Cost |
|---|---|---|---|---|---|
| Bandwidth decode GEMV kernel time **0.341** of dot-pad on Qwen3-30B-A3B's gate_up + down pair (below 1 = faster; kernel level, not a served speed) | `gnf4.kernel.k33-nf4-decode-gemv-bw.5090.2026-10-07` | gnf4 `5a60c37d` (v0.43.0) | RTX 5090 at 600 W, Threadripper PRO 7965WX | K33 via e4b `bench/k33/k33_drive.sh` (needs `GNF4_SHA`), `k33-5090-1` | $0.037 |
| Routed expert GEMM against unquantised bf16 `torch._grouped_mm` (Qwen3-30B-A3B, B=16): **2.1-6.0×** faster, against weights at higher precision: not a same-precision race (quote the worst case, 2.1×) | `gnf4.kernel.sm120-census-vs-grouped-mm` | gnf4 `4f1db333` (v0.18.0) | RTX 5090, two boxes | the census probes in `bench/sm120-census/` | UNVERIFIED |
| **A negative:** vLLM 0.28.0's Marlin is **1.99×** faster than grouped-nf4-gemm's own bf16 dequant path on o_proj at M=16 (16.5 → 8.28 µs; q_proj 1.62×), and still ahead at matched bytes | `gnf4.kernel.k15-marlin-comparator.5090.2026-09-11` | gnf4 `453bebf5` (v0.31.0) | RTX 5090 at 550 W | K15 via e4b `bench/k15/k15_drive.sh` (needs `GNF4_SHA`), `k15-marlin-4` | lane $1.1253 (over its $1.00 ceiling) |
| One-launch training backward: relative error **0.0029** against the exact per-expert loop (correctness, not speed) | `gnf4.kernel.dgrad` | gnf4 v0.7.0 | RTX A2000 | the 0.7.0 release entry | $0 |
| loggetta's planner memory estimate on Qwen3-30B-A3B resident: 22.09 vs 21.91 GiB measured (-0.8 %) | UNVERIFIED: loggetta has no register; its `docs/RESULTS.md` §5 | loggetta (`fp1-5090-1`) | RTX 5090 | FP1, `fp1-5090-1` | $0.848 |

## How far a number travels between boxes

- **Within one box, a ratio repeats to under 0.5 %.** The four ABBA pair ratios in each of P143, P149, P153, P158 and P161
  agree to a median 0.08 %, and at most 0.46 % (P161).
- **Across boxes, an absolute moves and a GPU-bound ratio holds.** P116's read and its reported second draw (void as a
  reading), on two rented RTX 5090s, differ by 15-18 % in absolute rate and by 2 % in the ratio: 1.2681 against 1.2417
  (the smaller of two ABBA pairs; `e4b.serve.p116.gemv-bw.qwen3.5090.2026-10-07`).
- **A ratio with host-side work in it moves too.** TC1's axolotl/e4b steady-state ratio (medians of two draws) read
  1.238 on one host and 1.146 on another (`e4b.train.h2h.axolotl.qwen3.5090.2026-10-03.native-steady-state`). Where
  e4b's step is host-launch-bound it read 2.775 against 1.979 (medians of two draws)
  (`e4b.train.h2h.axolotl.qwen3.5090.2026-10-03`).

Compare ratios read on one box. An absolute time from another box is that host's number, and so is a ratio whose path
includes CPU-side work.

## How to report a reading

Open an issue with:
- the claim id;
- your commit (or release) for e4b and grouped-nf4-gemm, torch, transformers, the competitor's version, the GPU with
  its power limit, and the host CPU;
- the receipt directory the controller wrote.

A reported reading goes into the record beside the published one, whichever way it reads.
