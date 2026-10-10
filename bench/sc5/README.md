# SC5: same-box serving on one RTX 5090, e4b against vLLM 0.31.0 and SGLang 0.5.21 on Qwen3-30B-A3B (read)

The registration is `SC5-PREREG.md` (experts4bit-qlora#1478 item 4; the serving campaign #846). This file reads its one
reading, `sc5-5090-2`, by the registered rule (`sc5_reduce.py`). The reading has no single headline. Each cell is read
on its own, and every losing cell is listed below.

**e4b's out-of-box serving (pip install, no environment) is NF4, and it is not measured here.** The e4b arm is the
documented int4 serving configuration in the registration's arm table.

**Formats, on which the quality rows depend.**
- vLLM and SGLang serve the one shared official checkpoint, `Qwen/Qwen3-30B-A3B-GPTQ-Int4` @ `9b534e43` (GPTQ 4-bit
  g128, Marlin kernels), the 4-bit format SC2 served on both.
- e4b serves its documented int4 serving configuration: int4-b32 at 4.50 bpw, packed on the box from the same bf16
  weights, `Qwen/Qwen3-30B-A3B` @ `ad44e777`.

## Runs

| run | what | outcome | cost |
|---|---|---|---|
| `sc5-prove-1` | proof | NOT_RUN: the box refused the ssh key | $0.001 |
| `sc5-prove-2` | proof | refused at launch: the receipt store's head was not its upstream (no rental) | $0.000 |
| `sc5-prove-3` | proof | rc 11: harness defects, fixed in #1530 | $0.445 |
| `sc5-prove-4` | proof | rc 23: harness defects, fixed in #1536 | $0.905 |
| `sc5-prove-5` | proof | PROVED; the SGLang capacity readout was fixed in #1547 | $1.531 |
| `sc5-ref-1` | the bf16 reference | OK, committed by #1557 | $0.367 |
| `sc5-5090-1` | reading | NOT_RUN: ssh never authenticated (host fault) | $0.031 |
| `sc5-5090-2` | **reading** | **READ** | $1.845 |

The lane spent $5.125 of its $10.84 ceiling.

**The reading's box.**
- One RTX 5090 (32,607 MiB, power limit 575 W, driver 590.48.01) with 128.8 GB of host RAM.
- Launched from main `53abde34`: the registration, #1547's readout fix and cost guard, #1551's lane liveness and the
  reference amendment.
- The stacks, from `versions.txt`:
  - experts4bit-qlora 0.52.0 and grouped-nf4-gemm 0.45.0, the release wheels by sha256, on torch 2.8.0+cu128;
  - vLLM 0.31.0 and SGLang 0.5.21 (tag `e00930c5`), from their hash locks, on torch 2.13.0+cu130.
- All 24 blocks ran: 2 draws × 2 memory settings × 6 cold-started blocks, in ABBA order. No block or quality row is
  VOID.

## Capacity readouts

| setting | e4b | vLLM | SGLang |
|---|---|---|---|
| default | 131,072 | 128,160, 131,888 | 88,303 |
| matched | 65,536 | 65,536 | 65,536 |

At the matched setting every framework read the registered 65,536 tokens, so the cost guard (#1547) let the reading
continue after block 9. At the default setting, the readout is each framework's own sizing:
- e4b: `E4B_PAGED_MAX_SEQS=auto` resolved to 32 slots × 4,096 tokens on this card;
- vLLM: its default GPU-memory utilization, which its `/metrics` reports as 0.92 at v0.31.0. The registration's table
  expected 0.9; the arm is the framework's default either way;
- SGLang: its default static memory fraction, 0.777 on this card (its server info).

## Measured values

Each cell is the median over the framework's four blocks (two draws × two blocks); the block range is in brackets.

### Default setting

| C | framework | output tok/s | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p50 (ms) | TPOT p95 (ms) |
|---|---|---|---|---|---|---|
| 1 | e4b | 234 [232–234] | 43.5 [43.4–43.8] | 43.8 [43.7–44.2] | 4.12 [4.11–4.17] | 4.15 [4.13–4.22] |
| 1 | vLLM | 226 [226–227] | 54.6 [54.4–55.2] | 57.0 [55.7–58.5] | 4.22 [4.21–4.23] | 4.23 [4.21–4.23] |
| 1 | SGLang | 263 [263–264] | 33.3 [33.2–33.4] | 33.8 [33.7–34.0] | 3.68 [3.68–3.68] | 3.68 [3.68–3.69] |
| 16 | e4b | 1315 [1307–1317] | 102.4 [102.4–102.5] | 103.7 [103.4–103.9] | 11.74 [11.69–11.78] | 11.97 [11.94–11.98] |
| 16 | vLLM | 1666 [1665–1669] | 153.8 [152.8–154.3] | 225.1 [222.8–226.2] | 9.00 [8.99–9.01] | 9.26 [9.24–9.27] |
| 16 | SGLang | 1678 [1677–1685] | 235.1 [190.7–279.4] | 305.8 [305.5–306.6] | 8.57 [8.52–8.66] | 9.26 [9.13–9.31] |
| 64 | e4b | 1618 [1612–1621] | 4528.0 [4503.1–4536.3] | 4599.9 [4568.1–4611.5] | 17.29 [17.24–17.33] | 17.62 [17.49–17.67] |
| 64 | vLLM | 3475 [3472–3477] | 270.3 [270.0–270.7] | 358.4 [358.2–358.9] | 17.17 [17.16–17.18] | 17.69 [17.69–17.70] |
| 64 | SGLang | 1258 [1244–1260] | 688.4 [687.2–689.6] | 1132.1 [1115.1–1136.9] | 48.39 [48.27–49.08] | 50.50 [50.44–51.10] |

### Matched setting (65,536 KV tokens: 64 slots × 1,024)

| C | framework | output tok/s | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p50 (ms) | TPOT p95 (ms) |
|---|---|---|---|---|---|---|
| 1 | e4b | 243 [243–243] | 43.5 [43.4–43.9] | 44.0 [43.7–44.3] | 3.95 [3.95–3.96] | 4.01 [3.98–4.01] |
| 1 | vLLM | 226 [226–226] | 54.6 [54.4–54.9] | 56.7 [56.5–57.3] | 4.23 [4.23–4.23] | 4.23 [4.23–4.24] |
| 1 | SGLang | 263 [263–263] | 33.3 [33.2–33.3] | 33.9 [33.7–34.1] | 3.68 [3.68–3.69] | 3.69 [3.69–3.69] |
| 16 | e4b | 1317 [1309–1319] | 102.0 [101.6–102.0] | 103.1 [102.9–103.3] | 11.77 [11.64–11.87] | 11.95 [11.89–12.05] |
| 16 | vLLM | 1668 [1665–1670] | 153.4 [153.3–153.7] | 223.7 [222.3–224.9] | 8.99 [8.97–9.01] | 9.24 [9.20–9.27] |
| 16 | SGLang | 1679 [1676–1681] | 213.5 [191.5–265.1] | 306.5 [303.1–308.7] | 8.63 [8.62–8.72] | 9.24 [9.23–9.27] |
| 64 | e4b | 2402 [2389–2405] | 113.5 [113.4–113.6] | 115.1 [114.8–115.8] | 25.53 [25.44–25.58] | 25.59 [25.53–25.69] |
| 64 | vLLM | 3477 [3476–3479] | 270.4 [269.9–270.8] | 358.5 [357.8–358.7] | 17.16 [17.16–17.17] | 17.69 [17.68–17.71] |
| 64 | SGLang | 1250 [1242–1257] | 677.2 [676.1–685.4] | 1199.2 [1197.5–1200.7] | 48.75 [48.48–49.01] | 50.88 [50.52–51.19] |

## The rule's labels

Each label reads from e4b's side; the same cell read from the competitor's side carries the opposite label. The ratio is
e4b's value over the competitor's, one per block: draw 1 (b1, b2) / draw 2 (b1, b2).
- For output tok/s, above 1 means e4b produced more.
- For TTFT and TPOT, below 1 means e4b was faster.
- A cell is LEADS or TRAILS only when every block clears the registered bound in the same direction: 5 % for tok/s and
  TPOT, 10 % for TTFT. Otherwise it is WITHIN NOISE.

Counts: **LEADS 24, TRAILS 30, WITHIN NOISE 6.** One cell was flagged noisy by its
own blocks: draw 2 matched C=16 ttft_p50_s: sglang blocks 0.265116 / 0.234419.

### Default setting

| C | metric | e4b vs vLLM | e4b vs SGLang |
|---|---|---|---|
| 1 | output tok/s | **WITHIN NOISE** (1.04, 1.03 / 1.03, 1.02) | **TRAILS** (0.89, 0.89 / 0.89, 0.88) |
| 1 | TTFT p50 | **LEADS** (0.79, 0.79 / 0.80, 0.80) | **TRAILS** (1.31, 1.30 / 1.31, 1.31) |
| 1 | TTFT p95 | **LEADS** (0.75, 0.77 / 0.77, 0.79) | **TRAILS** (1.30, 1.29 / 1.31, 1.29) |
| 1 | TPOT p50 | **WITHIN NOISE** (0.97, 0.98 / 0.97, 0.99) | **TRAILS** (1.12, 1.12 / 1.12, 1.13) |
| 1 | TPOT p95 | **WITHIN NOISE** (0.98, 0.98 / 0.99, 1.00) | **TRAILS** (1.12, 1.12 / 1.13, 1.14) |
| 16 | output tok/s | **TRAILS** (0.79, 0.79 / 0.78, 0.79) | **TRAILS** (0.78, 0.78 / 0.78, 0.78) |
| 16 | TTFT p50 | **LEADS** (0.67, 0.67 / 0.66, 0.67) | **LEADS** (0.54, 0.54 / 0.37, 0.37) |
| 16 | TTFT p95 | **LEADS** (0.46, 0.46 / 0.46, 0.46) | **LEADS** (0.34, 0.34 / 0.34, 0.34) |
| 16 | TPOT p50 | **TRAILS** (1.30, 1.31 / 1.31, 1.30) | **TRAILS** (1.35, 1.38 / 1.38, 1.36) |
| 16 | TPOT p95 | **TRAILS** (1.29, 1.29 / 1.29, 1.29) | **TRAILS** (1.29, 1.31 / 1.29, 1.29) |
| 64 | output tok/s | **TRAILS** (0.47, 0.47 / 0.47, 0.46) | **LEADS** (1.28, 1.29 / 1.29, 1.30) |
| 64 | TTFT p50 | **TRAILS** (16.79, 16.73 / 16.65, 16.76) | **TRAILS** (6.59, 6.58 / 6.53, 6.59) |
| 64 | TTFT p95 | **TRAILS** (12.87, 12.83 / 12.74, 12.82) | **TRAILS** (4.07, 4.06 / 4.10, 4.05) |
| 64 | TPOT p50 | **WITHIN NOISE** (1.01, 1.01 / 1.00, 1.01) | **LEADS** (0.36, 0.36 / 0.36, 0.35) |
| 64 | TPOT p95 | **WITHIN NOISE** (1.00, 1.00 / 0.99, 1.00) | **LEADS** (0.35, 0.35 / 0.35, 0.34) |

### Matched setting

| C | metric | e4b vs vLLM | e4b vs SGLang |
|---|---|---|---|
| 1 | output tok/s | **LEADS** (1.08, 1.08 / 1.08, 1.08) | **TRAILS** (0.92, 0.92 / 0.92, 0.92) |
| 1 | TTFT p50 | **LEADS** (0.80, 0.79 / 0.80, 0.80) | **TRAILS** (1.31, 1.30 / 1.31, 1.32) |
| 1 | TTFT p95 | **LEADS** (0.78, 0.76 / 0.78, 0.78) | **TRAILS** (1.29, 1.28 / 1.31, 1.31) |
| 1 | TPOT p50 | **LEADS** (0.93, 0.94 / 0.93, 0.93) | **TRAILS** (1.07, 1.07 / 1.07, 1.07) |
| 1 | TPOT p95 | **LEADS** (0.95, 0.95 / 0.95, 0.94) | **TRAILS** (1.09, 1.09 / 1.09, 1.08) |
| 16 | output tok/s | **TRAILS** (0.78, 0.79 / 0.79, 0.79) | **TRAILS** (0.78, 0.78 / 0.79, 0.79) |
| 16 | TTFT p50 | **LEADS** (0.66, 0.66 / 0.66, 0.67) | **WITHIN NOISE** (0.53, 0.53 / 0.38, 0.43) |
| 16 | TTFT p95 | **LEADS** (0.46, 0.46 / 0.46, 0.46) | **LEADS** (0.34, 0.34 / 0.33, 0.34) |
| 16 | TPOT p50 | **TRAILS** (1.32, 1.31 / 1.31, 1.29) | **TRAILS** (1.36, 1.36 / 1.37, 1.35) |
| 16 | TPOT p95 | **TRAILS** (1.31, 1.29 / 1.28, 1.30) | **TRAILS** (1.30, 1.29 / 1.29, 1.30) |
| 64 | output tok/s | **TRAILS** (0.69, 0.69 / 0.69, 0.69) | **LEADS** (1.94, 1.93 / 1.91, 1.91) |
| 64 | TTFT p50 | **LEADS** (0.42, 0.42 / 0.42, 0.42) | **LEADS** (0.17, 0.17 / 0.17, 0.17) |
| 64 | TTFT p95 | **LEADS** (0.32, 0.32 / 0.32, 0.32) | **LEADS** (0.10, 0.10 / 0.10, 0.10) |
| 64 | TPOT p50 | **TRAILS** (1.48, 1.49 / 1.49, 1.49) | **LEADS** (0.52, 0.52 / 0.53, 0.53) |
| 64 | TPOT p95 | **TRAILS** (1.44, 1.45 / 1.45, 1.45) | **LEADS** (0.50, 0.50 / 0.51, 0.51) |

### Cells where e4b trails (the 30 losing cells)

- default, C = 1, against SGLang: output tok/s, TPOT p50, TPOT p95, TTFT p50, TTFT p95
- default, C = 16, against vLLM: output tok/s, TPOT p50, TPOT p95
- default, C = 16, against SGLang: output tok/s, TPOT p50, TPOT p95
- default, C = 64, against vLLM: output tok/s, TTFT p50, TTFT p95
- default, C = 64, against SGLang: TTFT p50, TTFT p95
- matched, C = 1, against SGLang: output tok/s, TPOT p50, TPOT p95, TTFT p50, TTFT p95
- matched, C = 16, against vLLM: output tok/s, TPOT p50, TPOT p95
- matched, C = 16, against SGLang: output tok/s, TPOT p50, TPOT p95
- matched, C = 64, against vLLM: output tok/s, TPOT p50, TPOT p95

### Cells where e4b leads (24)

- default, C = 1, against vLLM: TTFT p50, TTFT p95
- default, C = 16, against vLLM: TTFT p50, TTFT p95
- default, C = 16, against SGLang: TTFT p50, TTFT p95
- default, C = 64, against SGLang: output tok/s, TPOT p50, TPOT p95
- matched, C = 1, against vLLM: output tok/s, TPOT p50, TPOT p95, TTFT p50, TTFT p95
- matched, C = 16, against vLLM: TTFT p50, TTFT p95
- matched, C = 16, against SGLang: TTFT p95
- matched, C = 64, against vLLM: TTFT p50, TTFT p95
- matched, C = 64, against SGLang: output tok/s, TPOT p50, TPOT p95, TTFT p50, TTFT p95

### Within noise (6)

- default, C = 1, against vLLM: output tok/s, TPOT p50, TPOT p95
- default, C = 64, against vLLM: TPOT p50, TPOT p95
- matched, C = 16, against SGLang: TTFT p50

## Two cells that need their setting stated

- **C = 64 at the default setting is a property of each framework's default setting.**
  - At its default, e4b 0.52.0 serves 32 slots (`E4B_PAGED_MAX_SEQS=auto` on this card; its `/health` reports 32 ×
    4,096). The registration starts vLLM with `--max-num-seqs 64` and SGLang with `--max-running-requests 64`.
  - With 64 requests in flight, half of e4b's wait for a slot. The default-setting rows at C = 64 measure that queueing.
  - At the matched setting, every framework serves 64 slots; those rows are in the matched table above. Both rows
    stand as measured.
- **SGLang at C = 64 is not diagnosed.**
  - In both settings, SGLang's TPOT at C = 64 is about 48–51 ms, against about 9 ms at C = 16. vLLM's is about 17 ms.
  - This reading does not say why. No census of SGLang's batch scheduling or graph configuration was registered or
    run.

## Quality against the one bf16 reference

Each row covers 64 windows × 128 positions = 8,192 positions, scored against the committed reference
`bench/sc5/ref/sc5_ref.json` (sha256 `783e1443`, verified on the box) over windows `5f6e00d8`.

| row | mean NLL delta (nats/position) | window range | argmax agreement | SC1 label |
|---|---|---|---|---|
| e4b, prefill-shaped (the cross-framework row) | +0.0219 | -0.147 to +0.136 | 0.8954 | FAR |
| e4b, decode-shaped (served graphs; reported) | +0.0232 | -0.156 to +0.134 | 0.8911 | FAR |
| vLLM (`prompt_logprobs`) | +0.0498 | -0.086 to +0.275 | 0.8964 | FAR |
| SGLang (`return_logprob`) | +0.0500 | -0.093 to +0.291 | 0.8960 | FAR |
| the ordering floor (bf16, chunked 256) | +0.0005 | -0.069 to +0.055 | 0.9722 | CLOSE |

- SC1's labels are CLOSE for a delta of at most 0.0095 nats and COMPARABLE for at most 0.02; anything larger is FAR.
- **FAR describes divergence from the bf16 reference, for all three engines, not a ranking of correctness.**
- The floor is the agreement of the bf16 model with itself when the same windows are scored in chunks of 256. Agreement
  below 100 % there comes from reordered MoE arithmetic flipping router choices. It is the floor that every argmax
  agreement is read against.

## What this reading cannot say

These were stated in the registration before any data:
- the competitors' decode-shaped arithmetic (not measured);
- whether e4b's server replays its first prefill piece through the same kernels as the eager scoring pass;
- e4b's NF4 out-of-box serving.

Added here: the cause of SGLang's C = 64 TPOT (not diagnosed).

## Reproduce

`receipts/sc5-5090-2/` holds:
- the record (`sc5.json`) and the box's verdict (`verdict.json`);
- the per-framework quality rows and comparisons (`quality/`);
- the box's `summary.txt`, `versions.txt` and `forensics.txt`;
- `SHA256SUMS`.

The per-block raw logs stay in the lane's private receipt store. `tests/test_sc5_read.py` re-runs main's reducer on the
committed record and requires the committed verdict, or by hand:

```
python bench/sc5/sc5_reduce.py --record bench/sc5/receipts/sc5-5090-2/sc5.json --out /tmp/verdict.json
cmp <(python -m json.tool /tmp/verdict.json) <(python -m json.tool bench/sc5/receipts/sc5-5090-2/verdict.json)
```

## Addendum (2026-10-10, after the read): SGLang's C = 64 cells

The read above calls SGLang's C = 64 TPOT not diagnosed. A later zero-rental source read accounts for it, and it changes
no label: every cell stands as measured.
- **The source.** SGLang 0.5.21's locked wheel (`sglang-0.5.21-cp312-cp312-manylinux_2_34_x86_64.whl`, sha256
  `ac300998…`) sets the default decode CUDA-graph `max_bs` to 48 in `sglang/srt/arg_groups/memory_hook.py` (lines
  100–116). That default applies on a card with 20 to 35 GiB at tensor parallel below 4, and the code's comment names
  the RTX 5090.
- **The box.** This reading's server info reports `cuda_graph_config.decode.max_bs` 48, with graphs captured for batch
  sizes 1–48.
- **The logs.** In blocks 3 and 9 (default and matched), every decode batch of 16 or fewer ran with the graph, and every
  batch above 48 ran without it. At C = 64, SGLang therefore decodes eagerly at its default setting.

The full trace is the appendix of `bench/decode-census/README.md`.
