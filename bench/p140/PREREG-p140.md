# P140 — does P137's layer-major gain travel to an RTX 5090 (sm_120, 32 GB)?

**Question.** P137 measured `layer_major_step` at S = 2.365 over the stock accumulation loop on an RTX 4090
(`e4b.train.p137.layer-major-offload.qwen3.4090.2026-10-18`). The setup was expert offload with
`enable_fast_train(dgrad=True)` on Qwen3-30B-A3B. Does the gain hold on an RTX 5090? The 5090 has a different compute
rate and, on many hosts, a faster host link.

The lane number is free: TC1 amendment 54 (#1286) used "P137–P140" as its own prediction ids, the same
namespace overlap P137 and P138 already live with. This lane is unrelated to those predictions.

This is a replication. The kit is P137's at Amendment 3, by named substitutions (`bench/p140/`), and only the card
changes. The licence, if earned, covers the RTX 5090 class (sm_120) at this recipe and nothing else.

## Seen data, disclosed (sets no bar)

`p137-4090-7` (RTX 4090, machine 51612):

| | median s/step |
|---|---|
| A0a | 7.641 |
| L1a | 3.231 |
| L1b | 3.173 |
| A0b | 7.514 |

- S = 2.365, self-pairs 0.983 / 0.982.
- Stagings 380 / 95, peaks 8.41 / 7.98 GB.
- Q3 (band [1.25, 2.0]) was falsified high.

These numbers inform the prediction below, as stated. They set no STRICT term: every bar is P137's, unchanged.

## The pins (P137's)

- e4b `E4B_PIN=b4366b9af2234730b5df783c5f3b0c7aef3ab41c` (0.55.0), the library under test.
- grouped-nf4-gemm 0.46.0 `GNF4_SHA=6b7c2e7474d20c8843814dd46dad6cfc3d94b440`.
- torch 2.8.0+cu128 (the image's), transformers 5.18.0, bitsandbytes 0.50.2, `huggingface_hub>=0.23,<2` (P137 Amendment 1).
- Qwen/Qwen3-30B-A3B @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
- The data slice: alpaca `train[:400]`, staged from the controller with
  `DATA_SHA=68e3d31c194ebd8fee4330808a46642f443aecf4e16b78e5eb39fd21b16079c5`, never downloaded on the box.

## The arms and the run (P137's, Amendments 1–3 folded in)

- **Arms:** A0 is the stock accumulation loop; L1 is `layer_major_step`.
- **Order:** ABBA, A0a L1a L1b A0b, in one process after one load. Every arm starts from one hash-checked adapter
  snapshot, after an untimed warm-up of both loops.
- **Recipe:** seq 2048, micro-batch 2 × accum 4, r 16 / α 16, AdamW 8-bit, lr 2e-4, seed 3407. Each arm runs 2 warm and
  32 timed steps.
- **Card:** the RTX 5090 class, compute capability 12.0. Any other card is refused (rc 15), as is any other stack.

## The rule (`p140_reduce.py`, P137's reducer, 41 self-test cases)

**VOID, STRICT**, if any of these holds:
- an arm record is missing, or an arm hits OOM;
- off-pin or mixed stacks;
- the patched or offloaded layer count is not 48;
- step 0's per-micro-batch losses are not bit-identical across arms;
- step 0 is not cold (A0 381, L1a = L1b = 96);
- steady stagings are not exactly 380 (A0) and 95 (L1);
- L1's peak is above A0's by more than **4.25 GiB**;
- an arm did not start from the snapshot, or the arm-start memory is more than 16 MiB off;
- the warm-up record is missing or differs across arms, or its layer-major stagings are not above 96;
- an arm has other than 32 timed steps;
- a self-pair median ratio is outside **[0.97, 1.03]**.

**Otherwise:** LICENSED iff S = min(A0a/L1a, A0b/L1b) ≥ **1.10**, and NO_GAIN if not.

**REPORTING:** held-out |L1 − A0| ≤ **0.01**, and S inside the predicted band **[1.8, 3.0]**.

## The prediction, set a priori from the host link

**Bytes.** Qwen3-30B-A3B has 128 experts per layer, gate_up [1536 × 2048] and down [2048 × 768]. That is 603,979,776
parameters per layer, 0.302 GB in NF4 plus 0.038 GB of fp32 absmax: 0.3397 GB per staging. One forward and backward pass
stages 2n − 1 = 95 layers, 32.3 GB. The stock loop makes that pass once per micro-batch (129.1 GB per step);
layer-major makes it once per step (32.3 GB).

**Model.** Take a step as non-overlapped compute C plus copies at the effective pinned H2D rate r. Then
S = (C + 4T) / (C + T), with T = 32.3 GB / r. S tends to 4 when copy-bound and to 1 when compute-bound.

**Ranges:**
- **r = 20–50 GB/s.** A PCIe 4.0 x16 host at the low end; PCIe 5.0 x16 pinned at the high end, which a 5090 can use when
  the host provides it. So T runs from 0.65 to 1.6 s.
- **C = 0.8–1.6 s.** The 4090 reading fits r ≈ 17–20 GB/s with C ≈ 1.3–1.6 s (seen data, used here only for the scale).
  The 5090 runs the same kernels on more, faster SMs, so its C is no larger.

**Corners:**

| C (s) | r (GB/s) | T (s) | S |
|---|---|---|---|
| 1.6 | 20 | 1.6 | 2.5 |
| 0.8 | 20 | 1.6 | 3.0 |
| 1.6 | 50 | 0.65 | 1.9 |
| 0.8 | 50 | 0.65 | 2.3 |

**Predicted band: S in [1.8, 3.0], point about 2.4** (REPORTING). A host-link probe (the box's
`report_offload_environment` pinned ceiling) is recorded beside the reading, so a miss can be read against r. It is
never used to re-band.

## Predictions (written before any data)

- **Q1.** Every arm fits; an OOM is VOID.
- **Q2.** Step 0 is bit-identical across the four arms; stagings are exactly 381 / 380 and 96 / 95.
- **Q3.** S is in [1.8, 3.0].
- **Q4.** L1's peak is at or below A0's.
- **Q5.** The held-out |Δ| is within the same-loop spread. It may exceed 0.01, as at P137's 34 steps; REPORTING.
- **Q6.** LICENSED, about 85 %.

## Consequence, registered now

- **LICENSED:** a claim row for the RTX 5090 class at this recipe, and one STATUS clause beside P137's naming both cards.
  No default changes.
- **NO_GAIN:** the P137 row's scope stays the 4090. A note in `bench/expert-tiling/README.md` records that the gain did
  not travel, with r and C.
- **VOID:** under P137's stop rules. Two consecutive box-started HARNESS_ERROR or VOID results stop the lane until a
  simplification amendment lands.

## Runs, cost and the guards

- **Run ids:** `p140-5090-1` is the reading. `p140-5090-2` to `p140-5090-4` are spares under the flicker rule: an id
  refused before the create, or NOT_RUN at the pre-flight, is consumed and is not an attempt.
  `p140-dry-<n>` are the $0 dry runs.
- **Guards:** guard 2.5 h; the runner's arm check is 4,500 s (P137 Amendment 3's arithmetic, on a faster card).
- **Cost:**
  - Rate cap $0.85/h for the GPU and 200 GB (the other 5090 lanes' cap; 5090 pins on 2026-10-18 ran $0.71–0.83/h),
    download at most $0.011/GB.
  - **Worst $2.80**: $0.85 × 2.5 h, plus 61.1 GB × $0.011.
  - Expected about $0.8–1.3.
  - Under the lane's standing authority, at most $15 per run, after this registration merges.
- **Before any box:** a $0 dry run (`p140-dry-1`) of this kit on the A2000, for correctness only.
- **Launches** use the launcher current at launch, `--validate` passing, the market rule (3 GETs over 40 s, first-posted
  owns), and rsync before any hand abort. The receipt is committed within minutes of teardown.

P137's own history (P137 Amendment 1: staged data; Amendment 2: one process; Amendment 3: 32 timed steps and both
warm-ups) is inherited, not re-registered: see `bench/p137/PREREG-p137.md`.
