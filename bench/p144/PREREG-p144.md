# P144 — does layer-major cut Mixtral-8x7B's step time under expert offload on an RTX 4090?

**Question.** On Qwen3-30B-A3B, layer-major accumulation runs the step 2.365× faster than the stock loop on an RTX 4090
(P137) and 2.311× faster on an RTX 5090 (P140). Since #1757, `layer_major_step` also runs on Mixtral, checked for exactness
at $0 but not read for speed.

Mixtral-8x7B's NF4 experts total about 25 GB. That fits resident on a 32 GB card, but not on a 24 GB RTX 4090, so the
4090 is where a Mixtral user must offload. This lane reads that configuration.

**Kit.** P137's kit at its Amendment 3, by named substitutions (`bench/p144/`): the model, the library pin, the registered
n and the band change. Every STRICT term is P137's.

**Lane number.** The number is free as a lane. TC1 amendment 54 used "P137–P140" as prediction ids; this lane is unrelated
to those.

## Seen data, disclosed (sets no bar)

| run | card | S | self-pairs | load |
|---|---|---|---|---|
| `p137-4090-7` (Qwen3-30B-A3B) | RTX 4090 | 2.365 | 0.983 / 0.982 | 2,078 s on the slow host that #1752 explains |
| `p140-5090-1` (Qwen3-30B-A3B) | RTX 5090 | 2.311 | 0.999 / 1.002 | 65 s |

Host-link fits of those two runs:
- P137: T = 1.459 s, C = 1.743 s.
- P140: T = 1.139 s, C = 1.461 s.

These inform the scale of the prediction below and nothing else. Every bar is P137's.

## The pins

- e4b `E4B_PIN=a9d140d88c296596ecda63e1e97dba5c7d04b23e`: main with #1757 (layer-major on Mixtral) and #1752 (the
  thread cap).
- grouped-nf4-gemm 0.46.0 `GNF4_SHA=6b7c2e7474d20c8843814dd46dad6cfc3d94b440`.
- torch 2.8.0+cu128 (the image's), transformers 5.18.0, bitsandbytes 0.50.2, `huggingface_hub>=0.23,<2` (P137 Amendment 1).
- `mistralai/Mixtral-8x7B-Instruct-v0.1` @ `eba92302a2861cdc0098cc54bc9f17cb2c47eb61`, the Mixtral of tp1's receipts.
- The data slice: alpaca `train[:400]`, staged from the controller, with
  `DATA_SHA=68e3d31c194ebd8fee4330808a46642f443aecf4e16b78e5eb39fd21b16079c5`.

## The arms and the run (P137's design at Amendment 3)

- **Arms:** A0 is the stock accumulation loop; L1 is `layer_major_step`.
- **Order:** ABBA, A0a L1a L1b A0b, in one process after one load. Every arm starts from one hash-checked adapter
  snapshot, after an untimed warm-up of both loops.
- **Recipe:** seq 2048, micro-batch 2 × accum 4, r 16 / α 16, AdamW 8-bit, lr 2e-4, seed 3407. Each arm runs 2 warm and
  32 timed steps.
- **Mixtral's config:** `router_jitter_noise` is 0, `output_router_logits` false and the dropouts 0, so `layer_major_step`
  takes it.
- **Card:** the RTX 4090 class, compute capability 8.9. Any other card is refused (rc 15).
- **Host floors:** 150 GB of disk (a 93 GB checkpoint) and 98 GB of host RAM. The pin screen also prefers at least 48
  effective cores; the load's host-side assembly is copy-bound, and #1752 caps its threads at the container's quota.
- **Premise:** `tests/test_layer_major.py`, `test_layer_major_families.py`, `test_offload_hold.py` and `test_offload.py`
  at the pin: exactly 56 passed, none skipped (read on an A2000 at `a9d140d8`).

## The rule (`p144_reduce.py`, P137's reducer at n = 32; 41 self-test cases)

**VOID, STRICT**, if any of these holds:
- an arm record is missing, or an arm hits OOM;
- off-pin or mixed stacks;
- the patched or offloaded layer count is not 32;
- step 0's per-micro-batch losses are not bit-identical across arms;
- step 0 is not cold (A0 M(2n − 1) + 1 = 253; L1a = L1b = 2n = 64);
- steady stagings are not exactly 252 (A0) and 63 (L1);
- L1's peak is above A0's by more than **4.25 GiB**;
- an arm did not start from the snapshot, or the arm-start memory is more than 16 MiB off;
- the warm-up record is missing or differs across arms, or its layer-major stagings are not above 64;
- an arm has other than 32 timed steps;
- a self-pair median ratio is outside **[0.97, 1.03]**.

**Otherwise:** LICENSED iff S = min(A0a/L1a, A0b/L1b) ≥ **1.10**, and NO_GAIN if not.

**REPORTING:** held-out |L1 − A0| ≤ **0.01**, and S inside the predicted band **[1.5, 2.6]**.

## The prediction, set a priori from the host link

**Bytes.** Mixtral-8x7B has 8 experts per layer, gate and up [14336 × 4096] and down [4096 × 14336]. That is
1,409,286,144 parameters per layer, 0.705 GB in NF4 plus 0.088 GB of fp32 absmax: **0.793 GB per staging**. One forward and
backward pass stages 2n − 1 = 63 layers, **49.9 GB**. The stock loop makes that pass once per micro-batch (199.8 GB per
step); layer-major makes it once per step (49.9 GB).

**Model** (P140's). A0 = C + 4T and L1 = C + T, with T = 49.9 GB / r, so S = (C + 4T) / (C + T).

**Ranges:**
- **r = 17–25 GB/s** on 4090 hosts. P137's 4090 fit gave 22.1 GB/s. So T = 2.0–2.9 s.
- **C = 3–8 s**, the non-overlapped compute. Mixtral activates about 13 B parameters per token against Qwen3-30B-A3B's
  about 3.3 B, so C is several times P137's 1.74 s. The width of this range is the main uncertainty.

**Corners:**

| C (s) | T (s) | S |
|---|---|---|
| 3 | 2.9 | 2.47 |
| 3 | 2.0 | 2.20 |
| 8 | 2.9 | 1.80 |
| 8 | 2.0 | 1.60 |

**Predicted band: S in [1.5, 2.6], point about 2.0** (REPORTING). The box records `lscpu`, the container's CPU quota
(#1752's load log) and the pinned host-to-device ceiling, so a miss can be read against C and r. None of them re-bands.

## Predictions (written before any data)

- **Q1.** Every arm fits in 24 GB under offload; an OOM is VOID.
- **Q2.** Step 0 is bit-identical across the four arms; stagings are exactly 253 / 252 and 64 / 63.
- **Q3.** S is in [1.5, 2.6].
- **Q4.** L1's peak is within the 4.25 GiB term of A0's.
- **Q5.** The held-out |Δ| is within the same-loop spread (REPORTING).
- **Q6.** LICENSED, about 85 %.

## Consequence, registered now

- **LICENSED:** a claim row for Mixtral-8x7B on the RTX 4090 class at this recipe, and one STATUS clause beside P137's.
  No default changes.
- **NO_GAIN:** the Mixtral support stays exactness-only. A note records C and r.
- **VOID:** under P137's stop rules. Two consecutive box-started HARNESS_ERROR or VOID results stop the lane until a
  simplification amendment lands.

## Runs, cost and the guards

- **Run ids:** `p144-4090-1` is the reading. `p144-4090-2` to `p144-4090-4` are spares under the flicker rule: an id
  refused before the create, or NOT_RUN at the pre-flight, is consumed and is not an attempt.
  `p144-dry-<n>` are the $0 dry runs.
- **Guards:** guard 2.5 h; the runner's arm check is 4,500 s.
  - Expected box time is about 50 min: a 93 GB fetch (about 15 min), a load of a few minutes with the thread cap, and
    four 34-step arms of roughly 25 min together.
- **Cost:**
  - Rate cap $0.49/h for the GPU and 200 GB, download at most $0.011/GB.
  - **Worst $2.25**: $0.49 × 2.5 h, plus 93 GB × $0.011.
  - Expected about $0.8–1.2.
  - Under the lane's standing authority, at most $15 per run, after this registration merges.
- **Before any box:** a $0 dry run (`p144-dry-1`) of this kit on the A2000, for correctness only.
- **Launches** use the launcher current at launch, `--validate` passing, the market rule (3 GETs over 40 s, first-posted
  owns), and rsync before any hand abort. The receipt is committed within minutes of teardown.

P137's own history (P137 Amendment 1: staged data; Amendment 2: one process; Amendment 3: 32 timed steps and both
warm-ups) is inherited, not re-registered: see `bench/p137/PREREG-p137.md`.
