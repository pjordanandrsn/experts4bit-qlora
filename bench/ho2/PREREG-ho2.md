# HO2: the planner's first held-out training read

This registers two rented boxes that measure rows for the held-out side of Loggetta's frozen planner split
(`evidence/planner-split/SPLIT.json`, loggetta#88). It changes no estimate, default or gate. It computes no band: its rows go to
the split's held-out side as a new versioned split file, and the band (loggetta's L2) reads them from there.

## Why

The split's held-out side has one row, so every family × card is unmeasured and the planner prints no band anywhere. A row
belongs on that side only if no planner constant was set or tuned after looking at it. HO2 measures rows of two MoE families
that no constant has seen, on two cards, through the path a first-time user takes.

## Families (named before any measurement)

- `LiquidAI/LFM2-8B-A1B` (`lfm2_moe`) at `c1c44ff9fc00db3ebf4516970563f5f383d23670`.
- `baidu/ERNIE-4.5-21B-A3B-PT` (`ernie4_5_moe`) at `87db95487941cb39592ee0abca3b9155a6d19c5c`.

Why these two:
- experts4bit-qlora's fast-train path supports both (lane MG1's read, 2026-10-04).
- No row of either family is in the split's derivation side, and no planner or estimator constant's history cites either.
- Both plan feasibly on 24 GB and 32 GB cards today (CPU, configs only). LFM2 plans resident on both. ERNIE plans resident on
  the 5090, and at 2,048 tokens it plans host offload on the 4090, so both placements are exercised.
- MG1's own rows are not used. They ran the experts4bit-qlora of 2026-10-04, older than today's plan, and a runtime that moved
  can flip a verdict by itself. HO1 found this, and the IN-SAMPLE re-plan of its rows (loggetta#89) shows it.

## The boxes

- **`ho2-5090-1`:** one RTX 5090 (compute capability 12.0, at least 30,000 MiB), declared at $1.05/h, the policy ceiling.
- **`ho2-4090-1`:** one RTX 4090 (compute capability 8.9, at least 23,000 MiB), declared at $0.75/h. The policy has no RTX 4090
  ceiling, so the declared rate binds. The rate comes from a read on 2026-10-21 of offers at reliability ≥ 0.98, $0.661–0.707.
- `-2` and `-3` of each are spares. A spare runs only after a refusal that is not the lane's own.
- Each box: the image's torch 2.8.0+cu128, 220 GB disk, at least 48 GiB host RAM, 120 GB free, a 2 h guard.

## Rows

On each box, four rows: each family at 1,024 and 2,048 tokens. Each row is one `loggetta train` call:

```
python -m loggetta train MODEL --revision REV \
  --dataset tatsu-lab/alpaca --dataset-revision dce01c9b08f87459cf36a430d809084718273017 --format text --text-field text \
  --seq SEQ --micro-batch 2 --steps 20 --seed 0 --out run/receipts_TAG --adapter-out run/adapter_TAG
```

- **The planner's own plan.** There is no `--fix` and no `--observations`, so the plan uses no receipts on file and the
  planner's defaults: what a first-time user is told.
- Every estimate-read switch is unset, so the estimate prices the defaults the run runs.
- **The data is pinned:** Alpaca's text column at its revision, so it is identical on both boxes. A row must train exactly
  20 × 2 × SEQ tokens in 40 blocks, or it is VOID: an early stop is not a plan miss.
- **The installed stack is pinned:**
  - experts4bit-qlora at the launch commit;
  - grouped-nf4-gemm at e4b CI's pin, `a83f6bb3`;
  - Loggetta at its 0.6.2 release commit, which carries the demonstration data fix, written into the runner at registration;
  - transformers 5.17.0, bitsandbytes 0.50.2 and `datasets` (Loggetta's data path imports it).
- **A tripwire refuses at install, before any fetch,** if a pin differs or if any name Loggetta's train path imports does not
  import. Loggetta's modules import only the standard library at module level and every third-party import is lazy, so importing
  Loggetta proves nothing. The tripwire imports each name itself: `datasets`, `transformers`, `numpy`, `bitsandbytes`,
  `safetensors`, experts4bit-qlora's recipe, LoRA and topology names, and grouped-nf4-gemm's `nf4_qlora` and `nf4_route`.

## The rule (bytes, per row)

From the receipt Loggetta writes (`comparison`). `loggetta train` plans, then executes that plan, and the receipt embeds it
(`plan.to_dict()`, `loggetta/execution.py`). So the plan total is the plan's own `selected.device_bytes`, and the allocator estimate
is that total less its CUDA context and allocator reserve lines (`execution.compare`).
- **COVERED** if the plan total ≥ the sampled driver peak AND the allocator estimate ≥ the allocated peak.
- **UNDER** if the driver peak is above the plan total.
- **SHORT** if the allocated peak is above the estimate. A row can be both.

The box reads COVERED when all four rows are COVERED, MISS otherwise, with every row reported.

A row is VOID, and the box is VOID, if:
- it has not exactly one receipt;
- its status is not OK;
- its model, revision, workload, data or card differs from the registration;
- it trained fewer tokens;
- any overhead line (allocator reserve, CUDA context, process baseline) is not the inferred default, meaning the plan learned
  from receipts on file.

`bench/ho2/ho2_reduce.py --self-test` holds 17 cases, and `tests/test_ho2.py` the rest. These include one real Loggetta training
receipt, with paths and GPU uuids redacted (`tests/fixtures/ho2/`), read through the reducer against its own run and against this
registration.

## Prediction (before any measurement)

All 8 rows COVERED: the planner's own claim for a family it has never seen. Any UNDER or SHORT row is reported as it stands. None
is fitted away, and none changes a constant here.

## What happens to the rows

COVERED, UNDER or SHORT, every valid row goes to the held-out side of a new split version (`SPLIT-v2.json`, its own review).
That gives two rows per family × card, the minimum for a band. The band itself is L2's.

## Budget

- **5090:** 2 h × $1.05/h + 100 GB × $0.011/GB download = **$3.20** worst.
- **4090:** 2 h × $0.75/h + 100 GB × $0.011/GB = **$2.60** worst.
- **Total worst: $5.80.** The checkpoints are about 60 GB (LFM2 about 17, ERNIE about 43); the 100 GB is the launcher's download
  estimate, not an enforced bound.

The fit: install 450 s + fetch 900 s + 4 × 480 s of training + 60 s reduce + 600 s reserve = 3,930 s, × 1.25 = 4,912 s, inside the
2 h guard with 300 s for the boot.
