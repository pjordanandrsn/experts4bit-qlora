# SC1g box J: e4b's MXFP4 cost on gpt-oss-20b chat text is the weights, not the route; it does not replicate across windows (lane SC1g of #846; 2026-10-05)

Pre-registration: [`../../sc2/SC1g-PREREG.md`](../../sc2/SC1g-PREREG.md). Box J is set by amendment A2 and re-run under A3;
the reads are its "A3 read" section. These are e4b-only diagnostic runs: no comparator was installed, so G1–G5 read
UNREAD by design.

| run (receipt) | e4b | host | outcome | $ |
|---|---|---|---|---|
| `sc1g-diag-1` (adertha-receipts `a6a16350`) | `3e133cf7` | machine 145701 | OK. A2's arms; 15 of 20 ran (the deadline dropped j16–j20) | 0.709 |
| `sc1g-diag-2` (`28cd4d15`) | `a7891300` | machine 145701 | OK. A3's arms; 10 of 19 ran (the deadline dropped k11–k19); attention check INERT | 0.726 |

**What is committed.**
- Per run: `summary.txt`, `versions.txt`, `box.json`, `forensics.txt`, `kernel_classes.json`, `quiesce_arms.json`,
  `logs/`, and `sc1g/`.
- `sc1g/` holds every arm's receipt, the pinned windows (`k8_window_*.json`), the route records (`routes/`), the kernel
  and attention checks, and the reducer's verdict.
- Trimmed: `sc1g-diag-1`'s captured decode activations (`capture_conv1.pt`, 3.7 MB). The kernel check's receipts from
  them are kept (`gemv_*_5090.json`).

**Reproduce** the verdict lines (G1–G6, J1–J6 for diag-1, K1–K5 and the across-window reads for diag-2, the attention
check) from this directory with the reducer at main:

```
python bench/sc2/sc1g_reduce.py --dir bench/h2h-2026-10-02/sc1g/receipts/sc1g-diag-2/sc1g
python bench/sc2/sc1g_reduce.py --dir bench/h2h-2026-10-02/sc1g/receipts/sc1g-diag-1/sc1g
```

The $0 A2000 correctness checks the reads also rest on are in [`../../sc2/sc1g-a2000/`](../../sc2/sc1g-a2000/): the GEMV
kernel, the fp8 KV pack, the NF4 M-tile at large M, and the served loop's KV write paths.

## Box R (amendment A4): the bf16-dequant reference, `R_NOT_OK`

**Box:** one H100 NVL at a declared $3.50/h, maintainer-approved 2026-10-05 per tc1c's precedent, guard 1.0 h.

**Launches.** R launched from `8a1513a4` (#1198), then from `9dc7b59a` (#1217, which made the egress probe log why it
refused). The registered rule is identical at both: `test_box_r_rule_is_the_registered_one` digest `ee122b74…`.

| attempt | receipt (adertha-receipts) | outcome | $ |
|---|---|---|---|
| `sc1g-r-1` | none written | REFUSED[97]: the CTO daily ceiling ($50, the manifests' base; the $100 figure is the training campaign's) | 0 |
| `sc1g-r-2` | `f6ec2329` | REFUSED: the store HEAD was ahead of its upstream (the mini's https push failed) | 0 |
| `sc1g-r-3` | `f483ad41` | REFUSED: the same window, ~30 s | 0 |
| `sc1g-r-4` | `6859a12d` | REFUSED: the manifest carried RTX 5090 exclusion receipts (an H100 NVL launch accepts only same-class ones) | 0 |
| `sc1g-r-5` | `1dac8166` | REFUSED: no Vast H100 NVL offer with ≥ 320 GB disk and ≥ 98 GB RAM | 0 |
| `sc1g-r-6` | `ae253242` | HARNESS_ERROR: RunPod; the curl egress probe read 0.0 MB/s with its stderr discarded (fixed by #1217) | 0.1348 |
| `sc1g-r-7` | none written | REFUSED[97]: a live peer run's nested `receipt.json` failed the store reconciler | 0 |
| `sc1g-r-8` | `2c49f7d5` | **ran to its verdict: `R_NOT_OK`** (rc 31; the launcher records HARNESS_ERROR); RunPod, from `9dc7b59a` | 0.5556 |

Box R total **$0.6904**; the lane is at **$4.482**.

**Reproduce R's verdict** (no GPU) from the committed receipt:

```
python bench/sc2/sc1g_ref.py --reverdict bench/h2h-2026-10-02/sc1g/receipts/sc1g-r-8/ref --k0 bench/h2h-2026-10-02/sc1g/receipts/sc1g-r-8/k0.json
```

It exits 0 only if the verdict and every check equal R's own `r_verdict.json`, and every `ref_<src>.npz` hashes to the sha R
recorded. The artifacts are kept as receipts only: R is not OK, so they grade nothing.


## Box R (amendment A5): the full-vocabulary reference, `R_OK`

**The box.** The same class, rate and guard as A4: one H100 NVL, declared $3.50/h, guard 1.0 h. It launched from A5's merge,
`dfc5bdaf` (#1223), with the registered rule digest `7f307c39…` and `staged-r.sha256` re-checked at that SHA.

| attempt | receipt (adertha-receipts) | outcome | $ |
|---|---|---|---|
| `sc1g-r5-1` | `49b14848` | REFUSED at the provider: no RunPod Secure H100 NVL stock (no instance) | 0 |
| `sc1g-r5-2` | `fde4f454` | **`R_OK`** (rc 0); Vast verified, machine 141791, $2.07/h; teardown proven | 0.826 |

- **Box R's total** is now **$1.5164**: $0.6904 under A4 plus $0.826 here. The lane is at **$5.308**.
- **The run took 22 min:** from 02:12:59Z to TP_DONE at 02:26:38Z, then about 8 min fetching the ~4.1 GB of full rows.

**What is committed** follows `sc1g-r-8`'s layout: `summary.txt`, `versions.txt`, `forensics.txt`, `outer.log`, `rsync.txt`,
`k0.json`, `logs/`, and `ref/` (`r_verdict.json`, `r_calib.json`, `SHA256SUMS`).

**Where the full rows are.** They are not in git (5 × 823,656,576 bytes, fp16 `[2048, 201088]`). They are in two places,
each re-hashed against `ref/SHA256SUMS`:
- the controller copy: `~/sc1g-ref-full/sc1g-r5-2/` on the mini;
- the durable copy: QNAP Pool 3, `/share/ZFS19_DATA/sc1g-ref-full/sc1g-r5-2/` (with its own `SHA256SUMS`).

**What box I reads.** `bench/sc1/sc1g_ref/` holds `ref_full_shas.json` (the registered shas), plus `r_verdict.json` and
`r_calib.json`, byte-identical to this receipt's. A test pins all three to it.

**Reproduce R's verdict** (no GPU) from the committed receipt:

```
python bench/sc2/sc1g_ref.py --reverdict bench/h2h-2026-10-02/sc1g/receipts/sc1g-r5-2/ref --k0 bench/h2h-2026-10-02/sc1g/receipts/sc1g-r5-2/k0.json
```

It reads `R_OK rule=A5 ... matches_recorded=True`. The full rows are re-hashed too, when they sit in `ref/full/` beside it.


## Box I (amendment A5): the proof and the reading, K-A REFUTED, L1 HOLDS, L2 HOLDS

Both runs are from `5d794520` (#1248), on Vast RTX 5090 machine 145701, from the adertha controller at `5037a602`
(adertha#182: each receipt records `environment.prereg_on_main` and `lane_commit_sha`).

| run | receipt (adertha-receipts) | outcome | $ |
|---|---|---|---|
| `sc1g-prove-a5-8` | `e0b46a90` | PROVED: e4b served, vLLM served and llama.cpp q8 conv1 rows VALID; zero masked reference mass on all three | 0.955 |
| `sc1g-5090-a5-1` | `1ac9c0ff` | the reading: K-A REFUTED, L1 HOLDS, L2 HOLDS (`SC1G_A5_*` in `summary.txt`) | 1.557 |

Box I's earlier proof attempts (`-a5-1` to `-a5-7`) and their causes are listed in the PREREG's "A5 box I proof record".

**What is committed** follows the diag runs' layout: `summary.txt`, `versions.txt`, `box.json`, `forensics.txt`, `logs/`
and `sc1g/`. `sc1g/` holds every arm's receipt, the KL records (`kl_*.npz` with their `.json` meta, and llama.cpp's raw
`kl_*.bin`), the windows, the route records and the reducer's verdict. The full-vocabulary reference rows are inputs
registered by sha; they are not here (see box R above).

**Reproduce the reading** (no GPU) from this directory with the reducer at main:

```
python bench/sc2/sc1g_reduce.py --dir bench/h2h-2026-10-02/sc1g/receipts/sc1g-5090-a5-1/sc1g
```

It reproduces the box's `verdict_sc1g.json` A5 section: every verdict, key and flag identical, every number to 1e-12 relative
(CI's Linux numpy rounds three pooled means one ULP from the box's).

## Box J (amendment A6): the prompt route, P1 PARTIAL, P2 FALSIFIED, P3 HELD

| run | receipt (adertha-receipts) | outcome | $ |
|---|---|---|---|
| `sc1g-diag-a6-1` | `04e7678f` | A6 reading: P1 PARTIAL, P2 FALSIFIED, P3 UNREAD (the deadline dropped the (c) arms and conv4's (a)); Vast 152440, from `1825cf02` (#1254) | 0.532 |
| `sc1g-diag-a6-2` | `2f93f90d` | A6 continuation: **P3 HELD** ((c)/(b) median 1.019 / 0.920 / 0.912 on conv1–conv3; conv4's (c) dropped by `can_run`'s reserve); (b) bit-identical to A5's and A6-1's; Vast 145701, from `993103b8` (#1256) | 0.753 |

The committed layout is the diag runs': `summary.txt`, `versions.txt`, `box.json`, `forensics.txt`, `logs/` and `sc1g/`.

**Reproduce** (no GPU): `python bench/sc2/sc1g_reduce.py --dir bench/h2h-2026-10-02/sc1g/receipts/sc1g-diag-a6-1/sc1g`, and
the same with `sc1g-diag-a6-2`. Every verdict is identical, and every number agrees to 1e-12 relative (pinned by
`test_the_a6_reading_rederives_from_its_committed_receipt` and `test_the_a6_continuation_reading_rederives_from_its_committed_receipt`).

## Box J (amendment A7): the router-flip instrument, FRAGILE_POSITIONS

| run | receipt (adertha-receipts) | outcome | $ |
|---|---|---|---|
| `sc1g-diag-a7-1` | `b655ab99` | NOT_RUN: pre-flight SSH banner timeout on Vast 151350 during the disk check (machine evidence; a pair class since adertha#183) | 0.007 |
| `sc1g-diag-a7-2` | `2889ce90` | REFUSED: the exclusion was passed as a run id, not a receipt path (self-inflicted; no instance) | 0 |
| `sc1g-diag-a7-3` | `74a00b5f` | REFUSED: that pre-flight was not yet a class `rent.py` admits for exclusion (no instance) | 0 |
| `sc1g-diag-a7-4` | `42e404dd` | A7 reading: **FRAGILE_POSITIONS**. conv2's (a)-top and (b)-top 1 % KL positions are both 21/21 flipped against 0.706 (p 6.4e-4); the perturbation control is BIT_IDENTICAL; Vast 145701, from `f008f6d7` (#1261) | 0.701 |

`sc1g/rid_*.npz` (1.6 MB each) are the capture's per-call expert ids, kept byte-identical because the reading
re-derives from them.

**Reproduce** (no GPU): `python bench/sc2/sc1g_reduce.py --dir bench/h2h-2026-10-02/sc1g/receipts/sc1g-diag-a7-4/sc1g`.
Pinned by `test_the_a7_reading_rederives_from_its_committed_receipt`.

