# Q235 results: one full-depth QLoRA optimizer step on Qwen3-235B-A22B-Instruct-2507 (e4b#835)

**Status: MILESTONE PASS** at `q235-pro6000-8`, under [Amendment 16](PREREG-q235.md). `q235_verdict.py` at e4b
`27200917` reads PASS, "rules 1-10 hold". The maintainer re-derived it on CPU from the private receipt store, and every rule
boolean and instrument check came out identical. The JSON is not byte-identical: 284 fp32 `u` values differ at about 1e-4
relative (CPU torch 2.8 against the lab's CUDA torch 2.11). The nearest margin is the routing plant's 0.846 against the
band's 0.775, about 2,000× that drift, so no outcome moves.

The step: all 94 decoder layers, every routed expert, one GPU, two optimizer steps of the registered Shape B (Amendment 6),
a fresh-process repeat of step 0, and the records the milestone lists. It ran on one RTX PRO 6000 Blackwell Workstation
Edition, the registered host class.

## The runs

| run | what spent it | cost | outcome |
|---|---|---|---|
| [`q235-pro6000-1`](receipts/q235-pro6000-1/README.md) | the launcher: the manifest's 8.54 h wall clock exceeded the 6.0 h teardown cap | $0 | refused before launch; no instance |
| [`q235-pro6000-2`](receipts/q235-pro6000-2/README.md) | the first box (Shape A), machine 8652, e4b `c7a40682` | $4.954 | **PASS**, rules 1-7 |
| [`q235-pro6000-3`](receipts/q235-pro6000-3/README.md) | the Shape B box (Amendment 6), machine 8652, e4b `13c05fce` | $5.534 | **PASS**, rules 1-8 |
| [`q235-pro6000-4`](receipts/q235-pro6000-4/README.md) | the provider: the offer was taken before the create (`no_such_ask`) | $0 | refused before any instance |
| [`q235-pro6000-5`](receipts/q235-pro6000-5/README.md) | the box's host-rate check: the link or host-copy rate was below the fit's floor | $0.104 | refused before the fetch; not a strike |
| [`q235-pro6000-6`](receipts/q235-pro6000-6/README.md) | the repeat's time-left guard (rc 142), machine 46209 | $3.848 | VOID, strike one |
| [`q235-pro6000-7`](receipts/q235-pro6000-7/README.md) | the box's time-left guard (rc 142), machine 115349 | $4.299 | VOID, strike two: the lane stopped for a simplification |
| [`q235-pro6000-8`](receipts/q235-pro6000-8/README.md) | the box under Amendment 16, machine 141671, e4b `27200917` | $4.567 | **PASS**, rules 1-10 |

Total $23.306. Runs 2 and 3 read the milestone's rules as they stood then, on one host (machine 8652). Amendment 8 added
rules 9 (the dX record) and 10 (the held-out read) and required the next box to run off machine 8652. After strike two,
Amendment 16 sized the box's cap on the box (half the time left), recorded the CPU topology and made a host-bound load
`VOID_BY_HOST`, not a strike. Strikes restarted at 0. `q235-pro6000-8` is the first box under those rules, and it read
PASS on all ten.

## The milestone read (`q235-pro6000-8`)

- **The box:** Vast instance 56688656 on machine 141671, 2026-10-19 07:11-09:03Z on the box, $4.567.
  - The card: an RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB, cc 12.0, driver 595.71.05, power limit 400 W).
  - The host: an Intel Xeon Gold 6330 (56 physical cores, 112 logical, CPU quota 53.76) and 503.5 GiB of RAM.
- **Code:** e4b `27200917` (Amendment 16's merge) and grouped-nf4-gemm `724ccc45`. The box recorded torch 2.11.0+cu128,
  transformers 5.19.0, bitsandbytes 0.50.2, grouped-nf4-gemm 0.45.0 and experts4bit-qlora 0.55.0.
- **The checkpoint:** Qwen/Qwen3-235B-A22B-Instruct-2507 at `ac9c66cc`, fetched on the box in 61 min. The load read
  470.2 GB in 736.9 s: assembly 386.5 s, offload 114.5 s, read 47.8 s at 9.8 GB/s. All 94 layers were patched, with the
  experts offloaded to host RAM. `load_layers.jsonl` records 94 of 94 layers.
- **Timeline:** the box phase took 2,047 s under its 8,653 s cap (Amendment 16: half of the 17,606 s left, less 300 s).
  The repeat took 843 s under its 1,439 s cap (Amendment 15).

### The milestone's records

| record | step 0 | step 1 |
|---|---|---|
| loss (Shape B: micro-batch 2 × 4, 1,006 valid tokens) | 1.233889 | 0.649796 |
| forward / backward / `optimizer.step()` | 19.87 / 22.60 / 0.18 s | 19.73 / 22.53 / 0.10 s |
| elapsed, synchronized | 44.40 s | 44.26 s |
| `max_memory_allocated` / `max_memory_reserved` | 21.29 / 21.84 GiB | 22.77 / 24.72 GiB |
| host peak RSS (VmHWM) | 340.1 GiB | 340.1 GiB |
| host-to-device bytes (offload staging) | 946.69 GiB in 748 copies | 946.69 GiB in 748 copies |
| NVMe read / written during the step | 5.5 / 6.1 MB | 0 / 23.9 MB |
| gradient global norm; all finite | 3.7723; yes | 1.3825; yes |
| adapter gradients nonzero | every B (every A is zero: B starts at zero) | every A and every B |
| adapter tensors changed by the step | every B (564) | all 1,128 |

- **Frozen-base hashes:** 2,541 frozen tensors. The sha256 after the load (`f32e22da…`) equals the sha256 after both steps.
- **The repeat** (a fresh process, step 0): loss 1.2399002313613892, equal to the box's in-process loss. The gap is 0.0
  against a tolerance of 0.000124.
- **Not recorded:** the box record carries torch's allocator peaks, not a separate driver-side peak for the process. It
  carries the process's VmHWM and RSS split (shared memory 170.4 GB, which holds the offloaded experts), not e4b's own
  pinned-bytes figure. The milestone lists both, and this read does not have them.

### Fidelity and the rules (`verdict.27200917.json`)

The reference is the full-depth layer-streamed bf16 reference on the box's own tokens (8 rows, 1,014 real tokens). It was
run on the lab's RTX A2000: 94 layers in 4,574.5 s, loss 1.2130016. Its 94 per-layer outputs are bitwise identical to
`q235-pro6000-2`'s reference. Runs 2, 3 and 8 share their tokens, and both references ran under the same lab software (torch
2.11.0+cu128, transformers 5.19.0), so this is the reference reproducing itself, not a check across versions.

| rule | read |
|---|---|
| 1 completed on the class | RTX PRO 6000 Blackwell Workstation Edition, exit 0 |
| 2 loss finite | yes |
| 3-5 strict (gradients, optimizer delta, frozen hashes) | no failures |
| 6 repeat | gap 0.0 (tolerance 0.000124) |
| 7 band | interior (layers 8-86) max `u` 0.358 against 0.775 |
| 8 accumulation (Amendment 6) | max_rel 0.00338 / 0.00384 against 0.0231; cosine 0.999997; floor 0.0 |
| 9 dX (Amendment 8) | finite and nonzero on all 94 layers at both steps |
| 10 held-out (Amendment 8) | 1.3044 → 1.0059 → 0.9534 (−26.9 %), one-sided bound 1.10× before; finite |

- **The instrument checks:** the null is bitwise (0.0 against the 0.001 floor). Both plants engage at layer 47 only and
  fail the band, as a plant must: routing 0.846, epilogue 1.234. Rule 8's plants are detected (mean 0.70, overwrite 1.10).
- **Reported, not gating:**
  - device peak 22.77 GiB against e4b's estimate of 24.76 GiB (−8 %, under the 15 % that opens an estimator issue);
  - host-to-device bytes 946.69 GiB a step against the estimate's link line of 237.94 GiB (4.0× the line);
  - the loss gap to the reference, 0.0269;
  - router overlap at least 0.926, and router set agreement at least 0.497;
  - Shape A against Shape B: 11,572 of 95,316 token-layers flipped a routing choice (12.1 %), in all 94 layers;
  - Amendment 7's MoE timing: overhead 0.74 s a pass (1.8 %), decision-grade;
  - the train rows' loss, 1.2399 → 0.6468 → 0.2949.

## What this read cannot say

The registration's limits stand unchanged. The step's elapsed time is a record, not a benchmark, and nothing compares
frameworks or cards. Two steps say nothing about training dynamics or quality after training. Nothing here covers other
shapes, sequence lengths, models or adapter scopes; the `grouped_mm` route (sm_90); serving; or any card class this lane
did not run. The host-to-device bytes at 4.0× the estimate's link line are a finding for the estimator, not a claim about
links.

## Receipts

`receipts/q235-pro6000-8/` holds the box's own outputs, byte-identical to the private receipt store, with `SHA256SUMS`
over every file there:
- `tc1/receipts/` (the box and repeat records, the plants, the caps, the CPU and host records, `load_layers.jsonl`);
- `tc1/summary.txt`, `tc1/outer.log` and `tc1/logs/`;
- the verdict, `verdict.27200917.json`;
- the read's record in `read/`.

Two manifests list the files that stay outside the repository: the 380 per-layer tensors (8.2 GB) in
`tc1/RELOCATED.sha256`, and the 16.0 MB reference tensor in `read/RELOCATED.sha256`. `STORE-ONLY.sha256` names the few
files kept only in the store, by sha256: the token and held-out tensors; the read's lab scripts, stage record and own
`SHA256SUMS`, which carry lab paths; and the fetch log, which carries carriage returns from the download's progress output. The launcher's receipts and ledger rows are in the private receipt store. The other runs' directories hold a README
each, and runs 2 and 3 also hold their verdict files.
