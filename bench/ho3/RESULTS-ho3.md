# HO3 read: a second box for HO2's cohorts, and the first host-offload held-out rows

HO3 (`PREREG-ho3.md`) registered rented boxes that add rows to the held-out side of Loggetta's planner split. Each box ran
HO2's four `loggetta train` rows: `LiquidAI/LFM2-8B-A1B` and `baidu/ERNIE-4.5-21B-A3B-PT`, each at 1,024 and 2,048 tokens.
Every row used the planner's own plan, at HO2's exact stack:
- experts4bit-qlora `f6e66fd6` (0.56.1);
- grouped-nf4-gemm `a83f6bb3` (0.49.0);
- Loggetta `9348f8b8` (0.6.2);
- transformers 5.17.0, bitsandbytes 0.50.2, torch 2.8.0+cu128.

Part A gives HO2's RTX 5090 and RTX 4090 cohorts a second machine each. Part B runs the same rows on two RTX A4000 machines,
where the planner keeps ERNIE's experts on the host: the first held-out rows with host offload.

## The read

| part | cohort | boxes read | rows covered | read |
|---|---|---|---|---|
| A | LFM2 × RTX 5090 | 1 | 2 of 2 | COVERED |
| A | ERNIE × RTX 5090 | 1 | 2 of 2 | COVERED |
| A | both × RTX 4090 | 0 (two started boxes VOID) | none read | not read: the part's stop |
| B | LFM2 × RTX A4000 (device) | 2, on two machines | 4 of 4 | COVERED on both boxes |
| B | ERNIE × RTX A4000 (host) | 2, on two machines | 1 of 4 | **MISS** on both boxes |

**ERNIE's host estimate is low.** On a 16 GB RTX A4000 the planner put ERNIE's experts on the host, as registered, and its
host estimate (15,451,815,936 bytes) was below the measured host peak on 3 of 4 rows across two machines:
- 1.0210× and 0.9959× on the first box (1,024 and 2,048 tokens);
- 1.0056× and 1.0548× on the second, the largest 808 MiB over.

**This is a host-side miss, not a device miss.** On all four ERNIE rows the device side held, with room:
- driver / plan 0.73 to 0.79;
- allocated / estimate 0.86 to 0.88.

The shortfall is the planner's estimate of host memory for a held-out family with its experts on the host. The registration
predicted every row COVERED, so the prediction failed on 3 of 4 ERNIE rows.

**Every other row held.** The prediction held for the RTX 5090 (all four rows) and for LFM2 on both RTX A4000 machines (all
eight rows). On every read box the plan matched the registration in residency and in both totals: no box recorded a plan
finding.

| box | card | row | placement | plan total GB | driver peak GB | driver / plan | allocator estimate GB | allocated peak GB | allocated / estimate | host estimate GB | host peak GB | host / estimate | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ho3-5090-r1` | RTX 5090 (cc 12.0) | lfm2_1024 | device | 9.768 | 8.510 | 0.8712 | 7.693 | 7.338 | 0.9539 | 3.221 | 2.033 | | COVERED |
| | | lfm2_2048 | device | 9.386 | 7.697 | 0.8200 | 7.375 | 6.748 | 0.9151 | 3.221 | 1.935 | | COVERED |
| | | ernie_1024 | device | 21.856 | 18.352 | 0.8397 | 17.766 | 17.100 | 0.9625 | 3.221 | 1.927 | | COVERED |
| | | ernie_2048 | device | 20.886 | 17.056 | 0.8166 | 16.958 | 16.143 | 0.9519 | 3.221 | 1.978 | | COVERED |
| `ho3-a4000-1` | RTX A4000 (cc 8.6) | lfm2_1024 | device | 9.768 | 8.101 | 0.8294 | 7.693 | 7.338 | 0.9539 | 3.221 | 1.850 | | COVERED |
| | | lfm2_2048 | device | 9.386 | 7.290 | 0.7766 | 7.375 | 6.755 | 0.9160 | 3.221 | 1.683 | | COVERED |
| | | ernie_1024 | host | 9.747 | 7.682 | 0.7881 | 7.675 | 6.584 | 0.8578 | 15.452 | 15.776 | 1.0210 | HOST_UNDER |
| | | ernie_2048 | host | 8.777 | 6.428 | 0.7323 | 6.867 | 6.049 | 0.8808 | 15.452 | 15.388 | 0.9959 | COVERED |
| `ho3-a4000-r2` | RTX A4000 (cc 8.6) | lfm2_1024 | device | 9.768 | 8.101 | 0.8294 | 7.693 | 7.338 | 0.9539 | 3.221 | 1.843 | | COVERED |
| | | lfm2_2048 | device | 9.386 | 7.286 | 0.7762 | 7.375 | 6.748 | 0.9150 | 3.221 | 1.668 | | COVERED |
| | | ernie_1024 | host | 9.747 | 7.682 | 0.7881 | 7.675 | 6.584 | 0.8578 | 15.452 | 15.538 | 1.0056 | HOST_UNDER |
| | | ernie_2048 | host | 8.777 | 6.430 | 0.7326 | 6.867 | 6.045 | 0.8803 | 15.452 | 16.299 | 1.0548 | HOST_UNDER |

GB = 1e9 bytes. Every number is from the boxes' reducer outputs (`receipts/<box>/ho3/run/verdict_ho3.json`). The host ratio
is the rule's only for a row with the experts on the host. A device row's host peak is recorded, but never ruled on.

## What binds every reading of these rows

**HO2's loss-branch line holds here too.** Each family's plan is one setup for both lengths. The 2,048-token rows (4,096
tokens a micro-batch) run experts4bit-qlora's chunked LM loss under its auto gate, and the 1,024-token rows run the stock
loss. The branch is inferred, not counted. So a row pair is not a length comparison, and the growth of ERNIE's host
shortfall with length on the second box is not read as a length effect.

**Two machines per cohort, not a distribution.** Part B's two RTX A4000 boxes ran on different machines, as registered,
and the plans and device-side numbers are near-identical across them. The host peaks differ by 226.6 MiB at 1,024 tokens and
869.0 MiB at 2,048. Two boxes say the host estimate is low on most rows; they do not say by how much on a
third machine.

## Part A's RTX 4090: two started boxes, both VOID

The stop: two strikes in a part end that part, and a box that started and was not read is a strike. Both RTX 4090 boxes
started and read VOID. They are not readings, and their rows go nowhere.
- **`ho3-4090-r1`:** every row's sampled driver peak is missing. The host's `nvidia-smi --query-compute-apps` listed no
  per-process memory, so Loggetta's sampler took no sample. A plan total, estimate or peak is missing, so every row is
  VOID. This is the box that led to Amendment 2's driver-peak probe, which since refuses such a host before the install
  (rc 19).
- **`ho3-4090-2`:** a numerics failure on the host. Both LFM2 rows read `ALARM`: each receipt's mean loss over the first
  third of training equals its mean over the last third, 11.0904, which is ln 65,536: uniform over LFM2's 65,536-wide
  head. Both ERNIE trains exited 1 and wrote no receipt.

The RTX 4090 cohort's second box is therefore not measured. HO2's RTX 4090 rows stand on their one box.

## Attempts and cost

From the private receipt store: each attempt's launcher receipt (`receipt.json`, its `cost_usd.actual` and
`runtime_seconds`). Those receipts are store-only, because they name the rented host, so a reader of this tree cannot
re-derive this table; the four boxes that never started have no other record here. Each started box's README repeats
its USD.

| box | outcome | started | USD | launcher wall time |
|---|---|---|---|---|
| `ho3-5090-1` | refused at create: the pinned offer was gone | no | 0 | 35 s |
| `ho3-5090-r1` | read: COVERED | yes | 0.472 | 916 s |
| `ho3-4090-1` | refused at create: the pinned offer was gone | no | 0 | 35 s |
| `ho3-4090-r1` | VOID (strike 1) | yes | 0.630 | 946 s |
| `ho3-4090-2` | VOID (strike 2: the RTX 4090's stop) | yes | 0.282 | 851 s |
| `ho3-a4000-1` | read: MISS | yes | 0.435 | 1,420 s |
| `ho3-a4000-2` | host refusal before the install (rc 19: the driver-peak probe took no sample) | no | 0.008 | 378 s |
| `ho3-a4000-r1` | refused at create (the offer billed above the declared rate), then NOT_RUN at the launcher's pre-flight (Hugging Face CDN 14.6 MB/s, under its 20 MB/s floor) | no | 0.014 | 305 s |
| `ho3-a4000-r2` | read: MISS | yes | 0.262 | 2,069 s |

**Total:** USD 2.103 against the registered USD 27.23 worst. The A4000 retry id `r1` appears twice because a refusal at
create is receipted under its own name and spends no id.

Every started box's verdict re-derives from the private receipt store: `ho3_reduce.py` re-run on the box's run directory
reproduces its `verdict_ho3.json` byte for byte. The maintainer re-derives them again on this change.

## What the rows are for

Every valid row goes to the held-out side of a new version of Loggetta's planner split (`SPLIT-v3.json`, under its own
review), with its box and machine, so a band can require two boxes per cohort. That includes ERNIE's HOST_UNDER rows: a
held-out row is recorded whatever it reads. HO3 computes no band (`band_computed: false`). Loggetta's L2 band reads the
split, and it is where the host shortfall has to show.

## Closed

HO3 is closed. Its pins were last asserted at `5790042a` (Amendment 3, e4b#2064), the launch commit of every box started
after it. `tests/test_ho3.py` reads its staged hashes and its runner's stack pin at that commit, never the live tree
(bench/common rule 6).

## Rebuild

Each started box's directory under `receipts/` holds:
- `ho3/`: the box's files that carry no host identifier, byte-identical: `run/verdict_ho3.json`, `run/phases.jsonl`,
  `summary.txt` (with the box's pins on its `KNOBS` line), `versions.txt`, `constraints.txt` and the three install logs;
- `public/`: the row receipts and the box's plans with their GPU-UUID values set to `null` (`ho3_public.py derive`);
- `SHA256SUMS`: every file of the box's receipt, with its sha256 in the private store;
- `STORE-ONLY.sha256`: the files kept only there. These are the launcher's files, the original receipts and plans, the logs
  that carry carriage returns, `forensics.txt` and `outer.log`, the run's markers and its staged copies of committed code.

```
python bench/ho3/ho3_public.py rebuild bench/ho3/receipts/ho3-a4000-r2 --out /tmp/verdict.json
```

This runs this checkout's `ho3_reduce.py` on the committed files with the derived receipts laid in. It rebuilds the box's
`verdict_ho3.json` byte for byte. `tests/test_ho3_public.py` checks all five rebuilds, the sums and their coverage, and that
no committed file names the host.
