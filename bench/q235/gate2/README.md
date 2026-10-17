# Q235 gate 2: the time-left fit, from the A2000 rehearsal

The A2000 leg of gate 1 ran `q235_run.sh` with `Q235_REHEARSAL=a2000` (a 4-layer slice of `ac9c66cc`, never science-eligible) at
4b301ec3. Its records, with no host paths:

- `run.json`, `phase-timings.jsonl`: the run and every phase's cap, elapsed time and exit code. Install and fetch ran on warm caches
  here. The cold-cache figures are the warm pass's: `warm-fetch.json` (27.05 GB fetched in 101 s, hashed in 23.7 s), and an install
  of 178 s.
- `hostrate.json`: pinned H2D 5.8 GB/s and D2H 6.6 GB/s on a PCIe gen 3 x8 slot, host copy 5.14 GB/s on 6 threads, O_DIRECT read
  1.63 GB/s.
- `a2000-leg-summary.json`: the box record's load profile and steps, Amendment 2's rules (9 of 9), the frozen base (111 tensors
  unchanged across both steps) and the fresh-process repeat (step 0's loss and per-layer outputs equal). The load took 116 s:
  - assemble 65.7 s (57 %), the host-side expert fusion, which bounds it;
  - read 30.1 s (22.39 GB at 0.743 GB/s effective);
  - other 11.5 s; offload 7.4 s; quantize 1.1 s.
- `fit-500-2.2.json`, `fit-500-1.6.json`: `q235_fit.py` for a host listing 500 Mbps at $2.20 and $1.60 an hour. Both fit:
  - each sealed cap covers its budget (box 8,820 s for 8,813; repeat 4,620 s for 4,583; fetch 12,000 s for 11,904);
  - the sealed floors equal their derivation (H2D 5.8 GB/s, host copy 5.14 GB/s, disk 0.743 GB/s), and the sealed pull floor
    (0.062 GB/s) is at least the rate at which the checkpoint, hashed at the rehearsal's rate, fits the fetch cap / 1.5.

  The worst-case wallclock is 8.25 h: $18.15 at $2.20 (above the $15 no-ask tier) and $13.20 at $1.60 (within it).

The fit is re-run for the pinned host's listed bandwidth and rate when the run is requested.

## Amendment 3: the repeat with the in-process null and plants

The repeat now runs the null and both plants in one process (`q235_box.py --repeat --plants`). The A2000 leg ran again at f7714702
(`run-a3.json`, `phase-timings-a3.jsonl`, `a2000-leg-a3-summary.json`), exit 0:
- Amendment 2's rules held (9 of 9), and the frozen base was unchanged across both steps.
- The null equals the box's step-0 output on every layer, bit for bit, at the same loss.
- Each plant left layers 0–1 bit for bit equal to the null and changed layer 2. The routing plant's forward took 1.689 s and the
  epilogue plant's 1.693 s.

That leg's load ran on a warm page cache (reads at 1.32 and 2.56 GB/s against the first leg's 0.743), so its phases would understate
a cold host. The fit therefore keeps the first leg as its basis and adds only the two plant forwards, 3.382 s, to its repeat before
scaling (`q235_fit.py --plants-from`):
- the repeat's budget rises from 4,583 s to 4,702 s, and its sealed cap from 4,620 s to **4,710 s**;
- the controller's allowance rises from 900 s to 1,800 s for the receipts: the box, the null and two plants, each 94 layers of
  2,648 × 4,096 bf16 outputs, about 8.2 GB.

`fit-a3-500-2.2.json` and `fit-a3-500-1.6.json` both fit, and the sealed floors still equal their derivation. The worst-case wallclock
is 8.53 h: **$18.76 at $2.20** an hour, above the $15 no-ask tier, and **$13.64 at $1.60**, within it. The first fit's files above
record the caps before this amendment.

`capacity-2648.json` (from `capacity.py`) is e4b's estimate at the run's shape, on main at 13b55990: 24.76 GiB of device memory and
158.62 GiB of pinned host memory for bf16 adapters (34.82 GiB of device memory for fp32), and 237.94 GiB host-to-device per
micro-batch.

## Amendments 4 and 5: the check copies on the host, and the host-RAM floor measured

The A2000 leg ran again at 9f6e0088 (Amendment 4: the box's fp32 check copies on the host, and a `host_memory` record), exit 0:
`run-am4.json`, `phase-timings-am4.jsonl` and `a2000-leg-am4-summary.json`.
- Amendment 2's rules held (9 of 9), and the frozen base was unchanged across both steps (111 tensors).
- The null equals the box's step 0, and each plant left layers 0–1 bit for bit equal to the null and changed layer 2.
- The device peak fell by 1.194 GiB at step 0 and 1.195 GiB at step 1, against 1.197 GiB predicted.
- At the step end, `RssAnon + RssShmem` was 13.53 GB, 1.26 GB above Amendment 4's terms at 4 layers.

`memslope.json` settles whether that excess grows with depth. The box's `--repeat` ran in a fresh process at 2, 4, 6 and 8 layers,
reading `host_memory` at the load end:
- shared memory grows 1.8119 GB per layer, the estimator's pinned homes to 4 decimals, with residuals of 52 kB;
- the excess is fixed: about 2.23 GB of shared memory and about 3.78 GB of anonymous memory, plus 15 MB per layer.

Projected to 94 layers with the check copies and held layer outputs, and assuming no heap reuse, that is 220.02 GB at the step end.
With the 18.5 GB allowance the science floor is 239 GB (Amendment 5).

## Amendments 6 and 7: the Shape B box and the timed pass

The A2000 leg ran at f17f0744 (Amendment 6's Shape B with rule 8's deterministic check, and Amendment 7's timed pass), exit 0:
`run-a7.json`, `phase-timings-a7.jsonl` and `a2000-leg-a7-summary.json`.
- Amendment 2's rules held (9 of 9), and the frozen base was unchanged across both steps.
- Rule 8's floor read exactly 0 at both steps, and its check 0.00278 and 0.00393 against BAR 0.0231.
- Amendment 7's overhead read: 0.0513 s against a drift noise of 0.0109 s, decision-grade.

That leg's load, like every leg after the first, ran on a warm page cache. Its time outside the load is cache-sensitive too: 123.8 s on
the cold first leg, against 81.9 s (f7714702) and 79.2 s (9f6e0088) on warm ones. So the fit keeps the first leg's whole box phase and
adds only the change, measured warm against warm (`q235_fit.py --box-from` this leg, `--box-baseline` the Amendment 4 leg):
- 222.0 − 79.2 = 142.8 s on the slice, a box basis of 392.8 s;
- the box's budget at 94 layers is 13,847 s, so its sealed cap goes from 8,820 s to **13,847 s**.

Every other phase still fits its sealed cap: the repeat's 4,702 s budget under 4,710; the fetch's 1,502 s (6,387 Mbps) to 2,499 s
(3,000 Mbps) budget under 12,000; and the install, under 1,800. The floors are unchanged. `fit-a7-*.json` hold the fit at the
last box's host listing (6,387 Mbps, $1.5139/h) and at the packet's bandwidth floor (3,000 Mbps, $1.70/h).

