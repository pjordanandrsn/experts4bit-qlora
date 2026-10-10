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
