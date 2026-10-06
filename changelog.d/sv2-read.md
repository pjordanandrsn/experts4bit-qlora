### Read: SV2 (#1207): the int4 serving levers measured beside the estimate on Qwen3-30B-A3B with decode graphs (bench and receipts only; licenses no change)

- `sv2-5090-1`: one RTX 5090, **$0.45**, teardown proven. Four arms OK, integrity clean.
- Every reading fell inside its registered expectation (`bench/sv2/RESULTS-sv2.md`):
  - V1: `q_exp` 0.8% under its estimate;
  - V2: the int4 stores +54.0 MiB at load against +54 priced;
  - V3: int4 attention +585.2 MiB against +585.0 priced (the prereg's "+571 MiB" was 0.571 GiB, a unit slip);
  - V5: the repack's host peak 3.30 GB against 6.9 GiB priced;
  - V6: after load, the int4 builds hold 1.5 GB less host memory than the NF4 build.
- V4, measured only: the prefill graph at int4 costs +571 MiB (pool 294 MiB). That equals SV1's NF4 figure; SC2b's
  +3.3 GiB (0.47.0) is not reproduced.
- **Process (maintainer note, 2026-10-06):** the box launched at 22:39:55Z, one minute after the maintainer asked for
  registered consequences, a reducer and an exit-code fix on #1208, from a registration not yet on main. #1208 and this read
  were then merged over those requests. The expectations were public before the data; no consequence was registered, so
  this read licenses no change to the estimate, the repack price or the trims. See `bench/sv2/RESULTS-sv2.md`.
