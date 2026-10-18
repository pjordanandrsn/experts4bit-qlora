# Amendment 12's evidence: the RTX PRO 6000 timing leg `q235-leg-pro6000-2`

The receipts `q235_fit.py` reads, copied from the run's receipt (`leg/receipts/`; the per-layer tensors stay in the receipt
store): the run record (mode `pro6000`, `science_eligible` false), the phase timings, the slice fetch, both host-rate samples,
the quiet gate, the card and host records, and the box record (its load and load profile). `SHA256SUMS` covers every file.

- Host: one rented RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB), PCIe gen 4 x16, 112 CPUs, 540 GB RAM.
- Quiet gate: 0 compute apps, utilization 0 % over 10 samples, load average 1.56.
- Host rates: pinned H2D 28.36 and 28.33 GB/s; host copy 76.64 and 70.20 GB/s (8.4 % apart); O_DIRECT disk 1.73 and 3.33 GB/s.
- Phases: install 155 s, hostrate 35, fetch 216 (27.05 GB), tokens 4, box 85 (its load 15.1), repeat 20, hostrate2 37.
- The box load read the slice at 17.36 GB/s from the page cache (the fetch had just read and hashed it): a warm read.

`python bench/q235/q235_fit.py --rehearsal <a copy of leg> --warm-install-s 155 --bandwidth-mbps 806 --usd-per-hour 1.627`
gives exactly the floors and caps `q235_run.sh` seals (`tests/test_q235_fit.py`).
