### HO3 read: ERNIE's host estimate is low with its experts on the host; every device row held

- Part B (two RTX A4000 machines, the first held-out rows with host offload): LFM2 COVERED 4 of 4. ERNIE, with its experts
  on the host as planned, is MISS on both boxes: its host estimate was under the measured host peak on 3 of 4 rows
  (1.0056× to 1.0548×, at most 808 MiB over), while its device side held on all four (driver / plan 0.73 to 0.79).
- Part A: the RTX 5090's second box COVERED 4 of 4. The RTX 4090's two started boxes both read VOID (a host whose
  `nvidia-smi` listed no per-process memory, then a numerics failure), which ends that part, so its cohort keeps one box.
- Every read box planned what the registration registered. USD 2.103 against the USD 27.23 worst.
- `bench/ho3/RESULTS-ho3.md`: the read. `bench/ho3/receipts/<box>/`: each started box's identifier-free files, byte-identical,
  with its row receipts and plans with the GPU UUID nulled (`public/`). `forensics.txt` and `outer.log` stay store-only.
- `bench/ho3/ho3_public.py` derives those copies and rebuilds each box's verdict byte for byte; `tests/test_ho3_public.py`.
- HO3 is closed: `tests/test_ho3.py` reads its staged pins at `5790042a` (bench/common rule 6).
