# ho3-a4000-r2: HO3 Part B's second RTX A4000 box, on another machine, the files that carry no host identifier

One rented RTX A4000, 2026-10-24. experts4bit-qlora `f6e66fd6` (0.56.1), grouped-nf4-gemm `a83f6bb3` (0.49.0)
and Loggetta `9348f8b8` (0.6.2), launched from `5790042a`. `SHA256SUMS` lists all 46 files of the box's receipt,
with their sha256 in the private receipt store.

- **Read:** MISS: LFM2 COVERED 2 of 2; ERNIE (experts on the host) 0 of 2, both HOST_UNDER.
- **Committed byte-identical (8):** under `ho3/`: `run/verdict_ho3.json`, `run/phases.jsonl`, `summary.txt` (with the box's
  pins on its `KNOBS` line), `versions.txt`, `constraints.txt` and the three install logs.
- **Store-only (38):** `STORE-ONLY.sha256` records each at the same sha256: the launcher's files, the original row receipts
  and plans (they carry the GPU's UUID), the logs that carry carriage returns, `forensics.txt` and `outer.log`, the run's
  markers, and its staged copies of committed code.
- **Derived (`public/`, 8):** `bench/ho3/ho3_public.py derive` wrote the row receipts and plans with their GPU-UUID values
  set to `null`, serialized as Loggetta writes them. `public/SHA256SUMS` pins them.

## Rebuild

```
python bench/ho3/ho3_public.py rebuild bench/ho3/receipts/ho3-a4000-r2 --out /tmp/verdict_ho3.json
```

This rebuilds `ho3/run/verdict_ho3.json` byte for byte. `tests/test_ho3_public.py` checks the rebuild and that no committed
file names the host.

- **Cost:** USD 0.262 actual against USD 2.09 worst.
- **No identifiers:** no file here names an instance, an offer, a machine, a host name, an endpoint or the GPU's UUID.
