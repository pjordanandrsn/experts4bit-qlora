# ho3-5090-r1: HO3 Part A's RTX 5090 box, the files that carry no host identifier

One rented RTX 5090, 2026-10-23. experts4bit-qlora `f6e66fd6` (0.56.1), grouped-nf4-gemm `a83f6bb3` (0.49.0)
and Loggetta `9348f8b8` (0.6.2), launched from `e4230f82`. `SHA256SUMS` lists all 44 files of the box's receipt,
with their sha256 in the private receipt store.

- **Read:** COVERED: both families, 4 of 4 rows.
- **Committed byte-identical (8):** under `ho3/`: `run/verdict_ho3.json`, `run/phases.jsonl`, `summary.txt` (with the box's
  pins on its `KNOBS` line), `versions.txt`, `constraints.txt` and the three install logs.
- **Store-only (36):** `STORE-ONLY.sha256` records each at the same sha256: the launcher's files, the original row receipts
  and plans (they carry the GPU's UUID), the logs that carry carriage returns, `forensics.txt` and `outer.log`, the run's
  markers, and its staged copies of committed code.
- **Derived (`public/`, 8):** `bench/ho3/ho3_public.py derive` wrote the row receipts and plans with their GPU-UUID values
  set to `null`, serialized as Loggetta writes them. `public/SHA256SUMS` pins them.

## Rebuild

```
python bench/ho3/ho3_public.py rebuild bench/ho3/receipts/ho3-5090-r1 --out /tmp/verdict_ho3.json
```

This rebuilds `ho3/run/verdict_ho3.json` byte for byte. `tests/test_ho3_public.py` checks the rebuild and that no committed
file names the host.

- **Cost:** USD 0.472 actual against USD 3.40 worst.
- **No identifiers:** no file here names an instance, an offer, a machine, a host name, an endpoint or the GPU's UUID.
