# tc1-4090-2: TC1 amendment 78's draw 2, the files that carry no host identifier

One rented RTX 4090 (memory.total 24,564 MiB; AMD EPYC 9684X, a 30.72-CPU quota). e4b `576d99e9` with grouped-nf4-gemm `efdda943`,
torch 2.12.1+cu130, 2026-10-24. `SHA256SUMS` lists all 51 files of the box's `tc1/` receipt with their sha256 in the private receipt store.

- **Committed byte-identical (29):** `tc1/`. These are the arm's GPU-clock and VRAM samplers, `versions.txt`, `ds_manifest.json` and the
  logs.
- **Store-only (22):** `STORE-ONLY.sha256`, each file at the same sha256. Some carry the rented host's identifiers: the instance id, the
  container's host name and the GPU's UUID. These are `box.json`, the cell receipt, `summary.txt`, `forensics.txt`, `outer.log`,
  `INSTANCE_ID` and `RESULTS-tc1.md` / `RESULTS.txt`. Two fetch logs carry carriage returns. The rest are the run's markers, its
  staged copies of committed code, and its dataset and token files. The launcher's receipt and its ledger row are only in the store.
- **Derived (`public/`):** `bench/tc1/tc1_public.py derive` wrote copies of `box.json`, the cell receipt and `summary.txt` with exactly the
  identifying fields removed. `public/SHA256SUMS` pins them.

**The load gate** (TC1-PREREG amendment 33): attempt 0 passed (load1 median 1.41 against 6.0).

## Rebuild

```
python bench/tc1/tc1_public.py rebuild bench/h2h-2026-10-02/tc1/receipts/tc1-4090-2 --md /tmp/RESULTS-tc1.md
```

This rebuilds `public/RESULTS-tc1.md` byte for byte. That file differs from the store's `RESULTS-tc1.md` only by the two `box.json`
lines removed. `tests/test_tc1_public.py` checks the rebuild and that no committed file names the host.

**This box's RESULTS reads the draw VOID, "the machine id is unread".** That is the box-time render. The runner writes its RESULTS
before the launcher's receipt exists, and the reducer reads the machine id from that receipt. The lane's verdict
([`../../RESULTS-tc1-pos78fit.md`](../../RESULTS-tc1-pos78fit.md), `tc1_reduce.py --pos78-lane`) is run on the store, where each run's
receipt sits beside its `tc1/`. It reads this draw COMPLETE, on a different machine from draw 1's. The machine ids stay in the store.

- **Cost:** USD 0.928 actual, against USD 3.50 worst (1.5 h × USD 1.00/h + 100 GB × USD 0.02/GB).
- **No identifiers:** no file here names an instance, an offer, a machine, a host name, an endpoint or the GPU's UUID.
