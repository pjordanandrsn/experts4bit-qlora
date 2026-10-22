# tc1-5090-154: TC1 amendment 74's box, the files that carry no host identifier

One rented RTX 5090 (AMD EPYC 7402P, a 23.04-CPU quota), e4b `cedddf84` (0.56.1) with grouped-nf4-gemm `a83f6bb3` (0.49.0), 2026-10-22.
`SHA256SUMS` lists all 58 files of the box's `tc1/` receipt with their sha256 in the private receipt store.

- **Committed byte-identical (33):** `tc1/`. The four arms' GPU-clock and VRAM samplers, `versions.txt`, `ds_manifest.json` and 23 logs.
- **Store-only (25):** `STORE-ONLY.sha256`, each file at the same sha256. These files carry the rented host's identifiers: the instance
  id, the container's host name and the GPU's UUID. They are `box.json`, the four cell receipts, `summary.txt`, `forensics.txt`,
  `outer.log`, `INSTANCE_ID` and `RESULTS-tc1.md` / `RESULTS.txt`. Two logs carry carriage returns. The rest are the run's markers, its
  staged copies of committed code, and its dataset and token files. The launcher's receipt and its ledger row are only in the store.
- **Derived (`public/`):** `bench/tc1/tc1_public.py derive` wrote copies of `box.json`, the four cell receipts and `summary.txt` with
  exactly the identifying fields removed. `public/SHA256SUMS` pins them.

## Rebuild

```
python bench/tc1/tc1_public.py rebuild bench/h2h-2026-10-02/tc1/receipts/tc1-5090-154 --md /tmp/RESULTS-tc1.md
```

This rebuilds `public/RESULTS-tc1.md` byte for byte. That file differs from the store's `RESULTS-tc1.md` only by the two `box.json`
lines removed. The store's own `RESULTS-tc1.md` re-derives byte for byte with main's `tc1_reduce.py`, and the maintainer re-derived the
read from the store. `tests/test_tc1_public.py` checks the rebuild and that no committed file names the host.

- **Cost:** USD 0.914 actual against USD 3.20 worst (2 h × USD 1.05/h + 100 GB × USD 0.011/GB).
- **No identifiers:** no file here names an instance, an offer, a host name, an endpoint or the GPU's UUID.
