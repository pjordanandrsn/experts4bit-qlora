# tc1-5090-152: TC1 amendment 73's box, the files that carry no host identifier

One rented RTX 5090, e4b `737f02ff` with grouped-nf4-gemm `8371c38d` (the field-device-ratio claim's commits), kit from experts4bit-qlora
`94a5198a`, 2026-10-22. `SHA256SUMS` lists all 50 files of the box's `tc1/` receipt with their sha256 in the private receipt store.

- **Committed byte-identical (25):** `tc1/`. Both arms' GPU-clock and VRAM samplers, `versions.txt`, `ds_manifest.json` and 19 logs.
- **Store-only (25):** `STORE-ONLY.sha256`, each file at the same sha256. These files carry the rented host's identifiers: the instance
  id, the container's host name and the GPU's UUID. They are `box.json`, the two cell receipts, `summary.txt`, `forensics.txt`,
  `outer.log`, `INSTANCE_ID` and `RESULTS-tc1.md` / `RESULTS.txt`. Four logs carry carriage returns. The rest are the run's markers, its
  staged copies of committed code, and its dataset and token files. The launcher's receipt and its ledger row are only in the store.
- **Derived (`public/`):** `bench/tc1/tc1_public.py derive` wrote copies of `box.json` (without `instance_id` and `hostname`), the two
  cell receipts (without `env.host`'s `uuid`, `vast_instance_id` and `container_host`) and `summary.txt` (without its BOX line). It
  removed nothing else, and it refuses a copy that still names the host. `public/SHA256SUMS` pins them.

## Rebuild

```
python bench/tc1/tc1_public.py rebuild bench/h2h-2026-10-02/tc1/receipts/tc1-5090-152 --md /tmp/RESULTS-tc1.md
```

This runs this checkout's `bench/tc1/tc1_reduce.py` on `tc1/` overlaid with `public/`. It rebuilds `public/RESULTS-tc1.md` byte for
byte. That file differs from the store's `RESULTS-tc1.md` only by the two `box.json` lines removed. The store's own `RESULTS-tc1.md`
re-derives byte for byte with main's `tc1_reduce.py`, and the maintainer re-derived the census from the store.
`tests/test_tc1_public.py` checks the rebuild and that no committed file names the host.

- **Cost:** USD 1.136 actual against USD 3.20 worst (2 h × USD 1.05/h + 100 GB × USD 0.011/GB).
- **No identifiers:** no file here names an instance, an offer, a host name, an endpoint or the GPU's UUID.
