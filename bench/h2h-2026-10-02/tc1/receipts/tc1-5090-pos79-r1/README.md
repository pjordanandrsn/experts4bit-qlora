# tc1-5090-pos79-r1: TC1 amendment 79's reading box, the files that carry no host identifier

One rented RTX 5090 (memory.total 32,607 MiB; Intel Xeon E5-2698 v4, 40 CPUs). The pinned release stack is:
- e4b 0.58.0 at `f15cb7d6` and grouped-nf4-gemm 0.51.0 at `30adb1bf`;
- unsloth and unsloth_zoo 2026.10.8, both proven against PyPI's published wheels before any arm ran (`tc1/release_proof_unsloth.json`);
- torch 2.12.1+cu130, run on 2026-10-24.

The kit was launched from e4b `1245fd0e` (experts4bit-qlora#2121). `SHA256SUMS` lists all 61 files of the box's `tc1/` receipt with their
sha256 in the private receipt store.

- **Committed byte-identical (35):** `tc1/`. These are the arm's GPU-clock and VRAM samplers, the four profiled arms' op lists
  (`*_profile.json`), Unsloth's release proof, `versions.txt`, `ds_manifest.json` and the logs.
- **Store-only (26):** `STORE-ONLY.sha256`, each file at the same sha256. Some carry the rented host's identifiers: the instance id, the
  container's host name and the GPU's UUID. These are `box.json`, the four cell receipts, `summary.txt`, `forensics.txt`, `outer.log`,
  `INSTANCE_ID` and `RESULTS-tc1.md` / `RESULTS.txt`. Two fetch logs carry carriage returns. The rest are the run's markers, its staged
  copies of committed code, and its dataset and token files. The launcher's receipt and its ledger row are only in the store.
- **Derived (`public/`):** `bench/tc1/tc1_public.py derive` wrote copies of `box.json`, the four cell receipts and `summary.txt` with
  exactly the identifying fields removed. `public/SHA256SUMS` pins them.

**The load gate** (TC1-PREREG amendment 33): every arm passed on attempt 0 (load1 medians 2.0 to 2.37, against 6.0).

## Rebuild

```
python bench/tc1/tc1_public.py rebuild bench/h2h-2026-10-02/tc1/receipts/tc1-5090-pos79-r1 --md /tmp/RESULTS-tc1.md
```

This rebuilds `public/RESULTS-tc1.md` byte for byte. That file differs from the store's `RESULTS-tc1.md` only by the two `box.json`
lines removed. `tests/test_tc1_public.py` checks the rebuild and that no committed file names the host.

**This box reads amendment 79 NOT_READ.** Both Unsloth draws are VOID on TC1's standing engagement check, so no ratio is quoted. The
cause and what is and is not known about it are in [`../../RESULTS-tc1-pos79.md`](../../RESULTS-tc1-pos79.md).

- **Cost:** USD 2.113 actual.
- **No identifiers:** no file here names an instance, an offer, a machine, a host name, an endpoint or the GPU's UUID.
