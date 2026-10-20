# hbd-5090-1: the HBD box's outputs that carry no host identifier

One RTX 5090, e4b `248c35ae`, 2026-10-20. Every file here is byte-identical to the private receipt store, and `SHA256SUMS`
covers them. `STORE-ONLY.sha256` lists everything else in the run's receipt with its sha256. That covers the cell receipts,
`summary.txt`, `box.json`, `forensics.txt` and `outer.log` (they carry the rented host's identifiers), six logs with carriage
returns, and the run's staged copies, dataset and token files. The launcher's receipt and its ledger row are only in the store.

- `hbd/rec_<cell>.json`: each cell's in-cell record. It holds the stored-state digest and per-slot hashes, the per-slot
  dequantized fingerprints, the slot structure, the `libbitsandbytes` mapped in the cell, and the versions and device.
- `hbd/probe_<cell>.json`: the sibling probes.
- `hbd_spec.json` and `hbd/result.json`: the reducer's spec and its output. The maintainer re-derived it from the store,
  byte-identical.
- `hp1/bulk_manifest.json` and `logs/`: the run's own records.

Each record's `per_slot_dequant` re-folds with `hp1_lib.fold_fingerprints` to its cell's dequantized digest.
`tests/test_hbd.py` checks this, and the one cross-build slot, on these files.
