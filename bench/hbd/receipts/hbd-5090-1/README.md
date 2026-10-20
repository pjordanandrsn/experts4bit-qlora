# hbd-5090-1: the HBD box's own outputs

One RTX 5090, e4b `248c35ae`, 2026-10-20. Every file here is byte-identical to the private receipt store. `SHA256SUMS`
covers them, and `STORE-ONLY.sha256` lists the six logs kept only in that store, because they carry carriage returns and
receipts are never rewritten. The launcher's receipt and its ledger row are only in the store.

- `qwen3hbdL2_<framework>_hbd_<cell>.json`: the eight cell receipts. `hp1.train_base` holds the dequantized digest; `hp1.hbd`
  holds the stored-state digest, the slot structure, the `libbitsandbytes` mapped in the cell, and the versions and device.
- `hbd/rec_<cell>.json`: each cell's in-cell record, with the per-slot stored-state hashes and the dequantized fingerprints.
- `hbd/probe_<cell>.json`: the sibling probes.
- `hbd_spec.json` and `hbd/result.json`: the reducer's spec and its output. Reproduce it with
  `python bench/hbd/hbd_reduce.py bench/hbd/receipts/hbd-5090-1/hbd_spec.json --out /tmp/result.json`, which equals
  `hbd/result.json`.
- `summary.txt`, `outer.log`, `logs/`, `box.json`, `forensics.txt`, `hp1/bulk_manifest.json`: the run's own log and records.
