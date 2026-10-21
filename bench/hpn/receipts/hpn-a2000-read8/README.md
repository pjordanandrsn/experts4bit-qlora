# hpn-a2000-read8: HPN's registered read, the public subset

One climb at L4 and L8 on the lab's RTX A2000, e4b `8dc66652`, 2026-10-21, $0.

- `hpn/result.json`, `hpn_spec.json`, `pin_proof_qwen3hpn.json`, `hp1/bulk_manifest.json` and `logs/` (12 logs) are
  byte-identical to the private receipt store. `SHA256SUMS` covers them and the derived files below.
- `proj/` and `hpn_spec.public.json` are derived, not copied. `bench/hpn/hpn_public_inloop.py` writes each reducer input's
  projection, holding only the fields `hpn_reduce.py` reads:
  - an arm receipt's `eval_curve`;
  - an evaluator output's bases (digest, slots, digest match, status, zero-adapter rows);
  - a reference output's runs.

  It also writes a spec that points at them. The registered reducer on that spec rebuilds `hpn/result.json`
  byte-identically: `python bench/hpn/hpn_reduce.py bench/hpn/receipts/hpn-a2000-read8/hpn_spec.public.json --out /tmp/r.json`.
- `STORE-ONLY.sha256` lists every other file of the run. That covers the original arm receipts and evaluator and reference
  outputs (they carry the host's identifiers or the run's own paths), the summary, box and forensics records, most logs, and
  the staged copies.
- **No identifiers:** no file here names an instance, an offer, a host name, an endpoint or the GPU's UUID.

Re-derive: `python bench/hpn/hpn_reduce.py bench/hpn/receipts/hpn-a2000-read8/hpn_spec.public.json --out {out}` writes `bench/hpn/receipts/hpn-a2000-read8/hpn/result.json`
