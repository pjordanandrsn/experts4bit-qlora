# ho2-4090-2: HO2's RTX 4090 box, the files that carry no host identifier

One rented RTX 4090. experts4bit-qlora `f6e66fd6` (0.56.1), grouped-nf4-gemm `a83f6bb3` (0.49.0) and Loggetta `9348f8b8` (0.6.2),
2026-10-22. `SHA256SUMS` lists all 36 files of the box's receipt, with their sha256 in the private receipt store.

- **Committed byte-identical (10):** under `ho2/`:
  - `run/verdict_ho2.json` (the box's verdict) and `run/phases.jsonl`;
  - `summary.txt` (with the box's pins on its `KNOBS` line), `outer.log`, `versions.txt`, `forensics.txt` and `constraints.txt`;
  - the three install logs.
- **Store-only (26):** `STORE-ONLY.sha256` records each file at the same sha256.
  - The launcher's files: the receipt, `instance.json`, `known_hosts`, `teardown-proof.json`, `guard-armed.json` and the rest at the
    top level. These carry the rented host's instance id, machine and offer ids, or its ssh endpoint.
  - The four original row receipts (they carry the GPU's UUID).
  - The five logs that carry carriage returns (the fetch and the four trains).
  - The run's markers, and its staged copies of committed code. `staged-ho2.sha256` is the kit pinned at `f6e66fd6`.
- **Derived (`public/`):** `bench/ho2/ho2_public.py derive` wrote the four row receipts with `hardware.gpu.uuid` and
  `plan.hardware.gpu.uuid` set to `null`, serialized as Loggetta writes them. Each differs from its original in those two lines
  only. `public/SHA256SUMS` pins them.

## Rebuild

```
python bench/ho2/ho2_public.py rebuild bench/ho2/receipts/ho2-4090-2 --out /tmp/verdict_ho2.json
```

This rebuilds `ho2/run/verdict_ho2.json` byte for byte. The maintainer re-derived the verdict from the store.
`tests/test_ho2_public.py` checks the rebuild and that no committed file names the host.

- **Cost:** USD 0.461 actual against USD 2.60 worst (2 h × USD 0.75/h + 100 GB × USD 0.011/GB).
- **No identifiers:** no file here names an instance, an offer, a machine, a host name, an endpoint or the GPU's UUID.
