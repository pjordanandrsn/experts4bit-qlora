Real Loggetta training receipt, from loggetta commit 50c1553a00bee4ace2c1a06b2113e7c457aba3bc, path
`evidence/2026-10-09-a2000-moe-train-residual/A2/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261009T014815Z-a52e0454.json`
(unchanged on loggetta main at 11b9d6d8). Original receipt SHA256 a9082fd2b28663a5785c2a836ea3a161d80fb225fe86136fe80ede6f362e7014.

Two kinds of value are redacted, and nothing else changes:
- every string that holds a local path (`/home/` or `/tmp/`) reads `"<local path>"`: 5 values (provenance argv and source files,
  the adapter path);
- the two GPU `uuid` values (`hardware.gpu`, `plan.hardware.gpu`) read `null`.

It is re-serialized the way Loggetta writes a receipt (`json.dump(..., indent=1, sort_keys=True)`). Fixture SHA256
22ca086ef4461090cebe5b0d45d039858a0e391701541f2f7abf59eeaa0c3903.

`tests/test_ho2.py` reads it through HO2's reducer against its own run's values (OLMoE-1B-7B, 512 tokens × 2, 12 steps, Alpaca, the
RTX A2000), so every field the reducer reads is a field a real receipt carries. It also reads it against HO2's registration, which
must VOID it.
