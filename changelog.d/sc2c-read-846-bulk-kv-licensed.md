### SC2c read (#846): bulk KV bookkeeping licensed as a default; capacity ceiling 2 → 4 req/s on one 5090 (bench only)

- **The reading** (`sc2c-5090-1`, $1.27; lane total $1.458 over 4 receipts): `E4B_PAGED_BULK_KV` OFF against ON on
  Qwen3-30B-A3B int4, paired, with the prefill graph at `auto` in both arms. Every gate passed, and output was
  byte-identical OFF against ON in both draws.
  - The bucket-controlled stall per prefill fell from 0.211 / 0.199 s to 0.040 / 0.039 s (P1).
  - Serial TTFT fell from 161 / 156 ms to 42.7 / 41.8 ms (P2, 3.78× and 3.74×).
  - The ceiling rose from 2 to 4 req/s (P4, P5).
  - P3 was refuted on the fast side (TPOT ON / OFF 0.943): OFF's first-decode block claims sit inside TPOT, while the
    decode step is unchanged by bucket.
  - **DEFAULT_LICENSED.** The flip is a separate PR.
- **The census** closes on the box: the direct stall and the fitted stall agree within 1 ms on every server.
  - OFF's prefill step is a 152–158 ms host flush over a 38.5 ms forward, plus 44–48 ms of first-decode claims.
  - ON's step is the forward.
- **Post hoc** (descriptive): `serve_capacity` on each server's own step costs reproduces 12 of 16 attainment cells
  within 0.03 and ON's ceiling. Under the model, 8 req/s needs slots or decode-step time; the forward comes third.
