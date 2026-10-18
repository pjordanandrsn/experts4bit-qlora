# P138 results: the verify split (#1313)

**Verdict: READ, H = REFUTE, BUILD = NO_BUILD.** This is `vsplit-5090-3`, the last box-started attempt. The maintainer
re-derived it independently on the committed records, with an identical verdict
(`verdict_vsplit.json` sha256 `025a6e9b908cdd4a…`). The reduced records are in `receipts/vsplit-5090-3/`, and their
digests are in `SHA256SUMS`.

## The run

| | |
|---|---|
| card | one RTX 5090 (`card.json`, `forensics.txt`) |
| subject | SD2's read server: e4b `88ae1cfd`, grouped-nf4-gemm v0.45.0 `724ccc45`, EAGLE-3 at k = 3 (head `6afc5aa2`), Qwen3-30B-A3B at `ad44e777`, 16 slots, graphs on |
| harness | e4b `bd51c7d9` (registration #1727, Amendment 1 #1729, Fix 1 #1733) |
| box | rc 0 in 1210 s; cost $0.367 |

## The reading

Every millisecond comes from this card's own two unprofiled timing passes. The census contributes only shares of its
own profiled period.

| | n = 2 | n = 3 | n = 4 |
|---|---:|---:|---:|
| verify_ms(n), mean of passes A and B | 7.890 | 8.371 | 8.771 |
| I_b(n) = verify_ms(n) − anchor (3.927 ms) | 3.963 | 4.444 | 4.844 |
| bytes(n) = (ρ − 1) × the n = 1 expert ms | 0.808 | 1.462 | 2.029 |
| excess floor, max(spread, 0.05 × I_b) | 0.198 | 0.222 | 0.242 |
| dExp(n), the expert path's increment | 1.569 | 2.044 | 2.546 |
| S(n) = dExp − bytes | 0.762 | 0.582 | 0.517 |
| **share = S / (I_b − bytes)** | **0.241** | **0.195** | **0.184** |
| need(n), SD2's speed bar with this box's terms | 2.259 | 1.690 | 1.674 |
| registered prediction for the share (lo / central / hi) | 0.782 / 0.873 / 0.990 | 0.760 / 0.856 / 0.979 | 0.735 / 0.836 / 0.966 |

- **H: REFUTE.** The share is below 0.50 at every n. Every n carries an excess above its floor, so none reads
  NO_EXCESS.
- **BUILD: NO_BUILD.** S(n) < need(n) at every n.
- **Gates:** A/A 1.0016 (pass A) and 1.0120 (pass B), within 3 %. The self-pair is within 1 % on the anchor and on
  every verify_ms. The canary passed. Census n = 1 is `ok`; n = 2 and 4 carry `PROFILER_INFLATED`, and n = 3 also
  `NODE_TRACE_INFLATED`. Those are reporting labels, read through under Fix 1.
- **Robustness (the maintainer's re-derivation note):** the census's profiler inflation at n ≥ 2 (census over
  unprofiled 1.129 / 1.134 / 1.157) is far smaller than either margin: S / need is about 0.33, and the shares are
  0.18–0.24 against 0.50.

**Reporting, never a rung.** As a fraction of I_b(n) at n = 2: `moe_expert` 0.463, `dense_gemm` 0.175, `norm_elem`
0.045, `moe_route` −0.067, `attn` −0.007. The part of I_b not attributed to any class is about 1.50 / 1.41 / 1.42 ms
at n = 2 / 3 / 4.

## The consequence

As registered, **NO_BUILD closes the B = 1 multi-token (speculative verify) lane for Qwen3-30B-A3B on this stack**,
with this split as its record. No 2–4-row expert kernel is registered or built, and no default changes. SD2's
`E4B_PAGED_SPEC` stays off.

## What this cannot say

Nothing about other cards, other models or stacks, B > 1 batched decode, or quality. The unattributed part above is
reported, not explained. Any lane that takes it up needs its own registration.
