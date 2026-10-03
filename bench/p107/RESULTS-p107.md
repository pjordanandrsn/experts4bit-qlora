# P107 — results: **`DEFAULT=flash`**. On an RTX 5090, Qwen3-30B-A3B at `max_seqs` 1: a 4096-token prefill's device time 906 ms → 378 ms; TTFT-4096 1.984 s → 1.742 s (1.14×) on a CPU-bound host; the served-prefill NLL −0.014 / −0.008 ppl against math on 12 fresh windows

Issue: experts4bit-qlora#960. Pre-registration: `bench/p107/PREREG-p107.md` (#967, merged `8d3c306`, the launch
commit). Code under test: `E4B_PAGED_PREFILL_ATTN` (#963). Receipts: `bench/p107/receipts/p107-5090-3/`.

**Verdict by `p107_reduce.py`: `DEFAULT=flash`**, on the first full draw `p107-5090-3`. No VOID reason fired.

## Engagement (one 512-token request per route under `torch.profiler`)

| route | flash kernels | fp32 SGEMMs |
|---|---:|---:|
| math | 0 | 97 |
| flash | 48 | 1 |

- `math` ran the registered path: no flash kernel, and the 96 SIMT SGEMMs of QK^T and PV across 48 layers.
- `flash` ran one flash kernel per layer for the one chunk.
- The 97th SGEMM appears on both routes (one under `flash`), so it is not attention.
- The build census: 48 int4 expert layers on `int4_b32`; the int4 prefill route reads `k19`.
- **The premise passed on this card:** 1 passed, none skipped, with `rel flash vs math 0.00208` (the A2000 read 0.00209).

## Speed (median of three rotated rounds, one `max_new_tokens=1` request per route and length per round)

| route | TTFT-512 | TTFT-4096 |
|---|---:|---:|
| math | 0.249 s | 1.984 s |
| **flash** | **0.235 s** | **1.742 s** |
| flash vs math | 1.06× | **1.14×** |

- Rounds were tight: the spread was ≤ 1.0 % of the median for every route and length, except flash at 512 tokens
  (2.9 %).
- The rule needs flash's TTFT-4096 to be at most 0.9 × math's (1.786 s). It is 1.742 s.
- **First tokens were identical in every timed draw.**

## Quality: the served-prefill NLL (flash − math, perplexity, 2,048 scored tokens per window)

| text | W | mean Δppl | SD | windows over 0.05 | ppl (math) | verdict |
|---|---:|---:|---:|---:|---:|---|
| c4val1 | 8 | −0.0140 | 0.0365 | 1 (k = 16: −0.079) | 13.993 | |
| wikitext | 4 | −0.0085 | 0.0228 | 0 | 8.969 | |
| | | | | | | **PASS** (both \|mean\| ≤ 0.05) |

- Per window, flash − math moved the mean NLL by −0.0045 to +0.0023 nats, with no consistent sign.
- Every window scored the same text under both routes (same `text_sha`, 2,048 steps).
- One 2,560-token scoring took 0.68 s under math and 0.59 s under flash.

## Where a prefill spends its time (descriptive; one 4096-token request per route under `torch.profiler`)

| item | math: device ms | flash: device ms |
|---|---:|---:|
| K19 (the experts) | 138.8 | 138.2 |
| fp32 SIMT SGEMMs (QK^T, PV) | 225.4 | — |
| fp32 `where` / `add` / `isneginf` around them | 181.9 | — |
| fp32 softmax | ≥ 69.7 | — |
| flash attention (`flash_fwd_splitkv_kernel` × 288 + `flash_fwd_kernel` × 96) | — | 40.2 |
| device-to-device memcpy | 38.8 (50,736) | 38.0 (50,352) |
| **device total** | **905.8** | **378.1** |
| wall, profiled | 2.44 s | 2.33 s |

- Flash removes 528 ms of device time from a 4096-token prefill, more than the ~400 ms the pre-registration expected.
- **The wall time moved by 242 ms, not 528, because this host is CPU-bound.** Under flash the GPU is busy for about
  a sixth of the profiled wall.
- The host was an AMD EPYC 7663 (2 NUMA nodes; the build's own log advises `numactl --interleave=all`).
  `p102-5090-6`'s host was a Ryzen 9 7900. On it, the same math route (k19) read TTFT-512 0.113 s and TTFT-4096
  1.373 s, with the same ~920 ms device time. Here, math reads 0.249 / 1.984 s.
- **The next lead for TTFT is host-side, not a kernel.** About 50,000 device-to-device copies per 4096-token prefill
  remain under both routes, and the GPU idles between launches. This lane does not attribute the host time.

## The predictions, scored

| prediction | reading | held? |
|---|---|---|
| math TTFT-4096 1.25–1.55 s and TTFT-512 0.10–0.13 s, if the host is in `p102-5090-6`'s class | 1.984 / 0.249 s, on an EPYC 7663 host | not scored: the condition failed (the host is not in that class) |
| flash TTFT-4096 0.85–1.10 s, 1.25–1.6× faster | 1.742 s, 1.14× | **no** (the device saving was larger than predicted; the host bounds the wall) |
| flash TTFT-512 0.09–0.11 s | 0.235 s | **no** (same host) |
| flash exactly 48 flash kernels; math none and 96 SGEMMs | 48; 0 and 97 | yes (the 97th SGEMM is not attention) |
| PASS, \|mean Δppl\| ≤ 0.01 on both texts, no window over 0.05 | PASS; −0.0140 / −0.0085; one window at −0.079 | PASS held; the 0.01 bound held on wikitext only, and one window was over 0.05 |
| first tokens identical in every draw | identical | yes |
| `DEFAULT=flash` | `DEFAULT=flash` | yes |
| under flash, K19 is the largest device item | K19 138 ms, largest | yes |

## Box and cost

| run | outcome | cost (`cost_usd.actual`) | receipt (adertha-receipts) |
|---|---|---:|---|
| `p107-5090-1` | REFUSED by the launcher: an exclusion receipt failed on `stuck loading`, which it does not accept (my list) | $0 | `1e4be8f` |
| `p107-5090-2` | REFUSED by the launcher: single bandwidth refusals do not name a machine (my list) | $0 | `5594556` |
| `p107-5090-3` | OK, **`DEFAULT=flash`**, 2026-10-03T13:13:50Z → 13:32:29Z, instance 54015502 destroyed and absent | $0.1634 | `1aba741` |

- **Lane total: $0.1634** (ceiling $1.50).
- The first two attempts rented nothing. The third used P102's exclusion list after checking it against the
  launcher's own validator.

**The registered consequence:** a PR makes `flash` `E4B_PAGED_PREFILL_ATTN`'s default, with `math` kept selectable,
plus a CHANGELOG entry and a `docs/SERVING.md` note citing this read. The flip ships in the next release.
