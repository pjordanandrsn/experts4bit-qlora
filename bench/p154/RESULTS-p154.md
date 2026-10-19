# P154 results: the W16 NF4 expert byte floor of Qwen3-30B-A3B's served step, on the replay's own route (e4b#1313, stream (B))

**Status:** closed. **READ** on the one run. REPORTING only: this prices headroom and names no lever. It changes no
default, no gate and no speed claim.

[P152](../p152/RESULTS-p152.md) named the grouped NF4 small-M expert GEMM at 71.2 % of the unprofiled 16-request step,
but its byte floor read UNREAD: the routing pass had run off the replay's route. [The registration](PREREG-p154.md) read
the floor on the replay's own route, and compared it with the GEMM times P152's census recorded, from P152's committed
verdict. Nothing was re-timed.

| run | what | cost | verdict |
|---|---|---|---|
| [`p154-5090-1`](receipts/p154-5090-1/README.md) | the routing pass on the replay's route, the store read, a copy rate | $0.399 | **READ** |

Total $0.399 on one RTX 5090 launch, inside the registered worst case of $1.72. The public receipt copy is under
[`receipts/`](receipts/), with a README and `SHA256SUMS`. The launcher's receipt and ledger row are in the private receipt
store. The maintainer re-derived `verdict_p154.json` byte for byte from the store: the registered reducer, its self-test
(14/14), and P152's pinned verdict.

## The read (`p154-5090-1`)

- **The box:** one RTX 5090 at a 575 W power limit read on the box. e4b `20910df4`, grouped-nf4-gemm v0.47.1 `38b30888`
  (P152's, on purpose: the K25 build P152 timed). It is the same card as P152's read, by the GPU UUID in both forensics.
- **The premise:** K25's row-exact GPU tests, 4 passed on the card.
- **The route, both witnesses exact:**
  - (singleton, device) = (False, True) at 16 rows;
  - **6,144** K25 calls in the window;
  - the library's own route record across the window, exactly {`nf4_k25|le256`: **3,072**}.
- **The store:** NF4 on all 48 MoE layers (packed `uint8` plus `float32` absmax per 64). Bytes per expert, read off the
  loaded tensors: **1,769,472** for gate_up and **884,736** for down, 0.5625 B a weight. The arena was this run's own bake.
- **The head:** bf16, P152's server (`E4B_INT8_HEAD_BW=0`). No faults.
- **The routing on the replay's route:** **58.07** unique experts per layer-step (42 to 81), 2,787 a step. Uniform routing
  would give 82.42. P152's eager, host-grouped pass read 57.02 over the same window.

## The floor

The floor is the mean unique experts a step x bytes per expert, over P141's copy rate of 1,498.9 GB/s, against P152's
recorded removable time for each GEMM line:

| | byte floor | recorded (P152) | fraction of copy | headroom |
|---|---|---|---|---|
| gate_up (`136x48`) | 3.2904 ms | 8.3951 ms | 0.392 | 5.1047 ms |
| down (`136x64`) | 1.6452 ms | 3.3530 ms | 0.491 | 1.7078 ms |
| **both** | **4.9356 ms** | **11.7480 ms** | **0.420** | **6.8124 ms** |

- **The headroom is 41.3 % of the unprofiled 16.497 ms W16 step.** The two GEMMs read their weights at 42 % of the copy
  rate, gate_up at 39 %.
- **Reported beside it:** over the step totals' range (2,521 to 3,097 unique a step) the combined floor runs 4.46 to 5.48
  ms. At the box's own copy rate (1,523 GB/s) it is 4.857 ms.
- **Against P152's W1 candidate:** the W16 headroom is 22.5x the W1 expert GEMV headroom of P152's re-read
  (0.303 ms), and 25x P142's original 0.269 ms.

## Predictions (graded, never ruling)

| prediction | read | grade |
|---|---|---|
| device-grouped K25 on NF4 on 48 layers; 6,144 K25 calls, 3,072 `nf4_k25\|le256` | exactly that | HELD |
| bytes per expert 1,769,472 / 884,736 | exactly that | HELD |
| 50-65 unique experts per layer-step | 58.07 | HELD |
| gate_up floor 2.7-3.7 ms at 0.32-0.45 of copy | 3.29 ms at 0.392 | HELD |
| down floor 1.35-1.85 ms at 0.40-0.55 | 1.65 ms at 0.491 | HELD |
| headroom over both at least 5 ms | 6.81 ms | HELD |
| box copy rate 1,350-1,650 GB/s | 1,523 | HELD |

## What it means for stream (B)

- **The lever is the grouped NF4 small-M GEMM at W16.** It reads its experts' bytes at 42 % of the copy rate, with
  6.8 ms, 41 % of the 16-request step, between its recorded time and its byte floor. The next line in the step,
  `Kernel2` 8x16, takes 1.11 ms in total, about a sixth of this headroom.
- **The next registration is a lever on that kernel's plan at Qwen3's W16 shapes** (its own registration, with a served
  read). Each layer routes about 2 real rows to each of about 58 experts, so the 16-row tiles carry mostly padding. Whether
  the gap is the tile geometry or something else is that lane's question; this read only prices it.

## What this lane cannot say

- **Nothing about speed or a lever.** It prices headroom; a kernel change and its gain are their own registrations.
- **The floor and the recorded GEMM times come from two rentals.** The routing and the store were read on `p154-5090-1`,
  and the GEMM times are P152's. Both ran on one card (the same GPU UUID) at the same pins, so card-to-card variation is
  out, but rental-to-rental conditions (clocks, co-tenancy, temperature) are not.
- **Not that the routing pass saw the replay's experts step for step.** It ran the replay's route eagerly, not the
  captured graph, and the window's tokens were not compared.
- **Nothing about other cards, families or loads:** one RTX 5090, Qwen3-30B-A3B, 16 rows in steady decode.

## Reproduce

From a checkout of e4b at `20910df4`, with this lane's receipt copy and P152's committed verdict:

```
python bench/p154/p154_reduce.py --self-test
python bench/p154/p154_reduce.py --route bench/p154/receipts/p154-5090-1/route_b16.public.json \
  --p152 bench/p152/receipts/p152-5090-2/verdict_p152.json --out verdict_p154.json
```

The output equals `receipts/p154-5090-1/verdict_p154.json` byte for byte.
