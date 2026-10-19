# `p131-5090-2`: the reading (M32_HELD M64_HELD DEFAULT_NONE)

One RTX 5090 (cc 12.0, driver 595.91.07, 32,607 MiB) on an AMD EPYC 7K62 host (96 CPUs, 503 GB RAM, a 200 GB disk):
Vast instance 56686506 on machine 152440 at $0.837/h, 2026-10-19 06:48Z to 07:52Z. Cost $1.048 (guard 2.5 h, worst
$2.80). Code: e4b `8e6471f5` (Amendment 2's merge, #1800), grouped-nf4-gemm `57f53e19`; torch 2.8.0+cu128, triton
3.4.0, transformers 5.17.0. Qwen/Qwen3-30B-A3B at `ad44e777`. The fetch gate (STOP-0) passed at the launch commit
(`p131_fetch_gate.json`). [`p131-5090-1`](../p131-5090-1/README.md) stalled at the pre-flight before this run.

**Verdict: M32_HELD M64_HELD DEFAULT_NONE.** Every integrity check held:
- the self-test read 44/44, and the premise read 9 passed;
- every capture was on, in every process, with none refused;
- the decode check was untouched under every arm;
- DETERMINISM held across all nine processes;
- FUNCTION read **BITWISE** for both arms on every speed window;
- Phase B: R repeated bit for bit, and S32 and S64 were `torch.equal` to R. The mutant was caught (bias 1.150 nats,
  argmax 0.576).

Median replay ms of the served first-chunk prefill graph at T = 512 (16 windows x 12 timed rounds per graph):

| process | m0 | m32 | m64 | m0 / m32 | m0 / m64 |
|---|---|---|---|---|---|
| 0 (warm-up, reported only) | 34.384 | 34.977 | 34.194 | 0.9831 | 1.0055 |
| 1 | 34.526 | 35.048 | 34.213 | 0.9851 | 1.0091 |
| 2 | 34.373 | 34.982 | 34.145 | 0.9826 | 1.0067 |
| 3 | 34.374 | 34.958 | 34.171 | 0.9833 | 1.0059 |
| 4 | 34.454 | 35.004 | 34.178 | 0.9843 | 1.0081 |
| 5 | 34.355 | 34.973 | 34.134 | 0.9823 | 1.0065 |
| 6 | 34.382 | 34.963 | 34.171 | 0.9834 | 1.0062 |
| 7 | 34.493 | 35.028 | 34.204 | 0.9847 | 1.0084 |
| 8 | 34.467 | 35.051 | 34.207 | 0.9833 | 1.0076 |

- **m32:** gain 0.9836 [0.9828, 0.9845] (t-interval over processes 1-8, 7 df). Slower in every process: NOT_FASTER.
- **m64:** gain 1.0073 [1.0063, 1.0083], under the x1.02 bar: NOT_FASTER. Saving 0.250 ms.
- **The order:** m64 over m32 x1.0241 [1.0236, 1.0246]. **The span:** m0 x1.0050 over processes 1-8.
- **The warm-up** (process 0, Amendment 2) read m0 at x0.999 of the scored median. The proof's first-process effect
  did not recur.
- **GPU state** (Amendment 2, reported only), the same in every process: 2392 MHz / 66 W / 35 C at the start of the
  speed pass; 2512 MHz / about 541 W / 54-55 C after the warm rounds and after the timed rounds, with clock event
  reason 0x4 (software power cap); 2752-2775 MHz / 245 W at the end. The timed rounds ran power-capped at one clock
  in all nine processes.
- **Eager** (reported, never gated): m0 90.0 ms, m32 105.5, m64 105.5 (process 1).
- **Load** 168-180 s a process; peak 20.97 GB (21.29 in process 0, with Phase B).

The maintainer re-derived the verdict from the private receipt store with the reducer at `8e6471f5` (equal to the
staged pin and the box's copy): the verdict, reasons, arms, order, determinism and default reproduce exactly. The
launcher's receipt and ledger row are in that store. `SHA256SUMS` covers every file here.
