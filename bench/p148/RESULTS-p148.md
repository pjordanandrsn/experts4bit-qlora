# P148 results: the dense bf16 decode GEMVs, a falsify-first microbench (e4b#1313)

**Status:** closed. **D1_GO+D2_GO.** A microbench premise, never the quoted gain: a GO starts kernel work and a served
read of its own, and no default, gate or speed claim changes. No claims row.

P142's per-name table put the dense bf16 GEMVs (cuBLAS `kernel`) at 41 % of the one-request step on an RTX 5090 (P =
4.380 ms, Qwen3-30B-A3B), the top removable lines after P143's attention cap. P148 asked whether two kernel levers clear
2 % of that step each, in one CUDA-graph harness against cuBLAS, before any of them is built:
- **D1:** a bandwidth-targeted bf16 GEMV for fused q/k/v (5120 x 2048) and o (2048 x 4096), 48 of each per step.
- **D2:** an int8 lm_head (151936 x 2048, per-row scale), half the bytes.

| run | what | cost | verdict |
|---|---|---|---|
| [`p148-5090-1`](receipts/p148-5090-1/README.md) | the microbench | $0 | refused at create (`no_such_ask`; not a strike) |
| [`p148-5090-2`](receipts/p148-5090-2/README.md) | the microbench, the registered spare | $0.025 | **D1_GO+D2_GO** |

The maintainer re-derived `verdict_p148.json` byte for byte from the receipt store (`6caf2dc2`; the reducer at
`caf5a28c`, equal to the staged pin and the box's copy), and recomputed it without the reducer from `p148.json`.

## The read

Every rung of the [registered rule](PREREG-p148.md) held:
- **The harness ran P142's kernel.** The profiled cuBLAS call launched P142's in-situ grid at every shape (1280, 512
  and 37984).
- **It transferred.** cuBLAS read 1.036, 1.061 and 1.009x P142's in-situ times, inside [0.8, 1.25]. It was slightly
  slower than in situ, not faster.
- **It was quiet.** Every self-pair was within 1.0008 of 1, inside [0.985, 1.015].
- **The contract held.** Every config passed (36, 36 and 18). q/k/v and o read cuBLAS's own fp64 error, and the int8
  head read 1.660e-3 against its dequantized weights, where cuBLAS reads 1.665e-3 on the bf16 head.

The bars rule on each shape's **projected in-situ saving**, the smaller pair's P142 time x (1 - B/A):

| lever | shapes | projected saving | share of P | bar | raw harness saving (reported) | verdict |
|---|---|---|---|---|---|---|
| D1 | q/k/v 0.0597 + o 0.0715 | **0.131137 ms** | 2.99 % | 0.087603 ms | 0.137712 ms | **GO** |
| D2 | head 0.1895 | **0.189546 ms** | 4.33 % | 0.087603 ms | 0.191328 ms | **GO** |

The choices ran at 83 % (q/k/v), 81 % (o) and 92 % (the int8 head) of the card's 1.792 TB/s. Per-shape medians and the
selection are in the [receipt's README](receipts/p148-5090-2/README.md).

Two things the read showed beyond the verdict (reported, never ruling):
- **Split-K lost at every shape.** The best split-K config was about 20 % slower than the best unsplit one at q/k/v
  and o (0.818 against 0.679 ms, and 0.696 against 0.558). The prototype sums the partials in a separate pass. Both
  shapes chose 2-row blocks over full-K chunks of 1024, unsplit.
- **The int8 head reads near the bandwidth floor at every config.** All 18 configs read between 0.188 and 0.192 ms,
  so in the microbench the configuration hardly matters. Quality is the next question, and the served read still owes
  the speed.

**The host.** Machine 40521 runs its 5090 at a 400 W power limit. P148 registered no power term. The verdict is ratios
on one host projected onto P142's in-situ times, and these GEMVs are DRAM-bound, so a power cap should move them least
of anything in the step. The served reads below carry a power term in their registrations.

## Predictions (graded, never ruling)

| prediction | read | grade |
|---|---|---|
| D1 saves 0.08-0.18 ms in total; D1_GO | 0.131 ms; GO | hit |
| D1's best configs reach 85-92 % of peak at q/k/v and o | 83 % and 81 % | miss |
| D2 saves 0.15-0.20 ms; D2_GO | 0.190 ms; GO | hit |
| every self-pair within [0.99, 1.01] | within 1.0008 | hit |

## What the GOs start

Routed by the maintainer after the re-derivation:
- **D2 first.** A served registration that rules the int8 head on **quality** at both of the split premise's gates (t1
  and k16: argmax >= 0.95 and d + 1.645 SE <= the bound) before any speed read. The microbench reads the int8 head
  near the bandwidth floor at every config; the served read still owes its speed, after the quality ruling. The
  earlier I8HEAD reads were reported, not ruled, and Qwen3's 151936-row head must earn its own.
- **D1.** A grouped-nf4-gemm dense bf16 GEMV with no split-K, under its own tolerance contract, then a served read in
  P143's form.

Each runs under the standing rule, with its own registration and review.
