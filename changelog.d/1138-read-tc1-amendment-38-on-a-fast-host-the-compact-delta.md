### Read: TC1 amendment 38 — on a fast host the compact delta is 1.3–1.6 % slower (P77, P78, P80 FALSIFIED); the peak holds (P79, P81 HELD); it stays opt-in

- `tc1-5090-85` ($2.64, EPYC 9655, machine 150700, 60-step load-gated draws): `NF4_QLORA_COMPACT_DELTA` 0 vs 1 with grouped-nf4-gemm#473.
  Qwen3 matched 1.016, shipped 1.013, Mixtral matched 1.013; peaks Qwen3 −0.312 GB, Mixtral −0.037 GB; held-out within 0.001.
- The flag stays opt-in. Across three hosts its speed follows how host-bound the step is (0.967–0.970 at 3.4–3.9 s steps, 1.013–1.016 at a
  2.17 s step): it trades host work for device work. Its memory saving holds on every host.
