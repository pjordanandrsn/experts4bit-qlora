### P160 registered (#1313): decode x plan, K39's decode at P156's carried tile plans against prmt at today's plan

P156 reported, never ruling, that prmt at its three carried plan pairs ran the two W16 K25 GEMMs about 1.6x faster than
prmt at today's plan (C1: gate_up 64/256/4/2, down 128/128/8/3, 1.6167 over P), selected with a 0.153 winner's curse.
P158 has since made prmt qwen3_moe's served default, so P is the control. P160 re-asks C1 under a registered rule.
- **Arms:** P and P2 (prmt at today's plan, the self-pair), C1-C3 (prmt at P156's carried pairs, named before data),
  T (the tree, reported). Only the plan moves between P and C1.
- **Data:** GO is read on S1, P153's second router-replay stream, which no K25 microbench has timed (126 reps); S0's
  held-out steps 64-123 re-read P156's figure, reported. S1 shares S0's prompts; its overlap is stated before data.
- **The defences:** an untimed spacer graph before every timed replay, a Williams order over the six arms, NVML clocks
  and the enforced limit per arm. GO needs C1's median P/C1 minus 2 SE >= 1.05 both raw and clock-adjusted (the whole
  clock gain taken from a faster-clocked C1); a sample under a 575 W enforced limit VOIDs.
- **Box:** one RTX 5090, grouped-nf4-gemm v0.50.0 (41ef086e), premise 27 tests (K25's row-exact tests and K39's compiled
  contract), guard 1.0 h with the pre-install fit at 25 % headroom from P156's box walls, worst $1.16. A $0 A2000
  shakedown ran end to end before the registration.

A GO opens an opt-in per-projection K25 plan in e4b and a served read, each its own PR. Files: `bench/p160/` and
`tests/test_p160.py`.
