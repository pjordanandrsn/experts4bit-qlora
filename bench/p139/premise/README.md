# P139's premise, read at $0 before the registration

Both ran on an RTX A2000 (sm_86) with grouped-nf4-gemm 0.46.1, torch 2.8.0+cu128 and transformers 5.17.0.

- `trace.json` (`k33_mixtral_trace.py`): a real-width Mixtral-8x7B (hidden 4096, intermediate 14336, 8 experts, top 2),
  two decoder layers with random weights, through the GPU serve smoke's own build (`bench/smoke/gpu_serve_smoke.py`
  `run_cell`) at e4b `9890a502`. `nf4_grouped._bw_engages` was wrapped before the build, so the record is what e4b's
  served decode hands K33's gate. It reads (28672, 4096) and (4096, 14336), 16 calls each (2 layers x 8 decode steps),
  each at T = 2.
- `numerics.json` (`k33_mixtral_numerics.py`): K33 at both shapes, 8 experts, 2 routed rows, every one of K33's twelve
  plans. Every plan meets K33's contract: `bw_prmt32` bitwise `bw_tree`, both within the fp32 reference's tolerance. The
  served token-row form (`gather_div=2`) is bitwise its expanded call. No plan is bitwise the scalar route.

The scripts are kept here as they ran. The box does not stage them; phase 1 (`p139_select.py`) re-checks the contract
on the 5090 before any timing.
