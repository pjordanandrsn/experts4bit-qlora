### SC1g A3 read (#846): MXFP4's cost on in-distribution chats is the weights, not e4b's route, and it does not replicate across windows; NF4's lower NLL is most likely entropy flattery (hypothesis: P44's KL; A4 adds a fidelity instrument) (bench only)

- **`sc1g-diag-2` ($0.726):**
  - MXFP4 served repeats box J bit for bit across hosts (0.904969107589033).
  - **K1 REFUTED:** the MXFP4-weights prefill reads +0.157 over NF4.
  - **K2 REFUTED:** the int8 activations carry 0.185 of the gap on `conv1` and none on `conv2`.
  - **K5 REFUTED across windows:** the gap is +0.169, +0.097 and −0.033 on `conv1`, `conv2` and `conv3`.
  - By the registered rules, nothing is filed on e4b.
  - The receipts of both box J runs are committed (`bench/h2h-2026-10-02/sc1g/`), and the lines re-derive from them with
    `sc1g_reduce.py`.
- **P44 already measured the native store as ten times closer** to a bf16 dequant reference than NF4 (KL 0.0019 vs 0.0222).
  So NF4's lower teacher-forced NLL on off-policy chat text is most likely entropy flattery. That is a hypothesis resting
  on P44's KL, untested in this lane. A4 will read cross-engine NLL as descriptive only and add a KL fidelity instrument.
- **e4b#1175:**
  - The 5090 attention check reads INERT. Synthetic N(0,1) sinks carry about 1/T of the softmax mass, so the no-sink
    mutation had no power. The kernel agrees within 1.7–1.9e-3 at k_groups 4, 8 and 16.
  - $0 on the A2000: the served loop's KV writes (`append_prompt`, then `append_many` per decode token) are bitwise equal
    to per-layer `append` at every group count (`sc1g_prompt_append_check.py`).
  - The kg16 regression is narrowed to the kernel under real sinks and lengths.
