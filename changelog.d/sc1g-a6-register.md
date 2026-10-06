### SC1g amendment A6 registered (#846): box J under A5's instrument tests whether e4b's NF4-prefilled prompt carries its excess KL, graded within the box by the median (bench and tests only)

- **Why.** These observations were found after A5's data, at $0, from the committed per-position records:
  - A5's mean KL is tail-dominated: the top 1% of positions carry 38–59% of each window.
  - At the median, e4b MXFP4 served is 4–6× further from the reference than vLLM on all four windows.
  - conv2's excess is front-loaded.
  - e4b's served stack (`E4B_INT4_KEEP_NF4=1`) runs the 512-token prompt's MoE on NF4. vLLM and box R use MXFP4.
- **The box.** Box J (e4b only, guard 1 h, about $1) reuses box R's registered rows and grades every arm against its own
  baseline (b): (a) the prompt on MXFP4 too (`KEEP_NF4=0`), (c) bf16 decode activations (`E4B_MXFP4_GEMV=0`), and (b') a
  determinism repeat. Each arm runs in its own process, with engagement gates on its routes.
- **Predictions.** P1 (prompt route), P2 (conv2 front-loading) and P3 (int8 activations) each have three-way outcomes and
  bars set before data.
- **Consequences.** P1 HELD licenses only a priced serve A/B, never a default change on fidelity alone.
- **The noise caveat.** Box R kept no per-position floor, so the medians may sit at the reference's arithmetic noise.
- **The wiring proof.** `tests/test_sc1g_a6.py` drives the real box script with stubbed engines, and the reducer's
  self-test covers A6 (45 cases). A3's box J is kept as `box_j_a3`.
