### SC1g A6 continuation registered (#846): box J reads P3, the bf16-activation arms the first A6 box's deadline dropped (bench and tests only)

- **What it adds.** No new hypothesis. It runs the arms P3 needs: (b) and (c) (`E4B_MXFP4_GEMV=0`) on conv1–conv4, plus
  the (b') repeat on conv1, placed third so the noise gate cannot leave P3 UNREAD.
- **The rules are A6's, unchanged:** the same predictions, bars, gates and reducer.
- **P1 and P2** stand from `sc1g-diag-a6-1`.
- **The box.** 9 arms, about 43 min inside the 1.0 h guard, about $0.6.
- **Code.** `box_j` now runs `i_arms_a6c`, and the first box's `i_arms_a6` is kept. The wiring test drives the
  continuation through the real box script.
