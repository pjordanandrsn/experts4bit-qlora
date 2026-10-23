### HO3 registered: a second box for HO2's cohorts, and the first host-offload held-out rows

- Part A: one more RTX 5090 and one more RTX 4090 run HO2's four rows on machines HO2 did not use (the driver refuses
  HO2's two), on HO2's exact stack, so each LFM2 and ERNIE cohort rests on two boxes.
- Part B: two RTX A4000 boxes, on two different machines, run the same rows. On a 16 GB card the planner keeps ERNIE's
  experts on the host: the first held-out rows with host offload. Each family is its own cohort result.
- HO2's rule per row, plus HOST_UNDER for a host-offload row whose host peak exceeds its host estimate. Each row's plan is
  compared with the registered one, and a difference is a finding beside the row. Every gate fails closed.
- Prediction: every row COVERED. Lane worst $26.16 at the started-box caps; expected about $2.50.
- `bench/ho3/`: the registration, the controller and runner (HO2's by counted edits), the reducer (25 self-test cases)
  and the staged pins; `tests/test_ho3.py`.
