### HO2 registered: the planner's first held-out training read

- Two rented boxes, one RTX 5090 and one RTX 4090, measure rows for the held-out side of Loggetta's frozen planner split. Each
  runs `loggetta train` on the planner's own plan (no receipts on file, no fixed fields) for two MoE families no planner constant
  has seen, LFM2-8B-A1B and ERNIE-4.5-21B-A3B, at 1,024 and 2,048 tokens on Alpaca's pinned text column.
- Each row reads COVERED, UNDER (driver peak above the plan) or SHORT (allocated peak above the estimate). Every valid row goes to
  the split's next version; no band is computed here. Prediction: all 8 COVERED. Worst case $5.80.
- `bench/ho2/`: the registration, the controller (P224's by counted edits), the runner, the reducer (17 self-test cases) and the
  staged pins; `tests/test_ho2.py`.
