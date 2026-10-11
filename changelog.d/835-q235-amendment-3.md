### Q235 Amendment 3: the depth read, the contribution statistic and the plants on the box (bench/q235)

- **The depth test** (`bench/q235/depth-30b/`): Amendment 1's full-state bands do not see either planted error at layer 24 of
  Qwen3-30B-A3B's 48 layers (routing max 0.241, epilogue max 0.371). The per-layer contribution error u_i sees both at the planted
  layer: 3.07× and 24.6× the unplanted box. An attention sink at position 0 carries 97–99.8 % of the reference's squared norm from
  layer 3, which is why the state statistic dilutes.
- **Rule 7 for a deep box** is a STRICT band on u_i: at most 0.775 (2 × the 30B box's interior maximum) on layers 8–86 of 94. The
  full state, the router overlap and the loss gap are reported.
- **`q235_box.py --repeat --plants`** runs the box's own plants in the fresh-process repeat. The clean step-0 forward is the null,
  then each planted error follows in the same process. Each plant's earlier layers are checked against the null within 1e-3
  relative (bit-for-bit equality is reported), and each unplant is checked structurally.
  On the 30B model at full depth one such process reproduces the three separate runs bit for bit.
- **`q235_run.sh`** seals `--plants` into the repeat and raises its cap to 4,710 s. `q235_fit.py --plants-from` adds the two
  measured plant forwards to the repeat's fitted time. The controller's allowance covers about 8.2 GB of receipts. The worst-case
  wallclock is 8.53 h: $18.76 at $2.20 an hour.
- **`gate2/capacity-2648.json`**: e4b's estimate at the run's shape, 8 × 331 positions: 24.76 GiB device and 158.62 GiB pinned host
  for bf16 adapters.
