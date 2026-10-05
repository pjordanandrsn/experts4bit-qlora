### SV2 registered (#1207): the int4 serving levers beside the estimate on Qwen3-30B-A3B with decode graphs (bench and prereg only)

- `bench/sv2/`:
  - `SV2-PREREG.md`;
  - `sv2_run.sh`: box side, `tc1_drive.sh`'s `TC1_RUNNER` contract. The arena and the checkpoint stay outside the
    fetched tree.
  - `sv2_measure.py`: one arm, `sv2-arm/1`.
- One RTX 5090, one arena, four arms: `q_nf4` (the anchor), `q_exp`, `q_both`, `q_exp_prefill`.
- The readings, registered with expectations:
  - V1: the estimate against `q_exp`'s peak, ±5%;
  - V2: the int4 stores' load delta, ±32 MiB of +54;
  - V3: int4 attention, ±64 MiB of +571;
  - V5: the repack's host peak at or under its 6,912 MiB price;
  - V6: the heap handed back, within 0.25 GB of NF4.
- One reading is measured only: V4, the prefill graph's pool at int4.
- Spend is capped at $5 by #1207.
