### DQ6 registered: DQ4's capacity read on a 24 GB RTX 4090 (bench and prereg only; #1083)

- **The read.** `bench/dq6/` measures the longest QLoRA sequence that trains on a 24 GB RTX 4090, resident against
  streamed, on DQ3's Qwen3-32B-architecture subject. Predicted from DQ4's receipts: resident about 1.5k tokens, streamed
  about 9k, a ratio of about 6 (DQ4's 32 GB 5090 read 2.00).
- **Reused unchanged:** DQ4's harness, configurations and rule.
- **The registered differences:**
  - a card gate: a 24 GB RTX 4090, otherwise rc 19;
  - DQ3's VRAM probe with a 21.5 GiB floor;
  - a 512-token ladder;
  - DQ4's reducer re-registered to the 4090, with a 24 GB total-memory check.
- **The runner's drift from `dq4_run.sh`** is pinned by `tests/test_dq6_lane.py`, which also checks that DQ4's committed
  read still re-derives byte for byte.
