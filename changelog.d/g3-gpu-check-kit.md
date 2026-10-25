### G3 GPU check kit: the shared server's prefill graph at 64 slots on Qwen3 int4, read from the server's own state

- `bench/kv-pool-g/g3-gpu-check/`, new: one RTX 5090 runs SC2j box Q's shared-arm server at a merged main sha that
  carries e4b#2087. The verdict binds on the capture evidence:
  - `prefill_graph.status == "on"` under `requested == "auto"`, which the runner sets only after the headroom gate;
  - `T == 512` and `replays > 0` after six long prompts;
  - 64 slots, and a shared pool smaller than claim-all, whose 64-slot candidate held back
    `prefill_graph_headroom_bytes > 0`.

  A refused graph (sc2j-5090-1's record) reads FAIL, and any missing field reads FAIL.
- The card is recorded with `card_clock.py`, but not gated on. The guard-fit check comes before any install. The fetch
  is `rsync --no-links`. `tests/test_g3_gpu_check.py` builds `/health` from the runtime's own functions.
