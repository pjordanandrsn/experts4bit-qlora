### SC2d registered (#846): the `E4B_PAGED_BULK_KV` default's engagement reads on Qwen3.6 (hybrid) and gpt-oss, box K (bench and tests only)

- **Why.** SC2c licensed the bulk-KV default on Qwen3-30B-A3B (#1166). As registered there, the flip first carries
  one `kv_bookkeeping` engagement read on a hybrid model (the pool holds its attention layers only) and one on gpt-oss
  (attention sinks). Neither model fits the 12 GB A2000, so the reads run on one 5090, for correctness only.
- **Box K** (`bench/sc2/sc2d_box_k.sh`):
  - **Models:** gpt-oss-20b on SC2g's e4b path; Qwen3.6-35B-A3B on the server's defaults over P98's NF4 arena
    (`p98_bake.py`, now staged and pinned).
  - **Per model:** an OFF server then an ON server, with 4 warm requests, 16 serial (plus a repeat on OFF), and each
    server's own `/health`.
  - **Records:** the checkpoint's layer types and sinks.
  - **Guard:** a tripwire refuses an e4b without #1174's `flush_bulk_fallback`.
- **The rule** (`sc2d_reduce.py`, 12 self-test cases), per model: SERVED, ARCH, PROMPTS, ENGAGED, DETERMINISM,
  IDENTITY. ENGAGED reads what ran: ON's `flush_bulk_fallback` must be 0. The flip is licensed iff both models are
  ENGAGED_IDENTICAL.
- **Harness:** `sc1_run.sh` / `sc1_drive.sh` / `make_pin.sh` gain box K, mirroring box H; the box-list tests are
  updated; `tests/test_sc2d_box.py` runs `k_arch` on hybrid- and sinks-shaped snapshots.
