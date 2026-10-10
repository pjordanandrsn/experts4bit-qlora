### Design note: a fused path for norm calls above 64 rows in e4b's prefill (bench and tests only)

- `bench/prefill-glue/DESIGN.md` is zero rental and for review. It maps the eager prefill chain onto the fused kernels
  that already exist, and proposes `E4B_FUSE_PREFILL_GLUE` (default 0) in three parts:
  - norms and rotary;
  - the bit-identical MoE dispatch;
  - the router epilogue.
- It also states the arithmetic choice behind the check: HF's two roundings against the decode kernels' one.
- It lists the CPU and A2000 checks it would register.
- `glue_map.py` assigns every kernel of P119's 512-token prefill profile to its source operation.
  `tests/test_prefill_glue_map.py` keeps the note equal to the script's output and checks that the map closes.
