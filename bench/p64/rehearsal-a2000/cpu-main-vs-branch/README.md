# Flag off against `main`: `_fused_over_stack` bit for bit (CPU; NOT a GPU reading)

`old_vs_new.py` loads `main`'s `experts4bit_qlora/engines/hot_residency.py` (at `94842a2`, saved beside it as
`/w/old_hot_residency.py` in the container) next to this branch's. With `E4B_INT4_DECODE_A16` unset, it runs both
through `_fused_over_stack` with `tests/test_int4_decode_a16.py`'s pure-torch int4 references and a stand-in NF4
GEMM.

- **Cases:** 25, five routes × five random draws: int4 T = 1 (singleton), int4 T > 1 host-grouped prefill, the
  int4 batched-decode GEMV (device grouping, ≤ 256 rows), NF4 grouped, NF4 singleton.
- **Checked per case:** that the outputs are `torch.equal` and the kernel-call sequences identical.
- **Result:** `old_vs_new.json`, `all_equal_and_same_calls: true` on all 25.
- **Where it ran:** the CPU-only `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` container, torch 2.8.0, with
  grouped-nf4-gemm v0.33.0's `int4_pack_ref` and `nf4_grouped` importable.
