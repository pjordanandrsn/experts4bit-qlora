# Async-reuse stability, RTX A2000 (2026-10-17): no reuse failure found

A correctness check, with no speed figures. Three of our paths reuse a device workspace across asynchronous launches. Each ran
under concurrent pinned host-to-device copy traffic (a side stream copying 256 MB at a time throughout, 8,300–12,408 copies per
run), with its workspace poisoned with NaN, and was checked against a path that does not reuse. The question is the one the
FlashInfer grouped-WGRAD race and the TensileLite StreamK partial-store bug raise: is a workspace reused while an earlier use
is still in flight, or do distinct partials land in one slot?

- Code: grouped-nf4-gemm `5c3d309` (its `kernel/` on `PYTHONPATH`) and experts4bit-qlora `3834e696`.
- Environment: torch 2.11.0+cu128, one RTX A2000 12 GB.
- `async_reuse_stress.py` is the script; `results.json` holds every count below, and the sha256 of the three raw outputs.

## Results

| part | path | denominator | result |
|---|---|---|---|
| A | grouped-nf4-gemm `int4_b32` with `fused_reduce=True` (lane K17): one `part`/`cnt`/`out` reused, `part` and `out` NaN-filled before every launch | 7 shapes (one with N % 128 ≠ 0) × R ∈ {1, 8, 16, 128} = 28 cases, all split-K > 1, × 300 launches = **8,400**; plus two streams racing on separate workspaces × 300 | **0 mismatches** against the two-launch path, bitwise; `cnt` all-zero after every launch; race: 0 |
| B | grouped-nf4-gemm fp8 split-KV paged attention, f32 path, in-kernel combine (per-stream arrival counters): the caching allocator NaN-poisoned before every launch | 4 shapes × `n_split` ∈ {2, 5, 8} = 12 cases × 300 = **3,600**; plus two streams × 300 | **0 mismatches** against the two-kernel combine, bitwise; race: 0 |
| C | experts4bit-qlora expert offload (pinned host homes, single-slot staging), a reduced Qwen3-30B-A3B (4 layers, 2 × 512 tokens): the allocator NaN-poisoned before every forward-backward | **30** offloaded passes, against 6 resident passes of the same model (same seeds, same adapters), twice: once with the default absmax, once with `E4B_ABSMAX_DQ=0` | **0 NaN** in 60 offloaded passes; one loss value per build; gradient spread of the same size as the resident model's own (table below) |

**Reading C.** The training kernels are not bitwise deterministic run to run: the resident model's gradients differ between its
own passes too. So C reads the size of the spread, not equality. All figures are max relative difference over the adapter tensors,
taken over the passes named:

| run | resident spread (5 passes vs its first) | offloaded spread (29 passes vs its first) | offloaded vs resident's first pass (30 passes) |
|---|---|---|---|
| default absmax (`out-C2.json`) | 0.0128 | 0.0131 | 0.0768, constant on all 30 (the absmax difference below) |
| `E4B_ABSMAX_DQ=0` (`out-C3-dq0.json`) | 0.0128 | 0.0137 | 0.0103–0.0147 |

The offloaded spread is the same size as the resident one (0.0131 and 0.0137 against 0.0128). It is not shown to be smaller: the
denominators differ (29 passes against 5), and a bf16 unit in the last place is 0.4–0.8 % of a value, so 0.013 is about two to three
bf16 steps on the largest elements. With no NaN anywhere and one loss value per build, the poisoning and the copy traffic left no
trace a reuse race would leave. This reads as no reuse failure found, not as a bitwise pass.

## The one actionable line: absmax storage differs between offloaded and resident layers

Offloaded layers keep the expert absmax in fp32. `compress_expert_absmax_` refuses an offloaded stack, because its absmax lives in
host RAM. A resident build double-quantizes it by default (`enable_fast_train`'s `absmax_dq` policy, TC1 amendment 56). So an
offloaded-versus-resident comparison of the same model is two different numeric paths:

| | resident vs offloaded loss | offloaded vs resident gradients (max rel.) |
|---|---|---|
| default (resident double-quantizes, offloaded fp32) | 12.358864 vs 12.358825 | **0.0768**, the same on all 30 passes |
| `E4B_ABSMAX_DQ=0` on both sides | **bitwise equal**, 12.358824729919434 | 0.0103–0.0147, the same size as each build's own spread |

**Any offloaded-versus-resident parity check must set `E4B_ABSMAX_DQ=0` on both sides, or assert that both rows' recorded
absmax storage is the same.**

- e4b's in-process test suite already does this. `tests/conftest.py`'s autouse fixture pins
  `fast.ABSMAX_DQ_DEFAULT = False` for every test, and `test_absmax_dq.py` turns it back on only where it tests the default.
- Its subprocess tests make no offloaded-versus-resident comparison.
- One reducer does: `bench/tc1/tc1_reduce.py`'s `resident_comparison`, which compares the frontier box's offload arm with TC1's
  resident `fused_attn4_m`. It did not check the receipts' `absmax_dq`. That is fixed separately, as TC3 amendment 6 (e4b
  #1684), with an audit of the record: its one scored pair was fp32 on both sides.
