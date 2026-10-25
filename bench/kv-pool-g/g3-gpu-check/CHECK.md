# G3 GPU check: does the shared server capture the prefill graph at 64 slots on Qwen3 int4?

A correctness check, not a reading. sc2j-5090-1 (SC2j strike 2) started the shared KV pool's server at 64 slots on
Qwen3-30B-A3B int4 on an RTX 5090, and its prefill graph refused for headroom: 724 MiB free after capture against the
745 MiB its gate wanted. e4b#2087 (merge `fc1f630a`) makes `size_shared_pool` hold back the graph's runtime headroom
with the same rule the gate applies (`serve_recipe.prefill_graph_headroom_bytes`). This check runs that server once on
the card and reads, from the server's own state, whether the graph was captured and replayed.

## The box

One RTX 5090, launched with `tools/pod-launch.sh` on a 1.0 h guard. The driver (`g3check_drive.sh`) refuses unless the
checkout is clean, HEAD is on `origin/main`, and HEAD carries `fc1f630a`. The box (`g3check_run.sh`):

1. runs the canonical guard-fit check before any install (rc 41, the cross-lane code, not a strike);
2. installs torch 2.8.0+cu128, e4b at HEAD and grouped-nf4-gemm at main's CI pin (`30adb1bf`), and runs a tripwire on
   the installed commits and #2087's rule;
3. records the card (`nvidia-smi` with the enforced power limit, and `bench/common/card_clock.py burn --floor 2700`).
   This is recorded, not a gate;
4. fetches Qwen3-30B-A3B at SC1's pin (`ad44e777`) and bakes the NF4 arena (`bench/p39/k8_bake.py`);
5. starts one server with the env of SC2j box Q's shared arm (`bench/sc2/sc2j_box_q.sh` `q_server_start`):
   - `E4B_PAGED_MAX_SEQS=auto`, `E4B_PAGED_KV_POOL=shared`, 4,096 tokens a slot;
   - SC1's `SPEEDENV` and `ROUTEENV`, plus `E4B_PAGED_FUSE_QKV=1`;
   - the prefill graph at its default, `auto`;
6. sends six prompts of about 1,000 tokens (`g3check.py drive`), each answered and each longer than the graph's 512
   tokens, so every first chunk is the graph's;
7. reads `/health` and gives the verdict (`g3check.py verdict`).

## PASS

PASS needs every one of these, and a missing field is a FAIL:

- the configuration, from the live server, never the launch env:
  - top-level `model == "Qwen/Qwen3-30B-A3B"`;
  - `engine.max_tokens_per_seq == 4096`;
  - int4 experts on every MoE layer (`levers.int4_expert_layers == levers.moe_layers > 0`, counted from the expert
    layers that carry an int4 store);
  - int4 attention on every projection, bound exactly from the served model's own layer count `L = levers.moe_layers`
    (every Qwen3-30B-A3B layer is an MoE layer):
    - `attn_int4_rtn_projections == 4L` (q, k, v and o swapped on every layer);
    - `fuse_qkv_n == L` (q, k and v fused into one on every layer);
    - `int4_attn_projections == 2L` (the `Int4Linear` modules counted on the model as served).

    sc2j-5090-1's record reads 192, 48 and 96 at `L = 48`;
- `engine.max_seqs == 64` and `engine.kv_pool == "shared"`;
- `engine.kv_pool_sizing.pool_rows < claim_all_rows`;
- the derivation's fitting 64-slot candidate carries `prefill_graph_headroom_bytes > 0`;
- `prefill_graph.requested == "auto"` and `prefill_graph.status == "on"`;
- `prefill_graph.T == 512` and `prefill_graph.replays > 0`.

"on" under "auto" is the capture evidence. `serve_paged` captures with
`require_headroom=(cfg.prefill_graph == "auto")`. `PagedModelRunner.enable_prefill_graph` assigns
`self._prefill_graph = pg` only after the gate
`free_after_bytes >= prefill_graph_headroom_bytes(pool_bytes, bulk_flush_bytes)`, which compares unrounded bytes.
`prefill_graph_stats` reports "on" only while that attribute is set. A refused graph reads FAIL. The rounded MiB
margin is recorded as information only.

`tests/test_g3_gpu_check.py` builds `/health` from the runtime's own functions on sc2j-5090-1's inputs: a captured
graph reads PASS. Each of these reads FAIL: the box's refusal, a graph with no replays, a forced graph, claim-all,
32 slots, a sizing with no hold-back, the wrong model, 2,048 tokens a slot, bf16 experts, bf16 attention, and 95 of 96 int4 attention projections.
sc2j-5090-1's own shared-arm `/health` (a fixture, pinned by sha256 to the receipt store) reads FAIL for its refusal
and its missing hold-back. With only a capture and the hold-back added, it reads PASS. That shows the verdict's field
names are the server's.

## Cost

The worst case is $1.72: a 1.0 h guard at $1.05/h, plus a 61 GB download at the $0.011/GB cap. That is under the
$15 per-run rule, and the figure is posted before the create. Receipts are fetched with
`rsync --no-links` before teardown (bench/common rule 8), and the receipt is committed to the store.
