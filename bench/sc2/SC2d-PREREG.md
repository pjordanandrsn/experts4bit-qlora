# SC2d: does bulk KV bookkeeping engage, and change nothing, on a hybrid model and on gpt-oss? The `E4B_PAGED_BULK_KV` default's engagement reads, OFF against ON, one RTX 5090 (lane SC2d of #846; box K; correctness only)

Registered before any box. The default this lane gates was licensed by SC2c
([`../h2h-2026-10-02/sc2c/README.md`](../h2h-2026-10-02/sc2c/README.md), #1166: DEFAULT_LICENSED, ceiling 2 → 4 req/s,
output identical) on Qwen3-30B-A3B. SC2c registered that the flip PR also carries one `/health` `kv_bookkeeping`
engagement read on a hybrid model and one on gpt-oss (#1131's maintainer review, point 3). This lane is those two
reads.

## Why a box, and why these two models

**The bulk path's two structural risks are a pool-layer subset and attention sinks.**
- **The hybrid (Qwen3.6-35B-A3B).** Its Gated DeltaNet layers keep no K/V, so the pool holds its attention layers
  only: 10 of 40, one in four. The bulk flush and claim run over `pool_layers`, a strict subset of the model's layers.
- **gpt-oss-20b.** Every layer carries attention sinks, and sliding-window layers alternate with full ones.

The bitwise tests cover layer subsets and mixed geometry on tiny models. Neither real checkpoint has been served with
the knob on.

**Neither fits the 12 GB NAS A2000.**
- gpt-oss read 26,316 MiB at ready in SC2g.
- Qwen3.6 is a 72 GB bf16 checkpoint with a 17 GB NF4 snapshot.

The household NAS has 33 GB of host RAM free, so the A2000's offload route does not fit either. Hence one rented
5090, read for correctness only. **No time from this lane is evidence of speed.**

**What the read must show.** Engagement is what *ran*, not what was called. Since #1174, `/health` counts
`flush_bulk_fallback`: bulk flushes that `Fp8PagedKV.append_prompt` wrote per layer after all. That happens for:
- a slot already holding tokens;
- prompts of different lengths across layers;
- a demoted arena.

Before #1174, `flush_bulk` counted the call. By code reading none of the three can happen here:
- prompts are 512 tokens, one chunk;
- sliding layers keep full K/V and mask at attention;
- the arena never demotes.

The read checks this rather than assuming it.

## The box (K) and the stack it pins

- **e4b:** the launch commit. Main must contain #1174 (the fallback counter). Box K's tripwire refuses (rc 9) before
  any fetch if the installed e4b lacks it, because every server would otherwise read `<missing>` and report a false
  NOT_ENGAGED.
- **The rest of the stack:** grouped-nf4-gemm v0.41.0 (`dc8f94ab`, as box H); torch 2.8.0+cu128 (A1's pin, on every
  python3 box); transformers 5.16.1 and bitsandbytes 0.50.1 (the harness's pins).
  - transformers 5.16.1 ships `qwen3_5_moe` (`Qwen3_5MoeGatedDeltaNet`), `cache_utils.LinearAttentionLayer` and
    `gpt_oss`.
  - e4b's `test_linear_state.py` and `test_gptoss_forward.py` pass on it (14 passed, CPU; checked 2026-10-05).
- **Routes and settings:** main's defaults (neither SC1 route pin is exported, as on boxes F–J). `env -u` clears
  `E4B_PAGED_PREFILL_GRAPH` and `E4B_PAGED_GRAPHS`, so the prefill graph and decode graphs are at `auto`.
- **gpt-oss-20b** at SC2g's pin (`openai/gpt-oss-20b@6cee5e81`), on SC2g's registered e4b path: `SC2G_E4B_ENV` (native
  MXFP4 decode, NF4 prefill) plus the folds, its NF4 arena baked on the box (`bake_gptoss`).
- **Qwen3.6-35B-A3B** at P98's pin (`Qwen/Qwen3.6-35B-A3B@995ad96e`), with the server's defaults on P98's NF4 arena
  (`p98_bake.py`, staged and pinned):
  - placement `all-vram`;
  - decode graphs `auto`;
  - the prefill graph `auto`, which refuses on a hybrid by design;
  - the int4 levers off.

  P98, P105 and P106 served this checkpoint through `serve_paged.build_engine`. This is its first run through the HTTP
  server.
- **Prompts:** the SC2 prompt pool (512-token prompts) built in each model's own tokenizer. The driver records each
  pool's sha256 in every run file.

**Per model, two servers, OFF (`E4B_PAGED_BULK_KV=0`) then ON (`=1`).** The knob is the only difference.
- **OFF:** 4 warm requests, the serial plan of 16 (seed 1), the same plan again (the determinism control). Then
  `/health`.
- **ON:** 4 warm requests, the serial plan of 16 (seed 1). Then `/health`.
- **Order:** gpt-oss runs end to end before Qwen3.6 is fetched. If Qwen3.6's fetch, bake, pool or arch record fails,
  that is recorded, gpt-oss's read stands, and the lane's rc is non-zero.

**The checkpoint's own facts** (`arch_<model>.json`), read on the box from the fetched snapshot's `config.json` and
weight index:
- layer types and which layers are attention;
- how many layers carry a `self_attn.sinks` tensor.

## The rule (`bench/sc2/sc2d_reduce.py`, self-tested on 12 cases)

Per model, the gates in order. The first that fails names the model's outcome.

| gate | passes when | otherwise |
|---|---|---|
| SERVED | OFF served every request VALID (warm, serial, repeat) | **UNREAD**: the knob is not implicated (a harness or model failure); the lane relaunches after a fix |
| | ON served every request VALID | **NOT_SERVED_ON**: a finding |
| ARCH | Qwen3.6: its attention layers are a strict subset of its layers, and the rest are linear. gpt-oss: a sinks tensor on every layer | VOID |
| PROMPTS | both arms ran the same pool | VOID |
| ENGAGED | `kv_bookkeeping` at each server's end, n = the requests it admitted. **OFF:** `requested`/`bulk` false, `flush_layers` n, `flush_bulk` = `flush_bulk_fallback` = `ready_bulk` = `ready_at_flush` = 0. **ON:** `requested`/`bulk` true, `flush_bulk` n, **`flush_bulk_fallback` 0**, `flush_layers` = `ready_layers` = 0 | **NOT_ENGAGED** |
| DETERMINISM | OFF serial against its repeat is IDENTICAL (`sc2_identity`) | VOID (identity unreadable) |
| IDENTITY | OFF serial against ON serial is IDENTICAL, every request's streamed text byte-equal | **DIFFERENT** |

A model that passes every gate is **ENGAGED_IDENTICAL**.

**The flip, as registered.**
- **FLIP_LICENSED** iff both models are ENGAGED_IDENTICAL. The flip PR then makes `E4B_PAGED_BULK_KV` default to `1`,
  keeping `0` as the escape, and cites this read with SC2c's.
- **FLIP_HELD** otherwise, naming each model's outcome. The flip waits. A family-scoped default (bulk off where a read
  failed) would need its own registration.
- **UNREAD** on either model is a harness result, not a reading: the lane relaunches after a fix.

**Prediction:** both models ENGAGED_IDENTICAL. The basis:
- the bulk path is bitwise the per-layer path in every pool byte, table and length, on layer subsets, mixed geometry
  and partial tail blocks (`tests/test_bulk_kv.py`);
- on SC2c's four Qwen3 servers it was byte-identical in output;
- none of `append_prompt`'s fallback conditions can arise for single-chunk prompts.

What could still fail is the server path itself on a real checkpoint: Qwen3.6 has never run through the HTTP server.
That is what the read is for.

**Recorded, no gate:**
- per server: `prefill_graph` (status, `bulk_flush_mib`, `free_after_mib`), decode-graph status, `ready_bulk` and
  `ready_at_flush`, VRAM at ready;
- per run: TTFT and TPOT, as the driver records them. **These are not read as speed.**

## Proof and budget

**Proof** (`sc2d-prove-*`, guard 1.0 h). The harness's common Granite proof (fetch, bake, the b1 and b16 smokes), then
box K's whole flow on Granite:
- the tripwire;
- the driver, identity and SC2d reducer self-tests;
- `p98_bake.py` parses;
- the prompt pool and arch record;
- both servers with every request;
- `sc2d_reduce.py --proof granite`, which is every gate but ARCH, and must read ENGAGED_IDENTICAL.

The proof does not fetch either target model.

**Reading** (`sc2d-5090-*`, guard 2.5 h, `--download-gb 120`):
- gpt-oss (13 GB) and Qwen3.6 (72 GB) fetches;
- two bakes (Qwen3.6's took 1–2.5 min in P98, P105 and P106);
- four servers, 112 requests in all (56 per model).

Expected ~1–1.5 h. Each run sits under #846's standing no-ask tier for a single run under $15.

## Out of scope

- Speed, and any comparison with SC2c's numbers.
- Other families: tests cover equivalence, and the per-server engagement counters report every run.
- Prompts over 512 tokens.
- The flip itself, which is a separate PR on FLIP_LICENSED.
