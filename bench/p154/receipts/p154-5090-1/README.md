# `p154-5090-1`: the W16 NF4 expert byte floor on the replay's route (READ)

One RTX 5090 (cc 12.0, 170 SMs, driver 595.91.07, a **575 W** power limit read on the box before any install) on an AMD
EPYC 7K62 host (96 CPUs, 503 GiB RAM), 2026-10-19T20:16:36Z to 20:31:49Z. Cost $0.399, at $0.9037/h including 200 GB.
- **Code:** e4b `20910df4` (the registration's merge, #1843), grouped-nf4-gemm v0.47.1 `38b30888` (P152's, on purpose).
- **Stack:** torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0, bitsandbytes 0.50.2; Qwen/Qwen3-30B-A3B @ `ad44e777`.
- **The same card as P152's read:** the GPU UUID in `forensics.txt` is the one in `p152-5090-2`'s. The floor and the
  GEMM times it is compared with come from two rentals of one card, not two cards.

The power floor, install, tripwire and self-tests passed: `p154_route.py` 24 cases, `p154_reduce.py` 14, P136's box 5 and
SC1b's window 3. The premise read **4 passed**: K25's row-exact GPU tests on this card. The arena bake, the prompts and the
routing pass completed with rc 0.

**The route, from both witnesses, as registered:**
- (singleton, device) = (False, True) at 16 rows, with `DEVICE_GROUPING` set by the pass and K25's mode `auto`;
- **6,144** K25 calls in the window (2 a MoE layer a step), counted by the wrapper on the kernel module;
- the library's own route record across the window: exactly **{`nf4_k25|le256`: 3,072}**, and no other key;
- **the store:** NF4 on all 48 MoE layers, packed `uint8` plus `float32` absmax, 128 experts a layer on `cuda:0`. Bytes per
  expert were read off the loaded tensors: **1,769,472** (gate_up) and **884,736** (down). The arena was written after the
  run started;
- **the head:** bf16 (`E4B_INT8_HEAD_BW=0`, no int8 head module), P152's server;
- faults: none.

**The routing:** a mean of **58.07** unique experts per layer-step (42 to 81), 2,787.2 a step (2,521 to 3,097), against
82.42 under uniform routing; the hottest expert holds 10.2 % of a layer's hits. The box's own copy rate is 1,523 GB/s
(reported).

**Verdict: READ** (`verdict_p154.json`), against P152's recorded GEMM times:

| | byte floor | recorded | fraction of copy | headroom |
|---|---|---|---|---|
| gate_up | 3.2904 ms | 8.3951 ms | 0.3919 | 5.1047 ms |
| down | 1.6452 ms | 3.3530 ms | 0.4907 | 1.7078 ms |
| both | 4.9356 ms | 11.7480 ms | 0.4201 | **6.8124 ms** |

The launcher's receipt and ledger row are in the private receipt store, and the teardown is proven there. `SHA256SUMS`
covers every file here. Some files stay only in that store, byte-exact, and `STORE-ONLY.sha256` lists them by their path
within the run's receipt:
- three logs that carry carriage returns from the box's progress output;
- the run's instance file and the launcher's teardown proof, which name the provider's instance;
- `route_b16.json`, whose store record carries the box's container hostname.

`route_b16.public.json` is that file with the one field removed, and nothing else changed:

```
python3 -c "import json,sys; d=json.load(open('route_b16.json')); del d['store']['host']; sys.stdout.write(json.dumps(d))"
```

The registered reducer rebuilds `verdict_p154.json` byte for byte from `route_b16.public.json` and P152's committed
verdict, since the reducer never reads the field.

The staged scripts are not copied: each is byte-identical to the file `staged-p154.sha256` pins at `20910df4`.
