# P55x — results

Pre-registration: [`P55X-PREREG.md`](P55X-PREREG.md). Every number below is read from a committed
receipt; nothing is recomputed from memory.

## The pack

- `pack_fingerprint` **`sha256:0c9955a9f06d83269050d1fc64571a2a0e78abd48b288fd422e4eef72ccbcd42`**
- Qwen/Qwen3-30B-A3B @ `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, layout `int4_b32.gate_first.v1`, 194 payloads
- gptq / rtn expert matrices: **11512 / 776**  — the bo6c/bo7 split, exactly (a diagnostic under #405, never the identity)
- `min_rows` 32, damping 0.01, solve device cuda, calibration token stream `sha256:e5d5eba524e6a…`

## Q1 — the registered two-text K8 gate, on the artifact

### `lic` — calibrated int4 attention (bo6c's configuration) — **primary**

| text | NF4 ppl | pack ppl | delta | budget +0.05 |
|---|---|---|---|---|
| wikitext | 6.41984 | 6.36709 | -0.05275 | PASS |
| c4val1 | 16.49703 | 16.43081 | -0.06622 | PASS |

**lic verdict: PASS**

### `licrtn` — RTN int4 attention (fully pinnable) — secondary

| text | NF4 ppl | pack ppl | delta | budget +0.05 |
|---|---|---|---|---|
| wikitext | 6.41984 | 6.35725 | -0.06259 | PASS |
| c4val1 | 16.49703 | 16.62939 | +0.13237 | **FAIL** |

**licrtn verdict: FAIL**

  Failure is on the **budget**: c4val1 exceeds +0.05.

**dump → load round trip (wikitext):** live stores 6.36709 vs the artifact 6.36709, delta +0.00000. The dumped bytes are the built bytes, or this number would not be zero.

## Q2 — is the recipe byte-deterministic on one box?

| | build 1 | build 2 |
|---|---|---|
| `pack_fingerprint` | `sha256:0c9955a9f06d83269050d1fc64571a2a0e78abd48b288fd422e4eef72ccbcd42` | `sha256:0c9955a9f06d83269050d1fc64571a2a0e78abd48b288fd422e4eef72ccbcd42` |
| gptq / rtn counts (read from each build's manifest) | 11512 / 776 | 11512 / 776 |
| `method_map_hash` | `sha256:1f2ba78a20c83c6bf…` | `sha256:1f2ba78a20c83c6bf…` |
| row-count-vector hash | `sha256:4ebf85d6389ed523c…` | `sha256:4ebf85d6389ed523c…` |

**Verdict: DETERMINISTIC** — bytes identical: True, classification identical: True.

What this extends: `e4b.serve.buildout.bo6c.qwen3.calib-deterministic.5090.2026-09-05` compared
mean_nll and counts for the **16k** arm, before the fingerprint existed. This is the 64k recipe
compared at the byte level.

## Q4 / STOP-1 — does the NF4 reference travel?

| text | this box | bo6c's box | drift |
|---|---|---|---|
| wikitext | 6.41984 | 6.41984 | +0.00000 |
| c4val1 | 16.49703 | 16.49703 | +0.00000 |

## The box, and the path the bytes took

- K0: NVIDIA GeForce RTX 5090, 320 GB free on `/root` (floor 200)
- upload probe: **10.67 MB/s** over 256 MB (floor 3, rsync rc 0) — the direction the rental
  pre-flight does not measure, and the one this lane's product has to travel
- artifact fetch, **sustained**: 3.33 MB/s over 15559 MB in 4667 s (rsync rc 0), against the probe's 10.67 MB/s on the same box minutes earlier. The probe **overstated** by 3.2×. A prior lane's probe went the other way (14.76 probe vs 38.70 sustained), so a short probe has **no consistent sign** against sustained rate — it is a disaster detector, and a transfer budget must come from a sustained row like this one.
- `NVIDIA GeForce RTX 5090, 32607 MiB, 575.57.08, GPU-7a6634c6-2afe-863b-7f96-d00691343cdc`
- `power.limit,clocks.max.sm 600.00 W, 3105 MHz`
- `Model name:                              AMD Ryzen 9 7950X 16-Core Processor`

```
e4b 0.36.4 @1c73a8d6fbf2a3c5bda5d4b30a110c7f897b15de
gnf4 0.32.1
torch 2.8.0+cu128
triton 3.4.0
transformers 5.16.1
bitsandbytes 0.50.1
```

## Arms as the box recorded them

```
k8 nf4 src=wikitext kind=nf4 rc=0 in 109s K8_PPL steps=2048 nll=1.85939 ppl=6.41984 compute=fp8 sha=9ef10d760ad9 out=/root/p55x/qwen3_ppl_nf4_wikitext.json

k8 nf4 src=c4val1 kind=nf4 rc=0 in 102s K8_PPL steps=2048 nll=2.80318 ppl=16.49703 compute=fp8 sha=4bcb55179b96 out=/root/p55x/qwen3_ppl_nf4_c4val1.json

INT4EXP licbuild_wikitext INT4EXP calibrated experts: 1954 gptq / 94 rtn (min_rows=32) over 8 layers
INT4EXP licbuild_wikitext INT4EXP calibrated streaming: 48 layers in 5 passes of <= 10 layer(s)
ATTNINT4 licbuild_wikitext ATTNINT4 calibrated: 192 projections
k8 licbuild src=wikitext kind=calibattn rc=0 in 1832s K8_PPL steps=2048 nll=1.85114 ppl=6.36709 compute=fp8 sha=9ef10d760ad9 out=/root/p55x/qwen3_ppl_licbuild_wikitext.json

ARTIFACT1 sha256:0c9955a9f06d83269050d1fc64571a2a0e78abd48b288fd422e4eef72ccbcd42
ARTIFACT1_SIZE 16G	/root/p55x/artifact1
ATTNINT4 lic_wikitext ATTNINT4 calibrated: 192 projections
k8 lic src=wikitext kind=calibattn rc=0 in 313s K8_PPL steps=2048 nll=1.85114 ppl=6.36709 compute=fp8 sha=9ef10d760ad9 out=/root/p55x/qwen3_ppl_lic_wikitext.json

ATTNINT4 lic_c4val1 ATTNINT4 calibrated: 192 projections
k8 lic src=c4val1 kind=calibattn rc=0 in 306s K8_PPL steps=2048 nll=2.79916 ppl=16.43081 compute=fp8 sha=4bcb55179b96 out=/root/p55x/qwen3_ppl_lic_c4val1.json

ATTNINT4 licrtn_wikitext ATTNINT4 rtn: 192 projections
k8 licrtn src=wikitext kind=rtnattn rc=0 in 130s K8_PPL steps=2048 nll=1.84960 ppl=6.35725 compute=fp8 sha=9ef10d760ad9 out=/root/p55x/qwen3_ppl_licrtn_wikitext.json

ATTNINT4 licrtn_c4val1 ATTNINT4 rtn: 192 projections
k8 licrtn src=c4val1 kind=rtnattn rc=0 in 131s K8_PPL steps=2048 nll=2.81117 ppl=16.62939 compute=fp8 sha=4bcb55179b96 out=/root/p55x/qwen3_ppl_licrtn_c4val1.json

GATE lic=PASS licrtn=FAIL
INT4EXP licbuild2_wikitext INT4EXP calibrated experts: 1954 gptq / 94 rtn (min_rows=32) over 8 layers
INT4EXP licbuild2_wikitext INT4EXP calibrated streaming: 48 layers in 5 passes of <= 10 layer(s)
ATTNINT4 licbuild2_wikitext ATTNINT4 calibrated: 192 projections
k8 licbuild2 src=wikitext kind=calibattn rc=0 in 1832s K8_PPL steps=2048 nll=1.85114 ppl=6.36709 compute=fp8 sha=9ef10d760ad9 out=/root/p55x/qwen3_ppl_licbuild2_wikitext.json

ARTIFACT2 sha256:0c9955a9f06d83269050d1fc64571a2a0e78abd48b288fd422e4eef72ccbcd42 SAME_BYTES
DETERMINISM DETERMINISTIC counts ['gptq', 'rtn'] ['gptq', 'rtn']
ARTIFACT2_PAYLOADS_REMOVED (fingerprint recorded; bytes not retained -- artifact1 is the one kept)
ARMS_WITH_RECEIPTS 8
```

## Where the bytes are, and how you would check a copy

The pack is **16.3 GB over 194 payloads** and is
not in this repository. It is held on an operator archive host, offline, and is available on
request. What is here is the manifest, the hashed identity and assignment payloads, and this
record — which is enough to check any copy you are given, because a loader pinned to the
fingerprint refuses anything else and never rebuilds from the recipe.

- **local**: `experts4bit_qlora.engines.pack_manifest.verify_artifact (e4b 0.36.4 @1c73a8d6)` → `OK`
- **on the archive host**: 194 payloads re-hashed by an independent implementation of the canonical rule → `MATCH`

Two implementations, two machines, one artifact, the same 64 hex characters. Either alone would
not be the check: the library can agree with itself, and the inline rule is trusted only because
the library computed the same number on the same bytes.

**This is a private holding, said as one.** It is weaker than publishing the bytes. The #405
design note proposed a dedicated artifact repository pinned by commit, which would let a reader
fetch as well as check; that remains the better answer and is the owner's call, not a step
inside a measurement lane.

## What this says

**There is a licensed pack again.** `sha256:0c9955a9f06d83269050d1fc64571a2a0e78abd48b288fd422e4eef72ccbcd42` passed the registered two-text gate on the bytes a
loader installs, so the licence is a property of those bytes and not of the recipe that made them.
That is the thing [#405](https://github.com/pjordanandrsn/experts4bit-qlora/issues/405) has been
open on since 2026-09-06: the machinery to license bytes existed, and no pack had been built,
gated and **kept**.

**And the recipe reproduces across boxes after all.** This pack is byte-identical to the one P39
recorded on 2026-09-10 — the same 64-hex `sha256:0c9955a9f06d8326905…` — built on a different rented
5090, twelve days earlier, under **e4b 0.35.3** where this ran under **0.36.4**. P39 had already
reproduced it between its own box 1 and box 2 on one day and one cut; this extends it across a
release boundary. Together with the two same-box builds below that is four builds and at least
three boxes agreeing on every byte.

That **narrows** #405 rather than closing it. The issue's framing — *the streamed calibration
recipe does not reproduce its licence across hosts* — was drawn from P37, which read 11522/766
and failed c4val1 at +0.109. On the evidence now available P37 is the **outlier**, not the rule,
and the open question is no longer *does the recipe reproduce* but *what was different about that
host*. What has not changed is the remedy: a licence still travels only as bytes, because nothing
here predicts which box will be the next P37.

**The half that cannot be pinned is the half carrying the quality.** The same pinned expert
bytes with RTN attention instead of calibrated move c4val1 from -0.06622 to +0.13237 — a
swing of +0.19858 ppl, and a budget failure. So the calibrated attention component is
load-bearing, and the residual this lane registered is not a formality: `int4_attn_calib.py`
has no serialisation, so those 192 projections are re-derived on every load, on whatever box
does the loading. `pack_fingerprint` names the expert bytes; it does not name those.

**The dumped bytes are the built bytes**, to all sixteen digits (6.367086902514706), so the
artifact is the pack and not a copy of it.

