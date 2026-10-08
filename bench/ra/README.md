# Release anchor

RA compares shipped e4b/gnf4 pairs in ABBA order on one RTX 5090.
[Registration](PREREG-ra.md) · [work item #1363](https://github.com/pjordanandrsn/experts4bit-qlora/issues/1363).

```bash
python bench/ra/ra_reduce.py --self-test
python bench/ra/ra_reduce.py --dir bench/ra/receipts/RUN --out bench/ra/receipts/RUN/verdict.json
```

`source-pins.json` records the released baseline commits, model revisions and
the reused instrument bytes read at registration. Before launch, the executor
adds wheel provenance, a complete common dependency lock and the launch
manifest; a version string alone is insufficient.

Input contract (`schema: 1`):

| file | contents |
|---|---|
| `run.json` | `kind` PROOF/READING, ABBA `order`, `old`/`new` package version/commit/wheel hashes, common `identity`, short/long/reps, quality continuation, training steps/layers/count and matched init digest |
| `arm_TAG.json` | `tag`, `status`, common `identity`, package `release`, isolated `venv`/`imports`, empty `feature_env`, matched init digest, own arena digest/resolved defaults, embedded records below |
| `training`, `training_profile` | TC1 native fields plus feature-resolution/call evidence; exactly 20 steps, the latter profiles steps 11–20 |
| `decode` | P109 workload walls/tokens/digests plus resolved graph/bucket/default observations |
| `capacity` | SC2 native requests/plan plus burst and end-health counters; the reducer recalculates goodput |
| `quality_1`, `quality_12` | P115 per-window R scores/engagement; the executor adds full `argmax_ids`, without changing scores |
| `features` | per observed feature: resolved `mode`, `patched`, measured `calls`, `fallback_calls`; enabled paths must be called |
| `SHA256SUMS` | every input/raw/log file; `verdict.json` is the reducer's output |

Exit 0 means CLEAR, exit 2 means another verdict. PROOF is always marked with
`release_clearance: false`. VOID carries its reason. A noisier metric cannot
hide another metric's regression. Output never serializes NaN/Infinity.

**Current scope:** the registration and reducer contract. Synthetic fixtures
exercise validation and the verdicts on CPU. The executor that obtains these
records, proves normalization against native files, and verifies each release's
defaults has not been implemented or run. Envelopes and caller-provided feature
observations are not independent proof of GPU engagement. No GPU measurement,
cost feasibility or release clearance is claimed here.
