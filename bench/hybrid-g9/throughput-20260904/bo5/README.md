# Build-out validation lane bo5 — 2026-09-04, one RTX 5090: the second text and the optimisation pass

**Box:** Vast.ai instance 49841214, RTX 5090 (sm_120, driver 580.119.02) on an AMD EPYC 9755 host (see
[`forensics.txt`](forensics.txt)); transformers 5.16.1, bitsandbytes 0.50.1 (the lane script's pins).
**When:** 2026-09-04 08:5x–12:33Z. Per [`logs/outer.log`](logs/outer.log): install 09:17:25Z, first arm
09:19Z, phase bo5 `TP_DONE` 11:11:35Z, bo5b `TP2_DONE` 11:57:05Z, bo5c `TP3_DONE` 12:32:06Z.
**Cuts:**

- **bo5** ([`logs/bo5_run.sh`](logs/bo5_run.sh)) = e4b integration-6 @`0535930324e5d11c35966ae0569f3c519a684355`
  = main 0.34.0 + #372 (native MXFP4 store) + #384 (C4-calibrated int4 experts) + #385 (`swiglu_rows` /
  `combine_rows` decode glue), with grouped-nf4-gemm @`587eb7aaf5618a045a92cf30d6a66bfb77507127` = 0.29.0 +
  `combine_rows`.
- **bo5b / bo5c** ([`logs/bo5b.sh`](logs/bo5b.sh), [`logs/bo5c.sh`](logs/bo5c.sh)) = e4b integration-7
  @`d090940ee10518872e4f8fd05dabe100f8dc1fa8` = main + #384 + #385 + #387 (fused q/k/v on norm-less attention +
  the fused rope-only fold), same grouped-nf4-gemm. NF4 numerics do not depend on the cut; the pairing matters
  for the speed pairs and for which NF4 wikitext control Mixtral's arms are read against (bo5c re-baked Mixtral
  and re-scored NF4 on the cut it ran).

**Protocol:** bo3's ([`../bo3/README.md`](../bo3/README.md)) — per family an NF4 bake of the released checkpoint,
then arms under `bench/hybrid-g9/step_decomp.py`: K8 teacher-forced NLL (2048 steps, sha-matched windows,
`--b1d-loop eager`), B=1 decode (512-token prompt, 128 generated, graph loop, timed window, fp8 paged KV,
placement all-vram; `--no-fuse-qkv` except on the `_fq*` arms, which are the #387 A/B) and B=16 aggregate
(graph loop, 70 steps) — **plus the second text**: `--ppl-source c4val1` (C4 validation) beside wikitext, so the
registered gate can be applied on every text an arm was scored on. The packs were calibrated on 8 batches of
C4 *train*; C4 validation is a different split of the same corpus, the weaker reading of "outside the
calibration domain" — every second-text verdict here is a FAIL, which a stricter second text would not soften.
Four families (Gemma-4 has no instrument and was not re-run); one arm per fusion; a refusal is a row.

## The table

[`RESULTS.md`](RESULTS.md) is the verbatim output of [`buildout_reduce.py`](buildout_reduce.py)
(`python buildout_reduce.py . --ref ../bo3`) followed by the reading of it. The reducer reads the JSON receipts
(`<family>_<kind>_<arm>.json`, one per arm; its result line is in `summary.txt`) and applies the registered rule
(`experts4bit_qlora.k8_gate`) **in its own units, perplexity, per text**: an uncalibrated arm passes when
|Δppl| ≤ 0.05 on every text; a calibrated pack passes when Δppl ≤ +0.05 on every text, and an improvement is
claimable only when it holds with the same sign on ≥ 2 texts. Nats are printed beside every verdict, read
against the family's arithmetic-order floor (Granite 0.0033, gpt-oss 0.0176, Qwen3 0.0095, Mixtral unmeasured),
and never change it. `--ref ../bo3` supplies only the NF4 *wikitext* K8 baselines this lane did not re-score
(Qwen3; Mixtral's integration-6 arms — the P30 NF4 row), sha-checked against the arm's window; speed ratios
never cross lanes.

## Layout

- `<family>_ppl_<arm>.json` — K8 receipt (`mean_nll`, `ppl`, `ppl_source`, `text_sha`); `_c4val` on the arm name
  = the second text. `<family>_b1_<arm>.json` — timed graph window (`step_ms_clean`);
  `<family>_b16_<arm>.json` — B=16 (`aggregate_tok_s`). 56 receipts + [`calib.json`](calib.json) (the
  calibration manifest, byte-identical to bo3's).
- `census_<family>_<arm>_cen.txt` — kernel censuses (`--replay-profile-out`, 8 replayed steps) for the
  Qwen3 glue A/B, the Granite #387 pair and the gpt-oss store route.
- [`logs/`](logs/) — the three lane scripts with their pip pins, and [`outer.log`](logs/outer.log) (the lane
  driver's console: timestamps, the install and tripwire lines, every arm's result line, the `SUMMARY` blocks).
  The 56 per-arm `run_*.log` files and the seven `bake*.log` arena bakes are not shipped (the repository ignores
  `*.log`; bo3 shipped its lane scripts only, the same way) — [`summary.txt`](summary.txt) is each run log's
  final result line, and the JSON receipt beside this file is the per-arm record.
- [`models.txt`](models.txt), [`forensics.txt`](forensics.txt) (from `outer.log`'s `nvidia-smi` / `lscpu` lines).
  The lane's `TP_DONE` / `TP2_DONE` / `TP3_DONE` markers are empty files and are not shipped.

## Reproduce

Rent the class in `forensics.txt`, stage `bench/hybrid-g9/step_decomp.py`, the lane's `k8_bake.py` and its
`hook/usercustomize.py` (the K8 int4-arm hook; the scripts assert on it and neither this bundle nor bo3 ships
the two helpers), put `calib.json` beside them, and run `logs/bo5_run.sh`, then `logs/bo5b.sh`, then
`logs/bo5c.sh` (each waits on the previous phase's marker). Every arm is one `step_decomp.py` invocation; the
`k8()` / `arm()` helpers in the scripts carry the flags, and `E4B_*` environment variables select the fusions
(`fenv` maps `0 / r1 / r12 / epi / r1epi / all` to `E4B_FUSE_T1_GLUE`, `E4B_FUSE_T1_GLUE_R2`,
`E4B_FUSE_ROUTER_EPI`; `E4B_SERVE_EXP_INT4`, `E4B_SERVE_ATTN_INT4_CALIB`, `E4B_SERVE_EXP_INT4_CALIB`,
`E4B_INT4_KEEP_NF4`, `E4B_FUSE_SWIGLU` / `E4B_FUSE_COMBINE` per arm as listed in `outer.log`). Then
`python buildout_reduce.py . --ref ../bo3`.

## What is NOT in the table

Nothing was probed on a short window this lane; every K8 row is the 2048-step window. The census arms
(`*_cen`) are full timed runs and appear as rows; their profiler tables are the `census_*.txt` files, whose
Self-CUDA totals and per-step launch counts the reducer prints under each family. bo3's Gemma-4 rows and its
gpt-oss K8 probes are not repeated here.
