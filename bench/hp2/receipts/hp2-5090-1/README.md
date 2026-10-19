# `hp2-5090-1`: the reading (G1 STOP)

One RTX 5090 (driver 590.48.01, 32,607 MiB) on an AMD EPYC 7B13 host (128 CPUs, 64 physical, a cgroup quota of about 61 CPUs,
252 GiB RAM, a 220 GB disk). A Vast RTX 5090 at $0.7411/h, 2026-10-19 06:45Z to 09:06Z. Cost $2.144
(guard 6 h, registered worst $6.09). Code: e4b `68801d3e` (the HP2 registration's merge, #1793), grouped-nf4-gemm `724ccc45`.
The e4b side ran torch 2.8.0+cu128 and transformers 5.18.0; the Unsloth side ran Unsloth 2026.9.14 with transformers 5.5.0
(`versions.txt`). Qwen/Qwen3-30B-A3B at `ad44e777`.

**Reading: `HP2_G1=STOP`** (`hp2/g1.json`, from `bench/hp2/hp2_reduce.py g1` on `hp2/eval_cal.json`):
- **The box-side shakedown passed.** Both 2-layer arms were SCORED, and both shake bases were OK (`hp1/eval_shake.json`).
- **The calibration rows.** The calibration tokens are held-out pool rows 97–396, and the pool's first 48 rows reproduce HP1's
  held-out set (`summary.txt`).
- **The calibration arms.** One e4b K00 step (`qwen3hp2_e4b_hp2cal_e4b_k00_s3407.json`) and one Unsloth step
  (`qwen3hp2_unsloth_hp2cal_unsloth_s3407.json`) each read CELL OK. Each base was exported and matched its training run by
  digest. The one evaluator then made its zero-adapter pass on each base over the 300 rows.
- **The gate.** δ0 = −0.002691, sd_d 0.041194, SE 0.002378:
  - |δ0| + 2 SE = 0.007448 ≤ 0.01, so the CLOSE tier passed;
  - m_need = 0.014895 > 0.005, so the STOP tier fired.
- **Per framework (reported, never gated):** b_e4b +0.000931 (sd 0.030206); b_unsloth +0.003622 (sd 0.028295). The per-row gaps'
  correlation between the frameworks is ρ 0.0094.
- **The driver ended the draw** on the reducer's exit code 4. No confirmation arm ran, and the box tore down.

**Re-derivation.** The maintainer re-derived the reading from the private receipt store with `hp2_reduce.py` at `68801d3e`
(equal to the box's copy): `g1` on the stored `hp2/eval_cal.json` reproduced `hp2/g1.json` byte-identically.

**What is here.** The files that decide the reading are byte-identical to the store: the G1 record and its evaluator output, the
shakedown evaluation, the four arm records, `summary.txt`, `versions.txt`, `forensics.txt`, `box.json`, `outer.log` and the box
logs. `hp1/bulk_manifest.json` lists, with their sha256s, the base exports and canonical adapters that stayed on the box.

**What is not here.**
- The launcher's receipt and ledger row are only in the receipt store.
- Four logs carry carriage returns from the box's progress output. They stay only in that store, byte-exact, and
  `STORE-ONLY.sha256` lists them.

`SHA256SUMS` covers every file here.
