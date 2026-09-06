# tp2 — training-side parity vs Unsloth, six MoE families, one RTX 5090 (2026-09-05/06)

Receipts of lane **tp2 / pre-registration P40** (`P40-PREREG.md`, verbatim): e4b (0.35.1 wheel + grouped-nf4-gemm 0.30.2) vs Unsloth 2026.9.2, per family, one box (Vast instance 50005568, RTX 5090 32 GB, Ryzen 7 5700X3D, box class pcie-full/launch-fast), one fixture (the sha-pinned clinical set tokenised per family, seq 512, r 8, α 16, lr 1e-4, accum 1, N = 60 steps, held-out every 20).

- `RESULTS-tp2.md` — the reducer's report (statuses, validity, positions, quality readings, P1–P7 scored mechanically). `RESULTS.txt` is its plain-text twin.
- `<family>_<framework>_<arm>.json` — one receipt per attempt (status vocabulary OK / REFUSED / OOM / INSTALL_FAILED / LOAD_FAULT / HARNESS_ERROR / ALARM / NOT_RUN; every attempt is a row).
- `summary.txt`, `outer.log` — the lane's own log; `forensics.txt`, `anchor.json` — box identity + the anchor gate; `tokens_<family>.json`, `ds_manifest.json` — the fixture as run; `tp2_arm.py`, `n9_datasets.py` — the harness that produced the receipts.

Receipts only. The measured claims, the `training_support` updates and the prose that quote these numbers land in a separate pull request that cites this directory (rule: receipts and code/claims in separate PRs so review is possible). Adapters and the model caches were not kept.

Headline, read from `RESULTS-tp2.md` (Unsloth/e4b s/step; VALID pairs only): Qwen3-30B-A3B **1.457** (e4b faster; agrees with P38's 1.413 within ±10 %; quality COMPARABLE); Mixtral-8x7B **0.361** (Unsloth resident 29.2 GB vs e4b expert-offload 3.2 GB — the registered design; P4's OOM prediction FALSIFIED; quality COMPARABLE). No position for Granite (Unsloth VOID: 2.6 M trainable vs e4b's 49.8 M — no expert LoRA), OLMoE (Unsloth died at engage), gpt-oss (both refuse, P5 HELD) or Gemma-4 (e4b attention-4bit arms failed their projection-count check, 100 of 120 — see experts4bit-qlora#412). e4b fused-vs-reference parity PASS on all five families that ran.
