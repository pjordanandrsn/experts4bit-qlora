### SC5 registered (#1478 item 4): same-box serving head-to-head on one RTX 5090 — e4b int4 against vLLM 0.31.0 and SGLang 0.5.21 on Qwen3-30B-A3B (bench and tests only)

- **The question.** Where `serve_paged` leads, matches or trails vLLM and SGLang, at 1, 16 and 64 concurrent requests,
  in TTFT, TPOT and throughput. Two memory settings: each framework's default, and 65,536 matched KV tokens. Quality is
  measured against one common bf16 reference.
- **The e4b arm is "e4b int4 (documented serving configuration)":** SC1's `SPEEDENV` plus `E4B_PAGED_FUSE_QKV=1`. e4b's
  out-of-box serving is NF4 and is not measured.
- **Provenance:** the e4b and grouped-nf4-gemm release wheels by sha256, and the competitors' whole closures
  hash-locked (`bench/sc5/make_locks.sh`, `bench/sc5/locks/`). SC1's vLLM and SGLang installers gain an opt-in lock
  mode; every other box installs as before.
- **The box:** SC1 box M (`bench/sc5/sc5_box_m.sh`):
  - ABBA cold-started blocks over 2 draws × 2 memory settings, closed-loop cells N = 48/160/320;
  - capacity readouts, a 1 Hz memory sampler, and the quality passes;
  - a reference phase and a proof. It is wired at every letter-enumerating site.
- **The rule** (`sc5_reduce.py`): LEADS or TRAILS only when both ABBA pairs clear the bound, 5 % for TPOT and throughput
  and 10 % for TTFT; every losing cell is listed.
- **Budget.** Proof $3.12, reference $2.05, reading $5.67 (ceilings); about $6–8 expected.
