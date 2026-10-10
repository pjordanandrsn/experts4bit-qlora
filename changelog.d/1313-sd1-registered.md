### SD1 registered (#1313): speculative decoding, Phase 0 -- how many tokens a verify step yields on the shipped default

- **What it measures.** One RTX 5090 run measures greedy acceptance on `Qwen/Qwen3-30B-A3B` at e4b's shipped default
  (e4b `7f044dd9` + grouped-nf4-gemm `d769d502`), by draft length k, for two draft sources:
  - the licensed EAGLE-3 head `RedHatAI/Qwen3-30B-A3B-speculator.eagle3` (apache-2.0), pinned at `6afc5aa2` by file hash;
  - prompt-lookup (n-gram) drafting.
- **The workloads:** raw wikitext completion, and chat with reasoning on and off.
- **The price.** Each route's B = 1 speedup comes from e4b's own verify-step model: P123's census, 0.151 ms per distinct
  expert a verify step reads.
- **The decision**, registered before any data:
  - an EAGLE-3 serving lane if S ≥ 1.15 on two of three workloads;
  - otherwise a prompt-lookup lane if S ≥ 1.05 on raw text;
  - otherwise stop.
- **The n-gram floor is already computed** from P127's receipts: 1.08× at k = 1 under independent routing.
- **The draft head is reimplemented in plain PyTorch** (`bench/sd1/sd1_eagle3.py`) and tested against a step-by-step
  reference. A stubbed dry run (`tests/test_sd1_dryrun.py`) takes the whole box path to success in CI.
