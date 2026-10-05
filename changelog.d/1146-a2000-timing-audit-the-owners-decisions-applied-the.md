### A2000-timing audit, the owner's decisions applied: the runtime warning and two lane files stop quoting A2000 timings (no output change)

- **Runtime warning.** `enable_fast_train`'s warning, raised when a hybrid family's recurrent blocks fall back to
  transformers' reference PyTorch, no longer cites the A2000's 36 % device-time cut. It still names the fallback and the
  three packages to install. Decision 3 on #1133.
- **MG1's box runner.** `bench/moegen/mg1_run.sh` installs mamba-ssm and causal-conv1d for the structural reason: without
  them the recurrent blocks run unfused reference PyTorch, so an arm that pays the fallback is not the fast path. The
  A2000 36 % is no longer the reason given. Decision 4.
- **TC3's lane README.** The 12 GB section drops the A2000's 70 s/step, tokens/s and J/step, matching the relabelled row
  `e4b.train.frontier.qwen3.a2000-12gb.2026-10-02`; the receipts keep them as the record. Decision 4.
- Not in this change: the energy rows' rented rerun (decision 1) and the TC lane's band erratum (decision 5), each its
  own follow-up.
