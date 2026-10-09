### `recipe.estimate_env()`: the environment switches the training estimate reads

- **What changed.** `estimate_env()` (also `experts4bit_qlora.estimate_env`) returns the environment switches
  `estimate_qlora_footprint` reads, each valued as the engine interprets it in this process. Today that is
  `E4B_CHUNKED_LM_LOSS`, which decides whether the loss branch is priced chunked or whole (#1491), reported as
  `{"chunk", "auto_gate_bytes"}` from the chunked-loss module's own accessors. Unset, empty and `auto` therefore report
  the same value, so a planner comparing two processes warns only on a real difference.
- **Why.** A planner that prices in one process and runs in another can record this beside its plan and compare it
  before the run. loggetta does (loggetta#52): its plans record it, and `execute` warns when the running process
  differs.
- **Test.** `tests/test_topology_recipe.py::test_estimate_env_reports_the_switches_the_estimate_reads`. Every switch it
  reports changes the estimate in the chunked regime.
