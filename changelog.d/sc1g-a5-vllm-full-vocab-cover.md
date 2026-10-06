### SC1g A5 (#846): vLLM's full vocabulary is verified by coverage, not a count; box I's proof record (bench and tests only)

- **The proof.** Box I's proof `sc1g-prove-a5-6` read e4b served and llama.cpp q8 VALID: zero masked reference mass and
  no void positions on all 2,048 positions. So no common-support rule is registered for either.
- **The defect.** vLLM's row was VOID, because `logprobs=-1` returned 201,089 entries for the 201,088-token vocabulary.
  vLLM 0.30.0's V2 runner returns the generated token first and then every token
  (`vllm/v1/worker/gpu/sample/logprob.py`, `compute_topk_scores`). The count check was wrong, not vLLM.
- **The fix.** `full_vocab_cover` replaces the count on A5's path, at request 0 and at every position:
  - every id must be present;
  - V + 1 entries are accepted only when the repeat is entry 0 with an identical log-prob;
  - anything else is still VOID, with no downgrade.
  - SC1's own arms are unchanged.
- **Tests.** Unit cases for the coverage check, and the served loop on vLLM's real vocab + 1 flat shape. The latter fails
  with the count check.
- **PREREG.** It records the six proof attempts (`-a5-1` to `-a5-6`, $1.03 in all) and the support decision.
