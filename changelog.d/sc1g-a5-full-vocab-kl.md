### SC1g amendment A5 registered (#846): engines graded by the full-vocabulary KL, each window above its own floor; box R's rule pinned beside A4's (bench and tests only)

- **The estimator.**
  - Box R stores the reference's decode-shaped log-softmax over all 201,088 tokens: fp16, computed in fp64, masked `−inf`
    kept exact, ~0.82 GB per window, sha-pinned, outside git.
  - Engines read KL(p_ref ‖ p_eng) in fp64 over the full vocabulary (`sc1g_kl.kl_full_rows`), with no truncation bound.
  - A4's coverage and calibration checks are dropped by name.
- **R's gate (`verdict_a5`).**
  - The fp16 storage error must be ≤ 0.1 × F on both of R's pairs, on every gradable window.
  - A conv window is graded only if its own F < 1e-2. Wikitext is never graded.
  - Fewer than 3 gradable windows gives the new `R_NO_GRADABLE` outcome, a cost gate: box I is not launched.
  - Digest `2d63276a…` is pinned beside A4's `verdict()` sha. `--reverdict` dispatches on the receipt's rule, and A4's
    committed read still re-derives.
- **Engines.**
  - e4b: the served proxy's full capture.
  - vLLM: `logprobs=-1` with `LLM(max_logprobs=-1)`, verified full on request 0, VOID with no downgrade.
  - llama.cpp: the harness's `--ref-full`/`--kl-out`.
  - SGLang: UNREAD by registration.
- **Predictions.** K-A, L1 and L2 are graded only where a KL exceeds its window's floor:
  - F is used as the upper bound of an unresolved MXFP4 denominator or e4b row;
  - a window is dropped when NF4, or the best comparator, is unresolved.
- **Storage.** The rows' durable copy goes to the QNAP's Pool 3. Box I stages them and refuses any KL arm whose rows do not
  match the registered sha. `sc1g_ref` joins the SC staging lists.
