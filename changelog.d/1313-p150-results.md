### P150 results (#1313): the int8 lm_head keeps the quality gate on held-out windows (GO)

Rounding only Qwen3-30B-A3B's lm_head to its per-row int8 grid (the lever P148 read in a microbench) keeps P125's
quality gate at both ruled gates. The read used wikitext-validation windows nobody had read, with their sha256s
registered before data:
- t1: argmax 0.989, d + 1.645 SE 0.0012 against a bound of 0.0052;
- k16: argmax 0.990, 0.0011 against 0.0057.

This is a ruling on quality only. The int8 head's speed is a served read of its own, after its kernel ships, and nothing
changes by default. Record: `bench/p150/RESULTS-p150.md`.
