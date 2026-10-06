### TC1 amendment 54 registered: two remedies for torch 2.8's host time at e4b's defaults (P137-P140)

- Token `qwen3ladder28` runs e4b's matched arm on packed rows in torch 2.8 at its defaults, against two remedies:
  - cuBLASLt's heuristics cache raised to 262,144 entries;
  - grouped-nf4-gemm's bucket ladder (#498).

  Each side runs two draws, profiled on steps 3-5.
- P137: ladder / defaults <= 0.95. P138: cache / defaults <= 0.97. P139: held-out within 0.005. P140: the ladder's peak rises at most
  1.5 GB.
- The arm's receipt records the ladder and cache settings, and the reducer adds the family, `ladder28_why` and `score_ladder28`. Self-test
  119.
- The RTX A2000 cuBLAS API count behind the *Why* is committed: `bench/tc1/bmm_api_count.py` and
  `bench/h2h-2026-10-02/tc1/a2000-cublas-api/`.
