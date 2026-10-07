### P115 registered (#1313): the registered B=1 fused stack on the default `serve_paged` server, speed and teacher-forced quality on one RTX 5090 (bench and tests only)

- **Why.** The default NF4 graph server decodes one Qwen3-30B-A3B request at 9.05–10.06 ms per token (P109, P111),
  2.3× e4b's own int4 route. SV2's census puts ~1.7 ms of that step in glue the opt-in folds remove; bo7 timed the folds
  at 8.68 → 6.16 ms at B=1, but they have no K8 on record (bo5's one-text FAIL-by-improving predates 0.37.5's router
  cast), and fused q/k/v has never been timed on the bf16 route.
- **Phase A, speed** (`bench/p115/p115_box.py`, P111's protocol): F0a F1a F1b F0b on the default graph server, F1 =
  `E4B_PAGED_FUSE_QKV=1` + the three folds; W16 and W1, p37's slope. The fusion census is checked against SC1's
  receipts (48 / 193 / [48, 48] / 48).
- **Phase B, quality** (`bench/p115/p115_quality.py`, P110's instrument): an OFF process scores the default server's
  arithmetic R, its floor (half, chunk, rep) and the scale mutant on wikitext and c4val1 and saves R's log-probs; an ON
  process builds the fused stack as the server does and scores against them. Engagement counts every glue kernel call
  (49 / 48 / 96 / 48 per decode step on Qwen3) and every `qkv_proj` call.
- **The rule** (`bench/p115/p115_reduce.py`, 27 self-test cases): VOID, NOISY, FUNCTION_FAIL (F1 against F1 bitwise),
  QUALITY_FAIL (P110's floor bar on both texts; K8 ±0.05 ppl gated on wikitext, reported on c4val1, where 0.05 ppl is
  below the instrument's floor), SLOWER (g1 < 1.10 or g16 < 1.00), DEFAULT_AUTO. Phase C (engagement under `auto` on
  gpt-oss-20b and Qwen3.6-35B-A3B) is registered and lands as an amendment before its box.
- **Tests:** `tests/test_p115_staged_pin.py`; `tests/test_p115_quality_box.py` runs both quality phases on CPU and
  reproduces the census and per-step tables on tiny Qwen3-MoE and GraniteMoe models with the glue kernels stood in;
  `tests/test_fused_glue_decode_graphs_gpu.py` (the lane's premise on the card) replays glue rounds 1 and 2 under
  bucketed graphs and asserts they decode exactly as the padded eager step.
