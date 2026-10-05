### SV1 registered (#1152): the serve estimate beside decode graphs and the prefill graph (bench and prereg only)

- `bench/sv1/` (`SV1-PREREG.md`, `sv1_run.sh`, `sv1_measure.py`) for one RTX 5090. `serve_paged`'s engine is built
  in-process with the environment `ServeSetup.to_env()` gives, all-VRAM, from arenas baked on the box.
- Arms: OLMoE-1B-7B with eager decode, decode graphs, and decode graphs + the prefill graph; then Qwen3-30B-A3B with
  decode graphs, and + the prefill graph.
- Readings:
  - S1: the estimate against the eager peak;
  - S2–S3: the two graph pools the estimate lists as not modelled;
  - S4: the same at 30B NF4.
