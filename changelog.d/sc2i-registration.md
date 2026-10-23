### SC2i registered (#846): does spacing or shortening prefill lift `serve_paged`'s served decode under load at 64 slots?

- Lane SC2i reads e4b#2017's two opt-in scheduler knobs on one RTX 5090, at 64 slots, on SC2h's `base` server (SC2f's
  `on`: SC2e's s64a at the release's defaults). Five arms:
  - `base` (both knobs unset);
  - `E4B_PAGED_PREFILL_EVERY` 2 and 4;
  - `E4B_PAGED_LOADED_CHUNK_TOKENS` 256 and 128.
- Two draws of six servers each. Each draw brackets the four knob arms between two `base` servers, which measure its
  noise. The ladder is 4, 8 and 12 req/s.
- Each knob arm reads FASTER, FLAT or REGRESSED on TPOT p50 at 12 req/s against the draw's `base` mean, with a bar of
  max(0.05, the `base` pair's spread). FASTER also needs no loss of attainment and no TTFT p99 rise past 1 s. Nothing is
  licensed: a FASTER arm goes to a quality read first, because chunking a prompt differently can change its tokens.
- Box P (`bench/sc2/sc2i_box_p.sh`, `bench/sc2/sc2i_reduce.py`), wired into `bench/sc1/sc1_run.sh` and `sc1_drive.sh`
  beside boxes N and O, at grouped-nf4-gemm v0.50.0. SC2e's ENGAGED gate is adapted for the loaded-chunk arms only: a
  capped first chunk runs eagerly there by design.
- `tests/test_sc2i_box.py` covers box P; the SC1/SC2 box tests and the staged pins read the new box letter.
- The box runs after SC2f's and SC2h's, on its own rental, inside the launcher's 6 h cap (reading guard 4.5 h); lane worst $13.21.
