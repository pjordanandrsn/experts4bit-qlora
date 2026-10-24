### P233 registered: does serving at the solver placement keep quality? (MAP 2)

- `bench/p233/PREREG-p233.md`: `serve_paged` at `E4B_PAGED_PLACEMENT=solver` with Qwen3-30B-A3B's experts across VRAM,
  host DRAM and NVMe (arm S), against the same server at `all-vram` (arm A), both eager, teacher-forced with P115 Phase
  B's `measure_phase` at group T = 1 and 8 over 24 wikitext windows x 128 positions, P153's bars per T. A VRAM+NVMe arm
  (C) is reported only, to split S's gap into the tiered forward's part and the DRAM tier's. Each arm's tier placement
  is proved from its record, or the read is VOID. The verdict (SOLVER_SANE / SOLVER_NOT_SANE) decides only whether a doc
  or lane may propose serving at `solver`; it says nothing about speed, and `solver` stays non-default. The kit follows
  as its own PR.
