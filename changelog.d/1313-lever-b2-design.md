### Added

- Lever B2 design note (`bench/lever-b2/DESIGN.md`, no code, no box): a multi-row K33 with an `M_ROWS` constexpr over
  the device tile table, opened by P227's B2 trigger (B over the byte floor at T = 8: Mixtral 2.00, Qwen3 1.59). It
  names the row grouping (`_tile_table_r1` with `BM = M_ROWS`, outputs in sorted order where K25 runs today), the
  bitwise-per-row contract against M = 1 and its GPU check, the kill criteria, and the order of work; the weight bytes
  per `M_ROWS` come from P227's pinned routing (Mixtral at T = 8: 2.213x the floor at M = 1, 1.027x at M = 4).
