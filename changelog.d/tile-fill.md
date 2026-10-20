### Tile fill and its amortization: rows per step are the lever (reported)

`bench/tile-fill/` reads P153's router replay under a construction committed first, with Amendment 1 after the first
width composition failed its positive control (kept and reported). At sixteen requests K25's 16-row tiles are about
14 % full. Composed to 32, 64 and 128 rows per step, tiles per row fall to 0.64, 0.39 and 0.26 of sixteen-wide (an
optimistic bound from 16 requests). Verify blocks of consecutive tokens add only a few percent of sharing on top
(0.96-0.98), and a request's tokens share about 2 % more experts than separate requests'. Reported only: a W32/W64 or
verify-width served read would test the premise that cost tracks tiles.
