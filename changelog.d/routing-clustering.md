### Cross-request routing clustering at sixteen requests: small, real, not a lever (reported)

`bench/routing-clustering/` reads P153's committed router replay against two shuffled nulls, written down before any
draw. Same-step requests share experts 0.26 % (S0) and 0.64 % (S1) more than the nulls, both intervals clear of 1, far
from the 2 % a scheduling change would need. The 16-row tiles hold about 2.24 rows each (about 14 % full), a property of
top-8 routing over 128 experts at sixteen rows that cross-request overlap barely moves. Reported only.
