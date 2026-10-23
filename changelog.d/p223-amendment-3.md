### P223 Amendment 3 (before any reading): no links on the fetch, a download probe, retry ids

P223's reading gains three pre-data guards, none a term of the read. The driver's fetch passes `rsync --no-links`
(bench/common rule 8). A download probe before the start refuses a host below 40 MB/s with rc 14 (HOST_REFUSED, never a
strike), writes `download_probe.json` on every box, and three consecutive refusals stop the renting. The retry ids
`p223-5090-r1` to `-r8` follow boxes that did not start. The lane worst is re-priced to USD 11.98 (2 starts × USD 4.81 +
9 non-starts × USD 0.26).
