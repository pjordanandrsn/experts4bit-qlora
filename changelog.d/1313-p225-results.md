### P225 results: FAIL on G3 alone, the latency control; the graded quality gates hold

`p225-5090-r3` (one RTX 5090, $1.163) answered P225's question, and the lane ended at that answer. A Loggetta attention
adapter trained 80/80 steps passed every adapter, engagement and identity check. It learned (5.21 nats on the assistant
positions), and on the paged server it held G1, the ratio and the SE as graded. The verdict is FAIL because of G3, the
two-sided latency control: the merged adapter reached the first token about 1 % faster than the base in both blocks
(2.748 ms against a 2.388 ms band in block 0; 2.360 ms, just inside, in block 1). The reported, ungraded lines also show
the merged path scoring better than the reference beyond its standard errors (all-token G1 would fail; both DiD flags
set). Why the merged serve path differs is an open question, read next at $0. `bench/p225/RESULTS-p225.md`, receipts
in `bench/p225/receipts/p225-5090-r3/`.
