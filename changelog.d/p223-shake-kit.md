### P223 shake kit: Amendment 1's shakedown ids on the one-layer cut

The P223 kit gains Amendment 1's shake mode, P149's Amendment 5 dry mode ported:
- **Ids and switch:** run ids `p223-shake-1` to `-3` set `P223_SHAKE=1` through the driver. The runner refuses a switch
  that does not match the id, either way, before any install (rc 78).
- **The run:** it fetches shards 1 and 19 and cuts them to one layer with `p149_slice.py`, now staged and pinned. Then it
  runs the full kit at the shake table: 4,890 s against a 1.75 h guard, the runner needing 5,737 s, and 2 timed reps.
  Every check is kept. Each arm's early read reports and goes on.
- **The reduce:** `p223_reduce.py --shake` holds every layer-dependent expectation at one layer: P149's `census(1)`, 4
  int8 projections with 12 calls a captured bucket reported, the slice's model and 2 reps. Its verdict is SHAKE, hard
  coded. It lists every instrument fault across the processes and reports the reads ungraded.
- **The runner's exit:** rc 0 only on SHAKE with no fault, else 29. A shake can never write READ, and a test proves it.

No default, gate or claim moves.
