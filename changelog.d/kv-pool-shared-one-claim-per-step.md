### Shared KV pool: one block claim per decode step, also when the step runs in pieces

- Under `E4B_PAGED_KV_POOL=shared` with bulk KV, a decode step wider than the largest captured bucket runs as
  consecutive replays of that bucket. Each piece made its own block claim (`Fp8PagedKV.claim_many`), and claims take
  alternate buffers of a two-buffer pinned staging ring. Under the decode lookahead, the next step's second piece
  therefore waited on an event recorded after this step's first replay, losing the host/device overlap the lookahead
  exists for.
- The synchronous path and the lookahead path now make ONE claim per step, over every piece's slots, before the first
  piece. Rows are popped in the same order as before (the pieces' slots in step order), so the block tables, and the
  tokens, are unchanged. Claim-all is unaffected.
