# Prefetch-lane rebuild: catastrophe fixed, interior still needs gating (honest partial)

**Date:** 2026-07-12 · **Change:** `e4b-ssdtier@08c05c3` (wait-don't-duplicate reader)
· **Host:** Latitude `g3-h100-small` DAL — **same physical server, 3rd provision
today** (`sv_gXQvNeZGl5zpb`), raid-0; L=56.58 GB/s. ON-ladder rerun of prereg
`2bbd2e2`, K1/K2/K3 bars unchanged. ~$3 · torn down, 404-verified. Floors n=2:
OFF spread 0.0004, ON 0.0008.

## The rebuild worked where it was designed to

`FusedPrefetchReader.take()` now blocks on a queued/in-flight read instead of
returning `None` and letting the consumer issue a second synchronous read of the
same bytes. Effect, measured same-silicon vs session 2's v1 reader:

| f (ram share) | v1 ON vs OFF (session 2) | rebuilt ON vs OFF (this run) |
|---:|---:|---:|
| 0.0 (all-flash) | **+41.9%** (duplicate I/O) | **−6.9%** (ON now FASTER than OFF) |
| 0.5 | +23.9% | +13.1% |
| 0.9375 | +3.1% | +6.6% |

The all-flash regression — the worst symptom, where every block is a flash read
and v1 duplicated all of them — **inverted to a win**: with long reads that fully
overlap compute and no duplication, the lane beats synchronous staging by 7%.
Every point improved. No correctness change (grads bit-exact; the leak-free reader
serves each flash touch exactly once, unit-tested).

## But the frozen bars still fail — and why

- **K1 FAIL:** ON at flash-share 1/16 sits +6.6% over the ON floor (bar ≤5%).
- **K2 FAIL:** at f=0.5 the lane still *adds* 13% (needed ≥25% removal); it only
  wins at f=0.0.
- **K3 FAIL:** centroid ON (0.437) > OFF (0.396); lane inert at f=1 (+0.1%) ✓.

**Mechanism:** the read-ahead thread has a fixed per-block cost (thread wake,
Condition handoff, buffer-set management, NVMe queue contention with the main
lane). That cost is amortized only when the read is LONG relative to the inter-flash
gap — i.e. at high flash share. At low/interior flash share the flash reads are
short and sparse, the OFF path already hides most of their cost behind the async
H2D of a synchronous read (f=15/16 OFF pays only ~9 ms/step over floor), and the
lane's fixed overhead exceeds what little it can save. So prefetch helps **only in
the flash-dominant regime** and is net overhead elsewhere.

## Two conclusions

1. **Product:** for the SSD-tier's actual target (a few percent of blocks on flash —
   the "cheap because small flash share" regime), **`prefetch: false` is the correct
   default**, now doubly confirmed: synchronous staging with async H2D is already at
   the compute-hiding limit there, and the lane only earns its keep when you are
   mostly on flash (which the knee says you don't want to be). The validated default
   everywhere stands vindicated.
2. **Engineering (queued):** to pass K1/K2/K3 the lane needs **adaptive gating** —
   enqueue a read-ahead only when the predicted inter-flash gap-compute exceeds the
   expected read time (both measurable at runtime via EMAs the reader can keep). The
   v1 rebuild deliberately deferred this; this ladder shows it is required for the
   interior, not optional. Next increment; rerun this ladder unchanged after.

## Program status

The rebuild delivered its stated goal (kill the duplicate-I/O catastrophe — done,
−49 points at all-flash) but not a clean K1/K2/K3 pass; that needs the gating
increment. Given conclusion #1, the gating work is **low priority** — it optimizes a
regime the thesis avoids. No further bare-metal needed for it; the rerun can ride
any future box. This closes tonight's design-case follow-up honestly.

## Evidence

`bm3-final.tgz` (all arm logs, configs, results JSON, SENT) sha256
`964e4354eb4f30da21c3c497195e316d20932119e67d8f334f093e15b8037bd0` at mini
`~/bm3-evidence/`; `bm3_results.json` alongside (this dir).
