# PREREG — B1d: the device-driven decode loop (425-single-stream, rung 2)

Registered 2026-08-24, before any measurement. The rung the b1c cert
licensed (`RESULTS-b1c-collapse.md`): the collapsed all-resident B=1
step is 49.8 ms of which ~13.3 ms is device work — the remaining
~36 ms is host launch/framework time on a path that now has NO
data-dependent host decisions. Static shapes + no host branching is
the shape `torch.cuda.CUDAGraph` captures. **Scope guard**: this line
binds to the collapsed all-resident single-stream point. T1b's
cudagraph refutation was at the graph-broken hybrid B=16 point and
does not bind here; this prereg neither reopens nor relies on it.

## Roofline frame (from measured receipts, restated as the campaign map)

5090 device triad 1573.9 GB/s; ~3.2 GB weights read per token ⇒
~2.0 ms/token ≈ 480-490 tok/s ceiling. This rung's own ceiling is the
CURRENT device floor: ~13.3 ms/step ≈ **75 tok/s** — it removes host
time, not device time. Rung 3 (M=1 kernel roofline) then owns the
13.3 → ~2.5-4 ms descent; rung 4 (speculative) crosses 425.

## Capture-cleanliness inventory (from code reading, the work list)

The decode step must become replayable: one graph, static buffers,
zero host decisions inside. Known offenders, each with its registered
treatment:

1. **KV page allocation** (`_ensure_blocks` inside `append_many`):
   host branching when a sequence crosses a page boundary. Treatment:
   pre-ensure pages for step t+1 OUTSIDE the graph after each replay;
   the in-graph append writes to pre-reserved pages only. Boundary
   steps that would allocate run the eager path (measured fraction
   reported; at BLOCK_TOKENS-page granularity this is a bounded
   fraction of steps).
2. **Block-table / seq-len indexing in `kv.attention`**: any per-call
   host-built index tensors must become device-resident state updated
   in-place (seq_lens incremented on device; block tables written at
   pre-ensure time). The attribution run (stage A) measures how much
   of the ~15-20 ms B=1 attention-host bracket this is.
3. **Token feedback**: `argmax` → next `input_ids` stays on device
   (in-place copy into the static input buffer); `position_ids`
   incremented in-place. The host reads tokens asynchronously for the
   transcript/identity gate — never on the critical path.
4. **The collapsed MoE path**: already clean (placement-static, no
   syncs) — the b1c construction was chosen for exactly this.
5. **Router/norms/dense**: stock torch ops, static shapes — clean.

## Stages

- **A (capture smoke + attribution)**: make one decode step capture
  under `torch.cuda.CUDAGraph` (manual capture, NO dynamo/compile —
  the T1/T1b failure modes were compile-layer). Deliverable: capture
  succeeds, replay produces bit-identical logits vs eager for 8
  consecutive steps on-box, and a bracket attribution of what remained
  outside the graph. STOP-AND-AMEND if capture cannot succeed without
  engine changes beyond the inventory above (the amendment names
  them BEFORE they are built).
- **B (the loop)**: replay-per-step serving with pre-ensure outside
  the graph; eager fallback on page-boundary steps.
- **C (cert)**: A/B/A vs the collapsed baseline (b1c C1 shape), same
  pinned command line, same G0/GS/G1 gates as b1c (token records
  non-empty by construction; C1-vs-graph tokens bitwise).

## Bars (before any number)

- **H-G (primary)**: median step ≤ **20 ms** (from 49.8; ≥ 60% cut —
  host time must mostly vanish, not merely shrink; 50 tok/s+).
  PARTIAL band (20, 30] ms; > 30 ms ⇒ REFUTED for the capture
  approach: revert, and rung 3 proceeds against the eager collapsed
  baseline (the ladder does not stall on this rung's failure).
- **H-D (device sanity)**: the graph-replay step's kernel occupancy
  within 15% of the eager collapsed path's (the graph must not have
  smuggled in extra device work).
- **G1**: decoded tokens bitwise vs the eager collapsed arm across
  the full window; the eager-fallback boundary steps included.
- **GS**: the B=16 certified point untouched (no flag leakage —
  the graph loop is a B=1 serving mode, never default for batch).
- Consequences: pass ⇒ CERTIFIED, the graph loop becomes the B=1
  serving default behind a placement gate (all-VRAM only), and rung 3
  registers against the new step floor. The eager collapsed path
  remains the fallback and the A/B arm forever.

## Verdict calculator

`b1d_verdict.py`, self-tested both directions (including a
boundary-step token-divergence fixture and an empty-token refusal)
before any box is rented. Receipts in `receipts-b1d/`.
