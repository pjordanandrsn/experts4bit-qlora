# PREREG — S3: capture-safe device-grouped verification

Registered 2026-08-25, before any implementation and before any
measurement. Basis: S2's singleton verify bound (47.71 ms) is a
reuse-DISABLED upper bound; S2's own controlled data shows 3.28–4.91
route assignments per loaded expert in real verification windows. S3
answers the narrower unresolved question: **what does speculative
verification cost when repeated expert selections across the window
actually share their packed-weight read?**

## Mechanism (registered before code exists)

Fully device-side T>1 expert grouping, CUDA-graph-capturable: no
`.item()`, no `unique_consecutive` host-size dependency, no dynamic
Python loop, no host synchronization. From router outputs `[T, top_k]`:

1. Flatten the `T*top_k` (expert_id, token_row, route_slot)
   assignments on device.
2. Sort/bin by expert id entirely on device (`torch.sort` /
   `scatter_add` — in-stream ops).
3. Produce fixed-size `counts[E]` and prefix offsets on device.
4. Gather token rows into expert-major order (`index_select` by the
   sort permutation — static shape `R = T*top_k`).
5. Execute each expert's packed weights ONCE against its `M_e` routed
   rows. **Reuse the existing grouped NF4/M-tile machinery**: the
   M-tile kernel already consumes per-tile `(t_row0, t_rows, t_group)`
   from device tensors — the work is constructing those tensors on
   device at a STATIC tile budget `TILES = ceil(R/BM) + E` (an upper
   bound on `sum(ceil(M_e/BM))`), with unused tiles carrying
   `rows = 0`, which the kernel's own `m_mask` turns into a no-op. A
   new GEMM is NOT licensed unless the existing kernel provably cannot
   express this fixed-shape captured form.
6. Scatter results back to `(token_row, route_slot)` by the inverse
   permutation, preserving the existing route-weight/reduction
   semantics.
7. Launch shapes static; `counts[e] == 0` handled by masking; the
   active-expert count NEVER returns to Python.

## Correctness gates, in order, before any timing

1. **Grouping parity (CPU-checkable)**: the device tile construction
   vs host `build_group_tiles` on identical routing, exact equality
   of (row0, rows, group) over randomized `M_e` distributions
   including zeros and `M_e > BM`.
2. **Numeric parity**: device-grouped verify vs the existing EAGER
   grouped T>1 path on identical routing — recorded max|Δ|, required
   identical greedy verify tokens.
3. **Post-rewind continuation identity** (as S2's gate).
4. **The S2 sequential-oracle gate** re-run through the new path.

Any correctness failure REFUSES timing.

## Measurement (same box, one session)

Shipped 7.41 ms-class anchor with A/A; K ∈ {16, 32, 64}. Per K,
recorded: `verify_graph_ms`; accepted tokens/step from the COMMITTED
table (2.948 / 3.447 / 3.926 — not re-derived); per-layer
distinct-expert count; full `M_e` occupancy histogram; total route
assignments; grouping/scatter overhead (timed separately);
grouped-expert device time; total verify time. Three-way comparison
at K=16: **singleton vs eager-grouped vs captured device-grouped**,
so the reuse-disabled artifact inside 47.71 ms is quantified.

## Adjudication (fixed now; calculator committed with this prereg)

Per K: `T_pred(K) = accept(K) / verify_graph_ms(K)` as a multiple of
the anchor's rate; decision on the max over K:

- **GO** — ≥ 1.5× ⇒ executor stage with bars tied to the prediction.
- **INCONCLUSIVE/LIMITED** — 1.2–1.5× ⇒ executor only for the best
  cell with the PARTIAL bar as PASS.
- **REFUTED** — < 1.2× ⇒ grouped verification cannot pay on this
  architecture; THEN AND ONLY THEN the composed single-stream
  ceiling is recomputed and a statement about 425 is made.
- REFUSE on: any correctness-gate failure, anchor A/A wider than
  half the GO margin, `verify_graph_ms(K) < 0.9×` anchor, or a
  missing committed acceptance for a measured K.

The negative hypothesis is registered as live: even perfect grouping
touches 5.19× ordinary decode's distinct experts for 2.948 tokens at
K=16 (6.73× for 3.447 at K=32) — this may still lose. The cycle
exists to measure the best available reuse mechanism directly rather
than infer the limit from an arm that disables it.
