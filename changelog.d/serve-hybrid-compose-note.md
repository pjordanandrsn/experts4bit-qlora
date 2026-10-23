### HC design note: serve_paged at the solver placement (MAP 2; USD 0, no box)

- `bench/serve-hybrid/DESIGN-HC.md`: what composes with the hybrid tier today, why a tiered layer cannot be captured
  into a decode graph (and why no capture can silently skip a cold expert), what is unread, and the one explore box.
- `tests/test_serve_hybrid_compose.py` (CPU) pins the startup matrix, the collapse gate and the sync-before-branch order.
