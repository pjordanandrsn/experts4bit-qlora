### Tests: the late-bound saving is measured as bytes held across the forward, against a same-config control (tests only; #1186)

- **Why:** #1183's power control compared a resident baseline with an offloaded model. At `0bf98cf3`, with the route
  disabled, it reported 42.8 MB saved, more than the toy's 25 MB of weights.
- **The new measurement:** `memory_allocated()` immediately before and after a warm forward, with `gc.collect()` first.
  The stock-bnb graph can sit in a reference cycle through its ctx.
- **The tests:**
  - route-off minus route-on in the same offloaded config must be at least (L−2) layers;
  - route-on must be at most resident + 2 layers.
- **On the A2000:** every config has 0.00 MB spread over 15 samples. The ctx-pin mutant fails all four tests.
