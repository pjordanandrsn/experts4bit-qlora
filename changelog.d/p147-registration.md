### Registered: P147, a quality premise for compressing Mixtral-8x7B's attention projections

`bench/p147/PREREG-p147.md`: does int8 or NF4 weight-only rounding of Mixtral-8x7B's attention projections keep the W8A16 split premise's gate (argmax agreement ≥ 0.95 and the one-sided NLL upper bound within K8's budget, at two gates)? The run is on the lab's RTX A2000 with expert offload, at $0, and reports no timing. A premise: a GO starts kernel work and a later 5090 speed registration that re-checks quality on the served card.
