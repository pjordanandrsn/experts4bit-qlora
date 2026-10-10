### Training routing trace: score the complete steps of a capture that ended early

`replay.py --complete-steps N --truncation-reason TEXT` scores the first N optimizer steps of a capture that stopped
early (for example, an out-of-memory error in a later step). The capture's own verdict may then be VOID; the
completeness guard and the forward == recompute check are re-run on exactly the kept steps, and a fault inside the cut
still refuses. N cannot exceed the optimizer steps the capture completed, and the capture's meta and a reason are
required. The output's `trace_info` records the capture's verdict and reasons, the planned steps, N and the reason. The
loader now also re-checks forward == recompute on every replayed trace.
