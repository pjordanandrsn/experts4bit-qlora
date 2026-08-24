"""The production slot controller — the C1/C2 rule as an engine-owned
component (PREREG-productionization.md).

Attach after `enable_hybrid_tier(..., swappable=True)`:

    ctrl = SlotController.attach(runner, states, prior)

The runner calls `on_decode_step()` once per decode step; every EPOCH
steps the controller re-estimates per-expert touch rates from its own
trailing window and performs the gain-gated, noise-aware, per-layer
swaps C1 certified and C2 measured in-engine (22.7% uniques, 22.2%
dram-bucket wall at the 10 GB operating point).

Change-point handling (`cp=True`): at a hard content switch the
persistence estimator errs 73–232% for ~one horizon (the online cycle's
boundary data). When the trailing-CP_WINDOW touch-mass of the CURRENT
hot set falls below CP_RATIO × its trailing-window mean, the history is
truncated to the last CP_WINDOW steps so re-convergence takes one epoch
instead of one horizon.

The controller is the sole consumer of the amortization series in
production and trims it as it reads (`trim_series=True`), so a
long-running server holds O(TRAIL) history, not O(steps). Receipt runs
that also dump the full series pass `trim_series=False`.
"""
from __future__ import annotations

import collections


class SlotController:
    EPOCH = 8
    TRAIL = 32
    PRIOR_FLOOR = 0.25
    THETA = 4.0 / 32.0
    CP_WINDOW = 8
    CP_RATIO = 0.5

    def __init__(self, states, prior, *, cp: bool = False,
                 trim_series: bool = True):
        self.states = list(states)
        self.L = len(self.states)
        self.E = int(self.states[0].mod.num_experts)
        assert len(prior) == self.L and len(prior[0]) == self.E
        self.prior = prior
        self.cp = cp
        self.trim = trim_series
        self.trail: collections.deque = collections.deque(maxlen=self.TRAIL)
        self.step = 0
        self.swaps = 0
        self.cp_resets = 0
        self.ns = 0                    # controller's own wall, reported
        for st in self.states:
            assert getattr(st, "swappable", False), \
                "SlotController needs enable_hybrid_tier(swappable=True)"
            assert st.amort is not None, \
                "SlotController needs arm_amortization(True)"

    @classmethod
    def attach(cls, runner, states, prior, **kw):
        ctrl = cls(states, prior, **kw)
        runner.slot_controller = ctrl
        return ctrl

    # -- called by the runner, once per decode step, between forwards -----
    def on_decode_step(self):
        import time
        t0 = time.perf_counter_ns()
        try:
            self._on_decode_step()
        finally:
            self.ns += time.perf_counter_ns() - t0

    def _on_decode_step(self):
        step_touch = []
        for l, st in enumerate(self.states):
            ser = st.amort["series"]
            ids = ser[-1].tolist()
            if self.trim:
                del ser[:-1]
            step_touch.append(ids)
        self.trail.append(step_touch)
        self.step += 1
        if self.cp and len(self.trail) > self.CP_WINDOW:
            self._maybe_reset()
        if self.step % self.EPOCH == 0 and len(self.trail) >= self.CP_WINDOW:
            self._tick()

    def _hot_mass(self, window):
        m = 0
        for step_touch in window:
            for l, ids in enumerate(step_touch):
                hot = self.states[l].is_hot
                for e in ids:
                    if bool(hot[e]):
                        m += 1
        return m / max(1, len(window))

    def _maybe_reset(self):
        recent = list(self.trail)[-self.CP_WINDOW:]
        older = list(self.trail)[:-self.CP_WINDOW]
        if not older:
            return
        r = self._hot_mass(recent)
        o = self._hot_mass(older)
        if o > 0 and r < self.CP_RATIO * o:
            keep = list(self.trail)[-self.CP_WINDOW:]
            self.trail.clear()
            self.trail.extend(keep)
            self.cp_resets += 1

    def _tick(self):
        nn = len(self.trail)
        E = self.E
        for l, st in enumerate(self.states):
            cnt = [0] * E
            for step_touch in self.trail:
                for e in step_touch[l]:
                    cnt[e] += 1
            pl = self.prior[l]
            est = [max(cnt[e] / nn, self.PRIOR_FLOOR * pl[e])
                   for e in range(E)]
            hot = set(st.hot_ids.tolist())
            ins = sorted((e for e in range(E) if e not in hot),
                         key=lambda e: -est[e])
            outs = sorted(hot, key=lambda e: est[e])
            k = 0
            while k < len(ins) and k < len(outs):
                a, b = est[ins[k]], est[outs[k]]
                sd = ((a * (1 - a) + b * (1 - b)) / nn) ** 0.5
                if a - b <= max(self.THETA, 3 * sd):
                    break
                st.swap_expert(ins[k], outs[k])
                self.swaps += 1
                k += 1
