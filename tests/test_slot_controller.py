"""CI unit tests for engines.slot_controller — pure CPU, mocked tier
states. These are the production component's standing guard: the gated
swap rule, the noise gate, per-layer isolation, series trimming, and the
change-point reset."""
import torch

from experts4bit_qlora.engines.slot_controller import SlotController


class MockState:
    def __init__(self, E, hot):
        class Mod:
            num_experts = E
        self.mod = Mod()
        self.swappable = True
        self.hot_ids = torch.tensor(sorted(hot), dtype=torch.long)
        self.is_hot = torch.zeros(E, dtype=torch.bool)
        self.is_hot[self.hot_ids] = True
        self.amort = {"series": []}
        self.swapped = []

    def swap_expert(self, promote, demote):
        assert bool(self.is_hot[demote]) and not bool(self.is_hot[promote])
        self.is_hot[promote] = True
        self.is_hot[demote] = False
        hm = self.hot_ids == demote
        self.hot_ids = torch.where(
            hm, torch.tensor(promote, dtype=self.hot_ids.dtype),
            self.hot_ids)
        self.swapped.append((promote, demote))

    def push(self, touched):
        self.amort["series"].append(torch.tensor(touched, dtype=torch.long))


def drive(ctrl, states, per_layer_touched, steps):
    for _ in range(steps):
        for st, ids in zip(states, per_layer_touched):
            st.push(ids)
        ctrl.on_decode_step()


def test_hot_shift_swaps_and_respects_layers():
    E = 16
    s0 = MockState(E, hot=[0, 1, 2, 3])
    s1 = MockState(E, hot=[0, 1, 2, 3])
    prior = [[0.5 if e < 4 else 0.01 for e in range(E)] for _ in range(2)]
    ctrl = SlotController([s0, s1], prior)
    # layer 0's routing moved to experts 8,9; layer 1 unchanged
    drive(ctrl, [s0, s1], [[8, 9], [0, 1]], 40)
    assert any(p in (8, 9) for p, _ in s0.swapped)
    assert s1.swapped == [] or all(p in (0, 1) for p, _ in s1.swapped)
    assert bool(s0.is_hot[8]) and bool(s0.is_hot[9])


def test_noise_gate_blocks_tied_margins():
    E = 8
    st = MockState(E, hot=[0, 1, 2, 3])
    prior = [[0.3] * E]
    ctrl = SlotController([st], prior)
    # everything touched every step: est identical everywhere -> no swaps
    drive(ctrl, [st], [list(range(E))], 40)
    assert st.swapped == []


def test_series_is_trimmed():
    E = 8
    st = MockState(E, hot=[0, 1])
    ctrl = SlotController([st], [[0.2] * E], trim_series=True)
    drive(ctrl, [st], [[0, 1]], 10)
    assert len(st.amort["series"]) == 1


def test_series_kept_when_asked():
    E = 8
    st = MockState(E, hot=[0, 1])
    ctrl = SlotController([st], [[0.2] * E], trim_series=False)
    drive(ctrl, [st], [[0, 1]], 10)
    assert len(st.amort["series"]) == 10


def test_change_point_resets_history():
    E = 16
    st = MockState(E, hot=[0, 1, 2, 3])
    prior = [[0.5 if e < 4 else 0.01 for e in range(E)]]
    ctrl = SlotController([st], prior, cp=True)
    drive(ctrl, [st], [[0, 1, 2, 3]], 32)      # steady on the hot set
    assert ctrl.cp_resets == 0
    drive(ctrl, [st], [[10, 11]], 12)          # hard switch off the hot set
    assert ctrl.cp_resets >= 1


def test_requires_swappable_and_armed():
    E = 8
    st = MockState(E, hot=[0])
    st.swappable = False
    try:
        SlotController([st], [[0.1] * E])
        raise SystemExit("should have refused")
    except AssertionError:
        pass
