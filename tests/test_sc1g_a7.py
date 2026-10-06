"""SC1g amendment A7 (bench/sc2/SC1g-PREREG.md): the router-flip instrument on box J. These tests are A7's wiring proof: the
arm list is driven through the real box script with only the engine calls stubbed (test_sc1g_a6's harness), the capture
(sc1g_k8.route_ids_capture) is run on CPU against a stand-in GEMV and read back through the reducer's own gate
(sc1g_reduce.route_ids), and the reading itself is self-tested on synthetic rows (sc1g_reduce.py --self-test)."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import types

import pytest

from test_sc1g_a6 import BOX, _drive

torch = pytest.importorskip("torch")
SC2 = pathlib.Path(__file__).resolve().parents[1] / "bench" / "sc2"
K8 = (SC2 / "sc1g_k8.py").read_text()


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, SC2 / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A7_ORDER = (["e4b_a7mx_served_conv2", "e4b_a7b_served_conv2", "e4b_serve_served_conv2"]
            + [n for s in ("conv1", "conv3", "conv4") for n in (f"e4b_a7mx_served_{s}", f"e4b_a7b_served_{s}")])


def test_box_j_runs_a7_and_keeps_the_a6_continuation_for_the_record():
    body = BOX[BOX.index("\nbox_j(){"):]
    body = body[:body.index("; }\n") + 4]
    assert "i_arms_a7\n" in body and "--out $W/sc1g/verdict_sc1g_a7.json" in body and "i_arms_a6c" not in body
    assert "SC1G_NCONV=4 i_windows || finish 19; i_ref_full_stage" in body and "install_" not in body
    kept = BOX[BOX.index("\nbox_j_a6c(){"):]
    assert "i_arms_a6c\n" in kept[:kept.index("; }\n")]


def test_a7_arms_run_conv2_first_with_the_uncaptured_control_third(tmp_path):
    """9 arms, each its own i_e4b call: conv2's (a) and (b) captured, then (b) UNcaptured (the perturbation control), then (a)
    and (b) captured on conv1, conv3, conv4. Each captured arm writes its own rid record; (a) differs from (b) only by
    KEEP_NF4; every arm reads box R's re-hashed rows."""
    got = [ln for ln in _drive(tmp_path, "i_arms_a7") if ln.startswith("ARM ")]
    names = [g.split("|")[0][4:] for g in got]
    assert names == A7_ORDER, names
    stacks = {}
    for g in got:
        name, stack, src = g[4:].split("|")
        assert f"SC1G_REF_FULL_FILE={tmp_path}/sc1g_ref_full/ref_full_{src}.npy" in stack and f"SC1G_KL_OUT={tmp_path}/sc1g/kl_{name}.npz" in stack
        cap = f"SC1G_ROUTE_IDS_OUT={tmp_path}/sc1g/rid_{name}.npz"
        assert (cap in stack) == (name != "e4b_serve_served_conv2"), (name, stack)
        assert ("E4B_INT4_KEEP_NF4=0" in stack) == name.startswith("e4b_a7mx_")
        stacks[name] = set(stack.split()) - {cap} - {w for w in stack.split() if w.startswith(("SC1G_REF_FULL", "SC1G_KL_OUT"))}
    assert stacks["e4b_a7mx_served_conv2"] ^ stacks["e4b_a7b_served_conv2"] == {"E4B_INT4_KEEP_NF4=0", "E4B_INT4_KEEP_NF4=1"}
    assert stacks["e4b_a7b_served_conv2"] == stacks["e4b_serve_served_conv2"]          # the control differs only by the capture


def test_a7_arms_are_refused_without_box_r_rows(tmp_path):
    calls = _drive(tmp_path, "i_arms_a7", norows=True)
    assert not [ln for ln in calls if ln.startswith("ARM ")]
    stubs = [ln for ln in calls if ln.startswith("STUB ")]
    assert len(stubs) == 9 and all(ln.endswith(" refused") for ln in stubs)


def test_box_j_stages_the_rows_runs_the_arms_then_the_a7_reading(tmp_path):
    calls = _drive(tmp_path, "box_j")
    stage = calls.index("STAGE")
    arms = [i for i, ln in enumerate(calls) if ln.startswith("ARM ")]
    reduce_ = [i for i, ln in enumerate(calls) if ln.startswith("PY ") and "verdict_sc1g_a7.json" in ln]
    assert len(arms) == 9 and reduce_ and stage < arms[0] and arms[-1] < reduce_[0]


def test_the_capture_wraps_the_kl_proxy_and_refuses_an_oracle_arm():
    """In k8(): route_ids_capture is installed AFTER full_capture / named_capture, so its boundary proxy wraps A5's (both see
    every scored row), and an --ppl-oracle arm (no served decode loop) is refused."""
    i_full, i_rid = K8.index("full_capture(step_decomp,"), K8.index("route_ids_capture(step_decomp,")
    i_named, i_main = K8.index("named_capture(step_decomp,"), K8.index("    step_decomp.main()")
    assert i_full < i_rid and i_named < i_rid < i_main
    blk = K8[K8.index('if os.environ.get("SC1G_ROUTE_IDS_OUT"):'):i_main]
    assert '"--ppl-oracle" in args' in blk and "refused" in blk


def test_route_ids_capture_records_each_scored_position_and_the_reducer_reads_it_back(tmp_path, monkeypatch):
    """CPU, end to end: warm decode steps, then per scored position one decode forward (24 layers x gate_up/down GEMV calls,
    ids in router order) and one [1, V] log_softmax row -- through an A5-like proxy already installed. The GEMV's output and
    the log-probs pass through unchanged, the inner proxy still sees every row, and the reducer's gate recovers each
    position's sorted sets exactly."""
    k8 = _mod("sc1g_k8")
    R = _mod("sc1g_reduce")
    monkeypatch.setattr(k8.atexit, "register", lambda f: f)
    P, L, warm = 7, 24, 3
    win = tmp_path / "win.json"
    win.write_text(json.dumps({"steps": P}))
    monkeypatch.setenv("SC1G_WINDOW_FILE", str(win))
    seen = []

    def gemv(xq, xs, blocks, scales, eids, N, K, part=None):
        return eids.to(torch.float32) * 2.0
    fake = types.ModuleType("mxfp4_grouped")
    fake.gemv_mxfp4_b32 = gemv
    monkeypatch.setitem(sys.modules, "mxfp4_grouped", fake)
    sd = types.SimpleNamespace(torch=k8._TorchProxy(torch, lambda x: seen.append(tuple(x.shape))))   # A5's proxy stand-in
    out = str(tmp_path / "rid_e4b_a7mx_served_conv2.npz")
    st = k8.route_ids_capture(sd, out, layers=L)
    g = torch.Generator().manual_seed(0)
    want = torch.stack([torch.stack([torch.randperm(32, generator=g)[:4] for _ in range(L)]) for _ in range(P)]).to(torch.int32)

    def forward(ids_per_layer):
        for lyr in range(L):
            for _which in ("gu", "dn"):
                e = ids_per_layer[lyr]
                y = fake.gemv_mxfp4_b32(None, None, None, None, e, 8, 8)
                assert torch.equal(y, e.to(torch.float32) * 2.0)
    for _ in range(warm):
        forward(torch.randint(0, 32, (L, 4), generator=g, dtype=torch.int32))
    for t in range(P):
        forward(want[t])
        row = torch.randn(1, 50, generator=g)
        assert torch.equal(sd.torch.log_softmax(row, -1), torch.log_softmax(row, -1))
    sd.torch.log_softmax(torch.randn(3, 50, generator=g), -1)                 # a non-row call: counted, not a position
    st["save"]()
    assert seen == [(1, 50)] * P + [(3, 50)]
    meta = json.load(open(out + ".json"))
    assert meta["gemv_calls"] == (warm + P) * 2 * L and meta["rows"] == P and meta["other_calls"] == 1 and meta["positions"] == P
    r = R.route_ids(str(tmp_path), "e4b_a7mx_served_conv2", positions=P)
    assert r["verdict"] == "VALID", r
    assert (r["sets"] == torch.sort(want, dim=-1).values.numpy()).all()


def test_route_ids_refuses_a_record_whose_positions_are_not_48_calls_apart(tmp_path, monkeypatch):
    """A replayed graph (or any decode row off the GEMV) shows up as a position without its 48 calls: VOID, never a short read."""
    k8 = _mod("sc1g_k8")
    R = _mod("sc1g_reduce")
    monkeypatch.setattr(k8.atexit, "register", lambda f: f)
    monkeypatch.delenv("SC1G_WINDOW_FILE", raising=False)
    fake = types.ModuleType("mxfp4_grouped")
    fake.gemv_mxfp4_b32 = lambda xq, xs, b, s, eids, N, K, part=None: eids
    monkeypatch.setitem(sys.modules, "mxfp4_grouped", fake)
    sd = types.SimpleNamespace(torch=torch)
    out = str(tmp_path / "rid_x.npz")
    st = k8.route_ids_capture(sd, out, layers=24)
    e = torch.arange(4, dtype=torch.int32)
    for t in range(3):
        for _ in range(48 if t != 1 else 24):                               # position 1's forward ran half its layers on the GEMV
            fake.gemv_mxfp4_b32(None, None, None, None, e, 8, 8)
        sd.torch.log_softmax(torch.zeros(1, 5), -1)
    st["save"]()
    r = R.route_ids(str(tmp_path), "x", positions=3)
    assert r["verdict"] == "VOID" and "exactly 48" in r["why"]
