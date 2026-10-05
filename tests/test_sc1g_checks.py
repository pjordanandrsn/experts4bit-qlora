"""SC1g A3's two side checks, their pure-torch parts (CPU): the attention reference sc1g_attn_check.py grades the fp8 decode
kernel against, and the routing / grouping sc1g_mtile_check.py feeds the NF4 M-tile. The GPU halves ran on the A2000
(bench/sc2/sc1g-a2000/a3_*) and ride box J on the 5090."""
import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")
SC2 = pathlib.Path(__file__).resolve().parents[1] / "bench" / "sc2"


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, SC2 / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


A = _mod("sc1g_attn_check")
M = _mod("sc1g_mtile_check")


def _qkv(T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(A.HQ, A.D, generator=g), torch.randn(T, A.HKV, A.D, generator=g),
            torch.randn(T, A.HKV, A.D, generator=g), torch.randn(A.HQ, generator=g))


def test_attend_with_zero_query_and_no_sinks_averages_the_values():
    q, K, V, s = _qkv(40)
    out = A.attend(torch.zeros_like(q), K, V, 0, s, A.D ** -0.5, use_sinks=False)
    want = V.mean(0).repeat_interleave(A.HQ // A.HKV, dim=0)
    assert torch.allclose(out, want, atol=1e-6)


def test_attend_window_reads_only_the_last_keys():
    q, K, V, s = _qkv(300)
    K2, V2 = K.clone(), V.clone()
    K2[:-128] = torch.randn_like(K2[:-128]) * 50          # anything outside the window must not matter
    V2[:-128] = 1e3
    sc = A.D ** -0.5
    assert torch.allclose(A.attend(q, K, V, 128, s, sc), A.attend(q, K2, V2, 128, s, sc), atol=1e-6)
    assert not torch.allclose(A.attend(q, K, V, 0, s, sc), A.attend(q, K2, V2, 0, s, sc), atol=1e-3)


def test_attend_sink_takes_mass_and_never_contributes_a_value():
    q, K, V, s = _qkv(64)
    sc = A.D ** -0.5
    huge = torch.full_like(s, 80.0)                        # a dominant sink absorbs nearly all the softmax mass
    assert A.attend(q, K, V, 0, huge, sc).abs().max() < 1e-6
    low = torch.full_like(s, -80.0)                        # a negligible sink == no sink
    assert torch.allclose(A.attend(q, K, V, 0, low, sc), A.attend(q, K, V, 0, s, sc, use_sinks=False), atol=1e-6)


def test_attend_maps_query_heads_to_kv_heads_in_blocks():
    q, K, V, s = _qkv(32)
    V2 = V.clone()
    V2[:, 1] += 7.0                                        # only kv head 1 changes: only q heads 8..15 may move
    sc = A.D ** -0.5
    d = (A.attend(q, K, V2, 0, s, sc) - A.attend(q, K, V, 0, s, sc)).abs().amax(-1)
    G = A.HQ // A.HKV
    assert torch.all(d[G:2 * G] > 1e-3) and torch.all(torch.cat([d[:G], d[2 * G:]]) < 1e-6)


@pytest.mark.parametrize("kind", ["uniform", "skewed"])
def test_route_gives_top4_distinct_experts_per_token(kind):
    ids = M.route(500, 32, kind, torch.Generator().manual_seed(1))
    assert ids.shape == (500, M.TOPK)
    assert all(len(set(r.tolist())) == M.TOPK for r in ids)
    top = torch.bincount(ids.reshape(-1), minlength=32).max().item()
    assert (top > 400) == (kind == "skewed")               # the skewed draw puts one expert on most tokens


def test_grouped_sorts_by_expert_and_order_inverts_it():
    g = torch.Generator().manual_seed(2)
    ids = M.route(50, 32, "uniform", g)
    x = torch.randn(50, 8, generator=g)
    xs, sizes, eids, order = M.grouped(x, ids)
    assert sum(sizes) == 50 * M.TOPK and eids == sorted(eids) and len(set(eids)) == len(eids)
    back = torch.empty_like(xs)
    back[order] = xs
    assert torch.equal(back, x.repeat_interleave(M.TOPK, dim=0))
    r0 = 0
    flat = ids.reshape(-1)
    for m, e in zip(sizes, eids):                          # each group's rows really are routed to that expert
        assert torch.all(flat[order[r0:r0 + m]] == e)
        r0 += m


# ---- A4: the e4b named-token capture (sc1g_k8's torch proxy) ----------------------------------------------------------

K8 = _mod("sc1g_k8")
KLM = _mod("sc1g_kl")


def test_torch_proxy_forwards_tensor_and_everything_else():
    rows = []
    px = K8._TorchProxy(torch, rows.append)
    x = px.ones(2, 3)
    assert px.Tensor is torch.Tensor and isinstance(x, px.Tensor) and isinstance(x, torch.Tensor)
    assert px.float32 is torch.float32 and px.no_grad is torch.no_grad and px.__version__ == torch.__version__
    y = px.log_softmax(torch.zeros(1, 5), -1)
    assert torch.allclose(y, torch.log_softmax(torch.zeros(1, 5), -1)) and len(rows) == 1


def test_named_capture_records_the_served_rows_and_reproduces_the_nll(tmp_path):
    import types
    g = torch.Generator().manual_seed(3)
    V, P = 300, 12
    ref_logits = torch.randn(P, V, generator=g) * 2
    targets = torch.randint(0, V, (P,), generator=g).numpy()
    rows = KLM.reference_rows(ref_logits, targets)
    ref_path = str(tmp_path / "ref_conv1.npz")
    sha = KLM.save_artifact(ref_path, rows, {"source": "conv1"})
    fake = types.SimpleNamespace(torch=torch)            # stands in for step_decomp: its loop calls module.torch.log_softmax
    out = str(tmp_path / "named.npz")
    st = K8.named_capture(fake, ref_path, sha, out)
    eng_logits = ref_logits + 0.1 * torch.randn(P, V, generator=g)
    nll = 0.0
    for t in range(P):                                    # step_decomp's served loop, verbatim in shape
        lg = eng_logits[t:t + 1].float()
        nll += -fake.torch.log_softmax(lg, -1)[0, int(targets[t])].item()
    fake.torch.log_softmax(torch.zeros(3, V), -1)         # a non-[1, V] call is counted, not recorded
    st["save"]()
    z = __import__("numpy").load(out)
    meta = __import__("json").load(open(out + ".json"))
    assert meta["calls"] == P and meta["other_calls"] == 1 and meta["ref_sha"] == sha
    assert abs(-z["eng_target_lp"].mean() - nll / P) < 1e-9
    want = torch.log_softmax(eng_logits.double(), -1).gather(1, torch.as_tensor(rows["ids"]).long()).numpy()
    assert abs(z["eng_lp"] - want).max() < 1e-5          # fp32 log_softmax vs fp64
    r = KLM.window_read(rows, z["eng_lp"], z["eng_target_lp"])
    assert r["positions"] == P and r["kl65_mean"] > 0


def test_named_capture_refuses_a_wrong_reference_sha(tmp_path):
    import types
    rows = KLM.reference_rows(torch.randn(4, 100), [1, 2, 3, 4])
    ref_path = str(tmp_path / "ref.npz")
    KLM.save_artifact(ref_path, rows, {})
    with pytest.raises(SystemExit):
        K8.named_capture(types.SimpleNamespace(torch=torch), ref_path, "0" * 64, str(tmp_path / "o.npz"))


def test_sc1g_kl_and_ref_self_tests_pass():
    import subprocess
    import sys
    for f in ("sc1g_kl.py", "sc1g_ref.py"):
        r = subprocess.run([sys.executable, str(SC2 / f), "--self-test"], capture_output=True, text=True, timeout=600)
        assert r.returncode == 0 and "self-test OK" in r.stdout, f + r.stdout + r.stderr


def test_box_r_staging_pin_matches_its_sources():
    """bench/sc2/sc1g-r/staged-r.sha256 names every file box R stages; each must hash to its pin (make_pin.sh regenerates)."""
    import hashlib
    repo = SC2.parents[1]
    win = repo / "bench" / "h2h-2026-10-02" / "sc1g" / "receipts" / "sc1g-diag-2" / "sc1g"
    rdir = SC2 / "sc1g-r"
    names = []
    for ln in (rdir / "staged-r.sha256").read_text().splitlines():
        want, name = ln.split()
        names.append(name)
        if name.startswith("windows/"):
            src = win / name.split("/", 1)[1]
        elif name in ("sc1g_ref.py", "sc1g_kl.py"):
            src = SC2 / name
        elif name == "kl_fidelity.py":
            src = repo / "bench" / name
        else:
            src = rdir / name
        assert hashlib.sha256(src.read_bytes()).hexdigest() == want, name
    assert {"sc1g_r_run.sh", "sc1g_ref.py", "sc1g_kl.py", "kl_fidelity.py", "window_shas.json"} <= set(names)
    assert sum(n.startswith("windows/") for n in names) == 5
    run = (rdir / "sc1g_r_run.sh").read_text()
    assert "sha256sum -c staged-r.sha256" in run and "kl_fidelity.py --controls" in run and '"verdict": "R_OK"' in run
