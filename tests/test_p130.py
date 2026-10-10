"""Lane P130's instrument on CPU (bench/p130/PREREG-p130.md; e4b#1313): speculative decoding, Phase 0.

- The EAGLE-3 draft's batched chains equal a step-by-step reference at every index (fp32 and bf16).
- The reducer self-tests, and reproduces the registered n-gram floor from p127-5090-1's committed receipts.
- The box's capture bookkeeping self-tests.
- The staged pin matches the repo, as bench/p130/p130_drive.sh checks before it stages anything.
- The runner and driver keep their registered shape, and the PREREG carries every pinned value.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p130"
sys.path.insert(0, str(LANE))
import p130_eagle3 as e3  # noqa: E402
import p130_reduce as rd  # noqa: E402

RUN = (LANE / "p130_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p130_drive.sh").read_text(encoding="utf-8")
PREREG = (LANE / "PREREG-p130.md").read_text(encoding="utf-8")
SOURCES = {
    "p130_run.sh": LANE / "p130_run.sh", "p130_box.py": LANE / "p130_box.py", "p130_eagle3.py": LANE / "p130_eagle3.py",
    "p130_reduce.py": LANE / "p130_reduce.py", "chat_prompts.json": LANE / "chat_prompts.json",
    "expect_w1.json": LANE / "expect_w1.json", "p109_box.py": REPO / "bench" / "p109" / "p109_box.py",
    "k8_bake.py": REPO / "bench" / "p39" / "k8_bake.py", "calib.json": REPO / "bench" / "p39" / "calib.json",
}
P127R = REPO / "bench" / "p127" / "receipts" / "p127-5090-1"


def _head(seed=0, H=16, NH=4, HD=8, NKV=2, V=50, DV=20, I=24):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    return {"d2t": torch.randint(0, V - DV, (DV,), generator=g), "embed_tokens.weight": r(V, H), "fc.weight": r(H, 3 * H) / 4,
            "layers.0.hidden_norm.weight": 1 + 0.1 * r(H), "layers.0.input_layernorm.weight": 1 + 0.1 * r(H),
            "layers.0.mlp.down_proj.weight": r(H, I) / 4, "layers.0.mlp.gate_proj.weight": r(I, H) / 4,
            "layers.0.mlp.up_proj.weight": r(I, H) / 4, "layers.0.post_attention_layernorm.weight": 1 + 0.1 * r(H),
            "layers.0.self_attn.k_proj.weight": r(NKV * HD, 2 * H) / 4, "layers.0.self_attn.o_proj.weight": r(H, NH * HD) / 4,
            "layers.0.self_attn.q_proj.weight": r(NH * HD, 2 * H) / 4, "layers.0.self_attn.v_proj.weight": r(NKV * HD, 2 * H) / 4,
            "lm_head.weight": r(DV, H), "norm.weight": 1 + 0.1 * r(H)}, (H, NH, HD, NKV, V)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batched_chains_equal_the_step_by_step_reference(dtype):
    t, (H, NH, HD, NKV, V) = _head()
    d = e3.Eagle3Draft(t, n_heads=NH, n_kv=NKV, head_dim=HD, dtype=dtype)
    g = torch.Generator().manual_seed(1)
    L = 24
    tokens = torch.randint(0, V, (L,), generator=g)
    aux = torch.randn(L - 1, 3 * H, generator=g)
    ch = d.chains(tokens, aux, K=5)
    assert tuple(ch.shape) == (L - 1, 5)
    for tt in range(L - 1):
        assert ch[tt].tolist() == d.chain_at(tokens, aux, tt, K=5), tt


def test_draft_ids_map_through_d2t_and_a_missing_tensor_is_refused():
    t, (H, NH, HD, NKV, V) = _head()
    d = e3.Eagle3Draft(t, n_heads=NH, n_kv=NKV, head_dim=HD, dtype=torch.float32)
    ch = d.chains(torch.arange(10) % V, torch.randn(9, 3 * H), K=3)
    targets = set((torch.arange(20) + t["d2t"]).tolist())
    assert set(ch.flatten().tolist()) <= targets, "every drafted id is a mapped target id"
    bad = dict(t)
    bad.pop("fc.weight")
    with pytest.raises(ValueError, match="fc.weight"):
        e3.Eagle3Draft(bad)
    assert e3.AUX_LAYERS == (2, 24, 45)


def test_the_reducer_self_tests():
    p = subprocess.run([sys.executable, str(LANE / "p130_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert p.returncode == 0 and "self-test OK" in p.stdout.splitlines()[-1], p.stdout


def test_the_registered_ngram_floor_reproduces_from_the_p127_receipts():
    """The PREREG's n-gram table, recomputed from p127-5090-1's committed prompts and W16 tokens, to 4 decimals."""
    prompts = json.load(open(P127R / "prompts.json"))["rows"]
    gen = json.load(open(P127R / "arm_A1.json"))["workloads"]["W16"]["tokens"]["160"]
    rows = [{"tokens": list(p) + list(g), "prompt_len": len(p)} for p, g in zip(prompts, gen)]
    want = {1: (1.2201, 1.0847, 1.1296), 2: (1.3033, 1.0723, 1.1411), 3: (1.3496, 1.0450, 1.1268),
            4: (1.3759, 1.0109, 1.0994), 5: (1.3917, 0.9754, 1.0662)}
    for k, (tau, s_ind, s_reu) in want.items():
        a = rd.simulate(rows, k, "ngram", rd.d_ind)
        b = rd.simulate(rows, k, "ngram", rd.d_reuse)
        assert (round(a["tau"], 4), round(a["S"], 4), round(b["S"], 4)) == (tau, s_ind, s_reu), k
        assert f"| {k} | {tau:.4f} |" in PREREG


def test_measured_dn_replaces_only_the_reuse_arm():
    seq = list(range(10)) + [7, 8, 9] * 10
    caps = {w: {"rows": [{"tokens": seq, "prompt_len": 10, "chains": [seq[t + 2:t + 7] + [0] * 5 for t in range(len(seq) - 1)]}]}
            for w in ("R", "C-think", "C-nothink")}
    a = rd.reduce(caps)
    b = rd.reduce(caps, {"2": 8.0, "3": 8.0, "4": 8.0, "5": 8.0, "6": 8.0})
    for w in caps:
        for k in ("1", "3"):
            assert a["routes"]["eagle3"][w][k]["S_independent"] == b["routes"]["eagle3"][w][k]["S_independent"]
            assert b["routes"]["eagle3"][w][k]["S_reuse"] >= a["routes"]["eagle3"][w][k]["S_reuse"]
    assert a["verdict"] == b["verdict"], "independence decides"


def test_the_box_self_tests():
    p = subprocess.run([sys.executable, str(LANE / "p130_box.py"), "--self-test"], capture_output=True, text=True)
    assert p.returncode == 0 and "self-test OK" in p.stdout, p.stdout + p.stderr


def test_the_staged_pin_matches_the_repo():
    pin = [l.split() for l in (LANE / "staged.sha256").read_text().splitlines() if l and not l.startswith("#")]
    names = [n for _, n in pin]
    assert sorted(names) == sorted(SOURCES), names
    for want, name in pin:
        assert hashlib.sha256(SOURCES[name].read_bytes()).hexdigest() == want, name
    stage = re.search(r'STAGE="([^"]+)"', DRIVE).group(1).split()
    assert sorted(pathlib.PurePosixPath(s).name for s in stage) == sorted(names + ["staged.sha256"])


def test_the_runner_keeps_its_registered_shape():
    assert "E4B_T=7f044dd9570b0f75ac4ebc1742e697256cd2e209 " in RUN and "GNF4_T=d769d5022c0fb7a2ada847f69a3cba6e0f45c77f " in RUN
    assert "REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39" in RUN and "E4B_H=$E4B_SHA " in RUN
    assert "HEAD_REV=6afc5aa2477b923467fb9a8d906782b984a9a6ba" in RUN
    assert "HEAD_SHA256=d2d6e2e63e09dc755053ae5c98cdececae3611ae5e202d4fa5411126dd3b1dfa; HEAD_BYTES=1044539336" in RUN
    assert re.search(r'ENGINE_ENV="[^"]*E4B_PAGED_MAX_SEQS=16 E4B_INT4_TILE_PROGRAMS=1 E4B_PAGED_GRAPHS=0 E4B_PAGED_PREFILL_GRAPH=0"', RUN)
    assert "for WL in R C-think C-nothink; do" in RUN and "--expect-w1 $W/expect_w1.json" in RUN
    assert "WATCHDOG=$W/src/e4b_H/bench/common/hf_fetch_watchdog.py" in RUN and "snapshot_download" not in RUN
    order = [RUN.index(s) for s in ("STAGED FILES DIFFER", "TRIPWIRE FAIL", "REDUCER SELF-TEST FAILED", 'say "fetch $MODEL',
                                    "HEAD MISMATCH", "BAKE FAIL", "PROMPTS FAIL (R)", "for WL in", 'say "reduce"')]
    assert order == sorted(order)


def test_the_driver_uses_the_shared_liveness_and_token_scope():
    assert 'LANE_HELPER="$REPO/bench/common/lane_liveness.sh"' in DRIVE and "lane_snapshot_verdict" in DRIVE
    assert "lane_write_host_fault" in DRIVE and "token_scope.py" in DRIVE
    assert "pgrep" not in DRIVE, "liveness by PID identity, never a pattern that can match its own probe"
    assert "P130_RUN_NONCE" in DRIVE and "bash p130_run.sh" in DRIVE


def test_the_pinned_inputs_carry_their_provenance():
    cp = json.load(open(LANE / "chat_prompts.json", encoding="ascii"))
    assert cp["source"]["revision"] == "8049631c405ae6576f93f445c6b8166f76f5505a" and len(cp["prompts"]) == 16
    assert hashlib.sha256(json.dumps([p["prompt"] for p in cp["prompts"]], ensure_ascii=True).encode()).hexdigest() == cp["prompts_sha256"]
    ex = json.load(open(LANE / "expect_w1.json"))
    w1 = json.load(open(P127R / "arm_B1.json"))["workloads"]["W1"]["tokens"]["160"][0]
    assert ex["tokens"] == w1 and ex["prompts_sha256"] == json.load(open(P127R / "prompts.json"))["prompts_sha256"]
    for v in ("6afc5aa2477b923467fb9a8d906782b984a9a6ba", "746d2d4cbcd0682b4818bc8aa65abb16410cb8fa556e00326e644cf9b391461a",
              "92a7a4fa8f0467960cb37ab837629814b79cd17ef28b7a8e969dfda03af0da4b",
              "d2d6e2e63e09dc755053ae5c98cdececae3611ae5e202d4fa5411126dd3b1dfa", "8049631c", "**1.40**", "**1.15**", "**1.05**"):
        assert v in PREREG, v
