# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane SC2e's box L (#846): the checks the box runs on a server's own /health, executed rather than grepped, and the
arm tables every file of the lane carries, held equal.

``slots_ok`` and ``burst_ok`` are Python heredocs inside ``bench/sc2/sc2e_box_l.sh``. These tests cut each heredoc out
and run it on /health fixtures:
* a server must report its arm's slots, bucket list and ``buckets_requested``, every bucket captured, and scratch slots
  sized by the largest bucket;
* after the burst, the arm's widest bucket must have replayed, no bucket may have run eagerly, and all 64 requests must
  be VALID with 512 prompt tokens.
"""
import importlib.util
import json
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SC2 = REPO / "bench" / "sc2"
BOX = (SC2 / "sc2e_box_l.sh").read_text()
RUN = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()
DRIVE = (REPO / "bench" / "sc1" / "sc1_drive.sh").read_text()
ARMS = {"s16": (16, [1, 2, 4, 8, 16], "default"), "s32a": (32, [1, 2, 4, 8, 16, 32], "auto"),
        "s64c": (64, [1, 2, 4, 8, 16], "default"), "s64a": (64, [1, 2, 4, 8, 16, 32, 64], "auto")}


def _heredoc(fn: str, tag: str) -> str:
    start = BOX.index(fn + "(){")
    body = BOX[BOX.index(f"<<'{tag}'", start):]
    body = body[body.index("\n") + 1:]
    return body[:body.index(f"\n{tag}\n")]


SLOTS_PY = _heredoc("slots_ok", "PYS")
BURST_PY = _heredoc("burst_ok", "PYB")


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, SC2 / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _run(script, args, tmp_path):
    p = tmp_path / "s.py"
    p.write_text(script)
    return subprocess.run([sys.executable, str(p), *args], capture_output=True, text=True, timeout=60)


def _start(tmp_path, arm, **over):
    m, b, req = ARMS[arm]
    e = {"max_seqs": m, "kv_slots": m, "buckets": b, "buckets_requested": req, "graph_status": {str(x): "graph" for x in b}}
    e.update(over.pop("engine", {}))
    h = {"engine": e, "levers": {"kv": {"scratch_slots": over.pop("scratch", b[-1]), "pool_mib": 1669.7}}}
    f = tmp_path / f"h_{arm}.json"
    f.write_text(json.dumps(h))
    return str(f)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_each_arm_reads_its_own_slots_and_buckets(tmp_path, arm):
    r = _run(SLOTS_PY, [_start(tmp_path, arm), arm], tmp_path)
    assert r.returncode == 0 and '"ok": true' in r.stdout, r.stdout + r.stderr


def test_slots_ok_stops_on_every_departure(tmp_path):
    cases = [("s32a", {"engine": {"buckets": [1, 2, 4, 8, 16]}}),                    # auto did not take: chained
             ("s32a", {"engine": {"buckets_requested": "default"}}),
             ("s64a", {"engine": {"graph_status": {**{str(x): "graph" for x in (1, 2, 4, 8, 16, 32)}, "64": "eager: OOM"}}}),
             ("s64a", {"engine": {"graph_status": {str(x): "graph" for x in (1, 2, 4, 8, 16, 32)}}}),   # 64 missing
             ("s64c", {"engine": {"kv_slots": 16}}),
             ("s64a", {"scratch": 16}),
             ("s16", {"engine": {"max_seqs": 32}})]
    for arm, over in cases:
        r = _run(SLOTS_PY, [_start(tmp_path, arm, **over), arm], tmp_path)
        assert r.returncode == 1 and '"ok": false' in r.stdout, (arm, over, r.stdout)


def _burst(tmp_path, arm, *, valid=64, n=64, pt=512, stats=None):
    b = ARMS[arm][1]
    gs = stats if stats is not None else {str(x): {"replays": 3, "eager_steps": 0, "rows": 9, "pad_rows": 1} for x in b}
    h = tmp_path / "hb.json"
    h.write_text(json.dumps({"engine": {"graph_stats": gs}}))
    j = tmp_path / "b.json"
    j.write_text(json.dumps({"requests": [{"valid": i < valid, "prompt_tokens": pt} for i in range(n)]}))
    return [str(h), str(j), arm]


def test_the_burst_must_replay_the_widest_bucket_with_no_eager_step(tmp_path):
    for arm in ARMS:
        r = _run(BURST_PY, _burst(tmp_path, arm), tmp_path)
        assert r.returncode == 0, (arm, r.stdout + r.stderr)
    top0 = {str(x): {"replays": 3, "eager_steps": 0, "rows": 3, "pad_rows": 0} for x in (1, 2, 4, 8, 16, 32)}
    top0["64"] = {"replays": 0, "eager_steps": 0, "rows": 0, "pad_rows": 0}
    eager = {str(x): {"replays": 3, "eager_steps": 0, "rows": 3, "pad_rows": 0} for x in (1, 2, 4, 8, 16)}
    eager["8"]["eager_steps"] = 2
    for args in (_burst(tmp_path, "s64a", stats=top0), _burst(tmp_path, "s64c", stats=eager),
                 _burst(tmp_path, "s32a", valid=63), _burst(tmp_path, "s16", n=63), _burst(tmp_path, "s16", pt=511),
                 _burst(tmp_path, "s16", stats={})):
        r = _run(BURST_PY, args, tmp_path)
        assert r.returncode == 1 and '"ok": false' in r.stdout, (args, r.stdout)


def test_the_burst_fills_the_widest_slot_count():
    """64 requests at once, each asking >= SC2_LO tokens: with SC2_LO >= the burst, the last admission (one prompt a
    step) finds every earlier request still decoding, so a 64-slot server runs a 64-row step."""
    lo = int(re.search(r"SC2_LO=(\d+)", (SC2 / "sc2_box_e.sh").read_text()).group(1))
    n = int(re.search(r"SC2E_BURST_N=(\d+)", BOX).group(1))
    rate = int(re.search(r"SC2E_BURST_RATE=(\d+)", BOX).group(1))
    assert n == max(m for m, _, _ in ARMS.values()) and lo >= n and rate >= 1000


def test_every_file_of_the_lane_carries_the_same_arms():
    from experts4bit_qlora import serve_recipe
    if not hasattr(serve_recipe, "default_buckets"):          # the box's tripwire refuses such an e4b (rc 9)
        pytest.skip("the installed e4b predates E4B_PAGED_BUCKETS=auto, the code under test")
    DEFAULT_BUCKETS, default_buckets = serve_recipe.DEFAULT_BUCKETS, serve_recipe.default_buckets
    red, cen, bas = _mod("sc2e_reduce"), _mod("sc2e_census"), _mod("sc2e_basis")
    for arm, (m, b, req) in ARMS.items():
        assert list(default_buckets(m) if req == "auto" else DEFAULT_BUCKETS) == b, arm      # what e4b captures
        assert red.SLOTS[arm] == (m, b, req) and list(cen.ARM_BUCKETS[arm]) == b and cen.ARM_SLOTS[arm] == m
        assert bas.ARMS[arm] == (m, tuple(b))
    assert re.search(r'l_slots\(\)\{ case "\$1" in s16\) echo 16;; s32a\) echo 32;; s64c\|s64a\) echo 64;;', BOX)
    assert re.search(r'l_buckets\(\)\{ case "\$1" in s32a\|s64a\) echo auto;; s16\|s64c\) echo "";;', BOX)
    assert red.RATES == (1, 2, 4, 8, 12, 16) and 'SC2E_RATES="1 2 4 8 12 16"' in BOX


@pytest.mark.parametrize("script", ["sc2e_reduce.py", "sc2e_census.py", "sc2e_basis.py"])
def test_the_lane_tools_pass_their_self_tests(script):
    r = subprocess.run([sys.executable, str(SC2 / script), "--self-test"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "self-test:" in r.stdout and "FAILED" not in r.stdout, r.stdout + r.stderr


def test_box_l_is_wired_like_box_h():
    assert 'case "$BOX" in A|B|C|D|E|F|G|H|I|J|K|L) ;;' in RUN
    assert 'case "$BOX" in L) GNF4_SHA=b4f93f1c62d1e3436ed45bec8ccd608c90433737;; esac' in RUN     # v0.42.0, CI's pin
    assert "C|D|E|F|G|H|I|J|K|L) BASEPY=python3;;" in RUN
    torch = next(x for x in RUN.splitlines() if '"torch==2.8.0"' in x and "pipx logs/pip_torch.log" in x)
    assert '[ "$BOX" = L ]' in torch                                     # SC2c A1's lesson: every python3 box pins torch
    assert RUN.count('os.environ["TRIP_BOX"] in ("F", "G", "H", "I", "J", "K", "L")') == 2   # main's routes, no pins
    assert "  L) . $W/sc2_box_e.sh; . $W/sc2c_box_h.sh; . $W/sc2e_box_l.sh; install_sc2_client ;;" in RUN
    assert 'L) PROVE_NEEDS="sc2client";;' in RUN and '[ "$BOX" = L ] && prove_l' in RUN and "L) box_l;; esac" in RUN
    assert 'case "$SC1_BOX" in A|B|C|D|E|F|G|H|I|J|K|L) ;;' in DRIVE
    assert "sc2e_box_l.sh sc2e_reduce.py sc2e_census.py sc2e_basis.py; do STAGE=" in DRIVE
    assert 'sc2c_*|sc2d_*|sc2e_*) src="$SC2/$name";;' in DRIVE
    pin = (REPO / "bench" / "sc1" / "staged.sha256").read_text()
    for name in ("sc2e_box_l.sh", "sc2e_reduce.py", "sc2e_census.py", "sc2e_basis.py"):
        assert re.search(rf"^[0-9a-f]{{64}}  {re.escape(name)}$", pin, re.M), name


def test_the_server_start_changes_only_the_two_knobs():
    start = BOX[BOX.index("l_server_start(){"):BOX.index("# l_burst")]
    assert "env -u E4B_INT4_PREFILL -u E4B_PAGED_PREFILL_ATTN -u E4B_PAGED_PREFILL_GRAPH -u E4B_PAGED_BULK_KV" in start
    assert "-u E4B_PAGED_GRAPHS -u E4B_PAGED_BUCKETS" in start
    assert "E4B_PAGED_MAX_SEQS=$M ${B:+E4B_PAGED_BUCKETS=$B}" in start
    assert "E4B_PAGED_STEP_TRACE=$W/sc2/steps_$TAG.jsonl" in start and "E4B_PAGED_TRACE=$W/sc2/trace_$TAG.jsonl" in start
    assert "E4B_PAGED_MAX_TOKENS_PER_SEQ=$SC2_MAXLEN" in start
    assert "return 47" in start and "return 49" in start
    assert '"$SPEEDENV E4B_PAGED_FUSE_QKV=1" "int4_k19|"; then' in BOX               # the reading: K19 taken
    assert '"$W/work_granite/nf4.arena" "$GR_ENV" ""; then' in BOX                  # the proof: any grouped route
    assert 'SC2E_ARMS_D1="s16 s32a s64c s64a"; SC2E_ARMS_D2="s64a s64c s32a s16"' in BOX
    assert '[ "$D" = 1 ] && [ "$ARM" = s16 ] && drive ${TAG}_serial_repeat' in BOX   # the determinism control
    assert "$(( D * 100 + r ))" in BOX                                              # SC2's seeds, paired across arms
