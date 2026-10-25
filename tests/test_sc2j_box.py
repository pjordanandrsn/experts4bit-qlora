# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane SC2j's box Q (#846): the POOL check the box runs, executed rather than grepped; the tripwire on this checkout;
the reducer's, the contract arm's and the long prompt pool's self-tests; the ladders, seeds, order, slots and tokens
every file of the lane carries, held equal; Guard Fit's static form for the proof and the reading (bench/common rule 5);
the registered cap, ids and bound (rules 2-4); box Q wired as box P is; and the controller's dry run.
"""
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SC2 = REPO / "bench" / "sc2"
G3KIT = REPO / "bench" / "kv-pool-g" / "g3-smoke"
BOX = (SC2 / "sc2j_box_q.sh").read_text()
RUN = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()
DRIVE = (REPO / "bench" / "sc1" / "sc1_drive.sh").read_text()
PREREG = (SC2 / "SC2j-PREREG.md").read_text()

# Guard Fit's static form (bench/common rule 5): boot + 1.25 x fit <= guard, boot >= ssh_ready_s 420 s + 120 s staging
BOOT_S = 540
GUARD = {"proof": 1.0 * 3600, "reading": 5.5 * 3600}
FIT_TOP = {"proof": 25 * 60, "reading": 231 * 60}      # the PREREG's top estimates (proof about 25 min; 192-231 min)


def _mod(name, *extra):
    for p in extra:
        sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(name, SC2 / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RED = _mod("sc2j_reduce")


def _selftest(script, *pythonpath):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(str(p) for p in pythonpath)}
    return subprocess.run([sys.executable, str(SC2 / script), "--self-test"], capture_output=True, text=True, timeout=600,
                          env=env)


def test_the_reducer_passes_its_self_test():
    r = _selftest("sc2j_reduce.py")
    assert r.returncode == 0 and "SC2J_REDUCE self-test PASS" in r.stdout, r.stdout + r.stderr


def test_the_contract_arm_passes_its_self_test():
    r = _selftest("sc2j_contract.py", G3KIT)
    assert r.returncode == 0 and "SC2J_CONTRACT self-test PASS" in r.stdout, r.stdout + r.stderr


def test_the_long_prompt_pool_passes_its_self_test():
    r = _selftest("sc2j_prompts.py")
    assert r.returncode == 0 and "SC2J_PROMPTS self-test PASS" in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("arm,width", [("ca_a", 32), ("sh", 64), ("ca_b", 32)])
def test_the_pool_check_passes_each_arm_as_registered(tmp_path, arm, width):
    h = tmp_path / "h.json"
    h.write_text(json.dumps(RED._health("sh" if arm == "sh" else "ca", width)))
    r = subprocess.run([sys.executable, str(SC2 / "sc2j_reduce.py"), "--pool", str(h), arm], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0 and '"ok": true' in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("arm,mut", [
    ("sh", lambda h: h["engine"].update(kv_pool="claim-all")),
    ("sh", lambda h: h["engine"].update(kv_pool_sizing=None)),
    ("ca_a", lambda h: h["engine"].update(kv_pool="shared")),
    ("ca_a", lambda h: h["engine"].update(buckets=[1, 2, 4, 8, 16])),           # not auto's list for 32
    ("sh", lambda h: h["levers"]["kv"].update(scratch_slots=16)),
    ("sh", lambda h: h["engine"]["graph_status"].update({"64": "eager"})),
])
def test_the_pool_check_stops_on_every_departure(tmp_path, arm, mut):
    h = RED._health("sh" if arm == "sh" else "ca", 64 if arm == "sh" else 32)
    mut(h)
    p = tmp_path / "h.json"
    p.write_text(json.dumps(h))
    r = subprocess.run([sys.executable, str(SC2 / "sc2j_reduce.py"), "--pool", str(p), arm], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 1 and '"ok": false' in r.stdout, r.stdout


def test_the_tripwire_passes_on_this_checkout(tmp_path):
    pytest.importorskip("torch")
    body = BOX[BOX.index('q_tripwire(){ "$PY" -c "') + len('q_tripwire(){ "$PY" -c "'):BOX.index('" \\\n    2>&1 | tail -1 | sed')]
    p = tmp_path / "trip.py"
    p.write_text(body)
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    r = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, timeout=300, cwd=REPO, env=env)
    assert r.returncode == 0 and "levers G1-G3 present" in r.stdout, r.stdout + r.stderr


def test_every_file_of_the_lane_carries_the_same_ladders_seeds_order_and_slots():
    assert RED.Q2_RATES == (1, 2, 4, 8, 12, 16) and 'SC2J_Q2="1 2 4 8 12 16"; SC2J_Q2_N=120; SC2J_SERIAL_N=24' in BOX
    assert "**1, 2, 4, 8, 12 and 16 req/s**, 120 requests each" in PREREG
    assert RED.Q3_RATES == (0.5, 1, 2, 4) and 'SC2J_Q3="0.5 1 2 4"; SC2J_Q3_N=96; SC2J_Q3_LO=768; SC2J_Q3_HI=1536' in BOX
    assert "Poisson at **0.5, 1, 2 and 4 req/s**, **96 requests each**" in PREREG and "`max_tokens` U[768, 1,536]" in PREREG
    assert RED.SLOT_ARMS == ("ca_a", "sh", "ca_b") and 'SC2J_ORDER="ca_a sh ca_b"' in BOX
    assert "Draw 1: `ca_a sh ca_b`; draw 2: `ca_a sh ca_b`" in PREREG
    assert "$(( D * 100 + r ))" in BOX and "seed draw × 100 + rate" in PREREG
    assert "D * 1000 + $(awk -v r=\"$r\" 'BEGIN{printf \"%d\", 10 * r}')" in BOX and "seed draw\n    × 1,000 + 10 × rate" in PREREG
    assert "SC2J_TOKENS=4096; SC2J_CONTRACT_SLOTS=32" in BOX and "E4B_PAGED_MAX_TOKENS_PER_SEQ=4096" in PREREG
    assert RED.Q3_PROMPT == 2048 and RED.T == 512
    assert (RED.ATT_TOL_IN, RED.ATT_TOL_REGRESS, RED.Q3_TOL_MIN, RED.TOKS_BAR) == (0.02, 0.05, 0.05, 1.25)
    assert "≥ 1.25 × `ca`'s at `ca`'s ceiling" in PREREG and "lower `ca` server's − 0.02" in PREREG


def test_the_server_start_changes_only_the_pool():
    start = BOX[BOX.index("q_server_start(){"):BOX.index("# q_burst TAG MODEL")]
    assert "E4B_PAGED_MAX_SEQS=auto E4B_PAGED_KV_POOL=$P" in start and "E4B_PAGED_MAX_TOKENS_PER_SEQ=$SC2J_TOKENS" in start
    assert "-u E4B_PAGED_BUCKETS -u E4B_PAGED_MAX_SEQS -u E4B_PAGED_KV_POOL" in start      # buckets auto, the default
    arm = BOX[BOX.index("q_arm(){"):BOX.index("# q_contract MODEL")]
    assert 'q_server_start "$TAG" "$S" "$MID" "$REV" "$QA" "$SPEEDENV E4B_PAGED_FUSE_QKV=1" "int4_k19|"' in arm
    assert "\nunset E4B_PAGED_KV_POOL E4B_PAGED_MAX_SEQS E4B_PAGED_BUCKETS E4B_PAGED_DECODE_LOOKAHEAD " in BOX
    spd = "".join(re.search(rf'^{v}="([^"]*)"', RUN, re.M).group(1) for v in ("SPEEDENV", "FOLDS", "ROUTEENV"))
    assert "KV_POOL" not in spd and "MAX_SEQS" not in spd and "BUCKETS" not in spd
    assert "l_tripwire; n_tripwire; q_tripwire" in BOX


def test_guard_fit_static_form_holds_for_the_proof_and_the_reading():
    for kind in ("proof", "reading"):
        assert BOOT_S + 1.25 * FIT_TOP[kind] <= GUARD[kind], kind
    assert "(`sc2j-prove-*`, guard 1.0 h)" in PREREG and "(`sc2j-5090-*`, guard 5.5 h)" in PREREG
    assert "9 + 1.25 × 231 = 298 min" in PREREG and "9 + 1.25 × 25 = 40 min" in PREREG
    assert "The fit is therefore about 192-231 min" in PREREG


def test_the_cap_ids_and_bound_are_registered():
    assert "**4** boxes may START: at most 2 proofs and at most 2 readings" in PREREG
    assert "`sc2j-prove-1` and `sc2j-prove-2`; readings `sc2j-5090-1` and `sc2j-5090-2`" in PREREG
    assert "`sc2j-r1` to `sc2j-r8`, shared across both kinds" in PREREG
    assert abs(2 * 1.15 + 2 * 6.61 + 8 * 0.72 - 21.28) < 1e-9 and "**$21.28**" in PREREG
    assert "**Two strikes** stop the lane" in PREREG


def test_box_q_is_wired_like_box_p():
    assert 'case "$BOX" in A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q) ;;' in RUN
    assert 'case "$BOX" in P) GNF4_SHA=30adb1bfa76f95f9ce4a4c29734b8f36de907eca;; Q) GNF4_SHA=41ef086e6a3e2be5b158ff1bb38e0de67862a949;; esac' in RUN      # Q stays on v0.50.0, SC2j's registered pin (SC2f Amendment 7 moved P alone)
    torch = next(x for x in RUN.splitlines() if '"torch==2.8.0"' in x and "pipx logs/pip_torch.log" in x)
    assert '[ "$BOX" = Q ]' in torch
    assert RUN.count('os.environ["TRIP_BOX"] in ("F", "G", "H", "I", "J", "K", "L", "N", "O", "P", "Q")') == 2
    assert ("  Q) . $W/sc2_box_e.sh; . $W/sc2c_box_h.sh; . $W/sc2e_box_l.sh; . $W/sc2f_box_n.sh; . $W/sc2j_box_q.sh; "
            "install_sc2_client ;;") in RUN
    assert 'Q) PROVE_NEEDS="sc2client";;' in RUN and '[ "$BOX" = Q ] && prove_q' in RUN and "Q) box_q;; esac" in RUN
    assert 'case "$SC1_BOX" in A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q) ;;' in DRIVE
    assert "sc2j_box_q.sh sc2j_reduce.py sc2j_contract.py sc2j_prompts.py sc2j_quality.py; do STAGE=" in DRIVE   # Am. 1
    assert '$REPO/bench/p117/p117_reduce.py $TESTS/test_decode_graph_buckets.py $REPO/bench/kv-pool-g/g3-smoke/g3_smoke.py' in DRIVE
    pin = (REPO / "bench" / "sc1" / "staged.sha256").read_text()
    for name in ("sc2j_box_q.sh", "sc2j_reduce.py", "sc2j_contract.py", "sc2j_prompts.py", "sc2j_quality.py", "p117_reduce.py",
                 "test_decode_graph_buckets.py", "g3_smoke.py"):
        assert re.search(rf"^[0-9a-f]{{64}}  {re.escape(name)}$", pin, re.M), name


def test_the_box_script_parses():
    assert subprocess.run(["bash", "-n", str(SC2 / "sc2j_box_q.sh")]).returncode == 0
    assert "bc -l" not in BOX                       # the CUDA image ships no bc: rates are computed with awk


def _dry_env(tmp_path, **extra):
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(tmp_path), "E4B_RENT_SSH_HOST": "h",
           "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-o UserKnownHostsFile=/run/known_hosts",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "sc2j-dry", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "0", "E4B_SHA": "0" * 40, "SC1_DRIVE_DRYRUN": "1", "SC1_BOX": "Q"}
    env.update(extra)
    return env


@pytest.mark.parametrize("prove", ["0", "1"])
def test_the_controller_dry_runs_box_q(tmp_path, prove):
    out = subprocess.run(["bash", str(REPO / "bench" / "sc1" / "sc1_drive.sh")], capture_output=True, text=True,
                         env=_dry_env(tmp_path, SC1_PROVE=prove))
    assert out.returncode == 0, out.stdout + out.stderr
    for name in ("sc2j_box_q.sh", "sc2j_reduce.py", "sc2j_contract.py", "sc2j_prompts.py", "g3_smoke.py", "p117_reduce.py",
                 "sc2j_quality.py"):
        assert name in out.stdout, name
    assert " SC1_BOX=Q " in out.stdout


FIX = REPO / "tests" / "fixtures" / "sc2i_shake1"


def test_engaged_passes_a_real_engaged_end_record():
    """A real end /health of an engaged 64-slot server (SC2i's shakedown, base arm: 90 prompts, no Q3) passes the
    fail-closed ENGAGED check; the same record's lc128 sibling, whose short first chunks ran eagerly, does not."""
    base = json.loads((FIX / "health_prove_base_end.json").read_text())
    assert RED.engagement_bad("prove_base", base, 90, 0) == []
    lc = json.loads((FIX / "health_prove_lc128_end.json").read_text())
    why = RED.engagement_bad("prove_lc128", lc, 90, 0)
    assert any("short_chunk" in x for x in why), why


@pytest.mark.parametrize("path,how", [
    (("engine", "graph_stats", "8"), "del"), (("engine", "graph_stats", "2", "eager_steps"), "del"),
    (("kv_bookkeeping", "flush_layers"), "del"), (("prefill_graph", "eager_reasons"), "del"),
    (("prefill_graph", "eager_chunks"), "del"), (("prefill_graph", "eager_reasons"), "empty"),
    (("engine", "graph_stats", "64"), "null"), (("kv_bookkeeping",), "null"),
])
def test_engaged_fails_closed_on_the_real_record(path, how):
    h = json.loads((FIX / "health_prove_base_end.json").read_text())
    node = h
    for k in path[:-1]:
        node = node[k]
    if how == "del":
        del node[path[-1]]
    elif how == "empty":
        node[path[-1]] = {}
    else:
        node[path[-1]] = None
    assert RED.engagement_bad("prove_base", h, 90, 0), (path, how)


# ---- Amendment 1: QUALITY reduced by P117's reducer and bar at box Q's two stack constants (sc2j_quality.py)
QREC = REPO / "tests" / "fixtures" / "sc2j_r3_quality"     # sc2j-r3's QUALITY record (the proof box, Granite, 8 windows)
QUAL = _mod("sc2j_quality")
# box Q's gnf4 pin in sc1_run.sh: "P|Q) GNF4_SHA=..." until SC2f Amendment 7 moved box P alone, then "Q) GNF4_SHA=..."
GNF4_Q = re.search(r'^case "\$BOX" in .*?\bQ\) GNF4_SHA=([0-9a-f]{40})', RUN, re.M).group(1)


def test_quality_wiring_exports_box_qs_gnf4_and_uses_the_wrapper():
    body = BOX[BOX.index("q_quality(){"):BOX.index("\n}\n", BOX.index("q_quality(){"))]
    assert "GNF4_SHA=$GNF4_SHA E4B_PAGED_MODEL=$MODEL" in body          # p117_box records os.environ["GNF4_SHA"]
    assert '$W/sc2j_quality.py --dir $QD --out $QD/verdict.json --e4b-sha "$E4B_SHA" --gnf4-sha "$GNF4_SHA"' in body
    assert "p117_reduce.py --dir" not in body                             # P117's own constants would VOID box Q
    assert "p117_reduce.py sc2j_quality.py; do" in BOX                   # the proof runs its self-test


def test_quality_constants_are_box_qs():
    assert QUAL.GNF4_SHA == GNF4_Q
    assert QUAL.WINDOWS == {"Qwen/Qwen3-30B-A3B": 64, "ibm-granite/granite-3.1-3b-a800m-instruct": 8}
    assert re.search(r"q_quality .* 64 128\n", BOX) and re.search(r"q_quality .* 8 32\n", BOX)
    assert QUAL.P117_GNF4_SHA != QUAL.GNF4_SHA                           # the reason the wrapper exists


def _qrun(tmp_path, box, gnf4):
    (tmp_path / "summary.txt").write_text((QREC / "summary.txt").read_text())
    (tmp_path / "box.json").write_text(json.dumps(box))
    return QUAL.reduce(tmp_path, box["e4b_sha"], gnf4)


def test_the_proof_record_read_void_under_p117s_constants():
    """What sc2j-r3 recorded: P117's reducer VOIDed it on its own stack pin and window count."""
    v = json.loads((QREC / "verdict.json").read_text())
    assert v["verdict"] == "VOID" and any("b4f93f1c" in r for r in v["reasons"]) and any("expected 40" in r for r in v["reasons"])


def test_the_proof_record_reads_at_box_qs_constants(tmp_path):
    """The same record, with the GNF4_SHA the box now exports, reads under P117's unchanged bar."""
    box = json.loads((QREC / "box.json").read_text())
    assert box["gnf4_sha"] is None and box["windows"] == 8               # as recorded: nothing exported GNF4_SHA
    box["gnf4_sha"] = GNF4_Q
    v = _qrun(tmp_path, box, GNF4_Q)
    assert v["verdict"] == "AT_PARITY", v.get("reasons")
    assert v["function"] == {"positions": 8 * (box["cont"] - 1), "differ": 0}


@pytest.mark.parametrize("edit", ["record_none", "record_p117", "installed_other", "windows_40"])
def test_the_proof_record_fails_closed(tmp_path, edit):
    box = json.loads((QREC / "box.json").read_text())
    box["gnf4_sha"], gnf4 = GNF4_Q, GNF4_Q
    if edit == "record_none":
        box["gnf4_sha"] = None
    elif edit == "record_p117":
        box["gnf4_sha"] = QUAL.P117_GNF4_SHA
    elif edit == "installed_other":
        gnf4 = "0" * 40
    else:
        box["windows"] = 40
    assert _qrun(tmp_path, box, gnf4)["verdict"] == "VOID", edit


def test_quality_self_test_passes():
    r = subprocess.run([sys.executable, str(SC2 / "sc2j_quality.py"), "--self-test"], capture_output=True, text=True)
    assert r.returncode == 0 and "SC2J_QUALITY self-test PASS (11/11)" in r.stdout, r.stdout + r.stderr
