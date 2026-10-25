# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane SC2h's box O (#846): the LOOKAHEAD check and the knee rule the box runs, executed rather than grepped; the
readings held equal to the code that produces them; the ladder, seeds and order every file of the lane carries, held
equal; and box O wired as box N is.

Box O stops a server (rc 44) unless its start /health reads its arm: ``engine.decode_lookahead`` true on ``la`` and false
on ``base``, and SC2f's ``on`` defaults on both (P130's two prefill defaults engaged). The knee rates are SC2f's reading
run through the registered rule; the box refuses to launch (rc 45) when the rule does.
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
BOX = (SC2 / "sc2h_box_o.sh").read_text()
RUN = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()
DRIVE = (REPO / "bench" / "sc1" / "sc1_drive.sh").read_text()


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, SC2 / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RED = _mod("sc2h_reduce")
FRED = _mod("sc2f_reduce")


def _health(arm, la=None, **over):
    w = FRED.WANT_FLIPS["on"]
    seen = {"moe_k19_dispatch": {"chained|lean|gt256": 240, "one_launch|lean|gt256": 144, "one_launch|lean|le256": 720},
            "prefill_folds": {f: 48 for f in FRED.FOLDS}}
    pr = {"prefill_glue_env": None, "prefill_lean_dispatch_env": None, "prefill_lean_dispatch": "auto", "seen": seen}
    pr.update(over.pop("routes", {}))
    return {"engine": {"decode_lookahead": (arm == "la") if la is None else la}, "prefill_routes": pr,
            "levers": {"prefill_glue": over.pop("glue", dict(w["glue"]))}}


def _run(tmp_path, *args):
    return subprocess.run([sys.executable, str(SC2 / "sc2h_reduce.py"), *args], capture_output=True, text=True, timeout=120)


def _check(tmp_path, health, arm, *extra):
    p = tmp_path / f"h_{arm}.json"
    p.write_text(json.dumps(health))
    return _run(tmp_path, "--lookahead", str(p), arm, *extra)


@pytest.mark.parametrize("arm", ["la", "base"])
def test_each_arm_reads_its_own_lookahead(tmp_path, arm):
    r = _check(tmp_path, _health(arm), arm)
    assert r.returncode == 0 and '"ok": true' in r.stdout, r.stdout + r.stderr


def test_the_lookahead_check_stops_on_every_departure(tmp_path):
    cases = [("la", {"la": False}), ("base", {"la": True}), ("la", {"la": None}),
             ("base", {"glue": {"mode": "0", "source": "explicit"}}),                      # off P130's P1 default
             ("la", {"routes": {"prefill_lean_dispatch": "0"}})]                          # off P130's P2 default
    for arm, over in cases:
        h = _health(arm, **{k: v for k, v in over.items() if k != "la"})
        if "la" in over:
            h["engine"]["decode_lookahead"] = over["la"]
        r = _check(tmp_path, h, arm)
        assert r.returncode == 1 and '"ok": false' in r.stdout, (arm, over, r.stdout)
    assert _check(tmp_path, {}, "la").returncode == 1
    assert _check(tmp_path, _health("la"), "neither").returncode == 1


def test_the_knee_rule_writes_the_rates_and_refuses_an_unread_sc2f(tmp_path):
    v = RED._sc2f_verdict({4: 1.0, 8: 1.0, 12: 0.99, 13: 0.97, 14: 0.9, 16: 0.6})
    p = tmp_path / "verdict_sc2f.json"
    p.write_text(json.dumps(v))
    out = tmp_path / "run"
    out.mkdir()
    r = _run(tmp_path, "--knee-rule", str(p), "--write", str(out))
    assert r.returncode == 0 and "SC2H_KNEE" in r.stdout, r.stdout + r.stderr
    kr = json.loads((out / "knee_rates.json").read_text())
    assert kr["rates"] == [13, 14, 15] and (out / "sc2f_verdict.json").read_bytes() == p.read_bytes()
    assert RED.knee_bad(str(out)) == ((13, 14, 15), None)
    p.write_text(json.dumps(RED._sc2f_verdict({r_: 0.99 for r_ in RED.SC2F_LADDER}, verdict="UNREAD")))
    assert _run(tmp_path, "--knee-rule", str(p)).returncode == 1
    assert _run(tmp_path, "--knee-rule", str(tmp_path / "missing.json")).returncode == 1


def test_the_steps_check_reads_the_lookahead_notes(tmp_path):
    rows = [{"decode_rows": 60, "bucket": 64, "dec_pieces": 1, "step_ms": 14.0, "lookahead": 1} for _ in range(3)]
    (tmp_path / "la.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    assert _run(tmp_path, "--steps", str(tmp_path / "la.jsonl"), "la").returncode == 0
    assert _run(tmp_path, "--steps", str(tmp_path / "la.jsonl"), "base").returncode == 1
    for x in rows:
        x.pop("lookahead")
    (tmp_path / "base.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    assert _run(tmp_path, "--steps", str(tmp_path / "base.jsonl"), "base").returncode == 0
    assert _run(tmp_path, "--steps", str(tmp_path / "base.jsonl"), "la").returncode == 1


def test_the_readings_are_what_the_code_reports():
    sp = (REPO / "experts4bit_qlora" / "serve_paged.py").read_text(encoding="utf-8")
    assert '"decode_lookahead": cfg.decode_lookahead,' in sp                       # /health's engine block
    assert 'decode_lookahead=_lookahead_env(env("E4B_PAGED_DECODE_LOOKAHEAD", "0"))' in sp
    pr = (REPO / "experts4bit_qlora" / "engines" / "paged_runner.py").read_text(encoding="utf-8")
    assert "tr.note(bucket=b, lookahead=1)" in pr                                 # every issued decode step's note
    assert pr.count("stats[\"replays\"] += 1") >= 2                              # issue_decode counts its replays too


def test_the_tripwire_passes_on_this_checkout(tmp_path):
    pytest.importorskip("torch")
    body = BOX[BOX.index('o_tripwire(){ "$PY" -c "') + len('o_tripwire(){ "$PY" -c "'):BOX.index('" \\\n    2>&1 | tail -1 | sed')]
    p = tmp_path / "trip.py"
    p.write_text(body)
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    r = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, timeout=300, cwd=REPO, env=env)
    assert r.returncode == 0 and "P118 lookahead present and off by default" in r.stdout, r.stdout + r.stderr


def test_every_file_of_the_lane_carries_the_same_ladder_seeds_and_order():
    assert RED.SC2F_LADDER == FRED.LADDER == (4, 8, 12, 13, 14, 16) and 'SC2H_LADDER0="4 8 12 13 14 16"' in BOX
    assert RED.KNEE_J == FRED.KNEE_J == (1, 2, 3, 4) and 'SC2H_KNEE_J="1 2 3 4"' in BOX
    assert "$(( D * 100 + r ))" in BOX and "$(( 10000 * D + 100 * r + j ))" in BOX
    assert f'SC2H_ORDER_D1="{" ".join(RED.ORDER[1])}"; SC2H_ORDER_D2="{" ".join(RED.ORDER[2])}"' in BOX
    assert 'SC2H_N=120; SC2H_SERIAL_N=24' in BOX and RED.SLOT_ARM == "s64a"
    assert RED.servers()[0] == ("e4b_la_d1a", "la", 1) and RED.servers()[4] == ("e4b_base_d2a", "base", 2)


def test_the_server_start_changes_only_the_lookahead():
    arm = BOX[BOX.index("o_arm(){"):BOX.index("# ---- the real lane")]
    assert 'LEV="$SPEEDENV E4B_PAGED_FUSE_QKV=1"; [ "$ARM" = la ] && LEV="$LEV $SC2H_LA"' in arm
    assert 'SC2H_LA="E4B_PAGED_DECODE_LOOKAHEAD=1"' in BOX and "\nunset E4B_PAGED_DECODE_LOOKAHEAD " in BOX
    assert 'l_server_start "$TAG" s64a "$MID" "$REV" "$QA" "$LEV" "int4_k19|" && o_la_ok "$TAG" "$ARM"' in arm
    assert 'h_early "$TAG" on $W/sc2/${TAG}_warm.json && l_burst "$TAG" s64a "$MID"' in arm
    assert "return 44" in BOX and "finish 45" in BOX and "l_tripwire; n_tripwire; o_tripwire; o_knee" in BOX
    spd = "".join(re.search(rf'^{v}="([^"]*)"', RUN, re.M).group(1) for v in ("SPEEDENV", "FOLDS", "ROUTEENV"))
    assert "LOOKAHEAD" not in spd and "PREFILL_GLUE" not in spd and "LEAN_DISPATCH" not in spd


@pytest.mark.parametrize("script", ["sc2h_reduce.py", "sc2h_basis.py"])
def test_the_lane_tools_pass_their_self_tests(script):
    r = subprocess.run([sys.executable, str(SC2 / script), "--self-test"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "self-test:" in r.stdout and "FAILED" not in r.stdout, r.stdout + r.stderr


def test_box_o_is_wired_like_box_n():
    assert 'case "$BOX" in A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q) ;;' in RUN
    assert 'case "$BOX" in N|O) GNF4_SHA=30adb1bfa76f95f9ce4a4c29734b8f36de907eca;; esac' in RUN      # v0.51.0 (SC2h Amendment 5; v0.50.0 under Amendment 1)
    torch = next(x for x in RUN.splitlines() if '"torch==2.8.0"' in x and "pipx logs/pip_torch.log" in x)
    assert '[ "$BOX" = O ]' in torch
    assert RUN.count('os.environ["TRIP_BOX"] in ("F", "G", "H", "I", "J", "K", "L", "N", "O", "P", "Q")') == 2
    assert ("  O) . $W/sc2_box_e.sh; . $W/sc2c_box_h.sh; . $W/sc2e_box_l.sh; . $W/sc2f_box_n.sh; . $W/sc2h_box_o.sh; "
            "install_sc2_client ;;") in RUN
    assert 'O) PROVE_NEEDS="sc2client";;' in RUN and '[ "$BOX" = O ] && prove_o' in RUN and "O) box_o;; P) box_p;; Q) box_q;; esac" in RUN
    assert 'case "$SC1_BOX" in A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q) ;;' in DRIVE
    assert "sc2h_box_o.sh sc2h_reduce.py sc2h_basis.py sc2i_box_p.sh sc2i_reduce.py sc2j_box_q.sh sc2j_reduce.py sc2j_contract.py sc2j_prompts.py sc2j_quality.py; do STAGE=" in DRIVE
    assert 'sc2f_*|sc2h_*|sc2i_*|sc2j_*) src="$SC2/$name";;' in DRIVE
    assert 'python3 "$SC2/sc2h_reduce.py" --knee-rule "$SC2H_SC2F_VERDICT"' in DRIVE
    assert '[ "$SC1_BOX" = O ] && [ "${SC1_PROVE:-0}" != 1 ]' in DRIVE
    pin = (REPO / "bench" / "sc1" / "staged.sha256").read_text()
    for name in ("sc2h_box_o.sh", "sc2h_reduce.py", "sc2h_basis.py"):
        assert re.search(rf"^[0-9a-f]{{64}}  {re.escape(name)}$", pin, re.M), name
