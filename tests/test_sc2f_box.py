# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane SC2f's box N (#846): the FLIPS check the box runs on a server's own /health, executed rather than grepped; the
knobs' readings held equal to the code that produces them; and the ladder, seeds and server order every file of the
lane carries, held equal.

Box N stops a server (rc 43) unless its start /health reads as its arm of P130's two knobs. ``on`` leaves both unset
(the release's defaults): the glue default must resolve from the family allowlist, and the startup captures must have
taken K19's lean dispatch on every call above 256 rows, the prefill capture's chained table among them, and all three
prefill folds. ``off`` sets both to 0: an explicit 0, the gather dispatch, no prefill fold. The table of the other calls
above 256 rows (bucket 64's decode capture: the one-launch table where ``E4B_INT4_WIDE_TILES=auto`` takes it) is not
read (Amendment 1); a split table is refused on either arm. ``tests/test_sc2f_real_records.py`` runs the gate on real
records first. The box runs ``sc2f_reduce.py --flips``, the reducer's own gate, so the two cannot drift.
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
BOX = (SC2 / "sc2f_box_n.sh").read_text()
RUN = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()
DRIVE = (REPO / "bench" / "sc1" / "sc1_drive.sh").read_text()


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, SC2 / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RED = _mod("sc2f_reduce")


def _health(arm, **over):
    w = RED.WANT_FLIPS[arm]
    d = w["dispatch"]          # an s64a start: the prefill capture (chained) and bucket 64's decode capture (one-launch)
    seen = {"moe_k19_dispatch": {f"chained|{d}|gt256": 240, f"one_launch|{d}|gt256": 144, "one_launch|lean|le256": 720},
            "prefill_folds": {f: 48 for f in RED.FOLDS} if w["folds"] else {}}
    seen.update(over.pop("seen", {}))
    pr = {"prefill_glue_env": w["glue_env"], "prefill_lean_dispatch_env": w["lean_env"], "prefill_lean_dispatch": w["lean"],
          "seen": seen}
    pr.update(over.pop("routes", {}))
    return {"prefill_routes": pr, "levers": {"prefill_glue": over.pop("glue", dict(w["glue"]))}}


def _flips(tmp_path, health, arm, *extra):
    p = tmp_path / f"h_{arm}.json"
    p.write_text(json.dumps(health))
    return subprocess.run([sys.executable, str(SC2 / "sc2f_reduce.py"), "--flips", str(p), arm, *extra],
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("arm", ["on", "off"])
def test_each_arm_reads_its_own_knobs(tmp_path, arm):
    r = _flips(tmp_path, _health(arm), arm)
    assert r.returncode == 0 and '"ok": true' in r.stdout, r.stdout + r.stderr


def test_the_flips_check_stops_on_every_departure(tmp_path):
    cases = [("on", {"glue": {"mode": "1", "source": "explicit"}}),                       # set, not the default
             ("on", {"glue": {"mode": "0", "source": "default-off"}}),                    # the family missed the allowlist
             ("on", {"routes": {"prefill_lean_dispatch_env": "auto"}}),
             ("on", {"seen": {"moe_k19_dispatch": {"chained|gather|gt256": 96}}}),         # the K19 lacks the options
             ("on", {"seen": {"moe_k19_dispatch": {"chained|lean|gt256": 90, "chained|gather|gt256": 6}}}),
             ("on", {"seen": {"moe_k19_dispatch": {"chained|lean|gt256": 240, "one_launch|gather|gt256": 144}}}),
             ("on", {"seen": {"moe_k19_dispatch": {"one_launch|lean|gt256": 144}}}),     # no prefill capture above 256
             ("on", {"seen": {"moe_k19_dispatch": {"chained|lean|gt256": 240, "split|lean|gt256": 4}}}),
             ("on", {"seen": {"prefill_folds": {"norm": 48, "layer": 48}}}),               # one fold never took prefill rows
             ("off", {"glue": {"mode": "1", "source": "default-allowlisted"}}),           # the 0 never reached the build
             ("off", {"routes": {"prefill_lean_dispatch": "auto"}}),
             ("off", {"seen": {"moe_k19_dispatch": {"chained|lean|gt256": 96}}}),
             ("off", {"seen": {"moe_k19_dispatch": {"chained|gather|gt256": 240, "one_launch|lean|gt256": 144}}}),
             ("off", {"seen": {"prefill_folds": {"attention": 1}}}),
             ("off", {"seen": {"moe_k19_dispatch": {}}})]                                  # nothing above 256 rows ran
    for arm, over in cases:
        r = _flips(tmp_path, _health(arm, **over), arm)
        assert r.returncode == 1 and '"ok": false' in r.stdout, (arm, over, r.stdout)
    r = _flips(tmp_path, {}, "on")
    assert r.returncode == 1 and "<missing>" in r.stdout
    r = _flips(tmp_path, _health("on"), "neither")
    assert r.returncode == 1


def test_the_proof_reads_granite_where_neither_knob_engages(tmp_path):
    for arm, glue in (("on", {"mode": "0", "source": "default-off"}), ("off", {"mode": "0", "source": "explicit"})):
        w = RED.WANT_FLIPS_PROOF[arm]
        h = {"prefill_routes": {"prefill_glue_env": w["glue_env"], "prefill_lean_dispatch_env": w["lean_env"],
                                "prefill_lean_dispatch": w["lean"], "seen": {"moe": {"nf4_grouped|gt256": 9}}},
             "levers": {"prefill_glue": glue}}
        assert _flips(tmp_path, h, arm, "--proof").returncode == 0, arm
        assert _flips(tmp_path, h, arm).returncode == 1, arm                 # the reading's form refuses it


def test_the_knob_readings_are_what_the_code_reports():
    src = {p.name: p.read_text() for p in (REPO / "experts4bit_qlora" / "engines").glob("*.py")}
    for fold in RED.FOLDS:                                                    # glue_fuse / glue_r2 count these names
        assert any(f'_seen_prefill("{fold}")' in s for s in src.values()), fold
    hr = src["hot_residency.py"]
    assert "key = f\"{table}|{dispatch}|{'le256' if rows <= 256 else 'gt256'}\"" in hr
    assert '"one_launch" if (_lean or _fused_tiles is not None) else "chained",' in hr
    assert '"lean" if (_lean or _pf_lean) else "gather", R_rows)' in hr
    sp = (REPO / "experts4bit_qlora" / "serve_paged.py").read_text()
    assert '"prefill_glue_env": os.environ.get("E4B_FUSE_PREFILL_GLUE", "") or None' in sp
    assert '"prefill_lean_dispatch_env": os.environ.get("E4B_PREFILL_LEAN_DISPATCH", "") or None' in sp
    assert '"prefill_lean_dispatch": hot_residency._prefill_lean_dispatch_env()' in sp


def test_the_glue_tables_match_resolve_prefill_glue():
    """Before #1615 merges the function does not exist and the box's tripwire refuses that e4b (rc 9)."""
    pytest.importorskip("torch")
    from experts4bit_qlora import serve_paged as s
    if not hasattr(s, "resolve_prefill_glue"):
        pytest.skip("the installed e4b predates #1615 (the P1 default): box N's tripwire refuses it")
    assert s.resolve_prefill_glue(s.FUSION_UNSET, "qwen3_moe") == tuple(RED.WANT_FLIPS["on"]["glue"].values())
    assert s.resolve_prefill_glue("0", "qwen3_moe") == tuple(RED.WANT_FLIPS["off"]["glue"].values())
    assert s.resolve_prefill_glue(s.FUSION_UNSET, "granitemoe") == tuple(RED.WANT_FLIPS_PROOF["on"]["glue"].values())


def test_the_tripwire_refuses_an_e4b_without_both_defaults(tmp_path):
    pytest.importorskip("torch")
    from experts4bit_qlora import serve_paged as s
    body = BOX[BOX.index('n_tripwire(){ "$PY" -c "') + len('n_tripwire(){ "$PY" -c "'):BOX.index('" \\\n    2>&1 | tail -1 | sed')]
    p = tmp_path / "trip.py"
    p.write_text(body)
    env = {**os.environ, "PYTHONPATH": str(REPO)}                       # the checkout's e4b, as the box's install is
    r = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, timeout=300, cwd=REPO, env=env)
    if hasattr(s, "resolve_prefill_glue"):
        assert r.returncode == 0 and "both P130 defaults present" in r.stdout, r.stdout + r.stderr
    else:
        assert r.returncode != 0 and "lacks the P1 default (#1615)" in r.stderr, r.stdout + r.stderr


def test_every_file_of_the_lane_carries_the_same_ladder_seeds_and_order():
    bas = _mod("sc2f_basis")
    assert RED.LADDER == bas.LADDER == (4, 8, 12, 13, 14, 16) and 'SC2F_RATES="4 8 12 13 14 16"' in BOX
    assert RED.KNEE == bas.KNEE == (12, 13, 14) and 'SC2F_KNEE="12 13 14"' in BOX
    assert RED.KNEE_J == bas.KNEE_J == (1, 2, 3, 4) and 'SC2F_KNEE_J="1 2 3 4"' in BOX
    assert "$(( D * 100 + r ))" in BOX and "$(( 10000 * D + 100 * r + j ))" in BOX
    assert bas.ladder_seed(2, 13) == 213 and bas.knee_seeds(2, 13)[1:] == [21301, 21302, 21303, 21304]
    assert f'SC2F_ORDER_D1="{" ".join(RED.ORDER[1])}"; SC2F_ORDER_D2="{" ".join(RED.ORDER[2])}"' in BOX
    assert RED.SLOT_ARM == "s64a" and bas.MAX_SEQS == 64 and bas.BUCKETS == (1, 2, 4, 8, 16, 32, 64)
    assert 'SC2F_N=120; SC2F_SERIAL_N=24' in BOX and bas.N == 120 and bas.SERIAL_N == 24


def test_the_server_start_changes_only_the_two_knobs():
    arm = BOX[BOX.index("n_arm(){"):BOX.index("# ---- the real lane")]
    assert 'LEV="$SPEEDENV E4B_PAGED_FUSE_QKV=1"; [ "$ARM" = off ] && LEV="$LEV $SC2F_OFF"' in arm
    assert 'SC2F_OFF="E4B_FUSE_PREFILL_GLUE=0 E4B_PREFILL_LEAN_DISPATCH=0"' in BOX
    assert "\nunset E4B_FUSE_PREFILL_GLUE E4B_PREFILL_LEAN_DISPATCH " in BOX            # the on arm reads the defaults
    assert 'l_server_start "$TAG" s64a "$MID" "$REV" "$QA" "$LEV" "int4_k19|" && n_flips_ok "$TAG" "$ARM"' in arm
    assert 'h_early "$TAG" on $W/sc2/${TAG}_warm.json && l_burst "$TAG" s64a "$MID"' in arm
    assert "return 43" in BOX and "--flips $W/sc2/health_${1}_start.json" in BOX
    assert "E4B_FUSE_PREFILL_GLUE" not in (SC2 / "sc2e_box_l.sh").read_text()        # l_server_start leaves them be
    spd = re.search(r'^SPEEDENV="([^"]*)"', RUN, re.M).group(1) + re.search(r'^FOLDS="([^"]*)"', RUN, re.M).group(1)
    assert "PREFILL_GLUE" not in spd and "LEAN_DISPATCH" not in spd


@pytest.mark.parametrize("script", ["sc2f_reduce.py", "sc2f_basis.py"])
def test_the_lane_tools_pass_their_self_tests(script):
    r = subprocess.run([sys.executable, str(SC2 / script), "--self-test"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "self-test:" in r.stdout and "FAILED" not in r.stdout, r.stdout + r.stderr


def test_box_n_is_wired_like_box_l():
    assert 'case "$BOX" in A|B|C|D|E|F|G|H|I|J|K|L|M|N|O) ;;' in RUN
    assert 'case "$BOX" in N|O) GNF4_SHA=41ef086e6a3e2be5b158ff1bb38e0de67862a949;; esac' in RUN      # v0.50.0 (Amendment 3)
    assert "C|D|E|F|G|H|I|J|K|L|M|N|O) BASEPY=python3;;" in RUN
    torch = next(x for x in RUN.splitlines() if '"torch==2.8.0"' in x and "pipx logs/pip_torch.log" in x)
    assert '[ "$BOX" = N ]' in torch
    assert RUN.count('os.environ["TRIP_BOX"] in ("F", "G", "H", "I", "J", "K", "L", "N", "O")') == 2   # main's routes, no pins
    assert "  N) . $W/sc2_box_e.sh; . $W/sc2c_box_h.sh; . $W/sc2e_box_l.sh; . $W/sc2f_box_n.sh; install_sc2_client ;;" in RUN
    assert 'N) PROVE_NEEDS="sc2client";;' in RUN and '[ "$BOX" = N ] && prove_n' in RUN and "N) box_n;; O) box_o;; esac" in RUN
    assert 'case "$SC1_BOX" in A|B|C|D|E|F|G|H|I|J|K|L|M|N|O) ;;' in DRIVE
    assert "sc2f_box_n.sh sc2f_reduce.py sc2f_basis.py sc2h_box_o.sh sc2h_reduce.py sc2h_basis.py; do STAGE=" in DRIVE
    assert 'sc2c_*|sc2d_*|sc2e_*|sc2f_*|sc2h_*) src="$SC2/$name";;' in DRIVE
    pin = (REPO / "bench" / "sc1" / "staged.sha256").read_text()
    for name in ("sc2f_box_n.sh", "sc2f_reduce.py", "sc2f_basis.py"):
        assert re.search(rf"^[0-9a-f]{{64}}  {re.escape(name)}$", pin, re.M), name
