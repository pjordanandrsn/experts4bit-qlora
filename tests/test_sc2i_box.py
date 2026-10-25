# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane SC2i's box P (#846): the KNOBS check the box runs, executed rather than grepped; the tripwire run on this
checkout; the readings held equal to the code that reports them; the ladder, seeds and order every file of the lane
carries, held equal; and box P wired as box O is.

Box P stops a server (rc 44) unless its start /health reads its arm: ``engine.prefill_every`` and
``engine.loaded_chunk_tokens`` the arm's (e4b#2017), the decode lookahead off, and SC2f's ``on`` defaults.
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
BOX = (SC2 / "sc2i_box_p.sh").read_text()
RUN = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()
DRIVE = (REPO / "bench" / "sc1" / "sc1_drive.sh").read_text()
PREREG = (SC2 / "SC2i-PREREG.md").read_text()


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, SC2 / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RED = _mod("sc2i_reduce")
FRED = _mod("sc2f_reduce")


def _health(arm, **over):
    w = FRED.WANT_FLIPS["on"]
    seen = {"moe_k19_dispatch": {"chained|lean|gt256": 240, "one_launch|lean|gt256": 144, "one_launch|lean|le256": 720},
            "prefill_folds": {f: 48 for f in FRED.FOLDS}}
    pr = {"prefill_glue_env": None, "prefill_lean_dispatch_env": None, "prefill_lean_dispatch": "auto", "seen": seen}
    pe, lc = RED.KNOBS[arm]
    eng = {"prefill_every": pe, "loaded_chunk_tokens": lc, "decode_lookahead": False}
    eng.update(over.pop("engine", {}))
    return {"engine": eng, "prefill_routes": pr, "levers": {"prefill_glue": over.pop("glue", dict(w["glue"]))}}


def _check(tmp_path, health, arm):
    p = tmp_path / f"h_{arm}.json"
    p.write_text(json.dumps(health))
    return subprocess.run([sys.executable, str(SC2 / "sc2i_reduce.py"), "--knobs", str(p), arm],
                          capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("arm", RED.ARMS)
def test_each_arm_reads_its_own_knobs(tmp_path, arm):
    r = _check(tmp_path, _health(arm), arm)
    assert r.returncode == 0 and '"ok": true' in r.stdout, r.stdout + r.stderr


def test_the_knobs_check_stops_on_every_departure(tmp_path):
    cases = [("pe2", {"engine": {"prefill_every": 1}}), ("base", {"engine": {"prefill_every": 2}}),
             ("lc128", {"engine": {"loaded_chunk_tokens": 256}}), ("lc256", {"engine": {"loaded_chunk_tokens": 0}}),
             ("pe4", {"engine": {"decode_lookahead": True}}), ("base", {"engine": {"prefill_every": None}}),
             ("base", {"glue": {"mode": "0", "source": "explicit"}})]                       # off P130's P1 default
    for arm, over in cases:
        r = _check(tmp_path, _health(arm, **json.loads(json.dumps(over))), arm)
        assert r.returncode == 1 and '"ok": false' in r.stdout, (arm, over, r.stdout)
    assert _check(tmp_path, {}, "base").returncode == 1


def test_the_readings_are_what_the_code_reports():
    sp = (REPO / "experts4bit_qlora" / "serve_paged.py").read_text(encoding="utf-8")
    assert '"prefill_every": cfg.prefill_every, "loaded_chunk_tokens": cfg.loaded_chunk_tokens,' in sp   # /health engine
    assert 'prefill_every=int(env("E4B_PAGED_PREFILL_EVERY", "1") or "1")' in sp
    assert 'loaded_chunk_tokens=int(env("E4B_PAGED_LOADED_CHUNK_TOKENS", "0") or "0")' in sp
    pr = (REPO / "experts4bit_qlora" / "engines" / "paged_runner.py").read_text(encoding="utf-8")
    assert 'self._note_eager("later_chunk" if start else "short_chunk")' in pr
    sc = (REPO / "experts4bit_qlora" / "engines" / "scheduler.py").read_text(encoding="utf-8")
    assert '"prefill_deferred": self.prefill_deferred' in sc


def test_the_tripwire_passes_on_this_checkout(tmp_path):
    pytest.importorskip("torch")
    body = BOX[BOX.index('p_tripwire(){ "$PY" -c "') + len('p_tripwire(){ "$PY" -c "'):BOX.index('" \\\n    2>&1 | tail -1 | sed')]
    p = tmp_path / "trip.py"
    p.write_text(body)
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    r = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, timeout=300, cwd=REPO, env=env)
    assert r.returncode == 0 and "e4b#2017 knobs present at their defaults" in r.stdout, r.stdout + r.stderr


def test_every_file_of_the_lane_carries_the_same_ladder_seeds_and_order():
    assert RED.LADDER == (4, 8, 12) and 'SC2I_LADDER="4 8 12"' in BOX and "ladder at 4, 8 and 12 req/s" in PREREG
    assert "$(( D * 100 + r ))" in BOX and "seed draw × 100 + rate" in PREREG
    assert f'SC2I_ORDER_D1="{" ".join(RED.ORDER[1])}"; SC2I_ORDER_D2="{" ".join(RED.ORDER[2])}"' in BOX
    assert f"Draw 1: `{' '.join(RED.ORDER[1])}`; draw 2: `{' '.join(RED.ORDER[2])}`" in PREREG
    assert "SC2I_N=120; SC2I_SERIAL_N=24" in BOX and (RED.N_RATE, RED.N_SERIAL) == (120, 24) and RED.SLOT_ARM == "s64a"
    assert RED.servers()[0] == ("e4b_base_a_d1", "base", 1, "base_a") and RED.servers()[7] == ("e4b_lc128_d2", "lc128", 2, "lc128")
    for arm, (pe, lc) in RED.KNOBS.items():
        env = subprocess.run(["bash", "-c", BOX[BOX.index("p_env(){"):BOX.index("# p_tripwire")] + f"\np_env {arm}"],
                             capture_output=True, text=True).stdout.strip()
        want = "" if arm == "base" else (f"E4B_PAGED_PREFILL_EVERY={pe}" if pe != 1 else f"E4B_PAGED_LOADED_CHUNK_TOKENS={lc}")
        assert env == want, (arm, env)


def test_the_server_start_changes_only_the_knobs():
    arm = BOX[BOX.index("p_arm(){"):BOX.index("# ---- the real lane")]
    assert 'LEV="$SPEEDENV E4B_PAGED_FUSE_QKV=1 $(p_env "$ARM")"' in arm
    assert "\nunset E4B_PAGED_PREFILL_EVERY E4B_PAGED_LOADED_CHUNK_TOKENS E4B_PAGED_DECODE_LOOKAHEAD " in BOX
    assert 'l_server_start "$TAG" s64a "$MID" "$REV" "$QA" "$LEV" "int4_k19|" && p_knobs_ok "$TAG" "$ARM"' in arm
    assert 'h_early "$TAG" on $W/sc2/${TAG}_warm.json && l_burst "$TAG" s64a "$MID"' in arm
    assert 'h_health_end "$TAG"; p_stats_end "$TAG"' in arm
    assert "return 44" in BOX and "l_tripwire; n_tripwire; p_tripwire" in BOX
    spd = "".join(re.search(rf'^{v}="([^"]*)"', RUN, re.M).group(1) for v in ("SPEEDENV", "FOLDS", "ROUTEENV"))
    assert "PREFILL_EVERY" not in spd and "LOADED_CHUNK" not in spd and "LOOKAHEAD" not in spd


def test_the_reducer_passes_its_self_test():
    r = subprocess.run([sys.executable, str(SC2 / "sc2i_reduce.py"), "--self-test"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "self-test OK" in r.stdout, r.stdout + r.stderr


def test_box_p_is_wired_like_box_o():
    assert 'case "$BOX" in A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q) ;;' in RUN
    assert 'case "$BOX" in P) GNF4_SHA=30adb1bfa76f95f9ce4a4c29734b8f36de907eca;; Q) GNF4_SHA=41ef086e6a3e2be5b158ff1bb38e0de67862a949;; esac' in RUN      # P on v0.51.0 (SC2i Amendment 4), Q unchanged
    torch = next(x for x in RUN.splitlines() if '"torch==2.8.0"' in x and "pipx logs/pip_torch.log" in x)
    assert '[ "$BOX" = P ]' in torch
    assert RUN.count('os.environ["TRIP_BOX"] in ("F", "G", "H", "I", "J", "K", "L", "N", "O", "P", "Q")') == 2
    assert ("  P) . $W/sc2_box_e.sh; . $W/sc2c_box_h.sh; . $W/sc2e_box_l.sh; . $W/sc2f_box_n.sh; . $W/sc2i_box_p.sh; "
            "install_sc2_client ;;") in RUN
    assert 'P) PROVE_NEEDS="sc2client";;' in RUN and '[ "$BOX" = P ] && prove_p' in RUN and "P) box_p;; Q) box_q;; esac" in RUN
    assert 'case "$SC1_BOX" in A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q) ;;' in DRIVE
    assert "sc2h_box_o.sh sc2h_reduce.py sc2h_basis.py sc2i_box_p.sh sc2i_reduce.py sc2j_box_q.sh sc2j_reduce.py sc2j_contract.py sc2j_prompts.py sc2j_quality.py; do STAGE=" in DRIVE
    assert 'sc2f_*|sc2h_*|sc2i_*|sc2j_*) src="$SC2/$name";;' in DRIVE
    pin = (REPO / "bench" / "sc1" / "staged.sha256").read_text()
    for name in ("sc2i_box_p.sh", "sc2i_reduce.py"):
        assert re.search(rf"^[0-9a-f]{{64}}  {re.escape(name)}$", pin, re.M), name


# ------------------------------------------------- the real records of the shakedown (tests/fixtures/sc2i_shake1) --
FIX = REPO / "tests" / "fixtures" / "sc2i_shake1"


def _fx(name):
    return json.loads((FIX / name).read_text())


def test_the_fixture_is_the_receipts_bytes():
    import hashlib
    man = _fx("manifest.json")
    for name in ("health_prove_lc128_start.json", "health_prove_lc128_end.json", "stats_prove_lc128_end.json",
                 "health_prove_base_end.json"):
        assert hashlib.sha256((FIX / name).read_bytes()).hexdigest() == man["sha256"][name], name
    assert man["admitted"] == {"base": 90, "pe2": 90, "lc128": 90}


def test_the_adapted_engaged_gate_on_the_real_lc128_record():
    e_red = _mod("sc2e_reduce")
    end = _fx("health_prove_lc128_end.json")
    g = end["prefill_graph"]
    assert (g["replays"], g["eager_reasons"]) == (12, {"later_chunk": 234, "short_chunk": 78})
    assert RED.engagement_bad(e_red, end, 90, "lc128") == []                 # replays + short_chunk = 90 admitted
    assert e_red.engagement(RED.SLOT_ARM, end, 90) != []                    # SC2e's unadapted check fails it
    assert RED.engagement_bad(e_red, end, 91, "lc128") != []                # a count off by one fails
    assert RED.engagement_bad(e_red, end, 90, "pe2") != []                  # a pe arm keeps SC2e's check


def test_the_real_base_record_passes_sc2es_unadapted_check():
    e_red = _mod("sc2e_reduce")
    assert RED.engagement_bad(e_red, _fx("health_prove_base_end.json"), 90, "base") == []


def test_the_census_check_is_narrow_to_the_lc_arms():
    lc, base = _fx("census_lc128.json"), _fx("census_base.json")
    assert lc["prompts"] == 12 and lc["decode_steps"]["64"]["n"] > 0         # the real lc census: 12 replayed prompts
    assert RED.census_bad(lc, "lc128") == []                                 # lc: the 64-row step is enough
    assert RED.census_bad(lc, "base") != [] and RED.census_bad(lc, "pe2") != []   # a base or pe arm without it fails
    assert RED.census_bad(base, "base") == [] and base["prompts"] >= 26
    lc_no64 = json.loads(json.dumps(lc))
    lc_no64["decode_steps"].pop("64")
    assert RED.census_bad(lc_no64, "lc128") != []


def test_the_knobs_check_on_the_real_lc128_start_record():
    h = _fx("health_prove_lc128_start.json")
    assert RED.knobs_bad(h, "lc128", proof=True) == {}
    assert RED.knobs_bad(h, "base", proof=True) != {}


def test_the_proof_uses_the_arm_scoped_census():
    prove = BOX[BOX.index("prove_p(){"):]
    assert '"$PY" $W/sc2i_reduce.py --census $W/sc2/steps_prove_$a.jsonl $a' in prove
    assert "--require-wide" not in prove and "for a in base pe2 lc128; do" in prove
