# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""SC2f Amendment 7 (SC2h Amendment 5, SC2i Amendment 4), on the CPU: qualify-at-proof's admission, its card checks and
its post-proof judge (``bench/sc2/sc2_qualify.py``), the controller wrapper's refusals on a stubbed box
(``bench/sc2/sc2_qualify_drive.sh``), the release pair in ``bench/sc1/sc1_run.sh``, and the default-dump records behind
the amendment's "what moved under the arms" (``bench/sc2/sc2f-am7/``)."""
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SC2 = REPO / "bench" / "sc2"
_spec = importlib.util.spec_from_file_location("sc2_qualify", SC2 / "sc2_qualify.py")
Q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(Q)
PREREG = (SC2 / "SC2f-PREREG.md").read_text()
AM7 = PREREG[PREREG.index("### Amendment 7"):]
RUN = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()
REC = SC2 / "sc2f-am7"
FIX = REPO / "tests" / "fixtures" / "sc2f_am7"


def test_the_self_test_passes():
    assert Q.self_test() == 0


# ---- the admission is the text
def test_the_admission_constants_are_the_texts():
    assert Q.WALK == ("52780", "145701", "55337")
    assert "**52780, then 145701, then 55337**" in AM7
    assert Q.EXCLUDED == ("13828", "37367") and Q.UNTIL_CREATE == ("26157",)
    assert "excluded: 13828 (clock-bound) and 37367" in AM7 and "26157 stays excluded until a create there" in AM7
    assert (Q.MAX_ATTEMPTS, Q.CEILING_USD_H, Q.MIN_DISK_GB, Q.STABLE_READS) == (3, 1.05, 320, 3)
    assert "**Every create consumes one of 3 proof attempts** (`sc2f-prove-11`, `-12`, `-13`)" in AM7
    assert Q.FIRST_ID == {"sc2f": 11, "sc2h": 1, "sc2i": 1}
    assert "each of\n  the last 3 consecutive reads" in AM7 or "the last 3 consecutive reads" in AM7.replace("\n  ", " ")
    assert Q.ENFORCED_W == 575.0 and "--floor 2700" in AM7


def _receipt(store, day, run_id, machine, instance="1", status="OK", result="pass"):
    d = store / day / run_id
    d.mkdir(parents=True)
    (d / "receipt.json").write_text(json.dumps({"instance_id": instance, "status": status, "result": result,
                                                "environment": {"vast_machine_id": machine}}))


def _reads(*machines, n=3, usd=0.9, disk=400):
    return [[{"machine_id": m, "offer_id": f"o{m}", "usd_h": usd, "disk_gb": disk} for m in machines]] * n


def _admit(store, lane, reads):
    return Q.admit(Q.lane_attempts(store, lane), reads, Q.creates_on(store, "26157"),
                   prefer=Q.earlier_hosts(store, lane))


def test_admission_from_the_receipt_store(tmp_path):
    s = tmp_path / "store"
    _receipt(s, "2026-10-18", "sc2f-prove-1", "26157", instance="none", status="REFUSED", result="invalid")
    assert _admit(s, "sc2f", _reads("52780", "145701"))["machine"] == "52780"         # pre-Am. 7 ids are not attempts
    _receipt(s, "2026-10-26", "sc2f-prove-11", "52780", status="HARNESS_ERROR", result="fail")   # a qualify refusal
    r = _admit(s, "sc2f", _reads("52780", "145701"))
    assert r["machine"] == "145701" and r["attempts_used"] == 1 and r["skipped"]["52780"].startswith("already tried")
    _receipt(s, "2026-10-26", "sc2f-prove-12-refused-1", "145701", instance="none", status="REFUSED", result="invalid")
    assert _admit(s, "sc2f", _reads("145701"))["machine"] == "145701"               # refused before create: no attempt
    _receipt(s, "2026-10-26", "sc2f-prove-12", "145701", status="HARNESS_ERROR", result="fail")
    _receipt(s, "2026-10-27", "sc2f-prove-13", "55337", status="HARNESS_ERROR", result="fail")
    assert _admit(s, "sc2f", _reads("52780", "145701", "55337"))["verdict"] == "PARK"


def test_a_proved_host_is_the_lanes_and_carries_to_sc2h(tmp_path):
    s = tmp_path / "store"
    _receipt(s, "2026-10-26", "sc2f-prove-11", "145701")
    assert _admit(s, "sc2f", _reads("52780", "145701"))["machine"] == "145701"
    assert _admit(s, "sc2f", _reads("52780"))["verdict"] == "NONE"                 # only the proved host, for the reading
    assert _admit(s, "sc2h", _reads("52780", "145701"))["machine"] == "145701"     # continuity: SC2f's host first
    assert _admit(s, "sc2h", _reads("52780"))["machine"] == "52780"                # unlisted: the walk


def test_26157_is_admissible_only_after_a_create_there(tmp_path):
    s = tmp_path / "store"
    assert not Q.creates_on(s, "26157")
    _receipt(s, "2026-10-01", "other-lane-1", "26157")
    assert Q.creates_on(s, "26157")


# ---- the card checks
@pytest.mark.parametrize("csv,want", [("575.00 W, 575.00 W, 600.00 W", "PASS"), ("575 W, 575 W, 575 W", "PASS"),
                                      ("574.99 W, 575.00 W, 600.00 W", "REFUSE"), ("[N/A], 575.00 W, 600.00 W", "REFUSE"),
                                      ("575.00 W, 575.00 W", "REFUSE"), ("", "REFUSE")])
def test_the_power_parse(csv, want):
    assert Q.power(csv)["verdict"] == want


SC2J_PROVE2_POWER_LINE = "power.limit,clocks.max.sm 575.00 W, 3105 MHz"   # sc2j-prove-2's forensics.txt, line 3
SC2J_PROVE2_HOST_FACTS_SHA256 = "48882954f8694a30887409ecbdd06eb5b49535496d80d970320fe0dc67e9cf56"


def test_the_post_proof_judge_on_a_real_proofs_records():
    raw = (FIX / "sc2j-prove-2_host_facts.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SC2J_PROVE2_HOST_FACTS_SHA256
    hf = json.loads(raw)
    forensics = "GPU 0: NVIDIA GeForce RTX 5090\n" + SC2J_PROVE2_POWER_LINE + "\n"
    assert Q.judge_proof(forensics, hf)["verdict"] == "PASS"
    assert Q.judge_proof(forensics.replace("575.00 W", "500.00 W"), hf)["verdict"] == "REFUSE"
    assert Q.judge_proof(forensics, dict(hf, gpu_power_limit_w=400.0))["verdict"] == "REFUSE"


# ---- the wrapper, on a stubbed box
def _tree(tmp_path, drive_rc=0):
    """A copy of the repo pieces the wrapper reads, with bench/sc1/sc1_drive.sh a stub that records its call."""
    t = tmp_path / "repo"
    (t / "bench" / "sc2").mkdir(parents=True)
    (t / "bench" / "sc1").mkdir(parents=True)
    (t / "bench" / "common").mkdir(parents=True)
    for f in ("sc2_qualify_drive.sh", "sc2_qualify.py"):
        shutil.copy(SC2 / f, t / "bench" / "sc2" / f)
    shutil.copy(REPO / "bench" / "common" / "card_clock.py", t / "bench" / "common" / "card_clock.py")
    (t / "bench" / "sc1" / "sc1_drive.sh").write_text(
        f'#!/bin/bash\necho DRIVE_CALLED >> "$DRIVE_LOG"\nmkdir -p "$E4B_RENT_RUN_DIR/sc1"\n'
        f'printf "power.limit,clocks.max.sm %s W, 3105 MHz\\n" "$PROOF_W" > "$E4B_RENT_RUN_DIR/sc1/forensics.txt"\n'
        f'exit {drive_rc}\n')
    return t


def _stubs(tmp_path, power, burn_rc):
    b = tmp_path / "bin"
    b.mkdir()
    box = tmp_path / "box"
    box.mkdir()
    (b / "ssh").write_text(f'''#!/bin/bash
for a in "$@"; do last=$a; done
case "$last" in
  nvidia-smi*) echo "{power}";;
  *card_clock.py*) mkdir -p {box}/q; echo {burn_rc} > {box}/q/burn.rc; echo "burn" > {box}/q/card_clock.log;;
  *) :;;
esac
''')
    (b / "scp").write_text("#!/bin/bash\nexit 0\n")
    (b / "rsync").write_text(f'#!/bin/bash\nfor a in "$@"; do last=$a; done\nmkdir -p "$last"; cp {box}/q/* "$last" 2>/dev/null; exit 0\n')
    for f in b.iterdir():
        os.chmod(f, 0o755)
    return b


def _wrap(tmp_path, power="575.00 W, 575.00 W, 600.00 W", burn_rc=0, drive_rc=0, prove="1", proof_w="575.00"):
    t, b = _tree(tmp_path, drive_rc), _stubs(tmp_path, power, burn_rc)
    run = tmp_path / "run"
    env = {"PATH": f"{b}:{os.environ['PATH']}", "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1",
           "E4B_RENT_SSH_OPTS": "-o X=1", "E4B_RENT_RUN_DIR": str(run), "E4B_RENT_RUN_ID": "sc2f-prove-11",
           "E4B_RENT_DEADLINE_EPOCH": "9999999999", "E4B_RENT_INSTANCE_ID": "1", "SC1_BOX": "N", "SC1_PROVE": prove,
           "DRIVE_LOG": str(tmp_path / "drive.log"), "PROOF_W": proof_w}
    out = subprocess.run(["bash", str(t / "bench" / "sc2" / "sc2_qualify_drive.sh")], capture_output=True, text=True,
                         env=env)
    return out, (tmp_path / "drive.log").exists(), run


def test_the_wrapper_refuses_a_400_w_card_before_anything(tmp_path):
    out, drove, run = _wrap(tmp_path, power="400.00 W, 575.00 W, 600.00 W")
    assert out.returncode == 17 and not drove, out.stdout + out.stderr
    assert json.loads((run / "qualify" / "power.json").read_text())["enforced_w"] == 400.0


def test_the_wrapper_refuses_a_slow_card_with_18(tmp_path):
    out, drove, _ = _wrap(tmp_path, burn_rc=18)
    assert out.returncode == 18 and not drove, out.stdout + out.stderr


def test_a_qualified_card_runs_the_lanes_driver_and_its_clean_proof_is_judged(tmp_path):
    out, drove, run = _wrap(tmp_path)
    assert out.returncode == 0 and drove, out.stdout + out.stderr
    assert json.loads((run / "qualify" / "proof_judge.json").read_text())["verdict"] == "PASS"


def test_a_clean_proof_on_a_500_w_record_is_a_host_fault(tmp_path):
    out, drove, _ = _wrap(tmp_path, proof_w="500.00")
    assert out.returncode == 18 and drove


def test_a_failed_proof_keeps_its_own_code_and_is_never_judged(tmp_path):
    out, drove, run = _wrap(tmp_path, drive_rc=23, proof_w="500.00")
    assert out.returncode == 23 and drove and not (run / "qualify" / "proof_judge.json").exists()


def test_a_reading_is_not_judged_after_the_box(tmp_path):
    out, drove, run = _wrap(tmp_path, prove="0", proof_w="500.00")
    assert out.returncode == 0 and drove and not (run / "qualify" / "proof_judge.json").exists()


def test_the_wrapper_refuses_boxes_outside_sc2f_h_i(tmp_path):
    t = _tree(tmp_path)
    env = {"PATH": os.environ["PATH"], "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "x",
           "E4B_RENT_RUN_DIR": str(tmp_path), "E4B_RENT_RUN_ID": "r", "E4B_RENT_DEADLINE_EPOCH": "1",
           "E4B_RENT_INSTANCE_ID": "1", "SC1_BOX": "Q"}
    out = subprocess.run(["bash", str(t / "bench" / "sc2" / "sc2_qualify_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 78


def test_the_wrapper_never_exits_a_launcher_host_code_for_a_setup_failure():
    src = (SC2 / "sc2_qualify_drive.sh").read_text()
    assert re.findall(r"exit 17\b", src) == ["exit 17"] and len(re.findall(r"exit 18\b", src)) == 2
    assert "rc 9, not a card verdict\"; exit 9" in src


# ---- the release pair
def test_boxes_n_o_p_on_gnf4_0_51_0_and_q_unchanged():
    assert 'case "$BOX" in N|O) GNF4_SHA=30adb1bfa76f95f9ce4a4c29734b8f36de907eca;; esac' in RUN
    assert 'case "$BOX" in P) GNF4_SHA=30adb1bfa76f95f9ce4a4c29734b8f36de907eca;; Q) GNF4_SHA=41ef086e6a3e2be5b158ff1bb38e0de67862a949;; esac' in RUN
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    assert "grouped-nf4-gemm.git@30adb1bfa76f95f9ce4a4c29734b8f36de907eca" in ci


def test_the_kit_hold_names_the_new_stack():
    from importlib.util import module_from_spec, spec_from_file_location
    sp = spec_from_file_location("skh", SC2 / "sc2_kit_hold.py")
    kh = module_from_spec(sp)
    sp.loader.exec_module(kh)
    assert kh.STACK == "605d21e67958fa94d5e73fc3a95ffcb5c8eae02d"
    for old, new in kh.GNF4_DELTA:
        assert new in RUN and old not in RUN
    assert kh.self_test() == 0


# ---- the default-dump records behind "what moved under the arms"
def _rec(name):
    return json.loads((REC / name).read_text())


@pytest.mark.parametrize("arm", ["on", "off"])
def test_what_moved_is_the_texts_list(arm):
    d = _rec(f"diff_{arm}_36cdd2d4_605d21e6.json")
    assert sorted(d["changed"]) == ["config.int8_head_rows", "env.E4B_NF4_SMALLM_BW"]
    assert d["changed"]["config.int8_head_rows"] == [1, "default"]
    assert d["only_ref"] == [] and d["why"] == []
    assert d["only_launch"] == ["config.attn_split_fixed", "config.kv_pool_resolution", "config.kv_pool_rows",
                                "config.merged_arena", "env.E4B_ATTN_SPLIT_FIXED", "env.E4B_DENSE_LT_PIN",
                                "env.E4B_PAGED_MERGED_ARENA", "env.E4B_TRAIN_DETERMINISTIC", "env.E4B_TRAIN_DET_GATHER"]
    new = _rec(f"defaults_{arm}_605d21e6.json")
    assert new["grouped_nf4_gemm"] == _rec(f"defaults_{arm}_36cdd2d4.json")["grouped_nf4_gemm"] == "0.51.0"
    lv = new["levers"]
    assert (lv["config.attn_split_fixed"], lv["config.merged_arena"], lv["config.kv_pool_rows"], lv["config.kv_pool"]) == (
        None, False, None, "claim-all")


@pytest.mark.parametrize("sha", ["36cdd2d4", "605d21e6"])
def test_the_on_off_contrast_is_exactly_p130s_two_knobs_in_both_trees(sha):
    on, off = _rec(f"defaults_on_{sha}.json"), _rec(f"defaults_off_{sha}.json")
    assert on["env_set"] == {} and off["env_set"] == {"E4B_FUSE_PREFILL_GLUE": "0", "E4B_PREFILL_LEAN_DISPATCH": "0"}
    a, b = on["levers"], off["levers"]
    moved = {k: [a.get(k), b.get(k)] for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert moved == {"config.prefill_glue": ["default", "0"],
                     "family.resolve_prefill_glue": [["1", "default-allowlisted"], ["0", "explicit"]],
                     "route._prefill_lean_dispatch_env": ["auto", "0"]}


def test_the_text_names_the_window_the_limit_and_the_deadline():
    for s in ("15:12Z to 22:35Z (7.4 h, 40 reads)", "only `offers_total`", "145701 alone was present in 2 of 40 reads (5 %)",
              "**upper bound**", "**The deadline is 72 h**", "whichever is\n  later", "**The G3 GPU check (e4b#2133) must PASS",
              "**Box Q stays at `v0.50.0`**", "**P130's two knobs: neither default moved.**",
              "**e4b#2087 moves neither arm's path.**", "$12.42"):
        assert s in AM7, s
    for lane, n in (("SC2h", 5), ("SC2i", 4)):
        t = (SC2 / f"{lane}-PREREG.md").read_text()
        assert f"### Amendment {n} (registered before any {lane} run): SC2f Amendment 7's" in t
        assert "If one lane later moves alone, its\n  cross-lane read is VOID." in t
