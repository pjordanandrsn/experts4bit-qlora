# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""SC2f Amendment 7 (SC2h Amendment 5, SC2i Amendment 4), on the CPU: qualify-at-proof's open-walk admission, its card
checks, its run sampler and its post-proof judge (``bench/sc2/sc2_qualify.py``), the controller wrapper's three-way exit
codes on a stubbed box (``bench/sc2/sc2_qualify_drive.sh``), the release pair in ``bench/sc1/sc1_run.sh``, the basis
window's stability recount, and the default-dump records behind "what moved under the arms" (``bench/sc2/sc2f-am7/``)."""
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
AM7_FLAT = re.sub(r"\s+", " ", AM7)
RUN = (REPO / "bench" / "sc1" / "sc1_run.sh").read_text()
REC = SC2 / "sc2f-am7"
FIX = REPO / "tests" / "fixtures" / "sc2f_am7"


def test_the_self_test_passes():
    assert Q.self_test() == 0


# ---- the admission is the text
def test_the_admission_constants_are_the_texts():
    assert Q.NAMED == ("52780", "145701") and "**52780, then 145701**" in AM7
    assert "The walk is **open**" in AM7_FLAT and "**any other machine whose offers list 575 W**" in AM7_FLAT
    assert Q.EXCLUDED == ("13828", "37367") and Q.UNTIL_CREATE == ("26157",)
    assert "Excluded throughout: 13828 (clock-bound) and 37367" in AM7_FLAT and "26157 until a create there succeeds" in AM7_FLAT
    assert (Q.MAX_ATTEMPTS, Q.CEILING_USD_H, Q.MIN_DISK_GB, Q.STABLE_READS) == (3, 1.05, 320, 3)
    assert 575.0 in Q.POWER_W                          # Amendment 7's 575 W; SC2f Amendment 10 widens the class (test_sc2f_am10)
    assert "**Every create consumes one of 3 proof attempts** (`sc2f-prove-11`, `-12`, `-13`), whatever its exit" in AM7_FLAT
    assert Q.FIRST_ID == {"sc2f": 11, "sc2h": 1, "sc2i": 1}
    assert "`vast_provider.hourly(offer)` at 320 GB disk" in AM7_FLAT and "`gpu_max_power` as `listed_power_w`" in AM7_FLAT
    assert 575.0 in Q.POWER_W and "--floor 2700" in AM7 and Q.SERVING_JUDGE_FLOOR_MHZ == 0   # SC2f Amendment 9 supersedes the 95 % bar


def _receipt(store, day, run_id, machine, instance="1", status="OK", result="pass"):
    d = store / day / run_id
    d.mkdir(parents=True)
    (d / "receipt.json").write_text(json.dumps({"instance_id": instance, "status": status, "result": result,
                                                "environment": {"vast_machine_id": machine}}))


def _reads(*machines, n=3, usd=0.9, disk=400, w=575.0):
    return [[{"machine_id": m, "offer_id": f"o{m}", "usd_h": usd, "disk_gb": disk, "listed_power_w": w}
             for m in machines]] * n


def _admit(store, lane, reads, clean_runs=None):
    return Q.admit(Q.lane_attempts(store, lane), reads, Q.creates_on(store, "26157"),
                   prefer=Q.earlier_hosts(store, lane), clean_runs=clean_runs, clocks=Q.recorded_clocks(store))


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
    assert _admit(s, "sc2f", _reads("52780", "145701", "999"))["machine"] == "999"  # the open walk
    _receipt(s, "2026-10-27", "sc2f-prove-13", "999", status="HARNESS_ERROR", result="fail")
    assert _admit(s, "sc2f", _reads("52780", "145701", "1000"))["verdict"] == "PARK"


def test_a_proved_host_is_the_lanes_and_carries_to_sc2h(tmp_path):
    s = tmp_path / "store"
    _receipt(s, "2026-10-26", "sc2f-prove-11", "145701")
    assert _admit(s, "sc2f", _reads("52780", "145701"))["machine"] == "145701"
    assert _admit(s, "sc2f", _reads("52780"))["verdict"] == "NONE"                 # only the proved host, for the reading
    assert _admit(s, "sc2h", _reads("52780", "145701"))["machine"] == "145701"     # continuity: SC2f's host first
    assert _admit(s, "sc2h", _reads("52780"))["machine"] == "52780"                # unlisted: the walk


def test_the_open_walk_orders_by_tier_then_clean_runs_then_id(tmp_path):
    s = tmp_path / "store"
    reads = [[{"machine_id": m, "offer_id": f"o{m}", "usd_h": 0.9, "disk_gb": 400, "listed_power_w": w}
              for m, w in (("55337", None), ("700", 575.0), ("600", 575.0), ("19317", 450.0), ("145701", 575.0))]] * 3
    r = _admit(s, "sc2f", reads, clean_runs={"600": 4, "700": 4, "55337": 9})
    assert r["walk"] == ["52780", "145701", "600", "700", "55337"] and r["machine"] == "145701"
    assert "19317" not in r["walk"]


def test_a_new_machine_appearing_mid_screen(tmp_path):
    s = tmp_path / "store"
    reads = _reads("52780", usd=1.2, n=2) + _reads("888", n=2)
    assert _admit(s, "sc2f", reads)["verdict"] == "NONE"                       # two reads: not yet
    assert _admit(s, "sc2f", reads + _reads("888", n=1))["machine"] == "888"   # its third read admits it


def test_the_price_is_the_all_in_hourly_never_a_base_rate(tmp_path):
    """``usd_h`` is the launcher's hourly at 320 GB (GPU plus storage); an offer whose base rate is under the ceiling but
    whose all-in hourly is over it is never admitted, whatever other field the read carries."""
    reads = [[{"machine_id": "52780", "offer_id": "1", "usd_h": 1.08, "base_usd_h": 0.95, "disk_gb": 400,
               "listed_power_w": 575.0}]] * 3
    assert _admit(tmp_path / "store", "sc2f", reads)["verdict"] == "NONE"


def test_26157_is_admissible_only_after_a_create_there(tmp_path):
    s = tmp_path / "store"
    assert not Q.creates_on(s, "26157")
    _receipt(s, "2026-10-01", "other-lane-1", "26157")
    assert Q.creates_on(s, "26157")


# ---- the basis window's stability recount (F4): the rule admission binds on, from the window's own record
def test_the_stability_recount_is_the_texts():
    rec = json.loads((REC / "sc2j_screen_reads.json").read_text())
    reads = rec["reads"]
    assert len(reads) == 40 and reads[0]["at"] == "2026-10-23T15:12:02Z" and reads[-1]["at"] == "2026-10-23T22:35:22Z"
    present, stable, prev = {}, {}, []
    for rd in reads:
        keys = {tuple(x) for x in rd["under_ceiling"]}
        for m, _ in keys:
            present[m] = present.get(m, 0) + 1
        hist = (prev + [keys])[-3:]
        if len(hist) == 3:
            for m, _ in set.intersection(*hist):
                stable[m] = stable.get(m, 0) + 1
        prev = hist[-2:]
    assert present == {"52780": 15, "145701": 2, "55337": 1, "36213": 1, "150147": 1, "19317": 2, "46990": 2}
    assert sum(1 for rd in reads if rd["under_ceiling"]) == 21
    assert sum(1 for rd in reads if any(rec["listed_power_w"].get(m) == 575.0 for m, _ in rd["under_ceiling"])) == 16
    assert "575 W under the $1.05/h ceiling in 16 of 40 reads (any under-ceiling offer in 21)" in AM7_FLAT
    assert stable == {"52780": 6}                       # every other machine: 0; without 52780, no admissible read
    assert "52780 in **6 of 40** reads" in AM7_FLAT and "**every other machine in 0**" in AM7_FLAT
    # the admission itself, replayed read by read over the window with each machine's listed power: only 52780 admits
    lp = rec["listed_power_w"]
    window = [[{"machine_id": m, "offer_id": o, "usd_h": 1.0, "disk_gb": 320, "listed_power_w": lp.get(m)}
               for m, o in rd["under_ceiling"]] for rd in reads]
    hits = {Q.admit([], window[:i + 1], False, clocks={}).get("machine") for i in range(len(window))} - {None}
    assert hits == {"52780"}
    no52780 = [[o for o in rd if o["machine_id"] != "52780"] for rd in window]
    assert all(Q.admit([], no52780[:i + 1], False, clocks={})["verdict"] == "NONE" for i in range(len(no52780)))
    assert lp["55337"] == 575.0 and "its one logged offer lists 575 W" in AM7_FLAT


# ---- the card checks
@pytest.mark.parametrize("csv,want", [("575.00 W, 575.00 W, 600.00 W", "PASS"), ("575 W, 575 W, 575 W", "PASS"),
                                      ("574.99 W, 575.00 W, 600.00 W", "REFUSE"), ("[N/A], 575.00 W, 600.00 W", "UNREAD"),
                                      ("575.00 W, 575.00 W", "UNREAD"), ("", "UNREAD")])
def test_the_power_parse_three_ways(csv, want):
    assert Q.power(csv)["verdict"] == want


SC2J_PROVE2_POWER_LINE = "power.limit,clocks.max.sm 575.00 W, 3105 MHz"   # sc2j-prove-2's forensics.txt, line 3
SC2J_PROVE2_HOST_FACTS_SHA256 = "48882954f8694a30887409ecbdd06eb5b49535496d80d970320fe0dc67e9cf56"


def test_the_post_proof_judge_on_a_real_proofs_records():
    raw = (FIX / "sc2j-prove-2_host_facts.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SC2J_PROVE2_HOST_FACTS_SHA256
    hf = json.loads(raw)
    forensics = "GPU 0: NVIDIA GeForce RTX 5090\n" + SC2J_PROVE2_POWER_LINE + "\n"
    qp = {"enforced_w": 575.0}                                                    # the qualify step's power.json
    assert Q.judge_proof(forensics, hf, qp)["verdict"] == "PASS"
    assert Q.judge_proof(forensics, dict(hf, gpu_power_limit_w=450.0), qp)["verdict"] == "REFUSE"
    assert Q.judge_proof(forensics, None, qp)["verdict"] == "UNREAD"                 # host_facts.json deleted


# ---- the wrapper, on a stubbed box
CLEAN = "0x0000000000000000"


def _tree(tmp_path, drive_rc=0):
    """A copy of the repo pieces the wrapper reads, with bench/sc1/sc1_drive.sh a stub that records its call and writes
    the proof's forensics line."""
    t = tmp_path / "repo"
    for d in ("sc2", "sc1", "common"):
        (t / "bench" / d).mkdir(parents=True)
    for f in ("sc2_qualify_drive.sh", "sc2_qualify.py"):
        shutil.copy(SC2 / f, t / "bench" / "sc2" / f)
    shutil.copy(REPO / "bench" / "common" / "card_clock.py", t / "bench" / "common" / "card_clock.py")
    (t / "bench" / "sc1" / "sc1_drive.sh").write_text(
        '#!/bin/bash\necho DRIVE_CALLED >> "$DRIVE_LOG"\nmkdir -p "$E4B_RENT_RUN_DIR/sc1"\n'
        '[ -n "$PROOF_W" ] && printf "power.limit,clocks.max.sm %s W, 3105 MHz\\n" "$PROOF_W" > "$E4B_RENT_RUN_DIR/sc1/forensics.txt"\n'
        '[ "${NO_NONCE:-0}" = 1 ] || echo nn > "$E4B_RENT_RUN_DIR/sc1/SC1_RUN_NONCE"\n'
        '[ "${EXIT_FILE-0}" = MISSING ] || echo "${EXIT_FILE-0}" > "$E4B_RENT_RUN_DIR/sc1/SC1_EXIT_CODE.nn"\n'
        '[ "${NO_MARKER:-0}" = 1 ] || : > "$E4B_RENT_RUN_DIR/sc1/SC1_SUCCESS.nn"\n'
        '[ "${PLANT_EMPTY:-0}" = 1 ] && { : > "$E4B_RENT_RUN_DIR/sc1/SC1_SUCCESS."; echo 0 > "$E4B_RENT_RUN_DIR/sc1/SC1_EXIT_CODE."; }\n'
        f'exit {drive_rc}\n')
    return t


def _stubs(tmp_path, power, burn_rc, burn_refusal, sampler):
    b, box = tmp_path / "bin", tmp_path / "box"
    b.mkdir()
    (box / "q").mkdir(parents=True)
    smp = tmp_path / "sampler.src"
    smp.write_text(sampler if sampler is not None else "")
    (b / "ssh").write_text(f'''#!/bin/bash
for a in "$@"; do last=$a; done
case "$last" in
  nvidia-smi\\ --query-gpu=enforced*) echo "{power}";;
  *card_clock.py\\ burn*) echo {burn_rc} > {box}/q/burn.rc; echo "burn" > {box}/q/card_clock.log
     echo '{{"refusal": "{burn_refusal}"}}' > {box}/q/card_clock.json;;
  *-lms\\ 1000*) {"cp " + str(smp) + " " + str(box) + "/q/sampler.csv" if sampler is not None else ":"};;
  *) :;;
esac
''')
    (b / "scp").write_text("#!/bin/bash\nexit 0\n")
    (b / "rsync").write_text(f'#!/bin/bash\nfor a in "$@"; do last=$a; done\nmkdir -p "$last"; cp {box}/q/* "$last" 2>/dev/null; exit 0\n')
    for f in b.iterdir():
        os.chmod(f, 0o755)
    return b


HDR = "timestamp, clocks.sm [MHz], clocks_event_reasons.active, power.draw [W]\n"


def _sampler(n=30, mask=CLEAN, bad_tail=None):
    rows = "".join(f"2026/10/26 10:00:{i % 60:02d}.000, 2800 MHz, {CLEAN}, 540.00 W\n" for i in range(n - 1))
    return HDR + rows + f"2026/10/26 10:01:00.000, 2800 MHz, {mask}, 540.00 W\n" + (bad_tail or "")


def _wrap(tmp_path, power="575.00 W, 575.00 W, 600.00 W", burn_rc=0, burn_refusal="", drive_rc=0, prove="1",
          proof_w="575.00", host_facts=575.0, sampler=None, box="N", no_marker=False, exit_file="0", no_nonce=False,
          plant_empty=False):
    sampler = _sampler() if sampler is None else sampler
    t, b = _tree(tmp_path, drive_rc), _stubs(tmp_path, power, burn_rc, burn_refusal, None if sampler == "MISSING" else sampler)
    run = tmp_path / "run"
    run.mkdir()
    if host_facts is not None:
        (run / "host_facts.json").write_text(json.dumps({"gpu_power_limit_w": host_facts}))
    env = {"PATH": f"{b}:{os.environ['PATH']}", "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1",
           "E4B_RENT_SSH_OPTS": "-o X=1", "E4B_RENT_RUN_DIR": str(run), "E4B_RENT_RUN_ID": "sc2f-prove-11",
           "E4B_RENT_DEADLINE_EPOCH": "9999999999", "E4B_RENT_INSTANCE_ID": "1", "SC1_BOX": box, "SC1_PROVE": prove,
           "DRIVE_LOG": str(tmp_path / "drive.log"), "PROOF_W": proof_w or "", "NO_MARKER": "1" if no_marker else "0",
           "EXIT_FILE": exit_file, "NO_NONCE": "1" if no_nonce else "0",
           "PLANT_EMPTY": "1" if plant_empty else "0"}
    out = subprocess.run(["bash", str(t / "bench" / "sc2" / "sc2_qualify_drive.sh")], capture_output=True, text=True, env=env)
    return out, (tmp_path / "drive.log").exists(), run


def test_a_400_w_card_is_refused_before_anything(tmp_path):
    out, drove, run = _wrap(tmp_path, power="400.00 W, 575.00 W, 600.00 W")
    assert out.returncode == 17 and not drove, out.stdout + out.stderr
    assert json.loads((run / "qualify" / "power.json").read_text())["enforced_w"] == 400.0


def test_an_unreadable_power_read_is_a_harness_fault(tmp_path):
    out, drove, _ = _wrap(tmp_path, power="[N/A], [N/A], [N/A]")
    assert out.returncode == 25 and not drove, out.stdout + out.stderr


def test_a_slow_card_is_refused_with_18(tmp_path):
    out, drove, _ = _wrap(tmp_path, burn_rc=18, burn_refusal="below_floor")
    assert out.returncode == 18 and not drove, out.stdout + out.stderr


def test_a_burn_that_could_not_read_the_card_is_a_harness_fault(tmp_path):
    out, drove, _ = _wrap(tmp_path, burn_rc=18, burn_refusal="probe_error")
    assert out.returncode == 25 and not drove, out.stdout + out.stderr


def test_a_qualified_proof_runs_the_driver_and_is_judged_twice(tmp_path):
    out, drove, run = _wrap(tmp_path)
    assert out.returncode == 0 and drove, out.stdout + out.stderr
    assert json.loads((run / "qualify" / "run_judge.json").read_text())["verdict"] == "PASS"
    assert json.loads((run / "qualify" / "proof_judge.json").read_text())["verdict"] == "PASS"


@pytest.mark.parametrize("prove", ["1", "0"])
def test_a_capped_sample_is_a_host_fault_on_proofs_and_readings(tmp_path, prove):
    out, drove, run = _wrap(tmp_path, prove=prove, sampler=_sampler(mask="0x0000000000000040"))   # HwThermalSlowdown
    assert out.returncode == 18 and drove, out.stdout + out.stderr
    assert json.loads((run / "qualify" / "run_judge.json").read_text())["refusal"] == "below_floor"


@pytest.mark.parametrize("prove", ["1", "0"])
def test_a_missing_sampler_is_a_harness_fault_on_proofs_and_readings(tmp_path, prove):
    out, drove, _ = _wrap(tmp_path, prove=prove, sampler="MISSING")
    assert out.returncode == 25 and drove, out.stdout + out.stderr


def test_a_clean_reading_is_judged_on_its_sampler_only(tmp_path):
    out, drove, run = _wrap(tmp_path, prove="0", proof_w="", host_facts=None)
    assert out.returncode == 0 and drove and not (run / "qualify" / "proof_judge.json").exists()


def test_a_proof_without_host_facts_is_a_harness_fault_never_a_pass(tmp_path):
    out, drove, run = _wrap(tmp_path, host_facts=None)
    assert out.returncode == 25 and drove, out.stdout + out.stderr
    assert json.loads((run / "qualify" / "proof_judge.json").read_text())["verdict"] == "UNREAD"


def test_a_proof_whose_host_facts_read_450_w_is_a_host_fault(tmp_path):
    out, _, _ = _wrap(tmp_path, host_facts=450.0)
    assert out.returncode == 18


def test_a_clean_proof_on_a_500_w_forensics_record_is_a_host_fault(tmp_path):
    out, _, _ = _wrap(tmp_path, proof_w="500.00")
    assert out.returncode == 18


def test_a_failed_run_keeps_its_own_code(tmp_path):
    out, drove, run = _wrap(tmp_path, drive_rc=23, proof_w="500.00", sampler=_sampler(mask="0x0000000000000040"))
    assert out.returncode == 23 and drove
    assert (run / "qualify" / "run_judge.json").exists()            # judged for the record only


def test_a_symlink_in_the_fetched_records_is_refused_before_any_read(tmp_path):
    """bench/common rule 8: the box plants a link among its qualify records; the fetch keeps it (a stub rsync that
    copies links), and the wrapper refuses with the harness code before reading anything."""
    t, b = _tree(tmp_path), _stubs(tmp_path, "575.00 W, 575.00 W, 600.00 W", 0, "", _sampler())
    (b / "rsync").write_text(f'#!/bin/bash\nfor a in "$@"; do last=$a; done\nmkdir -p "$last"; cp {tmp_path}/box/q/* "$last" 2>/dev/null; '
                             'ln -sf /etc/hosts "$last/card_clock.json"; exit 0\n')
    os.chmod(b / "rsync", 0o755)
    run = tmp_path / "run"
    run.mkdir()
    env = {"PATH": f"{b}:{os.environ['PATH']}", "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "x",
           "E4B_RENT_RUN_DIR": str(run), "E4B_RENT_RUN_ID": "r", "E4B_RENT_DEADLINE_EPOCH": "9999999999",
           "E4B_RENT_INSTANCE_ID": "1", "SC1_BOX": "N", "DRIVE_LOG": str(tmp_path / "drive.log"), "PROOF_W": "575.00"}
    out = subprocess.run(["bash", str(t / "bench" / "sc2" / "sc2_qualify_drive.sh")], capture_output=True, text=True, env=env)
    assert out.returncode == 25 and "symlink" in out.stdout and not (tmp_path / "drive.log").exists(), out.stdout + out.stderr


@pytest.mark.parametrize("prove", ["1", "0"])
def test_a_driver_exit_0_without_the_success_marker_is_never_a_pass(tmp_path, prove):
    """sc1_drive.sh (held) exits the box's rc even when it is 0 without SC1_SUCCESS.<nonce>; the wrapper exits 24."""
    out, drove, run = _wrap(tmp_path, prove=prove, no_marker=True)
    assert out.returncode == 24 and drove and "success marker" in out.stdout, out.stdout + out.stderr
    assert (run / "qualify" / "run_judge.json").exists()            # the sampler is still fetched and judged, for the record


@pytest.mark.parametrize("case,kw", [("marker present, exit file reads 23", {"exit_file": "23"}),
                                     ("marker present, exit file missing", {"exit_file": "MISSING"}),
                                     ("marker present, exit file not a number", {"exit_file": "x"}),
                                     ("marker present, no SC1_RUN_NONCE", {"no_nonce": True}),
                                     ("no SC1_RUN_NONCE, but files named SC1_SUCCESS. and SC1_EXIT_CODE. (0) exist",
                                      {"no_nonce": True, "plant_empty": True})])
def test_each_clause_of_the_marker_check_binds_alone(tmp_path, case, kw):
    """One clause at a time: each of the nonce, the marker and the exit file reading 0 is required on its own."""
    out, _, _ = _wrap(tmp_path, **kw)
    assert out.returncode == 24, (case, out.stdout + out.stderr)


def test_the_complete_record_is_a_clean_run(tmp_path):
    out, _, _ = _wrap(tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr


def test_the_wrapper_refuses_boxes_outside_sc2f_h_i(tmp_path):
    out, drove, _ = _wrap(tmp_path, box="Q")
    assert out.returncode == 78 and not drove


def test_the_wrapper_passes_host_facts_always():
    src = (SC2 / "sc2_qualify_drive.sh").read_text()
    assert '--host-facts "$RUN_DIR/host_facts.json"' in src and "HF=(" not in src


# ---- the release pair
def test_boxes_n_o_p_on_gnf4_0_51_0_and_q_unchanged():
    assert 'case "$BOX" in N|O) GNF4_SHA=30adb1bfa76f95f9ce4a4c29734b8f36de907eca;; esac' in RUN
    assert ('case "$BOX" in P) GNF4_SHA=30adb1bfa76f95f9ce4a4c29734b8f36de907eca;; '
            'Q) GNF4_SHA=41ef086e6a3e2be5b158ff1bb38e0de67862a949;; esac') in RUN
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    assert "grouped-nf4-gemm.git@30adb1bfa76f95f9ce4a4c29734b8f36de907eca" in ci


def test_the_kit_hold_names_the_new_stack():
    sp = importlib.util.spec_from_file_location("skh", SC2 / "sc2_kit_hold.py")
    kh = importlib.util.module_from_spec(sp)
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
    for s in ("15:12Z to 22:35Z (7.4 h, 40 reads", "only `offers_total`", "0 %, not the 5 %", "**The deadline is 72 h**",
              "whichever is later", "**The G3 GPU check (e4b#2133) must PASS", "**Box Q stays at `v0.50.0`**",
              "**P130's two knobs: neither default moved.**", "**e4b#2087 moves neither arm's path.**", "$12.42",
              "a failed burn setup (rc 9)", "rc 25, which voids the run and never excludes the machine"):
        assert s in AM7_FLAT, s
    for lane, n in (("SC2h", 5), ("SC2i", 4)):
        t = (SC2 / f"{lane}-PREREG.md").read_text()
        assert f"### Amendment {n} (registered before any {lane} run): SC2f Amendment 7's" in t
        assert "If one lane later moves alone, its\n  cross-lane read is VOID." in t
        assert "the same OPEN walk" in t
