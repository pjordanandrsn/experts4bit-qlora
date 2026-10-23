# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P231, lever B3: K33's GEMV at 2..8 tokens on Mixtral's served decode (bench/p231/PREREG-p231.md). The self-tests;
every runner call through its script's own argument check; the staged pins; the run ids across runner, driver and
PREREG; the grouped-nf4-gemm pin and the switch's boundary; the arms' palindrome; W8 reported only; the premise before
the fetch; the pre-install fit, its boot slack against the launcher's ssh wait, and the start marker's place; every
pre-start exit code classified. (Private names are the repository guards' job, not this file's.)"""
import hashlib
import importlib.util
import os
import pathlib
import re
import shlex
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p231"
PREREG = (LANE / "PREREG-p231.md").read_text(encoding="utf-8")
RUN = (LANE / "p231_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p231_drive.sh").read_text(encoding="utf-8")
BENCH = REPO / "bench"
SOURCES = {"p231_run.sh": LANE, "p231_box.py": LANE, "p231_reduce.py": LANE, "p109_box.py": BENCH / "p109",
           "p115_quality.py": BENCH / "p115", "p115_reduce.py": BENCH / "p115", "p110_box.py": BENCH / "p110",
           "p108_box.py": BENCH / "p108", "p97_box.py": BENCH / "p97", "k8_bake.py": BENCH / "p39",
           "calib.json": BENCH / "p39", "test_decode_graph_buckets.py": REPO / "tests"}
PATHS = [str(LANE), str(BENCH / "p109"), str(BENCH / "p115"), str(BENCH / "p110"), str(BENCH / "p108"),
         str(BENCH / "p97")]
ENV = {**os.environ, "PYTHONPATH": os.pathsep.join(PATHS)}
GNF4_V050 = "41ef086e6a3e2be5b158ff1bb38e0de67862a949"      # grouped-nf4-gemm v0.50.0
RUN_IDS = ["p231-5090-1", "p231-5090-2"] + [f"p231-5090-r{i}" for i in range(1, 9)]
SSH_READY_S, STAGING_MARGIN_S = 420, 300                     # the launch manifest's ssh_ready_s; staging and the checks
GUARD_H = 4.8


def _load(name):
    spec = importlib.util.spec_from_file_location(name, LANE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    sys.path[:0] = PATHS
    try:
        spec.loader.exec_module(mod)
    finally:
        del sys.path[:len(PATHS)]
    return mod


@pytest.mark.parametrize("name", ["p231_reduce", "p231_box"])
def test_self_tests_pass(name):
    r = subprocess.run([sys.executable, str(LANE / f"{name}.py"), "--self-test"], capture_output=True, text=True,
                       timeout=300, cwd=str(LANE), env=ENV)
    assert r.returncode == 0 and "self-test OK" in r.stdout, r.stdout + r.stderr


def _joined_lines(text):
    out, cur = [], ""
    for line in text.splitlines():
        if line.rstrip().endswith("\\"):
            cur += line.rstrip()[:-1] + " "
            continue
        out.append(cur + line)
        cur = ""
    return out


def _calls(script):
    calls = []
    for line in _joined_lines(RUN):
        m = re.search(r"python \$W/" + re.escape(script) + r"(?P<rest>.*)", line)
        if not m:
            continue
        rest = re.split(r"\s(?:\d?>|>>|\||;|&&|\|\|)", " " + m["rest"])[0]
        calls.append([re.sub(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?", "X", tok) for tok in shlex.split(rest)])
    return calls


NUMERIC = {"--short": "32", "--long": "160", "--reps": "3", "--windows": "24", "--cont": "128", "--group": "4"}


@pytest.mark.parametrize("script", ["p231_box.py", "p231_reduce.py"])
def test_every_runner_call_passes_the_scripts_own_argument_check(script):
    calls = [c for c in _calls(script) if "--self-test" not in c]
    assert calls, script
    if script == "p231_box.py":     # every mode the runner drives is reached
        modes = {c[c.index("--mode") + 1] for c in calls if "--mode" in c}
        assert modes == {"rows", "fit", "speed", "quality"} and any("--prompts-only" in c for c in calls) \
            and any("--premise" in c for c in calls), calls
    for argv in calls:
        argv = [NUMERIC.get(argv[i - 1], a) if a == "X" and i else a for i, a in enumerate(argv)]
        r = subprocess.run([sys.executable, str(LANE / script), *argv, "--check-args"], capture_output=True, text=True,
                           timeout=60, cwd=str(LANE), env=ENV)
        assert r.returncode == 0 and "ARGS OK" in r.stdout, (argv, r.stdout + r.stderr)


def test_staged_pins_match_their_sources_and_the_driver_stages_them():
    lines = [ln.split() for ln in (LANE / "staged-p231.sha256").read_text(encoding="ascii").splitlines()
             if ln.strip() and not ln.startswith("#")]
    assert sorted(n for _, n in lines) == sorted(SOURCES)
    for want, name in lines:
        got = hashlib.sha256((SOURCES[name] / name).read_bytes()).hexdigest()
        assert got == want, f"{name}: staged {want}, source {got}"
    stage = re.search(r'^STAGE="([^"]+)"', DRIVE, re.M).group(1).split()
    assert sorted(pathlib.PurePosixPath(s).name for s in stage) == sorted([*SOURCES, "staged-p231.sha256"])
    pre = RUN[:RUN.index("sha256sum -c staged-p231.sha256")]
    for name in SOURCES:
        if name != "p231_run.sh":
            assert name in pre, f"the runner does not check {name} is staged"


def test_run_ids_agree_across_runner_driver_and_prereg():
    case = "|".join(RUN_IDS) + ")"
    assert case in RUN and case in DRIVE
    for bad in ("p231-5090-3", "p231-5090-r9"):
        assert bad not in RUN and bad not in DRIVE, bad
    assert "`p231-5090-1`" in PREREG and "`p231-5090-r1` to `p231-5090-r8`" in PREREG


def test_the_runner_and_reducer_pin_gnf4_v050_mixtral_and_the_switch_boundary():
    R, B = _load("p231_reduce"), _load("p231_box")
    assert f"GNF4_SHA={GNF4_V050}" in RUN and R.GNF4_SHA == GNF4_V050
    assert 'md.version("grouped-nf4-gemm") == "0.50.0"' in RUN
    assert f"MIX_REV={R.REVS[R.MIXTRAL]}" in RUN and f"MIX={R.MIXTRAL};" in RUN
    assert 'ng._bw_engages(n, k, dev) for n, k in ((28672, 4096), (4096, 14336))' in RUN
    assert B.MIXTRAL["shapes"] == (28672, 4096, 4096, 14336) and B.KNOB == "E4B_NF4_SMALLM_BW"
    # the boundary the lane reads: unset, 2..8 tokens take device grouping; set, singleton groups; inert at 16
    assert "hr._collapsed_grouping(4, None) == (False, True)" in RUN
    assert "all(hr._collapsed_grouping(t, None) == (True, False) for t in (2, 4, 8))" in RUN
    assert "hr._collapsed_grouping(16, None) == (False, True)" in RUN
    assert R.SWITCHED_BUCKETS == ("2", "4", "8")


def test_the_arms_are_a_palindrome_and_differ_only_in_the_switch():
    R, B = _load("p231_reduce"), _load("p231_box")
    assert R.TAGS == ("B0a", "B1a", "B1b", "B0b") and "for TAG in B0a B1a B1b B0b; do" in RUN
    assert 'ARM=${TAG:0:2}; KN=""; [ "$ARM" = B1 ] && KN="E4B_NF4_SMALLM_BW=1"' in RUN
    assert B.ARMS == ("B0", "B1")
    unset = RUN[RUN.index("unset E4B_SERVE_EXP_INT4"):RUN.index(": > summary.txt")]
    assert "E4B_NF4_SMALLM_BW" in unset and "GNF4_GEMV_BW " in unset     # B0 is the shipped default


def test_w8_is_reported_only_and_the_bars_are_registered():
    R = _load("p231_reduce")
    assert R.RULED == ("W2", "W4") and R.CONTROLS == ("W1", "W16") and "W8" not in R.RULED + R.CONTROLS
    assert "W8" not in R.SELF_SMALL_W and R.WORKLOADS == ("W1", "W2", "W4", "W8", "W16")
    assert (R.SELF_SMALL, R.SELF_WIDE, R.GAIN_MIN, R.CONTROL_MIN) == ((0.985, 1.015), (0.96, 1.04), 1.03, 0.99)
    for s in ("[0.985, 1.015]", "[0.96, 1.04]", "1.03", "0.99", "8 tokens stay opt-in"):
        assert s in PREREG, s


def test_quality_is_read_at_group_4_only():
    R, B = _load("p231_reduce"), _load("p231_box")
    assert B.QUALITY_GROUP == 4 and R.GROUP == 4 and "GROUP=4" in RUN and "--group $GROUP" in RUN
    assert B.QUALITY_ARMS == {"B0": ("R", "rep", "half", "chunk", "mutant_scale"), "B1": ("ON",)}
    r = subprocess.run([sys.executable, str(LANE / "p231_box.py"), "--mode", "quality", "--out", "x", "--ref-dir", "y",
                        "--group", "2", "--check-args"], capture_output=True, text=True, timeout=60, cwd=str(LANE),
                       env=ENV)
    assert r.returncode == 2 and "ARGS BAD" in r.stdout, "a quality pass at any group but 4 is refused"


def test_function_fail_is_read_before_the_timing_rungs():
    src = (LANE / "p231_reduce.py").read_text(encoding="utf-8")
    body = src[src.index("    if fn:\n        out.update(verdict=\"FUNCTION_FAIL\""):]
    assert body.index("FUNCTION_FAIL") < body.index("NOISY") < body.index("QUALITY_FAIL") < body.index("SLOWER") \
        < body.index("LICENSED")
    assert PREREG.index("**FUNCTION_FAIL.**") < PREREG.index("**NOISY.**") < PREREG.index("**QUALITY_FAIL.**") \
        < PREREG.index("**SLOWER.**") < PREREG.index("**LICENSED.**")


def test_the_premise_runs_on_the_card_before_the_fetch():
    i_rows = RUN.index("python $W/p231_box.py --mode rows")
    i_served = RUN.index("python $W/p231_box.py --premise $W/logs/premise_served.xml")
    i_fetch = RUN.index("snapshot_download")
    assert RUN.index(": > $W/STARTED") < i_served < i_rows < i_fetch
    assert 'finish 25' in RUN[i_rows:i_fetch]
    assert "premise failed" in RUN[i_rows:i_fetch]


def test_the_pre_install_fit_meets_the_guard_with_boot_slack():
    fit = 600 + 3000 + 1500 + 600 + 4 * 900 + 2400 + 900 + 600
    fit_need = fit * 125 // 100
    guard_s = round(GUARD_H * 3600)
    assert fit == 13200 and fit_need == 16500 and guard_s == 17280
    assert guard_s - fit_need >= SSH_READY_S + STAGING_MARGIN_S, guard_s - fit_need       # slack 780 s
    assert GUARD_H <= 5.9 and abs(GUARD_H * 1.05 + (93 + 4.9) * 0.011 - 6.12) < 0.005   # the fetch + the probe's shard
    nonstart = (guard_s - fit_need) * 1.05 / 3600 + 4.9 * 0.011                # the boot term + the probe's shard
    assert nonstart <= 0.29
    assert abs(2 * 6.12 + (len(RUN_IDS) - 2) * 0.29 - 14.56) < 0.005 and 14.56 <= 15.0   # the ids bound the boxes at 10
    for s in ("13,200 s", "16,500 s", "Guard 4.8 h (17,280 s)", "780 s", "97.9 GB", "USD 6.12", "USD 0.29", "USD 14.56"):
        assert s in PREREG, s
    for need in ("NEED_FETCH=3000", "NEED_BAKE=1500", "NEED_FIT=600", "NEED_ARM=900", "NEED_QOFF=2400",
                 "NEED_QON=900", "INSTALL_TO_GATE_S=600", "RESERVE=600", "MIN_DL_MB_S=40"):
        assert need in RUN, need
    i_fit, i_start = RUN.index('[ "$LEFT" -ge "$FIT_NEED" ]'), RUN.index(": > $W/STARTED")
    assert RUN.index("finish 19") < i_fit < i_start < RUN.index("python -m pip install")
    # the probe's record is written on every box, refused or not, before the refusal reads it (the maintainer's (3))
    i_rec = RUN.index("[ -s $W/download_probe.json ] ||")
    assert RUN.index("json.dump(rec, open(os.environ[\"OUT\"]") < i_rec < RUN.index("finish 19")
    assert "3 consecutive rc 19" in PREREG and "675 s" in PREREG and "`download_probe.json`" in PREREG
    assert i_start < RUN.index("python $W/p231_reduce.py")


def test_every_exit_before_the_start_is_classified():
    R = _load("p231_reduce")
    pre = RUN[:RUN.index(": > $W/STARTED")]
    codes = {int(c) for c in re.findall(r"finish (\d+)", pre)} - {130}   # 130 is the trap's
    assert codes, "no pre-start exits found"
    unknown = sorted(c for c in codes if c not in R.HOST_CODES and c not in R.FIT_CODES and c not in R.KIT_CODES)
    assert not unknown, f"pre-start exit codes the classification does not name: {unknown}"
    assert {13, 15, 16, 18, 19} == set(R.HOST_CODES) and {41} == set(R.FIT_CODES)
    assert {9, 10, 28, 78, 130} <= set(R.KIT_CODES)
    for rc in R.HOST_CODES:
        assert R.classify(rc, False)["event"] == "HOST_REFUSED"
    fit = R.classify(41, False)
    assert fit["event"] == "FIT_REFUSED" and not fit["halt"] and not fit["started"]     # a slow boot never halts the lane
    post = RUN[RUN.index(": > $W/STARTED"):]
    for rc in (25, 31):
        assert f"finish {rc}" in post and R.classify(rc, True)["event"] == "HALT" and R.classify(rc, True)["halt"]
    assert R.classify(0, True)["event"] == "ANSWER"
    assert R.classify(27, True)["event"] == "NO_ANSWER"


#: P227's reading (bench/p227/receipts/p227-5090-1): e4b's ROUTE_SEEN and grouped-nf4-gemm v0.50.0's dispatch tally as
#: the engine wrote them, for the two routes P231 grades, at Mixtral's shapes on an RTX 5090 -- K (the tile route, B0's)
#: and B (E4B_NF4_SMALLM_BW=1, B1's). Pinned by the receipt's own SHA256SUMS line.
P227_RECEIPT = REPO / "bench" / "p227" / "receipts" / "p227-5090-1" / "bench_p227.json"
P227_SHA256 = "76750467dfdf552609cfd821ab3bd04ddeeb111d365d5a43689e639963a8d3dd"


def _p227_engaged():
    raw = P227_RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == P227_SHA256, "P227's receipt changed"
    import json
    d = json.loads(raw)
    assert d["gnf4"] == "0.50.0" and d["sm_count"] >= 160
    return {t: d["families"]["mixtral"]["cells"][t]["engaged"] for t in ("2", "4", "8")}


def _as_bucket(rec, R):
    """One real P227 record as a P231 bucket entry: its route tally verbatim, and its dispatch counters picked by name
    (P227 stored them beside the route; P231's box stores them under "dispatch"). No value is typed."""
    return {"route": rec["route"], "dispatch": {k: rec[k] for k in R.GNF4_DISPATCH_KEYS}}


def test_the_reducer_reads_real_engine_records():
    """The maintainer's check after (F): the gates run on ROUTE_SEEN and dispatch records real code produced, not on
    fixtures typed to match a belief about their format."""
    R = _load("p231_reduce")
    eng = _p227_engaged()
    assert all("|" in k for t in eng for arm in ("K", "B") for k in eng[t][arm]["route"]), "real keys carry the row class"
    for t in eng:                                    # each real record, read the way P231 reads it
        k, b = _as_bucket(eng[t]["K"], R), _as_bucket(eng[t]["B"], R)
        assert R._routes(k["route"]) == {"nf4_k25": 8} and R._bw_counted(k["dispatch"]) == 0
        assert R._routes(b["route"]) == {"nf4_singleton": 8} and R._bw_counted(b["dispatch"]) == 24
    # whole per-bucket records assembled ONLY from real entries: B (singleton + K33) where the rule registers the
    # per-row route, K (K25, K33 counted at 0) where it registers the tile route
    single, tile = _as_bucket(eng["2"]["B"], R), _as_bucket(eng["2"]["K"], R)
    by_t = {t: (_as_bucket(eng[t]["B"], R), _as_bucket(eng[t]["K"], R)) for t in eng}
    b1 = {"1": single, **{t: by_t[t][0] for t in eng}, "16": tile}
    b0 = {"1": single, **{t: by_t[t][1] for t in eng}, "16": tile}
    assert R.bucket_faults("B1a", {"by_bucket": b1}) == [] and R.bucket_faults("B0a", {"by_bucket": b0}) == []
    assert R.bucket_faults("B0a", {"by_bucket": b1}) and R.bucket_faults("B1a", {"by_bucket": b0})   # swapped: VOID
    # and through the whole reducer: the LICENSED fixture with its per-bucket records replaced by the real ones
    base = [[r * 100 + i for i in range(8)] for r in range(16)]
    rates = {"B0": {"W1": 50.0, "W2": 60.0, "W4": 80.0, "W8": 100.0, "W16": 140.0},
             "B1": {"W1": 50.0, "W2": 70.0, "W4": 90.0, "W8": 98.0, "W16": 140.0}}
    arms = {t: R._fake_arm(t, rates[t[:2]], base, byb=b1 if t.startswith("B1") else b0) for t in R.TAGS}
    on, off = R._fake_quality("on"), R._fake_quality("off")
    on["route_measure"], on["dispatch_measure"] = eng["4"]["B"]["route"], _as_bucket(eng["4"]["B"], R)["dispatch"]
    off["route_measure"], off["dispatch_measure"] = eng["4"]["K"]["route"], _as_bucket(eng["4"]["K"], R)["dispatch"]
    assert R.reduce(arms, off, on, R.E, fit={"status": "ok"})["verdict"] == "LICENSED"
    # a real record with one counter removed is refused: absence never reads as the negative
    partial = dict(tile, dispatch={k: v for k, v in tile["dispatch"].items() if k != "bw_splitk"})
    assert R.bucket_faults("B0a", {"by_bucket": {**b0, "16": partial}})


#: Review's real records from other lanes' committed receipts, each sha256-pinned as committed. Every dict below is read
#: verbatim from its receipt; none is subset or typed.
REAL = {
    "p226_k25": ("bench/p226/receipts/p226-5090-1/capture.json",
                 "7be6cdaac6c5f12fcb6c70c61708a834a68e1766eb34aca44b20ef5601258451"),
    "sv3_mixed": ("bench/sv3/receipts/sv3-5090-1/q36_g16.json",
                  "e641d49999e4c60c55e896b524bc1203d345c1e13a125ac4a9d8df3b6156d1b6"),
    "p115d_k33": ("bench/p115/receipts/p115d-5090-1/arm_D0a.json",
                  "7cce4e9cdead31c996ddd487d19ec69fcc01ccdf23510ffe0b63c38c488de307"),
    "fam_zero": ("bench/fam/receipts/fam-gptoss-1/fam_gptoss_anchor_ON_auto.json",
                 "f570fe4cf247a549e9fe7a03783b1e54c397f3da7a18da6df0bb3d0c57afd9af"),
}


def _real(name):
    import json
    path, sha = REAL[name]
    raw = (REPO / path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sha, f"{path} changed"
    return json.loads(raw)


def test_the_bucket_gates_on_other_lanes_real_records():
    """Review's set: a pure-K25 route tally (P226), a MIXED singleton + K25 tally (SV3), a dispatch tally with K33
    (P115d) and an all-zero one (fam). Pure singleton is P227's verbatim B record (test above)."""
    R = _load("p231_reduce")
    k25 = _real("p226_k25")["route"]["route_seen_in_window"]
    mixed = _real("sv3_mixed")["expert_routes_seen"]
    k33 = _real("p115d_k33")["dispatch_build"]
    zero = _real("fam_zero")["cells"]["wikitext|12|A"]["dispatch"]
    single = _p227_engaged()["2"]["B"]["route"]
    assert set(zero) == set(R.GNF4_DISPATCH_KEYS) == set(k33), "the receipts name exactly the counters P231 requires"
    assert R._routes(mixed) == {"nf4_singleton": 160, "nf4_k25": 480, "nf4_mtile_captured": 1280}
    tile, per_row = {"route": k25, "dispatch": zero}, {"route": single, "dispatch": k33}
    b0 = {"1": per_row, "2": tile, "4": tile, "8": tile, "16": tile}
    b1 = {"1": per_row, "2": per_row, "4": per_row, "8": per_row, "16": tile}
    assert R.bucket_faults("B0a", {"by_bucket": b0}) == [] and R.bucket_faults("B1a", {"by_bucket": b1}) == []
    # a real mixed tally where B1 registers the per-row route alone: refused
    assert R.bucket_faults("B1a", {"by_bucket": {**b1, "4": {"route": mixed, "dispatch": k33}}})
    # a real K33 dispatch where B0 registers the tile route: refused
    assert R.bucket_faults("B0a", {"by_bucket": {**b0, "4": {"route": k25, "dispatch": k33}}})
    # the quality gates on the same real records: ON on the mixed tally is refused, OFF with K33 is refused
    on, off = R._fake_quality("on"), R._fake_quality("off")
    on["route_measure"], on["dispatch_measure"] = mixed, k33
    assert R.quality_faults(R._fake_quality("off"), on, R.E)
    off["route_measure"], off["dispatch_measure"] = k25, k33
    assert R.quality_faults(off, R._fake_quality("on"), R.E)
    off["dispatch_measure"] = zero
    on["route_measure"] = single
    assert R.quality_faults(off, on, R.E) == []


def test_the_fetch_imports_no_links():
    """Amendment 1 (bench/common rule 8): every rsync in the driver passes --no-links, so a link left on a rented box is
    never copied to the controller, where the receipt writers would follow it."""
    calls = [ln for ln in _joined_lines(DRIVE) if re.match(r"\s*rsync\b", ln)]
    assert calls, "the driver's fetch is an rsync"
    for ln in calls:
        assert "--no-links" in ln.split(), ln
    assert "## Amendment 1 (pre-data): the fetch imports no links" in PREREG


def test_the_runner_finishes_on_the_verdict():
    assert ('case "$V" in LICENSED|SLOWER|QUALITY_FAIL|FUNCTION_FAIL) finish 0;; *) say "P231 $V -- no reading"; '
            'finish 27;; esac') in RUN
    R = _load("p231_reduce")
    assert R.ANSWERS == ("LICENSED", "SLOWER", "QUALITY_FAIL", "FUNCTION_FAIL")
