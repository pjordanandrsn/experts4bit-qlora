# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""P227, the per-row bandwidth GEMV against K25 at 2 to 8 decode tokens on two families (bench/p227/PREREG-p227.md).
The staged pins; the routing (Qwen3's re-derived from P153's replay, Mixtral's from P222's seed), with the table the
registration states; the arms and their call against the collapse's own; the runner's pins, refusals, tripwire, premise
and guard fit with 25 % headroom; the drive's refusals; no instance id or host name on the box; the self-tests and the
mutants they must kill."""
import hashlib
import json
import importlib.util
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LANE = REPO / "bench" / "p227"
REPLAY = REPO / "bench" / "p153" / "router-replay" / "rr.json.pt"
SOURCES = {"p227_bench.py": LANE, "p227_reduce.py": LANE, "p227_run.sh": LANE, "rr.json.pt": REPLAY.parent,
           "test_nf4_smallm_bw_gpu.py": REPO / "tests", "test_gemv_bw_served_gpu.py": REPO / "tests"}
RUN = (LANE / "p227_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p227_drive.sh").read_text(encoding="utf-8")
BENCH = (LANE / "p227_bench.py").read_text(encoding="utf-8")
PREREG = (LANE / "PREREG-p227.md").read_text(encoding="utf-8")
GUARD_S, BOOT_S, HEADROOM = 4500, 300, 1.25


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


R = _load("p227_reduce_under_test", LANE / "p227_reduce.py")


def test_every_staged_file_is_pinned_and_matches():
    pins = {}
    for line in (LANE / "staged-p227.sha256").read_text(encoding="ascii").splitlines():
        want, name = line.split()
        pins[name] = want
    assert set(pins) == set(SOURCES)
    for name, where in SOURCES.items():
        assert hashlib.sha256((where / name).read_bytes()).hexdigest() == pins[name], name
    assert pins["rr.json.pt"] == R.REPLAY_SHA256


def test_the_reducer_and_bench_self_tests_pass():
    R.self_test()
    r = subprocess.run([sys.executable, str(LANE / "p227_bench.py"), "--self-test"], capture_output=True, text=True)
    assert r.returncode == 0 and "self-test OK" in r.stdout, r.stdout + r.stderr


def test_qwen3_routing_pins_re_derive_from_p153s_replay():
    torch = pytest.importorskip("torch")
    picks = torch.load(REPLAY, weights_only=False)["picks"]
    assert tuple(picks.shape) == (48, 32, 128, 8)
    for T in R.TS:
        tab = [[[[int(e) for e in picks[li, R.QWEN3_ROWS0 + r, s].tolist()] for r in range(T)] for s in R.STEPS]
               for li in range(48)]
        assert R.route_sha256(tab) == R.ROUTE_SHA256["qwen3"][T], T
        assert sum(len({e for row in tab[li][s] for e in row}) for li in range(48) for s in R.STEPS) \
            == R.DISTINCT_TOTAL["qwen3"][T], T


def test_mixtral_routing_is_p222s_draw_byte_for_byte():
    p222 = _load("p222_reduce_for_p227", REPO / "bench" / "p222" / "p222_reduce.py")
    assert R.mixtral_picks() == p222.picks(2221)
    assert R.MIXTRAL_SEED == p222.SEEDS["B"]


def test_the_prereg_table_is_the_pinned_tables():
    for fam, L in (("qwen3", 48), ("mixtral", 8)):
        k = R.FAMILIES[fam]["topk"]
        for T in R.TS:
            mean = R.DISTINCT_TOTAL[fam][T] / (L * len(R.STEPS))
            assert f"| {T} | {T * k} | {mean:.2f} | {T * k / mean:.2f} |" in PREREG, (fam, T)


def test_the_registered_constants():
    assert (R.BAR, R.B2_TRIGGER, R.RULED_TS, R.TS, R.SELF_BAND, R.TOL, R.MIN_POWER_W) == \
        (0.70, 1.25, (2, 4), (2, 4, 8), (0.99, 1.01), 1.5e-2, 575)
    assert R.ARMS == ["K", "K2", "B"] and R.SPACER == "S" and R.ITERS == 10 and R.STEPS == tuple(range(60))
    assert R.FAMILIES["qwen3"]["k25"] == {"E4B_K25_LUT": "prmt", "E4B_K25_PLAN": "c1"}
    assert R.FAMILIES["mixtral"]["k25"] == {"E4B_K25_LUT": "tree", "E4B_K25_PLAN": "today"}
    assert R.FAMILIES["qwen3"]["shapes"] == [1536, 2048, 2048, 768] and R.FAMILIES["qwen3"]["layers"] == 48
    assert R.FAMILIES["mixtral"]["shapes"] == [28672, 4096, 4096, 14336] and R.FAMILIES["mixtral"]["layers"] == 8
    for s in ("**<= 0.70**", "**> 1.25**", "0.99-1.01", "1.5e-2", "under 575 W", "seed 2221", "`3c59e72f...`"):
        assert s in PREREG, s


def test_k_is_each_familys_served_default_in_the_tripwire():
    assert 'for mt, lut, plan in (("qwen3_moe", "prmt", "c1"), ("mixtral", "tree", "today")):' in RUN
    assert "assert hr.NF4_SMALLM_BW_MAX_T == 8 and callable(hr.resolve_nf4_smallm_bw)" in RUN
    assert "assert hr._nf4_smallm_bw_env() is False" in RUN


def test_the_runner_pins_the_kernel_package_and_refuses_before_the_install():
    assert "GNF4_SHA=41ef086e6a3e2be5b158ff1bb38e0de67862a949" in RUN and 'md.version("grouped-nf4-gemm") == "0.50.0"' in RUN
    p222 = (REPO / "bench" / "p222" / "p222_run.sh").read_text(encoding="utf-8")
    for var in ("NF4_SMALLM_SHA256", "INT4_B32_SHA256", "NF4_GROUPED_SHA256"):
        want = re.search(rf"^{var}=([0-9a-f]{{64}})", p222, re.M).group(1)
        assert f"{var}={want}" in RUN, var                          # P222's kernel files, byte for byte
    assert "MIN_POWER_W=575 " in RUN and "finish 17; }" in RUN
    for rid in ("p227-5090-1|p227-5090-2|p227-5090-3|", "p227-shake-1|p227-shake-2|p227-shake-3) SHAKE=1;;"):
        assert rid in RUN
    assert "E4B_K25_LUT E4B_K25_PLAN E4B_NF4_SMALLM_BW" in RUN, "every knob the lane moves starts unset"
    order = ["sha256sum -c staged-p227.sha256", "PL=$(nvidia-smi --query-gpu=power.limit", '[ "$LEFT" -ge "$FIT_NEED" ]',
             "pipx logs/pip_e4b.log", "TRIPWIRE FAIL", "grep -qE '^25 passed", "p227_bench.py --replay",
             "p227_reduce.py --bench"]
    at = [RUN.index(x) for x in order]
    assert at == sorted(at), dict(zip(order, at))
    assert "test_nf4_smallm_bw_gpu.py test_gemv_bw_served_gpu.py" in RUN and "**25 passed**" in PREREG
    # only a shakedown forces K33's gate, after the premise; the reducer VOIDs a reading that carries it
    assert RUN.index('[ "$SHAKE" = 1 ] && export GNF4_GEMV_BW=1') > RUN.index("grep -qE '^25 passed")
    assert 'if not (shake and k == "GNF4_GEMV_BW"):' in BENCH and "GNF4_GEMV_BW=1" in PREREG


def test_the_guard_fit_has_25_percent_headroom_and_the_prereg_states_it():
    get = lambda v: int(re.search(rf"^{v}=(\d+)", RUN, re.M).group(1))  # noqa: E731
    install, need, reserve = get("INSTALL_TO_GATE_S"), get("NEED_BENCH"), get("RESERVE")
    assert (install, need, reserve) == (600, 1500, 600)
    fit = install + need + reserve
    assert fit == 2700 and int(fit * HEADROOM) == 3375 and (BOOT_S + fit) * HEADROOM == 3750 <= GUARD_S
    assert '[ "$SHAKE" = 1 ] && NEED_BENCH=600' in RUN
    for s in ("`NEED_BENCH` = **1500 s**", "`FIT` = 600 + 1500 + 600 = 2700 s", "2700 x 1.25 = 3375 s",
              "(boot 300 s + 2700 s) x 1.25 = 3750 s <= 4500 s (1.25 h)", "**$1.71**"):
        assert s in PREREG, s
    assert abs(1.25 * 1.05 + 10 * 0.04 - 1.71) < 0.005


SSH_READY_S, STAGING_MARGIN_S = 420, 120   # pod-launch's longest wait for ssh on Vast; the drive's staging margin


def test_the_slack_clears_a_slow_ssh_host():
    """The deadline runs from the create, so the guard minus the runner's FIT_NEED must hold pod-launch's ssh wait plus
    the staging, or a slow-ssh host halts the lane at the fit check (rc 41) before the install (P225's r1 took 443 s)."""
    get = lambda v: int(re.search(rf"^{v}=(\d+)", RUN, re.M).group(1))  # noqa: E731
    fit_need = (get("INSTALL_TO_GATE_S") + get("NEED_BENCH") + get("RESERVE")) * get("FIT_HEADROOM_PCT") // 100
    guard_s = round(float(re.search(r"\*\*guard ([\d.]+) h\*\*", PREREG).group(1)) * 3600)
    assert (guard_s, fit_need) == (GUARD_S, 3375)
    assert guard_s - fit_need == 1125 >= SSH_READY_S + STAGING_MARGIN_S
    assert "guard - FIT_NEED = 4500 - 3375 = 1125 s" in PREREG


def test_the_drive_never_creates_compute_and_refuses_what_is_not_registered():
    assert "p227-5090-[1-3]|p227-5090-r[1-8]|p227-shake-[1-3]) :;;" in DRIVE and 'refusing unregistered run ID' in DRIVE
    assert "the checkout is dirty" in DRIVE
    assert "shasum -a 256 -c staged-p227.sha256" in DRIVE
    for word in ("vast.py", "destroy", "/api/", "rent-watch"):
        assert word not in DRIVE.split("set -uo pipefail", 1)[1], word


def _accepted(src, pattern_re, ids):
    """The ids a script's own reading case accepts, run through bash with the pattern as written."""
    pat = re.search(pattern_re, src, re.M).group(1)
    script = f'case "$1" in {pat}) exit 0;; *) exit 9;; esac'
    return {rid for rid in ids if subprocess.run(["bash", "-c", script, "_", rid]).returncode == 0}


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_amendment_4_adds_exactly_the_retry_ids_r1_to_r8():
    """Amendment 4: the runner and the drive accept exactly -1..-3 and r1..r8 (labels, not budget); -4 and r9 refuse."""
    probe = [f"p227-5090-{x}" for x in ("0", "1", "2", "3", "4", "10", "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7",
                                       "r8", "r9", "r10", "r01", "R1", "r", "r1x")]
    want = {f"p227-5090-{x}" for x in ("1", "2", "3")} | {f"p227-5090-r{i}" for i in range(1, 9)}
    assert _accepted(RUN, r"^  (p227-5090-1\|[^)]*)\) :;;", probe) == want
    assert _accepted(DRIVE, r'^case "\$E4B_RENT_RUN_ID" in ([^)]*)\) :;;', probe) == want
    assert "**`p227-5090-r1` to `p227-5090-r8`**, for ANY box" in PREREG
    assert "**The ids are labels, not budget.**" in PREREG and "no code parses an id's number" in PREREG
    assert ("| **refused at create** (Amendment 4) | the provider refused the create, so nothing was created ($0) | no |"
            " no; counts toward no cap, uses one id |") in PREREG
    assert "(two consecutive HARNESS_ERROR, VOID or NOT_RUN readings stop the lane, as registered)" in PREREG


def test_the_drive_follows_pod_launchs_contract():
    """pod-launch runs a reading's drive without E4B_SHA (it proves the checkout HEAD is heads.e4b) and owns the run
    directory's receipt files: the drive derives the launch commit from the clean HEAD, refuses an override, and fetches
    the box tree into $RUN_DIR/p227/ (P226's form)."""
    assert 'if [ -n "${E4B_SHA:-}" ]; then say "refusing: E4B_SHA override; use the clean checkout HEAD"; exit 78; fi' in DRIVE
    assert 'E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)' in DRIVE
    assert "E4B_RENT_DEADLINE_EPOCH E4B_SHA; do" not in DRIVE
    assert 'OUT="$RUN_DIR/p227"' in DRIVE and '"$RUN_DIR/"' not in DRIVE
    assert 'RC=$(cat "$OUT/P227_EXIT_CODE.$NONCE"' in DRIVE


def test_no_instance_id_or_host_name_is_written_on_the_box():
    for text in (RUN, BENCH, DRIVE, PREREG):
        assert not re.search(r"\b(instance|machine|offer)[ _-]?(id)?\s*[:=]?\s*\d{5,}", text, re.I), "an id"
        assert "vast.ai" not in text and "ssh1." not in text


def _runner_read_count(verdict, tmp_path):
    """The runner's own READ one-liner, run on a verdict file."""
    code = re.search(r'^READ=\$\(python -c "(.*)"\)$', RUN, re.M).group(1)
    path = tmp_path / "verdict_p227.json"
    path.write_text(json.dumps(verdict), encoding="utf-8")
    r = subprocess.run([sys.executable, "-c", code.replace("$W/verdict_p227.json", path.as_posix())],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return int(r.stdout.strip())


def test_a_record_with_no_family_read_is_not_a_reading(tmp_path):
    """The runner's exit: 0 only when a family is READ -- PASS, MISS or TOLERANCE_FAIL (final, never re-read, not a
    strike) -- or both families SHAKEDOWN in a shakedown; otherwise rc 27."""
    assert '0/*) [ "${READ:-0}" -ge 1 ] && finish 0; say "P227 $V -- no family read"; finish 27;;' in RUN
    assert '"1/qwen3=SHAKEDOWN mixtral=SHAKEDOWN") finish 0;;' in RUN and "1/*) say" in RUN
    assert R.READ_VERDICTS == ("PASS", "MISS", "TOLERANCE_FAIL")
    # qwen3=TOLERANCE_FAIL beside mixtral=VOID: one family read, so the run finishes 0 (and is not a strike)
    q = R._fake()
    q["families"]["qwen3"]["cells"]["2"]["reps"][4]["b_rel_diff"] = 0.05
    q["families"]["mixtral"]["cells"]["8"]["reps"][1]["k2_identical"] = False
    v = R.verdict(q)
    assert v["verdict"] == "qwen3=TOLERANCE_FAIL mixtral=VOID", v["verdict"]
    assert _runner_read_count(v, tmp_path) == 1
    # every family VOID: nothing read, rc 27
    q["families"]["qwen3"]["cells"]["2"]["reps"][4]["k2_identical"] = False
    v = R.verdict(q)
    assert v["verdict"] == "qwen3=VOID mixtral=VOID" and _runner_read_count(v, tmp_path) == 0
    for s in ("no family is read (PASS, MISS or TOLERANCE_FAIL) is not a reading", "It is FINAL for that family",
              "re-reads only the families not yet read"):
        assert s in PREREG, s


def test_a_missing_or_unreadable_bench_record_reduces_to_void(tmp_path):
    for content in (None, "{not json"):
        bench = tmp_path / "bench_p227.json"
        if content is not None:
            bench.write_text(content, encoding="utf-8")
        out = tmp_path / "v.json"
        r = subprocess.run([sys.executable, str(LANE / "p227_reduce.py"), "--bench", str(bench), "--out", str(out)],
                           capture_output=True, text=True)
        assert r.returncode == 0 and "P227_VERDICT VOID" in r.stdout, r.stdout + r.stderr
        assert '"verdict": "VOID"' in out.read_text(encoding="utf-8")


REDUCER_MUTANTS = [
    ("BAR = 0.70 ", "BAR = 0.75 "),                                                      # a looser bar
    ('ok = all(cells[T]["b_over_k"]["ub"] <= BAR and cells[T]["b_over_k"]["ub_clock_adj"] <= BAR for T in RULED_TS)',
     'ok = all(cells[T]["b_over_k"]["median"] <= BAR for T in RULED_TS)'),               # the median, not the bounds
    ("RULED_TS = (2, 4) ", "RULED_TS = (2,) "),                                            # one T rules
    ('s["ub_clock_adj"] = s["ub"] * max(1.0, ratio)', 's["ub_clock_adj"] = s["ub"] * min(1.0, ratio)'),  # clock credit
    ("    if outside and not shake:\n", "    if False:\n"),                                # the self-pair band dropped
    ('if rep.get("k2_identical") is not True:', 'if False:'),                              # K2's bits unchecked
    ('bad = [T for T, c in cells.items() if c["b_rel_diff_max"] > TOL]', 'bad = []'),     # the tolerance dropped
    ('or any(b.get(x) for x in ("dotpad"', 'or False and any(b.get(x) for x in ("dotpad"'),   # B could be dot-pad
    ('if k.get("k25_lut") != {f"{lut}|le256": L} or k.get("k25_plan") != {f"{plan}|le256": L}:',
     'if False:'),                                                                       # K's tallies unchecked
]


@pytest.mark.parametrize("old, new", REDUCER_MUTANTS, ids=[m[0][:28] for m in REDUCER_MUTANTS])
def test_the_self_test_kills_each_reducer_mutant(tmp_path, old, new):
    src = (LANE / "p227_reduce.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    shutil.copy(LANE / "p227_reduce.py", tmp_path / "orig.py")
    (tmp_path / "p227_reduce.py").write_bytes(src.replace(old, new).encode("utf-8"))
    r = subprocess.run([sys.executable, str(tmp_path / "p227_reduce.py"), "--self-test"], capture_output=True, text=True)
    assert r.returncode != 0 and "FAILED" in r.stdout, (old, r.stdout)


# ---- the real record first (the maintainer's gate-test rule): p227-shake-2's bench record, sha256-pinned
FIXTURE = LANE / "fixtures" / "p227-shake-2" / "bench_p227.json"
FIXTURE_SHA256 = "35b6a65e3d3e436d19170c3e8d47a7fc91fb1e3fd8fe0e069cb94c701ba18e36"


def _real():
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256, "the pinned shakedown record changed"
    return json.loads(raw)


def test_the_real_shakedown_record_reduces_to_shakedown_with_every_check_passed():
    rec = _real()
    assert rec["shakedown"] is True and rec["gnf4_gemv_bw_env"] == "1" and rec["replay_sha256"] == R.REPLAY_SHA256
    v = R.verdict(rec)
    assert v["verdict"] == "qwen3=SHAKEDOWN mixtral=SHAKEDOWN", v
    for fam, L in (("qwen3", 4), ("mixtral", 2)):
        lut, plan = R.FAMILIES[fam]["k25"]["E4B_K25_LUT"], R.FAMILIES[fam]["k25"]["E4B_K25_PLAN"]
        for T in R.TS:
            eng = rec["families"][fam]["cells"][str(T)]["engaged"]
            assert eng["K"]["k25_lut"] == {f"{lut}|le256": L} and eng["K"]["k25_plan"] == {f"{plan}|le256": L}
            assert eng["B"]["bw_prmt32"] == 2 * L and eng["B"]["route"] == {"nf4_singleton|le256": L}


DEPARTURES = [
    (lambda r: r["families"]["qwen3"]["cells"]["2"]["engaged"]["K"].pop("k25_plan"), "K's plan tally missing"),
    (lambda r: r["families"]["qwen3"]["cells"]["4"]["engaged"]["K"].update(k25_lut={"tree|le256": 4}), "K on the tree"),
    (lambda r: r["families"]["qwen3"]["cells"]["8"]["engaged"]["B"].update(dotpad=1), "B on dot-pad"),
    (lambda r: r["families"]["qwen3"]["cells"]["2"]["reps"][3].update(k2_identical=False), "K2 moved a bit"),
    (lambda r: r["families"]["qwen3"]["cells"]["4"]["reps"][2].update(order=["B", "K2", "K"]), "the order"),
]


@pytest.mark.parametrize("depart, what", DEPARTURES, ids=[d[1] for d in DEPARTURES])
def test_each_departure_from_the_real_record_voids_that_family_only(depart, what):
    rec = _real()
    depart(rec)
    fv = R.verdict(rec)["families"]
    assert fv["qwen3"]["verdict"] == "VOID" and fv["mixtral"]["verdict"] == "SHAKEDOWN", (what, fv["qwen3"]["why"])


def test_the_old_self_reported_k_check_would_pass_the_real_records_departure(tmp_path):
    """Review 1, item 1: before e4b's own tallies were read, a K that ran the wrong plan passed. The mutant restores
    the old rule and must ACCEPT the departure the current reducer VOIDs."""
    src = (LANE / "p227_reduce.py").read_text(encoding="utf-8")
    old = 'if k.get("k25_lut") != {f"{lut}|le256": L} or k.get("k25_plan") != {f"{plan}|le256": L}:'
    assert src.count(old) == 1
    (tmp_path / "p227_reduce.py").write_bytes(src.replace(old, "if False:").encode("utf-8"))
    mut = _load("p227_reduce_old_rule", tmp_path / "p227_reduce.py")
    rec = _real()
    rec["families"]["qwen3"]["cells"]["4"]["engaged"]["K"]["k25_lut"] = {"tree|le256": 4}
    assert mut.verdict(rec)["families"]["qwen3"]["verdict"] == "SHAKEDOWN"
    assert R.verdict(rec)["families"]["qwen3"]["verdict"] == "VOID"
