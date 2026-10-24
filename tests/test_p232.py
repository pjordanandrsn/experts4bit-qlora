# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P232 (bench/p232/PREREG-p232.md; lever K, K2b), the served read of the int8 lm_head at 2, 4 and 8 rows
(E4B_INT8_HEAD_ROWS=8, grouped-nf4-gemm K37): the staged pins (the P109 / P115 / P39 helpers at P153's pins, P229's
premise in the repo's layout under t/ at P229's), the registered constants as the PREREG states them, the runner's order
(the refusals, the download probe and the guard fit before the install; P229's premise at 11 and 45; the warm-up and eight
arms; the three quality phases), the guard fit and its slack, the drive's pod-launch contract and refusals (and that its
fetch imports no link), the self-tests run as the box runs them (one directory), P153's pieces carried verbatim, and the
reducer and box mutants the self-tests must kill.

The real fixture (Amendment 2): p232-shake-1's twelve reducer inputs, sha256-pinned, reduce to SHAKEDOWN with every
integrity check passed; departures from them each VOID; made reading-shaped (shakedown off, the registered rep count)
they VOID on the 575 W floor alone, since that host's board was capped at 500 W, and a reducer without the power check
reads a verdict from them."""
import ast
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
B = REPO / "bench"
LANE = B / "p232"
#: staged path (as the box sees it) -> the file in this checkout
SOURCES = {"p232_run.sh": LANE / "p232_run.sh", "p232_box.py": LANE / "p232_box.py",
           "p232_reduce.py": LANE / "p232_reduce.py", "p115_quality.py": B / "p115" / "p115_quality.py",
           "p109_box.py": B / "p109" / "p109_box.py", "p110_box.py": B / "p110" / "p110_box.py",
           "p108_box.py": B / "p108" / "p108_box.py", "p97_box.py": B / "p97" / "p97_box.py",
           "k8_bake.py": B / "p39" / "k8_bake.py", "calib.json": B / "p39" / "calib.json",
           "t/tests/test_int8_head_bw.py": REPO / "tests" / "test_int8_head_bw.py",
           "t/tests/test_int8_head_rows_interp.py": REPO / "tests" / "test_int8_head_rows_interp.py",
           "t/tests/conftest.py": REPO / "tests" / "conftest.py",
           "t/bench/i8dense/i8_fakequant.py": B / "i8dense" / "i8_fakequant.py",
           "t/bench/p148/p148_kernels.py": B / "p148" / "p148_kernels.py"}
RUN = (LANE / "p232_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p232_drive.sh").read_text(encoding="utf-8")
PREREG = (LANE / "PREREG-p232.md").read_text(encoding="utf-8")
#: the e4b commit P232 read at (p232-5090-1: the Amendment 2 merge)
READ_AT = "f7589a02"
FLAT = " ".join(PREREG.split())     # the PREREG's wrapped lines joined: a phrase reads across a line break
NEW_FILES = [LANE / f for f in ("p232_run.sh", "p232_box.py", "p232_reduce.py", "p232_drive.sh", "staged-p232.sha256")] + [
    REPO / "tests" / "test_p232.py", REPO / "changelog.d" / "p232-kit.md"]


def _pins(path):
    out = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip() and not line.startswith("#"):
            want, name = line.split()
            out[name] = want
    return out


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return mod


R = _load("p232_reduce_under_test", LANE / "p232_reduce.py")


def _staged_sha(path, read_at):
    """sha256 of the bytes the kit staged from ``path``: the file in this checkout, or, for a repo file outside the
    lane that has moved on since the reading (tests/test_int8_head_bw.py, when E4B_INT8_HEAD_ROWS took its per-family
    default), the file at ``read_at``, the e4b commit the reading ran at. The pins are a record of what was staged, so
    they are never rewritten; a file outside the lane is held to them at that commit (CI checks out full history)."""
    here = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.is_relative_to(LANE):
        return here
    rel = path.relative_to(REPO).as_posix()
    out = subprocess.run(["git", "cat-file", "blob", f"{read_at}:{rel}"], cwd=REPO, capture_output=True)
    if out.returncode != 0:
        pytest.skip(f"{rel} at {read_at} is not in this checkout's history")
    return hashlib.sha256(out.stdout).hexdigest()


def _run(script, *args, cwd=None):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=180, cwd=cwd)


def test_every_staged_file_is_pinned_and_matches():
    pins = _pins(LANE / "staged-p232.sha256")
    assert list(pins) == list(SOURCES)
    for name, path in SOURCES.items():
        here = hashlib.sha256(path.read_bytes()).hexdigest()
        assert (here if here == pins[name] else _staged_sha(path, READ_AT)) == pins[name], name


def test_the_shared_pieces_are_at_p153s_and_p229s_pins():
    """The staged copies of P109's box, P115's instrument (and the boxes it imports) and P39's bake are byte-identical to
    the ones P153 read with; P229's premise files are at P229's pins."""
    mine, p153, p229 = _pins(LANE / "staged-p232.sha256"), _pins(B / "p153" / "staged-p153.sha256"), _pins(B / "p229" / "staged-p229.sha256")
    flat = {"p115_quality.py", "p109_box.py", "p110_box.py", "p108_box.py", "p97_box.py", "k8_bake.py", "calib.json"}
    assert flat <= set(p153) and all(mine[n] == p153[n] for n in flat)
    premise = {n for n in p229 if n.startswith("t/")}
    assert len(premise) == 5 and all(mine[n] == p229[n] for n in premise)
    assert set(mine) == {"p232_run.sh", "p232_box.py", "p232_reduce.py"} | flat | premise


def test_the_premise_counts_are_the_suites_own():
    """45 and 11 are what the staged suites hold (P229's premise): the CPU suite's items, and the interpreter file's 7
    plus its 4 CUDA-graph replays (each parametrized test counted once per case)."""
    cpu = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                          str(REPO / "tests" / "test_int8_head_bw.py")], capture_output=True, text=True, cwd=REPO)
    assert re.search(r"^45 tests collected", cpu.stdout, re.M), cpu.stdout[-500:]
    src = (REPO / "tests" / "test_int8_head_rows_interp.py").read_text(encoding="utf-8")
    assert src.count("\ndef test_") == 5 and '@pytest.mark.parametrize("rows", [1, 2, 3, 8])' in src \
        and '@pytest.mark.parametrize("rows", [1, 2, 4, 8])' in src            # 1 + 4 + 1 + 1 + 4 = 11
    assert "grep -qE '^11 passed" in RUN and "grep -qE '^45 passed" in RUN


@pytest.fixture(scope="module")
def box_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("p232_box")
    for name, src in SOURCES.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, d / name)
    return d


@pytest.mark.parametrize("script", ["p232_box.py", "p232_reduce.py", "p115_quality.py"])
def test_the_self_tests_pass(box_dir, script):
    out = _run(box_dir / script, "--self-test", cwd=box_dir)
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


def test_the_registered_constants_and_the_prereg_agree():
    assert R.TAGS == ("Ra", "Ka", "Kb", "Rb", "Rc", "Kc", "Kd", "Rd") and R.WARMUP == "Rw"
    assert R.PAIRS == (("Ka", "Ra"), ("Kb", "Rb"), ("Kc", "Rc"), ("Kd", "Rd"))
    assert R.SELF_PAIRS == (("Rb", "Ra"), ("Kb", "Ka"), ("Rd", "Rc"), ("Kd", "Kc"))
    assert (R.NOISY_BAND, R.REPS, R.GROUPS, R.WORKLOADS) == ((0.985, 1.015), 5, (2, 4, 8), {"W2": 2, "W4": 4, "W8": 8})
    assert (R.Q_BIAS, R.Q_ARGMAX, R.FLIP_G, R.MIN_POWER_W) == (0.02, 0.95, 1.01, 575)
    assert (R.TRIP_BIAS, R.TRIP_ARGMAX, R.TRIP_KL_MULT, R.TRIP_KL_ABS) == (0.005, 0.98, 3.0, 0.002)
    assert (R.Q_WINDOWS, R.Q_CONT, R.SERVED_TOKENS) == (24, 128, 4096) and all(24 % T == 0 for T in R.GROUPS)
    assert R.GNF4_SHA == "41ef086e6a3e2be5b158ff1bb38e0de67862a949" and R.TORCH == "2.8.0+cu128"
    assert (R.MODEL, R.REV) == ("Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39")
    assert R.HEAD_SHAPE == [151936, 2048] and R.SMS == 170
    assert R.ROWS_RECORD == {"rows_max": 8, "kernel": "dense_i8.linear_i8 (K37)",
                             "plan": {str(m): [32, 256, 4] for m in range(2, 9)}}
    assert R.PHASE_ARMS == {"R": ("R", "rep"), "K": ("ON",), "B16": ("ON",)}
    assert set(R.VERDICTS_READ) == {"FLIP_ROWS_8", "FLIP_ROWS_4", "FLIP_ROWS_2", "NO_FLIP", "FUNCTION_FAIL"}
    for s in ("R_a K_a K_b R_b R_c K_c K_d R_d", "One unscored warm-up process", "Five timed reps",
              "T x 128 / (the min wall at 160 tokens - the min wall at 32)", "**24 wikitext windows x 128 positions**",
              "KV at the server's 4096 tokens", "11 passed", "45 passed", "grouped-nf4-gemm v0.50.0 `41ef086e`",
              "torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0", "Qwen/Qwen3-30B-A3B @ `ad44e777`",
              "**|mean NLL(K) - mean NLL(R)| <= 0.02 nats**", "**argmax agreement >= 0.95**", "**[0.985, 1.015]**",
              "**>= 1.01**", "**575 W**", "plan (32, 256, 4)", "A zero count is a VOID", "**B16** swaps nothing",
              "A MISS at 4 caps it at 2 even if 8 passes", "mean NLL(R)| > 0.005 nats**", "**argmax agreement < 0.98**",
              "max(3 x the R self-pair", "floor, 0.002 nats per position)", "`TRIPWIRE none`",
              "`p232-5090-1` to `-3`, and `p232-shake-1` to `-3`"):
        assert s in FLAT, s


def test_the_runner_refuses_before_the_install_and_reads_only_a_rule_verdict():
    assert "GNF4_SHA=41ef086e6a3e2be5b158ff1bb38e0de67862a949" in RUN and 'md.version("grouped-nf4-gemm") == "0.50.0"' in RUN
    assert "TORCH_WANT=2.8.0+cu128" in RUN and 'transformers.__version__ == "5.17.0"' in RUN
    assert 'triton.__version__ == "3.4.0"' in RUN and '"transformers==5.17.0"' in RUN
    assert "p232-5090-1|p232-5090-2|p232-5090-3) MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39;;" in RUN
    assert "p232-shake-1|p232-shake-2|p232-shake-3) SHAKE=1; REPS=3;" in RUN and R.SHAKE_REPS == 3
    assert "SHORT=32; LONG=160; REPS=5; CONT=128; WINDOWS=24" in RUN and "MIN_POWER_W=575 " in RUN
    assert 'finish 17; }' in RUN and 'echo "refused: power.limit unreadable" > REFUSAL; finish 15; }' in RUN
    assert "MIN_DL_MB_S=40" in RUN and 'below ${MIN_DL_MB_S} MB/s"; echo "refused: download ${DL:-none}" > REFUSAL; finish 19; }' in RUN
    assert "E4B_DENSE_LT_PIN E4B_INT8_HEAD_BW E4B_INT8_HEAD_ROWS\n" in RUN, "every knob starts unset"
    unset = RUN.split("\nunset ", 1)[1].split("\n: > summary.txt")[0]
    for knob in ("E4B_INT8_HEAD_ROWS", "E4B_INT8_HEAD_BW", "E4B_PAGED_SPEC", "E4B_ATTN_SPLIT_CAP", "E4B_DENSE_GEMV_BW",
                 "TRITON_INTERPRET", "E4B_PAGED_GRAPHS"):
        assert re.search(rf"\s{knob}(\s|\\|$)", unset), knob
    # the tripwire: the head's grid (K36's quantizer bytes and plan), its default, and K37's shipped plan per M
    assert "8ea6dd9ee1b52594419f40339272059e413956aaed2a2f02d382450f854e8c6c" in RUN and "dense_gemv.PLANS_I8 ==" in RUN
    assert 'sp.resolve_int8_head_bw(sp.FUSION_UNSET, "qwen3_moe", (151936, 2048)) == ("1", "default-allowlisted")' in RUN
    assert "assert all(tuple(dense_i8.plan(151936, 2048, m)) == (32, 256, 4) for m in range(2, 9))" in RUN
    assert 'sp.PagedServeConfig.__dataclass_fields__["int8_head_rows"].default == 1' in RUN
    order = ["sha256sum -c staged-p232.sha256", 'finish 28;; esac', "PL=$(nvidia-smi --query-gpu=power.limit",
             "DL=$(MIX=$MODEL", '[ "$LEFT" -ge "$FIT_NEED" ]', ": > $W/STARTED", "pipx logs/pip_e4b.log",
             "pipx logs/pip_gnf4.log", "TRIPWIRE FAIL", "p232_reduce.py --self-test", "p232_box.py --self-test",
             "p115_quality.py --self-test", "TRITON_INTERPRET=0 perl -e 'alarm 900; exec @ARGV' python -m pytest tests/test_int8_head_rows_interp.py",
             "grep -qE '^11 passed", "snapshot_download", "python $W/k8_bake.py", "--prompts-only",
             "for TAG in Rw Ra Ka Kb Rb Rc Kc Kd Rd; do", "for PH in R K B16; do", "python $W/p232_reduce.py --dir"]
    at = [RUN.index(x) for x in order]
    assert at == sorted(at), dict(zip(order, at))
    assert 'ARM=${TAG:0:1}; KN=""; [ "$ARM" = K ] && KN="E4B_INT8_HEAD_ROWS=8"' in RUN
    assert 'KN=""; [ "$PH" = K ] && KN="E4B_INT8_HEAD_ROWS=8"; [ "$PH" = B16 ] && KN="E4B_INT8_HEAD_BW=0"' in RUN
    code = "\n".join(ln for ln in RUN.splitlines() if not ln.lstrip().startswith("#"))
    assert code.count("E4B_INT8_HEAD_ROWS=8") == 2 and code.count("E4B_INT8_HEAD_BW=0") == 1, "R is the unset knob"
    assert "E4B_ATTN_SPLIT_CAP=" not in RUN and "E4B_DENSE_GEMV_BW=" not in RUN
    assert "E4B_PAGED_GRAPHS=0 P232_ARM=$PH" in RUN and "--ref-dir $W/work/ref" in RUN
    assert "  1/SHAKEDOWN) finish 0;;" in RUN
    assert "  0/FLIP_ROWS_8|0/FLIP_ROWS_4|0/FLIP_ROWS_2|0/NO_FLIP|0/FUNCTION_FAIL) finish 0;;" in RUN
    assert "rc 41: a refusal before the install, not a strike; two consecutive headroom refusals count as one strike" in RUN


def _get(v):
    return int(re.search(rf"^{v}=(\d+)", RUN, re.M).group(1))


SSH_READY_S, STAGING_MARGIN_S = 420, 120   # pod-launch's longest wait for ssh on Vast; the drive's staging margin


def test_the_guard_fit_is_the_prereg_arithmetic():
    """Amendment 1's budget: FIT = install 600 + fetch 1,500 + bake 400 + the warm-up and eight arms at 240 + three
    quality phases at 700 + reserve 600 = 7,360 s, x 1.25 = 9,200 s; (boot 300 + 7,360) x 1.25 = 9,575 s <= the 2.75 h
    guard. The guard read is the LAST one the PREREG states (the amendment's), never the superseded 2.5 h."""
    install, fetch, bake, arm, q, reserve = (_get(v) for v in ("INSTALL_TO_GATE_S", "NEED_FETCH", "NEED_BAKE", "NEED_ARM",
                                                                "NEED_Q", "RESERVE"))
    assert (install, fetch, bake, arm, q, reserve) == (600, 1500, 400, 240, 700, 600)
    assert re.search(r"\bMIN_DISK_GB=140\b", RUN), "the box refuses below 140 GB free (Amendment 1)"
    assert "FIT=$((INSTALL_TO_GATE_S + NEED_FETCH + NEED_BAKE + 9 * NEED_ARM + 3 * NEED_Q + RESERVE))" in RUN
    fit = install + fetch + bake + 9 * arm + 3 * q + reserve
    assert fit == 7360 and fit * _get("FIT_HEADROOM_PCT") // 100 == 9200 and (300 + fit) * 1.25 == 9575 <= 9900
    guard_s = round(float(re.findall(r"\*\*Guard ([\d.]+) h\*\*", PREREG)[-1]) * 3600)
    assert guard_s == 9900 and guard_s - 9200 >= SSH_READY_S + STAGING_MARGIN_S, "a slow-ssh host clears the fit check"
    assert 57e9 / 40e6 <= fetch, "57 GB at the probe's 40 MB/s floor fits the fetch need"
    assert abs(2.75 * 1.05 + 70 * 0.04 - 5.6875) < 1e-9
    for s in ("fetch 1,500 + bake 400", "at 240 (2,160)", "at 700 (2,100)", "reserve 600 = 7,360 s, x 1.25 = 9,200 s.**",
              "((boot 300 + 7,360) x 1.25 = 9,575 s <= 9,900 s", "**$5.69**", "below 140 GB free"):
        assert s in FLAT, s


def test_the_drive_follows_pod_launchs_contract():
    """pod-launch runs a reading's drive without E4B_SHA (it proves the checkout HEAD is heads.e4b) and owns the run
    directory's receipt files: the drive derives the launch commit from the clean HEAD, refuses an override, and fetches
    the box tree into $RUN_DIR/p232/, refusing a non-empty target (P229 Amendment 3's form)."""
    assert 'if [ -n "${E4B_SHA:-}" ]; then say "refusing: E4B_SHA override; use the clean checkout HEAD"; exit 78; fi' in DRIVE
    assert 'E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)' in DRIVE and "is dirty" in DRIVE
    assert "E4B_RENT_DEADLINE_EPOCH E4B_SHA" not in DRIVE
    assert 'OUT="$RUN_DIR/p232"' in DRIVE and '"$RUN_DIR/"' not in DRIVE
    assert 'mkdir -p "$OUT" && [ -z "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ] || { say "fetch failed: $OUT is not an empty directory"; exit 22; }' in DRIVE
    assert "p232-5090-1|p232-5090-2|p232-5090-3|p232-shake-1|p232-shake-2|p232-shake-3) :;;" in DRIVE
    assert 'mkdir -p "$STAGE/t/$(dirname "$f")"' in DRIVE and "staged-p232.sha256" in DRIVE
    for f in ("bench/p232/p232_run.sh", "bench/p232/p232_box.py", "bench/p232/p232_reduce.py", "bench/p109/p109_box.py",
              "bench/p115/p115_quality.py", "bench/p110/p110_box.py", "bench/p108/p108_box.py", "bench/p97/p97_box.py",
              "bench/p39/k8_bake.py", "bench/p39/calib.json", "tests/test_int8_head_bw.py",
              "tests/test_int8_head_rows_interp.py", "tests/conftest.py", "bench/i8dense/i8_fakequant.py",
              "bench/p148/p148_kernels.py"):
        assert f in DRIVE, f
    assert "--include 'work/bake.json' --exclude 'work/*'" in DRIVE, "the arena and the reference log-probs stay on the box"
    for word in ("vast.py", "destroy", "/api/", "rent-watch"):
        assert word not in DRIVE.split("set -uo pipefail", 1)[1], word


def _fetches_without_no_links(drive):
    """(rsync lines, the ones without --no-links): the drive's live rsync invocations, comments skipped."""
    lines = [ln for ln in drive.splitlines() if re.search(r"\brsync\b", ln) and not ln.lstrip().startswith("#")]
    return lines, [ln for ln in lines if not re.search(r"\brsync\s(?:[^|;&]*\s)?--no-links\b", ln)]


def test_the_fetch_never_imports_a_link(tmp_path):
    """bench/common rule 8 (P229 Amendment 6's form): every rsync in the drive carries --no-links, so no link on a rented
    box reaches the controller. A drive with no rsync fails (the check needs its input), and the old fetch line fails."""
    lines, bad = _fetches_without_no_links(DRIVE)
    assert len(lines) == 1 and not bad, (lines, bad)
    assert _fetches_without_no_links(DRIVE.replace(" --no-links", ""))[1]          # the old line is refused
    assert not _fetches_without_no_links("echo fetch\n")[0]                          # no rsync -> the count above fails
    if shutil.which("rsync") is None:
        return
    src, dst = tmp_path / "box", tmp_path / "out"
    src.mkdir()
    (src / "f.json").write_text("{}", encoding="utf-8")
    try:
        (src / "secret").symlink_to("../../../../etc/hosts")
        (src / "etc").symlink_to("/etc", target_is_directory=True)
    except OSError:
        pytest.skip("this host cannot make symlinks")
    flags = re.search(r"rsync (-az --no-links) -e", lines[0]).group(1).split()
    subprocess.run(["rsync", *flags, f"{src}/", f"{dst}/"], check=True, capture_output=True)
    assert (dst / "f.json").is_file()
    assert not [p for p in dst.rglob("*") if p.is_symlink()] and not (dst / "secret").exists() and not (dst / "etc").exists()


def _dry_env(tmp_path, run_id, **extra):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P232_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0", **extra}


def _clean_checkout():
    return subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout == ""


def _drive(tmp_path, run_id, **extra):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    return subprocess.run(["bash", str(LANE / "p232_drive.sh")], capture_output=True, text=True,
                          env=_dry_env(tmp_path, run_id, **extra), timeout=120)


@pytest.mark.parametrize("run_id", ["p232-5090-4", "p153-5090-1", "p232-shake-4", "p232-5090-r1"])
def test_the_driver_refuses_another_run_id(tmp_path, run_id):
    out = _drive(tmp_path, run_id)
    assert out.returncode == 78 and "is not p232-5090-N or p232-shake-N" in out.stdout + out.stderr


def test_the_driver_refuses_an_e4b_sha_override(tmp_path):
    out = _drive(tmp_path, "p232-5090-1", E4B_SHA="a" * 40)
    assert out.returncode == 78 and "E4B_SHA override" in out.stdout + out.stderr


@pytest.mark.parametrize("run_id", ["p232-5090-1", "p232-5090-3", "p232-shake-1"])
def test_the_drivers_dry_run_names_the_launch(tmp_path, run_id):
    if not _clean_checkout():
        pytest.skip("the driver refuses a dirty tree (it derives the launch commit from HEAD)")
    out = _drive(tmp_path, run_id)
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"DRYRUN stage {len(SOURCES) + 1} files -> root@h:/root/p232" in out.stdout and "bash p232_run.sh" in out.stdout
    assert f"fetch -> {tmp_path}/p232" in out.stdout


def _funcs(path, names):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    return {n.name: ast.get_source_segment(src, n) for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names}


def test_p153s_pieces_are_carried_verbatim():
    """A2's ServedKV, P153's telemetry and the reducer's helpers (rates, tokens, divergence, the GEMV tally, the power
    floor and the telemetry report) are P153's bytes; what changed is named in the module docstrings."""
    box = ("ServedKV", "parse_telemetry", "_telemetry", "_counts", "_delta", "_sms")
    mine, theirs = _funcs(LANE / "p232_box.py", box), _funcs(B / "p153" / "p153_box.py", box)
    assert set(mine) == set(box) and mine == theirs
    red = ("_rate", "_tok", "divergence", "gemv_faults", "power_faults", "telemetry_report", "_telem", "_dense_off")
    mine, theirs = _funcs(LANE / "p232_reduce.py", red), _funcs(B / "p153" / "p153_reduce.py", red)
    assert set(mine) == set(red) and mine == theirs


# ---- the real record first: p232-shake-1 (an RTX 5090 on a 500 W board, explore tier), sha256-pinned
FIXTURE = LANE / "fixtures" / "p232-shake-1"
FIXTURE_SHA256 = {
    "arm_Ka.json": "d7bee34c3e32d77004af08c8eaecf0058d1a911ada160090cbe93d233b3af20f",
    "arm_Kb.json": "05b5ed7c5743d2072ed062e7caae4eb3878dd078ba9d7dff39de5470b2f6120b",
    "arm_Kc.json": "016cdb8722b0868824c1eb6dd915d70ae037f80c49be613292546d3677521318",
    "arm_Kd.json": "84e5736f1b4ed0bd29bdb014db5720227ff09c46fc8c88fad73c0bae3f47339f",
    "arm_Ra.json": "b51705c486e6da96dfe874fe9d6631dc0dd7b0c7528b085a19395620c10ede2d",
    "arm_Rb.json": "d6739327dce12881f33de5587c979a754e4418d87f77fa1d598218b653da34a2",
    "arm_Rc.json": "4f72998a835763a313c427ad8304575e626d1d4bc4e7612b42b080f06dd2f5bd",
    "arm_Rd.json": "a40cfe6975515ab4c366a131195d76f8729d9b8882fcf38e61896d48eea788b3",
    "arm_Rw.json": "a3c89a55ea4b4029da4578fb5029e735ae00b0903d0de6eda0d1286d470107a0",
    "quality_B16.json": "5cbe7786a14205b87a974d44f6c4c6dcb715ce1d31a81fe875772fb4ed99ce57",
    "quality_K.json": "dfdbdbed101b4a36d164069af6497a2eb70952d8c249295cefd9a5c205925175",
    "quality_R.json": "53c7a1d72cf07448ae6a53a889c3bbeece3f438be4df98428e7b0d18c961317d",
}
FIXTURE_E4B = "65a33362608169e1df61fa56f879baef263fd0ba"     # the controller HEAD that staged p232-shake-1's kit


def _real():
    """(arms, quality) as the reducer's main() loads them, each file checked against its pin first."""
    assert sorted(p.name for p in FIXTURE.iterdir()) == sorted(FIXTURE_SHA256)
    recs = {}
    for name, want in FIXTURE_SHA256.items():
        raw = (FIXTURE / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == want, f"the pinned shakedown record {name} changed"
        recs[name] = json.loads(raw)
    return ({t: recs[f"arm_{t}.json"] for t in R.TAGS + (R.WARMUP,)},
            {a: recs[f"quality_{a}.json"] for a in R.QUALITY_ARMS})


def _reading_shaped(arms, quality, *, power_w=None):
    """The narrowest change that makes the shakedown's records a reading's: shakedown off on every record and the
    registered rep count on every arm. With power_w, every timed block's power.limit is also set to it."""
    for r in list(arms.values()) + list(quality.values()):
        r["shakedown"] = False
    for r in arms.values():
        r["reps"] = R.REPS
        if power_w is not None:
            for w in r["workloads"].values():
                for t in w["telemetry"].values():
                    for side in ("before", "after"):
                        t[side]["power.limit"] = power_w
    return arms, quality


def test_the_real_shakedown_record_reduces_to_shakedown_with_every_check_passed():
    arms, quality = _real()
    assert all(r["shakedown"] is True and r["reps"] == R.SHAKE_REPS for r in arms.values())
    assert all(r["shakedown"] is True for r in quality.values())
    v = R.reduce(arms, quality, FIXTURE_E4B)
    assert v["verdict"] == "SHAKEDOWN" and v["would_read"] == "FLIP_ROWS_8", v["reasons"][:4]
    assert v["reasons"][0] == "a shakedown record is never a reading" and v["tripwire"] == []
    assert [v["per_T"][str(T)]["result"] for T in R.GROUPS] == ["PASS", "PASS", "PASS"]
    assert R.reduce(arms, quality, "0" * 40)["verdict"] == "VOID"          # the e4b pin is read on the real bytes


def test_the_real_record_made_a_reading_voids_on_the_575_w_floor_alone():
    """p232-shake-1 ran on a board capped at 500 W, which a shakedown may (its power is not ruled). Made reading-shaped,
    every timed block of every arm fails the floor, and nothing else does: 9 arms x 3 workloads x 2 lengths x 2 sides."""
    arms, quality = _reading_shaped(*_real())
    v = R.reduce(arms, quality, FIXTURE_E4B)
    assert v["verdict"] == "VOID" and v["tripwire"] == ["unread"]
    assert len(v["reasons"]) == len(R.TAGS + (R.WARMUP,)) * len(R.WORKLOADS) * 2 * 2, v["reasons"][:3]
    assert all(re.fullmatch(r"\w+ W\d at \d+ tokens: power\.limit 500\.0 W (before|after) the block, the floor 575 W", x)
               for x in v["reasons"]), [x for x in v["reasons"] if "500.0 W" not in x][:3]
    # the same bytes with the recorded cap at the floor read a verdict: the 500 W reading is the only fault
    lifted = R.reduce(*_reading_shaped(*_real(), power_w=575.0), FIXTURE_E4B)
    assert lifted["verdict"] == "FLIP_ROWS_8", lifted["reasons"][:3]


def test_a_reducer_without_the_power_check_reads_a_verdict_from_the_500_w_record(tmp_path):
    """The old rule as a mutant: drop the reading's power_faults call and the capped record reads FLIP_ROWS_8."""
    src = (LANE / "p232_reduce.py").read_text(encoding="utf-8")
    old = "        if not shake:\n            void += power_faults(t, r)\n"
    assert src.count(old) == 1
    (tmp_path / "p232_reduce.py").write_bytes(src.replace(old, "").encode("utf-8"))
    mut = _load("p232_reduce_no_power", tmp_path / "p232_reduce.py")
    assert mut.reduce(*_reading_shaped(*_real()), FIXTURE_E4B)["verdict"] == "FLIP_ROWS_8"
    assert R.reduce(*_reading_shaped(*_real()), FIXTURE_E4B)["verdict"] == "VOID"


DEPARTURES = [
    (lambda a, q: q["K"]["phases"]["4"]["per_window"]["wikitext"]["ON"].pop(), "23 ON windows at T=4"),
    (lambda a, q: q["K"]["phases"]["2"]["per_window"]["wikitext"]["ON"][3].update(argmax_agree=0.99),
     "an agreement that is not k/128"),
    (lambda a, q: q["K"]["phases"]["8"]["per_window"]["wikitext"]["ON"][5].pop("kl"), "a window without its KL"),
    (lambda a, q: a["Kc"].update(shakedown=False), "one record not a shakedown"),
    (lambda a, q: a["Kb"]["int8_head_bw"].update(mode="0"), "a K arm with the int8 head off"),
    (lambda a, q: a["Rd"].update(reps=R.REPS), "an arm at the reading's rep count"),
]


@pytest.mark.parametrize("depart, what", DEPARTURES, ids=[d[1] for d in DEPARTURES])
def test_each_departure_from_the_real_record_voids_it(depart, what):
    arms, quality = _real()
    depart(arms, quality)
    v = R.reduce(arms, quality, FIXTURE_E4B)
    assert v["verdict"] == "VOID" and v["tripwire"] == ["unread"], (what, v["reasons"][:2])


def test_the_new_files_are_lf_only_and_the_shell_scripts_parse():
    for f in NEW_FILES:
        assert b"\r" not in f.read_bytes(), f
    if shutil.which("bash") is None:
        return
    for f in ("p232_run.sh", "p232_drive.sh"):
        out = subprocess.run(["bash", "-n", str(LANE / f)], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr


REDUCER_MUTANTS = [
    ("    return abs(k - round(k)) < 1e-6", "    return True"),
    # g_T from the best pair, not the worst of four
    ("    g = {T: min(pairs[w].values()) for w, T in WORKLOADS.items()}",
     "    g = {T: max(pairs[w].values()) for w, T in WORKLOADS.items()}"),
    # only the first ABBA block's pairs
    ('PAIRS = (("Ka", "Ra"), ("Kb", "Rb"), ("Kc", "Rc"), ("Kd", "Rd"))', 'PAIRS = (("Ka", "Ra"), ("Kb", "Rb"))'),
    # the NOISY band widened
    ("NOISY_BAND = (0.985, 1.015)", "NOISY_BAND = (0.98, 1.02)"),
    # the second block's self-pairs dropped
    ('SELF_PAIRS = (("Rb", "Ra"), ("Kb", "Ka"), ("Rd", "Rc"), ("Kd", "Kc"))', 'SELF_PAIRS = (("Rb", "Ra"), ("Kb", "Ka"))'),
    # NOISY never read
    ('    if noisy:\n        rung = ("NOISY"', '    if False:\n        rung = ("NOISY"'),
    # the warm-up's absence accepted (P131 Amendment 2's process 0)
    ("    missing = [t for t in TAGS + (WARMUP,) if t not in arms", "    missing = [t for t in TAGS if t not in arms"),
    # the warm-up read as a scored arm: its slow first pass would trip a self-pair
    ('SELF_PAIRS = (("Rb", "Ra"),', 'SELF_PAIRS = (("Ra", "Rw"), ("Rb", "Ra"),'),
    # the five reps never checked
    ('        if r.get("reps") != reps:', "        if False:"),
    # the gain bar lowered
    ("FLIP_G = 1.01", "FLIP_G = 1.00"),
    # P153's quality bars loosened, or the argmax bar dropped
    ("Q_BIAS, Q_ARGMAX = 0.02, 0.95 ", "Q_BIAS, Q_ARGMAX = 0.03, 0.95 "),
    ('        if not agree_at_least(tuple(x["argmax_counts"]), Q_ARGMAX):', "        if False:"),
    # the bars read the ROUNDED minima
    ("        if g[T] < FLIP_G:", "        if round(g[T], 4) < FLIP_G:"),
    # a MISS at 4 does not cap the verdict at 2
    ("        if not passes[T]:\n            break", "        if not passes[T]:\n            continue"),
    # K arms held to Ra's tokens (K37 rounds differently from cuBLAS: K may decode other tokens than R)
    ('        first = "Ra" if t.startswith("R") else "Ka"', '        first = "Ra"'),
    # the head's engagement never checked at the speed build, nor at the quality pass's T
    ("        void += head_faults(t, r, arm, captured=True)\n", ""),
    ("        out += quality_head_faults(name, p, arm, T)\n", ""),
    # a zero count (and another kernel) accepted for K's rows
    ("        if any(routed.get(t) != 1 or t in passed or kern.get(t) != K37 for t in rows)",
     "        if any(t in passed for t in rows)"),
    # R routing a row count accepted
    ("    elif any(t in routed or not passed.get(t) for t in rows):", "    elif any(not passed.get(t) for t in rows):"),
    # the one-row K36 route (and its fallback) unchecked
    ('    if eng.get("modules") != 1 or routed.get("1") != 1 or "1" in passed:', '    if eng.get("modules") != 1:'),
    # B16 may swap the head
    ('    if arm == "B16":\n        if (d.get("mode"), d.get("source"))', '    if arm == "B16":\n        if False and (d.get("mode"), d.get("source"))'),
    # the head swapped by an explicit 1 taken for the default
    ('    if (d.get("mode"), d.get("modules"), d.get("shape"), d.get("source")) != ("1", 1, HEAD_SHAPE, "default-allowlisted"):',
     '    if (d.get("mode"), d.get("modules"), d.get("shape")) != ("1", 1, HEAD_SHAPE):'),
    # K's route record (K37 at its shipped plan) unchecked
    ('    if d.get("rows") != want_rows:', "    if False:"),
    # bucket T never required to replay in W_T
    ('        if not gd.get("replays"):', "        if False:"),
    # the floor lowered to a capped board's, or an unreadable limit accepted
    ("MIN_POWER_W = 575 ", "MIN_POWER_W = 450 "),
    ("                if pl is None or pl < MIN_POWER_W:", "                if pl is not None and pl < MIN_POWER_W:"),
    # the quality pass at P115's 656 tokens accepted; another group accepted
    ("        served = (qk.get(\"max_tokens_per_seq\"), qk.get(\"blocks_per_seq\")) == ([SERVED_TOKENS], [SERVED_BLOCKS])",
     "        served = True"),
    ('        if p.get("group") != T or p.get("arms") != list(PHASE_ARMS[arm]):', '        if p.get("arms") != list(PHASE_ARMS[arm]):'),
    # the phases on different windows accepted
    ('        if len(shas) != 1 or "null" in shas:', "        if False:"),
    # rung 6: a zero floor falls to 0 (every KL above zero fires), the bias or argmax wire moved
    ('        bound = max(TRIP_KL_MULT * (x["floor_kl"] or 0.0), TRIP_KL_ABS)', '        bound = TRIP_KL_MULT * (x["floor_kl"] or 0.0)'),
    ("TRIP_BIAS, TRIP_ARGMAX, TRIP_KL_MULT, TRIP_KL_ABS = 0.005, 0.98, 3.0, 0.002",
     "TRIP_BIAS, TRIP_ARGMAX, TRIP_KL_MULT, TRIP_KL_ABS = 0.01, 0.98, 3.0, 0.002"),
    ("TRIP_BIAS, TRIP_ARGMAX, TRIP_KL_MULT, TRIP_KL_ABS = 0.005, 0.98, 3.0, 0.002",
     "TRIP_BIAS, TRIP_ARGMAX, TRIP_KL_MULT, TRIP_KL_ABS = 0.005, 0.97, 3.0, 0.002"),
    ("TRIP_BIAS, TRIP_ARGMAX, TRIP_KL_MULT, TRIP_KL_ABS = 0.005, 0.98, 3.0, 0.002",
     "TRIP_BIAS, TRIP_ARGMAX, TRIP_KL_MULT, TRIP_KL_ABS = 0.005, 0.98, 5.0, 0.002"),
    # the tripwire changing the verdict (it is reported, never ruling)
    ('            rung = (f"FLIP_ROWS_{bound}",', '            rung = ("NO_FLIP" if fired else f"FLIP_ROWS_{bound}",'),
    # a VOID printed as "TRIPWIRE none"
    ('    if t == ["unread"] or t is None:', "    if t is None:"),
    # a shakedown read as a reading, or mixed with one
    ('    if shake:\n        out.update(verdict="SHAKEDOWN"', '    if False:\n        out.update(verdict="SHAKEDOWN"'),
    ("    if len(flags) != 1:", "    if False:"),
]


@pytest.mark.parametrize("old,new", REDUCER_MUTANTS, ids=[f"m{i:02d}" for i in range(len(REDUCER_MUTANTS))])
def test_each_reducer_mutant_fails_the_self_test(tmp_path, old, new):
    src = (LANE / "p232_reduce.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p232_reduce.py").write_bytes(src.replace(old, new).encode("utf-8"))
    out = _run(tmp_path / "p232_reduce.py", "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


BOX_MUTANTS = [
    # K at another row count
    ('"K": {ROWS_KNOB: "8", HEAD_KNOB: None}', '"K": {ROWS_KNOB: "4", HEAD_KNOB: None}'),
    # R may name the head on (the default is the subject)
    ('ARM_VALUE = {"R": {ROWS_KNOB: None, HEAD_KNOB: None},', 'ARM_VALUE = {"R": {ROWS_KNOB: None},'),
    # the split cap knob may be set in an arm
    ("    bad.update({k: env.get(k) for k in (CAP_KNOB, DENSE_KNOB, SPEC_KNOB) + FUSION_KNOBS + GEMV_KNOBS",
     "    bad.update({k: env.get(k) for k in (DENSE_KNOB, SPEC_KNOB) + FUSION_KNOBS + GEMV_KNOBS"),
    # B16 in a speed arm
    ('SPEED_ARMS = ("R", "K")', 'SPEED_ARMS = ("R", "K", "B16")'),
    # an unreadable telemetry field read as zero
    ("        except (IndexError, ValueError):\n            out[k] = None", "        except (IndexError, ValueError):\n            out[k] = 0.0"),
    # the stability report compares bucket T with itself
    ("                y_2 = torch.cat([head(x[i:i + 2]) for i in range(0, T, 2)])", "                y_2 = head(x)"),
    # the call delta keeps zero rows
    ("        out[side] = {k: a[k] - b.get(k, 0) for k in a if a[k] - b.get(k, 0)}",
     "        out[side] = {k: a[k] - b.get(k, 0) for k in a}"),
]


@pytest.mark.parametrize("old,new", BOX_MUTANTS, ids=[f"b{i:02d}" for i in range(len(BOX_MUTANTS))])
def test_each_box_mutant_fails_the_self_test(box_dir, tmp_path, old, new):
    src = (LANE / "p232_box.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p232_box.py").write_bytes(src.replace(old, new).encode("utf-8"))
    shutil.copyfile(box_dir / "p109_box.py", tmp_path / "p109_box.py")
    out = _run(tmp_path / "p232_box.py", "--self-test", cwd=tmp_path)
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr
