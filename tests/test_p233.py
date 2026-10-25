# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Lane P233 (bench/p233/PREREG-p233.md, MAP 2), does serving at E4B_PAGED_PLACEMENT=solver keep quality against
all-vram: the staged pins (P115's instrument and the boxes it imports at P232's pins, the premise in the repo's layout
under t/), the registered constants as the PREREG states them (Amendment 1 included), the runner's order (the refusals,
the download probe and the guard fit before the install; the premise at 37 and 6; the three phases), the guard fit and
its slack, the drive's pod-launch contract and refusals (and that its fetch imports no link), the self-tests run as the
box runs them (one directory), P232's pieces carried verbatim, and the reducer and box mutants the self-tests must kill.

The maintainer's two requirements:
- arm A passes ``measure_phase`` exactly the arguments ``p115_quality.main()`` passes: read off main's AST (the keywords
  and their argparse defaults), and proven at run time by calling main and the box with the same stubs and comparing the
  captured calls; the box is then run against measure_phase's REAL record shape (a committed P232 quality record), and
  the reducer reads real per-window values;
- on CPU, a ``PagedServeConfig`` at ``E4B_PAGED_PLACEMENT=solver`` with ``E4B_INT8_HEAD_BW=1`` and
  ``E4B_INT8_HEAD_ROWS=8`` resolves the head on (an ``Int8HeadLinear``, rows 8, source ``explicit``) through
  build_engine's two head steps on a toy model carrying a tiered MoE state.

NO REAL FIXTURE YET: the shakedown's records become the reducer's real fixture by an amendment."""
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
import types

import pytest
from guard_outcomes import expect_outcome

REPO = pathlib.Path(__file__).resolve().parents[1]
B = REPO / "bench"
LANE = B / "p233"
#: staged path (as the box sees it) -> the file in this checkout
SOURCES = {"p233_run.sh": LANE / "p233_run.sh", "p233_box.py": LANE / "p233_box.py",
           "p233_reduce.py": LANE / "p233_reduce.py", "p115_quality.py": B / "p115" / "p115_quality.py",
           "p110_box.py": B / "p110" / "p110_box.py", "p108_box.py": B / "p108" / "p108_box.py",
           "p97_box.py": B / "p97" / "p97_box.py", "k8_bake.py": B / "p39" / "k8_bake.py",
           "calib.json": B / "p39" / "calib.json",
           "t/tests/test_hybrid_cold_dest.py": REPO / "tests" / "test_hybrid_cold_dest.py",
           "t/tests/test_serve_hybrid_compose.py": REPO / "tests" / "test_serve_hybrid_compose.py",
           "t/tests/conftest.py": REPO / "tests" / "conftest.py"}
RUN = (LANE / "p233_run.sh").read_text(encoding="utf-8")
DRIVE = (LANE / "p233_drive.sh").read_text(encoding="utf-8")
PREREG = (LANE / "PREREG-p233.md").read_text(encoding="utf-8")
FLAT = " ".join(PREREG.split())     # the PREREG's wrapped lines joined: a phrase reads across a line break
NEW_FILES = [LANE / f for f in ("p233_run.sh", "p233_box.py", "p233_reduce.py", "p233_drive.sh", "staged-p233.sha256")] + [
    REPO / "tests" / "test_p233.py"]
# The kit's changelog fragment (changelog.d/p233-kit.md) was folded into CHANGELOG.md by the 0.58.0 release, so it is
# no longer a file; its text is checked there with the rest of the release section.
#: a committed P232 quality record: measure_phase's real output (with P232's box fields beside it)
P232_FIXTURE = B / "p232" / "fixtures" / "p232-shake-1"


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


R = _load("p233_reduce_under_test", LANE / "p233_reduce.py")


def _run(script, *args, cwd=None):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=180, cwd=cwd)


def test_every_staged_file_is_pinned_and_matches():
    pins = _pins(LANE / "staged-p233.sha256")
    assert list(pins) == list(SOURCES)
    for name, path in SOURCES.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pins[name], name


def test_the_shared_pieces_are_at_p232s_pins():
    """P115's instrument, the boxes it imports, P39's bake and the suite's conftest are byte-identical to the ones P232
    read with; the two premise files are this lane's own, held to the current tree by the pins above."""
    mine, p232 = _pins(LANE / "staged-p233.sha256"), _pins(B / "p232" / "staged-p232.sha256")
    shared = {"p115_quality.py", "p110_box.py", "p108_box.py", "p97_box.py", "k8_bake.py", "calib.json", "t/tests/conftest.py"}
    assert shared <= set(p232) and all(mine[n] == p232[n] for n in shared)
    assert set(mine) == {"p233_run.sh", "p233_box.py", "p233_reduce.py", "t/tests/test_hybrid_cold_dest.py",
                         "t/tests/test_serve_hybrid_compose.py"} | shared


def _ast_count(path) -> int:
    """The items pytest collects from a test file: each top-level ``test_`` function once per case of its literal
    ``parametrize`` lists (the premise files use no other generator)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    n = 0
    for f in tree.body:
        if isinstance(f, ast.FunctionDef) and f.name.startswith("test_"):
            k = 1
            for d in f.decorator_list:
                if isinstance(d, ast.Call) and ast.unparse(d.func).endswith("parametrize"):
                    vals = d.args[1]
                    assert isinstance(vals, (ast.List, ast.Tuple)), ast.unparse(d)
                    k *= len(vals.elts)
            n += k
    return n


def test_the_premise_counts_are_the_suites_own():
    """37 and 6 are what the staged suites hold: the tier's GPU tests (31 functions, one parametrized over 7 bad
    destinations) and the solver composition's 6 CPU tests. Read off the files; where this host has the suites'
    dependencies, pytest's own collection must agree."""
    gpu, cpu = REPO / "tests" / "test_hybrid_cold_dest.py", REPO / "tests" / "test_serve_hybrid_compose.py"
    assert (_ast_count(gpu), _ast_count(cpu)) == (37, 6)
    assert "grep -qE '^37 passed" in RUN and "grep -qE '^6 passed" in RUN
    for path, want, deps in ((gpu, 37, ("nvme_arena", "nvme_residency", "cpu_grouped")), (cpu, 6, ("fastapi",))):
        if any(importlib.util.find_spec(m) is None for m in deps):
            continue                                       # collection would skip the module: the AST count stands
        out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(path)],
                             capture_output=True, text=True, cwd=REPO)
        assert re.search(rf"^{want} tests collected", out.stdout, re.M), out.stdout[-500:]


@pytest.fixture(scope="module")
def box_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("p233_box")
    for name, src in SOURCES.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, d / name)
    return d


@pytest.mark.parametrize("script", ["p233_box.py", "p233_reduce.py", "p115_quality.py"])
def test_the_self_tests_pass(box_dir, script):
    out = _run(box_dir / script, "--self-test", cwd=box_dir)
    assert out.returncode == 0 and "self-test OK" in out.stdout, out.stdout + out.stderr


def test_the_registered_constants_and_the_prereg_agree():
    assert R.ARMS == ("A", "S", "C") and (R.RULING, R.REFERENCE, R.REPORTED) == ("S", "A", "C")
    assert R.GROUPS == (1, 8) and (R.Q_WINDOWS, R.Q_PROMPT, R.Q_CONT) == (24, 512, 128)
    assert all(R.Q_WINDOWS % T == 0 and R.SHAKE_WINDOWS % T == 0 for T in R.GROUPS)
    assert (R.Q_BIAS, R.Q_ARGMAX) == (0.02, 0.95)
    assert (R.TRIP_BIAS, R.TRIP_ARGMAX, R.TRIP_KL_MULT, R.TRIP_KL_ABS) == (0.005, 0.98, 3.0, 0.002)
    assert R.PLACEMENT == {"A": "all-vram", "S": "solver", "C": "solver"}
    assert R.BUDGET_GB == {"S": (8.0, 3.0), "C": (8.0, 0.0)}
    assert (R.CC, R.SMS) == ("12.0", 170) and R.SERVED_TOKENS == 4096
    assert R.GNF4_SHA == "41ef086e6a3e2be5b158ff1bb38e0de67862a949" and R.TORCH == "2.8.0+cu128"
    assert (R.MODEL, R.REV) == ("Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39")
    assert (R.HEAD_MODE, R.HEAD_SOURCE, R.ROWS_MAX, R.ROWS_SOURCE) == ("1", "explicit", 8, "explicit")
    assert R.KNOBS == {"E4B_PAGED_GRAPHS": "0", "E4B_INT8_HEAD_BW": "1", "E4B_INT8_HEAD_ROWS": "8",
                       "E4B_PAGED_HOT_ROWS": "128"} and R.HOT_ROWS == 128
    assert set(R.VERDICTS_READ) == {"SOLVER_SANE", "SOLVER_NOT_SANE"}
    for s in ("`E4B_PAGED_GRAPHS=0` in every arm", "| **A** | `all-vram` | (unused)", "| **S** | `solver` | `8.0` / `3.0`",
              "| **C** | `solver` | `8.0` / `0`", "**the ruling arm**", "**reported only**",
              "**24 wikitext windows x 128 positions per T**", "after a 512-token prompt", "at the server's 4,096 tokens",
              "**Group T = 1 and T = 8, each read on its own.**", "its `rep` arm (A again on the first group) is the floor draw",
              "grouped-nf4-gemm v0.50.0 `41ef086e`", "torch 2.8.0+cu128, triton 3.4.0, transformers 5.17.0",
              "Qwen/Qwen3-30B-A3B @ `ad44e777`", "**|mean NLL(S) - mean NLL(A)| <= 0.02 nats**",
              "**argmax agreement >= 0.95**", "the pins, the card (cc 12.0, 170 SMs)", "**|mean NLL(S) - mean NLL(A)| > 0.005 nats**",
              "**argmax agreement < 0.98**", "max(3 x the A self-pair floor, 0.002 nats per position)", "`TRIPWIRE none`",
              "a zero or absent floor falls to the absolute bound, never to 0", "**SOLVER_SANE:**", "**SOLVER_NOT_SANE:**",
              "**`E4B_INT8_HEAD_BW=1`** and **`E4B_INT8_HEAD_ROWS=8`**", "`levers.int8_head_bw`",
              "`p233-5090-1` to `-3` for readings, plus `p233-5090-r1` to `-r6`", "`p233-shake-1` to `-3`",
              "(`tests/test_hybrid_cold_dest.py`, `tests/test_serve_hybrid_compose.py`), at counts the kit names, or rc 25",
              "**No power floor:**", "sets **`E4B_PAGED_HOT_ROWS=128`** (E)",
              "the read speaks for **solver placement with `E4B_PAGED_HOT_ROWS=128`**"):
        assert s in FLAT, s


def test_the_runner_refuses_before_the_install_and_reads_only_a_rule_verdict():
    assert "GNF4_SHA=41ef086e6a3e2be5b158ff1bb38e0de67862a949" in RUN and 'md.version("grouped-nf4-gemm") == "0.50.0"' in RUN
    assert "TORCH_WANT=2.8.0+cu128" in RUN and 'transformers.__version__ == "5.17.0"' in RUN
    assert 'triton.__version__ == "3.4.0"' in RUN and '"transformers==5.17.0"' in RUN
    ids = "p233-5090-1|p233-5090-2|p233-5090-3|" + "|".join(f"p233-5090-r{i}" for i in range(1, 7)) + ")"
    assert ids in RUN and "MODEL=Qwen/Qwen3-30B-A3B; REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39;;" in RUN
    assert "p233-shake-1|p233-shake-2|p233-shake-3) SHAKE=1; WINDOWS=8;" in RUN and R.SHAKE_WINDOWS == 8
    assert "CONT=128; WINDOWS=24" in RUN
    assert "MIN_RAM_GB=60" in RUN and "MIN_DISK_GB=140" in RUN and "MIN_DL_MB_S=40" in RUN
    for rc in ("finish 10", "finish 13", "finish 15", "finish 16", "finish 18", "finish 19", "finish 28", "finish 41"):
        assert rc in RUN, rc
    # no power floor: the limit is recorded and never refused on (there is no rc 17 and no floor constant)
    assert "finish 17" not in RUN and "MIN_POWER_W" not in RUN and "POWER_LIMIT_W ${PL:-unreadable} (recorded, never ruled)" in RUN
    unset = RUN.split("\nunset ", 1)[1].split("\n: > summary.txt")[0]
    for knob in ("E4B_INT8_HEAD_ROWS", "E4B_INT8_HEAD_BW", "E4B_PAGED_HOT_ROWS", "E4B_PAGED_SPEC", "E4B_ATTN_SPLIT_CAP", "E4B_DENSE_GEMV_BW",
                 "TRITON_INTERPRET", "E4B_PAGED_GRAPHS", "E4B_PAGED_PLACEMENT", "E4B_PAGED_VRAM_GB", "E4B_PAGED_DRAM_GB",
                 "E4B_ATTN_SPLIT_FIXED", "E4B_FUSE_PREFILL_GLUE", "E4B_NF4_SMALLM_BW"):
        assert re.search(rf"\s{knob}(\s|\\|$)", unset), knob
    order = ["sha256sum -c staged-p233.sha256", 'finish 28;; esac', "FREE_GB=$(df", "RAM_GB=$(awk",
             "PL=$(nvidia-smi --query-gpu=power.limit", "DL=$(MIX=$MODEL", '[ "$LEFT" -ge "$FIT_NEED" ]', ": > $W/STARTED",
             "pipx logs/pip_e4b.log", "pipx logs/pip_gnf4.log", "TRIPWIRE FAIL", "p233_reduce.py --self-test",
             "p233_box.py --self-test", "p115_quality.py --self-test", "python -m pytest tests/test_hybrid_cold_dest.py",
             "python -m pytest tests/test_serve_hybrid_compose.py", "grep -qE '^37 passed", "snapshot_download",
             "python $W/k8_bake.py", "for PH in A S C; do", "python $W/p233_reduce.py --dir"]
    at = [RUN.index(x) for x in order]
    assert at == sorted(at), dict(zip(order, at))
    # the arms, as registered: every arm eager with the head knobs named; the placements and their budgets
    assert ("E4B_PAGED_MAX_SEQS=16 E4B_PAGED_GRAPHS=0 E4B_INT8_HEAD_BW=1 E4B_INT8_HEAD_ROWS=8 E4B_PAGED_HOT_ROWS=128\""
            in RUN)
    assert 'A) PLACE="E4B_PAGED_PLACEMENT=all-vram"; NEED=$NEED_A' in RUN
    assert 'S) PLACE="E4B_PAGED_PLACEMENT=solver E4B_PAGED_VRAM_GB=8.0 E4B_PAGED_DRAM_GB=3.0"; NEED=$NEED_S' in RUN
    assert 'C) PLACE="E4B_PAGED_PLACEMENT=solver E4B_PAGED_VRAM_GB=8.0 E4B_PAGED_DRAM_GB=0"; NEED=$NEED_C' in RUN
    code = "\n".join(ln for ln in RUN.splitlines() if not ln.lstrip().startswith("#"))
    assert code.count("E4B_INT8_HEAD_ROWS=8") == 1 and code.count("E4B_PAGED_GRAPHS=0") == 1, "named once, for every arm"
    assert code.count("E4B_PAGED_HOT_ROWS=128") == 1, "the tier's rows named once, for every arm (Amendment 3)"
    # the tripwire ties the knob to the tier: the config reads it, build_engine hands it on
    assert 'hot_rows=int(env("E4B_PAGED_HOT_ROWS", "64"))' in RUN
    assert "enable_hybrid_tier(model, cfg.arena, man, hot_rows=cfg.hot_rows," in RUN
    assert "E4B_ATTN_SPLIT_CAP=" not in RUN and "E4B_DENSE_GEMV_BW=" not in RUN
    assert "--ref-dir $W/work/ref" in RUN and '"fastapi>=0.110"' in RUN
    assert "  1/SHAKEDOWN) finish 0;;" in RUN and "  0/SOLVER_SANE|0/SOLVER_NOT_SANE) finish 0;;" in RUN
    assert "rc 41: a refusal before the install, not a strike; two consecutive headroom refusals count as one strike" in RUN
    # the tripwire: the head (Amendment 1) explicit at both knobs, K36's quantizer bytes and K37's plan; eager at solver
    assert 'sp.resolve_int8_head_bw("1", "qwen3_moe", (151936, 2048)) == ("1", "explicit")' in RUN
    assert 'sp.resolve_int8_head_rows(8, "qwen3_moe", (151936, 2048)) == (8, "explicit")' in RUN
    assert "8ea6dd9ee1b52594419f40339272059e413956aaed2a2f02d382450f854e8c6c" in RUN
    assert 'sp._graphs_env("0", "cuda", "all-vram") is False and sp._graphs_env("0", "cuda", "solver") is False' in RUN


def _get(v):
    return int(re.search(rf"^{v}=(\d+)", RUN, re.M).group(1))


SSH_READY_S, STAGING_MARGIN_S = 420, 120   # pod-launch's longest wait for ssh on Vast; the drive's staging margin


def test_the_guard_fit_is_the_prereg_arithmetic():
    """The budget: FIT = install 600 + fetch 1,500 + bake 400 + phase A 1,026 (Amendment 4's re-fit; registered 600) +
    phase S 2,400 + phase C 2,400 + reserve 600 = 8,926 s, x 1.25 = 11,157 s; (boot 300 + 8,926) x 1.25 = 11,532.5 s
    <= the 3.25 h guard (11,700 s)."""
    terms = tuple(_get(v) for v in ("INSTALL_TO_GATE_S", "NEED_FETCH", "NEED_BAKE", "NEED_A", "NEED_S", "NEED_C", "RESERVE"))
    assert terms == (600, 1500, 400, 1026, 2400, 2400, 600)
    assert round(1.25 * (315.1 + 2 * (195.9 + 56.9)), 1) == 1025.9 and _get("NEED_A") == 1026   # 1.25 x A at 24 windows
    assert "FIT=$((INSTALL_TO_GATE_S + NEED_FETCH + NEED_BAKE + NEED_A + NEED_S + NEED_C + RESERVE))" in RUN
    fit = sum(terms)
    assert fit == 8926 and fit * _get("FIT_HEADROOM_PCT") // 100 == 11157 and (300 + fit) * 1.25 == 11532.5 <= 11700
    guard_s = round(float(re.findall(r"\*\*Guard ([\d.]+) h\*\*", PREREG)[-1]) * 3600)
    assert guard_s == 11700 and guard_s - 11157 >= SSH_READY_S + STAGING_MARGIN_S, "a slow-ssh host clears the fit check"
    assert 57e9 / 40e6 <= _get("NEED_FETCH"), "57 GB at the probe's 40 MB/s floor fits the fetch need"
    assert abs(3.25 * 1.05 + 70 * 0.04 - 6.2125) < 1e-9 and abs(2 * 6.2125 + 6 * 0.30 - 14.225) < 1e-9
    for s in ("install 600 s; fetch 1,500 s", "bake 400 s; phase A at two group sizes, about 600 s",
              "phase S about 2,400 s and phase C about 2,400 s", "**FIT = 8,500 s, x 1.25 = 10,625 s.**",
              "((boot 300 + 8,500) x 1.25 = 11,000 s <= 11,700 s)", "**$6.21**", "**$14.23**", "Host RAM at least **60 GiB**",
              "**8,926 s, x 1.25 = 11,157 s**", "(boot 300 + 8,926) x 1.25 = 11,532.5 s <= 11,700 s",
              "**re-fit to 1,026 s** (1.25 x 820.7)", "The TRIPWIRE's terms and bounds do not move:**"):
        assert s in FLAT, s


def test_the_drive_follows_pod_launchs_contract():
    """pod-launch runs a reading's drive without E4B_SHA (it proves the checkout HEAD is heads.e4b) and owns the run
    directory's receipt files: the drive derives the launch commit from the clean HEAD, refuses an override, and fetches
    the box tree into $RUN_DIR/p233/, refusing a non-empty target (P229 Amendment 3's form)."""
    assert 'if [ -n "${E4B_SHA:-}" ]; then say "refusing: E4B_SHA override; use the clean checkout HEAD"; exit 78; fi' in DRIVE
    assert 'E4B_SHA=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)' in DRIVE and "is dirty" in DRIVE
    assert "E4B_RENT_DEADLINE_EPOCH E4B_SHA" not in DRIVE
    assert 'OUT="$RUN_DIR/p233"' in DRIVE and '"$RUN_DIR/"' not in DRIVE
    assert 'mkdir -p "$OUT" && [ -z "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ] || { say "fetch failed: $OUT is not an empty directory"; exit 22; }' in DRIVE
    assert ("  p233-5090-1|p233-5090-2|p233-5090-3|p233-5090-r1|p233-5090-r2|p233-5090-r3|p233-5090-r4|p233-5090-r5|"
            "p233-5090-r6) :;;\n  p233-shake-1|p233-shake-2|p233-shake-3) :;;") in DRIVE
    assert 'mkdir -p "$STAGE/t/$(dirname "$f")"' in DRIVE and "staged-p233.sha256" in DRIVE
    for name, path in SOURCES.items():
        assert path.relative_to(REPO).as_posix() in DRIVE, name
    assert "bench/p109/p109_box.py" not in DRIVE, "no speed arms: P109's box is not staged"
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


def test_the_runner_never_collects_the_gpu_uuid():
    """The card's GPU UUID is store-only, so the runner never asks for it: no nvidia-smi query names uuid, and no
    nvidia-smi -L, -q or --list-gpus (which print it) runs. The old query line is refused."""
    def bad(src):
        return [ln for ln in src.splitlines() if not ln.lstrip().startswith("#") and "nvidia-smi" in ln
                and ("uuid" in ln.lower() or re.search(r"nvidia-smi\s+(-L|-q|--list-gpus)(\s|$)", ln))]
    assert sum("nvidia-smi --query-gpu=" in ln for ln in RUN.splitlines()) >= 2   # the check has its input
    assert not bad(RUN), bad(RUN)
    assert bad(RUN.replace("driver_version,compute_cap", "driver_version,uuid,compute_cap"))
    assert bad("nvidia-smi -L") and bad("nvidia-smi -q | tee f") and bad("nvidia-smi --list-gpus")


def _dry_env(tmp_path, run_id, **extra):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "P233_DRIVE_DRYRUN": "1",
            "E4B_RENT_SSH_HOST": "h", "E4B_RENT_SSH_PORT": "1", "E4B_RENT_SSH_OPTS": "-q", "E4B_RENT_RUN_DIR": str(tmp_path),
            "E4B_RENT_RUN_ID": run_id, "E4B_RENT_DEADLINE_EPOCH": "4102444800", "E4B_RENT_INSTANCE_ID": "0", **extra}


def _drive(tmp_path, run_id, **extra):
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    return subprocess.run(["bash", str(LANE / "p233_drive.sh")], capture_output=True, text=True,
                          env=_dry_env(tmp_path, run_id, **extra), timeout=120)


@pytest.mark.parametrize("run_id", ["p233-5090-4", "p232-5090-1", "p233-shake-4", "p233-5090-r7", "p233-5090-r0"])
def test_the_driver_refuses_another_run_id(tmp_path, run_id):
    out = _drive(tmp_path, run_id)
    assert out.returncode == 78 and "is not p233-5090-N, p233-5090-rN or p233-shake-N" in out.stdout + out.stderr


def test_the_driver_refuses_an_e4b_sha_override(tmp_path):
    out = _drive(tmp_path, "p233-5090-1", E4B_SHA="a" * 40)
    assert out.returncode == 78 and "E4B_SHA override" in out.stdout + out.stderr


def _clean_or_skip(dirty):
    """The driver refuses a dirty tree by design, so a dirty local checkout skips the dry run, named. Under CI the
    checkout is fresh: a dirty one is broken and fails, never a silent skip that drops the dry-run asserts."""
    if dirty.strip():
        why = f"the checkout is dirty ({dirty.splitlines()[0][:80]}): the driver refuses a dirty tree by design"
        if os.environ.get("CI"):
            pytest.fail(why + " (CI must run on a clean checkout)")
        pytest.skip(why)


@pytest.mark.parametrize("ci,outcome", [("1", pytest.fail.Exception), ("", pytest.skip.Exception)],
                         ids=["ci-fails", "local-skips"])
def test_a_dirty_checkout_fails_under_ci_and_skips_locally(monkeypatch, ci, outcome):
    if ci:
        monkeypatch.setenv("CI", ci)
    else:
        monkeypatch.delenv("CI", raising=False)
    with expect_outcome(outcome):
        _clean_or_skip(" M tests/test_p233.py\n")
    _clean_or_skip("")                                                   # a clean tree neither fails nor skips


@pytest.mark.parametrize("run_id", ["p233-5090-1", "p233-5090-r6", "p233-shake-1"])
def test_the_drivers_dry_run_names_the_launch(tmp_path, run_id):
    """The driver dry-runs on a clean checkout (a new lane: this checkout is its launch tree)."""
    if sys.platform == "win32" or shutil.which("bash") is None:
        pytest.skip("the driver needs a POSIX bash")
    _clean_or_skip(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True,
                                  text=True).stdout)
    out = _drive(tmp_path, run_id)
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"DRYRUN stage {len(SOURCES) + 1} files -> root@h:/root/p233" in out.stdout and "bash p233_run.sh" in out.stdout
    assert f"fetch -> {tmp_path}/p233" in out.stdout


def _funcs(path, names):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    return {n.name: ast.get_source_segment(src, n) for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names}


def test_p232s_pieces_are_carried_verbatim():
    """A2's ServedKV, the telemetry, the head's call reads and the reducer's pooled-count helpers are P232's bytes; what
    changed is named in the module docstrings."""
    box = ("ServedKV", "parse_telemetry", "_telemetry", "_counts", "_delta", "_sms", "head_calls", "calls_delta",
           "_engagement")
    mine, theirs = _funcs(LANE / "p233_box.py", box), _funcs(B / "p232" / "p232_box.py", box)
    assert set(mine) == set(box) and mine == theirs
    red = ("_mean", "integral_agree", "agree_counts", "agree_at_least", "compare", "tripwire_line")
    mine, theirs = _funcs(LANE / "p233_reduce.py", red), _funcs(B / "p232" / "p232_reduce.py", red)
    assert set(mine) == set(red) and mine == theirs


# ------------------------------------------------- the maintainer's first requirement: arm A is P115's instrument --

def _p115_main_call():
    """(keywords in order, their value expressions, the positional arguments) of main()'s measure_phase call, and main's
    argparse defaults, read off p115_quality.py's AST."""
    tree = ast.parse((B / "p115" / "p115_quality.py").read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "measure_phase"]
    assert len(calls) == 1
    c = calls[0]
    defaults = {}
    for n in ast.walk(main):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "add_argument":
            flag = n.args[0].value.lstrip("-").replace("-", "_")
            for k in n.keywords:
                if k.arg == "default":
                    defaults[flag] = ast.literal_eval(k.value) if isinstance(k.value, ast.Constant) else ast.unparse(k.value)
    return ([k.arg for k in c.keywords], [ast.unparse(k.value) for k in c.keywords], [ast.unparse(x) for x in c.args],
            defaults)


def test_arm_a_names_exactly_main_s_keywords_at_main_s_defaults():
    """The box's keyword set IS main's (names and order), its values come from the same places (main's ``a.<name>`` for
    every keyword but ``group``, the T, and ``device``, the server's), and its argparse defaults are main's."""
    names, values, pos, defaults = _p115_main_call()
    box = _load("p233_box_under_test", LANE / "p233_box.py")
    assert tuple(names) == box.P115_MAIN_KEYWORDS and pos == ["model", "windows"]
    assert values == ["a.phase", "a.prompt", "a.cont", "a.chunk", "a.floor_chunk", "a.group", "cfg.device", "a.ref_dir",
                      "counters", "fwd"]
    assert {k: defaults[k] for k in box.P115_DEFAULTS} == box.P115_DEFAULTS
    a = box._parser().parse_args([])
    assert {k: getattr(a, k) for k in box.P115_DEFAULTS} == box.P115_DEFAULTS
    # the box's one call site: measure_phase(model, windows, **measure_kwargs(...)), nothing else passed
    tree = ast.parse((LANE / "p233_box.py").read_text(encoding="utf-8"))
    qm = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "quality_main")
    calls = [n for n in ast.walk(qm) if isinstance(n, ast.Call) and ast.unparse(n.func) == "q.measure_phase"]
    assert len(calls) == 1 and [ast.unparse(x) for x in calls[0].args] == ["model", "windows"]
    assert [(k.arg, ast.unparse(k.value)) for k in calls[0].keywords] == [(None, "kw")]
    assert "kw = measure_kwargs(arm, T, a, cfg.device, a.ref_dir, counters, fwd)" in ast.unparse(qm)
    # the instrument's own defaults hold: the box passes no arms, ref_kw or stand_in, so A scores OFF_ARMS, S and C ON_ARMS
    consts = {n.targets[0].id: ast.literal_eval(n.value) for n in tree_p115().body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("OFF_ARMS", "ON_ARMS")}
    assert box.PHASE_ARMS == {"A": consts["OFF_ARMS"], "S": consts["ON_ARMS"], "C": consts["ON_ARMS"]}
    assert box.PHASE_ARMS == R.PHASE_ARMS


def tree_p115():
    return ast.parse((B / "p115" / "p115_quality.py").read_text(encoding="utf-8"))


def _measure_return_keys() -> list:
    """The keys of measure_phase's own record, read off its return statement."""
    tree = ast.parse((B / "p115" / "p115_quality.py").read_text(encoding="utf-8"))
    mp = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "measure_phase")
    ret = [n for n in ast.walk(mp) if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)]
    assert len(ret) == 1
    return [k.value for k in ret[0].value.keys]


def _real_phase(arm_file, T):
    """measure_phase's own record for one T, out of a committed P232 quality record (P232's box fields dropped)."""
    rec = json.loads((P232_FIXTURE / arm_file).read_bytes())["phases"][str(T)]
    keys = _measure_return_keys()
    assert set(keys) <= set(rec), set(keys) - set(rec)
    return {k: rec[k] for k in keys}


@pytest.fixture
def staged(box_dir, monkeypatch):
    """p115_quality, p108_box and p233_box imported from the flat stage, as the box imports them; the stage's modules
    are removed afterwards (the library's stay: a second copy of a class would fail isinstance elsewhere)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    monkeypatch.syspath_prepend(str(box_dir))
    import p108_box
    import p115_quality
    import p233_box
    yield p115_quality, p233_box, p108_box
    for n, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None) or ""
        if f and pathlib.Path(f).resolve().parent == pathlib.Path(box_dir).resolve():
            sys.modules.pop(n, None)


def _stub_world(monkeypatch, q, tmp_path, phase_record):
    """build_engine, the glue kernels, the loader, the card and measure_phase stood in, so p115_quality.main() and the
    box run to their measure_phase call and past it on CPU. Returns the captured calls."""
    import torch
    from experts4bit_qlora import serve_paged as sp
    seen = {"calls": [], "built": []}
    stub = types.ModuleType("int4_b32")
    for k in q.KERNELS:
        setattr(stub, k, lambda *a, **kw: None)
    monkeypatch.setitem(sys.modules, "int4_b32", stub)
    gate = types.ModuleType("p115_box")
    gate.arm_env_ok = lambda arm, env, prove: (True, "")          # main's own env gate (not an argument)
    monkeypatch.setitem(sys.modules, "p115_box", gate)
    plan = types.ModuleType("fp8_paged_attn")                     # grouped-nf4-gemm's split plan (reported beside T)
    plan.decode_split_plan = lambda *a, **kw: 11
    monkeypatch.setitem(sys.modules, "fp8_paged_attn", plan)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(model_type="qwen3_moe", num_hidden_layers=2, _commit_hash=R.REV)
            self.lm_head = torch.nn.Linear(4, 4)

        def get_output_embeddings(self):
            return self.lm_head
    model = Toy()

    def build(cfg):
        seen["built"].append(all(getattr(getattr(stub, k), "_p115_counted", False) for k in q.KERNELS))
        info = {"int8_head_bw": {"mode": "1", "source": "explicit", "rows": {"rows_max": 8, "source": "explicit"}},
                "attn_split_cap": {"cap": 16, "source": "default-allowlisted"}, "graph_status": None}
        return types.SimpleNamespace(runner=types.SimpleNamespace(model=model), tokenizer="tok", info=info)
    monkeypatch.setattr(sp, "build_engine", build)
    monkeypatch.setattr(sp, "_capability", lambda device: None)
    monkeypatch.setattr(sp, "_bytes_per_expert", lambda arena: 2654208)
    monkeypatch.setitem(q.LOADERS, "wikitext", lambda tok, n, P, C: [[(i * 7 + j) % 50 for j in range(P + C)] for i in range(n)])

    def measure(m, windows, **kw):
        seen["calls"].append({"model": m, "windows": windows, **kw})
        return json.loads(json.dumps(phase_record))
    monkeypatch.setattr(q, "measure_phase", measure)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "stub")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: types.SimpleNamespace(multi_processor_count=170))
    arena = tmp_path / "nf4.arena"
    arena.write_bytes(b"arena")
    (tmp_path / "nf4.arena.index.json").write_bytes(b"{}")
    for k in ("E4B_PAGED_SPEC", "E4B_PAGED_SPEC_K", "E4B_ATTN_SPLIT_CAP", "E4B_PAGED_FUSE_QKV", "E4B_FUSE_T1_GLUE",
              "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI", "E4B_DENSE_GEMV_BW", "E4B_PAGED_VRAM_GB", "E4B_PAGED_DRAM_GB",
              "E4B_ATTN_SPLIT_FIXED", "E4B_FUSE_PREFILL_GLUE", "E4B_NF4_SMALLM_BW", "E4B_SERVE_EXP_INT4",
              "GNF4_GEMV_BW", "GNF4_GEMV_BW_PLAN", "GNF4_GEMV_BW_DECODE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in {"E4B_PAGED_MODEL": R.MODEL, "E4B_PAGED_REVISION": R.REV, "E4B_PAGED_ARENA": str(arena),
                 "E4B_PAGED_CALIB": str(tmp_path / "calib.json"), "E4B_PAGED_MAX_SEQS": "16", "E4B_PAGED_GRAPHS": "0",
                 "E4B_INT8_HEAD_BW": "1", "E4B_INT8_HEAD_ROWS": "8", "E4B_PAGED_HOT_ROWS": "128",
                 "E4B_PAGED_PLACEMENT": "all-vram",
                 "E4B_PAGED_DEVICE": "cpu"}.items():
        monkeypatch.setenv(k, v)
    return seen, model


def _box_stubs(monkeypatch, box):
    monkeypatch.setattr(box, "_counts", lambda: {})
    monkeypatch.setattr(box, "_telemetry", lambda: {})
    monkeypatch.setattr(box, "_engagement", lambda parts: {})
    monkeypatch.setattr(box, "moe_states", lambda model: [])


def test_arm_a_calls_measure_phase_with_exactly_what_p115_main_passes(staged, monkeypatch, tmp_path):
    """At run time: p115_quality.main() at ``--phase off --texts wikitext --windows 24 --group T --ref-dir REF/T<T>`` and
    the box's arm A, both through the same stubs, hand measure_phase the same model, the same windows and the same
    keywords at every T; both install the kernel counters before the build and count forwards after it."""
    q, box, _ = staged
    seen, model = _stub_world(monkeypatch, q, tmp_path, _real_phase("quality_R.json", 8))
    _box_stubs(monkeypatch, box)
    ref = tmp_path / "ref"
    for T in box.GROUPS:
        assert q.main(["--phase", "off", "--out", str(tmp_path / f"main_T{T}.json"), "--ref-dir", str(ref / f"T{T}"),
                       "--texts", "wikitext", "--windows", "24", "--group", str(T)]) == 0
    main_calls = list(seen["calls"])
    seen["calls"].clear()
    monkeypatch.setenv("P233_ARM", "A")
    monkeypatch.setenv("E4B_SHA", "a" * 40)
    assert box.main(["--mode", "quality", "--out", str(tmp_path / "quality_A.json"), "--ref-dir", str(ref)]) == 0
    box_calls = seen["calls"]
    assert len(main_calls) == len(box_calls) == len(box.GROUPS) and all(seen["built"]) and len(seen["built"]) == 3

    def norm(c):
        return {k: (type(v).__name__ if k in ("counters", "fwd") else v) for k, v in c.items() if k not in ("model", "windows")}
    for m, b in zip(main_calls, box_calls):
        assert list(b) == list(m), "the same keywords, in main's order"
        assert norm(b) == norm(m)
        assert b["model"] is m["model"] is model and b["windows"] == m["windows"] and list(b["windows"]) == ["wikitext"]
        assert norm(b)["counters"] == "KernelCounters" and norm(b)["fwd"] == "ForwardCounter"
        assert b["phase"] == "off" and {k: b[k] for k in box.P115_DEFAULTS} == box.P115_DEFAULTS
    assert [c["group"] for c in box_calls] == [1, 8] and [c["ref_dir"] for c in box_calls] == [str(ref / "T1"), str(ref / "T8")]


def test_the_box_reads_measure_phases_real_record_shape(staged, monkeypatch, tmp_path):
    """The box's arms A and S run against measure_phase's REAL record (P232's committed shakedown records, P232's box
    fields dropped): every field the box and the reducer read is in it, the box keeps the instrument's record whole
    and adds its own beside it, and S's argmax read stays unread (never invented) when the scorer was not called."""
    q, box, _ = staged
    keys = _measure_return_keys()
    read = {"group", "arms", "windows", "windows_sha256", "prompt", "cont", "per_window", "engagement", "rep_identical",
            "seconds"}
    assert read <= set(keys)
    off, on = _real_phase("quality_R.json", 8), _real_phase("quality_K.json", 8)
    assert {"window", "nll", "argmax_agree"} <= set(off["per_window"]["wikitext"]["R"][0])
    assert {"window", "nll", "argmax_agree", "kl"} <= set(on["per_window"]["wikitext"]["ON"][0])
    assert {"window", "nll", "argmax_agree", "kl"} <= set(off["per_window"]["wikitext"]["rep"][0])
    e = off["engagement"]["wikitext"]["R"][0]
    assert {"graph_status", "graph_stats"} <= set(e) and all(v.startswith("eager") for v in e["graph_status"].values())
    for arm, rec, extra in (("A", off, {}), ("S", on, {"E4B_PAGED_PLACEMENT": "solver", "E4B_PAGED_VRAM_GB": "8.0",
                                                        "E4B_PAGED_DRAM_GB": "3.0"})):
        seen, _ = _stub_world(monkeypatch, q, tmp_path, rec)
        _box_stubs(monkeypatch, box)
        for k, v in {"P233_ARM": arm, "E4B_SHA": "a" * 40, **extra}.items():
            monkeypatch.setenv(k, v)
        out = tmp_path / f"quality_{arm}.json"
        assert box.main(["--mode", "quality", "--out", str(out), "--ref-dir", str(tmp_path / "ref")]) == 0
        got = json.loads(out.read_bytes())
        assert sorted(got["phases"]) == ["1", "8"] and got["status"] == "ok" and got["arm"] == arm
        for T in ("1", "8"):
            p = got["phases"][T]
            assert {k: p[k] for k in keys} == json.loads(json.dumps(rec, default=str)), "the instrument's record, whole"
            assert {"attn_split_cap", "quality_kv", "paths", "argmax", "head_calls", "telemetry", "wall_s"} <= set(p)
        assert got["config"]["placement"] == ("all-vram" if arm == "A" else "solver")
        assert got["levers"]["int8_head_bw"]["rows"] == {"rows_max": 8, "source": "explicit"}
        assert {"model", "isa", "cpu_kernel_threads"} <= set(got["host_cpu"]) and got["arena"]["sha256"]
        assert got["served_revision"] == R.REV and got["windows_sha256"] == q.windows_sha(seen["calls"][0]["windows"]["wikitext"])
        if arm == "S":
            assert "unread" in got["phases"]["1"]["argmax"], "the scorer was stubbed out: S's argmax read is unread"


def test_the_reducer_reads_real_per_window_values():
    """The reducer's arithmetic on real per-window records (P232's T = 8 R, rep and K, as A, its floor and S; B16 as C):
    S against A is the gap, pooled argmax and KL P232's own reducer reads from the same windows, and the read is SANE."""
    p232 = _load("p232_reduce_for_p233", B / "p232" / "p232_reduce.py")
    real = {a: json.loads((P232_FIXTURE / f"quality_{a}.json").read_bytes())["phases"]["8"]["per_window"]["wikitext"]
            for a in ("R", "K", "B16")}
    q = {a: R._fake_quality(a) for a in R.ARMS}
    for T in ("1", "8"):
        pa = q["A"]["phases"][T]["per_window"]["wikitext"]
        pa["R"] = real["R"]["R"]
        pa["rep"] = real["R"]["rep"][:int(T)]
        q["S"]["phases"][T]["per_window"]["wikitext"]["ON"] = real["K"]["ON"]
        q["C"]["phases"][T]["per_window"]["wikitext"]["ON"] = real["B16"]["ON"]
    v = R.reduce(q, R.E)
    assert v["verdict"] == "SOLVER_SANE", v["reasons"][:3]
    x = v["quality_unrounded"]["8"]
    ag = p232.agree_counts(real["K"]["ON"])
    assert x["bias"] == p232.compare(real["K"]["ON"], real["R"]["R"])["bias"]
    assert x["argmax_counts"] == list(ag) and x["kl"] == p232._mean([w["kl"] for w in real["K"]["ON"]])
    assert x["C_vs_A"]["bias"] == p232.compare(real["B16"]["ON"], real["R"]["R"])["bias"]


# ------------------------------------- the maintainer's second requirement: the int8 head on at the solver placement --

def _toy(state):
    """A model with a family, an output head (vocab x hidden) and one MoE layer's tier state, as build_engine sees it
    after its tier and before its head steps (tests/test_int8_head_rows_default.py's toy plus a tiered state)."""
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(model_type="qwen3_moe")
            self.embed = nn.Embedding(40, 16, dtype=torch.bfloat16)
            self.lm_head = nn.Linear(16, 40, bias=False, dtype=torch.bfloat16)
            self.moe = nn.Module()
            self.moe._hot_residency = state

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.lm_head

        def set_output_embeddings(self, m):
            self.lm_head = m
    return Toy()


@pytest.fixture
def k37(monkeypatch):
    """A grouped-nf4-gemm with K37: ``dense_i8`` importable, with stub ``linear_i8`` and ``plan`` (as
    tests/test_int8_head_rows_default.py's fixture)."""
    torch = pytest.importorskip("torch")
    fake = types.ModuleType("dense_i8")
    fake.linear_i8 = lambda x, q, s: ((x.float() @ q.float().T) * s).to(torch.bfloat16)
    fake.plan = lambda n, k, m: (32, 256, 4)
    monkeypatch.setitem(sys.modules, "dense_i8", fake)
    return fake


def _tiered_state():
    """The real ``_HotResidency`` predicate on a three-tier split (two of eight experts resident, two in DRAM, four on
    NVMe): ``_all_hot`` is False, so the layer takes the tiered forward."""
    torch = pytest.importorskip("torch")
    from experts4bit_qlora.engines import hot_residency as hr
    st = object.__new__(hr._HotResidency)
    st.is_hot = torch.tensor([True, True] + [False] * 6)
    st.g2h = torch.tensor([0, 1] + [-1] * 6)
    st.hot_ids, st.dram_ids = torch.tensor([0, 1]), torch.tensor([2, 3])
    st.nvme_set = frozenset(range(4, 8))
    st.collapse_resident, st.amort, st._all_hot_cache = True, None, None
    return st


def test_the_int8_head_resolves_on_at_the_solver_placement(monkeypatch, k37):
    """S's knobs read by the shipped config, then build_engine's two head steps in its order on a toy model whose MoE
    layer is tiered: the head is an Int8HeadLinear, its rows resolve to 8 on K37, both sources ``explicit``, as the
    reducer requires (Amendment 1). The real kernels (K36's gemv and quantizer, K37's linear_i8) and the real tier need
    CUDA; here they are stood in, so what this shows is that the head's resolution does not depend on the placement."""
    torch = pytest.importorskip("torch")
    from experts4bit_qlora import serve_paged as sp
    from experts4bit_qlora.engines import int8_head_bw as ih
    box = _load("p233_box_head", LANE / "p233_box.py")
    monkeypatch.setattr(sp, "_capability", lambda device: None)
    for k in ("E4B_PAGED_SPEC", "E4B_PAGED_SPEC_K", "E4B_PAGED_SPEC_HEAD", "E4B_PAGED_BUCKETS"):
        monkeypatch.delenv(k, raising=False)
    env = {"E4B_PAGED_PLACEMENT": "solver", "E4B_PAGED_VRAM_GB": "8.0", "E4B_PAGED_DRAM_GB": "3.0",
           "E4B_PAGED_GRAPHS": "0", "E4B_INT8_HEAD_BW": "1", "E4B_INT8_HEAD_ROWS": "8", "E4B_PAGED_MAX_SEQS": "16",
           "E4B_PAGED_HOT_ROWS": "128"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert box.arm_env_ok("S", {**env})[0]
    cfg = sp.PagedServeConfig.from_env()
    assert (cfg.placement, cfg.vram_gb, cfg.dram_gb, cfg.graphs) == ("solver", 8.0, 3.0, False)
    assert (cfg.int8_head_bw, cfg.int8_head_rows) == ("1", 8)
    assert cfg.hot_rows == 128, "the arm's named rows are the config's (Amendment 3)"
    st = _tiered_state()
    model = _toy(st)
    pl = box.placement_record([st], cfg.placement, cfg.vram_gb, cfg.dram_gb, 10)
    assert pl["experts"] == {"vram": 2, "dram": 2, "nvme": 4} and pl["layers"] == {"collapse": 0, "tiered": 1}

    def gemv(x, q, s):
        return (s * (q.float() @ x.float().reshape(-1))).to(torch.bfloat16).reshape(1, -1)

    def quantize(w):
        s = (w.float().abs().amax(dim=1) / 127.0).clamp_min(1e-12)
        return torch.round(w.float() / s[:, None]).clamp(-127, 127).to(torch.int8), s
    head = sp.apply_int8_head_bw_resolved(model, cfg, gemv=gemv, quantize=quantize)
    rows = sp.apply_int8_head_rows_resolved(model, cfg, head)
    assert isinstance(model.lm_head, ih.Int8HeadLinear)
    assert (head["mode"], head["source"], head["modules"]) == ("1", "explicit", 1)
    assert rows == {"rows_max": 8, "kernel": "dense_i8.linear_i8 (K37)", "plan": {str(m): [32, 256, 4] for m in range(2, 9)},
                    "source": "explicit"}
    assert model.lm_head.rows_max == 8 and model.lm_head._linear is k37.linear_i8
    assert box.static_path(st) == "tiered", "the head steps leave the tier as it was"
    rec = {"levers": {"int8_head_bw": {**head, "shape": R.HEAD_SHAPE, "rows": rows}}}
    assert R.head_faults("S", rec) == [], "the reducer reads this head as registered (the toy's shape aside)"
    # build_engine runs the two steps unconditionally, after the tier, whatever the placement
    tree = ast.parse(__import__("inspect").getsource(sp.build_engine).replace("\r\n", "\n"))
    fn = tree.body[0]
    top = [ast.unparse(s) for s in fn.body]
    assert "int8_head = apply_int8_head_bw_resolved(model, cfg)" in top
    assert "int8_head_rows = apply_int8_head_rows_resolved(model, cfg, int8_head)" in top
    assert top.index("int8_head = apply_int8_head_bw_resolved(model, cfg)") > next(
        i for i, s in enumerate(top) if s.startswith("n = enable_hybrid_tier("))


# ---------------------------------------------------------------------------------------------- files, mutants --

def test_the_new_files_are_lf_only_and_the_shell_scripts_parse():
    for f in NEW_FILES:
        assert b"\r" not in f.read_bytes(), f
    if shutil.which("bash") is None:
        return
    for f in ("p233_run.sh", "p233_drive.sh"):
        out = subprocess.run(["bash", "-n", str(LANE / f)], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr


REDUCER_MUTANTS = [
    ("    return abs(k - round(k)) < 1e-6", "    return True"),
    # P153's bars loosened, the argmax bar dropped, or read rounded
    ("Q_BIAS, Q_ARGMAX = 0.02, 0.95 ", "Q_BIAS, Q_ARGMAX = 0.03, 0.95 "),
    ("Q_BIAS, Q_ARGMAX = 0.02, 0.95 ", "Q_BIAS, Q_ARGMAX = 0.02, 0.94 "),
    ('        if not agree_at_least(tuple(x["argmax_counts"]), Q_ARGMAX):', "        if False:"),
    ('        if abs(x["bias"]) > Q_BIAS:', '        if round(abs(x["bias"]), 4) > Q_BIAS:'),
    # one T enough for the verdict
    ('        why_all += [f"T={T}: {y}" for y in why]', '        why_all += [f"T={T}: {y}" for y in why if T == 1]'),
    # C read as the ruling arm
    ("        s, c = _vs(ss, rr), _vs(cc, rr)", "        s, c = _vs(cc, rr), _vs(ss, rr)"),
    # the tripwire: a zero floor falls to 0, a wire moved, or the wire changing the verdict
    ('        bound = max(TRIP_KL_MULT * (x["floor_kl"] or 0.0), TRIP_KL_ABS)', '        bound = TRIP_KL_MULT * (x["floor_kl"] or 0.0)'),
    ("TRIP_BIAS, TRIP_ARGMAX, TRIP_KL_MULT, TRIP_KL_ABS = 0.005, 0.98, 3.0, 0.002",
     "TRIP_BIAS, TRIP_ARGMAX, TRIP_KL_MULT, TRIP_KL_ABS = 0.01, 0.98, 3.0, 0.002"),
    ("TRIP_BIAS, TRIP_ARGMAX, TRIP_KL_MULT, TRIP_KL_ABS = 0.005, 0.98, 3.0, 0.002",
     "TRIP_BIAS, TRIP_ARGMAX, TRIP_KL_MULT, TRIP_KL_ABS = 0.005, 0.97, 3.0, 0.002"),
    ("TRIP_BIAS, TRIP_ARGMAX, TRIP_KL_MULT, TRIP_KL_ABS = 0.005, 0.98, 3.0, 0.002",
     "TRIP_BIAS, TRIP_ARGMAX, TRIP_KL_MULT, TRIP_KL_ABS = 0.005, 0.98, 5.0, 0.002"),
    ("    if not why_all:\n", "    if not why_all and not fired:\n"),
    ('    if t == ["unread"] or t is None:', "    if t is None:"),
    # a shakedown read as a reading, mixed with one, or at the reading's windows
    ('    if shake:\n        out.update(verdict="SHAKEDOWN"', '    if False:\n        out.update(verdict="SHAKEDOWN"'),
    ("    if len(flags) != 1:", "    if False:"),
    ("    W = SHAKE_WINDOWS if shake else Q_WINDOWS\n    void = []", "    W = Q_WINDOWS\n    void = []"),
    # the card, the budgets, graphs, a captured pass
    ('    if not shake and (r.get("cc"), r.get("sm_count")) != (CC, SMS):', "    if False:"),
    ('    if arm in BUDGET_GB and (cfg.get("vram_gb"), cfg.get("dram_gb")) != BUDGET_GB[arm]:', "    if False:"),
    ('    if cfg.get("graphs") is not False:', '    if cfg.get("graphs") is True:'),
    ('            if any(v == "graph" for v in gs.values()) or any((v or {}).get("replays") for v in st.values()):',
     '            if any(v == "graph" for v in gs.values()):'),
    # the head (Amendment 1): its source or its rows unchecked
    ("!= (HEAD_MODE, HEAD_SOURCE, 1, HEAD_SHAPE):", "[:1] + (HEAD_SOURCE, 1, HEAD_SHAPE) != (HEAD_MODE, HEAD_SOURCE, 1, HEAD_SHAPE):"),
    ('    if (rows.get("rows_max"), rows.get("source")) != (ROWS_MAX, ROWS_SOURCE):', '    if rows.get("rows_max") != ROWS_MAX:'),
    # the placement proof: A's tiers, S's three, C's DRAM, every layer called, a path off its static one
    ('        if ex["vram"] != L * epl[0] or ex["dram"] or ex["nvme"] or by["dram"] or by["nvme"]:', "        if False:"),
    ('        if arm == "S" and not (by["vram"] > 0 and by["dram"] > 0 and by["nvme"] > 0):',
     '        if arm == "S" and not (by["vram"] > 0 and by["nvme"] > 0):'),
    ('        if arm == "C" and not (by["dram"] == 0 and ex["dram"] == 0 and by["nvme"] > 0):',
     '        if arm == "C" and not by["nvme"] > 0:'),
    ("    if idle:\n", "    if False:\n"),
    ("    if off:\n", "    if False:\n"),
    ('    if sum(ex[t] for t in TIERS) != L * epl[0] or any(by[t] != ex[t] * bpe for t in TIERS):', "    if False:"),
    ('        if (p.get("budget_gb") or {}) != {"vram": BUDGET_GB[arm][0], "dram": BUDGET_GB[arm][1]}:', "        if False:"),
    # the same model, windows and defaults in every phase
    ('    if len(shas) != 1 or "null" in next(iter(shas)):', "    if len(shas) != 1:"),
    ("    if len(revs) != 1:", "    if False:"),
    ("    if len(ar) != 1 or None in ar:", "    if len(ar) != 1:"),
    ('        if len({json.dumps(lv[a][k], sort_keys=True) for a in ARMS}) != 1:', "        if False:"),
    ('    if not lv["A"]["fusion_sources"] or any(not str(s).startswith("default") for s in src):', "    if False:"),
    # a missing value read as present, the window ids unchecked, the pass KV unchecked
    ('            if any(not _num(x.get("nll")) for x in xs):', "            if False:"),
    ('            if sorted(x.get("window") for x in xs if isinstance(x.get("window"), int)) != list(range(W)):', "            if False:"),
    ('        if (qk.get("max_tokens_per_seq"), qk.get("blocks_per_seq")) != ([SERVED_TOKENS], [SERVED_BLOCKS]) \\',
     '        if False \\'),
    ('        if p.get("group") != T or p.get("arms") != list(PHASE_ARMS[arm]):', '        if p.get("group") != T:'),
]


@pytest.mark.parametrize("old,new", REDUCER_MUTANTS, ids=[f"m{i:02d}" for i in range(len(REDUCER_MUTANTS))])
def test_each_reducer_mutant_fails_the_self_test(tmp_path, old, new):
    src = (LANE / "p233_reduce.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p233_reduce.py").write_bytes(src.replace(old, new).encode("utf-8"))
    out = _run(tmp_path / "p233_reduce.py", "--self-test")
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


BOX_MUTANTS = [
    # the arms' placements and budgets
    ('"S": {PLACEMENT_KNOB: "solver", VRAM_KNOB: "8.0", DRAM_KNOB: "3.0"}', '"S": {PLACEMENT_KNOB: "solver", VRAM_KNOB: "8.0", DRAM_KNOB: "2.0"}'),
    ('"C": {PLACEMENT_KNOB: "solver", VRAM_KNOB: "8.0", DRAM_KNOB: "0"}', '"C": {PLACEMENT_KNOB: "solver", VRAM_KNOB: "8.0", DRAM_KNOB: "3.0"}'),
    ('ARM_VALUE = {"A": {PLACEMENT_KNOB: "all-vram", VRAM_KNOB: None, DRAM_KNOB: None},', 'ARM_VALUE = {"A": {PLACEMENT_KNOB: "all-vram"},'),
    # every arm's knobs (Amendments 1 and 3), and the defaults left unset
    ('ROWS_KNOB: "8", HOT_ROWS_KNOB: "128"}', 'ROWS_KNOB: "4", HOT_ROWS_KNOB: "128"}'),
    ('EVERY_ARM = {GRAPHS_KNOB: "0", HEAD_KNOB: "1", ROWS_KNOB: "8",', 'EVERY_ARM = {GRAPHS_KNOB: "0", ROWS_KNOB: "8",'),
    ('ROWS_KNOB: "8", HOT_ROWS_KNOB: "128"}', 'ROWS_KNOB: "8"}'),
    ('ROWS_KNOB: "8", HOT_ROWS_KNOB: "128"}', 'ROWS_KNOB: "8", HOT_ROWS_KNOB: "64"}'),
    ("DEFAULT_KNOBS = (CAP_KNOB, ", "DEFAULT_KNOBS = ("),
    # measure_phase's keywords: an extra one, another floor chunk, another phase, the references not per T
    ('"ref_dir": os.path.join(ref_dir, f"T{T}"), "counters": counters, "fwd": fwd}',
     '"ref_dir": os.path.join(ref_dir, f"T{T}"), "counters": counters, "fwd": fwd, "arms": ("R", "rep")}'),
    ('"floor_chunk": a.floor_chunk,', '"floor_chunk": a.chunk,'),
    ('    p.add_argument("--floor-chunk", type=int, default=256)', '    p.add_argument("--floor-chunk", type=int, default=512)'),
    ('PHASE = {"A": "off", "S": "on", "C": "on"}', 'PHASE = {"A": "on", "S": "on", "C": "on"}'),
    ('"ref_dir": os.path.join(ref_dir, f"T{T}"),', '"ref_dir": ref_dir,'),
    # the placement record and the path counts
    ('    collapse = bool(getattr(st, "collapse_resident", False) and getattr(st, "amort", None) is None and st._all_hot())',
     '    collapse = bool(getattr(st, "collapse_resident", False) and getattr(st, "amort", None) is None)'),
    ('        nvme = len(nset) if nset is not None else E - vram - dram', '        nvme = E - vram'),
    ('"tiered": c["forward"] - c["_forward_collapsed"],', '"tiered": c["forward"],'),
    ("            vars(st).pop(name, None)", "            pass"),
    # the scorer tap and the argmax differences
    ("        self.mod._score = self.orig\n", "        pass\n"),
    ("    return [p for p, (x, y) in enumerate(zip(am, ref_am)) if x != y] + list(range(min(len(am), len(ref_am)),",
     "    return [p for p, (x, y) in enumerate(zip(am, ref_am)) if x != y] + list(range(max(len(am), len(ref_am)),"),
    # an unread ISA read as False
    ("           if seen else None)", "           if True else None)"),
    # an unreadable telemetry field read as zero
    ("        except (IndexError, ValueError):\n            out[k] = None", "        except (IndexError, ValueError):\n            out[k] = 0.0"),
]


@pytest.mark.parametrize("old,new", BOX_MUTANTS, ids=[f"b{i:02d}" for i in range(len(BOX_MUTANTS))])
def test_each_box_mutant_fails_the_self_test(tmp_path, old, new):
    src = (LANE / "p233_box.py").read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    (tmp_path / "p233_box.py").write_bytes(src.replace(old, new).encode("utf-8"))
    out = _run(tmp_path / "p233_box.py", "--self-test", cwd=tmp_path)
    assert out.returncode != 0 and "FAILED" in out.stdout, out.stdout + out.stderr


# ---- the real record first (Amendment 4): p233-shake-2, the explore-tier shakedown at Amendment 3's head, sha256-pinned
FIXTURE = LANE / "fixtures" / "p233-shake-2"
FIXTURE_SHA256 = {
    "quality_A.json": "a9fc989a2dc5f4f33ba9f3fce8e0fc72d3e6fad47b7be8212d189c56e722cac8",
    "quality_S.json": "b34419e945279e51aef1e656e2890e990fea64f968c418606b19f66722186a44",
    "quality_C.json": "267423c5dd519688855d9f8bb0b6c1d8f7416e8ce366018531a248f91ad8e2e3",
}
FIXTURE_E4B = "9d19079ec20fcefafc697fe6277c39ef75319f82"     # the controller HEAD that staged p233-shake-2's kit
FIXTURE_TRIPWIRE = [
    "T=1: |NLL gap| 0.005613 > 0.005",
    "T=1: argmax agreement 0.978516 < 0.98",
    "T=1: KL 0.003766 > max(3 x floor 0.0, 0.002) = 0.002000",
    "T=8: argmax agreement 0.974609 < 0.98",
    "T=8: KL 0.003739 > max(3 x floor 0.0, 0.002) = 0.002000",
]


def _real_quality():
    """The quality records as the reducer's main() loads them, each file checked against its pin first."""
    assert sorted(p.name for p in FIXTURE.iterdir()) == sorted(FIXTURE_SHA256)
    out = {}
    for name, want in FIXTURE_SHA256.items():
        raw = (FIXTURE / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == want, f"the pinned shakedown record {name} changed"
        out[name[len("quality_"):-len(".json")]] = json.loads(raw)
    return out


def test_the_real_shakedown_record_reduces_to_shakedown_solver_sane_with_its_five_tripwire_lines():
    q = _real_quality()
    assert all(r["shakedown"] is True and r["config"]["hot_rows"] == 128 for r in q.values())
    v = R.reduce(q, FIXTURE_E4B)
    assert v["verdict"] == "SHAKEDOWN" and v["would_read"] == "SOLVER_SANE", v["reasons"][:4]
    assert v["reasons"][:3] == ["a shakedown record is never a reading", "T=1: SANE", "T=8: SANE"]
    assert v["tripwire"] == FIXTURE_TRIPWIRE                     # the bounds, unmoved (Amendment 4), on the real bytes
    assert R.reduce(_real_quality(), "0" * 40)["verdict"] == "VOID"      # the e4b pin is read on the real bytes


def test_the_real_record_made_a_reading_voids_on_the_window_count_alone():
    """Made reading-shaped (shakedown off), the 8-window records VOID, and every reason comes from the window count:
    the windows' shape, the R and ON windows not 0..23, and the pass count W/T. Nothing else in the record is a
    fault."""
    q = _real_quality()
    for r in q.values():
        r["shakedown"] = False
    v = R.reduce(q, FIXTURE_E4B)
    assert v["verdict"] == "VOID" and v["tripwire"] == ["unread"] and len(v["reasons"]) == 22, v["reasons"][:3]
    window = (r"windows \{'wikitext': 8\} x 128 after 512, registered 24 wikitext windows x 128 after 512",
              r"8 (R|ON) windows, registered 24", r"the (R|ON) windows are not 0\.\.23 once each",
              r"the pass's KV \{'passes': (8|1), .*\}: registered 4096 tokens \(256 blocks\) "
              r"with the server's cap 16, "
              r"one KV per group of (1|8)")
    assert all(any(re.fullmatch(rf"quality_[ASC] T=(1|8): {w}", x) for w in window) for x in v["reasons"]), \
        [x for x in v["reasons"] if not any(re.fullmatch(rf"quality_[ASC] T=(1|8): {w}", x) for w in window)]
