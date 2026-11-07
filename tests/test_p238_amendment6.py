"""P238 amendment 6, on CPU: the three causes p238-shake-r6 found, each from its real record first.

1. The Unsloth engagement counters were never installed: tc1_arm calls the hook's ``post_import`` before its own ``import unsloth``,
   and unsloth_zoo 2026.10.8's package ``__init__`` (``:555-556``) refuses every import until ``unsloth`` has set
   ``UNSLOTH_IS_PRESENT``. ``p238_hook.post_import`` now imports unsloth first and records the message of any import that still
   fails. Tested on a stand-in unsloth_zoo laid out on disk as the wheel is (its ``__init__`` raises as ``:555`` does), with the
   REAL hook and the REAL counter, each in a fresh interpreter.
2. The card floor reads the measured window (the hook's first step edge to its last), with at least CLOCK_WINDOW_MIN_BUSY busy
   samples. On r6's real clock logs and step edges: the whole log refuses U0/U1/UA at 2647 MHz; the window passes all five.
3. The shakedown's op cells are one constant (p238_schedule.SHAKE_OP_ROWS): the runner passes it as --rows, and the reducer's shake
   mode checks exactly those cells. r6's real op records pass in shake mode and fail in read mode.
"""
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
P238 = ROOT / "bench" / "p238"
COMMON = ROOT / "bench" / "common"
FIX = ROOT / "tests" / "fixtures" / "p238_r6"
# the real records, from p238-shake-r6's receipt (bench/p238/receipts after the lane commits it; the private store until then):
# gpuclk_3..7.txt (loads/gpuclk_<i>.txt), op_1_e4b.json / op_2_unsloth.json (ops/) and verdict_p238.json (its in-box reduction)
# byte for byte; edges_t.json holds each 2k
# load's step-edge times (arm_hook.record.edges[].t, in order) extracted from that load's record, whose own sha256 it states.
FIXTURES = {"gpuclk_3.txt": "199da1c5385e597aa80b1142665b0db731cf61ed2e5610e6854cfd6b306660df",
            "gpuclk_4.txt": "b6ba177d12fadad509f62f4cda267d0d9eeed2b992e77d4441ed56f5fd9cb828",
            "gpuclk_5.txt": "147d2e477a8324f55512b08dcb8c898c7572fa672c3b6091ada648d667648a92",
            "gpuclk_6.txt": "540ff806b02a3d5c3472639658e9e6278be10db2cc127b1b83a0fc9550d63048",
            "gpuclk_7.txt": "9c0b3d725e88a839f0e0a3dd2f32191d95b1f278fbd7c344f39c2edb9a31da90",
            "op_1_e4b.json": "aa86febdaff13a6055758838b4bcc4287063f482165dd847861be7d47d7b6407",
            "op_2_unsloth.json": "e5378d899057eb99afa0c47c0f823670f922e29f3067c47b28ed6329ec3d8462",
            "edges_t.json": "f68ba584882f63322c3e0cfc696686810115e3639388989dcde482897caea983",
            "verdict_p238.json": "e1fcad0c91f55fc861dff7c0e1b48735cb61ee0b0f9f1be5206ee0dd1d39cf40"}
ARMS = {3: "E_F", 4: "E_A", 5: "U0", 6: "U1", 7: "UA"}
WINDOW_MEDIAN = {3: 3000, 4: 3000, 5: 2955, 6: 2947, 7: 2947}         # the busy SM median inside each load's measured window


def _load(name, rel):
    for d in (str(P238), str(COMMON)):
        if d not in sys.path:
            sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _edges(i):
    return [{"t": t} for t in json.loads((FIX / "edges_t.json").read_text(encoding="utf-8"))[str(i)]["t"]]


def test_the_fixtures_are_r6s_records():
    for name, sha in FIXTURES.items():
        assert hashlib.sha256((FIX / name).read_bytes()).hexdigest() == sha, name
    edges = json.loads((FIX / "edges_t.json").read_text(encoding="utf-8"))
    assert sorted(edges) == ["3", "4", "5", "6", "7"]
    for i, arm in ARMS.items():
        e = edges[str(i)]
        assert e["arm"] == arm and len(e["t"]) == 61 and e["t"] == sorted(e["t"]) and len(e["source_sha256"]) == 64


# -- 2. the card clock's measured window ---------------------------------------------------------------------------------------
def test_the_whole_log_refuses_the_unsloth_rows_and_the_window_passes_all_five():
    """r6's record: the whole log's busy median is 2647 MHz for U0/U1/UA (refused) and above the floor for E_F/E_A; inside the
    measured window all five pass, at 3000 / 3000 / 2955 / 2947 / 2947 MHz."""
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    import card_clock
    card = {"name": "NVIDIA GeForce RTX 5090", "cap": [12, 0], "sm_count": 170}
    for i, arm in ARMS.items():
        log = str(FIX / f"gpuclk_{i}.txt")
        whole = R.clock_facts(log)
        ok, why = card_clock.passes(whole, R.CLOCK_FLOOR_MHZ, mode=R.CARD_MODE)
        if arm.startswith("U"):
            assert not ok and whole["sm_median"] == 2647 and "2647" in why, (arm, why)
        else:
            assert ok and whole["sm_median"] >= R.CLOCK_FLOOR_MHZ, (arm, why)
        win = R.clock_window(_edges(i))
        clock = R.clock_facts(log, window=win)
        assert clock["window"] == list(win) and clock["whole_log_sm_median"] == whole["sm_median"]
        assert clock["sm_median"] == WINDOW_MEDIAN[i] and clock["busy_samples"] >= R.CLOCK_WINDOW_MIN_BUSY, (arm, clock)
        assert R.check_card({"card": card, "clock": clock}) == [], arm


def test_no_window_or_a_thin_window_voids():
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    card = {"name": "NVIDIA GeForce RTX 5090", "cap": [12, 0], "sm_count": 170}
    edges = _edges(5)
    for bad in (None, [], [edges[0]], [{"t": "x"}, {"t": 2}], [{"time": 1}, {"time": 2}], list(reversed(edges)), [edges[0], edges[0]]):
        assert R.clock_window(bad) is None, bad
    why = R.check_card({"card": card, "clock": {"window": None}})
    assert why and "no measured window" in why[0]
    assert R.check_card({"card": card, "clock": R.clock_facts(str(FIX / "gpuclk_5.txt"))})     # a whole-log summary: no window
    t0 = edges[0]["t"]
    thin = R.clock_facts(str(FIX / "gpuclk_5.txt"), window=(t0, t0 + 10))
    assert 0 < thin["busy_samples"] < R.CLOCK_WINDOW_MIN_BUSY
    why = R.check_card({"card": card, "clock": thin})
    assert why and "the card clock" in why[0], why


def test_the_row_carries_the_window_from_the_hooks_edges(tmp_path):
    """normalize_load reads the window from the load's own record (arm_hook.record.edges), and a record without edges VOIDs."""
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    for edges, want in ((_edges(5), WINDOW_MEDIAN[5]), (None, None)):
        w = tmp_path / ("e" if edges else "n")
        (w / "loads" / "5").mkdir(parents=True)
        (w / "loads" / "gpuclk_5.txt").write_bytes((FIX / "gpuclk_5.txt").read_bytes())
        cell = {"status": "ok", "env": {}, "arm_hook": {"record": {"edges": edges} if edges else {}}}
        (w / "loads" / "5" / "qwen3_5_unsloth_p238_5_U0_2k.json").write_text(json.dumps(cell), encoding="utf-8")
        row = R.normalize_load(str(w), {"index": 5, "arm": "U0", "point": "2k", "draw": 1, "position": 2}, {}, True)
        assert row["clock"].get("sm_median") == want and (row["clock"]["window"] is None) == (edges is None), row["clock"]


# -- 3. the shakedown's op cells -----------------------------------------------------------------------------------------------
def test_r6s_op_records_pass_in_shake_mode_and_fail_in_read_mode():
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    for name in ("op_1_e4b.json", "op_2_unsloth.json"):
        rec = json.loads((FIX / name).read_text(encoding="utf-8"))
        assert sorted({c["rows"] for c in rec["cells"]}) == [64]
        assert R.op_faults(rec, "shake") == [], (name, R.op_faults(rec, "shake"))
        read = R.op_faults(rec)
        assert read and "missing or unregistered" in read[0], (name, read)


def test_one_constant_names_the_shakedowns_op_cells():
    """The runner passes p238_schedule.py --shake-op-rows (no literal row count), the CLI prints SHAKE_OP_ROWS, and the reducer's
    shake mode checks exactly those rows under both routings."""
    S = _load("p238_schedule", "bench/p238/p238_schedule.py")
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    run = (P238 / "p238_run.sh").read_text(encoding="utf-8")
    line = [ln for ln in run.splitlines() if '[ "$P238_BOX" = shake ] && echo "--rows' in ln]
    assert len(line) == 1 and '--rows $(python3 $W/p238_schedule.py --shake-op-rows)' in line[0], line
    assert "--rows 64" not in run
    out = subprocess.run([sys.executable, str(P238 / "p238_schedule.py"), "--shake-op-rows"], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ",".join(map(str, S.SHAKE_OP_ROWS)) == "64"
    assert R.op_cells("shake") == {(64, "uniform"), (64, "skew")}
    assert R.op_cells("read") == R.op_cells("part") == {(r, k) for r in R.OP_ROWS for k in R.OP_ROUTINGS}


# -- 1. the counter's import order ----------------------------------------------------------------------------------------------
ZOO_INIT = '''import os
# unsloth_zoo 2026.10.8 __init__.py:555-556, the guard as the wheel writes it
if not ("UNSLOTH_IS_PRESENT" in os.environ):
    raise ImportError("Please install Unsloth via `pip install unsloth`!")
'''


def _stand_in(tmp_path, broken=None):
    """unsloth_zoo laid out on disk: the package __init__ with the wheel's guard, temporary_patches and the counter's three
    modules; unsloth's __init__ sets UNSLOTH_IS_PRESENT (unsloth 2026.10.8 __init__.py:14)."""
    pkgs = tmp_path / "pkgs"
    tp = pkgs / "unsloth_zoo" / "temporary_patches"
    tp.mkdir(parents=True)
    (pkgs / "unsloth_zoo" / "__init__.py").write_text(ZOO_INIT, encoding="utf-8")
    (tp / "__init__.py").write_text("", encoding="utf-8")
    for m in ("moe_grouped_fp16", "moe_nf4_grouped", "moe_utils"):
        body = f'raise ImportError("{m}: a stand-in failure")\n' if m == broken else ""
        (tp / f"{m}.py").write_text(body, encoding="utf-8")
    (pkgs / "unsloth").mkdir()
    (pkgs / "unsloth" / "__init__.py").write_text('import os\nos.environ["UNSLOTH_IS_PRESENT"] = "1"\n', encoding="utf-8")
    return pkgs


PROBE = '''import json, os, sys, types
sys.path[:0] = [sys.argv[1], sys.argv[2], sys.argv[3]]
os.environ.pop("UNSLOTH_IS_PRESENT", None)
if sys.argv[4] == "hook":
    import p238_hook
    h = p238_hook.P238Hook()
    h.post_import(types.SimpleNamespace(framework="unsloth", accum=1))
    c, errs = h.counter, h.facts.get("counter_import_errors")
else:                                   # r6's order: the counter installed before any `import unsloth`
    import p238_hook
    from unsloth_moe_counter import UnslothMoeCounter
    c = UnslothMoeCounter().install()
    errs = p238_hook.P238Hook._import_errors(c)
print(json.dumps({"failed": [a for a in list(c.absent) + list(c.dispatch_absent) if ": import failed: " in a], "errors": errs,
                  "unsloth_imported": "unsloth" in sys.modules}))
'''


def _probe(tmp_path, how, broken=None):
    pkgs = _stand_in(tmp_path, broken)
    out = subprocess.run([sys.executable, "-c", PROBE, str(pkgs), str(P238), str(COMMON), how], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_old_order_reproduces_r6s_failure_and_its_message_is_recorded(tmp_path):
    got = _probe(tmp_path, "old")
    assert sorted(got["failed"]) == sorted(f"unsloth_zoo.temporary_patches.{m}: import failed: ImportError"
                                           for m in ("moe_grouped_fp16", "moe_nf4_grouped", "moe_utils")), got
    assert not got["unsloth_imported"]
    assert set(got["errors"]) == {f"unsloth_zoo.temporary_patches.{m}" for m in ("moe_grouped_fp16", "moe_nf4_grouped", "moe_utils")}
    assert all(v == "ImportError: Please install Unsloth via `pip install unsloth`!" for v in got["errors"].values()), got


def test_the_hook_imports_unsloth_first_and_the_counter_installs_cleanly(tmp_path):
    got = _probe(tmp_path, "hook")
    assert got["failed"] == [] and got["errors"] == {} and got["unsloth_imported"], got


def test_the_hook_records_the_message_of_an_import_that_still_fails(tmp_path):
    """With unsloth imported, a module that still fails to import is recorded in the hook's facts with its message."""
    got = _probe(tmp_path, "hook", broken="moe_utils")
    assert got["failed"] == ["unsloth_zoo.temporary_patches.moe_utils: import failed: ImportError"], got
    assert got["errors"] == {"unsloth_zoo.temporary_patches.moe_utils": "ImportError: moe_utils: a stand-in failure"}, got


def test_the_shared_counter_and_p237_do_not_move():
    """The fix lives in P238's hook: the shared counter file is byte-identical to what P237 pins."""
    pin = dict(reversed(ln.split()) for ln in (ROOT / "bench" / "p237" / "staged-p237.sha256").read_text(encoding="utf-8").splitlines())
    assert hashlib.sha256((COMMON / "unsloth_moe_counter.py").read_bytes()).hexdigest() == pin["unsloth_moe_counter.py"]


# -- the exemption: r6's exact record ------------------------------------------------------------------------------------------
IMPORT_FAILED = "import failed: ImportError"


def r6_exact(verdict, started, rc):
    """The page's five conditions for r6's exact record (Amendment 6, "r6's exact record"), on its reduction (verdict_p238.json)
    and its STARTED / exit code. The a7 kit's stop is written against this function. Any shape it cannot read is no."""
    try:
        rows = {(r["point"], r["arm"]): r for r in verdict["rows"]}
        ops = verdict["op_rows"]
        return all((
            started is True and rc == 0,
            (verdict["verdict"], verdict["n_rows"], verdict["n_op_rows"], verdict["n_void"]) == ("VOID", 20, 2, 20),
            all(rows[("2k", a)]["verdict"] == "READ" for a in ("E_F", "E_A")),
            all(rows[("2k", a)]["verdict"] == "VOID" and any(IMPORT_FAILED in w for w in rows[("2k", a)]["why"])
                for a in ("U0", "U1", "UA")),
            len(ops) == 2 and all(o["verdict"] == "VOID" and len(o["why"]) == 1 and "missing or unregistered" in o["why"][0]
                                  for o in ops),
        ))
    except (KeyError, TypeError, IndexError):
        return False


def _verdict():
    return json.loads((FIX / "verdict_p238.json").read_text(encoding="utf-8"))


def test_r6s_real_record_meets_the_five_conditions():
    v = _verdict()
    assert r6_exact(v, True, 0)
    unscheduled = [r for r in v["rows"] if r["point"] != "2k"]
    assert len(unscheduled) == 15 and all(r["why"] == ["status None"] for r in unscheduled)   # not read by the conditions


@pytest.mark.parametrize("label, mutate, started, rc", [
    ("not STARTED", lambda v: None, False, 0),
    ("finished 27", lambda v: None, True, 27),
    ("another reduction line (19 VOID)", lambda v: v.update(n_void=19), True, 0),
    ("a READ box", lambda v: v.update(verdict="SHAKE_OK"), True, 0),
    ("an E 2k row VOID", lambda v: v["rows"][1].update(verdict="VOID", why=["the card clock: x"]), True, 0),
    ("a U 2k row without the ImportError", lambda v: v["rows"][3].update(why=[w for w in v["rows"][3]["why"] if IMPORT_FAILED not in w]),
     True, 0),
    ("a U 2k row READ", lambda v: v["rows"][4].update(verdict="READ", why=[]), True, 0),
    ("an op row with another reason", lambda v: v["op_rows"][0]["why"].append("LAYOUT/PARITY STOP: x"), True, 0),
    ("an op row READ", lambda v: v["op_rows"][1].update(verdict="READ", why=[]), True, 0),
    ("one op row", lambda v: v["op_rows"].pop(), True, 0),
    ("no 2k U0 row", lambda v: v["rows"].pop(2), True, 0),
    # each remaining part of a condition, changed alone (nothing above touches them)
    ("the other E 2k row VOID", lambda v: v["rows"][0].update(verdict="VOID", why=["the card clock: x"]), True, 0),
    ("another step-row count", lambda v: v.update(n_rows=19), True, 0),
    ("another op-row count", lambda v: v.update(n_op_rows=1), True, 0),
    ("an op row whose one reason is another", lambda v: v["op_rows"][0].update(why=["the layout check: parity STOP"]), True, 0),
])
def test_mutated_copies_of_r6s_record_fail_the_conditions(label, mutate, started, rc):
    v = _verdict()
    mutate(v)
    assert not r6_exact(v, started, rc), label


# -- the page -------------------------------------------------------------------------------------------------------------------
def test_the_page_registers_the_window_the_box_and_the_counts():
    page = (P238 / "PREREG-p238.md").read_text(encoding="utf-8")
    a6 = " ".join(page.split("## Amendment 6 (2026-11-07)", 1)[1].split())
    R = _load("p238_reduce", "bench/p238/p238_reduce.py")
    for want in ("`p238-shake-r9`", "`p238-shake-r10` and `p238-shake-r11`", "rises to four, for this one box only",
                 "strike two, and the lane ends", "`P238 VOID: 20 step rows, 2 op rows, 20 VOID`", "`edges[0].t` to `edges[-1].t`",
                 f"fewer than **{R.CLOCK_WINDOW_MIN_BUSY}** busy samples", "`p238_schedule.SHAKE_OP_ROWS = (64,)`",
                 "`bench/common/unsloth_moe_counter.py` does not change", "3000, 3000, 2955, 2947 and 2947 MHz",
                 "**USD 8.21**", "USD 3.86 in all", "this amendment's merge",
                 "every Unsloth row the shakedown schedules (U0, U1 and UA at 2k) VOIDs",
                 "each for its op cells alone (`missing or unregistered`)", "these conditions do not read them"):
        assert want in a6, want
