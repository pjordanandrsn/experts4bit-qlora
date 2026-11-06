"""PGX reading kit (bench/pgx/PREREG-pgx.md, Amendment 3) at $0, run against this commit.

- the schedule (pgx_schedule.py): each draw's order, the passes, the split by whole draws, the plan's K pinned as a constant,
  and every fit constant read back from the shakedown's committed receipt (bench/pgx/receipts/pgx-shake-r1/);
- the run ids: the runner's and the driver's patterns accept exactly the registered reading ids at the plan's K;
- the tier counters (hook_e4b): on e4b's REAL ``_HybridTier`` (pgx_dram_bench.py), every route wrapped, the decode/prefill
  split right, the cost under 0.1 % of the shakedown's E-hyb step; a hook missing any one route fails; the route list is
  pinned to ``_HybridTier``'s methods at this commit; per-pass deltas from the hook's own output match a direct count;
- the box helper's new readers (tier counters with their invariant, n+1 V8 probes, per-pass S residency, the batch check);
- the reducer: part mode on the shakedown's REAL lifetimes, "not measured" for a missing tier record, Q7's hit-rate half;
- the reading runner (pgx_read.sh): the same interpreter rules as pgx_run.sh, and a stubbed run of box 2 end to end.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PGX = REPO / "bench" / "pgx"
R1 = PGX / "receipts" / "pgx-shake-r1"
RUN = (PGX / "pgx_run.sh").read_text()
READ = (PGX / "pgx_read.sh").read_text()
DRIVE = (PGX / "pgx_drive.sh").read_text()
sys.path.insert(0, str(PGX))
import pgx_box as box  # noqa: E402
import pgx_reduce as red  # noqa: E402
import pgx_schedule as sch  # noqa: E402

REGISTERED = {0, 9, 10, 11, 12, 13, 15, 16, 18, 19, 25, 27, 28, 33, 40, 41, 78, 130}


# ---------------------------------------------------------------- the schedule and the plan
def test_each_draw_rotates_and_instruments_every_sglang_lifetime():
    for d in sch.DRAWS:
        rows = sch.draw_lifetimes(d)
        timed = [a for k, a, b in rows if k == "timed" and a in sch.ARMS]
        order = list(dict.fromkeys(timed))
        assert order[0] == sch.ARMS[d - 1] and sorted(order) == sorted(sch.ARMS)
        for arm in sch.ARMS:
            assert [b for k, a, b in rows if k == "timed" and a == arm] == list(sch.BUDGETS)   # budgets ascend
        for i, (k, a, b) in enumerate(rows):
            if k == "instr":
                assert a in sch.SGL and rows[i + 1] == ("timed", a, b)      # right before its own timed lifetime
        assert rows[-3:] == [("timed", "E-hyb-u", 8), ("timed", "E-allvram-g", "all"), ("timed", "E-allvram-e", "all")]
    assert len(sch.schedule()) == 3 * (12 + 6 + 3)


def test_the_passes_are_the_registered_ones():
    for arm in sch.ARMS:
        for b in sch.BUDGETS:
            assert sch.passes(arm, b) == red.registered_passes(b)
    assert {a: ps for a, b, ps in red.CONTROLS} == {a: sch.passes(a, b) for a, b, _ in red.CONTROLS}


def test_the_plan_is_k2_and_the_kit_carries_it():
    assert sch.PLAN_K == 2 == sch.fewest_k()
    p = sch.plan()
    by_k = {s["k"]: s for s in p["splits"]}
    assert not by_k[1]["fits"] and by_k[2]["fits"] and by_k[3]["fits"]
    assert [b["draws"] for b in by_k[2]["boxes"]] == [[1, 2], [3]]
    assert all(b["override"] for b in by_k[2]["boxes"])                  # both over 6 h: the per-run override
    assert all(b["worst_usd"] <= sch.RUN_WORST_USD for b in by_k[2]["boxes"])
    assert sch.box_cap_s() == int((15.0 - 2.40) / 0.85 * 3600)
    assert p["box_ids"] == ["pgx-4090-1-2-1", "pgx-4090-1-2-2"]


def test_the_box_need_and_guard():
    for spec, draws in (("2-1", [1, 2]), ("2-2", [3])):
        assert sorted({r["draw"] for r in sch.box_rows(spec)}) == draws
        need = sch.box_need_s(draws)
        assert sch.box_fit_s(draws) == need + 1500
        assert sch.main(["--need", spec]) == 0
    assert '"$PY_IMG" $W/pgx_schedule.py --need "$BOX_SPEC"' in RUN


def _r1_json(rel):
    return json.loads((R1 / rel).read_text())


def test_every_fit_constant_is_the_receipts():
    s = sch.SHAKE
    for arm in sch.ARMS:
        lt = _r1_json(f"shake/{arm}/8/d1/lifetime.json")
        p1, p2 = _r1_json(f"shake/{arm}/8/d1/P1.json"), _r1_json(f"shake/{arm}/8/d1/P2.json")
        assert s["wall_s"][arm] == lt["wall_s"]
        assert s["p1_s"][arm] == pytest.approx(p1["wall_s"], abs=0.01)
        assert s["p2_s"][arm] == pytest.approx(p2["wall_s"], abs=0.01)
        assert s["tpot_s"][arm] == pytest.approx(p2["tpot_p50"], abs=1e-6)
    for arm in sch.SGL:
        assert s["instr_i1_s"][arm] == pytest.approx(_r1_json(f"instr/{arm}/I1.json")["wall_s"], abs=0.01)
    pv = _r1_json("place_preview.json")
    for (arm, b), v in s["vram_share"].items():
        assert v == pytest.approx(pv[f"{arm}|{b}"]["vram_mass_share"], abs=1e-4)
    calib = (R1 / "logs" / "calibrate.log").read_text()
    assert f"h2d single-copy {s['link_bps'] / 1e9:.2f} GB/s" in calib
    outer = (R1 / "outer.log").read_text()

    def at(pat):
        m = re.search(r"\[2026-11-06T(\d\d):(\d\d):(\d\d)Z\] pgx: " + pat, outer)
        return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
    started = re.search(r"STARTED 2026-11-06T(\d\d):(\d\d):(\d\d)Z", outer)
    t0 = int(started[1]) * 3600 + int(started[2]) * 60 + int(started[3])
    assert at("instrument lifetime S4") - t0 == s["setup_s"]
    assert at("bake the NF4 arena") - at("fetch Qwen/Qwen3-30B-A3B @") == s["fetch_s"]
    assert at("lifetime S8") - at("instrument lifetime S8") == s["instr_wall_s"]["S8"]
    assert at("lifetime S4 \\(8 GB\\)") - at("instrument lifetime S4") == s["instr_wall_s"]["S4"]
    assert _r1_json("shake_verdict.json")["fit"]["peak_rss_max_gib"] == s["peak_rss_gib_max"]


def test_the_kit_cites_the_receipt_in_tree_not_the_private_store():
    """The fit cites bench/pgx/receipts/pgx-shake-r1/; the private store's own layout never appears in the kit."""
    assert sch.SHAKE["receipt"] == "bench/pgx/receipts/pgx-shake-r1" and (REPO / sch.SHAKE["receipt"]).is_dir()
    for f in ("pgx_schedule.py", "pgx_read.sh", "PREREG-pgx.md"):
        assert "receipts/experts4bit-qlora" not in (PGX / f).read_text(), f


# ---------------------------------------------------------------- the run ids
_CANDIDATES = [f"pgx-4090-{a}{s}" for a in ("1", "2", "r1", "r2", "r3", "r4") for s in
               ("", "-1-1", "-2-1", "-2-2", "-2-3", "-3-1", "-3-3", "-2-0")] + ["pgx-shake-1", "pgx-4090-1-2-1x"]


def _case_accepts(text: str, rid: str) -> bool:
    pat = re.search(r"\n\s*(pgx-4090-1-2-\[12\]\|pgx-4090-r\[1-3\]-2-\[12\])\)", text)[1]
    r = subprocess.run(["bash", "-c", f'case "$1" in {pat}) exit 0;; *) exit 1;; esac', "_", rid])
    return r.returncode == 0


def test_the_runner_and_driver_accept_exactly_the_plan_ids():
    def registered(rid):
        try:
            k, _i, _n = sch.parse_run_id(rid)
        except ValueError:
            return False
        return k == sch.PLAN_K
    for rid in _CANDIDATES:
        want = registered(rid)
        assert _case_accepts(RUN, rid) == want == _case_accepts(DRIVE, rid), rid
    assert sch.retry_id(2, 2, 2) == "pgx-4090-r2-2-2"
    assert sch.main(["--run-id", "pgx-4090-1-3-1"]) == 2 and sch.main(["--run-id", "pgx-4090-1-2-2"]) == 0


# ---------------------------------------------------------------- the tier counters, on e4b's real hybrid module
def _hybrid_importable() -> bool:
    r = subprocess.run([sys.executable, "-c", "import experts4bit_qlora.engines.hybrid"], capture_output=True,
                       env={**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")])})
    return r.returncode == 0


needs_e4b = pytest.mark.skipif(not _hybrid_importable(), reason="needs e4b importable (torch, bitsandbytes)")


def _bench(hook_dir: Path, tmp_path: Path, calls: int = 20_000):
    env = {**os.environ, "PGX_DRAM_TIME_OUT": str(tmp_path / "d.jsonl"), "PGX_DRAM_TIME_PERIOD_S": "3600",
           "PYTHONPATH": os.pathsep.join([str(hook_dir), str(REPO), os.environ.get("PYTHONPATH", "")])}
    r = subprocess.run([sys.executable, str(PGX / "pgx_dram_bench.py"), "--calls", str(calls)], env=env,
                       capture_output=True, text=True, cwd=str(tmp_path))
    return r.returncode, json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {"err": r.stderr}


@needs_e4b
def test_the_tier_counters_on_the_real_hybrid_module(tmp_path):
    rc, out = _bench(PGX / "hook_e4b", tmp_path)
    assert rc == 0 and out["ok"], out
    assert all(out["wrap"].values()) and out["bound"] < 0.001
    assert (tmp_path / "d.jsonl").exists()


@needs_e4b
@pytest.mark.parametrize("route", ["forward", "_cold_contrib", "_dram_contrib", "_dram_on_gpu", "_cold_on_cpu"])
def test_a_hook_missing_any_route_fails(tmp_path, route):
    mut = tmp_path / "hook_mut"
    mut.mkdir()
    src = (PGX / "hook_e4b" / "sitecustomize.py").read_text()
    loop = re.search(r"for r, fn in \((.*?)\):\n", src, re.S)
    entry = re.compile(rf'\("{route}", {route}\),?\s*')
    assert len(entry.findall(loop[1])) == 1, route
    mutant = src[:loop.start(1)] + entry.sub("", loop[1]) + src[loop.end(1):]
    (mut / "sitecustomize.py").write_text(mutant)
    rc, out = _bench(mut, tmp_path)
    assert rc == 1 and out.get("wrap", {}).get(route) is False, out


def test_the_route_list_is_pinned_to_the_hybrid_tier_at_this_commit():
    """Every _HybridTier method at this commit, by inspection: the five that take a routed row (the per-layer entry, the
    non-VRAM dispatcher, both DRAM routes, the host NVMe route) are the hook's ROUTES; the rest compute no routed row
    (helpers inside those five, construction, accounting, prefetch, swaps). A new method fails this until inspected."""
    src = (REPO / "experts4bit_qlora" / "engines" / "hybrid.py").read_text()
    body = src[src.index("class _HybridTier(_NvmeResidency):"):]
    body = body[:re.search(r"\n(class |def )", body[1:]).start() + 1]
    methods = re.findall(r"\n    def (\w+)\(", body)
    assert methods == ["__init__", "swap_expert", "expert_bytes", "arm_amortization", "_count_amortization", "forward",
                       "_submit_prefetch", "prefill_gpu_only", "_dram_on_gpu", "_split_cold_diet", "_cold_contrib",
                       "_dram_contrib", "_dram_contrib_inner", "_cold_to_cpu", "_group", "_cold_to_cpu_deadline",
                       "_cold_on_cpu", "_cpu_over_stacks"], methods
    hook = (PGX / "hook_e4b" / "sitecustomize.py").read_text()
    routes = re.search(r'ROUTES = \(([^)]*)\)', hook)[1]
    assert [r.strip().strip('"') for r in routes.split(",") if r.strip()] == [
        "forward", "_cold_contrib", "_dram_contrib", "_dram_on_gpu", "_cold_on_cpu"]
    # the helpers are reached only from inside a wrapped route
    for helper, caller in (("_dram_contrib_inner", "_dram_contrib"), ("_cpu_over_stacks", "_dram_contrib_inner"),
                           ("_split_cold_diet", "_cold_contrib")):
        assert f"self.{helper}(" in body, helper


@needs_e4b
def test_per_pass_deltas_match_a_direct_count(tmp_path):
    out = tmp_path / "d.jsonl"
    code = r'''
import json, sys, time
sys.path.insert(0, sys.argv[2])
import experts4bit_qlora.engines.hybrid as h
from pgx_dram_bench import _T
t = object.__new__(h._HybridTier); t.amort = None; t._dram_contrib_inner = lambda *a: None
def step(rows, routed, nonvram, dram):
    try: t.forward(_T(rows), _T(rows, routed), None)
    except Exception: pass
    try: t._cold_contrib(_T(rows), None, None, None, _T(nonvram))
    except Exception: pass
    t._dram_contrib(_T(rows), None, None, None, _T(dram), None, None, None)
h._pgx_dram_append(); time.sleep(1.1); w0 = int(time.time()) + 1; time.sleep(1.1)
for _ in range(10): step(1, 384, 40, 40)          # 10 decode tokens, a full 48 x 8 routed each
step(512, 512 * 8, 900, 900)                      # one prefill chunk: never in the decode counts
time.sleep(1.1); w1 = int(time.time()); time.sleep(1.1); h._pgx_dram_append()
h._pgx_dram_append(); w2 = w1 + 1; time.sleep(1.2); w3 = int(time.time()) + 1; time.sleep(1.2)
for _ in range(10): step(1, 380, 40, 40)          # a missed route: 4 slots a token the hook never sees
time.sleep(1.1); w4 = int(time.time()); time.sleep(1.1); h._pgx_dram_append()
print(json.dumps({"A": [w0, w1], "B": [w3, w4]}))
'''
    env = {**os.environ, "PGX_DRAM_TIME_OUT": str(out), "PGX_DRAM_TIME_PERIOD_S": "3600",
           "PYTHONPATH": os.pathsep.join([str(PGX / "hook_e4b"), str(REPO), os.environ.get("PYTHONPATH", "")])}
    r = subprocess.run([sys.executable, "-c", code, "x", str(PGX)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    win = {k: tuple(v) for k, v in json.loads(r.stdout.strip().splitlines()[-1]).items()}
    tiers = box.e_tiers(str(out), win, {"A": 10, "B": 10}, layers=48, top_k=8)
    a, b = tiers["A"], tiers["B"]
    assert (a["routed"], a["want"]) == (3840, 3840) and a["hit_rate"] == pytest.approx(1 - 400 / 3840)
    assert a["nonvram_is_dram"] is True and a["dram_s_per_token"] >= 0
    assert (b["routed"], b["want"]) == (3800, 3840) and b["hit_rate"] is None          # the invariant: not measured


# ---------------------------------------------------------------- the box helper's readers
def test_v8_counts_every_pass_between_its_probes(tmp_path):
    def line(vh, vt):
        return box._pager_line("model.layers.0.mlp.experts", vh, vt, vt, vt)
    parts = [line(10, 20), line(30, 50), line(40, 70), line(70, 120), line(80, 140)]
    log = tmp_path / "v.log"
    log.write_text("".join(parts))
    offs = [sum(len(x) for x in parts[:i + 1]) for i in range(5)]
    v = box.v_counters(str(log), offs, ("P1", "P2", "P3", "P4"))
    assert v["counters_ok"] and [v["passes"][p]["vram_hits"] for p in ("P1", "P2", "P3", "P4")] == [20, 10, 30, 10]
    assert box.v_counters(str(log), offs[:4], ("P1", "P2", "P3", "P4"))["counters_ok"] is False


def test_s_residency_per_pass(tmp_path):
    import pgx_residency as pr
    pol = pr._HostLRU(2)
    rows = []
    for i, (p, ids) in enumerate([("W0", [[0]]), ("P1", [[2]]), ("P1", [[0]]), ("P2", [[1]]), ("P3", [[3]])], 1):
        rows.append({"pass": p, "call": i, "layer": 0, "tokens": 1, "ids": ids, "decode": True,
                     "host_loads": pol.request(sorted({e for t in ids for e in t}))})
    (tmp_path / "route-1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    c = box.s_counters(str(tmp_path), 2, ("P1", "P2", "P3"))
    assert c["counters_ok"] and set(k for k in c if k.startswith("residency_")) == {"residency_P1", "residency_P2",
                                                                                     "residency_P3"}
    assert "residency_W0" not in c                     # the warmup warms the state, never scored
    assert box.s_counters(str(tmp_path), 2, ("P1", "P4"))["counters_ok"] is False      # a pass with no rows


def test_the_batch_check_windows_exclude_the_boundary_seconds(tmp_path):
    import time
    base = int(time.mktime(time.strptime("2026-11-06 07:00:00", "%Y-%m-%d %H:%M:%S")))

    def log(path, vals):
        path.write_text("".join(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base + s))}] Decode batch, "
                                f"#running-req: {n}, #token: 1\n" for s, n in vals))
    timed, instr = tmp_path / "t.log", tmp_path / "i.log"
    log(timed, [(0, 9), (5, 4), (6, 4), (10, 9)])          # the boundary seconds 0 and 10 hold another pass's lines
    log(instr, [(105, 4), (107, 4)])
    w_t, w_i = {"P3": (base, base + 10)}, {"P3": (base + 100, base + 110)}
    assert box.batch_check(str(timed), str(instr), w_t, w_i, ("P3",))["P3"]["state"] == "BATCH_MATCHED"
    log(instr, [(105, 4), (107, 3)])
    assert box.batch_check(str(timed), str(instr), w_t, w_i, ("P3",))["P3"]["state"] == "APPROXIMATE"


def test_a_missing_tier_record_reads_not_measured(tmp_path):
    lt = tmp_path / "lt"
    lt.mkdir()
    (lt / "health.json").write_text(json.dumps({"levers": {"moe_layers": 48, "top_k": 8, "placement_profile":
                                                           {"kind": "measured", "sha256": "ab" * 32}}}))
    (lt / "placement.json").write_text(json.dumps({"experts": {"vram": 3014, "dram": 3130, "nvme": 0}}))
    (lt / "step_trace.jsonl").write_text("{}\n")
    (lt / "windows.txt").write_text("P1 100 200\n")
    _, a = box._args(["lifetime", "--arm", "E-hyb", "--lt", str(lt), "--budget", "8", "--passes", "P1",
                      "--windows", str(lt / "windows.txt"), "--health", str(lt / "health.json"),
                      "--placement", str(lt / "placement.json"), "--step-trace", str(lt / "step_trace.jsonl"),
                      "--dram-time", str(lt / "dram_time.jsonl")])
    rec = box.lifetime(a)
    assert rec["counters_ok"] is True                      # the VOID-relevant counters are there
    assert rec["dram_tier_s_per_token"] == {"P1": None} and rec["vram_hit_rate"] == {"P1": None}


# ---------------------------------------------------------------- the reducer
def _reg():
    man = json.loads((PGX / "prompts" / "manifest.json").read_text())
    import hashlib
    return {"profile_sha": hashlib.sha256((PGX / "fit_profile.jsonl").read_bytes()).hexdigest(),
            "budget_bytes": {str(b): b * 10**9 for b in (4, 8, 12)},
            "sets": {k: v["prompts_sha"] for k, v in man["sets"].items()},
            "preview": {k: v["vram_mass_share"] for k, v in _r1_json("place_preview.json").items() if k.startswith("E-hyb")}}


def test_part_mode_on_the_shakedowns_real_lifetimes():
    """The shakedown's REAL records (r1, committed): its four 8 GB lifetimes ran P1 and P2. In a draw-1 part reduction,
    their P1 and P2 rows carry no fault (the real lifetime records pass every lifetime and pass check), and every other
    registered row of draw 1 faults as missing."""
    v = red.reduce([str(R1 / "shake")], _reg(), draws=(1,), part=True)
    assert v["verdict"] == "PART_FAULTS"
    real = {f"{a}|8|1|{p}" for a in sch.ARMS for p in ("P1", "P2")}
    assert not real & set(v["faults"]), {k: v["faults"][k] for k in real & set(v["faults"])}
    want = {"|".join(map(str, k)) for k in red.schedule((1,))} - real
    assert set(v["faults"]) == want


def test_part_mode_reads_part_ok_on_a_whole_draw(tmp_path):
    work = tmp_path / "read"
    for (arm, b, d, p) in red.schedule((3,)):
        src = "E-hyb" if arm.startswith("E-") else arm
        lt_dir = work / arm / str(b) / f"d{d}"
        lt_dir.mkdir(parents=True, exist_ok=True)
        lt = _r1_json(f"shake/{src}/8/d1/lifetime.json")
        lt.update(arm=arm, budget=b, draw=d)
        if arm == "E-hyb-u":
            lt["profile"] = {"kind": "uniform-assumed", "sha256": None}
        if arm in ("S4", "S8", "V8"):
            lt["k_registered"] = lt["k_measured"] = {"S4": {4: 33, 8: 67, 12: 101}, "S8": {4: 17, 8: 35, 12: 52},
                                                     "V8": {4: 17, 8: 35, 12: 52}}[arm][b]
        if arm == "E-hyb":
            lt["vram_tier_bytes"] = min(lt["vram_tier_bytes"], b * 10**9) if b != "all" else lt["vram_tier_bytes"]
        (lt_dir / "lifetime.json").write_text(json.dumps(lt))
        rec = _r1_json(f"shake/{src}/8/d1/P2.json")
        rec["prompts_sha"] = _reg()["sets"][red.PASS_SET[p]]
        (lt_dir / f"{p}.json").write_text(json.dumps(rec))
    v = red.reduce([str(work)], _reg(), draws=(3,), part=True)
    assert v["verdict"] == "PART_OK", v["faults"]
    assert red.main(["--work", str(work), "--registration", str(_write(tmp_path, _reg())), "--mode", "part", "--draws", "3",
                     "--out", str(tmp_path / "v.json")]) == 0


def _write(tmp_path, obj):
    p = tmp_path / "reg.json"
    p.write_text(json.dumps(obj))
    return p


def test_the_attribution_reads_not_measured_without_a_tier_record():
    def cell(t):
        return {"state": "READ", "median": {"tpot_p50": t}}
    cells = {("E-hyb", 8, "P2"): cell(0.19), ("S4", 8, "P2"): cell(0.006), ("S8", 8, "P2"): cell(0.02),
             ("V8", 8, "P2"): cell(0.018), ("E-allvram-g", "all", "P2"): cell(0.004),
             ("E-allvram-e", "all", "P2"): cell(0.008), ("E-hyb-u", 8, "P2"): cell(0.47)}
    rows = {("E-hyb", 8, d, "P2"): {"lt_dram": None} for d in sch.DRAWS}
    a = red.attribute(cells, rows, 8, "P2")
    assert a["state"] == "LOSS" and a["not_measured"] == ["dram_host"]
    assert "eager_tax" in a["shares"] and "placement" in a["shares"]
    rows = {("E-hyb", 8, d, "P2"): {"lt_dram": 0.15} for d in sch.DRAWS}
    assert "not_measured" not in red.attribute(cells, rows, 8, "P2")


def test_residency_labels():
    s = {"counters": {"residency_P3": {"hit_rate": 0.9}, "batch": {"P3": {"state": "APPROXIMATE"}}}}
    assert red.residency("S8", s, "P3") == (0.9, "APPROXIMATE")
    assert red.residency("E-hyb", {"vram_hit_rate": {"P2": None}}, "P2") == (None, "NOT_MEASURED")
    assert red.residency("E-hyb", {"vram_hit_rate": {"P2": 0.8}}, "P2") == (0.8, "TIER_COUNTERS")


# ---------------------------------------------------------------- the reading runner
def _code_lines(text):
    out, hd = [], None
    for ln in text.splitlines():
        if hd is not None:
            if ln.strip() == hd:
                hd = None
            continue
        if ln.lstrip().startswith("#"):
            continue
        out.append(ln)
        m = re.search(r"<<-?\s*'?([A-Z_]+)'?", ln)
        if m:
            hd = m.group(1)
    return out


def test_the_reading_runner_keeps_amendment_2s_interpreter_rules():
    bare = re.compile(r'(?<![\w/$"\'.-])python(?= +[-$/"])')
    lines = _code_lines(READ)
    assert not [ln for ln in lines if bare.search(ln)]
    assert not [ln for ln in lines if re.match(r"\s*(\.|source)\s", ln)]
    assert "sglang.env" not in READ
    assert {int(x) for x in re.findall(r"\bfinish (\d+)\b", READ)} <= REGISTERED
    assert "img_guard dram-bench" in READ and READ.index("img_guard dram-bench") < READ.index("pgx_dram_bench.py >")
    assert READ.count("e4b_start ") == 1                    # every E arm starts through pgx_run.sh's e4b_start, whose
    i, j = RUN.index("e4b_start(){"), RUN.index('"$PY_IMG" -m experts4bit_qlora.serve_paged')   # guard runs first
    assert i < RUN.index("img_guard e-hyb-server", i) < j
    # the runner sources it only on a read box, after the clock burn, and the shakedown's lifetimes never run there
    i = RUN.index('[ "$PGX_BOX" = read ] && { . $W/pgx_read.sh; exit 0; }')
    assert RUN.index("card_clock.py burn") < i < RUN.index("can_run $INSTR_S instrument_S4")


def test_the_reading_runner_carries_the_registered_k_table_and_knobs():
    import pgx_bytes as pb
    import pgx_place_preview as pp
    kt = pb.k_table(json.loads((PGX / "bytes_per_expert.json").read_text()))
    for arm in ("S4", "S8", "V8"):
        for b in (4, 8, 12):
            assert re.search(rf"\bK_{arm}_{b}={kt[arm][str(b)]}(?!\d)", RUN), (arm, b)
    for b in (4, 8, 12):
        assert re.search(rf"\bEHYB_GIB_{b}={pp.vram_gib(b)}\b", RUN), b


def _stub_box(tmp_path, spec):
    """pgx_read.sh with pgx_run.sh's helpers stubbed: servers that start, a drive that writes a record, a box helper that
    writes the lifetime record, and the REAL pgx_schedule.py and pgx_reduce.py."""
    w = tmp_path / "w"
    (w / "logs").mkdir(parents=True)
    (w / "rt" / "pgx").mkdir(parents=True)
    for f in ("pgx_schedule.py", "pgx_reduce.py", "pgx_read.sh", "pgx_dram_bench.py"):
        shutil.copy2(PGX / f, w / f)
    shutil.copy2(PGX / "pgx_reduce.py", w / "rt" / "pgx" / "pgx_reduce.py")
    (w / "prompts").mkdir()
    shutil.copy2(PGX / "prompts" / "manifest.json", w / "prompts" / "manifest.json")
    shutil.copy2(PGX / "fit_profile.jsonl", w / "fit_profile.jsonl")
    shutil.copy2(R1 / "place_preview.json", w / "place_preview.json")
    kt = json.loads((PGX / "bytes_per_expert.json").read_text())["k_table"]
    (w / "box_bytes.json").write_text(json.dumps({"k_table": kt}))
    stub_py = tmp_path / "stubpy"
    stub_py.write_text('#!/bin/bash\ncase "$1" in\n  */pgx_dram_bench.py) echo \'{"ok": true}\'; exit 0;;\n'
                       '  */pgx_box.py) echo "BOX $*" >> "$STUB_LOG"; exit 0;;\nesac\nexec ' + sys.executable + ' "$@"\n')
    stub_py.chmod(0o755)
    harness = tmp_path / "h.sh"
    harness.write_text(f'''
set -uo pipefail
W={w}; cd $W; PY_IMG={stub_py}; BOX_SPEC={spec}; RESERVE=0; PGX_DEADLINE_EPOCH=$(( $(date +%s) + ${{LEFT_S:-999999}} ))
PORT_SGL=1; PORT_VLLM=2; PORT_E4B=3; SRV_PID=""
INT4=i; INT4_REV=r; FP8=f; FP8_REV=r; EHYB_GIB_4=3.725; EHYB_GIB_8=7.451; EHYB_GIB_12=11.176
K_S4_4=33; K_S4_8=67; K_S4_12=101; K_S8_4=17; K_S8_8=35; K_S8_12=52; K_V8_4=17; K_V8_8=35; K_V8_12=52
say(){{ echo "say: $*"; }}
finish(){{ echo "FINISH $1"; exit "$1"; }}
img_guard(){{ echo "GUARD $1" >> "$STUB_LOG"; }}
can_run(){{ local left=$((PGX_DEADLINE_EPOCH - $(date +%s))); [ "$1" -le "$left" ] || {{ echo "SKIP $2" >> "$STUB_LOG"; return 1; }}; }}
sgl_start(){{ echo "START sgl $3 ${{SGL_DUMP_ENV:-timed}}" >> "$STUB_LOG"; SRV_PID=1; }}
vllm_start(){{ echo "START vllm $1" >> "$STUB_LOG"; SRV_PID=1; }}
e4b_start(){{ echo "START e4b $3 $4" >> "$STUB_LOG"; SRV_PID=1; }}
sampler_start(){{ echo 0; }}
sampler_stop(){{ :; }}
stop_pid(){{ :; }}
probe(){{ echo 1; }}
curl(){{ :; }}
drive(){{ echo "DRIVE $5 $3 c$6" >> "$STUB_LOG"; echo '{{"faults": 0, "out_tok_s": 1}}' > "$4"; }}
. $W/pgx_read.sh
''')
    return harness


def test_a_stubbed_reading_box_runs_its_schedule_in_order(tmp_path):
    h = _stub_box(tmp_path, "2-2")
    log = tmp_path / "stub.log"
    r = subprocess.run(["bash", str(h)], env={**os.environ, "STUB_LOG": str(log)}, capture_output=True, text=True)
    assert "FINISH 27" in r.stdout, r.stdout + r.stderr      # the stub writes no real counters: the part reduction faults
    calls = log.read_text().splitlines()
    starts = [c for c in calls if c.startswith("START")]
    want = sch.box_rows("2-2")
    assert len(starts) == len(want) == 21
    for c, row in zip(starts, want):
        if row["arm"] in sch.SGL:
            assert (" PGX_ROUTE_DUMP=" in c) == (row["kind"] == "instr"), (c, row)
    drives = [c for c in calls if c.startswith("DRIVE")]
    assert len(drives) == sum(len(r["passes"]) for r in want)
    assert all(c.endswith("c4") == (c.split()[2] == "P3") for c in drives)
    assert "GUARD dram-bench" in calls[0]
    assert re.search(r"part reduction \(2-2, draws 3 \)", r.stdout)


def test_a_lifetime_that_does_not_fit_is_skipped_and_the_box_faults(tmp_path):
    h = _stub_box(tmp_path, "2-2")
    log = tmp_path / "stub.log"
    r = subprocess.run(["bash", str(h)], env={**os.environ, "STUB_LOG": str(log), "LEFT_S": "1000"},
                       capture_output=True, text=True)
    assert "FINISH 27" in r.stdout
    assert any(c.startswith("SKIP ") for c in log.read_text().splitlines())


def test_the_page_states_the_plan_the_kit_computes():
    page = (PGX / "PREREG-pgx.md").read_text()
    am3 = page[page.index("## Amendment 3"):]
    p = sch.plan()
    assert f"Setup fit = 950 - 584 + 2,750 = **{p['setup_s']:,} s**" in am3
    assert f"**{p['draw_s'][1]:,}**" in am3
    assert f"= {p['cap_s']:,} s." in am3
    rows = {s["k"]: s for s in p["splits"]}
    for k, s in rows.items():
        for b in s["boxes"]:
            assert f"{b['guard_s']:,} s = {b['guard_h']} h" in am3, (k, b)
            assert f"{b['worst_usd']}" in am3, (k, b)
        assert f"{s['total_usd']}" in am3, k
    assert f"fits with\n  {p['cap_s'] - rows[2]['boxes'][0]['guard_s']} s to spare" in am3 or \
        f"{p['cap_s'] - rows[2]['boxes'][0]['guard_s']} s to spare" in am3.replace("\n  ", " ")
    assert f"**K = {sch.PLAN_K}.**" in am3
    staged = (PGX / "staged-pgx.sha256").read_text().splitlines()
    assert f"{len(staged)} files in `staged-pgx.sha256`" in am3
