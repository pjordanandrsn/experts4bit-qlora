"""PGX Z1: the pass driver against a stub OpenAI-compatible SSE server, and the reducer's every rule on synthetic
records (a full clean schedule READs; each registered fault VOIDs or flags exactly what it should). CPU, stdlib."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench", "pgx"))
import pgx_drive as drv  # noqa: E402
import pgx_reduce as red  # noqa: E402


# --------------------------------------------------------------- the stub server ------------------------------------
class _Stub(BaseHTTPRequestHandler):
    mode = "ok"            # ok | short | error
    seen: list = []
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.lock:
            self.seen.append(body)
        if self.mode == "error":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        n = body["max_tokens"] - (1 if self.mode == "short" else 0)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        seed = sum(body["prompt"])
        for i in range(n):
            self.wfile.write(f"data: {json.dumps({'choices': [{'text': f't{seed + i} '}]})}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': n}})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture()
def stub():
    _Stub.mode, _Stub.seen = "ok", []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


PROMPTS = [{"id": f"p{i}", "kind": "code", "token_ids": [i, i + 1, i + 2]} for i in range(6)]


def test_driver_sends_the_registered_request_and_times_every_token(stub):
    rec = drv.run_pass(stub, "m", PROMPTS, concurrency=2, max_tokens=8)
    assert rec["faults"] == 0 and rec["n"] == 6 and rec["per_token_events"]
    body = _Stub.seen[0]
    assert body["max_tokens"] == body["min_tokens"] == 8 and body["ignore_eos"] and body["temperature"] == 0
    assert body["stream"] and body["stream_options"] == {"include_usage": True}
    assert isinstance(body["prompt"], list)                      # token ids, never text
    assert [r["id"] for r in rec["requests"]] == [p["id"] for p in PROMPTS]
    assert all(len(r["events"]) == 8 and r["usage_tokens"] == 8 for r in rec["requests"])
    assert rec["prompts_sha"] == drv.prompts_sha(PROMPTS) and rec["out_tok_s"] > 0
    again = drv.run_pass(stub, "m", PROMPTS, concurrency=1, max_tokens=8)
    assert [r["text_sha"] for r in again["requests"]] == [r["text_sha"] for r in rec["requests"]]


def test_driver_records_a_short_output_and_an_http_error(stub):
    _Stub.mode = "short"
    rec = drv.run_pass(stub, "m", PROMPTS[:2], concurrency=1, max_tokens=8)
    assert rec["faults"] == 2 and "short output: 7 of 8" in rec["requests"][0]["fault"]
    _Stub.mode = "error"
    rec = drv.run_pass(stub, "m", PROMPTS[:1], concurrency=1, max_tokens=8)
    assert rec["faults"] == 1 and rec["requests"][0]["fault"].startswith("http 500")


def test_closed_loop_never_exceeds_the_concurrency(stub):
    drv.run_pass(stub, "m", PROMPTS, concurrency=3, max_tokens=4)
    assert len(_Stub.seen) == 6


def test_pct_is_nearest_rank():
    assert drv.pct([1, 2, 3, 4], 50) == 2 and drv.pct([1, 2, 3, 4], 99) == 4 and drv.pct([], 50) is None


# --------------------------------------------------------------- the reducer ----------------------------------------
REG = {"profile_sha": "f" * 64, "budget_bytes": {"4": 4_000_000_000, "8": 8_000_000_000, "12": 12_000_000_000},
       "sets": {"warm_c1": "w" * 64, "warm_c4": "v" * 64, "shift": "s" * 64}}
TPOT = {"S4": 0.020, "S8": 0.025, "V8": 0.026, "E-hyb": 0.040, "E-hyb-u": 0.060, "E-allvram-g": 0.010,
        "E-allvram-e": 0.030}


def _rec(arm, p, draw, tpot=None, texts=None):
    t = tpot if tpot is not None else TPOT[arm] * (1 + 0.005 * (draw - 2))
    return {"faults": 0, "prompts_sha": REG["sets"][red.PASS_SET[p]], "tpot_p50": t,
            "tpot_p99": t * 1.2, "ttft_p50": 0.1, "ttft_p99": 0.2, "out_tok_s": 1 / t,
            "requests": [{"id": f"r{i}", "text_sha": (texts or {}).get(i, f"{arm}-{i}")} for i in range(4)]}


def _lt(arm, bud, draw):
    lt = {"arm": arm, "budget": bud, "draw": draw, "k_registered": 31, "k_measured": 31, "counters_ok": True,
          "pager_giveups": 0, "nvme_experts": 0, "dram_tier_s_per_token": {"P2": 0.012, "P3": 0.012}}
    if arm in ("E-hyb", "E-hyb-u"):
        lt["vram_tier_bytes"] = 3_900_000_000 if bud == 4 else int(bud) * 990_000_000
        lt["profile"] = ({"kind": "measured", "sha256": REG["profile_sha"]} if arm == "E-hyb"
                         else {"kind": "uniform-assumed", "sha256": None})
    return lt


def _work(tmp_path, edit=None):
    """A complete clean schedule; ``edit(arm, budget, draw, pass, lt, rec)`` may change one entry in place."""
    root = tmp_path / "work"
    for arm, bud, draw, p in red.schedule():
        d = root / arm / str(bud) / f"d{draw}"
        d.mkdir(parents=True, exist_ok=True)
        lt_path = d / "lifetime.json"
        lt = json.loads(lt_path.read_text()) if lt_path.exists() else _lt(arm, bud, draw)
        rec = _rec(arm, p, draw)
        if edit:
            out = edit(arm, bud, draw, p, lt, rec)
            if out == "drop":
                continue
        lt_path.write_text(json.dumps(lt))
        (d / f"{p}.json").write_text(json.dumps(rec))
    return str(root)


def test_a_clean_schedule_reads(tmp_path):
    v = red.reduce([_work(tmp_path)], REG)
    assert v["verdict"] == "READ" and not v["void_lifetimes"] and not v["findings"]
    r = {(x["budget"], x["pass"]): x for x in v["readings"]}
    assert r[(8, "P2")]["ratios"]["S4"]["tpot"] == pytest.approx(0.040 / 0.020)
    a = {(x["budget"], x["pass"]): x for x in v["attribution"]}[(8, "P2")]
    # gap = 0.040 - 0.020 = 0.020: eager 0.020 / 0.020 = 1.0; dram 0.012 / 0.020 = 0.6; placement 0.020 / 0.040 = 0.5
    assert a["state"] == "LOSS"
    assert a["shares"] == pytest.approx({"eager_tax": 1.0, "dram_host": 0.6, "placement": 0.5})
    assert a["branches"] == ["capture-legal hybrid tier", "lean prefill / GPU paging",
                             "a better profile (placement quality)"]
    assert "placement" not in {(x["budget"], x["pass"]): x for x in v["attribution"]}[(4, "P2")]["shares"]


def test_each_pass_is_checked_against_its_own_set(tmp_path):
    """P3 runs the 16-request c4 set: a P3 record carrying the c1 set's sha is the wrong set, and VOIDs that pass."""
    def edit(a, b, d, p, lt, r):
        if (a, b, d, p) == ("S8", 12, 2, "P3"):
            r["prompts_sha"] = REG["sets"]["warm_c1"]
    v = red.reduce([_work(tmp_path, edit)], REG)
    assert v["cells"]["S8|12|P3"]["state"] == "VOID" and v["cells"]["S8|12|P2"]["state"] == "READ"
    assert v["cells"]["S8|8|P3"]["state"] == "READ"


@pytest.mark.parametrize("name, edit, void_key", [
    ("missing pass", lambda a, b, d, p, lt, r: "drop" if (a, b, d, p) == ("S8", 4, 2, "P3") else None, ("S8", 4, 2)),
    ("request fault", lambda a, b, d, p, lt, r: r.update(faults=1) if (a, b, d, p) == ("V8", 12, 1, "P2") else None,
     ("V8", 12, 1)),
    ("wrong prompt set", lambda a, b, d, p, lt, r: r.update(prompts_sha="x") if (a, b, d, p) == ("S4", 8, 3, "P4")
     else None, ("S4", 8, 3)),
    ("K mismatch", lambda a, b, d, p, lt, r: lt.update(k_measured=30) if (a, b, d) == ("S8", 8, 1) else None,
     ("S8", 8, 1)),
    ("counters missing", lambda a, b, d, p, lt, r: lt.update(counters_ok=False) if (a, b, d) == ("S4", 4, 1) else None,
     ("S4", 4, 1)),
    ("pager give-up", lambda a, b, d, p, lt, r: lt.update(pager_giveups=1) if (a, b, d) == ("V8", 4, 3) else None,
     ("V8", 4, 3)),
    ("NVMe tier used", lambda a, b, d, p, lt, r: lt.update(nvme_experts=3) if (a, b, d) == ("E-hyb", 12, 2) else None,
     ("E-hyb", 12, 2)),
    ("VRAM over budget", lambda a, b, d, p, lt, r: lt.update(vram_tier_bytes=4_100_000_000)
     if (a, b, d) == ("E-hyb", 4, 1) else None, ("E-hyb", 4, 1)),
    ("uniform profile on E-hyb", lambda a, b, d, p, lt, r: lt.update(profile={"kind": "uniform-assumed", "sha256": None})
     if (a, b, d) == ("E-hyb", 8, 2) else None, ("E-hyb", 8, 2)),
    ("measured profile on E-hyb-u", lambda a, b, d, p, lt, r: lt.update(profile={"kind": "measured", "sha256": "f" * 64})
     if (a, b, d) == ("E-hyb-u", 8, 1) else None, ("E-hyb-u", 8, 1)),
])
def test_each_registered_fault_voids_its_lifetime(tmp_path, name, edit, void_key):
    v = red.reduce([_work(tmp_path, edit)], REG)
    assert v["verdict"] == "VOID", name
    assert [tuple(x) for x in v["void_lifetimes"]] == [void_key], name


def test_token_invariance_rules_for_triton_and_reports_for_marlin(tmp_path):
    def edit(a, b, d, p, lt, r):
        if (b, d, p) == (8, 1, "P2") and a in ("S8", "S4"):
            r["requests"][0]["text_sha"] = "different"
    v = red.reduce([_work(tmp_path, edit)], REG)
    assert [tuple(x) for x in v["void_lifetimes"]] == [("S8", 8, 1)]
    kinds = {(f["kind"], f["arm"], f["rules"]) for f in v["findings"]}
    assert ("token_invariance", "S4", False) in kinds and ("token_invariance", "S8", True) in kinds


def test_an_e_hyb_p2_p1_difference_is_a_finding_not_a_void(tmp_path):
    """E-hyb pages nothing; its DRAM tier is multithreaded host arithmetic. A P2/P1 text difference is reported."""
    def edit(a, b, d, p, lt, r):
        if (b, d, p) == (8, 2, "P2") and a in ("E-hyb", "E-hyb-u"):
            r["requests"][1]["text_sha"] = "different"
    v = red.reduce([_work(tmp_path, edit)], REG)
    assert v["verdict"] == "READ" and not v["void_lifetimes"]
    kinds = {(f["kind"], f["arm"], f["rules"]) for f in v["findings"]}
    assert kinds == {("token_invariance", "E-hyb", False), ("token_invariance", "E-hyb-u", False)}
    assert {"S8", "V8", "E-allvram-g", "E-allvram-e"} == red.TOKEN_RULED


def test_cross_budget_tokens_are_reported_never_ruled(tmp_path):
    def edit(a, b, d, p, lt, r):
        if (a, b) == ("V8", 12) and p in ("P1", "P2"):
            for q in r["requests"]:
                q["text_sha"] = "v8-12-" + q["id"]
    v = red.reduce([_work(tmp_path, edit)], REG)
    assert v["verdict"] == "READ"
    assert {(f["kind"], f["arm"]) for f in v["findings"]} == {("cross_budget_tokens", "V8")}


def test_a_draw_spread_over_three_percent_is_unstable_not_void(tmp_path):
    def edit(a, b, d, p, lt, r):
        if (a, b, p) == ("S8", 12, "P3") and d == 3:
            r.update(_rec("S8", p, d, tpot=0.025 * 1.05))
    v = red.reduce([_work(tmp_path, edit)], REG)
    assert v["verdict"] == "READ"
    assert v["cells"]["S8|12|P3"]["state"] == "UNSTABLE"
    row = {(x["budget"], x["pass"]): x for x in v["readings"]}[(12, "P3")]
    assert "S8" not in row["ratios"] and "S4" in row["ratios"]


def test_a_draw_spread_of_exactly_three_percent_reads(tmp_path):
    """The rule is max/min <= 1.03: a cell sitting exactly on the boundary is READ, not UNSTABLE."""
    vals = {1: (1.0, 100.0), 2: (1.015, 101.5), 3: (1.03, 103.0)}

    def edit(a, b, d, p, lt, r):
        if (a, b, p) == ("S8", 12, "P2"):
            r.update(tpot_p50=vals[d][0], out_tok_s=vals[d][1])
    v = red.reduce([_work(tmp_path, edit)], REG)
    c = v["cells"]["S8|12|P2"]
    assert c["spread"] == red.STABLE and c["state"] == "READ"


def test_a_duplicated_lifetime_across_boxes_voids(tmp_path):
    a = _work(tmp_path / "a")
    b = tmp_path / "b" / "S4" / "4" / "d1"
    b.mkdir(parents=True)
    for f in os.listdir(os.path.join(a, "S4", "4", "d1")):
        (b / f).write_text(open(os.path.join(a, "S4", "4", "d1", f)).read())
    v = red.reduce([a, str(tmp_path / "b")], REG)
    assert ["S4", 4, 1] in v["void_lifetimes"]


def test_no_loss_names_no_branch(tmp_path):
    def edit(a, b, d, p, lt, r):
        if a == "E-hyb" and b == 12:
            r.update(_rec("E-hyb", p, d, tpot=0.015 * (1 + 0.005 * (d - 2))))
    v = red.reduce([_work(tmp_path, edit)], REG)
    a = {(x["budget"], x["pass"]): x for x in v["attribution"]}[(12, "P2")]
    assert a["state"] == "NO_LOSS" and a["branches"] == []


def test_the_schedule_is_the_registered_one():
    s = red.schedule()
    matched = [x for x in s if x[0] in red.ARMS]
    assert len(matched) == 4 * 3 * (3 + 4 + 3)           # arms x draws x passes over the 4 / 8 / 12 GB budgets
    assert len([x for x in s if x[0] == "E-hyb-u"]) == 3 * 4
    assert len([x for x in s if x[0].startswith("E-allvram")]) == 2 * 3 * 3
