#!/usr/bin/env python3
"""PGX's reading schedule, fit and split (PREREG-pgx.md, Amendment 3). The runner, the driver and the kit test read it, so the
order, the fit and the plan are written once.

- **The lifetimes of one draw**, in run order: the four matched arms rotated (draw d starts at arm d of S4, S8, V8, E-hyb),
  budgets ascending within an arm (4, 8, 12 GB), each S arm's lifetime preceded by its untimed graphs-off instrument lifetime
  at the same budget; then the controls, E-hyb-u (8 GB), E-allvram-g and E-allvram-e.
- **Passes per lifetime:** P1, P2 (c1, warm_c1), P3 (c4, warm_c4), and P4 (c1, shift) at 8 GB only; E-hyb-u runs all four;
  the two all-VRAM controls P1-P3. An instrument lifetime runs its timed lifetime's passes, graphs off, with the routing dump.
- **The fit** comes from pgx-shake-r1's receipt (``SHAKE``, below), not from the registration's INFERRED estimates. Each
  constant names where it was read. What the shakedown did not measure is modelled and stated (``lifetime_s``).
- **The split rule (P238's form):** the fewest K of 1, 2 or 3 boxes whose largest box fits the box cap; boxes are whole draws
  (K = 2: draws {1, 2} and {3}; K = 3: one draw a box). Box i of K runs as ``pgx-4090-1-<K>-<i>`` (K = 1: ``pgx-4090-1``);
  a box that does not start is retried as ``pgx-4090-r<n>-<K>-<i>``, n = 1..3 across all the boxes.

    python3 pgx_schedule.py --plan        # the fit per box and the K it picks
    python3 pgx_schedule.py --box K-i     # that box's lifetimes, in run order
    python3 pgx_schedule.py --need K-i    # that box's rc 41 need (its modelled work x 1.25), in s
    python3 pgx_schedule.py --run-id RID  # "K i" for a registered reading id at the plan's K, else exit 2
"""
from __future__ import annotations

import sys

ARMS = ("S4", "S8", "V8", "E-hyb")
BASELINES = ("S4", "S8", "V8")
SGL = ("S4", "S8")
BUDGETS = (4, 8, 12)
DRAWS = (1, 2, 3)
CONTROLS = ("E-hyb-u", "E-allvram-g", "E-allvram-e")
SPLITS = (1, 2, 3)
RUN_IDS = ("pgx-4090-1",)
RETRIES = (1, 2, 3)        # three retry ids in all across the boxes: pgx-4090-r<n>-<K>-<i> names the box it replaces

# ---- the box cap (PREREG "Run ids, order and the stop": worst <= USD 15, guard <= 18 h under the per-run override)
POLICY_MAX_WALLCLOCK_H = 6
RUN_OVERRIDE_MAX_H = 18
RUN_WORST_USD, USD_PER_H, DOWNLOAD_GB, USD_PER_GB = 15.0, 0.85, 120, 0.02
DOWNLOAD_USD = DOWNLOAD_GB * USD_PER_GB                     # 2.40
HEADROOM_PCT, LAUNCH_SLACK_S = 125, 1500                    # x 1.25, then Amendment 2 rule 5's launch slack

# ---- the shakedown's receipt (bench/pgx/receipts/pgx-shake-r1/: summary.txt, outer.log, shake/*/8/d1/*.json, instr/*/I1.json,
# place_preview.json), all on one box
SHAKE = {
    "receipt": "bench/pgx/receipts/pgx-shake-r1",
    # setup: STARTED 05:42:52Z -> the first lifetime 05:58:42Z = 950 s; the fetch inside it 05:46:46Z -> 05:56:30Z = 584 s
    "setup_s": 950, "fetch_s": 584,
    # c1 pass walls at 8 GB over warm_c1 (8 requests x 256 tokens): P1 and P2, and the median TPOT
    "p1_s": {"S4": 15.48, "S8": 48.09, "V8": 44.33, "E-hyb": 412.04},
    "p2_s": {"S4": 15.48, "S8": 48.11, "V8": 44.05, "E-hyb": 397.03},
    "tpot_s": {"S4": 0.006061, "S8": 0.020114, "V8": 0.018356, "E-hyb": 0.190672},
    # lifetime walls (start to stop); load = wall minus the two passes
    "wall_s": {"S4": 88.0, "S8": 170.0, "V8": 211.0, "E-hyb": 836.0},
    # the instrument lifetimes (graphs off, routing dump, I1 = the P1 prompts once): I1's wall and the lifetime's wall
    "instr_i1_s": {"S4": 85.67, "S8": 143.03}, "instr_wall_s": {"S4": 496.0, "S8": 220.0},
    # calibrate's link (h2d single-copy) and the placement preview's VRAM mass share on the box's calib.json
    "link_bps": 24.07e9,
    "vram_share": {("E-hyb", 4): 0.5691, ("E-hyb", 8): 0.8000, ("E-hyb", 12): 0.9014, ("E-hyb-u", 8): 0.4931},
    "peak_rss_gib_max": 49.96,                              # S8
}
FETCH_FLOOR_S = 2750                                        # 110 GB at the runner's 40 MB/s refusal floor (rc 19)
BYTES = {"S4": 2_470_912, "S8": 4_719_744, "V8": 4_719_744}  # bytes per expert (bytes_per_expert.json)
SLOTS_PER_TOKEN = 48 * 8                                    # MoE layers x top-k
# the registered LRU hit predictions for the warm set (code and dialogue, PREREG "Predictions"), mean of the two
HIT = {"S4": {4: (0.752 + 0.718) / 2, 8: (0.933 + 0.983) / 2, 12: (0.988 + 0.997) / 2},
       "S8": {4: (0.578 + 0.503) / 2, 8: (0.769 + 0.747) / 2, 12: (0.878 + 0.929) / 2}}
HIT["V8"] = HIT["S8"]
HIT_MARGIN = 0.05                                           # the predictions' own band, taken against the fit off 8 GB
EAGER_GPU_S = 0.00805                                       # P66's eager e4b decode, s/token (the registration's Q4): INFERRED
CONTROL_LIFETIME_S = 300                                    # E-allvram-g / -e: never run on the shakedown: INFERRED
JIT_ONCE_S = 353                                            # S4's first SGLang start on a box (Marlin JIT): 496 - 86 - 57


def passes(arm, budget):
    if arm == "E-hyb-u":
        return ("P1", "P2", "P3", "P4")
    if arm in ("E-allvram-g", "E-allvram-e"):
        return ("P1", "P2", "P3")
    return ("P1", "P2", "P3", "P4") if budget == 8 else ("P1", "P2", "P3")


def draw_lifetimes(d):
    """[(kind, arm, budget)] for draw d, in run order. kind: "instr" (untimed, graphs off) or "timed"."""
    i = ARMS.index(ARMS[(d - 1) % len(ARMS)])
    out = []
    for arm in ARMS[i:] + ARMS[:i]:
        for b in BUDGETS:
            if arm in SGL:
                out.append(("instr", arm, b))
            out.append(("timed", arm, b))
    out += [("timed", "E-hyb-u", 8), ("timed", "E-allvram-g", "all"), ("timed", "E-allvram-e", "all")]
    return out


def schedule():
    """Every lifetime of the reading: [{index, draw, kind, arm, budget, passes}] in run order."""
    rows = []
    for d in DRAWS:
        for kind, arm, b in draw_lifetimes(d):
            rows.append({"draw": d, "kind": kind, "arm": arm, "budget": b, "passes": passes(arm, b)})
    for i, r in enumerate(rows, 1):
        r["index"] = i
    return rows


def _tpot_ratio(arm, b):
    """TPOT at budget b over TPOT at 8 GB (the shakedown's budget).
    S/V arms: TPOT = fixed + misses x bytes / link. The fixed part is the 8 GB TPOT minus its predicted page-in time (this
    fits all three: S8's predicted page-ins are 18.2 of its 20.1 ms); at 4 and 12 GB the hit rate is the prediction minus
    the band (more page-ins, conservative).
    E-hyb arms: TPOT = the eager GPU decode + the CPU tier's routed mass x a per-mass cost fit at 8 GB (all of E-hyb's
    measured excess over the eager decode is charged to the CPU tier, so the cost scales with the CPU mass)."""
    if arm in BASELINES:
        def t(hit):
            return SLOTS_PER_TOKEN * (1 - hit) * BYTES[arm] / SHAKE["link_bps"]
        fixed = SHAKE["tpot_s"][arm] - t(HIT[arm][8])
        assert fixed > 0, arm
        return (fixed + t(HIT[arm][b] - (HIT_MARGIN if b != 8 else 0))) / SHAKE["tpot_s"][arm]
    t8 = SHAKE["tpot_s"]["E-hyb"]
    per_mass = (t8 - EAGER_GPU_S) / (1 - SHAKE["vram_share"][("E-hyb", 8)])
    return (EAGER_GPU_S + (1 - SHAKE["vram_share"][(arm, b)]) * per_mass) / t8


def lifetime_s(kind, arm, b):
    """A lifetime's modelled wall: the load (the shakedown lifetime minus its passes) + each pass. P1 and P2 are the
    shakedown's c1 passes scaled by the TPOT ratio; P3 (c4, 16 requests) is twice the c1 pass (serial-equivalent: c4 buys
    no throughput); P4 is the P2 pass. An instrument lifetime runs the same passes slowed by the shakedown's measured
    graphs-off ratio (I1 / P1)."""
    if arm in ("E-allvram-g", "E-allvram-e"):
        return CONTROL_LIFETIME_S
    src = "E-hyb" if arm == "E-hyb-u" else arm
    load = SHAKE["wall_s"][src] - SHAKE["p1_s"][src] - SHAKE["p2_s"][src]
    r = _tpot_ratio(arm, b)
    p1, p2 = SHAKE["p1_s"][src] * r, SHAKE["p2_s"][src] * r
    want = {"P1": p1, "P2": p2, "P3": 2 * p2, "P4": p2}
    run = sum(want[p] for p in passes(arm, b))
    if kind == "instr":
        run *= SHAKE["instr_i1_s"][src] / SHAKE["p1_s"][src]
    return load + run


def setup_s():
    """STARTED to the first lifetime, measured, with the fetch re-fit at the refusal floor (the margin)."""
    return SHAKE["setup_s"] - SHAKE["fetch_s"] + FETCH_FLOOR_S


def draw_s(d, instr=True):
    return sum(lifetime_s(k, a, b) for k, a, b in draw_lifetimes(d) if instr or k == "timed")


def box_work_s(draws, instr=True):
    """The box's modelled work: setup + S4's one JIT start + its draws."""
    return setup_s() + (JIT_ONCE_S if instr else 0) + sum(draw_s(d, instr) for d in draws)


def box_need_s(draws, instr=True):
    """What the box needs left at its rc 41 check: the work x 1.25 (integer s; the runner compares its deadline to this)."""
    return int(box_work_s(draws, instr) * HEADROOM_PCT / 100)


def box_fit_s(draws, instr=True):
    """The box's guard: its need + the launch slack (Amendment 2 rule 5), integer s."""
    return box_need_s(draws, instr) + LAUNCH_SLACK_S


def lifetime_need_s(kind, arm, b):
    """What one lifetime needs left before it starts (the runner's can_run): its modelled wall x 1.25."""
    return int(lifetime_s(kind, arm, b) * HEADROOM_PCT / 100)


def blocks(k):
    if k not in SPLITS:
        raise ValueError(f"K {k} is not one of {SPLITS}")
    return {1: [[1, 2, 3]], 2: [[1, 2], [3]], 3: [[1], [2], [3]]}[k]


def worst_usd(guard_s):
    return guard_s / 3600 * USD_PER_H + DOWNLOAD_USD


def box_cap_s():
    return int(min(RUN_OVERRIDE_MAX_H, (RUN_WORST_USD - DOWNLOAD_USD) / USD_PER_H) * 3600)


def fewest_k(instr=True):
    for k in SPLITS:
        if max(box_fit_s(b, instr) for b in blocks(k)) <= box_cap_s():
            return k
    return None


# THE PLAN, fixed by Amendment 3 from the committed receipt's fit: the kit carries K as a constant and nothing recomputes it
# at launch. Only a later amendment (a new receipt-based fit) moves it; K = 3 is the stated fallback.
PLAN_K = 2


def box_ids(k):
    return [f"{RUN_IDS[0]}-{k}-{i}" for i in range(1, k + 1)] if k > 1 else [RUN_IDS[0]]


def retry_id(n, k, i):
    """The n-th retry id of the lane (n = 1..3, across every box), replacing box i of K."""
    if n not in RETRIES or k not in SPLITS or not 1 <= i <= k:
        raise ValueError((n, k, i))
    return f"pgx-4090-r{n}" if k == 1 else f"pgx-4090-r{n}-{k}-{i}"


def parse_run_id(rid):
    """A reading run id -> (K, i, retry n or None); ValueError for anything unregistered."""
    import re
    m = re.fullmatch(r"pgx-4090-(?:1|r([1-3]))(?:-([23])-([1-3]))?", rid)
    if not m:
        raise ValueError(f"not a registered reading id: {rid}")
    n = int(m.group(1)) if m.group(1) else None
    k, i = (int(m.group(2)), int(m.group(3))) if m.group(2) else (1, 1)
    if i > k:
        raise ValueError(f"box {i} of {k}: {rid}")
    return k, i, n


def plan(instr=True):
    k = fewest_k(instr)
    rows = []
    for kk in SPLITS:
        boxes = [{"draws": b, "guard_s": box_fit_s(b, instr)} for b in blocks(kk)]
        for bx in boxes:
            bx["guard_h"] = round(bx["guard_s"] / 3600, 2)
            bx["worst_usd"] = round(worst_usd(bx["guard_s"]), 2)
            bx["override"] = bx["guard_s"] > POLICY_MAX_WALLCLOCK_H * 3600
            bx["fits"] = bx["guard_s"] <= box_cap_s()
        rows.append({"k": kk, "boxes": boxes, "total_usd": round(sum(b["worst_usd"] for b in boxes), 2),
                     "fits": all(b["fits"] for b in boxes)})
    return {"k": k, "box_ids": box_ids(k) if k else [], "cap_s": box_cap_s(), "setup_s": setup_s(),
            "draw_s": {d: round(draw_s(d, instr)) for d in DRAWS}, "splits": rows}


def box_rows(spec):
    """"all" or "K-i" -> that box's lifetimes in run order."""
    if spec == "all":
        return schedule()
    k, _, i = spec.partition("-")
    if not (k.isdigit() and i.isdigit()) or int(k) not in SPLITS[1:] or not 1 <= int(i) <= int(k):
        raise ValueError(f"box {spec!r}: 'all', or K-i with K in {SPLITS[1:]} and 1 <= i <= K")
    want = set(blocks(int(k))[int(i) - 1])
    return [r for r in schedule() if r["draw"] in want]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--plan" in argv:
        p = plan()
        print(f"setup {p['setup_s']} s; per draw {p['draw_s']}; box cap {p['cap_s']} s")
        for s in p["splits"]:
            print(f"K={s['k']}: " + "; ".join(f"draws {b['draws']} {b['guard_h']} h USD {b['worst_usd']}"
                                              f"{' (override)' if b['override'] else ''}{'' if b['fits'] else ' OVER THE CAP'}"
                                              for b in s["boxes"]) + f"; total USD {s['total_usd']}")
        print(f"PLAN K={p['k']} {' '.join(p['box_ids'])}" if p["k"] else "PLAN NO_FIT")
        return 0
    if "--need" in argv:                             # --need K-i -> the box's rc 41 need, in s
        spec = argv[argv.index("--need") + 1]
        print(box_need_s(sorted({r["draw"] for r in box_rows(spec)})))
        return 0
    if "--lifetime-need" in argv:                    # --lifetime-need KIND ARM BUDGET -> s
        kind, arm, b = argv[argv.index("--lifetime-need") + 1:][:3]
        print(lifetime_need_s(kind, arm, b if b == "all" else int(b)))
        return 0
    if "--run-id" in argv:                           # --run-id RID -> "K i" for a registered reading id at PLAN_K, else exit 2
        try:
            k, i, _n = parse_run_id(argv[argv.index("--run-id") + 1])
        except ValueError as exc:
            print(exc)
            return 2
        if k != PLAN_K:
            print(f"K {k} is not the registered plan's K = {PLAN_K}")
            return 2
        print(k, i)
        return 0
    rows = box_rows(argv[argv.index("--box") + 1]) if "--box" in argv else schedule()
    for r in rows:
        print(r["index"], r["draw"], r["kind"], r["arm"], r["budget"], ",".join(r["passes"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
