#!/usr/bin/env python3
"""p159_reduce.py -- P159's registered reads (bench/p159/PREREG-p159.md, "What is read"), from p159_box.py's
result.json. Stdlib only.

    python p159_reduce.py --result result.json --out verdict_p159.json
    python p159_reduce.py --self-test
"""
import argparse
import json
import math
import statistics

E_FULL = 128
FAST_S, SLOW_S = 1.5, 3.0
FUSE_FRESH_BYTES = 8053063680                           # 7.5 GiB at 128 experts (p159_box.FUSE_FRESH_BYTES)
N_HUGE_MAX = 64 << 20                                    # an N block's fuse huge-page bytes must stay under this
P_MINFLT_MAX = 400_000                                   # the median of a P process's timed fuse minor faults
NULL_LOG_MAX = math.log(1.5)
HI, LO = 0.8, 0.2
MIN_STATE_BLOCKS, MIN_SLOW_TRIPLES, MIN_B = 12, 6, 3
COUNTER_MULTIPLE = 5.0
B_RATIO = 1.25


def state(fuse_s, experts):
    v = fuse_s * E_FULL / experts
    return "FAST" if v <= FAST_S else "SLOW" if v >= SLOW_S else "MID"


def huge_gain(b):
    """AnonHugePages gained over the fuse (after fuse less before the block), in bytes, or None when unreadable."""
    after = (b.get("smaps_after_fuse") or {}).get("anon_huge_bytes")
    before = ((b.get("pre") or {}).get("smaps") or {}).get("anon_huge_bytes")
    return None if after is None or before is None else after - before


def huge_share(b):
    """REPORTED only: the huge-page share of the anonymous memory the fuse grew. The wiring run (p159-dry-1) showed why
    it cannot be read: the growth is the two stacks only (the temporaries are freed by then), and none at all when the
    allocator reuses freed memory, so a fully huge-paged block can read 0. READ-2 reads the THP fault counters."""
    after, before = b.get("smaps_after_fuse") or {}, (b.get("pre") or {}).get("smaps") or {}
    g, a = huge_gain(b), None
    if after.get("anon_bytes") is not None and before.get("anon_bytes") is not None:
        a = after["anon_bytes"] - before["anon_bytes"]
    return None if g is None or not a or a <= 0 else g / a


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def triples(proc):
    """An M process's 5 triples, each {arm: block}."""
    bl = proc["blocks"]
    return [{b["arm"]: b for b in bl[i:i + 3]} for i in range(0, len(bl) - 2, 3)]


def window(prev, nxt, experts):
    """FAST/SLOW if all 8 edge M and M' blocks agree (the last two triples before, the first two after), else MIXED."""
    if not prev or not nxt or prev.get("kind") != "M" or nxt.get("kind") != "M" or "blocks" not in prev \
            or "blocks" not in nxt:
        return "MIXED"
    edge = triples(prev)[-2:] + triples(nxt)[:2]
    states = {state(t[a]["fuse_s"], experts) for t in edge for a in ("M", "M2")}
    return states.pop() if len(states) == 1 and states <= {"FAST", "SLOW"} else "MIXED"


def engagement(res):
    """Per-arm VOID flags from the closed forms (scaled to the run's experts). Each fails closed: a reading that is
    missing, or a probe with no positive control, VOIDs the arm."""
    ex, procs = res["experts"], [p for p in res["processes"] if "blocks" in p]
    scale = ex / E_FULL
    void = {}
    n = [p for p in procs if p["kind"] == "N"]

    def n_block_bad(b):
        h = huge_gain(b)
        return h is None or h >= N_HUGE_MAX * scale
    void["N"] = bool(n) and any(not p.get("engaged", {}).get("thp_disabled")
                                or any(n_block_bad(b) for b in p["blocks"]) for p in n)
    pp = [p for p in procs if p["kind"] == "P"]
    # the median of each P process's timed blocks (a block that grows the heap once does not void the arm); the max is
    # reported beside it
    void["P"] = bool(pp) and any(not p.get("engaged", {}).get("tunables")
                                 or statistics.median(b["fuse"]["minflt"] for b in p["blocks"]) >= P_MINFLT_MAX * scale
                                 for p in pp)
    mp = [p for p in procs if p["kind"] == "M"]
    rb = [b for p in mp for b in p["blocks"] if b["arm"] == "R"]
    # R maps no slab, and the probe can see a mapping: every M process has an M or M' block that shows the slab mapped
    control = all(any(b.get("slab_mapped_during_read") is True for b in p["blocks"] if b["arm"] in ("M", "M2"))
                  for p in mp)
    void["R"] = bool(rb) and (not control or any(b.get("slab_mapped_during_read") is not False for b in rb))
    return void


def reduce(res):
    ex, procs = res["experts"], res["processes"]
    out = {"bytes_ok": bool(res.get("bytes_ok")), "void": engagement(res)}
    mprocs = [p for p in procs if p.get("kind") == "M" and "blocks" in p]
    mm = [b for p in mprocs for b in p["blocks"] if b["arm"] in ("M", "M2")]
    # the null (M vs M')
    tri = [t for p in mprocs for t in triples(p)]
    flips = [state(t["M"]["fuse_s"], ex) != state(t["M2"]["fuse_s"], ex) for t in tri]
    out["flip_base_rate"] = sum(flips) / len(flips) if flips else None
    agree = [abs(math.log(t["M2"]["fuse_s"] / t["M"]["fuse_s"])) for t in tri
             if state(t["M"]["fuse_s"], ex) == state(t["M2"]["fuse_s"], ex) != "MID"]
    out["null_median_abs_log"] = med(agree)
    out["void_instrument"] = bool(agree) and med(agree) > NULL_LOG_MAX
    # windows
    rows = []
    for i, p in enumerate(procs):
        if p.get("kind") in ("N", "P", "B") and "blocks" in p:
            w = window(procs[i - 1] if i else None, procs[i + 1] if i + 1 < len(procs) else None, ex)
            st = [state(b["fuse_s"], ex) for b in p["blocks"]]
            rows.append({"index": i, "kind": p["kind"], "window": w, "slow": st.count("SLOW"),
                         "fast": st.count("FAST"), "n": len(st)})
    out["windows"] = rows
    slow_seen = any(state(b["fuse_s"], ex) == "SLOW" for b in mm)

    def share(kind, win, key):
        rs = [r for r in rows if r["kind"] == kind and r["window"] == win]
        return (sum(r[key] for r in rs) / sum(r["n"] for r in rs), len(rs)) if rs else (None, 0)
    s_nf, n_nf = share("N", "FAST", "slow") if not out["void"]["N"] else (None, 0)
    f_ps, n_ps = share("P", "SLOW", "fast") if not out["void"]["P"] else (None, 0)
    out["read1_inputs"] = {"s_NF": s_nf, "n_NF": n_nf, "f_PS": f_ps, "n_PS": n_ps}
    if not slow_seen:
        r1 = "NOT_READ_STATE_NOT_OBSERVED"
    elif n_ps and f_ps <= LO:
        r1 = "REFUTED_FAULTS"
    elif n_nf and s_nf <= LO:
        r1 = "REFUTED_THP"
    elif n_nf and n_ps and s_nf >= HI and f_ps >= HI:
        r1 = "IDENTIFIED"
    elif (n_nf and s_nf >= HI and not n_ps) or (n_ps and f_ps >= HI and not n_nf):
        r1 = "PARTIAL"
    elif not n_nf and not n_ps:
        r1 = "NOT_READ_NO_WINDOW"
    else:
        r1 = "UNRESOLVED"
    out["read1"] = r1
    # READ-2, the counters (corroborating)
    slow = [b for b in mm if state(b["fuse_s"], ex) == "SLOW"]
    fast = [b for b in mm if state(b["fuse_s"], ex) == "FAST"]
    if len(slow) < MIN_STATE_BLOCKS or len(fast) < MIN_STATE_BLOCKS:
        r2 = "NOT_READ"
        out["read2_inputs"] = {"slow": len(slow), "fast": len(fast)}
    else:
        mf_s, mf_f = med(b["fuse"]["minflt"] for b in slow), med(b["fuse"]["minflt"] for b in fast)
        ta_s = med(b["fuse"].get("thp_fault_alloc") for b in slow)
        ta_f = med(b["fuse"].get("thp_fault_alloc") for b in fast)
        tf_s = med(b["fuse"].get("thp_fault_fallback") for b in slow)
        tf_f = med(b["fuse"].get("thp_fault_fallback") for b in fast)
        if None in (ta_s, ta_f, tf_s, tf_f):
            r2 = "NOT_READ"                              # a kernel without THP counters (the lab A2000's)
        else:
            ok = (mf_f is not None and mf_s >= COUNTER_MULTIPLE * max(mf_f, 1) and ta_f > ta_s and tf_s > tf_f)
            r2 = "PASS" if ok else "FAIL"
        out["read2_inputs"] = {"minflt_slow": mf_s, "minflt_fast": mf_f,
                               "thp_alloc_slow": ta_s, "thp_alloc_fast": ta_f,
                               "thp_fallback_slow": tf_s, "thp_fallback_fast": tf_f,
                               "huge_share_slow_reported": med(huge_share(b) for b in slow),
                               "huge_share_fast_reported": med(huge_share(b) for b in fast)}
    out["read2"] = r2
    out["call"] = ({"PASS": "IDENTIFIED_CORROBORATED", "FAIL": "CONFLICT", "NOT_READ": "IDENTIFIED_UNCORROBORATED"}[r2]
                   if r1 == "IDENTIFIED" else r1)
    # READ-3, slab residency
    both_slow = [t for t in tri if state(t["M"]["fuse_s"], ex) == state(t["M2"]["fuse_s"], ex) == "SLOW"]
    if out["void"]["R"]:
        r3 = "VOID"
    elif len(both_slow) < MIN_SLOW_TRIPLES:
        r3 = "NOT_READ"
    else:
        fr = sum(state(t["R"]["fuse_s"], ex) == "FAST" for t in both_slow) / len(both_slow)
        r3 = "SLAB_CARRIES" if fr >= HI else "NOT_SLAB" if fr <= LO else "UNRESOLVED"
        out["read3_inputs"] = {"r_fast_share": fr, "triples": len(both_slow)}
    out["read3"] = r3
    # READ-4, headroom at the floor
    bproc = [(i, p) for i, p in enumerate(procs) if p.get("kind") == "B"]
    if not bproc or all("blocks" not in p for _, p in bproc):
        r4 = "NOT_RUN"
    else:
        ratios = []
        for i, p in bproc:
            if "blocks" not in p:
                continue
            br = [procs[j] for j in (i - 1, i + 1) if 0 <= j < len(procs) and procs[j].get("kind") == "M"
                  and "blocks" in procs[j]]
            if len(br) != 2:
                continue
            m_off = statistics.mean(med(b["offload_s"] for b in q["blocks"] if b["arm"] in ("M", "M2")) for q in br)
            ratios.append(med(b["offload_s"] for b in p["blocks"]) / m_off)
        if len(ratios) < MIN_B:
            r4 = "NOT_READ"
        else:
            r4 = "FLOOR_SLOWS_OFFLOAD" if min(ratios) > B_RATIO else "FLOOR_HOLDS" if max(ratios) <= B_RATIO \
                else "UNRESOLVED"
        out["read4_inputs"] = {"ratios": ratios}
    out["read4"] = r4
    # reported, never read
    arms = {}
    for p in procs:
        for b in p.get("blocks", []):
            arms.setdefault(b["arm"] if p["kind"] != "B" else "B", []).append(b)
    out["per_arm"] = {a: {k: med(b[k] for b in bs) for k in ("read_s", "fuse_s", "h2d_s", "quant_s", "offload_s",
                                                              "block_s", "residual_s")} | {"n": len(bs)}
                      for a, bs in arms.items()}
    out["slow_window_observed"] = slow_seen
    out["verdict"] = "VOID_BYTES" if not out["bytes_ok"] else "VOID_INSTRUMENT" if out["void_instrument"] \
        else out["call"]
    return out


# ---------------------------------------------------------------- self-test

def _blk(arm, fuse, minflt=None, huge=None, fallback=None, offload=1.0, mapped=None, ex=128):
    fresh = FUSE_FRESH_BYTES * ex // E_FULL
    slow = fuse * E_FULL / ex >= SLOW_S
    minflt = minflt if minflt is not None else (2_040_000 if slow else 80_000)
    huge = huge if huge is not None else (0.0 if slow else 0.95)
    return {"arm": arm, "fuse_s": fuse, "offload_s": offload, "read_s": 0.01, "h2d_s": 0.3, "quant_s": 0.06,
            "block_s": fuse + 2, "residual_s": 0.9,
            "fuse": {"minflt": minflt, "majflt": 0, "thp_fault_fallback": fallback if fallback is not None
                     else (3000 if slow else 0), "thp_fault_alloc": 0 if slow else 3840},
            "pre": {"smaps": {"anon_huge_bytes": 0, "anon_bytes": 0}},
            "smaps_after_fuse": {"anon_huge_bytes": int(huge * fresh), "anon_bytes": fresh},
            "slab_mapped_during_read": (arm != "R") if mapped is None else mapped}


def _mproc(fuses, r_fuse=None, offload=1.0):
    """An M process: 15 blocks in the registered rotation, M/M' at `fuses` (per triple), R at r_fuse."""
    rots = (("M", "R", "M2"), ("R", "M2", "M"), ("M2", "M", "R"))
    bl = []
    for k in range(5):
        for arm in rots[k % 3]:
            f = fuses[k] if arm != "R" else (r_fuse[k] if r_fuse else fuses[k])
            bl.append(_blk(arm, f, offload=offload))
    return {"kind": "M", "blocks": bl, "engaged": {}}


def _xproc(kind, fuse, offload=1.0, engaged=None, **kw):
    arm = {"N": "N", "P": "P", "B": "M"}[kind]
    e = engaged if engaged is not None else {"N": {"thp_disabled": True}, "P": {"tunables": True}, "B": {}}[kind]
    return {"kind": kind, "engaged": e, "blocks": [_blk(arm, fuse, offload=offload, **kw) for _ in range(15)]}


def _run(slots, m_states, n=7.0, p=0.4, b_off=1.0, m_off=1.0, r=None, **kw):
    """25 processes: M_i states from m_states (13 entries, 'F'/'S'), X/Y/Z from slots, N/P/B fuse values."""
    f = {"F": 0.4, "S": 7.2}
    procs, mi = [], 0
    for c in slots:
        for x in c:
            procs.append(_mproc([f[m_states[mi]]] * 5, r_fuse=[r] * 5 if r else None, offload=m_off))
            mi += 1
            if x == "N":
                procs.append(_xproc("N", n if not callable(n) else n(len(procs)), **kw.get("N", {})))
            elif x == "P":
                procs.append(_xproc("P", p if not callable(p) else p(len(procs)), **kw.get("P", {})))
            else:
                procs.append(_xproc("B", f[m_states[mi - 1]], offload=b_off) if b_off is not None
                             else {"kind": "B", "not_run": "refused"})
    procs.append(_mproc([f[m_states[mi]]] * 5, offload=m_off))
    return {"experts": 128, "bytes_ok": True, "processes": procs}


SLOTS = (("N", "P", "B"), ("P", "B", "N"), ("B", "N", "P"), ("N", "P", "B"))


def self_test():
    cases = 0

    def check(res, **want):
        nonlocal cases
        got = reduce(res)
        for k, v in want.items():
            assert got[k] == v, (k, got[k], v, {x: got[x] for x in ("read1_inputs", "read2", "windows")})
        cases += 1
        return got
    # half the M processes FAST, half SLOW, so N and P each see both window kinds
    st = "FFFFFFSSSSSSS"
    check(_run(SLOTS, st), read1="IDENTIFIED", read2="PASS", call="IDENTIFIED_CORROBORATED", read3="NOT_SLAB",
          read4="FLOOR_HOLDS", verdict="IDENTIFIED_CORROBORATED")
    check(_run(SLOTS, st, p=7.0, P={"minflt": 80_000}), read1="REFUTED_FAULTS")   # pre-faulted (engaged), still slow
    check(_run(SLOTS, st, n=0.4, N={"huge": 0.0}), read1="REFUTED_THP")          # no huge pages (engaged), still fast
    check(_run(SLOTS, "F" * 13), read1="NOT_READ_STATE_NOT_OBSERVED", read3="NOT_READ")
    check(_run(SLOTS, st, r=0.4), read3="SLAB_CARRIES")
    check(_run(SLOTS, st, b_off=1.6), read4="FLOOR_SLOWS_OFFLOAD")
    check(_run(SLOTS, st, b_off=None), read4="NOT_RUN")
    # counters disagree with an identified mechanism: CONFLICT
    res = _run(SLOTS, st)
    for p in res["processes"]:
        for b in p.get("blocks", []):
            b["fuse"]["minflt"] = 80_000
    check(res, read1="IDENTIFIED", read2="FAIL", call="CONFLICT")
    # N not engaged (prctl failed): N's side is NOT_READ, so P alone is PARTIAL
    check(_run(SLOTS, st, N={"engaged": {"thp_disabled": False}}), read1="PARTIAL")
    # P not engaged (faults in timed blocks): P's side NOT_READ
    check(_run(SLOTS, st, P={"minflt": 2_000_000}), read1="PARTIAL")
    # R mapped the slab: READ-3 VOID
    res = _run(SLOTS, st)
    for p in res["processes"]:
        for b in p.get("blocks", []):
            if b["arm"] == "R":
                b["slab_mapped_during_read"] = True
    check(res, read3="VOID")
    # the null: M' wildly off M within a state -> VOID_INSTRUMENT
    res = _run(SLOTS, st)
    for p in res["processes"]:
        for b in p.get("blocks", []):
            if b["arm"] == "M2":
                b["fuse_s"] *= 1.9 if b["fuse_s"] > 3 else 1.0          # still SLOW, but 1.9x its M
    got = reduce(res)
    assert got["void_instrument"] and got["verdict"] == "VOID_INSTRUMENT", got["null_median_abs_log"]
    cases += 1
    # bytes
    check(_run(SLOTS, st) | {"bytes_ok": False}, verdict="VOID_BYTES")
    # mixed windows only: no readable window
    alt = "FSFSFSFSFSFSF"
    check(_run(SLOTS, alt), read1="NOT_READ_NO_WINDOW")
    # the bands' edges (plan indices: N at 1 is the one N in a FAST window, P at 17 and 21 the P's in SLOW windows)
    def mix(res, idx, n_slow, **kw):
        a = res["processes"][idx]["blocks"][0]["arm"]
        res["processes"][idx]["blocks"] = [_blk(a, 7.2 if i < n_slow else 0.4, **kw) for i in range(15)]
        return res
    check(mix(_run(SLOTS, st), 1, 11, huge=0.0), read1="UNRESOLVED")            # s_NF 0.733
    check(mix(_run(SLOTS, st), 1, 12, huge=0.0), read1="IDENTIFIED")            # s_NF 0.8, the band's edge
    check(mix(_run(SLOTS, st), 1, 3, huge=0.0), read1="REFUTED_THP")            # s_NF 0.2, the other edge
    check(mix(_run(SLOTS, st), 1, 4, huge=0.0), read1="UNRESOLVED")             # s_NF 0.267
    res = mix(mix(_run(SLOTS, st), 17, 11, minflt=80_000), 21, 11, minflt=80_000)   # f_PS 8/30 = 0.267
    check(res, read1="UNRESOLVED")
    # both refutations hold: REFUTED_FAULTS is read first
    check(_run(SLOTS, st, n=0.4, p=7.0, N={"huge": 0.0}, P={"minflt": 80_000}), read1="REFUTED_FAULTS")
    # READ-2 fails on each clause alone (the mechanism IDENTIFIED, so the call is CONFLICT)
    for clause in ("alloc", "fallback", "multiple"):
        res = _run(SLOTS, st)
        for q in res["processes"]:
            for b in q.get("blocks", []) if q["kind"] == "M" else []:
                if b["arm"] == "R":
                    continue
                slow = b["fuse_s"] >= SLOW_S
                if clause == "alloc" and not slow:
                    b["fuse"]["thp_fault_alloc"] = 0
                if clause == "fallback" and slow:
                    b["fuse"]["thp_fault_fallback"] = 0
                if clause == "multiple" and slow:
                    b["fuse"]["minflt"] = 4 * 80_000                       # 4x, under the 5x multiple
        check(res, read1="IDENTIFIED", read2="FAIL", call="CONFLICT")
    # a kernel without THP counters (the lab A2000's): READ-2 is NOT_READ, never a FAIL
    res = _run(SLOTS, st)
    for q in res["processes"]:
        for b in q.get("blocks", []):
            b["fuse"].pop("thp_fault_alloc", None)
            b["fuse"].pop("thp_fault_fallback", None)
    check(res, read1="IDENTIFIED", read2="NOT_READ", call="IDENTIFIED_UNCORROBORATED")
    # one under each minimum: 10 FAST M/M' blocks; 5 both-SLOW triples; 2 B processes run
    check(_run(SLOTS, "F" + "S" * 12), read2="NOT_READ")
    check(_run(SLOTS, "S" + "F" * 12), read3="NOT_READ")
    res = _run(SLOTS, st)
    for i in (5, 9):
        res["processes"][i] = {"kind": "B", "not_run": "refused"}
    check(res, read4="NOT_READ")
    # fail closed: an unreadable huge-page reading VOIDs N; a probe that never sees the slab mapped VOIDs R
    res = _run(SLOTS, st)
    for b in res["processes"][1]["blocks"]:
        b["smaps_after_fuse"]["anon_huge_bytes"] = None
    got = check(res, read1="PARTIAL")
    assert got["void"]["N"], got["void"]
    res = _run(SLOTS, st)
    for q in res["processes"]:
        for b in q.get("blocks", []):
            b["slab_mapped_during_read"] = False
    check(res, read3="VOID")

    print(f"p159_reduce self-test OK ({cases} cases)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--result")
    p.add_argument("--out")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    out = reduce(json.load(open(a.result)))
    s = json.dumps(out, indent=1, sort_keys=True)
    if a.out:
        open(a.out, "w").write(s + "\n")
    print(f"P159 {out['verdict']} " + json.dumps({k: out[k] for k in ("read1", "read2", "read3", "read4",
                                                                       "flip_base_rate", "slow_window_observed")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
