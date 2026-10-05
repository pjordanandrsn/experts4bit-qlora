#!/usr/bin/env python3
"""Lane RD1's table and registered bar (bench/moegen/rd1/RD1-PREREG.md) from rd_probe.py's JSON.

usage: rd_table.py <rd.json>  |  rd_table.py --gate-only <rd.json>  |  rd_table.py --strip-timing <in.json> <out.json>

A row is one MoE layer's frozen-expert GEMM work in one training step: the sum of its four calls (the up or gate_up and the
down projection, forward and dgrad), per arm. Peak is the largest per-call transient of the arm's four calls.

The correctness gate (registered with the bar): an arm passes a call when its relative error against the fp32 reference
(rd_probe.py's ``rel_err32``: dequant_ref in fp32 times the fp32 activations) is at most GATE_X times dense's on the same
call; it passes a cell when all four of its calls pass. A cell whose arm fails, or whose error fields are missing (unread),
never counts toward a bar, and every failure is listed.

The registered bar, read on the SKEWED draw only (uniform draws flatter a route that loses at low rows per expert):
  DECODED  decoded_cap's event time <= 0.85 x min(v1, v3, dense) at BOTH seq 512 and 2048, for at least 3 of the 7
           many-group families (olmoe, lfm2, ernie, graniteh, qwen3, nemotron, qwen36). decoded_cap's decode transient is
           <= the cap by construction; its measured peak is reported.
  V3       v3's event time <= 0.85 x v1 at both seqs, for at least 3 of the 7.
Decision (as registered, read only on an RTX 5090 receipt): DECODED -> an opt-in decoded route in grouped-nf4-gemm, then a TC1 full-step A/B. Else V3 -> a v3
training default is its own TC1 A/B. Else neither: recorded, no route.

A reading that decides also scores the registered predictions (P0, P1, P3-P5) on their own text -- see predictions() --
and prints the rows-per-expert observation beside them, labelled as not registered.
"""
import json
import sys

MANY = ("olmoe", "lfm2", "ernie", "graniteh", "qwen3", "nemotron", "qwen36")
ARMS = ("v1", "v3", "dense", "decoded", "decoded_cap")
GATE_X = 2.0


def gate_call(m, dense):
    """True / False against the fp32 reference, or None when either error field is missing (unread)."""
    if not m or not dense or "rel_err32" not in m or "rel_err32" not in dense:
        return None
    return m["rel_err32"] <= GATE_X * dense["rel_err32"]


def layer(c):
    tot = {a: {"device_ms": 0.0, "event_ms": 0.0, "peak_mib": 0.0, "rel_err": 0.0, "chunks": 0} for a in ARMS}
    missing = set()
    extra = {"chunks": 0, "ms": 0.0}            # decoded_cap minus decoded, over the calls the cap split
    gate = {a: True for a in ARMS}
    fails = []
    for call, res in c["proj"].items():
        for mode, r in res.items():
            for a in ARMS:
                m = r.get(a)
                if not m:
                    missing.add(a)
                    continue
                t = tot[a]
                t["device_ms"] += m.get("device_ms", 0.0)
                t["event_ms"] += m.get("event_ms", 0.0)
                t["peak_mib"] = max(t["peak_mib"], m["peak_mib"])
                t["rel_err"] = max(t["rel_err"], m.get("rel_err", 0.0))
                t["chunks"] = max(t["chunks"], m.get("chunks", 1))
            for a in ARMS:
                g = gate_call(r.get(a), r.get("dense"))
                if g is False:
                    fails.append(f"{a} {c['routing']}/{c['fam']}/{c['seq']} {call}/{mode}: rel_err32 {r[a]['rel_err32']} > "
                                 f"{GATE_X} x dense's {r['dense']['rel_err32']}")
                if g is not True:
                    gate[a] = False if (g is False or gate[a] is False) else None
            if r.get("decoded_cap") and r.get("decoded") and r["decoded_cap"]["chunks"] > 1:
                extra["chunks"] += r["decoded_cap"]["chunks"] - 1
                extra["ms"] += r["decoded_cap"].get("event_ms", 0.0) - r["decoded"].get("event_ms", 0.0)
    return tot, missing, extra, gate, fails


def post_anchor_ok(rec):
    """Amendment 3: a decision needs the post-probe anchor to have passed (rc 0) -- the box held its class through the probe.
    Host load is informational (amendment 1's gate did not predict the anchor across hosts)."""
    return (rec.get("anchor_post") or {}).get("rc") == 0


def main(path):
    rec = json.load(open(path))
    if "5090" not in rec["gpu"]:
        # Not the registered card: no timing column, no bar, no decision -- only the correctness gate (the QNAP A2000 is a
        # correctness testbed; a timing-derived bar printed from it, even labelled "no decision", is speed evidence).
        print(f"NOT THE REGISTERED CARD ({rec['gpu']}): correctness only -- the gate below, no timing, no bar\n")
        gate_only(path)
        return
    print(f"{rec['gpu']}, torch {rec['torch']}, triton {rec['triton']}; decode cap {rec['cap_mib']} MiB; {rec['reps']} reps")
    print(f"host load1 over the probe (informational): {rec.get('host_load1_probe', 'not recorded')}")
    print(f"post-probe anchor: {rec.get('anchor_post', 'not recorded')}\n")
    print("| routing | family | seq | groups (<16 rows) | v1 ev ms | v3 / v1 | dense / v1 | decoded / v1 | decoded_cap / v1 "
          "| decoded_cap / best(v1,v3,dense) | ev / dev (decoded_cap) | peak MiB v1 / v3 / dense / decoded / cap (chunks) | µs per extra chunk | gate cap / v3 / v1 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = {}
    info = {}
    all_fails = []
    for c in rec["cells"]:
        t, missing, extra, gate, fails = layer(c)
        all_fails.extend(fails)
        per_chunk = f"{1000 * extra['ms'] / extra['chunks']:+.0f}" if extra["chunks"] else "—"
        ev = {a: t[a]["event_ms"] for a in ARMS}
        # min over the arms a decoded route must beat: v1 and dense always; v3 only where it passes the gate
        pool = [ev[a] for a in ("v1", "dense") if a not in missing] + ([ev["v3"]] if gate["v3"] is True else [])
        best = min(pool) if pool else float("nan")
        q = ev["decoded_cap"] / best if "decoded_cap" not in missing else float("nan")
        v3 = ev["v3"] / ev["v1"] if "v3" not in missing else float("nan")
        inf = float("inf")
        rows[(c["routing"], c["fam"], c["seq"])] = (q if gate["decoded_cap"] is True else inf,
                                                    v3 if gate["v3"] is True else inf)
        gl = lambda a: {True: "ok", False: "FAIL", None: "unread"}[gate[a]]                       # noqa: E731
        evdev = t["decoded_cap"]["event_ms"] / t["decoded_cap"]["device_ms"] if t["decoded_cap"]["device_ms"] else float("nan")
        info[(c["routing"], c["fam"], c["seq"])] = {"ev": ev, "evdev": evdev, "gate": gate, "missing": missing, "cell": c}
        r = lambda a: f"{ev[a] / ev['v1']:.2f}" if a not in missing else "—"                      # noqa: E731
        print(f"| {c['routing']} | {c['fam']} | {c['seq']} | {c['groups']} ({c['groups_under_16_rows']}) | {ev['v1']:.2f} | "
              f"{r('v3')} | {r('dense')} | {r('decoded')} | {r('decoded_cap')} | {q:.2f} | {evdev:.2f} | "
              f"{t['v1']['peak_mib']:.0f} / {t['v3']['peak_mib']:.0f} / {t['dense']['peak_mib']:.0f} / "
              f"{t['decoded']['peak_mib']:.0f} / {t['decoded_cap']['peak_mib']:.0f} ({t['decoded_cap']['chunks']}) | {per_chunk} | {gl('decoded_cap')} / {gl('v3')} / {gl('v1')} |"
              + (f" missing: {sorted(missing)}" if missing else ""))
    seqs = sorted({k[2] for k in rows})
    dec = [f for f in MANY if all(rows.get(("skew", f, s), (9, 9))[0] <= 0.85 for s in seqs)]
    v3w = [f for f in MANY if all(rows.get(("skew", f, s), (9, 9))[1] <= 0.85 for s in seqs)]
    full = set(seqs) >= {512, 2048} and all(("skew", f, s) in rows for f in MANY for s in (512, 2048))
    print("\nGATE: " + (f"{len(all_fails)} call(s) FAILED the fp32-reference gate (those cells never count):" if all_fails
                         else f"every arm passed the fp32-reference gate (<= {GATE_X} x dense's error) on every call read"))
    for f in all_fails:
        print("  FAIL " + f)
    print(f"BAR DECODED (gated): {len(dec)}/7 many-group families pass on the skewed draw at seqs {seqs}: {dec} -> "
          f"{'HELD' if len(dec) >= 3 else 'NOT HELD'}")
    # the same 0.85 test at each length alone -- read beside the bar, never instead of it (the bar needs both lengths), so a
    # count that holds at one length only is not read as a family failing everywhere
    print("  DECODED by length (skewed, gated; the bar needs every length): " + "; ".join(
        f"seq {s} {len(p)}/7 {p}" for s in seqs for p in [[f for f in MANY if rows.get(("skew", f, s), (9, 9))[0] <= 0.85]]))
    print(f"BAR V3 (gated): {len(v3w)}/7 pass: {v3w} -> {'HELD' if len(v3w) >= 3 else 'NOT HELD'}")
    if not full:
        print("NOT THE REGISTERED GRID: the skewed draw at seq 512 and 2048 for all seven families is incomplete; no decision")
    elif "5090" not in rec["gpu"]:
        print(f"NOT THE REGISTERED CARD ({rec['gpu']}): the bar is scored as a filter only; no decision")
    elif not post_anchor_ok(rec):
        print(f"NOT A DECISION (amendment 3): the post-probe anchor was {rec.get('anchor_post', 'not recorded')} -- the box's "
              f"class was not re-confirmed after the probe; this draw decides nothing")
    else:
        print("DECISION: " + ("an opt-in decoded route in grouped-nf4-gemm, then a TC1 full-step A/B" if len(dec) >= 3 else
                              "a v3 training default, its own TC1 A/B" if len(v3w) >= 3 else "neither: recorded, no route"))
        predictions(info, rows, dec)


# The registered predictions (RD1-PREREG.md, "Predictions"), each scored mechanically on its own text, only on a reading that
# decides. A line naming families, seqs or draws holds only if every one it names does: DECODED holding does not make P1 hold.
P1_FAMS = ("graniteh", "qwen3", "qwen36")       # "DECODED holds, at least on graniteh, qwen3 and qwen36"
P3_MAX = 1.5                                    # decoded_cap event / device, every many-group skewed cell
P4_FAMS, P4_MIN = ("graniteh", "qwen3", "qwen36"), 1.2   # dense / v1 event, seq 512, skewed
P5_FAM = "mixtral"                              # dense <= decoded_cap on event time at both seqs (no draw named: every draw)


def predictions(info, rows, dec):
    seqs = sorted({k[2] for k in info})
    q = lambda f, s: rows[("skew", f, s)][0]                                                   # noqa: E731
    ratio = lambda k, a, b: info[k]["ev"][a] / info[k]["ev"][b]                                # noqa: E731
    print("\nPREDICTIONS (RD1-PREREG.md; each scored on its registered text):")
    gates = [g for i in info.values() for g in i["gate"].values()]
    nfail, nunread = gates.count(False), gates.count(None)
    print(f"P0 every arm passes the correctness gate on every call -> "
          f"{'REFUTED' if nfail else 'UNREAD' if nunread else 'HELD'} ({len(gates)} arm-cells: {nfail} failed, {nunread} unread)")
    named = ", ".join(f"{f} {'passes' if f in dec else 'fails'} (" + ", ".join(
        f"{q(f, s):.2f} at {s}" if q(f, s) != float("inf") else f"gate FAIL at {s}" for s in seqs) + ")" for f in P1_FAMS)
    p1 = len(dec) >= 3 and all(f in dec for f in P1_FAMS)
    print(f"P1 DECODED holds, at least on {', '.join(P1_FAMS)} -> {'HELD' if p1 else 'REFUTED'}: DECODED "
          f"{'held' if len(dec) >= 3 else 'not held'} ({len(dec)}/7); decoded_cap / best: {named}")
    print("V3 no prediction was registered")
    sk = [(i["evdev"], k) for k, i in info.items() if k[0] == "skew" and k[1] in MANY]
    worst = max(sk)
    print(f"P3 decoded_cap event / device <= {P3_MAX} on every many-group skewed cell -> "
          f"{'HELD' if worst[0] <= P3_MAX else 'REFUTED'} (max {worst[0]:.2f}, {worst[1][1]} at {worst[1][2]}; {len(sk)} cells)")
    p4 = {f: ratio(("skew", f, 512), "dense", "v1") for f in P4_FAMS}
    print(f"P4 dense / v1 event > {P4_MIN} at seq 512, skewed, on {', '.join(P4_FAMS)} -> "
          f"{'HELD' if all(v > P4_MIN for v in p4.values()) else 'REFUTED'} ("
          + ", ".join(f"{f} {v:.2f}" for f, v in p4.items()) + ")")
    draws = sorted({k[0] for k in info if k[1] == P5_FAM})
    per = {d: [(s, ratio((d, P5_FAM, s), "dense", "v1"), ratio((d, P5_FAM, s), "decoded_cap", "v1")) for s in seqs]
           for d in draws}
    ok = {d: all(dn <= dc for _, dn, dc in v) for d, v in per.items()}
    p5 = "UNREAD" if not ok else "HELD" if all(ok.values()) else "REFUTED"
    print(f"P5 on {P5_FAM}, dense <= decoded_cap on event time at both seqs -> {p5} ("
          + "; ".join(f"{d} {'holds' if ok[d] else 'fails'}: " + ", ".join(f"{s} dense {dn:.2f} vs cap {dc:.2f} x v1"
                                                                            for s, dn, dc in v) for d, v in per.items()) + ")")
    if all("E" in i["cell"] and "k" in i["cell"] for i in info.values()):
        print("OBSERVATION (not a registered line): rows per expert = seq x top-k / experts, against decoded_cap / best on "
              "the skewed draw (pass <= 0.85):")
        for s in seqs:
            fams = sorted(MANY, key=lambda f: (s * info[("skew", f, s)]["cell"]["k"] / info[("skew", f, s)]["cell"]["E"], f))
            print(f"  seq {s}: " + ", ".join(
                f"{f} {s * info[('skew', f, s)]['cell']['k'] / info[('skew', f, s)]['cell']['E']:.0f} ({q(f, s):.2f})"
                for f in fams))


TIMING =("device_ms", "event_ms", "wall_s", "device_ms_vs_fused", "event_ms_vs_fused")


def strip_timing(src, dst):
    """A receipt with every timing field removed -- what is committed from a card that may not be timed (the QNAP A2000 is a
    correctness testbed only: shared production box, timings not reproducible). Correctness fields, peak bytes (the
    allocator's, deterministic), group/row counts, chunk counts and skipped arms stay."""
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k not in TIMING}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    rec = clean(json.load(open(src)))
    rec["timing_stripped"] = "every timing field removed: this card is a correctness testbed only"
    json.dump(rec, open(dst, "w"), indent=1)


def gate_only(path):
    """The correctness gate per cell and arm, with peaks and chunk counts -- no timing, no bar, no decision."""
    rec = json.load(open(path))
    print(f"{rec['gpu']}, torch {rec['torch']}, triton {rec['triton']}; decode cap {rec['cap_mib']} MiB -- CORRECTNESS ONLY\n")
    print("| routing | family | seq | groups (<16 rows) | gate v1 / v3 / dense / decoded / cap | max rel_err32 / dense's "
          "(v1, v3, decoded, cap) | peak MiB v1 / v3 / dense / decoded / cap (chunks) |")
    print("|---|---|---|---|---|---|---|")
    all_fails = []
    for c in rec["cells"]:
        t, missing, extra, gate, fails = layer(c)
        all_fails.extend(fails)
        worst = {a: 0.0 for a in ("v1", "v3", "decoded", "decoded_cap")}
        for res in c["proj"].values():
            for r in res.values():
                d = (r.get("dense") or {}).get("rel_err32")
                for a in worst:
                    e = (r.get(a) or {}).get("rel_err32")
                    if d and e is not None:
                        worst[a] = max(worst[a], e / d)
        gl = lambda a: {True: "ok", False: "FAIL", None: "unread"}[gate[a]]                       # noqa: E731
        print(f"| {c['routing']} | {c['fam']} | {c['seq']} | {c['groups']} ({c['groups_under_16_rows']}) | "
              f"{gl('v1')} / {gl('v3')} / {gl('dense')} / {gl('decoded')} / {gl('decoded_cap')} | "
              + ", ".join(f"{worst[a]:.2f}" for a in worst) + " | "
              f"{t['v1']['peak_mib']:.0f} / {t['v3']['peak_mib']:.0f} / {t['dense']['peak_mib']:.0f} / "
              f"{t['decoded']['peak_mib']:.0f} / {t['decoded_cap']['peak_mib']:.0f} ({t['decoded_cap']['chunks']}) |"
              + (f" missing: {sorted(missing)}" if missing else ""))
    print("\nGATE: " + (f"{len(all_fails)} call(s) FAILED the fp32-reference gate:" if all_fails
                         else f"every arm passed the fp32-reference gate (<= {GATE_X} x dense's error) on every call read"))
    for f in all_fails:
        print("  FAIL " + f)
    print("No timing is read from this receipt; no bar; no decision.")


if __name__ == "__main__":
    if sys.argv[1] == "--gate-only":
        gate_only(sys.argv[2])
    elif sys.argv[1] == "--strip-timing":
        strip_timing(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1])
