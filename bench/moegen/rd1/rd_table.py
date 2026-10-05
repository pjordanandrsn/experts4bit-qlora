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


def main(path):
    rec = json.load(open(path))
    print(f"{rec['gpu']}, torch {rec['torch']}, triton {rec['triton']}; decode cap {rec['cap_mib']} MiB; {rec['reps']} reps\n")
    print("| routing | family | seq | groups (<16 rows) | v1 ev ms | v3 / v1 | dense / v1 | decoded / v1 | decoded_cap / v1 "
          "| decoded_cap / best(v1,v3,dense) | ev / dev (decoded_cap) | peak MiB v1 / v3 / dense / decoded / cap (chunks) | µs per extra chunk | gate cap / v3 / v1 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = {}
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
    print(f"BAR V3 (gated): {len(v3w)}/7 pass: {v3w} -> {'HELD' if len(v3w) >= 3 else 'NOT HELD'}")
    if not full:
        print("NOT THE REGISTERED GRID: the skewed draw at seq 512 and 2048 for all seven families is incomplete; no decision")
    elif "5090" not in rec["gpu"]:
        print(f"NOT THE REGISTERED CARD ({rec['gpu']}): the bar is scored as a filter only; no decision")
    else:
        print("DECISION: " + ("an opt-in decoded route in grouped-nf4-gemm, then a TC1 full-step A/B" if len(dec) >= 3 else
                              "a v3 training default, its own TC1 A/B" if len(v3w) >= 3 else "neither: recorded, no route"))


TIMING = ("device_ms", "event_ms", "wall_s", "device_ms_vs_fused", "event_ms_vs_fused")


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
