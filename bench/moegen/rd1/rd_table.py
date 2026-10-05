#!/usr/bin/env python3
"""Lane RD1's table and registered bar (bench/moegen/rd1/RD1-PREREG.md) from rd_probe.py's JSON.

A row is one MoE layer's frozen-expert GEMM work in one training step: the sum of its four calls (the up or gate_up and the
down projection, forward and dgrad), per arm. Peak is the largest per-call transient of the arm's four calls.

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


def layer(c):
    tot = {a: {"device_ms": 0.0, "event_ms": 0.0, "peak_mib": 0.0, "rel_err": 0.0, "chunks": 0} for a in ARMS}
    missing = set()
    extra = {"chunks": 0, "ms": 0.0}            # decoded_cap minus decoded, over the calls the cap split
    for res in c["proj"].values():
        for r in res.values():
            for a in ARMS:
                m = r.get(a)
                if not m:
                    missing.add(a)
                    continue
                t = tot[a]
                t["device_ms"] += m["device_ms"]
                t["event_ms"] += m["event_ms"]
                t["peak_mib"] = max(t["peak_mib"], m["peak_mib"])
                t["rel_err"] = max(t["rel_err"], m.get("rel_err", 0.0))
                t["chunks"] = max(t["chunks"], m.get("chunks", 1))
            if r.get("decoded_cap") and r.get("decoded") and r["decoded_cap"]["chunks"] > 1:
                extra["chunks"] += r["decoded_cap"]["chunks"] - 1
                extra["ms"] += r["decoded_cap"]["event_ms"] - r["decoded"]["event_ms"]
    return tot, missing, extra


def main(path):
    rec = json.load(open(path))
    print(f"{rec['gpu']}, torch {rec['torch']}, triton {rec['triton']}; decode cap {rec['cap_mib']} MiB; {rec['reps']} reps\n")
    print("| routing | family | seq | groups (<16 rows) | v1 ev ms | v3 / v1 | dense / v1 | decoded / v1 | decoded_cap / v1 "
          "| decoded_cap / best(v1,v3,dense) | ev / dev (decoded_cap) | peak MiB v1 / v3 / dense / decoded / cap (chunks) | µs per extra chunk |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = {}
    for c in rec["cells"]:
        t, missing, extra = layer(c)
        per_chunk = f"{1000 * extra['ms'] / extra['chunks']:+.0f}" if extra["chunks"] else "—"
        ev = {a: t[a]["event_ms"] for a in ARMS}
        best = min(ev["v1"], ev["v3"], ev["dense"]) if not missing & {"v1", "v3", "dense"} else float("nan")
        q = ev["decoded_cap"] / best if "decoded_cap" not in missing else float("nan")
        v3 = ev["v3"] / ev["v1"] if "v3" not in missing else float("nan")
        rows[(c["routing"], c["fam"], c["seq"])] = (q, v3)
        evdev = t["decoded_cap"]["event_ms"] / t["decoded_cap"]["device_ms"] if t["decoded_cap"]["device_ms"] else float("nan")
        r = lambda a: f"{ev[a] / ev['v1']:.2f}" if a not in missing else "—"                      # noqa: E731
        print(f"| {c['routing']} | {c['fam']} | {c['seq']} | {c['groups']} ({c['groups_under_16_rows']}) | {ev['v1']:.2f} | "
              f"{r('v3')} | {r('dense')} | {r('decoded')} | {r('decoded_cap')} | {q:.2f} | {evdev:.2f} | "
              f"{t['v1']['peak_mib']:.0f} / {t['v3']['peak_mib']:.0f} / {t['dense']['peak_mib']:.0f} / "
              f"{t['decoded']['peak_mib']:.0f} / {t['decoded_cap']['peak_mib']:.0f} ({t['decoded_cap']['chunks']}) | {per_chunk} |"
              + (f" missing: {sorted(missing)}" if missing else ""))
    seqs = sorted({k[2] for k in rows})
    dec = [f for f in MANY if all(rows.get(("skew", f, s), (9, 9))[0] <= 0.85 for s in seqs)]
    v3w = [f for f in MANY if all(rows.get(("skew", f, s), (9, 9))[1] <= 0.85 for s in seqs)]
    full = set(seqs) >= {512, 2048} and all(("skew", f, s) in rows for f in MANY for s in (512, 2048))
    print(f"\nBAR DECODED: {len(dec)}/7 many-group families pass on the skewed draw at seqs {seqs}: {dec} -> "
          f"{'HELD' if len(dec) >= 3 else 'NOT HELD'}")
    print(f"BAR V3: {len(v3w)}/7 pass: {v3w} -> {'HELD' if len(v3w) >= 3 else 'NOT HELD'}")
    if not full:
        print("NOT THE REGISTERED GRID: the skewed draw at seq 512 and 2048 for all seven families is incomplete; no decision")
    elif "5090" not in rec["gpu"]:
        print(f"NOT THE REGISTERED CARD ({rec['gpu']}): the bar is scored as a filter only; no decision")
    else:
        print("DECISION: " + ("an opt-in decoded route in grouped-nf4-gemm, then a TC1 full-step A/B" if len(dec) >= 3 else
                              "a v3 training default, its own TC1 A/B" if len(v3w) >= 3 else "neither: recorded, no route"))


if __name__ == "__main__":
    main(sys.argv[1])
