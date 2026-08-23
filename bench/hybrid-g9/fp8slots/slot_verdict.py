"""Verdict calculator for PREREG-slotvalue.md — pure arithmetic over the
sweep receipts. Frozen before the box runs.

  python slot_verdict.py --gate  sweep.json
  python slot_verdict.py --score sweep.json

Gate: per ladder point, A/A noise between pass 1 and pass 2 on the
dram_experts_host bucket. ACCEPT iff median noise <= 5% and every point
<= 10%. Score: uniques accounting (B1), time conversion at 58 us/unique
(B2), the tail per-slot value against the widened claimed band (B3), and
the could-it-have-failed spoiler. The FP8 KV accounting is reported, not
scored.
"""
import json
import sys

US_PER_UNIQUE = 58.0
TAIL_BAND_US = (12.0, 45.0)     # claimed ~17-35, widened 30% for its "~"
B1_FLOOR = 0.15
B2_FLOOR = 0.25


def load(path):
    d = json.load(open(path))
    ladder = sorted(d["ladder"])
    pts = d["points"]

    def point(pno, v):
        return pts[f"p{pno}_v{v:g}"]
    return ladder, point, d


def dram_ms(p):
    return p["decode_median_ms"]["dram_experts_host"]


def uniq_dram_per_step(p):
    steps = p["per_layer"][0]["steps"]
    tot = sum(pl["uniq_dram"] for pl in p["per_layer"])
    return tot / max(1, steps)


def vram_set(p):
    return set(map(tuple, p["manifest_vram"]))


def load_profile_probs(profile_path, top_k):
    from collections import defaultdict
    mass = defaultdict(float)
    tot = defaultdict(float)
    for line in open(profile_path):
        r = json.loads(line)
        if r.get("row") == "expert":
            key = (int(r["layer_id"]), int(r["expert_id"]))
            mass[key] += float(r["tokens_routed"])
            tot[int(r["layer_id"])] += float(r["tokens_routed"])
    return {k: min(1.0, top_k * m / tot[k[0]]) for k, m in mass.items()}


def gate(ladder, point, verbose=True):
    noises = []
    for v in ladder:
        a, b = dram_ms(point(1, v)), dram_ms(point(2, v))
        n = abs(a - b) / ((a + b) / 2)
        noises.append(n)
        if verbose:
            print("gate v=%-5g dram %7.2f / %7.2f ms  noise %5.1f%%%s"
                  % (v, a, b, n * 100, "  FAIL(>10%)" if n > 0.10 else ""))
    med = sorted(noises)[len(noises) // 2]
    ok = med <= 0.05 and max(noises) <= 0.10
    print("gate median noise %.1f%% (bar <= 5%%), worst %.1f%% (bar <= 10%%)"
          % (med * 100, max(noises) * 100))
    print("A/A GATE:", "ACCEPT" if ok else "REJECT -- destroy and re-hunt")
    return ok


def score(path):
    ladder, point, d = load(path)
    if not gate(ladder, point):
        print("VERDICT: VOID (gate failed; sweep should not be scored)")
        return
    p0 = point(1, ladder[0])
    top_k, batch = p0["top_k"], p0["batch"]
    probs = load_profile_probs(d["profile"], top_k)

    def p_touched(key):
        p = probs.get(key, 0.0)
        return 1.0 - (1.0 - p) ** batch

    ok = True
    tail_value = None
    for lo, hi in zip(ladder, ladder[1:]):
        added = [vram_set(point(pn, hi)) - vram_set(point(pn, lo))
                 for pn in (1, 2)]
        assert added[0] == added[1], "manifests differ between passes"
        du_pred = sum(p_touched(k) for k in added[0])
        du = [uniq_dram_per_step(point(pn, lo))
              - uniq_dram_per_step(point(pn, hi)) for pn in (1, 2)]
        dt = [dram_ms(point(pn, lo)) - dram_ms(point(pn, hi))
              for pn in (1, 2)]
        du_m = sum(du) / 2
        dt_m = sum(dt) / 2
        b1_allow = max(B1_FLOOR * du_pred, 3 * abs(du[0] - du[1]))
        b1 = abs(du_m - du_pred) <= b1_allow
        dt_pred = US_PER_UNIQUE * du_m / 1e3
        b2_allow = max(B2_FLOOR * dt_pred, 3 * abs(dt[0] - dt[1]))
        b2 = abs(dt_m - dt_pred) <= b2_allow
        ok &= b1 and b2
        slots = len(added[0])
        val = dt_m * 1e3 / max(1, slots)
        print("bracket %g->%g  +%d slots | dU pred %6.1f meas %6.1f "
              "(allow %5.1f) %s | dT pred %6.2f meas %6.2f ms "
              "(allow %5.2f) %s | %5.1f us/slot"
              % (lo, hi, slots, du_pred, du_m, b1_allow,
                 "B1-PASS" if b1 else "B1-FAIL", dt_pred, dt_m, b2_allow,
                 "B2-PASS" if b2 else "B2-FAIL", val))
        tail_value = val
    b3 = TAIL_BAND_US[0] <= tail_value <= TAIL_BAND_US[1]
    ok &= b3
    print("B3 tail bracket: %.1f us/slot vs band [%g, %g] -> %s"
          % (tail_value, *TAIL_BAND_US, "PASS" if b3 else "FAIL"))

    full = [dram_ms(point(pn, ladder[0])) - dram_ms(point(pn, ladder[-1]))
            for pn in (1, 2)]
    spread = abs(full[0] - full[1])
    full_m = sum(full) / 2
    if full_m <= 3 * spread:
        print("SPOILER: full-range dT %.2f ms <= 3x A/A spread %.2f ms"
              % (full_m, 3 * spread))
        print("VERDICT: VOID (the sweep could not have failed)")
        return
    print("spoiler ok: full-range dT %.2f ms > 3x spread %.2f ms"
          % (full_m, 3 * spread))
    print("VERDICT:", "CERTIFIED" if ok else "REFUTED")


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    if mode == "--gate":
        ladder, point, _ = load(path)
        sys.exit(0 if gate(ladder, point) else 1)
    score(path)
