"""Paired analysis: ratio WITHIN each round, then summarise across rounds.

Ratio-of-means is the wrong statistic here. If the machine drifts between early
and late rounds, a mean-of-all-host over mean-of-all-arena inherits the drift.
Taking each round's own host/arena ratio cancels anything that moved both arms
together, which is the entire point of interleaving them.

`host_self` is the same configuration timed twice in one round. Its ratio should
be ~1.000, and whatever spread it shows is the harness's resolution. Any effect
inside that spread is UNRESOLVABLE and must not be reported as a number --
0.17.1 published a 1.03x that died exactly this way.
"""
import json
import sys


def med(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else float("nan")


def main(path, ref="host", ctrl="host_self"):
    rows = [json.loads(l) for l in open(path) if l.strip().startswith("{")]
    good = [r for r in rows if r.get("ok") and r.get("scored")]
    if not good:
        print("no scored rows yet"); return 1
    rounds = sorted({r["round"] for r in good})
    labels = [l for l in dict.fromkeys(r["label"] for r in good)]
    by = {(r["round"], r["label"]): r for r in good}
    print(f"model {good[0]['model']}   gpu {good[0]['gpu']}   cores {good[0]['nproc']}")
    print(f"scored rounds {rounds} (round 0 dropped as warmup)\n")

    print(f"{'label':11} {'t_load s (per round)':30} {'step s median (per round)':30}")
    for lab in labels:
        L = [by[(r, lab)]["t_load_s"] for r in rounds if (r, lab) in by]
        S = [by[(r, lab)]["step_s_median"] for r in rounds if (r, lab) in by]
        print(f"  {lab:9} {str([round(x,2) for x in L]):30} {str([round(x,4) for x in S]):30}")

    def paired(lab, field):
        out = []
        for r in rounds:
            a, b = by.get((r, ref)), by.get((r, lab))
            if a and b and a[field]:
                out.append(b[field] / a[field])
        return out

    print(f"\nPAIRED ratios vs '{ref}', computed within each round:")
    print(f"  {'label':11} {'t_load':>22}   {'step':>22}")
    ctrl_load = ctrl_step = None
    for lab in labels:
        if lab == ref:
            continue
        rl, rs = paired(lab, "t_load_s"), paired(lab, "step_s_median")
        if lab == ctrl:
            ctrl_load, ctrl_step = rl, rs
        f = lambda v: f"{med(v):.3f} [{min(v):.3f}-{max(v):.3f}]" if v else "n/a"
        print(f"  {lab:9} {f(rl):>22}   {f(rs):>22}")

    if ctrl_step:
        spread = (max(ctrl_step) - min(ctrl_step)) / med(ctrl_step) * 100
        lspread = (max(ctrl_load) - min(ctrl_load)) / med(ctrl_load) * 100
        print(f"\nCONTROL '{ctrl}': step median {med(ctrl_step):.4f}, spread {spread:.1f}%"
              f"   |   load median {med(ctrl_load):.4f}, spread {lspread:.1f}%")
        print("  Registered gate: report a ratio only if the control spread is within +/-8%.")
        for name, sp in (("step", spread), ("load", lspread)):
            print(f"  -> {name}: {'RESOLVABLE' if sp <= 8 else 'NOT RESOLVABLE — report indistinguishable'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], *(sys.argv[2:])))
