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
    # A ledger holds MORE THAN ONE MODEL, with round numbers restarting per model
    # (the published timing-3090/timing-l40s files each carry OLMoE and Qwen3).
    # Keying by (round, label) alone let the second model overwrite the first's
    # host/host_self rows, so the paired ratios silently mixed models and could
    # not reproduce the receipt (Bugbot, PR #130). Analyse one model at a time.
    models = list(dict.fromkeys(r.get("model", "?") for r in good))
    if len(models) > 1:
        rc = 0
        for mdl in models:
            print("=" * 72)
            rc |= _one(mdl, [r for r in good if r.get("model") == mdl], ref, ctrl)
        return rc
    return _one(models[0], good, ref, ctrl)


def _one(model, good, ref, ctrl):
    rounds = sorted({r["round"] for r in good})
    labels = [l for l in dict.fromkeys(r["label"] for r in good)]
    by = {(r["round"], r["label"]): r for r in good}
    print(f"model {model}   gpu {good[0]['gpu']}   cores {good[0]['nproc']}")
    print(f"scored rounds {rounds} (round 0 dropped as warmup)\n")

    print(f"{'label':11} {'t_load s (per round)':30} {'step s median (per round)':30}")
    for lab in labels:
        loads = [by[(r, lab)]["t_load_s"] for r in rounds if (r, lab) in by]
        steps = [by[(r, lab)]["step_s_median"] for r in rounds if (r, lab) in by]
        print(f"  {lab:9} {str([round(x, 2) for x in loads]):30} "
              f"{str([round(x, 4) for x in steps]):30}")

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
        def fmt(v):
            return f"{med(v):.3f} [{min(v):.3f}-{max(v):.3f}]" if v else "n/a"
        print(f"  {lab:9} {fmt(rl):>22}   {fmt(rs):>22}")

    if ctrl_step:
        spread = (max(ctrl_step) - min(ctrl_step)) / med(ctrl_step) * 100
        lspread = (max(ctrl_load) - min(ctrl_load)) / med(ctrl_load) * 100
        print(f"\nCONTROL '{ctrl}': step {med(ctrl_step):.3f} [{min(ctrl_step):.3f}-{max(ctrl_step):.3f}]"
              f" (spread {spread:.1f}%)   load {med(ctrl_load):.3f}"
              f" [{min(ctrl_load):.3f}-{max(ctrl_load):.3f}] (spread {lspread:.1f}%)")
        # AMENDMENT 1: the gate is whether the EFFECT clears the control's range,
        # not whether the control's spread is under a fixed threshold. The old
        # rule called a 239% load effect unresolvable because a control moved
        # 8.4% -- it confused resolution with significance. A wide control now
        # WIDENS the band an effect must clear rather than voiding the run.
        print("  Gate (PREREG-timing-AMENDMENT-1): an effect is resolved when its"
              " per-round range does not overlap the control's.")
        for lab in labels:
            if lab in (ref, ctrl):
                continue
            for name, eff, ctl in (("load", paired(lab, "t_load_s"), ctrl_load),
                                   ("step", paired(lab, "step_s_median"), ctrl_step)):
                if not eff:
                    continue
                clear = min(eff) > max(ctl) or max(eff) < min(ctl)
                print(f"  -> {lab:9} {name:5} {med(eff):.3f} [{min(eff):.3f}-{max(eff):.3f}]"
                      f"  vs control [{min(ctl):.3f}-{max(ctl):.3f}]: "
                      f"{'RESOLVED' if clear else 'OVERLAPS — report indistinguishable'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], *(sys.argv[2:])))
