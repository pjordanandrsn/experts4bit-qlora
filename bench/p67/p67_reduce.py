#!/usr/bin/env python3
"""P67 reducer -- training parity read against a per-family FLOOR, not against zero.

Governed by ``bench/p67/P67-PREREG.md``. stdlib only; nothing in a results page is hand-transcribed.

The quantities are tp1's two, computed exactly as ``bench/tp4/tp4_reduce.py:parity`` computes them, on the
final TRAIN loss (``loss_last``) and the per-step TRAIN losses (``losses``) -- never the held-out loss, which
is reported beside them under its own name and gates nothing:

    D_final = |loss_last(arm) - loss_last(reference)|
    D_med   = median over steps of |loss_t(arm) - loss_t(reference)|

The FLOOR is how far the reference path lands from ITSELF under a change that is correct by construction:
the same session's ``reference_attn4`` against ``reference_attn4_repeat`` (a plain repeat) and
``reference_attn4_perm<s>`` (the per-expert loop in a fixed permuted order, ``E4B_REFERENCE_EXPERT_ORDER``).
A floor draw is admissible only if it is VALID, from the SAME session and fixture, carries proof of what it
ran, and is not bit-identical to the reference (a zero floor is the band against zero again).

    F_hi(q) = max over admissible floor draws of D_q          (needs >= N_MIN draws, else NO-FLOOR)
    B(q)    = max(TOL, K * F_hi(q))
    PASS    iff D_final <= B(final) and D_med <= B(med)

``carried_by`` says which term decided a PASS: ``tolerance`` (both D <= TOL, i.e. tp1's constant band alone
passes) or ``floor``. ``detectable`` says whether an arm is distinguishable from the floor (D > K * F_hi on
either quantity) -- reported, never a verdict by itself. With no floor measured, a constant-band PASS is still
a floor-band PASS (B >= TOL by construction); a constant-band FAIL is NO-FLOOR: undecidable, not FAIL.

Usage:
    p67_reduce.py --session DIR [--consistency DIR ...] [--md OUT.md] [--json OUT.json]   # the registered read
    p67_reduce.py --reread DIR [DIR ...] [--tp1 DIR] [--md OUT.md] [--json OUT.json]       # existing receipts
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import sys

PREREG = "bench/p67/P67-PREREG.md"
TOL = 0.05          # tp1's B2/C2 constant, kept as the practical tolerance (P67-PREREG "The margin")
K = 3.0             # the multiple on the floor's upper edge (P67-PREREG "The margin")
N_MIN = 3           # admissible floor draws a family needs before its floor exists (P67-PREREG "Draws")
STEP0_FLAG = 1e-4   # a floor draw whose step-0 train loss moves more than this is FLAGGED (Q3), never refused

REFERENCE = "reference_attn4"
FLOOR_REPEAT = "reference_attn4_repeat"
FLOOR_PERM_RE = re.compile(r"^reference_attn4_perm(\d+)$")
JUDGED = ("fused_attn4", "batched_attn4", "fused_attn4_nodgrad")
OK = ("ok", "OK")
# The fixture a pair must share to be read at all. tokens + init make it the same data from the same start.
FIXTURE_KEYS = ("fam", "steps", "seq", "micro_batch", "accum", "r", "alpha", "lr", "seed", "optimizer", "attn_4bit")


# ----------------------------------------------------------------------------- inputs
def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _receipt_dir(d):
    """A launcher run dir holds the box tree under tp4/; a curated bundle is flat."""
    return os.path.join(d, "tp4") if os.path.isdir(os.path.join(d, "tp4")) else d


def load_tp4(d, session=None):
    """Every e4b arm receipt in a tp4-shaped dir: {tag: rec}. ``rec`` = the JSON plus ``_session``/``_path``/``_sha``."""
    rd = _receipt_dir(d)
    out = {}
    for p in sorted(glob.glob(os.path.join(rd, "*_e4b_*.json"))):
        try:
            r = json.load(open(p))
        except (OSError, ValueError):
            continue
        if not isinstance(r, dict) or r.get("framework", "e4b") != "e4b" or not r.get("tag"):
            continue
        r = dict(r, _session=session or os.path.normpath(d), _path=p, _sha=_sha(p))
        out.setdefault(r.get("fam"), {})[r["tag"]] = r
    return out


def load_tp1(d):
    """tp1's bundle (``<fam>_train_<arm>.json``, clinical fixture, one box) mapped onto tp4's tag names."""
    tag_of = {"reference": REFERENCE, "fused": "fused_attn4", "batched": "batched_attn4"}
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*_train_*.json"))):
        m = re.match(r"^([a-z0-9]+)_train_([a-z_]+)\.json$", os.path.basename(p))
        if not m or m.group(2) not in tag_of:
            continue
        try:
            r = json.load(open(p))
        except (OSError, ValueError):
            continue
        fam = m.group(1)
        r = dict(r, fam=fam, tag=tag_of[m.group(2)], _session="tp1 (clinical, bf16 attention)", _path=p, _sha=_sha(p), _tp1=True)
        out.setdefault(fam, {})[r["tag"]] = r
    return out


# ----------------------------------------------------------------------------- validity
def fixture(r):
    t = (r.get("tokens") or {}).get("sha256") if not r.get("_tp1") else "tp1-clinical"
    return tuple(r.get(k) for k in FIXTURE_KEYS) + (t, r.get("init_sha"))


def valid_run(r):
    """A receipt that trained: status ok, C1 bit-exact, one loss per step."""
    why = []
    if r is None:
        return False, ["missing"]
    if r.get("status") not in OK:
        why.append(f"status {r.get('status')!r}")
    if not r.get("C1_bit_exact"):
        why.append("C1 frozen bytes not bit-exact")
    L = r.get("losses") or []
    if not L or (r.get("steps") and len(L) != r.get("steps")):
        why.append(f"{len(L)} losses for {r.get('steps')} steps")
    return (not why), why


def same_pair(ref, arm):
    """What any pair must share: fixture, tokens, init, trainable count, step count."""
    why = []
    if fixture(arm) != fixture(ref):
        diff = [k for k, a, b in zip(FIXTURE_KEYS + ("tokens_sha", "init_sha"), fixture(arm), fixture(ref)) if a != b]
        why.append(f"fixture differs from the reference ({', '.join(diff)})")
    if arm.get("trainable_params") != ref.get("trainable_params"):
        why.append(f"trainable {arm.get('trainable_params')} != reference {ref.get('trainable_params')}")
    if len(arm.get("losses") or []) != len(ref.get("losses") or []):
        why.append("step counts differ")
    return why


def judged_valid(ref, arm):
    """tp4's validity for an accelerated arm (tp4_reduce.parity), plus P56's engagement rule for batched."""
    ok, why = valid_run(arm)
    if not ok:
        return False, why
    why = same_pair(ref, arm)
    n = arm.get("n_patched") or 0
    if n == 0:
        why.append("n_patched == 0 (the accelerated path was never installed)")
    A = int(arm.get("accum") or 1)
    kmin = arm.get("kernel_calls_per_step_min")
    if arm["tag"] != "batched_attn4" and kmin is not None and kmin < 2 * n * A:
        why.append(f"kernel calls/step min {kmin} < 2*n_patched*accum={2 * n * A}")
    if arm["tag"] == "batched_attn4":
        st = arm.get("batched_stats")
        if st is None and arm.get("_tp1"):
            if kmin is not None and kmin < 2 * n:   # tp1's own rule for its batched rows (no counter existed)
                why.append(f"kernel calls/step min {kmin} < 2*n_patched={2 * n} (tp1: fallback took layers)")
        elif not st:
            why.append("batched arm carries no batched_stats (engagement unproven)")
        else:
            if not st.get("calls"):
                why.append("batched_stats records 0 calls")
            if st.get("fallback_calls"):
                why.append(f"batched path fell back on {st['fallback_calls']}/{st.get('calls')} calls")
    return (not why), why


# ----------------------------------------------------------------------------- the two registered quantities
def deltas(ref, arm):
    """tp4_reduce.parity's two numbers on the TRAIN loss, plus the held-out and step-0 deltas, NAMED."""
    la, lr = arm["losses"], ref["losses"]
    d_final = abs(arm["loss_last"] - ref["loss_last"])
    d_med = statistics.median(abs(a - b) for a, b in zip(la, lr))
    ea, er = arm.get("eval_loss_final"), ref.get("eval_loss_final")
    # Unrounded: tp4_reduce and p56_reduce round once, at 5 decimals, and a second rounding here would print
    # 0.124205 as 0.12420 against the standing row's 0.12421. The renderer rounds, once.
    return {"d_final_train": d_final, "d_med_train": d_med,
            "d_heldout": (abs(ea - er) if ea is not None and er is not None else None),
            "d_step0_train": abs(la[0] - lr[0]),
            "first_differing_step": next((i for i, (a, b) in enumerate(zip(la, lr)) if a != b), None),
            "sign_final": (1 if arm["loss_last"] > ref["loss_last"] else -1 if arm["loss_last"] < ref["loss_last"] else 0)}


# ----------------------------------------------------------------------------- floor draws
def floor_kind(tag):
    if tag == FLOOR_REPEAT:
        return "repeat", None
    m = FLOOR_PERM_RE.match(tag or "")
    return ("perm", f"perm:{int(m.group(1))}") if m else (None, None)


def floor_draw(ref, alt):
    """One floor draw: the session's reference against its repeat or a perturbed repeat."""
    kind, want = floor_kind(alt.get("tag"))
    d = {"tag": alt.get("tag"), "kind": kind, "admissible": False, "why": []}
    ok, why = valid_run(alt)
    if not ok:
        d["why"] = why
        return d
    d["why"] = same_pair(ref, alt)
    if alt.get("_session") != ref.get("_session"):
        d["why"].append("a different session from the reference (floor draws are same-box, same-session)")
    # proof of what ran: the switch's own counter, read from the loop (tp4_arm.py `reference_order`)
    ro_ref, ro = ref.get("reference_order") or {}, alt.get("reference_order")
    if ro_ref.get("calls"):
        d["why"].append(f"the reference itself ran a non-default order ({ro_ref.get('order')})")
    if ro is None:
        d["why"].append("no reference_order on the receipt (what the loop ran is unproven)")
    elif kind == "perm":
        if ro.get("requested") != want or ro.get("order") != want or not ro.get("calls"):
            d["why"].append(f"perturbation not proven: requested {ro.get('requested')!r}, applied {ro.get('order')!r} "
                            f"on {ro.get('calls')} calls (want {want!r} on > 0)")
    elif kind == "repeat":
        if ro.get("requested") or ro.get("calls"):
            d["why"].append(f"the repeat ran a non-default order ({ro.get('order')!r} on {ro.get('calls')} calls)")
    d.update(deltas(ref, alt))
    d["identical"] = alt["losses"] == ref["losses"] and alt["loss_last"] == ref["loss_last"]
    if d["identical"] and not d["why"]:
        d["why"].append("bit-identical to the reference at print precision: a floor of ZERO is the band against zero again")
    d["step0_flag"] = d["d_step0_train"] > STEP0_FLAG
    d["admissible"] = not d["why"]
    return d


def floor_of(draws):
    adm = [d for d in draws if d["admissible"]]
    f = {"n_admissible": len(adm), "n_candidates": len(draws)}
    if len(adm) < N_MIN:
        f.update({"exists": False, "why": f"{len(adm)} admissible floor draw(s) < N_MIN={N_MIN}"})
        return f
    for q in ("final", "med"):
        vals = [d[f"d_{q}_train"] for d in adm]
        hi = max(vals)
        f[q] = {"hi": hi, "lo": min(vals), "values": vals, "spread_ratio": (round(hi / min(vals), 3) if min(vals) > 0 else None),
                "band": max(TOL, K * hi)}
    f["exists"] = True
    return f


def verdict(dq, floor):
    """The registered rule, applied to one judged arm's two deltas."""
    d_f, d_m = dq["d_final_train"], dq["d_med_train"]
    const_pass = d_f <= TOL and d_m <= TOL
    if not floor.get("exists"):
        if const_pass:
            return {"verdict": "PASS", "carried_by": "tolerance", "detectable": None,
                    "note": "no floor measured; B >= TOL by construction, so the constant band's PASS stands"}
        return {"verdict": "NO-FLOOR", "carried_by": None, "detectable": None,
                "note": f"the constant band fails and no floor exists ({floor.get('why')}): undecidable, not FAIL"}
    bf, bm = floor["final"]["band"], floor["med"]["band"]
    ok = d_f <= bf and d_m <= bm
    det = d_f > K * floor["final"]["hi"] or d_m > K * floor["med"]["hi"]
    out = {"verdict": "PASS" if ok else "FAIL", "carried_by": (("tolerance" if const_pass else "floor") if ok else None),
           "detectable": det, "band_final": bf, "band_med": bm,
           "ratio_final": (round(d_f / floor["final"]["hi"], 3) if floor["final"]["hi"] else None),
           "ratio_med": (round(d_m / floor["med"]["hi"], 3) if floor["med"]["hi"] else None)}
    return out


def constant_verdict(dq):
    return "PASS" if dq["d_final_train"] <= TOL and dq["d_med_train"] <= TOL else "FAIL"


# ----------------------------------------------------------------------------- the registered read (one session)
def reduce_session(session_dir, consistency_dirs=()):
    fams = load_tp4(session_dir)
    others = [load_tp4(d) for d in consistency_dirs]
    res = {}
    for fam, recs in sorted(fams.items(), key=lambda kv: str(kv[0])):
        ref = recs.get(REFERENCE)
        ok, why = valid_run(ref)
        if not ok:
            res[fam] = {"reference": None, "why": "no valid reference arm: " + "; ".join(why), "reading": "NO-REF"}
            continue
        draws = [floor_draw(ref, r) for t, r in sorted(recs.items()) if floor_kind(t)[0]]
        rep = next((d for d in draws if d["kind"] == "repeat"), None)
        # the determinism finding: only a repeat that is otherwise admissible can answer it
        rep_other = [w for w in (rep or {}).get("why", []) if not w.startswith("bit-identical")]
        rep_identical = rep["identical"] if (rep and "identical" in rep and not rep_other) else None
        fl = floor_of(draws)
        judged = []
        for tag in JUDGED:
            arm = recs.get(tag)
            if arm is None:
                continue
            jv, jwhy = judged_valid(ref, arm)
            row = {"tag": tag, "status": "VALID" if jv else "VOID", "why": jwhy}
            if jv:
                dq = deltas(ref, arm)
                row.update(dq, constant=constant_verdict(dq), **verdict(dq, fl))
            judged.append(row)
        # consistency: the same fixture's judged pairs from OTHER sessions, read against THIS session's floor
        cons, skipped = [], []
        for d, o in zip(consistency_dirs, others):
            orecs = o.get(fam) or {}
            oref = orecs.get(REFERENCE)
            if not valid_run(oref)[0] or fixture(oref) != fixture(ref):
                # said, never silent: a consistency session that cannot be read against this one is listed
                skipped.append({"session": os.path.normpath(d), "why": ("no valid reference arm" if not valid_run(oref)[0]
                                                                        else "a different fixture (" + ", ".join(same_pair(ref, oref)) + ")")})
                continue
            for tag in JUDGED:
                arm = orecs.get(tag)
                if arm is None or not judged_valid(oref, arm)[0]:
                    continue
                dq = deltas(oref, arm)
                cons.append({"session": oref["_session"], "tag": tag, **dq, **verdict(dq, fl)})
        mixed = [c for c in cons for j in judged if j["tag"] == c["tag"] and j.get("verdict") and c["verdict"] != j["verdict"]]
        # the family's reading word, which the prereg's decision table keys on: MIXED beats everything; without a
        # floor the band could not be applied, whatever the judged rows' constant-band verdicts were
        reading = "MIXED" if mixed else ("READ" if fl["exists"] else "NO-FLOOR")
        res[fam] = {"reference": {"loss_last": ref["loss_last"], "eval_loss_final": ref.get("eval_loss_final"),
                                  "s_per_step": ref.get("s_per_step_median_11plus"), "session": ref["_session"]},
                    "floor_draws": draws, "floor": fl,
                    "reference_bit_identical_on_repeat": rep_identical,
                    "judged": judged, "consistency": cons, "consistency_skipped": skipped, "reading": reading}
    return res


# ----------------------------------------------------------------------------- the re-read of existing receipts
def reduce_reread(dirs, tp1_dir=None):
    """Every family, every fixture group: the same-session judged pairs under the band, the floor draws the
    registered rule admits (none exist before P67's own draw), and -- DISCLOSED, never admitted -- every
    cross-session same-path pair, which is the evidence that a floor is not zero."""
    sessions = [load_tp4(d, session=os.path.basename(os.path.normpath(d))) for d in dirs]
    if tp1_dir:
        sessions.append(load_tp1(tp1_dir))
    groups = {}
    for s in sessions:
        for fam, recs in s.items():
            for tag, r in recs.items():
                if valid_run(r)[0]:
                    groups.setdefault((fam, fixture(r)), []).append(r)
    out = []
    for (fam, fx), rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        refs = [r for r in rs if r["tag"] == REFERENCE]
        entry = {"fam": fam, "fixture": dict(zip(FIXTURE_KEYS + ("tokens_sha", "init_sha"), fx)), "sessions": sorted({r["_session"] for r in rs}),
                 "judged": [], "floor_draws": [], "cross_session_pairs": []}
        for ref in refs:
            same = [r for r in rs if r["_session"] == ref["_session"]]
            draws = [floor_draw(ref, r) for r in same if floor_kind(r["tag"])[0]]
            entry["floor_draws"] += [dict(d, session=ref["_session"]) for d in draws]
            fl = floor_of(draws)
            for r in same:
                if r["tag"] not in JUDGED:
                    continue
                jv, jwhy = judged_valid(ref, r)
                row = {"session": ref["_session"], "tag": r["tag"], "status": "VALID" if jv else "VOID", "why": jwhy}
                if jv:
                    dq = deltas(ref, r)
                    row.update(dq, constant=constant_verdict(dq), **verdict(dq, fl))
                entry["judged"].append(row)
        # disclosed only: the same TAG in two different sessions of the same fixture
        by_tag = {}
        for r in rs:
            by_tag.setdefault(r["tag"], []).append(r)
        for tag, lst in sorted(by_tag.items()):
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    a, b = lst[i], lst[j]
                    if a["_session"] == b["_session"] or same_pair(a, b):
                        continue
                    entry["cross_session_pairs"].append({"tag": tag, "a": a["_session"], "b": b["_session"], **deltas(a, b),
                                                         "admissible": False, "why": "different sessions (disclosed, never a floor draw)"})
        entry["n_admissible_floor_draws"] = sum(1 for d in entry["floor_draws"] if d["admissible"])
        out.append(entry)
    inputs = sorted({(r["_session"], os.path.basename(r["_path"]), r["_sha"]) for s in sessions for recs in s.values() for r in recs.values()})
    return {"groups": out, "inputs": [{"session": s, "file": f, "sha256": h} for s, f, h in inputs]}


# ----------------------------------------------------------------------------- rendering
def _f(x, nd=5):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render_session(res):
    L = [f"P67 registered read ({PREREG}): TOL={TOL}, K={K:g}, N_MIN={N_MIN}. Quantities on the TRAIN loss; held-out beside, never gated.", ""]
    for fam, R in res.items():
        L.append(f"### {fam} — reading {R['reading']}")
        if R.get("reference") is None:
            L.append(f"- {R['why']}")
            continue
        L.append(f"- reference: loss_last {_f(R['reference']['loss_last'])}, held-out {_f(R['reference']['eval_loss_final'])}")
        rb = R["reference_bit_identical_on_repeat"]
        L.append(f"- plain repeat bit-identical to the reference on this box: {rb}")
        L.append("| floor draw | admissible | D_final train | D_med train | Δ held-out | Δ step-0 | first differing step | why not |")
        L.append("|---|---|---|---|---|---|---|---|")
        for d in R["floor_draws"]:
            L.append(f"| `{d['tag']}` | {d['admissible']} | {_f(d.get('d_final_train'))} | {_f(d.get('d_med_train'))} | {_f(d.get('d_heldout'))} | "
                     f"{_f(d.get('d_step0_train'), 6)}{' FLAG' if d.get('step0_flag') else ''} | {d.get('first_differing_step')} | {'; '.join(d['why'])} |")
        fl = R["floor"]
        if fl["exists"]:
            L.append(f"- floor ({fl['n_admissible']} draws): F_hi final {_f(fl['final']['hi'])} (lo {_f(fl['final']['lo'])}) -> band {_f(fl['final']['band'])}; "
                     f"F_hi med {_f(fl['med']['hi'])} (lo {_f(fl['med']['lo'])}) -> band {_f(fl['med']['band'])}")
        else:
            L.append(f"- floor: NONE -- {fl['why']}")
        L.append("| judged arm | status | D_final train | D_med train | Δ held-out | constant 0.05 | **floor band** | carried by | detectable | D/F_hi (final, med) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for j in R["judged"]:
            if j["status"] != "VALID":
                L.append(f"| `{j['tag']}` | VOID | — | — | — | — | — | — | — | {'; '.join(j['why'])} |")
                continue
            L.append(f"| `{j['tag']}` | VALID | {_f(j['d_final_train'])} | {_f(j['d_med_train'])} | {_f(j['d_heldout'])} | {j['constant']} | **{j['verdict']}** | "
                     f"{j.get('carried_by') or '—'} | {j.get('detectable')} | {_f(j.get('ratio_final'), 2)}, {_f(j.get('ratio_med'), 2)} |")
        for c in R["consistency"]:
            L.append(f"- consistency ({c['session']}): `{c['tag']}` D_final {_f(c['d_final_train'])} D_med {_f(c['d_med_train'])} -> {c['verdict']}")
        for c in R.get("consistency_skipped", []):
            L.append(f"- consistency session NOT read ({c['session']}): {c['why']}")
        L.append("")
    return "\n".join(L)


def render_reread(res):
    L = [f"P67 re-read of existing receipts ({PREREG}): TOL={TOL}, K={K:g}, N_MIN={N_MIN}. TRAIN-loss quantities; held-out beside.", ""]
    L.append("| family | fixture | session | judged arm | status | D_final train | D_med train | Δ held-out | constant 0.05 | floor band | carried by | admissible floor draws |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g in res["groups"]:
        fx = g["fixture"]
        fxs = f"N={fx['steps']} seq={fx['seq']} mb={fx['micro_batch']} acc={fx['accum']} r={fx['r']} tok={str(fx['tokens_sha'])[:8]} init={str(fx['init_sha'])[:8]}"
        for j in g["judged"]:
            if j["status"] != "VALID":
                L.append(f"| {g['fam']} | {fxs} | {j['session']} | `{j['tag']}` | VOID | — | — | — | — | — | — | {g['n_admissible_floor_draws']} ({'; '.join(j['why'])}) |")
                continue
            L.append(f"| {g['fam']} | {fxs} | {j['session']} | `{j['tag']}` | VALID | {_f(j['d_final_train'])} | {_f(j['d_med_train'])} | {_f(j['d_heldout'])} | "
                     f"{j['constant']} | **{j['verdict']}** | {j.get('carried_by') or '—'} | {g['n_admissible_floor_draws']} |")
    L += ["", "Disclosed, never admitted (different sessions): the same arm on two boxes of the same fixture.", "",
          "| family | arm | session a | session b | D_final train | D_med train | Δ held-out | Δ step-0 | first differing step |", "|---|---|---|---|---|---|---|---|---|"]
    for g in res["groups"]:
        for p in g["cross_session_pairs"]:
            L.append(f"| {g['fam']} | `{p['tag']}` | {p['a']} | {p['b']} | {_f(p['d_final_train'])} | {_f(p['d_med_train'])} | {_f(p['d_heldout'])} | "
                     f"{_f(p['d_step0_train'], 6)} | {p['first_differing_step']} |")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session", help="the registered draw's run dir (one box, one session)")
    g.add_argument("--reread", nargs="+", help="existing tp4-shaped run dirs, each one session")
    ap.add_argument("--consistency", nargs="*", default=[], help="with --session: other sessions' same-fixture pairs")
    ap.add_argument("--tp1", help="with --reread: tp1's bundle (bench/train-parity-20260905/tp1)")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.session:
        res = reduce_session(a.session, a.consistency)
        md = render_session(res)
    else:
        res = reduce_reread(a.reread, a.tp1)
        md = render_reread(res)
    print(md)
    if a.md:
        open(a.md, "w").write(md + "\n")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
