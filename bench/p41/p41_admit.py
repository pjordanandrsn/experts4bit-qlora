#!/usr/bin/env python3
"""bench/p41/p41_admit.py -- the lane's admission and footprint rules, executable (pre-registration: bench/p41/P41-PREREG.md).

`admit`     -- after an arm: the validity rules (PREREG "Validity rules", "Expected trainable counts", reading rule 2) applied
               to the harness's receipt. A receipt that trained but fails a rule is rewritten as a VOID row (status `void`,
               the harness's status kept as `status_harness`, `void_class` + `void_reason` stated) -- it never enters a
               reading. Exit 0 admitted, 1 VOID (rewritten), 2 the harness's own non-ok row (oom / refused / alarm /
               harness_error / ... -- left as written), 3 no readable receipt (the caller writes the harness_error stub).
`footprint` -- after the p41c probe: the family's footprint row (PREREG "p41c"): fit_status in P40's vocabulary, allocator
               peak, nvidia-smi max from the lane's sampler, seq/batch/offload/checkpointing. Feeds the footprint row only.
`verdict`   -- after the fetch, controller-side: the run's verdict computed from the PRE-REGISTERED criteria that the lane
               itself recorded, never from whether a process exited cleanly (e4b#495). A VOID row and a fired STOP are the
               pre-registration's own failure conditions ("STOP rules": a stop is reported, not worked around; "Validity
               rules": a VOID never enters a reading) -- a run in which either fired cannot be `pass`. Exit 0 pass, 1 fail,
               2 inconclusive, 3 invalid; the driver maps those onto its own exit status so no caller that derives a
               receipt's `result` from the exit code can read a fired criterion as a success.
Stdlib only; runs on the box's image python and in the repo's tests."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

VOID_CLASSES = (
    "steps",
    "tokens",
    "trainable",
    "attn4",
    "engagement",
    "c1",
)  # STOP-3 counts VOIDs per class per family
HARNESS_VOIDS = {"void_trainable": "trainable", "void_attn4": "attn4", "tokens_mismatch": "tokens", "c1_failed": "c1"}
STOP_RULES = ("STOP-1", "STOP-2", "STOP-3", "STOP-4", "STOP-5")  # PREREG "STOP rules"
RESULT_ENUM = ("pass", "fail", "inconclusive", "invalid")
VERDICT_EXIT = {"pass": 0, "fail": 1, "inconclusive": 2, "invalid": 3}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: str, obj) -> None:
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".admit-", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)



def rules(
    rec: dict, *, steps: int, tokens_sha: str, expect_trainable: int, n_layers: int, attn4_census: int, arm: str
) -> list[tuple[str, str]]:
    """Every failed rule as (class, reason); an empty list admits. `arm` is fused or reference (the fused-only rules)."""
    fails: list[tuple[str, str]] = []
    done = rec.get("step_ms")
    n_done = len(done) if isinstance(done, list) else None
    if rec.get("steps") != steps or n_done != steps:
        fails.append(("steps", f"step count {n_done} / registered {rec.get('steps')} != N={steps}"))
    got_sha = (rec.get("tokens") or {}).get("sha256")
    if not tokens_sha or got_sha != tokens_sha:
        fails.append(("tokens", f"tokens sha {str(got_sha)[:12]} != the cell's file {str(tokens_sha)[:12]}"))
    if rec.get("trainable_params") != expect_trainable or rec.get("trainable_mismatch") is not None:
        fails.append(
            (
                "trainable",
                f"trainable {rec.get('trainable_params')} != expected {expect_trainable} (mismatch={rec.get('trainable_mismatch')})",
            )
        )
    if rec.get("n_attn4") != attn4_census or rec.get("structural_expected_n_attn4") != attn4_census:
        fails.append(
            (
                "attn4",
                f"n_attn4 {rec.get('n_attn4')} / census {rec.get('structural_expected_n_attn4')} != registered census {attn4_census}",
            )
        )
    if arm == "fused":
        # P41 is an e4b-only lane. Its registered engagement evidence is the
        # patched-module count plus the per-step kernel-call floor. The former
        # banner check required an Unsloth-only console line and therefore
        # VOIDed every valid e4b fused receipt by construction (#494).
        if rec.get("n_patched") != n_layers:
            fails.append(("engagement", f"n_patched {rec.get('n_patched')} != n_layers {n_layers}"))
        kmin = rec.get("kernel_calls_per_step_min") or 0
        if kmin < 2 * n_layers:
            fails.append(("engagement", f"kernel_calls_per_step_min {kmin} < 2 x n_layers {2 * n_layers}"))
    if rec.get("C1_bit_exact") is not True:
        fails.append(("c1", f"C1 not bit-exact ({rec.get('C1_experts_changed')} frozen tensors changed)"))
    return fails


def admit(a) -> int:
    try:
        rec = json.load(open(a.receipt))
    except Exception as e:  # noqa: BLE001 -- any unreadable receipt is the same row
        print(f"ADMIT MISSING {os.path.basename(a.receipt)}: {type(e).__name__}: {e}")
        return 3
    st = str(rec.get("status"))
    if st in HARNESS_VOIDS:  # the harness already wrote a VOID of a known class; name the class for STOP-3
        rec.update({"admitted": False, "void_class": HARNESS_VOIDS[st], "admitted_at": _utc()})
        _write(a.receipt, rec)
        print(
            f"ADMIT VOID class={HARNESS_VOIDS[st]} {os.path.basename(a.receipt)}: harness status {st}: {str(rec.get('reason'))[:160]}"
        )
        return 1
    if st != "ok":
        rec.update({"admitted": False, "admitted_at": _utc()})
        _write(a.receipt, rec)
        print(f"ADMIT {st.upper()} {os.path.basename(a.receipt)}: the harness's own row, left as written")
        return 2
    fails = rules(
        rec,
        steps=a.steps,
        tokens_sha=a.tokens_sha,
        expect_trainable=a.expect_trainable,
        n_layers=a.n_layers,
        attn4_census=a.attn4_census,
        arm=a.arm,
    )
    if fails:
        rec.update(
            {
                "status": "void",
                "status_harness": st,
                "admitted": False,
                "void_class": fails[0][0],
                "void_reason": "; ".join(f"{c}: {r}" for c, r in fails)[:800],
                "admitted_at": _utc(),
            }
        )
        _write(a.receipt, rec)
        print(f"ADMIT VOID class={fails[0][0]} {os.path.basename(a.receipt)}: {rec['void_reason'][:200]}")
        return 1
    rec.update(
        {
            "admitted": True,
            "admitted_at": _utc(),
            "admission_rules": "P41-PREREG.md validity rules (steps, tokens sha, trainable, attn4 census, engagement, C1)",
        }
    )
    _write(a.receipt, rec)
    print(
        f"ADMIT OK {os.path.basename(a.receipt)}: s/step {rec.get('s_per_step_median_11plus')} peak {rec.get('peak_vram_gb')} GB held-out {rec.get('eval_loss_final')}"
    )
    return 0


FIT = {
    "ok": "OK",
    "oom": "OOM",
    "refused": "REFUSED",
    "alarm": "ALARM",
    "void": "VOID",
    "harness_error": "HARNESS_ERROR",
    "load_fault": "LOAD_FAULT",
    "not_run": "NOT_RUN",
}


def nvsmi_max_gb(path: str | None) -> float | None:
    """The lane's sampler lines: `<epoch> <memory.used MiB>, <util>, <power>`; the max memory.used in GB (1024-based)."""
    if not path or not os.path.exists(path):
        return None
    best = None
    for line in open(path, errors="replace"):
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        try:
            mib = float(parts[1].split(",")[0])
        except ValueError:
            continue
        best = mib if best is None or mib > best else best
    return round(best / 1024, 3) if best is not None else None


def footprint(a) -> int:
    try:
        rec = json.load(open(a.receipt))
    except Exception as e:  # noqa: BLE001
        rec = {"status": "harness_error", "reason": f"probe receipt unreadable: {type(e).__name__}: {e}"}
    st = str(rec.get("status", "harness_error"))
    row = {
        "row": "p41c footprint",
        "fam": a.fam,
        "probe": "p41c",
        "seq_probe": a.seq,
        "r": a.r,
        "batch": 1,
        "fit_status": FIT.get(st, st.upper()),
        "harness_status": st,
        "reason": rec.get("reason") or rec.get("void_reason"),
        "peak_vram_gb_allocator": rec.get("peak_vram_gb"),
        "peak_vram_gb_nvsmi": nvsmi_max_gb(a.vram),
        "offload_design": "expert-cpu" if rec.get("offload") else "none",
        "checkpointing_mode": rec.get("grad_ckpt") or "hf-nonreentrant",
        "tokens_per_step": (rec.get("tokens_per_step") or [None])[0]
        if isinstance(rec.get("tokens_per_step"), list)
        else rec.get("tokens_per_step"),
        "feeds": "the footprint row only -- fit, never speed; no ratio is derived (PREREG p41c)",
        "prereg": rec.get("prereg"),
        "written_by": "p41_admit.py footprint",
        "at": _utc(),
    }
    _write(a.out, row)
    print(
        f"FOOTPRINT {a.fam} seq {a.seq} r {a.r}: {row['fit_status']} allocator {row['peak_vram_gb_allocator']} GB nvidia-smi max {row['peak_vram_gb_nvsmi']} GB"
    )
    return 0


def budget(a) -> int:
    """STOP-4 (PREREG "STOP rules"): projected spend = (actual elapsed + the remaining cells AT THE PLANNING CURVE) x the rate;
    due when it exceeds 1.5 x the approved estimate. Never the alarm sum (the alarms carry x1.5 and 900 s each and would fire
    STOP-4 before the first arm at any honest estimate -- CEO HIGH-1 on e4b#466). Prints one line; exit 0 = due, 1 = not due."""
    proj = (a.elapsed + a.remaining) / 3600.0 * a.rate
    limit = 1.5 * a.est
    due = proj > limit
    print(
        f"STOP-4 {'DUE' if due else 'ok'}: projected ${proj:.2f} (elapsed {a.elapsed:.0f} s + remaining at the planning curve {a.remaining:.0f} s at ${a.rate}/h) vs 1.5 x estimate ${a.est} = ${limit:.2f}"
    )
    return 0 if due else 1


def is_void(rec: dict) -> bool:
    """A VOID row in the pre-registration's vocabulary: rewritten by `admit`, or the harness's own void status."""
    status, void_class = rec.get("status"), rec.get("void_class")
    return (status == "void" and void_class in VOID_CLASSES) or (
        status in HARNESS_VOIDS and HARNESS_VOIDS[status] == void_class
    )


def _stops_fired(root: str) -> tuple[list[str], list[str]]:
    """The STOP rules this run reports, from BOTH artifacts the lane writes -- the `STOPn` marker files and
    `stop_state.json` -- unioned, plus any problem that stops the criteria being read at all. A stop that can be
    seen in one artifact and not the other still counts as fired: the union fails closed."""
    problems: list[str] = []
    fired = {rule for rule in STOP_RULES if os.path.exists(os.path.join(root, rule.replace("-", "")))}
    path = os.path.join(root, "stop_state.json")
    if os.path.exists(path):
        try:
            with open(path) as fh:
                stops = json.load(fh).get("stops")
            if not isinstance(stops, list):
                raise ValueError("no `stops` list")
            for entry in stops:
                rule = entry.get("rule") if isinstance(entry, dict) else None
                if rule in STOP_RULES:
                    fired.add(rule)
                else:
                    problems.append(f"stop_state.json names an unregistered stop rule {rule!r}")
        except Exception as exc:  # noqa: BLE001 -- unreadable STOP evidence is `invalid`, never a silent pass
            problems.append(f"stop_state.json unreadable: {type(exc).__name__}: {exc}")
    return sorted(fired), problems


def _summary_criteria(root: str) -> tuple[dict, list[str]]:
    """What `summary.txt` -- the artifact a reader actually opens -- says the criteria did. `admit` writes
    `ADMIT OK` / `ADMIT VOID`; a FIRED stop is written by `stop_now` as `STOP-n: <reason>` (the advisory
    `STOP-4 ok:` / `STOP-4 DUE:` / `STOP-1 UNDECIDED:` lines are deliberately NOT that shape)."""
    path = os.path.join(root, "summary.txt")
    try:
        with open(path, errors="replace") as fh:
            lines = fh.read().splitlines()
    except Exception as exc:  # noqa: BLE001
        return {}, [f"summary.txt unreadable: {type(exc).__name__}: {exc}"]
    return {
        "admit_ok_lines": sum(line.startswith("ADMIT OK") for line in lines),
        "admit_void_lines": sum(line.startswith("ADMIT VOID") for line in lines),
        "stop_lines": sorted({rule for rule in STOP_RULES for line in lines if line.startswith(rule + ": ")}),
    }, []


def _verdict(a) -> int:
    """The run's verdict from the pre-registration's own criteria, not from a clean exit (e4b#495).

    `invalid` -- the criteria cannot be located, parsed or reconciled. Failing closed is the point: an
                 unreadable criterion is not a satisfied one.
    `fail`    -- a VOID row appeared. PREREG "Validity rules" / "STOP rules": a VOID is reported as a row and
                 never enters a reading, and R1 was closed `FAIL / STOP-1` on exactly this shape.
    `inconclusive` -- no VOID, but a STOP fired or no arm was admitted: the lane behaved, and there is still
                 nothing to read. A legitimate, valuable outcome -- and not a `pass`.
    `pass`    -- at least one admitted arm, no VOID row, no STOP fired, every criterion readable.
    """
    root, nonce, expected = a.run_dir, a.nonce, a.expected
    problems: list[str] = []
    reasons: list[str] = []
    admitted = void = other = actual = None

    records: list[dict] = []
    for path in sorted(glob.glob(os.path.join(root, "*_e4b_*.json"))):
        try:
            with open(path) as fh:
                records.append(json.load(fh))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"receipt unreadable {os.path.basename(path)}: {type(exc).__name__}: {exc}")
    if any(type(rec.get("admitted")) is not bool for rec in records):
        problems.append("a receipt carries no Boolean `admitted` classification")
    admitted = sum(rec.get("admitted") is True for rec in records)
    void = sum(is_void(rec) for rec in records)
    actual = len(records)
    other = actual - admitted - void

    manifest_path = os.path.join(root, f"P41_OUTCOME_COUNTS.{nonce}.json")
    try:
        with open(manifest_path) as fh:
            manifest = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        manifest = None
        problems.append(f"outcome manifest unreadable ({os.path.basename(manifest_path)}): {type(exc).__name__}: {exc}")
    if manifest is not None:
        observed = {"expected": expected, "actual": actual, "admitted": admitted, "void": void, "other": other}
        if any(type(manifest.get(key)) is not int for key in observed):
            problems.append("outcome counts are not exact integers")
        elif any(manifest[key] != observed[key] for key in observed):
            problems.append(f"outcome counts do not match fetched receipts: manifest {manifest} vs {observed}")
        if manifest.get("run_nonce") != nonce or manifest.get("gate_pass") is not True:
            problems.append("outcome manifest is not bound to this successful run")
    if expected <= 0:
        problems.append(f"the plan expected {expected} rows")
    elif actual != expected:
        problems.append(f"{actual} receipts fetched, the plan expected {expected}")

    stops, stop_problems = _stops_fired(root)
    problems.extend(stop_problems)
    summary, summary_problems = _summary_criteria(root)
    problems.extend(summary_problems)
    # the harm in e4b#495 is a criterion firing in `summary.txt` that the verdict never sees; a criterion
    # visible there and nowhere else is an unreconcilable record, not a pass.
    if summary and summary.get("admit_void_lines", 0) > 0 and void == 0:
        problems.append(f"summary.txt reports {summary['admit_void_lines']} ADMIT VOID line(s) but no receipt is VOID")
    if summary and set(summary.get("stop_lines", [])) - set(stops):
        problems.append(
            f"summary.txt reports {sorted(set(summary['stop_lines']) - set(stops))} that no STOP marker or stop_state.json entry carries"
        )

    if problems:
        result = "invalid"
        reasons = problems
    else:
        if void > 0:
            reasons.append(
                f"{void} VOID row(s) -- the pre-registration reports a VOID as a row and never reads it (PREREG 'Validity rules')"
            )
        if stops:
            reasons.append(f"{', '.join(stops)} fired -- a stop is reported, not worked around (PREREG 'STOP rules')")
        if admitted == 0:
            reasons.append("no arm was admitted -- there is no reading to pass")
        result = "fail" if void > 0 else ("inconclusive" if reasons else "pass")
        if not reasons:
            reasons.append(f"{admitted} admitted arm(s), no VOID row, no STOP fired")

    row = {
        "row": "p41 verdict",
        "verdict": result,
        "result_enum": list(RESULT_ENUM),
        "computed_from": "the pre-registered criteria the lane recorded (admitted arms, VOID rows, STOP rules) -- never a process exit code (e4b#495)",
        "run_nonce": nonce,
        "expected": expected,
        "actual": actual,
        "admitted": admitted,
        "void": void,
        "other": other,
        "stops_fired": stops,
        "summary_observed": summary or None,
        "reasons": reasons,
        "prereg": "bench/p41/P41-PREREG.md",
        "written_by": "p41_admit.py verdict",
        "at": _utc(),
    }
    if a.out:
        _write(a.out, row)
    print(f"outcome: {actual} rows: {admitted} admitted, {void} VOID, {other} other")
    print(f"VERDICT {result}: " + "; ".join(reasons))
    return VERDICT_EXIT[result]


def verdict(a) -> int:
    """A verdict that cannot be computed at all is `invalid` -- never `fail`, and never `pass`. Without this the
    tool's own uncaught exception would leave Python's generic exit 1, which the driver maps to `fail`: still not a
    success, but the wrong name for it, and this issue is about a run being given the wrong name (e4b#495)."""
    try:
        return _verdict(a)
    except Exception as exc:  # noqa: BLE001
        print(f"VERDICT invalid: the verdict could not be computed: {type(exc).__name__}: {exc}")
        return VERDICT_EXIT["invalid"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("admit")
    q.add_argument("receipt")
    q.add_argument("--steps", type=int, required=True)
    q.add_argument("--tokens-sha", required=True)
    q.add_argument("--expect-trainable", type=int, required=True)
    q.add_argument("--n-layers", type=int, required=True)
    q.add_argument("--attn4-census", type=int, required=True)
    q.add_argument("--arm", choices=("fused", "reference"), required=True)
    q.set_defaults(fn=admit)
    f = sub.add_parser("footprint")
    f.add_argument("receipt")
    f.add_argument("--fam", required=True)
    f.add_argument("--seq", type=int, required=True)
    f.add_argument("--r", type=int, default=8)
    f.add_argument("--vram", default=None)
    f.add_argument("--out", required=True)
    f.set_defaults(fn=footprint)
    b = sub.add_parser("budget")
    b.add_argument("--elapsed", type=float, required=True)
    b.add_argument("--remaining", type=float, required=True)
    b.add_argument("--rate", type=float, required=True)
    b.add_argument("--est", type=float, required=True)
    b.set_defaults(fn=budget)
    v = sub.add_parser("verdict")
    v.add_argument("run_dir")
    v.add_argument("--nonce", required=True)
    v.add_argument("--expected", type=int, required=True)
    v.add_argument("--out", default=None)
    v.set_defaults(fn=verdict)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
