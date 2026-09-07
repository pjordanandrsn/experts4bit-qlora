#!/usr/bin/env python3
"""bench/p41/p41_admit.py -- the lane's admission and footprint rules, executable (pre-registration: bench/p41/P41-PREREG.md).

`admit`     -- after an arm: the validity rules (PREREG "Validity rules", "Expected trainable counts", reading rule 2) applied
               to the harness's receipt. A receipt that trained but fails a rule is rewritten as a VOID row (status `void`,
               the harness's status kept as `status_harness`, `void_class` + `void_reason` stated) -- it never enters a
               reading. Exit 0 admitted, 1 VOID (rewritten), 2 the harness's own non-ok row (oom / refused / alarm /
               harness_error / ... -- left as written), 3 no readable receipt (the caller writes the harness_error stub).
`footprint` -- after the p41c probe: the family's footprint row (PREREG "p41c"): fit_status in P40's vocabulary, allocator
               peak, nvidia-smi max from the lane's sampler, seq/batch/offload/checkpointing. Feeds the footprint row only.
Stdlib only; runs on the box's image python and in the repo's tests."""

from __future__ import annotations

import argparse
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


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: str, obj) -> None:
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".admit-", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


E4B_FUSED_MARKER = "fused TRAINING path"   # what e4b's loader prints when the fused path engages (#494)


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
        # Engagement evidence is per FRAMEWORK, which is what P38-PREREG.md:72 says in terms: "e4b's arm shows
        # the same engagement as tp1 (enable_fast_train … patched, the kernel counter …); Unsloth's arm shows its
        # OWN banners (`Enabling LoRA on MoE parameters`, the `Unsloth: MoE bnb4bit …` backend line …)". The
        # banner is Unsloth's console line and e4b never prints it. Requiring it of e4b's fused arm voided the
        # only arm that ever reached admission (R1 attempt 10, 2026-09-07) — with n_patched 32 == n_layers, the
        # fused counter at 128 calls on all 60 steps, the exact trainable count and C1 bit-exact — and would void
        # every fused arm in the campaign by construction. The same cell in tp2, which IS in the register, has
        # engagement_banners [] with identical counters and was admitted. #494.
        framework = str(rec.get("framework") or "")
        if rec.get("n_patched") != n_layers:
            fails.append(("engagement", f"n_patched {rec.get('n_patched')} != n_layers {n_layers}"))
        kmin = rec.get("kernel_calls_per_step_min") or 0
        if kmin < 2 * n_layers:
            fails.append(("engagement", f"kernel_calls_per_step_min {kmin} < 2 x n_layers {2 * n_layers}"))
        if framework == "unsloth":
            banners = rec.get("engagement_banners") or []
            missing = [b for b in banners if str(b).startswith("NO '")]
            if not banners or missing:
                fails.append(("engagement", "engagement banner missing (a green skipped path is not evidence)"))
        else:
            # e4b's third piece of evidence, in place of the banner it cannot emit: the loader must SAY it took
            # the fused training path. A silent fallback to the reference path is exactly what the banner rule
            # was defending against, so this keeps the defence rather than dropping it.
            # Match the loader's own marker, not the word "fused": a reason reading "no fused kernel" contains
            # it. The observed string on an engaged arm is
            #   "[e4b.fast] fused TRAINING path on 32 ExpertsLoRA module(s) (dgrad kernel backward)"
            # and a reference arm records "" — so the marker separates them, and a negation cannot sneak past.
            why = str(rec.get("enable_reason") or "")
            if E4B_FUSED_MARKER not in why:
                fails.append(("engagement",
                              f"enable_reason does not carry {E4B_FUSED_MARKER!r} (a green skipped path is not "
                              f"evidence): {why[:120]!r}"))
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
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
