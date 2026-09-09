#!/usr/bin/env python3
"""reduce_support.py -- turn probe rows into the support table, and check the claim.

Two jobs, deliberately in one file because they read the same rows:

**Report** (default) prints a markdown table from ``bench/support/rows/*.json``.
``docs/ARCHITECTURE_SUPPORT.md`` is hand-maintained and dated 2026-08-12; the
reason it aged silently is that nothing regenerates it. This does.

**Check** (``--check``) compares ``SUPPORTED_ARCHITECTURES`` against the rows and
fails when a family is claimed with *less* evidence than the committed baseline
records. It is deliberately a **regression** gate, not an absolute one: six of
the nine claimed families have no reference-tier row today, so an absolute gate
would fail on arrival and get switched off. Gating the delta holds the line
against a *new* unevidenced claim while the existing gap is worked down, and it
puts the gap in the tree as data instead of in a doc nobody re-runs.

Evidence ranking, strongest first -- the ranking is the opinion in this file, so
it is written down rather than implied:

  reference-ok  loads, verifies and forwards on a real released checkpoint
  toy-ok        same, on a published checkpoint too small to be representative
  blocked       no published checkpoint can exercise it (below the blocksize)
  refused       the loader declines it, with a reason
  error         a fault
  copy-broken   a shard was short of its declared length: says nothing at all
                about the family, only that the transfer failed
  host-limited  the probe was killed before it finished (OOM): says the HOST
                could not complete the attempt, not that the family fails
  none          no row at all -- the state that reads exactly like validated

`reference-ok` on CPU is full evidence for the loader claim and none at all for
capture or throughput. The table says which device produced each row for that
reason.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROWS = Path(__file__).resolve().parent / "rows"
BASELINE = Path(__file__).resolve().parent / "coverage-baseline.json"

# "copy-broken" sits just above "none" because it carries NO information about
# the family -- a shard short of its declared length says the transfer failed.
# Grading it as "error" would file a transfer fault as a support fault, which is
# the exact misreading this probe exists to prevent.
# Two grades sit between "no row" and any family verdict, because neither says
# anything about the family: `copy-broken` is a failed transfer, `host-limited`
# is a probe the OS killed (an OOM on the largest checkpoints). Both must
# outrank `none` -- an attempt was made and the outcome is known -- and both must
# rank BELOW `error` and `refused`, which are statements about the code.
RANK = ["none", "copy-broken", "host-limited", "error", "refused", "blocked",
        "toy-ok", "reference-ok"]


def _grade(row: dict) -> str:
    st = row.get("stages", {})
    code = row.get("exit_code")
    if code == 5:
        return "copy-broken"
    if code == 9:
        return "host-limited"
    if code == 4:
        return "blocked"
    if code == 3:
        return "refused"
    if code != 0:
        return "error"
    if any(v.get("status") == "error" for v in st.values()):
        return "error"
    ok = all(st.get(s, {}).get("status") == "ok" for s in ("load", "verify", "forward"))
    if not ok:
        return "error"
    return "reference-ok" if row.get("tier") == "reference" else "toy-ok"


def _load_rows() -> list[dict]:
    out = []
    for p in sorted(ROWS.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception as e:                          # noqa: BLE001
            print(f"warning: {p.name} does not parse: {e}", file=sys.stderr)
            continue
        d["_file"] = p.name
        d["_grade"] = _grade(d)
        out.append(d)
    return out


def _best(rows: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for r in rows:
        mt = r.get("model_type")
        if not mt:
            continue
        cur = best.get(mt)
        if cur is None or RANK.index(r["_grade"]) > RANK.index(cur["_grade"]):
            best[mt] = r
    return best


def _claimed() -> dict[str, str]:
    sys.path.insert(0, str(ROOT))
    from experts4bit_qlora.loader import SUPPORTED_ARCHITECTURES
    return dict(SUPPORTED_ARCHITECTURES)


# Families whose model_type is not a shape any published checkpoint carries, so a
# checkpoint-based probe can only ever grade them `none`. That reads as
# "untested", which is the wrong conclusion, so the reason is printed instead of
# left to be inferred. These are NOT counted as evidence: sharing a convention
# record with an evidenced family is an argument, not a load.
NO_PUBLISHED_CHECKPOINT = {
    "gemma4_text": (
        "the text tower of a `gemma4` config -- `moe_conventions.py:328` gives ONE "
        "convention record (`GEMMA4`) for `{gemma4_text, gemma4}`, and its own comment "
        "calls `gemma4_text` \"what the loader constructs from `text_config`\". Every "
        "released gemma-4 (26B-A4B, 26B-A4B-it, 31B, E4B) reports `model_type: gemma4`; "
        "`Gemma4TextConfig.model_type` exists in transformers but Google ships "
        "`Gemma4Config`. The `gemma4` row therefore already exercised this record's "
        "`model.language_model.` -> `model.` rename -- without it the load would have "
        "found no experts at all -- and `test_gemma4_text_only_tree_strips_the_prefix_"
        "and_drops_the_vision_tower` covers the text-only tree. What is NOT covered is a "
        "real-weight load entered through a top-level `gemma4_text` config, which no "
        "download can supply."
    ),
}


def report(rows: list[dict]) -> None:
    claimed = _claimed()
    best = _best(rows)

    print("## Probe rows\n")
    print("| model_type | checkpoint | tier | route | device | load | verify | forward | grade |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r.get("model_type") or "", r["_file"])):
        st = r.get("stages", {})
        route = ("SUPPORTED_ARCHITECTURES" if r.get("claim", {}).get("in_supported_architectures")
                 else "convention" if r.get("claim", {}).get("reachable_by_convention") else "—")
        ver = st.get("verify", {})
        vtxt = (f"{ver.get('n_quantized')}q/{ver.get('n_unquantized')}u"
                if ver.get("status") == "ok" else ver.get("status", "—"))
        print(f"| `{r.get('model_type')}` | `{r.get('model')}` | {r.get('tier','—')} | {route} "
              f"| {r.get('device')} | {st.get('load',{}).get('status','—')} | {vtxt} "
              f"| {st.get('forward',{}).get('status','—')} | **{r['_grade']}** |")

    print("\n## Claimed vs evidenced\n")
    print(f"`SUPPORTED_ARCHITECTURES` claims **{len(claimed)}** families.\n")
    print("| claimed model_type | best evidence | checkpoint |")
    print("|---|---|---|")
    for mt in claimed:
        r = best.get(mt)
        print(f"| `{mt}` | **{r['_grade'] if r else 'none'}** | "
              f"{('`' + r['model'] + '`') if r else '—'} |")

    unclaimed = sorted(set(best) - set(claimed))
    if unclaimed:
        print("\nProbed but **not** in the claimed list (reachable by convention, or not "
              "supported at all — a passing row here is not a reason to claim it):\n")
        for mt in unclaimed:
            print(f"- `{mt}` — {best[mt]['_grade']} on `{best[mt]['model']}`")

    n_ref = sum(1 for mt in claimed if best.get(mt) and best[mt]["_grade"] == "reference-ok")
    print(f"\n**{n_ref} of {len(claimed)}** claimed families have a reference-tier passing row.")

    footnotes = [mt for mt in claimed
                 if mt in NO_PUBLISHED_CHECKPOINT and not best.get(mt)]
    if footnotes:
        print("\nWhy some rows above read `none` and always will:\n")
        for mt in footnotes:
            print(f"- **`{mt}`** — {NO_PUBLISHED_CHECKPOINT[mt]}")
        print("\nThese stay uncounted on purpose. The count tracks loads, not arguments.")


def check(rows: list[dict]) -> int:
    claimed = _claimed()
    best = _best(rows)
    current = {mt: (best[mt]["_grade"] if mt in best else "none") for mt in claimed}

    if not BASELINE.exists():
        BASELINE.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
        print(f"wrote initial baseline {BASELINE.name} with {len(current)} families")
        return 0

    baseline = json.loads(BASELINE.read_text())
    problems = []
    for mt, grade in current.items():
        was = baseline.get(mt)
        if was is None:
            problems.append(
                f"{mt!r} is newly claimed in SUPPORTED_ARCHITECTURES with evidence {grade!r} and "
                f"no baseline entry. Add a probe row, or record the gap by updating "
                f"{BASELINE.name} deliberately -- a new claim should not arrive silently.")
        elif RANK.index(grade) < RANK.index(was):
            problems.append(
                f"{mt!r} regressed: baseline {was!r}, now {grade!r}. Either the row got worse or "
                f"it disappeared; both mean the shipped claim is weaker than it was.")

    for mt in sorted(set(baseline) - set(current)):
        print(f"note: {mt!r} is in the baseline but no longer claimed -- prune it from "
              f"{BASELINE.name} when the removal is intentional.")

    if problems:
        print("support coverage check FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    improved = [mt for mt, g in current.items()
                if RANK.index(g) > RANK.index(baseline.get(mt, "none"))]
    if improved:
        print("coverage IMPROVED (update the baseline to lock it in): "
              + ", ".join(f"{mt} {baseline.get(mt,'none')}->{current[mt]}" for mt in improved))
    print(f"OK: {len(current)} claimed families, none weaker than baseline")
    return 0


DOC = ROOT / "docs" / "ARCHITECTURE_SUPPORT.md"
BEGIN = "<!-- BEGIN GENERATED: bench/support/reduce_support.py --write-doc -->"
END = "<!-- END GENERATED -->"


def _generated_block(rows: list[dict]) -> str:
    """The table, as the doc should contain it.

    Kept separate from report() so the doc block and the terminal report cannot
    disagree about what the rows say -- the test regenerates this and compares
    bytes, which is the whole mechanism that stops the doc ageing again.
    """
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(rows)
    body = buf.getvalue().strip()
    return (f"{BEGIN}\n\n"
            f"*Generated from `bench/support/rows/*.json`. Do not edit by hand — run\n"
            f"`python bench/support/reduce_support.py --write-doc`. The hand-written\n"
            f"sections above and below remain authoritative for everything these rows do\n"
            f"not cover, notably CUDA-graph capture and any throughput figure.*\n\n"
            f"{body}\n\n{END}")


def write_doc(rows: list[dict]) -> int:
    block = _generated_block(rows)
    text = DOC.read_text()
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        new_text = head + block + tail
    else:
        # First insertion: append rather than guess where it belongs. The doc's
        # hand-written taxonomy and evidence-scope sections stay untouched.
        new_text = text.rstrip() + "\n\n## Probe evidence (generated)\n\n" + block + "\n"
    if new_text == text:
        print(f"{DOC.relative_to(ROOT)} already current")
        return 0
    DOC.write_text(new_text)
    print(f"updated {DOC.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="fail if a claimed family has weaker evidence than the baseline")
    ap.add_argument("--write-doc", action="store_true",
                    help="refresh the generated block in docs/ARCHITECTURE_SUPPORT.md")
    args = ap.parse_args()
    rows = _load_rows()
    if not rows:
        print(f"no rows under {ROWS} -- nothing to reduce", file=sys.stderr)
        return 1
    if args.check:
        return check(rows)
    if args.write_doc:
        return write_doc(rows)
    report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
