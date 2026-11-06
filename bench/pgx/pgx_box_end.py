"""PGX Amendment 4: the end class of one reading box, read from its fetched run directory (CONTROLLER side).

    python bench/pgx/pgx_box_end.py RUN_DIR/pgx [--run-id RID] [--rc N]

A reading box's end is one of three classes (bench/pgx/PREREG-pgx.md, Amendment 4):

  PART_OK      (a) rc 0, with a part record that verifies and reads PART_OK. Not a strike.
  PART_FAULTS  (b) rc 27, with a part record that verifies and reads PART_FAULTS. Not a strike: its faulted cells are
                   VOID under the registered rules, and the next box may launch.
  STRIKE       (c) anything else after STARTED: no part record, one that does not verify, a crash, any other code.

"Verifies" means the record re-derives exactly: this module reruns the box's own part reduction
(``pgx_reduce.reduce(..., part=True)``) on the fetched ``read/`` directory with the box's own ``registration.json`` for
the box's registered draws, and the result must equal the record field for field. So the record covers every scheduled
lifetime of the box's draws, skipped ones included (a skipped lifetime reduces as missing), and it is the record of the
directory it came with. The box must also have run its registered schedule (``box_schedule.txt`` equals
``pgx_schedule.py --box K-i``).

A box that never STARTED reads NOT_STARTED: the registered pre-STARTED rules apply, not this module.

Amendment 4's second rule, the shakedown's "STOP and report" carried into the reading: ``no_serve`` lists every
(arm, budget) of the box at which no lifetime served (each of that arm's scheduled lifetimes at that budget recorded a
startup failure, or completed no request in any pass). A non-empty list stops the lane for a person before the next box.
A lifetime skipped for time is not a serve failure.

Exit: 0 PART_OK; 10 PART_FAULTS; 11 PART_FAULTS with no_serve (stop for a person); 12 STRIKE; 13 NOT_STARTED. The last
stdout line is the result as JSON. Any other exit (an unreadable directory raises) is not a class: the caller stops for a
person. Stdlib only; tested by tests/test_pgx_box_end.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pgx_reduce as red  # noqa: E402
import pgx_schedule as sch  # noqa: E402

PART_OK, PART_FAULTS, STRIKE, NOT_STARTED = "PART_OK", "PART_FAULTS", "STRIKE", "NOT_STARTED"
EXIT = {PART_OK: 0, PART_FAULTS: 10, STRIKE: 12, NOT_STARTED: 13}
EXIT_NO_SERVE = 11
PREFIX = "PGX"
NONCE_RE = re.compile(r"[0-9a-f]{16}")                # the driver's nonce form


def _text(path):
    """A file's stripped text, or None when it is absent. Any other read error propagates (the caller stops)."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def _present(path) -> bool:
    """True if the path exists. Only FileNotFoundError means absent: any other stat error (PermissionError, ...) propagates,
    since absence of STARTED is the permissive outcome (NOT_STARTED) and must never be inferred from an unreadable path."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def _json(path):
    """(object, None), or (None, why) when the file is absent or not JSON."""
    t = _text(path)
    if t is None:
        return None, f"{os.path.basename(path)} missing"
    try:
        return json.loads(t), None
    except ValueError as exc:
        return None, f"{os.path.basename(path)} is not JSON ({exc.__class__.__name__})"


def schedule_lines(spec: str) -> list:
    """``pgx_schedule.py --box spec``'s output lines (the file the box writes as box_schedule.txt)."""
    return [f"{r['index']} {r['draw']} {r['kind']} {r['arm']} {r['budget']} {','.join(r['passes'])}"
            for r in sch.box_rows(spec)]


def _served(lt_dir: str, lt: dict) -> bool:
    """False when the lifetime recorded a startup failure, or no pass of it completed a single request."""
    if lt.get("startup_failed"):
        return False
    for p in ("P1", "P2", "P3", "P4"):
        rec, _ = _json(os.path.join(lt_dir, f"{p}.json"))
        if isinstance(rec, dict) and isinstance(rec.get("n"), int) and (rec.get("faults") or 0) < rec["n"]:
            return True
    return False


def no_serve(read_dir: str, draws) -> list:
    """[[arm, budget]] at which every scheduled lifetime of the box's draws ran and did not serve. A lifetime with no
    record (skipped for time) is not a serve failure, so an (arm, budget) with any such lifetime is not listed."""
    lts = sorted({(a, b) for a, b, _, _ in red.schedule(tuple(draws))}, key=str)
    out = []
    for arm, b in lts:
        failed = []
        for d in draws:
            lt_dir = os.path.join(read_dir, arm, str(b), f"d{d}")
            lt, _ = _json(os.path.join(lt_dir, "lifetime.json"))
            failed.append(isinstance(lt, dict) and not _served(lt_dir, lt))
        if failed and all(failed):
            out.append([arm, b])
    return out


def classify(box_dir: str, run_id: str | None = None, rc: int | None = None) -> dict:
    out = {"class": STRIKE, "why": [], "run_id": None, "rc": None, "spec": None, "draws": None, "no_serve": [],
           "stop_for_person": False}
    why = out["why"]
    if not _present(os.path.join(box_dir, "STARTED")):
        out["class"] = NOT_STARTED
        why.append("no STARTED: the registered pre-STARTED rules apply")
        return out
    # the run id and the box it names
    rid = _text(os.path.join(box_dir, "RUN_ID"))
    out["run_id"] = rid
    if rid is None:
        why.append("RUN_ID missing")
        return out
    if run_id is not None and rid != run_id:
        why.append(f"RUN_ID {rid} is not the launched id {run_id}")
        return out
    try:
        k, i, _n = sch.parse_run_id(rid)
    except ValueError as exc:
        why.append(str(exc))
        return out
    if k != sch.PLAN_K:
        why.append(f"K {k} is not the registered plan's K = {sch.PLAN_K}")
        return out
    spec = f"{k}-{i}"
    draws = sorted({r["draw"] for r in sch.box_rows(spec)})
    out.update(spec=spec, draws=draws)
    # the box's own exit code, for its own nonce
    nonce = _text(os.path.join(box_dir, f"{PREFIX}_RUN_NONCE"))
    if nonce is None or not NONCE_RE.fullmatch(nonce):       # read from the box (untrusted): validated before any path use
        why.append(f"{PREFIX}_RUN_NONCE {'missing' if nonce is None else 'malformed'}")
        return out
    code = _text(os.path.join(box_dir, f"{PREFIX}_EXIT_CODE.{nonce}"))
    if code is None or not code.isdigit():
        why.append(f"no readable {PREFIX}_EXIT_CODE for the box's nonce")
        return out
    code = int(code)
    out["rc"] = code
    if rc is not None and rc != code:
        why.append(f"the box's exit code {code} is not the code read by the caller {rc}")
        return out
    if code not in (0, 27):
        why.append(f"rc {code}: only 0 and 27 can end a reading box without a strike")
        return out
    if code == 0 and not os.path.exists(os.path.join(box_dir, f"{PREFIX}_SUCCESS.{nonce}")):
        why.append("rc 0 without the nonce's success marker")
        return out
    # the registered schedule ran
    got = _text(os.path.join(box_dir, "box_schedule.txt"))
    if got is None or got.splitlines() != schedule_lines(spec):
        why.append(f"box_schedule.txt {'missing' if got is None else 'differs'} from pgx_schedule.py --box {spec}")
        return out
    # the part record, re-derived from the box's own directory
    rec, err = _json(os.path.join(box_dir, "part_verdict.json"))
    reg, err2 = _json(os.path.join(box_dir, "registration.json"))
    read_dir = os.path.join(box_dir, "read")
    if err or err2 or not os.path.isdir(read_dir):
        why += [e for e in (err, err2) if e] + ([] if os.path.isdir(read_dir) else ["read/ missing"])
        return out
    try:
        want = json.loads(json.dumps(red.reduce([read_dir], reg, draws=tuple(draws), part=True), default=str))
    except (ValueError, KeyError, TypeError, AttributeError, FileNotFoundError, NotADirectoryError) as exc:
        why.append(f"the part reduction does not rerun on the fetched directory ({exc.__class__.__name__}: {exc})")
        return out
    if rec != want:
        diff = (sorted(k for k in set(rec) | set(want) if rec.get(k) != want.get(k)) if isinstance(rec, dict)
                else ["<not an object>"])
        why.append(f"part_verdict.json does not verify against read/ (differs in {', '.join(diff)})")
        return out
    verdict = rec["verdict"]
    if (code, verdict) == (0, "PART_OK"):
        out["class"] = PART_OK
    elif (code, verdict) == (27, "PART_FAULTS"):
        out["class"] = PART_FAULTS
        out["no_serve"] = no_serve(read_dir, draws)
        out["stop_for_person"] = bool(out["no_serve"])
    else:
        why.append(f"rc {code} with a {verdict} record")
        return out
    why.append(f"rc {code} with a verified {verdict} record over draws {draws}")
    return out


def exit_code(res: dict) -> int:
    if res["class"] == PART_FAULTS and res["stop_for_person"]:
        return EXIT_NO_SERVE
    return EXIT[res["class"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("box_dir")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--rc", type=int, default=None)
    a = ap.parse_args(argv)
    res = classify(a.box_dir, a.run_id, a.rc)
    print(f"BOX_END {res['class']} {res['run_id']} rc {res['rc']}: {'; '.join(res['why'])}"
          + (f"; NO SERVE at {res['no_serve']}: stop for a person" if res["stop_for_person"] else ""))
    print(json.dumps(res, sort_keys=True))
    return exit_code(res)


if __name__ == "__main__":
    sys.exit(main())
