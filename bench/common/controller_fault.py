#!/usr/bin/env python3
"""A controller-side fault: a rented box that the CONTROLLER's own guard destroyed while the controller's network to the provider
was down (PGX2 Amendment 1, P238 Amendment 7). Read from the box's receipt directory alone; stdlib only.

The record it is drawn from is ``pgx2-4090-r1-2-1`` (2026-11-07). A network outage at the controller stopped its launch
process's heartbeat. The controller-side guard read that as a dead controller and destroyed a healthy box mid-run, and
nothing was fetched. Such a box says nothing about the box, the host or the code under test, so a lane does not read it and
does not count it as a strike. It is not a STARTED box either (no lane directory was fetched, so whether it started is
unknown).

    python3 controller_fault.py RUN_DIR LANE   -> "CONTROLLER_FAULT ..." (exit 0) or "NOT_CONTROLLER_FAULT: <why>" (exit 1)

A box is a controller fault only when ALL of these hold, each read from the record (any unreadable file, any other shape:
NOT a controller fault, and the lane's own rules apply):

1. ``guard-firing.json``: the guard fired, and its ``reason`` is exactly ``heartbeat-loss``;
2. ``teardown-proof.guard.json``, the guard's own proof written AT the firing: its ``reason`` is ``heartbeat-loss`` and its
   evidence reads ``list_after`` ``"UNKNOWN"``: the guard's listing right after its destroy got no answer, so the controller
   could not reach the provider then. A heartbeat lost while the provider still answered (a stalled or crashed controller
   process) leaves ``list_after`` a list, and that is not a controller fault;
3. ``teardown-proof.json``, the final proof: ``complete`` is true and its evidence reads ``instance_absent`` true (the box
   is gone, so no spend continues);
4. ``receipt.json``: ``status`` is ``ALARM``;
5. no lane directory (``RUN_DIR/<LANE>``) was fetched: a box whose directory came back is read by the lane's own rules.

The guard's log is not read: its network-failure lines carry no time, and they also appear in runs with no outage.
"""
import json
import os
import sys


def _json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _evidence(proof):
    """A teardown proof's ``evidence`` as a dict (the proofs store it as a JSON string); anything else raises."""
    ev = proof["evidence"]
    ev = json.loads(ev) if isinstance(ev, str) else ev
    if not isinstance(ev, dict):
        raise TypeError("evidence is not an object")
    return ev


def controller_fault(run_dir, lane):
    """(True, summary) when the five conditions hold; else (False, why). Never raises on a malformed record."""
    try:
        firing = _json(os.path.join(run_dir, "guard-firing.json"))
        at_firing = _json(os.path.join(run_dir, "teardown-proof.guard.json"))
        final = _json(os.path.join(run_dir, "teardown-proof.json"))
        receipt = _json(os.path.join(run_dir, "receipt.json"))
    except (OSError, ValueError) as e:
        return False, f"the record is unreadable ({type(e).__name__})"
    why = []
    if not isinstance(firing, dict) or firing.get("reason") != "heartbeat-loss":
        why.append(f"the guard did not fire on heartbeat-loss ({firing.get('reason') if isinstance(firing, dict) else None!r})")
    try:
        unanswered = at_firing["reason"] == "heartbeat-loss" and _evidence(at_firing).get("list_after") == "UNKNOWN"
    except (KeyError, TypeError, ValueError):
        unanswered = False
    if not unanswered:
        why.append("the guard's own proof does not show the provider unreachable at the firing (reason heartbeat-loss, "
                   "list_after UNKNOWN)")
    try:
        absent = final["complete"] is True and _evidence(final).get("instance_absent") is True
    except (KeyError, TypeError, ValueError):
        absent = False
    if not absent:
        why.append("the final teardown proof does not show the instance absent")
    if not isinstance(receipt, dict) or receipt.get("status") != "ALARM":
        why.append(f"the receipt status is not ALARM ({receipt.get('status') if isinstance(receipt, dict) else None!r})")
    try:                                         # absence permits: FileNotFoundError alone means no lane directory
        os.lstat(os.path.join(run_dir, lane))
        why.append(f"a lane directory ({lane}/) was fetched: the lane's own rules read it")
    except FileNotFoundError:
        pass
    except OSError as e:
        why.append(f"the lane directory cannot be checked ({type(e).__name__})")
    if why:
        return False, "; ".join(why)
    return True, (f"guard fired heartbeat-loss at {firing.get('at')}; its listing after the destroy did not answer "
                  f"(proof at {at_firing.get('at')}); instance absent; receipt ALARM; nothing fetched")


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if len(a) != 2:
        print(__doc__.strip().splitlines()[0])
        return 2
    ok, why = controller_fault(a[0], a[1])
    print(("CONTROLLER_FAULT " if ok else "NOT_CONTROLLER_FAULT: ") + why)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
