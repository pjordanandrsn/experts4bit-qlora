#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2_qualify.py -- SC2f Amendment 7's host gate, qualify-at-proof (SC2h Amendment 5 and SC2i Amendment 4 take it by
reference): which machine a lane may pin, and whether the rented card qualifies. Nothing here creates, destroys or
approves compute; the launcher (``tools/pod-launch.sh``) creates, and applies its own power exclusions at create.

  admit --store <receipts root> --lane sc2f|sc2h|sc2i --reads READS.json [--out FILE]   exit 0 ADMIT, 1 NONE, 2 PARK
      The pin admission, the amendment's text as code:
      - the walk is OPEN; prior clean runs only ORDER it (SC2f Amendment 7, as ruled 2026-10-25). Its tiers, in order:
        host continuity first (SC2h, then SC2i, put the earlier lanes' PROVED host first); then 52780, then 145701
        (ruled first, from the recount); then ANY other machine whose offers list 575 W, by prior clean runs
        descending, then machine id; then machines whose listed power is unknown (55337 among them), in the same
        order, LAST. A machine whose offers list any power other than 575 W is never admitted. Listed power is an upper
        bound, never the gate: the enforced read on the card is;
      - excluded: 13828 (clock-bound) and 37367 (a 400 W cap in its clocks); 26157 until a create there has
        succeeded (a receipt on 26157 with an instance), since it answered every create with ``no_such_ask``;
      - ATTEMPTS: every create on the lane since the amendment (a receipt for ``<lane>-prove-<n>``, n >= the lane's
        first Amendment 7 id, whose instance was created) consumes one of 3, a qualify refusal included; a refusal
        before any create (no instance) consumes none. A machine already tried is never admitted again, whatever its
        outcome, unless its proof PROVED (status OK, result pass): that machine is then the lane's host, and only it
        is admitted. With 3 attempts consumed and no proof PROVED, the lane PARKS;
      - STABILITY: a machine is admitted only when an offer on it at or under the $1.05/h ceiling with at least
        320 GB of disk is present in each of the last 3 reads, by offer id (``READS.json``: a list of reads, oldest
        first, each a list of ``{"machine_id", "offer_id", "usd_h", "disk_gb", "listed_power_w"}``;
        ``listed_power_w`` null when the listing carries none). ``--clean-runs FILE`` (``{machine: n}``, from the
        launcher's receipt ranking) supplies the order; a machine absent from it has 0. The first admissible machine
        in walk order is admitted.
  power --csv FILE [--out FILE]                                              exit 0 PASS, 17 REFUSE, 25 UNREAD
      ``nvidia-smi --query-gpu=enforced.power.limit,power.limit,power.max_limit --format=csv,noheader`` on the
      card (``575.00 W, 575.00 W, 600.00 W``): PASS iff the enforced limit parses to 575.0 exactly (compared as a
      float, never as text). All three are recorded: a card can enforce 575 W and cap elsewhere, which the clock burn
      and the run sampler catch. A row READ at another limit, or more than one GPU, is a host fault (rc 17); nothing
      readable (no row, ``[N/A]``) is probe_error, a harness fault (rc 25).
  judge-proof --forensics sc1/forensics.txt --host-facts host_facts.json [--out FILE]     exit 0 PASS, 18 REFUSE, 25 UNREAD
      After a CLEAN proof (the driver exits 0; any other exit keeps its own code and is never read here), the proof's
      own host records: the box's ``forensics.txt`` line ``power.limit,clocks.max.sm <W> W, <MHz> MHz`` and the
      launcher's ``host_facts.json`` ``gpu_power_limit_w`` must each read 575.0 W. Both flags are always passed. A
      record READ at another limit is a host fault (rc 18); a record missing, unreadable or without its field is
      probe_error, a HARNESS fault (rc 25) that voids the run and never excludes the machine.
  judge-run --sampler sampler.csv [--out FILE]                                       exit 0 PASS, 18 REFUSE, 25 UNREAD
      The controller-side sampler over the whole timed run (``nvidia-smi --query-gpu=timestamp,clocks.sm,
      clocks_event_reasons.active,power.draw --format=csv -lms 1000``, started on the box before the lane's driver and
      fetched after it), judged by ``bench/common/card_clock.py``'s own ``passes(..., mode="clock")`` (TC1 amendment 83,
      SC2f Amendment 9) at a floor of 0 MHz (``SERVING_JUDGE_FLOOR_MHZ``): a hardware or thermal slowdown on ANY sample
      (idle ones included) refuses, Reliability refuses only on >= 95 % of the BUSY samples (SM above card_clock's
      1000 MHz, so idle install and download time cannot dilute it), and SwPowerCap is recorded, never refusing. The
      clock is NOT ruled here: a serving run's busy SM median sits under the 5090 floor (2,415 MHz over the store's 51
      serving snapshots), so the pre-install burn at 2700 is the only clock screen; the median and the SwPowerCap and
      Reliability shares are recorded. Fewer than 10 busy samples means the sampler never saw the load: probe_error
      (rc 25). A sampler that READ and failed is below_floor, a host fault
      (rc 18). A missing or unreadable file (no header, fewer than 10 readable rows, or more than 5 % of rows
      unreadable) is probe_error, a HARNESS fault (rc 25): it voids the run and never excludes the machine.
  --self-test

Exit 17 (power cap) and 18 (host floor) are the launcher's host-limited codes: it records a host fault, never a
strike, and excludes the machine from its next draws.
"""
import argparse
import json
import re
import sys
from pathlib import Path

NAMED = ("52780", "145701")                       # ruled first, from the recount; the walk is otherwise open
EXCLUDED = ("13828", "37367")
UNTIL_CREATE = ("26157",)
MAX_ATTEMPTS = 3
CEILING_USD_H = 1.05
MIN_DISK_GB = 320
STABLE_READS = 3
ENFORCED_W = 575.0
FIRST_ID = {"sc2f": 11, "sc2h": 1, "sc2i": 1}      # the first <lane>-prove-<n> under Amendment 7 (see each PREREG)
LANE_ORDER = ("sc2f", "sc2h", "sc2i")             # SC2f Amendment 3: the three run in this order and share a host
LISTED_W = 575.0
SAMPLER_MIN_ROWS = 10
SAMPLER_MIN_BUSY = 10                             # card_clock.passes' min_busy
SAMPLER_MAX_BAD = 0.05
#: SC2f Amendment 9: the run judge passes card_clock's clock rule a floor of 0 MHz, so the clock never binds here (a serving run's
#: busy SM median sits under the 5090's 2700 floor on healthy cards: 2,415 MHz over the store's 51 serving snapshots) while the
#: bit clauses and min_busy still do. The card's clock is screened once, by the pre-install burn at 2700 in clock mode.
SERVING_JUDGE_FLOOR_MHZ = 0
EXIT_ADMIT, EXIT_NONE, EXIT_PARK, EXIT_POWER, EXIT_FLOOR, EXIT_UNREAD = 0, 1, 2, 17, 18, 25


def _created(r: dict) -> bool:
    return str(r.get("instance_id") or "").strip().lower() not in ("", "none", "null")


def lane_attempts(store: Path, lane: str) -> list:
    """Every Amendment 7 proof create on the lane, from the receipt store: ``[{run_id, machine, created, proved}]``,
    in run-id order. A ``<run>-refused-*`` directory (X2: refused before any create) is read as not created."""
    pat = re.compile(rf"^{re.escape(lane)}-prove-(\d+)(?:-refused-.*)?$")
    out = []
    for rec in sorted(Path(store).glob("*/*/receipt.json")):
        m = pat.match(rec.parent.name)
        if not m or int(m.group(1)) < FIRST_ID[lane]:
            continue
        try:
            r = json.loads(rec.read_text())
        except (OSError, ValueError):
            continue
        created = _created(r)
        out.append({"run_id": rec.parent.name, "n": int(m.group(1)),
                    "machine": str((r.get("environment") or {}).get("vast_machine_id") or ""),
                    "created": created, "proved": created and r.get("status") == "OK" and r.get("result") == "pass"})
    return sorted(out, key=lambda a: (a["n"], a["run_id"]))


def creates_on(store: Path, machine: str) -> bool:
    """Has any create on ``machine`` succeeded, in any lane (a receipt on it with an instance)?"""
    for rec in Path(store).glob("*/*/receipt.json"):
        try:
            r = json.loads(rec.read_text())
        except (OSError, ValueError):
            continue
        if str((r.get("environment") or {}).get("vast_machine_id") or "") == machine and _created(r):
            return True
    return False


def earlier_hosts(store: Path, lane: str) -> list:
    """The PROVED hosts of the lanes before ``lane`` in LANE_ORDER, latest lane first (host continuity)."""
    out = []
    for prev in reversed(LANE_ORDER[:LANE_ORDER.index(lane)]):
        for a in lane_attempts(store, prev):
            if a["proved"] and a["machine"] and a["machine"] not in out:
                out.append(a["machine"])
    return out


def _stable(machine: str, reads: list) -> list:
    """The offers on ``machine`` that pass the ceiling and disk in EACH of the last STABLE_READS reads (by offer id)."""
    if len(reads) < STABLE_READS:
        return []
    keep = None
    for rd in reads[-STABLE_READS:]:
        ok = {str(o.get("offer_id")) for o in rd if str(o.get("machine_id")) == machine
              and isinstance(o.get("usd_h"), (int, float)) and o["usd_h"] <= CEILING_USD_H
              and isinstance(o.get("disk_gb"), (int, float)) and o["disk_gb"] >= MIN_DISK_GB}
        keep = ok if keep is None else keep & ok
    return sorted(keep or ())


def _listed(machine: str, reads: list):
    """``575`` when every listed power on the machine's offers in the last reads that carry one is 575 W, ``None`` when
    none carries one, else the other value(s): such a machine is never admitted."""
    vals = set()
    for rd in reads[-STABLE_READS:]:
        for o in rd:
            if str(o.get("machine_id")) == machine:
                v = o.get("listed_power_w")
                vals.add(float(v) if isinstance(v, (int, float)) else None)
    known = vals - {None}
    if not known:
        return None
    return LISTED_W if known == {LISTED_W} else sorted(known)


def _id_key(m: str):
    return (0, int(m), m) if m.isdigit() else (1, 0, m)


def walk_order(reads: list, clean_runs=None, prefer=()) -> list:
    """The walk: continuity first, then NAMED, then listed-575 W machines by clean runs (desc) and id, then listed-power
    unknown machines in the same order. Machines listing another power are left out (``walk_rejects`` says why)."""
    clean_runs = {str(k): int(v) for k, v in (clean_runs or {}).items()}
    seen = sorted({str(o.get("machine_id")) for rd in reads[-STABLE_READS:] for o in rd if o.get("machine_id") is not None},
                  key=_id_key)
    by_runs = lambda ms: sorted(ms, key=lambda m: (-clean_runs.get(m, 0), _id_key(m)))  # noqa: E731
    rest = [m for m in seen if m not in NAMED and m not in prefer]
    listed = by_runs(m for m in rest if _listed(m, reads) == LISTED_W)
    unknown = by_runs(m for m in rest if _listed(m, reads) is None)
    return list(dict.fromkeys([*prefer, *NAMED, *listed, *unknown]))


def admit(attempts: list, reads: list, created_26157: bool, prefer=(), clean_runs=None) -> dict:
    created = [a for a in attempts if a["created"]]
    proved = [a for a in created if a["proved"]]
    tried = {a["machine"] for a in created}
    rec = {"attempts_used": len(created), "attempts_max": MAX_ATTEMPTS, "tried": sorted(tried),
           "proved": proved[0]["machine"] if proved else None, "skipped": {}}
    if proved:
        walk = [proved[0]["machine"]]                 # the lane's host: only it, for the reading and any re-run
    elif len(created) >= MAX_ATTEMPTS:
        return dict(rec, verdict="PARK", why=f"{len(created)} creates and no proof PROVED")
    else:
        walk = walk_order(reads, clean_runs, prefer)
    rec["walk"] = walk
    for m in walk:
        if m in EXCLUDED:
            rec["skipped"][m] = "excluded"
            continue
        if m in UNTIL_CREATE and not created_26157:
            rec["skipped"][m] = "excluded until a create there succeeds"
            continue
        if not proved and m in tried:
            rec["skipped"][m] = "already tried: never revisited"
            continue
        lp = _listed(m, reads)
        if lp not in (LISTED_W, None):
            rec["skipped"][m] = f"listed power {lp} W, not {LISTED_W:.0f} W"
            continue
        offers = _stable(m, reads)
        if offers:
            return dict(rec, verdict="ADMIT", machine=m, offers=offers, listed_power_w=lp)
        rec["skipped"][m] = f"no offer <= ${CEILING_USD_H}/h with >= {MIN_DISK_GB} GB in each of the last {STABLE_READS} reads"
    return dict(rec, verdict="NONE", why="no walk machine admissible on these reads")


_W = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*W\s*$")


def power(csv_text: str) -> dict:
    """PASS, REFUSE (read: an enforced limit other than 575 W, or more than one GPU; a host fault, rc 17) or UNREAD
    (nothing readable: probe_error, a harness fault, rc 25, never excluding the machine)."""
    rows = [ln for ln in (csv_text or "").strip().splitlines() if ln.strip()]
    if not rows:
        return {"verdict": "UNREAD", "refusal": "probe_error", "why": "no nvidia-smi row", "raw": csv_text}
    if len(rows) != 1:
        return {"verdict": "REFUSE", "refusal": "below_floor", "why": f"{len(rows)} GPU rows, expected 1", "raw": csv_text}
    cells = [c.strip() for c in rows[0].split(",")]
    vals = [(float(m.group(1)) if (m := _W.match(c)) else None) for c in cells]
    rec = {"raw": rows[0], "enforced_w": vals[0] if vals else None,
           "power_limit_w": vals[1] if len(vals) > 1 else None, "power_max_limit_w": vals[2] if len(vals) > 2 else None}
    if len(cells) != 3 or any(v is None for v in vals):
        return dict(rec, verdict="UNREAD", refusal="probe_error", why="unparseable: expected 3 cells of '<number> W'")
    if vals[0] != ENFORCED_W:
        return dict(rec, verdict="REFUSE", refusal="below_floor", why=f"enforced power limit {vals[0]} W, not {ENFORCED_W} W")
    return dict(rec, verdict="PASS", refusal=None, why="")


_FPL = re.compile(r"^power\.limit,clocks\.max\.sm\s+([0-9]+(?:\.[0-9]+)?)\s*W", re.M)


def judge_proof(forensics_text, host_facts) -> dict:
    """PASS, REFUSE (both records READ and one shows a power limit other than 575 W: a host fault, rc 18) or UNREAD
    (a record missing, unreadable or without its power field: probe_error, a harness fault, rc 25). ``forensics_text``
    or ``host_facts`` None means the file is missing; ``host_facts`` must be a dict to count as read."""
    unread, wrong, rec = [], [], {}
    m = _FPL.search(forensics_text) if isinstance(forensics_text, str) else None
    rec["forensics_power_limit_w"] = float(m.group(1)) if m else None
    if m is None:
        unread.append("forensics.txt missing or without its power.limit line")
    elif rec["forensics_power_limit_w"] != ENFORCED_W:
        wrong.append(f"forensics.txt power.limit {rec['forensics_power_limit_w']} W, not {ENFORCED_W} W")
    hf = None
    if isinstance(host_facts, dict):
        try:
            hf = float(host_facts.get("gpu_power_limit_w"))
        except (TypeError, ValueError):
            hf = None
    rec["host_facts_power_limit_w"] = hf
    if hf is None:
        unread.append("host_facts.json missing, unreadable or without gpu_power_limit_w")
    elif hf != ENFORCED_W:
        wrong.append(f"host_facts.json gpu_power_limit_w {hf} W, not {ENFORCED_W} W")
    if wrong:
        return dict(rec, verdict="REFUSE", refusal="below_floor", why=wrong + unread)
    if unread:
        return dict(rec, verdict="UNREAD", refusal="probe_error", why=unread)
    return dict(rec, verdict="PASS", refusal=None, why=[])


def _card_clock():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "common" / "card_clock.py"
    spec = importlib.util.spec_from_file_location("card_clock", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def judge_run(text) -> dict:
    """The controller-side sampler (csv with header) over the whole run, by card_clock's bits: PASS, REFUSE (read,
    below_floor: a host fault) or UNREAD (probe_error: a harness fault)."""
    cc = _card_clock()
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines or "clocks_event_reasons.active" not in lines[0]:
        return {"verdict": "UNREAD", "refusal": "probe_error", "why": ["no sampler header"], "rows": 0}
    head = [h.strip().split(" ")[0] for h in lines[0].split(",")]
    try:
        ri, ci = head.index("clocks_event_reasons.active"), head.index("clocks.sm")
    except ValueError:
        return {"verdict": "UNREAD", "refusal": "probe_error", "why": [f"sampler header {lines[0][:120]!r}"], "rows": 0}
    masks, bad = [], 0
    for ln in lines[1:]:
        cells = [c.strip() for c in ln.split(",")]
        try:
            masks.append((int(cells[ci].split()[0]), int(cells[ri], 16)))
        except (ValueError, IndexError):
            bad += 1
    rows = len(masks) + bad
    rec = {"rows": rows, "unreadable_rows": bad}
    if len(masks) < SAMPLER_MIN_ROWS or (rows and bad / rows > SAMPLER_MAX_BAD):
        return dict(rec, verdict="UNREAD", refusal="probe_error",
                    why=[f"{len(masks)} readable rows of {rows} (need >= {SAMPLER_MIN_ROWS}, <= {SAMPLER_MAX_BAD:.0%} unreadable)"])
    summ = cc.summarize([(c, 0, 0, m) for c, m in masks])   # card_clock's own summary: its busy samples, counts and any-sample bits
    rec.update(busy_samples=summ["busy_samples"], sm_max=summ["sm_max"], busy_sm_median=summ["sm_median"],   # reported, not ruled
               any_sample_bits_seen=summ["any_sample_bits_seen"], busy_reliability_samples=summ["busy_reliability_samples"],
               busy_reliability_fraction=summ["busy_reliability_fraction"], busy_swpowercap_fraction=summ["busy_swpowercap_fraction"])
    if summ["busy_samples"] < SAMPLER_MIN_BUSY:
        return dict(rec, verdict="UNREAD", refusal="probe_error",
                    why=[f"{summ['busy_samples']} busy samples (SM > {cc.BUSY_MHZ} MHz) < {SAMPLER_MIN_BUSY}: the sampler never saw the load"])
    ok, why = cc.passes(summ, SERVING_JUDGE_FLOOR_MHZ, min_busy=SAMPLER_MIN_BUSY, mode="clock")   # TC1 am83's rule, never copied
    return dict(rec, verdict="PASS" if ok else "REFUSE", refusal=None if ok else "below_floor", why=[why] if why else [])


def self_test() -> int:
    a = lambda n, m, created=True, proved=False: {"run_id": f"sc2f-prove-{n}", "n": n, "machine": m,  # noqa: E731
                                                     "created": created, "proved": proved}
    fx = "GPU 0: NVIDIA GeForce RTX 5090\npower.limit,clocks.max.sm 575.00 W, 3105 MHz\n"
    def rd(*offers):     # (machine, offer, usd_h, disk_gb, listed_power_w)
        return [{"machine_id": m, "offer_id": o, "usd_h": p, "disk_gb": d, "listed_power_w": w} for m, o, p, d, w in offers]
    named = [rd(("52780", "1", 0.9, 400, 575.0), ("145701", "2", 0.95, 400, 575.0))] * 3
    hdr = "timestamp, clocks.sm [MHz], clocks_event_reasons.active, power.draw [W]\n"
    row = lambda r="0x0000000000000000", mhz=2800: f"2026/10/26 10:00:00.000, {mhz} MHz, {r}, 540.00 W\n"  # noqa: E731
    cases = [
        ("admit: 52780 first on stable reads", admit([], named, False).get("machine") == "52780"),
        ("admit: 52780 tried -> 145701", admit([a(11, "52780")], named, False).get("machine") == "145701"),
        ("admit: a refusal consumes an attempt", admit([a(11, "52780")], named, False)["attempts_used"] == 1),
        ("admit: a pre-create refusal consumes nothing", admit([a(11, "52780", created=False)], named, False)
         .get("machine") == "52780"),
        ("admit: 3 creates and no PROVED -> PARK", admit([a(11, "52780"), a(12, "145701"), a(13, "9")], named, False)
         ["verdict"] == "PARK"),
        ("admit: PROVED pins that host only", admit([a(11, "145701", proved=True)], named, False).get("machine")
         == "145701" and admit([a(11, "145701", proved=True)], [rd(("52780", "1", 0.9, 400, 575.0))] * 3, False)
         ["verdict"] == "NONE"),
        ("admit: the walk is OPEN: an unnamed 575 W machine is admitted",
         admit([], [rd(("777", "7", 0.9, 400, 575.0))] * 3, False).get("machine") == "777"),
        ("admit: a machine appearing mid-screen is admitted once stable",
         admit([], [rd(("52780", "1", 1.2, 400, 575.0))] * 2 + [rd(("888", "8", 0.9, 400, 575.0))] * 3, False)
         .get("machine") == "888"),
        ("admit: ... and not before its third read",
         admit([], [rd()] + [rd(("888", "8", 0.9, 400, 575.0))] * 2, False)["verdict"] == "NONE"),
        ("admit: 575 W machines by clean runs, then id", admit([], [rd(("300", "a", 0.9, 400, 575.0),
                                                                        ("200", "b", 0.9, 400, 575.0),
                                                                        ("100", "c", 0.9, 400, 575.0))] * 3, False,
                                                                clean_runs={"300": 5, "200": 5, "100": 1})
         .get("machine") == "200"),
        ("admit: unknown listed power comes LAST", admit([], [rd(("100", "c", 0.9, 400, None),
                                                                  ("900", "d", 0.9, 400, 575.0))] * 3, False,
                                                          clean_runs={"100": 9}).get("machine") == "900"),
        ("admit: unknown listed power is admissible when nothing else is",
         admit([], [rd(("100", "c", 0.9, 400, None))] * 3, False).get("machine") == "100"),
        ("admit: another listed power is never admitted",
         admit([], [rd(("19317", "x", 0.6, 400, 450.0))] * 3, False)["verdict"] == "NONE"),
        ("admit: exclusions bind in the open walk too",
         admit([], [rd(("13828", "x", 0.5, 900, 575.0), ("37367", "y", 0.5, 900, 575.0),
                       ("26157", "z", 0.5, 900, 575.0))] * 3, False)["verdict"] == "NONE"),
        ("admit: 26157 after a create there", admit([], [rd(("26157", "z", 0.5, 900, 575.0))] * 3, True)
         .get("machine") == "26157"),
        ("admit: the LAST 3 reads decide", admit([], [rd()] + named, False).get("machine") == "52780"),
        ("admit: 2 of the last 3 reads is not stable", admit([], [rd()] + named[:2], False)["verdict"] == "NONE"
         and admit([], named[:2] + [rd()], False)["verdict"] == "NONE" and admit([], named[:2], False)["verdict"] == "NONE"),
        ("admit: over the ceiling (the all-in usd_h) is not admissible",
         admit([], [rd(("52780", "1", 1.06, 400, 575.0))] * 3, False)["verdict"] == "NONE"),
        ("admit: under 320 GB disk is not admissible", admit([], [rd(("52780", "1", 0.9, 319, 575.0))] * 3, False)
         ["verdict"] == "NONE"),
        ("admit: an offer must persist by id", admit([], [rd(("52780", "1", 0.9, 400, 575.0)),
                                                         rd(("52780", "2", 0.9, 400, 575.0)),
                                                         rd(("52780", "1", 0.9, 400, 575.0))], False)["verdict"] == "NONE"),
        ("admit: continuity puts the earlier lane's host first",
         admit([], [rd(("52780", "1", 0.9, 400, 575.0), ("777", "7", 0.9, 400, 575.0))] * 3, False,
               prefer=("777",)).get("machine") == "777"),
        ("power: 575.00 W enforced passes", power("575.00 W, 575.00 W, 600.00 W\n")["verdict"] == "PASS"),
        ("power: 575 W (no decimals) passes as a float", power("575 W, 575 W, 600 W")["verdict"] == "PASS"),
        ("power: 574.99 W refuses", power("574.99 W, 575.00 W, 600.00 W")["verdict"] == "REFUSE"),
        ("power: 400 W refuses", power("400.00 W, 575.00 W, 600.00 W")["verdict"] == "REFUSE"),
        ("power: two GPUs refuse", power("575.00 W, 575.00 W, 600.00 W\n575.00 W, 575.00 W, 600.00 W")["verdict"]
         == "REFUSE"),
        ("power: [N/A] is UNREAD (harness)", power("[N/A], 575.00 W, 600.00 W")["verdict"] == "UNREAD"),
        ("power: no row is UNREAD (harness)", power("")["verdict"] == "UNREAD"),
        ("power: all three recorded", power("575.00 W, 575.00 W, 600.00 W")["power_max_limit_w"] == 600.0),
        ("judge: both records at 575 pass", judge_proof(fx, {"gpu_power_limit_w": 575.0})["verdict"] == "PASS"),
        ("judge: forensics READ at 500 W refuses", judge_proof(fx.replace("575.00 W", "500.00 W"),
                                                               {"gpu_power_limit_w": 575.0})["verdict"] == "REFUSE"),
        ("judge: host facts READ at 450 W refuse", judge_proof(fx, {"gpu_power_limit_w": 450.0})["verdict"] == "REFUSE"),
        ("judge: host facts missing is UNREAD", judge_proof(fx, None)["verdict"] == "UNREAD"),
        ("judge: host facts unreadable is UNREAD", judge_proof(fx, "unreadable")["verdict"] == "UNREAD"),
        ("judge: host facts without the field is UNREAD", judge_proof(fx, {})["verdict"] == "UNREAD"),
        ("judge: forensics missing is UNREAD", judge_proof(None, {"gpu_power_limit_w": 575.0})["verdict"] == "UNREAD"),
        ("judge: no power line is UNREAD", judge_proof("GPU 0\n", {"gpu_power_limit_w": 575.0})["verdict"] == "UNREAD"),
        ("judge: a READ wrong record outranks a missing one", judge_proof(fx.replace("575.00 W", "500.00 W"), None)
         ["verdict"] == "REFUSE"),
        ("run: a clean sampler passes", judge_run(hdr + row() * 30)["verdict"] == "PASS"),
        ("run: one HwSlowdown sample refuses", judge_run(hdr + row() * 29 + row("0x0000000000000008"))["verdict"]
         == "REFUSE"),
        # SC2f Amendment 9: TC1 amendment 83's clock rule at a 0 MHz floor -- Reliability refuses only when persistent, SwPowerCap
        # is recorded, the clock is never ruled here
        ("run: one Reliability sample passes (recorded)", judge_run(hdr + row() * 29 + row("0x0000000000000400"))["verdict"]
         == "PASS"),
        ("run: Reliability on 95 % of busy samples refuses (persistent)",
         judge_run(hdr + row() + row("0x0000000000000400") * 19)["verdict"] == "REFUSE"),
        ("run: Reliability on 94 of 100 busy samples passes",
         judge_run(hdr + row() * 6 + row("0x0000000000000400") * 94)["verdict"] == "PASS"),
        ("run: SwPowerCap on every busy sample passes (recorded, never refusing)",
         judge_run(hdr + row("0x0000000000000004") * 30)["verdict"] == "PASS"),
        ("run: a serving-shaped run at a 2,400 MHz busy median with clean bits passes (the clock is not ruled here)",
         judge_run(hdr + row(mhz=2400) * 30)["verdict"] == "PASS" and SERVING_JUDGE_FLOOR_MHZ == 0),
        ("run: GpuIdle is no refusal", judge_run(hdr + row("0x0000000000000001", 210) * 30 + row() * 10)["verdict"]
         == "PASS"),
        ("run: half idle plus Reliability on 95 % of busy samples refuses (idle never dilutes)",
         judge_run(hdr + row("0x0000000000000001", 210) * 100 + row() * 5 + row("0x0000000000000400") * 95)["verdict"]
         == "REFUSE"),
        ("run: the record carries am83's shares", judge_run(hdr + row() * 20 + row("0x0000000000000004") * 10)
         .get("busy_swpowercap_fraction") == round(10 / 30, 4)),
        ("run: a hard bit on an IDLE sample still refuses", judge_run(hdr + row("0x0000000000000008", 210) + row() * 30)
         ["verdict"] == "REFUSE"),
        ("run: fewer than 10 busy samples is UNREAD", judge_run(hdr + row(mhz=210) * 40 + row() * 9)["verdict"] == "UNREAD"),
        ("run: a missing file is UNREAD", judge_run(None)["verdict"] == "UNREAD"),
        ("run: no header is UNREAD", judge_run(row() * 30)["verdict"] == "UNREAD"),
        ("run: 9 rows is UNREAD", judge_run(hdr + row() * 9)["verdict"] == "UNREAD"),
        ("run: >5 % unreadable rows is UNREAD", judge_run(hdr + row() * 18 + "garbage\n" * 2)["verdict"] == "UNREAD"),
    ]
    bad = [n for n, ok in cases if not ok]
    for n, ok in cases:
        print(f"SC2_QUALIFY self-test {'ok ' if ok else 'BAD'} {n}")
    print(f"SC2_QUALIFY self-test {'OK' if not bad else 'FAILED'} ({len(cases) - len(bad)}/{len(cases)})")
    return 0 if not bad else 1


def _emit(tag: str, rec: dict, out) -> None:
    if out:
        with open(out, "w") as f:
            json.dump(rec, f, indent=1, sort_keys=True)
    print(f"{tag} " + json.dumps(rec, sort_keys=True), flush=True)


def _read_text(path):
    try:
        return Path(path).read_text(errors="replace")
    except (OSError, TypeError):
        return None


def _code(rec: dict, refuse: int) -> int:
    return {"PASS": EXIT_ADMIT, "REFUSE": refuse}.get(rec["verdict"], EXIT_UNREAD)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", nargs="?", choices=("admit", "power", "judge-proof", "judge-run"))
    ap.add_argument("--store")
    ap.add_argument("--lane", choices=sorted(FIRST_ID))
    ap.add_argument("--reads")
    ap.add_argument("--clean-runs")
    ap.add_argument("--csv")
    ap.add_argument("--forensics")
    ap.add_argument("--host-facts")
    ap.add_argument("--sampler")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.cmd == "admit":
        if not (a.store and a.lane and a.reads):
            ap.error("admit needs --store, --lane and --reads")
        cr = json.loads(Path(a.clean_runs).read_text()) if a.clean_runs else {}
        rec = admit(lane_attempts(Path(a.store), a.lane), json.loads(Path(a.reads).read_text()),
                    creates_on(Path(a.store), UNTIL_CREATE[0]), prefer=earlier_hosts(Path(a.store), a.lane), clean_runs=cr)
        rec["lane"] = a.lane
        _emit("SC2_QUALIFY_ADMIT", rec, a.out)
        return {"ADMIT": EXIT_ADMIT, "NONE": EXIT_NONE}.get(rec["verdict"], EXIT_PARK)
    if a.cmd == "power":
        if not a.csv:
            ap.error("power needs --csv")
        rec = power(_read_text(a.csv) or "")
        _emit("SC2_QUALIFY_POWER", rec, a.out)
        return _code(rec, EXIT_POWER)
    if a.cmd == "judge-proof":
        if not (a.forensics and a.host_facts):
            ap.error("judge-proof needs --forensics and --host-facts (both, always)")
        ht = _read_text(a.host_facts)
        try:
            hf = json.loads(ht) if ht is not None else None
        except ValueError:
            hf = "unreadable"
        rec = judge_proof(_read_text(a.forensics), hf)
        _emit("SC2_QUALIFY_PROOF", rec, a.out)
        return _code(rec, EXIT_FLOOR)
    if a.cmd == "judge-run":
        if not a.sampler:
            ap.error("judge-run needs --sampler")
        rec = judge_run(_read_text(a.sampler))
        _emit("SC2_QUALIFY_RUN", rec, a.out)
        return _code(rec, EXIT_FLOOR)
    ap.error("nothing to do")
    return 2


if __name__ == "__main__":
    sys.exit(main())
