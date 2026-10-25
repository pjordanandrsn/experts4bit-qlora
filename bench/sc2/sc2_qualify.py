#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2_qualify.py -- SC2f Amendment 7's host gate, qualify-at-proof (SC2h Amendment 5 and SC2i Amendment 4 take it by
reference): which machine a lane may pin, and whether the rented card qualifies. Nothing here creates, destroys or
approves compute; the launcher (``tools/pod-launch.sh``) creates, and applies its own power exclusions at create.

  admit --store <receipts root> --lane sc2f|sc2h|sc2i --reads READS.json [--out FILE]   exit 0 ADMIT, 1 NONE, 2 PARK
      The pin admission, the amendment's text as code:
      - the walk, in order: 52780, then 145701, then 55337 (last: its listed power is unknown, so it is outside the
        "2 qualifying" count, and only the enforced read on the card can admit it). Host continuity (SC2f Amendment 3):
        SC2h, then SC2i, put the earlier lanes' PROVED host (SC2f's, then SC2h's) first, ahead of the walk;
      - excluded: 13828 (clock-bound) and 37367 (a 400 W cap in its clocks); 26157 until a create there has
        succeeded (a receipt on 26157 with an instance), since it answered every create with ``no_such_ask``;
      - ATTEMPTS: every create on the lane since the amendment (a receipt for ``<lane>-prove-<n>``, n >= the lane's
        first Amendment 7 id, whose instance was created) consumes one of 3, a qualify refusal included; a refusal
        before any create (no instance) consumes none. A machine already tried is never admitted again, whatever its
        outcome, unless its proof PROVED (status OK, result pass): that machine is then the lane's host, and only it
        is admitted. With 3 attempts consumed and no proof PROVED, the lane PARKS;
      - STABILITY: a machine is admitted only when an offer on it at or under the $1.05/h ceiling with at least
        320 GB of disk is present in each of the last 3 reads (``READS.json``: a list of reads, oldest first, each a
        list of ``{"machine_id", "offer_id", "usd_h", "disk_gb"}``). The first walk machine that passes is admitted.
  power --csv FILE [--out FILE]                                                       exit 0 PASS, 17 REFUSE
      ``nvidia-smi --query-gpu=enforced.power.limit,power.limit,power.max_limit --format=csv,noheader`` on the
      card (``575.00 W, 575.00 W, 600.00 W``): PASS iff the enforced limit parses to 575.0 exactly (compared as a
      float, never as text). All three are recorded: a card can enforce 575 W and cap elsewhere, which the clock burn
      catches. One GPU only; anything unparseable refuses.
  judge-proof --forensics sc1/forensics.txt [--host-facts host_facts.json] [--out FILE]   exit 0 PASS, 18 REFUSE
      After a CLEAN proof (the driver exits 0; any other exit keeps its own code and is never read here), the proof's
      own host records: the box's ``forensics.txt`` line ``power.limit,clocks.max.sm <W> W, <MHz> MHz`` must read
      575.0 W, and the launcher's ``host_facts.json`` (when present) ``gpu_power_limit_w`` must read 575.0 too. An
      SC1 proof records no per-block clock samples, so the bar's "no capped block" term is carried by the pre-install
      clock burn on the same card (``card_clock.py``: no hard reason bit on any sample, SwPowerCap clear on >= 95 %).
  --self-test

Exit 17 (power cap) and 18 (host floor) are the launcher's host-limited codes: it records a host fault, never a
strike, and excludes the machine from its next draws.
"""
import argparse
import json
import re
import sys
from pathlib import Path

WALK = ("52780", "145701", "55337")
EXCLUDED = ("13828", "37367")
UNTIL_CREATE = ("26157",)
MAX_ATTEMPTS = 3
CEILING_USD_H = 1.05
MIN_DISK_GB = 320
STABLE_READS = 3
ENFORCED_W = 575.0
PIN_FLOOR_W = 560.0
FIRST_ID = {"sc2f": 11, "sc2h": 1, "sc2i": 1}      # the first <lane>-prove-<n> under Amendment 7 (see each PREREG)
LANE_ORDER = ("sc2f", "sc2h", "sc2i")             # SC2f Amendment 3: the three run in this order and share a host
EXIT_ADMIT, EXIT_NONE, EXIT_PARK, EXIT_POWER, EXIT_FLOOR = 0, 1, 2, 17, 18


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


def earlier_hosts(store: Path, lane: str) -> list:
    """The PROVED hosts of the lanes before ``lane`` in LANE_ORDER, latest lane first (host continuity)."""
    out = []
    for prev in reversed(LANE_ORDER[:LANE_ORDER.index(lane)]):
        for a in lane_attempts(store, prev):
            if a["proved"] and a["machine"] and a["machine"] not in out:
                out.append(a["machine"])
    return out


def admit(attempts: list, reads: list, created_26157: bool, prefer=()) -> dict:
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
        walk = list(dict.fromkeys([*prefer, *WALK]))   # the earlier lanes' host first, then the walk, each once
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
        offers = _stable(m, reads)
        if offers:
            return dict(rec, verdict="ADMIT", machine=m, offers=offers)
        rec["skipped"][m] = f"no offer <= ${CEILING_USD_H}/h with >= {MIN_DISK_GB} GB in each of the last {STABLE_READS} reads"
    return dict(rec, verdict="NONE", why="no walk machine admissible on these reads")


_W = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*W\s*$")


def power(csv_text: str) -> dict:
    rows = [ln for ln in csv_text.strip().splitlines() if ln.strip()]
    if len(rows) != 1:
        return {"verdict": "REFUSE", "why": f"{len(rows)} GPU rows, expected 1", "raw": csv_text}
    cells = [c.strip() for c in rows[0].split(",")]
    vals = [(float(m.group(1)) if (m := _W.match(c)) else None) for c in cells]
    rec = {"raw": rows[0], "enforced_w": vals[0] if vals else None,
           "power_limit_w": vals[1] if len(vals) > 1 else None, "power_max_limit_w": vals[2] if len(vals) > 2 else None}
    if len(cells) != 3 or any(v is None for v in vals):
        return dict(rec, verdict="REFUSE", why="unparseable: expected 3 cells of '<number> W'")
    if vals[0] != ENFORCED_W:
        return dict(rec, verdict="REFUSE", why=f"enforced power limit {vals[0]} W, not {ENFORCED_W} W")
    return dict(rec, verdict="PASS", why="")


_FPL = re.compile(r"^power\.limit,clocks\.max\.sm\s+([0-9]+(?:\.[0-9]+)?)\s*W", re.M)


def judge_proof(forensics_text, host_facts=None) -> dict:
    why, rec = [], {}
    m = _FPL.search(forensics_text or "")
    rec["forensics_power_limit_w"] = float(m.group(1)) if m else None
    if rec["forensics_power_limit_w"] != ENFORCED_W:
        why.append(f"forensics.txt power.limit {rec['forensics_power_limit_w']!r}, not {ENFORCED_W} W")
    if host_facts is not None:
        try:
            hf = float(host_facts.get("gpu_power_limit_w"))
        except (TypeError, ValueError, AttributeError):
            hf = None
        rec["host_facts_power_limit_w"] = hf
        if hf != ENFORCED_W:
            why.append(f"host_facts.json gpu_power_limit_w {hf!r}, not {ENFORCED_W} W")
    return dict(rec, verdict="PASS" if not why else "REFUSE", why=why)


def _rd(*offers):
    return [{"machine_id": m, "offer_id": o, "usd_h": p, "disk_gb": d} for m, o, p, d in offers]


def self_test() -> int:
    ok_reads = [_rd(("52780", "1", 0.9, 400), ("145701", "2", 0.95, 400))] * 3
    a = lambda n, m, created=True, proved=False: {"run_id": f"sc2f-prove-{n}", "n": n, "machine": m,  # noqa: E731
                                                     "created": created, "proved": proved}
    fx = "GPU 0: NVIDIA GeForce RTX 5090\npower.limit,clocks.max.sm 575.00 W, 3105 MHz\n"
    cases = [
        ("admit: the first walk machine on stable reads", admit([], ok_reads, False).get("machine") == "52780"),
        ("admit: 52780 tried -> 145701", admit([a(11, "52780")], ok_reads, False).get("machine") == "145701"),
        ("admit: a qualify refusal consumes an attempt and the host", admit([a(11, "52780")], ok_reads, False)
         ["attempts_used"] == 1),
        ("admit: a pre-create refusal consumes nothing", admit([a(11, "52780", created=False)], ok_reads, False)
         .get("machine") == "52780"),
        ("admit: 3 creates and no PROVED -> PARK", admit([a(11, "52780"), a(12, "145701"), a(13, "55337")], ok_reads,
                                                         False)["verdict"] == "PARK"),
        ("admit: PROVED pins that host only", admit([a(11, "145701", proved=True)], ok_reads, False).get("machine")
         == "145701" and admit([a(11, "145701", proved=True)], [_rd(("52780", "1", 0.9, 400))] * 3, False)
         ["verdict"] == "NONE"),
        ("admit: the LAST 3 reads decide", admit([], [_rd()] + ok_reads, False).get("machine") == "52780"),
        ("admit: 2 of the last 3 reads is not stable", admit([], [_rd()] + ok_reads[:2], False)["verdict"] == "NONE"
         and admit([], ok_reads[:2] + [_rd()], False)["verdict"] == "NONE" and admit([], ok_reads[:2], False)
         ["verdict"] == "NONE"),
        ("admit: over the ceiling is not admissible", admit([], [_rd(("52780", "1", 1.06, 400))] * 3, False)
         ["verdict"] == "NONE"),
        ("admit: under 320 GB disk is not admissible", admit([], [_rd(("52780", "1", 0.9, 319))] * 3, False)
         ["verdict"] == "NONE"),
        ("admit: an offer must persist by id", admit([], [_rd(("52780", "1", 0.9, 400)), _rd(("52780", "2", 0.9, 400)),
                                                         _rd(("52780", "1", 0.9, 400))], False)["verdict"] == "NONE"),
        ("admit: off-walk machines are never admitted", admit([], [_rd(("99999", "9", 0.5, 900))] * 3, False)
         ["verdict"] == "NONE"),
        ("admit: 55337 last", admit([a(11, "52780"), a(12, "145701")], [_rd(("55337", "5", 0.9, 400))] * 3, False)
         .get("machine") == "55337"),
        ("admit: continuity puts the earlier lane's host first", admit([], [_rd(("52780", "1", 0.9, 400),
                                                                           ("777", "7", 0.9, 400))] * 3, False,
                                                                      prefer=("777",)).get("machine") == "777"),
        ("admit: an earlier lane's host that is unlisted falls to the walk", admit([], ok_reads, False, prefer=("777",))
         .get("machine") == "52780"),
        ("admit: an earlier lane's host is still excluded if excluded", admit([], [_rd(("13828", "1", 0.5, 900))] * 3,
                                                                              False, prefer=("13828",))["verdict"] == "NONE"),
        ("power: 575.00 W enforced passes", power("575.00 W, 575.00 W, 600.00 W\n")["verdict"] == "PASS"),
        ("power: 575 W (no decimals) passes as a float", power("575 W, 575 W, 600 W")["verdict"] == "PASS"),
        ("power: 574.99 W refuses", power("574.99 W, 575.00 W, 600.00 W")["verdict"] == "REFUSE"),
        ("power: 400 W enforced refuses", power("400.00 W, 575.00 W, 600.00 W")["verdict"] == "REFUSE"),
        ("power: [N/A] refuses", power("[N/A], 575.00 W, 600.00 W")["verdict"] == "REFUSE"),
        ("power: two GPUs refuse", power("575.00 W, 575.00 W, 600.00 W\n575.00 W, 575.00 W, 600.00 W")["verdict"]
         == "REFUSE"),
        ("power: all three recorded", power("575.00 W, 575.00 W, 600.00 W")["power_max_limit_w"] == 600.0),
        ("judge: 575.00 W in forensics passes", judge_proof(fx)["verdict"] == "PASS"),
        ("judge: forensics and host facts at 575 pass", judge_proof(fx, {"gpu_power_limit_w": 575.0})["verdict"] == "PASS"),
        ("judge: 500 W in forensics refuses", judge_proof(fx.replace("575.00 W", "500.00 W"))["verdict"] == "REFUSE"),
        ("judge: host facts at 500 W refuse", judge_proof(fx, {"gpu_power_limit_w": 500.0})["verdict"] == "REFUSE"),
        ("judge: host facts without the field refuse", judge_proof(fx, {})["verdict"] == "REFUSE"),
        ("judge: no power line refuses", judge_proof("GPU 0: NVIDIA GeForce RTX 5090\n")["verdict"] == "REFUSE"),
        ("judge: an empty record refuses", judge_proof("")["verdict"] == "REFUSE"),
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", nargs="?", choices=("admit", "power", "judge-proof"))
    ap.add_argument("--store")
    ap.add_argument("--lane", choices=sorted(FIRST_ID))
    ap.add_argument("--reads")
    ap.add_argument("--csv")
    ap.add_argument("--forensics")
    ap.add_argument("--host-facts")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.cmd == "admit":
        if not (a.store and a.lane and a.reads):
            ap.error("admit needs --store, --lane and --reads")
        rec = admit(lane_attempts(Path(a.store), a.lane), json.loads(Path(a.reads).read_text()),
                    creates_on(Path(a.store), UNTIL_CREATE[0]), prefer=earlier_hosts(Path(a.store), a.lane))
        rec["lane"] = a.lane
        _emit("SC2_QUALIFY_ADMIT", rec, a.out)
        return {"ADMIT": EXIT_ADMIT, "NONE": EXIT_NONE}.get(rec["verdict"], EXIT_PARK)
    if a.cmd == "power":
        if not a.csv:
            ap.error("power needs --csv")
        try:
            text = Path(a.csv).read_text()
        except OSError as e:
            text = f"<unreadable: {e}>"
        rec = power(text)
        _emit("SC2_QUALIFY_POWER", rec, a.out)
        return EXIT_ADMIT if rec["verdict"] == "PASS" else EXIT_POWER
    if a.cmd == "judge-proof":
        if not a.forensics:
            ap.error("judge-proof needs --forensics")
        try:
            text = Path(a.forensics).read_text(errors="replace")
        except OSError:
            text = ""
        hf = None
        if a.host_facts:
            try:
                hf = json.loads(Path(a.host_facts).read_text())
            except (OSError, ValueError):
                hf = {}
        rec = judge_proof(text, hf)
        _emit("SC2_QUALIFY_PROOF", rec, a.out)
        return EXIT_ADMIT if rec["verdict"] == "PASS" else EXIT_FLOOR
    ap.error("nothing to do")
    return 2


if __name__ == "__main__":
    sys.exit(main())
