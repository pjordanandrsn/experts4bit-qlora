#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2_qualify.py -- SC2f Amendment 7's host gate, qualify-at-proof (SC2h Amendment 5 and SC2i Amendment 4 take it by
reference): which machine a lane may pin, and whether the rented card qualifies. Nothing here creates, destroys or
approves compute; the launcher (``tools/pod-launch.sh``) creates, and applies its own power exclusions at create.

  admit --store <receipts root> --lane sc2f|sc2h|sc2i --reads READS.json [--out FILE]   exit 0 ADMIT, 1 NONE, 2 PARK
      The pin admission, the amendment's text as code:
      - the walk is OPEN; prior clean runs only ORDER it (SC2f Amendment 7, as ruled 2026-10-25). Its tiers, in order:
        host continuity first (SC2h, then SC2i, put the earlier lanes' PROVED host first); then 52780, then 145701
        (ruled first, from the recount); then ANY other machine whose offers list 575 or 600 W (SC2f Amendment 10:
        one class), recorded-pass machines first, then unrecorded ones, each by prior clean runs descending, then
        machine id; then machines whose listed power is unknown (55337 among them), in the same order, LAST. A machine
        whose offers list any power outside {575, 600} W is never admitted. Listed power is an upper bound, never the
        gate: the enforced read on the card is;
      - the RECORDED CLOCK (SC2f Amendment 10, P238 amendment 2's rule): outside NAMED and continuity, a machine whose
        latest receipt with a loaded clock record (a run's ``gpuclk_*.txt``, or a pre-install burn's
        ``card_clock_burn.json`` / ``card_clock.json``, each with >= 10 busy samples) has any such record that FAILS
        ``card_clock.passes(summary, 2700, mode="clock")`` is never admitted. A store error other than a missing file
        makes the census UNREAD, and then only continuity and NAMED are walked (fail closed). Clean runs count host
        outcomes, never clocks, so the clean-runs order alone never excludes a recorded fail;
      - excluded: 13828 (clock-bound) and 37367 (a 400 W cap in its clocks); 26157 until a create there has
        succeeded (a receipt on 26157 with an instance), since it answered every create with ``no_such_ask``;
      - ATTEMPTS: every create on the lane since the amendment (a receipt for ``<lane>-prove-<n>``, n >= the lane's
        first Amendment 7 id, whose instance was created) consumes one of 3, a qualify refusal included; a refusal
        before any create (no instance) consumes none. A machine already tried is never admitted again, whatever its
        outcome, unless its proof PROVED (status OK, result pass): that machine is then the lane's host, and only it
        is admitted. With 3 attempts consumed and no proof PROVED, the lane PARKS;
      - STABILITY: a machine is admitted only when an offer on it at or under the $1.25/h ceiling (SC2f Amendment 11, one
        ceiling for SC2f, SC2h and SC2i) with at least
        320 GB of disk is present in each of the last 3 reads, by offer id (``READS.json``: a list of reads, oldest
        first, each a list of ``{"machine_id", "offer_id", "usd_h", "disk_gb", "listed_power_w"}``;
        ``listed_power_w`` null when the listing carries none). ``--clean-runs FILE`` (``{machine: n}``, from the
        launcher's receipt ranking) supplies the order; a machine absent from it has 0. The first admissible machine
        in walk order is admitted.
  power --csv FILE [--out FILE]                                              exit 0 PASS, 17 REFUSE, 25 UNREAD
      ``nvidia-smi --query-gpu=enforced.power.limit,power.limit,power.max_limit --format=csv,noheader`` on the
      card (``575.00 W, 575.00 W, 600.00 W``): PASS iff the enforced limit parses to 575.0 or 600.0 exactly (SC2f
      Amendment 10; compared as a float, never as text). All three are recorded: a card can enforce 575 W and cap elsewhere, which the clock burn
      and the run sampler catch. A row READ at another limit, or more than one GPU, is a host fault (rc 17); nothing
      readable (no row, ``[N/A]``) is probe_error, a harness fault (rc 25).
  judge-proof --forensics sc1/forensics.txt --host-facts host_facts.json --qualify-power qualify/power.json [--out FILE]
                                                                                     exit 0 PASS, 18 REFUSE, 25 UNREAD
      After a CLEAN proof (the driver exits 0; any other exit keeps its own code and is never read here), the proof's
      own host records: the box's ``forensics.txt`` line ``power.limit,clocks.max.sm <W> W, <MHz> MHz`` and the
      launcher's ``host_facts.json`` ``gpu_power_limit_w`` must each read 575.0 or 600.0 W AND equal the enforced limit
      the qualify step read on the card (its ``power.json``, SC2f Amendment 10). All three flags are always passed. A
      record READ at another limit, or at one that differs from the qualify reading, is a host fault (rc 18); a record
      missing, unreadable or without its field is probe_error, a HARNESS fault (rc 25) that voids the run (the create
      still uses an attempt and the host is not revisited, as ``admit`` reads the store).
  judge-run --sampler sampler.csv [--out FILE]                                       exit 0 PASS, 18 REFUSE, 25 UNREAD
      The controller-side sampler over the whole timed run (``nvidia-smi --query-gpu=timestamp,clocks.sm,
      clocks_event_reasons.active,power.draw --format=csv,noheader,nounits -lms 1000``, read by POSITION, SC2f Amendment 10),
      started on the box before the lane's driver and
      fetched after it), judged by ``bench/common/card_clock.py``'s own ``passes(..., mode="clock")`` (TC1 amendment 83,
      SC2f Amendment 9) at a floor of 0 MHz (``SERVING_JUDGE_FLOOR_MHZ``): a hardware or thermal slowdown on ANY sample
      (idle ones included) refuses, Reliability refuses only on >= 95 % of the BUSY samples (SM above card_clock's
      1000 MHz, so idle install and download time cannot dilute it), and SwPowerCap is recorded, never refusing. The
      clock is NOT ruled here: a serving run's busy SM median sits under the 5090 floor (2,415 MHz over the store's 51
      serving snapshots), so the pre-install burn at 2700 is the only clock screen; the median and the SwPowerCap and
      Reliability shares are recorded. Fewer than 10 busy samples means the sampler never saw the load: probe_error
      (rc 25). A sampler that READ and failed is below_floor, a host fault
      (rc 18). A missing or unreadable file (no header, fewer than 10 readable rows, or more than 5 % of rows
      unreadable) is probe_error, a HARNESS fault (rc 25): it voids the run. Every create still uses one of the lane's
      attempts and the host is not revisited, whatever the exit (``admit``, SC2f Amendment 7).
  --self-test

Exit 17 (power cap) and 18 (host floor) are the launcher's host-limited codes: it records a host fault, never a
strike, and excludes the machine from its next draws.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

NAMED = ("52780", "145701")                       # ruled first, from the recount; the walk is otherwise open
EXCLUDED = ("13828", "37367")
UNTIL_CREATE = ("26157",)
MAX_ATTEMPTS = 3
#: SC2f Amendment 11: the per-hour offer ceiling (the launcher's all-in hourly at 320 GB), one for SC2f, SC2h and SC2i ($1.05 before it).
#: Every run's worst stays <= $15 (the largest, SC2i's reading, $6.36); lane totals are stated, not capped.
CEILING_USD_H = 1.25
MIN_DISK_GB = 320
STABLE_READS = 3
#: SC2f Amendment 10: the listed-power class and the enforced limits that pass, one set (Amendment 7's 575 W alone before it)
POWER_W = (575.0, 600.0)
FIRST_ID = {"sc2f": 11, "sc2h": 1, "sc2i": 1}      # the first <lane>-prove-<n> under Amendment 7 (see each PREREG)
LANE_ORDER = ("sc2f", "sc2h", "sc2i")             # SC2f Amendment 3: the three run in this order and share a host
CLOCK_FLOOR_MHZ = 2700                            # the burn's floor; the recorded-clock rule reads records at the same floor
RECORD_MIN_BUSY = 10                              # a loaded clock record needs >= 10 busy samples (P238 amendment 2's rule)
CLOCK_RECORDS = ("card_clock_burn.json", "card_clock.json")   # pre-install burns; with gpuclk_*.txt run logs
#: SC2f Amendment 10: the run sampler's query, read by POSITION (``--format=csv,noheader,nounits``, as card_clock's burn and TC1's
#: sampler read theirs). nvidia-smi rewrites queried names in a CSV header (535.154.05, 610.43.02 and 615.71.09 all print
#: ``clocks.sm`` as ``clocks.current.sm [MHz]``), so no header name is ever looked up.
SAMPLER_FIELDS = ("timestamp", "clocks.sm", "clocks_event_reasons.active", "power.draw")
SAMPLER_QUERY = f"nvidia-smi --query-gpu={','.join(SAMPLER_FIELDS)} --format=csv,noheader,nounits -lms 1000"
SAMPLER_SM_MIN, SAMPLER_SM_MAX = 100, 4000
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
    """The sorted listed powers on the machine's offers in the last reads that carry one, ``None`` when none carries one.
    The machine is in the class (``_in_class``) only when every one of them is in POWER_W."""
    vals = set()
    for rd in reads[-STABLE_READS:]:
        for o in rd:
            if str(o.get("machine_id")) == machine:
                v = o.get("listed_power_w")
                vals.add(float(v) if isinstance(v, (int, float)) else None)
    known = vals - {None}
    return sorted(known) if known else None


def _in_class(listed) -> bool:
    return listed is not None and set(listed) <= set(POWER_W)


def _id_key(m: str):
    return (0, int(m), m) if m.isdigit() else (1, 0, m)


def walk_order(reads: list, clean_runs=None, prefer=(), *, clocks) -> list:
    """The walk: continuity first, then NAMED, then machines listing 575 or 600 W (recorded-pass, then unrecorded, each by
    clean runs desc and id), then listed-power-unknown machines in the same order. Left out: machines listing a power
    outside the class, and (outside continuity and NAMED) any machine whose latest loaded clock record fails. ``clocks``
    is ``recorded_clocks()``: keyword-only with no default, so a caller always says what the census read; ``None``
    (UNREAD) walks continuity and NAMED only."""
    clean_runs = {str(k): int(v) for k, v in (clean_runs or {}).items()}
    seen = sorted({str(o.get("machine_id")) for rd in reads[-STABLE_READS:] for o in rd if o.get("machine_id") is not None},
                  key=_id_key)
    by_runs = lambda ms: sorted(ms, key=lambda m: (-clean_runs.get(m, 0), _id_key(m)))  # noqa: E731
    if clocks is None:
        return list(dict.fromkeys([*prefer, *NAMED]))
    rest = [m for m in seen if m not in NAMED and m not in prefer and (m not in clocks or clocks[m]["ok"])]
    tiers = []
    for tier in ([m for m in rest if _in_class(_listed(m, reads))], [m for m in rest if _listed(m, reads) is None]):
        tiers += by_runs(m for m in tier if m in clocks) + by_runs(m for m in tier if m not in clocks)
    return list(dict.fromkeys([*prefer, *NAMED, *tiers]))


def _is_5090(run_dir: Path, receipt: dict) -> bool:
    env = receipt.get("environment") if isinstance(receipt.get("environment"), dict) else {}
    name = str(env.get("vast_gpu_name") or receipt.get("gpu_model") or "")
    if name:
        return "5090" in name
    for fx in (run_dir / "forensics.txt", *sorted(run_dir.glob("*/forensics.txt"))):
        try:
            g = re.search(r"RTX (\d{4})", fx.read_text(errors="replace").split("\n", 1)[0])
        except FileNotFoundError:
            continue
        if g:
            return g.group(1) == "5090"
    return False


def _utc(value):
    """``started_at`` as a sortable datetime, or None when missing or not ``YYYY-MM-DDTHH:MM:SSZ`` (fractional seconds allowed)."""
    import datetime
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.\d+)?Z", str(value or ""))
    if not m:
        return None
    try:
        return datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def recorded_clocks(store: Path):
    """SC2f Amendment 10's recorded-clock census (P238 amendment 2's rule): ``{machine: {"ok", "run", "why"}}`` for every RTX
    5090 machine with a loaded clock record, read from its LATEST receipt (by ``started_at``, then day, then run id) that
    has one. ``ok`` is True only when EVERY loaded record in that receipt passes ``card_clock.passes(summary, 2700,
    mode="clock")``; a record that does not parse fails, and a burn summary written before TC1 amendment 83 fails (its
    Reliability clause reads UNREAD). A record with fewer than 10 busy samples (a harness error summary among them) is
    no record. Returns ``None`` (UNREAD) on any store error other than a missing file, and when a run with a clock record
    has a ``receipt.json`` that exists but does not parse as an object, or a ``started_at`` that is missing or does not
    parse as ``YYYY-MM-DDTHH:MM:SSZ`` (no defined "latest"): a skipped or mis-sorted run could hide a machine's failing
    latest record behind an older passing one. A run with no ``receipt.json`` names no machine and vouches for none."""
    cc = _card_clock()

    def _err(e):
        if not isinstance(e, FileNotFoundError):
            raise e

    store = Path(store)
    try:
        runs = {}
        for root, _dirs, files in os.walk(store, onerror=_err):
            for fn in files:
                if (fn.startswith("gpuclk_") and fn.endswith(".txt")) or fn in CLOCK_RECORDS:
                    rel = Path(root, fn).relative_to(store).parts
                    if len(rel) >= 3:
                        runs.setdefault(store / rel[0] / rel[1], []).append(Path(root, fn))
        latest = {}
        for rd, files in sorted(runs.items()):
            try:
                rj = json.loads((rd / "receipt.json").read_text())
            except FileNotFoundError:
                continue
            except ValueError:
                return None
            if not isinstance(rj, dict):
                return None
            env = rj.get("environment") if isinstance(rj.get("environment"), dict) else {}
            m = str(env.get("vast_machine_id") or "")
            if not m or not _is_5090(rd, rj):
                continue
            recs = []
            for f in sorted(files):
                try:
                    summ = json.loads(f.read_text()) if f.suffix == ".json" else cc.summarize_log(str(f))
                except FileNotFoundError:
                    continue
                except ValueError:
                    recs.append((False, f"unparseable record {f.name}"))
                    continue
                if not isinstance(summ, dict) or (summ.get("busy_samples") or 0) < RECORD_MIN_BUSY or summ.get("sm_median") is None:
                    continue
                recs.append(cc.passes(summ, CLOCK_FLOOR_MHZ, min_busy=RECORD_MIN_BUSY, mode="clock"))
            started = _utc(rj.get("started_at"))
            if started is None:
                return None
            key = (started, rd.parent.name, rd.name)
            if recs and (m not in latest or key > latest[m]["key"]):
                latest[m] = {"key": key, "run": f"{rd.parent.name}/{rd.name}", "ok": all(ok for ok, _ in recs),
                             "why": "; ".join(sorted({w for ok, w in recs if not ok}))}
    except OSError:
        return None
    return {m: {k: v for k, v in r.items() if k != "key"} for m, r in latest.items()}


def admit(attempts: list, reads: list, created_26157: bool, prefer=(), clean_runs=None, *, clocks) -> dict:
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
        walk = walk_order(reads, clean_runs, prefer, clocks=clocks)
        rec["clock_census"] = "UNREAD" if clocks is None else "read"
        rec["clock_refused"] = {} if clocks is None else {
            m: f"latest loaded clock record {clocks[m]['run']} fails: {clocks[m]['why']}"
            for m in sorted({str(o.get("machine_id")) for rd in reads[-STABLE_READS:] for o in rd}, key=_id_key)
            if m in clocks and not clocks[m]["ok"] and m not in NAMED and m not in prefer}
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
        if lp is not None and not _in_class(lp):
            rec["skipped"][m] = f"listed power {lp} W, not within {list(POWER_W)} W"
            continue
        offers = _stable(m, reads)
        if offers:
            return dict(rec, verdict="ADMIT", machine=m, offers=offers, listed_power_w=lp)
        rec["skipped"][m] = f"no offer <= ${CEILING_USD_H}/h with >= {MIN_DISK_GB} GB in each of the last {STABLE_READS} reads"
    return dict(rec, verdict="NONE", why="no walk machine admissible on these reads")


_W = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*W\s*$")


def power(csv_text: str) -> dict:
    """PASS, REFUSE (read: an enforced limit outside {575, 600} W, or more than one GPU; a host fault, rc 17) or UNREAD
    (nothing readable: probe_error, a harness fault, rc 25: the run is void)."""
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
    if vals[0] not in POWER_W:
        return dict(rec, verdict="REFUSE", refusal="below_floor", why=f"enforced power limit {vals[0]} W, not within {list(POWER_W)} W")
    return dict(rec, verdict="PASS", refusal=None, why="")


_FPL = re.compile(r"^power\.limit,clocks\.max\.sm\s+([0-9]+(?:\.[0-9]+)?)\s*W", re.M)


def _watts(d: dict, key: str):
    if not isinstance(d, dict):
        return None
    try:
        return float(d.get(key))
    except (TypeError, ValueError):
        return None


def judge_proof(forensics_text, host_facts, qualify_power) -> dict:
    """PASS, REFUSE (a record READ at a power limit outside {575, 600} W, or at one that differs from the qualify step's
    enforced reading: a host fault, rc 18) or UNREAD (a record missing, unreadable or without its power field:
    probe_error, a harness fault, rc 25). ``forensics_text``, ``host_facts`` or ``qualify_power`` None means the file is
    missing; ``host_facts`` and ``qualify_power`` (the qualify step's ``power.json``) must be dicts to count as read."""
    unread, wrong, rec = [], [], {}
    qw = _watts(qualify_power, "enforced_w")
    rec["qualify_enforced_w"] = qw
    if qw is None:
        unread.append("qualify power.json missing, unreadable or without enforced_w")
    m = _FPL.search(forensics_text) if isinstance(forensics_text, str) else None
    hf = _watts(host_facts, "gpu_power_limit_w")
    rec["forensics_power_limit_w"] = float(m.group(1)) if m else None
    rec["host_facts_power_limit_w"] = hf
    for name, v, absent in (("forensics.txt power.limit", rec["forensics_power_limit_w"],
                             "forensics.txt missing or without its power.limit line"),
                            ("host_facts.json gpu_power_limit_w", hf,
                             "host_facts.json missing, unreadable or without gpu_power_limit_w")):
        if v is None:
            unread.append(absent)
        elif v not in POWER_W:
            wrong.append(f"{name} {v} W, not within {list(POWER_W)} W")
        elif qw is not None and v != qw:
            wrong.append(f"{name} {v} W differs from the qualify reading {qw} W")
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


_SAMPLER_TS = re.compile(r"^\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?$")
_SAMPLER_SM = re.compile(r"^(\d+)(?: MHz)?$")
_SAMPLER_HEX = re.compile(r"^0x[0-9a-fA-F]+$")
_SAMPLER_W = re.compile(r"^\d+(?:\.\d+)?(?: W)?$")


def _sampler_row(line: str):
    """One sampler row read by POSITION in SAMPLER_QUERY's order (SC2f Amendment 10): ``(sm_mhz, reasons_mask)``, or None
    when the row fails a sanity check: the column count, the timestamp, the SM column as an integer MHz in
    [SAMPLER_SM_MIN, SAMPLER_SM_MAX], the reasons column as 0x-hex, the power column as watts. A unit suffix
    (`` MHz``, `` W``, from a sampler run without ``nounits``) is accepted; a header name never matters."""
    cells = [c.strip() for c in line.split(",")]
    if len(cells) != len(SAMPLER_FIELDS):
        return None
    ts, sm, reasons, watts = cells
    m = _SAMPLER_SM.match(sm)
    if not (_SAMPLER_TS.match(ts) and m and _SAMPLER_HEX.match(reasons) and _SAMPLER_W.match(watts)):
        return None
    mhz = int(m.group(1))
    if not SAMPLER_SM_MIN <= mhz <= SAMPLER_SM_MAX:
        return None
    return mhz, int(reasons, 16)


def judge_run(text) -> dict:
    """The controller-side sampler over the whole run, by card_clock's bits: PASS, REFUSE (read, below_floor: a host
    fault) or UNREAD (probe_error: a harness fault). Rows are read by position (SC2f Amendment 10): a legacy header
    line is skipped only as the FIRST line and only when it fails the row checks; a non-numeric row anywhere after
    it is UNREAD outright (a restarted or concatenated sampler), and rows failing a check count against the 5 %
    unreadable limit (a torn last line)."""
    cc = _card_clock()
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return {"verdict": "UNREAD", "refusal": "probe_error", "why": ["no sampler rows"], "rows": 0}
    header = None
    if _sampler_row(lines[0]) is None and not any(ch.isdigit() for ch in lines[0].split(",")[0]):
        header, lines = lines[0], lines[1:]
    masks, bad = [], 0
    for ln in lines:
        r = _sampler_row(ln)
        if r is not None:
            masks.append(r)
        elif not any(ch.isdigit() for ch in ln.split(",")[0]):
            return {"verdict": "UNREAD", "refusal": "probe_error", "rows": len(lines), "header_skipped": header,
                    "why": [f"a non-numeric row after the first line: {ln[:120]!r}"]}
        else:
            bad += 1
    rows = len(masks) + bad
    rec = {"rows": rows, "unreadable_rows": bad, "header_skipped": header}
    if len(masks) < SAMPLER_MIN_ROWS or (rows and bad / rows > SAMPLER_MAX_BAD):
        return dict(rec, verdict="UNREAD", refusal="probe_error",
                    why=[f"{len(masks)} readable rows of {rows} (need >= {SAMPLER_MIN_ROWS}, <= {SAMPLER_MAX_BAD:.0%} unreadable; "
                         f"each row: {len(SAMPLER_FIELDS)} columns in the order {','.join(SAMPLER_FIELDS)})"])
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
    adm = lambda *x, **k: admit(*x, **{"clocks": {}, **k})  # noqa: E731 -- an empty census: nothing recorded
    QP = {"enforced_w": 575.0}                         # the qualify step's power.json on a 575 W card
    hdr = ""                                           # SC2f Amendment 10: the sampler runs noheader,nounits and is read by position
    row = lambda r="0x0000000000000000", mhz=2800: f"2026/10/26 10:00:00.000, {mhz}, {r}, 540.00\n"  # noqa: E731
    legacy = "timestamp, clocks.current.sm [MHz], clocks_event_reasons.active, power.draw [W]\n"   # 610.43.02's real header
    cases = [
        ("admit: 52780 first on stable reads", adm([], named, False).get("machine") == "52780"),
        ("admit: 52780 tried -> 145701", adm([a(11, "52780")], named, False).get("machine") == "145701"),
        ("admit: a refusal consumes an attempt", adm([a(11, "52780")], named, False)["attempts_used"] == 1),
        ("admit: a pre-create refusal consumes nothing", adm([a(11, "52780", created=False)], named, False)
         .get("machine") == "52780"),
        ("admit: 3 creates and no PROVED -> PARK", adm([a(11, "52780"), a(12, "145701"), a(13, "9")], named, False)
         ["verdict"] == "PARK"),
        ("admit: PROVED pins that host only", adm([a(11, "145701", proved=True)], named, False).get("machine")
         == "145701" and adm([a(11, "145701", proved=True)], [rd(("52780", "1", 0.9, 400, 575.0))] * 3, False)
         ["verdict"] == "NONE"),
        ("admit: the walk is OPEN: an unnamed 575 W machine is admitted",
         adm([], [rd(("777", "7", 0.9, 400, 575.0))] * 3, False).get("machine") == "777"),
        ("admit: a machine appearing mid-screen is admitted once stable",
         adm([], [rd(("52780", "1", 1.2, 400, 575.0))] * 2 + [rd(("888", "8", 0.9, 400, 575.0))] * 3, False)
         .get("machine") == "888"),
        ("admit: ... and not before its third read",
         adm([], [rd()] + [rd(("888", "8", 0.9, 400, 575.0))] * 2, False)["verdict"] == "NONE"),
        ("admit: 575 W machines by clean runs, then id", adm([], [rd(("300", "a", 0.9, 400, 575.0),
                                                                        ("200", "b", 0.9, 400, 575.0),
                                                                        ("100", "c", 0.9, 400, 575.0))] * 3, False,
                                                                clean_runs={"300": 5, "200": 5, "100": 1})
         .get("machine") == "200"),
        ("admit: unknown listed power comes LAST", adm([], [rd(("100", "c", 0.9, 400, None),
                                                                  ("900", "d", 0.9, 400, 575.0))] * 3, False,
                                                          clean_runs={"100": 9}).get("machine") == "900"),
        ("admit: unknown listed power is admissible when nothing else is",
         adm([], [rd(("100", "c", 0.9, 400, None))] * 3, False).get("machine") == "100"),
        ("admit: another listed power is never admitted",
         adm([], [rd(("19317", "x", 0.6, 400, 450.0))] * 3, False)["verdict"] == "NONE"),
        ("admit: exclusions bind in the open walk too",
         adm([], [rd(("13828", "x", 0.5, 900, 575.0), ("37367", "y", 0.5, 900, 575.0),
                       ("26157", "z", 0.5, 900, 575.0))] * 3, False)["verdict"] == "NONE"),
        ("admit: 26157 after a create there", adm([], [rd(("26157", "z", 0.5, 900, 575.0))] * 3, True)
         .get("machine") == "26157"),
        ("admit: the LAST 3 reads decide", adm([], [rd()] + named, False).get("machine") == "52780"),
        ("admit: 2 of the last 3 reads is not stable", adm([], [rd()] + named[:2], False)["verdict"] == "NONE"
         and adm([], named[:2] + [rd()], False)["verdict"] == "NONE" and adm([], named[:2], False)["verdict"] == "NONE"),
        ("admit: over the ceiling (the all-in usd_h) is not admissible",
         adm([], [rd(("52780", "1", 1.26, 400, 575.0))] * 3, False)["verdict"] == "NONE"),
        ("admit: under 320 GB disk is not admissible", adm([], [rd(("52780", "1", 0.9, 319, 575.0))] * 3, False)
         ["verdict"] == "NONE"),
        ("admit: an offer must persist by id", adm([], [rd(("52780", "1", 0.9, 400, 575.0)),
                                                         rd(("52780", "2", 0.9, 400, 575.0)),
                                                         rd(("52780", "1", 0.9, 400, 575.0))], False)["verdict"] == "NONE"),
        ("admit: continuity puts the earlier lane's host first",
         adm([], [rd(("52780", "1", 0.9, 400, 575.0), ("777", "7", 0.9, 400, 575.0))] * 3, False,
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
        ("judge: both records at 575 pass", judge_proof(fx, {"gpu_power_limit_w": 575.0}, QP)["verdict"] == "PASS"),
        ("judge: forensics READ at 500 W refuses", judge_proof(fx.replace("575.00 W", "500.00 W"),
                                                               {"gpu_power_limit_w": 575.0}, QP)["verdict"] == "REFUSE"),
        ("judge: host facts READ at 450 W refuse", judge_proof(fx, {"gpu_power_limit_w": 450.0}, QP)["verdict"] == "REFUSE"),
        ("judge: host facts missing is UNREAD", judge_proof(fx, None, QP)["verdict"] == "UNREAD"),
        ("judge: host facts unreadable is UNREAD", judge_proof(fx, "unreadable", QP)["verdict"] == "UNREAD"),
        ("judge: host facts without the field is UNREAD", judge_proof(fx, {}, QP)["verdict"] == "UNREAD"),
        ("judge: forensics missing is UNREAD", judge_proof(None, {"gpu_power_limit_w": 575.0}, QP)["verdict"] == "UNREAD"),
        ("judge: no power line is UNREAD", judge_proof("GPU 0\n", {"gpu_power_limit_w": 575.0}, QP)["verdict"] == "UNREAD"),
        ("judge: a READ wrong record outranks a missing one", judge_proof(fx.replace("575.00 W", "500.00 W"), None, QP)
         ["verdict"] == "REFUSE"),
        # SC2f Amendment 10: the {575, 600} W class, the recorded-clock rule, and the post-proof records equal to the qualify reading
        ("am10 admit: a 600 W listing is in the class", adm([], [rd(("600", "6", 0.9, 400, 600.0))] * 3, False).get("machine") == "600"),
        ("am10 admit: a machine listing 575 and 600 W is in the class",
         adm([], [rd(("55678", "a", 0.9, 400, 575.0), ("55678", "b", 0.9, 400, 600.0))] * 3, False).get("machine") == "55678"),
        ("am10 admit: 500, 599.9 and 600.1 W listings are never admitted",
         all(adm([], [rd(("9", "x", 0.9, 400, w))] * 3, False)["verdict"] == "NONE" for w in (500.0, 599.9, 600.1))),
        ("am10 admit: a 600 W machine listing 500 W on another offer is out", adm([], [rd(("36204", "a", 0.9, 400, 600.0),
                                                                                        ("36204", "b", 0.9, 400, 500.0))] * 3, False)["verdict"] == "NONE"),
        ("am10 admit: 575 and 600 W interleave by clean runs", adm([], [rd(("100", "a", 0.9, 400, 575.0), ("200", "b", 0.9, 400, 600.0),
                                                                        ("300", "c", 0.9, 400, 575.0))] * 3, False,
                                                                    clean_runs={"100": 1, "200": 3, "300": 2})["walk"][2:] == ["200", "300", "100"]),
        ("am10 admit: a recorded clock FAIL is never admitted", adm([], [rd(("200", "b", 0.9, 400, 600.0))] * 3, False,
                                                                    clocks={"200": {"ok": False, "run": "d/r", "why": "x"}})["verdict"] == "NONE"),
        ("am10 admit: ... and is named in clock_refused", "200" in adm([], [rd(("200", "b", 0.9, 400, 600.0))] * 3, False,
                                                                       clocks={"200": {"ok": False, "run": "d/r", "why": "x"}})["clock_refused"]),
        ("am10 admit: recorded-pass before unrecorded, whatever the clean runs",
         adm([], [rd(("100", "a", 0.9, 400, 575.0), ("200", "b", 0.9, 400, 600.0))] * 3, False, clean_runs={"100": 9},
             clocks={"200": {"ok": True, "run": "d/r", "why": ""}}).get("machine") == "200"),
        ("am10 admit: a recorded fail in the listed-unknown tier is out too", adm([], [rd(("100", "a", 0.9, 400, None))] * 3, False,
                                                                                 clocks={"100": {"ok": False, "run": "d/r", "why": "x"}})["verdict"] == "NONE"),
        ("am10 admit: NAMED is walked as before, whatever its clock record",
         adm([], named, False, clocks={"52780": {"ok": False, "run": "d/r", "why": "x"}}).get("machine") == "52780"),
        ("am10 admit: an UNREAD census walks continuity and NAMED only",
         adm([], [rd(("200", "b", 0.9, 400, 600.0))] * 3, False, clocks=None)["verdict"] == "NONE"
         and adm([], named, False, clocks=None).get("machine") == "52780"),
        ("am10 power: 600.00 W enforced passes", power("600.00 W, 600.00 W, 600.00 W")["verdict"] == "PASS"),
        ("am10 power: 500, 599.9 and 600.1 W refuse", all(power(f"{w} W, 600.00 W, 600.00 W")["verdict"] == "REFUSE"
                                                          for w in ("500.00", "599.9", "600.1"))),
        ("am10 judge: both records at 600 W with a 600 W qualify reading pass",
         judge_proof(fx.replace("575.00 W", "600.00 W"), {"gpu_power_limit_w": 600.0}, {"enforced_w": 600.0})["verdict"] == "PASS"),
        ("am10 judge: a record in the class but not the qualify reading refuses",
         judge_proof(fx.replace("575.00 W", "600.00 W"), {"gpu_power_limit_w": 600.0}, QP)["verdict"] == "REFUSE"
         and judge_proof(fx, {"gpu_power_limit_w": 600.0}, QP)["verdict"] == "REFUSE"),
        ("am10 judge: a missing qualify power.json is UNREAD", judge_proof(fx, {"gpu_power_limit_w": 575.0}, None)["verdict"] == "UNREAD"
         and judge_proof(fx, {"gpu_power_limit_w": 575.0}, "unreadable")["verdict"] == "UNREAD"
         and judge_proof(fx, {"gpu_power_limit_w": 575.0}, {})["verdict"] == "UNREAD"),
        # SC2f Amendment 11: one ceiling, $1.25, for the three lanes
        ("am11 admit: an offer at $1.25 is admitted", adm([], [rd(("152938", "x", 1.25, 400, 600.0))] * 3, False)
         .get("machine") == "152938" and CEILING_USD_H == 1.25),
        ("am11 admit: $1.26 is refused", adm([], [rd(("152938", "x", 1.26, 400, 600.0))] * 3, False)["verdict"] == "NONE"),
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
        ("am10 run: a noheader,nounits sampler passes", judge_run(row() * 30)["verdict"] == "PASS"),
        ("am10 run: a legacy header line with unit-suffixed rows passes",
         judge_run(legacy + "2026/10/26 10:00:00.000, 2800 MHz, 0x0000000000000000, 540.00 W\n" * 30)["verdict"] == "PASS"),
        ("am10 run: the header NAME never matters", judge_run(legacy.replace("clocks.current.sm", "anything.else") + row() * 30)
         ["verdict"] == "PASS"),
        ("am10 run: SM and reasons swapped is UNREAD", judge_run("2026/10/26 10:00:00.000, 0x0000000000000000, 2800, 540.00\n" * 30)
         ["verdict"] == "UNREAD"),
        ("am10 run: SM and power swapped is UNREAD", judge_run("2026/10/26 10:00:00.000, 540.00, 0x0000000000000000, 2800\n" * 30)
         ["verdict"] == "UNREAD"),
        ("am10 run: a short query (3 columns) is UNREAD", judge_run("2026/10/26 10:00:00.000, 2800, 0x0000000000000000\n" * 30)
         ["verdict"] == "UNREAD"),
        ("am10 run: a header line after the first is UNREAD", judge_run(row() * 20 + legacy + row() * 20)["verdict"] == "UNREAD"),
        ("am10 run: an SM reading outside [100, 4000] MHz is unreadable", judge_run(row(mhz=40000) * 30)["verdict"] == "UNREAD"),
        ("am10 run: a torn last line is tolerated within 5 %", judge_run(row() * 30 + "2026/10/26 10:00:01.000, 28")["verdict"] == "PASS"),
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
    ap.add_argument("--qualify-power")
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
                    creates_on(Path(a.store), UNTIL_CREATE[0]), prefer=earlier_hosts(Path(a.store), a.lane), clean_runs=cr,
                    clocks=recorded_clocks(Path(a.store)))
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
        if not (a.forensics and a.host_facts and a.qualify_power):
            ap.error("judge-proof needs --forensics, --host-facts and --qualify-power (all three, always)")

        def _load(path):
            t = _read_text(path)
            try:
                return json.loads(t) if t is not None else None
            except ValueError:
                return "unreadable"
        rec = judge_proof(_read_text(a.forensics), _load(a.host_facts), _load(a.qualify_power))
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
