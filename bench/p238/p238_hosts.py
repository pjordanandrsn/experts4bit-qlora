"""P238 amendment 3: the ONE recorded-good function. The shakedown screen and the census both call ``recorded_good``, so they
cannot disagree about a host.

A machine is recorded-good when its LATEST receipt that holds any loaded record satisfies BOTH:
- every loaded record in that receipt passes, where the records are
  - each ``card_clock*.json`` burn (P238's ``card_clock_burn.json``, SC2's ``qualify/box/card_clock.json``), RE-JUDGED by
    ``card_clock.passes(summary, 2700, mode="clock")`` from its summary, never through its stored ``pass`` field (a burn written
    before TC1 amendment 83 has no Reliability count, so it re-judges UNREAD and fails: it cannot vouch for a host);
  - each ``gpuclk_*.txt`` run log, summarized by ``card_clock.summarize_log`` and judged the same way;
  - each ``run_judge.json`` (SC2's run sampler verdict): a ``REFUSE`` is a FAILING record, whatever the burn read (a power-capped
    burn masks Reliability; a run judge can still see it);
  burns and run logs count with >= 10 busy samples;
- the card is shown able to hold the clock under the GATE'S OWN LOAD: that receipt recorded a power limit >= 575 W, or one of its
  passing records is a burn (a passing clock-mode burn is necessarily an amendment-83-era record).

Why the power clause: P238's gate is a 20 s full-power bf16 matmul burn, which is power-bound. Run logs under a lighter training load
overstate a capped card's burn clock (p238-shake-r3: 2,752-2,827 MHz in training at 475 W, 2,587 MHz at the burn). All seven RTX 5090
burns in the store split exactly at 575 W.

The read rule: only FileNotFoundError means a file is absent. Any other OSError while reading the store propagates, and the caller
(the screen) treats the whole census as UNREAD and pins nothing. A record that does not parse FAILS. A receipt that names no
machine vouches for nothing.

    python3 p238_hosts.py STORE_RECEIPTS_DIR CARD_CLOCK_PY     (prints {machine: verdict}; private: machine ids are not public)
"""
from __future__ import annotations

import calendar
import json
import os
import re
import sys
import time

CLOCK_FLOOR_MHZ = 2700
POWER_MIN_W = 575.0
MIN_BUSY = 10


def _walk_err(e):
    if not isinstance(e, FileNotFoundError):
        raise e


def _json(path):
    """The JSON object at ``path``; {} when absent or not an object. A ValueError (it does not parse) is returned as None."""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except (FileNotFoundError, NotADirectoryError):   # absent (a path through a file is absent too)
        return {}
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


def _epoch(s):
    try:
        return calendar.timegm(time.strptime(str(s)[:20], "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        return 0


def _first_line(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.readline()
    except (FileNotFoundError, NotADirectoryError):
        return ""


def _card(rd):
    """(GPU class or None, power limit W or None) for a run dir: host_facts.json, then SC2's qualify/power.json, then
    forensics.txt's first line, then box.json's name."""
    hf = _json(os.path.join(rd, "host_facts.json")) or {}
    w = hf.get("gpu_power_limit_w")
    if w is None:
        pj = _json(os.path.join(rd, "qualify", "power.json")) or {}
        w = pj.get("enforced_w", pj.get("power_limit_w"))
    name = None
    r = _json(os.path.join(rd, "receipt.json")) or {}
    env = r.get("environment") if isinstance(r.get("environment"), dict) else {}
    for label in (env.get("vast_gpu_name"), (_json(os.path.join(rd, "instance.json")) or {}).get("gpu")):
        g = re.search(r"RTX (\d{4})", str(label or ""))
        if g:
            name = g.group(1)
            break
    for sub in ("", "p238", "tc1", "sc1"):
        line = _first_line(os.path.join(rd, sub, "forensics.txt"))
        g = re.search(r"RTX (\d{4})", line)
        if g:
            name = name or g.group(1)
            if w is None:
                p = re.search(r",\s*([\d.]+) W", line)
                w = float(p.group(1)) if p else None
            break
    if name is None:
        try:
            subs = sorted(os.listdir(rd))
        except FileNotFoundError:
            subs = []
        for sub in subs:
            box = _json(os.path.join(rd, sub, "box.json")) or {}
            g = re.search(r"RTX (\d{4})", str(box.get("name", "")))
            if g:
                name = g.group(1)
                break
    return name, (float(w) if w is not None else None)


def _machine(rd):
    r = _json(os.path.join(rd, "receipt.json")) or {}
    env = r.get("environment")
    m = env.get("vast_machine_id") if isinstance(env, dict) else None
    if m is None:
        m = (_json(os.path.join(rd, "instance.json")) or {}).get("machine_id")
    return (str(m) if m is not None else None), _epoch(r.get("started_at"))


def judge_records(rd, files, cc):
    """[(kind, ok, why)] for one run dir's loaded records."""
    recs = []
    for f in sorted(files):
        base = os.path.basename(f)
        if base == "run_judge.json":
            d = _json(f)
            if d is None:
                recs.append(("run_judge", False, f"unparseable {base}"))
            elif d:
                v = str(d.get("verdict", ""))
                if v == "REFUSE":
                    recs.append(("run_judge", False, "run judge REFUSE: " + "; ".join(map(str, d.get("why") or []))[:120]))
            continue
        if base.endswith(".json"):
            s, kind = _json(f), "burn"
            if s is None:
                recs.append((kind, False, f"unparseable {base}"))
                continue
        else:
            kind = "run_log"
            try:
                s = cc.summarize_log(f)
            except FileNotFoundError:
                continue
        if not s or (s.get("busy_samples") or 0) < MIN_BUSY or s.get("sm_median") is None:
            continue
        ok, why = cc.passes(s, CLOCK_FLOOR_MHZ, mode="clock")
        recs.append((kind, ok, why))
    return recs


def recorded_good(receipts_root, cc):
    """{machine: {"good", "run", "t", "power_w", "records", "why"}} over every RTX 5090 run dir under ``receipts_root``
    (``.../receipts``), each machine judged on its latest receipt with any loaded record. Raises OSError on any read error but a
    missing file (the read rule)."""
    runs = {}
    for root, _dirs, fs in os.walk(receipts_root, onerror=_walk_err):
        for fn in fs:
            if (fn.startswith("card_clock") and fn.endswith(".json")) or (fn.startswith("gpuclk_") and fn.endswith(".txt")) \
                    or fn == "run_judge.json":
                parts = os.path.join(root, fn).split(os.sep)
                i = parts.index("receipts")
                runs.setdefault(os.sep.join(parts[: i + 4]), []).append(os.path.join(root, fn))
    latest = {}
    for rd, files in sorted(runs.items()):
        m, t = _machine(rd)
        if m is None:
            continue
        name, w = _card(rd)
        if name != "5090":
            continue
        recs = judge_records(rd, files, cc)
        if not recs:
            continue
        key = (t, os.path.basename(rd))
        if m in latest and key <= (latest[m]["t"], latest[m]["run"]):
            continue
        all_pass = all(ok for _, ok, _ in recs)
        burn_pass = any(ok and kind == "burn" for kind, ok, _ in recs)
        power_ok = w is not None and w >= POWER_MIN_W
        why = [w_ for _, ok, w_ in recs if not ok]
        if all_pass and not (power_ok or burn_pass):
            why.append(f"power limit {w} W < {POWER_MIN_W:.0f} W and no passing burn: run logs do not predict the burn")
        latest[m] = {"good": all_pass and (power_ok or burn_pass), "run": os.path.basename(rd), "t": t, "power_w": w,
                     "records": [[k, ok] for k, ok, _ in recs], "why": "; ".join(why)[:240]}
    return latest


def main(argv=None):
    import importlib.util
    argv = sys.argv[1:] if argv is None else argv
    root, cc_path = argv[0], argv[1]
    spec = importlib.util.spec_from_file_location("card_clock", cc_path)
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    out = recorded_good(root, cc)
    print(json.dumps({"good": sum(v["good"] for v in out.values()), "machines": len(out), "verdicts": out}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
