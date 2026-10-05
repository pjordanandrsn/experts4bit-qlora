#!/usr/bin/env python3
"""sc1g_k8.py -- lane SC1g (#846): P39's step_decomp.py and SC1's K8 window rule, run UNMODIFIED, with two things set in
this process that the shared harness files must not carry (they are byte-pinned by other lanes: P86/P88's step_decomp,
SC1's sc1_prompts.py, P42's hook):

  1. SC1G_CHAT_DATE=YYYY-MM-DD pins the chat template's date. gpt-oss's template writes "Current date: <today>" through
     transformers' strftime_now, so the chat-framed window -- and its sha -- would change with the day, and a run that
     crossed midnight would VOID its own rows. strftime_now reads chat_template_utils.datetime at render time; this
     replaces that one global, and the windows mode checks the rendered prefix carries the pinned date.
  2. SC1G_ROUTE_OUT=<prefix> writes <prefix>.<pid>.json at exit: e4b#1129's hot_residency.ROUTE_SEEN and
     paged_attention.ATTN_SEEN -- only if hot_residency was imported, atomically. atexit does not run on a signal or
     os._exit (the arms run under perl alarm), so a missing record is a FAIL to the reader (sc1g_reduce.py), never a pass.

  sc1g_k8.py k8 -- <step_decomp.py args...>        run step_decomp.py (beside this file) as __main__
  sc1g_k8.py windows --model M --rev R --out DIR --suffix S [--steps 2048]
                                                   k8_window_{wikitext,c4val1}.json: sc1_prompts.window_record over
                                                   step_decomp._k8_window with the chat frame
"""
from __future__ import annotations

import argparse
import atexit
import datetime as _dt
import json
import os
import re
import runpy
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ("wikitext", "c4val1")
PROMPT_LEN = 512


def pin_chat_date(day: str | None = None) -> str | None:
    """Make transformers' strftime_now render `day` (YYYY-MM-DD). Returns the pinned day, or None when unset."""
    day = day or os.environ.get("SC1G_CHAT_DATE", "")
    if not day:
        return None
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise SystemExit(f"SC1G_CHAT_DATE={day!r}: expected YYYY-MM-DD")
    y, m, d = (int(x) for x in day.split("-"))
    import transformers.utils.chat_template_utils as ctu

    class _Pinned(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return _dt.datetime(y, m, d, 12, 0, 0, tzinfo=tz)

    if getattr(ctu, "datetime", None) is not _dt.datetime:
        raise SystemExit(f"transformers' chat_template_utils.datetime is {getattr(ctu, 'datetime', None)!r}, not datetime.datetime: "
                         "strftime_now cannot be pinned this way on this transformers")
    ctu.datetime = _Pinned
    return day


def _dump_routes():
    hr = sys.modules.get("experts4bit_qlora.engines.hot_residency")
    if hr is None:
        return
    pa = sys.modules.get("experts4bit_qlora.engines.paged_attention")
    rec = {"pid": os.getpid(),
           "route_seen": dict(hr.ROUTE_SEEN) if hasattr(hr, "ROUTE_SEEN") else None,    # None: e4b predates #1129
           "attn_seen": dict(pa.ATTN_SEEN) if pa is not None and hasattr(pa, "ATTN_SEEN") else None}
    out = f"{os.environ['SC1G_ROUTE_OUT']}.{os.getpid()}.json"
    with open(out + ".tmp", "w") as f:
        json.dump(rec, f, sort_keys=True)
    os.replace(out + ".tmp", out)


def k8(args) -> None:
    pin_chat_date()
    if os.environ.get("SC1G_ROUTE_OUT"):
        atexit.register(_dump_routes)
    sd = os.path.join(HERE, "step_decomp.py")
    sys.argv = [sd] + list(args)
    sys.path.insert(0, HERE)
    runpy.run_path(sd, run_name="__main__")


def windows(a) -> int:
    day = pin_chat_date()
    sys.path.insert(0, a.harness_dir)
    import sc1_prompts
    import step_decomp
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.rev)
    pre = tok.decode(step_decomp._chat_prefix_ids(tok, a.suffix).tolist())
    dates = re.findall(r"Current date: (\d{4}-\d{2}-\d{2})", pre)
    if day and dates != [day]:
        raise SystemExit(f"the chat prefix carries dates {dates}, not [{day}]: the pin did not take")
    os.makedirs(a.out, exist_ok=True)
    for src in SOURCES:
        ns = types.SimpleNamespace(ppl_source=src, ppl_chat=True, ppl_chat_suffix=a.suffix, prompt_offset=0, prompt_span=0,
                                   prompt_len=PROMPT_LEN, batch=1, ppl_steps=a.steps)
        _, _, _, ppl_ids, ppl_sha = step_decomp._k8_window(ns, tok)
        win = ppl_ids[:PROMPT_LEN + a.steps + 1].tolist()
        rec = sc1_prompts.window_record(src, win, ppl_sha, PROMPT_LEN, a.steps, a.model, a.rev)
        rec.update(chat=True, chat_suffix=a.suffix, chat_date=day, chat_prefix_dates=dates)
        json.dump(rec, open(os.path.join(a.out, f"k8_window_{src}.json"), "w"))
        print(f"WINDOW src={src} text_sha={ppl_sha} n_ids={len(win)} prompt_len={PROMPT_LEN} steps={a.steps} chat_date={day}", flush=True)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["k8"]:
        rest = argv[1:]
        k8(rest[1:] if rest[:1] == ["--"] else rest)
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("windows")
    w.add_argument("--model", required=True)
    w.add_argument("--rev", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--suffix", required=True)
    w.add_argument("--steps", type=int, default=2048)
    w.add_argument("--harness-dir", default=HERE)
    a = ap.parse_args(argv)
    return windows(a)


if __name__ == "__main__":
    sys.exit(main())
