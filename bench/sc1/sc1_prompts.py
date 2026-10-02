#!/usr/bin/env python3
"""sc1_prompts.py -- the prompt files and the quality windows of lane SC1 (experts4bit-qlora#846), cut by
``step_decomp._k8_window`` byte for byte (P58's dump, bench/p58/p58_run.sh "prompt dump"), so every engine on every box
sees the same token ids and every quality row scores the same window.

Writes into --out (the box's $W):
  prompts_b1.json, prompts_b16.json   512-token rows, B DISTINCT windows of wikitext-2 test, P58's record shape:
                                      {model, revision, source, batch, prompt_len, corpus_tokens, row_step, rows_sha256,
                                       prompts_sha256, prompts}  (prompts_sha256 = sha256(json.dumps(prompts)), p37's digest)
  prompts_b1_4096.json                one 4096-token window (the TTFT-4096 arm)
  prompts_b16_same.json               16 copies of prompts_b16's row 0 -- the SAMEPROMPT control; the record SAYS so
                                      (rows_distinct_in_file false, sameprompt {source_row, copies, source_file_sha256})
  k8_window_wikitext.json, k8_window_c4val1.json
                                      {ids: ids[:prompt_len+steps+1], text_sha, source, prompt_len, steps, model, revision,
                                       digest_rule}; text_sha IS _k8_window's own ppl_sha (sha256 of the int64 tensor's bytes)
                                      and is re-derived here from the id list with the stdlib (little-endian int64) and
                                      asserted equal, so a scorer without torch (llama.cpp's harness, the HTTP drivers) can
                                      refuse a foreign window by the same digest.
Prints one `PROMPTS ...` line per prompt file and one `WINDOW ...` line per window; the runner tees them into summary.txt.
``step_decomp.py`` is imported from --harness-dir (default: this file's directory; on the box both sit in $W).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import types

PROMPT_LEN, STEPS, SAME_COPIES = 512, 2048, 16
SOURCES = ("wikitext", "c4val1")
SOURCE_DESC = "wikitext-2-raw-v1 test via step_decomp._k8_window"
DIGEST_RULE = ("text_sha = sha256(ids[:prompt_len+steps+1] as int64 little-endian bytes) == step_decomp._k8_window's ppl_sha "
               "(torch int64 tensor .numpy().tobytes()); a quality row whose sha differs from this file's is VOID")


def prompts_digest(prompts) -> str:
    """p37's expression, byte for byte."""
    return hashlib.sha256(json.dumps(prompts).encode()).hexdigest()


def row_digests(prompts) -> list:
    return [hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts]


def window_digest(ids) -> str:
    """K8's digest from a plain id list: int64 little-endian bytes (== torch.long tensor .numpy().tobytes() on x86/arm64)."""
    return hashlib.sha256(struct.pack(f"<{len(ids)}q", *ids)).hexdigest()


def window_args(source: str, prompt_len: int, batch: int, steps: int):
    """The namespace _k8_window reads (bench/p39/step_decomp.py:616); the same fields p58_run.sh passed."""
    return types.SimpleNamespace(ppl_source=source, ppl_chat=False, ppl_chat_suffix="", prompt_offset=0, prompt_span=0,
                                 prompt_len=prompt_len, batch=batch, ppl_steps=steps)


def prompt_record(model, revision, batch, prompt_len, corpus_tokens, row_step, prompts, source=SOURCE_DESC) -> dict:
    """P58's record, key for key."""
    if len(prompts) != batch:
        raise ValueError(f"{len(prompts)} rows != batch {batch}")
    if not all(len(p) == prompt_len for p in prompts):
        raise ValueError(f"row lengths {[len(p) for p in prompts]} != {prompt_len}")
    if len({tuple(p) for p in prompts}) != batch:
        raise ValueError("rows must be distinct (P20's distinct-prompt rule)")
    return {"model": model, "revision": revision, "source": source, "batch": batch, "prompt_len": prompt_len,
            "corpus_tokens": int(corpus_tokens), "row_step": int(row_step), "rows_sha256": row_digests(prompts),
            "prompts_sha256": prompts_digest(prompts), "prompts": prompts}


def same_prompt_record(rec16: dict, copies: int = SAME_COPIES) -> dict:
    """The SAMEPROMPT control file: `copies` copies of row 0 of the B=16 file; every field says so."""
    row0 = list(rec16["prompts"][0])
    prompts = [list(row0) for _ in range(copies)]
    return {"model": rec16["model"], "revision": rec16["revision"],
            "source": rec16["source"] + " -- SAMEPROMPT control: row 0 replicated", "batch": copies,
            "prompt_len": rec16["prompt_len"], "corpus_tokens": rec16["corpus_tokens"], "row_step": 0,
            "rows_sha256": row_digests(prompts), "prompts_sha256": prompts_digest(prompts), "prompts": prompts,
            "rows_distinct_in_file": False,
            "sameprompt": {"source_row": 0, "copies": copies, "source_file_sha256": rec16["prompts_sha256"],
                           "source_row_sha256": rec16["rows_sha256"][0]}}


def window_record(source, ids, text_sha, prompt_len, steps, model, revision) -> dict:
    ids = [int(x) for x in ids]
    if len(ids) != prompt_len + steps + 1:
        raise ValueError(f"window has {len(ids)} ids, expected {prompt_len + steps + 1}")
    got = window_digest(ids)
    if got != text_sha:
        raise ValueError(f"window digest {got[:12]} != _k8_window's ppl_sha {text_sha[:12]}: the stdlib re-derivation disagrees "
                         "with the harness -- refusing to write a window file whose sha would VOID every row")
    return {"ids": ids, "text_sha": text_sha, "source": source, "prompt_len": prompt_len, "steps": steps,
            "n_ids": len(ids), "model": model, "revision": revision, "digest_rule": DIGEST_RULE,
            "scored_targets": f"ids[{prompt_len + 1}..{prompt_len + steps}] (K8's cont[t+1], t in 0..{steps - 1})"}


def dump(a) -> int:
    sys.path.insert(0, a.harness_dir)
    import step_decomp  # the staged harness; its own window rule, never a re-implementation
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.rev)
    os.makedirs(a.out, exist_ok=True)
    recs = {}
    for name, (B, L) in (("prompts_b1", (1, PROMPT_LEN)), ("prompts_b16", (16, PROMPT_LEN)), ("prompts_b1_4096", (1, 4096))):
        ids, step, prompts, _, _ = step_decomp._k8_window(window_args("wikitext", L, B, 0), tok)
        rec = prompt_record(a.model, a.rev, B, L, ids.numel(), step, prompts)
        json.dump(rec, open(os.path.join(a.out, name + ".json"), "w"))
        recs[name] = rec
        print(f"PROMPTS file={name}.json B={B} prompt_len={L} corpus={ids.numel()} step={step} sha={rec['prompts_sha256']}", flush=True)
    same = same_prompt_record(recs["prompts_b16"])
    json.dump(same, open(os.path.join(a.out, "prompts_b16_same.json"), "w"))
    print(f"PROMPTS file=prompts_b16_same.json B=16 prompt_len={PROMPT_LEN} copies_of_row0=16 sha={same['prompts_sha256']} "
          f"source_sha={recs['prompts_b16']['prompts_sha256']}", flush=True)
    for src in SOURCES:
        _, _, _, ppl_ids, ppl_sha = step_decomp._k8_window(window_args(src, PROMPT_LEN, 1, a.steps), tok)
        win = ppl_ids[:PROMPT_LEN + a.steps + 1].tolist()
        rec = window_record(src, win, ppl_sha, PROMPT_LEN, a.steps, a.model, a.rev)
        json.dump(rec, open(os.path.join(a.out, f"k8_window_{src}.json"), "w"))
        print(f"WINDOW src={src} text_sha={ppl_sha} n_ids={len(win)} prompt_len={PROMPT_LEN} steps={a.steps}", flush=True)
    return 0


def _raises(exc, fn, *args, **kw) -> bool:
    try:
        fn(*args, **kw)
    except exc:
        return True
    return False


def selftest() -> int:
    """CPU, no torch, no datasets: the digests and the record shapes on synthetic ids."""
    import array
    checks = []
    ids = [(i * 7919 + 13) % 151936 for i in range(PROMPT_LEN + 64 + 1)]
    # the stdlib digest equals the native int64 buffer's bytes (what torch.long .numpy().tobytes() gives on LE hosts)
    checks.append(window_digest(ids) == hashlib.sha256(array.array("q", ids).tobytes()).hexdigest())
    rec = window_record("wikitext", ids, window_digest(ids), PROMPT_LEN, 64, "m", "r")
    checks.append(rec["n_ids"] == PROMPT_LEN + 65 and rec["text_sha"] == window_digest(ids))
    checks.append(_raises(ValueError, window_record, "wikitext", ids, "0" * 64, PROMPT_LEN, 64, "m", "r"))             # a foreign sha
    checks.append(_raises(ValueError, window_record, "wikitext", ids[:-1], window_digest(ids[:-1]), PROMPT_LEN, 64, "m", "r"))  # a short window
    rows = [[(r * 31 + i) % 1000 for i in range(PROMPT_LEN)] for r in range(16)]
    r16 = prompt_record("m", "r", 16, PROMPT_LEN, 100000, 6000, rows)
    checks.append(set(r16) == {"model", "revision", "source", "batch", "prompt_len", "corpus_tokens", "row_step", "rows_sha256",
                               "prompts_sha256", "prompts"})                                   # P58's keys, no more, no fewer
    checks.append(r16["prompts_sha256"] == hashlib.sha256(json.dumps(rows).encode()).hexdigest())
    checks.append(_raises(ValueError, prompt_record, "m", "r", 2, PROMPT_LEN, 1, 1, [rows[0], rows[0]]))             # identical rows
    same = same_prompt_record(r16)
    checks.append(same["batch"] == 16 and all(p == rows[0] for p in same["prompts"]) and same["rows_distinct_in_file"] is False)
    checks.append(same["prompts_sha256"] != r16["prompts_sha256"] and same["sameprompt"]["source_file_sha256"] == r16["prompts_sha256"])
    checks.append(len(set(same["rows_sha256"])) == 1 and same["rows_sha256"][0] == r16["rows_sha256"][0])
    a = window_args("c4val1", PROMPT_LEN, 1, STEPS)
    checks.append((a.ppl_source, a.prompt_len, a.batch, a.ppl_steps, a.ppl_chat, a.prompt_offset, a.prompt_span)
                  == ("c4val1", 512, 1, 2048, False, 0, 0))
    failed = [i for i, ok in enumerate(checks) if not ok]
    if failed:
        print(f"sc1_prompts selftest FAILED cases {failed}")
        return 1
    print(f"sc1_prompts selftest OK ({len(checks)} cases)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--rev", default="ad44e777bcd18fa416d9da3bd8f70d33ebb85d39")
    ap.add_argument("--out", default=".")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--harness-dir", default=os.environ.get("SC1_HARNESS_DIR", os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    return selftest() if a.selftest else dump(a)


if __name__ == "__main__":
    raise SystemExit(main())
