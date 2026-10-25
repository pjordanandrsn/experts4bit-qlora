#!/usr/bin/env python3
"""sc2j_prompts.py -- lane SC2j's long-ladder prompt pool (bench/sc2/SC2j-PREREG.md, Q3): SC2's pool builder
(``sc2_prompts.rows_from_ids``) at 2,048 tokens a prompt, from the same wikitext-2-raw test split and the same row
offsets, so row k of this pool starts where row k of SC2's 512-token pool starts. Writes the same JSON shape the SC2
driver reads (``rows``, ``prompts_sha256``, ``eos_ids``).

    python sc2j_prompts.py --model M --revision R --out prompts_2048.json
    python sc2j_prompts.py --self-test
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sc2_prompts  # noqa: E402

PROMPT = 2048
ROWS, OFFSET = sc2_prompts.ROWS, sc2_prompts.OFFSET


def build(ids):
    """The pool's rows from the corpus token ids: ROWS rows of PROMPT tokens at OFFSET apart (refuses a short corpus)."""
    return sc2_prompts.rows_from_ids(ids, n=ROWS, prompt=PROMPT, offset=OFFSET)


def self_test() -> int:
    ids = list(range(ROWS * OFFSET + 10))
    rows = build(ids)
    ok = (len(rows) == ROWS and all(len(r) == PROMPT for r in rows) and rows[1][0] == OFFSET
          and rows[0][:512] == sc2_prompts.rows_from_ids(ids)[0])
    try:
        build(list(range(ROWS * OFFSET - 1)))
        short_refused = False
    except SystemExit:
        short_refused = True
    print(f"SC2J_PROMPTS self-test {'PASS' if ok and short_refused else 'FAIL'}")
    return 0 if ok and short_refused else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--revision", default="")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision or None)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    rows = build(tok(text, return_tensors="pt").input_ids[0].tolist())
    eos = sorted({int(x) for x in ([tok.eos_token_id] if tok.eos_token_id is not None else [])})
    json.dump({"model": a.model, "revision": a.revision, "rows": rows, "prompts_sha256": sc2_prompts.digest(rows),
               "offset": OFFSET, "prompt": PROMPT, "eos_ids": eos}, open(a.out, "w"))
    print(f"SC2J_PROMPTS rows={len(rows)} prompt={PROMPT} sha256={sc2_prompts.digest(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
