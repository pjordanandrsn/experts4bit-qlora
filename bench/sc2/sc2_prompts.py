#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc2_prompts.py -- lane SC2 (#846): the prompt pool every engine is offered, as token ids.

ROWS distinct rows of PROMPT tokens of wikitext-2-raw test, row k from token k * OFFSET, the corpus joined as P97's
``wikitext_windows`` joins it, tokenised ONCE with the bf16 checkpoint's tokenizer. Every engine receives these exact
ids (``/v1/completions`` with a token-id prompt), so no engine re-tokenises a string differently.

  sc2_prompts.py --model M --revision R --out prompts.json
  sc2_prompts.py --self-test
"""
import argparse
import hashlib
import json
import sys

ROWS, PROMPT, OFFSET = 64, 512, 2048


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, separators=(",", ":")).encode()).hexdigest()


def rows_from_ids(ids, n=ROWS, prompt=PROMPT, offset=OFFSET):
    out = []
    for k in range(n):
        w = ids[k * offset:k * offset + prompt]
        if len(w) != prompt:
            raise SystemExit(f"REFUSED: row {k} has {len(w)} tokens; the corpus is too short for {n} rows")
        out.append([int(t) for t in w])
    if len({tuple(r) for r in out}) != n:
        raise SystemExit("REFUSED: rows are not distinct")
    return out


def self_test() -> int:
    ids = list(range(ROWS * OFFSET + PROMPT))
    rows = rows_from_ids(ids)
    ok = [len(rows) == ROWS and all(len(r) == PROMPT for r in rows) and rows[1][0] == OFFSET,
          digest(rows) == digest(rows_from_ids(ids))]
    try:
        rows_from_ids(list(range(100)))
        ok.append(False)
    except SystemExit:
        ok.append(True)
    print(f"sc2_prompts self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


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
    rows = rows_from_ids(tok(text, return_tensors="pt").input_ids[0].tolist())
    eos = sorted({int(x) for x in ([tok.eos_token_id] if tok.eos_token_id is not None else [])})
    json.dump({"model": a.model, "revision": a.revision, "rows": rows, "prompts_sha256": digest(rows), "offset": OFFSET,
               "prompt": PROMPT, "eos_ids": eos}, open(a.out, "w"))
    print(f"SC2_PROMPTS rows={len(rows)} prompt={PROMPT} sha256={digest(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
