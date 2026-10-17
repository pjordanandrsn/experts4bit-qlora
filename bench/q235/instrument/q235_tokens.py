# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The step's token ids, from TC1's field-recipe rows (``tp4_alpaca.py``'s file) rendered as TC1 renders them (``tc1_arm.render_row``,
the Alpaca template with EOS), tokenized by the checkpoint's own tokenizer, truncated to ``--seq`` and right-padded to the longest row.
Writes ``input_ids``, ``attention_mask`` and ``labels`` (-100 on pads) as one torch file, and their sha256 beside it.

``--heldout-out`` (Amendment 8) also writes the registered held-out rows: the next ``--rows`` rows of the same split, from
``--first + --rows``, rendered, tokenized and padded the same way and disjoint from the step's by construction."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve()


def _tc1_arm():
    """TC1's renderer: beside this file on a box (the flat staged closure), else in the repository's bench/tc1."""
    beside = HERE.with_name("tc1_arm.py")
    spec = importlib.util.spec_from_file_location("tc1_arm_for_q235", beside if beside.is_file() else HERE.parents[2] / "tc1" / "tc1_arm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(ds_path, tokenizer_dir, rows, seq, first=0):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    ds = json.load(open(ds_path))
    arm = _tc1_arm()
    picked = ds["train"][first:first + rows]
    ids = arm.encode_rows(tok, picked, seq, template="alpaca", eos=tok.eos_token or "")
    assert len(ids) == rows, f"{rows - len(ids)} rows tokenized under 8 tokens"
    L = max(len(x) for x in ids)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    input_ids = torch.full((rows, L), pad, dtype=torch.long)
    mask = torch.zeros((rows, L), dtype=torch.long)
    for i, x in enumerate(ids):
        input_ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    labels = input_ids.masked_fill(mask == 0, -100)
    t = {"input_ids": input_ids, "attention_mask": mask, "labels": labels}
    sha = {k: hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k, v in t.items()}
    return t, {"ds_sha256": hashlib.sha256(open(ds_path, "rb").read()).hexdigest(), "rows": rows, "first": first, "seq": seq,
               "padded_len": L, "real_tokens": int(mask.sum()), "pad_id": pad, "sha256": sha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds", help="tp4_alpaca.py's output (ds_alpaca.json)")
    ap.add_argument("tokenizer", help="the checkpoint directory (its tokenizer files)")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout-out", default=None, help="Amendment 8: also write the next --rows rows, the held-out read's")
    a = ap.parse_args()
    t, rec = build(a.ds, a.tokenizer, a.rows, a.seq, a.first)
    torch.save(t, a.out)
    json.dump(rec, open(a.out + ".json", "w"), indent=1)
    if a.heldout_out:
        h, hrec = build(a.ds, a.tokenizer, a.rows, a.seq, a.first + a.rows)
        torch.save(h, a.heldout_out)
        json.dump(hrec, open(a.heldout_out + ".json", "w"), indent=1)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
