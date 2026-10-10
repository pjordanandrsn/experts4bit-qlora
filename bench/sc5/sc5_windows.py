#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc5_windows.py -- lane SC5 (#1478 item 4): the quality windows, built once and committed before any SC5 data.

The text is joined as ``bench/p117/p117_box.windows`` (and K8) join it: the non-blank lines of wikitext-2-raw-v1 test,
joined with a blank line between them, then tokenized ONCE with the Qwen3 tokenizer. Window ``k`` is the token ids
``[k * stride, k * stride + prompt_len + steps + 1)``. ``prompt_len + steps + 1`` follows K8's convention, which
``sc5_ref`` and ``sc5_quality`` score: the targets are ``ids[prompt_len + 1 .. prompt_len + steps]``.

The output carries the windows, their sha256 (``sc5_ref.windows_sha256``: compact JSON) and the dataset and tokenizer
revisions. Every framework and the reference score exactly these token ids.

    sc5_windows.py --tokenizer Qwen/Qwen3-30B-A3B --tokenizer-revision REV [--dataset-revision REV]
                   --n 64 --prompt-len 512 --steps 128 [--stride 3072] --out windows.json
    sc5_windows.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys


def join_text(lines) -> str:
    """The K8 / P117 corpus join: the non-blank lines, separated by a blank line."""
    return "\n\n".join(t for t in lines if t.strip())


def cut(ids: list, n: int, prompt_len: int, steps: int, stride: int) -> list:
    width = prompt_len + steps + 1
    if n < 1 or prompt_len < 1 or steps < 1 or stride < width:
        raise ValueError(f"bad windows: n={n} prompt_len={prompt_len} steps={steps} stride={stride} (width {width})")
    out = []
    for k in range(n):
        w = ids[k * stride:k * stride + width]
        if len(w) != width:
            raise ValueError(f"window {k} has {len(w)} ids of {width}: the corpus is too short for n={n}")
        out.append([int(x) for x in w])
    return out


def windows_sha256(windows: list) -> str:
    return hashlib.sha256(json.dumps(windows, separators=(",", ":")).encode()).hexdigest()


def self_test() -> int:
    ok = []
    ok.append(join_text(["a", "  ", "", "b\n", " c"]) == "a\n\nb\n\n\n c")
    ids = list(range(1000))
    w = cut(ids, 3, 4, 3, 100)
    ok.append(w == [list(range(0, 8)), list(range(100, 108)), list(range(200, 208))])
    for bad in ((0, 4, 3, 100), (3, 4, 3, 7), (20, 4, 3, 100)):
        try:
            cut(ids, *bad)
            ok.append(False)
        except ValueError:
            ok.append(True)
    ok.append(windows_sha256(w) == windows_sha256(json.loads(json.dumps(w))) and windows_sha256(w) != windows_sha256(w[:2]))
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location("sc5_ref", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                          "sc5_ref.py"))
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    ok.append(ref.windows_sha256(w) == windows_sha256(w))                  # the reference checks the same digest
    print(f"sc5_windows self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--tokenizer-revision")
    ap.add_argument("--dataset-revision", default=None)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--stride", type=int, default=3072)
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.tokenizer_revision and a.out):
        ap.error("--tokenizer-revision --out, or --self-test")
    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", revision=a.dataset_revision)
    tok = AutoTokenizer.from_pretrained(a.tokenizer, revision=a.tokenizer_revision)
    ids = list(tok(join_text(ds["text"])).input_ids)          # the same ids as P117's return_tensors="pt" path, without torch
    ws = cut(ids, a.n, a.prompt_len, a.steps, a.stride)
    rec = {"windows": ws, "prompt_len": a.prompt_len, "steps": a.steps, "stride": a.stride, "n": a.n,
           "windows_sha256": windows_sha256(ws), "tokenizer": {"repo": a.tokenizer, "revision": a.tokenizer_revision},
           "dataset": {"repo": "Salesforce/wikitext", "config": "wikitext-2-raw-v1", "split": "test",
                       "revision": a.dataset_revision}}
    json.dump(rec, open(a.out, "w"), separators=(",", ":"))
    print(f"SC5_WINDOWS n={a.n} width={a.prompt_len + a.steps + 1} sha256={rec['windows_sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
