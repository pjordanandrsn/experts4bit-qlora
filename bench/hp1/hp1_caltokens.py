#!/usr/bin/env python3
"""bench/hp1/hp1_caltokens.py -- HP1-PREREG Amendment 4: the calibration box's 48 CALIBRATION rows.

The registered dataset (tp4_alpaca.py's ds_alpaca.json) carries exactly 48 held-out rows, all of them HP1's. So the calibration
rows come from tc1_arm.py's own held-out POOL (pack_pools, TC1 amendment 39). It starts with the registered 48 held-out rows and
continues in tp4_alpaca.py's shuffled order from the sha-pinned source, beyond every train row. Here those examples are encoded
UNPACKED, exactly as HP1's held-out rows are (encode_rows, the same template, EOS and seq). Rows 1-48 must then reproduce HP1's
registered held-out rows byte-for-byte: that check ties the pool to HP1's tokens file, and rows 49-96 are the calibration rows,
disjoint from HP1's train and held-out sets.

tc1_arm.py's functions are used as they are (the file is pinned by a closure); nothing in it changes. Refuses (exit 13) on a short
pool, a prefix that does not reproduce HP1's rows, or a tokens file that is packed.
"""
import argparse
import hashlib
import importlib.util
import json
import sys


def load_tc1_arm(path):
    spec = importlib.util.spec_from_file_location("tc1_arm_for_caltokens", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def calibration_tokens(T, tok, ds, pack_src, hp1_tokens, n, start=None, count=None):
    """(tokens dict, record). `T` is tc1_arm; `hp1_tokens` the HP1 tokens file (its eval = HP1's n held-out rows, unpacked).
    The calibration rows are pool rows [start, start + count) (0-based; default [n, 2n), HP1 Amendment 4's rows 49-96). HP2 takes
    [96, 396), rows 97-396: past HP1's held-out rows and past hp1cal-5090-2's."""
    start = n if start is None else start
    count = n if count is None else count
    if start < n:
        raise ValueError(f"calibration rows must start at or after row {n + 1} (HP1's held-out rows are 1-{n})")
    if hp1_tokens.get("pack"):
        raise ValueError("the HP1 tokens file is packed; the calibration rows are unpacked like HP1's")
    if len(hp1_tokens["eval"]) != n:
        raise ValueError(f"the HP1 tokens file carries {len(hp1_tokens['eval'])} held-out rows, need {n}")
    template = hp1_tokens.get("template", "alpaca")
    eos = hp1_tokens.get("eos", "") or ""
    seq = int(hp1_tokens["seq"])
    _tpool, epool, pool_rec = T.pack_pools(pack_src, ds)
    rows = T.encode_rows(tok, epool, seq, template, eos)
    if len(rows) < start + count:
        raise ValueError(f"the held-out pool encodes to {len(rows)} rows of >= 8 tokens, need {start + count}")
    if rows[:n] != hp1_tokens["eval"]:
        bad = next(i for i in range(n) if rows[i] != hp1_tokens["eval"][i])
        raise ValueError(f"the pool's first {n} rows do not reproduce HP1's held-out rows (first difference at row {bad})")
    cal = dict(hp1_tokens)
    cal["eval"] = rows[start:start + count]
    cal["n_eval"] = count
    cal["sha256"] = hashlib.sha256(json.dumps({"train": cal["train"], "eval": cal["eval"]}, separators=(",", ":")).encode()).hexdigest()
    rec = {"held_out_pool_rows": [start + 1, start + count], "pool": pool_rec, "hp1_tokens_sha256": hp1_tokens.get("sha256"),
           "prefix_reproduces_hp1_held_out": True}
    cal["hp1_calibration"] = rec
    return cal, rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tc1-arm", required=True, help="the staged tc1_arm.py")
    ap.add_argument("--data", required=True, help="tp4_alpaca.py's ds_alpaca.json")
    ap.add_argument("--data-sha", required=True)
    ap.add_argument("--pack-src", required=True, help="tp4_alpaca.py's alpaca_data_cleaned.json (sha-pinned by tc1_arm.pack_pools)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--tokens", required=True, help="the HP1 tokens file (its 48 held-out rows)")
    ap.add_argument("--n", type=int, default=48, help="HP1's held-out rows (the pool's prefix that must reproduce them)")
    ap.add_argument("--start", type=int, default=None, help="the first calibration row, 0-based (default --n)")
    ap.add_argument("--count", type=int, default=None, help="how many calibration rows (default --n)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    T = load_tc1_arm(a.tc1_arm)
    if T.sha_bytes(open(a.data, "rb").read()) != a.data_sha:
        print(f"HP1 CALTOKENS REFUSED: {a.data} is not the registered dataset")
        return 13
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    try:
        cal, rec = calibration_tokens(T, tok, json.load(open(a.data)), a.pack_src, json.load(open(a.tokens)), a.n, a.start, a.count)
    except ValueError as e:
        print(f"HP1 CALTOKENS REFUSED: {e}")
        return 13
    json.dump(cal, open(a.out, "w"))
    print(f"HP1 CALTOKENS rows {rec['held_out_pool_rows'][0]}-{rec['held_out_pool_rows'][1]} of the held-out pool; prefix reproduces "
          f"HP1's {a.n}; sha={cal['sha256']}")
    print(cal["sha256"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
