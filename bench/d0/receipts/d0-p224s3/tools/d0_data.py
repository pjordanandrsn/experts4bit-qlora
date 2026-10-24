"""D0 Step 2 data: prove the held-out rows unseen from a fixture's receipt, then encode them as the fixtures trained (Alpaca's pinned text column; see the amendment).

PREREG-d0.md "Data": re-encoding rows [0, examples_used) through Loggetta's own demonstration path must reproduce the
receipt's token_stream_sha256 (that proves which rows were fed, in order, and that this dataset copy has the same
bytes); training must have read only rows below 51,938; the held-out rows are 51,938-52,001, each encoded by the
demo's own encode_example (alpaca text + EOS), truncated at 512, scored on every token after the first.

    python d0_data.py --receipt RUN_ID.json --tokenizer ADAPTER_DIR --out heldout.json
"""
import argparse
import json
import sys

DATASET = "tatsu-lab/alpaca"
DATASET_REVISION = "dce01c9b08f87459cf36a430d809084718273017"
HELD_OUT = (51938, 52002)
MAX_LEN = 512


def find(d, key):
    """The first value under ``key`` anywhere in a nested receipt."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find(v, key)
            if r is not None:
                return r
    elif isinstance(d, list):
        for v in d:
            r = find(v, key)
            if r is not None:
                return r
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from loggetta.data import TrainingData, encode_example, load_rows, prepare_data, resolve_format

    rec = json.load(open(a.receipt))
    data = rec["data"]
    opts = dict(data["options"])
    if opts.get("source") != DATASET or data.get("shuffled") or data.get("repeated") or data.get("passes") != 1:
        print("D0_DATA VOID: not the unshuffled single-pass demo " + json.dumps(
            {"source": opts.get("source"), **{k: data.get(k) for k in ("shuffled", "repeated", "passes")}}))
        return 1
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    if opts.get("revision") != DATASET_REVISION:
        print(f"D0_DATA VOID: dataset revision {opts.get('revision')} is not the pinned {DATASET_REVISION}")
        return 1
    spec = TrainingData.from_dict(opts)
    profile = (rec.get("plan") or {}).get("data_profile")
    # The plan recorded the tokenizer by its hub NAME; ours is the same files read from the fixture folder, so its
    # name_or_path is a path. Every other identity field must match as recorded; only then is the name aligned, and
    # Loggetta's own profile check (encoded_sha256, rows, tokens, format, options) still decides.
    from loggetta.data import tokenizer_identity
    want, have = dict((profile or {}).get("tokenizer") or {}), tokenizer_identity(tok)
    if {k: v for k, v in have.items() if k != "name"} != {k: v for k, v in want.items() if k != "name"}:
        print("D0_DATA VOID: tokenizer identity differs from the plan's " + json.dumps({"plan": want, "ours": have}))
        return 1
    tok.name_or_path = want.get("name")
    prepared = prepare_data(tok, data["blocks"], data["seq_len"], spec, seed=data["seed"], expect=profile)
    got = prepared.info
    out = {"dataset": DATASET, "dataset_revision": DATASET_REVISION,
           "receipt": {k: data.get(k) for k in ("examples_used", "token_stream_sha256", "tokens", "blocks", "seq_len")},
           "reencoded": {k: got.get(k) for k in ("examples_used", "token_stream_sha256", "tokens")},
           "fingerprint_receipt": find(data, "dataset_fingerprint"), "fingerprint_now": find(got, "dataset_fingerprint")}
    out["hash_ok"] = got["token_stream_sha256"] == data["token_stream_sha256"] and got["examples_used"] == data["examples_used"]
    out["unseen_ok"] = data["examples_used"] <= HELD_OUT[0]
    out["fingerprint_ok"] = (out["fingerprint_receipt"] is None) or out["fingerprint_receipt"] == out["fingerprint_now"]
    ds, _ = load_rows(spec)
    fmt = resolve_format(spec, getattr(ds, "column_names", None) or list(ds[0]))
    out["format"] = fmt
    rows = []
    for i in range(*HELD_OUT):
        ids = encode_example(tok, ds[i], spec, fmt)[:MAX_LEN]
        rows.append({"row": i, "ids": ids})
    out["rows"] = rows
    out["scored_tokens"] = sum(len(r["ids"]) - 1 for r in rows)
    out["ok"] = out["hash_ok"] and out["unseen_ok"] and out["fingerprint_ok"]
    json.dump(out, open(a.out, "w"))
    print("D0_DATA " + json.dumps({k: out[k] for k in ("hash_ok", "unseen_ok", "fingerprint_ok", "ok", "scored_tokens")}))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
