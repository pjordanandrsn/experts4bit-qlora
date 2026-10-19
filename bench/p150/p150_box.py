#!/usr/bin/env python3
"""p150_box.py -- lane P150's box (bench/p150/PREREG-p150.md; e4b#1313): the W8A16 split premise's ``i8_box.py``,
staged at its merged bytes, with ONE change: its wikitext windows. P150 rules on a HELD-OUT window set nobody has read.

P125's loader (``p125_box.wikitext_windows``) reads wikitext-2-raw-v1 TEST, and the split box already read I8HEAD on its
windows 0-195 (t1) and 0-122 (k16). This box reads the same corpus's VALIDATION split (disjoint articles) with the same
join and tokenisation. It keeps the same stride (1024), counts (t1 196, k16 123; k16's windows are t1's first 123) and
window (512 + 128). The windows' sha256s were registered before any data (``HELDOUT``). The windows check refuses the
box when they differ, and every quality record carries the hashes of the windows it read, which the reducer checks. c4
stays P125's (16 windows, reported only).

    python p150_box.py --windows-check --model M --revision R      (refuses unless the held-out hashes match)
    python p150_box.py --mode quality --out quality_X.json --ref-dir work/ref     (the arm from I8_ARM, as i8_box)
    python p150_box.py --self-test
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i8_box as ib  # noqa: E402  (staged at bench/i8dense's bytes: the arms, the round trip, P125's instrument)
import p125_box as pb  # noqa: E402  (staged at P125's pinned bytes)

#: registered before any data: the held-out windows' sha256 (JSON of each gate's token-id windows, no spaces), measured
#: with transformers 5.17.0 and Qwen3-30B-A3B's tokenizer at ad44e777. The same pipeline reproduces P125's 298,938 test
#: tokens; the validation split holds 262,210, room for 256 windows at stride 1024.
HELDOUT = {"split": "validation", "stride": 1024,
           "sha256": {"t1": "5c444982538d1aaba2b4f47c00509c0604a72b3845f5cb98c7b30d07b81a76ec",
                      "k16": "7ab52a502ffc5292802db52e4325f025688deeed735a8c04ce33fba5789acded"}}
RULED = {"t1": 196, "k16": 123}
_SEEN: dict = {}     # text -> {window count: sha256} for every window set this process read


def windows_sha256(ws) -> str:
    """The registered form: sha256 of the JSON of the windows' token ids, no spaces."""
    return hashlib.sha256(json.dumps([[int(t) for t in w] for w in ws], separators=(",", ":")).encode()).hexdigest()


def heldout_wikitext_windows(tok, n, prompt, cont, stride=HELDOUT["stride"]):
    """``p125_box.wikitext_windows`` with one change, the split: the same corpus, join and tokenisation; window k starts
    at token k * ``stride``."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=HELDOUT["split"])
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    out = []
    for k in range(n):
        w = ids[k * stride:k * stride + prompt + cont]
        assert w.numel() == prompt + cont, f"held-out window {k} has {w.numel()} tokens (stride {stride}, {ids.numel()} tokens)"
        out.append(w.tolist())
    return out


def windows(tok, text, n, a):
    """i8_box's ``windows`` with the held-out wikitext; records each window set's hash."""
    if text == "wikitext":
        ws = heldout_wikitext_windows(tok, n, a.prompt, a.cont, stride=a.wikitext_stride)
    else:
        ws = pb.loaders()[text](tok, n, a.prompt, a.cont)
    _SEEN.setdefault(text, {})[len(ws)] = windows_sha256(ws)
    return ws


def ref_sha256(ref_dir) -> dict:
    """Per gate, the sha256 of the reference log-probs (R) arm A wrote under ``ref_dir/<gate>/``: every file's name and
    bytes, in name order. A records it; I8HEAD records what it scored against, and the reducer holds the two equal."""
    out = {}
    if not ref_dir or not os.path.isdir(ref_dir):
        return out
    for gate in sorted(os.listdir(ref_dir)):
        d = os.path.join(ref_dir, gate)
        if not os.path.isdir(d):
            continue
        h = hashlib.sha256()
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                h.update(name.encode() + b"\0" + open(p, "rb").read() + b"\0")
        out[gate] = h.hexdigest()
    return out


def install() -> None:
    """Point i8_box at the held-out windows: its quality_main and windows_check call the module's ``windows``."""
    ib.windows = windows


def read_heldout() -> dict:
    """What this process read, in the registered form."""
    wiki = _SEEN.get("wikitext", {})
    return {"split": HELDOUT["split"], "stride": HELDOUT["stride"], "sha256": {g: wiki.get(n) for g, n in RULED.items()}}


def self_test() -> int:
    install()
    hexes = [h for h in HELDOUT["sha256"].values()]
    cases = [
        ("the held-out split is not P125's", HELDOUT["split"] == "validation"),
        ("the registered hashes are sha256s", all(len(h) == 64 and int(h, 16) >= 0 for h in hexes)),
        ("the ruled gates and counts are the split premise's", RULED == {g: ib.COUNTS[g] for g in ("t1", "k16")}),
        ("the stride is the split premise's", HELDOUT["stride"] == ib.WIKITEXT_STRIDE),
        ("the window is P125's", pb.WINDOW == 640),
        ("the hash is the registered form", windows_sha256([[1, 2], [3]]) == hashlib.sha256(b"[[1,2],[3]]").hexdigest()),
        ("i8_box reads the held-out windows", ib.windows is windows),
        ("nothing read yet records no hash", read_heldout()["sha256"] == {"t1": None, "k16": None}),
    ]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "t1"))
        open(os.path.join(d, "t1", "R_0.pt"), "wb").write(b"abc")
        h0 = ref_sha256(d)
        open(os.path.join(d, "t1", "R_0.pt"), "wb").write(b"abd")
        cases.append(("R's hash binds its bytes", set(h0) == {"t1"} and ref_sha256(d)["t1"] != h0["t1"]))
        cases.append(("no reference directory hashes to nothing", ref_sha256(os.path.join(d, "absent")) == {}))
    bad = [n for n, ok in cases if not ok]
    print(f"p150_box self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--mode")
    p.add_argument("--out")
    p.add_argument("--windows-check", action="store_true")
    p.add_argument("--ref-dir")
    a, _ = p.parse_known_args(argv)
    install()
    scoring = a.mode == "quality" and os.environ.get("I8_ARM", "A") != "A"
    before = ref_sha256(a.ref_dir) if scoring else None      # the R this arm will score against (A wrote it)
    rc = ib.main(argv)
    got = read_heldout()
    if a.windows_check:
        ok = rc == 0 and got == HELDOUT
        print("P150_HELDOUT " + json.dumps({"read": got, "registered": HELDOUT, "match": ok}), flush=True)
        return 0 if ok else 1
    if a.mode == "quality" and rc == 0:
        refs = ref_sha256(a.ref_dir)
        if scoring and refs != before:
            raise SystemExit(f"REFUSED: the reference log-probs changed while this arm scored ({before} -> {refs})")
        rec = json.load(open(a.out))
        rec["heldout"], rec["ref_sha256"] = got, refs
        json.dump(rec, open(a.out, "w"), indent=1, default=str)
        print("P150_QUALITY_HELDOUT " + json.dumps({"heldout": got, "ref_sha256": refs}), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
