# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""HC explore box client (bench/serve-hybrid/DESIGN-HC.md, section 5). EXPLORE, never a reading.

``run``: send the fixed prompts to one serve_paged, serially, greedy (temperature 0, ignore_eos, a fixed max_tokens), and
record each completion's text and wall time. ``compare``: the texts of one arm against another's, per prompt: identical, or
the first diverging character and the first diverging token (both texts re-tokenized with the model's tokenizer). ``tiers``:
the solver's own split for an arm's environment, recomputed as build_engine computes it (the server does not report it).
``--self-test``: the compare rule on fixed cases, no network, no model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

PROMPTS = [
    "The capital of France is",
    "Write a haiku about the ocean.",
    "Explain what a hash table is in one paragraph.",
    "List the first ten prime numbers.",
    "def fibonacci(n):",
    "Translate 'good morning' into Spanish, French and German.",
    "What causes the seasons on Earth?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "A train leaves at 3 pm travelling 60 km/h. How far has it gone by 5:30 pm?",
    "Give three tips for writing clear code.",
]


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run(args) -> int:
    out = {"arm": args.arm, "max_tokens": args.max_tokens, "prompts": PROMPTS, "results": []}
    for i, p in enumerate(PROMPTS):
        t0 = time.time()
        r = post(f"{args.url}/v1/completions", {"model": args.model, "prompt": p, "max_tokens": args.max_tokens,
                                                "temperature": 0, "ignore_eos": True}, args.timeout)
        text = r["choices"][0]["text"]
        out["results"].append({"i": i, "text": text, "wall_s": round(time.time() - t0, 3),
                               "usage": r.get("usage")})
        print(f"[hc] {args.arm} prompt {i}: {len(text)} chars in {time.time() - t0:.1f} s", flush=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1)
    return 0


def first_diff(a, b):
    """Index of the first differing element of two sequences, or None when equal (a strict prefix differs at its end)."""
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k
    return None if len(a) == len(b) else min(len(a), len(b))


def compare_texts(base: list, other: list, encode=None) -> dict:
    """Per prompt: identical, or the first diverging character (and token, when an encoder is given). Fails closed: the
    two runs must hold the same number of prompts, and an empty run is refused."""
    if not base or len(base) != len(other):
        raise ValueError(f"runs not comparable: {len(base)} vs {len(other)} results")
    rows = []
    for a, b in zip(base, other):
        c = first_diff(a, b)
        row = {"identical": c is None, "first_char": c}
        if encode is not None and c is not None:
            row["first_token"] = first_diff(encode(a), encode(b))
        rows.append(row)
    return {"n": len(rows), "identical": sum(r["identical"] for r in rows), "rows": rows}


def compare(args) -> int:
    base = json.load(open(args.base, encoding="utf-8"))
    other = json.load(open(args.other, encoding="utf-8"))
    enc = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    res = compare_texts([r["text"] for r in base["results"]], [r["text"] for r in other["results"]], enc)
    res.update({"base": base["arm"], "other": other["arm"]})
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=1)
    firsts = [r.get("first_token", r["first_char"]) for r in res["rows"] if not r["identical"]]
    print(f"[hc] {other['arm']} vs {base['arm']}: {res['identical']}/{res['n']} identical"
          + (f"; first diverging token per differing prompt: {firsts}" if firsts else ""), flush=True)
    return 0


def tiers(args) -> int:
    from experts4bit_qlora.engines.placement import solve_placement
    from experts4bit_qlora.serve_paged import _bytes_per_expert
    cfg = json.load(open(args.config, encoding="utf-8"))
    L, E, k = cfg["num_hidden_layers"], cfg["num_experts"], cfg["num_experts_per_tok"]
    man = solve_placement(n_layers=L, n_experts=E, bytes_per_expert=_bytes_per_expert(args.arena),
                          vram_budget_bytes=int(args.vram_gb * 2**30), dram_budget_bytes=int(args.dram_gb * 2**30),
                          calibration=json.load(open(args.calib, encoding="utf-8")), profile_path=None,
                          batch=1, top_k=k, cpu_us_fixed=None, cpu_us_per_row=None)
    split = {t: len(man["tiers"][t]) for t in ("vram", "dram", "nvme")}
    json.dump({"arm": args.arm, "vram_gb": args.vram_gb, "dram_gb": args.dram_gb, "experts": split,
               "masses": man.get("masses")}, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"[hc] {args.arm} solver split (experts): {split}", flush=True)
    return 0


def self_test() -> int:
    n = 0
    for a, b, want in (("abc", "abc", None), ("abc", "abd", 2), ("ab", "abc", 2)):
        assert first_diff(a, b) == want, (a, b)
        n += 1
    r = compare_texts(["x", "hello"], ["x", "help!"], encode=lambda s: list(s))
    assert r["identical"] == 1 and r["rows"][1] == {"identical": False, "first_char": 3, "first_token": 3}
    n += 1
    for bad in (([], []), (["a"], ["a", "b"])):
        try:
            compare_texts(*bad)
        except ValueError:
            n += 1
        else:
            raise AssertionError(f"compare_texts accepted {bad}")
    print(f"hc_client self-test OK ({n} checks)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    for name in ("--url", "--model", "--arm", "--out"):
        r.add_argument(name, required=True)
    r.add_argument("--max-tokens", type=int, default=32)
    r.add_argument("--timeout", type=float, default=600)
    c = sub.add_parser("compare")
    for name in ("--base", "--other", "--out"):
        c.add_argument(name, required=True)
    c.add_argument("--tokenizer", default="")
    t = sub.add_parser("tiers")
    for a in ("--arm", "--config", "--arena", "--calib", "--out"):
        t.add_argument(a, required=True)
    for name in ("--vram-gb", "--dram-gb"):
        t.add_argument(name, type=float, required=True)
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    return {"run": run, "compare": compare, "tiers": tiers}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
