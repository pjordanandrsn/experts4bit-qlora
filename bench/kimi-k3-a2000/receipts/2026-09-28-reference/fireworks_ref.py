"""Reference scores for the A2000 Kimi-K3 run, from Fireworks' serverless kimi-k3.

Fireworks serves K3 from its native MXFP4 expert weights (the bytes the A2000 arena
holds) with MXFP8 activations, where the A2000 path runs bf16 activations. Its raw
completions endpoint takes a prompt with NO chat template and, with `echo`, returns the
log probability of every prompt token, so it can score exactly what k3_run_rel.py scored:

  gen -- greedy continuation of the raw prompt, with each token's probability
  ppl -- mean NLL over the same paragraph's tokens (position 0 has no prediction)

Stdlib only. The key is read from FIREWORKS_API_KEY or ~/.fireworks/secrets.env and is
never printed. Writes fireworks_ref.json (raw responses + derived numbers), and compares
against the A2000 JSONs when their paths are given.

    python fireworks_ref.py [--ours-gen k3_gen_n4_pin0.json] [--ours-ppl k3_ppl_n0_pin0.json]
"""
import argparse
import json
import math
import os
import pathlib
import time
import urllib.request

URL = "https://api.fireworks.ai/inference/v1/completions"
MODEL = "accounts/fireworks/models/kimi-k3"
PROMPT = "The capital city of France is"
PPL_TEXT = (
    "The capital city of France is Paris. It sits on the river Seine and has "
    "been the political and cultural centre of the country for centuries. The "
    "city is known for its museums, its boulevards, and the cathedral of Notre "
    "Dame, which stands on an island in the middle of the river. Millions of "
    "visitors arrive each year to see the Louvre and the Eiffel Tower, and the "
    "surrounding region remains the most densely populated part of France.")


def key():
    k = os.environ.get("FIREWORKS_API_KEY")
    if k:
        return k
    p = pathlib.Path.home() / ".fireworks" / "secrets.env"
    if not p.exists():
        raise SystemExit(f"no FIREWORKS_API_KEY in the environment and no {p}")
    for line in p.read_text().splitlines():
        if line.startswith("FIREWORKS_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit(f"no FIREWORKS_API_KEY in the environment or {p}")


def post(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {key()}", "Content-Type": "application/json",
        "Accept": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.loads(r.read())
    out["_wall_s"] = time.time() - t0
    return out


def token_logprobs(choice):
    """(tokens, logprobs, top) from either logprobs schema Fireworks may return."""
    lp = choice.get("logprobs") or {}
    if "tokens" in lp:                                   # legacy completions shape
        return lp["tokens"], lp["token_logprobs"], lp.get("top_logprobs")
    content = lp.get("content")
    if content is not None:                              # chat-style shape
        return ([c["token"] for c in content], [c["logprob"] for c in content],
                [{t["token"]: t["logprob"] for t in c.get("top_logprobs", [])}
                 for c in content])
    raise SystemExit(f"unrecognised logprobs shape: {json.dumps(lp)[:400]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours-gen")
    ap.add_argument("--ours-ppl")
    ap.add_argument("--out", default="fireworks_ref.json")
    a = ap.parse_args()

    gen = post({"model": MODEL, "prompt": PROMPT, "max_tokens": 4, "temperature": 0,
                "logprobs": True, "top_logprobs": 5, "echo": False})
    toks, lps, top = token_logprobs(gen["choices"][0])
    g = {"text": gen["choices"][0]["text"], "tokens": toks,
         "probs": [math.exp(x) for x in lps], "top": top}
    print(f"gen: {g['text']!r}")
    for t, p in zip(toks, g["probs"]):
        print(f"  {t!r:14s} p={p*100:.2f}%")

    ppl = post({"model": MODEL, "prompt": PPL_TEXT, "max_tokens": 1, "temperature": 0,
                "logprobs": True, "top_logprobs": 5, "echo": True})
    ptoks, plps, _ = token_logprobs(ppl["choices"][0])
    n_prompt = ppl.get("usage", {}).get("prompt_tokens")
    # With echo, the prompt tokens come first, then the 1 generated token. Position 0 of
    # the prompt has no prediction (null logprob), exactly like the A2000 run's window.
    body = list(zip(ptoks[:n_prompt], plps[:n_prompt])) if n_prompt else list(zip(ptoks, plps))
    scored = [lp for _t, lp in body if lp is not None]
    nll = -sum(scored) / len(scored)
    p = {"prompt_tokens": n_prompt, "first_tokens": [t for t, _ in body[:3]],
         "predictions": len(scored), "nll": nll, "perplexity": math.exp(nll),
         "per_token": body}
    print(f"ppl: {len(body)} prompt tokens, {len(scored)} predictions, "
          f"NLL {nll:.4f}, perplexity {math.exp(nll):.3f}   first tokens {p['first_tokens']}")

    cmp = {}
    if a.ours_gen:
        o = json.loads(pathlib.Path(a.ours_gen).read_text())
        cmp["gen"] = {"ours_text": o["text"], "ref_text": g["text"],
                      "text_agrees": o["text"] == g["text"],
                      "ours_probs": o.get("probs"), "ref_probs": g["probs"]}
        print(f"compare gen: ours {o['text']!r} vs ref {g['text']!r}")
    if a.ours_ppl:
        o = json.loads(pathlib.Path(a.ours_ppl).read_text())
        cmp["ppl"] = {"ours": {"tokens": o["tokens"], "predictions": o["predictions"],
                               "nll": o["nll"], "perplexity": o["perplexity"]},
                      "ref": {"prompt_tokens": n_prompt, "predictions": len(scored),
                              "nll": nll, "perplexity": math.exp(nll)},
                      "same_token_count": o["tokens"] == len(body)}
        print(f"compare ppl: ours {o['perplexity']:.3f} ({o['tokens']} tokens) vs ref "
              f"{math.exp(nll):.3f} ({len(body)} tokens)"
              + ("" if o["tokens"] == len(body) else
                 "   *** TOKEN COUNTS DIFFER -- tokenization or a BOS differs; "
                 "the two NLLs are not over the same predictions ***"))

    pathlib.Path(a.out).write_text(json.dumps({
        "model": MODEL, "endpoint": URL,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gen": g, "ppl": p, "compare": cmp, "raw": {"gen": gen, "ppl": ppl}}, indent=1))
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
