#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc5_ref.py -- lane SC5 (#1478 item 4): the ONE common bf16 reference, computed once and stored with its sha256.

**The windows.** Each window is ``prompt_len + steps + 1`` token ids from the lane's committed windows file. The scored
targets are ``ids[prompt_len + 1 .. prompt_len + steps]`` (K8's convention, and SC1's ``prefill_nll``). That is
``sc5_quality``'s range ``start = prompt_len + 1``, ``count = steps``.

**The forward.** ``Qwen/Qwen3-30B-A3B`` in bf16 through transformers, with ``device_map="auto"`` over the card and the host
RAM and eager attention. It is ONE forward per window, with no cache and no chunking, in the manner of
``step_decomp.ppl_oracle_score_full``. Chunking reorders an MoE's arithmetic and flips a few percent of router choices
(6.77 % on Qwen3, METHODOLOGY 13.1), so the full forward is the order-free reference. ``--chunked`` scores the same
windows through the cache in chunks instead: the bf16 ordering floor that every argmax agreement is read against. Row
``j`` of the logits predicts ``ids[j + 1]``.

**Recorded per scored position:** the NLL of the true token, from fp32 log-softmax, and the argmax id. The output names
the windows' sha256 and the model revision, and its own sha256 is written beside it (``<out>.sha256``). Every SC5 draw
verifies that hash before scoring.

    sc5_ref.py --windows windows.json --model Qwen/Qwen3-30B-A3B --revision REV --out ref.json [--chunked 256]
    sc5_ref.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys


def score_full(model, ids, prompt_len: int, steps: int) -> list:
    """[(nll, argmax_id)] for the targets ids[prompt_len+1 .. prompt_len+steps], from one forward with no cache."""
    import torch
    end = prompt_len + steps + 1
    if len(ids) < end:
        raise ValueError(f"window holds {len(ids)} ids, needs {end}")
    dev = next(model.parameters()).device
    t = torch.tensor(ids[:end], dtype=torch.long, device=dev)[None]
    with torch.no_grad():
        logits = model(input_ids=t, use_cache=False).logits[0, prompt_len:end - 1].float().cpu()
    return _pairs(logits, ids[prompt_len + 1:end])


def score_chunked(model, ids, prompt_len: int, steps: int, chunk: int) -> list:
    """The same targets through the model's cache: prefill ids[:prompt_len+1], then the rest in ``chunk``-token pieces.
    The same mathematics in another order: the bf16 ordering floor."""
    import torch
    end = prompt_len + steps + 1
    if len(ids) < end or chunk < 1:
        raise ValueError(f"window holds {len(ids)} ids, needs {end}; chunk {chunk}")
    dev = next(model.parameters()).device
    t = torch.tensor(ids[:end], dtype=torch.long, device=dev)
    rows = []
    with torch.no_grad():
        out = model(input_ids=t[None, :prompt_len + 1], use_cache=True)
        cache = out.past_key_values
        rows.append(out.logits[0, -1:].float().cpu())                  # predicts ids[prompt_len + 1]
        pos = prompt_len + 1
        while pos < end - 1:
            n = min(chunk, end - 1 - pos)
            out = model(input_ids=t[None, pos:pos + n], use_cache=True, past_key_values=cache,
                        position_ids=torch.arange(pos, pos + n, device=dev)[None])
            cache = out.past_key_values
            rows.append(out.logits[0].float().cpu())
            pos += n
    return _pairs(torch.cat(rows), ids[prompt_len + 1:end])


def _pairs(logits, targets) -> list:
    import torch
    lp = torch.log_softmax(logits, -1)
    tgt = torch.tensor(targets, dtype=torch.long)
    if lp.shape[0] != tgt.numel():
        raise ValueError(f"{lp.shape[0]} logit rows for {tgt.numel()} targets")
    nll = (-lp.gather(1, tgt[:, None])[:, 0]).tolist()
    am = lp.argmax(-1).tolist()
    return [(round(float(a), 6), int(b)) for a, b in zip(nll, am)]


def windows_sha256(windows: list) -> str:
    return hashlib.sha256(json.dumps(windows, separators=(",", ":")).encode()).hexdigest()


def build(model, wf: dict, chunk: int | None) -> dict:
    ws, pl, st = wf["windows"], int(wf["prompt_len"]), int(wf["steps"])
    if windows_sha256(ws) != wf["windows_sha256"]:
        raise ValueError("the windows file's sha256 does not match its windows")
    pos = [score_chunked(model, w, pl, st, chunk) if chunk else score_full(model, w, pl, st) for w in ws]
    return {"windows_sha256": wf["windows_sha256"], "prompt_len": pl, "steps": st, "windows": len(ws),
            "order": f"chunked {chunk}" if chunk else "full forward", "positions": pos}


def write(rec: dict, out: str) -> str:
    body = json.dumps(rec, separators=(",", ":"), sort_keys=True).encode()
    open(out, "wb").write(body)
    sha = hashlib.sha256(body).hexdigest()
    open(out + ".sha256", "w").write(f"{sha}  {out.replace(chr(92), '/').rsplit('/', 1)[-1]}\n")
    return sha


def _tiny_model(seed: int = 0):
    """A two-layer random Qwen3-MoE on the CPU in fp32: the self-test's stand-in for the 61 GB checkpoint."""
    import torch
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3MoeConfig(vocab_size=97, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         num_experts=8, num_experts_per_tok=2, max_position_embeddings=256, decoder_sparse_step=1,
                         mlp_only_layers=[])
    cfg._attn_implementation = "eager"
    return Qwen3MoeForCausalLM(cfg).eval()


def self_test() -> int:
    import torch
    ok = []
    model = _tiny_model()
    g = torch.Generator().manual_seed(1)
    ws = [torch.randint(0, 97, (12,), generator=g).tolist() for _ in range(3)]
    pl, st = 6, 5
    full = [score_full(model, w, pl, st) for w in ws]
    ok.append(all(len(p) == st for p in full))
    with torch.no_grad():                                         # the convention, against a hand computation
        lg = torch.log_softmax(model(input_ids=torch.tensor(ws[0])[None]).logits[0].float(), -1)
    want = [(round(float(-lg[j, ws[0][j + 1]]), 6), int(lg[j].argmax())) for j in range(pl, pl + st)]
    ok.append(full[0] == want)
    chunked = [score_chunked(model, w, pl, st, 2) for w in ws]
    ok.append(all(len(p) == st for p in chunked) and all(abs(a[0] - b[0]) < 1e-4 for f, c in zip(full, chunked)
                                                          for a, b in zip(f, c)))   # fp32: the orders agree closely
    wf = {"windows": ws, "prompt_len": pl, "steps": st, "windows_sha256": windows_sha256(ws)}
    rec = build(model, wf, None)
    ok.append(rec["positions"] == full and rec["order"] == "full forward" and rec["windows"] == 3)
    bad = dict(wf, windows_sha256="0" * 64)
    try:
        build(model, bad, None)
        ok.append(False)
    except ValueError:
        ok.append(True)
    try:
        score_full(model, ws[0][:8], pl, st)
        ok.append(False)
    except ValueError:
        ok.append(True)
    import os
    import tempfile
    d = tempfile.mkdtemp()
    out = os.path.join(d, "ref.json")
    sha = write(rec, out)
    ok.append(hashlib.sha256(open(out, "rb").read()).hexdigest() == sha
              and open(out + ".sha256").read().split()[0] == sha and write(build(model, wf, None), out) == sha)
    print(f"sc5_ref self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--revision")
    ap.add_argument("--out")
    ap.add_argument("--chunked", type=int, default=0, help="score through the cache in chunks: the ordering floor")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.windows and a.revision and a.out):
        ap.error("--windows --revision --out, or --self-test")
    import torch
    import transformers
    model = transformers.AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=torch.bfloat16,
                                                              device_map="auto", attn_implementation="eager").eval()
    rec = build(model, json.load(open(a.windows)), a.chunked or None)
    rec.update(model=a.model, revision=a.revision, transformers=transformers.__version__, torch=torch.__version__)
    sha = write(rec, a.out)
    print(f"SC5_REF {rec['order']} windows={rec['windows']} sha256={sha}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
