#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""sc5_e4b_quality.py -- lane SC5 (#1478 item 4): e4b's per-position quality on the SERVED weights and kernels.

Both columns score the windows file's targets ``ids[prompt_len + 1 .. prompt_len + steps]`` (K8's convention). Each
position yields (the true token's fp32 NLL, the argmax id), the same pair ``sc5_ref`` and ``sc5_quality`` produce.

- **Prefill-shaped**, the cross-framework comparison (decided 2026-10-09). The whole window goes through the paged
  runner's ``run_prefill`` in the served ``chunk``-token pieces (512), and the logit rows for the scored positions are
  kept from whichever piece holds them. The 641-token window is 512 + 129 here, where vLLM and SGLang take it in one
  piece at their default chunk sizes. That is e4b's served prefill arithmetic, stated in the registration.
  The scoring runner takes the first piece eagerly. The server replays that piece through its first-chunk prefill CUDA
  graph, which replays the captured forward's kernels, so the arithmetic is expected to be identical. That is an
  expectation, not a measurement, and the proof run checks it (graph against eager on the first piece's last row).
- **Decode-shaped**, reported as the served arithmetic: ``bench/p117/p117_box.paged_pass`` at its registered bytes, with
  P = ``prompt_len + 1`` and C = ``steps``. The prefill's last logit predicts ``ids[prompt_len + 1]``, and each decode
  step predicts the next.

The model is the one ``serve_paged``'s ``build_engine`` serves at every default, built by the SC5 box. This module only
scores it. ``pairs`` and ``rows_for`` are pure and CPU-tested; the passes need the card.

The box's command, run with the SAME environment the server was started with (the E4B_PAGED_* knobs, the arena, the
calibration), and no server running:

    sc5_e4b_quality.py run --windows windows.json --prefill-out q_e4b.json [--decode-out q_e4b_decode.json]
    sc5_e4b_quality.py --self-test

``run`` builds the engine through ``serve_paged.build_engine(PagedServeConfig.from_env())``, so the weights, kernels,
chunk size and buckets are what the server serves, then scores the runner's model. Each output has the shape
``sc5_quality.score`` writes (``windows_sha256``, ``prompt_len``, ``steps``, ``positions``, or ``error``).
"""
from __future__ import annotations

import sys


def rows_for(chunks: list, prompt_len: int, steps: int) -> list:
    """From ``[(start, logits [take, V])]`` in order, the logit rows for positions prompt_len .. prompt_len+steps-1.
    Row r of a chunk starting at ``start`` is position start + r, which predicts ids[start + r + 1]."""
    want = range(prompt_len, prompt_len + steps)
    got = {}
    for start, lg in chunks:
        for r in range(lg.shape[0]):
            p = start + r
            if p in want:
                if p in got:
                    raise ValueError(f"position {p} appears in two chunks")
                got[p] = lg[r]
    missing = [p for p in want if p not in got]
    if missing:
        raise ValueError(f"no logit row for positions {missing[:3]}{'...' if len(missing) > 3 else ''}")
    return [got[p] for p in want]


def pairs(rows: list, targets: list) -> list:
    """[(nll, argmax_id)] from fp32 log-softmax of each row against its target."""
    import torch
    if len(rows) != len(targets):
        raise ValueError(f"{len(rows)} rows for {len(targets)} targets")
    lp = torch.log_softmax(torch.stack([r.float() for r in rows]), -1)
    tgt = torch.tensor(targets, dtype=torch.long)
    nll = (-lp.gather(1, tgt[:, None])[:, 0]).tolist()
    return [(round(float(a), 6), int(b)) for a, b in zip(nll, lp.argmax(-1).tolist())]


def prefill_pass(model, ws: list, prompt_len: int, steps: int, chunk: int, device) -> list:
    """Prefill-shaped: each window through the paged runner's served prefill pieces; per window [(nll, argmax)]."""
    import torch
    from experts4bit_qlora.engines.fp8_paged_kv import Fp8PagedKV
    from experts4bit_qlora.engines.paged_runner import PagedModelRunner, kv_layers
    from experts4bit_qlora.serve_paged import _kv_geometry
    cfg = getattr(model.config, "text_config", None) or model.config
    hkv, hd = _kv_geometry(model.config)
    width = prompt_len + steps + 1
    kv = Fp8PagedKV(kv_layers(model, int(cfg.num_hidden_layers)), hkv, hd, batch=len(ws), max_tokens_per_seq=width + 16,
                    device=device, scratch_slots=1)
    runner = PagedModelRunner(model, kv, device=device)
    inner, kept = model.forward, []

    def keep(*a, **kw):
        o = inner(*a, **kw)
        kept.append(o.logits)
        return o
    out = []
    try:
        model.forward = keep
        with torch.no_grad():
            for rid, w in enumerate(ws):
                if len(w) != width:
                    raise ValueError(f"window {rid} holds {len(w)} ids, expected {width}")
                runner.bind(rid, rid, w)
                chunks = []
                for s in range(0, width, chunk):
                    take = min(chunk, width - s)
                    kept.clear()
                    runner.run_prefill([(rid, s, take)])
                    if len(kept) != 1 or kept[0].shape[-2] != take:
                        raise RuntimeError(f"window {rid} piece at {s}: expected one forward of {take} rows, got "
                                           f"{[tuple(k.shape) for k in kept]} (a graph replay or last-logits prefill)")
                    chunks.append((s, kept[0][0].float().cpu()))
                out.append(pairs(rows_for(chunks, prompt_len, steps), w[prompt_len + 1:width]))
    finally:
        model.forward = inner
    del runner, kv
    return out


def decode_pass(model, ws: list, prompt_len: int, steps: int, chunk: int, device, buckets) -> list:
    """Decode-shaped: P117's ``paged_pass`` (staged at its registered bytes) with P = prompt_len + 1, C = steps."""
    import p117_box
    lps, _tk, _eng = p117_box.paged_pass(model, [w[:prompt_len + steps + 1] for w in ws], prompt_len + 1, steps, chunk,
                                         device, buckets=buckets)
    return [pairs(list(lp), w[prompt_len + 1:prompt_len + steps + 1]) for lp, w in zip(lps, ws)]


def _record(wf: dict, order: str, fn) -> dict:
    out = {"framework": "e4b", "windows_sha256": wf["windows_sha256"], "prompt_len": int(wf["prompt_len"]),
           "steps": int(wf["steps"]), "windows": len(wf["windows"]), "order": order}
    try:
        out["positions"] = [[(round(float(a), 6), int(b)) for a, b in w] for w in fn()]
    except Exception as e:  # noqa: BLE001 - an incomplete pass is that row's VOID, recorded with its reason
        out["error"] = f"{type(e).__name__}: {e}"[:500]
    return out


def run(windows: str, prefill_out: str, decode_out: str | None) -> int:
    import json

    import torch
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    wf = json.load(open(windows))
    pl, st, ws = int(wf["prompt_len"]), int(wf["steps"]), [list(w) for w in wf["windows"]]
    cfg = PagedServeConfig.from_env()
    parts = build_engine(cfg)
    model, device = parts.runner.model, torch.device(cfg.device)
    served = f"served chunk {cfg.chunk_tokens}, buckets {list(cfg.buckets)}, max_seqs {cfg.max_seqs}"
    rc = 0
    for path, order, fn in (
            (prefill_out, f"prefill-shaped ({served})",
             lambda: prefill_pass(model, ws, pl, st, cfg.chunk_tokens, device)),
            (decode_out, f"decode-shaped, P117 paged_pass ({served})",
             lambda: decode_pass(model, ws, pl, st, cfg.chunk_tokens, device, list(cfg.buckets)))):
        if not path:
            continue
        rec = _record(wf, order, fn)
        json.dump(rec, open(path, "w"), separators=(",", ":"), sort_keys=True)
        print(f"SC5_E4B_QUALITY {order.split(' ')[0]} " + (f"ERROR {rec['error']}" if "error" in rec
              else f"positions={sum(len(x) for x in rec['positions'])}"), flush=True)
        rc |= 1 if "error" in rec else 0
    return rc


def self_test() -> int:
    import torch
    ok = []
    g = torch.Generator().manual_seed(0)
    V, width, pl, st = 11, 9, 4, 4
    full = torch.randn(width, V, generator=g)
    chunks = [(0, full[0:5]), (5, full[5:9])]                       # a window cut 5 + 4, as 512 + 129 is
    rows = rows_for(chunks, pl, st)
    ok.append(all(torch.equal(a, b) for a, b in zip(rows, [full[p] for p in range(pl, pl + st)])))
    ids = torch.randint(0, V, (width,), generator=g).tolist()
    pr = pairs(rows, ids[pl + 1:pl + st + 1])
    lp = torch.log_softmax(full.float(), -1)
    ok.append(pr == [(round(float(-lp[p, ids[p + 1]]), 6), int(lp[p].argmax())) for p in range(pl, pl + st)])
    for bad in ([(0, full[0:5])], [(0, full[0:6]), (5, full[5:9])]):         # a position missing / in two chunks
        try:
            rows_for(bad, pl, st)
            ok.append(False)
        except ValueError:
            ok.append(True)
    try:
        pairs(rows, ids[:2])
        ok.append(False)
    except ValueError:
        ok.append(True)
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location("sc5_ref", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                          "sc5_ref.py"))
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    ok.append(ref._pairs(full[pl:pl + st], ids[pl + 1:pl + st + 1]) == pr)          # the reference's pairing exactly
    print(f"sc5_e4b_quality self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("run",))
    ap.add_argument("--windows")
    ap.add_argument("--prefill-out")
    ap.add_argument("--decode-out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.cmd == "run" and a.windows and a.prefill_out:
        return run(a.windows, a.prefill_out, a.decode_out)
    ap.error("run --windows --prefill-out [--decode-out] | --self-test")
    return 2


if __name__ == "__main__":
    sys.exit(main())
