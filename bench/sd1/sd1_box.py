#!/usr/bin/env python3
"""sd1_box.py -- lane SD1 (e4b#1313), the box side (bench/sd1/PREREG-sd1.md): speculative decoding's acceptance on the
shipped default's target.

**`--chat-prompts`** turns the 16 pinned UltraChat prompts (`chat_prompts.json`) into token rows. It uses the model's own
chat template, with `enable_thinking` True for C-think and False for C-nothink, and writes `prompts_<W>.json`.

**`--capture W`** builds the engine as the shipped server builds it (`PagedServeConfig.from_env()` + `build_engine`). Its
decode and prefill are eager (`E4B_PAGED_GRAPHS=0`, `E4B_PAGED_PREFILL_GRAPH=0`), so forward pre-hooks fire.
- **The hooks.** They sit on decoder layers 2, 24 and 45 and record the residual stream entering them: EAGLE-3's
  auxiliary states, vLLM's default (2, L // 2, L - 3).
- **The rows.** Each row runs alone as one request, to exactly N new tokens. Its prefill plus its N - 1 decode steps
  forward L - 1 = P + N - 1 positions, and the capture must account for every one of them, or the box refuses (rc 19).
- **The draft.** The pinned EAGLE-3 head (`sd1_eagle3.py`) then computes a greedy 5-token chain at every index. The
  capture file keeps tokens and chains, not hidden states.
- **R's tripwire.** R's row 0 must equal `p127-5090-1`'s W1 tokens (B = 7f044dd9 + d769d502, graph mode). This is a
  REPORTED check, not a gate: P109 read W1 identical in every arm, eager included, and SC2b read prefill graphs as
  byte-identical text. tau is a quality quantity either way.

**`--self-test`** checks the capture bookkeeping on CPU with a fake layer stack.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sd1_eagle3 as e3  # noqa: E402

WORKLOADS = {"R": 160, "C-think": 256, "C-nothink": 256}
HEAD_REPO = "RedHatAI/Qwen3-30B-A3B-speculator.eagle3"
HEAD_REV = "6afc5aa2477b923467fb9a8d906782b984a9a6ba"


def digest(obj) -> str:
    """P109's digest, so its prompts.json verifies here: compact JSON, no key sorting."""
    return hashlib.sha256(json.dumps(obj, separators=(",", ":")).encode()).hexdigest()


class AuxCapture:
    """Forward pre-hooks on the decoder layers in AUX_LAYERS. Each call appends that call's [T, H] input. rows() joins
    them per call, in order, into [positions, 3 * H] (low|mid|high)."""

    def __init__(self, layers, ids=e3.AUX_LAYERS):
        self.ids, self.calls, self.handles = tuple(ids), {i: [] for i in ids}, []
        for i in self.ids:
            self.handles.append(layers[i].register_forward_pre_hook(self._hook(i), with_kwargs=True))

    def _hook(self, i):
        def h(_mod, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            self.calls[i].append(x.detach().reshape(-1, x.shape[-1]).to("cpu"))
        return h

    def reset(self):
        for i in self.ids:
            self.calls[i].clear()

    def take(self):
        import torch
        counts = {i: sum(c.shape[0] for c in self.calls[i]) for i in self.ids}
        if len(set(counts.values())) != 1:
            raise AssertionError(f"layers saw different position counts: {counts}")
        aux = torch.cat([torch.cat(self.calls[i], 0) for i in self.ids], -1)
        self.reset()
        return aux

    def close(self):
        for h in self.handles:
            h.remove()


def chat_main(a) -> int:
    from transformers import AutoTokenizer
    src = json.load(open(a.chat_prompts, encoding="ascii"))
    texts = [p["prompt"] for p in src["prompts"]]
    if digest_list(texts) != src["prompts_sha256"]:
        raise SystemExit("REFUSED: chat_prompts.json does not match its own digest")
    tok = AutoTokenizer.from_pretrained(a.snapshot)
    for w, think in (("C-think", True), ("C-nothink", False)):
        rows = [tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                        enable_thinking=think, tokenize=True) for t in texts]
        rows = [_ids(r) for r in rows]
        json.dump({"workload": w, "enable_thinking": think, "rows": rows, "prompts_sha256": digest(rows),
                   "source_sha256": src["prompts_sha256"]}, open(os.path.join(a.outdir, f"prompts_{w}.json"), "w"))
        print(f"SD1_PROMPTS {w} rows={len(rows)} min_len={min(map(len, rows))} max_len={max(map(len, rows))} "
              f"sha256={digest(rows)}", flush=True)
    return 0


def _ids(r) -> list:
    """apply_chat_template(tokenize=True) returns a list, or (transformers 5) a dict-like with input_ids, possibly batched."""
    x = r if isinstance(r, list) else r["input_ids"]
    if hasattr(x, "tolist"):
        x = x.tolist()
    if x and isinstance(x[0], list):
        if len(x) != 1:
            raise SystemExit(f"REFUSED: a chat template returned {len(x)} sequences for one conversation")
        x = x[0]
    return [int(t) for t in x]


def digest_list(texts) -> str:
    return hashlib.sha256(json.dumps(texts, ensure_ascii=True).encode()).hexdigest()


def capture_main(a) -> int:
    import torch
    from safetensors.torch import load_file
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    w, n_new = a.capture, WORKLOADS[a.capture]
    pf = json.load(open(a.prompts))
    rows = pf["rows"]
    if digest(rows) != pf["prompts_sha256"]:
        raise SystemExit("REFUSED: the prompts file does not match its own digest")
    cfg = PagedServeConfig.from_env()
    if cfg.graphs or cfg.prefill_graph != "0":
        raise SystemExit(f"REFUSED: capture needs eager decode and prefill (graphs={cfg.graphs}, "
                         f"prefill_graph={cfg.prefill_graph})")
    t0 = time.perf_counter()
    parts = build_engine(cfg)
    load_s = round(time.perf_counter() - t0, 2)
    layers = parts.runner.model.model.layers
    if len(layers) != 48:
        raise SystemExit(f"REFUSED: {len(layers)} decoder layers, expected 48")
    cap = AuxCapture(layers)
    head = e3.Eagle3Draft(load_file(a.head), device="cuda")
    sched = parts.scheduler
    out_rows = []
    for ri, row in enumerate(rows):
        before = len(sched.done)
        cap.reset()
        rid = sched.add_request(list(row), max_new_tokens=n_new)
        sched.run_until_idle()
        q = [r for r in sched.done[before:] if r.rid == rid][0]
        gen = [int(t) for t in q.out]
        if len(gen) != n_new:
            raise SystemExit(f"REFUSED: row {ri} produced {len(gen)} tokens, expected {n_new}")
        aux = cap.take()
        L = len(row) + n_new
        extra = aux.shape[0] == L                 # Amendment 1: one benign extra forward of the last token is dropped
        if extra:
            aux = aux[:L - 1]
        if aux.shape[0] != L - 1:
            print(f"CAPTURE FAIL row {ri}: {aux.shape[0]} positions captured, expected {L - 1}", flush=True)
            return 19
        tokens = torch.tensor(list(row) + gen)
        chains = head.chains(tokens, aux, K=5)
        out_rows.append({"prompt_len": len(row), "tokens": list(row) + gen, "chains": chains.tolist(), "extra_forward": extra})
        print(f"SD1_ROW {w} {ri} prompt={len(row)} new={n_new} positions={aux.shape[0]}", flush=True)
    cap.close()
    rec = {"workload": w, "new_tokens": n_new, "rows": out_rows, "prompts_sha256": pf["prompts_sha256"],
           "e4b_sha": os.environ.get("E4B_SHA"), "gnf4_sha": os.environ.get("GNF4_SHA"), "load_s": load_s,
           "aux_layers": list(e3.AUX_LAYERS), "head": {"repo": HEAD_REPO, "revision": HEAD_REV,
                                                        "sha256": hashlib.sha256(open(a.head, "rb").read()).hexdigest()},
           "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items() if k != "token"},
           "max_memory_allocated": int(torch.cuda.max_memory_allocated())}
    if w == "R" and a.expect_w1:
        ex = json.load(open(a.expect_w1))
        if ex["prompts_sha256"] != pf["prompts_sha256"]:
            raise SystemExit("REFUSED: expect_w1.json belongs to other prompts")
        exp = ex["tokens"]
        got = out_rows[0]["tokens"][out_rows[0]["prompt_len"]:]
        same = sum(x == y for x, y in zip(got, exp))
        rec["w1_tripwire"] = {"equal": got == exp, "agree": same, "of": len(exp)}
        print(f"SD1_TRIPWIRE R row0 equals p127-5090-1 W1: {got == exp} ({same}/{len(exp)})", flush=True)
    json.dump(rec, open(a.out, "w"))
    print("SD1_CAPTURE " + json.dumps({"workload": w, "rows": len(out_rows), "load_s": load_s}), flush=True)
    return 0


def self_test() -> int:
    import torch
    bad = []

    class Layer(torch.nn.Module):
        def forward(self, hidden_states, **kw):
            return hidden_states + 1

    layers = torch.nn.ModuleList([Layer() for _ in range(48)])
    cap = AuxCapture(layers)
    x = torch.zeros(1, 7, 4)
    for i, layer in enumerate(layers):            # one "prefill" of 7 positions
        x = layer(x)
    for _ in range(3):                            # three "decode" steps, one keyword call
        y = torch.zeros(1, 1, 4)
        for layer in layers:
            y = layer(hidden_states=y)
    aux = cap.take()
    if tuple(aux.shape) != (10, 12):
        bad.append(f"shape {tuple(aux.shape)}")
    if not torch.equal(aux[0, :4], torch.full((4,), 2.0)) or not torch.equal(aux[0, 8:], torch.full((4,), 45.0)):
        bad.append("layer order or values")
    cap.close()
    for got, want in ((_ids([1, 2]), [1, 2]), (_ids({"input_ids": [3, 4]}), [3, 4]), (_ids({"input_ids": [[5, 6]]}), [5, 6]),
                      (_ids({"input_ids": torch.tensor([[7, 8]])}), [7, 8])):
        if got != want:
            bad.append(f"_ids {got} != {want}")
    if bad:
        print("sd1_box self-test FAILED:", bad)
        return 1
    print("sd1_box self-test OK (capture bookkeeping, chat-template ids)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--chat-prompts")
    ap.add_argument("--snapshot")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--capture", choices=sorted(WORKLOADS))
    ap.add_argument("--prompts")
    ap.add_argument("--head")
    ap.add_argument("--expect-w1")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.chat_prompts:
        return chat_main(a)
    if a.capture:
        return capture_main(a)
    ap.print_usage()
    return 2


if __name__ == "__main__":
    sys.exit(main())
