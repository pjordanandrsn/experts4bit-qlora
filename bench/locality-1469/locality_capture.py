#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT.
"""locality_capture.py -- #1469 item 1: Qwen3-30B-A3B's routing trace through e4b's NF4 host-residency path.

**Why this path.** The RTX A2000 (sm_86) cannot run the fp8 paged runner. The trace therefore runs through
``load_moe_4bit_streaming(offload=True, pin=True, prefetch=True)``: NF4 experts are homed in pinned host RAM, and the
pipelined residency engine keeps a static hot set of experts per layer.

**What it reuses.** grouped-nf4-gemm's ``bench/cold-engine/routing-trace/capture_routing.py``, imported from a pinned
checkout, unchanged. That covers the four prompts (read from its source with ``ast``, not copied), router discovery by
probe through its ``routed_ids``, its rank invariants and its environment fingerprint, so the Qwen3 trace stays
comparable with the 12 committed traces.

**Phases, traced separately and labelled:**
- **calibration:** a profile pass over wikitext-2-raw-v1 *validation* windows picks the hot set per layer (top
  ``--hot-per-layer`` by tokens routed, ``hot_sets_from_profile``). That keeps it out of sample for the decode prompts.
- **decode:** for each prompt kind, ``--steps`` greedy autoregressive steps after a ``--prompt-tokens`` prompt, as
  ``capture_routing`` runs them, with the residency engine on and its traffic counted (``E4B_PIPELINED_TRAFFIC=1``).
- **prefill:** ``--prefill-windows`` teacher-forced forwards over wikitext-2-raw-v1 *test* windows (P117's join and
  stride), recording every row.

**Outputs** in ``--out-dir`` (the format agreed with the owner of #1469 items 2 and 3):
- ``decode.npz`` and ``prefill.npz``: ``expert_ids`` int16 ``[n_tokens, n_layers, top_k]`` in token order, ``seq_offsets``
  int32, ``labels`` and ``n_experts``;
- ``decode_<prompt>.jsonl``: ``capture_routing``'s shape (one meta line, then one line per decode step), so the trace
  sits next to the 12 existing ones;
- ``manifest.json``: the model and revision, versions, the environment fingerprint, text provenance with token-id
  hashes, the geometry, the residency setup and counters, and the exact command.

    locality_capture.py --phase calibrate --revision REV --out-dir OUT --hot-profile OUT/calibration_profile.jsonl
    locality_capture.py --phase census --revision REV --trace-dir GNF4/bench/cold-engine/routing-trace --out-dir OUT
                        --hot-profile OUT/calibration_profile.jsonl [--steps 512 --prompt-tokens 64 --hot-per-layer 8]
    locality_capture.py --self-test
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
import time

PROMPT_KINDS = ("prose", "code", "math", "dialogue")


def load_capture_routing(trace_dir: str):
    """grouped-nf4-gemm's capture_routing.py as a module, and its PROMPTS dict read from the source by ``ast``."""
    path = os.path.join(trace_dir, "capture_routing.py")
    spec = importlib.util.spec_from_file_location("capture_routing", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = open(path, encoding="utf-8").read()
    prompts = None
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PROMPTS" for t in node.targets):
            prompts = ast.literal_eval(node.value)
    if not isinstance(prompts, dict) or set(prompts) != set(PROMPT_KINDS):
        raise ValueError(f"capture_routing.py's PROMPTS is not the four kinds {PROMPT_KINDS}: {prompts!r:.120}")
    return mod, prompts, hashlib.sha256(src.encode()).hexdigest()


def prompt_ids(tok, text: str, n: int) -> list:
    """capture_routing's construction: the text repeated until it covers ``n`` tokens, then cut to ``n``."""
    reps = max(2, -(-n // max(1, len(tok(text).input_ids))) + 1)
    return tok(text * reps).input_ids[:n]


def find_routers(model, k: int, cr, device) -> list:
    """One router module per layer, found as capture_routing finds them: candidates by name, kept by a probe forward
    whose output its ``routed_ids`` can read."""
    import torch
    cands = [(n, m) for n, m in model.named_modules()
             if n.rsplit(".", 1)[-1] in ("gate", "router", "layer", "gate_proj") or n.endswith("router.layer")]
    probe = {}
    hs = [m.register_forward_hook((lambda nm: lambda _m, _i, out: probe.__setitem__(nm, out))(n)) for n, m in cands]
    try:
        with torch.no_grad():
            model(torch.tensor([[1, 2]], device=device))
    finally:
        for h in hs:
            h.remove()
    seen, gates = {}, []
    for n, m in cands:
        out = probe.get(n)
        if out is None:
            continue
        try:
            cr.routed_ids(out, k)
        except Exception:                                   # noqa: BLE001  (not a router: capture_routing's rule)
            continue
        key = ".".join(n.split(".")[:3])
        if key not in seen:
            seen[key] = n
            gates.append((n, m))
    return gates


class Recorder:
    """Forward hooks on every router. While armed, each call's routed ids, every row, are kept per layer."""

    def __init__(self, gates, k, cr):
        self.k, self.cr, self.armed, self.calls = k, cr, False, [[] for _ in gates]
        self.hooks = [m.register_forward_hook(self._mk(i)) for i, (_n, m) in enumerate(gates)]

    def _mk(self, i):
        def hook(_m, _inp, out):
            if self.armed:
                self.calls[i].append(self.cr.routed_ids(out, self.k).reshape(-1, self.k).detach().to("cpu"))
        return hook

    def take(self):
        """The armed calls' ids as [T, n_layers, k] int16 (every layer saw the same rows), then clear."""
        import torch
        per = [torch.cat(c) if c else None for c in self.calls]
        if any(p is None for p in per) or len({p.shape[0] for p in per}) != 1:
            raise RuntimeError(f"layers recorded different row counts: {[None if p is None else p.shape[0] for p in per]}")
        self.calls = [[] for _ in self.calls]
        return torch.stack(per, 1).to(torch.int16)

    def close(self):
        for h in self.hooks:
            h.remove()


def decode_trace(model, rec: Recorder, ids: list, steps: int, device, cr, n_layers: int):
    """capture_routing's loop: one prefill forward (not recorded), then ``steps`` greedy decode forwards with a cache.
    Returns (ids [steps, L, k] int16, capture_routing-shape records, the generated tokens)."""
    import torch
    rows, recs, toks = [], [], []
    with torch.no_grad():
        past, cur = None, torch.tensor([ids], device=device)
        for s in range(steps + 1):
            rec.armed = s > 0
            r = model(cur, past_key_values=past, use_cache=True)
            past, cur = r.past_key_values, r.logits[:, -1:].argmax(-1)
            if s == 0:
                rec.calls = [[] for _ in rec.calls]
                continue
            t = rec.take()                                   # [1, L, k]: one decode token
            if t.shape[0] != 1:
                raise RuntimeError(f"decode step {s - 1} recorded {t.shape[0]} rows")
            rows.append(t[0])
            tok = int(cur.reshape(-1)[-1].item())
            toks.append(tok)
            ranked = {str(i): t[0, i].tolist() for i in range(n_layers)}
            r_ = {"step": s - 1, "token": tok, "routed": {i: sorted(v) for i, v in ranked.items()}, "routed_rank": ranked}
            err = cr.check_rank_invariants(r_, n_layers)
            if err:
                raise RuntimeError(err)
            recs.append(r_)
    rec.armed = False
    return torch.stack(rows), recs, toks


def prefill_trace(model, rec: Recorder, window: list, device):
    """One teacher-forced forward over the window, every row recorded: [len(window), L, k] int16."""
    import torch
    rec.armed = True
    try:
        with torch.no_grad():
            model(torch.tensor([window], device=device), use_cache=False)
        return rec.take()
    finally:
        rec.armed = False


def wikitext_windows(tok, split: str, n: int, width: int, stride: int, revision=None):
    """P117's corpus join (non-blank lines joined by a blank line), tokenized once; window k = ids[k*stride, +width)."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split, revision=revision)
    ids = tok("\n\n".join(t for t in ds["text"] if t.strip())).input_ids
    out = [ids[k * stride:k * stride + width] for k in range(n)]
    if any(len(w) != width for w in out):
        raise ValueError(f"wikitext {split} is too short for {n} windows of {width} at stride {stride}")
    return out


def ids_sha256(seqs) -> str:
    return hashlib.sha256(json.dumps([list(map(int, s)) for s in seqs], separators=(",", ":")).encode()).hexdigest()


def write_npz(path, seqs: list, labels: list, n_experts: int):
    import numpy as np
    import torch
    ids = torch.cat(seqs).numpy().astype(np.int16)
    off = np.cumsum([0] + [s.shape[0] for s in seqs]).astype(np.int32)
    np.savez_compressed(path, expert_ids=ids, seq_offsets=off, labels=np.array(labels), n_experts=np.int32(n_experts))
    return ids.shape


def capture(model, tok, cr, prompts: dict, *, out_dir, steps, prompt_tokens, device, prefill_windows=(),
            residency=None, meta_extra=None) -> dict:
    """The model-agnostic census: routers, decode per prompt kind, prefill per window, and the files. ``residency``,
    when given, is ``(before_step_fn, counters_fn)``. ``counters_fn()`` returns {layer: (cold_pcie_bytes, row_bytes)},
    cumulative, and per-prompt deltas are taken around each decode."""
    cfg = getattr(model.config, "text_config", None) or model.config
    k = int(getattr(cfg, "num_experts_per_tok"))
    n_exp = int(next(getattr(cfg, n) for n in ("num_experts", "num_local_experts") if getattr(cfg, n, None)))
    gates = find_routers(model, k, cr, device)
    n_layers = len(gates)
    rec = Recorder(gates, k, cr)
    os.makedirs(out_dir, exist_ok=True)
    dec_seqs, dec_labels, cold, row_b, total_steps, prompt_meta = [], [], {}, {}, 0, {}
    try:
        for kind in PROMPT_KINDS:
            ids = prompt_ids(tok, prompts[kind], prompt_tokens)
            before = residency[1]() if residency else None
            t0 = time.time()
            rows, recs, toks = decode_trace(model, rec, ids, steps, device, cr, n_layers)
            wall = round(time.time() - t0, 1)
            if residency:
                after = residency[1]()
                for layer, (cb, rb) in after.items():
                    cold[layer] = cold.get(layer, 0) + cb - before[layer][0]
                    row_b[layer] = rb
            total_steps += steps
            dec_seqs.append(rows)
            dec_labels.append(kind)
            prompt_meta[kind] = {"prompt_ids_sha256": ids_sha256([ids]), "distinct_tokens": len(set(toks)),
                                 "wall_s": wall}
            meta = {"model": (meta_extra or {}).get("model"), "prompt": kind, "steps": len(recs), "layers": n_layers,
                    "top_k": k, "n_experts": n_exp, "top_k_native": None, "top_k_overridden": False,
                    "distinct_tokens": len(set(toks)), "prompt_tokens": len(ids), "decode": True,
                    "env": (meta_extra or {}).get("env"), "path": "e4b NF4 host residency (#1469 item 1)"}
            with open(os.path.join(out_dir, f"decode_{kind}.jsonl"), "w") as f:
                f.write(json.dumps({"meta": meta}) + "\n")
                for r in recs:
                    f.write(json.dumps(r) + "\n")
        pre_seqs = [prefill_trace(model, rec, w, device) for w in prefill_windows]
    finally:
        rec.close()
    shapes = {"decode": write_npz(os.path.join(out_dir, "decode.npz"), dec_seqs, dec_labels, n_exp)}
    if pre_seqs:
        shapes["prefill"] = write_npz(os.path.join(out_dir, "prefill.npz"), pre_seqs,
                                      [f"window{i}" for i in range(len(pre_seqs))], n_exp)
    man = {"geometry": {"n_layers": n_layers, "top_k": k, "n_experts": n_exp, "routers": [n for n, _ in gates][:2]},
           "decode": {"prompts": prompt_meta, "steps_per_prompt": steps, "prompt_tokens": prompt_tokens,
                      "batch": 1, "shape": list(shapes["decode"])},
           "prefill": {"windows": len(pre_seqs), "window_ids_sha256": ids_sha256(prefill_windows) if pre_seqs else None,
                       "shape": list(shapes.get("prefill", ())), "batch": 1},
           "residency_counters": ({"steps": total_steps, "cold_pcie_bytes": {str(i): cold[i] for i in cold},
                                   "row_bytes": {str(i): row_b[i] for i in row_b}} if residency else None)}
    man.update(meta_extra or {})
    json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1, sort_keys=True)
    return man


# ------------------------------------------------------------------------------------------------ the A2000 run --
def _load(a):
    import torch
    from transformers import AutoTokenizer

    from experts4bit_qlora.loader import load_moe_4bit_streaming
    # --model-path: a local snapshot of --model at --revision (e.g. a read-only mounted cache), loaded by path so the
    # hub cache can stay a separate writable directory for the dataset; the manifest still names repo and revision.
    src, rev = (a.model_path, None) if a.model_path else (a.model, a.revision)
    tok = AutoTokenizer.from_pretrained(src, revision=rev)
    model, _cfg = load_moe_4bit_streaming(src, "cuda", torch.bfloat16, r=8, alpha=16, offload=True, pin=True,
                                          prefetch=True, quant_type="nf4", revision=rev)
    return model.eval(), tok


def run_calibrate(a) -> int:
    """Process 1, as serve.py documents the flow: the expert profile over the calibration windows, then exit. The
    residency engine is never enabled on a model with the profiler attached."""
    import torch
    os.makedirs(a.out_dir, exist_ok=True)
    os.environ["E4B_EXPERT_PROFILE"] = a.hot_profile
    os.environ["E4B_PROFILE_PHASE"] = "calibration"
    from experts4bit_qlora.engines import expert_profile
    model, tok = _load(a)
    if not expert_profile.attach(model):
        raise RuntimeError("expert_profile.attach found no dispatched experts modules")
    calib = wikitext_windows(tok, "validation", a.calib_windows, a.window_width, a.stride)
    with torch.no_grad():
        for w in calib:
            model(torch.tensor([w], device="cuda"), use_cache=False)
    expert_profile.flush()
    json.dump({"windows": a.calib_windows, "width": a.window_width, "stride": a.stride, "split": "validation",
               "window_ids_sha256": ids_sha256(calib), "command": " ".join(sys.argv)},
              open(a.hot_profile + ".meta.json", "w"), indent=1)
    print(f"LOCALITY_CALIBRATE profile -> {a.hot_profile}", flush=True)
    return 0


def run_census(a) -> int:
    """Process 2: a fresh load, the pipelined engine from the calibration profile, then the trace."""
    import torch

    import experts4bit_qlora
    from experts4bit_qlora.engines.expert_profile import hot_sets_from_profile
    from experts4bit_qlora.engines.pipelined import enable_pipelined_residency
    cr, prompts, cr_sha = load_capture_routing(a.trace_dir)
    os.environ["E4B_PIPELINED_TRAFFIC"] = "1"
    os.environ.pop("E4B_EXPERT_PROFILE", None)
    model, tok = _load(a)
    hot = hot_sets_from_profile(a.hot_profile, a.hot_per_layer)
    n_eng = enable_pipelined_residency(model, hot, "cuda")
    engines = [m._pipelined for _n, m in model.named_modules() if getattr(m, "_pipelined", None) is not None]
    if not engines:
        raise RuntimeError("enable_pipelined_residency engaged no module")

    def counters():
        return {i: (e.traffic()["cold_pcie_bytes"], int(e.row_bytes)) for i, e in enumerate(engines)}
    if a.model_path:
        if os.path.basename(os.path.normpath(a.model_path)) != a.revision:
            raise ValueError(f"--model-path {a.model_path} is not the snapshot of revision {a.revision}")
        snap = a.model_path
    else:
        from huggingface_hub import snapshot_download
        snap = snapshot_download(a.model, revision=a.revision, local_files_only=True)
    test = wikitext_windows(tok, "test", a.prefill_windows, a.window_width, a.stride)
    calib_meta = json.load(open(a.hot_profile + ".meta.json"))
    extra = {"model": a.model, "revision": a.revision, "env": cr.env_fingerprint(snap, model, a.model),
             "versions": {"experts4bit_qlora": getattr(experts4bit_qlora, "__version__", None),
                          "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
             "capture_routing_sha256": cr_sha,
             "text": {"decode_prompts": "capture_routing.PROMPTS (prose, code, math, dialogue)",
                      "prefill": {"dataset": "Salesforce/wikitext wikitext-2-raw-v1", "split": "test",
                                  "join": "non-blank lines joined by a blank line (P117)", "stride": a.stride,
                                  "width": a.window_width},
                      "calibration": calib_meta},
             "residency": {"engine": "pipelined (enable_pipelined_residency)", "hot_per_layer": a.hot_per_layer,
                           "hot_sets_from": "hot_sets_from_profile over the calibration windows (wikitext validation, "
                                            "out of sample), profiled in a separate process",
                           "engines": n_eng, "offload": True, "pin": True, "prefetch": True,
                           "decode_only": "the engine serves T == 1; prefill takes the offload reference path"},
             "command": " ".join(sys.argv)}
    capture(model, tok, cr, prompts, out_dir=a.out_dir, steps=a.steps, prompt_tokens=a.prompt_tokens, device="cuda",
            prefill_windows=test, residency=(None, counters), meta_extra=extra)
    print(f"LOCALITY_CAPTURE done -> {a.out_dir}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ self-test --
_FAKE_CR = '''
import torch
def routed_ids(out, k):
    ts = [t for t in (out if isinstance(out, (tuple, list)) else [out]) if isinstance(t, torch.Tensor)]
    idx = [t for t in ts if t.dtype in (torch.int32, torch.int64) and t.shape[-1] == k]
    if len(idx) == 1:
        return idx[0]
    raise RuntimeError("no ids")
def check_rank_invariants(rec, n_layers):
    for i in range(n_layers):
        if sorted(rec["routed_rank"][str(i)]) != sorted(rec["routed"][str(i)]):
            return "rank"
    return None
def env_fingerprint(model_path, model, repo_id=None):
    return {"repo_id": repo_id}
def main():
    PROMPTS = {"prose": "the cat sat on the mat. ", "code": "def f(x):\\n    return x\\n",
               "math": "1 + 1 = 2. ", "dialogue": "A: hi\\nB: hello\\n"}
'''


def self_test() -> int:
    import tempfile

    import numpy as np
    import torch
    ok = []
    d = tempfile.mkdtemp()
    open(os.path.join(d, "capture_routing.py"), "w", encoding="utf-8").write(_FAKE_CR)
    cr, prompts, sha = load_capture_routing(d)
    ok.append(set(prompts) == set(PROMPT_KINDS) and len(sha) == 64)
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3MoeConfig(vocab_size=97, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_experts=8,
                         num_experts_per_tok=2, max_position_embeddings=512, decoder_sparse_step=1, mlp_only_layers=[])
    model = Qwen3MoeForCausalLM(cfg).eval()

    class Tok:
        def __call__(self, text, **kw):
            return type("E", (), {"input_ids": [ord(c) % 97 for c in text]})()
    out = os.path.join(d, "out")
    man = capture(model, Tok(), cr, prompts, out_dir=out, steps=6, prompt_tokens=10, device="cpu",
                  prefill_windows=[list(range(1, 21)), list(range(30, 50))], meta_extra={"model": "tiny"})
    z = np.load(os.path.join(out, "decode.npz"))
    ok.append(z["expert_ids"].shape == (24, 3, 2) and z["expert_ids"].dtype == np.int16
              and list(z["seq_offsets"]) == [0, 6, 12, 18, 24] and list(z["labels"]) == list(PROMPT_KINDS))
    p = np.load(os.path.join(out, "prefill.npz"))
    ok.append(p["expert_ids"].shape == (40, 3, 2) and list(p["seq_offsets"]) == [0, 20, 40])
    lines = open(os.path.join(out, "decode_code.jsonl")).read().splitlines()
    m0 = json.loads(lines[0])["meta"]
    ok.append(len(lines) == 7 and m0["layers"] == 3 and m0["top_k"] == 2 and m0["n_experts"] == 8 and m0["steps"] == 6
              and json.loads(lines[1])["routed"]["0"] == sorted(json.loads(lines[1])["routed_rank"]["0"]))
    ok.append(man["geometry"] == {"n_layers": 3, "top_k": 2, "n_experts": 8,
                                  "routers": ["model.layers.0.mlp.gate", "model.layers.1.mlp.gate"]}
              and man["residency_counters"] is None and man["model"] == "tiny")
    rec = json.loads(lines[1])                                             # the npz and the jsonl agree
    ok.append(sorted(z["expert_ids"][6, 0].tolist()) == rec["routed"]["0"])
    with torch.no_grad():                                                   # prefill ids equal the router's own top-k
        h = {}
        hk = model.model.layers[1].mlp.gate.register_forward_hook(lambda _m, _i, o: h.setdefault("o", o))
        model(torch.tensor([list(range(1, 21))]), use_cache=False)
        hk.remove()
    ok.append(torch.equal(cr.routed_ids(h["o"], 2).reshape(-1, 2).to(torch.int16), torch.from_numpy(p["expert_ids"][:20, 1])))
    st = 0

    def counters():
        nonlocal st
        st += 1
        return {0: (st * 64, 16), 1: (0, 16)}
    man2 = capture(model, Tok(), cr, prompts, out_dir=os.path.join(d, "o2"), steps=4, prompt_tokens=8, device="cpu",
                   residency=(None, counters))
    ok.append(man2["residency_counters"] == {"steps": 16, "cold_pcie_bytes": {"0": 256, "1": 0},
                                             "row_bytes": {"0": 16, "1": 16}})
    try:
        bad = os.path.join(d, "bad")
        os.makedirs(bad)
        open(os.path.join(bad, "capture_routing.py"), "w").write("def main():\n    PROMPTS = {'prose': 'x'}\n")
        load_capture_routing(bad)
        ok.append(False)
    except ValueError:
        ok.append(True)
    print(f"locality_capture self-test {'OK' if all(ok) else 'FAILED'} ({sum(ok)}/{len(ok)} cases)")
    return 0 if all(ok) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--revision")
    ap.add_argument("--model-path", help="a local snapshot directory of --model at --revision")
    ap.add_argument("--trace-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--hot-per-layer", type=int, default=8)
    ap.add_argument("--calib-windows", type=int, default=8)
    ap.add_argument("--prefill-windows", type=int, default=16)
    ap.add_argument("--window-width", type=int, default=641)
    ap.add_argument("--stride", type=int, default=3072)
    ap.add_argument("--phase", choices=("calibrate", "census"))
    ap.add_argument("--hot-profile", help="the calibration profile JSONL (written by --phase calibrate)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.phase and a.revision and a.out_dir and a.hot_profile and (a.trace_dir or a.phase == "calibrate")):
        ap.error("--phase calibrate|census --revision --out-dir --hot-profile [--trace-dir], or --self-test")
    return run_calibrate(a) if a.phase == "calibrate" else run_census(a)


if __name__ == "__main__":
    sys.exit(main())
