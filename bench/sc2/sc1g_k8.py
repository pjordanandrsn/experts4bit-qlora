#!/usr/bin/env python3
"""sc1g_k8.py -- lane SC1g (#846): P39's step_decomp.py and SC1's K8 window rule, run UNMODIFIED, with two things set in
this process that the shared harness files must not carry (they are byte-pinned by other lanes: P86/P88's step_decomp,
SC1's sc1_prompts.py, P42's hook):

  1. SC1G_CHAT_DATE=YYYY-MM-DD pins the chat template's date. gpt-oss's template writes "Current date: <today>" through
     transformers' strftime_now, so the chat-framed window -- and its sha -- would change with the day, and a run that
     crossed midnight would VOID its own rows. strftime_now reads chat_template_utils.datetime at render time; this
     replaces that one global, and the windows mode checks the rendered prefix carries the pinned date.
  2. SC1G_ROUTE_OUT=<prefix> writes <prefix>.<pid>.json at exit: e4b#1129's hot_residency.ROUTE_SEEN and
     paged_attention.ATTN_SEEN -- only if hot_residency was imported, atomically. atexit does not run on a signal or
     os._exit (the arms run under perl alarm), so a missing record is a FAIL to the reader (sc1g_reduce.py), never a pass.

  3. (A1) SC1G_WINDOW_FILE=<k8_window_*.json> makes step_decomp score THAT window's ids: its _k8_window is replaced by one
     that returns the file's ids and refuses unless their sha is the file's text_sha. Every e4b arm then scores the exact
     ids every other engine scores, whatever the text (step_decomp itself only knows wikitext and C4).
  4. (A1) SC1G_CAPTURE_OUT=<path.pt> records, for the first SC1G_CAPTURE_STEPS (16) decode steps, the raw activations
     gnf4's quant_x_rows quantises and the expert ids gemv_mxfp4_b32 serves, at SC1G_CAPTURE_LAYERS (0,6,12,18,23), for
     the gate_up and down GEMVs: the input of sc1g_gemv_check.py. Both gnf4 functions are wrapped, not edited (e4b looks
     them up at call time).
  5. (A4) SC1G_REF_FILE=<ref_<src>.npz> (+ SC1G_REF_SHA, refused on a mismatch) and SC1G_NAMED_OUT=<path.npz> record, at
     each served step, e4b's log-probs on the reference's 64 named tokens for that position and on the target
     (sc1g_kl.py's KL65). step_decomp's module global `torch` is replaced by a forwarding proxy whose log_softmax records
     [1, V] rows -- in the served loop (no --ppl-oracle) that is the one call per scored step -- and passes every other
     attribute (torch.Tensor included) through untouched. The reader proves the alignment: the recorded target log-probs
     must reproduce step_decomp's own mean NLL to 1e-9, and the call count must equal the steps, else the row is VOID.

  sc1g_k8.py k8 -- <step_decomp.py args...>        import step_decomp (beside this file) and run its main()
  sc1g_k8.py windows --model M --rev R --out DIR --suffix S [--steps 2048]
                     [--ultrachat PARQUET --n-conv 2] k8_window_wikitext.json (the chat-framed out-of-distribution
                                                   control) and, with --ultrachat, k8_window_conv{1..n}.json: the first n
                                                   test_sft conversations, by index, whose single rendering in gpt-oss's
                                                   chat template reaches prompt_len + steps + 1 tokens
"""
from __future__ import annotations

import argparse
import atexit
import datetime as _dt
import json
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ("wikitext", "c4val1")
PROMPT_LEN = 512


def pin_chat_date(day: str | None = None) -> str | None:
    """Make transformers' strftime_now render `day` (YYYY-MM-DD). Returns the pinned day, or None when unset."""
    day = day or os.environ.get("SC1G_CHAT_DATE", "")
    if not day:
        return None
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise SystemExit(f"SC1G_CHAT_DATE={day!r}: expected YYYY-MM-DD")
    y, m, d = (int(x) for x in day.split("-"))
    import transformers.utils.chat_template_utils as ctu

    class _Pinned(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return _dt.datetime(y, m, d, 12, 0, 0, tzinfo=tz)

    if getattr(ctu, "datetime", None) is not _dt.datetime:
        raise SystemExit(f"transformers' chat_template_utils.datetime is {getattr(ctu, 'datetime', None)!r}, not datetime.datetime: "
                         "strftime_now cannot be pinned this way on this transformers")
    ctu.datetime = _Pinned
    return day


def _dump_routes():
    hr = sys.modules.get("experts4bit_qlora.engines.hot_residency")
    if hr is None:
        return
    pa = sys.modules.get("experts4bit_qlora.engines.paged_attention")
    rec = {"pid": os.getpid(),
           "route_seen": dict(hr.ROUTE_SEEN) if hasattr(hr, "ROUTE_SEEN") else None,    # None: e4b predates #1129
           "attn_seen": dict(pa.ATTN_SEEN) if pa is not None and hasattr(pa, "ATTN_SEEN") else None}
    out = f"{os.environ['SC1G_ROUTE_OUT']}.{os.getpid()}.json"
    with open(out + ".tmp", "w") as f:
        json.dump(rec, f, sort_keys=True)
    os.replace(out + ".tmp", out)


def window_from_file(path):
    """A replacement for step_decomp._k8_window returning `path`'s ids (same tuple shape, same digest rule)."""
    import hashlib
    import torch
    rec = json.load(open(path))

    def _k8w(a, tok):
        ids = torch.tensor(rec["ids"], dtype=torch.long)
        if a.prompt_len != rec["prompt_len"] or a.ppl_steps != rec["steps"]:
            raise SystemExit(f"{path}: window is prompt_len {rec['prompt_len']} / steps {rec['steps']}, the arm asks "
                             f"{a.prompt_len} / {a.ppl_steps}")
        step = max(1, (ids.numel() - a.prompt_len) // max(1, a.batch))
        prompts = [ids[i * step:i * step + a.prompt_len].tolist() for i in range(a.batch)]
        sha = hashlib.sha256(ids[:a.prompt_len + max(a.ppl_steps, 0) + 1].numpy().tobytes()).hexdigest()
        if sha != rec["text_sha"]:
            raise SystemExit(f"{path}: ids hash to {sha[:12]}, the file says {rec['text_sha'][:12]}")
        print(f"K8 window from file {os.path.basename(path)} sha={sha[:12]} source={rec.get('source')}", flush=True)
        return ids, step, prompts, ids, sha
    return _k8w


def capture(out, steps=16, layers=(0, 6, 12, 18, 23)):
    """Wrap gnf4's quant_x_rows / gemv_mxfp4_b32: keep the raw x, xq, xs and eids of the first `steps` decode steps at
    `layers`. One decode step calls the GEMV twice per layer (gate_up: N = 2 * I, then down), layers in order."""
    import torch
    import int4_b32
    import mxfp4_grouped
    q0, g0 = int4_b32.quant_x_rows, mxfp4_grouped.gemv_mxfp4_b32
    st = {"last_x": None, "calls": 0, "rows": []}

    def q(x):
        st["last_x"] = x.detach()
        return q0(x)

    def g(xq, xs, blocks, scales, eids, N, K, part=None):
        i = st["calls"]
        st["calls"] += 1
        step, k = divmod(i, 48)
        layer, which = k // 2, ("gu", "dn")[k % 2]
        if step < steps and layer in layers and st["last_x"] is not None:
            st["rows"].append({"step": step, "layer": layer, "which": which, "N": int(N), "K": int(K),
                               "x": st["last_x"].to("cpu", copy=True), "eids": eids.detach().to("cpu", copy=True)})
        return g0(xq, xs, blocks, scales, eids, N, K, part=part)

    int4_b32.quant_x_rows, mxfp4_grouped.gemv_mxfp4_b32 = q, g

    def _save():
        if st["rows"]:
            torch.save({"rows": st["rows"], "gemv_calls": st["calls"], "layers": list(layers), "steps": steps}, out + ".tmp")
            os.replace(out + ".tmp", out)
    atexit.register(_save)


class _TorchProxy(types.ModuleType):
    """`torch`, with log_softmax observed. Every other attribute resolves to the real module (torch.Tensor is the real
    class, so isinstance checks are unaffected)."""

    def __init__(self, real, on_rows):
        super().__init__("torch")
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_on_rows", on_rows)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)

    def log_softmax(self, *a, **k):
        out = object.__getattribute__(self, "_real").log_softmax(*a, **k)
        object.__getattribute__(self, "_on_rows")(out)
        return out


def named_capture(module, ref_path: str, ref_sha: str | None, out: str):
    """Install the proxy on `module` (step_decomp): record each [1, V] log-prob row's values at the reference's named ids
    and target for that position; write `out` at exit (atomically)."""
    import numpy as np
    sys.path.insert(0, HERE)
    import sc1g_kl
    ref = sc1g_kl.load_artifact(ref_path, ref_sha or None)
    P = int(ref["ids"].shape[0])
    st = {"calls": 0, "other": 0, "lp": np.full(ref["ids"].shape, np.nan), "tlp": np.full(P, np.nan)}
    real = module.torch
    ids_dev = {}

    def on_rows(x):
        if x.dim() != 2 or x.shape[0] != 1:
            st["other"] += 1
            return
        t = st["calls"]
        st["calls"] += 1
        if t >= P:
            return
        if x.device not in ids_dev:
            ids_dev[x.device] = (real.as_tensor(ref["ids"], dtype=real.long, device=x.device),
                                 real.as_tensor(ref["target"], dtype=real.long, device=x.device))
        ids, tg = ids_dev[x.device]
        row = x[0].double()
        st["lp"][t] = row.index_select(0, ids[t]).cpu().numpy()
        st["tlp"][t] = float(row[tg[t]])

    def _save():
        np.savez(out + ".tmp.npz", eng_lp=st["lp"], eng_target_lp=st["tlp"])
        os.replace(out + ".tmp.npz", out)
        json.dump({"calls": st["calls"], "other_calls": st["other"], "positions": P, "ref_file": os.path.basename(ref_path),
                   "ref_sha": sc1g_kl.file_sha(ref_path)}, open(out + ".json", "w"), indent=1, sort_keys=True)
    atexit.register(_save)
    st["save"] = _save
    module.torch = _TorchProxy(real, on_rows)
    return st


def full_capture(module, ref_path: str, ref_sha: str | None, window_file: str, out: str):
    """A5: the same proxy, reading box R's FULL-vocabulary fp16 rows (ref_full_<src>.npy, memory-mapped, refused unless its
    sha is `ref_sha`): at each served step, KL(p_ref || p_e4b) over every token, in fp64 on the row's device, and the
    target's log-prob (targets from the arm's own window file). Written to `out` at exit."""
    import numpy as np
    sys.path.insert(0, HERE)
    import sc1g_kl
    if ref_sha and sc1g_kl.file_sha(ref_path) != ref_sha:
        raise SystemExit(f"{ref_path}: sha differs from the registered {ref_sha[:16]} -- refusing the reference")
    ref = np.load(ref_path, mmap_mode="r")
    rec = json.load(open(window_file))
    P0, S = int(rec["prompt_len"]), int(rec["steps"])
    targets = [int(x) for x in rec["ids"][P0 + 1:P0 + S + 1]]
    if ref.shape[0] != S:
        raise SystemExit(f"{ref_path}: {ref.shape[0]} rows for a {S}-step window -- refused")
    st = {"calls": 0, "other": 0, "kl": np.full(S, np.nan), "tlp": np.full(S, np.nan)}
    real = module.torch

    def on_rows(x):
        if x.dim() != 2 or x.shape[0] != 1:
            st["other"] += 1
            return
        t = st["calls"]
        st["calls"] += 1
        if t >= S:
            return
        if x.shape[1] != ref.shape[1]:
            raise SystemExit(f"e4b row has {x.shape[1]} logits, the reference {ref.shape[1]} -- refused")
        st["kl"][t] = float(sc1g_kl.kl_full_rows(real.as_tensor(np.asarray(ref[t:t + 1])), x.detach())[0])
        st["tlp"][t] = float(x[0, targets[t]])

    def _save():
        np.savez(out + ".tmp.npz", eng_kl=st["kl"], eng_target_lp=st["tlp"])
        os.replace(out + ".tmp.npz", out)
        json.dump({"calls": st["calls"], "other_calls": st["other"], "positions": S, "ref_file": os.path.basename(ref_path),
                   "ref_sha": ref_sha, "estimator": "full-vocabulary KL (A5)"}, open(out + ".json", "w"), indent=1, sort_keys=True)
    atexit.register(_save)
    st["save"] = _save
    module.torch = _TorchProxy(real, on_rows)
    return st


def k8(args) -> None:
    pin_chat_date()
    if os.environ.get("SC1G_ROUTE_OUT"):
        atexit.register(_dump_routes)
    sys.argv = [os.path.join(HERE, "step_decomp.py")] + list(args)
    sys.path.insert(0, HERE)
    if os.environ.get("SC1G_CAPTURE_OUT"):
        lay = tuple(int(x) for x in os.environ.get("SC1G_CAPTURE_LAYERS", "0,6,12,18,23").split(","))
        capture(os.environ["SC1G_CAPTURE_OUT"], int(os.environ.get("SC1G_CAPTURE_STEPS", "16")), lay)
    import step_decomp
    if os.environ.get("SC1G_WINDOW_FILE"):
        step_decomp._k8_window = window_from_file(os.environ["SC1G_WINDOW_FILE"])
    if os.environ.get("SC1G_REF_FULL_FILE"):
        if not (os.environ.get("SC1G_KL_OUT") and os.environ.get("SC1G_WINDOW_FILE")):
            raise SystemExit("SC1G_REF_FULL_FILE needs SC1G_KL_OUT and SC1G_WINDOW_FILE")
        if "--ppl-oracle" in args:
            raise SystemExit("SC1G_REF_FULL_FILE reads the SERVED loop's rows; an --ppl-oracle arm scores elsewhere -- refused")
        full_capture(step_decomp, os.environ["SC1G_REF_FULL_FILE"], os.environ.get("SC1G_REF_FULL_SHA"),
                     os.environ["SC1G_WINDOW_FILE"], os.environ["SC1G_KL_OUT"])
    elif os.environ.get("SC1G_REF_FILE"):
        if not os.environ.get("SC1G_NAMED_OUT"):
            raise SystemExit("SC1G_REF_FILE needs SC1G_NAMED_OUT")
        if "--ppl-oracle" in args:
            raise SystemExit("SC1G_REF_FILE reads the SERVED loop's rows; an --ppl-oracle arm scores elsewhere -- refused")
        named_capture(step_decomp, os.environ["SC1G_REF_FILE"], os.environ.get("SC1G_REF_SHA"), os.environ["SC1G_NAMED_OUT"])
    step_decomp.main()


def windows(a) -> int:
    day = pin_chat_date()
    sys.path.insert(0, a.harness_dir)
    import sc1_prompts
    import step_decomp
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.rev)
    pre = tok.decode(step_decomp._chat_prefix_ids(tok, a.suffix).tolist())
    dates = re.findall(r"Current date: (\d{4}-\d{2}-\d{2})", pre)
    if day and dates != [day]:
        raise SystemExit(f"the chat prefix carries dates {dates}, not [{day}]: the pin did not take")
    os.makedirs(a.out, exist_ok=True)
    srcs = ("wikitext",) if a.ultrachat else SOURCES
    for src in srcs:
        ns = types.SimpleNamespace(ppl_source=src, ppl_chat=True, ppl_chat_suffix=a.suffix, prompt_offset=0, prompt_span=0,
                                   prompt_len=PROMPT_LEN, batch=1, ppl_steps=a.steps)
        _, _, _, ppl_ids, ppl_sha = step_decomp._k8_window(ns, tok)
        win = ppl_ids[:PROMPT_LEN + a.steps + 1].tolist()
        rec = sc1_prompts.window_record(src, win, ppl_sha, PROMPT_LEN, a.steps, a.model, a.rev)
        rec.update(chat=True, chat_suffix=a.suffix, chat_date=day, chat_prefix_dates=dates)
        json.dump(rec, open(os.path.join(a.out, f"k8_window_{src}.json"), "w"))
        print(f"WINDOW src={src} text_sha={ppl_sha} n_ids={len(win)} prompt_len={PROMPT_LEN} steps={a.steps} chat_date={day}", flush=True)
    if a.ultrachat:
        ultrachat_windows(a, tok, sc1_prompts, day)
    return 0


def roles(tok, ids):
    """Per token: the role whose message CONTENT it is ('system', 'user', 'assistant', ...), or 'markup' for the template's
    special tokens and message headers. Read from the special tokens, since gpt-oss's template does not render a
    conversation prefix-stably (the last assistant turn ends <|return|>, earlier ones <|end|>)."""
    sp = {tok.convert_tokens_to_ids(x): x for x in ("<|start|>", "<|message|>", "<|end|>", "<|return|>", "<|channel|>")}
    out, role, mode, head = [], None, None, []
    for t in ids:
        s = sp.get(t)
        if s == "<|start|>":
            mode, head = "head", []
            out.append("markup")
        elif s == "<|message|>" and mode == "head":
            name = tok.decode(head).strip()
            role, mode = (name.split("<|channel|>")[0].strip() or "?"), "body"
            out.append("markup")
        elif s in ("<|end|>", "<|return|>"):
            mode = None
            out.append("markup")
        elif mode == "head":
            head.append(t)
            out.append("markup")
        else:
            out.append(role if mode == "body" else "markup")
    return out


def ultrachat_windows(a, tok, sc1_prompts, day):
    """The graded windows: the first a.n_conv test_sft conversations, by index, whose single rendering reaches the window
    length. Never concatenated; each window is that conversation's first prompt_len + steps + 1 ids."""
    import hashlib
    import struct
    import pyarrow.parquet as pq
    need = PROMPT_LEN + a.steps + 1
    tab = pq.read_table(a.ultrachat)
    n = 0
    for i in range(tab.num_rows):
        r = tab.slice(i, 1).to_pylist()[0]
        msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
        x = tok.apply_chat_template(msgs, tokenize=True)
        ids = list(x["input_ids"] if hasattr(x, "keys") else x)
        if len(ids) < need:
            continue
        n += 1
        win = [int(v) for v in ids[:need]]
        sha = hashlib.sha256(struct.pack(f"<{len(win)}q", *win)).hexdigest()
        rec = sc1_prompts.window_record(f"conv{n}", win, sha, PROMPT_LEN, a.steps, a.model, a.rev)
        own = roles(tok, win)[PROMPT_LEN + 1:need]
        share = {k: round(own.count(k) / len(own), 4) for k in sorted(set(own))}
        rec.update(chat=True, chat_date=day, dataset="HuggingFaceH4/ultrachat_200k test_sft", dataset_index=i,
                   prompt_id=r.get("prompt_id"), conversation_tokens=len(ids), messages=len(msgs), scored_target_roles=share)
        json.dump(rec, open(os.path.join(a.out, f"k8_window_conv{n}.json"), "w"))
        print(f"WINDOW src=conv{n} text_sha={sha} dataset_index={i} conversation_tokens={len(ids)} messages={len(msgs)} "
              f"scored_target_roles={json.dumps(share)}", flush=True)
        if n >= a.n_conv:
            return
    raise SystemExit(f"only {n} test_sft conversations reach {need} tokens; {a.n_conv} needed")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["k8"]:
        rest = argv[1:]
        k8(rest[1:] if rest[:1] == ["--"] else rest)
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("windows")
    w.add_argument("--model", required=True)
    w.add_argument("--rev", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--suffix", required=True)
    w.add_argument("--steps", type=int, default=2048)
    w.add_argument("--harness-dir", default=HERE)
    w.add_argument("--ultrachat", default="", help="A1: the pinned ultrachat_200k test_sft parquet")
    w.add_argument("--n-conv", type=int, default=2)
    a = ap.parse_args(argv)
    return windows(a)


if __name__ == "__main__":
    sys.exit(main())
