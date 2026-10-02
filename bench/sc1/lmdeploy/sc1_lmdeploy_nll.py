"""sc1_lmdeploy_nll.py -- teacher-forced NLL of a K8 window through LMDeploy 0.18.0 / TurboMind (SC1 Phase D).

Input: k8_window_<src>.json = {"ids": [...], "text_sha": "..."}; text_sha = sha256 of ids[:P+S+1] as int64 LE bytes
(bench/p39/step_decomp.py:655-657), recomputed and REFUSED on mismatch. P=512, S=2048; scoring = step_decomp's K8
loop (step_decomp.py:1620-1641): position P+t scores ids[P+t+1], fp32 log_softmax, mean over S, ppl = exp.

What TurboMind exposes (file:line in InternLM/lmdeploy @ v0.18.0, 110965c7):
  * prefill-shaped logits for EVERY input position: GenerationConfig(output_logits='all', max_new_tokens=1)
    (messages.py:200; turbomind.py:616-622 offset 0; engine/model_request.cc:73-75 allocates the buffer in the engine's
    activation dtype -- fp16/bf16, NOT fp32 -- on the host); upstream's own `AsyncEngine.async_get_logits` uses exactly
    this (serve/core/async_engine.py:866-900). The driver up-casts to fp32 and records `logits_dtype`.
  * an engine-side fp32 cross-entropy over the whole input: GenerationConfig(return_ppl=True) -> `ce_loss` (float32,
    model_request.cc:78-80; kernels/cross_entropy_kernels.cu:30-60 reads the fp16/bf16 logits and reduces in fp32;
    models/output_processor.cc:197-205 scores position i against token_ids[i+1]). `Pipeline.get_ppl(input_ids)` returns
    ce_loss / (len-1), i.e. the MEAN CROSS-ENTROPY despite its name (pipeline.py:288-312, async_engine.py:902-936), over ALL
    positions, so it cannot read the window directly -- but causal prefixes share context, so
    nll_window = (ce_loss(ids[:P+S+1]) - ce_loss(ids[:P+1])) / S is exact up to engine nondeterminism. Reported as `ce_check`.
  * NO token-granular decode-shaped path: TurboMind has no "step one token with the cache" API; its prefix cache is
    block-granular (cache_block_seq_len=64, messages.py:400). The registered served-shape row for such engines is the
    one-token-extension design with prefix caching ON (`--mode decode_prefix`, labelled partial-block M <= 64:
    request k sends ids[:P+1+k] with output_logits='generation', max_new_tokens=1 -> the one logits row predicts
    ids[P+1+k]; `cached_tokens` per request gives the recomputed suffix). It is NOT T == 1 and the receipt says so.

Env: SC1_WINDOW, SC1_OUT, SC1_MODEL / SC1_REV / SC1_MODEL_DIR / SC1_LMD_FORMAT / SC1_LMD_DTYPE / SC1_LMD_QUANT_POLICY as the
arm; SC1_NLL_MODE or --mode = prefill (default) | decode_prefix.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import struct
import sys
import time

PROMPT_LEN, STEPS = 512, 2048
ENGINE = "lmdeploy-turbomind"
MODE_AVAILABLE = ["prefill", "decode_prefix(partial-block M<=64)"]
DECODE_NOTE = ("decode (T == 1) NOT available: TurboMind exposes no token-granular cache-stepping API; "
               "decode_prefix re-prefills a 1..64-token suffix per request (cache_block_seq_len=64)")
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
DEFAULT_REV = "9b534e4318b7ebc3c961a839f13eb18b1833f441"


class Refusal(SystemExit):
    pass


def text_sha(ids, prompt_len: int = PROMPT_LEN, steps: int = STEPS) -> str:
    n = prompt_len + steps + 1
    return hashlib.sha256(struct.pack(f"<{n}q", *[int(x) for x in ids[:n]])).hexdigest()


def load_window(path: str, prompt_len: int = PROMPT_LEN, steps: int = STEPS) -> dict:
    w = json.load(open(path))
    ids = [int(x) for x in w["ids"]]
    need = prompt_len + steps + 1
    if len(ids) < need:
        raise Refusal(f"REFUSED: window holds {len(ids)} ids, {need} needed for P={prompt_len} S={steps}")
    sha = text_sha(ids, prompt_len, steps)
    if sha != w.get("text_sha"):
        raise Refusal(f"REFUSED: text_sha {sha[:12]} != file {str(w.get('text_sha'))[:12]} (different window or tokenizer)")
    return {"ids": ids[:need], "text_sha": sha}


def nll_from_logits_rows(rows, targets) -> float:
    """rows: float32 tensor [S, V] of logits at positions P..P+S-1; targets: [S] ids. fp32 log_softmax mean NLL."""
    import torch
    lp = torch.log_softmax(rows.float(), dim=-1)
    return float(-lp.gather(1, torch.as_tensor(targets, device=lp.device).unsqueeze(1)).sum().item() / len(targets))


def ce_window(ce_long: float, ce_short: float, steps: int = STEPS) -> float:
    return (ce_long - ce_short) / steps


def receipt(mode: str, mean_nll: float, sha: str, wall_s: float, extra: dict | None = None) -> dict:
    r = {"engine": ENGINE, "mode": mode, "mode_available": MODE_AVAILABLE, "decode_note": DECODE_NOTE, "steps": STEPS,
         "prompt_len": PROMPT_LEN, "mean_nll": mean_nll, "ppl": math.exp(mean_nll), "tokens_scored": STEPS, "text_sha": sha,
         "wall_s": round(wall_s, 2),
         "basis": ("teacher-forced; prefill: output_logits='all' over ids[:P+S+1], positions P..P+S-1 score ids[P+1..P+S], "
                   "fp32 log_softmax over the engine-dtype logits; ce_check: engine-side fp32 ce_loss(prefix P+S+1) - ce_loss(prefix P+1)")}
    if extra:
        r.update(extra)
    return r


RECEIPT_KEYS = ("mean_nll", "ppl", "steps", "prompt_len", "mode", "mode_available", "text_sha", "engine", "version",
                "logits_dtype", "wall_s")


# ------------------------------------------------------------------------------------------------ engine side

def resolve_model_dir(model: str, rev: str) -> str:
    d = os.environ.get("SC1_MODEL_DIR")
    if d:
        return d
    from huggingface_hub import snapshot_download
    return snapshot_download(model, revision=rev, max_workers=4)


class TM:
    def __init__(self, model_dir: str, session_len: int, prefix_caching: bool):
        from lmdeploy import TurbomindEngineConfig
        from lmdeploy.turbomind import is_available
        if not is_available():
            from lmdeploy.turbomind import _import_error
            raise Refusal(f"REFUSED/UNSUPPORTED: lmdeploy.turbomind binding not importable: {_import_error!r}")
        from lmdeploy.turbomind import TurboMind
        self.kwargs = dict(model_format=os.environ.get("SC1_LMD_FORMAT") or None, session_len=session_len, max_batch_size=1,
                           cache_max_entry_count=float(os.environ.get("SC1_LMD_CACHE_FRAC") or 0.8),
                           enable_prefix_caching=prefix_caching, quant_policy=int(os.environ.get("SC1_LMD_QUANT_POLICY") or 0),
                           dtype=os.environ.get("SC1_LMD_DTYPE") or "auto", tp=1, enable_metrics=True)
        self.tm = TurboMind.from_pretrained(model_dir, engine_config=TurbomindEngineConfig(**self.kwargs))
        self.serial = 0

    def run(self, ids, **gen_kwargs):
        """One request on a fresh instance; returns the final EngineOutput (logits/ce_loss/req_metrics attached)."""
        from lmdeploy import GenerationConfig
        gc = GenerationConfig(top_k=1, top_p=1.0, temperature=1.0, ignore_eos=True, **gen_kwargs)

        async def go():
            self.serial += 1
            inst = self.tm.create_instance()
            last = None
            async for out in inst.async_stream_infer(session_id=self.serial, input_ids=list(ids), gen_config=gc, stream_output=False):
                last = out
            return last

        return asyncio.run(go())


def score_prefill(tm: TM, ids, chunk: int = 128) -> tuple[float, str, float, dict]:
    import torch
    out = tm.run(ids, max_new_tokens=1, output_logits="all")
    logits = out.logits                                     # [P+S+1 (+1), V] host tensor in the engine dtype
    dtype = str(logits.dtype)
    if logits.shape[0] < PROMPT_LEN + STEPS:
        raise Refusal(f"REFUSED: engine returned {logits.shape[0]} logits rows, {PROMPT_LEN + STEPS} needed")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tgt = torch.tensor(ids[PROMPT_LEN + 1:PROMPT_LEN + STEPS + 1], device=dev)
    tot = 0.0
    for s in range(0, STEPS, chunk):
        e = min(STEPS, s + chunk)
        rows = logits[PROMPT_LEN + s:PROMPT_LEN + e].to(dev)
        tot += nll_from_logits_rows(rows, tgt[s:e]) * (e - s)
    nll = tot / STEPS
    ce_l = tm.run(ids[:PROMPT_LEN + STEPS + 1], max_new_tokens=1, return_ppl=True).ce_loss
    ce_s = tm.run(ids[:PROMPT_LEN + 1], max_new_tokens=1, return_ppl=True).ce_loss
    ce = ce_window(float(ce_l), float(ce_s))
    return nll, dtype, ce, {"ce_loss_long_sum": float(ce_l), "ce_loss_short_sum": float(ce_s), "ce_minus_logits_nll": ce - nll}


def score_decode_prefix(tm: TM, ids) -> tuple[float, str, dict]:
    tot, dtype, cached, suffix = 0.0, None, [], []
    for k in range(STEPS):
        pre = ids[:PROMPT_LEN + 1 + k]
        out = tm.run(pre, max_new_tokens=1, output_logits="generation")
        row = out.logits[0:1]
        dtype = dtype or str(row.dtype)
        tot += nll_from_logits_rows(row, [ids[PROMPT_LEN + 1 + k]])
        rm = getattr(out, "req_metrics", None)
        c = getattr(rm, "cached_tokens", None) if rm is not None else None
        cached.append(c)
        suffix.append(None if c is None else len(pre) - int(c))
    known = [s for s in suffix if s is not None]
    return tot / STEPS, dtype, {"cache_granularity_tokens": 64, "mean_recomputed_suffix": (sum(known) / len(known)) if known else None,
                                "max_recomputed_suffix": max(known) if known else None, "cached_tokens_sample": cached[:8],
                                "label": "partial-block (M <= 64), not T == 1"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prefill", "decode_prefix"], default=os.environ.get("SC1_NLL_MODE", "prefill"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    win = load_window(os.environ["SC1_WINDOW"])
    model_id, rev = os.environ.get("SC1_MODEL", DEFAULT_MODEL), os.environ.get("SC1_REV", DEFAULT_REV)
    model_dir = resolve_model_dir(model_id, rev)
    import torch
    import lmdeploy
    t0 = time.perf_counter()
    tm = TM(model_dir, session_len=PROMPT_LEN + STEPS + 8, prefix_caching=(a.mode == "decode_prefix"))
    load_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    extra = {}
    if a.mode == "prefill":
        nll, dtype, ce, extra = score_prefill(tm, win["ids"])
        extra["ce_check"] = {"mean_nll_ce": ce, "ppl_ce": math.exp(ce)}
        mode = "prefill"
    else:
        nll, dtype, extra = score_decode_prefix(tm, win["ids"])
        mode = "decode_prefix(partial-block M<=64)"
    wall = time.perf_counter() - t1
    ec = tm.tm.engine_config
    rec = receipt(mode, nll, win["text_sha"], wall, {
        "version": lmdeploy.__version__, "torch": torch.__version__, "torch_cuda": torch.version.cuda, "logits_dtype": dtype,
        "load_s": round(load_s, 1), "model": model_id, "revision": rev, "model_dir": model_dir,
        "engine_config": {"model_format": str(ec.model_format), "dtype": str(ec.dtype), "quant_policy": int(ec.quant_policy),
                          "session_len": ec.session_len, "enable_prefix_caching": ec.enable_prefix_caching,
                          "cache_block_seq_len": ec.cache_block_seq_len, "cache_max_entry_count": ec.cache_max_entry_count},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, **extra})
    json.dump(rec, open(os.environ["SC1_OUT"], "w"), indent=1)
    print(f"SC1LMD_NLL mode={mode} nll={nll:.5f} ppl={rec['ppl']:.5f} dtype={dtype} sha={win['text_sha'][:12]}", flush=True)
    return 0


def selftest() -> int:
    import tempfile
    ids = [(i * 17 + 3) % 151936 for i in range(PROMPT_LEN + STEPS + 1)]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"ids": ids, "text_sha": text_sha(ids)}, f)
    w = load_window(f.name)
    assert len(w["ids"]) == PROMPT_LEN + STEPS + 1
    with open(f.name, "w") as g:
        json.dump({"ids": ids[:100], "text_sha": text_sha(ids)}, g)
    try:
        load_window(f.name)
        raise AssertionError("short-window refusal did not fire")
    except Refusal:
        pass
    assert abs(ce_window(4096.0, 1024.0) - 1.5) < 1e-12
    try:
        import torch
        rows = torch.zeros(4, 1000)                      # uniform logits -> ln V
        nll = nll_from_logits_rows(rows, [1, 2, 3, 4])
        assert abs(nll - math.log(1000)) < 1e-5, nll
        dtype = "torch.float32"
    except ImportError:
        nll, dtype = math.log(1000), "n/a"
    r = receipt("prefill", nll, w["text_sha"], 0.1, {"version": "fake", "logits_dtype": dtype})
    missing = [k for k in RECEIPT_KEYS if k not in r]
    assert not missing and "decode_note" in r, missing
    print("SELFTEST OK sc1_lmdeploy_nll")
    return 0


if __name__ == "__main__":
    sys.exit(main())
