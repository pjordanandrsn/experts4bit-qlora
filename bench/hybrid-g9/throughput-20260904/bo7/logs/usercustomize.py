"""Generalization lane hook, model-agnostic. The int4 expert stores need
the hot-residency (all-VRAM) state, which the harness builds AFTER the
model loads, so the lanes are applied right after the hybrid tier is
enabled -- the same ordering the Qwen lanes had by hooking fuse_qkv,
without depending on a Qwen-only entry point. Refusals print a banner
and re-raise so the arm FAILS loudly and the summary records why.

  E4B_SERVE_EXP_INT4=1          int4-b32 expert stores (per model_type)
  E4B_SERVE_EXP_INT4_CALIB=1    + per-expert GPTQ calibration (hook v4); E4B_CALIB_NSEQ=N sizes the set (hook v5)
  E4B_SERVE_ATTN_INT4_CALIB=1   calibrated int4 attention (structural)
  E4B_CALIB_SOURCE=c4|wikitext  calibration text (default c4)
"""
import os
if (os.environ.get("E4B_SERVE_EXP_INT4", "0") == "1" or os.environ.get("E4B_SERVE_ATTN_INT4_CALIB", "0") == "1"
        or os.environ.get("E4B_SERVE_LMHEAD_INT4_CALIB", "0") == "1" or os.environ.get("E4B_SERVE_DENSE_INT4_CALIB", "0") == "1"):
    import importlib

    def _calib_batches(tok, n_seq=None, seq_len=512, bsz=4):
        # hook v5: E4B_CALIB_NSEQ scales the calibration set (default 32 x 512 tokens)
        n_seq = int(os.environ.get("E4B_CALIB_NSEQ", "32")) if n_seq is None else n_seq
        from datasets import load_dataset
        import torch
        src = os.environ.get("E4B_CALIB_SOURCE", "c4")
        if src == "c4":
            ds = load_dataset("allenai/c4", data_files={"v": "en/c4-validation.00000-of-00008.json.gz"}, split="v")
            text = "\n\n".join(ds["text"][:4000])
        else:
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            text = "\n\n".join(t for t in ds["text"] if t.strip())
        text = text[:6_000_000]
        ids = tok(text, return_tensors="pt").input_ids[0]
        step = max(1, (ids.numel() - seq_len) // n_seq)
        rows = [ids[i * step:i * step + seq_len] for i in range(n_seq)]
        return [torch.stack(rows[i:i + bsz]) for i in range(0, n_seq, bsz)]

    def _apply_lanes(model):
        mt = getattr(getattr(model, "config", None), "model_type", "?")
        model_id = getattr(getattr(model, "config", None), "_name_or_path", None) or os.environ.get("E4B_MODEL_ID")
        if os.environ.get("E4B_SERVE_EXP_INT4", "0") == "1":
            from huggingface_hub import snapshot_download
            from experts4bit_qlora.engines.int4_experts import enable_serve_experts_int4
            try:
                src = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors"])
                if os.environ.get("E4B_SERVE_EXP_INT4_CALIB", "0") == "1":
                    # hook v4: calibrated (GPTQ-style) int4 experts -- per-expert Hessians from the
                    # fused forward's tap over the same C4 batches the attention lane uses
                    from transformers import AutoTokenizer
                    from experts4bit_qlora.engines.int4_experts import calibrate_expert_hessians
                    tok = AutoTokenizer.from_pretrained(model_id)
                    batches = _calib_batches(tok)
                    try:  # hook v6: stream calibration + packing per layer chunk (bounded host memory)
                        from experts4bit_qlora.engines.int4_experts import enable_serve_experts_int4_calibrated
                    except ImportError:
                        enable_serve_experts_int4_calibrated = None
                    if enable_serve_experts_int4_calibrated is not None:
                        print("INT4EXP calibrating (streamed):", len(batches), "batches of", os.environ.get("E4B_CALIB_SOURCE", "c4"),
                              "budget GB", os.environ.get("E4B_INT4_HESSIAN_BUDGET_GB", "24"), flush=True)
                        n = enable_serve_experts_int4_calibrated(model, src, batches)
                    else:
                        hs = calibrate_expert_hessians(model, src, batches)
                        n_exp = sum(len(v) for v in hs.values())
                        print("INT4EXP hessians:", len(hs), "layers,", n_exp, "(layer, expert) pairs from", len(batches), "batches of",
                              os.environ.get("E4B_CALIB_SOURCE", "c4"), flush=True)
                        n = enable_serve_experts_int4(model, src, expert_hessians=hs)
                else:
                    n = enable_serve_experts_int4(model, src)
                print("INT4EXP enabled:", n, "layers (model_type=%s)" % mt, flush=True)
            except Exception as e:
                print("INT4EXP REFUSED (model_type=%s): %s" % (mt, repr(e)[:300]), flush=True)
                raise
        _attn = os.environ.get("E4B_SERVE_ATTN_INT4_CALIB", "0") == "1"
        _head = os.environ.get("E4B_SERVE_LMHEAD_INT4_CALIB", "0") == "1"
        _dense = os.environ.get("E4B_SERVE_DENSE_INT4_CALIB", "0") == "1"
        if _attn or _head or _dense:
            from transformers import AutoTokenizer
            from experts4bit_qlora.engines.int4_attn_calib import (
                calibrate_attention_hessians, enable_serve_attn_int4_calib)
            try:
                tok = AutoTokenizer.from_pretrained(model_id)
                batches = _calib_batches(tok)
                hs = calibrate_attention_hessians(model, batches, include_attention=_attn, include_head=_head, include_dense_mlp=_dense)
                print("ATTNINT4 calibrated:", enable_serve_attn_int4_calib(model, hs, include_attention=_attn, include_head=_head, include_dense_mlp=_dense),
                      "projections (attn=%d head=%d dense=%d; hessians from" % (_attn, _head, _dense), len(batches), "batches of",
                      os.environ.get("E4B_CALIB_SOURCE", "c4") + "; model_type=%s)" % mt, flush=True)
            except Exception as e:
                print("ATTNINT4 REFUSED (model_type=%s): %s" % (mt, repr(e)[:300]), flush=True)
                raise

    _patched = []
    for _modname in ("experts4bit_qlora.engines.hybrid", "experts4bit_qlora.engines.hot_residency",
                     "experts4bit_qlora.engines.nvme_experts", "experts4bit_qlora.engines.placement"):
        try:
            _m = importlib.import_module(_modname)
        except Exception:
            continue
        _orig = getattr(_m, "enable_hybrid_tier", None)
        if _orig is None or getattr(_orig, "_gen_hooked", False):
            continue

        def _make(orig):
            def _wrapped(model, *a, **k):
                out = orig(model, *a, **k)
                _apply_lanes(model)
                return out
            _wrapped._gen_hooked = True
            return _wrapped
        setattr(_m, "enable_hybrid_tier", _make(_orig)); _patched.append(_modname)
    print("GEN HOOK: lanes attached after enable_hybrid_tier in", _patched or "NOTHING (hook point missing!)", flush=True)
