#!/usr/bin/env python3
"""sc1g_ref.py -- lane SC1g amendment A4 (#846), box R: the reference side of the fidelity instrument.

gpt-oss-20b's shipped MXFP4 bytes dequantised to bf16 (P44's reference, `Mxfp4Config(dequantize=True)`, refused if any
packed tensor survives) scores each registered window DECODE-SHAPED (kl_fidelity.decode_teacher_forced_logits: one token
per forward over a KV cache), and for each window writes the artifact box I consumes (sc1g_kl.reference_rows): per scored
position the reference's top-64 token ids and fp64 log-probs, its rest mass, the target and its log-prob.

On the same box, where both full distributions exist, the estimator is calibrated (sc1g_kl.calibrate) on two real
perturbations of the reference:
  (i)  self   the reference scored PREFILL-shaped against its decode-shaped self: the arithmetic-order floor F (P44's
              self-consistency control; F must stay under SELF_CONSISTENCY_MAX = 1e-2 or the decode scorer is refused);
  (ii) nf4    the reference with every expert matrix fake-quantised to NF4 in place (per 64-element K-block absmax,
              nearest NF4 code -- gnf4's quantize_pack_nf4 + dequant_ref, cross-checked against gnf4 on the box), both
              scored prefill-shaped: the requantisation scale e4b's NF4 path sits at.
KL65 / KL_full >= 0.90 on both pairs, on every window, is the instrument's gate. R's own verdict (r_verdict.json) is
decided here, before anything consumes the artifacts: per check OK / UNREAD / VOID, and R_OK only when every check is OK.

The windows are the committed ones (bench/h2h-2026-10-02/sc1g/receipts/sc1g-diag-2/sc1g/k8_window_*.json); each must
hash to its registered text_sha or R refuses (VOID). Descriptive, not predicted: the reference's own NLL per window, and
the NF4 fake-quant's, so the flattery hypothesis's direction is read off the reference itself.

  sc1g_ref.py --model openai/gpt-oss-20b --rev 6cee5e81... --windows DIR --shas SHAS.json --k0 k0.json --out DIR
  sc1g_ref.py --self-test       (CPU: a tiny random gpt-oss through the whole flow)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (HERE, os.path.join(HERE, "..")):              # staged flat on the box; bench/ in the repo
    if os.path.exists(os.path.join(cand, "kl_fidelity.py")):
        sys.path.insert(0, cand)
        break
sys.path.insert(0, HERE)
import sc1g_kl as KL  # noqa: E402
from kl_fidelity import decode_teacher_forced_logits, teacher_forced_logits  # noqa: E402

SELF_CONSISTENCY_MAX = 1e-2          # P44 amendment 5 (bench/p44/kl_serve.py): the reference must agree with itself
SRCS = ("conv1", "conv2", "conv3", "conv4", "wikitext")
NF4_LUT = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
           -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
           0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]
BLOCK = 64


def load_reference(model_id: str, revision: str, dev: str):
    """P44's gpt-oss reference (bench/p44/kl_serve.py load_reference, family gptoss), verbatim in substance."""
    from transformers import AutoModelForCausalLM, Mxfp4Config
    m = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16, device_map=dev,
                                             quantization_config=Mxfp4Config(dequantize=True))
    low = [n for n, t in list(m.named_parameters()) + list(m.named_buffers())
           if t.dtype in (torch.uint8, torch.int8) or "blocks" in n.split(".")[-1] and t.dtype not in (torch.bfloat16, torch.float32, torch.float16)]
    if low:
        raise RuntimeError(f"gpt-oss reference still carries packed/quantised tensors ({len(low)}: {low[:3]}) -- not the dequant path")
    for n, t in m.named_parameters():
        if "experts" in n and t.dtype != torch.bfloat16:
            raise RuntimeError(f"gpt-oss reference expert tensor {n} is {t.dtype}, expected bf16 after dequantize=True")
    m.config.use_cache = True
    return m.eval()


def window_ids(path: str, want_sha: str):
    """The window's ids, refused unless they hash (int64 little-endian over ids[:prompt_len+steps+1], step_decomp's
    digest) to the registered sha. Returns (ids, prompt_len, steps)."""
    rec = json.load(open(path))
    P, S = int(rec["prompt_len"]), int(rec["steps"])
    ids = np.asarray(rec["ids"][:P + S + 1], dtype="<i8")
    sha = hashlib.sha256(ids.tobytes()).hexdigest()
    if sha != want_sha or rec.get("text_sha") != want_sha:
        raise SystemExit(f"{path}: ids hash to {sha[:16]}, file says {str(rec.get('text_sha'))[:16]}, registered {want_sha[:16]}")
    return torch.as_tensor(ids, dtype=torch.long), P, S


def fake_nf4(w: torch.Tensor) -> torch.Tensor:
    """dequant(quant(w)) for an [N, K] matrix, blocks along K: what gnf4's quantize_pack_nf4 + dequant_ref return."""
    N, K = w.shape
    if K % BLOCK:
        raise ValueError(f"K={K} is not a multiple of {BLOCK}")
    lut = torch.tensor(NF4_LUT, dtype=torch.float32, device=w.device)
    out = torch.empty(N, K, dtype=torch.float32, device=w.device)
    for r in range(0, N, 512):                                # the argmin materialises [rows, K, 16] fp32
        b = w[r:r + 512].float().reshape(-1, K // BLOCK, BLOCK)
        am = b.abs().amax(dim=2).clamp_min(1e-12)
        codes = ((b / am[:, :, None]).reshape(-1, K, 1) - lut).abs().argmin(dim=2)
        out[r:r + 512] = (lut[codes].reshape(-1, K // BLOCK, BLOCK) * am[:, :, None]).reshape(-1, K)
    return out


def gnf4_crosscheck(w: torch.Tensor):
    """On the box: fake_nf4 must equal gnf4's own quantize_pack_nf4 -> dequant_ref on a real expert matrix. None when
    gnf4 cannot be imported (CPU tests)."""
    try:
        from nf4_grouped import dequant_ref
        from nf4_pack_ref import quantize_pack_nf4
    except Exception as e:                                   # noqa: BLE001 -- recorded, decided by the caller
        return {"available": False, "why": repr(e)[:160]}
    N, K = w.shape
    pk, am = quantize_pack_nf4(w.float())
    ref = dequant_ref(pk, am, N, K)
    mine = fake_nf4(w)
    return {"available": True, "max_abs_diff": float((ref.to(mine.device) - mine).abs().max()), "equal": bool(torch.equal(ref.to(mine.device), mine))}


def fake_nf4_experts_(model) -> dict:
    """Every gpt-oss expert matrix fake-quantised to NF4 IN PLACE. gate_up_proj [E, H, 2I] and down_proj [E, I, H] are
    used as x @ W[e] (transformers' GptOssExperts), so K is dim 1 of W[e]: quantise W[e].T [N, K] and write back."""
    n, first = 0, None
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not (name.endswith("mlp.experts.gate_up_proj") or name.endswith("mlp.experts.down_proj")):
                continue
            for e in range(p.shape[0]):
                wt = p.data[e].t()
                if first is None:
                    first = gnf4_crosscheck(wt)
                p.data[e].copy_(fake_nf4(wt).t().to(p.dtype))
                n += 1
    if n == 0:
        raise RuntimeError("no gpt-oss expert matrices found to fake-quantise (parameter names changed?)")
    return {"matrices": n, "gnf4_crosscheck": first}


def nll_of(rows):
    return float(-np.mean(rows["target_lp"]))


def score(model, srcs, windows, out_dir, dev, nf4_pair=True, log=print) -> dict:
    """The whole R flow on an already-loaded reference `model`. `windows`: {src: (ids, prompt_len, steps)}."""
    os.makedirs(out_dir, exist_ok=True)
    res = {"windows": {}, "artifacts": {}}
    prefill_keep = {}
    for src in srcs:
        ids, P, S = windows[src]
        x = ids[:P + S].to(dev)                               # the logit at i predicts ids[i + 1]; scored i = P .. P+S-1
        tg = ids[P + 1:P + S + 1].numpy()
        t0 = time.time()
        dec = decode_teacher_forced_logits(model, x)[P:P + S]
        t_dec = time.time() - t0
        rows = KL.reference_rows(dec, tg)
        sha = KL.save_artifact(os.path.join(out_dir, f"ref_{src}.npz"), rows,
                               {"source": src, "prompt_len": P, "steps": S, "shape": "decode", "k": KL.K_NAMED,
                                "reference": "gpt-oss-20b MXFP4 dequant-to-bf16 (P44)", "metric": "sc1g_kl KL65"})
        res["artifacts"][src] = sha
        pre = teacher_forced_logits(model, x)[P:P + S]
        c_self = KL.calibrate(dec, pre)
        pre_rows = KL.reference_rows(pre, tg)
        res["windows"][src] = {"positions": int(S), "decode_s": round(t_dec, 1), "reference_nll_decode": nll_of(rows),
                               "reference_nll_prefill": nll_of(pre_rows),
                               "coverage_ref_top64": 1.0 - float(np.mean(np.maximum(rows["rest"], 0.0))),
                               "calib_self": c_self, "floor_F": c_self["kl_full_mean"]}
        prefill_keep[src] = pre.to("cpu")
        log(f"R {src}: nll dec {nll_of(rows):.5f} pre {nll_of(pre_rows):.5f} F {c_self['kl_full_mean']:.3e} "
            f"self ratio {c_self['ratio']} cov {res['windows'][src]['coverage_ref_top64']:.5f} sha {sha[:12]}")
        del dec, pre
    if nf4_pair:
        res["nf4"] = fake_nf4_experts_(model)
        for src in srcs:
            ids, P, S = windows[src]
            x = ids[:P + S].to(dev)
            q = teacher_forced_logits(model, x)[P:P + S]
            base = prefill_keep[src].to(q.device)
            c_nf4 = KL.calibrate(base, q)
            res["windows"][src]["calib_nf4"] = c_nf4
            res["windows"][src]["nf4_fakequant_nll_prefill"] = nll_of(KL.reference_rows(q, ids[P + 1:P + S + 1].numpy()))
            log(f"R {src}: nf4 KL_full {c_nf4['kl_full_mean']:.3e} ratio {c_nf4['ratio']}")
            del q, base
    return res


def verdict(res: dict, k0: dict | None, srcs) -> dict:
    """R's own gate: per check OK / UNREAD / VOID; R_OK only when every check is OK."""
    w = res["windows"]
    chk = {}
    chk["k0_controls"] = ("OK" if k0 and k0.get("all_passed") else "VOID")
    chk["windows_complete"] = "OK" if all(s in w for s in srcs) else "VOID"
    Fs = [w[s]["floor_F"] for s in srcs if s in w]
    chk["floor_F"] = "OK" if Fs and max(Fs) < SELF_CONSISTENCY_MAX else "UNREAD"
    rs = [w[s]["calib_self"]["ratio"] for s in srcs if s in w]
    chk["calib_self"] = "OK" if rs and all(KL.calib_verdict(r) == "OK" for r in rs) else "UNREAD"
    rn = [w[s].get("calib_nf4", {}).get("ratio") for s in srcs if s in w]
    chk["calib_nf4"] = "OK" if rn and all(KL.calib_verdict(r) == "OK" for r in rn) else "UNREAD"
    cv = [w[s]["coverage_ref_top64"] for s in srcs if s in w]
    chk["coverage"] = "OK" if cv and min(cv) >= KL.COVERAGE_MIN else "UNREAD"
    xc = (res.get("nf4") or {}).get("gnf4_crosscheck") or {}
    # VOID unless gnf4 imported AND matched bit for bit: an import failure must not leave the nf4 pair unverified
    chk["nf4_matches_gnf4"] = "OK" if (xc.get("available") and xc.get("equal")) else "VOID"
    ok = all(v == "OK" for v in chk.values())
    return {"verdict": "R_OK" if ok else "R_NOT_OK", "checks": chk, "floor_F_max": max(Fs) if Fs else None,
            "calib_self_min": min((r for r in rs if r is not None), default=None),
            "calib_nf4_min": min((r for r in rn if r is not None), default=None), "coverage_min": min(cv) if cv else None,
            "artifacts": res.get("artifacts", {})}


def _tiny_model():
    from transformers import GptOssConfig, GptOssForCausalLM
    cfg = GptOssConfig(vocab_size=320, hidden_size=64, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=16, num_local_experts=4, num_experts_per_tok=2, sliding_window=8,
                       layer_types=["sliding_attention", "full_attention"], max_position_embeddings=256)
    torch.manual_seed(0)
    m = GptOssForCausalLM(cfg).eval()
    m.config.use_cache = True
    return m


def self_test() -> int:
    import tempfile
    cases = []
    w = torch.randn(96, 128)
    q = fake_nf4(w)
    am = w.reshape(96, 2, 64).abs().amax(-1)
    lut = torch.tensor(NF4_LUT)
    on_grid = ((q.reshape(96, 2, 64) / am[..., None]).unsqueeze(-1) - lut).abs().min(-1).values.max() < 1e-6
    cases.append(("fake_nf4 lands on the NF4 grid, blockwise", bool(on_grid) and float((q - w).abs().max()) < float(am.max())))
    m = _tiny_model()
    n_before = {n: p.detach().clone() for n, p in m.named_parameters() if "experts" in n}
    g = torch.Generator().manual_seed(1)
    wins = {}
    for s in ("conv1", "wikitext"):
        ids = torch.randint(0, 320, (12 + 20 + 1,), generator=g)
        wins[s] = (ids, 12, 20)
    with tempfile.TemporaryDirectory() as d:
        res = score(m, list(wins), wins, d, "cpu", nf4_pair=True, log=lambda *_: None)
        art = KL.load_artifact(os.path.join(d, "ref_conv1.npz"), res["artifacts"]["conv1"])
        cases.append(("artifact: 20 positions x 64", art["ids"].shape == (20, 64)))
        cases.append(("decode vs prefill floor is tiny on fp32 CPU", res["windows"]["conv1"]["floor_F"] < 1e-6))
        changed = any(not torch.equal(n_before[n], p) for n, p in m.named_parameters() if "experts" in n)
        cases.append(("nf4 pass changed the experts", changed and res["nf4"]["matrices"] == 2 * 2 * 4))
        nf = res["windows"]["conv1"]["calib_nf4"]
        cases.append(("nf4 pair: KL65 <= full, ratio in (0, 1]", nf["kl65_le_full_everywhere"] and 0 < nf["ratio"] <= 1 + 1e-9))
        v = verdict(res, {"all_passed": True}, list(wins))
        cases.append(("verdict vocabulary", set(v["checks"].values()) <= {"OK", "UNREAD", "VOID"}))
        unver = dict(res, nf4={"matrices": 1, "gnf4_crosscheck": {"available": False, "why": "ImportError"}})
        cases.append(("gnf4 unavailable -> VOID, not R_OK", verdict(unver, {"all_passed": True}, list(wins))["checks"]["nf4_matches_gnf4"] == "VOID"
                      and verdict(unver, {"all_passed": True}, list(wins))["verdict"] == "R_NOT_OK"))
        cases.append(("no K0 receipt -> VOID, not R_OK", verdict(res, None, list(wins))["verdict"] == "R_NOT_OK"))
        bad = dict(res, windows={k: dict(x, floor_F=0.5) for k, x in res["windows"].items()})
        cases.append(("floor over 1e-2 -> UNREAD", verdict(bad, {"all_passed": True}, list(wins))["checks"]["floor_F"] == "UNREAD"))
    bad_cases = [n for n, ok in cases if not ok]
    print(f"sc1g_ref self-test {'OK' if not bad_cases else 'FAILED ' + str(bad_cases)} ({len(cases)} cases)")
    return 0 if not bad_cases else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--rev", default="")
    ap.add_argument("--windows", help="dir holding k8_window_<src>.json")
    ap.add_argument("--shas", help="json {src: registered text_sha}")
    ap.add_argument("--k0", help="kl_fidelity --controls receipt from THIS host")
    ap.add_argument("--srcs", default=",".join(SRCS))
    ap.add_argument("--out")
    ap.add_argument("--no-nf4-pair", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.windows and a.shas and a.out and a.rev):
        ap.error("--windows, --shas, --rev and --out are required")
    srcs = [s for s in a.srcs.split(",") if s]
    shas = json.load(open(a.shas))
    wins = {s: window_ids(os.path.join(a.windows, f"k8_window_{s}.json"), shas[s]) for s in srcs}
    print("SC1G_R windows " + " ".join(f"{s}={shas[s][:12]}" for s in srcs), flush=True)
    k0 = json.load(open(a.k0)) if a.k0 and os.path.exists(a.k0) else None
    t0 = time.time()
    model = load_reference(a.model, a.rev, "cuda")
    print(f"SC1G_R reference loaded in {time.time() - t0:.0f} s", flush=True)
    res = score(model, srcs, wins, a.out, "cuda", nf4_pair=not a.no_nf4_pair, log=lambda s: print(s, flush=True))
    v = verdict(res, k0, srcs)
    json.dump(res, open(os.path.join(a.out, "r_calib.json"), "w"), indent=1, sort_keys=True)
    json.dump(v, open(os.path.join(a.out, "r_verdict.json"), "w"), indent=1, sort_keys=True)
    print(f"SC1G_R_VERDICT {v['verdict']} " + " ".join(f"{k}={x}" for k, x in v["checks"].items())
          + f" F_max={v['floor_F_max']} self_min={v['calib_self_min']} nf4_min={v['calib_nf4_min']} cov_min={v['coverage_min']}", flush=True)
    for s in srcs:
        x = res["windows"][s]
        print(f"SC1G_R_NLL {s} reference_decode={x['reference_nll_decode']:.5f} reference_prefill={x['reference_nll_prefill']:.5f} "
              f"nf4_fakequant_prefill={x.get('nf4_fakequant_nll_prefill')}", flush=True)
    return 0 if v["verdict"] == "R_OK" else 1


if __name__ == "__main__":
    sys.exit(main())
