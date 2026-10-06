#!/usr/bin/env python3
"""sc1g_kl.py -- lane SC1g amendment A4 (#846): the fidelity estimator. KL from a bf16-dequant reference of gpt-oss-20b to
an engine, read on a 65-bucket partition of the vocabulary, plus everything around it that A4 registers.

THE ESTIMATOR (KL65). At each scored position the reference box R stores its own top-64 tokens (ids, fp64 log-probs) and
the mass on everything else (`rest`). The engine is asked for its log-probs on exactly those 64 named tokens (vLLM
`logprob_token_ids`, SGLang `token_ids_logprob`, llama.cpp's harness, e4b's full row) and its own rest is 1 - sum. Then

    KL65 = sum_k p_k (log p_k - log q_k) + p_rest (log p_rest - log q_rest)

which is KL exactly on the partition {64 named tokens, rest}. By the data-processing inequality it is a LOWER bound on the
full-vocabulary KL: truncation biases downward, the direction kl_fidelity.assert_full_vocab refuses. A4 therefore measures
that bias where both full distributions exist (on R: `calibrate`) and registers KL65 / KL_full >= 0.90 on two real
perturbation pairs as the instrument's gate. A comparator whose deviation has a different tail shape is covered by the
lower-bound property, not by that calibration.

The registered rules applied here:
  * the rest bucket is clamped at REST_EPS (1e-9) on either side; a window whose engine-side clamps exceed CLAMP_SHARE_MAX
    (1 %) of its positions is UNREAD;
  * a window whose reference top-64 mass averages under COVERAGE_MIN (0.99) is UNREAD;
  * the top-K FALLBACK (an engine or shape that cannot name tokens): the engine's own top-K (K >= 256); coverage = the
    reference top-64 mass falling inside it, a window under COVERAGE_MIN is UNREAD; an uncovered reference token gets the
    engine's residual mass spread evenly over the vocabulary it did not return (`spread`), and the bound that gives it
    the whole residual (`bound`) is reported beside it.

A5 (after box R read R_NOT_OK under A4: KL65 under-read full KL by 10-25 % on these texts) moves to A4's registered
FALLBACK, the full-vocabulary KL: R stores each window's reference log-softmax rows in fp16 (`full_rows_fp16`, -inf kept
exact for masked entries), the engines compute KL(p_ref || p_engine) over all 201,088 tokens per position (`kl_full_rows`),
and R measures what the fp16 storage costs against the fp64 rows on both calibration pairs (`storage_error`): it must stay
under STORAGE_F_FRACTION x the window's floor F. A window is graded only if its own F < GRADABLE_F_MAX.

  sc1g_kl.py --self-test
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

K_NAMED = 64
REST_EPS = 1e-9
CLAMP_SHARE_MAX = 0.01
COVERAGE_MIN = 0.99
CALIB_RATIO_MIN = 0.90
TOPK_FALLBACK_MIN = 256
ARTIFACT_KEYS = ("ids", "lp", "rest", "target", "target_lp")


# ----------------------------------------------------------------------------------------------- the reference artifact

def reference_rows(logits, targets, k: int = K_NAMED, chunk: int = 128) -> dict:
    """From reference logits [P, V] (any float dtype, any device) and targets [P]: per position the top-k ids and fp64
    log-probs, the rest mass (computed as its own log-sum-exp, not as 1 - sum, so it keeps its digits when tiny), the
    target's id and log-prob. Returns numpy arrays (the artifact's content)."""
    import torch
    P = int(logits.shape[0])
    ids = np.empty((P, k), np.int32)
    lp = np.empty((P, k), np.float64)
    rest = np.empty(P, np.float64)
    tlp = np.empty(P, np.float64)
    tg = torch.as_tensor(np.asarray(targets), dtype=torch.long)
    for a in range(0, P, chunk):
        b = min(P, a + chunk)
        x = logits[a:b].to(torch.float64)
        ls = torch.log_softmax(x, dim=-1)
        top = torch.topk(ls, k, dim=-1)
        ids[a:b] = top.indices.to(torch.int32).cpu().numpy()
        lp[a:b] = top.values.cpu().numpy()
        mask = torch.ones_like(ls, dtype=torch.bool)
        mask.scatter_(1, top.indices, False)
        rest[a:b] = torch.logsumexp(ls.masked_fill(~mask, float("-inf")), dim=-1).exp().cpu().numpy()
        tlp[a:b] = ls.gather(1, tg[a:b, None].to(ls.device))[:, 0].cpu().numpy()
    return {"ids": ids, "lp": lp, "rest": rest, "target": np.asarray(targets, np.int32), "target_lp": tlp}


def save_artifact(path: str, rows: dict, meta: dict) -> str:
    """Write the arrays (npz) and the meta (json sidecar); return the npz's sha256 (what A4 pins)."""
    for key in ARTIFACT_KEYS:
        if key not in rows:
            raise ValueError(f"artifact lacks {key}")
    with open(path + ".tmp", "wb") as f:
        np.savez(f, **{key: rows[key] for key in ARTIFACT_KEYS})
    os.replace(path + ".tmp", path)
    sha = file_sha(path)
    json.dump(dict(meta, sha256=sha, positions=int(rows["ids"].shape[0]), k=int(rows["ids"].shape[1])),
              open(path + ".json", "w"), indent=1, sort_keys=True)
    return sha


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_artifact(path: str, want_sha: str | None = None) -> dict:
    """Read a reference artifact; refuse (SystemExit) when `want_sha` is given and the file's sha differs."""
    if want_sha is not None:
        got = file_sha(path)
        if got != want_sha:
            raise SystemExit(f"{path}: sha {got[:16]} != the registered {want_sha[:16]} -- refusing the reference")
    z = np.load(path)
    return {key: z[key] for key in ARTIFACT_KEYS}


# ----------------------------------------------------------------------------------------------------------- KL65

def kl65(ref: dict, eng_lp, eps: float = REST_EPS) -> dict:
    """Per-position KL on the 65-bucket partition. `ref`: the artifact arrays; `eng_lp` [P, 64]: the engine's log-probs
    on ref["ids"], same order. Returns per-position arrays: kl, the engine rest, and which positions were clamped."""
    eng_lp = np.asarray(eng_lp, np.float64)
    if eng_lp.shape != ref["lp"].shape:
        raise ValueError(f"engine log-probs {eng_lp.shape} vs reference {ref['lp'].shape}: not aligned")
    if not np.all(np.isfinite(eng_lp)):
        raise ValueError("engine log-probs carry non-finite values on named tokens")
    p = np.exp(ref["lp"])
    named = np.sum(p * (ref["lp"] - eng_lp), axis=1)
    q_rest_raw = 1.0 - np.sum(np.exp(eng_lp), axis=1)
    clamped = q_rest_raw < eps
    q_rest = np.where(clamped, eps, q_rest_raw)
    p_rest = np.maximum(ref["rest"], 0.0)
    rest_term = np.where(p_rest > eps, p_rest * (np.log(np.maximum(p_rest, eps)) - np.log(q_rest)), 0.0)
    kl = named + rest_term
    if np.any(kl < -1e-9):
        i = int(np.argmin(kl))
        raise FloatingPointError(f"negative KL65 {kl[i]:.3e} at position {i}: the engine's named log-probs are not a "
                                 "sub-distribution aligned with the reference (alignment or units are wrong)")
    return {"kl": np.maximum(kl, 0.0), "q_rest": q_rest_raw, "clamped": clamped}


def window_read(ref: dict, eng_lp, eng_target_lp=None, eps: float = REST_EPS) -> dict:
    """One window's reading under the registered rules: mean KL65 (+ p50 / p95 / max), clamp share, coverage, the
    engine's NLL from its target log-probs when given, and the verdict (VALID or UNREAD with the reason)."""
    r = kl65(ref, eng_lp, eps)
    cov = 1.0 - float(np.mean(np.maximum(ref["rest"], 0.0)))
    clamp_share = float(np.mean(r["clamped"]))
    why = []
    if cov < COVERAGE_MIN:
        why.append(f"reference top-64 mass {cov:.4f} < {COVERAGE_MIN}")
    if clamp_share > CLAMP_SHARE_MAX:
        why.append(f"rest clamped on {clamp_share:.2%} of positions > {CLAMP_SHARE_MAX:.0%}")
    out = {"verdict": "UNREAD" if why else "VALID", "why": "; ".join(why) or None, "positions": int(r["kl"].size),
           "kl65_mean": float(np.mean(r["kl"])), "kl65_p50": float(np.median(r["kl"])),
           "kl65_p95": float(np.quantile(r["kl"], 0.95)), "kl65_max": float(np.max(r["kl"])),
           "coverage_ref_top64": cov, "rest_clamped": int(np.sum(r["clamped"])), "rest_clamp_share": clamp_share}
    if eng_target_lp is not None:
        t = np.asarray(eng_target_lp, np.float64)
        out["engine_nll"] = float(-np.mean(t))
        out["reference_nll"] = float(-np.mean(ref["target_lp"]))
    return out


# ----------------------------------------------------------------------------------------- the top-K fallback rule

def topk_fallback(ref: dict, eng_ids, eng_lp_topk, vocab: int, mode: str = "spread") -> tuple:
    """An engine that returns only ITS OWN top-K [P, K] (K >= 256): map onto the reference's 64 named tokens. Covered
    tokens take the engine's log-prob; an uncovered one gets the engine residual spread evenly over the vocabulary it
    did not return (`spread`) or the whole residual (`bound`). Returns (eng_lp [P, 64], coverage per window: the share
    of the reference top-64 mass whose tokens the engine returned)."""
    eng_ids = np.asarray(eng_ids)
    eng_lp_topk = np.asarray(eng_lp_topk, np.float64)
    P, K = eng_ids.shape
    if K < TOPK_FALLBACK_MIN:
        raise ValueError(f"top-K fallback needs K >= {TOPK_FALLBACK_MIN}, got {K}")
    resid = np.maximum(1.0 - np.sum(np.exp(eng_lp_topk), axis=1), REST_EPS)
    fill = np.log(resid / max(vocab - K, 1)) if mode == "spread" else np.log(resid)
    out = np.empty(ref["ids"].shape, np.float64)
    covered_mass, total_mass = 0.0, 0.0
    p = np.exp(ref["lp"])
    for i in range(P):
        look = dict(zip(eng_ids[i].tolist(), eng_lp_topk[i].tolist()))
        for j, t in enumerate(ref["ids"][i].tolist()):
            if t in look:
                out[i, j] = look[t]
                covered_mass += p[i, j]
            else:
                out[i, j] = fill[i]
            total_mass += p[i, j]
    return out, (covered_mass / total_mass if total_mass > 0 else 0.0)


# ----------------------------------------------------------------------------------- calibration on R (full vocab)

def calibrate(ref_logits, test_logits, k: int = K_NAMED, chunk: int = 128) -> dict:
    """Both distributions in FULL ([P, V] logits): the full-vocabulary KL (kl_fidelity's definition: fp64, masked at
    p == 0) and KL65 built from the reference's top-k with the test side's log-probs at those ids. The ratio of the
    means is what A4 registers (>= 0.90)."""
    import torch
    P = int(ref_logits.shape[0])
    full, part = [], []
    for a in range(0, P, chunk):
        b = min(P, a + chunk)
        lp_r = torch.log_softmax(ref_logits[a:b].to(torch.float64), -1)
        lp_t = torch.log_softmax(test_logits[a:b].to(torch.float64).to(lp_r.device), -1)
        pr = lp_r.exp()
        term = torch.where(pr > 0, pr * (lp_r - lp_t), torch.zeros((), dtype=pr.dtype, device=pr.device))
        full.append(term.sum(-1).cpu())
        top = torch.topk(lp_r, k, dim=-1).indices
        mask = torch.ones_like(lp_r, dtype=torch.bool)
        mask.scatter_(1, top, False)
        rr = {"lp": lp_r.gather(1, top).cpu().numpy(),
              "rest": torch.logsumexp(lp_r.masked_fill(~mask, float("-inf")), -1).exp().cpu().numpy()}
        part.append(torch.as_tensor(kl65(rr, lp_t.gather(1, top).cpu().numpy())["kl"]))
    full_t, part_t = torch.cat(full), torch.cat(part)
    if bool((full_t < -1e-9).any()):
        raise FloatingPointError("negative full-vocabulary KL: the pair is misaligned")
    fm, pm = float(full_t.clamp_min(0).mean()), float(part_t.mean())
    return {"positions": P, "kl_full_mean": fm, "kl65_mean": pm, "ratio": (pm / fm if fm > 0 else None),
            "kl65_le_full_everywhere": bool((part_t <= full_t.clamp_min(0) + 1e-9).all())}


# ------------------------------------------------------------------------- A5: the full-vocabulary estimator

GRADABLE_F_MAX = 1e-2          # a window is graded only if its own floor F (R's decode-vs-prefill full KL) is under this
STORAGE_F_FRACTION = 0.1       # |KL from the fp16 rows - KL from fp64| <= this x F on every gradable graded window


def full_rows_fp16(logits, chunk: int = 128) -> np.ndarray:
    """[P, V] logits -> fp16 log-softmax rows (computed in fp64). Every stored entry is finite, or exactly -inf where the
    logit itself was -inf (a masked / padded vocabulary entry). NaN, or a finite logit whose log-prob underflows fp16 to
    -inf, is refused: either would silently change the reference distribution."""
    import torch
    P, V = int(logits.shape[0]), int(logits.shape[1])
    out = np.empty((P, V), np.float16)
    for a in range(0, P, chunk):
        b = min(P, a + chunk)
        x = logits[a:b]
        masked = torch.isinf(x) & (x < 0)
        f16 = torch.log_softmax(x.to(torch.float64), -1).to(torch.float16)
        bad = ~torch.isfinite(f16) & ~masked
        if bool(bad.any()) or bool(torch.isnan(f16).any()) or not bool(torch.isinf(f16[masked]).all() if masked.any() else True):
            raise FloatingPointError(f"rows {a}..{b}: an fp16 log-prob is NaN, or -inf where the logit was finite -- refusing to store")
        out[a:b] = f16.cpu().numpy()
    return out


def kl_full_rows(ref_rows, eng, chunk: int = 64) -> np.ndarray:
    """Per-position KL(p_ref || p_eng) in fp64 over the FULL vocabulary. `ref_rows` [P, V]: log-probs (fp16 rows from R,
    renormalised on load; -inf entries carry p = 0 and stay out of the sum). `eng` [P, V]: the engine's logits or log-probs
    (renormalised too). An engine -inf where the reference has mass is an infinite KL: raised, so the row is VOID."""
    import torch
    P = int(ref_rows.shape[0])
    out = np.empty(P, np.float64)
    dev = eng.device if torch.is_tensor(eng) else torch.device("cpu")      # computed where the engine's rows live
    for a in range(0, P, chunk):
        b = min(P, a + chunk)
        rr = ref_rows[a:b]
        r = (rr if torch.is_tensor(rr) else torch.as_tensor(np.asarray(rr))).to(device=dev, dtype=torch.float64)
        e = torch.as_tensor(eng[a:b]).to(device=dev, dtype=torch.float64)
        r = r - torch.logsumexp(r, -1, keepdim=True)
        e = e - torch.logsumexp(e, -1, keepdim=True)
        pr = r.exp()
        term = torch.where(pr > 0, pr * (r - e), torch.zeros((), dtype=r.dtype, device=r.device))
        if not bool(torch.isfinite(term).all()):
            raise FloatingPointError(f"rows {a}..{b}: non-finite KL term (the engine puts -inf where the reference has mass)")
        kl = term.sum(-1)
        if bool((kl < -1e-9).any()):
            raise FloatingPointError(f"rows {a}..{b}: negative full KL {float(kl.min()):.3e} -- misaligned")
        out[a:b] = kl.clamp_min(0).cpu().numpy()
    return out


def storage_error(ref_logits, test_logits, chunk: int = 128) -> dict:
    """What fp16 storage of the reference costs: mean KL(ref || test) from fp64 log-probs vs from the stored fp16 rows
    (chunked, on the tensors' device)."""
    import torch
    k64, k16 = [], []
    for a in range(0, int(ref_logits.shape[0]), chunk):
        b = min(int(ref_logits.shape[0]), a + chunk)
        lp64 = torch.log_softmax(ref_logits[a:b].to(torch.float64), -1)
        lp16 = torch.as_tensor(full_rows_fp16(ref_logits[a:b])).to(lp64.device)
        k64.append(kl_full_rows(lp64, test_logits[a:b].to(lp64.device)))
        k16.append(kl_full_rows(lp16, test_logits[a:b].to(lp64.device)))
    m64, m16 = float(np.mean(np.concatenate(k64))), float(np.mean(np.concatenate(k16)))
    return {"kl_fp64_mean": m64, "kl_fp16_mean": m16, "abs_err": abs(m16 - m64)}


def calib_verdict(ratio) -> str:
    return "UNREAD" if ratio is None else ("OK" if ratio >= CALIB_RATIO_MIN else "UNREAD")


# ------------------------------------------------------------------------------------------------------- self-test

def self_test() -> int:
    import torch
    g = torch.Generator().manual_seed(0)
    cases = []
    V, P = 600, 40
    ref_logits = torch.randn(P, V, generator=g) * 3.0
    test_logits = ref_logits + torch.randn(P, V, generator=g) * 0.2
    targets = torch.randint(0, V, (P,), generator=g).numpy()
    rows = reference_rows(ref_logits, targets)
    p_full = torch.softmax(ref_logits.double(), -1)
    cases.append(("rest = 1 - top64 mass", np.allclose(rows["rest"], 1 - np.exp(rows["lp"]).sum(1), atol=1e-12)))
    cases.append(("target lp", np.allclose(rows["target_lp"], np.log(p_full[torch.arange(P), torch.as_tensor(targets)].numpy()))))
    eng_lp = torch.log_softmax(test_logits.double(), -1).gather(1, torch.as_tensor(rows["ids"]).long()).numpy()
    c = calibrate(ref_logits, test_logits)
    r = kl65(rows, eng_lp)
    cases.append(("KL65 matches calibrate's KL65", abs(float(np.mean(r["kl"])) - c["kl65_mean"]) < 1e-12))
    cases.append(("KL65 <= full KL (data processing)", c["kl65_le_full_everywhere"] and c["kl65_mean"] <= c["kl_full_mean"] + 1e-12))
    # with a vocabulary of exactly 65 tokens the partition IS the vocabulary: KL65 == full KL
    small_r, small_t = torch.randn(8, 65, generator=g), torch.randn(8, 65, generator=g)
    cs = calibrate(small_r, small_t)
    cases.append(("V == 65: KL65 == full", abs(cs["kl65_mean"] - cs["kl_full_mean"]) < 1e-10))
    # identity reads zero
    cases.append(("identity reads 0", float(np.max(kl65(rows, rows["lp"])["kl"])) < 1e-12))
    # an engine whose named mass exceeds 1 (wrong units) is clamped, flagged, and a window with > 1 % clamps is UNREAD
    bad = rows["lp"] + 0.05
    wr = window_read(rows, bad)
    cases.append(("over-unity named mass -> clamped -> UNREAD", wr["rest_clamp_share"] > 0.01 and wr["verdict"] == "UNREAD"))
    # coverage floor: a flat reference (top-64 mass well under 0.99) is UNREAD
    flat = reference_rows(torch.zeros(4, V), np.zeros(4, np.int32))
    cases.append(("flat reference -> coverage UNREAD", window_read(flat, flat["lp"])["verdict"] == "UNREAD"))
    # the top-K fallback: an engine returning its own top-300 covers the reference's top-64 here; spread <= bound
    lt = torch.log_softmax(test_logits.double(), -1)
    tk = torch.topk(lt, 300, dim=-1)
    e_spread, cov = topk_fallback(rows, tk.indices.numpy(), tk.values.numpy(), V, "spread")
    e_bound, _ = topk_fallback(rows, tk.indices.numpy(), tk.values.numpy(), V, "bound")
    ks, kb = float(np.mean(kl65(rows, e_spread)["kl"])), float(np.mean(kl65(rows, e_bound)["kl"]))
    cases.append(("top-K fallback: coverage measured, bound <= spread", 0.0 < cov <= 1.0 and kb <= ks + 1e-12))
    # a misaligned engine (named log-probs read at the wrong positions) shows up as a large KL, not as agreement
    rolled = np.roll(eng_lp, 1, axis=0)
    cases.append(("misalignment is visible", float(np.mean(kl65(rows, rolled)["kl"])) > 10 * float(np.mean(r["kl"]))))
    # artifact round trip + sha refusal
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        pth = os.path.join(d, "ref_x.npz")
        sha = save_artifact(pth, rows, {"source": "x"})
        back = load_artifact(pth, sha)
        ok = all(np.array_equal(back[key], rows[key]) for key in ARTIFACT_KEYS)
        try:
            load_artifact(pth, "0" * 64)
            refused = False
        except SystemExit:
            refused = True
        cases.append(("artifact round trip + sha refusal", ok and refused))
    cases.append(("calibration verdicts", calib_verdict(0.95) == "OK" and calib_verdict(0.85) == "UNREAD" and calib_verdict(None) == "UNREAD"))
    # A5: full-vocabulary KL equals the direct fp64 computation; fp16 storage costs little; -inf handled exactly
    full = kl_full_rows(torch.log_softmax(ref_logits.double(), -1).numpy(), test_logits)
    p_ = torch.softmax(ref_logits.double(), -1)
    direct = (p_ * (torch.log_softmax(ref_logits.double(), -1) - torch.log_softmax(test_logits.double(), -1))).sum(-1).numpy()
    cases.append(("A5 full KL == direct fp64", np.allclose(full, direct, atol=1e-12) and abs(float(np.mean(full)) - c["kl_full_mean"]) < 1e-12))
    se = storage_error(ref_logits, test_logits)
    cases.append(("A5 fp16 storage error is small", se["abs_err"] < 1e-3 * max(se["kl_fp64_mean"], 1e-9) + 1e-6))
    masked = ref_logits.clone()
    masked[:, -5:] = float("-inf")                      # padded vocab entries
    rows16 = full_rows_fp16(masked)
    ok_inf = bool(np.isneginf(rows16[:, -5:]).all()) and bool(np.isfinite(rows16[:, :-5]).all())
    kmask = kl_full_rows(rows16, test_logits)            # the masked entries carry p = 0: finite KL
    cases.append(("A5 -inf stored exactly and kept out of the sum", ok_inf and bool(np.all(np.isfinite(kmask)))))
    eng_bad = test_logits.clone()
    eng_bad[:, 3] = float("-inf")                        # the engine zeroes a token the reference gives mass to
    try:
        kl_full_rows(torch.log_softmax(ref_logits.double(), -1).numpy(), eng_bad)
        raised = False
    except FloatingPointError:
        raised = True
    cases.append(("A5 engine -inf on reference mass is refused", raised))
    try:
        full_rows_fp16(torch.full((2, 8), float("nan")))
        nan_ok = False
    except FloatingPointError:
        nan_ok = True
    cases.append(("A5 NaN rows refused", nan_ok))
    bad_cases = [n for n, ok in cases if not ok]
    print(f"sc1g_kl self-test {'OK' if not bad_cases else 'FAILED ' + str(bad_cases)} ({len(cases)} cases)")
    return 0 if not bad_cases else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    print(__doc__)
    sys.exit(2)
