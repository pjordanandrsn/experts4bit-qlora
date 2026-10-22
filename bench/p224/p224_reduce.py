#!/usr/bin/env python3
"""p224_reduce.py -- lane P224's registered rule (bench/p224/PREREG-p224.md): one trained attention adapter served by
the paged server (``E4B_PAGED_ADAPTER``, merged at load) against the same adapter under Loggetta's own reference path,
on one RTX 5090 with Qwen3-30B-A3B. Pure Python, no torch.

Arms (``p224_box.py`` writes one record each):
  HF0      the base under the reference path (transformers, teacher-forced), no adapter
  REF      the canary adapter under ``loggetta.load_adapter`` (trained attention-only, attn_4bit=false, E4B_ABSMAX_DQ=0)
  P0       the paged server, no adapter
  PA       the paged server with REF's adapter (``E4B_PAGED_ADAPTER_ACCEPT_UNVERIFIED_BASE=experts``)

Amendment 3 dropped canary B (REF-DQ, PA-DQ): an attention-only canary never runs the fused expert path, so its frozen
expert absmax is fp32 by construction and F1 does not arise for it.

Losses are mean next-token NLL in nats per held-out chat-formatted sequence, teacher-forced, over every token and over
the assistant positions (the canary's own tokens); the paged arms' come from the paged runner's PREFILL logits, so the
loss checks gate PREFILL numerics. Amendment 5: every statistic is taken over DISTINCT sequences (``per_seq_ids_sha256``),
G1 and its SE are graded on the ASSISTANT positions, the all-token G1 is reported beside it, and G3 is graded at W1 on
the per-token wall time (``itl_wall_ms``, ``ttft_any_ms``, ``tok_s``), every other cost cell reported.
The margin's yardstick is the paged base's own gap against the reference path, |loss(P0) - loss(HF0)|: it absorbs the
fp8 KV cache, the int8 lm_head (on by default for qwen3_moe), the arena's NF4 experts and the paged attention, so the
adapter must add no NEW gap beyond it.

The rule, first rung that applies:
  VOID  a record missing or not ok; the arms' scored token ids differ (``ids_sha256``); the pins (e4b, the model revision, the canary data's sha256) differ from the
        registered; not an RTX 5090 (cc 12.0), unless a shakedown; the held-out set is not the registered count; a
        training record that did not engage as registered (setup: train_experts false, train_attention true, attn_4bit
        false; Amendment 3: the setup's expert_kernel reference and the engaged expert absmax form fp32); Amendment 5: the
        held-out sequences not all distinct, or a held-out row equal to a training row; the canary not learned
        (REF's improvement over HF0 under MIN_REF_GAIN nats); THE PAGED BASE NOT FAITHFUL: |loss(P0) - loss(HF0)| >
        FAITHFUL ("paged base not faithful enough to judge an adapter on"); the SE of the per-sequence ASSISTANT-position
        loss difference PA - REF at or above SE_FRACTION of the floor (the held-out set too small to judge); the ABBA
        cost record not the registered structure.
  FAIL  (a finding against the served adapter, each failed gate named):
        G1 effect, on the ASSISTANT positions (Amendment 5): PA's improvement over P0 below RATIO_BAR of REF's over
           HF0, or |loss(PA) - loss(REF)| above the margin min(CAP, max(FLOOR, 2 x |loss(P0) - loss(HF0)|)), whose
           yardstick is the ALL-TOKEN base gap (the better-measured one);
        G2 HTTP: a trigger prompt that does not yield the canary on PA through /v1/completions and /v1/chat/completions,
           streamed and not, token-identical across the four; a control prompt that yields it on PA; any prompt that
           yields it on P0;
        G3 cost, GRADED at W1 on ttft_any_ms, itl_wall_ms and tok_s (Amendment 5): band = max(the max over the ABBA
           blocks of |A1 - A2| (the two P0 slots), BAND_FLOOR x the mean of the four A slots), both reported with the one
           that governed; PA (B) fails when |B - mean(A1, A2)| > band in any block; peak VRAM |PA - P0| / P0 > VRAM_TOL.
           Every other cost cell (W16, and the text-cadence ttft_ms / itl_ms) is computed the same way and REPORTED;
        G4 identity: PA's /health or /v1/models lacks the adapter id (the trained adapter.safetensors sha256), the
           manifest sha256 or base_match; the served name is not <model>+<id[:12]>; a request naming the base is not a
           404; the merge did not cover every attention projection;
        G5 the gates on the box: the registered CUDA offload fingerprint test did not pass, or a quant_fingerprint scope
           recorded no positive seconds.
  PASS  otherwise.
  SHAKEDOWN  a shakedown record: every VOID check that does not depend on the card or on the reading's thresholds runs;
        the reading's thresholds (MIN_REF_GAIN, FAITHFUL, SE_FRACTION) and the gates are computed and REPORTED
        (``thresholds_would_void``, ``gates_would_fail``), never graded.

Reported, never ruling (Amendment 5): the all-token G1 and its SE; on both position sets the signed difference in
differences DiD = (PA - REF) - (P0 - HF0) with its paired SE, FLAGGED when |DiD| > 2 SE and |DiD| > the margin, and
over-application, FLAGGED when the effect ratio exceeds 1 / RATIO_BAR; the SE of P0 - HF0; on a set with duplicates,
the SE both ways (every row, and over distinct rows).
Reported, never ruling: HTTP canary; for P0 and PA, the max |decode-step logit - prefill logit| over 8 held-out sequences, flagged when PA's
exceeds both twice P0's and P0's + DECODE_FLAG_ABS (the decode-path question for a later read); the merge seconds and
max merge error; each fingerprint scope's seconds and module count.

    python p224_reduce.py --dir RUN_DIR --out verdict_p224.json
    python p224_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

ARMS = ("HF0", "REF", "P0", "PA")
N_HELDOUT = 256                # held-out sequences (p224_canary.py, Amendment 5: 32 topics x 8 templates, all distinct)
N_DECODE_CHECK = 8             # held-out sequences whose decode-step logits are compared with the prefill's
RATIO_BAR = 0.95               # PA's improvement over P0 must reach this share of REF's over HF0
FLOOR, CAP = 0.01, 0.05        # the margin: min(CAP, max(FLOOR, 2 x the paged base's own gap)), nats
FAITHFUL = 0.025               # the paged base's own gap above this: VOID, never a loose pass
SE_FRACTION = 0.2              # the SE of the PA - REF difference must be under this share of FLOOR
MIN_REF_GAIN = 0.05            # nats: REF must improve on HF0 at least this much, or the canary did not learn
VRAM_TOL = 0.01
BAND_FLOOR = 0.01              # G3's band is never under 1 % of the A slots' mean (the maintainer's ruling)
DECODE_FLAG_ABS = 0.05
COST_METRICS = ("ttft_ms", "itl_ms", "tok_s", "ttft_any_ms", "itl_wall_ms")   # every cell the box records
COST_WS = ("W1", "W16")
#: Amendment 5: the GRADED cells. At W1 top-8 of 64 fixes the expert work per token, so the step is comparable across
#: arms; at W16 the arms decode different text, which routes to a different spread of experts (reported, with that reason)
GRADED_COST = {("ttft_any_ms", "W1"), ("itl_wall_ms", "W1"), ("tok_s", "W1")}
DID_SE_FLAG = 2.0              # Amendment 5: the DiD is flagged at |DiD| > 2 SE and |DiD| > the margin
ABBA_BLOCKS = 2                # two ABBA blocks: A1 B A2 each (P0, PA, P0)
EXPECTED_ATTN_PROJECTIONS = {"Qwen/Qwen3-30B-A3B": 48 * 4}
TRAIN_FORM = {"REF": "fp32"}               # Amendment 3: canary A, the only canary
TRAIN_KERNEL = "reference"                  # Amendment 3: an attention-only train never takes the fused expert path


def _ok(rec):
    return isinstance(rec, dict) and rec.get("ok") is True


def margin_of(p0, hf0):
    return min(CAP, max(FLOOR, 2 * abs(p0 - hf0)))


def distinct_index(keys) -> list:
    """The index of each distinct sequence's first occurrence (Amendment 5: every statistic counts a sequence once)."""
    seen, idx = set(), []
    for i, k in enumerate(keys):
        if k not in seen:
            seen.add(k)
            idx.append(i)
    return idx


def _se(xs):
    return statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else math.inf


def loss_stats(per: dict, asst: dict, keys) -> dict:
    """G1, its SE and the reported statistics, over DISTINCT sequences. ``per`` / ``asst``: arm -> per-sequence loss
    (every token / the assistant positions); ``keys``: each sequence's identity. Graded on the assistant positions; the
    margin's yardstick is the all-token base gap."""
    idx = distinct_index(keys)
    sets = {"all": per, "assistant": asst}
    st = {}
    for name, v in sets.items():
        pick = {a: [v[a][i] for i in idx] for a in ARMS}
        L = {a: statistics.fmean(x) for a, x in pick.items()}
        pr = [x - y for x, y in zip(pick["PA"], pick["REF"])]
        base = [x - y for x, y in zip(pick["P0"], pick["HF0"])]
        did = [x - y for x, y in zip(pr, base)]
        ref_gain = L["HF0"] - L["REF"]
        st[name] = {"loss": L, "base_gap": L["P0"] - L["HF0"], "se_base": _se(base),
                    "pa_minus_ref": L["PA"] - L["REF"], "se_pa_ref": _se(pr),
                    "sd_pa_ref": statistics.stdev(pr) if len(pr) > 1 else None,
                    "ratio": (L["P0"] - L["PA"]) / ref_gain if ref_gain > 0 else None,
                    "ref_gain": ref_gain, "did": statistics.fmean(did), "se_did": _se(did)}
        if len(idx) < len(keys):                                  # duplicates: the SE every row would have given
            st[name]["se_pa_ref_every_row"] = _se([x - y for x, y in zip(v["PA"], v["REF"])])
    margin = margin_of(st["all"]["loss"]["P0"], st["all"]["loss"]["HF0"])
    out = {"rows": len(keys), "distinct": len(idx), "margin": margin, "sets": st, "fails": [], "reported": {}}
    a = st["assistant"]
    if a["ratio"] is None or a["ratio"] < RATIO_BAR:
        out["fails"].append(f"G1 assistant effect ratio {a['ratio'] if a['ratio'] is None else round(a['ratio'], 4)} "
                            f"< {RATIO_BAR}")
    if abs(a["pa_minus_ref"]) > margin:
        out["fails"].append(f"G1 assistant |loss(PA) - loss(REF)| = {abs(a['pa_minus_ref']):.4f} > margin {margin:.4f}")
    al = st["all"]
    out["reported"]["all_token_g1"] = {
        "ratio": al["ratio"], "abs_pa_minus_ref": abs(al["pa_minus_ref"]), "margin": margin,
        "would_fail": [w for w in (
            f"ratio {al['ratio']} < {RATIO_BAR}" if al["ratio"] is None or al["ratio"] < RATIO_BAR else None,
            f"|PA - REF| {abs(al['pa_minus_ref']):.4f} > margin {margin:.4f}" if abs(al["pa_minus_ref"]) > margin else None)
            if w]}
    for name, x in st.items():
        out["reported"][f"did_{name}"] = {
            "did": x["did"], "se": x["se_did"],
            "flag": abs(x["did"]) > DID_SE_FLAG * x["se_did"] and abs(x["did"]) > margin}
        out["reported"][f"over_application_{name}"] = {
            "ratio": x["ratio"], "flag": x["ratio"] is not None and x["ratio"] > 1 / RATIO_BAR}
    return out


def cost_grade(blocks: list, metrics=COST_METRICS) -> dict:
    """G3: per (metric, W) the band and each block's |B - mean(A1, A2)|; the GRADED cells fail, the rest are reported."""
    band, raw, floor, governed, fails, reported = {}, {}, {}, {}, [], []
    for m in metrics:
        for w in COST_WS:
            cells = [b[s][m][w] for b in blocks for s in ("A1", "B", "A2")
                     if isinstance((b[s].get(m) or {}).get(w), (int, float))]
            if len(cells) != 3 * len(blocks):
                if (m, w) in GRADED_COST:
                    fails.append(f"G3 {m} {w}: a graded cost cell is unread")
                continue
            r = max(abs(b["A1"][m][w] - b["A2"][m][w]) for b in blocks)
            f = BAND_FLOOR * statistics.fmean(b[s][m][w] for b in blocks for s in ("A1", "A2"))
            raw.setdefault(m, {})[w], floor.setdefault(m, {})[w] = r, f
            band.setdefault(m, {})[w] = max(r, f)
            governed.setdefault(m, {})[w] = "floor" if f > r else "raw"
            for i, b in enumerate(blocks):
                dev = abs(b["B"][m][w] - (b["A1"][m][w] + b["A2"][m][w]) / 2)
                if dev > band[m][w]:
                    msg = f"G3 block {i} {m} {w}: |B - mean(A)| = {dev:.4g} > band {band[m][w]:.4g}"
                    (fails if (m, w) in GRADED_COST else reported).append(msg)
    out = {"band": band, "raw_max_abs_a1_a2": raw, "floor_1pct": floor, "governed": governed,
           "fails": fails, "reported_outside_band": reported}
    graded = [governed[m][w] for (m, w) in GRADED_COST if m in governed and w in governed[m]]
    if graded and all(g == "floor" for g in graded):
        out["summary"] = "G3 read at the 1% floor: noise below resolution"
    return out


def _void(why, out):
    return dict(out, verdict="VOID", why=why)


def verdict(recs: dict, pins: dict) -> dict:
    """``recs``: arm -> record (plus "train_REF", "cost", "gates", "meta"); ``pins``: the registered pins."""
    shake = bool(recs.get("meta", {}).get("shakedown"))
    out = {"lane": "P224", "shakedown": shake}
    why = []
    need = list(ARMS) + ["train_REF", "cost", "gates", "meta"]
    missing = [k for k in need if not _ok(recs.get(k))]
    if missing:
        return _void([f"record missing or not ok: {missing}"], out)
    meta = recs["meta"]
    for key in ("e4b_sha", "model_revision", "canary_sha256"):
        if meta.get(key) != pins.get(key):
            why.append(f"pin {key} is {meta.get(key)!r}, registered {pins.get(key)!r}")
    if not shake and ("5090" not in str(meta.get("device", "")) or meta.get("cc") != [12, 0]):
        why.append(f"device {meta.get('device')} cc {meta.get('cc')}: the reading registers an RTX 5090 (sm_120)")
    for arm in ARMS:
        r = recs[arm]
        for key in ("per_seq_loss", "per_seq_loss_assistant", "per_seq_ids_sha256"):
            if len(r.get(key) or []) != N_HELDOUT:
                why.append(f"{arm}: {len(r.get(key) or [])} held-out {key}, registered {N_HELDOUT}")
    shas = {arm: recs[arm].get("ids_sha256") for arm in ARMS}
    if None in shas.values() or len(set(shas.values())) != 1:
        why.append(f"the arms did not score the same token ids: {shas}")
    keys = recs["REF"].get("per_seq_ids_sha256") or []
    if any((recs[arm].get("per_seq_ids_sha256") or []) != keys for arm in ARMS):
        why.append("the arms' per-sequence ids differ")
    elif keys and len(set(keys)) != len(keys):
        why.append(f"the held-out sequences are not all distinct: {len(set(keys))} distinct of {len(keys)} (Amendment 5)")
    if meta.get("heldout_train_overlap") != 0:
        why.append(f"held-out rows equal to a training row: {meta.get('heldout_train_overlap')!r} (Amendment 5)")
    for name, form in TRAIN_FORM.items():
        t = recs[f"train_{name}"]
        s = t.get("setup") or {}
        if (s.get("train_experts"), s.get("train_attention"), s.get("attn_4bit")) != (False, True, False):
            why.append(f"train_{name}: setup {s} is not attention-only with attn_4bit=false")
        got = ((t.get("engaged") or {}).get("expert_absmax") or {}).get("form")
        if got != form:
            why.append(f"train_{name}: engaged expert absmax form {got!r}, registered {form!r}")
        if s.get("expert_kernel") != TRAIN_KERNEL:
            why.append(f"train_{name}: expert_kernel {s.get('expert_kernel')!r}, registered {TRAIN_KERNEL!r} (Amendment 3)")
    if why:
        return _void(why, out)
    ls = loss_stats({a: recs[a]["per_seq_loss"] for a in ARMS}, {a: recs[a]["per_seq_loss_assistant"] for a in ARMS},
                    keys)
    L, La = ls["sets"]["all"]["loss"], ls["sets"]["assistant"]["loss"]
    out["loss"], out["loss_assistant"] = L, La
    gap = abs(L["P0"] - L["HF0"])
    margin = ls["margin"]
    se = ls["sets"]["assistant"]["se_pa_ref"]
    out.update(base_gap=gap, margin=margin, se_pa_ref=se, sd_pa_ref=ls["sets"]["assistant"]["sd_pa_ref"],
               heldout_tokens={arm: recs[arm].get("tokens") for arm in ARMS}, distinct=ls["distinct"])
    ref_gain = L["HF0"] - L["REF"]
    # the reading's thresholds: a VOID on a reading; on a shakedown (another model, another card) they are reported
    thr = []
    if ref_gain < MIN_REF_GAIN:
        thr.append(f"the canary did not learn: REF improves on HF0 by {ref_gain:.4f} nats (< {MIN_REF_GAIN})")
    if gap > FAITHFUL:
        thr.append(f"paged base not faithful enough to judge an adapter on: |loss(P0) - loss(HF0)| = {gap:.4f} > "
                   f"{FAITHFUL}")
    if se >= SE_FRACTION * FLOOR:
        thr.append(f"the SE of loss(PA) - loss(REF) on the assistant positions is {se:.5f}, not under {SE_FRACTION} x "
                   f"{FLOOR}: the held-out set is too small to judge")
    if thr and not shake:
        return _void(thr, out)
    out["thresholds_would_void"] = thr
    cost = recs["cost"]
    blocks = cost.get("blocks") or []
    if len(blocks) != ABBA_BLOCKS or any(set(b) < {"A1", "B", "A2"} for b in blocks):
        return _void([f"the cost record has {len(blocks)} blocks, registered {ABBA_BLOCKS} ABBA blocks of A1 B A2"], out)

    fails, reported = [], {}
    # G1 effect, graded on the assistant positions (Amendment 5)
    sa = ls["sets"]["assistant"]
    out["effect"] = {"ref_gain": sa["ref_gain"], "pa_gain": La["P0"] - La["PA"], "ratio": sa["ratio"],
                     "pa_minus_ref": sa["pa_minus_ref"], "position_set": "assistant"}
    fails.extend(ls["fails"])
    reported.update(ls["reported"])
    reported["loss_sets"] = ls["sets"]
    # G2 HTTP
    http = recs["PA"].get("canary") or {}
    if not http.get("all_triggers_all_modes") or not http.get("modes_token_identical"):
        fails.append("G2 a trigger prompt did not yield the canary on PA in all four modes, token-identical")
    if http.get("control_hits", 1) != 0:
        fails.append(f"G2 {http.get('control_hits')} control prompt(s) yielded the canary on PA")
    if (recs["P0"].get("canary") or {}).get("any_hits", 1) != 0:
        fails.append("G2 the base (P0) yielded the canary")
    # G3 cost
    cg = cost_grade(blocks)
    out["cost_band"] = {k: cg[k] for k in ("band", "raw_max_abs_a1_a2", "floor_1pct", "governed")}
    if "summary" in cg:
        out["cost_band"]["summary"] = cg["summary"]
    fails.extend(cg["fails"])
    reported["cost_outside_band_not_graded"] = cg["reported_outside_band"]
    v0, va = cost.get("peak_vram_mib", {}).get("P0"), cost.get("peak_vram_mib", {}).get("PA")
    if not (isinstance(v0, (int, float)) and isinstance(va, (int, float)) and v0 > 0):
        fails.append("G3 peak VRAM unread")
    elif abs(va - v0) / v0 > VRAM_TOL:
        fails.append(f"G3 peak VRAM PA {va} vs P0 {v0} MiB: |delta| > {VRAM_TOL:.0%}")
    # G4 identity
    ident = recs["PA"].get("identity") or {}
    want_id = recs["train_REF"].get("adapter_sha256")
    if ident.get("adapter_id") != want_id:
        fails.append(f"G4 /health adapter_id {ident.get('adapter_id')!r} is not the trained adapter {want_id!r}")
    if not ident.get("manifest_sha256") or not ident.get("base_match"):
        fails.append("G4 /health lacks the manifest sha256 or base_match")
    if want_id and ident.get("served_name") != f"{meta.get('model')}+{want_id[:12]}":
        fails.append(f"G4 served name {ident.get('served_name')!r}")
    if ident.get("base_name_status") != 404:
        fails.append(f"G4 a request naming the base returned {ident.get('base_name_status')}, not 404")
    want_n = EXPECTED_ATTN_PROJECTIONS.get(meta.get("model"), meta.get("attention_projections"))
    if (ident.get("merge") or {}).get("merged") != want_n:
        fails.append(f"G4 merged {(ident.get('merge') or {}).get('merged')} attention projections, expected {want_n}")
    # G5 gates on the box
    g = recs["gates"]
    if g.get("cuda_offload_test") != "passed":
        fails.append(f"G5 the registered CUDA offload fingerprint test: {g.get('cuda_offload_test')!r}")
    for scope in ("attention", "experts"):
        sec = ((g.get("fingerprint") or {}).get(scope) or {}).get("seconds")
        if not (isinstance(sec, (int, float)) and sec > 0):
            fails.append(f"G5 quant_fingerprint({scope}) recorded no positive seconds")
    # reported, never ruling
    dp = {arm: recs[arm].get("decode_vs_prefill_max_abs") for arm in ("P0", "PA")}
    reported["decode_vs_prefill"] = dict(dp, flag=(isinstance(dp["PA"], (int, float)) and isinstance(dp["P0"], (int, float))
                                                   and dp["PA"] > 2 * dp["P0"] and dp["PA"] > dp["P0"] + DECODE_FLAG_ABS))
    reported["merge"] = {k: (ident.get("merge") or {}).get(k) for k in ("seconds", "max_err")}
    reported["fingerprint"] = g.get("fingerprint")
    out["reported"] = reported
    if shake:
        return dict(out, verdict="SHAKEDOWN", why=["a shakedown record is never a reading"], gates_would_fail=fails)
    return dict(out, verdict="FAIL", why=fails) if fails else dict(out, verdict="PASS", why=[])


# ------------------------------------------------------------------------------------------------------------ records

def load(run_dir):
    recs = {}
    for name in list(ARMS) + ["train_REF", "cost", "gates", "meta"]:
        path = os.path.join(run_dir, f"{name}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                recs[name] = json.load(f)
    return recs


# ------------------------------------------------------------------------------------------------------------- self-test

PINS = {"e4b_sha": "e" * 40, "model_revision": "r" * 40, "canary_sha256": "c" * 64}
ADAPTER = "a" * 64


def _fake(**kw):
    """A passing record set; ``kw`` edits it (callables receive the record dict). Each arm carries the same per-sequence
    loss on every token and on the assistant positions (``both`` edits the two together) and distinct sequence ids."""
    import random

    rng = random.Random(224)
    base = [3.0 + rng.gauss(0, 0.3) for _ in range(N_HELDOUT)]
    ids = [f"{i:064x}" for i in range(N_HELDOUT)]
    def arm(shift, jitter=0.0):
        losses = [x + shift + rng.gauss(0, jitter) for x in base]
        return {"ok": True, "tokens": 5000, "ids_sha256": "s" * 64, "per_seq_ids_sha256": list(ids),
                "per_seq_loss": losses, "per_seq_loss_assistant": list(losses)}
    recs = {
        "HF0": arm(0.0), "REF": arm(-0.40),
        "P0": arm(0.005, 0.0002), "PA": arm(-0.395, 0.0002),
        "train_REF": {"ok": True, "adapter_sha256": ADAPTER, "setup": {"train_experts": False, "train_attention": True,
                                                                       "attn_4bit": False, "expert_kernel": "reference"},
                      "engaged": {"expert_absmax": {"form": "fp32"}}},
        "cost": {"ok": True, "peak_vram_mib": {"P0": 30000, "PA": 30010},
                 "blocks": [{s: {m: {w: 10.0 + (0.2 if s == "A2" else 0.0) + (0.05 if s == "B" else 0.0)
                                     for w in COST_WS} for m in COST_METRICS} for s in ("A1", "B", "A2")}
                            for _ in range(ABBA_BLOCKS)]},
        "gates": {"ok": True, "cuda_offload_test": "passed",
                  "fingerprint": {"attention": {"seconds": 1.2, "modules": 192}, "experts": {"seconds": 6.0, "modules": 96}}},
        "meta": {"ok": True, "shakedown": False, "device": "NVIDIA GeForce RTX 5090", "cc": [12, 0],
                 "model": "Qwen/Qwen3-30B-A3B", "heldout_train_overlap": 0, **PINS},
    }
    recs["PA"].update(canary={"all_triggers_all_modes": True, "modes_token_identical": True, "control_hits": 0},
                      identity={"adapter_id": ADAPTER, "manifest_sha256": "m" * 64,
                                "base_match": {"attention": "exact by construction", "experts": "unverified"},
                                "served_name": f"Qwen/Qwen3-30B-A3B+{ADAPTER[:12]}", "base_name_status": 404,
                                "merge": {"merged": 192, "seconds": 0.8, "max_err": 0.004}},
                      decode_vs_prefill_max_abs=0.03)
    recs["P0"].update(canary={"any_hits": 0}, decode_vs_prefill_max_abs=0.03)
    for k, f in kw.items():
        f(recs)
    return recs


def both(rec, f):
    """Apply ``f`` to an arm's every-token AND assistant-position losses (the graded set is the assistant positions)."""
    rec["per_seq_loss"] = f(rec["per_seq_loss"])
    rec["per_seq_loss_assistant"] = f(rec["per_seq_loss_assistant"])


def self_test() -> int:
    v = lambda recs: verdict(recs, PINS)["verdict"]          # noqa: E731
    o = verdict(_fake(), PINS)
    assert o["verdict"] == "PASS", o["why"]
    assert o["margin"] == margin_of(o["loss"]["P0"], o["loss"]["HF0"]) and FLOOR <= o["margin"] <= CAP
    assert o["effect"]["ratio"] > RATIO_BAR and o["effect"]["position_set"] == "assistant" and o["distinct"] == N_HELDOUT
    assert margin_of(3.0, 3.03) == min(CAP, 0.06) and margin_of(3.0, 3.001) == FLOOR and margin_of(3.0, 3.0) == FLOOR
    assert distinct_index(["a", "b", "a", "c", "b"]) == [0, 1, 3]
    # VOID rungs
    assert v(_fake(a=lambda r: r.pop("PA"))) == "VOID"
    assert v(_fake(a=lambda r: r["meta"].update(e4b_sha="0" * 40))) == "VOID"
    assert v(_fake(a=lambda r: r["meta"].update(device="NVIDIA L4", cc=[8, 9]))) == "VOID"
    assert v(_fake(a=lambda r: r["REF"]["per_seq_loss"].pop())) == "VOID"
    assert v(_fake(a=lambda r: r["REF"]["per_seq_loss_assistant"].pop())) == "VOID"
    assert v(_fake(a=lambda r: r["PA"].update(ids_sha256="t" * 64))) == "VOID"
    assert v(_fake(a=lambda r: r["train_REF"]["engaged"]["expert_absmax"].update(form="double-quantized"))) == "VOID"
    assert v(_fake(a=lambda r: r["train_REF"]["setup"].update(expert_kernel="grouped_nf4"))) == "VOID"   # Amendment 3
    assert v(_fake(a=lambda r: r["train_REF"]["setup"].update(attn_4bit=True))) == "VOID"
    # Amendment 5: duplicates, a training row in the held-out set, per-sequence ids that disagree across arms
    def dup(r):
        for a in ARMS:
            r[a]["per_seq_ids_sha256"][1] = r[a]["per_seq_ids_sha256"][0]
    o = verdict(_fake(a=dup), PINS)
    assert o["verdict"] == "VOID" and any("not all distinct" in w for w in o["why"]), o["why"]
    o = verdict(_fake(a=lambda r: r["meta"].update(heldout_train_overlap=1)), PINS)
    assert o["verdict"] == "VOID" and any("training row" in w for w in o["why"])
    assert v(_fake(a=lambda r: r["meta"].pop("heldout_train_overlap"))) == "VOID"          # unread is not zero
    assert v(_fake(a=lambda r: r["PA"]["per_seq_ids_sha256"].reverse())) == "VOID"
    o = verdict(_fake(a=lambda r: r.__setitem__("REF", dict(r["HF0"], per_seq_loss=[x - 0.01 for x in r["HF0"]["per_seq_loss"]]))), PINS)
    assert o["verdict"] == "VOID" and "did not learn" in o["why"][0]
    o = verdict(_fake(a=lambda r: r["P0"].update(per_seq_loss=[x + 0.03 for x in r["HF0"]["per_seq_loss"]])), PINS)
    assert o["verdict"] == "VOID" and "not faithful" in o["why"][0]
    import random
    rng = random.Random(1)
    # the SE is the ASSISTANT positions': noise there VOIDs; the same noise on every token only does not
    o = verdict(_fake(a=lambda r: r["PA"].update(per_seq_loss_assistant=[x + rng.gauss(0, 0.05) for x in r["PA"]["per_seq_loss_assistant"]])), PINS)
    assert o["verdict"] == "VOID" and "too small to judge" in o["why"][0] and "assistant" in o["why"][0]
    o = verdict(_fake(a=lambda r: r["PA"].update(per_seq_loss=[x + rng.gauss(0, 0.05) for x in r["PA"]["per_seq_loss"]])), PINS)
    assert o["verdict"] == "PASS", o["why"]
    assert v(_fake(a=lambda r: r["cost"]["blocks"].pop())) == "VOID"
    # FAIL gates: G1 on the assistant positions
    o = verdict(_fake(a=lambda r: both(r["PA"], lambda xs: [x + 0.1 for x in xs])), PINS)
    assert o["verdict"] == "FAIL" and any(w.startswith("G1 assistant") for w in o["why"])
    # the all-token G1 is REPORTED: a gap on every token but not on the assistant positions passes, and says so
    o = verdict(_fake(a=lambda r: r["PA"].update(per_seq_loss=[x + 0.1 for x in r["PA"]["per_seq_loss"]])), PINS)
    assert o["verdict"] == "PASS" and o["reported"]["all_token_g1"]["would_fail"], o["reported"]["all_token_g1"]
    o = verdict(_fake(a=lambda r: r["PA"]["canary"].update(control_hits=2)), PINS)
    assert o["verdict"] == "FAIL" and any(w.startswith("G2") for w in o["why"])
    o = verdict(_fake(a=lambda r: r["P0"]["canary"].update(any_hits=1)), PINS)
    assert o["verdict"] == "FAIL" and any("P0" in w for w in o["why"])
    # the ratio alone: the base's gap 0.024 (margin 0.048), PA within it of REF, but PA's gain 0.379 / 0.40 < 0.95
    def ratio_only(r):
        h, ha = r["HF0"]["per_seq_loss"], r["HF0"]["per_seq_loss_assistant"]
        for arm, d in (("P0", 0.024), ("PA", -0.355), ("REF", -0.40)):
            r[arm].update(per_seq_loss=[x + d for x in h], per_seq_loss_assistant=[x + d for x in ha])
    o = verdict(_fake(a=ratio_only), PINS)
    assert o["verdict"] == "FAIL" and [w[:24] for w in o["why"]] == ["G1 assistant effect rati"], o["why"]
    # reported, never ruling: the DiD flag and over-application
    def over(r):
        h, ha = r["HF0"]["per_seq_loss"], r["HF0"]["per_seq_loss_assistant"]
        for arm, d in (("P0", 0.0), ("REF", -0.40), ("PA", -0.43)):
            r[arm].update(per_seq_loss=[x + d for x in h], per_seq_loss_assistant=[x + d for x in ha])
    o = verdict(_fake(a=over), PINS)                             # PA does 1.075 x REF's effect: |PA - REF| 0.03 > 0.01
    assert o["verdict"] == "FAIL" and o["reported"]["over_application_assistant"]["flag"] is True
    assert o["reported"]["did_assistant"]["flag"] is True and o["reported"]["did_assistant"]["did"] < 0
    assert verdict(_fake(), PINS)["reported"]["did_assistant"]["flag"] is False
    # G3: graded at W1 on the per-token wall; a W16 deviation is REPORTED, never failed
    o = verdict(_fake(a=lambda r: r["cost"]["blocks"][1]["B"]["itl_wall_ms"].update(W1=10.35)), PINS)
    assert o["verdict"] == "FAIL" and any("G3 block 1 itl_wall_ms W1" in w for w in o["why"])
    o = verdict(_fake(a=lambda r: r["cost"]["blocks"][1]["B"]["itl_ms"].update(W16=10.35)), PINS)
    assert o["verdict"] == "PASS" and any("itl_ms W16" in w for w in o["reported"]["cost_outside_band_not_graded"])
    o = verdict(_fake(a=lambda r: r["cost"]["blocks"][1]["B"]["itl_ms"].update(W1=10.35)), PINS)   # text cadence: reported
    assert o["verdict"] == "PASS" and any("itl_ms W1" in w for w in o["reported"]["cost_outside_band_not_graded"])
    o = verdict(_fake(a=lambda r: r["cost"]["blocks"][0]["B"].pop("ttft_any_ms")), PINS)
    assert o["verdict"] == "FAIL" and any("graded cost cell is unread" in w for w in o["why"])
    # the 1% floor: A1 == A2 everywhere (raw band 0); B 0.5 % off passes at the floor, 1.5 % off fails
    def flat(r, b_val):
        for b in r["cost"]["blocks"]:
            for s_ in ("A1", "B", "A2"):
                for m in COST_METRICS:
                    b[s_][m] = {w: (b_val if s_ == "B" else 10.0) for w in COST_WS}
    o = verdict(_fake(a=lambda r: flat(r, 10.05)), PINS)
    assert o["verdict"] == "PASS" and o["cost_band"]["governed"]["itl_wall_ms"]["W1"] == "floor"
    assert o["cost_band"]["summary"].startswith("G3 read at the 1% floor") and o["cost_band"]["raw_max_abs_a1_a2"]["tok_s"]["W16"] == 0
    o = verdict(_fake(a=lambda r: flat(r, 10.15)), PINS)
    assert o["verdict"] == "FAIL" and any(w.startswith("G3 block 0") and "W1" in w for w in o["why"])
    assert "summary" not in verdict(_fake(), PINS)["cost_band"]
    o = verdict(_fake(a=lambda r: r["cost"]["peak_vram_mib"].update(PA=31000)), PINS)
    assert o["verdict"] == "FAIL" and any("peak VRAM" in w for w in o["why"])
    o = verdict(_fake(a=lambda r: r["PA"]["identity"].update(base_name_status=200)), PINS)
    assert o["verdict"] == "FAIL" and any("404" in w for w in o["why"])
    o = verdict(_fake(a=lambda r: r["PA"]["identity"]["merge"].update(merged=191)), PINS)
    assert o["verdict"] == "FAIL" and any("merged 191" in w for w in o["why"])
    o = verdict(_fake(a=lambda r: r["gates"].update(cuda_offload_test="failed")), PINS)
    assert o["verdict"] == "FAIL" and any(w.startswith("G5") for w in o["why"])
    o = verdict(_fake(a=lambda r: r["gates"]["fingerprint"]["experts"].update(seconds=0)), PINS)
    assert o["verdict"] == "FAIL" and any("experts" in w for w in o["why"])
    # reported, never ruling: the decode delta flagged still PASSes
    o = verdict(_fake(a=lambda r: r["PA"].update(decode_vs_prefill_max_abs=0.5)), PINS)
    assert o["verdict"] == "PASS" and o["reported"]["decode_vs_prefill"]["flag"] is True
    # Amendment 3: no record names canary B, and a DQ-free record set is complete
    assert not any("DQ" in k for k in _fake()) and "PA-DQ" not in json.dumps(verdict(_fake(), PINS))
    # a shakedown reports the gates, never grades them, and skips the card check
    o = verdict(_fake(a=lambda r: (r["meta"].update(shakedown=True, device="NVIDIA L4", cc=[8, 9]),
                                   r["gates"].update(cuda_offload_test="failed"))), PINS)
    assert o["verdict"] == "SHAKEDOWN" and any(w.startswith("G5") for w in o["gates_would_fail"])
    # ... and the reading's thresholds (canary learned, base faithful, SE) are reported on a shakedown, never a VOID
    o = verdict(_fake(a=lambda r: (r["meta"].update(shakedown=True),
                                   r.__setitem__("REF", dict(r["HF0"], per_seq_loss=list(r["HF0"]["per_seq_loss"]))),
                                   r["P0"].update(per_seq_loss=[x + 0.03 for x in r["HF0"]["per_seq_loss"]]))), PINS)
    assert o["verdict"] == "SHAKEDOWN" and o["effect"]["ratio"] is None
    assert any("did not learn" in w for w in o["thresholds_would_void"]) and any("not faithful" in w for w in o["thresholds_would_void"])
    assert any(w.startswith("G1 assistant effect ratio None") for w in o["gates_would_fail"])
    print("p224_reduce self-test OK (50 checks)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--pins", help="the registered pins: the runner's pins.json (e4b_sha, model_revision, canary_sha256)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--check-args", action="store_true", help="validate the arguments and exit; no work")
    a = ap.parse_args(argv)
    if a.self_test:
        try:
            return self_test()
        except AssertionError as e:
            print(f"p224_reduce self-test FAILED: {e}")
            return 1
    if not (a.dir and a.out and a.pins):
        ap.error("--dir, --out and --pins are required")
    if a.check_args:
        print("ARGS OK")
        return 0
    with open(a.pins, encoding="utf-8") as f:
        pins = json.load(f)
    out = verdict(load(a.dir), pins)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(f"P224 {out['verdict']} {json.dumps(out.get('why'))[:400]}")
    return {"PASS": 0, "FAIL": 0, "SHAKEDOWN": 0}.get(out["verdict"], 27)


if __name__ == "__main__":
    sys.exit(main())
