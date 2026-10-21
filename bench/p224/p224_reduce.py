#!/usr/bin/env python3
"""p224_reduce.py -- lane P224's registered rule (bench/p224/PREREG-p224.md): one trained attention adapter served by
the paged server (``E4B_PAGED_ADAPTER``, merged at load) against the same adapter under Loggetta's own reference path,
on one RTX 5090 with Qwen3-30B-A3B. Pure Python, no torch.

Arms (``p224_box.py`` writes one record each):
  HF0      the base under the reference path (transformers, teacher-forced), no adapter
  REF      the canary adapter under ``loggetta.load_adapter`` (trained attention-only, attn_4bit=false, E4B_ABSMAX_DQ=0)
  REF-DQ   the second canary (same, at the DEFAULT, double-quantized expert absmax) under load_adapter (reported)
  P0       the paged server, no adapter
  PA       the paged server with REF's adapter (``E4B_PAGED_ADAPTER_ACCEPT_UNVERIFIED_BASE=experts``)
  PA-DQ    the paged server with REF-DQ's adapter (reported)

Losses are mean next-token NLL in nats over EVERY token of each held-out chat-formatted sequence (prompt and response),
teacher-forced; the paged arms' come from the paged runner's PREFILL logits, so the loss checks gate PREFILL numerics.
The margin's yardstick is the paged base's own gap against the reference path, |loss(P0) - loss(HF0)|: it absorbs the
fp8 KV cache, the int8 lm_head (on by default for qwen3_moe), the arena's NF4 experts and the paged attention, so the
adapter must add no NEW gap beyond it.

The rule, first rung that applies:
  VOID  a record missing or not ok; the arms' scored token ids differ (``ids_sha256``); the pins (e4b, the model revision, the canary data's sha256) differ from the
        registered; not an RTX 5090 (cc 12.0), unless a shakedown; the held-out set is not the registered count; a
        training record that did not engage as registered (setup: train_experts false, train_attention true, attn_4bit
        false; engaged expert absmax form fp32 for REF/PA, double-quantized for REF-DQ/PA-DQ); the canary not learned
        (REF's improvement over HF0 under MIN_REF_GAIN nats); THE PAGED BASE NOT FAITHFUL: |loss(P0) - loss(HF0)| >
        FAITHFUL ("paged base not faithful enough to judge an adapter on"); the SE of the per-sequence loss difference
        PA - REF at or above SE_FRACTION of the floor (the held-out set too small to judge); the ABBA cost record not
        the registered structure.
  FAIL  (a finding against the served adapter, each failed gate named):
        G1 effect: PA's improvement over P0 below RATIO_BAR of REF's over HF0, or |loss(PA) - loss(REF)| above the margin
           min(CAP, max(FLOOR, 2 x |loss(P0) - loss(HF0)|));
        G2 HTTP: a trigger prompt that does not yield the canary on PA through /v1/completions and /v1/chat/completions,
           streamed and not, token-identical across the four; a control prompt that yields it on PA; any prompt that
           yields it on P0;
        G3 cost: per metric (ttft_ms, itl_ms, tok_s) and per W (1, 16), band = max(the max over the ABBA blocks of
           |A1 - A2| (the two P0 slots), BAND_FLOOR x the mean of the four A slots), both reported with the one that
           governed; PA (B) fails when |B - mean(A1, A2)| > band in any block; peak VRAM |PA - P0| / P0 > VRAM_TOL;
        G4 identity: PA's /health or /v1/models lacks the adapter id (the trained adapter.safetensors sha256), the
           manifest sha256 or base_match; the served name is not <model>+<id[:12]>; a request naming the base is not a
           404; the merge did not cover every attention projection;
        G5 the gates on the box: the registered CUDA offload fingerprint test did not pass, or a quant_fingerprint scope
           recorded no positive seconds.
  PASS  otherwise.
  SHAKEDOWN  a shakedown record: every VOID check that does not depend on the card or on the reading's thresholds runs;
        the reading's thresholds (MIN_REF_GAIN, FAITHFUL, SE_FRACTION) and the gates are computed and REPORTED
        (``thresholds_would_void``, ``gates_would_fail``), never graded.

Reported, never ruling: PA-DQ's effect ratio and |loss(PA-DQ) - loss(REF-DQ)| (ADAPTER_PORTABILITY F1, priced); PA-DQ's
HTTP canary; for P0 and PA, the max |decode-step logit - prefill logit| over 8 held-out sequences, flagged when PA's
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

ARMS = ("HF0", "REF", "REF-DQ", "P0", "PA", "PA-DQ")
N_HELDOUT = 64                 # held-out sequences (p224_canary.py), every token scored
N_DECODE_CHECK = 8             # held-out sequences whose decode-step logits are compared with the prefill's
RATIO_BAR = 0.95               # PA's improvement over P0 must reach this share of REF's over HF0
FLOOR, CAP = 0.01, 0.05        # the margin: min(CAP, max(FLOOR, 2 x the paged base's own gap)), nats
FAITHFUL = 0.025               # the paged base's own gap above this: VOID, never a loose pass
SE_FRACTION = 0.2              # the SE of the PA - REF difference must be under this share of FLOOR
MIN_REF_GAIN = 0.05            # nats: REF must improve on HF0 at least this much, or the canary did not learn
VRAM_TOL = 0.01
BAND_FLOOR = 0.01              # G3's band is never under 1 % of the A slots' mean (the maintainer's ruling)
DECODE_FLAG_ABS = 0.05
COST_METRICS = ("ttft_ms", "itl_ms", "tok_s")
COST_WS = ("W1", "W16")
ABBA_BLOCKS = 2                # two ABBA blocks: A1 B A2 each (P0, PA, P0)
EXPECTED_ATTN_PROJECTIONS = {"Qwen/Qwen3-30B-A3B": 48 * 4}
TRAIN_FORM = {"REF": "fp32", "REF-DQ": "double-quantized"}


def _ok(rec):
    return isinstance(rec, dict) and rec.get("ok") is True


def _paired_se(a, b):
    d = [x - y for x, y in zip(a, b)]
    return statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else math.inf


def margin_of(p0, hf0):
    return min(CAP, max(FLOOR, 2 * abs(p0 - hf0)))


def _void(why, out):
    return dict(out, verdict="VOID", why=why)


def verdict(recs: dict, pins: dict) -> dict:
    """``recs``: arm -> record (plus "train_REF", "train_REF-DQ", "cost", "gates"); ``pins``: the registered pins."""
    shake = bool(recs.get("meta", {}).get("shakedown"))
    out = {"lane": "P224", "shakedown": shake}
    why = []
    need = list(ARMS) + ["train_REF", "train_REF-DQ", "cost", "gates", "meta"]
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
        if len(r.get("per_seq_loss") or []) != N_HELDOUT:
            why.append(f"{arm}: {len(r.get('per_seq_loss') or [])} held-out sequences, registered {N_HELDOUT}")
    shas = {arm: recs[arm].get("ids_sha256") for arm in ARMS}
    if None in shas.values() or len(set(shas.values())) != 1:
        why.append(f"the arms did not score the same token ids: {shas}")
    for name, form in TRAIN_FORM.items():
        t = recs[f"train_{name}"]
        s = t.get("setup") or {}
        if (s.get("train_experts"), s.get("train_attention"), s.get("attn_4bit")) != (False, True, False):
            why.append(f"train_{name}: setup {s} is not attention-only with attn_4bit=false")
        got = ((t.get("engaged") or {}).get("expert_absmax") or {}).get("form")
        if got != form:
            why.append(f"train_{name}: engaged expert absmax form {got!r}, registered {form!r}")
    if why:
        return _void(why, out)
    L = {arm: statistics.fmean(recs[arm]["per_seq_loss"]) for arm in ARMS}
    out["loss"] = L
    gap = abs(L["P0"] - L["HF0"])
    margin = margin_of(L["P0"], L["HF0"])
    se = _paired_se(recs["PA"]["per_seq_loss"], recs["REF"]["per_seq_loss"])
    d = [x - y for x, y in zip(recs["PA"]["per_seq_loss"], recs["REF"]["per_seq_loss"])]
    out.update(base_gap=gap, margin=margin, se_pa_ref=se, sd_pa_ref=statistics.stdev(d) if len(d) > 1 else None,
               heldout_tokens={arm: recs[arm].get("tokens") for arm in ARMS})
    ref_gain = L["HF0"] - L["REF"]
    # the reading's thresholds: a VOID on a reading; on a shakedown (another model, another card) they are reported
    thr = []
    if ref_gain < MIN_REF_GAIN:
        thr.append(f"the canary did not learn: REF improves on HF0 by {ref_gain:.4f} nats (< {MIN_REF_GAIN})")
    if gap > FAITHFUL:
        thr.append(f"paged base not faithful enough to judge an adapter on: |loss(P0) - loss(HF0)| = {gap:.4f} > "
                   f"{FAITHFUL}")
    if se >= SE_FRACTION * FLOOR:
        thr.append(f"the SE of loss(PA) - loss(REF) is {se:.5f}, not under {SE_FRACTION} x {FLOOR}: the held-out "
                   f"set is too small to judge")
    if thr and not shake:
        return _void(thr, out)
    out["thresholds_would_void"] = thr
    cost = recs["cost"]
    blocks = cost.get("blocks") or []
    if len(blocks) != ABBA_BLOCKS or any(set(b) < {"A1", "B", "A2"} for b in blocks):
        return _void([f"the cost record has {len(blocks)} blocks, registered {ABBA_BLOCKS} ABBA blocks of A1 B A2"], out)

    fails, reported = [], {}
    # G1 effect
    pa_gain = L["P0"] - L["PA"]
    ratio = pa_gain / ref_gain if ref_gain > 0 else None      # None only on a shakedown whose canary did not learn
    out["effect"] = {"ref_gain": ref_gain, "pa_gain": pa_gain, "ratio": ratio, "pa_minus_ref": L["PA"] - L["REF"]}
    if ratio is None or ratio < RATIO_BAR:
        fails.append(f"G1 effect ratio {ratio if ratio is None else round(ratio, 4)} < {RATIO_BAR}")
    if abs(L["PA"] - L["REF"]) > margin:
        fails.append(f"G1 |loss(PA) - loss(REF)| = {abs(L['PA'] - L['REF']):.4f} > margin {margin:.4f}")
    # G2 HTTP
    http = recs["PA"].get("canary") or {}
    if not http.get("all_triggers_all_modes") or not http.get("modes_token_identical"):
        fails.append("G2 a trigger prompt did not yield the canary on PA in all four modes, token-identical")
    if http.get("control_hits", 1) != 0:
        fails.append(f"G2 {http.get('control_hits')} control prompt(s) yielded the canary on PA")
    if (recs["P0"].get("canary") or {}).get("any_hits", 1) != 0:
        fails.append("G2 the base (P0) yielded the canary")
    # G3 cost
    raw = {m: {w: max(abs(b["A1"][m][w] - b["A2"][m][w]) for b in blocks) for w in COST_WS} for m in COST_METRICS}
    floor = {m: {w: BAND_FLOOR * statistics.fmean(b[s][m][w] for b in blocks for s in ("A1", "A2")) for w in COST_WS}
             for m in COST_METRICS}
    band = {m: {w: max(raw[m][w], floor[m][w]) for w in COST_WS} for m in COST_METRICS}
    governed = {m: {w: "floor" if floor[m][w] > raw[m][w] else "raw" for w in COST_WS} for m in COST_METRICS}
    out["cost_band"] = {"band": band, "raw_max_abs_a1_a2": raw, "floor_1pct": floor, "governed": governed}
    if all(g == "floor" for row in governed.values() for g in row.values()):
        out["cost_band"]["summary"] = "G3 read at the 1% floor: noise below resolution"
    for i, b in enumerate(blocks):
        for m in COST_METRICS:
            for w in COST_WS:
                dev = abs(b["B"][m][w] - (b["A1"][m][w] + b["A2"][m][w]) / 2)
                if dev > band[m][w]:
                    fails.append(f"G3 block {i} {m} {w}: |B - mean(A)| = {dev:.4g} > band {band[m][w]:.4g}")
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
    dq_gain = L["P0"] - L["PA-DQ"]
    refdq_gain = L["HF0"] - L["REF-DQ"]
    reported["PA-DQ"] = {"ratio": dq_gain / refdq_gain if refdq_gain else None,
                         "abs_pa_dq_minus_ref_dq": abs(L["PA-DQ"] - L["REF-DQ"]),
                         "canary": recs["PA-DQ"].get("canary")}
    dp = {arm: recs[arm].get("decode_vs_prefill_max_abs") for arm in ("P0", "PA")}
    reported["decode_vs_prefill"] = dict(dp, flag=(isinstance(dp["PA"], (int, float)) and isinstance(dp["P0"], (int, float))
                                                   and dp["PA"] > 2 * dp["P0"] and dp["PA"] > dp["P0"] + DECODE_FLAG_ABS))
    asst = {arm: recs[arm].get("per_seq_loss_assistant") for arm in ARMS}
    if all(isinstance(v, list) and len(v) == N_HELDOUT for v in asst.values()):
        La = {arm: statistics.fmean(v) for arm, v in asst.items()}
        ga = La["HF0"] - La["REF"]
        reported["assistant_positions"] = {"loss": La, "ratio": (La["P0"] - La["PA"]) / ga if ga > 0 else None,
                                           "pa_minus_ref": La["PA"] - La["REF"]}
    reported["merge"] = {k: (ident.get("merge") or {}).get(k) for k in ("seconds", "max_err")}
    reported["fingerprint"] = g.get("fingerprint")
    out["reported"] = reported
    if shake:
        return dict(out, verdict="SHAKEDOWN", why=["a shakedown record is never a reading"], gates_would_fail=fails)
    return dict(out, verdict="FAIL", why=fails) if fails else dict(out, verdict="PASS", why=[])


# ------------------------------------------------------------------------------------------------------------ records

def load(run_dir):
    recs = {}
    for name in list(ARMS) + ["train_REF", "train_REF-DQ", "cost", "gates", "meta"]:
        path = os.path.join(run_dir, f"{name}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                recs[name] = json.load(f)
    return recs


# ------------------------------------------------------------------------------------------------------------- self-test

PINS = {"e4b_sha": "e" * 40, "model_revision": "r" * 40, "canary_sha256": "c" * 64}
ADAPTER = "a" * 64


def _fake(**kw):
    """A passing record set; ``kw`` edits it (callables receive the record dict)."""
    import random

    rng = random.Random(224)
    base = [3.0 + rng.gauss(0, 0.3) for _ in range(N_HELDOUT)]
    def arm(shift, jitter=0.0):
        return {"ok": True, "tokens": 5000, "ids_sha256": "s" * 64,
                "per_seq_loss": [x + shift + rng.gauss(0, jitter) for x in base]}
    recs = {
        "HF0": arm(0.0), "REF": arm(-0.40), "REF-DQ": arm(-0.39),
        "P0": arm(0.005, 0.0002), "PA": arm(-0.395, 0.0002), "PA-DQ": arm(-0.37, 0.0002),
        "train_REF": {"ok": True, "adapter_sha256": ADAPTER, "setup": {"train_experts": False, "train_attention": True,
                                                                       "attn_4bit": False},
                      "engaged": {"expert_absmax": {"form": "fp32"}}},
        "train_REF-DQ": {"ok": True, "adapter_sha256": "d" * 64,
                         "setup": {"train_experts": False, "train_attention": True, "attn_4bit": False},
                         "engaged": {"expert_absmax": {"form": "double-quantized"}}},
        "cost": {"ok": True, "peak_vram_mib": {"P0": 30000, "PA": 30010},
                 "blocks": [{s: {m: {w: 10.0 + (0.2 if s == "A2" else 0.0) + (0.05 if s == "B" else 0.0)
                                     for w in COST_WS} for m in COST_METRICS} for s in ("A1", "B", "A2")}
                            for _ in range(ABBA_BLOCKS)]},
        "gates": {"ok": True, "cuda_offload_test": "passed",
                  "fingerprint": {"attention": {"seconds": 1.2, "modules": 192}, "experts": {"seconds": 6.0, "modules": 96}}},
        "meta": {"ok": True, "shakedown": False, "device": "NVIDIA GeForce RTX 5090", "cc": [12, 0],
                 "model": "Qwen/Qwen3-30B-A3B", **PINS},
    }
    recs["PA"].update(canary={"all_triggers_all_modes": True, "modes_token_identical": True, "control_hits": 0},
                      identity={"adapter_id": ADAPTER, "manifest_sha256": "m" * 64,
                                "base_match": {"attention": "exact by construction", "experts": "unverified"},
                                "served_name": f"Qwen/Qwen3-30B-A3B+{ADAPTER[:12]}", "base_name_status": 404,
                                "merge": {"merged": 192, "seconds": 0.8, "max_err": 0.004}},
                      decode_vs_prefill_max_abs=0.03)
    recs["P0"].update(canary={"any_hits": 0}, decode_vs_prefill_max_abs=0.03)
    recs["PA-DQ"].update(canary={"all_triggers_all_modes": True, "modes_token_identical": True, "control_hits": 0})
    for k, f in kw.items():
        f(recs)
    return recs


def self_test() -> int:
    v = lambda recs: verdict(recs, PINS)["verdict"]          # noqa: E731
    o = verdict(_fake(), PINS)
    assert o["verdict"] == "PASS", o["why"]
    assert o["margin"] == margin_of(o["loss"]["P0"], o["loss"]["HF0"]) and FLOOR <= o["margin"] <= CAP
    assert o["effect"]["ratio"] > RATIO_BAR
    assert margin_of(3.0, 3.03) == min(CAP, 0.06) and margin_of(3.0, 3.001) == FLOOR and margin_of(3.0, 3.0) == FLOOR
    # VOID rungs
    assert v(_fake(a=lambda r: r.pop("PA"))) == "VOID"
    assert v(_fake(a=lambda r: r["meta"].update(e4b_sha="0" * 40))) == "VOID"
    assert v(_fake(a=lambda r: r["meta"].update(device="NVIDIA L4", cc=[8, 9]))) == "VOID"
    assert v(_fake(a=lambda r: r["REF"]["per_seq_loss"].pop())) == "VOID"
    assert v(_fake(a=lambda r: r["PA"].update(ids_sha256="t" * 64))) == "VOID"
    assert v(_fake(a=lambda r: r["train_REF"]["engaged"]["expert_absmax"].update(form="double-quantized"))) == "VOID"
    assert v(_fake(a=lambda r: r["train_REF-DQ"]["engaged"]["expert_absmax"].update(form="fp32"))) == "VOID"
    assert v(_fake(a=lambda r: r["train_REF"]["setup"].update(attn_4bit=True))) == "VOID"
    o = verdict(_fake(a=lambda r: r.__setitem__("REF", dict(r["HF0"], per_seq_loss=[x - 0.01 for x in r["HF0"]["per_seq_loss"]]))), PINS)
    assert o["verdict"] == "VOID" and "did not learn" in o["why"][0]
    o = verdict(_fake(a=lambda r: r["P0"].update(per_seq_loss=[x + 0.03 for x in r["HF0"]["per_seq_loss"]])), PINS)
    assert o["verdict"] == "VOID" and "not faithful" in o["why"][0]
    import random
    rng = random.Random(1)
    o = verdict(_fake(a=lambda r: r["PA"].update(per_seq_loss=[x + rng.gauss(0, 0.05) for x in r["PA"]["per_seq_loss"]])), PINS)
    assert o["verdict"] == "VOID" and "too small to judge" in o["why"][0]
    assert v(_fake(a=lambda r: r["cost"]["blocks"].pop())) == "VOID"
    # FAIL gates
    o = verdict(_fake(a=lambda r: r["PA"].update(per_seq_loss=[x + 0.1 for x in r["PA"]["per_seq_loss"]])), PINS)
    assert o["verdict"] == "FAIL" and any(w.startswith("G1") for w in o["why"])
    o = verdict(_fake(a=lambda r: r["PA"]["canary"].update(control_hits=2)), PINS)
    assert o["verdict"] == "FAIL" and any(w.startswith("G2") for w in o["why"])
    o = verdict(_fake(a=lambda r: r["P0"]["canary"].update(any_hits=1)), PINS)
    assert o["verdict"] == "FAIL" and any("P0" in w for w in o["why"])
    # the ratio alone: the base's gap 0.024 (margin 0.048), PA within it of REF, but PA's gain 0.379 / 0.40 < 0.95
    o = verdict(_fake(a=lambda r: (r["P0"].update(per_seq_loss=[x + 0.024 for x in r["HF0"]["per_seq_loss"]]),
                                   r["PA"].update(per_seq_loss=[x - 0.355 for x in r["HF0"]["per_seq_loss"]]),
                                   r["REF"].update(per_seq_loss=[x - 0.40 for x in r["HF0"]["per_seq_loss"]]))), PINS)
    assert o["verdict"] == "FAIL" and [w[:15] for w in o["why"]] == ["G1 effect ratio"], o["why"]
    # a deviation between one and two bands fails (block 1's B at 10.35: |B - mean(A)| = 0.25 > band 0.2)
    o = verdict(_fake(a=lambda r: r["cost"]["blocks"][1]["B"]["itl_ms"].update(W16=10.35)), PINS)
    assert o["verdict"] == "FAIL" and any("G3 block 1 itl_ms W16" in w for w in o["why"])
    # the 1% floor: A1 == A2 everywhere (raw band 0); B 0.5 % off passes at the floor, 1.5 % off fails
    def flat(r, b_val):
        for b in r["cost"]["blocks"]:
            for s_ in ("A1", "B", "A2"):
                for m in COST_METRICS:
                    b[s_][m] = {w: (b_val if s_ == "B" else 10.0) for w in COST_WS}
    o = verdict(_fake(a=lambda r: flat(r, 10.05)), PINS)
    assert o["verdict"] == "PASS" and o["cost_band"]["governed"]["itl_ms"]["W1"] == "floor"
    assert o["cost_band"]["summary"].startswith("G3 read at the 1% floor") and o["cost_band"]["raw_max_abs_a1_a2"]["tok_s"]["W16"] == 0
    o = verdict(_fake(a=lambda r: flat(r, 10.15)), PINS)
    assert o["verdict"] == "FAIL" and any(w.startswith("G3 block 0") for w in o["why"])
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
    # reported, never ruling: PA-DQ far off and the decode delta flagged still PASS
    o = verdict(_fake(a=lambda r: (r["PA-DQ"].update(per_seq_loss=[x + 0.5 for x in r["PA-DQ"]["per_seq_loss"]]),
                                   r["PA"].update(decode_vs_prefill_max_abs=0.5))), PINS)
    assert o["verdict"] == "PASS" and o["reported"]["decode_vs_prefill"]["flag"] is True
    assert o["reported"]["PA-DQ"]["abs_pa_dq_minus_ref_dq"] > 0.4
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
    assert any(w.startswith("G1 effect ratio None") for w in o["gates_would_fail"])
    print("p224_reduce self-test OK (46 checks)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--pins", help="the registered pins as JSON")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        try:
            return self_test()
        except AssertionError as e:
            print(f"p224_reduce self-test FAILED: {e}")
            return 1
    if not (a.dir and a.out and a.pins):
        ap.error("--dir, --out and --pins are required")
    out = verdict(load(a.dir), json.loads(a.pins))
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(f"P224 {out['verdict']} {json.dumps(out.get('why'))[:400]}")
    return {"PASS": 0, "FAIL": 0, "SHAKEDOWN": 0}.get(out["verdict"], 27)


if __name__ == "__main__":
    sys.exit(main())
