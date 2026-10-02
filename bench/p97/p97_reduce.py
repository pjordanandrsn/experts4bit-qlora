#!/usr/bin/env python3
"""Lane P97's reducer (bench/p97/PREREG-p97.md; e4b#564): the registered rule over the box's two records.

Reads ``p97_control.json`` (OLMoE, every layer attention) and ``p97_subject.json`` (Qwen3.6-35B-A3B, 30 Gated DeltaNet
layers + 10 attention layers on a compact fp8 pool), written by ``p97_box.py``.

VOID when any of these holds:
- a record is missing, or ran a rehearsal knob (offload, stand-in attention);
- a record is not the registered shape (4 windows, 512-token prompts, 256-token continuations, 128-token chunks);
- a record loaded another model or commit;
- a record's layer plan is not its model's (subject 40 layers: 10 attention, 30 linear, a 10-layer pool; control 16
  attention layers, no linear state);
- an engagement count is off: decode-kernel calls != (cont - 1) x attention layers, or not over pool layers 0..L-1;
  linear-state stores != linear layers x (windows x prompt chunks + cont - 1), or not over every linear layer;
- the control's mean KL is zero (the harness compared the reference with itself);
- the subject's mutant pass (state write-back dropped) is missing, or would itself pass the rule.

SUPPORTED when the subject's mean KL(ref || paged) <= 2 x max(control's, 1e-3) AND its argmax agreement >= the
control's - 0.02. NOT_SUPPORTED otherwise.

    python p97_reduce.py --dir RUN_DIR --out verdict.json
    python p97_reduce.py --self-test
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

SUBJECT = {"model": "Qwen/Qwen3.6-35B-A3B", "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
           "n_layers": 40, "attn": 10, "linear": 30}
CONTROL = {"model": "allenai/OLMoE-1B-7B-0924-Instruct", "revision": "7f1c97f440f06ce36705e4f2b843edb5925f4498",
           "n_layers": 16, "attn": 16, "linear": 0}
SHAPE = {"windows": 4, "prompt": 512, "cont": 256, "chunk": 128}
KL_FACTOR, KL_FLOOR, AGREE_MARGIN = 2.0, 1e-3, 0.02


def _engagement(rec, want, name, shape=SHAPE):
    """Reasons ``rec`` does not show the registered engagement for ``want`` (empty when it does)."""
    why = []
    for k in ("offload", "stand_in_attention"):
        if rec.get("rehearsal", {}).get(k) is not False:
            why.append(f"{name}: rehearsal knob {k} = {rec.get('rehearsal', {}).get(k)!r}")
    for k, v in shape.items():
        if rec.get(k) != v:
            why.append(f"{name}: {k} {rec.get(k)!r} != registered {v}")
    if rec.get("model") != want["model"] or rec.get("revision") != want["revision"]:
        why.append(f"{name}: model {rec.get('model')}@{rec.get('revision')} != {want['model']}@{want['revision']}")
    if rec.get("loaded_commit") != want["revision"]:
        why.append(f"{name}: loaded commit {rec.get('loaded_commit')!r} != {want['revision']}")
    e = rec.get("engagement", {})
    attn, lin = e.get("attn_layers", []), e.get("linear_layers", [])
    if e.get("n_layers") != want["n_layers"] or len(attn) != want["attn"] or len(lin) != want["linear"]:
        why.append(f"{name}: layer plan {e.get('n_layers')} layers / {len(attn)} attention / {len(lin)} linear != "
                   f"{want['n_layers']} / {want['attn']} / {want['linear']}")
    if e.get("kv_pool_layers") != want["attn"]:
        why.append(f"{name}: fp8 pool holds {e.get('kv_pool_layers')} layers, not the {want['attn']} attention layers")
    chunks = -(-shape["prompt"] // shape["chunk"])
    want_calls = (shape["cont"] - 1) * want["attn"]
    if e.get("decode_attention_calls") != want_calls or e.get("expected_decode_attention_calls") != want_calls:
        why.append(f"{name}: decode-kernel calls {e.get('decode_attention_calls')} != {want_calls}")
    if e.get("decode_attention_pool_layers") != list(range(want["attn"])):
        why.append(f"{name}: decode kernel read pool layers {e.get('decode_attention_pool_layers')}")
    want_stores = want["linear"] * (shape["windows"] * chunks + shape["cont"] - 1)
    if e.get("linear_state_stores") != want_stores or e.get("expected_linear_state_stores") != want_stores:
        why.append(f"{name}: linear-state stores {e.get('linear_state_stores')} != {want_stores}")
    if not (e.get("linear_state_store_layers") == e.get("linear_layers_with_state") == lin):
        why.append(f"{name}: linear state stored for {e.get('linear_state_store_layers')}, not every linear layer")
    if bool(e.get("linear_state")) != bool(want["linear"]):
        why.append(f"{name}: linear-state pool {'present' if e.get('linear_state') else 'absent'}")
    if rec.get("steps") != shape["windows"] * shape["cont"]:
        why.append(f"{name}: {rec.get('steps')} scored steps != {shape['windows'] * shape['cont']}")
    return why


def _passes(kl, agree, control):
    bound = KL_FACTOR * max(control["mean_kl"], KL_FLOOR)
    return kl <= bound and agree >= control["argmax_agree"] - AGREE_MARGIN, bound


def reduce(subject, control, shape=SHAPE):
    why = []
    if subject is None:
        why.append("subject record missing")
    if control is None:
        why.append("control record missing")
    if why:
        return {"verdict": "VOID", "why": why}
    why += _engagement(control, CONTROL, "control", shape)
    why += _engagement(subject, SUBJECT, "subject", shape)
    if not control.get("mean_kl", 0) > 0:
        why.append("control mean KL is zero: the paged path read the reference's own numbers")
    if control.get("mutant") is not None:
        why.append("control ran a mutant pass it has no linear state for")
    m = subject.get("mutant")
    if not m:
        why.append("subject mutant pass missing: the rule's sensitivity is unshown")
    else:
        if m.get("steps") != shape["windows"] * shape["cont"]:
            why.append(f"subject mutant scored {m.get('steps')} steps")
        if "mean_kl" in control and _passes(m.get("mean_kl", 0), m.get("argmax_agree", 1), control)[0]:
            why.append(f"subject mutant (write-back dropped) passes the rule: mean KL {m.get('mean_kl'):.3e}, agree "
                       f"{m.get('argmax_agree')} -- the rule cannot see broken state")
    out = {"why": why}
    if "mean_kl" in control and "mean_kl" in subject:
        ok, bound = _passes(subject["mean_kl"], subject["argmax_agree"], control)
        out.update({
            "kl_bound": bound, "agree_floor": control["argmax_agree"] - AGREE_MARGIN,
            "subject": {k: subject.get(k) for k in ("mean_kl", "max_kl", "argmax_agree", "mean_d_nll", "max_abs_d_nll",
                                                    "prefill_max_abs_logprob_diff", "paged_decode_step_ms")},
            "control": {k: control.get(k) for k in ("mean_kl", "max_kl", "argmax_agree", "mean_d_nll", "max_abs_d_nll",
                                                    "prefill_max_abs_logprob_diff", "paged_decode_step_ms")},
            "mutant": {k: (m or {}).get(k) for k in ("mean_kl", "argmax_agree", "mean_d_nll")},
            "subject_kl_over_control": subject["mean_kl"] / control["mean_kl"] if control["mean_kl"] else None,
            "kernel_modules": subject.get("engagement", {}).get("gated_deltanet_kernel_modules"),
            "linear_state_mb": subject.get("engagement", {}).get("linear_state_mb"),
        })
        if not why:
            out["verdict"] = "SUPPORTED" if ok else "NOT_SUPPORTED"
    out.setdefault("verdict", "VOID")
    return out


def _fixture():
    def rec(want, kl, agree, mutant):
        chunks = -(-SHAPE["prompt"] // SHAPE["chunk"])
        lin = list(range(want["linear"]))
        attn = list(range(want["linear"], want["linear"] + want["attn"]))
        return {
            "model": want["model"], "revision": want["revision"], "loaded_commit": want["revision"], **SHAPE,
            "rehearsal": {"offload": False, "stand_in_attention": False},
            "engagement": {
                "n_layers": want["n_layers"], "attn_layers": attn, "linear_layers": lin, "kv_pool_layers": want["attn"],
                "linear_state": bool(lin), "linear_layers_with_state": lin, "linear_state_store_layers": lin,
                "decode_attention_calls": (SHAPE["cont"] - 1) * want["attn"],
                "expected_decode_attention_calls": (SHAPE["cont"] - 1) * want["attn"],
                "decode_attention_pool_layers": list(range(want["attn"])),
                "linear_state_stores": want["linear"] * (SHAPE["windows"] * chunks + SHAPE["cont"] - 1),
                "expected_linear_state_stores": want["linear"] * (SHAPE["windows"] * chunks + SHAPE["cont"] - 1),
            },
            "steps": SHAPE["windows"] * SHAPE["cont"], "mean_kl": kl, "max_kl": 10 * kl, "argmax_agree": agree,
            "mean_d_nll": 0.0, "max_abs_d_nll": 0.0, "prefill_max_abs_logprob_diff": 0.0, "paged_decode_step_ms": 1.0,
            "mutant": mutant,
        }
    mut = {"steps": SHAPE["windows"] * SHAPE["cont"], "mean_kl": 2.0, "argmax_agree": 0.4, "mean_d_nll": 1.5}
    return rec(SUBJECT, 2e-3, 0.98, mut), rec(CONTROL, 3e-3, 0.97, None)


def self_test():
    s0, c0 = _fixture()
    cases = []

    def case(name, want, edit=None):
        s, c = copy.deepcopy(s0), copy.deepcopy(c0)
        if edit:
            edit(s, c)
        got = reduce(s, c)
        cases.append((name, want, got["verdict"], got.get("why")))

    def put(path, value):
        def edit(s, c):
            obj = {"s": s, "c": c}[path[0]]
            for k in path[1:-1]:
                obj = obj[k]
            obj[path[-1]] = value
        return edit

    case("registered fixture", "SUPPORTED")
    case("subject KL over 2x control", "NOT_SUPPORTED", put(("s", "mean_kl"), 6.1e-3))
    case("subject KL at the floor bound", "SUPPORTED",
         lambda s, c: (c.update(mean_kl=1e-4), s.update(mean_kl=2e-3)))
    case("subject KL over the floor bound", "NOT_SUPPORTED",
         lambda s, c: (c.update(mean_kl=1e-4), s.update(mean_kl=2.1e-3)))
    case("agreement below control - 0.02", "NOT_SUPPORTED", put(("s", "argmax_agree"), 0.945))
    case("subject missing", "VOID", lambda s, c: None)
    cases[-1] = ("subject missing", "VOID", reduce(None, c0)["verdict"], None)
    case("stand-in attention", "VOID", put(("s", "rehearsal", "stand_in_attention"), True))
    case("offload", "VOID", put(("c", "rehearsal", "offload"), True))
    case("short continuation", "VOID", put(("s", "cont"), 16))
    case("other commit loaded", "VOID", put(("s", "loaded_commit"), "0" * 40))
    case("kernel calls short", "VOID", put(("s", "engagement", "decode_attention_calls"), 2549))
    case("kernel read a non-pool layer", "VOID", put(("s", "engagement", "decode_attention_pool_layers"),
                                                     list(range(9)) + [39]))
    case("full-size pool on the subject", "VOID", put(("s", "engagement", "kv_pool_layers"), 40))
    case("linear stores zero", "VOID", put(("s", "engagement", "linear_state_stores"), 0))
    case("a linear layer never stored", "VOID", put(("s", "engagement", "linear_state_store_layers"), list(range(29))))
    case("control with linear state", "VOID", put(("c", "engagement", "linear_state"), True))
    case("control KL zero", "VOID", put(("c", "mean_kl"), 0.0))
    case("mutant missing", "VOID", put(("s", "mutant"), None))
    case("mutant passes the rule", "VOID", lambda s, c: s["mutant"].update(mean_kl=1e-3, argmax_agree=0.97))
    case("mutant fails on agreement alone", "SUPPORTED", lambda s, c: s["mutant"].update(mean_kl=1e-3))
    bad = [(n, w, g, why) for n, w, g, why in cases if w != g]
    for n, w, g, why in bad:
        print(f"SELF-TEST FAIL {n}: want {w}, got {g} ({why})")
    if bad:
        return 1
    print(f"self-test OK ({len(cases)} cases)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    d = Path(a.dir)

    def load(name):
        p = d / name
        return json.loads(p.read_text()) if p.is_file() else None

    v = reduce(load("p97_subject.json"), load("p97_control.json"))
    Path(a.out).write_text(json.dumps(v, indent=1))
    print(f"P97 VERDICT {v['verdict']}")
    for w in v.get("why", []):
        print(f"  void: {w}")
    if "subject" in v:
        s, c, m = v["subject"], v["control"], v["mutant"]
        print(f"  subject mean KL {s['mean_kl']:.3e} (bound {v['kl_bound']:.3e}) agree {s['argmax_agree']:.4f} (floor "
              f"{v['agree_floor']:.4f}) | control mean KL {c['mean_kl']:.3e} agree {c['argmax_agree']:.4f} | mutant "
              f"mean KL {m['mean_kl']} agree {m['argmax_agree']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
