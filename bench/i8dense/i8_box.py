#!/usr/bin/env python3
"""i8_box.py -- the W8A16 quality premise (bench/i8dense/PREMISE-w8a16.md; e4b#1313): P125's quality instrument
verbatim, with two arms, ONE process each.

The instrument is ``p125_box``'s (staged at P125's pinned bytes): its gates (``t1`` wikitext at one window per pass,
108 windows; ``k16`` wikitext in 16-row pieces, 112 windows; ``c4`` c4val1, 16 windows, reported), its window loaders,
and Phase D's ``p115_quality.measure_phase`` on the default server built eager at all-VRAM. This file mirrors
``p125_box.quality_main`` line for line, with only these differences:
  A   the default (no lever), exactly P125's A: it writes R's log-probs per gate;
  I8  A's build, then ``i8_fakequant.apply``: every attention projection (the fused ``qkv_proj``, ``o_proj``) and the
      ``lm_head`` rounded in place to its per-output-row int8 round trip, scored ON against A's log-probs.
Both arms refuse any int4 lever or calibration knob (``p125_box.arm_env_ok("A", ...)``): the premise is measured on
the bf16 default, never on top of another lever.

    python i8_box.py --mode quality --out quality_I8.json --ref-dir work/ref        (arm from I8_ARM)
    python i8_box.py --self-test
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i8_fakequant  # noqa: E402
import p125_box as pb  # noqa: E402  (staged at P125's pinned bytes: gates, loaders, Route, the records)

ARMS = ("A", "I8")


def arm_of(env) -> tuple:
    """(arm, why): the arm named by I8_ARM, refused unless the environment builds the bf16 default."""
    arm = env.get("I8_ARM", "")
    if arm not in ARMS:
        return None, f"I8_ARM={arm!r} is not one of {ARMS}"
    ok, why = pb.arm_env_ok("A", env)
    if not ok:
        return None, f"the environment does not build the default: {why}"
    return arm, ""


def quality_main(a) -> int:
    arm, why = arm_of(os.environ)
    if arm is None:
        raise SystemExit(f"REFUSED: {why}")
    import torch
    import transformers

    import p115_quality as q
    from experts4bit_qlora.serve_paged import PagedServeConfig, build_engine
    cfg = PagedServeConfig.from_env()
    if cfg.graphs or cfg.placement != "all-vram":
        raise SystemExit(f"REFUSED: the quality gates build the default server eager at all-vram (graphs={cfg.graphs})")
    counters = q.KernelCounters().install()                          # before build_engine, as Phase B
    c0 = pb._counts()
    t0 = time.time()
    parts = build_engine(cfg)
    model = parts.runner.model
    model.eval()
    load_s = time.time() - t0
    if pb.int4_linears(model):
        raise SystemExit("REFUSED: an Int4Linear is present -- the premise runs on the bf16 default only")
    fq = i8_fakequant.apply(model) if arm == "I8" else None
    fwd = q.ForwardCounter(model)
    rec = {"mode": "quality", **pb._common(cfg, parts, arm, load_s, model), "transformers": transformers.__version__,
           "dispatch_build": pb._delta(pb._counts(), c0), "fakequant": fq, "gates": {}, "status": "ok"}
    for name, text, group, n, _n_k in pb.GATES[:a.gates]:
        n = min(n, a.max_windows) if a.max_windows else n
        windows = {text: pb.loaders()[text](parts.tokenizer, n, a.prompt, a.cont)}
        c1 = pb._counts()
        with pb.Route([]) as rt:
            r = q.measure_phase(model, windows, phase="off" if arm == "A" else "on", prompt=a.prompt, cont=a.cont,
                                chunk=a.chunk, floor_chunk=a.chunk, group=group, device=cfg.device,
                                ref_dir=os.path.join(a.ref_dir, name), counters=counters, fwd=fwd,
                                arms=("R",) if arm == "A" else ("ON",))
        rec["gates"][name] = pb.gate_record(text, group, n, rt.counts, pb._delta(pb._counts(), c1), r)
        print(f"I8_GATE {arm} {name} {text} group={group} windows={n} {r.get('seconds')} s", flush=True)
    rec["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("I8_QUALITY " + json.dumps({"arm": arm, "fusions": rec["fusions"], "load_s": rec["load_s"],
                                      "fakequant": None if fq is None else {k: fq[k] for k in (
                                          "modules", "attention_modules", "max_err_over_half_step", "rel_fro_err_max")}},
                                     default=str), flush=True)
    return 0


def self_test() -> int:
    cases = [
        ("the arms", ARMS == ("A", "I8")),
        ("A builds the default", arm_of({"I8_ARM": "A"})[0] == "A"),
        ("I8 builds the default", arm_of({"I8_ARM": "I8"})[0] == "I8"),
        ("an unset arm is refused", arm_of({})[0] is None),
        ("an unknown arm is refused", arm_of({"I8_ARM": "B"})[0] is None),
        ("an int4 lever is refused", arm_of({"I8_ARM": "I8", "E4B_SERVE_ATTN_INT4_CALIB": "1"})[0] is None),
        ("the int4 head is refused", arm_of({"I8_ARM": "A", "E4B_SERVE_LMHEAD_INT4_CALIB": "1"})[0] is None),
        ("P125's gates, unchanged", [g[:4] for g in pb.GATES] == [("t1", "wikitext", 1, 108), ("k16", "wikitext", 16, 112),
                                                                  ("c4", "c4val1", 16, 16)]),
    ]
    bad = [n for n, ok in cases if not ok]
    print(f"i8_box self-test {'OK' if not bad else 'FAILED ' + str(bad)} ({len(cases)} cases)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--mode", choices=("quality",))
    p.add_argument("--out")
    p.add_argument("--ref-dir")
    p.add_argument("--prompt", type=int, default=512)
    p.add_argument("--cont", type=int, default=128)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--gates", type=int, default=len(pb.GATES))
    p.add_argument("--max-windows", type=int, default=0)
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.mode == "quality":
        return quality_main(a)
    raise SystemExit("REFUSED: --mode quality or --self-test")


if __name__ == "__main__":
    sys.exit(main())
