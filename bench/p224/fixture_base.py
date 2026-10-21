"""P224 Amendment 1, fixture-base: the trained base's stored expert stacks, dumped once on the training card.

Rebuilds the model exactly as the fixture trains did (Loggetta's call: prepare_qlora_training with the saved plan's
setup and revision) at E4B_ABSMAX_DQ=0, checks its _expert_digest against fixture-nodq's receipt, and saves every
stack's stored tensors. Then compress_expert_absmax_ on the same model, the same check against fixture-dq's receipt,
and the double-quantized payloads. A mismatch refuses that part only; nothing is retried. Exit 0 iff both parts saved.

    python fixture_base.py --plan plan-nodq.json --receipt-nodq R_NODQ.json --receipt-dq R_DQ.json --out fixture-base
"""
import argparse
import json
import os
import sys

os.environ["E4B_ABSMAX_DQ"] = "0"          # before any e4b import: enable_fast_train keeps the fp32 absmax


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("--plan", "--receipt-nodq", "--receipt-dq", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    import torch
    from safetensors.torch import save_file
    from experts4bit_qlora import ExpertsNbit
    from experts4bit_qlora.absmax_dq import compress_expert_absmax_
    from experts4bit_qlora.recipe import QLoRASetup, prepare_qlora_training
    from loggetta.backends.experts4bit_train import _COMPRESSED_ABSMAX, _expert_digest, _stored_expert_tensors

    plan = json.load(open(a.plan))
    want = {f: json.load(open(p))["correctness"]["frozen_expert_digest"]
            for f, p in (("nodq", a.receipt_nodq), ("dq", a.receipt_dq))}
    torch.cuda.set_device(0)
    prep = prepare_qlora_training(plan["model"]["model"], QLoRASetup(**plan["selected"]["setup"]), device="cuda",
                                  revision=plan["model"].get("revision"))
    stacks = [m for m in prep.model.modules() if isinstance(m, ExpertsNbit)]
    os.makedirs(a.out, exist_ok=False)
    rec = {"stacks": len(stacks), "want": want}

    def dump(name, keep):
        tensors = {f"{i}.{n}": t.detach().contiguous().cpu() for i, m in enumerate(stacks)
                   for n, t in _stored_expert_tensors(m).items() if keep(n)}
        save_file(tensors, os.path.join(a.out, name))
        return len(tensors)

    rec["got_nodq"] = _expert_digest(prep.model)
    rec["nodq_ok"] = rec["got_nodq"] == want["nodq"]
    if rec["nodq_ok"]:                       # packed codes + fp32 absmax: fixture-nodq's base, and the codes of both
        rec["stacks_tensors"] = dump("stacks.safetensors", lambda n: True)
    rec["compressed"] = compress_expert_absmax_(prep.model)
    rec["got_dq"] = _expert_digest(prep.model)
    rec["dq_ok"] = rec["nodq_ok"] and rec["got_dq"] == want["dq"]
    if rec["dq_ok"]:                         # the double-quantized absmax payloads only
        rec["dq_tensors"] = dump("absmax_dq.safetensors", lambda n: n.endswith(_COMPRESSED_ABSMAX))
    with open(os.path.join(a.out, "fixture_base.json"), "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    print("FIXTURE_BASE " + json.dumps({k: rec[k] for k in ("stacks", "nodq_ok", "compressed", "dq_ok")}))
    return 0 if rec["dq_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
