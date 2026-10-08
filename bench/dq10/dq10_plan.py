"""The registered policy is selected before actual planner admission; no plan overlay."""
from __future__ import annotations

import json
from pathlib import Path

SHAPES = tuple((subject, placement, seq) for subject in ("mistral7b_v03", "smollm3_3b")
               for seq in (512, 2048, 4096) for placement in ("device", "stream"))


def make_plan(directory, placement, seq, data=None, *, policy=True):
    from loggetta import Constraints, Workload, plan
    from loggetta.backends import dense
    from loggetta.hardware import probe

    topology = dense.describe(str(directory))
    workload = Workload(seq_len=seq, steps=2, learning_rate=2e-4, lr_schedule="constant", data=data)
    constraints = Constraints(allow_development_executor=True,
                              dense_reserve_policy="dq10" if policy else None,
                              allocator_profile="default" if policy else None,
                              fixed={"base": "nf4", "placement": placement, "r": 16, "alpha": 32,
                                     "adapter_dtype": "fp32", "targets": ["attn_in", "attn_out", "mlp_in", "mlp_out"],
                                     "attn_impl": "sdpa", "loss_chunk": 0})
    result = plan(topology, probe(), workload, constraints, backends=(dense,))
    if result.status != "feasible":
        raise ValueError("registered setup is refused: " + result.render())
    return result


def admit(configs, out):
    root = Path(out).parent / "admission-configs"
    root.mkdir(exist_ok=False)
    rows = []
    for subject, placement, seq in SHAPES:
        directory = root / subject
        directory.mkdir(exist_ok=True)
        (directory / "config.json").write_bytes((Path(configs) / (subject + ".json")).read_bytes())
        plan = make_plan(directory, placement, seq)
        rows.append({"subject": subject, "placement": placement, "seq": seq, "plan": plan.to_dict()})
    with Path(out).open("x") as stream:
        json.dump({"schema": "dq10-admission/1", "status": "PASS", "arms": rows}, stream, indent=2)
        stream.write("\n")
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("configs")
    parser.add_argument("out")
    args = parser.parse_args()
    print(json.dumps({"admitted_shapes": len(admit(args.configs, args.out))}))
