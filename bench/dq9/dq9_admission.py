"""Actual current planner admission for every registered shape, before full checkpoint generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dq7_arm import make_plan
from dq9_reduce import SUBJECT_ARMS


def admit(configs, out):
    root = Path(out).parent / "admission-configs"
    root.mkdir(exist_ok=False)
    rows = []
    for subject, placement, seq in SUBJECT_ARMS:
        directory = root/subject
        directory.mkdir(exist_ok=True)
        (directory/"config.json").write_bytes((Path(configs)/(subject+".json")).read_bytes())
        plan = make_plan(directory, placement, seq)
        rows.append({"subject": subject, "placement": placement, "seq": seq, "plan": plan.to_dict()})
    with Path(out).open("x") as f:
        json.dump({"schema": "dq9-admission/1", "status": "PASS", "arms": rows}, f, indent=2)
        f.write("\n")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("configs")
    p.add_argument("out")
    a = p.parse_args()
    print(json.dumps({"admitted_shapes": len(admit(a.configs, a.out))}))
