"""DQ9: paired fresh-process readings through actual execute, with phase/cache telemetry."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path

from dq7_arm import make_plan
from dq7_subject import CONFIG_HASHES
from dq9_phase import PhaseCensus


def read(directory, subject, placement, seq, cache_mode, out):
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from loggetta import execute
    from loggetta.backends import dense_loader, dense_train
    from loggetta.data import TrainingData, prepare_data

    output = Path(out)
    config_hash = hashlib.sha256((Path(directory)/"config.json").read_bytes()).hexdigest()
    if config_hash != CONFIG_HASHES[subject]:
        raise ValueError("config differs from registration")
    datafile = output.parent/f"{subject}-{seq}.jsonl"
    text = " ".join(f"w{3+(i % 509)}" for i in range(2*seq+16))
    datafile.write_text(json.dumps({"text": text})+"\n")
    spec = TrainingData(str(datafile), format="text", packing="concat", loss="all")
    with prepare_data(AutoTokenizer.from_pretrained(directory), 2, seq, spec, seed=731) as prepared:
        assert prepared.blocks.shape == (2, seq) and np.all(prepared.blocks >= 3)
        rows_sha = hashlib.sha256(prepared.blocks.tobytes()).hexdigest()
    p = make_plan(directory, placement, seq, spec.to_dict())
    census = PhaseCensus(torch, cache_mode)
    with census.installed(dense_loader, dense_train):
        result = execute(p, seed=731, out_dir=str(output.parent/"execution"),
                         adapter_dir=str(output.parent.parent/"adapters"/output.stem))
    assert result["data"]["tokens"] == 2*seq and result["data"]["blocks"] == 2
    assert result["data"]["token_stream_sha256"] == rows_sha
    phases = census.report()
    assert phases["cache_calls"] == int(cache_mode == "empty")
    assert census.step == 2
    result["dq9"] = {"subject": subject, "synthetic": True, "pretrained": False, "seq": seq,
                     "placement": placement, "cache_mode": cache_mode, "allocator": "default",
                     "config_sha256": config_hash, "row_tokens_sha256": rows_sha,
                     "checkpoint_manifest_sha256": hashlib.sha256((Path(directory)/"dq7-subject.json").read_bytes()).hexdigest(),
                     "runtime": {k: metadata.version(k) for k in ("torch", "bitsandbytes", "transformers", "peft")},
                     "real_tokens_per_row": seq, "padded_tokens": 0,
                     "loggetta_sha": os.environ["LOGGETTA_SHA"], "e4b_sha": os.environ["E4B_SHA"]}
    result["phases"] = phases
    with output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--subject", choices=tuple(CONFIG_HASHES), required=True)
    p.add_argument("--placement", choices=("device", "stream"), required=True)
    p.add_argument("--seq", type=int, required=True)
    p.add_argument("--cache-mode", choices=("baseline", "empty"), required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    r = read(a.checkpoint, a.subject, a.placement, a.seq, a.cache_mode, a.out)
    print(json.dumps({"out": a.out, "status": r["status"]}))
