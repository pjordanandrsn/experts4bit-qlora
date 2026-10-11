# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Q235 Amendment 3: e4b's footprint estimate for the box as registered -- shape A as run (8 rows right-padded to the longest,
2,648 positions in one micro-batch), r 16 / alpha 16, NF4 attention, experts in pinned host RAM, the grouped NF4 kernel, AdamW8bit."""
import json
import subprocess
import sys

from transformers import AutoConfig

from experts4bit_qlora.arch.topology import describe_moe
from experts4bit_qlora.recipe import QLoRASetup, estimate_qlora_footprint

cfg = AutoConfig.from_pretrained(sys.argv[1])
topo = describe_moe(cfg)
GiB = 2 ** 30
rows = {}
for dt in ("bf16", "fp32"):
    s = QLoRASetup(r=16, alpha=16, adapter_dtype=dt, attn_4bit=True, expert_residency="host", pin=True, expert_kernel="grouped_nf4")
    fp = estimate_qlora_footprint(topo, s, tokens_per_microbatch=2648, optimizer="adamw_8bit")
    rows[dt] = {"device_gib": round(fp.device_bytes / GiB, 2), "host_pinned_gib": round(fp.host_bytes / GiB, 2),
                "link_gib_per_microbatch": round(fp.total("link") / GiB, 2), "refusals": list(fp.refusals),
                "items": {i.name: round(i.bytes / GiB, 3) for i in fp.items if i.where == "device"}}
e4b = subprocess.run(["git", "-C", sys.argv[2], "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print(json.dumps({"schema": "q235-capacity/1", "e4b": e4b, "layers": cfg.num_hidden_layers, "tokens_per_microbatch": 2648,
                  "setup": "r 16 / alpha 16, NF4 attention, experts in pinned host RAM, grouped_nf4, AdamW8bit", "rows": rows}, indent=1))
