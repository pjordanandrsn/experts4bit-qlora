#!/usr/bin/env python3
"""Generate the gen1 arm configs (prereg_gen1_hunt.json arm set) into sys.argv[1]."""
import pathlib
import sys

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

BASE = """base_model: allenai/OLMoE-1B-7B-0924
trust_remote_code: true
plugins: [{plugin}]
expert_offload: {offload}
load_in_4bit: true
quantize_moe_experts: true
adapter: qlora
lora_r: 8
lora_alpha: 16
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
datasets: [{{path: tatsu-lab/alpaca, type: alpaca}}]
dataset_prepared_path: /root/dq/prep
dataset_processes: 16
val_set_size: 0.0
output_dir: /root/dq/outputs/{name}
sequence_len: 64
sample_packing: true
gradient_accumulation_steps: 1
micro_batch_size: 1
max_steps: 40
seed: 42
optimizer: adamw_torch_fused
learning_rate: 0.0001
bf16: auto
gradient_checkpointing: true
gradient_checkpointing_kwargs: {{use_reentrant: false}}
logging_steps: 1
save_strategy: "no"
wandb_mode: offline
expert_offload_staging: whole_layer
{extra}"""

PUB = "axolotl.integrations.expert_offload.ExpertOffloadPlugin"
PRIV = "e4b_ssdtier.axolotl_plugin.FusedOffloadPlugin"


def w(name, plugin, offload, extra):
    (OUT / f"{name}.yaml").write_text(
        BASE.format(name=name, plugin=plugin, offload=offload, extra=extra)
    )


w("warmup", PUB, "true", "expert_offload_store: ram\n")
for suffix in ("a", "b"):
    w(f"v_{suffix}", PUB, "false", "")
    w(f"r_{suffix}", PUB, "true", "expert_offload_store: ram\n")
for fv in (0.0, 0.25, 0.5, 0.75, 1.0):
    name = f"fv{int(fv * 100):03d}"
    w(
        name, PRIV, "true",
        f"fused_ram_fraction: {1.0 - fv}\nfused_vram_fraction: {fv}\n"
        f"fused_policy: interleaved\nfused_prefetch: false\n"
        f"fused_store_dir: /root/dq/fstore_{name}\n",
    )
print(f"wrote {len(list(OUT.glob('*.yaml')))} configs to {OUT}")
