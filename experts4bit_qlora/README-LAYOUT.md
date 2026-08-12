# Package layout

Three axes were growing as flat siblings in one namespace, so there was no visible
shape telling a contributor where a new file goes. That is the problem this split
solves — not code quality.

| directory | holds | add here when… |
|---|---|---|
| `arch/` | architecture seams — keymaps, conventions, per-family expert layouts (`deepseek_v4`, `glm5`, `axk1`, `glimmer*`, `mixtral`, `gptoss`, `moe_*`) | you are teaching e4b a new **model family**: where its experts live on disk and how its keys map |
| `formats/` | on-disk quantization formats and readers (`awq`, `gptq`, `nvfp4`, `mxfp4`, `compressed_int`, `fp8_blocks`, `dense_disk`) | you are teaching e4b to **read new bytes** — a released checkpoint's quantization scheme |
| `engines/` | execution paths and residency policy (`pipelined`, `cold_engine`, `hot_residency`, `offload`, `dense_offload`, `nvme_experts`, `nvme_train`, `fast`, `batched`, `capture`, `kv_cache`, `speculative`, `expert_profile`) | you are changing **how compute or memory is scheduled**, not what is read |

Staying at the top level: `loader`, `lora`, `train`, `infer`, `serve`, `verify`, `util` —
the entry points and the primitives everything else composes.

**A new family usually touches two of these**, and that is the point: the storage
convention goes in `arch/`, the byte reader in `formats/`, and neither belongs in an
engine. If a change wants to live in all three, it is probably two changes.

Import paths: the public API is `experts4bit_qlora.__all__` and did not move. Old
submodule paths (`experts4bit_qlora.awq`) still resolve via aliases installed in
`__init__.py`, so existing code keeps working.
