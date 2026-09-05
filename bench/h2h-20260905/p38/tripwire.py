import importlib.metadata as md, torch, transformers, bitsandbytes, peft
import unsloth, unsloth_zoo
from unsloth import FastLanguageModel
from unsloth_zoo.temporary_patches.common import is_transformers_v5_moe_quantization_available
from unsloth_zoo.temporary_patches.moe_utils_bnb4bit import forward_moe_backend_bnb4bit, _is_bnb4bit_param
from unsloth_zoo.temporary_patches.moe_utils import select_moe_backend, _should_use_separated_lora
assert is_transformers_v5_moe_quantization_available(), "the transformers-v5 4-bit MoE path is NOT available in this environment: Unsloth would not train 4-bit MoE here"
tri = None
try:
    import triton; tri = triton.__version__
except Exception: pass
print("p38 tripwire OK (unsloth):", unsloth.__version__, "zoo", unsloth_zoo.__version__, "torch", torch.__version__, "triton", tri, "transformers", transformers.__version__,
      "bnb", bitsandbytes.__version__, "peft", peft.__version__, "moe_backend", select_moe_backend(), "separated_lora", _should_use_separated_lora())
open("/root/p38/versions.txt", "a").write(f"unsloth {unsloth.__version__}\nunsloth_zoo {unsloth_zoo.__version__}\ntorch(unsloth) {torch.__version__}\ntriton(unsloth) {tri}\ntransformers(unsloth) {transformers.__version__}\nbitsandbytes(unsloth) {bitsandbytes.__version__}\npeft {peft.__version__}\nmoe_backend {select_moe_backend()}\n")
