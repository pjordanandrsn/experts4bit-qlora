"""experts4bit-qlora — QLoRA fine-tuning of fused 4-bit Mixture-of-Experts on a single small GPU.

``Experts4bit`` resolves to the upstream bitsandbytes class once it ships in a release
(bitsandbytes#1965); until then it falls back to a vendored copy, so this package works on a
stock ``pip install bitsandbytes`` today. When upstream releases it, bump the bitsandbytes floor
and delete ``experts4bit_qlora/_vendor/`` — no API change for callers.
"""

try:
    from bitsandbytes.nn import (
        Experts4bit,
    )  # upstream (bitsandbytes#1965), once released
except ImportError:  # stock bitsandbytes: use the vendored copy
    from ._vendor.experts import Experts4bit

from .lora import ExpertsLoRA, LoRALinear, add_attention_lora

__all__ = ["Experts4bit", "ExpertsLoRA", "LoRALinear", "add_attention_lora"]
__version__ = "0.1.0"
