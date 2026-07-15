"""Post-load sanity check: are a loaded model's fused MoE experts actually low-bit?

The failure this guards against is [bitsandbytes#1849]: a stock
``AutoModelForCausalLM.from_pretrained(..., quantization_config=BitsAndBytesConfig(load_in_4bit=True))``
only replaces ``nn.Linear`` modules, so it silently skips a fused-MoE's 3-D expert
``nn.Parameter`` stacks — the overwhelming majority of the weights stay in full precision and OOM
the card. :func:`experts4bit_qlora.loader.load_moe_4bit_streaming` quantizes exactly those stacks;
this helper lets a caller *confirm* it after the fact. It is the read-only, any-model counterpart
to the loader's own zero-quantized-layer guard (``loader.py``): that one fires when e4b's own path
finds nothing to quantize; this one inspects any model, including one loaded the stock way.

[bitsandbytes#1849]: https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1849
"""

from . import Experts4bit, ExpertsNbit


def verify_moe_4bit(model, *, strict=False):
    """Classify every fused-expert stack in ``model`` as low-bit-quantized or still high-precision.

    Returns a report ``dict``::

        {
          "quantized":   [{"module": str, "quant_type": str}, ...],
          "unquantized": [{"module": str, "dtype": str, "shape": tuple}, ...],
          "n_quantized": int,
          "n_unquantized": int,
        }

    *Quantized* = an :class:`~experts4bit_qlora.ExpertsNbit` (or its 4-bit :class:`Experts4bit`
    subclass) stack — what e4b's loader installs, either directly or as the ``.base`` of an
    ``ExpertsLoRA`` wrapper (the wrapper's base is what gets counted, so wrapping does not
    double-count). *Unquantized* = a transformers fused-expert module (class name containing
    ``Experts`` — ``Qwen3MoeExperts``, ``OlmoeExperts``, the Gemma-4 / GraniteMoe equivalents)
    still holding a floating-point 3-D ``nn.Parameter``: the bitsandbytes#1849 silent-skip.

    With ``strict=True``, raises :class:`RuntimeError` naming the count and the fix when any stack is
    still high-precision — so ``verify_moe_4bit(model, strict=True)`` is a one-line assertion that a
    load actually quantized the experts. Detection is a heuristic keyed on transformers' fused-expert
    module naming plus a 3-D float parameter; a newly added MoE family may need its module class
    recognized here.
    """
    quantized = []
    unquantized = []
    for name, module in model.named_modules():
        if isinstance(module, (Experts4bit, ExpertsNbit)):
            quantized.append({"module": name, "quant_type": getattr(module, "quant_type", "?")})
            continue
        # A transformers fused-expert module e4b never replaced holds its experts as one fused 3-D
        # float Parameter per stack. Skip our own ExpertsLoRA wrapper — its quantized `.base` is the
        # Experts4bit/ExpertsNbit counted above when named_modules yields it.
        cls = type(module).__name__
        if cls == "ExpertsLoRA" or "Experts" not in cls:
            continue
        for pname, param in module.named_parameters(recurse=False):
            if param is not None and param.dim() == 3 and param.is_floating_point():
                unquantized.append(
                    {
                        "module": f"{name}.{pname}" if name else pname,
                        "dtype": str(param.dtype).replace("torch.", ""),
                        "shape": tuple(param.shape),
                    }
                )
                break

    report = {
        "quantized": quantized,
        "unquantized": unquantized,
        "n_quantized": len(quantized),
        "n_unquantized": len(unquantized),
    }
    if strict and unquantized:
        shown = ", ".join(u["module"] for u in unquantized[:4])
        more = "" if len(unquantized) <= 4 else f" (+{len(unquantized) - 4} more)"
        raise RuntimeError(
            f"{len(unquantized)} fused-expert stack(s) are still high-precision "
            f"({len(quantized)} quantized): {shown}{more}. A stock "
            "`from_pretrained(..., load_in_4bit=True)` only quantizes nn.Linear and skips the fused "
            "experts (bitsandbytes#1849) — load with "
            "`experts4bit_qlora.loader.load_moe_4bit_streaming(...)` instead."
        )
    return report
