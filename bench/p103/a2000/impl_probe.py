"""Which Gated DeltaNet implementation did transformers resolve at import (torch reference, fla, causal_conv1d)?"""
import inspect
import transformers.models.qwen3_5.modeling_qwen3_5 as m
for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update"):
    fn = getattr(m, name)
    impl = inspect.getclosurevars(fn).nonlocals.get("implementation", None)
    where = getattr(impl, "__module__", "?") if impl is not None else "?"
    print("IMPL", name, "->", where, getattr(impl, "__name__", impl))
