### D0 registered: how much of a trained expert LoRA delta survives repacking into a serving format

`bench/d0/PREREG-d0.md` registers a CPU-only premise read before any number. On two stored OLMoE-1B-7B fixtures
(Loggetta defaults, one with the double-quantized and one with the fp32 expert absmax), it measures per expert how much
of the trained delta ΔW = s·B·A survives a repack T(S + ΔW) into NF4, int4-b32 and MXFP4, from the trained base and
from raw W, and the effect on held-out loss. The trained base comes from the training card, digest-verified. It
decides nothing by itself: it is the premise for choosing between a merged, re-quantized expert arena and a fused
low-rank epilogue. No default, gate or claim moves.
