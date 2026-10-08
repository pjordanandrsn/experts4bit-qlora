### Diagnose DQ7 full-logit loss workspace

Preserve a same-version CPU/CUDA operator census: three distinct fp32 full-logit tensors overlap in log-softmax backward, giving twelve bytes per logit rather than the current ten-byte allowance. This names a missing term without refitting coefficients. Load-phase reserved memory was not recorded, so the streamed cache hypothesis remains unmeasured; no reread or calibration is implied.
