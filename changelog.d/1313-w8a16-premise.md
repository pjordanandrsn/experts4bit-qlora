### A W8A16 quality premise on the B = 1 dense path (#1313)

`bench/i8dense/` adds a one-box premise: does int8 weight-only rounding of every attention projection and the
lm_head keep P125's quality gate (argmax agreement >= 0.95, K8's +0.05 ppl budget)? At B = 1 the dense GEMM class is
the largest single-stream class, and P125's int4 version failed on argmax. The premise runs P125's subject and quality
instrument byte for byte, with one arm added: A's build with those weights replaced by their per-output-row int8
round trip. GO starts the kernel work; PARK closes the lever with the number. It is a premise, never a reading.
