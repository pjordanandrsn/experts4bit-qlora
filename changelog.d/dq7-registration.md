### Register the Loggetta dense executor proof and out-of-sample capacity reading

DQ7 specifies a deterministic CUDA proof before architecture-only capacity measurements on Qwen3-14B and
Llama-3.1-8B, with Qwen3-32B as the in-sample DQ4 anchor. It checks the actual Loggetta plan/executor and reports
itemized allocator estimates, peaks and residuals. CPU instrument tests cover refusal, setup and row mismatches,
underestimates and anchor misses. No new measurement, general capacity bound or planner calibration is claimed.
