### Tiny CUDA backward and alternate-family correctness smoke

Add an offline smoke for mixed-tier adapter gradients, pinned-host checkpoint
backward, and gpt-oss bias / DeepSeek-V4 clamp residency. Every cell requires
real execution and path engagement, then rejects a planted path mutation;
skips, missing backends, and undetected mutations fail the controller.
