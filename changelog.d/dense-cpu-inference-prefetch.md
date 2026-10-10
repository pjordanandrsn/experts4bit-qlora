### Fix streamed dense adapter inference on CPU

Streamed dense adapters reloaded for CPU inference could attempt to create a CUDA
prefetch stream and crash. Non-CUDA inference now uses the existing synchronous
staging path. The CUDA prefetch branch is unchanged.
