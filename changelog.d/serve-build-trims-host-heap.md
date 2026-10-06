### `serve_paged.build_engine` hands its freed host heap back

- The build churns through host buffers it then frees, among them the hybrid tier's setup tier and the stacks'
  one-shot reads. glibc kept those blocks resident for the life of the server.
- `build_engine` now calls `engines.host_heap.release_freed_host_heap()` once it is built, and reports
  `host_heap_trimmed` in its `info`.
- Measured on OLMoE-1B-7B (RTX A2000 host, NF4, all-VRAM):
  - anonymous host memory after the build fell from 0.90 GB to 0.56 GB;
  - a second trim afterwards returns nothing;
  - the loader alone leaves ~4 MB, so the build is where the trim belongs.
- On Qwen3-30B-A3B (lane SV2, a 48-core host) the NF4 build held 3.21 GB after load. The int4 builds, whose levers
  already trimmed (#1182), held 1.68 GB.
