### Serve estimate: the graph pools it leaves unpriced now cite what SV1 measured

- `estimate_serve_footprint` still lists the decode graphs' pools and the first-chunk prefill graph's pool as not
  modelled. Its notes now carry lane SV1's NF4 measurements (`bench/sv1/RESULTS-sv1.md`):
  - decode graphs: +60 MiB allocated (OLMoE-1B-7B, 16 sequences);
  - the prefill graph: +0.24 GiB (OLMoE-1B-7B) and +0.57 GiB (Qwen3-30B-A3B);
  - beside SC2b's +3.3 GiB for the int4 stack.
- No number is priced from two points.
