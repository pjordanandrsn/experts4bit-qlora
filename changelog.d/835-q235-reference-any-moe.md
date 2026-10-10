### Changed

- Q235's layer-streamed bf16 reference (`bench/q235/instrument/layer_stream_ref.py`) now reads any MoE checkpoint through
  transformers' own conversion mapping, the one `from_pretrained` uses, never through this package's MoE conventions, so it stays
  independent of what it checks. Mixtral (`block_sparse_moe` w1/w3/w2) loads as well as Qwen3, whose path is byte-identical to
  before. `mk_slice.py`, which builds the depth-truncated checkpoint view gate 1 runs on, is now in the repository (#835).
