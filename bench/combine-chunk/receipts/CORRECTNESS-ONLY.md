# Correctness and memory only

`combine_chunk_a2000.json` was produced on an RTX A2000, a correctness-only testbed. Its byte-equality results and its allocator peak
bytes are evidence; nothing in it is a timing. What the row chunks cost or save in a training step is a registered TC1 lane's to read.
