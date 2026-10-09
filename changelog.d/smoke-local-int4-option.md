### CUDA serving smoke can test local expert-int4 source reads

`--int4-source local` selects each int4 cell's synthetic checkpoint directory
directly and records the source selection. The existing offline-Hub fixture
mode remains the default; both modes use local random weights without downloads.
