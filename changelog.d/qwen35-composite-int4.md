### Qwen3.5 text serving reads expert-int4 sources from native composite checkpoints

The native Qwen3.5-MoE convention now covers the served text configuration.
For a text-only tree, source planning maps the declared composite text root,
records unused tensors outside that root, and excludes auxiliary expert stacks
by structure. Unknown text tensors, missing weights, and mixed plain/composite
text roots refuse before repacking. Native gate/up orientation is unchanged.
