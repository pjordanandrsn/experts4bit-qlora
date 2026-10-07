### The routed-expert combine's backward keeps 0.4 GB fewer fp32 temporaries, with the same gradient bytes (TC1 amendment 57)

- TC1 amendment 57's training-phase census put 1.07 GB of e4b's packed-row training peak in `_ScatterCombine`'s backward. The backward
  materialised the incoming gradient's expanded `[tokens × top-k, hidden]` fp32 image before indexing it, and kept a second fp32 product
  alive.
- It now reads the image by index (`g[order // k]`), and computes the weight gradient before scaling the down gradient in place. Both
  gradients are the same bytes as before, and the same as autograd's composite (`tests/test_moe_keep.py`).
- At a packed row's shape (4,096 tokens, top-8, hidden 2,048) the backward's peak falls from 1.208 GB to 0.805 GB on CUDA. That is an
  allocator measurement on an RTX A2000; the training-step peak is for a TC1 box to read.
