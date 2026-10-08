# DQ7 diagnosis: the full-logit workspace line misses a live fp32 tensor

This follows the unchanged **VOID** reading in [RESULTS-dq7.md](RESULTS-dq7.md). It identifies a named tensor term;
it neither changes the historical estimator nor fits an activation or reserve coefficient. A full-model reread,
cache intervention and replacement calibration require a separate reviewed registration.

## Full-logit backward: twelve bytes per logit

The exact DQ7 environment (torch 2.8.0+cu128, transformers 5.18.0) runs `ForCausalLMLoss` on a frozen bf16 head,
with only `.loss` surviving the forward expression, matching the shared executor. A transparent TorchDispatchMode
records shapes, dtypes and distinct storage addresses at `aten._log_softmax_backward_data.default`, after the
operator creates its output and before either input can be released. It stores metadata only, with no persistent
tensor references. This observes three different fp32 `[tokens, vocab]` storages simultaneously:

1. Saved log-softmax forward output.
2. NLL backward output, which is the incoming log-softmax gradient.
3. New log-softmax backward output, which is the logits gradient.

Their total is **3 × 4 × tokens × vocab = 12 bytes per logit**. The current dense estimate prices 10 bytes per
logit (bf16 logits, fp32 upcast and fp32 gradient); that forward-oriented accounting misses the third fp32 buffer
at backward. The correction would be **2 × tokens × vocab** additional bytes in the full-logit loss workspace,
with the activation coefficient unchanged. This term comes from tensor shapes and liveness, not a fit to DQ7 residuals.

The [CPU receipt](diagnosis/loss-cpu.json) proves three distinct tensors at vocabulary 1024 and sequences 8/16.
The [CUDA receipt](diagnosis/loss-cuda.json) proves the same operator on the A2000 with Llama's vocabulary 128256:

| tokens | each distinct fp32 tensor bytes | three live tensor bytes | current ten-byte allowance | named missing bytes |
|---:|---:|---:|---:|---:|
| 512 | 262668288 | 788004864 | 656670720 | 131334144 |
| 1024 | 525336576 | 1576009728 | 1313341440 | 262668288 |

Both cases ran sequentially in one process, so the receipt's global baseline/peak fields are descriptive and may
retain prior process allocations. The decisive evidence is the distinct simultaneous tensor storage count and its
exact bytes. No timing, whole-model capacity, allocator reserve or general family guarantee is derived from the
component proof. Frozen-head forward/backward and finite hidden gradients are checked. The instrument was run in
an isolated existing environment; the A2000 holder explicitly authorized simultaneous correctness use. No shared
environment was changed, and the census process and child resource claim ended before this publication.

The executed [source](diagnosis/dense_loss_census.executed.py.txt) is byte-identical to its recorded source hash
`fb7d2acefaba8b63418d9e724d68d543b6a4d7ad299ac1a4daafea2f80bae2cd` (its original basename was
`dense-loss-census.py`). The runnable [instrument](diagnosis/dense_loss_census.py) has only whitespace repairs; its AST is identical to the executed source. Transformers' loss_utils.py hash is
`83db16b24ce5c3a0642097624aa9a0ae7eb72445c41d8b4d5059a129862fdbbe`. Public source supports the mechanism:
[PyTorch 2.8 cross entropy](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/LossNLL.cpp)
composes log-softmax with NLL loss; [backward metadata](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/SoftMax.cpp)
allocates its full-shaped output. The receipts supply the runtime storage observation.

## Arithmetic against the failed Llama readings, not a new reading

For Llama-3.1-8B's vocabulary 128256, the source-derived extra term is 525336576 bytes at 2048 tokens and
1,050,673,152 bytes at 4096. If added to the unchanged DQ7 allocator estimates, it would leave a positive estimate
surplus of 327844355 bytes (resident 2048), 328368643 (streamed 2048), 388502022 (resident 4096) and 389026310
(streamed 4096). Those are **post-read arithmetic**, not new GPU observations, a licensed bracket, a new out-of-sample
pass or calibration. Their nonzero residual is retained; neither the correction nor the activation coefficient is
adjusted to make the residual vanish. Placement-independent errors identify a common loss workspace term; this
component observation supplies its concrete shape and live dtype.

## Streamed reserved memory: cause still unknown

DQ7 records load-phase **allocated peak**, driver use before training, and training-phase allocated/reserved/driver
peaks. It does **not** record load-phase reserved memory, post-loader cache state or a counterfactual cache clear.
A peak-stat reset does not free cached allocations. These receipts therefore cannot establish how much of the
streamed driver deficit is loader cache versus training fragmentation. The load-cache hypothesis is plausible and
remains unmeasured. It is not absorbed into a reserve fit.

Before a reread, register phase allocated/reserved/current/peak counters at loader completion, setup completion,
optimizer creation, forward/loss, backward, clip and optimizer update, plus loader-cache bytes. A matched baseline
and `empty_cache` intervention after loading/setup must use separate fresh processes, identical tensor/config/row
hashes, and validate unchanged loss/gradients/update/export and stream engagement. Keep the old receipts, name any
cache intervention, and select an anchor rung whose **actual plan including headroom** fits. The Qwen3-32B resident
4096 anchor cannot be silently forced through by dropping admission headroom.
