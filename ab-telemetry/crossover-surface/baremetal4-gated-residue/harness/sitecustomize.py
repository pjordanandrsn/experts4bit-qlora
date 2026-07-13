# cuDNN SDPA crashes at step 0 on cu130-class torch ("No valid execution plans built");
# ride PYTHONPATH so accelerate's child process inherits the disable too.
try:
    import torch

    torch.backends.cuda.enable_cudnn_sdp(False)
except Exception:
    pass
