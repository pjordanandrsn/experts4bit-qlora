# e4b-serve: HTTP serving shim for experts4bit-qlora (see experts4bit_qlora/serve.py).
# Base matches the NAS's gpu-dev container byte-for-byte — driver compatibility on the target
# host is already proven. CUDA libs come from torch's bundled nvidia/* wheels (runtime image is
# for parity, not necessity); python3.11 is the tested interpreter.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Ubuntu 22.04's own `python3.11` package is 3.11.0rc1 (Aug 2022) — it predates the
# sys.get_int_max_str_digits / set_int_max_str_digits backport. torch 2.6's dynamo
# polyfills decorate BOTH at import time (torch/_dynamo/polyfills/sys.py), so the first
# `import torch._dynamo` in the image raises AttributeError partway through registration.
# The failed module is evicted from sys.modules while the already-registered sys.intern
# rule survives, and every later import — e.g. transformers.integrations.deepgemm's
# `@torch._dynamo.allow_in_graph`, reached by loading any MoE model that touches
# transformers.integrations.moe — dies with the misleading
#   ValueError: Duplicate dispatch rule for <built-in function intern>
# while the serve container keeps reporting healthy. Install a real 3.11 from deadsnakes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-distutils python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

# Build-time guard: if the interpreter is ever an rc again, fail the IMAGE — not the
# first production model load half a symptom-chain later.
RUN python3.11 -c "import sys; sys.get_int_max_str_digits; sys.set_int_max_str_digits"

WORKDIR /app

# Layer the big pinned wheels before the (frequently-edited) package source so code iterations
# don't re-download torch.
COPY deploy/requirements.txt /app/deploy/requirements.txt
RUN python3.11 -m pip install --no-cache-dir -r /app/deploy/requirements.txt

COPY . /app
RUN python3.11 -m pip install --no-cache-dir --no-deps -e .

# Serving defaults; compose overrides per deployment. OFFLOAD_EXPERTS=1 is the shim's default
# too — restated here so a bare `docker run` is honest about what it does.
ENV OFFLOAD_EXPERTS=1 PREFETCH=1 E4B_PORT=8777
EXPOSE 8777

CMD ["python3.11", "-m", "experts4bit_qlora.serve"]
