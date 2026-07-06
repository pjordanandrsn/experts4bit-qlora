# e4b-serve: HTTP serving shim for experts4bit-qlora (see experts4bit_qlora/serve.py).
# Base matches the NAS's gpu-dev container byte-for-byte — driver compatibility on the target
# host is already proven. CUDA libs come from torch's bundled nvidia/* wheels (runtime image is
# for parity, not necessity); python3.11 is the tested interpreter.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.11 python3.11-venv python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

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
