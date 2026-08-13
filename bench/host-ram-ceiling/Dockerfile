FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
# Triton JITs its kernels through a host C compiler; the -runtime image ships
# without one. Adding gcc here rather than at container start keeps every capped
# run identical and offline.
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
 && rm -rf /var/lib/apt/lists/*
ENV CC=/usr/bin/gcc CXX=/usr/bin/g++
