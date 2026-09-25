#!/bin/bash
# P70 rehearsal on the NAS A2000 (NOT a reading): install the pinned deps into /w/pydeps once, then run the lane's
# own runner with the rehearsal knobs (the runner marks the run REHEARSAL). Tree: the uncommitted P70 branch on
# e4b abb198d3e1261a67607b412bebeb4a2a5d66cade; grouped-nf4-gemm 0.33.0 (5ca1897, the registered cut) from PyPI.
set -u
export PIP_CACHE_DIR=/w/pipcache HF_HOME=/w/hf
if [ ! -f /w/pydeps/.done ]; then
  python -m pip install -q --no-input --target /w/pydeps "transformers==5.16.1" datasets sentencepiece tiktoken safetensors "huggingface_hub>=0.23" psutil > /w/logs/pip1.log 2>&1 || { echo PIPFAIL1; tail -5 /w/logs/pip1.log; exit 9; }
  python -m pip install -q --no-input --no-deps --target /w/pydeps "bitsandbytes==0.50.1" accelerate "grouped-nf4-gemm==0.33.0" > /w/logs/pip2.log 2>&1 || { echo PIPFAIL2; tail -5 /w/logs/pip2.log; exit 9; }
  touch /w/pydeps/.done
fi
export PYTHONPATH=/w/pydeps:/w/src/e4b
cd /root/p70
export P70_RUN_ID=reh1 P70_RUN_NONCE=5feb2901a687aa1b5267ff5e10d4a0c38285961f114c34518775515bfccc0029 P70_DEADLINE_EPOCH=1790324808 P70_INSTANCE_ID=nas-a2000 E4B_SHA=abb198d3e1261a67607b412bebeb4a2a5d66cade GNF4_SHA=5ca1897585f9f456f99ea504b2a1be0ea91db496
export P70_MODEL=allenai/OLMoE-1B-7B-0924 P70_REVISION=6d84c48581ece794365f2b8e9cfb043c68ade9c5 P70_GPU_CLASS=A2000 P70_MIN_DISK_GB=50
export P70_CALIB_NSEQ=8 P70_HESSIAN_BUDGET_GB=2 P70_BUILD_PPL_STEPS=64 P70_ROWS=4 P70_SKIP_INSTALL=1 P70_REHEARSAL=1 P70_FIRST_CHUNK_S=3600
bash p70_run.sh > outer.log 2>&1
echo "RUNNER_RC=$?"
