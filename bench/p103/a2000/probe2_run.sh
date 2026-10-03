#!/bin/bash
trap "" HUP
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker; C=gdnk2-probe; R=/share/Container/scripts/gdnk2; ES=$1; GS=$2
L=$R/probe2.log; : > $L
say(){ echo "[$(date -u +%FT%TZ)] $*" >> $L; }
$D rm -f $C >/dev/null 2>&1
$D run -d --name $C --runtime nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel sleep 7200 >/dev/null
U=$($D exec $C nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d " ")
say "preflight utilization ${U:-?} %"
[ -n "$U" ] && [ "$U" -le 20 ] || { say "ABORT: GPU busy"; echo GDNK2_DONE >> $L; $D rm -f $C >/dev/null 2>&1; exit 1; }
$D exec $C bash -c "apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y git >/dev/null 2>&1; echo torch==2.8.0 > /root/c.txt; git clone -q https://github.com/pjordanandrsn/experts4bit-qlora.git /root/e4b && git -C /root/e4b checkout -q $ES && cd /root/e4b && pip install -q -c /root/c.txt -e . 'grouped-nf4-gemm @ git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GS' transformers==5.17.0 bitsandbytes==0.50.2 accelerate safetensors pytest > /root/pip.log 2>&1; python -c 'import torch, transformers, pytest, experts4bit_qlora; assert torch.cuda.is_available() and torch.__version__.startswith(\"2.8.0\"); print(\"READY\", torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))' || echo NOT_READY" >> $L 2>&1
grep -q "^READY" $L || { say "ABORT: environment not ready"; echo GDNK2_DONE >> $L; $D rm -f $C >/dev/null 2>&1; exit 1; }
$D cp $R/probe2.py $C:/root/probe2.py
arm(){ say "ARM $1"; $D exec $C bash -c "cd /root && python probe2.py 2>&1 | grep -E 'VARIANT|POOL|Error|error' | tail -12" >> $L 2>&1; }
arm T
$D exec $C bash -c "pip install -q -c /root/c.txt flash-linear-attention==0.5.2 > /root/pip_fla.log 2>&1; python -c 'import fla; print(\"FLA\", fla.__version__)'" >> $L 2>&1
arm F
$D exec $C bash -c "pip install -q -c /root/c.txt --no-build-isolation causal-conv1d==1.7.0 > /root/pip_cc.log 2>&1; python -c 'import causal_conv1d; print(\"CC\", causal_conv1d.__version__)'" >> $L 2>&1
arm FC
$D rm -f $C >/dev/null 2>&1
say done; echo GDNK2_DONE >> $L
