#!/bin/bash
trap "" HUP
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker; C=gdnk-probe; R=/share/Container/scripts/gdnk; ES=$1; GS=$2
L=$R/probe.log; : > $L
say(){ echo "[$(date -u +%FT%TZ)] $*" >> $L; }
$D rm -f $C >/dev/null 2>&1
$D run -d --name $C --runtime nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel sleep 7200 >/dev/null
say "preflight"; $D exec $C nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader >> $L 2>&1
U=$($D exec $C nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d " ")
[ -n "$U" ] && [ "$U" -le 20 ] || { say "ABORT: GPU busy (utilization ${U:-unknown} %) -- another session may be timing"; echo GDNK_DONE >> $L; $D rm -f $C >/dev/null 2>&1; exit 1; }
$D exec $C bash -c "apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y git >/dev/null 2>&1; echo torch==2.8.0 > /root/c.txt; git clone -q https://github.com/pjordanandrsn/experts4bit-qlora.git /root/e4b && git -C /root/e4b checkout -q $ES && cd /root/e4b && pip install -q -c /root/c.txt -e . 'grouped-nf4-gemm @ git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GS' transformers==5.17.0 bitsandbytes==0.50.2 accelerate safetensors pytest > /root/pip.log 2>&1; python -c 'import torch, transformers, pytest, experts4bit_qlora; assert torch.cuda.is_available() and torch.__version__.startswith(\"2.8.0\"); print(\"READY\", torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))' || echo NOT_READY" >> $L 2>&1
grep -q "^READY" $L || { say "ABORT: environment not ready"; echo GDNK_DONE >> $L; $D rm -f $C >/dev/null 2>&1; exit 1; }
for f in check_multibucket.py impl_probe.py compare.py; do $D cp $R/$f $C:/root/$f; done
arm(){ local A=$1
  say "ARM $A"
  $D exec $C bash -c "cd /root && python impl_probe.py 2>&1 | grep IMPL" >> $L 2>&1
  $D exec $C bash -c "cd /root/e4b && python -m pytest tests/test_linear_state_graph_gpu.py -q -p no:cacheprovider 2>&1 | tail -2" >> $L 2>&1
  $D exec $C bash -c "cd /root && OUT=/root/out_$A.pt python check_multibucket.py 2>&1 | grep -E 'STEPS|SAVED|replay !=|Error|error' | tail -8" >> $L 2>&1
}
arm T
say "install flash-linear-attention"
$D exec $C bash -c "pip install -c /root/c.txt flash-linear-attention > /root/pip_fla.log 2>&1; echo rc=\$?; tail -1 /root/pip_fla.log; python -c 'import torch, fla; print(\"FLA\", fla.__version__, \"torch\", torch.__version__)'" >> $L 2>&1
arm F
$D exec $C bash -c "cd /root && python compare.py out_T.pt out_F.pt" >> $L 2>&1
say "install causal-conv1d (capped 1500 s)"
$D exec $C bash -c "timeout 1500 pip install -c /root/c.txt --no-build-isolation causal-conv1d > /root/pip_cc.log 2>&1; echo rc=\$?; tail -2 /root/pip_cc.log; python -c 'import causal_conv1d; print(\"CC\", causal_conv1d.__version__)'" >> $L 2>&1
if $D exec $C python -c "import causal_conv1d" >/dev/null 2>&1; then
  arm FC
  $D exec $C bash -c "cd /root && python compare.py out_T.pt out_FC.pt && python compare.py out_F.pt out_FC.pt" >> $L 2>&1
fi
$D exec $C bash -c "pip list 2>/dev/null | grep -iE '^(torch|triton|transformers|flash-linear|causal|fla-core) '" >> $L 2>&1
$D rm -f $C >/dev/null 2>&1
say "done"; echo GDNK_DONE >> $L
