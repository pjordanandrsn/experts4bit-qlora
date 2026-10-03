#!/bin/bash
# P106 feasibility on the NAS A2000 ($0): the GdnToggle probe, kernel-free then fla then fla + causal-conv1d, one container.
# Ran 2026-10-03 (log stamps 08:17:41Z-08:21:59Z) at e4b 4bddf92 with toggle_probe.py's FIRST DRAFT, which imported the
# dense parity test's helpers from the e4b checkout's tests/ (/root/e4b/tests) and had no --json-out. The bench version
# imports them from its own directory; p106_run.sh stages them side by side.
trap "" HUP
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker; C=p106-tog; R=/share/Container/scripts/p106tog; ES=$1; GS=$2
L=$R/probe.log; : > $L
say(){ echo "[$(date -u +%FT%TZ)] $*" >> $L; }
$D rm -f $C >/dev/null 2>&1
$D run -d --name $C --runtime nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel sleep 10800 >/dev/null
U=$($D exec $C nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d " ")
APPS=$($D exec $C nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
say "preflight utilization ${U:-?} % compute apps ${APPS:-?}"
[ -n "$U" ] && [ "$U" -le 20 ] || { say "ABORT: GPU busy"; echo PROBE_DONE >> $L; $D rm -f $C >/dev/null 2>&1; exit 1; }
$D exec $C bash -c "apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y git >/dev/null 2>&1; echo torch==2.8.0 > /root/c.txt; git clone -q https://github.com/pjordanandrsn/experts4bit-qlora.git /root/e4b && git -C /root/e4b checkout -q $ES && cd /root/e4b && pip install -q -c /root/c.txt -e . 'grouped-nf4-gemm @ git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GS' transformers==5.17.0 bitsandbytes==0.50.2 accelerate safetensors pytest > /root/pip.log 2>&1; python -c 'import torch, transformers, experts4bit_qlora, importlib.util as u; assert torch.cuda.is_available() and torch.__version__.startswith(\"2.8.0\"); assert not any(u.find_spec(m) for m in (\"fla\", \"causal_conv1d\", \"kernels\")); print(\"READY\", torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))' || echo NOT_READY" >> $L 2>&1
grep -q "^READY" $L || { say "ABORT: environment not ready"; echo PROBE_DONE >> $L; $D rm -f $C >/dev/null 2>&1; exit 1; }
$D exec $C mkdir -p /root/p106
$D cp $R/gdn_toggle.py $C:/root/p106/gdn_toggle.py
$D cp $R/toggle_probe.py $C:/root/p106/toggle_probe.py
say "SAVE (kernel-free process)"
$D exec $C bash -c "cd /root/p106 && python toggle_probe.py --save /root/p106/ref.pt 2>&1 | grep -E 'TOGGLE|SAVED|Error|error' | cut -c1-600" >> $L 2>&1
$D exec $C bash -c "pip install -q -c /root/c.txt flash-linear-attention==0.5.2 > /root/pip_fla.log 2>&1; python -c 'import fla; print(\"FLA\", fla.__version__)'" >> $L 2>&1
say "COMPARE under fla"
$D exec $C bash -c "cd /root/p106 && python toggle_probe.py --compare /root/p106/ref.pt 2>&1 | grep -E 'TOGGLE|STEP|Error|error' | cut -c1-600" >> $L 2>&1
$D exec $C bash -c "pip install -q -c /root/c.txt --no-build-isolation causal-conv1d==1.7.0 > /root/pip_cc.log 2>&1; python -c 'import causal_conv1d; print(\"CC\", causal_conv1d.__version__)'" >> $L 2>&1
say "COMPARE under fla + causal-conv1d"
$D exec $C bash -c "cd /root/p106 && python toggle_probe.py --compare /root/p106/ref.pt 2>&1 | grep -E 'TOGGLE|STEP|Error|error' | cut -c1-600" >> $L 2>&1
$D rm -f $C >/dev/null 2>&1
say done; echo PROBE_DONE >> $L
