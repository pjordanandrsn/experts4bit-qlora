#!/bin/bash
# SC1g A3 read (#846, e4b#1175): the served loop's KV writes (append_prompt for the prompt, append_many per decode token) vs per-layer append at
# k_groups 4 / 8 / 16 on the QNAP A2000 (sm_86), $0, correctness only. Arm B must read BITWISE_EQUAL to clear the bulk
# path; arm BM (layers' K in reversed order on the per-layer side) must read DIFFERENT.
trap "" HUP
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker
E4B=a7891300bdfa9d65526bb8787831d062f61453e8
GNF4=dc8f94abfd868f149178623f6eb403dc8b892b02
C=sc1g-a3read
S=/share/Container/scripts
OUT=$S/sc1g-a3read-out; mkdir -p $OUT
say(){ echo "[$(date -u +%FT%TZ)] sc1g-a3read: $*"; }
RT=$($D info --format '{{json .Runtimes}}' | grep -o 'nvidia[a-z-]*' | head -1); say "runtime ${RT:-none}"
$D rm -f $C >/dev/null 2>&1
$D run -d --name $C --runtime "${RT:-nvidia-runtime}" -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
   -v $OUT:/out pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime sleep 3600 >/dev/null || { say "RUN FAIL"; echo SC1G_A3READ_DONE rc=90; exit 90; }
U=$($D exec $C nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | tr -d ' ')
say "pre-flight gpu util,used,total: $U"
case "${U%%,*}" in ''|*[!0-9]*) ;; *) [ "${U%%,*}" -gt 20 ] && { say "GPU busy (${U%%,*}%) -- not running"; $D rm -f $C >/dev/null; echo SC1G_A3READ_DONE rc=95; exit 95; };; esac
$D exec $C bash -c "apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y git gcc g++ >/dev/null 2>&1; gcc --version | head -1" || { say "APT FAIL"; echo SC1G_A3READ_DONE rc=91; exit 91; }
$D exec $C bash -c "set -e; pip install -q 'grouped-nf4-gemm @ git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4' 2>&1 | tail -2; git clone -q https://github.com/pjordanandrsn/experts4bit-qlora.git /w && git -C /w checkout -q $E4B; echo torch==2.8.0 > /c.txt; pip install -q -c /c.txt -e /w safetensors 2>&1 | tail -2" || { say "INSTALL FAIL"; echo SC1G_A3READ_DONE rc=92; exit 92; }
$D cp $S/sc1g_prompt_append_check.py $C:/w/sc1g_prompt_append_check.py
$D exec -w /w $C python -c "import torch, fp8_kv, experts4bit_qlora.engines.fp8_paged_kv; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))" \
   || { say "IMPORT/CUDA FAIL"; echo SC1G_A3READ_DONE rc=93; exit 93; }
say "arm B: append_prompt vs per-layer append, k_groups 4 / 8 / 16"
$D exec -w /w $C python sc1g_prompt_append_check.py --out /out/a2000_prompt_append.json 2>&1 | tail -3; brc=${PIPESTATUS[0]}
say "arm BM (mutation): per-layer side gets the layers' K reversed -- must read DIFFERENT"
$D exec -w /w $C python sc1g_prompt_append_check.py --mutate --out /out/a2000_prompt_append_mutate.json 2>&1 | tail -3; bmrc=${PIPESTATUS[0]}
$D rm -f $C >/dev/null 2>&1
echo "SC1G_A3READ_DONE armB=$brc armBM=$bmrc (armBM must be nonzero)"
