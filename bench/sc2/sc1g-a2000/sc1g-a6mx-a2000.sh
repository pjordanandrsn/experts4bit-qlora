#!/bin/bash
# SC1g A6 (#846), the conv2 lead: a $0 correctness check on the QNAP A2000 (sm_86); no timing is read or reported.
#  X   sc1g_mxfp4_prefill_check.py on layers 0, 11 and 23 of gpt-oss-20b's released MXFP4 experts: e4b's KEEP_NF4=0 prompt
#      route (gemm_mxfp4_grouped with sizes [1]*M) against the tiled kernel, the int8-activation decode GEMV, the NF4 M-tile and
#      cuBLAS bf16, each vs its own fp64 reference, at M = 16 / 256 / 2048 (the maintainer's question: is the > 256-row prompt
#      path out of line with the others?). Must exit 0 (IN_LINE and AT_FLOOR).
#  XM  the mutation (the reference decodes the high nibble first), layer 0 -- must exit 1 (ABOVE_FLOOR).
# The check script is staged at $W/sc1g_mxfp4_prefill_check.py; outputs land in $W/out.
trap "" HUP
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker
GNF4=dc8f94abfd868f149178623f6eb403dc8b892b02
C=sc1g-mxcheck
W=/share/Container/sc1g-mxcheck
OUT=$W/out; mkdir -p $OUT
say(){ echo "[$(date -u +%FT%TZ)] sc1g-a6mx: $*"; }
RT=$($D info --format '{{json .Runtimes}}' | grep -o 'nvidia[a-z-]*' | head -1); say "runtime ${RT:-none}"
$D rm -f $C >/dev/null 2>&1
$D run -d --name $C --runtime "${RT:-nvidia-runtime}" -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
   -v /share/models/gpt-oss-20b:/models:ro -v $W:/work pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel sleep 5400 >/dev/null \
   || { say "RUN FAIL"; echo SC1G_A6MX_DONE rc=90; exit 90; }
U=$($D exec $C nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits | tr -d ' ')
say "pre-flight gpu util,free MiB: $U"
case "${U%%,*}" in ''|*[!0-9]*) ;; *) [ "${U%%,*}" -gt 20 ] && { say "GPU busy (${U%%,*}%) -- another bench may be timing; not running"; $D rm -f $C >/dev/null; echo SC1G_A6MX_DONE rc=95; exit 95; };; esac
case "${U##*,}" in ''|*[!0-9]*) ;; *) [ "${U##*,}" -lt 3000 ] && { say "only ${U##*,} MiB free -- not contending with the home services"; $D rm -f $C >/dev/null; echo SC1G_A6MX_DONE rc=96; exit 96; };; esac
$D exec $C bash -c "command -v git >/dev/null || { apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y git >/dev/null 2>&1; }; pip install -q safetensors 'grouped-nf4-gemm @ git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4' 2>&1 | tail -2" || { say "INSTALL FAIL"; echo SC1G_A6MX_DONE rc=92; exit 92; }
$D exec -w /work $C python -c "import torch, triton, mxfp4_grouped, nf4_grouped, int4_b32, nf4_pack_ref, safetensors; assert torch.cuda.is_available(); print('torch', torch.__version__, 'triton', triton.__version__, torch.cuda.get_device_name(0))" \
   || { say "IMPORT/CUDA FAIL"; $D rm -f $C >/dev/null; echo SC1G_A6MX_DONE rc=93; exit 93; }
say "arm X: layers 0, 11, 23; gate_up and down; M = 16 / 256 / 2048"
$D exec -w /work $C python sc1g_mxfp4_prefill_check.py --model /models --out /work/out/a6mx_a2000_check.json --layers 0,11,23; xrc=$?
say "arm XM (mutation): the reference decodes the high nibble first, layer 0 -- must exit 1"
$D exec -w /work $C python sc1g_mxfp4_prefill_check.py --model /models --out /work/out/a6mx_a2000_check_mutate.json --layers 0 --mutate; xmrc=$?
$D rm -f $C >/dev/null 2>&1
echo "SC1G_A6MX_DONE armX=$xrc armXM=$xmrc (X must be 0, XM must be 1)"
