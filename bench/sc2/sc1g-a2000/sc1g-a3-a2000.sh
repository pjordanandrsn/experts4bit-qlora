#!/bin/bash
# SC1g A3 (#846): two $0 correctness side checks on the QNAP A2000 (sm_86); no timing is read or reported.
#  P   e4b's fp8 KV pack at k_groups 4 / 8 / 16 on gpt-oss's attention geometry (Fp8PagedKV.append, then reference_kv): the
#      stored K/V vs the bf16 originals. The A2000 cannot run the fp8 decode kernel itself (sm_89+), so this is the half of
#      e4b#1175's check that it can run; the kernel half rides box J on the 5090. PM is its mutation arm (must PACK_BAD).
#  T   gnf4's NF4 M-tile (gemm_4bit_grouped) on one layer's real experts: a whole 2561-token window's rows in ONE call vs
#      128-token chunks vs an fp32 reference (the maintainer's request: does box J's NF4 full anchor lose accuracy at large
#      M?). TM is its mutation arm (must DISAGREE).
trap "" HUP
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker
E4B=16074418a7e171f55dd983c4a1ba5c7d91df9948
GNF4=dc8f94abfd868f149178623f6eb403dc8b892b02
C=sc1g-a3
S=/share/Container/scripts
OUT=$S/sc1g-a3-out; mkdir -p $OUT
say(){ echo "[$(date -u +%FT%TZ)] sc1g-a3: $*"; }
RT=$($D info --format '{{json .Runtimes}}' | grep -o 'nvidia[a-z-]*' | head -1); say "runtime ${RT:-none}"
$D rm -f $C >/dev/null 2>&1
$D run -d --name $C --runtime "${RT:-nvidia-runtime}" -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
   -v /share/models/gpt-oss-20b:/models/gpt-oss-20b:ro -v $OUT:/out pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime sleep 5400 >/dev/null \
   || { say "RUN FAIL"; echo SC1G_A3_DONE rc=90; exit 90; }
U=$($D exec $C nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | tr -d ' ')
say "pre-flight gpu util,used,total: $U"
case "${U%%,*}" in ''|*[!0-9]*) ;; *) [ "${U%%,*}" -gt 20 ] && { say "GPU busy (${U%%,*}%) -- another bench may be timing; not running"; $D rm -f $C >/dev/null; echo SC1G_A3_DONE rc=95; exit 95; };; esac
$D exec $C bash -c "apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y git gcc g++ >/dev/null 2>&1; gcc --version | head -1" || { say "APT FAIL"; echo SC1G_A3_DONE rc=91; exit 91; }
$D exec $C bash -c "set -e; pip install -q 'grouped-nf4-gemm @ git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4' 2>&1 | tail -2; git clone -q https://github.com/pjordanandrsn/experts4bit-qlora.git /w && git -C /w checkout -q $E4B; echo torch==2.8.0 > /c.txt; pip install -q -c /c.txt -e /w safetensors 2>&1 | tail -2" || { say "INSTALL FAIL"; echo SC1G_A3_DONE rc=92; exit 92; }
for f in sc1g_attn_check.py sc1g_mtile_check.py sc1g_gemv_check.py; do $D cp $S/$f $C:/w/$f; done
$D exec -w /w $C python -c "import torch, triton, nf4_grouped, nf4_pack_ref, mxfp4_pack_ref, fp8_kv, experts4bit_qlora.engines.fp8_paged_kv; assert torch.cuda.is_available(); print('torch', torch.__version__, 'triton', triton.__version__, torch.cuda.get_device_name(0))" \
   || { say "IMPORT/CUDA FAIL"; echo SC1G_A3_DONE rc=93; exit 93; }
say "arm P: the fp8 KV pack at k_groups 4 / 8 / 16"
$D exec -w /w $C python sc1g_attn_check.py --pack-only --out /out/a2000_pack.json 2>&1 | tail -2; prc=${PIPESTATUS[0]}
say "arm PM (mutation): recon against the originals shifted one token -- must PACK_BAD"
$D exec -w /w $C python sc1g_attn_check.py --pack-only --mutate-pack --out /out/a2000_pack_mutate.json 2>&1 | tail -2; pmrc=${PIPESTATUS[0]}
say "arm T: the NF4 M-tile, one call vs 128-token chunks vs fp32, layers 0 and 12"
$D exec -w /w $C python sc1g_mtile_check.py --model-dir /models/gpt-oss-20b --layers 0,12 --out /out/a2000_mtile.json 2>&1 | tail -18; trc=${PIPESTATUS[0]}
say "arm TM (mutation): the reference reads the next expert -- must DISAGREE"
$D exec -w /w $C python sc1g_mtile_check.py --model-dir /models/gpt-oss-20b --layers 0 --mutate --out /out/a2000_mtile_mutate.json 2>&1 | tail -2; tmrc=${PIPESTATUS[0]}
$D rm -f $C >/dev/null 2>&1
echo "SC1G_A3_DONE armP=$prc armPM=$pmrc armT=$trc armTM=$tmrc (PM and TM must be nonzero)"
