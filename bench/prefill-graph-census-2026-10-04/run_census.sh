#!/bin/bash
# Prefill-graph feasibility census on the NAS A2000 ($0). bash run_census.sh <e4b_sha> <gnf4_sha>
trap "" HUP
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker; C=pgcensus; R=/share/Container/scripts/pgcensus; [ -n "$3" ] && R=$R/$3; ES=$1; GS=$2
L=$R/run.log; : > $L
say(){ echo "[$(date -u +%FT%TZ)] $*" >> $L; }
fin(){ mkdir -p $R/out; $D cp $C:/root/pg/out/. $R/out/ >/dev/null 2>&1; $D rm -f $C >/dev/null 2>&1; say "$1"; echo PG_DONE >> $L; exit 0; }
$D rm -f $C >/dev/null 2>&1
$D run -d --name $C --runtime nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e HF_HUB_DISABLE_XET=1 pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel sleep 14400 >/dev/null
U=$($D exec $C nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1)
say "preflight util,mem_used,mem_total: $U; apps: $($D exec $C nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader | tr '\n' ';')"
UT=$(echo "$U" | cut -d, -f1 | tr -d ' ')
[ -n "$UT" ] && [ "$UT" -le 20 ] || fin "ABORT: GPU busy ($U)"
$D exec $C bash -c "mkdir -p /root/pg/out"
$D cp $R/stage/. $C:/root/pg/
$D exec $C bash -c "apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y git >/dev/null 2>&1; git --version" >> $L 2>&1
$D exec -w /root $C bash -c "set -e; echo torch==2.8.0 > c.txt
  pip install -q -c c.txt 'grouped-nf4-gemm @ git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GS' 2>&1 | tail -2
  git clone -q https://github.com/pjordanandrsn/experts4bit-qlora.git e4b && git -C e4b checkout -q $ES && git -C e4b log --oneline -1
  pip install -q -c c.txt -e ./e4b transformers==5.17.0 bitsandbytes==0.50.2 accelerate safetensors huggingface_hub fastapi 2>&1 | tail -2" >> $L 2>&1
$D exec $C python -c "import torch, transformers, experts4bit_qlora, int4_smallm; assert torch.cuda.is_available(); print('GUARD ok', torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))" >> $L 2>&1
grep -q "^GUARD ok" $L || fin "ABORT: install guard failed"
say "tiny model + arena"
$D exec -w /root/pg $C bash -c "python make_tiny.py /root/pg/tiny 2>&1 | tail -1; mkdir -p arena; K8_WORK=/root/pg/arena K8_MODEL=/root/pg/tiny python k8_bake.py 2>&1 | tail -2; ls -la arena | head" >> $L 2>&1
$D exec $C test -f /root/pg/arena/nf4.arena || fin "ABORT: no arena"
STACK="E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1 E4B_PAGED_FUSE_QKV=1"
ROUTE="E4B_INT4_GROUPED_SMALLM=auto E4B_INT4_LEAN_GLUE=auto E4B_NF4_GROUPED_SMALLM=0 E4B_MXFP4_GROUPED_SMALLM=auto"
BASE="E4B_PAGED_MODEL=/root/pg/tiny E4B_PAGED_ARENA=/root/pg/arena/nf4.arena E4B_PAGED_CALIB=/root/pg/calib.json E4B_PAGED_PLACEMENT=all-vram PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
run(){ local tag=$1 mode=$2 extra=$3; say "arm $tag $mode ($extra)"
  $D exec -w /root/pg $C bash -c "env -u E4B_INT4_PREFILL -u E4B_PAGED_PREFILL_ATTN $BASE $STACK $ROUTE $extra perl -e 'alarm 1500; exec @ARGV' python census_prefill_sync.py $mode out/${tag}_$mode.json > out/${tag}_$mode.log 2>&1; echo RC=\$?; grep -aE '^(CENSUS|CAPTURE|  )' out/${tag}_$mode.log | cut -c1-400 | head -60; grep -aE 'Error|Traceback' out/${tag}_$mode.log | tail -4 | cut -c1-600" >> $L 2>&1; }
run A census ""
run A replay ""

run M replay "E4B_INT4_PREFILL=mtile"
fin done
