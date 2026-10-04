#!/bin/bash
# E4B_PAGED_PREFILL_GRAPH on the NAS A2000 ($0): GPU tests, then the knob end to end. bash run_knob.sh <e4b_sha> <gnf4_sha> <dir>
trap "" HUP
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker; C=pgknob; R=/share/Container/scripts/pgcensus/$3; ES=$1; GS=$2
L=$R/run.log; : > $L
say(){ echo "[$(date -u +%FT%TZ)] $*" >> $L; }
fin(){ mkdir -p $R/out; $D cp $C:/root/pg/out/. $R/out/ >/dev/null 2>&1; $D rm -f $C >/dev/null 2>&1; say "$1"; echo PG_DONE >> $L; exit 0; }
$D rm -f $C >/dev/null 2>&1
$D run -d --name $C --runtime nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e HF_HUB_DISABLE_XET=1 pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel sleep 14400 >/dev/null
U=$($D exec $C nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1)
say "preflight util,mem_used,mem_total: $U"
UT=$(echo "$U" | cut -d, -f1 | tr -d ' ')
[ -n "$UT" ] && [ "$UT" -le 20 ] || fin "ABORT: GPU busy ($U)"
$D exec $C bash -c "mkdir -p /root/pg/out"
$D cp $R/stage/. $C:/root/pg/
$D exec $C bash -c "apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y git >/dev/null 2>&1; git --version" >> $L 2>&1
$D exec -w /root $C bash -c "set -e; echo torch==2.8.0 > c.txt
  pip install -q -c c.txt 'grouped-nf4-gemm @ git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GS' 2>&1 | tail -1
  git clone -q https://github.com/pjordanandrsn/experts4bit-qlora.git e4b && git -C e4b checkout -q $ES && git -C e4b log --oneline -1
  pip install -q -c c.txt -e ./e4b transformers==5.17.0 bitsandbytes==0.50.2 accelerate safetensors huggingface_hub fastapi httpx pytest jsonschema 2>&1 | tail -1" >> $L 2>&1
$D exec $C python -c "import torch, transformers, pytest, experts4bit_qlora, int4_smallm, fp8_kv; assert torch.cuda.is_available(); print('GUARD ok', torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))" >> $L 2>&1
grep -q "^GUARD ok" $L || fin "ABORT: install guard failed"
say "arm T1: the new GPU tests, verbose"
$D exec -w /root/e4b $C bash -c "python -m pytest -v -p no:cacheprovider tests/test_prefill_graph_gpu.py 2>&1 | grep -E 'PASSED|FAILED|SKIPPED|ERROR|passed|failed' | cut -c1-200" >> $L 2>&1
say "arm T2: neighbours"
$D exec -w /root/e4b $C bash -c "python -m pytest -q -rs -p no:cacheprovider tests/test_prefill_graph.py tests/test_serve_paged.py tests/test_rt_cache_graph_gpu.py tests/test_decode_graph_buckets.py tests/test_kv_step_select.py tests/test_scheduler.py 2>&1 | tail -8 | cut -c1-200" >> $L 2>&1
say "tiny models + arenas"
$D exec -w /root/pg $C bash -c "python make_tiny.py /root/pg/tiny 2>&1 | tail -1; mkdir -p arena; K8_WORK=/root/pg/arena K8_MODEL=/root/pg/tiny python k8_bake.py 2>&1 | tail -1
  python make_tiny_granite.py /root/pg/tinyg 2>&1 | tail -3; mkdir -p arenag; K8_WORK=/root/pg/arenag K8_MODEL=/root/pg/tinyg python k8_bake.py 2>&1 | tail -1; grep -o '\"err\": \"[^\"]*' arenag/bake.json | head -1" >> $L 2>&1
STACK="E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1 E4B_PAGED_FUSE_QKV=1"
GR="E4B_SERVE_EXP_INT4=0 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1"
ROUTE="E4B_INT4_GROUPED_SMALLM=auto E4B_INT4_LEAN_GLUE=auto E4B_NF4_GROUPED_SMALLM=0 E4B_MXFP4_GROUPED_SMALLM=auto"
Q="E4B_PAGED_MODEL=/root/pg/tiny E4B_PAGED_ARENA=/root/pg/arena/nf4.arena E4B_PAGED_CALIB=/root/pg/calib.json E4B_PAGED_PLACEMENT=all-vram PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
G="E4B_PAGED_MODEL=/root/pg/tinyg E4B_PAGED_ARENA=/root/pg/arenag/nf4.arena E4B_PAGED_CALIB=/root/pg/calib.json E4B_PAGED_PLACEMENT=all-vram PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
run(){ local tag=$1 extra=$2; say "arm $tag ($extra)"
  $D exec -w /root/pg $C bash -c "env -u E4B_INT4_PREFILL -u E4B_PAGED_PREFILL_ATTN $extra perl -e 'alarm 1500; exec @ARGV' python census_prefill_sync.py knob out/$tag.json > out/$tag.log 2>&1; echo RC=\$?; grep -aE '^(KNOB|  \{)|PREFILL_GRAPH on' out/$tag.log | cut -c1-700 | head -20; grep -aE 'Error|Traceback' out/$tag.log | grep -v CompilationError | tail -3 | cut -c1-600" >> $L 2>&1; }
run K1_qwen3_int4_on "$Q $STACK $ROUTE E4B_PAGED_PREFILL_GRAPH=1"
run K0_qwen3_int4_maxseqs1_refusal "$Q $STACK $ROUTE E4B_PAGED_PREFILL_GRAPH=1 E4B_PAGED_MAX_SEQS=1"
run KG_granite_nf4_on "$G $GR $ROUTE E4B_PAGED_PREFILL_GRAPH=1"
fin done
