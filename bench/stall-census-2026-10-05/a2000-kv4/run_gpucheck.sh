#!/bin/bash
# Stall census on the NAS A2000 ($0): the bulk-KV branch's tests under CUDA, then the bookkeeping bench with the
# library's own bulk arm.  bash run_gpucheck.sh <e4b_sha> <gnf4_sha> <tag> ["bench args"]
trap "" HUP
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker; C=stallcensus; R=/share/Container/scripts/stallcensus; S=$R/stage; R=$R/$3; ES=$1; GS=$2; XA="$4"
L=$R/run.log; mkdir -p $R; : > $L
say(){ echo "[$(date -u +%FT%TZ)] $*" >> $L; }
fin(){ mkdir -p $R/out; $D cp $C:/root/sc/out/. $R/out/ >/dev/null 2>&1; $D rm -f $C >/dev/null 2>&1; say "$1"; echo SC_DONE >> $L; exit 0; }
$D rm -f $C >/dev/null 2>&1
$D run -d --name $C --runtime nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel sleep 7200 >/dev/null
U=$($D exec $C nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1)
say "preflight util,mem_used,mem_total: $U; host load: $(cat /proc/loadavg)"
UT=$(echo "$U" | cut -d, -f1 | tr -d ' ')
[ -n "$UT" ] && [ "$UT" -le 20 ] || fin "ABORT: GPU busy ($U)"
$D exec $C bash -c "mkdir -p /root/sc/out"
$D cp $S/. $C:/root/sc/
$D exec $C bash -c "apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y git >/dev/null 2>&1; git --version" >> $L 2>&1
$D exec -w /root $C bash -c "set -e; echo torch==2.8.0 > c.txt
  pip install -q -c c.txt 'grouped-nf4-gemm @ git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GS' 2>&1 | tail -2
  git clone -q https://github.com/pjordanandrsn/experts4bit-qlora.git e4b && git -C e4b checkout -q $ES && git -C e4b log --oneline -1
  pip install -q -c c.txt -e ./e4b transformers==5.17.0 bitsandbytes==0.50.2 accelerate safetensors huggingface_hub pytest fastapi uvicorn httpx 2>&1 | tail -2" >> $L 2>&1
$D exec $C python -c "import torch, experts4bit_qlora, fp8_kv, row_pool; assert torch.cuda.is_available(); print('GUARD ok', torch.__version__, torch.cuda.get_device_name(0))" >> $L 2>&1
grep -q "^GUARD ok" $L || fin "ABORT: install guard failed"
say "pytest (CUDA)"
$D exec -w /root/e4b $C bash -c "perl -e 'alarm 1500; exec @ARGV' python -m pytest -q -p no:cacheprovider tests/test_bulk_kv.py tests/test_fp8_paged_kv.py tests/test_fp8_paged_kv_mixed_geometry.py tests/test_kv_append_batch.py tests/test_kv_graph_append.py tests/test_kv_step_select.py tests/test_decode_graph_buckets.py tests/test_serve_paged.py tests/test_scheduler.py tests/test_prefill_graph.py tests/test_prefill_graph_gpu.py > /root/sc/out/pytest.log 2>&1; echo PYTEST_RC=\$?; tail -4 /root/sc/out/pytest.log" >> $L 2>&1
say "kv bookkeeping bench"
$D exec -w /root/sc $C bash -c "perl -e 'alarm 1800; exec @ARGV' python kv_bookkeeping_bench.py --out out/kv_bench.json $XA > out/kv_bench.log 2>&1; echo RC=\$?; grep -aE '^(INFO|PARITY|LIBRARY|BULK|DONE)' out/kv_bench.log | cut -c1-1500; grep -aE 'Error|Traceback' out/kv_bench.log | tail -6 | cut -c1-800" >> $L 2>&1
say "host load after: $(cat /proc/loadavg)"
fin done
