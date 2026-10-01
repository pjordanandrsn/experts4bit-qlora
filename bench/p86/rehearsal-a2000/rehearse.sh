#!/bin/bash
trap "" HUP
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker
C=p86-rehearsal; R=/share/Container/scripts/p86r
$D rm -f $C >/dev/null 2>&1
$D run -d --name $C --runtime nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e HF_HUB_DISABLE_XET=1 pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel sleep 7200 >/dev/null
echo "== card before:"; $D exec $C nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
$D exec $C bash -c "mkdir -p /root/p86r; apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y git >/dev/null 2>&1"
for f in p86_vllm_census.py mkprompts.py; do $D cp $R/$f $C:/root/p86r/$f; done
$D exec $C bash -c "cd /root/p86r && python mkprompts.py && pip install -q --no-input 'vllm==0.11.0' 'transformers>=4.56,<5' > pip.log 2>&1; echo pip_rc=\$?; python -c 'import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__)'"
for B in 1 4; do
  $D exec -w /root/p86r $C env P86_BATCH=$B P86_PROMPTS=/root/p86r/prompts_b$B.json P86_MODEL=Qwen/Qwen3-0.6B P86_REV=main P86_OUT=/root/p86r/census_b$B.json P86_SHORT=8 P86_LONG=24 P86_GPU_UTIL=0.5 P86_MAX_LEN=512 python p86_vllm_census.py > $R/census_b$B.log 2>&1
  echo "CENSUS B=$B rc=$?"
done
# control: the same census with enforce_eager (no graphs) -- if graph replays were invisible to the profiler, the
# graph arm's per-step kernel time would be ~0 while the eager arm's is not
$D exec -w /root/p86r $C bash -c "sed 's/enforce_eager=False/enforce_eager=True/' p86_vllm_census.py > p86_eager.py && env P86_BATCH=4 P86_PROMPTS=/root/p86r/prompts_b4.json P86_MODEL=Qwen/Qwen3-0.6B P86_REV=main P86_OUT=/root/p86r/census_b4_eager.json P86_SHORT=8 P86_LONG=24 P86_GPU_UTIL=0.5 P86_MAX_LEN=512 python p86_eager.py" > $R/census_b4_eager.log 2>&1
echo "CENSUS EAGER B=4 rc=$?"
for f in census_b1.json census_b4.json census_b4_eager.json pip.log; do $D cp $C:/root/p86r/$f $R/$f 2>/dev/null; done
$D rm -f $C >/dev/null 2>&1
echo P86R_DONE
