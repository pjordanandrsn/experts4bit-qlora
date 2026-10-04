#!/bin/bash
# inside the container: install, record versions, run the unchanged energy harness 3x with a per-process GPU monitor
set -u
cd /root/e392
E4B=$1
command -v git >/dev/null || (DEBIAN_FRONTEND=noninteractive apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git) > apt.log 2>&1
echo "torch==$(python3 -c 'import torch;print(torch.__version__.split("+")[0])')" > c.txt
pip install -q --no-input -c c.txt "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B" "bitsandbytes==0.50.2" > pip.log 2>&1 || { tail -5 pip.log; echo PIPFAIL; exit 9; }
python3 - > versions.txt <<'PY'
import json, importlib.metadata as md, torch, bitsandbytes, subprocess
d = json.loads(md.distribution("experts4bit-qlora").read_text("direct_url.json") or "{}")
print("experts4bit-qlora", md.version("experts4bit-qlora"), d.get("vcs_info", {}).get("commit_id"))
print("bitsandbytes", bitsandbytes.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("driver", subprocess.run(["nvidia-smi","--query-gpu=driver_version,power.limit","--format=csv,noheader"],capture_output=True,text=True).stdout.strip())
PY
cat versions.txt
for rep in 1 2 3; do
  nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader > pre_$rep.txt
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >> pre_$rep.txt 2>&1
  ( nvidia-smi pmon -s u -d 1 > pmon_$rep.txt 2>&1 ) & MON=$!
  date -u +%FT%TZ > t0_$rep.txt
  python3 bench_energy.py > run_$rep.txt 2>&1
  date -u +%FT%TZ > t1_$rep.txt
  kill $MON 2>/dev/null; wait $MON 2>/dev/null
  echo "rep $rep done: $(grep -c '|' run_$rep.txt) table lines"
done
echo E392_DONE
