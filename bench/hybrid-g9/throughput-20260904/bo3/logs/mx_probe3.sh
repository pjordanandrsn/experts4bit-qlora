#!/bin/bash
cd /root/bo3; say(){ echo "[$(date -u +%FT%TZ)] mxprobe3: $*"; }
say waiting for BO3E_DONE; while [ ! -f BO3E_DONE ]; do sleep 30; done
while ps -eo args | grep -E "^python .*(step_decomp|k8_bake)" | grep -v grep >/dev/null; do sleep 10; done
say "route test: NF4 path vs MXFP4 store vs reference on the lane's own wrapper state"
E4B_SERVE_EXP_INT4=1 E4B_INT4_KEEP_NF4=0 E4B_MODEL_ID=openai/gpt-oss-20b PYTHONPATH=/root/bo3/hook_v2 perl -e 'alarm 1500; exec @ARGV' python mx_probe3.py 2>&1 | grep -v "Warning\|warn\|Fetching\|it/s" | tee mx_probe3.txt | grep -a "ROUTE\|store kind\|hybrid tier\|wrapper\|default\|Error\|error" 
say "done"
