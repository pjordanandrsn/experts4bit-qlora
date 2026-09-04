#!/bin/bash
cd /root/bo3; say(){ echo "[$(date -u +%FT%TZ)] mxprobe: $*"; }
say "waiting for fixinstall to finish"; while ! grep -q "fixinstall: done" fixinstall.log 2>/dev/null; do sleep 15; done
while ps -eo args | grep -E "^python .*(step_decomp|k8_bake)" | grep -v grep >/dev/null; do sleep 10; done
say "running the kernel-vs-reference probe on gpt-oss layer 0, expert 3"
PYTHONPATH=/root/bo3/hook_v2 perl -e 'alarm 600; exec @ARGV' python mx_probe.py 2>&1 | grep -v Warning | tee mx_probe.txt
say "done"
