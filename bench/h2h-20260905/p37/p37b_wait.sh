#!/bin/bash
# box-side: wait for p37's TP_DONE, then run p37b.sh (P37 amendment 2: resume the GPTQ fetch, run the vLLM comparator arms).
cd /root/p37 || exit 1
while [ ! -f /root/p37/TP_DONE ]; do sleep 60; done
sleep 30
bash /root/p37/p37b.sh >> /root/p37/outer.log 2>&1
