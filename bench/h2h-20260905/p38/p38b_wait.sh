#!/bin/bash
# box-side: wait for p38's TP_DONE, then run p38b.sh (P38 amendment 1: the Unsloth comparator in a venv with its own resolution).
cd /root/p38 || exit 1
while [ ! -f /root/p38/TP_DONE ]; do sleep 60; done
sleep 30
bash /root/p38/p38b.sh >> /root/p38/outer.log 2>&1
