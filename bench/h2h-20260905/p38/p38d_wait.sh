#!/bin/bash
cd /root/p38 || exit 1
while [ ! -f /root/p38/TP3_DONE ]; do sleep 60; done
sleep 30
bash /root/p38/p38d.sh >> /root/p38/outer.log 2>&1
