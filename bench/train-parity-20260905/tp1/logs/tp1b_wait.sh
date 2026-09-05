#!/bin/bash
# box-side: wait for tp1's TP_DONE, then run tp1b.sh (P36 amendment 3). Started with setsid nohup so it survives the ssh session.
cd /root/tp1 || exit 1
while [ ! -f /root/tp1/TP_DONE ]; do sleep 60; done
sleep 30
bash /root/tp1/tp1b.sh >> /root/tp1/outer.log 2>&1
