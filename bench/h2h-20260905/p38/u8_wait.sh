#!/bin/bash
# box-side: after TP4_DONE (GPU free), run the U8 proof once; marker U8_DONE.
cd /root/p38 || exit 1
while [ ! -f /root/p38/TP4_DONE ]; do sleep 30; done
sleep 20
echo "[$(date -u +%FT%TZ)] u8 proof (amendment 4)" >> outer.log
HF_HUB_OFFLINE=1 /root/p38/venv-unsloth/bin/python /root/p38/u8_proof.py > logs/u8_proof.log 2>&1; echo "u8 proof rc=$?" >> outer.log
grep -aE "n_pred_true|wrap_depth|param_kinds|Traceback|Error" logs/u8_proof.log | tail -8 >> outer.log
echo "[$(date -u +%FT%TZ)] U8_DONE" >> outer.log; touch U8_DONE
