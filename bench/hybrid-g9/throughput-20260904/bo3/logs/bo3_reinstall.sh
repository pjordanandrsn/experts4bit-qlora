#!/bin/bash
# bo3_reinstall: refresh e4b to buildout-integration-2 @1b2aace (the Gemma-4 multimodal planning fix, #369) in the
# Mixtral download window -- the only stretch of the lane with no e4b process running.
cd /root/bo3; say(){ echo "[$(date -u +%FT%TZ)] reinstall: $*"; }
say "waiting for the mixtral section"; while ! grep -q "=============== mixtral" outer.log; do sleep 15; done
sleep 20
if pgrep -f "step_decomp.py|k8_bake.py" >/dev/null; then say "an e4b process is running; waiting"; while pgrep -f "step_decomp.py|k8_bake.py" >/dev/null; do sleep 10; done; fi
say "installing e4b @1b2aace"
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@1b2aace" > pip_reinstall.log 2>&1 && \
  python -c "from experts4bit_qlora.arch.moe_conventions import convention_for; c=convention_for('gemma4_text'); assert c.drop_re is not None and c.renames; print('REINSTALL OK: gemma4 convention has rename + drop_re')" || { say "REINSTALL FAILED"; tail -3 pip_reinstall.log; }
say "done"
