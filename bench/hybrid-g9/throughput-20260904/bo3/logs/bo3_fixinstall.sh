#!/bin/bash
# bo3_fixinstall: in the Gemma-4 download window of bo3d (no e4b process running), install integration-3 pinned by SHA
# with verbose pip, and prove the new symbols import. Diagnoses why the bo3b install was a no-op.
cd /root/bo3; say(){ echo "[$(date -u +%FT%TZ)] fixinstall: $*"; }
say "waiting for BO3C_DONE"; while [ ! -f BO3C_DONE ]; do sleep 20; done
say "waiting for bo3d's download to start"; while [ ! -f dl_gemma4_redo.log ]; do sleep 10; done; sleep 30
while ps -eo args | grep -E "^python .*(step_decomp|k8_bake)" | grep -v grep >/dev/null; do sleep 10; done
say "before: $(python -c 'import experts4bit_qlora as e, os; t=open(os.path.join(os.path.dirname(e.__file__),\"engines/int4_attn_calib.py\")).read(); print(\"LM_HEAD\", \"_LM_HEAD_ENV\" in t)')"
perl -e 'alarm 900; exec @ARGV' python -m pip install -v --no-input --force-reinstall --no-deps --no-cache-dir "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@2747fe7" > pip_fixinstall.log 2>&1; echo "pip rc=$?"
grep -a "Resolved\|Cloning\|Running command git\|checkout\|Successfully\|Building\|Created wheel\|ERROR\|error" pip_fixinstall.log | cut -c1-160 | head -12
say "after: $(python -c 'import experts4bit_qlora as e, os; t=open(os.path.join(os.path.dirname(e.__file__),\"engines/int4_attn_calib.py\")).read(); print(\"LM_HEAD\", \"_LM_HEAD_ENV\" in t)')"
python -c "from experts4bit_qlora.engines.int4_attn_calib import _LM_HEAD_ENV; from experts4bit_qlora.engines.int4_experts import _mxfp4_store_layout; print('FIXINSTALL OK: head lane + mxfp4 store both present')" || say "FIXINSTALL FAILED"
say "done"
