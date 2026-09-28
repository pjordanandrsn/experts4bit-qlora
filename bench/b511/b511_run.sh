#!/bin/bash
# bench/b511/b511_run.sh -- lane B511, BOX side (bench/b511/PREREG-b511.md; experts4bit-qlora#511). Pattern:
# bench/b393/b393_run.sh. No model download: tiny random models built by the tests themselves.
#
# Installs grouped-nf4-gemm at GNF4_SHA (no deps: the image's torch and triton stay), clones experts4bit-qlora at
# E4B_SHA (the launcher-proven heads.e4b) and installs it editable with [test] under a constraints file that pins
# the image's torch and triton, then runs two arms: A = the registered test files; M = the one registered mutation
# (mut_b511.py) and the replay-vs-padded-eager test alone. The verdict is read from armA.xml / armM.xml against the
# pre-registration; this script only says whether both arms ran.
set -uo pipefail
W=/root/b511; mkdir -p $W/logs; cd $W
say(){ echo "[$(date -u +%FT%TZ)] b511: $*" | tee -a summary.txt; }
NONCE=${B511_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/B511_RUN_NONCE.tmp && mv $W/B511_RUN_NONCE.tmp $W/B511_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > B511_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > B511_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
: > summary.txt
for v in B511_RUN_ID B511_DEADLINE_EPOCH B511_INSTANCE_ID GNF4_SHA E4B_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
for s in "$GNF4_SHA" "$E4B_SHA"; do
  case "$s" in *[!0-9a-f]*|"") say "refusing: $s is not hex"; finish 78;; esac
  [ ${#s} -eq 40 ] || { say "refusing: $s is not a 40-char sha"; finish 78; }
done
echo "$B511_INSTANCE_ID" > INSTANCE_ID
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/b511/staged.sha256"; finish 9; }

python -c "import torch; assert torch.cuda.is_available()" || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,uuid,compute_cap --format=csv,noheader | tee forensics.txt
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in *5090*) ;; *) say "REFUSED: card is '$GPU', the lane registers the RTX 5090 class"; echo "refused: class $GPU" > REFUSAL; finish 15;; esac
command -v gcc >/dev/null || { say "no gcc in the image (the lane registers the -devel image)"; finish 9; }

TV=$(python -c "import torch; print(torch.__version__)"); XV=$(python -c "import triton; print(triton.__version__)")
printf 'torch==%s\ntriton==%s\n' "$TV" "$XV" > constraints.txt
say "image torch $TV triton $XV (pinned by constraints.txt)"
say "install gnf4 @$GNF4_SHA"
perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --no-deps \
  "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 \
  || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
say "clone e4b @$E4B_SHA"
perl -e 'alarm 900; exec @ARGV' git clone -q https://github.com/pjordanandrsn/experts4bit-qlora.git e4b > logs/clone.log 2>&1 \
  && git -C e4b checkout -q "$E4B_SHA" || { tail -3 logs/clone.log; say "CLONE FAIL"; finish 9; }
[ "$(git -C e4b rev-parse HEAD)" = "$E4B_SHA" ] || { say "TRIPWIRE: clone is not $E4B_SHA"; finish 9; }
perl -e 'alarm 1200; exec @ARGV' python -m pip install -q --no-input -c constraints.txt -e "./e4b[test]" > logs/pip_e4b.log 2>&1 \
  || { tail -5 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
(cd $W/e4b && python -c "import experts4bit_qlora as e, torch, triton, transformers, bitsandbytes as b, importlib.metadata as m; print('e4b', e.__file__); print('gnf4', m.version('grouped-nf4-gemm')); print('torch', torch.__version__, 'triton', triton.__version__, 'transformers', transformers.__version__, 'bitsandbytes', b.__version__); print('cc', torch.cuda.get_device_capability())") > versions.txt 2>&1 \
  || { cat versions.txt; say "TRIPWIRE FAIL (import)"; finish 9; }
grep -q "^e4b /root/b511/e4b/" versions.txt || { cat versions.txt; say "TRIPWIRE: experts4bit_qlora does not resolve to the clone"; finish 9; }
grep -q "^torch $TV " versions.txt || { cat versions.txt; say "TRIPWIRE: torch changed during install"; finish 9; }
cat versions.txt >> summary.txt

FILES="tests/test_decode_graph_buckets.py tests/test_capture_quantized_moe.py tests/test_int4_attn_pack.py tests/test_hybrid_cold_dest.py tests/test_capture.py tests/test_fast_lora.py tests/test_router_epilogue.py"
left=$(( B511_DEADLINE_EPOCH - $(date +%s) - 300 )); [ "$left" -lt 300 ] && { say "STOP: no time left before arm A"; finish 30; }
say "arm A: $FILES"
(cd $W/e4b && perl -e 'alarm 1500; exec @ARGV' python -m pytest $FILES -v -rs -p no:cacheprovider --junitxml=$W/armA.xml) > armA.txt 2>&1; arc=$?
tail -n 3 armA.txt | tee -a summary.txt
say "arm A pytest rc=$arc"
[ -s armA.xml ] || { say "arm A wrote no junit xml"; finish 9; }

left=$(( B511_DEADLINE_EPOCH - $(date +%s) - 180 )); [ "$left" -lt 180 ] && { say "STOP: no time left before arm M"; finish 31; }
say "arm M: the registered mutation, then the replay-vs-padded-eager test alone"
cp mut_b511.py e4b/ && (cd $W/e4b && python mut_b511.py) >> summary.txt 2>&1 || { say "mutation did not apply"; finish 9; }
(cd $W/e4b && perl -e 'alarm 900; exec @ARGV' python -m pytest tests/test_decode_graph_buckets.py -v -rs -p no:cacheprovider -k padded_eager --junitxml=$W/armM.xml) > armM.txt 2>&1; mrc=$?
tail -n 3 armM.txt | tee -a summary.txt
say "arm M pytest rc=$mrc (the registered reading is a FAIL here)"
git -C e4b checkout -q -- . && rm -f e4b/mut_b511.py
[ -s armM.xml ] || { say "arm M wrote no junit xml"; finish 9; }
finish 0
