#!/bin/bash
# bench/p64/p64_run.sh -- lane P64, BOX side (bench/p64/P64-PREREG.md; e4b#709). Started detached by p64_drive.sh with
# the run's nonce; P64_RUN_NONCE first, then P64_EXIT_CODE.<nonce> + TP_DONE.<nonce> on every exit and
# P64_SUCCESS.<nonce> only when the registered read can be made.
#
# What the int4 serve lane's int8 activation step costs at decode: ONE served stack (Qwen3-30B-A3B, P55x's licensed
# calibrated int4 expert pack loaded by fingerprint + calibrated int4 attention + folds + router epilogue, bo6c's
# configuration), scored decode-shaped at B=1 under passes that differ only in the arithmetic of the T == 1 forwards
# (kl_a16.py: a8 = shipped, a16 = E4B_INT4_DECODE_A16, a16_all, the determinism control, the arithmetic-order floor),
# on two texts (wikitext-2 test, C4 validation shard 1: step_decomp._k8_window's rows), 16 rows x 128 decode positions
# each; the NF4 anchor in its own process. p64_reduce.py reads KL and NLL on the box. Nothing here changes a default.
#
# Knobs (every one recorded in summary.txt; any value off its registered default marks the run a REHEARSAL, NOT a
# reading): P64_MODEL P64_REVISION P64_LICENSED_FP P64_GPU_CLASS P64_MIN_DISK_GB P64_CALIB_NSEQ P64_HESSIAN_BUDGET_GB
# P64_BUILD_PPL_STEPS P64_BUILD_VIA P64_ROWS P64_SKIP_INSTALL P64_REHEARSAL.
# P64_PROVE=1 is the PROVING RUN (P64-PREREG "Box and cost"): refusals, the install at the pins, the tripwire, the
# scorer's self-test, K0, the flag on this card's real kernels and an egress probe -- no model, no pack, no KL; exit 0.
set -uo pipefail
W=/root/p64; mkdir -p $W/logs; cd $W || exit 78
say(){ echo "[$(date -u +%FT%TZ)] p64: $*"; }
NONCE=${P64_RUN_NONCE:?}; printf '%s\n' "$NONCE" > $W/P64_RUN_NONCE.tmp && mv $W/P64_RUN_NONCE.tmp $W/P64_RUN_NONCE
finish(){ local rc=$1; printf '%s\n' "$rc" > P64_EXIT_CODE.$NONCE; [ "$rc" = 0 ] && : > P64_SUCCESS.$NONCE; say "TP_DONE rc=$rc"; : > TP_DONE.$NONCE; exit "$rc"; }
trap 'finish 130' INT TERM
for v in P64_RUN_ID P64_DEADLINE_EPOCH P64_INSTANCE_ID E4B_SHA GNF4_SHA; do [ -n "${!v:-}" ] || { say "refusing: $v unset"; finish 78; }; done
for v in E4B_SHA GNF4_SHA; do case "${!v}" in *[!0-9a-f]*|"") say "refusing: $v is not hex"; finish 78;; esac; done
[ ${#E4B_SHA} -eq 40 ] && [ ${#GNF4_SHA} -eq 40 ] || { say "refusing: a pin is not a 40-char sha"; finish 78; }
# ---- registered defaults (P64-PREREG "Box and cost"); a rehearsal overrides them and says so
D_MODEL=Qwen/Qwen3-30B-A3B; D_REV=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
D_FP=sha256:0c9955a9f06d83269050d1fc64571a2a0e78abd48b288fd422e4eef72ccbcd42     # P55x's licensed pack
MID=${P64_MODEL:-$D_MODEL}; REV=${P64_REVISION:-$D_REV}; LIC_FP=${P64_LICENSED_FP:-$D_FP}
GPU_CLASS=${P64_GPU_CLASS:-5090}; MIN_DISK_GB=${P64_MIN_DISK_GB:-200}; NSEQ=${P64_CALIB_NSEQ:-128}
HBUDGET=${P64_HESSIAN_BUDGET_GB:-24}; BUILD_STEPS=${P64_BUILD_PPL_STEPS:-2048}; BUILD_VIA=${P64_BUILD_VIA:-step_decomp}
ROWS=${P64_ROWS:-16}; SKIP_INSTALL=${P64_SKIP_INSTALL:-0}; REHEARSAL=${P64_REHEARSAL:-0}
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$W/hook${PYTHONPATH:+:$PYTHONPATH} E4B_INT4_GPTQ_DEVICE=cuda E4B_INT4_HESSIAN_BUDGET_GB=$HBUDGET
export E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID=$MID
unset E4B_INT4_DECODE_A16    # the served stack is built with the shipped default; kl_a16.py flips the flag per pass
: > summary.txt; echo "$P64_INSTANCE_ID" > INSTANCE_ID
{ echo "KNOBS model=$MID rev=$REV licensed_fp=$LIC_FP gpu_class=$GPU_CLASS min_disk_gb=$MIN_DISK_GB calib_nseq=$NSEQ hessian_budget_gb=$HBUDGET build_ppl_steps=$BUILD_STEPS build_via=$BUILD_VIA rows=$ROWS skip_install=$SKIP_INSTALL"; } | tee -a summary.txt
if [ "$REHEARSAL" != 0 ] || [ "$MID" != "$D_MODEL" ] || [ "$REV" != "$D_REV" ] || [ "$LIC_FP" != "$D_FP" ] || [ "$GPU_CLASS" != 5090 ] \
   || [ "$NSEQ" != 128 ] || [ "$HBUDGET" != 24 ] || [ "$BUILD_STEPS" != 2048 ] || [ "$BUILD_VIA" != step_decomp ] || [ "$ROWS" != 16 ] || [ "$SKIP_INSTALL" != 0 ]; then
  echo "REHEARSAL -- NOT a reading: a knob is off its registered default (see KNOBS)" | tee -a summary.txt; : > REHEARSAL
fi
# ---- staged pieces, byte-for-byte
for f in kl_a16.py p64_reduce.py kl_b16.py p59_reduce.py serve_stack.py kl_fidelity.py step_decomp.py k8_bake.py calib.json hook/usercustomize.py staged.sha256; do
  [ -s $W/$f ] || { say "STAGE MISSING: $f"; finish 9; }; done
(cd $W && sha256sum -c staged.sha256 >/dev/null) || { say "STAGED FILES DIFFER FROM bench/p64/staged.sha256"; finish 9; }
# ---- refusals before anything is fetched: the card class and the disk the working set needs (host-limited: 13)
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "DUD BOX"; finish 10; }
nvidia-smi --query-gpu=name,memory.total,driver_version,uuid --format=csv,noheader | tee forensics.txt
nvidia-smi --query-gpu=power.limit,clocks.max.sm --format=csv,noheader | sed "s/^/power.limit,clocks.max.sm /" | tee -a forensics.txt
lscpu | grep -E "Model name" | tee -a forensics.txt; free -g | head -2 | tee -a forensics.txt; df -h $W | tail -1 | tee -a forensics.txt
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *"$GPU_CLASS"*) ;; *) say "REFUSED: card is '$GPU_NAME', the lane registers the RTX $GPU_CLASS class"; echo "refused: class $GPU_NAME" > REFUSAL; finish 15;; esac
FREE_GB=$(df -BG --output=avail $W 2>/dev/null | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge "$MIN_DISK_GB" ] || { say "REFUSED: ${FREE_GB:-?} GB free < ${MIN_DISK_GB} GB (the checkpoint + arena + pack + logits; overlay is not machine disk)"; echo "refused: disk ${FREE_GB:-?} GB" > REFUSAL; finish 13; }
# ---- deadline guard (P54's): never start a step that cannot finish 10 min before teardown
can_run(){ local need=$1 now; now=$(date +%s); [ $((now + need + 600)) -le "$P64_DEADLINE_EPOCH" ] || { say "STOP: $2 needs ${need}s, only $((P64_DEADLINE_EPOCH - now))s left -- skipped (host-limited)"; echo "SKIPPED $2 host-limited deadline" >> summary.txt; return 1; }; }
arm_alarm(){ local left=$(( P64_DEADLINE_EPOCH - $(date +%s) - 600 )); [ "$left" -lt 600 ] && left=600; echo "$left"; }
# ---- install: e4b pinned + P37's toolchain pins (image python); gnf4 at the registered cut
export DEBIAN_FRONTEND=noninteractive
if [ "$SKIP_INSTALL" = 0 ]; then
  say "install e4b @$E4B_SHA (image python; P37 pins) + gnf4 @$GNF4_SHA"
  perl -e 'alarm 1800; exec @ARGV' python -m pip install -q --no-input --prefer-binary "git+https://github.com/pjordanandrsn/experts4bit-qlora.git@$E4B_SHA" \
    "transformers==5.16.1" "bitsandbytes==0.50.1" datasets accelerate sentencepiece tiktoken safetensors "huggingface_hub>=0.23" > logs/pip_e4b.log 2>&1 || { tail -4 logs/pip_e4b.log; say "PIP FAIL (e4b)"; finish 9; }
  perl -e 'alarm 900; exec @ARGV' python -m pip install -q --no-input --force-reinstall --no-deps "git+https://github.com/pjordanandrsn/grouped-nf4-gemm.git@$GNF4_SHA" > logs/pip_gnf4.log 2>&1 || { tail -3 logs/pip_gnf4.log; say "PIP FAIL (gnf4)"; finish 9; }
else
  say "install SKIPPED (P64_SKIP_INSTALL=1): the importable packages are whatever the environment provides -- recorded below"
fi
python - <<'PYT' || { say "TRIPWIRE FAIL"; finish 9; }
import site, os, inspect, importlib.metadata as md
assert site.ENABLE_USER_SITE, "usercustomize would not load (venv?)"
from experts4bit_qlora.engines import hot_residency as hr
assert hasattr(hr, "DECODE_A16") and hasattr(hr, "_decode_a16_default"), "e4b cut lacks E4B_INT4_DECODE_A16 (P64): the lever cannot run"
assert hr.DECODE_A16[0] is False, "DECODE_A16 is on at import: the served stack must be built with the shipped default"
from experts4bit_qlora.engines.int4_attn import Int4Linear
assert Int4Linear.GEMV_ROWS_MAX == 1, f"Int4Linear.GEMV_ROWS_MAX={Int4Linear.GEMV_ROWS_MAX}: not the route this lane registers"
from experts4bit_qlora.engines.int4_experts import enable_serve_experts_int4_calibrated as f
for k in ("artifact_dir", "expected_fingerprint", "dump_artifact_dir"):
    assert k in inspect.signature(f).parameters, f"e4b cut lacks the #405 knob {k!r}"
from experts4bit_qlora.engines.pack_manifest import verify_artifact  # noqa: F401
import int4_b32, int4_pack_ref  # noqa: F401
assert callable(int4_b32.quant_x_rows) and callable(int4_pack_ref.dequant_int4_ref)
import usercustomize  # noqa: F401
import experts4bit_qlora as e, torch, triton, transformers
gnf4 = md.version("grouped-nf4-gemm")
open("/root/p64/versions.txt", "w").write(f"e4b {e.__version__} @{os.environ['E4B_SHA']} (from {os.path.dirname(e.__file__)})\ngnf4 {gnf4} @{os.environ['GNF4_SHA']} (int4_b32 from {int4_b32.__file__})\n"
                                          f"torch {torch.__version__}\ntriton {triton.__version__}\ntransformers {transformers.__version__}\nbitsandbytes {md.version('bitsandbytes')}\n")
print("tripwire OK: e4b", e.__version__, "gnf4", gnf4, "torch", torch.__version__, "triton", triton.__version__)
PYT
cat versions.txt | tee -a summary.txt
# ---- the scorer's own CPU self-test (alignment, phase census, NLL index) and K0 (no KL row without it, on THIS host)
perl -e 'alarm 600; exec @ARGV' python $W/kl_a16.py --self-test > logs/selftest.log 2>&1 || { tail -5 logs/selftest.log; say "SCORER SELF-TEST FAILED"; finish 21; }
grep -a "self-test OK" logs/selftest.log | tee -a summary.txt
perl -e 'alarm 600; exec @ARGV' python $W/kl_fidelity.py --controls --out $W/k0.json > logs/k0.log 2>&1 || { tail -5 logs/k0.log; say "K0 CONTROLS FAILED -- no KL row is produced"; echo "K0 FAILED" >> summary.txt; finish 16; }
python -c "import json; r=json.load(open('$W/k0.json')); assert r['all_passed']; print('K0 all_passed', r.get('metric_version'))" | tee -a summary.txt || { say "K0 receipt not passing"; finish 16; }
if [ "${P64_PROVE:-0}" = 1 ]; then
  echo "PROVE -- the proving run: no model, no pack, no KL row" | tee -a summary.txt
  say "PROVE: the flag on this card's real kernels (kl_a16.py --prove-flag)"
  perl -e 'alarm 300; exec @ARGV' python -u $W/kl_a16.py --prove-flag --out $W/prove_flag.json > logs/prove_flag.log 2>&1; prc=$?
  grep -a "PROVEFLAG" logs/prove_flag.log | cut -c1-400 | tee -a summary.txt
  [ "$prc" = 0 ] || { tail -5 logs/prove_flag.log; say "PROVE: the flag check FAILED (rc=$prc)"; finish 27; }
  say "PROVE: HF CDN egress probe (50 MB range, 20 s cap; recorded, not a refusal)"
  BPS=$(curl -sSL --max-time 20 -r 0-52428800 -o /dev/null -w '%{speed_download}' https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors 2>/dev/null || echo 0)
  echo "PROVE hf_cdn_mbps=$(python3 -c "print(round(float('${BPS:-0}')/1e6,1))")" | tee -a summary.txt forensics.txt
  : > PROVED; finish 0
fi
# ---- fetch (pinned), bake (bo7's k8_bake.py), prompts (step_decomp's own window, both texts) -- as P59
say "fetch $MID @ $REV"
perl -e "alarm $(arm_alarm); exec @ARGV" python -c "from huggingface_hub import snapshot_download as s; print(s('$MID', revision='$REV', allow_patterns=['*.safetensors','*.json','tokenizer*','*.model','*.txt','merges.txt','vocab.json'], max_workers=4))" > logs/fetch.log 2>&1 || { tail -2 logs/fetch.log; say "DL FAIL"; finish 11; }
bake_fail(){ say "BAKE FAIL ($1)"; python -c "import json; r=json.load(open('$W/work/bake.json')); print('BAKE_ERR', r.get('status'), r.get('step')); print(r.get('err')); print(r.get('tb'))" 2>/dev/null | tee -a logs/bake.log | tail -n 25 | cut -c1-400; finish 12; }
QA=$W/work/nf4.arena
if [ -e "$QA" ] && [ -e "$QA.index.json" ] && python -c "import json,sys; sys.exit(0 if json.load(open('$W/work/bake.json')).get('status')=='OK' else 1)" 2>/dev/null; then
  say "NF4 arena already baked on this box (work/bake.json status OK) -- not re-baked"; echo "BAKE reused (status OK)" >> summary.txt
else
  say "bake NF4 arena"; mkdir -p $W/work
  K8_MODEL="$MID" K8_WORK="$W/work" perl -e "alarm $(arm_alarm); exec @ARGV" python $W/k8_bake.py > logs/bake.log 2>&1 || { tail -3 logs/bake.log; bake_fail rc; }
  [ -e "$QA" ] || bake_fail "no arena"
fi
say "prompt dump: step_decomp._k8_window at B=16 on both texts"
python - "$MID" "$REV" <<'PYP' > logs/prompts.log 2>&1 || { tail -5 logs/prompts.log; say "PROMPT DUMP FAIL"; finish 19; }
import sys, json, hashlib, types
sys.path.insert(0, "/root/p64")
import step_decomp
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], revision=sys.argv[2])
B = 16
for src, label in (("wikitext", "wikitext-2-raw-v1 test via step_decomp._k8_window"), ("c4val1", "allenai/c4 en/c4-validation.00001-of-00008 (first 2000 docs) via step_decomp._k8_window")):
    a = types.SimpleNamespace(ppl_source=src, ppl_chat=False, ppl_chat_suffix="", prompt_offset=0, prompt_span=0, prompt_len=512, batch=B, ppl_steps=0)
    ids, step, prompts, _, _ = step_decomp._k8_window(a, tok)
    assert all(len(p) == 512 for p in prompts) and len(prompts) == B and len(set(tuple(p) for p in prompts)) == B
    rec = {"model": sys.argv[1], "revision": sys.argv[2], "source": label, "ppl_source": src, "batch": B, "prompt_len": 512,
           "corpus_tokens": int(ids.numel()), "row_step": int(step), "prompts_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(), "prompts": prompts}
    json.dump(rec, open(f"/root/p64/prompts_{src}.json", "w"))
    print(f"PROMPTS {src} B={B} corpus={ids.numel()} step={step} sha={rec['prompts_sha256']}")
PYP
grep -a PROMPTS logs/prompts.log | tee -a summary.txt
# P59's wikitext rows, same tokenizer and revision: the digest is on record (f67e7e4d...) -- recorded, not gating
[ "$MID" = "$D_MODEL" ] && { grep -aq "PROMPTS wikitext .* sha=f67e7e4d592b002b70c50e98f12e991a598ef3fdea54adbe3d4c5a2a776b36d2" logs/prompts.log && echo "WIKITEXT ROWS = P59's (f67e7e4d)" >> summary.txt || echo "WIKITEXT ROWS DIFFER from P59's f67e7e4d" >> summary.txt; }

# ---- the pack: P55x's build, verbatim (step_decomp licbuild: the streamed 64k recipe, calibrated attention, folds,
# the wikitext K8 from the live stores, the artifact dumped). Four builds of this recipe on >= 3 boxes gave the same
# bytes (P55x); the fingerprint is the check, never the recipe.
CAL="E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=$NSEQ E4B_CALIB_LAYERS_PER_PASS=10"
fp_of(){ python -c "import json; print(json.load(open('$1/manifest.json'))['pack_fingerprint'])" 2>/dev/null; }
# The library's own check (every payload re-hashed, the root fingerprint recomputed): the pack's validity is THIS,
# never the build process's exit code -- the build also runs P55x's wikitext K8 from the live stores, a cross-check
# that is not an input to anything here (the A2000 rehearsal: the dump completed, then the K8's fp8 paged-KV kernel
# refused sm_86, and the old gate on the exit code threw a complete pack away).
pack_verifies(){ python -c "from experts4bit_qlora.engines.pack_manifest import verify_artifact as v; v('$1', expected_model_revision='$REV')" > logs/verify_$(basename $1).log 2>&1; }
first_chunk_watchdog(){ local pid=$1 log=$2 budget=${P64_FIRST_CHUNK_S:-1500} t0; t0=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    grep -qa "INT4EXP calibrated experts" "$log" 2>/dev/null && { say "build: calibration chunk 1 in $(( $(date +%s) - t0 ))s"; return 0; }
    if [ $(( $(date +%s) - t0 )) -ge "$budget" ]; then
      say "HOST-LIMITED: no calibration chunk in ${budget}s (a good host: ~520s) -- killing the build"; echo "HOSTLIMITED build no calibration chunk in ${budget}s" >> summary.txt
      kill -TERM "$pid" 2>/dev/null; sleep 10; kill -KILL "$pid" 2>/dev/null; return 30
    fi
    sleep 20
  done; return 0; }
if [ -s $W/artifact1/manifest.json ] && pack_verifies $W/artifact1; then
  say "pack already built on this box and it verifies -- not rebuilt"; echo "BUILD reused (verify_artifact OK)" >> summary.txt
else
rm -rf $W/artifact1
can_run 3600 build || finish 20
say "build the pack ($BUILD_VIA) -> artifact1"
{ echo "P64 build via=$BUILD_VIA at=$(date -u +%FT%TZ)"; } > logs/build.log
if [ "$BUILD_VIA" = step_decomp ]; then
  env $CAL E4B_INT4_DUMP_ARTIFACT_DIR=$W/artifact1 E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=1 E4B_SERVE_ATTN_INT4=0 E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1 \
    perl -e "alarm $(arm_alarm); exec @ARGV" python $W/step_decomp.py --model "$MID" --arena "$QA" --calib $W/calib.json \
      --placement-override all-vram --amort off --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps $BUILD_STEPS \
      --b1d-loop eager --no-fuse-qkv --ppl-source wikitext --out $W/build_ppl_wikitext.json >> logs/build.log 2>&1 &
else
  env $CAL E4B_INT4_DUMP_ARTIFACT_DIR=$W/artifact1 E4B_SERVE_EXP_INT4=1 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_SERVE_ATTN_INT4=0 E4B_CALIB_SOURCE=c4 \
    perl -e "alarm $(arm_alarm); exec @ARGV" python -u $W/kl_a16.py --build-only --model "$MID" --arena "$QA" --calib $W/calib.json --out $W/build >> logs/build.log 2>&1 &
fi
pid=$!; brc=0; first_chunk_watchdog "$pid" logs/build.log || brc=$?; wrc=0; wait "$pid" 2>/dev/null || wrc=$?; [ "$brc" = 0 ] && brc=$wrc
grep -aE "K8_PPL|INT4EXP|ATTNINT4|REFUSED|Error" logs/build.log | tail -6 | sed "s/^/    /"
grep -aoE "INT4EXP (calibrated|loaded|assignment)[^|]{0,120}" logs/build.log | tail -2 | sed "s/^/BUILD /" >> summary.txt
grep -aE "K8_PPL" logs/build.log | tail -1 | sed "s/^/BUILD /" >> summary.txt
[ "$brc" = 30 ] && finish 30
if [ -s $W/artifact1/manifest.json ] && pack_verifies $W/artifact1; then
  [ "$brc" = 0 ] || { say "the build process exited rc=$brc AFTER dumping a pack that verifies (the K8 cross-check failed; see logs/build.log) -- continuing on the pack"; echo "BUILD rc=$brc after a verified dump: K8 cross-check failed, pack OK" >> summary.txt; }
else
  say "the build failed (rc=$brc) and left no pack that verifies (logs/verify_artifact1.log) -- there is no pack"; echo "BUILD FAILED rc=$brc" >> summary.txt; finish 20
fi
fi
FP=$(fp_of $W/artifact1); [ -n "$FP" ] || { say "no artifact fingerprint after the build"; echo "NO_FINGERPRINT" >> summary.txt; finish 20; }
if [ "$FP" = "$LIC_FP" ]; then echo "PACK $FP = the licensed pack (P55x)" | tee -a summary.txt; PACK_LICENSED=1
else echo "PACK $FP != the licensed $LIC_FP -- the read is labelled 'not the licensed pack'" | tee -a summary.txt; PACK_LICENSED=0; fi
printf '{"fingerprint": "%s", "licensed_fingerprint": "%s", "licensed": %s}\n' "$FP" "$LIC_FP" "$([ $PACK_LICENSED = 1 ] && echo true || echo false)" > pack.json
rm -rf $W/work/nf4snap 2>/dev/null   # the bake's intermediate snapshot (~16 GB): the arena is what every later step reads

# ---- the served stack, loaded by fingerprint, scored under every registered pass (one process: same bytes throughout)
PROMPTS="wikitext=$W/prompts_wikitext.json,c4val1=$W/prompts_c4val1.json"
LOAD="E4B_SERVE_EXP_INT4=1 E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=$NSEQ E4B_INT4_ARTIFACT_DIR=$W/artifact1 E4B_INT4_EXPECTED_FINGERPRINT=$FP"
rc_any=0; note(){ [ "$rc_any" = 0 ] && rc_any=$1; }
if can_run 1800 served; then
  say "served stack: every pass (kl_a16.py; deadline-aware inside)"
  env $LOAD E4B_SERVE_ATTN_INT4_CALIB=1 E4B_SERVE_ATTN_INT4=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=1 E4B_FUSE_T1_GLUE_R2=1 E4B_FUSE_ROUTER_EPI=1 \
    perl -e "alarm $(arm_alarm); exec @ARGV" python -u $W/kl_a16.py --model "$MID" --arena "$QA" --calib $W/calib.json --prompts "$PROMPTS" \
      --rows $ROWS --prefix 384 --out $W/out --deadline-epoch $P64_DEADLINE_EPOCH --margin-s 900 > logs/run_served.log 2>&1 || note $?
  grep -aE "P64BUILD|P64PASS|P64 SKIPPED|P64 REFUSED|INT4EXP|ATTNINT4|Error|Traceback" logs/run_served.log | tail -20 | cut -c1-300 | sed "s/^/    /"
  grep -aE "P64PASS|P64 SKIPPED|P64 REFUSED" logs/run_served.log | cut -c1-300 >> summary.txt
else note 20; fi
# ---- the NF4 anchor (informational), its own process: no int4, no folds
if can_run 900 anchor; then
  say "NF4 anchor"
  env E4B_SERVE_EXP_INT4=0 E4B_SERVE_ATTN_INT4_CALIB=0 E4B_SERVE_ATTN_INT4=0 E4B_CALIB_SOURCE=c4 E4B_FUSE_T1_GLUE=0 E4B_FUSE_T1_GLUE_R2=0 E4B_FUSE_ROUTER_EPI=0 \
    perl -e "alarm $(arm_alarm); exec @ARGV" python -u $W/kl_a16.py --anchor --model "$MID" --arena "$QA" --calib $W/calib.json --prompts "$PROMPTS" \
      --rows $ROWS --prefix 384 --out $W/out_nf4 --deadline-epoch $P64_DEADLINE_EPOCH --margin-s 600 > logs/run_nf4.log 2>&1 || echo "ANCHOR rc=$?" >> summary.txt
  grep -aE "P64PASS|P64 SKIPPED|P64 REFUSED" logs/run_nf4.log | cut -c1-300 >> summary.txt
  grep -aqE "INT4EXP|ATTNINT4" logs/run_nf4.log && { echo "INT4 LEAKED into the NF4 anchor" >> summary.txt; note 44; }
fi
# ---- reduce on the box (the logits stay here); the reducer applies the registered rule, this script only says whether it ran
say "reduce"; python $W/p64_reduce.py $W --md $W/RESULTS-p64-generated.md --json $W/p64_rep.json > logs/reduce.log 2>&1 || { tail -5 logs/reduce.log; note 43; }
grep -aE "^\*\*(VALIDITY|VERDICT)|^- \*\*" $W/RESULTS-p64-generated.md 2>/dev/null | cut -c1-240 | tee -a summary.txt
# ---- the terminal marker encodes the outcome: the primary pair, the determinism control and a floor sample on BOTH
# texts must exist and have engaged as registered, and the reducer must have produced a verdict it can read.
MISSING=""
for p in a8 a16 a8_rep a8_pc64; do for t in wikitext c4val1; do
  [ -s "$W/out/$p.$t.pt" ] && python -c "import json,sys; sys.exit(0 if json.load(open('$W/out/$p.$t.census.json')).get('engaged_as_registered') else 1)" 2>/dev/null || MISSING="$MISSING $p/$t"
done; done
[ -z "$MISSING" ] || { say "the registered read cannot be made -- missing or not engaged:$MISSING"; echo "READ_INCOMPLETE:$MISSING" >> summary.txt; note 42; }
python -c "import json,sys; r=json.load(open('$W/p64_rep.json')); v=r.get('validity',{}).get('status'); sys.exit(0 if v=='VALID' else 1)" 2>/dev/null || { echo "VALIDITY NOT MET (see RESULTS-p64-generated.md)" >> summary.txt; note 41; }
say "----- summary -----"; cat summary.txt; echo "----- versions -----"; cat versions.txt
finish "$rc_any"
