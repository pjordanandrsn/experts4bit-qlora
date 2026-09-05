#!/bin/bash
# p37c (P37 amendment 3, pre-registered 2026-09-05 20:30Z): the registered K8 gate on THIS box's re-derived 64k pack. The speed
# lane's e4b licensed arms are VOID under the fingerprint rule (11522/766 vs the licensed 11512/776); the direct test of the rule's
# intent is the gate itself: nf4 and the full stack on wikitext and c4val1, bo6c's k8() verbatim. Verdict rule (unchanged):
# ppl(all) - ppl(nf4) <= +0.05 on BOTH texts => LICENSED (gate_verdict.json); else VOID stands. Nothing else is re-run. Marker TP3_DONE.
set -uo pipefail
LANE=p37; W=/root/$LANE; cd $W
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W/hook E4B_INT4_GPTQ_DEVICE=cuda E4B_INT4_HESSIAN_BUDGET_GB=24
say(){ echo "[$(date -u +%FT%TZ)] p37c: $*"; }
fenv(){ case "$1" in 0) echo "0 0 0";; r1) echo "1 0 0";; r12) echo "1 1 0";; epi) echo "0 0 1";; r1epi) echo "1 0 1";; all) echo "1 1 1";; *) echo "0 0 0";; esac; }
# k8 TAG ARM EXP CALIBATTN FUSE MODEL ARENA [extra env...]   (bo6c's helper, verbatim apart from the log directory)
k8(){ local TAG=$1 ARM=$2 EXP=$3 CA=$4 FU=$5 MID=$6 AR=$7; shift 7; read G R E <<<"$(fenv $FU)"
  say "K8 $TAG/$ARM (exp=$EXP calib=$CA fuse=$FU src=${PPLSRC:-wikitext} $*)"
  env "$@" E4B_RECOMPILE_LIMIT=64 E4B_ACCUM_RECOMPILE_LIMIT=64 E4B_MODEL_ID="$MID" E4B_SERVE_EXP_INT4=$EXP E4B_SERVE_ATTN_INT4_CALIB=$CA E4B_CALIB_SOURCE=c4 \
      E4B_FUSE_T1_GLUE=$G E4B_FUSE_T1_GLUE_R2=$R E4B_FUSE_ROUTER_EPI=$E \
    perl -e 'alarm 5400; exec @ARGV' python $W/step_decomp.py --model "$MID" --arena "$AR" --calib $W/calib.json --placement-override all-vram --amort off \
      --batch 1 --prompt-len 512 --gen-tokens 16 --ppl-steps 2048 --b1d-loop eager --no-fuse-qkv --ppl-source ${PPLSRC:-wikitext} --out $W/${TAG}_ppl_${ARM}_${PPLSRC:-wikitext}.json > logs/run_${TAG}_ppl_${ARM}_${PPLSRC:-wikitext}.log 2>&1
  local rc=$?; grep -aE "K8_PPL|INT4EXP calibrated experts|ATTNINT4|REFUSED|Error" logs/run_${TAG}_ppl_${ARM}_${PPLSRC:-wikitext}.log | tail -8 | sed "s/^/    /"
  { echo -n "k8 $TAG/$ARM src=${PPLSRC:-wikitext} rc=$rc "; grep -aE "K8_PPL" logs/run_${TAG}_ppl_${ARM}_${PPLSRC:-wikitext}.log | tail -1 | cut -c1-300; echo; } >> summary.txt; }
Q=Qwen/Qwen3-30B-A3B; QA=$W/work_qwen3/nf4.arena; [ -e "$QA" ] || { say "NO ARENA at $QA"; touch TP3_DONE; exit 9; }
[ -f TP_DONE ] || { say "speed lane not finished"; exit 9; }
echo "AMENDMENT 3 (p37c): K8 gate on this box's re-derived pack; verdict = ppl(all)-ppl(nf4) <= +0.05 on wikitext AND c4val1" | tee -a summary.txt
PPLSRC=wikitext k8 qwen3 nf4 0 0 0 $Q $QA
PPLSRC=c4val1   k8 qwen3 nf4 0 0 0 $Q $QA
PPLSRC=wikitext k8 qwen3 all 1 1 all $Q $QA E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=128
PPLSRC=c4val1   k8 qwen3 all 1 1 all $Q $QA E4B_SERVE_EXP_INT4_CALIB=1 E4B_CALIB_NSEQ=128
say "verdict"
python - <<'PYV' | tee -a summary.txt
import json, os, re
W = "/root/p37"
def ppl(arm, src):
    p = f"{W}/qwen3_ppl_{arm}_{src}.json"
    if not os.path.exists(p): return None, "missing receipt"
    d = json.load(open(p))
    for k in ("ppl", "k8_ppl", "perplexity"):
        if k in d: return float(d[k]), "ok"
    m = re.search(r"K8_PPL[^\n]*?ppl[=: ]+([0-9.]+)", open(f"{W}/logs/run_qwen3_ppl_{arm}_{src}.log", errors="ignore").read())
    return (float(m.group(1)), "from log") if m else (None, "no ppl field")
out = {"rule": "ppl(all) - ppl(nf4) <= +0.05 on wikitext AND c4val1", "texts": {}}
ok = True
for src in ("wikitext", "c4val1"):
    a, sa = ppl("all", src); n, sn = ppl("nf4", src)
    d = (a - n) if (a is not None and n is not None) else None
    v = ("PASS" if d is not None and d <= 0.05 else "FAIL")
    ok = ok and v == "PASS"
    out["texts"][src] = {"nf4": n, "all": a, "delta": d, "verdict": v, "sources": [sn, sa]}
out["verdict"] = "LICENSED" if ok else "NOT LICENSED (VOID stands)"
json.dump(out, open(f"{W}/gate_verdict.json", "w"), indent=1); print("GATE", json.dumps(out))
PYV
say "P37C DONE"; touch TP3_DONE
