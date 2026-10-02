#!/bin/bash
# bench/sc1/sglang/one_batch.sh -- SGLang's OWN engine-level decode instrument, labelled native (SC1-PREREG "Validity", SGLang).
#
#   one_batch.sh <model_dir_or_id> <revision> <out_json> <log> [extra args passed through]
#
# Runs python -m sglang.benchmark.one_batch (python/sglang/benchmark/one_batch.py at v0.5.20; `sglang.bench_one_batch` is the
# deprecated shim) with --batch-size 1 16 --input-len 512 --output-len 128 --disable-radix-cache --cuda-graph-bs-decode 1 16
# --moe-runner-backend auto. CLI verified at the tag: BenchArgs.add_cli_args (one_batch.py:229-247) declares --run-name,
# --batch-size (nargs +), --input-len (nargs +), --output-len (nargs +), --prompt-filename, --result-filename (default
# result.jsonl), --log-decode-step, --profile...; every ServerArgs flag is accepted too (cli_main, one_batch.py:1092-1098).
# What it measures (latency_test_run_once, one_batch.py:752-892): prefill_latency / prefill_throughput for the extend, then
# output_len-1 decode steps each wrapped in model_runner.synchronize(); `median_decode_latency` = np.median over those step
# latencies at the FIXED batch size and `median_decode_throughput` = batch_size / that median (lines 877-882);
# total_latency, overall_throughput. One JSON line per (bs, input_len, output_len) is APPENDED to --result-filename
# (lines 1005-1009), after an untimed warmup at the first point with min(32, output_len) steps (lines 925-944).
# Prompts are RANDOM token ids, np.random.randint(0, 10000, (bs, input_len)) (prepare_synthetic_inputs_for_latency_test,
# lines 456-480); --prompt-filename takes TEXT and re-tokenises (latency_test, `tokenizer.encode(p.strip())`), so the SC1
# prompt ids CANNOT be fed -- this row is labelled native/random-ids and is never ratioed against a prompt-file arm.
# main() merges decode.max_bs = max(batch_size) = 16 into cuda_graph_config (one_batch.py:1016-1046); the explicit
# --cuda-graph-bs-decode 1 16 capture list survives because that JSON only sets max_bs (cuda_graph_hook.py:37-105:
# convenience flags first, JSON keys override per key). No HTTP server, no scheduler: the GPU must be free -- stop the
# server first (server.sh sglang_server_stop).
set -uo pipefail
MODEL=${1:?usage: one_batch.sh <model> <revision> <out_json> <log> [extra...]}; REV=${2:?}; OUT=${3:?}; LOG=${4:?}; shift 4
SGLANG_WORK=${SGLANG_WORK:-/root/sc1}
SGLANG_VENV=${SGLANG_VENV:-$SGLANG_WORK/venv-sglang}
SGLANG_ENV_FILE=${SGLANG_ENV_FILE:-$SGLANG_WORK/sglang.env}
ALARM=${SGLANG_ONE_BATCH_ALARM:-1800}
[ -f "$SGLANG_ENV_FILE" ] && . "$SGLANG_ENV_FILE"
export SGLANG_CACHE_DIR=${SGLANG_CACHE_DIR:-$SGLANG_WORK/sglang-cache} SGLANG_JIT_CACHE_DIR=${SGLANG_JIT_CACHE_DIR:-$SGLANG_WORK/sglang-cache/jit}
export PATH=$SGLANG_VENV/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH
[ -x "$SGLANG_VENV/bin/python" ] || { echo "one_batch: no venv at $SGLANG_VENV"; exit 78; }
JSONL=${OUT%.json}.jsonl; rm -f -- "${JSONL:?}"
CMD=("$SGLANG_VENV/bin/python" -m sglang.benchmark.one_batch --model-path "$MODEL" --revision "$REV"
     --batch-size 1 16 --input-len 512 --output-len 128 --disable-radix-cache --cuda-graph-bs-decode 1 16 --moe-runner-backend auto
     --dtype bfloat16 --random-seed 0 --run-name sc1_sglang_native --result-filename "$JSONL" "$@")
{ echo "# sc1 sglang one_batch at=$(date -u +%FT%TZ) alarm=${ALARM}s"; echo "# cmd: ${CMD[*]}"; } > "$LOG"
echo "[$(date -u +%FT%TZ)] one_batch: ${CMD[*]}"
perl -e "alarm $ALARM; exec @ARGV" "${CMD[@]}" 2>&1 | tee -a "$LOG" | grep -aE "Prefill\.|Decode\.|Total\.|skipping|Error|error|Warmup|Benchmark" | cut -c1-200
rc=${PIPESTATUS[0]}
SC1_RC=$rc SC1_JSONL=$JSONL SC1_OUT=$OUT SC1_LOG=$LOG SC1_CMD="${CMD[*]}" SC1_MODEL=$MODEL SC1_REV=$REV "$SGLANG_VENV/bin/python" - <<'PYT'
import json, os
rows = []
try:
    rows = [json.loads(l) for l in open(os.environ["SC1_JSONL"]) if l.strip()]
except OSError:
    pass
by_bs = {}
for r in rows:
    if r.get("input_len") == 512 and r.get("output_len") == 128 and "median_decode_latency" in r:
        bs = int(r["batch_size"])
        by_bs[str(bs)] = {"decode_ms_per_step": round(r["median_decode_latency"] * 1e3, 4),
                          "decode_tok_s": round(r["median_decode_throughput"], 1),
                          "prefill_latency_s": round(r["prefill_latency"], 5), "prefill_tok_s": round(r["prefill_throughput"], 1),
                          "total_latency_s": round(r["total_latency"], 4), "overall_tok_s": round(r["overall_throughput"], 1)}
text = open(os.environ["SC1_LOG"], errors="replace").read()
out = {"engine": "sglang", "arm": "one_batch_native", "instrument": "python -m sglang.benchmark.one_batch (v0.5.20)",
       "label": "native: engine-level, no server/scheduler, RANDOM token ids in [0, 10000) -- not the SC1 prompt file; informational, never ratioed",
       "model": os.environ["SC1_MODEL"], "revision": os.environ["SC1_REV"], "rc": int(os.environ["SC1_RC"]), "cmd": os.environ["SC1_CMD"],
       "method": "median over the output_len-1 synchronised decode steps at a fixed batch size (one_batch.py:752-892)",
       "input_len": 512, "output_len": 128, "batch_sizes": [1, 16], "by_batch": by_bs, "rows": rows,
       "gptq_marlin_banner": "The model is convertible to gptq_marlin during runtime. Using gptq_marlin kernel." in text,
       "jit_build_failures": [l for l in text.splitlines() if "Failed to build JIT module" in l][:3],
       "status": ("ok" if by_bs and int(os.environ["SC1_RC"]) == 0 else ("alarm" if int(os.environ["SC1_RC"]) == 142 else "failed"))}
json.dump(out, open(os.environ["SC1_OUT"], "w"), indent=1)
print("SC1_SGLANG_ONE_BATCH " + json.dumps({"status": out["status"], "by_batch": by_bs, "rc": out["rc"]}))
PYT
exit "$rc"
