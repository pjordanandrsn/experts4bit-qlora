#!/bin/bash
# Resilient OLMoE-1B-7B QLoRA expert-offload A/B supervisor.
#
# Resilience properties:
#   * Detached: launched via `setsid nohup … </dev/null &` so it lives in its own
#     session — a TaskStop / launcher-shell exit does NOT cascade to training
#     (that was the earlier failure: trainer sat in the bg-task's process group).
#   * Persistent: all state under /home/node/work (the durable deliverable volume),
#     NOT /tmp scratch — survives a scratch wipe.
#   * Idempotent/resumable: each phase gated by a state/<phase>.done sentinel that is
#     written ONLY after a valid "summary" record lands in the phase JSONL. Re-running
#     the supervisor skips completed phases — so a restart resumes instead of redoing.
#   * Self-healing: a phase that exits WITHOUT a summary (OOM, GPU reclaim, crash) is
#     retried up to MAX_ATTEMPTS with backoff; the JSONL is truncated per attempt so a
#     partial+retry never concatenates into a corrupt curve.
#   * GPU-aware: waits for enough free VRAM before each phase (shared card).
set -u

BASE=/home/node/work/experts4bit-qlora/ab-telemetry
PY=/home/node/e4b-venv/bin/python
REPO=/home/node/work/experts4bit-qlora
MAX_ATTEMPTS=3
NEED_MIB=7000        # require >= this much free VRAM before launching a phase
cd "$REPO" || exit 1
mkdir -p "$BASE/runs" "$BASE/logs" "$BASE/state" "$BASE/wandb" "$BASE/charts"

SUPLOG="$BASE/supervisor.log"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$SUPLOG"; }

# Common training config — same for both phases (seeded, so data order is identical).
COMMON=(MODEL=allenai/OLMoE-1B-7B-0924 STEPS=60 EVAL_EVERY=20 SEQ=256 N_TRAIN=3000
        GRAD_ACCUM=4 LR=1e-4 R=8 ALPHA=16 DO_GEN=0
        WANDB=1 WANDB_MODE=offline WANDB_DIR="$BASE/wandb" WANDB_SILENT=true)

wait_gpu() {  # wait until >= NEED_MIB free, up to ~15 min, then proceed anyway
    local i tot used free
    for i in $(seq 1 30); do
        read -r tot used < <(nvidia-smi --query-gpu=memory.total,memory.used \
            --format=csv,noheader,nounits 2>/dev/null | head -1 | tr ',' ' ')
        [ -z "${tot:-}" ] && { log "  gpu query failed, proceeding"; return 0; }
        free=$((tot - used))
        if [ "$free" -ge "$NEED_MIB" ]; then log "  GPU ok: ${free} MiB free"; return 0; fi
        log "  GPU busy: ${free} MiB free (< ${NEED_MIB}), poll ${i}/30"; sleep 30
    done
    log "  GPU wait capped; proceeding anyway"; return 0
}

run_phase() {  # $1=phase(off|on)  $2=OFFLOAD_EXPERTS(0|1)  $3=run name
    local phase=$1 offload=$2 runname=$3
    local jsonl="$BASE/runs/olmoe_${phase}.jsonl"
    local plog="$BASE/logs/olmoe_${phase}.log"
    local done="$BASE/state/${phase}.done"
    if [ -f "$done" ]; then log "PHASE ${phase}: already complete (sentinel present) — skip"; return 0; fi
    local attempt
    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        log "PHASE ${phase} attempt ${attempt}/${MAX_ATTEMPTS}: awaiting GPU"
        wait_gpu
        rm -f "$jsonl"      # fresh file per attempt — never concatenate partials
        log "PHASE ${phase} attempt ${attempt}: launching (offload=${offload})"
        env "${COMMON[@]}" OFFLOAD_EXPERTS="$offload" RUN_NAME="$runname" \
            METRICS_JSONL="$jsonl" OUT="$BASE/out-${phase}" \
            "$PY" -m experts4bit_qlora.train > "$plog" 2>&1
        local ec=$?
        if grep -q '"event": "summary"' "$jsonl" 2>/dev/null; then
            log "PHASE ${phase} attempt ${attempt}: summary present (exit=${ec}) -> DONE"
            touch "$done"; return 0
        fi
        log "PHASE ${phase} attempt ${attempt}: FAILED (exit=${ec}, no summary); backoff $((attempt*20))s"
        sleep $((attempt * 20))
    done
    log "PHASE ${phase}: EXHAUSTED ${MAX_ATTEMPTS} attempts — giving up"
    return 1
}

log "=== supervisor start (pid $$) — full matched A/B, 60 steps each ==="
if run_phase off 0 olmoe-offload-off && run_phase on 1 olmoe-offload-on; then
    touch "$BASE/state/ab.done"
    log "=== AB_DONE: both phases complete ==="
else
    log "=== AB_INCOMPLETE: a phase failed after retries ==="
fi
