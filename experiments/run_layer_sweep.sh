#!/bin/bash
# Comprehensive layer depth sweep for all models
# Estimated time: 6-8 hours

set -e  # Exit on error
set -o pipefail  # Exit on pipe failures too

LOG_FILE="layer_sweep_$(date +%Y%m%d_%H%M%S).log"
CACHE_DIR="/workspace/.cache/huggingface/hub"

log() {
    echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG_FILE"
}

clear_cache() {
    log "Clearing HuggingFace cache to free storage..."
    rm -rf "$CACHE_DIR"/*
    log "✓ Cache cleared"
}

log "=========================================="
log "LAYER DEPTH SWEEP - STARTING"
log "=========================================="
log ""

# Models to test
QWEN_MODELS=("qwen3-4b" "qwen3-8b" "qwen3-14b" "qwen3-4b-base" "qwen3-8b-base" "qwen3-14b-base")
LLAMA_MODELS=("llama2-7b" "llama2-13b" "llama2-7b-chat" "llama2-13b-chat")

# Layer depths to test (including 0.75 to regenerate with new datasets)
DEPTHS=("0.25" "0.375" "0.50" "0.625" "0.75" "0.875")

# Batch size to prevent RAM overflow
BATCH_SIZE=6

# ==========================================
# STEP 1: Extract vectors for missing models
# ==========================================

log "STEP 1: Extracting probe vectors for missing models"
log ""

# Check Qwen models
for model in "${QWEN_MODELS[@]}"; do
    if [ ! -f "output/comparison/$model/probe_vectors.jsonl" ]; then
        clear_cache
        log "Extracting vectors: $model (batch-size=$BATCH_SIZE)"
        if python experiments/extract_vectors.py \
            --model "$model" \
            --batch-size $BATCH_SIZE \
            2>&1 | tee -a "$LOG_FILE"; then
            log "✓ Completed: $model"
        else
            log "✗ FAILED: $model - check log for errors"
            exit 1
        fi
        log ""
    else
        log "⊘ Skipping $model (vectors already exist)"
        log ""
    fi
done

# Check LLaMA models
for model in "${LLAMA_MODELS[@]}"; do
    if [ ! -f "output/comparison/$model/probe_vectors.jsonl" ]; then
        clear_cache
        log "Extracting vectors: $model (batch-size=$BATCH_SIZE)"
        if python experiments/extract_vectors.py \
            --model "$model" \
            --batch-size $BATCH_SIZE \
            2>&1 | tee -a "$LOG_FILE"; then
            log "✓ Completed: $model"
        else
            log "✗ FAILED: $model - check log for errors"
            exit 1
        fi
        log ""
    else
        log "⊘ Skipping $model (vectors already exist)"
        log ""
    fi
done

log ""
log "=========================================="
log "STEP 2: Running layer depth evaluations"
log "=========================================="
log ""
log "Using MULTI-LAYER mode: Extracting all 6 depths in ONE pass per model"
log "  Depths: 25%, 37.5%, 50%, 62.5%, 75%, 87.5%"
log "  Expected speedup: ~4-5× faster than sequential extraction"
log ""

# Counter for progress
TOTAL_MODELS=$((${#QWEN_MODELS[@]} + ${#LLAMA_MODELS[@]}))
CURRENT=0

# ==========================================
# Qwen3 Models
# ==========================================

log "Testing Qwen3 models..."
log ""

for model in "${QWEN_MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    log "[$CURRENT/$TOTAL_MODELS] Evaluating: $model (all 6 depths)"

    # Check if any layer already exists (simple heuristic)
    if [ -d "output/comparison/$model/layer_h9" ] && \
       [ -d "output/comparison/$model/layer_h13" ] && \
       [ -d "output/comparison/$model/layer_h18" ] && \
       [ -d "output/comparison/$model/layer_h22" ] && \
       [ -d "output/comparison/$model/layer_h27" ] && \
       [ -d "output/comparison/$model/layer_h31" ]; then
        log "  ⊘ Skipping (all layers already exist)"
    else
        clear_cache

        if python experiments/evaluate_vectors.py \
            --model "$model" \
            --all-layers \
            --batch-size $BATCH_SIZE \
            --eval-limit 2000 \
            2>&1 | tee -a "$LOG_FILE"; then
            log "  ✓ Completed all 6 layers"
        else
            log "  ✗ FAILED - check log for errors"
            exit 1
        fi
    fi
    log ""
done

# ==========================================
# LLaMA-2 Models
# ==========================================

log "Testing LLaMA-2 models..."
log ""

for model in "${LLAMA_MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    log "[$CURRENT/$TOTAL_MODELS] Evaluating: $model (all 6 depths)"

    # Check if any layer already exists (approximate layer numbers)
    # LLaMA-7B: h8, h12, h16, h20, h24, h28
    # LLaMA-13B: h10, h15, h20, h25, h30, h35
    if [[ "$model" == *"7b"* ]]; then
        LAYER_CHECK_DIRS="layer_h8 layer_h12 layer_h16 layer_h20 layer_h24 layer_h28"
    else
        LAYER_CHECK_DIRS="layer_h10 layer_h15 layer_h20 layer_h25 layer_h30 layer_h35"
    fi

    ALL_EXIST=true
    for layer_dir in $LAYER_CHECK_DIRS; do
        if [ ! -d "output/comparison/$model/$layer_dir" ]; then
            ALL_EXIST=false
            break
        fi
    done

    if [ "$ALL_EXIST" = true ]; then
        log "  ⊘ Skipping (all layers already exist)"
    else
        clear_cache

        if python experiments/evaluate_vectors.py \
            --model "$model" \
            --all-layers \
            --batch-size $BATCH_SIZE \
            --eval-limit 2000 \
            2>&1 | tee -a "$LOG_FILE"; then
            log "  ✓ Completed all 6 layers"
        else
            log "  ✗ FAILED - check log for errors"
            exit 1
        fi
    fi
    log ""
done

# ==========================================
# Summary
# ==========================================

log ""
log "=========================================="
log "LAYER DEPTH SWEEP - COMPLETED"
log "=========================================="
log ""
log "Results saved in layer-specific subdirectories:"
log "  output/comparison/{model}/layer_h{N}/probe_evaluate-v2.jsonl"
log ""
log "Next steps:"
log "  1. Visualize results: python experiments/visualize.py --model {model}"
log "  2. Script will auto-select best layer across all tested depths"
log ""
log "Full log saved to: $LOG_FILE"
