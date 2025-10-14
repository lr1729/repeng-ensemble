#!/bin/bash
# Comprehensive layer depth sweep for all models
# Estimated time: 6-8 hours

set -e  # Exit on error

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
BATCH_SIZE=8

# ==========================================
# STEP 1: Extract vectors for missing models
# ==========================================

log "STEP 1: Extracting probe vectors for missing models"
log ""

for model in "llama2-7b" "llama2-7b-chat"; do
    if [ ! -f "output/comparison/$model/probe_vectors.jsonl" ]; then
        clear_cache
        log "Extracting vectors: $model (batch-size=$BATCH_SIZE)"
        python experiments/extract_vectors.py \
            --model "$model" \
            --batch-size $BATCH_SIZE \
            2>&1 | tee -a "$LOG_FILE"
        log "✓ Completed: $model"
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

# Counter for progress
TOTAL_EVALS=$((${#QWEN_MODELS[@]} * ${#DEPTHS[@]} + ${#LLAMA_MODELS[@]} * ${#DEPTHS[@]}))
CURRENT=0

# Track current loaded model to avoid unnecessary cache clears
CURRENT_LOADED_MODEL=""

# ==========================================
# Qwen3 Models
# ==========================================

log "Testing Qwen3 models at multiple depths..."
log ""

for model in "${QWEN_MODELS[@]}"; do
    # Determine number of layers for this model
    if [ "$model" == "qwen3-14b" ] || [ "$model" == "qwen3-14b-base" ]; then
        NUM_LAYERS=40
    else
        NUM_LAYERS=36
    fi

    for depth in "${DEPTHS[@]}"; do
        CURRENT=$((CURRENT + 1))
        log "[$CURRENT/$TOTAL_EVALS] Evaluating: $model at ${depth} depth"

        TARGET_LAYER=$(python3 -c "import math; print(int($NUM_LAYERS * $depth))")

        # Check if nearby layers already exist
        EXISTS=false
        for offset in -2 -1 0 1 2; do
            CHECK_LAYER=$((TARGET_LAYER + offset))
            if [ -f "output/comparison/$model/layer_h$CHECK_LAYER/probe_evaluate-v2.jsonl" ]; then
                log "  ⊘ Skipping (layer_h$CHECK_LAYER already exists)"
                EXISTS=true
                break
            fi
        done

        if [ "$EXISTS" = false ]; then
            # Clear cache only when switching models
            if [ "$CURRENT_LOADED_MODEL" != "$model" ]; then
                clear_cache
                CURRENT_LOADED_MODEL="$model"
            fi

            python experiments/evaluate_vectors.py \
                --model "$model" \
                --layer-depth "$depth" \
                --batch-size $BATCH_SIZE \
                --eval-limit 2000 \
                2>&1 | tee -a "$LOG_FILE"
            log "  ✓ Completed"
        fi
        log ""
    done
done

# ==========================================
# LLaMA-2 Models
# ==========================================

log "Testing LLaMA-2 models at multiple depths..."
log ""

for model in "${LLAMA_MODELS[@]}"; do
    # LLaMA models have 32 layers (7B) or 40 layers (13B)
    if [[ "$model" == *"7b"* ]]; then
        NUM_LAYERS=32
    else
        NUM_LAYERS=40
    fi

    for depth in "${DEPTHS[@]}"; do
        CURRENT=$((CURRENT + 1))
        log "[$CURRENT/$TOTAL_EVALS] Evaluating: $model at ${depth} depth"

        TARGET_LAYER=$(python3 -c "import math; print(int($NUM_LAYERS * $depth))")

        # Check if nearby layers already exist
        EXISTS=false
        for offset in -2 -1 0 1 2; do
            CHECK_LAYER=$((TARGET_LAYER + offset))
            if [ -f "output/comparison/$model/layer_h$CHECK_LAYER/probe_evaluate-v2.jsonl" ]; then
                log "  ⊘ Skipping (layer_h$CHECK_LAYER already exists)"
                EXISTS=true
                break
            fi
        done

        if [ "$EXISTS" = false ]; then
            # Clear cache only when switching models
            if [ "$CURRENT_LOADED_MODEL" != "$model" ]; then
                clear_cache
                CURRENT_LOADED_MODEL="$model"
            fi

            python experiments/evaluate_vectors.py \
                --model "$model" \
                --layer-depth "$depth" \
                --batch-size $BATCH_SIZE \
                --eval-limit 2000 \
                2>&1 | tee -a "$LOG_FILE"
            log "  ✓ Completed"
        fi
        log ""
    done
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
