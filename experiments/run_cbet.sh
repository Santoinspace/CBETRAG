#!/bin/bash
# Run CBET full experiment on all three datasets
# Usage: bash experiments/run_cbet.sh

# === Current: 3B model config ===
DATASETS=("hotpotqa" "musique" "2wikimultihopqa")
N_SAMPLES=50
MODEL_CONFIG=""
VLLM_URL="http://localhost:8000/v1"
VLLM_MODEL="/models/qwen"

# === 切换到 7B 时取消注释以下行 ===
# MODEL_CONFIG=configs/cbet_7b.yaml
# N_SAMPLES=500
# VLLM_URL="http://localhost:8000/v1"
# VLLM_MODEL="/models/qwen"

for DATASET in "${DATASETS[@]}"; do
    echo "Running CBET on $DATASET..."
    python -m src.cbet_controller \
        --dataset "$DATASET" \
        --n_samples $N_SAMPLES \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --model_path ./models/ \
        --theta 0.75 \
        --tau 0.5 \
        --backend vllm \
        --vllm_url "$VLLM_URL" \
        --vllm_model "$VLLM_MODEL" \
        --output_dir "experiments/results/cbet_${DATASET}.json" \
        --log_dir experiments/results/logs/
done
