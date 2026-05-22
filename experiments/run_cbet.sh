#!/bin/bash
# Run CBET full experiment on all three datasets

DATASETS=("hotpotqa" "musique" "2wikimultihopqa")
N_SAMPLES=500

for DATASET in "${DATASETS[@]}"; do
    echo "Running CBET on $DATASET..."
    python -m src.cbet_controller \
        --dataset "$DATASET" \
        --n_samples $N_SAMPLES \
        --model_path ./models/ \
        --theta 0.75 \
        --tau 0.5 \
        --output_dir "experiments/results/cbet_${DATASET}.json" \
        --log_dir experiments/results/logs/
done
