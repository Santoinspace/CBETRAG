#!/bin/bash
# Run AdaRAGUE baselines (DRAGIN, SeaKR, FLARE, AdaptiveRAG)
# Delegates to AdaRAGUE's own run scripts — we do not modify them.

DATASETS=("hotpotqa" "musique" "2wikimultihopqa")
N_SAMPLES=500

cd AdaRAGUE || { echo "AdaRAGUE directory not found"; exit 1; }

for DATASET in "${DATASETS[@]}"; do
    for METHOD in DRAGIN SeaKR FLARE Adaptive_Rag; do
        echo "Running $METHOD on $DATASET..."
        python -m Method.${METHOD}.main \
            --dataset "$DATASET" \
            --n_samples $N_SAMPLES \
            --output_dir "../experiments/results/baseline_${METHOD}_${DATASET}.json" \
            2>&1 | tee "../experiments/results/logs/baseline_${METHOD}_${DATASET}.log"
    done
done
