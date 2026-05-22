#!/bin/bash
# Five ablation experiments (paper Section 5.4) — run on HotpotQA

DATASET="hotpotqa"
N_SAMPLES=500
BASE="experiments/results"

run() {
    python -m src.cbet_controller \
        --dataset "$DATASET" \
        --n_samples $N_SAMPLES \
        --model_path ./models/ \
        --ablation "$1" \
        --output_dir "${BASE}/${2}.json" \
        --log_dir "${BASE}/logs/" \
        "${@:3}"
}

run full            ablation_full
run no_cross_branch ablation_no_cross
run no_override     ablation_no_override
run entropy_only    ablation_entropy
run fixed_rounds    ablation_fixed --max_iterations 3
