#!/bin/bash
# Five ablation experiments (paper Section 5.4) — run on HotpotQA
# Usage: bash experiments/run_ablations.sh

DATASET="hotpotqa"
N_SAMPLES=500
BASE="experiments/results"

run() {
    python -m src.cbet_controller \
        --dataset "$DATASET" \
        --n_samples $N_SAMPLES \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --model_path ./models/ \
        --theta 0.75 \
        --tau 0.5 \
        --ablation "$1" \
        --output_dir "${BASE}/${2}.json" \
        --log_dir "${BASE}/logs/" \
        "${@:3}"
}

# Ablation 1: Full CBET (baseline)
run full            ablation_full

# Ablation 2: Remove cross-branch NLI → single-branch only
run no_cross_branch ablation_no_cross

# Ablation 3: Remove epistemic override (keep stopping)
run no_override     ablation_no_override

# Ablation 4: Token Entropy replaces NLI (original ΔH approach)
run entropy_only    ablation_entropy

# Ablation 5: Fixed retrieval rounds (max_iter=3, no CS threshold)
run fixed_rounds    ablation_fixed --max_iterations 3
