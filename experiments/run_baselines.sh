#!/bin/bash
# Run AdaRAGUE baselines: DRAGIN, SeaKR, AdaptiveRAG
#
# Prerequisites:
#   - ElasticSearch running on localhost:9200 with a "wiki" index
#   - Models downloaded to ./models/
#   - AdaRAGUE/data/ contains the dataset CSVs
#   - Python environment with AdaRAGUE dependencies installed
#
# Notes:
#   - FLARE is not available in this AdaRAGUE clone (skipped)
#   - SeaKR requires vllm with custom async engine (see section below)
#   - AdaptiveRAG requires separate LLM server + retriever server (see section below)
#   - DRAGIN runs directly (generates config JSONs on the fly)
#
# Usage: bash experiments/run_baselines.sh

set -euo pipefail

DATASETS=("hotpotqa" "musique" "2wikimultihopqa")
N_SAMPLES=500
BASE_RESULTS="experiments/results"
BASE_LOGS="${BASE_RESULTS}/logs"
MODEL_PATH="./models/Qwen2.5-7B-Instruct-AWQ"

mkdir -p "$BASE_RESULTS" "$BASE_LOGS"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DRAGIN baseline
#    Runs via AdaRAGUE/dragin/src/main.py with a generated JSON config.
#    Model is loaded via HuggingFace transformers (full bf16 — may OOM on 8GB).
#    If VRAM is insufficient, set DRAGIN_SKIP=1 in environment.
# ═══════════════════════════════════════════════════════════════════════════════

run_dragin() {
    local DATASET="$1"
    local CONFIG_FILE
    CONFIG_FILE="$(mktemp)"
    local DATA_PATH="AdaRAGUE/data/adaptive_rag_${DATASET}/test.csv"
    local OUTPUT_DIR="${BASE_RESULTS}/baseline_DRAGIN_${DATASET}"

    # Map dataset name to the form used by AdaRAGUE dragin
    local DS_NAME="$DATASET"
    case "$DATASET" in
        musique)       DS_NAME="musiq" ;;
        2wikimultihopqa) DS_NAME="2wiki" ;;
    esac

    cat > "$CONFIG_FILE" << JSONEOF
{
    "model_name_or_path": "${MODEL_PATH}",
    "method": "dragin",
    "dataset": "${DS_NAME}",
    "data_path": "${DATA_PATH}",
    "fewshot": 8,
    "sample": ${N_SAMPLES},
    "shuffle": false,
    "generate_max_length": 100,
    "query_formulation": "real_words",
    "retrieve_keep_top_k": 35,
    "output_dir": "${OUTPUT_DIR}",
    "retriever": "BM25",
    "es_index_name": "wiki",
    "retrieve_topk": 3,
    "hallucination_threshold": 10.0,
    "check_real_words": true,
    "use_counter": true
}
JSONEOF

    echo "[DRAGIN] Running on ${DATASET} (config: ${CONFIG_FILE})"
    cd AdaRAGUE/dragin
    python src/main.py -c "../../${CONFIG_FILE}" 2>&1 | tee "../../${BASE_LOGS}/baseline_DRAGIN_${DATASET}.log"
    cd - > /dev/null
    rm -f "$CONFIG_FILE"
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SeaKR baseline (OPTIONAL — requires vllm custom build)
#
#    SeaKR uses a custom fork of vllm that computes eigen-scores from
#    intermediate layers. To run it:
#
#    1. Install the custom vllm:  cd AdaRAGUE/SeaKR/vllm_uncertainty && pip install -e .
#    2. Ensure ElasticSearch is running on the configured port (default 9200)
#    3. Run:
#       python AdaRAGUE/SeaKR/main_multihop.py \
#           --dataset_name hotpotqa \
#           --retriever_port 9200 \
#           --model_name_or_path ./models/Qwen2.5-7B-Instruct-AWQ \
#           --served_model_name Qwen2.5-7B-Instruct-AWQ \
#           --save_dir experiments/results/baseline_SeaKR_hotpotqa
#
#    If you want to run SeaKR automatically, set SEAKR_ENABLED=1 and configure
#    SEAKR_MODEL_PATH, SEAKR_PORT below.
# ═══════════════════════════════════════════════════════════════════════════════

run_seakr() {
    local DATASET="$1"
    local SAVE_DIR="${BASE_RESULTS}/baseline_SeaKR_${DATASET}"

    echo "[SeaKR] Running on ${DATASET}"
    python AdaRAGUE/SeaKR/main_multihop.py \
        --dataset_name "$DATASET" \
        --retriever_port 9200 \
        --model_name_or_path "${SEAKR_MODEL_PATH:-./models/Qwen2.5-7B-Instruct-AWQ}" \
        --served_model_name "${SEAKR_SERVED_NAME:-Qwen2.5-7B-Instruct-AWQ}" \
        --save_dir "$SAVE_DIR" \
        2>&1 | tee "${BASE_LOGS}/baseline_SeaKR_${DATASET}.log"
}

# ═══════════════════════════════════════════════════════════════════════════════
# 3. AdaptiveRAG baseline (OPTIONAL — requires LLM + retriever servers)
#
#    AdaptiveRAG needs:
#      - An LLM server (vllm / TGI) running on a configured port
#      - An ElasticSearch retriever server
#
#    See AdaRAGUE/Adaptive_Rag/README.md for full setup instructions.
#    Once servers are up:
#
#    cd AdaRAGUE/Adaptive_Rag
#    python run.py <experiment_name> \
#        --instantiation_scheme oner_qa \
#        --prompt_set 1 \
#        --set_name test \
#        --llm_port_num 8000 \
#        predict
#
#    If you want to run AdaptiveRAG automatically, set ADAPTIVERAG_ENABLED=1
#    and configure the server ports below.
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Main dispatch
# ═══════════════════════════════════════════════════════════════════════════════

echo "=== Running AdaRAGUE Baselines ==="
echo "Datasets: ${DATASETS[*]}"
echo "Samples per dataset: ${N_SAMPLES}"
echo ""

# DRAGIN
if [[ "${DRAGIN_SKIP:-0}" != "1" ]]; then
    for DS in "${DATASETS[@]}"; do
        run_dragin "$DS" || echo "[WARN] DRAGIN failed on $DS, continuing..."
    done
else
    echo "[SKIP] DRAGIN (DRAGIN_SKIP=1)"
fi

# SeaKR
if [[ "${SEAKR_ENABLED:-0}" == "1" ]]; then
    for DS in "${DATASETS[@]}"; do
        run_seakr "$DS" || echo "[WARN] SeaKR failed on $DS, continuing..."
    done
else
    echo "[SKIP] SeaKR (set SEAKR_ENABLED=1 to run — requires vllm custom build)"
fi

# AdaptiveRAG
if [[ "${ADAPTIVERAG_ENABLED:-0}" == "1" ]]; then
    echo "[TODO] AdaptiveRAG requires LLM + retriever servers. See script comments."
else
    echo "[SKIP] AdaptiveRAG (set ADAPTIVERAG_ENABLED=1 — requires LLM + retriever servers)"
fi

# FLARE — not available in this AdaRAGUE clone
echo "[SKIP] FLARE (not included in this AdaRAGUE clone; reference numbers from AdaRAGUE paper)"

echo ""
echo "=== Baselines complete ==="
echo "Results in ${BASE_RESULTS}/"
