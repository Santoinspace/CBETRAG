# Running Experiments — CBET-RAG

All scripts support **interrupt/resume**: if the output JSON already exists, processed
`(method, qid)` or `(variant, qid)` pairs are loaded and skipped on restart.

## Prerequisites

1. **vLLM serving Qwen2.5-7B-Instruct**:
   ```bash
   # Confirm model is loaded
   curl -s http://localhost:8000/v1/models | python -m json.tool
   # Should return model id: "qwen25-7b"
   ```

2. **DeBERTa NLI model**: `./models/nli-deberta-v3-base` (already downloaded)

3. **ElasticSearch** (only for Exp4):
   ```bash
   docker ps  # confirm ES container is running
   curl -s http://localhost:9200/wiki/_count | python -m json.tool  # should show ~21M docs
   ```

---

## Experiment 1: Main Comparison (3-5h)

Compares NoRAG / SingleRAG / IterativeRAG / CBET on HotpotQA (500) + MuSiQue (500).
Uses DatasetPassageRetriever (gold+distractor passages) — isolates method behavior from retrieval quality.

```bash
uv run python experiments/run_exp1_main.py \
    --datasets hotpotqa musique \
    --n_samples 500 \
    --vllm_model qwen25-7b \
    --output experiments/results/exp1_main.json
```

**Output**: `experiments/results/exp1_main.json` + `_summary.json`

**Pass criteria (paper Table 1)**:
- CBET F1 > SingleRAG F1 on both datasets
- CBET EM > IterativeRAG EM

---

## Experiment 2: Ablation Study (2-3h)

Runs 5 CBET variants on HotpotQA (200 samples):
- `full` — complete CBET
- `no_cross_branch` — GCS always 1.0, no cross-branch NLI
- `no_override` — epistemic override disabled
- `entropy_only` — no NLI claim extraction
- `fixed_rounds` — no CS stopping (always 3 iterations)

```bash
uv run python experiments/run_exp2_ablation.py \
    --n_samples 200 \
    --vllm_model qwen25-7b \
    --output experiments/results/exp2_ablation.json
```

**Output**: `experiments/results/exp2_ablation.json` + `_summary.json`

**Pass criteria (paper Table 2)**:
- `full` F1 > all other variants
- `no_cross_branch` shows largest F1 drop (validates cross-branch NLI)

---

## Experiment 3: θ Sensitivity (1h)

Sweeps θ ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9} on HotpotQA (100 samples).

```bash
uv run python experiments/run_exp3_theta.py \
    --n_samples 100 \
    --vllm_model qwen25-7b \
    --output experiments/results/exp3_theta.json
```

**Output**: `experiments/results/exp3_theta.json` + `_summary.json`

**Pass criteria (paper Figure 2)**:
- F1 curve has a clear peak (optimal θ)
- EarlyStop% increases monotonically with θ

Plot with: `python analysis/sensitivity_theta.py` (reads `exp3_theta_summary.json`)

---

## Experiment 4: Open-Domain with ElasticSearch (2-3h)

Tests CBET with real BM25 retrieval over 21M Wikipedia passages.

```bash
uv run python experiments/run_exp4_es.py \
    --n_samples 200 \
    --vllm_model qwen25-7b \
    --es_host localhost --es_port 9200 --es_index wiki \
    --output experiments/results/exp4_es.json
```

**Output**: `experiments/results/exp4_es.json` + `_summary.json`

**Pass criteria**:
- CBET F1 > SingleRAG F1 in open-domain setting
- CS scores show evidence sufficiency detection works even with noisy retrieval

---

## Monitoring Progress

Each script prints progress every 10-20 questions:
```
[42/500] EM=1 CS=0.423 DAG=3
[saved checkpoint: 126 rows]
```

To check intermediate results without stopping:
```bash
python -c "
import json
with open('experiments/results/exp1_main.json') as f:
    data = json.load(f)
print(f'Total rows: {len(data)}')
from collections import Counter
print('Methods:', Counter(r['method'] for r in data))
"
```

---

## Troubleshooting

**vLLM unreachable**: `ConnectionRefusedError` → restart vLLM container, script will resume.

**ElasticSearch error** (Exp4 only): check `docker ps`, restart ES container.

**OOM / CUDA error**: vLLM should handle OOM on its side. If it crashes, restart container and rerun — resume will skip completed samples.

**Keyboard interrupt**: safe to Ctrl+C anytime. All results up to the last checkpoint (every 10-20 samples) are saved.

---

## Execution Checklist

```
[ ] Exp1: main comparison (HotpotQA 500 + MuSiQue 500) — 3-5h
[ ] Exp2: ablation (HotpotQA 200 × 5 variants) — 2-3h
[ ] Exp3: theta sensitivity (HotpotQA 100 × 7 θ) — 1h
[ ] Exp4: open-domain ES (HotpotQA 200) — 2-3h
[ ] Generate theta sensitivity plot: python analysis/sensitivity_theta.py
[ ] Update CLAUDE.md PROJECT STATE with final results
[ ] Prepare paper Table 1 (main), Table 2 (ablation), Figure 2 (theta)
```

Total estimated runtime: 8-12h (can be run across multiple sessions).
