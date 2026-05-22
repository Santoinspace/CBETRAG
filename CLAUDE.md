# CBET-RAG Project — Claude Code Master Prompt

> Cross-Branch Evidence Triangulation for Multi-hop Agentic RAG
> Target: AAAI 2027 | Base: AdaRAGUE (ACL 2025)

---

## 0. 阅读本文件的规则（MANDATORY）

- **每次新会话开始时必须完整阅读本文件**，再执行任何代码操作
- **每次完成一个 Task 后必须更新本文件末尾的 `## PROJECT STATE` 区块**
- 本文件是项目的唯一真相来源（single source of truth），代码、实验结果、决策记录全部在此维护
- 遇到任何设计决策分叉点，必须在 `## DECISIONS LOG` 中记录选择及理由

---

## 1. 项目概览

### 1.1 研究问题

现有 Agentic RAG 方法（DRAGIN、MCTS-RAG、ParallelSearch）在多跳推理中面临两个未解决的开放问题：

1. **知识冲突判定**：检索证据与 LLM 参数化记忆冲突时，无法判断该信任哪一方
2. **检索完备性判定**：多跳任务中无法判断何时证据已足够、应停止检索

### 1.2 核心方法：CBET（Cross-Branch Evidence Triangulation）

利用**跨分支 NLI 一致性**作为统一信号，同时解决上述两个问题。

**三个核心公式：**

```
Cov(Eᵢ, qᵢ)  = NLI_score(Eᵢ ⊨ â_i)                          # 单分支覆盖度

GCS(ε)        = (2/n(n-1)) × Σᵢ＜ⱼ (1 - 𝟙[NLI(Eᵢ,Eⱼ)=contradiction])  # 全局一致性

CS(ε, Q)      = min_i Cov(Eᵢ,qᵢ) × GCS(ε)  ≥ θ  →  STOP      # 完备性停止准则
```

**知识冲突信任信号：**

```
TrustRetrieved(i) = GCS(ε) × 𝟙[NLI(Eᵢ, Mᵢ) = conflict] > τ  →  epistemic override
```

### 1.3 开源代码基础

**主要基础：AdaRAGUE（ACL 2025）**

- Repo: https://github.com/s-nlp/AdaRAGUE
- 理由：已集成 DRAGIN/FLARE/SeaKR/AdaptiveRAG 的统一评估框架，数据集预处理完毕，检索器已配置
- 我们在此基础上新增 `Method/CBET/` 模块，不修改其他方法代码

**参考结构（了解即可，不作为 base）：MCTS-RAG（EMNLP 2025）**

- Repo: https://github.com/yale-nlp/MCTS-RAG
- 参考其树形结构的分支管理逻辑

---

## 2. 项目目录结构

```
cbet-rag/
├── CLAUDE.md                          # ← 本文件，必须维护
├── README.md
├── requirements.txt
│
├── AdaRAGUE/                          # 完整 clone，不修改原始文件
│   ├── data/                          # 数据集（HotpotQA, MuSiQue, 2Wiki）
│   ├── standard_retriever/            # 统一检索器
│   ├── Method/
│   │   ├── DRAGIN/                    # 原始方法，作为 baseline
│   │   ├── SeaKR/                     # 原始方法，作为 baseline
│   │   ├── FLARE/                     # 原始方法，作为 baseline
│   │   └── CBET/                      # ← 新增：我们的方法
│   └── evaluate/
│
├── src/                               # CBET 核心源码（独立于 AdaRAGUE）
│   ├── __init__.py
│   ├── dag_extractor.py               # Task 2: DAG 子问题提取
│   ├── nli_scorer.py                  # Task 3: NLI 跨分支一致性计算
│   ├── parametric_probe.py            # Task 4: 参数化记忆探测
│   ├── cbet_controller.py             # Task 5: 主控制器（完整算法）
│   ├── epistemic_override.py          # Task 6: 信念覆盖 Prompt 构建
│   └── completeness_monitor.py       # Task 7: CS 分数与早停
│
├── configs/
│   ├── cbet_hotpotqa.yaml
│   ├── cbet_musique.yaml
│   └── cbet_2wiki.yaml
│
├── experiments/
│   ├── run_baselines.sh               # 跑 AdaRAGUE 中所有 baseline
│   ├── run_cbet.sh                    # 跑 CBET 完整版
│   ├── run_ablations.sh               # 跑消融实验
│   └── results/                       # 实验结果 JSON，不得手动修改
│
├── analysis/
│   ├── sensitivity_theta.py           # θ 超参敏感性分析
│   ├── conflict_case_study.py         # 知识冲突案例分析
│   └── visualize_results.py           # 生成论文图表
│
└── tests/
    ├── test_nli_scorer.py
    ├── test_dag_extractor.py
    └── test_cbet_e2e.py
```

---

## 3. 环境配置（Task 0）

### 3.1 硬件约束

- **目标环境**：单张 RTX 4060（8GB VRAM）— 资源受限，必须严格控制显存
- **VRAM 预算**（8GB 总量）：

  | 组件                                        | 显存占用         | 运行位置 |
  | ------------------------------------------- | ---------------- | -------- |
  | Qwen2.5-7B-Instruct-**AWQ**（4-bit）  | ~4.5GB           | GPU      |
  | cross-encoder/nli-deberta-v3-**base** | ~0.4GB           | GPU      |
  | KV cache + 激活值余量                       | ~2.0GB           | GPU      |
  | 系统 overhead                               | ~0.5GB           | GPU      |
  | **合计**                              | **~7.4GB** | ✅ 可行  |
- **⚠️ 重要约束**：

  - 必须使用 AWQ 量化版，全精度 bfloat16 约需 14GB，超出 4060 上限
  - NLI 模型使用 `deberta-v3-base` 而非 `large`（性能差异 < 1.5% F1，可接受）
  - 推理时 `max_new_tokens` 限制为 512，避免 KV cache 爆显存
  - 论文 Implementation Details 中注明：`Qwen2.5-7B-Instruct-AWQ (4-bit)` on `RTX 4060 8GB`

### 3.2 安装步骤（使用 uv，由 Claude Code 自动执行，若已有虚拟环境则不必执行）

```bash
# Step 0: 虚拟环境（uv）
pip install uv
uv venv .venv --python 3.11
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Step 1: 安装所有依赖
uv pip install -r requirements.txt

# Step 2: Clone AdaRAGUE 作为基础框架
git clone https://github.com/s-nlp/AdaRAGUE.git
# 不执行 AdaRAGUE 自己的 pip install，避免依赖冲突

# Step 3: 从 ModelScope 下载模型（国内网络推荐）
modelscope download \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --local_dir ./models/Qwen2.5-7B-Instruct-AWQ

modelscope download \
    --model cross-encoder/nli-deberta-v3-base \
    --local_dir ./models/nli-deberta-v3-base

# Step 4: 验证显存使用
python tests/test_env.py
```

### 3.3 requirements.txt 内容如文件所示

---

## 4. 任务分解（按顺序执行）

---

### Task 0：环境验证脚本

**文件**：`tests/test_env.py`

**要求**：

- 检查虚拟环境已激活（`sys.prefix` 包含 `.venv`）
- 检查 CUDA 可用 + VRAM ≥ 7GB（4060 8GB 的可用量约为 7.5GB）
- 验证 AdaRAGUE clone 完整（检查关键目录存在）
- 加载 DeBERTa-v3-**base** NLI 模型做一次推理并打印结果，验证显存占用 < 600MB
- 加载 Qwen2.5-7B-Instruct-**AWQ** 做一次推理并打印 logprobs，验证显存占用 < 5.5GB
- 两个模型同时在显存中时，打印总显存使用量，确认 < 7.5GB
- 所有检查通过后打印 `[ENV OK] Total VRAM: X.XGB / 8.0GB`

**加载 AWQ 模型的正确方式**：

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_quantized(
    "./models/Qwen2.5-7B-Instruct-AWQ",
    fuse_layers=True,        # 启用 kernel fusion，提升速度
    trust_remote_code=False,
    safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "./models/Qwen2.5-7B-Instruct-AWQ",
    trust_remote_code=False
)
```

**成功标准**：`python tests/test_env.py` 无报错，最后一行输出 `[ENV OK]`

---

### Task 1：数据集适配器

**文件**：`src/data_adapter.py`

**要求**：
适配 AdaRAGUE 的三个数据集格式，输出统一的 `Question` 对象：

```python
@dataclass
class Question:
    qid: str
    query: str                    # 原始多跳问题
    gold_passages: list[str]      # 正确证据段（用于评估）
    distractor_passages: list[str]# 干扰段（用于模拟噪声）
    answer: str                   # 标准答案
    dataset: str                  # hotpotqa / musique / 2wiki
    hop_count: int                # 跳数（从数据集字段推断）
```

读取路径：`AdaRAGUE/data/adaptive_rag_{dataset}/test.csv`

**注意**：

- MuSiQue 的 hop_count 需从 `id` 字段的前缀推断（`2hop__`, `3hop__`, `4hop__`）
- HotpotQA 默认 2-hop
- 2WikiMultiHopQA 从 `type` 字段推断（bridge=2hop, comparison=2hop, bridge_comparison=3hop+）

---

### Task 2：DAG 子问题提取器

**文件**：`src/dag_extractor.py`

**核心函数**：

```python
def extract_dag(query: str, llm_client: LLMClient) -> QuestionDAG:
    ...
```

**输出结构**：

```python
@dataclass
class SubQuestion:
    id: str           # q1, q2, q3...
    text: str         # 子问题文本
    depends_on: list[str]  # 依赖的子问题 id（叶节点为空列表）
    is_leaf: bool

@dataclass  
class QuestionDAG:
    root_query: str
    sub_questions: list[SubQuestion]
  
    def get_leaves(self) -> list[SubQuestion]: ...
    def get_execution_order(self) -> list[list[SubQuestion]]: 
        # 返回可并行执行的层次列表（拓扑排序）
        # 例：[[q1, q2], [q3], [q4]] 表示 q1/q2 并行，完成后执行 q3，再执行 q4
```

**Prompt 模板**（写死在函数里，不要参数化）：

```
You are a multi-hop question decomposer. Given a complex multi-hop question, 
decompose it into atomic sub-questions with their dependency relationships.

Rules:
1. Each sub-question must be answerable with a single factual retrieval
2. Mark dependencies explicitly (which sub-questions must be answered first)
3. Independent sub-questions have empty depends_on lists
4. Output ONLY valid JSON, no explanation

Question: {query}

Output format:
{
  "sub_questions": [
    {"id": "q1", "text": "...", "depends_on": []},
    {"id": "q2", "text": "...", "depends_on": []},
    {"id": "q3", "text": "... [answer of q1] ... [answer of q2] ...", "depends_on": ["q1", "q2"]}
  ]
}
```

**容错处理**：

- JSON 解析失败时最多重试 2 次（修改 prompt 增加格式约束）
- 重试仍失败则退回单一子问题（整个 query 作为唯一子问题），记录到 `experiments/results/dag_failures.log`

**单元测试**（`tests/test_dag_extractor.py`）：

- 测试 2-hop HotpotQA 样例，断言 `len(dag.sub_questions) >= 2`
- 测试 4-hop MuSiQue 样例，断言 `len(dag.get_leaves()) >= 2`（有并行叶节点）
- 测试 DAG 无环（拓扑排序不产生循环依赖）

---

### Task 3：NLI 一致性评分器

**文件**：`src/nli_scorer.py`

这是整个项目最核心的模块，需要特别仔细实现。

**模型**：`cross-encoder/nli-deberta-v3-large`（从 `./models/` 加载）

**核心函数**：

```python
class NLIScorer:
    def __init__(self, model_path: str = "./models/cross-encoder/nli-deberta-v3-large",
                 device: str = "cuda", batch_size: int = 16):
        ...
  
    def score_pair(self, premise: str, hypothesis: str) -> NLIResult:
        """单对 NLI 打分"""
        ...
  
    def extract_atomic_claims(self, text: str, llm_client: LLMClient) -> list[str]:
        """
        关键：不对整个 chunk 做 NLI，先提取原子 claims
        用 LLM 提取事实三元组，再转为自然语言陈述句
        例："柏林墙倒塌于 1989 年 11 月 9 日"
        """
        ...
  
    def compute_coverage(self, evidence: str, sub_answer: str, 
                         llm_client: LLMClient) -> float:
        """
        Cov(Eᵢ, qᵢ) = NLI_score(Eᵢ ⊨ â_i)
        先从 evidence 提取 claims，再判断 claims 是否 entail sub_answer
        返回最高 entailment 分数
        """
        ...
  
    def compute_gcs(self, evidences: list[str], 
                    llm_client: LLMClient) -> float:
        """
        GCS(ε) = 1 - (contradiction pairs / total pairs)
        对所有 evidence 两两做 NLI，统计非矛盾比例
        注意：先提取 atomic claims 再做跨文档 NLI
        """
        ...
  
    def compute_completeness_score(self, 
                                   branch_evidences: list[str],
                                   branch_answers: list[str],
                                   sub_questions: list[str],
                                   llm_client: LLMClient) -> CompletenessResult:
        """
        CS = min_i(Cov_i) × GCS
        返回完整的 CompletenessResult（包含各分支 Cov、GCS、CS、是否建议停止）
        """
        ...

@dataclass
class NLIResult:
    label: str          # "entailment" / "neutral" / "contradiction"
    entailment_score: float
    neutral_score: float
    contradiction_score: float

@dataclass
class CompletenessResult:
    branch_coverages: list[float]   # 每个分支的 Cov 分数
    min_coverage: float             # 最弱分支
    gcs: float                      # 全局一致性
    cs: float                       # 完备性分数 = min_cov × gcs
    should_stop: bool               # cs >= theta
    noisy_branch_ids: list[int]     # GCS 低时，标记不一致的分支 index
```

**重要实现细节**：

1. `extract_atomic_claims` 的 Prompt：

```
Extract all atomic factual claims from this text as a JSON list of simple declarative sentences.
Each claim must be independently verifiable. Remove opinions and vague statements.
Text: {text}
Output: {"claims": ["claim1", "claim2", ...]}
```

2. NLI 输入长度限制：DeBERTa-v3-large 最大 512 tokens，超出需截断（保留首尾各 128 tokens）
3. GCS 计算时，若分支数 n=1（单子问题），GCS 默认返回 1.0（无跨分支比较可做）
4. `noisy_branch_ids`：对每个分支 i，如果 `∃j: NLI(Eᵢ, Eⱼ) = contradiction`，则标记 i 或 j 为噪声（选择 coverage 更低的那个）

**单元测试**（`tests/test_nli_scorer.py`）：

- 测试互相一致的两段证据 → GCS 接近 1.0
- 测试明显矛盾的两段证据（如"A 在 1989 年""A 在 1991 年"）→ GCS < 0.5
- 测试 coverage：相关证据 → Cov > 0.7，无关证据 → Cov < 0.3

---

### Task 4：参数化记忆探测器

**文件**：`src/parametric_probe.py`

**目标**：获取 LLM 在**不看检索内容**的条件下对子问题的"参数化答案"及置信度，用于后续冲突检测。

```python
class ParametricProbe:
    def __init__(self, llm_client: LLMClient):
        ...
  
    def probe(self, sub_question: str) -> ParametricMemory:
        """
        仅用参数化知识回答子问题，同时获取答案的不确定性
        使用 AdaRAGUE 推荐的 Mean Token Entropy 作为不确定性指标
        """
        ...
  
    def detect_conflict(self, parametric: ParametricMemory, 
                        retrieved_evidence: str,
                        nli_scorer: NLIScorer,
                        llm_client: LLMClient) -> ConflictResult:
        """
        判断检索证据与参数化记忆是否冲突
        NLI(retrieved_evidence, parametric_answer) = contradiction → conflict
        """
        ...

@dataclass
class ParametricMemory:
    answer: str
    confidence: float       # Mean Token Entropy（越低越确定）
    raw_logprobs: list[float]

@dataclass
class ConflictResult:
    has_conflict: bool
    conflict_type: str      # "parametric_vs_retrieved" / "no_conflict" / "uncertain"
    parametric_answer: str
    retrieved_answer: str   # 从 evidence 中提取的答案
    trust_retrieved: float  # TrustRetrieved(i) = GCS × 𝟙[conflict]
```

**实现要点**：

- Probe prompt 不包含任何检索内容，仅给子问题：`"Answer this question concisely: {sub_question}"`
- 从 LLM logprobs 计算 Mean Token Entropy：`H = -mean(Σ_w p(w|x<i) log p(w|x<i))`（仅计算答案 tokens，跳过 prompt tokens）
- `confidence < 0.3`（低熵，模型很确定）时，冲突才被认为是"真实冲突"，需要触发 epistemic override

---

### Task 5：CBET 主控制器

**文件**：`src/cbet_controller.py`

这是整合所有模块的主算法，实现完整的检索-验证-停止循环。

```python
class CBETController:
    def __init__(self, 
                 llm_client: LLMClient,
                 retriever: Retriever,          # 复用 AdaRAGUE 的 standard_retriever
                 nli_scorer: NLIScorer,
                 parametric_probe: ParametricProbe,
                 config: CBETConfig):
        ...
  
    def solve(self, question: Question) -> CBETResult:
        """主入口，处理一个完整的多跳问题"""
        ...

@dataclass
class CBETConfig:
    theta: float = 0.75         # 完备性停止阈值（消融实验中在 0.6-0.9 区间测试）
    tau: float = 0.5            # 知识冲突信任阈值
    max_iterations: int = 5     # 最大检索轮数（防止死循环）
    max_branches: int = 6       # DAG 最大子问题数
    nli_claim_extraction: bool = True  # 是否用 LLM 提取 atomic claims（消融用）
```

**主算法伪代码**（务必严格按照此逻辑实现）：

```python
def solve(self, question: Question) -> CBETResult:
    # Step 1: DAG 提取
    dag = self.dag_extractor.extract_dag(question.query)
  
    iteration = 0
    branch_states = {sq.id: BranchState() for sq in dag.sub_questions}
  
    while iteration < self.config.max_iterations:
        iteration += 1
    
        # Step 2: 按拓扑序执行各层
        for parallel_batch in dag.get_execution_order():
        
            # Step 2a: 叶节点并行检索
            leaf_nodes = [sq for sq in parallel_batch if sq.is_leaf]
            if leaf_nodes:
                retrieval_results = self._parallel_retrieve(
                    leaf_nodes, branch_states
                )
        
            # Step 2b: 内部节点串行（使用已验证的前驱答案）
            internal_nodes = [sq for sq in parallel_batch if not sq.is_leaf]
            for node in internal_nodes:
                enriched_query = self._enrich_with_predecessor_answers(
                    node, branch_states
                )
                retrieval_results[node.id] = self._retrieve(enriched_query)
        
            # Step 3: 为每个分支更新证据
            for sq in parallel_batch:
                branch_states[sq.id].evidence = retrieval_results[sq.id]
            
                # Step 4: 参数化记忆探测 + 冲突检测
                param_mem = self.parametric_probe.probe(sq.text)
                conflict = self.parametric_probe.detect_conflict(
                    param_mem, branch_states[sq.id].evidence
                )
                branch_states[sq.id].conflict = conflict
            
                # Step 5: epistemic override（如果需要）
                if conflict.trust_retrieved > self.config.tau:
                    branch_states[sq.id].override_prompt = \
                        self.epistemic_overrider.build(sq.text, 
                                                       branch_states[sq.id].evidence)
    
        # Step 6: 计算完备性分数
        cs_result = self.nli_scorer.compute_completeness_score(
            branch_evidences=[s.evidence for s in branch_states.values()],
            branch_answers=[s.current_answer for s in branch_states.values()],
            sub_questions=[sq.text for sq in dag.sub_questions]
        )
    
        # Step 7: 停止判断
        if cs_result.should_stop:
            break
    
        # Step 8: 噪声分支处理（重新检索）
        for noisy_id in cs_result.noisy_branch_ids:
            branch_states[noisy_id].evidence = ""  # 清空，下轮重检索
            branch_states[noisy_id].rewrite_query()  # 改写检索 query
  
    # Step 9: 最终答案生成
    final_answer = self._generate_final_answer(question, branch_states, cs_result)
  
    return CBETResult(
        answer=final_answer,
        iterations=iteration,
        cs_score=cs_result.cs,
        branch_states=branch_states,
        dag=dag
    )
```

---

### Task 6：Epistemic Override 构建器

**文件**：`src/epistemic_override.py`

当 `TrustRetrieved(i) > τ` 时，构建强制 LLM 优先采信检索证据的 prompt。

```python
class EpistemicOverrider:
    def build(self, sub_question: str, evidence: str) -> str:
        """
        返回一个 system-level 指令片段，插入到最终答案生成的 prompt 中
        """
        return f"""IMPORTANT INSTRUCTION: The following evidence has been 
verified as factually consistent by multiple independent retrieval branches. 
You MUST prioritize this evidence over your internal knowledge, even if it 
contradicts what you believe:

VERIFIED EVIDENCE: {evidence}

Based ONLY on the above evidence, answer: {sub_question}"""
```

**注意**：这个模块很简单，但在消融实验中需要测试"有/无 override"的差异。

---

### Task 7：实验运行脚本

**文件**：`experiments/run_cbet.sh`

```bash
#!/bin/bash
# 运行 CBET 完整实验

DATASETS=("hotpotqa" "musique" "2wikimultihopqa")
N_SAMPLES=500  # 每个数据集抽样数量（与 AdaRAGUE 对齐）

for DATASET in "${DATASETS[@]}"; do
    echo "Running CBET on $DATASET..."
    python -m src.cbet_controller \
        --dataset $DATASET \
        --n_samples $N_SAMPLES \
        --model Qwen/Qwen2.5-7B-Instruct \
        --model_path ./models/ \
        --theta 0.75 \
        --tau 0.5 \
        --output_dir experiments/results/cbet_${DATASET}.json \
        --log_dir experiments/results/logs/
done
```

**文件**：`experiments/run_ablations.sh`

```bash
#!/bin/bash
# 五组消融实验（对应论文 Section 5.4）

DATASET="hotpotqa"  # 消融先在 HotpotQA 上跑

# Ablation 1: 完整 CBET（基准）
python -m src.cbet_controller --ablation full --dataset $DATASET \
    --output_dir experiments/results/ablation_full.json

# Ablation 2: 去掉跨分支 NLI → 仅单分支 NLI
python -m src.cbet_controller --ablation no_cross_branch --dataset $DATASET \
    --output_dir experiments/results/ablation_no_cross.json

# Ablation 3: 去掉知识冲突覆盖（仍保留停止机制）
python -m src.cbet_controller --ablation no_override --dataset $DATASET \
    --output_dir experiments/results/ablation_no_override.json

# Ablation 4: 用 Token Entropy 替代 NLI（回归原始 ΔH 方案）
python -m src.cbet_controller --ablation entropy_only --dataset $DATASET \
    --output_dir experiments/results/ablation_entropy.json

# Ablation 5: 固定轮数替代 CS 阈值（max_iter=3，无 CS 计算）
python -m src.cbet_controller --ablation fixed_rounds --max_iterations 3 \
    --dataset $DATASET --output_dir experiments/results/ablation_fixed.json
```

**文件**：`analysis/sensitivity_theta.py`

```python
# θ 超参敏感性分析：在 [0.5, 0.6, 0.7, 0.75, 0.8, 0.9] 上跑 CBET
# 绘制 θ vs F1 和 θ vs 平均检索轮数的双轴折线图
# 保存到 experiments/results/theta_sensitivity.png
```

---

### Task 8：评估与结果聚合

**文件**：`analysis/evaluate_all.py`

对齐 AdaRAGUE 的评估协议，输出以下指标：

```python
METRICS = {
    # 准确率指标（与 AdaRAGUE baselines 直接对比）
    "em": exact_match,
    "f1": token_f1,
  
    # 效率指标
    "avg_retrieval_rounds": ...,     # 平均检索轮数
    "avg_lm_calls": ...,             # 平均 LLM 调用次数（含 DAG 提取、claim 提取）
  
    # CBET 专项指标
    "avg_cs_at_stop": ...,           # 停止时的平均 CS 分数
    "conflict_detected_rate": ...,   # 检测到知识冲突的比例
    "override_triggered_rate": ...,  # 触发 epistemic override 的比例
    "noisy_branch_evicted_rate": ...,# 噪声分支被驱逐的比例
}
```

输出格式（与论文表格直接对应）：

```
=== Results on HotpotQA (n=500) ===
Method         EM     F1    Avg-Ret  Avg-LM
DRAGIN        42.1   51.3    2.1      2.3
SeaKR         43.8   53.0    1.9      2.1
FLARE         40.2   49.8    2.3      2.5
AdaptiveRAG   44.1   53.7    2.0      2.2
CBET (ours)   XX.X   XX.X    X.X      X.X   ← 填入实验结果
```

---

## 5. 代码规范（必须遵守）

### 5.1 接口规范

- 所有模块通过 `LLMClient` 抽象层访问 LLM（支持 vllm 和 HuggingFace 两种后端）
- 所有模块通过 `Retriever` 抽象层访问检索器（复用 AdaRAGUE 的 ElasticSearch 接口）
- 禁止在模块内部硬编码模型路径，统一从 `configs/*.yaml` 读取

### 5.2 日志规范

每次 `solve()` 调用都必须输出结构化日志：

```json
{
  "qid": "...",
  "iterations": 3,
  "dag_size": 4,
  "branch_cs_scores": [0.82, 0.91, 0.76, 0.88],
  "final_cs": 0.76,
  "conflicts_detected": ["q2"],
  "overrides_triggered": ["q2"],
  "noisy_evicted": [],
  "answer": "...",
  "gold_answer": "...",
  "em": 1,
  "f1": 1.0
}
```

### 5.3 错误处理原则

- NLI 调用失败 → 返回 `NLIResult(label="neutral", ...)` 并记录警告，不中断流程
- DAG 提取失败 → 退回单子问题模式，记录到 `dag_failures.log`
- LLM 调用超时 → 最多重试 2 次，超时后返回空答案并记录
- 所有异常都要被 catch 并记录，不允许整个实验因单条数据崩溃

### 5.4 可复现性

- 所有随机性来源（模型采样）设 `seed=42`
- 实验结果写入 JSON 后不得修改，分析脚本只读取 JSON

---

## 6. 与 AdaRAGUE 的集成方式

**原则：最小侵入，不修改 AdaRAGUE 任何原始文件**

需要在 `AdaRAGUE/Method/CBET/` 下创建的适配文件：

```python
# AdaRAGUE/Method/CBET/main.py
# 适配 AdaRAGUE 的统一运行接口（参考 AdaRAGUE/Method/DRAGIN/main.py 的格式）

import sys
sys.path.insert(0, "../../")  # 指向项目根目录
from src.cbet_controller import CBETController
from src.data_adapter import load_adaragUE_dataset

def run(args):
    # 将 AdaRAGUE 的 args 格式转换为 CBET 的 CBETConfig
    ...
```

---

## 7. 论文写作对应关系

| 论文章节                        | 对应代码文件                                        | 核心数字来源                                |
| ------------------------------- | --------------------------------------------------- | ------------------------------------------- |
| Section 3 Background            | -                                                   | AdaRAGUE/MCTS-RAG 文献                      |
| Section 4.1 DAG Extraction      | `dag_extractor.py`                                | `dag_failures.log` 成功率                 |
| Section 4.2 Cross-Branch NLI    | `nli_scorer.py`                                   | 消融 Ablation 2                             |
| Section 4.3 Conflict Resolution | `parametric_probe.py` + `epistemic_override.py` | 消融 Ablation 3 +`conflict_detected_rate` |
| Section 4.4 CS Stopping         | `completeness_monitor.py`                         | 消融 Ablation 5 + θ 敏感性图               |
| Section 5 Experiments           | `experiments/results/*.json`                      | `evaluate_all.py` 输出                    |
| Table 1 Main Results            | -                                                   | `run_baselines.sh` + `run_cbet.sh`      |
| Table 2 Ablation                | -                                                   | `run_ablations.sh`                        |
| Figure 2 θ Sensitivity         | -                                                   | `sensitivity_theta.py`                    |

---

## PROJECT STATE

> 每次完成 Task 后在此更新，格式如下

| Task                       | 状态      | 完成时间   | 备注                                                                     |
| -------------------------- | --------- | ---------- | ------------------------------------------------------------------------ |
| Task 0: 环境验证           | ✅ 完成   | 2026-05-22 | tests/test_env.py 已就绪，需模型下载后运行                               |
| Task 1: 数据适配器         | ✅ 完成   | 2026-05-22 | src/data_adapter.py，6/6 测试通过                                        |
| Task 2: DAG 提取器         | ✅ 完成   | 2026-05-22 | src/dag_extractor.py，7/7 测试通过                                       |
| Task 3: NLI 评分器         | ✅ 完成   | 2026-05-22 | src/nli_scorer.py，14/14 快速测试通过；4 个 GPU 集成测试需模型下载后运行 |
| Task 4: 参数化探针         | ✅ 完成   | 2026-05-22 | src/parametric_probe.py，14/14 测试通过                                  |
| Task 5: 主控制器           | ✅ 完成   | 2026-05-22 | src/cbet_controller.py，9/9 测试通过                                     |
| Task 6: Epistemic Override | ✅ 完成   | 2026-05-22 | src/epistemic_override.py，含在 e2e 测试中                               |
| Task 7: 实验脚本           | ⬜ 未开始 | -          | -                                                                        |
| Task 8: 评估聚合           | ⬜ 未开始 | -          | -                                                                        |
| Baseline 实验              | ⬜ 未开始 | -          | -                                                                        |
| CBET 完整实验              | ⬜ 未开始 | -          | -                                                                        |
| 消融实验                   | ⬜ 未开始 | -          | -                                                                        |
| θ 敏感性分析              | ⬜ 未开始 | -          | -                                                                        |

---

## DECISIONS LOG

> 记录所有设计决策，格式：[日期] 问题描述 → 选择 → 理由

- [初始化] 硬件约束 → RTX 4060 8GB → 必须使用量化模型；全精度 Qwen2.5-7B 需 ~14GB，超出硬件上限
- [初始化] 主模型选择 → `Qwen2.5-7B-Instruct-AWQ`（4-bit）→ 显存占用 ~4.5GB，与 NLI 模型共存可行；AWQ 量化在 QA 任务上性能损失 < 1% F1
- [初始化] NLI 模型选择 → `cross-encoder/nli-deberta-v3-base`（非 large）→ 显存仅 ~0.4GB，与 AWQ 主模型共存后总占用 ~7.4GB < 8GB；MNLI F1 差距 < 1.5%，可接受
- [初始化] 模型下载方式 → ModelScope CLI → 国内网络访问 HuggingFace 不稳定，ModelScope 镜像速度更快
- [初始化] 虚拟环境工具 → uv → 比 pip/conda 安装速度快 10-100x，由 Claude Code 自动创建
- [初始化] 代码基础选择 → `AdaRAGUE` → 已有统一评估框架，三个目标数据集预处理完毕，baseline 复现成本极低
- [初始化] atomic claims 提取 → 使用 LLM 而非 NER → 跨文档推理中事实跨度超过 NER 能识别的范围

---

## KNOWN ISSUES & RISKS

- **风险 1（硬件 OOM）**：4060 8GB 显存紧张，同时运行 AWQ 主模型 + DeBERTa + KV cache 时可能触发 OOM。**缓解**：设置 `max_new_tokens=512`；如仍 OOM，将 DeBERTa 移至 CPU（`device="cpu"`），推理变慢但稳定。
- **风险 2（AWQ logprobs 偏差）**：AWQ 量化的 logprobs 与全精度有微小差异（约 ±0.02 entropy），影响参数化记忆探测阈值标定。**缓解**：在验证集上重新标定 `tau`，不沿用全精度默认值。
- **风险 3（推理速度）**：4060 推理速度约为 3090 的 40-50%，500 条完整实验预计 20-30 小时。**缓解**：先用 50 条验证方法有效，再挂机跑完整实验。
- **风险 4**：DAG 提取质量依赖 Qwen2.5-7B 的 instruction following 能力，4-hop+ 问题可能产生不合理分解。**缓解**：设置 `max_branches=6`，超出时合并。
- **风险 5**：`compute_gcs` 的两两 NLI 调用量为 O(n²)，分支数 6 时需 15 次调用。**缓解**：DeBERTa batch_size=16，批量推理约 0.3s/batch on 4060，可接受。
- **风险 6**：θ 阈值需要在验证集上标定。**计划**：先用 HotpotQA 验证集（100条）做 grid search，再固定用于所有数据集。
