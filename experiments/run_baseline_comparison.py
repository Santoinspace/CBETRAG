"""Self-contained baseline comparison experiment.

Runs NoRAG, SingleRAG, IterativeRAG, and CBET on the same questions
using the vLLM model and a keyword-based retriever over dataset passages.

Works without ElasticSearch, NLI model, or GPU — only needs the vLLM server.

Usage:
    python experiments/run_baseline_comparison.py \
        --dataset hotpotqa --n_samples 20 \
        --output_dir experiments/results/baseline_comparison.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import re
import unicodedata
from dataclasses import dataclass, field
from collections import Counter
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_client import VLLMClient
from src.data_adapter import load_dataset, Question
from src.dag_extractor import extract_dag


def safe_str(s: str, max_len: int = 80) -> str:
    """Truncate and replace non-ASCII chars for Windows console safety."""
    s = s[:max_len]
    # Replace characters that can't be encoded in GBK
    result = []
    for ch in s:
        try:
            ch.encode('gbk')
            result.append(ch)
        except UnicodeEncodeError:
            result.append('?')
    return ''.join(result)


# ═══════════════════════════════════════════════════════════════════════════════
# Keyword Retriever (replaces ElasticSearch for self-contained experiments)
# ═══════════════════════════════════════════════════════════════════════════════

class KeywordRetriever:
    """Retrieve passages by keyword overlap (TF-like scoring) against a corpus."""

    def __init__(self, corpus: list[str]):
        self._corpus = corpus

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        query_words = set(query.lower().split())
        if not query_words:
            return self._corpus[:top_k]
        scored = []
        for passage in self._corpus:
            p_words = set(passage.lower().split())
            overlap = len(query_words & p_words)
            if overlap > 0:
                scored.append((overlap, passage))
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored[:top_k]]


class DatasetPassageRetriever:
    """Return all dataset-provided context passages (gold + distractor) directly.

    Eliminates retrieval quality as a variable — uses the pre-collected passages
    that come with each question, consistent with AdaRAGUE's standard protocol.
    """

    def __init__(self, corpus: list[str]):
        self._corpus = corpus

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        """Return all available passages regardless of query (up to top_k)."""
        return self._corpus[:top_k]


def build_corpus_from_question(q: Question) -> list[str]:
    """Collect all passages from a question's context field."""
    return q.gold_passages + q.distractor_passages


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline Strategies
# ═══════════════════════════════════════════════════════════════════════════════

def _run_llm(llm: VLLMClient, prompt: str, max_tokens: int = 128) -> str:
    try:
        resp = llm.generate(prompt, max_new_tokens=max_tokens, temperature=0.0)
        return resp.text.strip()
    except Exception as e:
        print(f"    [WARN] LLM call failed: {e}")
        return ""


def baseline_norag(q: Question, llm: VLLMClient) -> dict:
    """Direct answer — no retrieval at all."""
    prompt = f"Answer this question in 1-5 words. Do not explain.\n\nQuestion: {q.query}\nAnswer:"
    start = time.time()
    answer = _run_llm(llm, prompt)
    elapsed = time.time() - start
    return {
        "method": "NoRAG",
        "qid": q.qid,
        "answer": answer,
        "gold_answer": q.answer,
        "time_seconds": elapsed,
        "retrieval_rounds": 0,
        "lm_calls": 1,
        "contains": int(q.answer.lower() in answer.lower()),
    }


def baseline_singlerag(q: Question, llm: VLLMClient, retriever: DatasetPassageRetriever) -> dict:
    """One round of retrieval, then answer with evidence."""
    start = time.time()
    passages = retriever.retrieve(q.query, top_k=5)
    evidence = "\n".join(f"- {p[:300]}" for p in passages)
    prompt = (
        f"Use the following evidence to answer the question in 1-5 words.\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Question: {q.query}\n"
        f"Do not explain. Just the answer.\nAnswer:"
    )
    answer = _run_llm(llm, prompt)
    elapsed = time.time() - start
    return {
        "method": "SingleRAG",
        "qid": q.qid,
        "answer": answer,
        "gold_answer": q.answer,
        "time_seconds": elapsed,
        "retrieval_rounds": 1,
        "lm_calls": 1,
        "contains": int(q.answer.lower() in answer.lower()),
    }


def baseline_iterativerag(q: Question, llm: VLLMClient, retriever: DatasetPassageRetriever,
                          rounds: int = 3) -> dict:
    """Multiple rounds: retrieve → partial answer → re-retrieve → final answer."""
    start = time.time()
    current_query = q.query
    all_evidence: list[str] = []
    lm_calls = 0

    for r in range(rounds):
        # Retrieve
        passages = retriever.retrieve(current_query, top_k=3)
        all_evidence.extend(passages)

        # Generate intermediate answer
        evidence_text = "\n".join(f"- {p[:300]}" for p in all_evidence[-6:])
        if r < rounds - 1:
            prompt = (
                f"Based on the evidence, provide a partial answer and identify "
                f"what additional information is needed.\n\n"
                f"Evidence:\n{evidence_text}\n\n"
                f"Question: {q.query}\nPartial answer:"
            )
        else:
            prompt = (
                f"Based on the evidence, answer the question in 1-5 words. "
                f"Do not explain. Just the answer.\n\n"
                f"Evidence:\n{evidence_text}\n\n"
                f"Question: {q.query}\nFinal answer:"
            )
        result = _run_llm(llm, prompt)
        lm_calls += 1

        if r < rounds - 1:
            # Use result to refine next query
            current_query = f"{q.query} {result[:200]}"

    elapsed = time.time() - start
    return {
        "method": "IterativeRAG",
        "qid": q.qid,
        "answer": result,
        "gold_answer": q.answer,
        "time_seconds": elapsed,
        "retrieval_rounds": rounds,
        "lm_calls": lm_calls,
        "contains": int(q.answer.lower() in result.lower()),
    }


def baseline_cbet(q: Question, llm: VLLMClient, retriever: DatasetPassageRetriever) -> dict:
    """Full CBET pipeline with keyword retriever + real DeBERTa NLI (CPU)."""
    from src.cbet_controller import CBETController, CBETConfig, BranchState
    from src.dag_extractor import extract_dag
    from src.parametric_probe import ParametricProbe
    from src.epistemic_override import EpistemicOverrider
    from src.nli_scorer import NLIScorer

    start = time.time()

    nli = NLIScorer(model_path="./models/nli-deberta-v3-base", device="cpu", theta=0.75)
    probe = ParametricProbe(llm)
    config = CBETConfig(theta=0.75, tau=0.5, max_iterations=3, max_branches=6)
    controller = CBETController(llm, retriever, nli, probe, config)

    result = controller.solve(q)
    elapsed = time.time() - start

    return {
        "method": "CBET",
        "qid": q.qid,
        "answer": result.answer,
        "gold_answer": q.answer,
        "time_seconds": elapsed,
        "retrieval_rounds": result.iterations,
        "lm_calls": result.log.get("total_lm_calls", 0),
        "dag_size": len(result.dag.sub_questions),
        "final_cs": result.cs_score,
        "conflicts_detected": len(result.log.get("conflicts_detected", [])),
        "overrides_triggered": len(result.log.get("overrides_triggered", [])),
        "contains": int(q.answer.lower() in result.answer.lower()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def em_score(pred: str, gold: str) -> int:
    return int(pred.strip().lower() == gold.strip().lower())


def f1_score(pred: str, gold: str) -> float:
    p_tok = pred.strip().lower().split()
    g_tok = gold.strip().lower().split()
    if not p_tok or not g_tok:
        return 0.0
    common = set(p_tok) & set(g_tok)
    if not common:
        return 0.0
    prec = len(common) / len(p_tok)
    rec = len(common) / len(g_tok)
    return 2 * prec * rec / (prec + rec)


def summarize(results: list[dict], method_name: str) -> dict:
    subset = [r for r in results if r["method"] == method_name]
    n = len(subset)
    if n == 0:
        return {}
    em_vals = [em_score(r["answer"], r["gold_answer"]) for r in subset]
    f1_vals = [f1_score(r["answer"], r["gold_answer"]) for r in subset]
    contains_vals = [r.get("contains", 0) for r in subset]
    cs_vals = [r.get("final_cs", 0) for r in subset if r.get("final_cs") is not None]
    return {
        "method": method_name,
        "n": n,
        "em": 100 * sum(em_vals) / n,
        "f1": 100 * sum(f1_vals) / n,
        "contains_rate": 100 * sum(contains_vals) / n,
        "avg_retrieval_rounds": sum(r["retrieval_rounds"] for r in subset) / n,
        "avg_lm_calls": sum(r["lm_calls"] for r in subset) / n,
        "avg_cs": sum(cs_vals) / len(cs_vals) if cs_vals else 0,
        "avg_time_seconds": sum(r["time_seconds"] for r in subset) / n,
    }


def print_table(summaries: list[dict]) -> None:
    print()
    print(f"{'Method':<18} {'EM':>6} {'F1':>6} {'Contains%':>10} {'Avg-Ret':>8} {'Avg-LM':>8} {'Avg-CS':>8} {'Time(s)':>8}")
    print("-" * 78)
    for s in summaries:
        if not s:
            continue
        print(f"{s['method']:<18} {s['em']:>6.1f} {s['f1']:>6.1f} {s.get('contains_rate', 0):>10.1f} "
              f"{s['avg_retrieval_rounds']:>8.1f} {s['avg_lm_calls']:>8.1f} "
              f"{s.get('avg_cs', 0):>8.3f} {s['avg_time_seconds']:>8.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run baseline comparison experiment")
    parser.add_argument("--dataset", default="hotpotqa",
                        choices=["hotpotqa", "musique", "2wikimultihopqa"])
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--output_dir", default="experiments/results/baseline_comparison.json")
    parser.add_argument("--vllm_url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm_model", default="/models/qwen")
    parser.add_argument("--methods", default="NoRAG,SingleRAG,IterativeRAG,CBET",
                        help="Comma-separated method names to run")
    parser.add_argument("--iterative_rounds", type=int, default=3,
                        help="Rounds for IterativeRAG")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_dir)), exist_ok=True)

    print(f"Baseline Comparison Experiment")
    print(f"  Dataset: {args.dataset}")
    print(f"  Samples: {args.n_samples}")
    print(f"  Methods: {args.methods}")
    print(f"  vLLM: {args.vllm_url} / {args.vllm_model}")
    print()

    # Load data
    questions = load_dataset(args.dataset, n_samples=args.n_samples)
    print(f"Loaded {len(questions)} questions")

    # Init LLM
    llm = VLLMClient(base_url=args.vllm_url, model=args.vllm_model)

    methods_to_run = set(args.methods.split(","))

    all_results: list[dict] = []

    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {safe_str(q.query[:100])}...")
        corpus = build_corpus_from_question(q)
        retriever = DatasetPassageRetriever(corpus)
        print(f"  Corpus: {len(corpus)} passages")

        if "NoRAG" in methods_to_run:
            r = baseline_norag(q, llm)
            all_results.append(r)
            print(f"  NoRAG: '{safe_str(r['answer'])}' (EM={em_score(r['answer'], q.answer)})")

        if "SingleRAG" in methods_to_run:
            r = baseline_singlerag(q, llm, retriever)
            all_results.append(r)
            print(f"  SingleRAG: '{safe_str(r['answer'])}' (EM={em_score(r['answer'], q.answer)})")

        if "IterativeRAG" in methods_to_run:
            r = baseline_iterativerag(q, llm, retriever, rounds=args.iterative_rounds)
            all_results.append(r)
            print(f"  IterativeRAG: '{safe_str(r['answer'])}' (EM={em_score(r['answer'], q.answer)})")

        if "CBET" in methods_to_run:
            r = baseline_cbet(q, llm, retriever)
            all_results.append(r)
            print(f"  CBET: '{safe_str(r['answer'])}' (EM={em_score(r['answer'], q.answer)}, "
                  f"DAG={r.get('dag_size','?')}, CS={r.get('final_cs',0):.3f})")

    # Save raw results
    with open(args.output_dir, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nRaw results saved to {args.output_dir}")

    # Summarize
    summaries = []
    for method in ["NoRAG", "SingleRAG", "IterativeRAG", "CBET"]:
        s = summarize(all_results, method)
        if s:
            summaries.append(s)

    print_table(summaries)

    # Also save summaries
    summary_path = args.output_dir.replace(".json", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")

    return summaries


if __name__ == "__main__":
    main()
