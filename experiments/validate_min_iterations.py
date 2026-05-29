"""Validate min_iterations fix: compare min_iterations=1 vs min_iterations=2.

This script runs 50 HotpotQA samples with both configurations and reports
objective metrics without directional claims.

Usage:
    uv run python experiments/validate_min_iterations.py --n_samples 50
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_client import VLLMClient
from src.data_adapter import load_dataset, Question
from src.cbet_controller import CBETController, CBETConfig
from src.parametric_probe import ParametricProbe
from src.nli_scorer import NLIScorer


class DatasetPassageRetriever:
    def __init__(self, corpus: list[str]):
        self._corpus = corpus

    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        return self._corpus[:top_k]


def run_cbet(q: Question, llm: VLLMClient, nli: NLIScorer,
             min_iter: int, theta: float = 0.50) -> dict:
    corpus = q.gold_passages + q.distractor_passages
    retriever = DatasetPassageRetriever(corpus)
    probe = ParametricProbe(llm)
    config = CBETConfig(
        theta=theta, tau=0.5, max_iterations=3, max_branches=6,
        min_iterations=min_iter,
        gcs_conflict_threshold=0.35
    )    controller = CBETController(llm, retriever, nli, probe, config)
    result = controller.solve(q)

    return {
        "qid": q.qid,
        "min_iterations": min_iter,
        "answer": result.answer,
        "gold_answer": q.answer,
        "em": int(result.answer.strip().lower() == q.answer.strip().lower()),
        "f1": _f1(result.answer, q.answer),
        "iterations": result.iterations,
        "final_cs": result.cs_score,
        "early_stopped": result.log.get("early_stopped", False),
        "total_lm_calls": result.log.get("total_lm_calls", 0),
        "dag_size": len(result.dag.sub_questions),
    }


def _f1(pred: str, gold: str) -> float:
    p = pred.strip().lower().split()
    g = gold.strip().lower().split()
    if not p or not g:
        return 0.0
    common = set(p) & set(g)
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--vllm_url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm_model", default="qwen25-7b")
    parser.add_argument("--theta", type=float, default=0.50)
    parser.add_argument("--output", type=str,
                        default="experiments/results/min_iterations_validation.json")
    args = parser.parse_args()

    print(f"Validating min_iterations: {args.n_samples} HotpotQA samples")
    print(f"vLLM: {args.vllm_url} / {args.vllm_model}, theta={args.theta}")
    print("=" * 80)

    llm = VLLMClient(base_url=args.vllm_url, model=args.vllm_model)

    # Shared NLIScorer: GPU + cache reused across all samples
    nli = NLIScorer(model_path="./models/nli-deberta-v3-base", device="auto",
                     theta=args.theta, gcs_conflict_threshold=0.35)

    questions = load_dataset("hotpotqa", n_samples=args.n_samples)

    results_m1 = []
    results_m2 = []
    t_start = time.time()

    for i, q in enumerate(questions):
        # Run with min_iterations=1
        r1 = run_cbet(q, llm, nli, min_iter=1, theta=args.theta)
        results_m1.append(r1)

        # Run with min_iterations=2
        r2 = run_cbet(q, llm, nli, min_iter=2, theta=args.theta)
        results_m2.append(r2)

        print(f"[{i+1}/{args.n_samples}] min_iter=1: EM={r1['em']} F1={r1['f1']:.3f} "
              f"CS={r1['final_cs']:.3f} early_stop={r1['early_stopped']} | "
              f"min_iter=2: EM={r2['em']} F1={r2['f1']:.3f} "
              f"CS={r2['final_cs']:.3f} early_stop={r2['early_stopped']}")

    elapsed = time.time() - t_start

    # Aggregate metrics
    def aggregate(results: list[dict]) -> dict:
        n = len(results)
        em = [r["em"] for r in results]
        f1 = [r["f1"] for r in results]
        iters = [r["iterations"] for r in results]
        cs = [r["final_cs"] for r in results]
        lm = [r["total_lm_calls"] for r in results]
        estop = [r["early_stopped"] for r in results]
        estop_wrong = [r for r in results if r["early_stopped"] and r["em"] == 0]

        return {
            "n": n,
            "em": 100 * sum(em) / n,
            "f1": 100 * sum(f1) / n,
            "avg_iterations": sum(iters) / n,
            "avg_cs": sum(cs) / n,
            "avg_lm": sum(lm) / n,
            "early_stop_pct": 100 * sum(estop) / n,
            "early_stop_wrong": len(estop_wrong),
        }

    m1_agg = aggregate(results_m1)
    m2_agg = aggregate(results_m2)

    print(f"\nElapsed: {elapsed:.0f}s ({elapsed/args.n_samples:.1f}s/sample)")
    print(f"LLM: {llm.cache_stats()}")
    print(f"NLI: {nli.cache_stats()}")
    print("\n" + "=" * 80)
    print("=== Validation Results ===")
    print(f"{'Metric':<35} {'min_iter=1':>15} {'min_iter=2':>15}")
    print("-" * 70)
    print(f"{'CBET EM':<35} {m1_agg['em']:>15.1f} {m2_agg['em']:>15.1f}")
    print(f"{'CBET F1':<35} {m1_agg['f1']:>15.1f} {m2_agg['f1']:>15.1f}")
    print(f"{'EarlyStop%':<35} {m1_agg['early_stop_pct']:>14.1f}% {m2_agg['early_stop_pct']:>14.1f}%")
    print(f"{'Avg-Ret (iterations)':<35} {m1_agg['avg_iterations']:>15.2f} {m2_agg['avg_iterations']:>15.2f}")
    print(f"{'Avg-LM':<35} {m1_agg['avg_lm']:>15.1f} {m2_agg['avg_lm']:>15.1f}")
    print(f"{'EM=0 & CS>=0.5 (wrong early stop)':<35} {m1_agg['early_stop_wrong']:>12}/{args.n_samples} {m2_agg['early_stop_wrong']:>12}/{args.n_samples}")
    print("=" * 80)

    # Save results (overwrite existing)
    output = {
        "config": {
            "n_samples": args.n_samples,
            "vllm_url": args.vllm_url,
            "vllm_model": args.vllm_model,
            "theta": args.theta,
            "elapsed_seconds": elapsed,
        },
        "min_iterations_1": {"results": results_m1, "aggregate": m1_agg},
        "min_iterations_2": {"results": results_m2, "aggregate": m2_agg},
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
