"""Exp3: θ sensitivity analysis — CBET with different θ thresholds.

Sweeps θ ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9} on 100 HotpotQA samples.
Reports F1, EarlyStop%, Avg-iterations per θ value.

Expected runtime: ~1h on RTX 4060 + vLLM 7B.
Supports interrupt/resume: if output JSON exists, processed (theta, qid) pairs are skipped.

Usage:
    uv run python experiments/run_exp3_theta.py \
        --n_samples 100 \
        --vllm_model qwen25-7b \
        --workers 2
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
from src.experiment_runner import run_experiment_parallel


THETA_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def em_score(pred: str, gold: str) -> int:
    return int(pred.strip().lower() == gold.strip().lower())


def f1_score(pred: str, gold: str) -> float:
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


class DatasetPassageRetriever:
    def __init__(self, corpus: list[str]):
        self._corpus = corpus
    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        return self._corpus[:top_k]


def run_cbet_theta(q: Question, llm: VLLMClient, nli: NLIScorer,
                   theta: float) -> dict:
    """Run one (question, theta) pair with shared NLIScorer."""
    t0 = time.time()
    retriever = DatasetPassageRetriever(q.gold_passages + q.distractor_passages)
    probe = ParametricProbe(llm)
    config = CBETConfig(theta=theta, tau=0.5, max_iterations=3, min_iterations=1,
                        max_branches=6, gcs_conflict_threshold=0.35)
    controller = CBETController(llm, retriever, nli, probe, config)
    result = controller.solve(q)
    early_stopped = (result.iterations < config.max_iterations
                     and result.cs_score >= theta)
    return {
        "theta": theta, "qid": q.qid, "dataset": q.dataset,
        "answer": result.answer, "gold_answer": q.answer,
        "retrieval_rounds": result.iterations,
        "lm_calls": result.log.get("total_lm_calls", 0),
        "final_cs": result.cs_score,
        "early_stopped": early_stopped,
        "time_seconds": time.time() - t0,
        "contains": int(q.answer.lower() in result.answer.lower()),
    }


def load_completed(path: str) -> tuple[list[dict], set[tuple[float, str]]]:
    if not os.path.exists(path):
        return [], set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        done = {(r["theta"], r["qid"]) for r in data}
        print(f"  Resume: {len(data)} rows, {len(done)} (theta,qid) pairs")
        return data, done
    except Exception as e:
        print(f"  Resume: could not load {path}: {e}")
        return [], set()


def summarize(results: list[dict], theta: float) -> dict:
    sub = [r for r in results if r["theta"] == theta]
    if not sub:
        return {}
    n = len(sub)
    em = [em_score(r["answer"], r["gold_answer"]) for r in sub]
    f1 = [f1_score(r["answer"], r["gold_answer"]) for r in sub]
    early = [r.get("early_stopped", False) for r in sub]
    return {
        "theta": theta, "n": n,
        "em": 100 * sum(em) / n, "f1": 100 * sum(f1) / n,
        "early_stop_pct": 100 * sum(early) / n,
        "avg_ret": sum(r["retrieval_rounds"] for r in sub) / n,
        "avg_lm": sum(r["lm_calls"] for r in sub) / n,
        "avg_cs": sum(r.get("final_cs", 0) for r in sub) / n,
    }


def print_table(rows: list[dict]) -> None:
    print()
    print(f"{'Theta':>6} {'N':>4} {'EM':>6} {'F1':>6} {'EarlyStop%':>11} "
          f"{'Avg-Ret':>8} {'Avg-LM':>8} {'Avg-CS':>8}")
    print("-" * 65)
    for r in rows:
        if not r:
            continue
        print(f"{r['theta']:>6.1f} {r['n']:>4} {r['em']:>6.1f} {r['f1']:>6.1f} "
              f"{r['early_stop_pct']:>11.1f} {r['avg_ret']:>8.2f} "
              f"{r['avg_lm']:>8.1f} {r['avg_cs']:>8.3f}")


def main():
    parser = argparse.ArgumentParser(description="Exp3: theta sensitivity")
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--thetas", nargs="+", type=float, default=THETA_VALUES)
    parser.add_argument("--vllm_url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm_model", default="qwen25-7b")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel workers (2 conservative)")
    parser.add_argument("--output", default="experiments/results/exp3_theta.json")
    args = parser.parse_args()

    print("=" * 80)
    print(f"Exp3 — Theta Sensitivity: {args.dataset}, n={args.n_samples}")
    print(f"Thetas: {args.thetas}")
    print(f"vLLM: {args.vllm_url} / {args.vllm_model}, workers={args.workers}")
    print("=" * 80)

    # Shared resources
    llm = VLLMClient(base_url=args.vllm_url, model=args.vllm_model)

    # Model consistency check
    try:
        models = llm._client.models.list()
        server_models = [m.id for m in models.data]
        if args.vllm_model not in server_models:
            print(f"[ERROR] Expected '{args.vllm_model}' not on server: {server_models}")
            sys.exit(1)
        print(f"  [OK] vLLM model: '{args.vllm_model}'")
    except Exception as e:
        print(f"  [WARN] Could not verify model: {e}")

    # Shared NLIScorer (GPU, cache shared across all theta runs)
    nli = NLIScorer(model_path="./models/nli-deberta-v3-base", device="auto",
                     theta=0.5, gcs_conflict_threshold=0.35)

    questions = load_dataset(args.dataset, n_samples=args.n_samples)
    print(f"Loaded {len(questions)} questions")

    all_results, done = load_completed(args.output)

    for theta in args.thetas:
        print(f"\n=== theta={theta} ===")

        # Build task list for this theta
        tasks = [(q, theta) for q in questions if (theta, q.qid) not in done]
        print(f"  Pending: {len(tasks)}/{len(questions)}")

        if not tasks:
            continue

        def process_task(task):
            q, t = task
            return run_cbet_theta(q, llm, nli, t)

        batch_results = run_experiment_parallel(
            questions=tasks,
            method_fn=process_task,
            max_workers=args.workers,
            checkpoint_every=20,
            checkpoint_fn=None,
            desc=f"theta={theta}",
        )

        all_results.extend(batch_results)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {len(all_results)} total results")

    # Final save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_results)} results to {args.output}")
    print(f"LLM: {llm.cache_stats()}")
    print(f"NLI: {nli.cache_stats()}")

    rows = [summarize(all_results, t) for t in args.thetas]
    rows = [r for r in rows if r]
    print_table(rows)

    summary_path = args.output.replace(".json", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
