"""Exp2: CBET ablation study — 5 variants on HotpotQA.

Variants (matches paper Section 5.4):
  full            — complete CBET
  no_cross_branch — disable cross-branch GCS (always 1.0)
  no_override     — disable epistemic override
  entropy_only    — disable NLI claim extraction (token-entropy only)
  fixed_rounds    — disable CS stopping (always max iterations)

Expected runtime: 2-3h on RTX 4060 + vLLM 7B.
Supports interrupt/resume: if output JSON exists, processed (variant, qid) pairs are skipped.

Usage:
    uv run python experiments/run_exp2_ablation.py \
        --n_samples 200 \
        --vllm_model qwen25-7b
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


ABLATION_VARIANTS = ["full", "no_cross_branch", "no_override", "entropy_only", "fixed_rounds"]


def safe_str(s: str, max_len: int = 80) -> str:
    return "".join(ch if ord(ch) < 128 else "?" for ch in s[:max_len])


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


def build_config(variant: str, theta: float) -> CBETConfig:
    config = CBETConfig(theta=theta, tau=0.5, max_iterations=3, min_iterations=1,
                        max_branches=6, gcs_conflict_threshold=0.35)
    if variant == "no_cross_branch":
        config.skip_cross_branch_nli = True
    elif variant == "no_override":
        config.tau = 2.0  # never triggers
    elif variant == "entropy_only":
        config.nli_claim_extraction = False
    elif variant == "fixed_rounds":
        config.theta = 2.0  # never stops early
    return config


def run_variant(q: Question, llm: VLLMClient, variant: str, theta: float) -> dict:
    t0 = time.time()
    retriever = DatasetPassageRetriever(q.gold_passages + q.distractor_passages)
    nli = NLIScorer(model_path="./models/nli-deberta-v3-base", device="cpu",
                     theta=theta, gcs_conflict_threshold=0.35)
    probe = ParametricProbe(llm)
    config = build_config(variant, theta)
    controller = CBETController(llm, retriever, nli, probe, config)
    result = controller.solve(q)
    return {
        "variant": variant, "method": f"CBET-{variant}", "qid": q.qid,
        "answer": result.answer, "gold_answer": q.answer,
        "retrieval_rounds": result.iterations,
        "lm_calls": result.log.get("total_lm_calls", 0),
        "dag_size": len(result.dag.sub_questions),
        "final_cs": result.cs_score,
        "gcs": result.log.get("gcs", 0.0),
        "conflicts_detected": len(result.log.get("conflicts_detected", [])),
        "overrides_triggered": len(result.log.get("overrides_triggered", [])),
        "noisy_evicted": len(result.log.get("noisy_evicted", [])),
        "time_seconds": time.time() - t0,
        "contains": int(q.answer.lower() in result.answer.lower()),
    }


def load_completed(path: str) -> tuple[list[dict], set[tuple[str, str]]]:
    if not os.path.exists(path):
        return [], set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        done = {(r["variant"], r["qid"]) for r in data}
        print(f"  Resume: {len(data)} rows, {len(done)} (variant,qid) pairs")
        return data, done
    except Exception as e:
        print(f"  Resume: could not load {path}: {e}")
        return [], set()


def summarize(results: list[dict], variant: str) -> dict:
    sub = [r for r in results if r["variant"] == variant]
    if not sub:
        return {}
    n = len(sub)
    em = [em_score(r["answer"], r["gold_answer"]) for r in sub]
    f1 = [f1_score(r["answer"], r["gold_answer"]) for r in sub]
    cont = [r.get("contains", 0) for r in sub]
    cs = [r.get("final_cs", 0) for r in sub]
    return {
        "variant": variant, "n": n,
        "em": 100 * sum(em) / n, "f1": 100 * sum(f1) / n,
        "contains": 100 * sum(cont) / n,
        "avg_ret": sum(r["retrieval_rounds"] for r in sub) / n,
        "avg_lm": sum(r["lm_calls"] for r in sub) / n,
        "avg_cs": sum(cs) / n,
        "avg_conflicts": sum(r.get("conflicts_detected", 0) for r in sub) / n,
        "avg_overrides": sum(r.get("overrides_triggered", 0) for r in sub) / n,
        "avg_time": sum(r["time_seconds"] for r in sub) / n,
    }


def print_table(rows: list[dict]) -> None:
    print()
    print(f"{'Variant':<20} {'N':>4} {'EM':>6} {'F1':>6} {'Contains%':>10} "
          f"{'Avg-Ret':>8} {'Avg-LM':>8} {'Avg-CS':>8} {'Conflicts':>10}")
    print("-" * 95)
    for r in rows:
        if not r:
            continue
        print(f"{r['variant']:<20} {r['n']:>4} {r['em']:>6.1f} {r['f1']:>6.1f} "
              f"{r['contains']:>10.1f} {r['avg_ret']:>8.2f} {r['avg_lm']:>8.1f} "
              f"{r['avg_cs']:>8.3f} {r['avg_conflicts']:>10.2f}")


def main():
    parser = argparse.ArgumentParser(description="Exp2: CBET ablation study")
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--variants", default=",".join(ABLATION_VARIANTS))
    parser.add_argument("--vllm_url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm_model", default="qwen25-7b")
    parser.add_argument("--theta", type=float, default=0.50)
    parser.add_argument("--output", default="experiments/results/exp2_ablation.json")
    args = parser.parse_args()

    variants = args.variants.split(",")
    print("=" * 80)
    print(f"Exp2 — Ablation Study: {args.dataset}, n={args.n_samples}")
    print(f"Variants: {variants}")
    print(f"vLLM: {args.vllm_url} / {args.vllm_model}, theta={args.theta}")
    print("=" * 80)

    llm = VLLMClient(base_url=args.vllm_url, model=args.vllm_model)
    questions = load_dataset(args.dataset, n_samples=args.n_samples)
    print(f"Loaded {len(questions)} questions")

    all_results, done = load_completed(args.output)

    for variant in variants:
        print(f"\n=== Variant: {variant} ===")
        for i, q in enumerate(questions):
            if (variant, q.qid) in done:
                continue
            r = run_variant(q, llm, variant, args.theta)
            all_results.append(r)
            done.add((variant, q.qid))
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(questions)}] EM={em_score(r['answer'], q.answer)} "
                      f"CS={r.get('final_cs',0):.3f} DAG={r.get('dag_size')}")
            # Periodic save
            if (i + 1) % 20 == 0:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_results)} results to {args.output}")

    rows = [summarize(all_results, v) for v in variants]
    rows = [r for r in rows if r]
    print_table(rows)

    summary_path = args.output.replace(".json", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
