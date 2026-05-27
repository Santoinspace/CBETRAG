"""Main experiment: NoRAG / SingleRAG / IterativeRAG / CBET on HotpotQA + MuSiQue.

Dataset-passage retriever (gold+distractor) — isolates method behavior from retrieval quality.
Expected runtime: 3-5h on RTX 4060 + vLLM 7B.

Supports interrupt/resume: if the output JSON already exists, processed (method, qid) pairs
are loaded and skipped.

Usage:
    uv run python experiments/run_exp1_main.py \
        --datasets hotpotqa musique \
        --n_samples 500 \
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
from src.cbet_controller import CBETController, CBETConfig, BranchState
from src.dag_extractor import extract_dag
from src.parametric_probe import ParametricProbe
from src.epistemic_override import EpistemicOverrider
from src.nli_scorer import NLIScorer


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def safe_str(s: str, max_len: int = 80) -> str:
    s = s[:max_len]
    return "".join(ch if ord(ch) < 128 else "?" for ch in s)


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


def _run_llm(llm: VLLMClient, prompt: str, max_tokens: int = 128) -> str:
    try:
        return llm.generate(prompt, max_new_tokens=max_tokens, temperature=0.0).text.strip()
    except Exception as e:
        print(f"    [WARN] LLM call failed: {e}")
        return ""


class DatasetPassageRetriever:
    def __init__(self, corpus: list[str]):
        self._corpus = corpus
    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        return self._corpus[:top_k]


def build_corpus(q: Question) -> list[str]:
    return q.gold_passages + q.distractor_passages


# ═══════════════════════════════════════════════════════════════════════════════
# Baselines
# ═══════════════════════════════════════════════════════════════════════════════

def run_norag(q: Question, llm: VLLMClient) -> dict:
    t0 = time.time()
    prompt = f"Answer this question in 1-5 words. Do not explain.\n\nQuestion: {q.query}\nAnswer:"
    ans = _run_llm(llm, prompt)
    return {
        "method": "NoRAG", "qid": q.qid, "dataset": q.dataset,
        "answer": ans, "gold_answer": q.answer,
        "retrieval_rounds": 0, "lm_calls": 1,
        "time_seconds": time.time() - t0,
        "contains": int(q.answer.lower() in ans.lower()),
    }


def run_singlerag(q: Question, llm: VLLMClient, retriever: DatasetPassageRetriever) -> dict:
    t0 = time.time()
    passages = retriever.retrieve(q.query, top_k=5)
    evidence = "\n".join(f"- {p[:300]}" for p in passages)
    prompt = (f"Use the following evidence to answer the question in 1-5 words.\n\n"
              f"Evidence:\n{evidence}\n\nQuestion: {q.query}\n"
              f"Do not explain. Just the answer.\nAnswer:")
    ans = _run_llm(llm, prompt)
    return {
        "method": "SingleRAG", "qid": q.qid, "dataset": q.dataset,
        "answer": ans, "gold_answer": q.answer,
        "retrieval_rounds": 1, "lm_calls": 1,
        "time_seconds": time.time() - t0,
        "contains": int(q.answer.lower() in ans.lower()),
    }


def run_iterativerag(q: Question, llm: VLLMClient, retriever: DatasetPassageRetriever,
                      rounds: int = 3) -> dict:
    t0 = time.time()
    current_query = q.query
    all_ev: list[str] = []
    lm_calls = 0
    result = ""
    for r in range(rounds):
        passages = retriever.retrieve(current_query, top_k=3)
        all_ev.extend(passages)
        ev_text = "\n".join(f"- {p[:300]}" for p in all_ev[-6:])
        if r < rounds - 1:
            prompt = (f"Based on the evidence, provide a partial answer and identify "
                      f"what additional information is needed.\n\nEvidence:\n{ev_text}\n\n"
                      f"Question: {q.query}\nPartial answer:")
        else:
            prompt = (f"Based on the evidence, answer the question in 1-5 words. "
                      f"Do not explain. Just the answer.\n\nEvidence:\n{ev_text}\n\n"
                      f"Question: {q.query}\nFinal answer:")
        result = _run_llm(llm, prompt)
        lm_calls += 1
        if r < rounds - 1:
            current_query = f"{q.query} {result[:200]}"
    return {
        "method": "IterativeRAG", "qid": q.qid, "dataset": q.dataset,
        "answer": result, "gold_answer": q.answer,
        "retrieval_rounds": rounds, "lm_calls": lm_calls,
        "time_seconds": time.time() - t0,
        "contains": int(q.answer.lower() in result.lower()),
    }


def run_cbet(q: Question, llm: VLLMClient, retriever: DatasetPassageRetriever,
             theta: float = 0.50) -> dict:
    t0 = time.time()
    nli = NLIScorer(model_path="./models/nli-deberta-v3-base", device="cpu",
                     theta=theta, gcs_conflict_threshold=0.35)
    probe = ParametricProbe(llm)
    config = CBETConfig(theta=theta, tau=0.5, max_iterations=3, max_branches=6,
                        gcs_conflict_threshold=0.35)
    controller = CBETController(llm, retriever, nli, probe, config)
    result = controller.solve(q)
    return {
        "method": "CBET", "qid": q.qid, "dataset": q.dataset,
        "answer": result.answer, "gold_answer": q.answer,
        "retrieval_rounds": result.iterations,
        "lm_calls": result.log.get("total_lm_calls", 0),
        "dag_size": len(result.dag.sub_questions),
        "final_cs": result.cs_score,
        "conflicts_detected": len(result.log.get("conflicts_detected", [])),
        "overrides_triggered": len(result.log.get("overrides_triggered", [])),
        "time_seconds": time.time() - t0,
        "contains": int(q.answer.lower() in result.answer.lower()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Resume / Summary
# ═══════════════════════════════════════════════════════════════════════════════

def load_completed(path: str) -> tuple[list[dict], set[tuple[str, str]]]:
    """Return (existing_results, set of (method, qid) already processed)."""
    if not os.path.exists(path):
        return [], set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        done = {(r["method"], r["qid"]) for r in data}
        print(f"  Resume: loaded {len(data)} existing results, {len(done)} (method,qid) pairs")
        return data, done
    except Exception as e:
        print(f"  Resume: could not load {path}: {e}")
        return [], set()


def save_results(path: str, results: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def summarize(results: list[dict], method: str, dataset: str) -> dict:
    sub = [r for r in results if r["method"] == method and r.get("dataset") == dataset]
    if not sub:
        return {}
    n = len(sub)
    em = [em_score(r["answer"], r["gold_answer"]) for r in sub]
    f1 = [f1_score(r["answer"], r["gold_answer"]) for r in sub]
    cont = [r.get("contains", 0) for r in sub]
    cs_vals = [r.get("final_cs", 0) for r in sub if r.get("final_cs") is not None]
    return {
        "method": method, "dataset": dataset, "n": n,
        "em": 100 * sum(em) / n, "f1": 100 * sum(f1) / n,
        "contains": 100 * sum(cont) / n,
        "avg_ret": sum(r["retrieval_rounds"] for r in sub) / n,
        "avg_lm": sum(r["lm_calls"] for r in sub) / n,
        "avg_cs": sum(cs_vals) / len(cs_vals) if cs_vals else 0,
        "avg_time": sum(r["time_seconds"] for r in sub) / n,
    }


def print_table(rows: list[dict]) -> None:
    print()
    print(f"{'Dataset':<12} {'Method':<15} {'N':>4} {'EM':>6} {'F1':>6} "
          f"{'Contains%':>10} {'Avg-Ret':>8} {'Avg-LM':>8} {'Avg-CS':>8} {'Time(s)':>8}")
    print("-" * 95)
    for r in rows:
        if not r:
            continue
        print(f"{r['dataset']:<12} {r['method']:<15} {r['n']:>4} "
              f"{r['em']:>6.1f} {r['f1']:>6.1f} {r.get('contains',0):>10.1f} "
              f"{r['avg_ret']:>8.2f} {r['avg_lm']:>8.1f} "
              f"{r.get('avg_cs',0):>8.3f} {r['avg_time']:>8.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exp1: main comparison")
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "musique"])
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--methods", default="NoRAG,SingleRAG,IterativeRAG,CBET")
    parser.add_argument("--vllm_url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm_model", default="qwen25-7b")
    parser.add_argument("--theta", type=float, default=0.50)
    parser.add_argument("--output", default="experiments/results/exp1_main.json")
    args = parser.parse_args()

    methods = set(args.methods.split(","))
    print("=" * 80)
    print(f"Exp1 — Main Comparison: {args.datasets}, n={args.n_samples}, methods={sorted(methods)}")
    print(f"vLLM: {args.vllm_url} / {args.vllm_model}, theta={args.theta}")
    print("=" * 80)

    llm = VLLMClient(base_url=args.vllm_url, model=args.vllm_model)
    all_results, done = load_completed(args.output)

    for ds in args.datasets:
        print(f"\n=== Dataset: {ds} ===")
        questions = load_dataset(ds, n_samples=args.n_samples)
        print(f"Loaded {len(questions)} questions")

        for i, q in enumerate(questions):
            print(f"\n[{i+1}/{len(questions)}] {safe_str(q.query[:100])}...")
            retriever = DatasetPassageRetriever(build_corpus(q))

            if "NoRAG" in methods and ("NoRAG", q.qid) not in done:
                r = run_norag(q, llm)
                all_results.append(r); done.add(("NoRAG", q.qid))
                print(f"  NoRAG: '{safe_str(r['answer'], 40)}' EM={em_score(r['answer'], q.answer)}")

            if "SingleRAG" in methods and ("SingleRAG", q.qid) not in done:
                r = run_singlerag(q, llm, retriever)
                all_results.append(r); done.add(("SingleRAG", q.qid))
                print(f"  SingleRAG: '{safe_str(r['answer'], 40)}' EM={em_score(r['answer'], q.answer)}")

            if "IterativeRAG" in methods and ("IterativeRAG", q.qid) not in done:
                r = run_iterativerag(q, llm, retriever)
                all_results.append(r); done.add(("IterativeRAG", q.qid))
                print(f"  IterRAG: '{safe_str(r['answer'], 40)}' EM={em_score(r['answer'], q.answer)}")

            if "CBET" in methods and ("CBET", q.qid) not in done:
                r = run_cbet(q, llm, retriever, theta=args.theta)
                all_results.append(r); done.add(("CBET", q.qid))
                print(f"  CBET: '{safe_str(r['answer'], 40)}' EM={em_score(r['answer'], q.answer)} "
                      f"DAG={r.get('dag_size')} CS={r.get('final_cs',0):.3f}")

            # Periodic save every 10 questions
            if (i + 1) % 10 == 0:
                save_results(args.output, all_results)
                print(f"  [saved checkpoint: {len(all_results)} rows]")

    save_results(args.output, all_results)
    print(f"\nSaved {len(all_results)} results to {args.output}")

    # Summary table
    rows = []
    for ds in args.datasets:
        for m in ["NoRAG", "SingleRAG", "IterativeRAG", "CBET"]:
            s = summarize(all_results, m, ds)
            if s:
                rows.append(s)
    print_table(rows)

    # Save summary
    summary_path = args.output.replace(".json", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
