"""Exp4: Open-domain CBET with ElasticSearch Wikipedia index (21M passages).

Tests CBET in a realistic open-domain setting where retrieval quality is not given.
Compares NoRAG, SingleRAG (BM25 top-5), and CBET with ElasticRetriever.

Expected runtime: 2-3h on RTX 4060 + vLLM 7B (ES retrieval adds ~0.1s/query).
Supports interrupt/resume: if output JSON exists, processed (method, qid) pairs are skipped.

Usage:
    uv run python experiments/run_exp4_es.py \
        --n_samples 200 \
        --vllm_model qwen25-7b \
        --es_host localhost --es_port 9200 --es_index wiki

Requires: docker ElasticSearch container running with wiki index (21M passages).
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
from src.retriever import ElasticRetriever


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


def _run_llm(llm: VLLMClient, prompt: str, max_tokens: int = 128) -> str:
    try:
        return llm.generate(prompt, max_new_tokens=max_tokens, temperature=0.0).text.strip()
    except Exception as e:
        print(f"    [WARN] LLM call failed: {e}")
        return ""


def run_norag(q: Question, llm: VLLMClient) -> dict:
    t0 = time.time()
    prompt = f"Answer this question in 1-5 words. Do not explain.\n\nQuestion: {q.query}\nAnswer:"
    ans = _run_llm(llm, prompt)
    return {
        "method": "NoRAG", "qid": q.qid,
        "answer": ans, "gold_answer": q.answer,
        "retrieval_rounds": 0, "lm_calls": 1,
        "time_seconds": time.time() - t0,
        "contains": int(q.answer.lower() in ans.lower()),
    }


def run_singlerag(q: Question, llm: VLLMClient, retriever: ElasticRetriever) -> dict:
    t0 = time.time()
    passages = retriever.retrieve(q.query, top_k=5)
    evidence = "\n".join(f"- {p[:300]}" for p in passages)
    prompt = (f"Use the following evidence to answer the question in 1-5 words.\n\n"
              f"Evidence:\n{evidence}\n\nQuestion: {q.query}\n"
              f"Do not explain. Just the answer.\nAnswer:")
    ans = _run_llm(llm, prompt)
    return {
        "method": "SingleRAG", "qid": q.qid,
        "answer": ans, "gold_answer": q.answer,
        "retrieval_rounds": 1, "lm_calls": 1,
        "time_seconds": time.time() - t0,
        "contains": int(q.answer.lower() in ans.lower()),
        "passages_retrieved": len(passages),
    }


def run_cbet(q: Question, llm: VLLMClient, retriever: ElasticRetriever,
             theta: float = 0.50) -> dict:
    t0 = time.time()
    nli = NLIScorer(model_path="./models/nli-deberta-v3-base", device="cpu",
                     theta=theta, gcs_conflict_threshold=0.35)
    probe = ParametricProbe(llm)
    config = CBETConfig(theta=theta, tau=0.5, max_iterations=3, min_iterations=1,
                        max_branches=6, gcs_conflict_threshold=0.35)
    controller = CBETController(llm, retriever, nli, probe, config)
    result = controller.solve(q)
    return {
        "method": "CBET", "qid": q.qid,
        "answer": result.answer, "gold_answer": q.answer,
        "retrieval_rounds": result.iterations,
        "lm_calls": result.log.get("total_lm_calls", 0),
        "dag_size": len(result.dag.sub_questions),
        "final_cs": result.cs_score,
        "gcs": result.log.get("gcs", 0.0),
        "time_seconds": time.time() - t0,
        "contains": int(q.answer.lower() in result.answer.lower()),
    }


def load_completed(path: str) -> tuple[list[dict], set[tuple[str, str]]]:
    if not os.path.exists(path):
        return [], set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        done = {(r["method"], r["qid"]) for r in data}
        print(f"  Resume: {len(data)} rows, {len(done)} (method,qid) pairs")
        return data, done
    except Exception as e:
        print(f"  Resume: could not load {path}: {e}")
        return [], set()


def summarize(results: list[dict], method: str) -> dict:
    sub = [r for r in results if r["method"] == method]
    if not sub:
        return {}
    n = len(sub)
    em = [em_score(r["answer"], r["gold_answer"]) for r in sub]
    f1 = [f1_score(r["answer"], r["gold_answer"]) for r in sub]
    cont = [r.get("contains", 0) for r in sub]
    cs = [r.get("final_cs", 0) for r in sub if r.get("final_cs") is not None]
    return {
        "method": method, "n": n,
        "em": 100 * sum(em) / n, "f1": 100 * sum(f1) / n,
        "contains": 100 * sum(cont) / n,
        "avg_ret": sum(r["retrieval_rounds"] for r in sub) / n,
        "avg_lm": sum(r["lm_calls"] for r in sub) / n,
        "avg_cs": sum(cs) / len(cs) if cs else 0,
        "avg_time": sum(r["time_seconds"] for r in sub) / n,
    }


def print_table(rows: list[dict]) -> None:
    print()
    print(f"{'Method':<15} {'N':>4} {'EM':>6} {'F1':>6} {'Contains%':>10} "
          f"{'Avg-Ret':>8} {'Avg-LM':>8} {'Avg-CS':>8} {'Time(s)':>8}")
    print("-" * 80)
    for r in rows:
        if not r:
            continue
        print(f"{r['method']:<15} {r['n']:>4} {r['em']:>6.1f} {r['f1']:>6.1f} "
              f"{r['contains']:>10.1f} {r['avg_ret']:>8.2f} {r['avg_lm']:>8.1f} "
              f"{r.get('avg_cs',0):>8.3f} {r['avg_time']:>8.1f}")


def main():
    parser = argparse.ArgumentParser(description="Exp4: open-domain with ES")
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--methods", default="NoRAG,SingleRAG,CBET")
    parser.add_argument("--vllm_url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm_model", default="qwen25-7b")
    parser.add_argument("--theta", type=float, default=0.50)
    parser.add_argument("--es_host", default="localhost")
    parser.add_argument("--es_port", type=int, default=9200)
    parser.add_argument("--es_index", default="wiki")
    parser.add_argument("--output", default="experiments/results/exp4_es.json")
    args = parser.parse_args()

    methods = set(args.methods.split(","))
    print("=" * 80)
    print(f"Exp4 — Open-Domain (ES): {args.dataset}, n={args.n_samples}")
    print(f"Methods: {sorted(methods)}")
    print(f"vLLM: {args.vllm_url} / {args.vllm_model}, theta={args.theta}")
    print(f"ES: {args.es_host}:{args.es_port}/{args.es_index}")
    print("=" * 80)

    llm = VLLMClient(base_url=args.vllm_url, model=args.vllm_model)
    es_retriever = ElasticRetriever(index_name=args.es_index, host=args.es_host,
                                     port=args.es_port, top_k=5)

    # Quick ES health check
    try:
        test_hits = es_retriever.retrieve("Berlin Wall", top_k=1)
        if test_hits:
            print(f"ES OK: '{safe_str(test_hits[0][:80])}'")
        else:
            print("ES WARNING: no hits for 'Berlin Wall' — index may be empty")
    except Exception as e:
        print(f"ES ERROR: {e}")
        print("Aborting — ElasticSearch must be running for this experiment.")
        sys.exit(1)

    questions = load_dataset(args.dataset, n_samples=args.n_samples)
    print(f"Loaded {len(questions)} questions")

    all_results, done = load_completed(args.output)

    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {safe_str(q.query[:100])}...")

        if "NoRAG" in methods and ("NoRAG", q.qid) not in done:
            r = run_norag(q, llm)
            all_results.append(r); done.add(("NoRAG", q.qid))
            print(f"  NoRAG: '{safe_str(r['answer'], 40)}' EM={em_score(r['answer'], q.answer)}")

        if "SingleRAG" in methods and ("SingleRAG", q.qid) not in done:
            r = run_singlerag(q, llm, es_retriever)
            all_results.append(r); done.add(("SingleRAG", q.qid))
            print(f"  SingleRAG: '{safe_str(r['answer'], 40)}' EM={em_score(r['answer'], q.answer)}")

        if "CBET" in methods and ("CBET", q.qid) not in done:
            r = run_cbet(q, llm, es_retriever, theta=args.theta)
            all_results.append(r); done.add(("CBET", q.qid))
            print(f"  CBET: '{safe_str(r['answer'], 40)}' EM={em_score(r['answer'], q.answer)} "
                  f"DAG={r.get('dag_size')} CS={r.get('final_cs',0):.3f}")

        if (i + 1) % 10 == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"  [saved checkpoint: {len(all_results)} rows]")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_results)} results to {args.output}")

    rows = [summarize(all_results, m) for m in ["NoRAG", "SingleRAG", "CBET"]]
    rows = [r for r in rows if r]
    print_table(rows)

    summary_path = args.output.replace(".json", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
