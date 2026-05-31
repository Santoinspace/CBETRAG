"""Exp4: Oracle (Dataset) vs Open-domain (ES) retrieval comparison.

Core purpose: prove early-stopping works in real retrieval (ES returns different
content each round) vs oracle retrieval (Dataset returns fixed gold+distractor).

4 configurations:
  IterativeRAG (Dataset) — baseline, fixed 3 rounds
  CBET        (Dataset) — oracle retrieval with early stopping
  IterativeRAG (ES)     — open-domain baseline, fixed 3 rounds
  CBET        (ES)      — open-domain with early stopping

Expected runtime: ~1h on RTX 4060 + vLLM 7B + ES.
Supports interrupt/resume: processed (config, qid) pairs are skipped.

Usage:
    uv run python experiments/run_exp4_es.py \
        --n_samples 200 \
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
from src.retriever import ElasticRetriever
from src.experiment_runner import run_experiment_parallel


# ── configs ───────────────────────────────────────────────────────────────────

CONFIGS = [
    {"retriever": "dataset",  "method": "IterativeRAG"},
    {"retriever": "dataset",  "method": "CBET"},
    {"retriever": "elastic",  "method": "IterativeRAG"},
    {"retriever": "elastic",  "method": "CBET"},
]

CONFIG_NAMES = {
    ("dataset", "IterativeRAG"): "IterRAG-Dataset",
    ("dataset", "CBET"):        "CBET-Dataset",
    ("elastic", "IterativeRAG"): "IterRAG-ES",
    ("elastic", "CBET"):        "CBET-ES",
}


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


# ── retrievers ────────────────────────────────────────────────────────────────

class DatasetPassageRetriever:
    def __init__(self, corpus: list[str]):
        self._corpus = corpus
    def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        return self._corpus[:top_k]


# ── method runners ────────────────────────────────────────────────────────────

def run_iterativerag(q: Question, llm: VLLMClient, retriever, rounds: int = 3) -> dict:
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
        "retrieval_rounds": rounds, "lm_calls": lm_calls,
        "time_seconds": time.time() - t0,
        "answer": result,
        "contains": int(q.answer.lower() in result.lower()),
    }


def run_cbet(q: Question, llm: VLLMClient, retriever, nli: NLIScorer,
             theta: float = 0.50) -> dict:
    t0 = time.time()
    probe = ParametricProbe(llm)
    config = CBETConfig(theta=theta, tau=0.5, max_iterations=3, min_iterations=1,
                        max_branches=6, gcs_conflict_threshold=0.35)
    controller = CBETController(llm, retriever, nli, probe, config)
    result = controller.solve(q)
    early_stopped = (result.iterations < config.max_iterations
                     and result.cs_score >= theta)
    return {
        "retrieval_rounds": result.iterations,
        "lm_calls": result.log.get("total_lm_calls", 0),
        "dag_size": len(result.dag.sub_questions),
        "dag_success": result.log.get("dag_success", not result.dag.fallback),
        "dag_fallback": result.log.get("dag_fallback", result.dag.fallback),
        "dag_hop_count": result.log.get("dag_hop_count", result.dag.get_hop_count()),
        "final_cs": result.cs_score,
        "gcs": result.log.get("gcs", 0.0),
        "edge_scores": result.log.get("edge_scores", []),
        "conflicts_detected": len(result.log.get("conflicts_detected", [])),
        "overrides_triggered": len(result.log.get("overrides_triggered", [])),
        "early_stopped": early_stopped,
        "time_seconds": time.time() - t0,
        "answer": result.answer,
        "contains": int(q.answer.lower() in result.answer.lower()),
    }


# ── resume / output ──────────────────────────────────────────────────────────

def load_completed(path: str) -> tuple[list[dict], set[tuple[str, str]]]:
    """Return (results, set of (config_name, qid) already done)."""
    if not os.path.exists(path):
        return [], set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        done = {(r["config"], r["qid"]) for r in data}
        print(f"  Resume: {len(data)} rows, {len(done)} (config,qid) pairs")
        return data, done
    except Exception as e:
        print(f"  Resume: could not load {path}: {e}")
        return [], set()


def summarize(results: list[dict], config_name: str) -> dict:
    sub = [r for r in results if r["config"] == config_name]
    if not sub:
        return {}
    n = len(sub)
    em = [em_score(r["answer"], r["gold_answer"]) for r in sub]
    f1 = [f1_score(r["answer"], r["gold_answer"]) for r in sub]
    cont = [r.get("contains", 0) for r in sub]
    cs = [r.get("final_cs", 0) for r in sub if r.get("final_cs") is not None]
    early = [r.get("early_stopped", False) for r in sub]
    return {
        "config": config_name, "n": n,
        "em": 100 * sum(em) / n, "f1": 100 * sum(f1) / n,
        "contains": 100 * sum(cont) / n,
        "avg_ret": sum(r["retrieval_rounds"] for r in sub) / n,
        "avg_lm": sum(r["lm_calls"] for r in sub) / n,
        "avg_cs": sum(cs) / len(cs) if cs else 0,
        "early_stop_pct": 100 * sum(early) / n,
        "avg_time": sum(r["time_seconds"] for r in sub) / n,
    }


def print_comparison_table(rows: list[dict]) -> None:
    """Print the Oracle vs Open-domain comparison table."""
    print()
    print("=== Table: Oracle vs Open-domain Retrieval (HotpotQA) ===")
    print(f"{'Setting':<22} {'N':>4} {'EM':>6} {'F1':>6} {'Contains%':>10} "
          f"{'Avg-Ret':>8} {'EStop%':>7} {'Avg-CS':>8} {'Avg-LM':>8}")
    print("-" * 92)

    # Group by retriever type
    dataset_rows = [r for r in rows if "Dataset" in r.get("config", "")]
    es_rows = [r for r in rows if "ES" in r.get("config", "")]

    for r in dataset_rows:
        estop = f"{r['early_stop_pct']:.1f}" if r.get("early_stop_pct") else "—"
        avg_cs = f"{r['avg_cs']:.3f}" if r.get("avg_cs") else "—"
        print(f"{r['config']:<22} {r['n']:>4} {r['em']:>6.1f} {r['f1']:>6.1f} "
              f"{r['contains']:>9.1f}% {r['avg_ret']:>8.2f} {estop:>7} {avg_cs:>8} "
              f"{r['avg_lm']:>8.1f}")

    if dataset_rows and es_rows:
        print("-" * 92)

    for r in es_rows:
        estop = f"{r['early_stop_pct']:.1f}" if r.get("early_stop_pct") else "—"
        avg_cs = f"{r['avg_cs']:.3f}" if r.get("avg_cs") else "—"
        print(f"{r['config']:<22} {r['n']:>4} {r['em']:>6.1f} {r['f1']:>6.1f} "
              f"{r['contains']:>9.1f}% {r['avg_ret']:>8.2f} {estop:>7} {avg_cs:>8} "
              f"{r['avg_lm']:>8.1f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp4: Oracle vs ES comparison")
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--vllm_url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm_model", default="qwen25-7b")
    parser.add_argument("--theta", type=float, default=0.50)
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel workers (2 conservative for ES)")
    parser.add_argument("--es_host", default="localhost")
    parser.add_argument("--es_port", type=int, default=9200)
    parser.add_argument("--es_index", default="wiki")
    parser.add_argument("--output", default="experiments/results/exp4_es.json")
    args = parser.parse_args()

    print("=" * 80)
    print(f"Exp4 — Oracle vs Open-domain: {args.dataset}, n={args.n_samples}")
    print(f"vLLM: {args.vllm_url} / {args.vllm_model}, theta={args.theta}, workers={args.workers}")
    print(f"ES: {args.es_host}:{args.es_port}/{args.es_index}")
    print("=" * 80)

    # ── Shared resources ──────────────────────────────────────────────────
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

    # ES health check
    es_retriever = ElasticRetriever(index_name=args.es_index, host=args.es_host,
                                     port=args.es_port, top_k=5)
    if not es_retriever.is_available():
        print("[ERROR] ElasticSearch not available!")
        print("  Start with: docker start elasticsearch")
        sys.exit(1)
    doc_count = es_retriever.get_doc_count()
    print(f"  [OK] ES: {doc_count:,} passages")
    if doc_count < 20_000_000:
        print(f"  [WARN] Index may be incomplete (< 20M docs)")
    test_hits = es_retriever.retrieve("Who directed Titanic?", top_k=1)
    print(f"  [OK] ES retrieval test: {len(test_hits)} hit(s)")

    # Shared NLIScorer (GPU, cache shared across all configs)
    nli = NLIScorer(model_path="./models/nli-deberta-v3-base", device="auto",
                     theta=args.theta, gcs_conflict_threshold=0.35)

    questions = load_dataset(args.dataset, n_samples=args.n_samples)
    print(f"Loaded {len(questions)} questions")

    # ── Build task list ───────────────────────────────────────────────────
    all_results, done = load_completed(args.output)

    tasks = []
    for cfg in CONFIGS:
        rtype = cfg["retriever"]
        method = cfg["method"]
        cname = CONFIG_NAMES[(rtype, method)]
        pending = [(q, rtype, method) for q in questions
                   if (cname, q.qid) not in done]
        tasks.extend(pending)
        print(f"  {cname}: {len(pending)}/{len(questions)} pending")

    print(f"\nTotal pending tasks: {len(tasks)}")

    if not tasks:
        print("All configs complete!")
    else:
        # Group by config name for sequential execution (avoid ES overload)
        for cfg in CONFIGS:
            rtype = cfg["retriever"]
            method = cfg["method"]
            cname = CONFIG_NAMES[(rtype, method)]

            cfg_tasks = [(q, rt, m) for (q, rt, m) in tasks
                         if rt == rtype and m == method]
            if not cfg_tasks:
                continue

            print(f"\n=== {cname} ({len(cfg_tasks)} tasks) ===")

            def process_task(task, _rtype=rtype, _method=method):
                q, rt, m = task
                if _rtype == "dataset":
                    retriever = DatasetPassageRetriever(
                        q.gold_passages + q.distractor_passages)
                else:
                    retriever = es_retriever

                if _method == "IterativeRAG":
                    r = run_iterativerag(q, llm, retriever)
                else:
                    r = run_cbet(q, llm, retriever, nli, theta=args.theta)

                r.update({
                    "config": CONFIG_NAMES[(_rtype, _method)],
                    "retriever_type": _rtype,
                    "method": _method,
                    "qid": q.qid,
                    "dataset": q.dataset,
                    "gold_answer": q.answer,
                })
                return r

            batch_results = run_experiment_parallel(
                questions=cfg_tasks,
                method_fn=process_task,
                max_workers=args.workers,
                checkpoint_every=20,
                checkpoint_fn=None,
                desc=cname,
            )

            all_results.extend(batch_results)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {len(all_results)} total results")

    # ── Final save + summary ──────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_results)} results to {args.output}")
    print(f"LLM: {llm.cache_stats()}")
    print(f"NLI: {nli.cache_stats()}")
    if es_retriever:
        print(f"ES:  {es_retriever.cache_stats()}")

    # Summary table
    config_names_ordered = [CONFIG_NAMES[(c["retriever"], c["method"])]
                            for c in CONFIGS]
    rows = [summarize(all_results, cn) for cn in config_names_ordered]
    rows = [r for r in rows if r]
    print_comparison_table(rows)

    # Save summary
    summary_path = args.output.replace(".json", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
