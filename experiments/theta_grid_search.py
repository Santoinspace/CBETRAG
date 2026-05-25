"""θ grid search — run CBET once per sample, record per-iteration CS, simulate θ thresholds.

Usage:
    uv run python experiments/theta_grid_search.py --dataset hotpotqa --n_samples 30 \
        --output_dir experiments/results/theta_grid.json
"""
from __future__ import annotations
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_client import VLLMClient
from src.nli_scorer import NLIScorer
from src.cbet_controller import CBETController, CBETConfig, BranchState
from src.parametric_probe import ParametricProbe
from src.data_adapter import load_dataset, Question
from experiments.run_baseline_comparison import DatasetPassageRetriever, build_corpus_from_question, em_score, f1_score


def run_single_with_cs_tracking(q: Question, llm: VLLMClient, nli: NLIScorer,
                                 probe: ParametricProbe) -> dict:
    """Run CBET on one question and record per-iteration CS scores.

    We patch CBETController.solve to record CS after each iteration
    while still allowing the loop to continue to max_iterations.
    """
    from src.dag_extractor import extract_dag, QuestionDAG
    from src.epistemic_override import EpistemicOverrider
    from src.nli_scorer import CompletenessResult

    corpus = build_corpus_from_question(q)
    retriever = DatasetPassageRetriever(corpus)
    # Use low theta so CS never stops early; we control iterations manually
    config = CBETConfig(theta=0.0, tau=0.5, max_iterations=3, max_branches=6)
    controller = CBETController(llm, retriever, nli, probe, config)

    overrider = EpistemicOverrider()

    # ── Inline solve() with per-iteration CS tracking ──
    dag = extract_dag(q.query, llm)
    if len(dag.sub_questions) > config.max_branches:
        dag.sub_questions = dag.sub_questions[:config.max_branches]

    branch_states = {sq.id: BranchState() for sq in dag.sub_questions}
    per_iter_cs: list[float] = []
    cs_result = None

    for iteration in range(1, config.max_iterations + 1):
        for parallel_batch in dag.get_execution_order():
            leaf_nodes = [sq for sq in parallel_batch if sq.is_leaf]
            retrieval_results = {}
            if leaf_nodes:
                retrieval_results.update(
                    controller._parallel_retrieve(leaf_nodes, branch_states)
                )
            for node in [sq for sq in parallel_batch if not sq.is_leaf]:
                enriched = controller._enrich_with_predecessor_answers(node, branch_states)
                retrieval_results[node.id] = controller._retrieve(enriched)

            for sq in parallel_batch:
                state = branch_states[sq.id]
                if retrieval_results.get(sq.id):
                    state.evidence = retrieval_results[sq.id]

                if not state.probed:
                    param_mem = probe.probe(sq.text)
                    gcs_for_conflict = cs_result.gcs if cs_result is not None else 1.0
                    conflict = probe.detect_conflict(
                        param_mem, state.evidence, nli, llm, gcs=gcs_for_conflict)
                    state.conflict = conflict
                    state.probed = True
                    if conflict.trust_retrieved > config.tau:
                        state.override_prompt = overrider.build(sq.text, state.evidence)

                if state.current_answer and state.evidence == state._last_evidence_for_answer:
                    pass
                else:
                    state.current_answer = controller._answer_branch(sq, state)
                    state._last_evidence_for_answer = state.evidence

        cs_result = nli.compute_completeness_score(
            branch_evidences=[branch_states[sq.id].evidence for sq in dag.sub_questions],
            branch_answers=[branch_states[sq.id].current_answer for sq in dag.sub_questions],
            sub_questions=[sq.text for sq in dag.sub_questions],
            llm_client=llm,
            skip_gcs=config.skip_cross_branch_nli,
        )
        per_iter_cs.append(cs_result.cs)

    final_answer = controller._generate_final_answer(q, branch_states, cs_result, dag)

    return {
        "qid": q.qid,
        "answer": final_answer,
        "gold_answer": q.answer,
        "dag_size": len(dag.sub_questions),
        "per_iter_cs": per_iter_cs,
        "final_cs": cs_result.cs if cs_result else 0.0,
        "em": em_score(final_answer, q.answer),
        "f1": f1_score(final_answer, q.answer),
        "contains": int(q.answer.lower() in final_answer.lower()),
    }


def simulate_theta(per_sample_data: list[dict], theta: float, max_iterations: int = 3) -> dict:
    """For a given θ, compute what would have happened on each sample."""
    n = len(per_sample_data)
    total_iters = 0
    early_stops = 0
    f1_sum = 0.0
    em_sum = 0
    cs_sum = 0.0

    for d in per_sample_data:
        cs_seq = d["per_iter_cs"]
        stopped_at = max_iterations
        for i, cs in enumerate(cs_seq):
            if cs >= theta:
                stopped_at = i + 1
                early_stops += 1
                break
        total_iters += stopped_at
        cs_sum += cs_seq[stopped_at - 1] if stopped_at <= len(cs_seq) else cs_seq[-1]
        f1_sum += d["f1"]
        em_sum += d["em"]

    return {
        "theta": theta,
        "n": n,
        "em": 100 * em_sum / n,
        "f1": 100 * f1_sum / n,
        "avg_retrieval_rounds": total_iters / n,
        "avg_lm_calls": total_iters / n * 6 + 1,  # approximate: probe + claims + branch per iter
        "avg_cs": cs_sum / n,
        "early_stop_rate": 100 * early_stops / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--output_dir", default="experiments/results/theta_grid.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_dir)), exist_ok=True)

    print(f"θ Grid Search")
    print(f"  Dataset: {args.dataset}, Samples: {args.n_samples}")
    print()

    questions = load_dataset(args.dataset, n_samples=args.n_samples)
    print(f"Loaded {len(questions)} questions")

    llm = VLLMClient()
    nli = NLIScorer(model_path="./models/nli-deberta-v3-base", device="cpu", theta=0.75)
    probe = ParametricProbe(llm)

    per_sample_data = []
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q.query[:80]}...")
        d = run_single_with_cs_tracking(q, llm, nli, probe)
        per_sample_data.append(d)
        cs_seq = d["per_iter_cs"]
        print(f"  CS seq: {[f'{x:.3f}' for x in cs_seq]}  Answer: \"{d['answer']}\"  Gold: \"{d['gold_answer']}\"")

    # Save raw data
    with open(args.output_dir, "w", encoding="utf-8") as f:
        json.dump(per_sample_data, f, indent=2, ensure_ascii=False)

    # Simulate θ thresholds
    thetas = [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80]
    print(f"\n{'θ':>8} {'F1':>8} {'EM':>8} {'Avg-Ret':>10} {'Avg-LM':>10} {'Avg-CS':>10} {'EarlyStop%':>12}")
    print("-" * 72)
    results = []
    for theta in thetas:
        r = simulate_theta(per_sample_data, theta)
        results.append(r)
        print(f"{r['theta']:>8.2f} {r['f1']:>8.1f} {r['em']:>8.1f} {r['avg_retrieval_rounds']:>10.2f} "
              f"{r['avg_lm_calls']:>10.1f} {r['avg_cs']:>10.3f} {r['early_stop_rate']:>11.1f}%")

    # Select best θ
    base_f1 = results[0]["f1"]  # θ=0.3 F1
    for r in results:
        if r["f1"] >= base_f1 - 2 and r["early_stop_rate"] > 30:
            print(f"\nSelected θ = {r['theta']:.2f}: F1={r['f1']:.1f}, EarlyStop={r['early_stop_rate']:.1f}%")
            break

    # Save summary
    summary_path = args.output_dir.replace(".json", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
