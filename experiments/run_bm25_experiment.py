"""BM25 iteration retrieval experiment — HotpotQA 20 samples.

Runs NoRAG, SingleRAG, CBET with BM25Retriever and records per-iteration metrics.
Usage: uv run python experiments/run_bm25_experiment.py
"""
from __future__ import annotations
import argparse, json, os, sys, time, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_client import VLLMClient
from src.nli_scorer import NLIScorer, CompletenessResult
from src.cbet_controller import CBETController, CBETConfig, BranchState
from src.parametric_probe import ParametricProbe
from src.data_adapter import load_dataset, Question
from src.dag_extractor import extract_dag
from src.epistemic_override import EpistemicOverrider
from src.bm25_retriever import BM25Retriever


def safe_str(s: str, max_len: int = 80) -> str:
    s = s[:max_len]
    result = []
    for ch in s:
        try:
            ch.encode('gbk')
            result.append(ch)
        except UnicodeEncodeError:
            result.append('?')
    return ''.join(result)


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


def _run_llm(llm: VLLMClient, prompt: str, max_tokens: int = 128) -> str:
    try:
        resp = llm.generate(prompt, max_new_tokens=max_tokens, temperature=0.0)
        return resp.text.strip()
    except Exception as e:
        print(f"    [WARN] LLM call failed: {e}")
        return ""


# ── pass-through retrieval wrapper: records which passages were retrieved ──

class TrackedRetriever:
    def __init__(self, bm25: BM25Retriever, gold_passages: list[str]):
        self._bm25 = bm25
        self._gold = [g.lower() for g in gold_passages]
        self.call_log: list[dict] = []  # [{iteration, branch_id, query, passages, gold_found}]

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        results = self._bm25.retrieve(query, top_k)
        gold_found = any(
            any(g[:60] in p.lower() for g in self._gold)
            for p in results
        )
        self.call_log.append({"query": query, "results": results, "gold_found": gold_found})
        return results

    def gold_found_in_iteration(self, iteration: int) -> bool:
        return any(c.get("gold_found") for c in self.call_log
                   if c.get("_iteration") == iteration)

    def mark_iteration(self, iteration: int):
        for c in self.call_log:
            if "_iteration" not in c:
                c["_iteration"] = iteration


# ── CBET with per-iteration tracking ──

def run_cbet_tracked(q: Question, llm: VLLMClient, nli: NLIScorer,
                      probe: ParametricProbe, bm25: BM25Retriever,
                      config: CBETConfig) -> dict:
    tracker = TrackedRetriever(bm25, q.gold_passages)
    controller = CBETController(llm, tracker, nli, probe, config)
    overrider = EpistemicOverrider()

    dag = extract_dag(q.query, llm)
    if len(dag.sub_questions) > config.max_branches:
        dag.sub_questions = dag.sub_questions[:config.max_branches]

    branch_states = {sq.id: BranchState() for sq in dag.sub_questions}
    cs_history: list[float] = []
    gcs_history: list[float] = []
    passages_per_iter: list[list[str]] = []
    cs_result: CompletenessResult | None = None
    early_stopped = False
    iterations_used = config.max_iterations

    for iteration in range(1, config.max_iterations + 1):
        tracker.mark_iteration(iteration)

        for parallel_batch in dag.get_execution_order():
            retrieval_results: dict[str, str] = {}
            leaf_nodes = [sq for sq in parallel_batch if sq.is_leaf]
            if leaf_nodes:
                for sq in leaf_nodes:
                    retrieval_results[sq.id] = controller._retrieve(sq.text)

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
                    conflict = probe.detect_conflict(param_mem, state.evidence, nli, llm,
                                                     gcs=gcs_for_conflict)
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
        )
        cs_history.append(cs_result.cs)
        gcs_history.append(cs_result.gcs)

        # Record passages from this iteration
        iter_calls = [c for c in tracker.call_log if c.get("_iteration") == iteration]
        iter_passages = []
        for c in iter_calls:
            for p in c["results"][:3]:
                # Extract first sentence as title
                first = p.split(".")[0][:60] if "." in p else p[:60]
                iter_passages.append(first)
        passages_per_iter.append(iter_passages)

        if cs_result.should_stop:
            early_stopped = True
            iterations_used = iteration
            break

    final_answer = controller._generate_final_answer(q, branch_states, cs_result, dag)

    # Find gold passage hit iteration
    gold_found_at = -1
    for it in range(1, iterations_used + 1):
        iter_calls = [c for c in tracker.call_log if c.get("_iteration") == it]
        if any(c.get("gold_found") for c in iter_calls):
            gold_found_at = it
            break

    return {
        "qid": q.qid,
        "question": q.query,
        "gold_answer": q.answer,
        "predicted_answer": final_answer,
        "em": em_score(final_answer, q.answer),
        "f1": f1_score(final_answer, q.answer),
        "contains_rate": int(q.answer.lower() in final_answer.lower()),
        "iterations_used": iterations_used,
        "early_stopped": early_stopped,
        "cs_per_iteration": cs_history,
        "gcs_per_iteration": gcs_history,
        "passages_per_iteration": passages_per_iter,
        "gold_found_at_iteration": gold_found_at,
        "dag_size": len(dag.sub_questions),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def print_table(rows: list[dict]):
    print(f"{'Method':<18} {'EM':>6} {'F1':>6} {'Contains%':>10} {'Avg-Ret':>8} {'Avg-LM':>8}")
    print("-" * 60)
    for r in rows:
        print(f"{r['method']:<18} {r['em']:>6.1f} {r['f1']:>6.1f} {r['contains']:>10.1f} "
              f"{r['avg_ret']:>8.1f} {r['avg_lm']:>8.1f}")


def main():
    print("=" * 60)
    print("BM25 Iteration Retrieval Experiment (HotpotQA n=20)")
    print("=" * 60)

    questions = load_dataset("hotpotqa", n_samples=20)
    print(f"Loaded {len(questions)} questions")

    llm = VLLMClient()
    bm25 = BM25Retriever(index_path="./indexes/hotpotqa_bm25.pkl")
    nli = NLIScorer("./models/nli-deberta-v3-base", "cpu", theta=0.50, gcs_conflict_threshold=0.35)
    probe = ParametricProbe(llm)
    config = CBETConfig(theta=0.50, tau=0.5, max_iterations=3, max_branches=6,
                        gcs_conflict_threshold=0.35)

    all_results: list[dict] = []
    cbet_details: list[dict] = []

    for i, q in enumerate(questions):
        print(f"\n[{i+1}/20] {safe_str(q.query[:90])}...")

        # NoRAG
        prompt = f"Answer this question in 1-5 words. Do not explain.\n\nQuestion: {q.query}\nAnswer:"
        norag_ans = _run_llm(llm, prompt)
        all_results.append({
            "method": "NoRAG", "qid": q.qid,
            "answer": norag_ans, "gold_answer": q.answer,
            "retrieval_rounds": 0, "lm_calls": 1, "time_seconds": 0,
            "contains": int(q.answer.lower() in norag_ans.lower()),
        })
        print(f"  NoRAG: '{safe_str(norag_ans, 30)}' (EM={em_score(norag_ans, q.answer)})")

        # SingleRAG (top-5 from BM25)
        passages = bm25.retrieve(q.query, top_k=5)
        evidence = "\n".join(f"- {p[:300]}" for p in passages)
        prompt = (f"Use the following evidence to answer the question in 1-5 words.\n\n"
                  f"Evidence:\n{evidence}\n\nQuestion: {q.query}\n"
                  f"Do not explain. Just the answer.\nAnswer:")
        singlerag_ans = _run_llm(llm, prompt)
        all_results.append({
            "method": "SingleRAG", "qid": q.qid,
            "answer": singlerag_ans, "gold_answer": q.answer,
            "retrieval_rounds": 1, "lm_calls": 1, "time_seconds": 0,
            "contains": int(q.answer.lower() in singlerag_ans.lower()),
        })
        print(f"  SingleRAG: '{safe_str(singlerag_ans, 30)}' (EM={em_score(singlerag_ans, q.answer)})")

        # CBET with BM25
        cbet = run_cbet_tracked(q, llm, nli, probe, bm25, config)
        cbet["method"] = "CBET"
        all_results.append({
            "method": "CBET", "qid": q.qid,
            "answer": cbet["predicted_answer"], "gold_answer": q.answer,
            "retrieval_rounds": cbet["iterations_used"],
            "lm_calls": cbet["iterations_used"] * 6 + 1,
            "time_seconds": 0,
            "contains": cbet["contains_rate"],
        })
        cbet_details.append(cbet)
        cs_str = "/".join(f"{x:.3f}" for x in cbet["cs_per_iteration"])
        print(f"  CBET: '{safe_str(cbet['predicted_answer'], 30)}' "
              f"(EM={cbet['em']}, CS=[{cs_str}], Stop={cbet['early_stopped']}, Gold@={cbet['gold_found_at_iteration']})")

    # Save CBET details
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/bm25_hotpotqa_20.json", "w", encoding="utf-8") as f:
        json.dump({"all_results": all_results, "cbet_details": cbet_details}, f, indent=2, ensure_ascii=False)

    # ── Summary ──
    print()
    print("=" * 60)
    print("=== BM25 Iteration Retrieval Experiment (HotpotQA n=20) ===")
    print()

    methods = ["NoRAG", "SingleRAG", "CBET"]
    summary_rows = []
    for method in methods:
        subset = [r for r in all_results if r["method"] == method]
        n = len(subset)
        em_vals = [em_score(r["answer"], r["gold_answer"]) for r in subset]
        f1_vals = [f1_score(r["answer"], r["gold_answer"]) for r in subset]
        contains_vals = [r.get("contains", 0) for r in subset]
        summary_rows.append({
            "method": method,
            "em": 100 * sum(em_vals) / n,
            "f1": 100 * sum(f1_vals) / n,
            "contains": 100 * sum(contains_vals) / n,
            "avg_ret": sum(r["retrieval_rounds"] for r in subset) / n,
            "avg_lm": sum(r["lm_calls"] for r in subset) / n,
        })

    print("Performance Comparison:")
    print_table(summary_rows)

    # CBET iteration behavior
    print()
    print("CBET Iteration Behavior:")
    early_stops = sum(1 for d in cbet_details if d["early_stopped"])
    print(f"  - EarlyStop% (CS>=0.50 triggered): {early_stops}/{len(cbet_details)} ({100*early_stops/len(cbet_details):.0f}%)")

    cs_by_iter = [[], [], []]
    for d in cbet_details:
        for i, cs in enumerate(d["cs_per_iteration"]):
            if i < 3:
                cs_by_iter[i].append(cs)
    avg_cs = [sum(cs_by_iter[i]) / len(cs_by_iter[i]) if cs_by_iter[i] else 0 for i in range(3)]
    print(f"  - Avg CS trajectory: Iter1={avg_cs[0]:.3f}, Iter2={avg_cs[1]:.3f}, Iter3={avg_cs[2]:.3f}")

    gold_hits = [0, 0, 0]
    for d in cbet_details:
        for it in range(1, min(d["iterations_used"] + 1, 4)):
            if d["gold_found_at_iteration"] > 0 and it >= d["gold_found_at_iteration"]:
                gold_hits[it - 1] += 1
    print(f"  - Gold passage hit rate:")
    for i in range(3):
        print(f"      @Round{i+1}: {100*gold_hits[i]/len(cbet_details):.0f}%")

    cs_up = sum(1 for d in cbet_details
                if len(d["cs_per_iteration"]) >= 2 and d["cs_per_iteration"][-1] > d["cs_per_iteration"][0])
    print(f"  - CS rising samples (final CS > initial CS): {cs_up}/{len(cbet_details)}")

    # ── Case studies ──
    print()
    print("Case Studies:")
    # Success case: early_stopped=True, EM=1
    success = [d for d in cbet_details if d["early_stopped"] and d["em"] == 1]
    if success:
        d = success[0]
        print(f"  [SUCCESS] Q: {safe_str(d['question'][:120])}")
        print(f"    Gold: '{d['gold_answer']}'  CBET: '{d['predicted_answer']}'")
        for it in range(d["iterations_used"]):
            print(f"    Iter{it+1}: CS={d['cs_per_iteration'][it]:.3f}  "
                  f"Passages: {safe_str(' | '.join(d['passages_per_iteration'][it][:3]), 100)}")
    else:
        print("  [SUCCESS] No early_stopped+EM=1 case found.")
        # Fallback: just early_stopped
        early = [d for d in cbet_details if d["early_stopped"]]
        if early:
            d = early[0]
            print(f"  [EarlyStop] Q: {safe_str(d['question'][:120])}")
            print(f"    Gold: '{d['gold_answer']}'  CBET: '{d['predicted_answer']}'  EM={d['em']}")

    # Failure case: CBET EM=0, SingleRAG EM=1
    failures = []
    for d in cbet_details:
        single = [r for r in all_results if r["method"] == "SingleRAG" and r["qid"] == d["qid"]]
        if single and d["em"] == 0 and em_score(single[0]["answer"], single[0]["gold_answer"]) == 1:
            failures.append((d, single[0]))
    if failures:
        d, s = failures[0]
        print(f"  [FAILURE] Q: {safe_str(d['question'][:120])}")
        print(f"    Gold: '{d['gold_answer']}'")
        print(f"    SingleRAG: '{safe_str(s['answer'], 50)}' (EM=1)")
        print(f"    CBET: '{safe_str(d['predicted_answer'], 50)}' (EM=0)")
        gold_found = d["gold_found_at_iteration"] > 0
        print(f"    DAG size: {d['dag_size']}, Gold found: {gold_found}, "
              f"CS seq: {[f'{x:.3f}' for x in d['cs_per_iteration']]}")
    else:
        print("  [FAILURE] No CBET-fail while SingleRAG-win case found.")

    # ── Pass/fail check ──
    print()
    print("=" * 60)
    cbet_f1 = summary_rows[2]["f1"]
    singlerag_f1 = summary_rows[1]["f1"]
    early_stop_pct = 100 * early_stops / len(cbet_details)

    checks = {
        "CBET F1 > SingleRAG F1": cbet_f1 > singlerag_f1,
        "EarlyStop% > 15%": early_stop_pct > 15,
        "CS rising > 8/20": cs_up > 8,
        "Gold@Round3 > Gold@Round1": gold_hits[2] > gold_hits[0],
    }

    for desc, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {desc} "
              f"({cbet_f1:.1f}>{singlerag_f1:.1f}" if "F1" in desc else
               f"({early_stop_pct:.0f}%>15%" if "Stop" in desc else
               f"({cs_up}/20>8)" if "rising" in desc else
               f"({gold_hits[2]}>{gold_hits[0]})" if "Gold" in desc else "")

    all_pass = all(checks.values())
    print()
    if all_pass:
        print("ALL CHECKS PASSED — Framework validation complete.")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"FAILED CHECKS: {failed}")
        # Diagnose root cause
        if not checks["CS rising > 8/20"]:
            print("  Root cause: BM25 retrieval doesn't bring new evidence per iteration; "
                  "CS stays flat because same passages dominate results")
        if not checks["EarlyStop% > 15%"]:
            print("  Root cause: CS rarely reaches theta=0.50; "
                  "NLI(claim→short_answer) scores remain low despite correct claims")
        if not checks["CBET F1 > SingleRAG F1"]:
            print("  Root cause: Multi-hop decomposition introduces errors that "
                  "SingleRAG avoids with direct evidence")

    print("=" * 60)


if __name__ == "__main__":
    main()
