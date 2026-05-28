"""Task 5: CBET main controller — integrates all modules into the full algorithm."""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field

from src.data_adapter import Question
from src.dag_extractor import extract_dag, QuestionDAG, SubQuestion
from src.epistemic_override import EpistemicOverrider
from src.llm_client import LLMClient
from src.nli_scorer import NLIScorer, CompletenessResult
from src.parametric_probe import ParametricProbe, ConflictResult
from src.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass
class CBETConfig:
    theta: float = 0.50
    tau: float = 0.5
    max_iterations: int = 5
    max_branches: int = 6
    nli_claim_extraction: bool = True
    skip_cross_branch_nli: bool = False  # ablation: no_cross_branch
    gcs_conflict_threshold: float = 0.35  # density-based: branch pair contradictory if conflict_ratio > this


@dataclass
class BranchState:
    evidence: str = ""
    current_answer: str = ""
    conflict: ConflictResult | None = None
    override_prompt: str = ""
    query_rewrite_count: int = 0
    probed: bool = False  # True after first parametric probe
    _last_evidence_for_answer: str = ""  # avoid regenerating when evidence unchanged

    def rewrite_query(self) -> None:
        self.query_rewrite_count += 1


@dataclass
class CBETResult:
    answer: str
    iterations: int
    cs_score: float
    branch_states: dict[str, BranchState]
    dag: QuestionDAG
    log: dict = field(default_factory=dict)


def _exact_match(pred: str, gold: str) -> int:
    return int(pred.strip().lower() == gold.strip().lower())


def _token_f1(pred: str, gold: str) -> float:
    p_tokens = pred.strip().lower().split()
    g_tokens = gold.strip().lower().split()
    if not p_tokens or not g_tokens:
        return 0.0
    common = set(p_tokens) & set(g_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(p_tokens)
    recall = len(common) / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


class CBETController:
    def __init__(
        self,
        llm_client: LLMClient,
        retriever: Retriever,
        nli_scorer: NLIScorer,
        parametric_probe: ParametricProbe,
        config: CBETConfig | None = None,
    ):
        self.llm = llm_client
        self.retriever = retriever
        self.nli_scorer = nli_scorer
        self.parametric_probe = parametric_probe
        self.config = config or CBETConfig()
        self.epistemic_overrider = EpistemicOverrider()
        self._lm_call_count: dict[str, int] = {}

    def _record_lm_call(self, component: str) -> None:
        self._lm_call_count[component] = self._lm_call_count.get(component, 0) + 1

    # ── retrieval helpers ─────────────────────────────────────────────────────

    def _retrieve(self, query: str) -> str:
        try:
            passages = self.retriever.retrieve(query, top_k=5)
            return " ".join(passages)
        except Exception as e:
            logger.warning("retrieve failed for '%s': %s", query[:60], e)
            return ""

    def _parallel_retrieve(
        self, nodes: list[SubQuestion], branch_states: dict[str, BranchState]
    ) -> dict[str, str]:
        # Sequential execution (parallelism can be added via ThreadPoolExecutor later)
        return {sq.id: self._retrieve(sq.text) for sq in nodes}

    def _enrich_with_predecessor_answers(
        self, node: SubQuestion, branch_states: dict[str, BranchState]
    ) -> str:
        query = node.text
        for dep_id in node.depends_on:
            dep_answer = branch_states[dep_id].current_answer
            if dep_answer:
                query = query.replace(f"[answer of {dep_id}]", dep_answer)
        return query

    # ── answer generation ─────────────────────────────────────────────────────

    def _answer_branch(self, sq: SubQuestion, state: BranchState) -> str:
        """Generate answer for a single branch, using override prompt if set."""
        self._record_lm_call("answer_branch")
        if state.override_prompt:
            prompt = state.override_prompt
        else:
            prompt = (
                f"Use the following evidence to answer the question concisely.\n"
                f"Evidence: {state.evidence}\n"
                f"Question: {sq.text}\n"
                f"Answer (1-5 words only):"
            )
        try:
            resp = self.llm.generate(prompt, max_new_tokens=128, temperature=0.0)
            return resp.text.strip()
        except Exception as e:
            logger.warning("branch answer generation failed: %s", e)
            return ""

    def _generate_final_answer(
        self,
        question: Question,
        branch_states: dict[str, BranchState],
        cs_result: CompletenessResult,
        dag: QuestionDAG,
    ) -> str:
        self._record_lm_call("final_answer")
        # Build context from all branch evidence + answers
        branch_context = "\n".join(
            f"[{sq.id}] Q: {sq.text}\nEvidence: {branch_states[sq.id].evidence}\nA: {branch_states[sq.id].current_answer}"
            for sq in dag.sub_questions
            if branch_states[sq.id].current_answer
        )
        prompt = (
            f"Based on the following sub-question answers, answer the original question.\n\n"
            f"{branch_context}\n\n"
            f"Original question: {question.query}\n"
            f"Provide your final answer as a short phrase (1-5 words maximum). "
            f"Do not explain. Just the answer.\nFinal answer:"
        )
        try:
            resp = self.llm.generate(prompt, max_new_tokens=128, temperature=0.0)
            return resp.text.strip()
        except Exception as e:
            logger.warning("final answer generation failed: %s", e)
            # Fallback: return the last branch answer
            for sq in reversed(dag.sub_questions):
                ans = branch_states[sq.id].current_answer
                if ans:
                    return ans
            return ""

    # ── main algorithm ────────────────────────────────────────────────────────

    def solve(self, question: Question) -> CBETResult:
        self._lm_call_count.clear()
        self.nli_scorer.lm_call_count.clear()
        self.parametric_probe.lm_call_count.clear()

        self._record_lm_call("dag_extract")
        dag = extract_dag(question.query, self.llm)

        # Cap branches to max_branches
        if len(dag.sub_questions) > self.config.max_branches:
            dag.sub_questions = dag.sub_questions[: self.config.max_branches]

        branch_states: dict[str, BranchState] = {
            sq.id: BranchState() for sq in dag.sub_questions
        }

        conflicts_detected: list[str] = []
        overrides_triggered: list[str] = []
        noisy_evicted: list[str] = []
        cs_result: CompletenessResult | None = None

        iteration = 0
        while iteration < self.config.max_iterations:
            iteration += 1

            # Step 2: execute DAG in topological order
            for parallel_batch in dag.get_execution_order():

                # Step 2a: leaf nodes — retrieve in parallel
                leaf_nodes = [sq for sq in parallel_batch if sq.is_leaf]
                retrieval_results: dict[str, str] = {}
                if leaf_nodes:
                    retrieval_results.update(
                        self._parallel_retrieve(leaf_nodes, branch_states)
                    )

                # Step 2b: internal nodes — enrich query with predecessor answers
                for node in [sq for sq in parallel_batch if not sq.is_leaf]:
                    enriched = self._enrich_with_predecessor_answers(node, branch_states)
                    retrieval_results[node.id] = self._retrieve(enriched)

                # Step 3-5: update evidence, probe, detect conflict, override
                for sq in parallel_batch:
                    state = branch_states[sq.id]
                    if retrieval_results.get(sq.id):
                        state.evidence = retrieval_results[sq.id]

                    # Step 4: parametric probe + conflict detection (once per branch)
                    if not state.probed:
                        param_mem = self.parametric_probe.probe(sq.text)
                        gcs_for_conflict = (
                            cs_result.gcs if cs_result is not None else 1.0
                        )
                        conflict = self.parametric_probe.detect_conflict(
                            param_mem,
                            state.evidence,
                            self.nli_scorer,
                            self.llm,
                            gcs=gcs_for_conflict,
                        )
                        state.conflict = conflict
                        state.probed = True
                        if conflict.has_conflict and sq.id not in conflicts_detected:
                            conflicts_detected.append(sq.id)

                        # Step 5: epistemic override
                        if conflict.trust_retrieved > self.config.tau:
                            state.override_prompt = self.epistemic_overrider.build(
                                sq.text, state.evidence
                            )
                            if sq.id not in overrides_triggered:
                                overrides_triggered.append(sq.id)

                    # Generate branch answer (skipped if evidence unchanged from last iteration)
                    if state.current_answer and state.evidence == state._last_evidence_for_answer:
                        pass  # reuse previous answer
                    else:
                        state.current_answer = self._answer_branch(sq, state)
                        state._last_evidence_for_answer = state.evidence

            # Step 6: completeness score with DAG dependency edges
            # Build dependency pairs: (src_idx, tgt_idx) for each DAG edge
            sq_id_to_idx = {sq.id: idx for idx, sq in enumerate(dag.sub_questions)}
            dependency_pairs: list[tuple[int, int]] = []
            for tgt_idx, sq in enumerate(dag.sub_questions):
                for dep_id in sq.depends_on:
                    if dep_id in sq_id_to_idx:
                        dependency_pairs.append((sq_id_to_idx[dep_id], tgt_idx))

            cs_result = self.nli_scorer.compute_completeness_score(
                branch_evidences=[branch_states[sq.id].evidence for sq in dag.sub_questions],
                branch_answers=[branch_states[sq.id].current_answer for sq in dag.sub_questions],
                sub_questions=[sq.text for sq in dag.sub_questions],
                llm_client=self.llm,
                skip_gcs=self.config.skip_cross_branch_nli,
                dependency_pairs=dependency_pairs,
            )

            # Step 7: stop if complete
            if cs_result.should_stop:
                break

            # Step 8: evict noisy branches for re-retrieval next iteration
            sq_ids = [sq.id for sq in dag.sub_questions]
            for noisy_idx in cs_result.noisy_branch_ids:
                if noisy_idx < len(sq_ids):
                    nid = sq_ids[noisy_idx]
                    branch_states[nid].evidence = ""
                    branch_states[nid].current_answer = ""
                    branch_states[nid].override_prompt = ""
                    branch_states[nid].rewrite_query()
                    if nid not in noisy_evicted:
                        noisy_evicted.append(nid)

        # Step 9: final answer
        final_answer = self._generate_final_answer(question, branch_states, cs_result, dag)

        em = _exact_match(final_answer, question.answer)
        f1 = _token_f1(final_answer, question.answer)

        # Aggregate LM call counts across all components
        total_lm_calls = sum(self._lm_call_count.values())
        total_lm_calls += sum(self.nli_scorer.lm_call_count.values())
        total_lm_calls += sum(self.parametric_probe.lm_call_count.values())

        lm_breakdown = {
            **self._lm_call_count,
            **self.nli_scorer.lm_call_count,
            **self.parametric_probe.lm_call_count,
        }

        log = {
            "qid": question.qid,
            "iterations": iteration,
            "dag_size": len(dag.sub_questions),
            "branch_cs_scores": cs_result.branch_coverages if cs_result else [],
            "final_cs": cs_result.cs if cs_result else 0.0,
            "gcs": cs_result.gcs if cs_result else 0.0,
            "gcs_method": cs_result.gcs_method if cs_result else "none",
            "edge_scores": cs_result.edge_scores if cs_result else [],
            "conflicts_detected": conflicts_detected,
            "overrides_triggered": overrides_triggered,
            "noisy_evicted": noisy_evicted,
            "answer": final_answer,
            "gold_answer": question.answer,
            "em": em,
            "f1": f1,
            "lm_breakdown": lm_breakdown,
            "total_lm_calls": total_lm_calls,
        }
        logger.info(json.dumps(log, ensure_ascii=False))

        return CBETResult(
            answer=final_answer,
            iterations=iteration,
            cs_score=cs_result.cs if cs_result else 0.0,
            branch_states=branch_states,
            dag=dag,
            log=log,
        )


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_controller(args) -> "CBETController":
    """Instantiate CBETController from CLI args."""
    from src.llm_client import build_client
    from src.nli_scorer import NLIScorer
    from src.parametric_probe import ParametricProbe
    from src.retriever import ElasticRetriever
    import yaml, os

    # Load config yaml if present
    cfg_path = f"configs/cbet_{args.dataset}.yaml"
    yaml_cfg: dict = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            yaml_cfg = yaml.safe_load(f) or {}

    backend = getattr(args, "backend", None) or yaml_cfg.get("backend", "awq")
    nli_path = os.path.join(args.model_path, "nli-deberta-v3-base")

    if backend == "vllm":
        vllm_url = getattr(args, "vllm_url", "http://localhost:8000/v1")
        vllm_model = getattr(args, "vllm_model", "/models/qwen")
        llm = build_client("vllm", "",
                           base_url=vllm_url, model=vllm_model)
    else:
        model_path = os.path.join(args.model_path, "Qwen2.5-7B-Instruct-AWQ")
        llm = build_client(backend, model_path)
    gcs_conflict_threshold = yaml_cfg.get("gcs_conflict_threshold", 0.35)
    nli = NLIScorer(model_path=nli_path, theta=args.theta,
                     gcs_conflict_threshold=gcs_conflict_threshold)
    probe = ParametricProbe(llm)
    es_index = getattr(args, "es_index_name", None) or yaml_cfg.get("es_index_name", "wiki")
    retriever = ElasticRetriever(index_name=es_index)

    config = CBETConfig(
        theta=args.theta,
        tau=args.tau,
        max_iterations=args.max_iterations,
        nli_claim_extraction=(args.ablation != "entropy_only"),
        gcs_conflict_threshold=gcs_conflict_threshold,
    )
    # Ablation: no_cross_branch → skip cross-branch GCS (always 1.0)
    if args.ablation == "no_cross_branch":
        config.skip_cross_branch_nli = True
    # Ablation: no_override → disable epistemic override by setting tau > 1
    if args.ablation == "no_override":
        config.tau = 2.0
    # Ablation: fixed_rounds → disable CS stopping by setting theta > 1
    if args.ablation == "fixed_rounds":
        config.theta = 2.0

    return CBETController(llm, retriever, nli, probe, config)


def main():
    import argparse, json, os
    from src.data_adapter import load_dataset

    parser = argparse.ArgumentParser(description="Run CBET on a dataset")
    parser.add_argument("--dataset",       required=True,
                        choices=["hotpotqa", "musique", "2wikimultihopqa"])
    parser.add_argument("--n_samples",     type=int, default=500)
    parser.add_argument("--model",         default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--model_path",    default="./models/")
    parser.add_argument("--theta",         type=float, default=0.75)
    parser.add_argument("--tau",           type=float, default=0.5)
    parser.add_argument("--max_iterations",type=int,   default=5)
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--log_dir",       default="experiments/results/logs/")
    parser.add_argument("--ablation",      default="full",
                        choices=["full", "no_cross_branch", "no_override",
                                 "entropy_only", "fixed_rounds"])
    parser.add_argument("--es_index_name", default=None,
                        help="ElasticSearch index name (default: wiki)")
    parser.add_argument("--backend",       default=None,
                        choices=["awq", "hf", "vllm"],
                        help="LLM backend (default: awq)")
    parser.add_argument("--vllm_url",      default="http://localhost:8000/v1",
                        help="vLLM server base URL")
    parser.add_argument("--vllm_model",    default="/models/qwen",
                        help="Model name registered in vLLM server")
    parser.add_argument("--mode",          default="batch", choices=["batch", "single"])
    parser.add_argument("--query",         default=None)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_dir)), exist_ok=True)

    ctrl = _build_controller(args)

    if args.mode == "single":
        from src.data_adapter import Question
        q = Question(qid="cli_single", query=args.query, gold_passages=[],
                     distractor_passages=[], answer="", dataset=args.dataset, hop_count=2)
        result = ctrl.solve(q)
        print(json.dumps(result.log, indent=2, ensure_ascii=False))
        return

    questions = load_dataset(args.dataset, n_samples=args.n_samples)
    results = []
    for i, q in enumerate(questions):
        try:
            r = ctrl.solve(q)
            results.append(r.log)
        except Exception as e:
            logging.error("Failed on qid=%s: %s", q.qid, e)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(questions)}] done")

    with open(args.output_dir, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} results to {args.output_dir}")


if __name__ == "__main__":
    main()
