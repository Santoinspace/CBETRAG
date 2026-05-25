"""Task 8: CBET integration test with vLLM Qwen model.

Tests the full CBET pipeline using the vLLM-hosted Qwen model
and mock retriever/NLI components.

Usage:
    python tests/test_cbet_integration_vllm.py
"""
from __future__ import annotations
import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_client import VLLMClient, LLMResponse
from src.dag_extractor import extract_dag
from src.parametric_probe import ParametricProbe, ParametricMemory
from src.epistemic_override import EpistemicOverrider
from src.retriever import PassageListRetriever
from src.nli_scorer import NLIScorer
from src.cbet_controller import CBETController, CBETConfig
from src.data_adapter import Question

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ── mock NLI scorer (no GPU model needed) ────────────────────────────────────

class MockNLIScorer:
    """Lightweight NLI scorer for testing — returns synthetic scores.

    Uses keyword-based heuristics without requiring LLM calls for
    claim extraction. Returns generous entailment for evidence
    that has any keyword overlap with the hypothesis, enabling
    the CBET pipeline to complete in testing environments.
    """

    def __init__(self, theta: float = 0.75):
        self.theta = theta
        self.device = "cpu"

    def score_pair(self, premise: str, hypothesis: str):
        from src.nli_scorer import NLIResult
        p_words = set(premise.lower().split())
        h_words = set(hypothesis.lower().split())
        if not h_words:
            return NLIResult(label="neutral", entailment_score=0.5,
                             neutral_score=0.5, contradiction_score=0.0)
        overlap = len(p_words & h_words) / len(h_words)
        if overlap > 0.3:
            return NLIResult(label="entailment", entailment_score=0.6 + 0.3 * min(overlap, 1.0),
                             neutral_score=0.1, contradiction_score=0.0)
        elif overlap > 0.05:
            return NLIResult(label="neutral", entailment_score=0.3,
                             neutral_score=0.6, contradiction_score=0.1)
        else:
            return NLIResult(label="neutral", entailment_score=0.15,
                             neutral_score=0.8, contradiction_score=0.05)

    def _sentence_split(self, text: str) -> list[str]:
        """Simple sentence split without LLM call."""
        return [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 10]

    def extract_atomic_claims(self, text: str, llm_client=None) -> list[str]:
        """Sentence-split evidence (no LLM call needed for mock)."""
        return self._sentence_split(text)

    def compute_coverage(self, evidence: str, sub_answer: str, llm_client=None) -> float:
        """Compute coverage based on keyword overlap between evidence and answer."""
        if not evidence or not sub_answer:
            return 0.0
        try:
            claims = self._sentence_split(evidence)
            if not claims:
                return 0.5  # default for non-empty evidence
            scores = [self.score_pair(claim, sub_answer).entailment_score for claim in claims]
            return max(scores) if scores else 0.5
        except Exception:
            return 0.5

    def compute_gcs(self, evidences: list[str], llm_client=None) -> float:
        """Always return 1.0 for mock (skip cross-branch NLI)."""
        return 1.0

    def compute_completeness_score(self, branch_evidences, branch_answers,
                                   sub_questions, llm_client, skip_gcs=False):
        from src.nli_scorer import CompletenessResult
        n = len(branch_evidences)
        coverages = [
            self.compute_coverage(branch_evidences[i], branch_answers[i])
            if branch_answers[i] else 0.0
            for i in range(n)
        ]
        # Boost coverage floor so CS can reach theta
        gcs = 1.0
        min_cov = min(coverages) if coverages else 0.0
        cs = min_cov * gcs
        return CompletenessResult(
            branch_coverages=coverages, min_coverage=min_cov,
            gcs=gcs, cs=cs, should_stop=cs >= self.theta, noisy_branch_ids=[])


# ── test questions ───────────────────────────────────────────────────────────

SAMPLE_QUESTIONS = [
    Question(
        qid="test_1", query="Who directed the movie Inception and when was this director born?",
        gold_passages=[], distractor_passages=[], answer="Christopher Nolan, 1970",
        dataset="hotpotqa", hop_count=2,
    ),
    Question(
        qid="test_2",
        query="What is the capital of France and what is its population?",
        gold_passages=[], distractor_passages=[],
        answer="Paris, 2.1 million", dataset="hotpotqa", hop_count=2,
    ),
]

# Known facts for mock retrieval
MOCK_PASSAGES = [
    # Test 1: Inception director + birth
    "Inception is a 2010 science fiction action film written and directed by Christopher Nolan.",
    "Christopher Nolan was born on July 30, 1970 in Westminster, London, England.",
    "Inception was produced by Emma Thomas and Christopher Nolan.",
    # Test 2: France capital + population
    "Paris is the capital and most populous city of France.",
    "The official population of Paris is approximately 2.1 million as of 2023.",
    "France is a country located primarily in Western Europe.",
]


def test_vllm_client():
    """Verify VLLMClient can connect and generate text + logprobs."""
    print("--- Test: VLLMClient ---")
    client = VLLMClient()
    resp = client.generate("What is 2+2? Answer with just the number.", max_new_tokens=8)
    assert resp.text, "VLLMClient should return non-empty text"
    print(f"  Generated: '{resp.text}'")
    # logprobs may be empty if model doesn't support, but text must be non-empty
    print("  OK")


def test_dag_extraction():
    """Verify DAG extraction with vLLM model."""
    print("\n--- Test: DAG Extraction ---")
    client = VLLMClient()
    dag = extract_dag(SAMPLE_QUESTIONS[0].query, client)
    assert len(dag.sub_questions) >= 1, "DAG should have at least 1 sub-question"
    print(f"  Query: {SAMPLE_QUESTIONS[0].query}")
    print(f"  Sub-questions: {len(dag.sub_questions)}")
    for sq in dag.sub_questions:
        print(f"    [{sq.id}] leaf={sq.is_leaf} deps={sq.depends_on}: {sq.text[:80]}...")

    # Verify topological order
    order = dag.get_execution_order()
    assert len(order) >= 1, "Should have at least 1 execution batch"
    total_in_order = sum(len(batch) for batch in order)
    assert total_in_order == len(dag.sub_questions), "All sub-questions should be in execution order"
    print(f"  Execution batches: {len(order)}")
    for i, batch in enumerate(order):
        print(f"    Batch {i}: {[sq.id for sq in batch]}")
    print("  OK")


def test_parametric_probe():
    """Verify parametric probe returns answers and logprobs."""
    print("\n--- Test: Parametric Probe ---")
    client = VLLMClient()
    probe = ParametricProbe(client)

    memory = probe.probe("What is the capital of France?")
    assert memory.answer, "Probe should return a non-empty answer"
    print(f"  Question: What is the capital of France?")
    print(f"  Answer: '{memory.answer}'")
    print(f"  Confidence (entropy): {memory.confidence:.4f}")
    print(f"  Logprobs: {len(memory.raw_logprobs)} tokens")

    # Test conflict detection with a mock NLI
    mock_nli = MockNLIScorer()
    conflict = probe.detect_conflict(
        memory,
        "Paris is the capital and most populous city of France.",
        mock_nli,
        client,
        gcs=1.0,
    )
    print(f"  Conflict: has_conflict={conflict.has_conflict}, type={conflict.conflict_type}")
    print(f"  Parametric answer: '{conflict.parametric_answer}'")
    print(f"  Retrieved answer: '{conflict.retrieved_answer}'")
    print(f"  Trust retrieved: {conflict.trust_retrieved}")
    print("  OK")


def test_epistemic_override():
    """Verify epistemic override prompt builder."""
    print("\n--- Test: Epistemic Override ---")
    overrider = EpistemicOverrider()
    prompt = overrider.build("Who directed Inception?", "Inception was directed by Christopher Nolan.")
    assert "Inception was directed by Christopher Nolan" in prompt
    assert "VERIFIED EVIDENCE" in prompt
    assert "Who directed Inception?" in prompt
    print(f"  Prompt length: {len(prompt)} chars")
    print("  OK")


def test_cbet_controller_mock():
    """Full CBET pipeline with vLLM + mock retriever + mock NLI."""
    print("\n--- Test: CBET Controller (mock) ---")
    client = VLLMClient()
    retriever = PassageListRetriever(MOCK_PASSAGES)
    mock_nli = MockNLIScorer(theta=0.75)
    probe = ParametricProbe(client)
    config = CBETConfig(theta=0.75, tau=0.5, max_iterations=3, max_branches=6)
    controller = CBETController(client, retriever, mock_nli, probe, config)

    for q in SAMPLE_QUESTIONS:
        print(f"\n  Question: {q.query}")
        result = controller.solve(q)
        print(f"  Answer: '{result.answer}'")
        print(f"  Iterations: {result.iterations}")
        print(f"  CS score: {result.cs_score:.3f}")
        print(f"  DAG size: {len(result.dag.sub_questions)}")
        assert result.answer, "Should produce a non-empty answer"
        assert result.iterations >= 1
        assert result.iterations <= config.max_iterations

        # Verify log structure
        log = result.log
        assert "qid" in log
        assert "em" in log
        assert "f1" in log
        assert "iterations" in log
        assert "dag_size" in log
        assert "final_cs" in log
        assert "conflicts_detected" in log
        assert "overrides_triggered" in log
        assert "noisy_evicted" in log
        print(f"  Log keys: {sorted(log.keys())}")
        print(f"  EM={log['em']}, F1={log['f1']:.3f}")
    print("  OK")


def test_evaluate_aggregation_with_live_results():
    """Feed live CBET results into evaluate_all.aggregate()."""
    print("\n--- Test: Evaluate Aggregation with Live Results ---")
    from analysis.evaluate_all import aggregate

    client = VLLMClient()
    retriever = PassageListRetriever(MOCK_PASSAGES)
    mock_nli = MockNLIScorer(theta=0.75)
    probe = ParametricProbe(client)
    config = CBETConfig(max_iterations=2)
    controller = CBETController(client, retriever, mock_nli, probe, config)

    logs = []
    for q in SAMPLE_QUESTIONS:
        result = controller.solve(q)
        logs.append(result.log)

    metrics = aggregate(logs)
    required = ["n", "em", "f1", "avg_retrieval_rounds", "avg_lm_calls",
                "avg_cs_at_stop", "conflict_detected_rate",
                "override_triggered_rate", "noisy_branch_evicted_rate"]
    for key in required:
        assert key in metrics, f"Missing required metric: {key}"

    print(f"  n={metrics['n']}")
    print(f"  EM={metrics['em']:.1f}%  F1={metrics['f1']:.1f}%")
    print(f"  Avg retrieval rounds: {metrics['avg_retrieval_rounds']:.1f}")
    print(f"  Avg LM calls: {metrics['avg_lm_calls']:.1f}")
    print(f"  Avg CS at stop: {metrics['avg_cs_at_stop']:.3f}")
    print(f"  Conflict rate: {metrics['conflict_detected_rate']:.1f}%")
    print(f"  Override rate: {metrics['override_triggered_rate']:.1f}%")
    print(f"  Noisy evicted rate: {metrics['noisy_branch_evicted_rate']:.1f}%")
    print("  OK")


def main():
    print("=" * 60)
    print("CBET Integration Tests (vLLM + Mock NLI/Retriever)")
    print("=" * 60)

    tests = [
        ("VLLMClient", test_vllm_client),
        ("DAG Extraction", test_dag_extraction),
        ("Parametric Probe", test_parametric_probe),
        ("Epistemic Override", test_epistemic_override),
        ("CBET Controller (mock)", test_cbet_controller_mock),
        ("Evaluate Aggregation", test_evaluate_aggregation_with_live_results),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{passed+failed} passed, {failed} failed")
    if failed == 0:
        print("[INTEGRATION OK] All CBET integration tests passed")
    else:
        print("[INTEGRATION FAIL] Some tests failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
