"""Tests for Task 5: cbet_controller.py — fully mocked, no GPU required."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock
from src.llm_client import LLMClient, LLMResponse
from src.nli_scorer import NLIResult, CompletenessResult
from src.parametric_probe import ParametricMemory, ConflictResult
from src.retriever import PassageListRetriever
from src.data_adapter import Question
from src.cbet_controller import CBETController, CBETConfig, BranchState
from src.epistemic_override import EpistemicOverrider


# ── helpers ───────────────────────────────────────────────────────────────────

def _question(answer: str = "Paris") -> Question:
    return Question(
        qid="test_001",
        query="Where was the director of Titanic born?",
        gold_passages=["James Cameron was born in Kapuskasing, Ontario."],
        distractor_passages=[],
        answer=answer,
        dataset="hotpotqa",
        hop_count=2,
    )


def _make_llm(dag_json: str, answers: list[str]) -> LLMClient:
    """LLM that returns dag_json on first call, then cycles through answers."""
    call_count = [0]
    responses = [dag_json] + answers

    class _LLM(LLMClient):
        def __init__(self):
            import tempfile
            super().__init__(cache_dir=tempfile.mkdtemp(prefix="test_llm_cache_"))
        def _generate(self, prompt, max_new_tokens=512, temperature=0.0, **kw):
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            return LLMResponse(text=responses[idx], logprobs=[-0.1, -0.1])

    return _LLM()


def _make_nli_scorer(should_stop: bool = True) -> MagicMock:
    scorer = MagicMock()
    scorer.compute_completeness_score.return_value = CompletenessResult(
        branch_coverages=[0.85, 0.90],
        min_coverage=0.85,
        gcs=0.95,
        cs=0.85 * 0.95,
        should_stop=should_stop,
        noisy_branch_ids=[],
    )
    scorer.score_pair.return_value = NLIResult("entailment", 0.9, 0.05, 0.05)
    return scorer


def _make_probe(has_conflict: bool = False) -> MagicMock:
    probe = MagicMock()
    probe.probe.return_value = ParametricMemory(answer="some answer", confidence=0.1)
    probe.detect_conflict.return_value = ConflictResult(
        has_conflict=has_conflict,
        conflict_type="parametric_vs_retrieved" if has_conflict else "no_conflict",
        parametric_answer="some answer",
        retrieved_answer="retrieved answer",
        trust_retrieved=0.9 if has_conflict else 0.0,
    )
    return probe


_DAG_JSON = json.dumps({
    "sub_questions": [
        {"id": "q1", "text": "Who directed Titanic?", "depends_on": []},
        {"id": "q2", "text": "Where was [answer of q1] born?", "depends_on": ["q1"]},
    ]
})

_RETRIEVER = PassageListRetriever([
    "James Cameron directed Titanic.",
    "James Cameron was born in Kapuskasing, Ontario, Canada.",
])


# ── tests ─────────────────────────────────────────────────────────────────────

def test_solve_returns_cbet_result():
    llm = _make_llm(_DAG_JSON, ["James Cameron", "Kapuskasing", "Kapuskasing"])
    ctrl = CBETController(llm, _RETRIEVER, _make_nli_scorer(), _make_probe(), CBETConfig())
    result = ctrl.solve(_question())
    assert result.answer
    assert result.iterations >= 1
    assert result.dag is not None


def test_solve_stops_when_cs_sufficient():
    llm = _make_llm(_DAG_JSON, ["James Cameron"] * 10)
    ctrl = CBETController(llm, _RETRIEVER, _make_nli_scorer(should_stop=True), _make_probe(),
                          CBETConfig(min_iterations=1))
    result = ctrl.solve(_question())
    assert result.iterations == 1  # stops after first iteration (min_iterations=1)


def test_solve_runs_max_iterations_when_never_stops():
    llm = _make_llm(_DAG_JSON, ["answer"] * 20)
    ctrl = CBETController(
        llm, _RETRIEVER,
        _make_nli_scorer(should_stop=False),
        _make_probe(),
        CBETConfig(max_iterations=3),
    )
    result = ctrl.solve(_question())
    assert result.iterations == 3


def test_log_contains_required_fields():
    llm = _make_llm(_DAG_JSON, ["James Cameron", "Kapuskasing"] * 5)
    ctrl = CBETController(llm, _RETRIEVER, _make_nli_scorer(), _make_probe())
    result = ctrl.solve(_question())
    for key in ("qid", "iterations", "dag_size", "final_cs",
                 "conflicts_detected", "overrides_triggered", "noisy_evicted",
                 "answer", "gold_answer", "em", "f1"):
        assert key in result.log, f"Missing log key: {key}"


def test_em_f1_computed():
    llm = _make_llm(_DAG_JSON, ["Paris"] * 10)
    ctrl = CBETController(llm, _RETRIEVER, _make_nli_scorer(), _make_probe())
    result = ctrl.solve(_question(answer="Paris"))
    assert result.log["em"] == 1
    assert result.log["f1"] == pytest.approx(1.0)


def test_conflict_triggers_override():
    llm = _make_llm(_DAG_JSON, ["wrong answer"] * 10)
    probe = _make_probe(has_conflict=True)  # trust_retrieved=0.9 > tau=0.5
    ctrl = CBETController(llm, _RETRIEVER, _make_nli_scorer(), probe, CBETConfig(tau=0.5))
    result = ctrl.solve(_question())
    # At least one branch should have triggered override
    assert len(result.log["overrides_triggered"]) >= 1


def test_noisy_branch_evicted_and_requeried():
    llm = _make_llm(_DAG_JSON, ["answer"] * 20)
    scorer = MagicMock()
    # First iteration: noisy branch 0; second: stop
    scorer.compute_completeness_score.side_effect = [
        CompletenessResult([0.4, 0.9], 0.4, 0.5, 0.2, False, [0]),
        CompletenessResult([0.85, 0.9], 0.85, 0.95, 0.8, True, []),
    ]
    scorer.score_pair.return_value = NLIResult("entailment", 0.9, 0.05, 0.05)
    ctrl = CBETController(llm, _RETRIEVER, scorer, _make_probe(), CBETConfig(max_iterations=5))
    result = ctrl.solve(_question())
    assert result.iterations == 2
    assert len(result.log["noisy_evicted"]) >= 1


def test_max_branches_cap():
    many_sqs = [{"id": f"q{i}", "text": f"sub q {i}", "depends_on": []} for i in range(1, 10)]
    dag_json = json.dumps({"sub_questions": many_sqs})
    llm = _make_llm(dag_json, ["answer"] * 30)
    ctrl = CBETController(llm, _RETRIEVER, _make_nli_scorer(), _make_probe(),
                          CBETConfig(max_branches=6))
    result = ctrl.solve(_question())
    assert len(result.dag.sub_questions) <= 6


def test_epistemic_overrider_build():
    eo = EpistemicOverrider()
    prompt = eo.build("Who directed Titanic?", "James Cameron directed Titanic.")
    assert "VERIFIED EVIDENCE" in prompt
    assert "James Cameron directed Titanic." in prompt
    assert "Who directed Titanic?" in prompt
