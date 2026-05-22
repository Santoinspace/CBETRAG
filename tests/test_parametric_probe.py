"""Tests for Task 4: parametric_probe.py — no GPU required."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock
from src.llm_client import LLMClient, LLMResponse
from src.nli_scorer import NLIResult
from src.parametric_probe import (
    ParametricProbe, ParametricMemory, ConflictResult,
    _mean_token_entropy, _CERTAINTY_THRESHOLD,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _llm(text: str, logprobs: list[float] = None) -> LLMClient:
    m = MagicMock(spec=LLMClient)
    m.generate.return_value = LLMResponse(text=text, logprobs=logprobs or [])
    return m


def _nli(label: str) -> MagicMock:
    scorer = MagicMock()
    scorer.score_pair.return_value = NLIResult(
        label=label,
        entailment_score=0.9 if label == "entailment" else 0.05,
        neutral_score=0.05,
        contradiction_score=0.9 if label == "contradiction" else 0.05,
    )
    return scorer


# ── _mean_token_entropy ───────────────────────────────────────────────────────

def test_entropy_empty_logprobs():
    assert _mean_token_entropy([]) == 1.0


def test_entropy_certain(  ):
    # logprob close to 0 → low entropy (model very confident)
    lp = [-0.01, -0.02, -0.01]
    h = _mean_token_entropy(lp)
    assert h < _CERTAINTY_THRESHOLD


def test_entropy_uncertain():
    # logprob very negative → high entropy
    lp = [-3.0, -4.0, -3.5]
    h = _mean_token_entropy(lp)
    assert h > _CERTAINTY_THRESHOLD


def test_entropy_formula():
    lp = [-1.0, -2.0]
    expected = (1.0 + 2.0) / 2
    assert _mean_token_entropy(lp) == pytest.approx(expected)


# ── ParametricProbe.probe ─────────────────────────────────────────────────────

def test_probe_returns_answer():
    probe = ParametricProbe(_llm("Paris", [-0.1, -0.05]))
    mem = probe.probe("What is the capital of France?")
    assert mem.answer == "Paris"
    assert mem.raw_logprobs == [-0.1, -0.05]


def test_probe_confidence_low_when_certain():
    probe = ParametricProbe(_llm("Paris", [-0.01, -0.02]))
    mem = probe.probe("What is the capital of France?")
    assert mem.confidence < _CERTAINTY_THRESHOLD


def test_probe_confidence_high_when_uncertain():
    probe = ParametricProbe(_llm("Maybe Paris?", [-4.0, -3.5, -4.2]))
    mem = probe.probe("What is the capital of France?")
    assert mem.confidence > _CERTAINTY_THRESHOLD


def test_probe_fallback_on_error():
    llm = MagicMock(spec=LLMClient)
    llm.generate.side_effect = RuntimeError("GPU OOM")
    probe = ParametricProbe(llm)
    mem = probe.probe("question")
    assert mem.answer == ""
    assert mem.confidence == 1.0


# ── ParametricProbe.detect_conflict ──────────────────────────────────────────

def test_no_conflict_when_entailment():
    probe = ParametricProbe(_llm("extracted answer"))
    mem = ParametricMemory(answer="Paris", confidence=0.1, raw_logprobs=[-0.1])
    result = probe.detect_conflict(mem, "Paris is the capital.", _nli("entailment"), _llm("Paris"))
    assert result.has_conflict is False
    assert result.conflict_type == "no_conflict"
    assert result.trust_retrieved == 0.0


def test_real_conflict_when_contradiction_and_certain():
    probe = ParametricProbe(_llm("extracted answer"))
    # confidence < threshold → model is certain
    mem = ParametricMemory(answer="1989", confidence=0.05, raw_logprobs=[-0.05])
    result = probe.detect_conflict(mem, "The event was in 1991.", _nli("contradiction"), _llm("1991"), gcs=0.9)
    assert result.has_conflict is True
    assert result.conflict_type == "parametric_vs_retrieved"
    assert result.trust_retrieved == pytest.approx(0.9)


def test_uncertain_conflict_when_contradiction_but_low_confidence():
    probe = ParametricProbe(_llm("extracted answer"))
    # confidence >= threshold → model is uncertain → not a real conflict
    mem = ParametricMemory(answer="maybe 1989?", confidence=0.8, raw_logprobs=[-3.0])
    result = probe.detect_conflict(mem, "The event was in 1991.", _nli("contradiction"), _llm("1991"))
    assert result.has_conflict is False
    assert result.conflict_type == "uncertain"
    assert result.trust_retrieved == 0.0


def test_trust_retrieved_zero_when_no_conflict():
    probe = ParametricProbe(_llm("Paris"))
    mem = ParametricMemory(answer="Paris", confidence=0.05, raw_logprobs=[-0.05])
    result = probe.detect_conflict(mem, "Paris is the capital.", _nli("entailment"), _llm("Paris"), gcs=0.95)
    assert result.trust_retrieved == 0.0


def test_empty_parametric_answer_returns_uncertain():
    probe = ParametricProbe(_llm(""))
    mem = ParametricMemory(answer="", confidence=1.0, raw_logprobs=[])
    result = probe.detect_conflict(mem, "some evidence", _nli("contradiction"), _llm("answer"))
    assert result.has_conflict is False
    assert result.conflict_type == "uncertain"


def test_conflict_result_fields_populated():
    probe = ParametricProbe(_llm("retrieved_ans"))
    mem = ParametricMemory(answer="param_ans", confidence=0.05, raw_logprobs=[-0.05])
    result = probe.detect_conflict(mem, "evidence text", _nli("contradiction"), _llm("retrieved_ans"), gcs=0.8)
    assert result.parametric_answer == "param_ans"
    assert result.retrieved_answer  # non-empty
